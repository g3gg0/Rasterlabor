import jsfeat from 'jsfeat';

const wrap = angle => Math.atan2(Math.sin(angle), Math.cos(angle));
const rotate = (x, y, angle) => ({ x: Math.cos(angle) * x - Math.sin(angle) * y,
  y: Math.sin(angle) * x + Math.cos(angle) * y });

function featureLevel(image, minimumEdge) {
  return image.levels.findLast(level => Math.max(level.width, level.height) >= minimumEdge) ?? image.levels[0];
}

function validPatch(level, x, y, radius = 17) {
  if (x < radius || y < radius || x >= level.width - radius || y >= level.height - radius) return false;
  return [[0, 0], [-radius, 0], [radius, 0], [0, -radius], [0, radius]].every(([dx, dy]) =>
    level.gray[Math.round(y + dy) * level.width + Math.round(x + dx)] >= 0);
}

function features(image, maximum = 700, minimumEdge = 240) {
  const level = featureLevel(image, minimumEdge);
  const matrix = new jsfeat.matrix_t(level.width, level.height, jsfeat.U8_t | jsfeat.C1_t);
  for (let index = 0; index < level.gray.length; index++) matrix.data[index] = Math.max(0, level.gray[index]);
  jsfeat.imgproc.gaussian_blur(matrix, matrix, 3, 0);
  const capacity = Math.min(level.width * level.height, 20000);
  const corners = Array.from({ length: capacity }, () => new jsfeat.keypoint_t(0, 0, 0, 0, 0));
  jsfeat.yape06.laplacian_threshold = 20;
  jsfeat.yape06.min_eigen_value_threshold = 16;
  const count = Math.min(capacity, jsfeat.yape06.detect(matrix, corners, 20));
  const selected = corners.slice(0, count).filter(point => validPatch(level, point.x, point.y))
    .sort((first, second) => second.score - first.score).slice(0, maximum);
  const descriptors = new jsfeat.matrix_t(32, selected.length, jsfeat.U8_t | jsfeat.C1_t);
  jsfeat.orb.describe(matrix, selected, selected.length, descriptors);
  return { points: selected.map(point => ({ x: (point.x + 0.5) * level.scale - image.width / 2,
    y: (point.y + 0.5) * level.scale - image.height / 2 })), descriptors: descriptors.data, scale: level.scale };
}

function hamming(first, firstOffset, second, secondOffset) {
  let distance = 0;
  for (let byte = 0; byte < 32; byte++) {
    let value = first[firstOffset + byte] ^ second[secondOffset + byte];
    value -= (value >>> 1) & 0x55;
    value = (value & 0x33) + ((value >>> 2) & 0x33);
    distance += (value + (value >>> 4)) & 0x0f;
  }
  return distance;
}

function descriptorMatches(first, second) {
  const nearest = (source, target, sourceIndex) => {
    let best = -1; let bestDistance = Infinity; let secondDistance = Infinity;
    for (let targetIndex = 0; targetIndex < target.points.length; targetIndex++) {
      const distance = hamming(source.descriptors, sourceIndex * 32, target.descriptors, targetIndex * 32);
      if (distance < bestDistance) { secondDistance = bestDistance; bestDistance = distance; best = targetIndex; }
      else if (distance < secondDistance) secondDistance = distance;
    }
    return { best, bestDistance, secondDistance };
  };
  const forward = first.points.map((_, index) => nearest(first, second, index));
  const reverse = second.points.map((_, index) => nearest(second, first, index));
  return forward.flatMap((match, firstIndex) => match.best >= 0 && match.bestDistance <= 80 &&
    match.bestDistance < match.secondDistance * 0.82 && reverse[match.best]?.best === firstIndex ?
    [{ current: first.points[firstIndex], reference: second.points[match.best], distance: match.bestDistance }] : []);
}

function poseFromTwo(first, second) {
  const currentAngle = Math.atan2(second.current.y - first.current.y, second.current.x - first.current.x);
  const referenceAngle = Math.atan2(second.reference.y - first.reference.y, second.reference.x - first.reference.x);
  const angle = wrap(referenceAngle - currentAngle);
  const point = rotate(first.current.x, first.current.y, angle);
  return { x: first.reference.x - point.x, y: first.reference.y - point.y, angle };
}

function residual(match, pose) {
  const point = rotate(match.current.x, match.current.y, pose.angle);
  return Math.hypot(point.x + pose.x - match.reference.x, point.y + pose.y - match.reference.y);
}

function refine(matches) {
  const current = { x: 0, y: 0 }; const reference = { x: 0, y: 0 };
  for (const match of matches) {
    current.x += match.current.x; current.y += match.current.y;
    reference.x += match.reference.x; reference.y += match.reference.y;
  }
  current.x /= matches.length; current.y /= matches.length;
  reference.x /= matches.length; reference.y /= matches.length;
  let dot = 0; let cross = 0;
  for (const match of matches) {
    const ax = match.current.x - current.x, ay = match.current.y - current.y;
    const bx = match.reference.x - reference.x, by = match.reference.y - reference.y;
    dot += ax * bx + ay * by; cross += ax * by - ay * bx;
  }
  const angle = Math.atan2(cross, dot); const center = rotate(current.x, current.y, angle);
  return { x: reference.x - center.x, y: reference.y - center.y, angle };
}

export function registerFeatureOverlap(current, reference, referencePose,
  { iterations = 1200, minimumFeatureEdge = 240, maximumFeatures = 700 } = {}) {
  const currentFeatures = features(current, maximumFeatures, minimumFeatureEdge);
  const referenceFeatures = features(reference, maximumFeatures, minimumFeatureEdge);
  const matches = descriptorMatches(currentFeatures, referenceFeatures);
  const diagnostics = { featureScale: currentFeatures.scale,
    currentFeatures: currentFeatures.points.length, referenceFeatures: referenceFeatures.points.length, matches: matches.length };
  const minimumMatches = 8;
  if (matches.length < minimumMatches) return { ...diagnostics, accepted: false, reason: 'Zu wenige Feature-Paare' };
  const threshold = Math.max(4, currentFeatures.scale * 1.5);
  let best = [];
  let state = 0x9e3779b9;
  for (let iteration = 0; iteration < iterations; iteration++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0; const first = state % matches.length;
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0; const second = state % matches.length;
    if (first === second) continue;
    const model = poseFromTwo(matches[first], matches[second]);
    const inliers = matches.filter(match => residual(match, model) <= threshold);
    if (inliers.length > best.length) best = inliers;
  }
  if (best.length < minimumMatches || best.length < matches.length * 0.12) {
    return { ...diagnostics, accepted: false, reason: 'Kein robuster Feature-Konsens', inliers: best.length };
  }
  const measurement = refine(best);
  const errors = best.map(match => residual(match, measurement)).sort((first, second) => first - second);
  const translation = rotate(measurement.x, measurement.y, referencePose.rotation);
  const pose={x:referencePose.x+translation.x,y:referencePose.y+translation.y,
    rotation:wrap(referencePose.rotation+measurement.angle)};
  return { ...diagnostics, accepted: true, method: 'ORB + RANSAC', inliers: best.length,
    residual: errors[Math.floor(errors.length / 2)], score: best.length / matches.length,
    pose };
}
