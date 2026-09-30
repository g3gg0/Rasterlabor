import { composePose } from './pcb-realignment.js';

const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

// Small dark cores survive changes in reflections better than ORB descriptors
// on the otherwise smooth solder mask. Large rings and traces are excluded.
export function pcbVias(image) {
  const level = image.levels.findLast(item => item.scale <= 2) ?? image.levels[0];
  const { width, height, scale, gray } = level;
  const stride = width + 1, integral = new Float64Array(stride * (height + 1));
  const valid = new Uint32Array(integral.length);
  for (let y = 0; y < height; y++) {
    let sum = 0, count = 0;
    for (let x = 0; x < width; x++) {
      const value = gray[y * width + x], index = (y + 1) * stride + x + 1;
      if (value >= 0) { sum += value; count++; }
      integral[index] = integral[index - stride] + sum;
      valid[index] = valid[index - stride] + count;
    }
  }
  const radius = Math.ceil(16 / scale), contrast = new Float32Array(gray.length);
  for (let y = radius; y < height - radius; y++) for (let x = radius; x < width - radius; x++) {
    const index = y * width + x;
    if (gray[index] < 0) continue;
    const a = (y - radius) * stride + x - radius, b = a + 2 * radius + 1;
    const c = (y + radius + 1) * stride + x - radius, d = c + 2 * radius + 1;
    const count = valid[d] - valid[b] - valid[c] + valid[a];
    if (count < (2 * radius + 1) ** 2 * 0.6) continue;
    contrast[index] = (integral[d] - integral[b] - integral[c] + integral[a]) / count - gray[index];
  }
  const visited = new Uint8Array(gray.length), points = [];
  for (let start = 0; start < gray.length; start++) {
    if (visited[start] || contrast[start] < 8) continue;
    const queue = [start]; visited[start] = 1;
    let minX = width, maxX = 0, minY = height, maxY = 0, sx = 0, sy = 0, mass = 0;
    for (let cursor = 0; cursor < queue.length; cursor++) {
      const index = queue[cursor], x = index % width, y = Math.floor(index / width), weight = contrast[index];
      minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      sx += (x + 0.5) * weight; sy += (y + 0.5) * weight; mass += weight;
      for (const next of [index - 1, index + 1, index - width, index + width]) {
        if (next < 0 || next >= gray.length || visited[next] || contrast[next] < 8) continue;
        visited[next] = 1; queue.push(next);
      }
    }
    const w = (maxX - minX + 1) * scale, h = (maxY - minY + 1) * scale;
    if (Math.min(w, h) < 8 || Math.max(w, h) > 50 || Math.max(w, h) / Math.min(w, h) > 1.8 ||
        queue.length / ((maxX - minX + 1) * (maxY - minY + 1)) < 0.4) continue;
    const point = { x: sx / mass * scale - image.width / 2, y: sy / mass * scale - image.height / 2, width: w, height: h };
    const descriptor = [];
    for (let dy = -16; dy <= 16; dy += 4) for (let dx = -16; dx <= 16; dx += 4) {
      const x = Math.floor((point.x + image.width / 2 + dx) / scale), y = Math.floor((point.y + image.height / 2 + dy) / scale);
      descriptor.push(x >= 0 && y >= 0 && x < width && y < height ? gray[y * width + x] : -1);
    }
    Object.defineProperty(point, 'descriptor', { value: descriptor });
    points.push(point);
  }
  return points;
}

function fit(pairs) {
  const a = { x: 0, y: 0 }, b = { x: 0, y: 0 };
  for (const pair of pairs) { a.x += pair.source.x; a.y += pair.source.y; b.x += pair.target.x; b.y += pair.target.y; }
  for (const point of [a, b]) { point.x /= pairs.length; point.y /= pairs.length; }
  let dot = 0, cross = 0;
  for (const pair of pairs) {
    const x = pair.source.x - a.x, y = pair.source.y - a.y;
    dot += x * (pair.target.x - b.x) + y * (pair.target.y - b.y);
    cross += x * (pair.target.y - b.y) - y * (pair.target.x - b.x);
  }
  const rotation = Math.atan2(cross, dot), origin = composePose({ x: 0, y: 0, rotation }, a);
  return { x: b.x - origin.x, y: b.y - origin.y, rotation };
}

function viaPatchScore(first, second, source, target, angle) {
  const a = first.levels.findLast(level => level.scale <= 2), b = second.levels.findLast(level => level.scale <= 2);
  const samples = [];
  for (let y = -16; y <= 16; y += 2) for (let x = -16; x <= 16; x += 2) {
    const rotated = composePose({ x: 0, y: 0, rotation: angle }, { x, y });
    const ax = Math.floor((source.x + x + first.width / 2) / a.scale), ay = Math.floor((source.y + y + first.height / 2) / a.scale);
    const bx = Math.floor((target.x + rotated.x + second.width / 2) / b.scale), by = Math.floor((target.y + rotated.y + second.height / 2) / b.scale);
    if (ax < 0 || ay < 0 || bx < 0 || by < 0 || ax >= a.width || ay >= a.height || bx >= b.width || by >= b.height) continue;
    const av = a.gray[ay * a.width + ax], bv = b.gray[by * b.width + bx];
    if (av >= 0 && bv >= 0) samples.push([av, bv]);
  }
  if (samples.length < 128) return -1;
  const am = samples.reduce((sum, pair) => sum + pair[0], 0) / samples.length;
  const bm = samples.reduce((sum, pair) => sum + pair[1], 0) / samples.length;
  let aa = 0, bb = 0, ab = 0;
  for (const [av, bv] of samples) { aa += (av - am) ** 2; bb += (bv - bm) ** 2; ab += (av - am) * (bv - bm); }
  return ab / Math.sqrt(aa * bb);
}

export function recoverPcbViaPair(referenceImage, currentImage, referenceCenter, currentCenter, seed, limits) {
  const reference = limits.referencePoints ?? pcbVias(referenceImage), current = limits.currentPoints ?? pcbVias(currentImage);
  const targets = reference.map(point => composePose(referenceCenter, point));
  const sources = current.map(point => composePose(seed, point));
  const nearest = (point, candidates) => candidates.reduce((best, candidate, index) =>
    distance(point, candidate) < best.distance ? { index, distance: distance(point, candidate) } : best,
  { index: -1, distance: 192 });
  const pairs = [];
  for (const [index, point] of sources.entries()) {
    const target = nearest(point, targets);
    if (target.index < 0 || nearest(targets[target.index], sources).index !== index) continue;
    if (limits.minimumDescriptor !== undefined && descriptorScore(current[index].descriptor,
      reference[target.index].descriptor) < limits.minimumDescriptor) continue;
    pairs.push({ source: current[index], target: targets[target.index], reference: reference[target.index] });
  }
  const result = { accepted: false, reference, current, matches: pairs.length };
  if (pairs.length === 1) {
    const pair = pairs[0], offset = composePose({ x: 0, y: 0, rotation: currentCenter.rotation }, pair.source);
    const pose = { x: pair.target.x - offset.x, y: pair.target.y - offset.y, rotation: currentCenter.rotation };
    const score = viaPatchScore(currentImage, referenceImage, pair.source, pair.reference,
      pose.rotation - referenceCenter.rotation);
    return { ...result, accepted: score >= 0.7 && distance(pose, currentCenter) <= limits.coarseRadius,
      pose, translationOnly: true, inliers: 1, residual: 0, patchScores: [score],
      pairs: [{ source: pair.source, reference: pair.reference }] };
  }
  let best = [];
  for (let first = 0; first < pairs.length; first++) for (let second = first + 1; second < pairs.length; second++) {
    if (distance(pairs[first].source, pairs[second].source) < 500) continue;
    const pose = fit([pairs[first], pairs[second]]);
    const angle = Math.abs(Math.atan2(Math.sin(pose.rotation - currentCenter.rotation), Math.cos(pose.rotation - currentCenter.rotation))) * 180 / Math.PI;
    if (angle > 5 || distance(pose, currentCenter) > limits.coarseRadius) continue;
    const inliers = pairs.filter(pair => distance(composePose(pose, pair.source), pair.target) <= 8);
    if (inliers.length > best.length) best = inliers;
  }
  if (best.length < 2 || best.length < pairs.length * (limits.minimumConsensus ?? 0.6)) return result;
  const span = Math.max(...best.flatMap(a => best.map(b => distance(a.source, b.source))));
  if (span < (best.length === 2 ? 1000 : 500)) return result;
  const pose = fit(best), residual = Math.sqrt(best.reduce((sum, pair) => sum + distance(composePose(pose, pair.source), pair.target) ** 2, 0) / best.length);
  const patchScores = best.map(pair => viaPatchScore(currentImage, referenceImage, pair.source, pair.reference,
    pose.rotation - referenceCenter.rotation));
  Object.assign(result, { accepted: residual <= 4 && patchScores.every(score => score >= (limits.minimumPatchScore ?? 0.7)), pose, inliers: best.length, span, residual, patchScores,
    pairs: best.map(pair => ({ source: pair.source, reference: pair.reference })) });
  return result;
}

function descriptorScore(a, b) {
  const pairs = a.map((value, index) => [value, b[index]]).filter(pair => pair[0] >= 0 && pair[1] >= 0);
  if (pairs.length < 40) return -1;
  const am = pairs.reduce((sum, pair) => sum + pair[0], 0) / pairs.length;
  const bm = pairs.reduce((sum, pair) => sum + pair[1], 0) / pairs.length;
  let aa = 0, bb = 0, ab = 0;
  for (const [av, bv] of pairs) { aa += (av - am) ** 2; bb += (bv - bm) ** 2; ab += (av - am) * (bv - bm); }
  return ab / Math.sqrt(aa * bb);
}

// Different reference frames can each see a different via. Their tracked poses
// put both landmarks into one reference system without inventing image pixels.
export function recoverPcbViaPool(currentImage, referenceFrames, currentCenter, seed, limits) {
  const clusters = [];
  for (const frame of referenceFrames) for (const point of frame.points) {
    const world = composePose(frame.center, point);
    const cluster = clusters.find(item => distance(item.target, world) <= 16);
    if (cluster) {
      cluster.observations.push({ frame: frame.frame, point, world });
      cluster.target.x = cluster.observations.reduce((sum, item) => sum + item.world.x, 0) / cluster.observations.length;
      cluster.target.y = cluster.observations.reduce((sum, item) => sum + item.world.y, 0) / cluster.observations.length;
    } else clusters.push({ target: { x: world.x, y: world.y }, observations: [{ frame: frame.frame, point, world }] });
  }
  const stable = clusters.filter(cluster => new Set(cluster.observations.map(item => item.frame)).size >= 2);
  const current = pcbVias(currentImage), sources = current.map(point => composePose(seed, point));
  const nearest = (point, candidates) => candidates.reduce((best, candidate, index) =>
    distance(point, candidate) < best.distance ? { index, distance: distance(point, candidate) } : best,
  { index: -1, distance: 192 });
  const pairs = [];
  for (const [index, point] of sources.entries()) {
    const match = nearest(point, stable.map(cluster => cluster.target));
    if (match.index < 0 || nearest(stable[match.index].target, sources).index !== index) continue;
    const cluster = stable[match.index];
    const patchScore = Math.max(...cluster.observations.map(item => descriptorScore(current[index].descriptor, item.point.descriptor)));
    if (patchScore < 0.7) continue;
    pairs.push({ source: current[index], target: cluster.target, patchScore,
      references: cluster.observations.map(item => item.frame),
      observations:cluster.observations.map(item=>({frame:item.frame,world:item.world})) });
  }
  const result = { accepted: false, matches: pairs.length, referenceFrames: referenceFrames.length };
  let best = [];
  for (let first = 0; first < pairs.length; first++) for (let second = first + 1; second < pairs.length; second++) {
    if (distance(pairs[first].source, pairs[second].source) < 500) continue;
    const pose = fit([pairs[first], pairs[second]]);
    const angle = Math.abs(Math.atan2(Math.sin(pose.rotation - currentCenter.rotation), Math.cos(pose.rotation - currentCenter.rotation))) * 180 / Math.PI;
    if (angle > 5 || distance(pose, currentCenter) > limits.coarseRadius) continue;
    const inliers = pairs.filter(pair => distance(composePose(pose, pair.source), pair.target) <= 8);
    if (inliers.length > best.length) best = inliers;
  }
  if (best.length < 2 || best.length < pairs.length * 0.6) return result;
  const span = Math.max(...best.flatMap(a => best.map(b => distance(a.source, b.source))));
  if (span < (best.length === 2 ? 1000 : 500)) return result;
  const pose = fit(best), residual = Math.sqrt(best.reduce((sum, pair) => sum + distance(composePose(pose, pair.source), pair.target) ** 2, 0) / best.length);
  return { ...result, accepted: residual <= 4, pose, inliers: best.length, span, residual, pairs: best,
    references: [...new Set(best.flatMap(pair => pair.references))] };
}
