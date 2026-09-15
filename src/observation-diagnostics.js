function percentile(values, fraction) {
  if (!values.length) return 0;
  const sorted = [...values].sort((first, second) => first - second);
  return sorted[Math.min(sorted.length - 1, Math.round((sorted.length - 1) * fraction))];
}

function framePose(points, step) {
  let weightSum = 0;
  let sourceX = 0;
  let sourceY = 0;
  let targetX = 0;
  let targetY = 0;
  for (const point of points) {
    const weight = point.confidence ?? 1;
    weightSum += weight;
    sourceX += weight * step * point.col;
    sourceY += weight * step * point.row;
    targetX += weight * point.x;
    targetY += weight * point.y;
  }
  if (!(weightSum > 0)) return null;
  sourceX /= weightSum;
  sourceY /= weightSum;
  targetX /= weightSum;
  targetY /= weightSum;
  let dot = 0;
  let cross = 0;
  let sourceVariance = 0;
  for (const point of points) {
    const weight = point.confidence ?? 1;
    const qx = step * point.col - sourceX;
    const qy = step * point.row - sourceY;
    const px = point.x - targetX;
    const py = point.y - targetY;
    dot += weight * (qx * px + qy * py);
    cross += weight * (qx * py - qy * px);
    sourceVariance += weight * (qx * qx + qy * qy);
  }
  const theta = Math.atan2(cross, dot);
  const cosine = Math.cos(theta);
  const sine = Math.sin(theta);
  const scale = sourceVariance > 0 ? Math.hypot(dot, cross) / sourceVariance : 1;
  return { sourceX, sourceY, targetX, targetY, cosine, sine, scale };
}

function residual(point, pose, step, scale) {
  const qx = step * point.col - pose.sourceX;
  const qy = step * point.row - pose.sourceY;
  const predictedX = pose.targetX + scale * (pose.cosine * qx - pose.sine * qy);
  const predictedY = pose.targetY + scale * (pose.sine * qx + pose.cosine * qy);
  return Math.hypot(point.x - predictedX, point.y - predictedY);
}

export function frameGeometry(points, step = 1) {
  if (points.length < 3) return null;
  let weightSum = 0;
  let sourceX = 0;
  let sourceY = 0;
  let targetX = 0;
  let targetY = 0;
  for (const point of points) {
    const weight = point.confidence ?? 1;
    weightSum += weight;
    sourceX += weight * step * point.col;
    sourceY += weight * step * point.row;
    targetX += weight * point.x;
    targetY += weight * point.y;
  }
  if (!(weightSum > 0)) return null;
  sourceX /= weightSum;
  sourceY /= weightSum;
  targetX /= weightSum;
  targetY /= weightSum;
  let sourceXX = 0;
  let sourceXY = 0;
  let sourceYY = 0;
  let targetXSourceX = 0;
  let targetXSourceY = 0;
  let targetYSourceX = 0;
  let targetYSourceY = 0;
  for (const point of points) {
    const weight = point.confidence ?? 1;
    const sourceDx = step * point.col - sourceX;
    const sourceDy = step * point.row - sourceY;
    const targetDx = point.x - targetX;
    const targetDy = point.y - targetY;
    sourceXX += weight * sourceDx * sourceDx;
    sourceXY += weight * sourceDx * sourceDy;
    sourceYY += weight * sourceDy * sourceDy;
    targetXSourceX += weight * targetDx * sourceDx;
    targetXSourceY += weight * targetDx * sourceDy;
    targetYSourceX += weight * targetDy * sourceDx;
    targetYSourceY += weight * targetDy * sourceDy;
  }
  const determinant = sourceXX * sourceYY - sourceXY * sourceXY;
  if (!(determinant > 1e-12)) return null;
  const inverseXX = sourceYY / determinant;
  const inverseXY = -sourceXY / determinant;
  const inverseYY = sourceXX / determinant;
  const a = targetXSourceX * inverseXX + targetXSourceY * inverseXY;
  const b = targetXSourceX * inverseXY + targetXSourceY * inverseYY;
  const c = targetYSourceX * inverseXX + targetYSourceY * inverseXY;
  const d = targetYSourceX * inverseXY + targetYSourceY * inverseYY;
  const areaScale = a * d - b * c;
  if (!(areaScale > 0)) return null;
  const scale = Math.sqrt(areaScale);
  const trace = a * a + b * b + c * c + d * d;
  const discriminant = Math.sqrt(Math.max(0, trace * trace - 4 * areaScale * areaScale));
  const maximumSingular = Math.sqrt(Math.max(0, (trace + discriminant) / 2));
  const minimumSingular = Math.sqrt(Math.max(0, (trace - discriminant) / 2));
  return { scale, anisotropy: minimumSingular > 0 ? maximumSingular / minimumSingular : Infinity, matrix: { a, b, c, d } };
}

export function selectConsistentFrames(frames, step = 1, scaleTolerance = 0.01, anisotropyTolerance = 0.01) {
  const candidates = frames.map(frame => ({ frame, geometry: frameGeometry(frame.points || [], step) })).filter(candidate => candidate.geometry);
  if (!candidates.length) return { selectedIds: new Set(), rejectedIds: new Set(), center: null, candidates: [] };
  const scaleRadius = Math.log1p(scaleTolerance);
  const anisotropyRadius = Math.log1p(anisotropyTolerance);
  const descriptors = candidates.map(candidate => ({ ...candidate, logScale: Math.log(candidate.geometry.scale), logAnisotropy: Math.log(candidate.geometry.anisotropy) }));
  let center = descriptors[0];
  let maximumNeighbors = -1;
  for (const candidate of descriptors) {
    const neighbors = descriptors.reduce((count, other) => count + Number(
      Math.abs(candidate.logScale - other.logScale) <= scaleRadius &&
      Math.abs(candidate.logAnisotropy - other.logAnisotropy) <= anisotropyRadius
    ), 0);
    if (neighbors > maximumNeighbors) { center = candidate; maximumNeighbors = neighbors; }
  }
  const selectedIds = new Set();
  const rejectedIds = new Set();
  for (const candidate of descriptors) {
    const selected = Math.abs(candidate.logScale - center.logScale) <= scaleRadius &&
      Math.abs(candidate.logAnisotropy - center.logAnisotropy) <= anisotropyRadius;
    (selected ? selectedIds : rejectedIds).add(candidate.frame.id);
  }
  return { selectedIds, rejectedIds, center: { scale: center.geometry.scale, anisotropy: center.geometry.anisotropy }, candidates: descriptors.map(candidate => ({
    id: candidate.frame.id, scale: candidate.geometry.scale, anisotropy: candidate.geometry.anisotropy, selected: selectedIds.has(candidate.frame.id)
  })) };
}

export function analyzeObservations(frames, width, height, step = 1, cols = 90, rows = 160) {
  const selected = [...frames].filter(frame => frame.enabled && frame.points?.length >= 3).sort((first, second) => first.id - second.id);
  const frameCounts = new Uint32Array(cols * rows);
  const residualSums = new Float64Array(cols * rows);
  const residualCounts = new Uint32Array(cols * rows);
  const rigidRms = [];
  const similarityRms = [];
  const scales = [];
  const movements = [];
  const directions = new Set();
  let nearDuplicates = 0;
  let previousCenter = null;
  for (const frame of selected) {
    const pose = framePose(frame.points, step);
    if (!pose) continue;
    let rigidSquared = 0;
    let similaritySquared = 0;
    const occupied = new Set();
    for (const point of frame.points) {
      const rigidError = residual(point, pose, step, 1);
      const similarityError = residual(point, pose, step, pose.scale);
      rigidSquared += rigidError * rigidError;
      similaritySquared += similarityError * similarityError;
      const col = Math.max(0, Math.min(cols - 1, Math.floor(point.x / width * cols)));
      const row = Math.max(0, Math.min(rows - 1, Math.floor(point.y / height * rows)));
      const index = row * cols + col;
      occupied.add(index);
      residualSums[index] += rigidError;
      residualCounts[index]++;
    }
    for (const index of occupied) frameCounts[index]++;
    rigidRms.push(Math.sqrt(rigidSquared / frame.points.length));
    similarityRms.push(Math.sqrt(similaritySquared / frame.points.length));
    scales.push(pose.scale);
    const center = { x: pose.targetX, y: pose.targetY };
    if (previousCenter) {
      const dx = center.x - previousCenter.x;
      const dy = center.y - previousCenter.y;
      const distance = Math.hypot(dx, dy);
      movements.push(distance);
      if (distance < Math.max(1, step * 0.1)) nearDuplicates++;
      if (distance >= 1) {
        directions.add((Math.round(Math.atan2(dy, dx) / (Math.PI / 4)) + 8) % 8);
      }
    }
    previousCenter = center;
  }
  return { cols, rows, frameCounts, residualSums, residualCounts, metrics: {
    frames: selected.length,
    points: selected.reduce((sum, frame) => sum + frame.points.length, 0),
    rigidMedian: percentile(rigidRms, 0.5), rigidP95: percentile(rigidRms, 0.95),
    similarityMedian: percentile(similarityRms, 0.5), similarityP95: percentile(similarityRms, 0.95),
    scaleP05: percentile(scales, 0.05), scaleP95: percentile(scales, 0.95),
    movementMedian: percentile(movements, 0.5), directionBins: directions.size,
    nearDuplicateFraction: movements.length ? nearDuplicates / movements.length : 0
  } };
}
