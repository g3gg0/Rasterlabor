import { evaluate } from './spline.js';

const median = values => [...values].sort((first, second) => first - second)[Math.floor(values.length / 2)] ?? 0;

function rigidPose(pairs) {
  const weightSum = pairs.reduce((sum, pair) => sum + pair.weight, 0);
  if (!(weightSum > 0) || pairs.length < 3) return null;
  const mean = key => pairs.reduce((sum, pair) => sum + pair.weight * pair[key], 0) / weightSum;
  const sourceX = mean('sourceX');
  const sourceY = mean('sourceY');
  const targetX = mean('targetX');
  const targetY = mean('targetY');
  let cosineSum = 0;
  let sineSum = 0;
  for (const pair of pairs) {
    const fromX = pair.sourceX - sourceX;
    const fromY = pair.sourceY - sourceY;
    const toX = pair.targetX - targetX;
    const toY = pair.targetY - targetY;
    cosineSum += pair.weight * (fromX * toX + fromY * toY);
    sineSum += pair.weight * (fromX * toY - fromY * toX);
  }
  const theta = Math.atan2(sineSum, cosineSum);
  const cosine = Math.cos(theta);
  const sine = Math.sin(theta);
  return { theta, cosine, sine, tx: targetX - cosine * sourceX + sine * sourceY,
    ty: targetY - sine * sourceX - cosine * sourceY };
}

export function fitCameraPose(points, field, center = { x: field.width / 2, y: field.height / 2 }) {
  const inField = (px, py) => Number.isFinite(px) && Number.isFinite(py) &&
    px >= 0 && py >= 0 && px <= field.width - 1 && py <= field.height - 1;
  let pairs = points.filter(point => inField(point.col, point.row) && inField(point.x, point.y)).map(point => {
    const source = evaluate(field, point.col, point.row);
    const target = evaluate(field, point.x, point.y);
    return { sourceX: source.x, sourceY: source.y, targetX: target.x, targetY: target.y,
      weight: point.confidence ?? 1 };
  }).filter(pair => Object.values(pair).every(Number.isFinite) && pair.weight > 0);
  let pose = null;
  for (let iteration = 0; iteration < 3 && pairs.length >= 3; iteration++) {
    pose = rigidPose(pairs);
    if (!pose) return null;
    const residuals = pairs.map(pair => Math.hypot(
      pose.cosine * pair.sourceX - pose.sine * pair.sourceY + pose.tx - pair.targetX,
      pose.sine * pair.sourceX + pose.cosine * pair.sourceY + pose.ty - pair.targetY));
    const middle = median(residuals);
    const limit = Math.max(0.35, middle * 3);
    const retained = pairs.filter((_, index) => residuals[index] <= limit);
    if (retained.length === pairs.length || retained.length < 3) break;
    pairs = retained;
  }
  pose = rigidPose(pairs);
  if (!pose) return null;
  const shiftedX = center.x - pose.tx;
  const shiftedY = center.y - pose.ty;
  const cameraX = pose.cosine * shiftedX + pose.sine * shiftedY - center.x;
  const cameraY = -pose.sine * shiftedX + pose.cosine * shiftedY - center.y;
  const residuals = pairs.map(pair => Math.hypot(
    pose.cosine * pair.sourceX - pose.sine * pair.sourceY + pose.tx - pair.targetX,
    pose.sine * pair.sourceX + pose.cosine * pair.sourceY + pose.ty - pair.targetY));
  return { x: cameraX, y: cameraY, rotation: -pose.theta, points: pairs.length,
    rms: Math.sqrt(residuals.reduce((sum, value) => sum + value ** 2, 0) / residuals.length) };
}

export function stabilizePose(path, windowSize) {
  const valid = sample => sample?.raw &&
    ['x', 'y', 'rotation', 'points'].every(key => Number.isFinite(sample.raw[key]));
  if (!valid(path.at(-1))) return null;
  const samples = path.slice(-Math.max(1, Math.round(windowSize))).filter(valid);
  if (!samples.length) return null;
  const weightSum = samples.reduce((sum, sample) => sum + Math.max(1, sample.raw.points), 0);
  const weighted = key => samples.reduce((sum, sample) => sum + Math.max(1, sample.raw.points) * sample.raw[key], 0) / weightSum;
  const sine = samples.reduce((sum, sample) => sum + Math.max(1, sample.raw.points) * Math.sin(sample.raw.rotation), 0);
  const cosine = samples.reduce((sum, sample) => sum + Math.max(1, sample.raw.points) * Math.cos(sample.raw.rotation), 0);
  return { x: weighted('x'), y: weighted('y'), rotation: Math.atan2(sine, cosine) };
}