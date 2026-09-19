import { Matrix, solve } from 'ml-matrix';
import { MASK_SEARCH } from './patch-mask.js';
import { WebGpuContextTracker } from './webgpu-context-tracker.js';
import { remapRGBA } from './maps.js';
import { applyLoopClosure } from './tracking-data.js';
import { refitGray } from './refit-preprocess.js';

const wrap = angle => Math.atan2(Math.sin(angle), Math.cos(angle));
const rotate = (x, y, angle) => ({ x: Math.cos(angle) * x - Math.sin(angle) * y, y: Math.sin(angle) * x + Math.cos(angle) * y });
const median = values => [...values].sort((first, second) => first - second)[Math.floor(values.length / 2)];
export const CYCLE_STRICT_PX = 1.5;
export const CYCLE_CONDITIONAL_PX = 7.5;

export function shouldRunContext(frame, previousFrame, interval = 8, score = Infinity) {
  if (!Number.isInteger(interval) || interval < 1) throw new Error('Ungueltiges Umfeldintervall.');
  return previousFrame === null || frame - previousFrame >= interval || score < 0.75;
}

function worldPoint(pose, point) {
  const rotated = rotate(point.x, point.y, pose.rotation);
  return { x: pose.x + rotated.x, y: pose.y + rotated.y };
}

function localPoint(pose, point) {
  return rotate(point.x - pose.x, point.y - pose.y, -pose.rotation);
}

function percentile(values, fraction) {
  const sorted = [...values].sort((first, second) => first - second);
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position); const upper = Math.ceil(position);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

export function cycleErrorMetrics(expected, measured, points) {
  const translation = Math.hypot(measured.x - expected.x, measured.y - expected.y);
  const rotation = Math.abs(wrap(measured.rotation - expected.rotation));
  const errors = points.map(point => {
    const first = worldPoint(expected, point); const second = worldPoint(measured, point);
    return Math.hypot(first.x - second.x, first.y - second.y);
  });
  return { translation, rotation, rotationDegrees: rotation * 180 / Math.PI,
    median: errors.length ? median(errors) : translation,
    rms: errors.length ? Math.sqrt(errors.reduce((sum, value) => sum + value ** 2, 0) / errors.length) : translation,
    p95: errors.length ? percentile(errors, 0.95) : translation, samples: errors.length };
}

function overlapReferencePoints(cells, currentPose, referencePose, width, height) {
  return cells.map(tile => {
    const current = { x: ((tile % 16 + 0.5) / 16 - 0.5) * width,
      y: ((Math.floor(tile / 16) + 0.5) / 16 - 0.5) * height };
    return localPoint(referencePose, worldPoint(currentPose, current));
  });
}

export function contextImage(image, mask = null, preprocessing = null) {
  const { width, height } = image;
  if (mask && (mask.sourceWidth !== width || mask.sourceHeight !== height)) throw new Error('Umfeldmaske passt nicht zum entzerrten Bild.');
  const gray = new Int16Array(width * height);
  for (let index = 0; index < gray.length; index++) {
    const allowed = !mask || mask.data[Math.floor(Math.floor(index / width) / mask.cellSize) * mask.width + Math.floor((index % width) / mask.cellSize)] === MASK_SEARCH;
    gray[index] = allowed && image.data[index * 4 + 3] === 255 ? (preprocessing ?
      refitGray(image.data[index * 4], image.data[index * 4 + 1], image.data[index * 4 + 2], preprocessing) :
      Math.floor((299 * image.data[index * 4] + 587 * image.data[index * 4 + 1] + 114 * image.data[index * 4 + 2] + 500) / 1000)) : -1;
  }
  const levels = [{ width, height, gray, scale: 1 }];
  while (levels.length < 6 && Math.min(levels.at(-1).width, levels.at(-1).height) >= 96) {
    const previous = levels.at(-1);
    const next = { width: Math.floor(previous.width / 2), height: Math.floor(previous.height / 2), scale: previous.scale * 2 };
    next.gray = new Int16Array(next.width * next.height);
    for (let row = 0; row < next.height; row++) for (let column = 0; column < next.width; column++) {
      const offset = row * 2 * previous.width + column * 2;
      const pixels = [previous.gray[offset], previous.gray[offset + 1], previous.gray[offset + previous.width], previous.gray[offset + previous.width + 1]];
      next.gray[row * next.width + column] = pixels.every(value => value >= 0) ? Math.round(pixels.reduce((sum, value) => sum + value, 0) / 4) : -1;
    }
    levels.push(next);
  }
  return { width, height, levels, bytes: levels.reduce((sum, level) => sum + level.gray.byteLength, 0) };
}

function pixel(level, x, y) {
  if (x < 1 || y < 1 || x >= level.width - 2 || y >= level.height - 2) return null;
  const left = Math.floor(x); const top = Math.floor(y); const offset = top * level.width + left;
  const values = [level.gray[offset], level.gray[offset + 1], level.gray[offset + level.width], level.gray[offset + level.width + 1]];
  if (values.some(value => value < 0)) return null;
  const fractionX = x - left; const fractionY = y - top;
  return (1 - fractionY) * ((1 - fractionX) * values[0] + fractionX * values[1]) +
    fractionY * ((1 - fractionX) * values[2] + fractionX * values[3]);
}

function relative(current, reference) {
  return { ...rotate(current.x - reference.x, current.y - reference.y, -reference.rotation), angle: wrap(current.rotation - reference.rotation) };
}

function correlation(pairs) {
  const count = pairs.length;
  if (count < 16) return null;
  let sumFirst = 0; let sumSecond = 0; let squareFirst = 0; let squareSecond = 0; let product = 0;
  for (const pair of pairs) {
    sumFirst += pair.value; sumSecond += pair.valueSecond;
    squareFirst += pair.value ** 2; squareSecond += pair.valueSecond ** 2; product += pair.value * pair.valueSecond;
  }
  const varianceFirst = squareFirst - sumFirst ** 2 / count; const varianceSecond = squareSecond - sumSecond ** 2 / count;
  if (Math.min(varianceFirst, varianceSecond) / count < 4) return null;
  const covariance = product - sumFirst * sumSecond / count;
  return { score: covariance / Math.sqrt(varianceFirst * varianceSecond), gain: covariance / varianceFirst,
    bias: (sumSecond - covariance / varianceFirst * sumFirst) / count };
}

function partialCorrelation(pairs, scale, width, height) {
  const tileSize = Math.max(64, scale * 16, Math.min(width, height) / 6); const tiles = new Map();
  for (const pair of pairs) {
    const key = `${Math.floor(pair.x / tileSize)}:${Math.floor(pair.y / tileSize)}`;
    if (!tiles.has(key)) tiles.set(key, []);
    tiles.get(key).push(pair);
  }
  const ranked = [...tiles.values()].map(tilePairs => ({ pairs: tilePairs, stats: correlation(tilePairs) }))
    .filter(tile => tile.stats).sort((first, second) => second.stats.score - first.stats.score);
  if (ranked.length < 2) return null;
  const selected = ranked.slice(0, 2);
  const support = selected.reduce((sum, tile) => sum + tile.pairs.length, 0);
  if (support < 128) return null;
  return { score: selected.reduce((sum, tile) => sum + tile.stats.score * tile.pairs.length, 0) / support,
    pairs: selected.flatMap(tile => tile.pairs.map(pair => ({ ...pair, gain: tile.stats.gain, bias: tile.stats.bias }))) };
}

function* overlapSteps(current, reference, prediction, referencePose, {
  radius = 32, angle = 1, coarseStep = 0, coarseRadiusFactor = 0.8, partial = false
} = {}) {
  const initial = relative(prediction, referencePose);
  const angleRadius = angle * Math.PI / 180;
  const lever = Math.hypot(current.width, current.height) / 2;
  let estimate = { ...initial }; let evaluated = 0; let finalScore = -1; let support = 0;
  let finalEvaluate = null; let hadEnoughSamples = false; let initialized = false;
  for (let levelIndex = Math.min(current.levels.length, reference.levels.length) - 1; levelIndex >= 0; levelIndex--) {
    const source = current.levels[levelIndex]; const target = reference.levels[levelIndex];
    const samples = [];
    const stride = Math.max(1, Math.ceil(Math.sqrt(source.width * source.height / 6000)));
    for (let row = 2; row < source.height - 2; row += stride) for (let column = 2; column < source.width - 2; column += stride) {
      const value = source.gray[row * source.width + column];
      if (value < 0) continue;
      const x = (column + 0.5) * source.scale - current.width / 2;
      const y = (row + 0.5) * source.scale - current.height / 2;
      samples.push({ x, y, value });
    }
    if (samples.length < 128) continue;
    hadEnoughSamples = true;
    yield { type: 'prepare', samples, target, reference, lever };
    const evaluate = function* (pose) {
      evaluated++;
      if (Math.hypot(pose.x - initial.x, pose.y - initial.y) > radius || Math.abs(wrap(pose.angle - initial.angle)) > angleRadius) return null;
      return yield { type: 'evaluate', pose, cpu: () => {
      const pairs = [];
      for (const sample of samples) {
        const transformed = rotate(sample.x, sample.y, pose.angle);
        const px = (transformed.x + pose.x + reference.width / 2) / target.scale - 0.5;
        const py = (transformed.y + pose.y + reference.height / 2) / target.scale - 0.5;
        const value = pixel(target, px, py);
        if (value === null) continue;
        pairs.push({ ...sample, px, py, valueSecond: value, rotatedX: transformed.x, rotatedY: transformed.y });
      }
      const count = pairs.length;
      if (count < 128 || count < samples.length * 0.2) return null;
      const result = partial ? partialCorrelation(pairs, target.scale, current.width, current.height) : correlation(pairs);
      return result ? { ...result, pairs: result.pairs ?? pairs } : null;
      } };
    };
    let best = yield* evaluate(estimate);
    if (coarseStep > 0 && !initialized) {
      if (!best) {
        const centered = { x: 0, y: 0, angle: initial.angle };
        const trial = yield* evaluate(centered);
        if (trial) { best = trial; estimate = centered; }
      }
      const overlapRadius = Math.hypot(current.width, current.height) * coarseRadiusFactor;
      const gridRadius = Math.min(radius, overlapRadius);
      const gridStep = Math.max(target.scale, gridRadius / (best ? 8 : 16));
      for (let offsetY = -gridRadius; offsetY <= gridRadius; offsetY += gridStep) {
        for (let offsetX = -gridRadius; offsetX <= gridRadius; offsetX += gridStep) {
          if (Math.hypot(offsetX, offsetY) > gridRadius) continue;
          const proposed = { ...initial, x: initial.x + offsetX, y: initial.y + offsetY };
          const trial = yield* evaluate(proposed);
          if (trial && (!best || trial.score > best.score)) { best = trial; estimate = proposed; }
        }
      }
      for (let stepSize = gridStep / 2; stepSize >= target.scale; stepSize /= 2) {
        const center = estimate;
        for (const [dx, dy] of [[-stepSize, -stepSize], [0, -stepSize], [stepSize, -stepSize],
          [-stepSize, 0], [stepSize, 0], [-stepSize, stepSize], [0, stepSize], [stepSize, stepSize]]) {
          const proposed = { ...center, x: center.x + dx, y: center.y + dy };
          const trial = yield* evaluate(proposed);
          if (trial && (!best || trial.score > best.score)) { best = trial; estimate = proposed; }
        }
      }
    }
    if (!best) continue;
    initialized = true;
    if (coarseStep > 0) {
      for (const stepSize of [target.scale * 2, target.scale]) {
        const center = estimate;
        for (const [dx, dy] of [[-stepSize, -stepSize], [0, -stepSize], [stepSize, -stepSize],
          [-stepSize, 0], [stepSize, 0], [-stepSize, stepSize], [0, stepSize], [stepSize, stepSize]]) {
          const proposed = { ...center, x: center.x + dx, y: center.y + dy };
          const trial = yield* evaluate(proposed);
          if (trial && trial.score > best.score) { best = trial; estimate = proposed; }
        }
      }
    }
    for (let iteration = 0; iteration < 16; iteration++) {
      const { normal, gradient } = yield { type: 'normal', pose: estimate, best, lever, cpu: () => {
      const normal = Matrix.zeros(3, 3); const gradient = Matrix.zeros(3, 1);
      for (const pair of best.pairs) {
        const left = pixel(target, pair.px - 1, pair.py); const right = pixel(target, pair.px + 1, pair.py);
        const top = pixel(target, pair.px, pair.py - 1); const bottom = pixel(target, pair.px, pair.py + 1);
        if ([left, right, top, bottom].some(value => value === null)) continue;
        const gx = (right - left) / (2 * target.scale); const gy = (bottom - top) / (2 * target.scale);
        const jacobian = [gx, gy, (-gx * pair.rotatedY + gy * pair.rotatedX) / lever];
        const residual = pair.valueSecond - (pair.gain ?? best.gain) * pair.value - (pair.bias ?? best.bias);
        const weight = Math.min(1, 15 / Math.max(1, Math.abs(residual)));
        for (let axis = 0; axis < 3; axis++) {
          gradient.set(axis, 0, gradient.get(axis, 0) - weight * jacobian[axis] * residual);
          for (let other = 0; other < 3; other++) normal.set(axis, other, normal.get(axis, other) + weight * jacobian[axis] * jacobian[other]);
        }
      }
      return { normal, gradient };
      } };
      for (let axis = 0; axis < 3; axis++) normal.set(axis, axis, normal.get(axis, axis) + 1e-4);
      const step = solve(normal, gradient, true).to1DArray();
      if (!step.every(Number.isFinite)) break;
      const magnitude = Math.hypot(step[0], step[1], step[2]);
      if (magnitude < 0.02) break;
      let improved = false;
      for (const fraction of [1, 0.5, 0.25, 0.125]) {
        const factor = fraction * Math.min(1, 3 * target.scale / magnitude);
        const proposed = { x: estimate.x + step[0] * factor, y: estimate.y + step[1] * factor, angle: estimate.angle + step[2] * factor / lever };
        const trial = yield* evaluate(proposed);
        if (trial && trial.score > best.score + 1e-7) { best = trial; estimate = proposed; improved = true; break; }
      }
      if (!improved) break;
    }
    finalScore = best.score; support = best.count ?? best.pairs.length; finalEvaluate = evaluate;
  }
  if (!finalEvaluate) return { accepted: false,
    reason: hadEnoughSamples ? 'Unzureichende Struktur oder Ueberlappung' : 'Zu kleine maskierte Schnittmenge', evaluated };
  const boundary = Math.hypot(estimate.x - initial.x, estimate.y - initial.y) >= radius * 0.9 || Math.abs(wrap(estimate.angle - initial.angle)) >= angleRadius * 0.9;
  const competitors = [];
  for (const [dx, dy] of [[4, 0], [-4, 0], [0, 4], [0, -4]]) {
    competitors.push((yield* finalEvaluate({ ...estimate, x: estimate.x + dx, y: estimate.y + dy }))?.score ?? -1);
  }
  const margin = finalScore - Math.max(...competitors);
  const translation = rotate(estimate.x, estimate.y, referencePose.rotation);
  const pose = { x: referencePose.x + translation.x, y: referencePose.y + translation.y,
    rotation: prediction.rotation + wrap(referencePose.rotation + estimate.angle - prediction.rotation) };
  return { accepted: finalScore >= 0.9 && margin >= 0.002 && !boundary, pose, score: finalScore, margin, support, evaluated,
    reason: boundary ? 'Suchgrenze' : finalScore < 0.9 ? 'Korrelation' : margin < 0.002 ? 'Mehrdeutig' : '' };
}

export function registerOverlap(...args) {
  const steps = overlapSteps(...args);
  let step = steps.next();
  while (!step.done) step = steps.next(step.value.cpu?.());
  return step.value;
}

export async function registerOverlapAsync(backend, ...args) {
  const steps = overlapSteps(...args);
  let step = steps.next();
  while (!step.done) step = steps.next(await backend.execute(step.value));
  return step.value;
}

export function chooseRegistrationBackend(cpu, gpu, cpuMs, gpuMs, width, height) {
  const difference = cpu.accepted && gpu.accepted ? Math.hypot(cpu.pose.x - gpu.pose.x, cpu.pose.y - gpu.pose.y) +
    Math.abs(wrap(cpu.pose.rotation - gpu.pose.rotation)) * Math.hypot(width, height) / 2 : 0;
  if (cpu.accepted !== gpu.accepted || !Number.isFinite(difference) || difference > 0.1) return { backend: 'CPU', reason: 'GPU-Paarmessung weicht von CPU ab' };
  if (gpuMs >= cpuMs) return { backend: 'CPU', reason: `CPU-Paarmessung schneller (${cpuMs.toFixed(1)} / GPU ${gpuMs.toFixed(1)} ms)` };
  return { backend: 'WebGPU', reason: '' };
}

export function consensusPose(prediction, matches, width, height, tolerance = 2) {
  const usable = matches.filter(match => match.accepted && [match.pose.x, match.pose.y, match.pose.rotation].every(Number.isFinite));
  const distance = (first, second) => Math.hypot(first.x - second.x, first.y - second.y) +
    Math.abs(wrap(first.rotation - second.rotation)) * Math.hypot(width, height) / 2;
  const groups = usable.map(match => usable.filter(other => distance(match.pose, other.pose) <= tolerance));
  groups.sort((first, second) => second.length - first.length);
  const group = groups[0] ?? [];
  if (group.length < 2 || group.length <= usable.length / 2) return { pose: prediction, applied: false, inliers: [] };
  const pose = { ...prediction, x: median(group.map(match => match.pose.x)), y: median(group.map(match => match.pose.y)),
    rotation: prediction.rotation + median(group.map(match => wrap(match.pose.rotation - prediction.rotation))) };
  return { pose, applied: true, inliers: group.map(match => match.frame) };
}

function referenceGroups(matches, maximumGap = 8) {
  const sorted = [...matches].sort((first, second) => first.frame - second.frame);
  const groups = [];
  for (const match of sorted) {
    if (!groups.length || match.frame - groups.at(-1).at(-1).frame > maximumGap) groups.push([]);
    groups.at(-1).push(match);
  }
  return groups;
}

export function spatialConsensusPose(prediction, matches, width, height, tolerance = CYCLE_CONDITIONAL_PX) {
  const usable = matches.filter(match => match.kind === 'spatial' && (match.accepted || match.conditionallyAccepted) &&
    [match.pose?.x, match.pose?.y, match.pose?.rotation].every(Number.isFinite));
  const distance = (first, second) => Math.hypot(first.x - second.x, first.y - second.y) +
    Math.abs(wrap(first.rotation - second.rotation)) * Math.hypot(width, height) / 2;
  const neighborhoods = usable.map(match => usable.filter(other => distance(match.pose, other.pose) <= tolerance));
  neighborhoods.sort((first, second) => second.length - first.length);
  const inliers = neighborhoods[0] ?? [];
  if (inliers.length < 2 || inliers.length <= usable.length / 2) {
    return { pose: prediction, supported: false, inliers: [], referenceGroups: [] };
  }
  const pose = { ...prediction, x: median(inliers.map(match => match.pose.x)), y: median(inliers.map(match => match.pose.y)),
    rotation: prediction.rotation + median(inliers.map(match => wrap(match.pose.rotation - prediction.rotation))) };
  return { pose, supported: true, inliers: inliers.map(match => match.frame),
    referenceGroups: referenceGroups(inliers).map(group => group.map(match => match.frame)) };
}

export function confirmSpatialClosure(previous, frame, prediction, spatial, width, height, tolerance = CYCLE_CONDITIONAL_PX) {
  if (!spatial.supported) return { confirmed: false, evidence: null, diagnostics: null };
  const correction = { x: spatial.pose.x - prediction.x, y: spatial.pose.y - prediction.y,
    rotation: wrap(spatial.pose.rotation - prediction.rotation) };
  const correctionDistance = previous ? Math.hypot(correction.x - previous.correction.x, correction.y - previous.correction.y) +
    Math.abs(wrap(correction.rotation - previous.correction.rotation)) * Math.hypot(width, height) / 2 : Infinity;
  const independentGroups = spatial.referenceGroups.length >= 2;
  const relatedReferences = Boolean(previous && previous.referenceGroups.some(previousGroup =>
    spatial.referenceGroups.some(group => previousGroup.some(first => group.some(second => Math.abs(first - second) <= 8)))));
  const consecutiveFrames = Boolean(previous && previous.frame < frame && relatedReferences && correctionDistance <= tolerance);
  return { confirmed: independentGroups || consecutiveFrames,
    evidence: { frame, correction, inliers: spatial.inliers, referenceGroups: spatial.referenceGroups },
    diagnostics: { supported: true, independentGroups, consecutiveFrames, relatedReferences, correctionDistance,
      referenceGroups: spatial.referenceGroups, previousFrame: previous?.frame ?? null } };
}

export function contextSearchRadii(initial, width, height, kind, maximumFraction = 0.8) {
  const radii = [initial];
  if (kind !== 'spatial') return radii;
  const maximum = Math.max(initial, Math.ceil(Math.hypot(width, height) * maximumFraction));
  while (radii.at(-1) < maximum) radii.push(Math.min(maximum, radii.at(-1) * 2));
  return radii;
}

function descriptor(image, frame, pose) {
  const sharpness = new Float32Array(256); const counts = new Uint32Array(256); const allowed = new Uint32Array(256);
  const source = image.levels[0];
  const stride = Math.max(1, Math.floor(Math.min(image.width, image.height) / 128));
  for (let row = 2; row < image.height - 2; row += stride) for (let column = 2; column < image.width - 2; column += stride) {
    const tile = Math.floor(row * 16 / image.height) * 16 + Math.floor(column * 16 / image.width);
    counts[tile]++;
    const offset = row * image.width + column;
    const values = [source.gray[offset], source.gray[offset - 1], source.gray[offset + 1], source.gray[offset - image.width], source.gray[offset + image.width]];
    if (values.some(value => value < 0)) continue;
    allowed[tile]++;
    sharpness[tile] += (values[1] + values[2] + values[3] + values[4] - 4 * values[0]) ** 2;
  }
  const coverage = new Float32Array(256);
  for (let tile = 0; tile < 256; tile++) {
    coverage[tile] = allowed[tile] / Math.max(1, counts[tile]);
    sharpness[tile] /= Math.max(1, allowed[tile]);
  }
  return { frame, pose: { ...pose }, width: image.width, height: image.height, sharpness, coverage };
}

export function selectContextReferences(current, history, recentCount, spatialCount) {
  const minimumOverlapCells = Math.ceil(256 * 0.2);
  const candidateFor = reference => {
    const transform = relative(current.pose, reference.pose); const cells = []; let sharpness = 0;
    for (let tile = 0; tile < 256; tile++) {
      if (current.coverage[tile] < 0.5) continue;
      const x = ((tile % 16 + 0.5) / 16 - 0.5) * current.width;
      const y = ((Math.floor(tile / 16) + 0.5) / 16 - 0.5) * current.height;
      const rotated = rotate(x, y, transform.angle);
      const px = rotated.x + transform.x + reference.width / 2; const py = rotated.y + transform.y + reference.height / 2;
      if (px < 0 || py < 0 || px >= reference.width || py >= reference.height) continue;
      const other = Math.floor(py * 16 / reference.height) * 16 + Math.floor(px * 16 / reference.width);
      if (reference.coverage[other] < 0.5) continue;
      cells.push(tile); sharpness += Math.sqrt(reference.sharpness[other]);
    }
    return { reference, cells, quality: cells.length ? sharpness / cells.length * Math.sqrt(cells.length / 256) : 0 };
  };
  const recent = new Set((recentCount > 0 ? history.slice(-recentCount) : []).map(frame => frame.frame));
  const candidates = history.map(candidateFor).filter(candidate => candidate.cells.length >= minimumOverlapCells && candidate.quality > 0);
  const selected = candidates.filter(candidate => recent.has(candidate.reference.frame)).map(candidate => ({ ...candidate, kind: 'recent' }));
  const spatial = candidates.filter(candidate => !recent.has(candidate.reference.frame))
    .sort((first, second) => Math.hypot(current.pose.x - second.reference.pose.x, current.pose.y - second.reference.pose.y) -
      Math.hypot(current.pose.x - first.reference.pose.x, current.pose.y - first.reference.pose.y) || second.quality - first.quality)
    .slice(0, spatialCount);
  return [...selected, ...spatial.map(candidate => ({ ...candidate, kind: 'spatial' }))];
}

export class ContextTracker {
  constructor(cacheLimit = null, gpu = new WebGpuContextTracker()) {
    this.fixedCacheLimit = cacheLimit; this.gpu = gpu; this.reset();
  }
  reset() {
    this.cacheLimit = this.fixedCacheLimit ?? 192 * 1024 * 1024;
    this.gpu.reset(); this.registrationChoice = null; this.history = []; this.cache = new Map(); this.cacheBytes = 0; this.pending = null;
    this.closureEvidence = null;
  }

  sizeCache(bytes, options) {
    if (this.fixedCacheLimit !== null) return;
    this.cacheLimit = Math.min(768 * 1024 * 1024, Math.max(192 * 1024 * 1024,
      bytes * (1 + options.contextRecent + options.contextSpatial)));
    while (this.cacheBytes > this.cacheLimit) {
      const oldest = this.cache.keys().next().value;
      this.cacheBytes -= this.cache.get(oldest).bytes; this.cache.delete(oldest);
    }
  }

  async image(image, options) {
    let failedGpuMs = 0;
    if (options.useWebGpu && !this.gpu.failure && this.registrationChoice?.backend !== 'CPU') {
      const started = performance.now();
      try { return { ...await this.gpu.image(image, options.imageMask), accelerator: 'WebGPU' }; }
      catch (error) { failedGpuMs = performance.now() - started; this.gpu.disable(error); }
    }
    const started = performance.now();
    let rgba = image;
    if (image.frame) {
      const orientation = image.orientation;
      const canvas = new OffscreenCanvas(orientation.width, orientation.height);
      const context = canvas.getContext('2d', { willReadFrequently: true });
      context.setTransform(orientation.a, orientation.b, orientation.c, orientation.d, orientation.translateX, orientation.translateY);
      context.drawImage(image.frame, 0, 0);
      rgba = remapRGBA(context.getImageData(0, 0, canvas.width, canvas.height), image.maps);
    }
    const result = contextImage(rgba, options.imageMask);
    return { ...result, rgba: image.readRgba ? rgba : undefined, accelerator: 'CPU',
      pyramidTiming: { cpuImages: 1, gpuImages: 0, cpuMs: performance.now() - started, failedGpuMs } };
  }

  async register(current, reference, prediction, referencePose, limits, useWebGpu) {
    if (useWebGpu && !this.gpu.failure && this.registrationChoice?.backend !== 'CPU') {
      try {
        const started = performance.now();
        const accelerated = await registerOverlapAsync(this.gpu, current, reference, prediction, referencePose, limits);
        const gpuMs = performance.now() - started;
        if (!this.registrationChoice) {
          const cpuStarted = performance.now();
          const cpu = registerOverlap(current, reference, prediction, referencePose, limits);
          const cpuMs = performance.now() - cpuStarted;
          this.registrationChoice = { ...chooseRegistrationBackend(cpu, accelerated, cpuMs, gpuMs, current.width, current.height), cpuMs, gpuMs };
          if (this.registrationChoice.backend === 'CPU') { this.gpu.retainImages = false; this.gpu.release(); return { ...cpu, accelerator: 'CPU' }; }
        }
        return { ...accelerated, accelerator: 'WebGPU' };
      }
      catch (error) { this.gpu.disable(error); }
    }
    return { ...registerOverlap(current, reference, prediction, referencePose, limits), accelerator: 'CPU' };
  }

  remember(frame, image) {
    const previous = this.cache.get(frame);
    if (previous) { this.cache.delete(frame); this.cacheBytes -= previous.bytes; }
    if (image.bytes > this.cacheLimit) return;
    while (this.cacheBytes + image.bytes > this.cacheLimit) {
      const oldest = this.cache.keys().next().value;
      this.cacheBytes -= this.cache.get(oldest).bytes; this.cache.delete(oldest);
      if (this.pending) this.pending.cacheEvictions++;
    }
    this.cache.set(frame, image); this.cacheBytes += image.bytes;
  }

  async begin(image, frame, prediction, options, prepared = null) {
    if (this.pending) throw new Error('Umfeldregistrierung noch nicht abgeschlossen.');
    if (![options.contextRecent, options.contextSpatial].every(value => Number.isInteger(value) && value >= 0 && value <= 8) ||
      !Number.isFinite(options.contextRadius) || options.contextRadius < 4 || options.contextRadius > 256 ||
      !Number.isFinite(options.contextAngle) || options.contextAngle < 0.1 || options.contextAngle > 5 ||
      !Number.isFinite(options.contextCycleStrict ?? CYCLE_STRICT_PX) || (options.contextCycleStrict ?? CYCLE_STRICT_PX) <= 0 ||
      !Number.isFinite(options.contextCycleConditional ?? CYCLE_CONDITIONAL_PX) ||
      (options.contextCycleConditional ?? CYCLE_CONDITIONAL_PX) < (options.contextCycleStrict ?? CYCLE_STRICT_PX) ||
      (options.contextCycleConditional ?? CYCLE_CONDITIONAL_PX) > 50) throw new Error('Ungueltige Umfeldparameter.');
    const started = performance.now();
    const current = prepared ?? await this.image(image, options);
    this.sizeCache(current.bytes, options);
    const pyramidMs = performance.now() - started + (prepared?.preparationMs || 0);
    const metadata = descriptor(current, frame, prediction);
    const selected = selectContextReferences(metadata, this.history, options.contextRecent, options.contextSpatial);
    this.pending = { current, metadata, prediction, options, selected, matches: [], milliseconds: performance.now() - started + (prepared?.preparationMs || 0),
      pyramidMs, pyramidProfile: { ...current.pyramidTiming }, registrationMs: 0, cacheHits: 0, cacheMisses: 0, cacheEvictions: 0,
      accelerators: new Set([current.accelerator]) };
    const needed = [];
    for (const candidate of selected) {
      const cached = this.cache.get(candidate.reference.frame);
      if (cached) { this.pending.cacheHits++; this.remember(candidate.reference.frame, cached); await this.match(candidate, cached); }
      else { this.pending.cacheMisses++; needed.push(candidate.reference.frame); }
    }
    return needed;
  }

  async match(candidate, reference) {
    const started = performance.now();
    const { current, prediction, options } = this.pending;
    const limits = { radius: options.contextRadius, angle: options.contextAngle };
    const attempts = [];
    let backward = null;
    let result;
    try {
      for (const radius of contextSearchRadii(limits.radius, current.width, current.height, candidate.kind)) {
        result = await this.register(current, reference, prediction, candidate.reference.pose,
          { ...limits, radius, coarseStep: candidate.kind === 'spatial' && radius > limits.radius ? limits.radius : 0 }, options.useWebGpu);
        result.searchRadius = radius;
        attempts.push({ ...structuredClone(result), angle: limits.angle,
          coarseStep: candidate.kind === 'spatial' && radius > limits.radius ? limits.radius : 0 });
        const expandable = result.reason === 'Suchgrenze' || result.reason === 'Korrelation';
        if (result.accepted || candidate.kind !== 'spatial' || !expandable) break;
      }
      this.pending.accelerators.add(result.accelerator);
      if (result.accepted) {
        const reverse = await this.register(reference, current, candidate.reference.pose, result.pose, limits, options.useWebGpu);
        backward = { ...structuredClone(reverse), searchRadius: limits.radius, angle: limits.angle };
        this.pending.accelerators.add(reverse.accelerator);
        result.reverseAccelerator = reverse.accelerator;
        const distance = reverse.accepted ? Math.hypot(reverse.pose.x - candidate.reference.pose.x, reverse.pose.y - candidate.reference.pose.y) +
          Math.abs(wrap(reverse.pose.rotation - candidate.reference.pose.rotation)) * Math.hypot(current.width, current.height) / 2 : Infinity;
        result.reverseDistance = distance;
        if (reverse.accepted) {
          const points = overlapReferencePoints(candidate.cells, result.pose, candidate.reference.pose, current.width, current.height);
          result.cycleError = cycleErrorMetrics(candidate.reference.pose, reverse.pose, points);
        }
        const strictLimit = options.contextCycleStrict ?? CYCLE_STRICT_PX;
        const conditionalLimit = options.contextCycleConditional ?? CYCLE_CONDITIONAL_PX;
        result.cycleLimits = { strict: strictLimit, conditional: conditionalLimit };
        result.cycleQuality = distance <= strictLimit ? 'strict' :
          candidate.kind === 'spatial' && distance <= conditionalLimit ? 'conditional' : 'rejected';
        if (distance > strictLimit) {
          result.accepted = false;
          result.conditionallyAccepted = result.cycleQuality === 'conditional';
          result.reason = result.conditionallyAccepted ? 'Rueckwaertspruefung bedingt' : 'Rueckwaertspruefung';
        }
      }
    } catch (error) { result = { accepted: false, reason: error.message }; }
    this.pending.matches.push({ ...result, frame: candidate.reference.frame, kind: candidate.kind, overlap: candidate.cells.length / 256,
      prediction: { ...prediction }, referencePose: { ...candidate.reference.pose }, attempts, backward,
      milliseconds: performance.now() - started });
    const milliseconds = performance.now() - started;
    this.pending.milliseconds += milliseconds; this.pending.registrationMs += milliseconds;
  }

  async provide(frame, image, error = null) {
    const candidate = this.pending?.selected.find(item => item.reference.frame === frame);
    if (!candidate || this.pending.matches.some(match => match.frame === frame)) throw new Error('Unerwarteter Umfeldframe.');
    if (!error && (!image || image.width !== this.pending.current.width || image.height !== this.pending.current.height)) error = 'Umfeldframe hat andere Abmessungen.';
    if (error) { this.pending.matches.push({ frame, kind: candidate.kind, accepted: false, reason: error,
      prediction: { ...this.pending.prediction }, referencePose: { ...candidate.reference.pose }, attempts: [], backward: null }); return; }
    const started = performance.now();
    const reference = await this.image(image, this.pending.options);
    for (const [key, value] of Object.entries(reference.pyramidTiming ?? {})) this.pending.pyramidProfile[key] = (this.pending.pyramidProfile[key] || 0) + value;
    const milliseconds = performance.now() - started;
    this.pending.milliseconds += milliseconds; this.pending.pyramidMs += milliseconds;
    this.pending.accelerators.add(reference.accelerator);
    await this.match(candidate, reference); this.remember(frame, reference);
  }

  finish() {
    const pending = this.pending;
    if (!pending || pending.matches.length !== pending.selected.length) throw new Error('Umfeldreferenzen fehlen.');
    const started = performance.now();
    let result = consensusPose(pending.prediction, pending.matches, pending.current.width, pending.current.height);
    const conditionalLimit = pending.options.contextCycleConditional ?? CYCLE_CONDITIONAL_PX;
    const spatial = spatialConsensusPose(pending.prediction, pending.matches, pending.current.width, pending.current.height, conditionalLimit);
    let spatialConfirmation = null;
    if (spatial.supported) {
      const confirmation = confirmSpatialClosure(this.closureEvidence, pending.metadata.frame, pending.prediction, spatial,
        pending.current.width, pending.current.height, conditionalLimit);
      spatialConfirmation = confirmation.diagnostics;
      if (confirmation.confirmed) result = { pose: spatial.pose, applied: true, inliers: spatial.inliers,
        confirmation: spatialConfirmation.independentGroups ? 'independent-reference-groups' : 'consecutive-frames' };
      this.closureEvidence = confirmation.evidence;
    } else this.closureEvidence = null;
    const spatialInliers = pending.matches.filter(match => match.kind === 'spatial' && result.inliers.includes(match.frame));
    if (result.applied && spatialInliers.length) {
      result.loopClosure = { anchorFrame: Math.max(...spatialInliers.map(match => match.frame)), frame: pending.metadata.frame,
        dx: result.pose.x - pending.prediction.x, dy: result.pose.y - pending.prediction.y,
        rotation: wrap(result.pose.rotation - pending.prediction.rotation) };
      applyLoopClosure(this.history, result.loopClosure);
    }
    pending.metadata.pose = { ...result.pose };
    this.history.push(pending.metadata); this.remember(pending.metadata.frame, pending.current);
    this.pending = null;
    return { ...result, spatialConfirmation, matches: pending.matches, selected: pending.selected.length, milliseconds: pending.milliseconds + performance.now() - started,
      pyramidMs: pending.pyramidMs, registrationMs: pending.registrationMs, cacheHits: pending.cacheHits, cacheMisses: pending.cacheMisses,
      pyramidProfile: pending.pyramidProfile, cacheEvictions: pending.cacheEvictions, cacheEntries: this.cache.size, cacheLimit: this.cacheLimit,
      pyramidBytes: pending.current.bytes, imageWidth: pending.current.width, imageHeight: pending.current.height,
      accelerator: [...pending.accelerators].sort().join(' + '), registrationChoice: this.registrationChoice,
      fallback: pending.options.useWebGpu ? this.gpu.failure || this.registrationChoice?.reason || '' : '', gpuCacheBytes: this.gpu.cacheBytes,
      historyFrames: this.history.length, cacheBytes: this.cacheBytes };
  }
}