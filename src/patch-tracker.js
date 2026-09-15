import { MASK_SEARCH, patchAllowed } from './patch-mask.js';
import { WebGpuPatchLocator } from './webgpu-patch-tracker.js';

const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));

function grayscale(image) {
  const gray = new Float32Array(image.width * image.height);
  for (let index = 0; index < gray.length; index++) gray[index] = image.data.length === gray.length ? image.data[index] :
    0.299 * image.data[4 * index] + 0.587 * image.data[4 * index + 1] + 0.114 * image.data[4 * index + 2];
  return gray;
}

function structureScore(gray, width, px, py, size) {
  const radius = Math.floor(size / 2);
  const stride = Math.max(1, Math.floor(size / 16));
  const aperture = Math.max(2, Math.floor(size / 16));
  let xx = 0; let xy = 0; let yy = 0; let count = 0;
  for (let offsetY = -radius; offsetY <= radius; offsetY += stride) {
    for (let offsetX = -radius; offsetX <= radius; offsetX += stride) {
      const centerX = px + offsetX;
      const centerY = py + offsetY;
      let gx = 0; let gy = 0;
      for (let across = -aperture; across <= aperture; across++) {
        gx += gray[(centerY + across) * width + centerX + aperture] - gray[(centerY + across) * width + centerX - aperture];
        gy += gray[(centerY + aperture) * width + centerX + across] - gray[(centerY - aperture) * width + centerX + across];
      }
      gx /= 2 * aperture + 1;
      gy /= 2 * aperture + 1;
      xx += gx * gx; xy += gx * gy; yy += gy * gy; count++;
    }
  }
  const trace = xx + yy;
  const root = Math.sqrt(Math.max(0, (xx - yy) ** 2 + 4 * xy ** 2));
  const minimum = (trace - root) / 2 / count;
  const maximum = (trace + root) / 2 / count;
  return { score: minimum, dimensionality: minimum / Math.max(maximum, 1e-9) };
}

function correlation(previous, current, width, height, fromX, fromY, toX, toY, size, sourceWidth = width) {
  const radius = Math.floor(size / 2) - 1;
  const stride = Math.max(1, Math.floor(size / 12));
  if (toX - radius < 1 || toY - radius < 1 || toX + radius >= width - 1 || toY + radius >= height - 1) return -1;
  let sumA = 0; let sumB = 0; let sumAA = 0; let sumBB = 0; let sumAB = 0; let count = 0;
  for (let offsetY = -radius; offsetY <= radius; offsetY += stride) {
    for (let offsetX = -radius; offsetX <= radius; offsetX += stride) {
      const first = previous[Math.round(fromY + offsetY) * sourceWidth + Math.round(fromX + offsetX)];
      const second = current[Math.round(toY + offsetY) * width + Math.round(toX + offsetX)];
      sumA += first; sumB += second; sumAA += first * first; sumBB += second * second; sumAB += first * second; count++;
    }
  }
  const covariance = sumAB - sumA * sumB / count;
  const variance = (sumAA - sumA ** 2 / count) * (sumBB - sumB ** 2 / count);
  return variance > 1e-6 ? covariance / Math.sqrt(variance) : -1;
}

function locate(previous, current, width, height, feature, size, searchRadius, reference = feature) {
  const scoreAt = (px, py) => correlation(previous, current, width, height,
    reference.x, reference.y, px, py, size, reference.width ?? width);
  let centerX = feature.x;
  let centerY = feature.y;
  let step = Math.max(2, Math.ceil(searchRadius / 6));
  let range = searchRadius;
  let bestScore = -1;
  while (step >= 1) {
    let bestX = centerX;
    let bestY = centerY;
    bestScore = -1;
    for (let offsetY = -range; offsetY <= range; offsetY += step) {
      for (let offsetX = -range; offsetX <= range; offsetX += step) {
        const candidateX = Math.round(centerX + offsetX);
        const candidateY = Math.round(centerY + offsetY);
        const score = scoreAt(candidateX, candidateY);
        if (score > bestScore) { bestScore = score; bestX = candidateX; bestY = candidateY; }
      }
    }
    centerX = bestX; centerY = bestY;
    if (step === 1) break;
    range = step;
    step = Math.max(1, Math.floor(step / 2));
  }
  if (bestScore < 0.72) return null;
  const left = scoreAt(centerX - 1, centerY);
  const right = scoreAt(centerX + 1, centerY);
  const top = scoreAt(centerX, centerY - 1);
  const bottom = scoreAt(centerX, centerY + 1);
  const refine = (low, middle, high) => {
    const curvature = low - 2 * middle + high;
    return Math.abs(curvature) > 1e-6 ? clamp(0.5 * (low - high) / curvature, -0.75, 0.75) : 0;
  };
  return { x: centerX + refine(left, bestScore, right), y: centerY + refine(top, bestScore, bottom), score: bestScore };
}

function rigidPose(points, sourceX, sourceY, targetX, targetY) {
  if (points.length < 2) return null;
  const mean = axis => points.reduce((sum, point) => sum + axis(point), 0) / points.length;
  const fromX = mean(sourceX); const fromY = mean(sourceY);
  const toX = mean(targetX); const toY = mean(targetY);
  let cosineSum = 0; let sineSum = 0;
  for (const point of points) {
    const qx = sourceX(point) - fromX; const qy = sourceY(point) - fromY;
    const px = targetX(point) - toX; const py = targetY(point) - toY;
    cosineSum += qx * px + qy * py;
    sineSum += qx * py - qy * px;
  }
  const theta = Math.atan2(sineSum, cosineSum);
  const cosine = Math.cos(theta); const sine = Math.sin(theta);
  return { cosine, sine, tx: toX - cosine * fromX + sine * fromY, ty: toY - sine * fromX - cosine * fromY };
}

function inversePose(pose, px, py) {
  const x = px - pose.tx; const y = py - pose.ty;
  return { x: pose.cosine * x + pose.sine * y, y: -pose.sine * x + pose.cosine * y };
}

function median(values) {
  const ordered = [...values].sort((first, second) => first - second);
  return ordered[Math.floor(ordered.length / 2)] ?? 0;
}

function selectCandidates(gray, width, height, size, threshold, occupied, limit, mask) {
  const radius = Math.floor(size / 2) + Math.max(2, Math.floor(size / 16)) + 1;
  const spacing = Math.max(12, Math.round(size * 0.75));
  const candidates = [];
  for (let py = radius; py < height - radius; py += Math.max(4, Math.floor(spacing / 2))) {
    for (let px = radius; px < width - radius; px += Math.max(4, Math.floor(spacing / 2))) {
      if (!patchAllowed(mask, px, py, size, true)) continue;
      if (occupied.some(feature => Math.hypot(feature.x - px, feature.y - py) < spacing)) continue;
      const quality = structureScore(gray, width, px, py, size);
      if (quality.score >= threshold ** 2 * 0.35 && quality.dimensionality >= 0.08) candidates.push({ x: px, y: py, ...quality });
    }
  }
  candidates.sort((first, second) => second.score - first.score);
  const selected = [];
  for (const candidate of candidates) {
    if (selected.some(feature => Math.hypot(feature.x - candidate.x, feature.y - candidate.y) < spacing)) continue;
    selected.push(candidate);
    if (selected.length >= limit) break;
  }
  return selected;
}

export class PatchTracker {
  reset() {
    this.gpuLocator?.reset();
    this.gpuLocator = new WebGpuPatchLocator();
    this.previous = null;
    this.features = [];
    this.index = null;
    this.signature = null;
    this.generation = 0;
    this.patchArchive = new Map();
    this.archiveBytes = 0;
    this.nextPatchId = 0;
  }

  rememberPatches(current, width, height, size, index) {
    const tileSize = size + 6;
    const half = Math.floor(tileSize / 2);
    for (const feature of this.features) {
      feature.id ??= this.nextPatchId++;
      const known = this.patchArchive.get(feature.id);
      if (known) { known.lastSeen = index; continue; }
      const left = Math.floor(feature.x) - half;
      const top = Math.floor(feature.y) - half;
      if (left < 0 || top < 0 || left + tileSize > width || top + tileSize > height) continue;
      const pixels = new Float32Array(tileSize * tileSize);
      for (let row = 0; row < tileSize; row++) pixels.set(current.subarray((top + row) * width + left,
        (top + row) * width + left + tileSize), row * tileSize);
      this.patchArchive.set(feature.id, { id: feature.id, col: feature.col, row: feature.row, pixels,
        x: feature.x - left, y: feature.y - top, width: tileSize, lastSeen: index, lastAttempt: -1 });
      this.archiveBytes += pixels.byteLength;
      while (this.patchArchive.size > 2000 || this.archiveBytes > 32 * 1024 * 1024) {
        const oldest = this.patchArchive.keys().next().value;
        this.archiveBytes -= this.patchArchive.get(oldest).pixels.byteLength;
        this.patchArchive.delete(oldest);
      }
    }
  }

  rediscoverPatches(current, width, height, size, searchRadius, mask, index) {
    if (this.features.length < 3) return 0;
    const pose = rigidPose(this.features, point => point.col, point => point.row, point => point.x, point => point.y);
    const active = new Set(this.features.map(point => point.id));
    const margin = Math.floor(size / 2) + 2;
    const candidates = [...this.patchArchive.values()].filter(entry => !active.has(entry.id) && entry.lastSeen < index - 1)
      .map(entry => ({ entry, x: pose.cosine * entry.col - pose.sine * entry.row + pose.tx,
        y: pose.sine * entry.col + pose.cosine * entry.row + pose.ty }))
      .filter(point => point.x >= margin && point.y >= margin && point.x < width - margin && point.y < height - margin &&
        patchAllowed(mask, point.x, point.y, size) && !this.features.some(activePoint => activePoint.id < point.entry.id &&
          Math.hypot(activePoint.x - point.x, activePoint.y - point.y) < size * 0.5))
      .sort((first, second) => first.entry.lastAttempt - second.entry.lastAttempt || first.entry.id - second.entry.id)
      .slice(0, 48);
    const matches = [];
    for (const predicted of candidates) {
      const entry = predicted.entry;
      entry.lastAttempt = index;
      const found = locate(entry.pixels, current, width, height, predicted, size, searchRadius, entry);
      if (!found || found.score < 0.9 || Math.hypot(found.x - predicted.x, found.y - predicted.y) > searchRadius ||
        !patchAllowed(mask, found.x, found.y, size)) continue;
      let competing = -1;
      for (let offsetY = -searchRadius; offsetY <= searchRadius; offsetY += 3) {
        for (let offsetX = -searchRadius; offsetX <= searchRadius; offsetX += 3) {
          const candidateX = Math.round(predicted.x + offsetX);
          const candidateY = Math.round(predicted.y + offsetY);
          if (Math.hypot(candidateX - found.x, candidateY - found.y) < 4) continue;
          competing = Math.max(competing, correlation(entry.pixels, current, width, height,
            entry.x, entry.y, candidateX, candidateY, size, entry.width));
        }
      }
      if (found.score - competing < 0.04) continue;
      const backward = locate(current, entry.pixels, entry.width, entry.width, entry, size, 2, { ...found, width });
      if (!backward || backward.score < 0.9 || Math.hypot(backward.x - entry.x, backward.y - entry.y) > 1.5) continue;
      if (matches.some(point => Math.hypot(point.x - found.x, point.y - found.y) < size * 0.5)) continue;
      matches.push({ id: entry.id, col: entry.col, row: entry.row, x: found.x, y: found.y,
        predictedX: predicted.x, predictedY: predicted.y, confidence: Math.min(found.score, backward.score) });
    }
    if (matches.length < 3) return 0;
    const deltaX = median(matches.map(point => point.x - point.predictedX));
    const deltaY = median(matches.map(point => point.y - point.predictedY));
    const consistent = matches.filter(point => Math.hypot(point.x - point.predictedX - deltaX,
      point.y - point.predictedY - deltaY) <= Math.max(2, size * 0.08));
    if (consistent.length < 3) return 0;
    this.features = this.features.filter(point => !consistent.some(match => Math.hypot(point.x - match.x, point.y - match.y) < size * 0.5));
    this.features = [...consistent.map(({ predictedX, predictedY, ...point }) => point), ...this.features].slice(0, 400);
    return consistent.length;
  }

  process(image, index, options = {}) {
    const size = clamp(Math.round(options.patchSize || 32), 16, 256);
    const searchRadius = clamp(Math.round(options.patchSearchRadius || size / 2), 4, 128);
    const mask = options.patchMask;
    const roi = { x: 0, y: 0, width: image.width, height: image.height };
    const signature = `${image.width}:${image.height}:${size}:${searchRadius}:${mask?.revision || 0}:${Boolean(options.rediscoverPatches)}`;
    const current = options.currentGray || grayscale(image);
    if (mask?.data && !mask.data.includes(MASK_SEARCH)) {
      return { points: [], lines: [], rejected: [], roi, step: 1, success: false,
        reason: 'Keine gruene Patch-Suchflaeche markiert. Mit dem gruenen Pinsel geeignete Bildbereiche auswaehlen.',
        coverage: 0, confidence: 0, patchSize: size, initialized: false, generation: this.generation || 0 };
    }
    if (this.previous && signature === this.signature && index < this.index) {
      return { points: [], lines: [], rejected: [], roi, step: 1, success: false,
        reason: 'Image-Patch-Sequenz laeuft nicht vorwaerts. Analyse fortsetzen oder Messframes zuruecksetzen.',
        coverage: 0, confidence: 0, patchSize: size, initialized: false, generation: this.generation, interrupted: true };
    }
    const initialized = !this.previous || signature !== this.signature || index > this.index + 1;
    let rediscovered = 0;
    if (initialized) {
      this.patchArchive = new Map();
      this.archiveBytes = 0;
      this.nextPatchId = 0;
      this.generation = (this.generation || 0) + 1;
      this.features = selectCandidates(current, image.width, image.height, size, options.threshold || 16, [], 400, mask)
        .map(candidate => ({ x: candidate.x, y: candidate.y, col: candidate.x, row: candidate.y,
          confidence: clamp(candidate.dimensionality * 2, 0.1, 1) }));
    } else {
      let tracked = options.trackedFeatures || this.features.map(feature => {
        const tracked = locate(this.previous, current, image.width, image.height, feature, size, searchRadius);
        if (!tracked) return null;
        const backward = locate(current, this.previous, image.width, image.height, tracked, size, searchRadius);
        if (!backward || Math.hypot(backward.x - feature.x, backward.y - feature.y) > 1.5) return null;
        return { ...feature, previousX: feature.x, previousY: feature.y, x: tracked.x, y: tracked.y,
          confidence: clamp(Math.min(tracked.score, backward.score), 0.1, 1) };
      }).filter(feature => feature && patchAllowed(mask, feature.x, feature.y, size));
      if (tracked.length >= 3) {
        const medianX = median(tracked.map(point => point.x - point.previousX));
        const medianY = median(tracked.map(point => point.y - point.previousY));
        const translationResiduals = tracked.map(point => Math.hypot(point.x - point.previousX - medianX, point.y - point.previousY - medianY));
        const translationLimit = Math.max(3, median(translationResiduals) * 3, size * 0.12);
        tracked = tracked.filter((point, local) => translationResiduals[local] <= translationLimit);
      }
      if (tracked.length >= 3) {
        const motion = rigidPose(tracked, point => point.previousX, point => point.previousY, point => point.x, point => point.y);
        const residuals = tracked.map(point => Math.hypot(
          motion.cosine * point.previousX - motion.sine * point.previousY + motion.tx - point.x,
          motion.sine * point.previousX + motion.cosine * point.previousY + motion.ty - point.y));
        const maximumResidual = Math.max(2, median(residuals) * 3, size * 0.08);
        tracked = tracked.filter((point, local) => residuals[local] <= maximumResidual);
      }
      this.features = tracked.map(({ previousX, previousY, ...feature }) => feature);
      if (options.rediscoverPatches) rediscovered = this.rediscoverPatches(current, image.width, image.height, size,
        Math.min(searchRadius, 32), mask, index);
      if (this.features.length >= 3 && this.features.length < 400) {
        const planePose = rigidPose(this.features, point => point.col, point => point.row, point => point.x, point => point.y);
        const additions = selectCandidates(current, image.width, image.height, size, options.threshold || 16, this.features, 400 - this.features.length, mask);
        for (const candidate of additions) {
          const plane = inversePose(planePose, candidate.x, candidate.y);
          const nearest = [...this.features].sort((first, second) =>
            Math.hypot(first.x - candidate.x, first.y - candidate.y) - Math.hypot(second.x - candidate.x, second.y - candidate.y)).slice(0, 8);
          let correctionX = 0; let correctionY = 0; let weightSum = 0;
          for (const feature of nearest) {
            const predicted = inversePose(planePose, feature.x, feature.y);
            const weight = 1 / Math.max(4, Math.hypot(feature.x - candidate.x, feature.y - candidate.y));
            correctionX += weight * (feature.col - predicted.x); correctionY += weight * (feature.row - predicted.y); weightSum += weight;
          }
          this.features.push({ x: candidate.x, y: candidate.y, col: plane.x + correctionX / weightSum,
            row: plane.y + correctionY / weightSum, confidence: clamp(candidate.dimensionality, 0.1, 0.5) });
        }
      }
    }
    this.features = this.features.filter(feature => patchAllowed(mask, feature.x, feature.y, size));
    if (options.rediscoverPatches) this.rememberPatches(current, image.width, image.height, size, index);
    this.previous = current;
    this.index = index;
    this.signature = signature;
    const occupied = new Set(this.features.map(point => `${Math.floor(point.x / image.width * 12)},${Math.floor(point.y / image.height * 9)}`));
    const success = this.features.length >= 6 && occupied.size >= 4;
    return { points: this.features.map(feature => ({ ...feature })), lines: [], rejected: [], roi, step: 1,
      success, reason: success ? '' : 'Zu wenige zweidimensional strukturierte Image-Patches. Patchgroesse, Kontrast oder Bildausschnitt anpassen.',
      coverage: occupied.size / 108, confidence: this.features.length ? this.features.reduce((sum, feature) => sum + feature.confidence, 0) / this.features.length : 0,
      patchSize: size, initialized, generation: this.generation, accelerator: options.trackerAccelerator || 'CPU',
      rediscovered, archivedPatches: this.patchArchive.size, archiveBytes: this.archiveBytes };
  }

  async processAsync(image, index, options = {}) {
    const started = performance.now();
    const size = clamp(Math.round(options.patchSize || 32), 16, 256);
    const searchRadius = clamp(Math.round(options.patchSearchRadius || size / 2), 4, 128);
    const signature = `${image.width}:${image.height}:${size}:${searchRadius}:${options.patchMask?.revision || 0}:${Boolean(options.rediscoverPatches)}`;
    const consecutive = this.previous && signature === this.signature && index === this.index + 1;
    if (!options.useWebGpu || !consecutive) {
      if (!consecutive) this.gpuLocator?.reset();
      const result = this.process(image, index, options);
      const elapsedMs = performance.now() - started;
      result.timing = { totalMs: elapsedMs, cpuMs: elapsedMs };
      return result;
    }
    const grayscaleStarted = performance.now();
    const current = grayscale(image);
    const grayscaleMs = performance.now() - grayscaleStarted;
    try {
      this.gpuLocator ||= new WebGpuPatchLocator();
      const gpuResult = await this.gpuLocator.track(this.previous, current, image.width, image.height, this.features, size, searchRadius);
      if (!gpuResult) return this.process(image, index, { ...options, currentGray: current });
      const trackedFeatures = gpuResult.pairs.map(pair => {
        if (!pair.backward || Math.hypot(pair.backward.x - pair.feature.x, pair.backward.y - pair.feature.y) > 1.5) return null;
        return { ...pair.feature, previousX: pair.feature.x, previousY: pair.feature.y, x: pair.tracked.x, y: pair.tracked.y,
          confidence: clamp(Math.min(pair.tracked.score, pair.backward.score), 0.1, 1) };
      }).filter(Boolean);
      const postStarted = performance.now();
      const result = this.process(image, index, { ...options, currentGray: current, trackedFeatures, trackerAccelerator: 'WebGPU' });
      result.timing = { grayscaleMs, ...gpuResult.profile, postprocessMs: performance.now() - postStarted, totalMs: performance.now() - started };
      return result;
    } catch {
      this.gpuLocator?.reset();
      const fallbackStarted = performance.now();
      const result = this.process(image, index, { ...options, currentGray: current });
      result.timing = { grayscaleMs, cpuMs: performance.now() - fallbackStarted, totalMs: performance.now() - started };
      return result;
    }
  }
}
