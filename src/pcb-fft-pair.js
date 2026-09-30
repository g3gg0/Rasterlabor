import { fftPatchShift } from './pcb-fft.js';
import { composePose } from './pcb-realignment.js';
import { maskIncludes } from './patch-mask.js';

const rotate = (x, y, angle) => ({ x: Math.cos(angle) * x - Math.sin(angle) * y,
  y: Math.sin(angle) * x + Math.cos(angle) * y });
function imageGeometry(image, pose, offset) {
  const inverseCosine = Math.cos(-pose.rotation), inverseSine = Math.sin(-pose.rotation);
  const world = (x, y) => {
    const point = rotate(x + offset[0], y + offset[1], pose.rotation);
    return { x: pose.x + point.x, y: pose.y + point.y };
  };
  const local = point => {
    const x = point.x - pose.x, y = point.y - pose.y;
    return { x: inverseCosine * x - inverseSine * y - offset[0],
      y: inverseSine * x + inverseCosine * y - offset[1] };
  };
  const corners = [[0, 0], [image.width, 0], [image.width, image.height], [0, image.height]].map(([x, y]) => world(x, y));
  return { world, local, corners, inverseCosine, inverseSine };
}
function imageBounds(corners) {
  return { minX: Math.min(...corners.map(point => point.x)), maxX: Math.max(...corners.map(point => point.x)),
    minY: Math.min(...corners.map(point => point.y)), maxY: Math.max(...corners.map(point => point.y)) };
}
function sampleGray(image, x, y, mask) {
  const left = Math.floor(x), top = Math.floor(y);
  if (left < 0 || top < 0 || left + 1 >= image.width || top + 1 >= image.height) return null;
  const fx = x - left, fy = y - top;
  const stride = image.width * 4, offset = (top * image.width + left) * 4, data = image.data;
  if (data[offset + 3] !== 255 || data[offset + 7] !== 255 ||
      data[offset + stride + 3] !== 255 || data[offset + stride + 7] !== 255 ||
      (mask && (!maskIncludes(mask, left, top) || !maskIncludes(mask, left + 1, top) ||
        !maskIncludes(mask, left, top + 1) || !maskIncludes(mask, left + 1, top + 1)))) return null;
  const gray = index => 0.299 * data[index] + 0.587 * data[index + 1] + 0.114 * data[index + 2];
  return (1 - fy) * ((1 - fx) * gray(offset) + fx * gray(offset + 4)) +
    fy * ((1 - fx) * gray(offset + stride) + fx * gray(offset + stride + 4));
}
function jointUsableBounds(first, second, firstGeometry, secondGeometry, overlap, cellSize, mask) {
  const step = Math.max(16, cellSize / 2);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let y = overlap.minY + step / 2; y < overlap.maxY; y += step)
    for (let x = overlap.minX + step / 2; x < overlap.maxX; x += step) {
      const world = { x, y };
      const firstPoint = firstGeometry.local(world), secondPoint = secondGeometry.local(world);
      if (sampleGray(first, firstPoint.x, firstPoint.y, mask) === null ||
          sampleGray(second, secondPoint.x, secondPoint.y, mask) === null) continue;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
  if (!Number.isFinite(minX)) return null;
  return { minX: Math.max(overlap.minX, minX - step / 2), maxX: Math.min(overlap.maxX, maxX + step / 2),
    minY: Math.max(overlap.minY, minY - step / 2), maxY: Math.min(overlap.maxY, maxY + step / 2) };
}
function patch(image, geometry, center, size, mask) {
  const values = new Float64Array(size * size);
  const start = geometry.local({ x: center.x - size / 2, y: center.y - size / 2 });
  const cosine = geometry.inverseCosine, sine = geometry.inverseSine;
  for (let row = 0; row < size; row++) {
    const rowX = start.x - row * sine, rowY = start.y + row * cosine;
    for (let column = 0; column < size; column++) {
      const value = sampleGray(image, rowX + column * cosine, rowY + column * sine, mask);
      if (value === null) return null;
      values[row * size + column] = value;
    }
  }
  return values;
}
function fitRigidCells(cells, cellSize, translationOnly = false) {
  let total = 0, sourceX = 0, sourceY = 0, targetX = 0, targetY = 0;
  for (const cell of cells) {
    const weight = Math.min(cell.ownedSupportPixels, cellSize * 2);
    total += weight;
    sourceX += weight * (cell.center.x + cell.dx);
    sourceY += weight * (cell.center.y + cell.dy);
    targetX += weight * cell.center.x;
    targetY += weight * cell.center.y;
  }
  if (total <= 0) return null;
  sourceX /= total; sourceY /= total; targetX /= total; targetY /= total;
  if (translationOnly) return {x:targetX-sourceX,y:targetY-sourceY,rotation:0};
  let dot = 0, cross = 0, spread = 0;
  for (const cell of cells) {
    const weight = Math.min(cell.ownedSupportPixels, cellSize * 2);
    const x = cell.center.x + cell.dx - sourceX;
    const y = cell.center.y + cell.dy - sourceY;
    const targetDx = cell.center.x - targetX, targetDy = cell.center.y - targetY;
    dot += weight * (x * targetDx + y * targetDy);
    cross += weight * (x * targetDy - y * targetDx);
    spread += weight * (targetDx * targetDx + targetDy * targetDy);
  }
  if (spread / total < cellSize ** 2 / 16) return null;
  const rotation = Math.atan2(cross, dot);
  const cosine = Math.cos(rotation), sine = Math.sin(rotation);
  return { x: targetX - cosine * sourceX + sine * sourceY,
    y: targetY - sine * sourceX - cosine * sourceY, rotation };
}
function cellResidual(cell, correction) {
  const point = rotate(cell.center.x + cell.dx, cell.center.y + cell.dy, correction.rotation);
  return Math.hypot(point.x + correction.x - cell.center.x,
    point.y + correction.y - cell.center.y);
}

export function registerPcbFftPair(first, second, firstPose, secondPose, firstOffset, secondOffset,
  { cellSize = 64, cellsPerAxis = 3, searchRadius = 16, minimumPsr = 6, residualLimit = 2,
    mask = null, adaptiveCells = false, candidateGrid: requestedGrid = null, translationOnly = false,
    excludedRegions = [], measureShift = fftPatchShift } = {}) {
  if (!Number.isInteger(cellSize) || cellSize < 16 || cellSize > 256 ||
      !Number.isInteger(cellsPerAxis) || cellsPerAxis < 2 || cellsPerAxis > 8)
    throw new Error('Ungueltige FFT-Zellparameter.');
  const firstGeometry = imageGeometry(first, firstPose, firstOffset);
  const secondGeometry = imageGeometry(second, secondPose, secondOffset);
  const a = imageBounds(firstGeometry.corners), b = imageBounds(secondGeometry.corners);
  let overlap = { minX: Math.max(a.minX, b.minX), maxX: Math.min(a.maxX, b.maxX),
    minY: Math.max(a.minY, b.minY), maxY: Math.min(a.maxY, b.maxY) };
  const half = cellSize / 2 + 2;
  if (overlap.maxX - overlap.minX < cellSize + 4 || overlap.maxY - overlap.minY < cellSize + 4)
    return { accepted: false, reason: 'Zu schmale Ueberlappung', cells: [] };
  overlap = jointUsableBounds(first, second, firstGeometry, secondGeometry, overlap, cellSize, mask);
  if (!overlap || overlap.maxX - overlap.minX < cellSize + 4 ||
      overlap.maxY - overlap.minY < cellSize + 4)
    return { accepted: false, reason: 'Zu wenig gemeinsamer Bildinhalt', cells: [] };
  const cells = [], candidates = [];
  const candidateGrid = requestedGrid === null ?
    (mask || adaptiveCells ? Math.min(12, Math.max(7, cellsPerAxis * 3)) : cellsPerAxis) : requestedGrid;
  if (!Number.isInteger(candidateGrid) || candidateGrid < 2 || candidateGrid > 32)
    throw new Error('Ungueltiges FFT-Suchraster.');
  for (let row = 0; row < candidateGrid; row++) for (let column = 0; column < candidateGrid; column++) {
    const center = { x: overlap.minX + half + column / (candidateGrid - 1) * (overlap.maxX - overlap.minX - 2 * half),
      y: overlap.minY + half + row / (candidateGrid - 1) * (overlap.maxY - overlap.minY - 2 * half) };
    if (excludedRegions.some(region => Math.hypot(center.x-region.center.x,center.y-region.center.y) <
        region.radius+cellSize/Math.SQRT2)) {
      cells.push({cellId:row*candidateGrid+column,center,accepted:false,reason:'Kreisfoermige Referenz'});
      continue;
    }
    const firstPatch = patch(first, firstGeometry, center, cellSize, mask);
    const secondPatch = patch(second, secondGeometry, center, cellSize, mask);
    if (!firstPatch || !secondPatch) { cells.push({ cellId: row * candidateGrid + column, center, accepted: false, reason: 'Maske oder Rand' }); continue; }
    const ownerBounds = {
      minX: overlap.minX + column * (overlap.maxX - overlap.minX) / candidateGrid - center.x + cellSize / 2,
      maxX: overlap.minX + (column + 1) * (overlap.maxX - overlap.minX) / candidateGrid - center.x + cellSize / 2,
      minY: overlap.minY + row * (overlap.maxY - overlap.minY) / candidateGrid - center.y + cellSize / 2,
      maxY: overlap.minY + (row + 1) * (overlap.maxY - overlap.minY) / candidateGrid - center.y + cellSize / 2
    };
    let texture = 0;
    for (let index = cellSize + 1; index < firstPatch.length; index += 4) {
      const differenceX = firstPatch[index] - firstPatch[index - 1];
      const differenceY = firstPatch[index] - firstPatch[index - cellSize];
      texture += differenceX ** 2 + differenceY ** 2;
    }
    candidates.push({ cellId: row * candidateGrid + column, center, firstPatch, secondPatch,
      ownerBounds, texture: Math.sqrt(texture / Math.max(1, firstPatch.length / 4)) });
  }
  const strongestTexture = Math.max(0, ...candidates.map(candidate => candidate.texture));
  const structural = candidates.filter(candidate =>
    candidate.texture >= Math.max(1.8, Math.min(8, strongestTexture * 0.4)));
  const selected = [], remaining = (mask || adaptiveCells) && structural.length >= 3 ? [...structural] : [...candidates];
  const maximumCells = Math.min(remaining.length, cellsPerAxis ** 2 * 2);
  while (selected.length < maximumCells) {
    let bestIndex = 0, bestScore = -Infinity;
    for (const [index, candidate] of remaining.entries()) {
      const distribution = selected.length ? Math.min(...selected.map(other => {
        const dx = (candidate.center.x - other.center.x) / (overlap.maxX - overlap.minX);
        const dy = (candidate.center.y - other.center.y) / (overlap.maxY - overlap.minY);
        return dx * dx + dy * dy;
      })) : 1;
      const score = distribution * candidate.texture * candidate.texture;
      if (score > bestScore) { bestScore = score; bestIndex = index; }
    }
    selected.push(remaining.splice(bestIndex, 1)[0]);
  }
  for (const candidate of candidates.filter(candidate => !selected.includes(candidate))) cells.push({ cellId: candidate.cellId,
    center: candidate.center, texture: candidate.texture,
    accepted: false, reason: 'Nicht ausgewaehlt' });
  for (const candidate of selected) {
    const measurement = measureShift(candidate.firstPatch, candidate.secondPatch, cellSize, cellSize,
      { searchRadius, minimumPsr, ownerBounds: candidate.ownerBounds });
    cells.push({ cellId: candidate.cellId, center: candidate.center,
      texture: candidate.texture, ...measurement });
  }
  cells.sort((first, second) => first.cellId - second.cellId);
  const usable = cells.filter(cell => cell.accepted && cell.ownedSupportPixels >= 8);
  if (usable.length < 3) return { accepted: false, reason: 'Zu wenig unabhaengige Strukturzellen', cells };
  let best = null;
  for (let firstIndex = 0; firstIndex < usable.length; firstIndex++) for (let secondIndex = firstIndex + 1;
    secondIndex < usable.length; secondIndex++) {
    const hypothesis = fitRigidCells([usable[firstIndex], usable[secondIndex]], cellSize, translationOnly);
    if (!hypothesis) continue;
    const inliers = usable.filter(cell => cellResidual(cell, hypothesis) <= residualLimit);
    const area = inliers.reduce((sum, cell) => sum + Math.min(cell.ownedSupportPixels, cellSize * 2), 0);
    const score = inliers.length * cellSize * 2 + area;
    if (!best || score > best.score || (score === best.score && area > best.area))
      best = { score, area, inliers };
  }
  if (!best || best.inliers.length < 3 || best.inliers.length <= usable.length / 2)
    return { accepted: false, reason: 'Kein raeumlicher Mehrheitskonsens', cells };
  let inliers = best.inliers, correction = null;
  for (let pass = 0; pass < 3; pass++) {
    correction = fitRigidCells(inliers, cellSize, translationOnly);
    if (!correction) return { accepted: false, reason: 'Rotation unterbestimmt', cells };
    const kept = usable.filter(cell => cellResidual(cell, correction) <= residualLimit);
    if (kept.length === inliers.length && kept.every((cell, index) => cell === inliers[index])) break;
    if (kept.length < 3) return { accepted: false, reason: 'Kein Flaechenkonsens', cells };
    inliers = kept;
  }
  correction = fitRigidCells(inliers, cellSize, translationOnly);
  if (translationOnly && Math.hypot(
      Math.max(...inliers.map(cell => cell.center.x)) - Math.min(...inliers.map(cell => cell.center.x)),
      Math.max(...inliers.map(cell => cell.center.y)) - Math.min(...inliers.map(cell => cell.center.y))) < cellSize)
    return {accepted:false, reason:'Zu wenig verteilte Struktur', cells};
  const residualRms = Math.sqrt(inliers.reduce((sum, cell) =>
    sum + cellResidual(cell, correction) ** 2, 0) / inliers.length);
  const pose = composePose(correction, secondPose);
  return { accepted: true, cellSize, translationOnly, pose, translation: { x: -correction.x, y: -correction.y },
    rotation: -correction.rotation, cells, inlierCells: inliers.map(cell => cell.cellId),
    uniqueSupportArea: inliers.reduce((sum, cell) => sum + cell.ownedSupportPixels, 0),
    residualRms, spatialCoverage: inliers.length / cells.length,
    pointPairs: inliers.map(cell => ({ cellId: cell.cellId,
      reference: rotate(cell.center.x - firstPose.x, cell.center.y - firstPose.y, -firstPose.rotation),
      current: rotate(cell.center.x + cell.dx - secondPose.x, cell.center.y + cell.dy - secondPose.y, -secondPose.rotation),
      psr: cell.psr, support: cell.ownedSupportPixels })) };
}

// Seed rotations about the image centre: rotating about the world origin couples
// a small angle to an arbitrarily large translation on long tracking paths.
export function registerPcbRotationPair(first, second, firstPose, secondPose, firstOffset, secondOffset,
  { angle = 2, radius = 128, ...options } = {}) {
  if (!(angle > 0) || !Number.isFinite(angle)) return null;
  const pivot = { x: secondOffset[0] + second.width / 2, y: secondOffset[1] + second.height / 2 };
  const initialPivot = rotate(pivot.x, pivot.y, secondPose.rotation);
  const cellSize = Math.min(128, options.cellSize ?? 256);
  const settings = { ...options, cellSize, adaptiveCells: true,
    searchRadius: Math.min(radius, cellSize / 2 - 1) };
  let best = null;
  for (const fraction of [0, -.5, .5, -1, 1]) {
    const rotation = secondPose.rotation + fraction * angle * Math.PI / 180;
    const offset = rotate(pivot.x, pivot.y, rotation);
    const seed = { x: secondPose.x + initialPivot.x - offset.x,
      y: secondPose.y + initialPivot.y - offset.y, rotation };
    let result = registerPcbFftPair(first, second, firstPose, seed, firstOffset, secondOffset, settings);
    if (!result.accepted || result.inlierCells.length < 5 || result.uniqueSupportArea < 1024) continue;
    // A second measurement removes the within-cell angular approximation.
    const refined = registerPcbFftPair(first, second, firstPose, result.pose, firstOffset, secondOffset,
      { ...settings, searchRadius: Math.min(31, settings.searchRadius) });
    if (!refined.accepted || refined.inlierCells.length < 5 || refined.uniqueSupportArea < 1024) continue;
    result = refined;
    const delta = Math.atan2(Math.sin(result.pose.rotation - secondPose.rotation),
      Math.cos(result.pose.rotation - secondPose.rotation));
    const finalPivot = rotate(pivot.x, pivot.y, result.pose.rotation);
    const movement = Math.hypot(result.pose.x + finalPivot.x - secondPose.x - initialPivot.x,
      result.pose.y + finalPivot.y - secondPose.y - initialPivot.y);
    if (Math.abs(delta) > angle * Math.PI / 180 || movement > radius) continue;
    if (!best || result.inlierCells.length > best.inlierCells.length ||
        (result.inlierCells.length === best.inlierCells.length && result.residualRms < best.residualRms))
      best = { ...result, method: 'FFT-Rotationssuche' };
  }
  return best;
}

// Low resolution whole-image phase correlation proposes large offsets before
// local cells refine the pose. These hypotheses are never accepted on their own.
export function coarsePcbFftSeeds(first, second, firstPose, secondPose, firstOffset, secondOffset,
  { angle = 5, radius = 384, size = 128, mask = null } = {}) {
  const started = performance.now();
  const scale = Math.max(first.width, first.height, second.width, second.height) / size;
  const firstGeometry = imageGeometry(first, firstPose, firstOffset);
  const center = firstGeometry.world(first.width / 2, first.height / 2);
  const raster = geometry => {
    const values = new Float64Array(size * size), valid = new Uint8Array(values.length);
    let total = 0, count = 0;
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const local = geometry.local({ x: center.x + (x - size / 2) * scale,
        y: center.y + (y - size / 2) * scale });
      const value = sampleGray(geometry.image, local.x, local.y, mask);
      if (value === null) continue;
      const index = y * size + x; values[index] = value; valid[index] = 1;
      total += value; count++;
    }
    const mean = count ? total / count : 0;
    for (let index = 0; index < values.length; index++) if (!valid[index]) values[index] = mean;
    return { values, coverage: count / values.length };
  };
  const reference = raster({ ...firstGeometry, image: first });
  if (reference.coverage < .05) return { seeds: [], ms: performance.now() - started };
  const pivot = { x: secondOffset[0] + second.width / 2, y: secondOffset[1] + second.height / 2 };
  const originalPivot = rotate(pivot.x, pivot.y, secondPose.rotation);
  const seeds = [];
  const fractions = angle > 0 ? [0, -.25, .25, -.5, .5, -.75, .75, -1, 1] : [0];
  for (const fraction of fractions) {
    const rotation = secondPose.rotation + fraction * angle * Math.PI / 180;
    const movedPivot = rotate(pivot.x, pivot.y, rotation);
    const pose = { x: secondPose.x + originalPivot.x - movedPivot.x,
      y: secondPose.y + originalPivot.y - movedPivot.y, rotation };
    const current = raster({ ...imageGeometry(second, pose, secondOffset), image: second });
    if (current.coverage < .05) continue;
    const measured = fftPatchShift(reference.values, current.values, size, size,
      { searchRadius: Math.min(size / 2 - 1, Math.max(2, radius / scale + 2)),
        minimumStructure: 1, minimumPsr: 5 });
    if (!measured.accepted || Math.hypot(measured.dx, measured.dy) * scale > radius) continue;
    seeds.push({ pose: { ...pose, x: pose.x - measured.dx * scale,
      y: pose.y - measured.dy * scale }, psr: measured.psr, angle: fraction * angle,
      shift: { x: -measured.dx * scale, y: -measured.dy * scale } });
  }
  seeds.sort((a, b) => b.psr - a.psr);
  return { seeds: seeds.slice(0, 3), scale, size, ms: performance.now() - started };
}

export function registerPcbCoarsePair(first, second, firstPose, secondPose, firstOffset, secondOffset,
  { angle = 5, radius = 384, ...options } = {}) {
  const coarse = coarsePcbFftSeeds(first, second, firstPose, secondPose, firstOffset, secondOffset,
    { angle, radius, mask: options.mask });
  let best = null;
  for (const seed of coarse.seeds) {
    const settings = { ...options, cellSize: Math.min(128, options.cellSize ?? 256), adaptiveCells: true };
    settings.searchRadius = Math.min(63, settings.cellSize / 2 - 1);
    let result = registerPcbFftPair(first, second, firstPose, seed.pose, firstOffset, secondOffset, settings);
    if (!result.accepted || result.inlierCells.length < 5 || result.uniqueSupportArea < 1024) continue;
    result = registerPcbFftPair(first, second, firstPose, result.pose, firstOffset, secondOffset,
      { ...settings, searchRadius: Math.min(31, settings.searchRadius) });
    if (!result.accepted || result.inlierCells.length < 5 || result.uniqueSupportArea < 1024) continue;
    const delta = Math.atan2(Math.sin(result.pose.rotation - secondPose.rotation),
      Math.cos(result.pose.rotation - secondPose.rotation));
    if (Math.abs(delta) > angle * Math.PI / 180 ||
        Math.hypot(result.pose.x - secondPose.x, result.pose.y - secondPose.y) > radius) continue;
    if (!best || result.inlierCells.length > best.inlierCells.length ||
        (result.inlierCells.length === best.inlierCells.length && result.residualRms < best.residualRms))
      best = { ...result, method: 'FFT-Grobsuche', coarseSearch: coarse };
  }
  return best;
}
