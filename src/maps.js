import { evaluate, inversePoint } from './spline.js';
import { geometryCheck, tick, statistics } from './solver.js';
import { MASK_FORBIDDEN } from './patch-mask.js';

const MAP_ARRAY_BUDGET = 768 * 1024 * 1024;

function forbidden(mask, x, y) {
  if (!mask?.data || x < 0 || y < 0 || x >= mask.sourceWidth || y >= mask.sourceHeight) return false;
  const col = Math.min(mask.width - 1, Math.floor((x + 1e-4) / mask.cellSize));
  const row = Math.min(mask.height - 1, Math.floor((y + 1e-4) / mask.cellSize));
  return mask.data[row * mask.width + col] === MASK_FORBIDDEN;
}

function cropOutputMaps(maps, bounds) {
  const [minimumX, minimumY, maximumX, maximumY] = bounds;
  const width = maximumX - minimumX + 1;
  const height = maximumY - minimumY + 1;
  const copy = (source, Constructor) => {
    const target = new Constructor(width * height);
    if (Constructor === Float32Array) target.fill(-1);
    for (let row = 0; row < height; row++) {
      const start = (row + minimumY) * maps.outputWidth + minimumX;
      target.set(source.subarray(start, start + width), row * width);
    }
    return target;
  };
  return { ...maps, inverseX: copy(maps.inverseX, Float32Array), inverseY: copy(maps.inverseY, Float32Array),
    valid: copy(maps.valid, Uint8Array), numericalValid: copy(maps.numericalValid, Uint8Array),
    outputWidth: width, outputHeight: height, origin: [maps.origin[0] + minimumX, maps.origin[1] + minimumY] };
}

export function coverageGrid(frames, width, height, step, cols = 24, rows = 18, patchSize = 0) {
  const counts = new Uint8Array(cols * rows);
  for (const frame of frames.filter(entry => entry.enabled && entry.role === 'train')) {
    const occupied = new Set();
    for (const point of frame.points) {
      if (patchSize) {
        const minCol = Math.max(0, Math.floor((point.x - patchSize / 2) / width * cols));
        const maxCol = Math.min(cols - 1, Math.floor((point.x + patchSize / 2) / width * cols));
        const minRow = Math.max(0, Math.floor((point.y - patchSize / 2) / height * rows));
        const maxRow = Math.min(rows - 1, Math.floor((point.y + patchSize / 2) / height * rows));
        for (let row = minRow; row <= maxRow; row++) for (let col = minCol; col <= maxCol; col++) occupied.add(row * cols + col);
        continue;
      }
      const radius = step * 0.8;
      const minCol = Math.max(0, Math.floor((point.x - radius) / width * cols));
      const maxCol = Math.min(cols - 1, Math.floor((point.x + radius) / width * cols));
      const minRow = Math.max(0, Math.floor((point.y - radius) / height * rows));
      const maxRow = Math.min(rows - 1, Math.floor((point.y + radius) / height * rows));
      for (let row = minRow; row <= maxRow; row++) {
        for (let col = minCol; col <= maxCol; col++) {
          const px = (col + 0.5) / cols * width;
          const py = (row + 0.5) / rows * height;
          if (Math.hypot(px - point.x, py - point.y) < radius) occupied.add(row * cols + col);
        }
      }
    }
    for (const index of occupied) counts[index] = Math.min(255, counts[index] + 1);
  }
  return { cols, rows, counts };
}

export function heatmapGridSize(width, height, cellSize = 24, maximum = 160) {
  return { cols: Math.min(maximum, Math.max(24, Math.ceil(width / cellSize))),
    rows: Math.min(maximum, Math.max(18, Math.ceil(height / cellSize))) };
}

export function relativeCoverageScale(counts) {
  let minimum = Infinity;
  let maximum = 0;
  for (const count of counts) {
    if (count > 0) minimum = Math.min(minimum, count);
    maximum = Math.max(maximum, count);
  }
  if (!maximum) minimum = 0;
  const colors = Array.from({ length: 256 }, (_, count) => {
    if (!count) return [135, 142, 138, 255];
    const ratio = maximum === minimum ? 0.5 : Math.max(0, Math.min(1, (count - minimum) / (maximum - minimum)));
    return [Math.round(235 - 210 * ratio), Math.round(180 - 35 * ratio), Math.round(60 + 105 * ratio), 255];
  });
  return { minimum, maximum, colors };
}

export async function buildMaps(calibration, frames, notify = () => {}, cancelled = () => false, inverseBuilder = null) {
  const { field } = calibration;
  const geometry = geometryCheck(field, calibration.parameters?.tau ?? 0.12);
  if (!geometry.valid) throw new Error(geometry.reason);
  const origin = [Math.floor(geometry.bounds[0]), Math.floor(geometry.bounds[1])];
  const outputWidth = Math.ceil(geometry.bounds[2]) - origin[0] + 1;
  const outputHeight = Math.ceil(geometry.bounds[3]) - origin[1] + 1;
  const sourceCount = field.width * field.height;
  const outputCount = outputWidth * outputHeight;
  if (!Number.isSafeInteger(outputCount) || sourceCount * 9 + outputCount * 10 > MAP_ARRAY_BUDGET) {
    throw new Error('Maps benoetigen mehr als das 768-MiB-Arraybudget. Kleinere feste Zielaufloesung oder kleinere Videoaufloesung verwenden.');
  }
  const forward = new Float32Array(sourceCount * 2);
  const sourceCoverage = new Uint8Array(sourceCount);
  const patchSize = calibration.parameters?.pattern === 'patches' ? calibration.parameters.patchSize : 0;
  const heatmap = heatmapGridSize(field.width, field.height);
  const coverage = coverageGrid(frames, field.width, field.height, calibration.parameters?.approxStep || calibration.step,
    heatmap.cols, heatmap.rows, patchSize);
  for (let py = 0; py < field.height; py++) {
    for (let px = 0; px < field.width; px++) {
      const index = py * field.width + px;
      const value = evaluate(field, px, py);
      forward[index * 2] = value.x;
      forward[index * 2 + 1] = value.y;
      sourceCoverage[index] = coverage.counts[Math.min(coverage.rows - 1, Math.floor(py / field.height * coverage.rows)) * coverage.cols +
        Math.min(coverage.cols - 1, Math.floor(px / field.width * coverage.cols))];
    }
    if (py % 64 === 0) {
      notify({ stage: 'forward', done: py, total: field.height });
      await tick();
      if (cancelled()) throw new Error('Berechnung abgebrochen.');
    }
  }
  let inverseX;
  let inverseY;
  let valid;
  let numericalValid;
  let roundtrip = [];
  let numericalCount = 0;
  let validCount = 0;
  let maxRoundtrip = 0;
  const patchMask = calibration.parameters?.patchMask;
  let cropMinimumX = outputWidth;
  let cropMinimumY = outputHeight;
  let cropMaximumX = -1;
  let cropMaximumY = -1;
  let accelerator = 'CPU';
  let timing = null;
  let accelerated = null;
  if (inverseBuilder) {
    try { accelerated = await inverseBuilder(field, geometry, origin, outputWidth, outputHeight, sourceCoverage,
      patchMask, notify, cancelled); }
    catch (error) {
      if (cancelled()) throw error;
    }
  }
  if (accelerated) {
    ({ inverseX, inverseY, valid, numericalValid, numericalCount, validCount, maxRoundtrip, roundtrip } = accelerated);
    [cropMinimumX, cropMinimumY, cropMaximumX, cropMaximumY] = accelerated.cropBounds;
    accelerator = accelerated.accelerator;
    timing = accelerated.timing;
  } else {
    inverseX = new Float32Array(outputCount).fill(-1);
    inverseY = new Float32Array(outputCount).fill(-1);
    valid = new Uint8Array(outputCount);
    numericalValid = new Uint8Array(outputCount);
    for (let triangleIndex = 0; triangleIndex < geometry.triangles.length; triangleIndex++) {
    const [first, second, third] = geometry.triangles[triangleIndex];
    const minimumX = Math.max(0, Math.ceil(Math.min(first.x, second.x, third.x) - origin[0]));
    const maximumX = Math.min(outputWidth - 1, Math.floor(Math.max(first.x, second.x, third.x) - origin[0]));
    const minimumY = Math.max(0, Math.ceil(Math.min(first.y, second.y, third.y) - origin[1]));
    const maximumY = Math.min(outputHeight - 1, Math.floor(Math.max(first.y, second.y, third.y) - origin[1]));
    const determinant = (second.y - third.y) * (first.x - third.x) + (third.x - second.x) * (first.y - third.y);
    for (let py = minimumY; py <= maximumY; py++) {
      for (let px = minimumX; px <= maximumX; px++) {
        const index = py * outputWidth + px;
        if (numericalValid[index]) continue;
        const qx = origin[0] + px;
        const qy = origin[1] + py;
        const firstWeight = ((second.y - third.y) * (qx - third.x) + (third.x - second.x) * (qy - third.y)) / determinant;
        const secondWeight = ((third.y - first.y) * (qx - third.x) + (first.x - third.x) * (qy - third.y)) / determinant;
        const thirdWeight = 1 - firstWeight - secondWeight;
        if (Math.min(firstWeight, secondWeight, thirdWeight) < -1e-7) continue;
        const source = inversePoint(field, qx, qy, firstWeight * first.px + secondWeight * second.px + thirdWeight * third.px,
          firstWeight * first.py + secondWeight * second.py + thirdWeight * third.py, 0.002);
        if (!source || source.x < 0 || source.y < 0 || source.x >= field.width - 1 || source.y >= field.height - 1) continue;
        inverseX[index] = source.x;
        inverseY[index] = source.y;
        if (inverseX[index] >= field.width - 1 || inverseY[index] >= field.height - 1) {
          inverseX[index] = inverseY[index] = -1;
          continue;
        }
        const stored = evaluate(field, inverseX[index], inverseY[index]);
        const storedError = Math.hypot(stored.x - qx, stored.y - qy);
        if (storedError > 0.01) { inverseX[index] = inverseY[index] = -1; continue; }
        if (forbidden(patchMask, source.x, source.y)) { inverseX[index] = inverseY[index] = -1; continue; }
        numericalValid[index] = 1;
        cropMinimumX = Math.min(cropMinimumX, px);
        cropMinimumY = Math.min(cropMinimumY, py);
        cropMaximumX = Math.max(cropMaximumX, px);
        cropMaximumY = Math.max(cropMaximumY, py);
        numericalCount++;
        maxRoundtrip = Math.max(maxRoundtrip, storedError);
        if (index % 101 === 0) roundtrip.push(storedError);
        const sourceIndex = Math.floor(source.y) * field.width + Math.floor(source.x);
        if ([sourceIndex, sourceIndex + 1, sourceIndex + field.width, sourceIndex + field.width + 1].every(location => sourceCoverage[location] >= 3)) {
          valid[index] = 1;
          validCount++;
        }
      }
    }
    if (triangleIndex % 128 === 0) {
      notify({ stage: 'inverse', done: triangleIndex, total: geometry.triangles.length });
      await tick();
      if (cancelled()) throw new Error('Berechnung abgebrochen.');
    }
  }
  }
  let maps = { forward, sourceCoverage, inverseX, inverseY, valid, numericalValid, outputWidth, outputHeight, origin,
    coverageGrid: coverage, roundtrip: { ...statistics(roundtrip), maximum: maxRoundtrip, numericalCount, validCount,
      validFraction: validCount / outputCount, tolerance: 0.01 }, accelerator, timing };
  if (patchMask?.data?.includes(MASK_FORBIDDEN)) {
    if (cropMaximumX < cropMinimumX || cropMaximumY < cropMinimumY) throw new Error('Die rote Maske schliesst den gesamten Ausgabebereich aus.');
    maps = cropOutputMaps(maps, [cropMinimumX, cropMinimumY, cropMaximumX, cropMaximumY]);
    maps.roundtrip.validFraction = validCount / (maps.outputWidth * maps.outputHeight);
  }
  return maps;
}

export function remapRGBA(image, maps, sourceMask = null) {
  const output = new Uint8ClampedArray(maps.outputWidth * maps.outputHeight * 4);
  const maskData = sourceMask?.data;
  const cellSize = sourceMask?.cellSize;
  const includes = (x, y) => !maskData || maskData[Math.floor(y / cellSize) * sourceMask.width + Math.floor(x / cellSize)] === 1;
  for (let index = 0; index < maps.valid.length; index++) {
    if (!maps.valid[index]) continue;
    const px = maps.inverseX[index];
    const py = maps.inverseY[index];
    const col = Math.floor(px);
    const row = Math.floor(py);
    if (maskData && (col < 0 || row < 0 || col + 1 >= sourceMask.sourceWidth || row + 1 >= sourceMask.sourceHeight ||
        !includes(col, row) || !includes(col + 1, row) || !includes(col, row + 1) || !includes(col + 1, row + 1))) continue;
    const localX = px - col;
    const localY = py - row;
    const sourceIndex = (row * image.width + col) * 4;
    for (let component = 0; component < 3; component++) {
      output[4 * index + component] = (1 - localY) * ((1 - localX) * image.data[sourceIndex + component] + localX * image.data[sourceIndex + 4 + component]) +
        localY * ((1 - localX) * image.data[sourceIndex + 4 * image.width + component] + localX * image.data[sourceIndex + 4 * image.width + 4 + component]);
    }
    output[4 * index + 3] = 255;
  }
  return { width: maps.outputWidth, height: maps.outputHeight, data: output };
}
