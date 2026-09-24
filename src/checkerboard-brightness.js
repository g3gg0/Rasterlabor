const linear = value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
const table = Float32Array.from({ length: 256 }, (_, value) => linear(value / 255));

const median = values => {
  if (!values.length) return NaN;
  values.sort((first, second) => first - second);
  return (values[Math.floor(values.length / 2)] + values[Math.floor((values.length - 1) / 2)]) / 2;
};

function insetCell(cell, fraction) {
  return cell.corners.map(point => ({ x: cell.x + (point.x - cell.x) * (1 - fraction),
    y: cell.y + (point.y - cell.y) * (1 - fraction) }));
}

function insideConvex(points, x, y) {
  let sign = 0;
  for (let index = 0; index < points.length; index++) {
    const first = points[index], second = points[(index + 1) % points.length];
    const cross = (second.x - first.x) * (y - first.y) - (second.y - first.y) * (x - first.x);
    if (Math.abs(cross) < 1e-7) continue;
    if (sign && Math.sign(cross) !== sign) return false;
    sign = Math.sign(cross);
  }
  return Boolean(sign);
}

function blockLuminance(image, column, row, block, allowed) {
  let sum = 0, squared = 0, count = 0, minimum = Infinity, maximum = -Infinity;
  for (let y = row * block; y < Math.min(image.height, (row + 1) * block); y++) {
    for (let x = column * block; x < Math.min(image.width, (column + 1) * block); x++) {
      const offset = (y * image.width + x) * 4;
      if ((allowed && !allowed(x, y)) || image.data[offset + 3] !== 255 ||
        Math.max(image.data[offset], image.data[offset + 1], image.data[offset + 2]) >= 253) return null;
      const value = 0.2126 * table[image.data[offset]] + 0.7152 * table[image.data[offset + 1]] + 0.0722 * table[image.data[offset + 2]];
      if (value <= 0.01) return null;
      sum += value; squared += value * value; minimum = Math.min(minimum, value); maximum = Math.max(maximum, value); count++;
    }
  }
  const mean = sum / count, deviation = Math.sqrt(Math.max(0, squared / count - mean * mean));
  return deviation / mean <= 0.035 && (maximum - minimum) / mean <= 0.16 ? mean : null;
}

export function sampleCheckerboardBrightness(image, cells, frame, block = 8, allowed = null, inset = 0.24) {
  const width = Math.floor(image.width / block), height = Math.floor(image.height / block);
  const gray = new Float32Array(width * height), uniform = new Uint8Array(width * height);
  const candidates = [[], []], measured = [];
  for (const cell of cells) {
    const polygon = insetCell(cell, inset);
    const column = Math.floor(cell.x / block), row = Math.floor(cell.y / block);
    const value = column >= 0 && row >= 0 && column < width && row < height ? blockLuminance(image, column, row, block, allowed) : null;
    if (value) candidates[(cell.col + cell.row) & 1].push(value);
    measured.push({ cell, polygon });
  }
  const levels = candidates.map(values => median(values));
  if (!levels.every(Number.isFinite) || Math.max(...levels) < Math.min(...levels) * 1.35) {
    return { frame, width, height, block, gray, uniform, whiteParity: null, cells: 0 };
  }
  const whiteParity = levels[1] > levels[0] ? 1 : 0;
  let acceptedCells = 0;
  for (const { cell, polygon } of measured) {
    if (((cell.col + cell.row) & 1) !== whiteParity) continue;
    const minimumX = Math.max(0, Math.floor(Math.min(...polygon.map(point => point.x)) / block));
    const maximumX = Math.min(width - 1, Math.floor(Math.max(...polygon.map(point => point.x)) / block));
    const minimumY = Math.max(0, Math.floor(Math.min(...polygon.map(point => point.y)) / block));
    const maximumY = Math.min(height - 1, Math.floor(Math.max(...polygon.map(point => point.y)) / block));
    let accepted = false;
    for (let row = minimumY; row <= maximumY; row++) for (let column = minimumX; column <= maximumX; column++) {
      const margin = 0.5;
      if (![[column * block + margin, row * block + margin], [(column + 1) * block - margin, row * block + margin],
        [(column + 1) * block - margin, (row + 1) * block - margin], [column * block + margin, (row + 1) * block - margin]]
        .every(([x, y]) => insideConvex(polygon, x, y))) continue;
      const value = blockLuminance(image, column, row, block, allowed);
      if (!value) continue;
      const index = row * width + column;
      gray[index] = value; uniform[index] = 255; accepted = true;
    }
    if (accepted) acceptedCells++;
  }
  return { frame, width, height, block, gray, uniform, whiteParity, cells: acceptedCells };
}

function fieldValue(values, width, height, block, x, y) {
  const column = Math.max(0, Math.min(width - 1, x / block - 0.5));
  const row = Math.max(0, Math.min(height - 1, y / block - 0.5));
  const left = Math.floor(column), top = Math.floor(row);
  const right = Math.min(width - 1, left + 1), bottom = Math.min(height - 1, top + 1);
  const fx = column - left, fy = row - top;
  return (1 - fy) * ((1 - fx) * values[top * width + left] + fx * values[top * width + right]) +
    fy * ((1 - fx) * values[bottom * width + left] + fx * values[bottom * width + right]);
}

function smoothField(values, weights, width, height, amount = 0.12) {
  const result = values.slice();
  for (let row = 1; row < height - 1; row++) for (let column = 1; column < width - 1; column++) {
    const index = row * width + column;
    if (!weights[index]) continue;
    let sum = 0, count = 0;
    for (const neighbor of [index - 1, index + 1, index - width, index + width]) if (weights[neighbor]) {
      sum += values[neighbor]; count++;
    }
    if (count) result[index] = values[index] * (1 - amount) + sum / count * amount;
  }
  return result;
}

function fillField(values, weights, width, height) {
  let result = values.slice(), support = Uint8Array.from(weights, value => value ? 1 : 0);
  for (let iteration = 0; iteration < width + height && support.some(value => !value); iteration++) {
    const next = result.slice(), nextSupport = support.slice(); let changed = false;
    for (let row = 0; row < height; row++) for (let column = 0; column < width; column++) {
      const index = row * width + column;
      if (support[index]) continue;
      let sum = 0, count = 0;
      for (const neighbor of [column ? index - 1 : -1, column + 1 < width ? index + 1 : -1,
        row ? index - width : -1, row + 1 < height ? index + width : -1]) if (neighbor >= 0 && support[neighbor]) {
        sum += result[neighbor]; count++;
      }
      if (count) { next[index] = sum / count; nextSupport[index] = 1; changed = true; }
    }
    result = next; support = nextSupport;
    if (!changed) break;
  }
  return result;
}

function blurField(values, width, height, radius = 1, passes = 2) {
  let result = values;
  for (let pass = 0; pass < passes; pass++) {
    const horizontal = new Float32Array(result.length), vertical = new Float32Array(result.length);
    for (let row = 0; row < height; row++) {
      let sum = 0;
      for (let column = -radius; column <= radius; column++) sum += result[row * width + Math.max(0, Math.min(width - 1, column))];
      for (let column = 0; column < width; column++) {
        horizontal[row * width + column] = sum / (radius * 2 + 1);
        sum += result[row * width + Math.min(width - 1, column + radius + 1)] - result[row * width + Math.max(0, column - radius)];
      }
    }
    for (let column = 0; column < width; column++) {
      let sum = 0;
      for (let row = -radius; row <= radius; row++) sum += horizontal[Math.max(0, Math.min(height - 1, row)) * width + column];
      for (let row = 0; row < height; row++) {
        vertical[row * width + column] = sum / (radius * 2 + 1);
        sum += horizontal[Math.min(height - 1, row + radius + 1) * width + column] - horizontal[Math.max(0, row - radius) * width + column];
      }
    }
    result = vertical;
  }
  return result;
}

export function fitCheckerboardBrightness(frames, heldOut = new Set(), { outputWidth, outputHeight, maxGain = 1.6, allowed = null } = {}) {
  if (!frames.length) throw new Error('Keine Checkerboard-Helligkeitsframes vorhanden.');
  const fieldWidth = frames[0].width, fieldHeight = frames[0].height, fieldSize = fieldWidth * fieldHeight;
  const observations = frames.map(frame => Array.from(frame.uniform, (value, index) => value && frame.gray[index] > 0 ?
    { index, value: Math.log(frame.gray[index]) } : null).filter(Boolean));
  const trainingCount = observations.reduce((sum, rows, index) => sum + (heldOut.has(index) ? 0 : rows.length), 0);
  const validationCount = observations.reduce((sum, rows, index) => sum + (heldOut.has(index) ? rows.length : 0), 0);
  if (trainingCount < 200 || validationCount < 40) throw new Error(`Zu wenige weisse Checkerboard-Bloecke: ${trainingCount} Training, ${validationCount} Validierung.`);
  let values = new Float32Array(fieldSize), weights = new Float32Array(fieldSize);
  for (let iteration = 0; iteration < 12; iteration++) {
    const sums = new Float64Array(fieldSize), nextWeights = new Float32Array(fieldSize);
    for (let frameIndex = 0; frameIndex < observations.length; frameIndex++) {
      if (heldOut.has(frameIndex)) continue;
      const rows = observations[frameIndex];
      const exposure = median(rows.map(row => row.value - values[row.index]));
      for (const row of rows) { sums[row.index] += row.value - exposure; nextWeights[row.index]++; }
    }
    const next = values.slice();
    for (let index = 0; index < fieldSize; index++) if (nextWeights[index]) next[index] = sums[index] / nextWeights[index];
    values = smoothField(next, nextWeights, fieldWidth, fieldHeight, 0.08);
    weights = nextWeights;
    let mean = 0, count = 0;
    for (let index = 0; index < fieldSize; index++) if (weights[index]) { mean += values[index]; count++; }
    mean /= count;
    for (let index = 0; index < fieldSize; index++) if (weights[index]) values[index] -= mean;
  }
  const residuals = (frameIndexes, corrected) => {
    let squared = 0, count = 0;
    for (const frameIndex of frameIndexes) {
      const rows = observations[frameIndex].filter(row => weights[row.index]);
      const offset = median(rows.map(row => row.value - (corrected ? values[row.index] : 0)));
      for (const row of rows) { const residual = row.value - (corrected ? values[row.index] : 0) - offset; squared += residual ** 2; count++; }
    }
    return { rms: Math.sqrt(squared / count), count };
  };
  const validationFrames = [...heldOut];
  const baseline = residuals(validationFrames, false), validation = residuals(validationFrames, true);
  if (!Number.isFinite(validation.rms)) throw new Error('Checkerboard-Helligkeitsmodell ist numerisch instabil.');
  const filled = blurField(fillField(values, weights, fieldWidth, fieldHeight), fieldWidth, fieldHeight);
  const gain = new Float32Array(outputWidth * outputHeight).fill(1), supported = new Uint8Array(gain.length);
  for (let y = 0; y < outputHeight; y++) for (let x = 0; x < outputWidth; x++) {
    const blockColumn = Math.min(fieldWidth - 1, Math.floor(x / frames[0].block));
    const blockRow = Math.min(fieldHeight - 1, Math.floor(y / frames[0].block));
    const index = y * outputWidth + x;
    if (!allowed || allowed(x, y)) {
      gain[index] = Math.max(1 / maxGain, Math.min(maxGain, Math.exp(-fieldValue(filled, fieldWidth, fieldHeight, frames[0].block, x + 0.5, y + 0.5))));
      if (weights[blockRow * fieldWidth + blockColumn] >= 2) supported[index] = 255;
    }
  }
  return { version: 2, width: outputWidth, height: outputHeight, gain, supported,
    model: { representation: 'checkerboard-white-native-block-field', solver: 'alternating-frame-exposure', block: frames[0].block,
      smoothing: { iterative: 0.08, radius: 1, passes: 2 }, completion: 'nearest-smoothed', maxGain, exposureMode: 'per-frame-offset' },
    metrics: { accelerator: 'CPU', equations: trainingCount, validationEquations: validation.count,
      trainingFrames: frames.length - heldOut.size, validationFrames: heldOut.size,
      baselineValidationRms: baseline.rms, validationRms: validation.rms,
      covered: supported.reduce((sum, value) => sum + Number(Boolean(value)), 0) / supported.length } };
}