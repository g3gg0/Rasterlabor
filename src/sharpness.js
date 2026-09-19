const luma = (data, index) => 0.2126 * data[index] + 0.7152 * data[index + 1] + 0.0722 * data[index + 2];

export function laplacianVariance(data, width, height, region = { x: 0, y: 0, width, height }) {
  const firstX = Math.max(1, Math.floor(region.x) + 1);
  const firstY = Math.max(1, Math.floor(region.y) + 1);
  const lastX = Math.min(width - 1, Math.ceil(region.x + region.width) - 1);
  const lastY = Math.min(height - 1, Math.ceil(region.y + region.height) - 1);
  let count = 0, sum = 0, squared = 0, luminance = 0, luminanceSquared = 0;
  for (let y = firstY; y < lastY; y++) {
    for (let x = firstX; x < lastX; x++) {
      const index = (y * width + x) * 4;
      const center = luma(data, index);
      const value = 4 * center - luma(data, index - 4) - luma(data, index + 4) -
        luma(data, index - width * 4) - luma(data, index + width * 4);
      count++; sum += value; squared += value * value;
      luminance += center; luminanceSquared += center * center;
    }
  }
  if (!count) return { variance: 0, contrast: 0, samples: 0 };
  return { variance: Math.max(0, squared / count - (sum / count) ** 2),
    contrast: Math.sqrt(Math.max(0, luminanceSquared / count - (luminance / count) ** 2)), samples: count };
}

export function sharpnessFromTileGrid(data, tileSize, columns = 3, rows = 3) {
  const width = tileSize * columns, height = tileSize * rows;
  if (!(data instanceof Uint8ClampedArray) || data.length !== width * height * 4 || tileSize < 3) {
    throw new Error('Ungueltige Pixeldaten fuer die Schaerfemessung.');
  }
  return sharpnessFromGrid(data, width, height, columns, rows);
}

export function sharpnessFromGrid(data, width, height, columns = 3, rows = 3) {
  if (!(data instanceof Uint8ClampedArray) || data.length !== width * height * 4 ||
      width < columns * 3 || height < rows * 3) throw new Error('Ungueltige Pixeldaten fuer die Schaerfemessung.');
  const tiles = [];
  for (let row = 0; row < rows; row++) for (let col = 0; col < columns; col++) {
    const x = Math.floor(col * width / columns), y = Math.floor(row * height / rows);
    const right = Math.floor((col + 1) * width / columns), bottom = Math.floor((row + 1) * height / rows);
    tiles.push(laplacianVariance(data, width, height, { x, y, width: right - x, height: bottom - y }));
  }
  const variances = tiles.map(tile => tile.variance).sort((a, b) => a - b);
  const contrasts = tiles.map(tile => tile.contrast).sort((a, b) => a - b);
  return { score: variances[Math.floor(variances.length / 2)],
    mean: variances.reduce((sum, value) => sum + value, 0) / variances.length,
    minimum: variances[0], maximum: variances.at(-1),
    contrast: contrasts[Math.floor(contrasts.length / 2)],
    samples: tiles.reduce((sum, tile) => sum + tile.samples, 0), sampleWidth: width, sampleHeight: height,
    method: 'median-laplacian-variance-9-downsampled-regions' };
}

export function validSharpness(value) {
  return value && typeof value === 'object' &&
    ['score', 'mean', 'minimum', 'maximum', 'contrast'].every(key => Number.isFinite(value[key]) && value[key] >= 0) &&
    Number.isSafeInteger(value.samples) && value.samples > 0 &&
    Number.isSafeInteger(value.sampleWidth) && value.sampleWidth >= 9 &&
    Number.isSafeInteger(value.sampleHeight) && value.sampleHeight >= 9 &&
    value.method === 'median-laplacian-variance-9-downsampled-regions';
}

let canvas = null;
export async function measureFrameSharpness(frame, useWebGpu = false) {
  if (useWebGpu) {
    try {
      const gpu = await measureFrameSharpnessGpu(frame);
      if (gpu) return gpu;
    } catch {}
  }
  const sourceWidth = frame.displayWidth, sourceHeight = frame.displayHeight;
  const scale = Math.min(1, 1024 / Math.max(sourceWidth, sourceHeight));
  const width = Math.max(9, Math.floor(sourceWidth * scale / 3) * 3);
  const height = Math.max(9, Math.floor(sourceHeight * scale / 3) * 3);
  canvas ??= new OffscreenCanvas(width, height);
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('2D-Canvas fuer die Schaerfemessung ist nicht verfuegbar.');
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(frame, 0, 0, width, height);
  return { ...sharpnessFromGrid(context.getImageData(0, 0, width, height).data, width, height), accelerator: 'Canvas' };
}
import { measureFrameSharpnessGpu } from './webgpu-sharpness.js';
