import test from 'node:test';
import assert from 'node:assert/strict';
import { laplacianVariance, sharpnessFromTileGrid, validSharpness } from '../src/sharpness.js';

function image(width, height, value) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const gray = value(x, y), index = (y * width + x) * 4;
    data.set([gray, gray, gray, 255], index);
  }
  return data;
}

test('Laplacian sharpness is zero for a flat frame and responds to native-pixel edges', () => {
  const flat = laplacianVariance(image(24, 24, () => 90), 24, 24);
  const soft = laplacianVariance(image(24, 24, x => Math.min(255, x * 11)), 24, 24);
  const sharp = laplacianVariance(image(24, 24, (x, y) => (x + y) % 2 ? 255 : 0), 24, 24);
  assert.equal(flat.variance, 0);
  assert.ok(sharp.variance > soft.variance * 100);
  assert.equal(sharp.samples, 22 * 22);
});

test('tile-grid score uses the median so one sharp outlier cannot dominate a frame', () => {
  const tile = 8, width = tile * 3, height = tile * 3;
  const data = image(width, height, (x, y) => x < tile && y < tile && (x + y) % 2 ? 255 : 80);
  const result = sharpnessFromTileGrid(data, tile);
  assert.equal(result.score, 0);
  assert.ok(result.maximum > 0);
  assert.equal(result.method, 'median-laplacian-variance-9-downsampled-regions');
  assert.equal(validSharpness(result), true);
  assert.equal(validSharpness({ ...result, score: NaN }), false);
});
