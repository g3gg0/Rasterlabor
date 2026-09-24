import test from 'node:test';
import assert from 'node:assert/strict';
import { fitCheckerboardBrightness, sampleCheckerboardBrightness } from '../src/checkerboard-brightness.js';

test('white checkerboard interiors recover sensor shading without black fields or edges', () => {
  const width = 320, height = 240, cellSize = 40, block = 8;
  const samples = Array.from({ length: 20 }, (_, frame) => {
    const shiftX = (frame % 5) * 8 - 16, shiftY = (Math.floor(frame / 5) % 2) * 8 - 4;
    const data = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const col = Math.floor((x - shiftX) / cellSize), row = Math.floor((y - shiftY) / cellSize);
      const white = ((col + row) & 1) === 0;
      const ring = 1 - 0.13 * Math.exp(-(((Math.hypot(x - 225, y - 120) - 25) / 7) ** 2));
      const value = Math.round((white ? 205 : 35) * ring * (1 + frame * 0.004));
      data.set([value, value, value, 255], (y * width + x) * 4);
    }
    const cells = [];
    for (let row = 0; row < 6; row++) for (let col = 0; col < 8; col++) {
      const x = shiftX + col * cellSize, y = shiftY + row * cellSize;
      cells.push({ col, row, x: x + cellSize / 2, y: y + cellSize / 2, corners: [
        { x, y }, { x: x + cellSize, y }, { x: x + cellSize, y: y + cellSize }, { x, y: y + cellSize }
      ] });
    }
    return sampleCheckerboardBrightness({ width, height, data }, cells, frame, block);
  });
  assert.equal(samples[0].whiteParity, 0);
  assert.equal(samples[0].uniform[samples[0].width], 255);
  assert.equal(samples[0].uniform[samples[0].width + 5], 0);
  const result = fitCheckerboardBrightness(samples, new Set([18, 19]), { outputWidth: width, outputHeight: height });
  assert.ok(result.metrics.validationRms < result.metrics.baselineValidationRms * 0.45, JSON.stringify(result.metrics));
  const ring = [], center = [];
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const index = y * width + x, radius = Math.hypot(x - 225, y - 120);
    if (!result.supported[index]) continue;
    if (radius >= 22 && radius <= 28) ring.push(result.gain[index]);
    if (radius <= 8) center.push(result.gain[index]);
  }
  const median = values => values.toSorted((first, second) => first - second)[Math.floor(values.length / 2)];
  assert.ok(Math.max(...ring) > median(center) * 1.06, JSON.stringify({ ring: Math.max(...ring), center: median(center) }));
  assert.equal(result.model.representation, 'checkerboard-white-native-block-field');
});

test('checkerboard field fills unobserved blocks without marking them as measured', () => {
  const frames = Array.from({ length: 6 }, (_, frame) => {
    const gray = new Float32Array(100), uniform = new Uint8Array(100);
    for (let index = 0; index < 50; index++) {
      gray[index] = Math.exp(frame * 0.02 + (index % 10) * 0.04);
      uniform[index] = 255;
    }
    return { width: 10, height: 10, block: 8, gray, uniform };
  });
  const result = fitCheckerboardBrightness(frames, new Set([5]), { outputWidth: 80, outputHeight: 80 });
  const filled = 70 * result.width + 70;
  assert.equal(result.supported[filled], 0);
  assert.notEqual(result.gain[filled], 1);
  assert.equal(result.model.completion, 'nearest-smoothed');
});