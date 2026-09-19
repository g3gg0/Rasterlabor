import test from 'node:test';
import assert from 'node:assert/strict';
import { detectGrid } from '../src/detector.js';

function checkerboardImage(width = 360, height = 280, curved = true) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      const ux = px + (curved ? 5 * Math.sin(py / 85) + 2 * Math.sin(px / 100) : 0);
      const uy = py + (curved ? 4 * Math.sin(px / 110) : 0);
      const value = (Math.floor(ux / 40) + Math.floor(uy / 40)) % 2 ? 225 : 25;
      const index = (py * width + px) * 4;
      data[index] = data[index + 1] = data[index + 2] = value;
      data[index + 3] = 255;
    }
  }
  return { width, height, data };
}

test('empty and one-dimensional images are rejected; checkerboard spacing can be estimated', () => {
  const image = checkerboardImage(240, 200, false);
  assert.equal(detectGrid(image).success, true);
  image.data.fill(255);
  assert.equal(detectGrid(image, { approxStep: 40 }).success, false);
  for (let py = 0; py < image.height; py++) {
    for (let px = 0; px < image.width; px++) {
      const value = px % 40 < 3 ? 20 : 240;
      const index = (py * image.width + px) * 4;
      image.data[index] = image.data[index + 1] = image.data[index + 2] = value;
    }
  }
  assert.equal(detectGrid(image, { approxStep: 40 }).success, false);
});

test('chessboard uses saddle corners rather than treating it as line paper', () => {
  const result = detectGrid(checkerboardImage(280, 240, false), { approxStep: 40 });
  assert.equal(result.success, true, result.reason);
  assert.ok(result.points.length >= 20);
});

test('chessboard accepts a precomputed dominant angle', () => {
  const result = detectGrid(checkerboardImage(280, 240, false), { approxStep: 40, precomputedAngle: 0 });
  assert.equal(result.success, true, result.reason);
  assert.ok(result.points.length >= 20);
  assert.equal(result.angle, 0);
});

test('chessboard refines precomputed corner candidates and assigns their topology', () => {
  const precomputedCandidates = [];
  for (let y = 40; y < 240; y += 40) {
    for (let x = 40; x < 280; x += 40) precomputedCandidates.push({ x, y, response: 255 });
  }
  const result = detectGrid(checkerboardImage(280, 240, false), {
    approxStep: 40, precomputedAngle: 0, precomputedCandidates
  });
  assert.equal(result.success, true, result.reason);
  assert.equal(result.points.length, precomputedCandidates.length);
  assert.equal(result.cols, 6);
  assert.equal(result.rows, 5);
});

test('chessboard assigns topology directly from GPU-refined corners', () => {
  const precomputedCorners = [];
  for (let y = 40; y < 240; y += 40) {
    for (let x = 40; x < 280; x += 40) precomputedCorners.push({ x, y, confidence: 1 });
  }
  const result = detectGrid(checkerboardImage(280, 240, false), {
    approxStep: 40, precomputedAngle: 0, precomputedCandidates: [], precomputedCorners
  });
  assert.equal(result.success, true, result.reason);
  assert.equal(result.points.length, precomputedCorners.length);
  assert.ok(result.detectorTiming.refinementMs < 1);
});

test('pattern size rejection reports the detected crossing counts', () => {
  const result = detectGrid(checkerboardImage(), { approxStep: 40, columns: 3, rows: 3 });
  assert.equal(result.success, false);
  assert.ok(result.cols > 3 && result.rows > 3);
  assert.ok(result.reason.includes(`${result.cols} Spalten und ${result.rows} Zeilen`));
});