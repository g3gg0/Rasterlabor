import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { detectGrid } from '../src/detector.js';

export function gridImage(width = 360, height = 280, curved = true, chessboard = false) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      const ux = px + (curved ? 5 * Math.sin(py / 85) + 2 * Math.sin(px / 100) : 0);
      const uy = py + (curved ? 4 * Math.sin(px / 110) : 0);
      const distanceX = Math.abs(((ux + 20) % 40) - 20);
      const distanceY = Math.abs(((uy + 20) % 40) - 20);
      const value = chessboard ? ((Math.floor(ux / 40) + Math.floor(uy / 40)) % 2 ? 225 : 25) :
        235 - 200 * Math.exp(-(Math.min(distanceX, distanceY) ** 2) / 3);
      const index = (py * width + px) * 4;
      data[index] = data[index + 1] = data[index + 2] = value;
      data[index + 3] = 255;
    }
  }
  return { width, height, data };
}

test('real image detector follows curved dark centerlines and subpixel intersections', () => {
  const result = detectGrid(gridImage(), { approxStep: 40, threshold: 18 });
  assert.equal(result.success, true, result.reason);
  assert.ok(result.points.length >= 25, `${result.points.length} points`);
  assert.ok(result.lines.length >= 10);
  const errors = result.points.map(point => {
    const ux = point.x + 5 * Math.sin(point.y / 85) + 2 * Math.sin(point.x / 100);
    const uy = point.y + 4 * Math.sin(point.x / 110);
    return Math.hypot(ux - Math.round(ux / 40) * 40, uy - Math.round(uy / 40) * 40);
  });
  assert.ok(Math.max(...errors) < 0.7, `max subpixel error ${Math.max(...errors)}`);
});

test('empty and one-dimensional images are rejected; spacing can be estimated', () => {
  const image = gridImage(240, 200, false);
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
  const result = detectGrid(gridImage(280, 240, false, true), { pattern: 'chessboard', approxStep: 40 });
  assert.equal(result.success, true, result.reason);
  assert.ok(result.points.length >= 20);
});

test('chessboard accepts a precomputed dominant angle', () => {
  const result = detectGrid(gridImage(280, 240, false, true), { pattern: 'chessboard', approxStep: 40, precomputedAngle: 0 });
  assert.equal(result.success, true, result.reason);
  assert.ok(result.points.length >= 20);
  assert.equal(result.angle, 0);
});

test('chessboard refines precomputed corner candidates and assigns their topology', () => {
  const precomputedCandidates = [];
  for (let y = 40; y < 240; y += 40) {
    for (let x = 40; x < 280; x += 40) precomputedCandidates.push({ x, y, response: 255 });
  }
  const result = detectGrid(gridImage(280, 240, false, true), {
    pattern: 'chessboard', approxStep: 40, precomputedAngle: 0, precomputedCandidates
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
  const result = detectGrid(gridImage(280, 240, false, true), {
    pattern: 'chessboard', approxStep: 40, precomputedAngle: 0, precomputedCandidates: [], precomputedCorners
  });
  assert.equal(result.success, true, result.reason);
  assert.equal(result.points.length, precomputedCorners.length);
  assert.ok(result.detectorTiming.refinementMs < 1);
});

test('pattern size rejection reports the detected crossing counts', () => {
  const result = detectGrid(gridImage(), { approxStep: 40, columns: 3, rows: 3 });
  assert.equal(result.success, false);
  assert.ok(result.cols > 3 && result.rows > 3);
  assert.ok(result.reason.includes(`${result.cols} Spalten und ${result.rows} Zeilen`));
});

test('thick curved lines remain connected across crossings inside a circular field', () => {
  const width = 640;
  const height = 640;
  const data = new Uint8ClampedArray(width * height);
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      const ux = px + 6 * Math.sin(py / 100);
      const uy = py + 5 * Math.sin(px / 110);
      const distanceX = Math.abs(((ux + 24) % 48 + 48) % 48 - 24);
      const distanceY = Math.abs(((uy + 24) % 48 + 48) % 48 - 24);
      data[py * width + px] = Math.hypot(px - 320, py - 320) > 270 ? 0 :
        220 - 200 * Math.exp(-(Math.min(distanceX, distanceY) ** 2) / 12);
      if (Math.abs(ux - 240) < 10 && uy > 250 && uy < 274) data[py * width + px] = 220;
    }
  }
  const result = detectGrid({ width, height, data }, { approxStep: 48 });
  assert.equal(result.success, true, result.reason);
  const missing = [];
  for (let row = 3; row <= 10; row++) {
    for (let col = 3; col <= 10; col++) {
      const found = result.points.some(point => Math.hypot(point.x + 6 * Math.sin(point.y / 100) - col * 48,
        point.y + 5 * Math.sin(point.x / 110) - row * 48) < 1);
      if (!found) missing.push([col, row]);
    }
  }
  assert.deepEqual(missing, []);
});

test('microscope video frame retains curved tracks through crossing disturbances', () => {
  const data = gunzipSync(readFileSync(new URL('./microscope-grid.gray.gz', import.meta.url)));
  assert.equal(data.length, 1080 * 1080);
  const result = detectGrid({ width: 1080, height: 1080, data });
  assert.equal(result.success, true, result.reason);
  assert.equal(result.cols, 17);
  assert.equal(result.rows, 17);
  assert.ok(result.points.length >= 235, `${result.points.length} crossings`);
  for (const [px, py] of [[716, 293], [211, 626], [446, 854]]) {
    assert.ok(result.points.some(point => Math.hypot(point.x - px, point.y - py) < 8), `Missing crossing near ${px}, ${py}`);
  }
});