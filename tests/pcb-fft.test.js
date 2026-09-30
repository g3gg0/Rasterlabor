import test from 'node:test';
import assert from 'node:assert/strict';
import { fftPatchShift } from '../src/pcb-fft.js';

const width = 64, height = 64;
const scene = () => {
  const values = new Float64Array(width * height);
  for (let row = 0; row < height; row++) for (let column = 0; column < width; column++) {
    const trace = Math.abs(row - 18 - 5 * Math.sin(column / 9)) < 1.4 ||
      Math.abs(column - 38 - 4 * Math.sin(row / 7)) < 1.3 ||
      Math.abs(row - 47 + 3 * Math.sin(column / 5)) < 1;
    values[row * width + column] = 80 + (trace ? 65 : 0) + 8 * Math.sin(column * 0.29 + row * 0.41);
  }
  return values;
};
const translated = (values, dx, dy, brightness = 0) => {
  const result = new Float64Array(values.length);
  for (let row = 0; row < height; row++) for (let column = 0; column < width; column++) {
    const sourceX = column - dx, sourceY = row - dy;
    result[row * width + column] = brightness + (sourceX >= 0 && sourceX < width && sourceY >= 0 && sourceY < height ?
      values[sourceY * width + sourceX] : 80);
  }
  return result;
};

test('FFT patch translation preserves both signs and tolerates brightness offset', () => {
  const first = scene();
  for (const [dx, dy] of [[4, -3], [-5, 2]]) {
    const result = fftPatchShift(first, translated(first, dx, dy, 35), width, height, { searchRadius: 12 });
    assert.equal(result.accepted, true, JSON.stringify(result));
    assert.ok(Math.hypot(result.dx - dx, result.dy - dy) < 0.6, JSON.stringify(result));
  }
});

test('FFT patch rejects flat and independent noise instead of inventing a secure shift', () => {
  const flat = new Float64Array(width * height).fill(100);
  assert.equal(fftPatchShift(flat, flat, width, height).reason, 'Unzureichende Struktur');
  const noise = seed => {
    let value = seed;
    return Float64Array.from({ length: width * height }, () => {
      value = (1664525 * value + 1013904223) >>> 0;
      return value / 2 ** 32 * 255;
    });
  };
  assert.equal(fftPatchShift(noise(1), noise(2), width, height, { minimumPsr: 8 }).accepted, false);
});
