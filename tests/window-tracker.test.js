import test from 'node:test';
import assert from 'node:assert/strict';
import { WindowTracker } from '../src/window-tracker.js';

function image(dx = 0, dy = 0, seed = 123) {
  const width = 192; const height = 160;
  const texture = new Uint8Array(width * height);
  for (let index = 0; index < texture.length; index++) { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; texture[index] = seed >>> 24; }
  const data = new Uint8ClampedArray(width * height * 4);
  for (let row = 0; row < height; row++) for (let column = 0; column < width; column++) {
    const value = texture[((row - dy + height) % height) * width + (column - dx + width) % width];
    data.set([value, value, value, 255], 4 * (row * width + column));
  }
  return { width, height, data };
}
const rectangle = { x: 32, y: 32, width: 128, height: 64 };

function rigidImage(dx, dy, angle, width = 256, height = 192) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let row = 0; row < height; row++) for (let column = 0; column < width; column++) {
    const shiftedX = column + 0.5 - width / 2 - dx; const shiftedY = row + 0.5 - height / 2 - dy;
    const sourceX = Math.cos(angle) * shiftedX + Math.sin(angle) * shiftedY;
    const sourceY = -Math.sin(angle) * shiftedX + Math.cos(angle) * shiftedY;
    const value = 125 + 28 * Math.sin(sourceX * 0.31 + sourceY * 0.19) + 32 * Math.cos(sourceX * 0.13 - sourceY * 0.37) +
      25 * Math.sin(sourceX * 0.051 + sourceY * 0.11) + 18 * Math.cos(sourceX * 0.43 + sourceY * 0.07);
    data.set([value, value, value, 255], 4 * (row * width + column));
  }
  return { width, height, data };
}

test('window tracking iterates translation and rotation and composes camera poses', () => {
  const tracker = new WindowTracker();
  const window = { x: 32, y: 32, width: 192, height: 128 };
  tracker.process(rigidImage(0, 0, 0), 0, window);
  for (const [index, dx, dy, degrees] of [[1, 4, -3, 2], [2, 7, 1, 3], [3, 2, 4, -1]]) {
    const angle = degrees * Math.PI / 180;
    const result = tracker.process(rigidImage(dx, dy, angle), index, window);
    assert.equal(result.success, true, JSON.stringify(result));
    assert.ok(Math.abs(result.raw.rotation + angle) < 0.15 * Math.PI / 180, JSON.stringify(result.raw));
    assert.ok(Math.abs(result.raw.x + Math.cos(angle) * dx + Math.sin(angle) * dy) < 0.35);
    assert.ok(Math.abs(result.raw.y - Math.sin(angle) * dx + Math.cos(angle) * dy) < 0.35);
    assert.ok(result.timing.refinementMs > 0);
  }
});

test('window tracking measures rectangular translation and accumulates inverse camera movement', () => {
  const tracker = new WindowTracker();
  assert.equal(tracker.process(image(), 10, rectangle).initial, true);
  const result = tracker.process(image(5, -3), 11, rectangle);
  assert.equal(result.success, true);
  assert.ok(Math.abs(result.raw.x + 5) < 0.15);
  assert.ok(Math.abs(result.raw.y - 3) < 0.15);
  const next = tracker.process(image(2, 1), 12, rectangle);
  assert.equal(next.success, true);
  assert.ok(Math.abs(next.raw.x + 2) < 0.2);
  assert.ok(Math.abs(next.raw.y + 1) < 0.2);
  assert.ok(next.timing.totalMs >= next.timing.sampleMs);
});

test('large window refinement evaluates a bounded seed when the FFT shift exceeds the search radius', () => {
  const tracker = new WindowTracker();
  const window = { x: 32, y: 32, width: 1024, height: 1024 };
  tracker.process(rigidImage(0, 0, 0, 1088, 1088), 0, window, 4);
  const result = tracker.process(rigidImage(8, 8, 0, 1088, 1088), 1, window, 4);
  assert.equal(result.success, false);
  assert.ok(result.score > -1, JSON.stringify(result));
  assert.ok(Math.abs(result.dx) <= 4 && Math.abs(result.dy) <= 4);
});

test('window tracking rejects missing structure, invalid pixels, frame gaps and unrelated appearance', () => {
  const flat = image(); flat.data.fill(255);
  assert.equal(new WindowTracker().process(flat, 0, rectangle).success, false);
  const invalid = image(); invalid.data[4 * (40 * invalid.width + 40) + 3] = 0;
  assert.throws(() => new WindowTracker().process(invalid, 0, rectangle), /ungueltige/);
  const tracker = new WindowTracker(); tracker.process(image(), 0, rectangle);
  assert.throws(() => tracker.process(image(), 2, rectangle), /Framefolge/);
  const rejected = tracker.process(image(0, 0, 999), 1, rectangle);
  assert.equal(rejected.success, false);
  assert.match(rejected.reason, /Korrelation zu niedrig/);
  assert.match(rejected.reason, /NCC .*FFT-Start: .*PSR/);
  assert.throws(() => tracker.process(image(), 1, rectangle), /verloren/);
  tracker.reset(); assert.equal(tracker.process(image(), 0, rectangle).success, true);
});