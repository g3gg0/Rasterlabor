import test from 'node:test';
import assert from 'node:assert/strict';
import { frameGeometry, localSelectionMask, localSelectionDistance, localSelectionSupport, nearestPathEntry, applyPixelMask, edgeFeatherMask, applyEdgeFeather, accumulateFrame, approximateTopFrames, averagedFrames, sharpestFramesFirst, sparsePathFrames, pointBounds, evenlySpaced } from '../src/path-support.js';
import { contextImage } from '../src/context-tracker.js';

test('point bounds handle tracking paths larger than the call argument limit', () => {
  const points = Array.from({ length: 100_000 }, (_, index) => ({ x: index - 40_000, y: 50_000 - index }));
  assert.deepEqual(pointBounds([points], { x: 0, y: 0 }),
    { minX: -40_000, minY: -49_999, maxX: 59_999, maxY: 50_000 });
});

test('long diagnostic paths are sampled evenly with bounded drawing work', () => {
  assert.deepEqual(evenlySpaced(Array.from({ length: 100_000 }, (_, index) => index), 5),
    [0, 25_000, 50_000, 74_999, 99_999]);
});

test('sparse path frames retain endpoints and spatially distant coverage', () => {
  const geometry = (frame, x, y) => ({ entry: { frame }, width: 10, height: 10,
    world: () => ({ x, y }) });
  const frames = [geometry(0, 0, 0), geometry(1, 1, 0), geometry(2, 2, 0),
    geometry(3, 10, 10), geometry(4, 3, 0), geometry(5, 4, 0)];
  assert.deepEqual(sparsePathFrames(frames, 3).map(item => item.entry.frame), [0, 3, 5]);
  assert.deepEqual(sparsePathFrames(frames, 1).map(item => item.entry.frame), [5]);
});

test('local selection support agrees with the masked matcher sample grid', () => {
  const width = 2182, height = 3742;
  const geometry = { width, height, local: point => point };
  const baseMask = localSelectionMask(geometry, { x: 1100, y: 1900 }, 0.7);
  const allowed = (x, y) => x > 500 && y > 700;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let row = 0; row < height; row++) for (let column = 0; column < width; column++) {
    data[(row * width + column) * 4 + 3] = allowed(column, row) ? 255 : 0;
  }
  for (const point of [{ x: 1100, y: 1900 }, { x: 1100, y: 0 }, { x: -100, y: 1900 }]) {
    const mask = localSelectionMask(geometry, point, 0.4, baseMask);
    const level = contextImage({ width, height, data }, mask).levels[0];
    const stride = Math.ceil(Math.sqrt(width * height / 6000));
    let expected = 0;
    for (let row = 2; row < height - 2; row += stride) for (let column = 2; column < width - 2; column += stride) {
      if (level.gray[row * width + column] >= 0) expected++;
    }
    assert.equal(localSelectionSupport(geometry, point, 0.4, baseMask, allowed), expected);
    if (point.y === 1900 && point.x === 1100) assert.ok(expected >= 128);
    else assert.equal(expected, 0);
  }
});

test('local refit mask follows the selected point in frame coordinates', () => {
  const geometry = { width: 100, height: 80, local: point => ({ x: point.x - 10, y: point.y - 20 }) };
  const mask = localSelectionMask(geometry, { x: 60, y: 60 }, 0.5);
  const valueAt = (x, y) => mask.data[Math.floor(y / mask.cellSize) * mask.width + Math.floor(x / mask.cellSize)];
  assert.equal(valueAt(50, 40), 1);
  assert.equal(valueAt(5, 5), 0);
});

test('local selection distance measures the gap outside a frame', () => {
  const geometry = { width: 100, height: 80, local: point => point };
  assert.equal(localSelectionDistance(geometry, { x: 40, y: 30 }), 0);
  assert.equal(localSelectionDistance(geometry, { x: 106, y: 88 }), 10);
});

test('frame edges fade smoothly over ten percent of image width', () => {
  const rgba = new Uint8ClampedArray(20 * 8 * 4).fill(255);
  applyEdgeFeather(rgba, 20, 8);
  const alpha = (x, y) => rgba[(y * 20 + x) * 4 + 3];
  assert.equal(alpha(0, 4), 40);
  assert.equal(alpha(1, 4), 215);
  assert.equal(alpha(2, 4), 255);
  assert.equal(alpha(10, 0), 40);
  assert.equal(alpha(10, 1), 215);
  assert.equal(alpha(10, 4), 255);
  const sum = new Float32Array(4);
  accumulateFrame(sum, new Uint8ClampedArray([120, 80, 40, 128]));
  assert.deepEqual([...averagedFrames(sum, true)], [120, 80, 40, 128]);
});

test('frame edge fade accepts zero and custom widths', () => {
  const none = new Uint8ClampedArray(20 * 8 * 4).fill(255);
  applyEdgeFeather(none, 20, 8, 0);
  assert.ok(none.every(value => value === 255));
  const wide = new Uint8ClampedArray(20 * 20 * 4).fill(255);
  applyEdgeFeather(wide, 20, 20, 0.25);
  const alpha = (x, y) => wide[(y * 20 + x) * 4 + 3];
  assert.ok(alpha(2, 10) < 255);
  assert.equal(alpha(5, 10), 255);
  const landscape = new Uint8ClampedArray(20 * 8 * 4).fill(255);
  applyEdgeFeather(landscape, 20, 8, 0.5);
  const landscapeAlpha = (x, y) => landscape[(y * 20 + x) * 4 + 3];
  assert.equal(landscapeAlpha(10, 3), 255);
  assert.equal(landscapeAlpha(10, 4), 255);
  const landscapeMask = edgeFeatherMask(20, 8, () => true, 0.5);
  assert.equal(landscapeMask[3 * 20 + 10], 255);
  assert.equal(landscapeMask[4 * 20 + 10], 255);
});

test('frame edge fade follows internal validity boundaries', () => {
  const width = 20, height = 20;
  const weights = edgeFeatherMask(width, height, (x, y) => x < 10 || y < 10, 0.25);
  const weight = (x, y) => weights[y * width + x];
  assert.equal(weight(15, 15), 0);
  assert.ok(weight(9, 15) < weight(5, 15));
  assert.ok(weight(15, 9) < weight(15, 5));
  assert.equal(weight(5, 5), 255);
});

test('sharpest frames are ordered once with deterministic ties and unknown scores last', () => {
  const geometries = [
    { entry: { frame: 0 } }, { entry: { frame: 3, sharpness: { score: 10 } } },
    { entry: { frame: 2, sharpness: { score: 20 } } }, { entry: { frame: 1, sharpness: { score: 10 } } },
    { entry: { frame: 4, sharpness: { score: NaN } } }
  ];
  assert.deepEqual(sharpestFramesFirst(geometries).map(geometry => geometry.entry.frame), [2, 1, 3, 0, 4]);
  assert.equal(geometries[0].entry.frame, 0);
});

test('approximate top frames retain sharp coverage across space without redundant decoding', () => {
  const rectangle = (frame, score, left, right) => ({
    entry: { frame, sharpness: { score } },
    corners: [{ x: left, y: 0 }, { x: right, y: 0 }, { x: right, y: 10 }, { x: left, y: 10 }],
    supports: point => point.x >= left && point.x < right && point.y >= 0 && point.y < 10
  });
  const frames = [rectangle(2, 10, 0, 10), rectangle(3, 5, 8, 18), rectangle(0, 30, 0, 10), rectangle(1, 20, 0, 10)];
  assert.deepEqual(approximateTopFrames(frames, 2, null, { x: 9, y: 5 }, 2).map(frame => frame.entry.frame), [0, 1, 3]);
});

test('default spatial approximation ignores shifts smaller than the feather width', () => {
  const rectangle = (frame, score, left) => ({
    width: 1000, entry: { frame, sharpness: { score } },
    corners: [{ x: left, y: 0 }, { x: left + 1000, y: 0 }, { x: left + 1000, y: 500 }, { x: left, y: 500 }],
    supports: point => point.x >= left && point.x < left + 1000 && point.y >= 0 && point.y < 500
  });
  const frames = [rectangle(0, 30, 0), rectangle(1, 20, 10), rectangle(2, 10, 30)];
  assert.deepEqual(approximateTopFrames(frames, 1, null, { x: 100, y: 100 }).map(frame => frame.entry.frame), [0]);
});

test('top n weighted coverage blends feathered edges before reaching the limit', () => {
  const frames = [
    new Uint8ClampedArray([200, 0, 0, 255, 0, 0, 0, 0, 100, 0, 0, 128]),
    new Uint8ClampedArray([100, 0, 0, 255, 80, 0, 0, 255, 200, 0, 0, 255]),
    new Uint8ClampedArray([0, 0, 0, 255, 40, 0, 0, 255, 0, 0, 0, 255])
  ];
  for (const limit of [1, 2, 4, 0]) {
    const sum = new Float32Array(12), counts = new Uint32Array(3);
    for (const frame of frames) accumulateFrame(sum, frame, counts, limit);
    const output = averagedFrames(sum);
    assert.equal(output[0], limit === 1 ? 200 : limit === 2 ? 150 : 100);
    assert.equal(output[4], limit === 1 ? 80 : 60);
    assert.equal(output[8], limit === 1 ? 150 : limit === 2 ? 125 : 100);
    assert.deepEqual([...counts], limit === 0 ? [3, 2, 3] : [Math.min(limit, 3), Math.min(limit, 2), limit === 1 ? 2 : 3]);
  }
});

test('paint mask removes excluded pixels before averaging', () => {
  const rgba = new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255]);
  applyPixelMask(rgba, 2, x => x === 1);
  assert.deepEqual([...rgba], [10, 20, 30, 0, 40, 50, 60, 255]);
});

test('equal frame weights survive thousands of frames and transparent holes', () => {
  const sum = new Float32Array(8);
  for (let i = 0; i < 1000; i++) accumulateFrame(sum, new Uint8ClampedArray([i % 2 ? 200 : 100, 60, 40, 255, 90, 90, 90, i % 2 ? 255 : 0]));
  assert.deepEqual([...averagedFrames(sum)], [150, 60, 40, 255, 90, 90, 90, 255]);
});

test('rotated translated frames select actual valid pixels, using measured pose', () => {
  const maps = { outputWidth: 4, outputHeight: 2, origin: [10, 20], valid: new Uint8Array(8).fill(1) };
  maps.valid[1] = 0;
  const entry = { mode: 'window', raw: { x: 30, y: 40, rotation: Math.PI / 2 }, pose: { x: 1000, y: 1000, rotation: 0 } };
  const g = frameGeometry(entry, {}, maps);
  assert.deepEqual(g.world(2, 1), { x: 1000, y: 1000 });
  assert.equal(g.supports(g.world(0.5, 0.5)), true);
  assert.equal(g.supports(g.world(1.5, 0.5)), false);
  assert.equal(g.supports(g.world(-0.5, 0.5)), false);
  assert.equal(g.supports(g.world(4.5, 0.5)), false);
  assert.equal(g.supports(g.world(0.5, 0.5), () => false), false);
});

test('patch coordinates include cropped map origin and source image centre', () => {
  const maps = { outputWidth: 4, outputHeight: 2, origin: [10, 20], valid: new Uint8Array(8).fill(1) };
  const entry = { raw: { x: 5, y: -3, rotation: 0 } };
  const g = frameGeometry(entry, { width: 100, height: 80 }, maps);
  assert.deepEqual(g.world(0, 0), { x: -35, y: -23 });
  assert.equal(g.supports({ x: -34.5, y: -22.5 }), true);
  assert.equal(frameGeometry({ ...entry, success: false }, {}, maps), null);
});

test('hover uses screen distance and ignores missing poses', () => {
  const entries = [{ pose: null }, { frame: 1, pose: { x: 2, y: 3 } }];
  const project = p => ({ x: p.x * 10, y: p.y * 10 });
  assert.equal(nearestPathEntry(entries, project, { x: 22, y: 31 }), entries[1]);
  assert.equal(nearestPathEntry(entries, project, { x: 80, y: 90 }), null);
});
