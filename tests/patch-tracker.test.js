import test from 'node:test';
import assert from 'node:assert/strict';
import { PatchTracker } from '../src/patch-tracker.js';
import { createPatchMask, MASK_FORBIDDEN, MASK_SEARCH, paintPatchMask, patchAllowed, validatePointsAgainstMask } from '../src/patch-mask.js';
import { bestLocations } from '../src/webgpu-patch-tracker.js';

test('GPU score layout selects and refines one location per patch', () => {
  const scores = new Float32Array(18).fill(-1);
  scores[5] = 0.8; scores[4] = 0.6; scores[6] = 0.7; scores[2] = 0.6; scores[8] = 0.7;
  scores[9 + 4] = 0.9; scores[9 + 3] = 0.8; scores[9 + 5] = 0.8;
  const result = bestLocations(scores, [{ x: 20, y: 30 }, { x: 50, y: 60 }], 1);
  assert.equal(result[0].score, scores[5]);
  assert.ok(result[0].x > 21 && result[0].y > 30);
  assert.deepEqual(result[1], { x: 50, y: 60, score: scores[13] });
});

test('patch mask limits new centers and rejects any footprint touching forbidden terrain', () => {
  const mask = createPatchMask(160, 120, 4);
  assert.equal(patchAllowed(mask, 40, 40, 16, true), false);
  assert.equal(patchAllowed(null, 40, 40, 16, true), true);
  paintPatchMask(mask, 40, 40, 18, MASK_SEARCH);
  assert.equal(patchAllowed(mask, 40, 40, 16, true), true);
  assert.equal(patchAllowed(mask, 100, 80, 16, true), false);
  paintPatchMask(mask, 48, 40, 3, MASK_FORBIDDEN);
  assert.equal(patchAllowed(mask, 40, 40, 16, true), false);
  const points = [{ x: 40, y: 40 }, { x: 100, y: 80 }];
  assert.equal(validatePointsAgainstMask(points, mask, 16), 1);
  assert.deepEqual(points.map(point => point.maskValid), [false, true]);
});

test('patch tracker explains when no green search area is marked', () => {
  const tracker = new PatchTracker();
  const width = 96;
  const height = 96;
  const image = { width, height, data: new Uint8ClampedArray(width * height * 4) };
  const result = tracker.process(image, 0, { patchSize: 24, threshold: 16, patchMask: createPatchMask(width, height) });
  assert.equal(result.success, false);
  assert.match(result.reason, /Keine gruene Patch-Suchflaeche/);
});

test('red contact at every patch edge and corner is forbidden', () => {
  for (const [redX, redY] of [[32, 40], [48, 40], [40, 32], [40, 48], [32, 32], [48, 48]]) {
    const mask = createPatchMask(80, 80, 1);
    mask.data[redY * mask.width + redX] = MASK_FORBIDDEN;
    assert.equal(patchAllowed(mask, 40, 40, 16), false, `red contact at ${redX},${redY}`);
  }
  const clear = createPatchMask(80, 80, 1);
  clear.data[40 * clear.width + 49] = MASK_FORBIDDEN;
  assert.equal(patchAllowed(clear, 40, 40, 16), true);
});

function texturedImage(width, height, shiftX = 0, shiftY = 0, lineAngle = null) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      const sourceX = px - shiftX;
      const sourceY = py - shiftY;
      const distance = lineAngle === null ? Infinity : Math.abs((sourceX - width / 2) * Math.cos(lineAngle) + (sourceY - height / 2) * Math.sin(lineAngle));
      const value = lineAngle !== null ? (distance < 3 ? 20 : 220) :
        128 + 55 * Math.sin(sourceX * 0.17) + 45 * Math.cos(sourceY * 0.13) + 25 * Math.sin((sourceX + sourceY) * 0.31);
      const index = 4 * (py * width + px);
      data[index] = data[index + 1] = data[index + 2] = Math.max(0, Math.min(255, value));
      data[index + 3] = 255;
    }
  }
  return { width, height, data };
}

test('patch tracker searches only green mask cells and excludes red footprints', () => {
  const mask = createPatchMask(192, 144, 4);
  for (let row = 0; row < mask.height; row++) {
    for (let col = 0; col < mask.width / 2; col++) mask.data[row * mask.width + col] = MASK_SEARCH;
  }
  mask.revision++;
  const tracker = new PatchTracker();
  const greenOnly = tracker.process(texturedImage(192, 144), 0, { patchSize: 32, threshold: 16, patchMask: mask });
  assert.ok(greenOnly.points.length >= 6);
  assert.ok(greenOnly.points.every(point => point.x < 96));
  paintPatchMask(mask, 48, 72, 32, MASK_FORBIDDEN);
  const excluded = tracker.process(texturedImage(192, 144), 0, { patchSize: 32, threshold: 16, patchMask: mask });
  assert.ok(excluded.points.every(point => Math.hypot(point.x - 48, point.y - 72) > 16));
});

function lowNoiseImage(width, height) {
  const data = new Uint8ClampedArray(width * height * 4);
  let state = 0x12345678;
  for (let index = 0; index < width * height; index++) {
    state = (1664525 * state + 1013904223) >>> 0;
    const value = 128 + (state % 7) - 3;
    data[4 * index] = data[4 * index + 1] = data[4 * index + 2] = value;
    data[4 * index + 3] = 255;
  }
  return { width, height, data };
}

test('image patches reject one-dimensional structure and select textured regions', () => {
  for (const degrees of [0, 30, 70, 90]) {
    const tracker = new PatchTracker();
    const rejected = tracker.process(texturedImage(192, 144, 0, 0, degrees * Math.PI / 180), 0, { patchSize: 32, threshold: 16 });
    assert.equal(rejected.success, false, `${degrees} degree line`);
  }
  const tracker = new PatchTracker();
  const accepted = tracker.process(texturedImage(192, 144), 0, { patchSize: 32, threshold: 16 });
  assert.equal(accepted.success, true, accepted.reason);
  assert.ok(accepted.points.length >= 12, `${accepted.points.length} patches`);
});

test('large patches do not turn low-level image noise into structure', () => {
  for (const patchSize of [32, 112, 192]) {
    const tracker = new PatchTracker();
    const detection = tracker.process(lowNoiseImage(384, 320), 0, { patchSize, threshold: 16 });
    assert.equal(detection.success, false, `${patchSize}px selected ${detection.points.length} noise patches`);
  }
});

test('image patches retain identities and track consecutive translation', () => {
  const tracker = new PatchTracker();
  const first = tracker.process(texturedImage(192, 144), 10, { patchSize: 32, patchSearchRadius: 12, threshold: 16 });
  const second = tracker.process(texturedImage(192, 144, 5, -3), 11, { patchSize: 32, patchSearchRadius: 12, threshold: 16 });
  assert.equal(second.success, true, second.reason);
  assert.ok(second.points.length >= first.points.length * 0.8);
  const byIdentity = new Map(first.points.map(point => [`${point.col},${point.row}`, point]));
  const errors = second.points.map(point => {
    const origin = byIdentity.get(`${point.col},${point.row}`);
    return Math.hypot(point.x - origin.x - 5, point.y - origin.y + 3);
  });
  assert.ok(errors.sort((a, b) => a - b)[Math.floor(errors.length / 2)] < 0.75, `median error ${errors}`);
});

test('image patches replenish newly visible image areas in the shared coordinate system', () => {
  const tracker = new PatchTracker();
  const first = tracker.process(texturedImage(256, 176), 0, { patchSize: 32, patchSearchRadius: 48, threshold: 16 });
  const identities = new Set(first.points.map(point => `${point.col},${point.row}`));
  const second = tracker.process(texturedImage(256, 176, -36, 0), 1, { patchSize: 32, patchSearchRadius: 48, threshold: 16 });
  const additions = second.points.filter(point => !identities.has(`${point.col},${point.row}`));
  assert.ok(additions.length > 0, 'expected replenished patches');
  assert.ok(additions.some(point => point.x > 200), 'expected a replenished patch near the newly visible right edge');
  assert.ok(second.points.length >= first.points.length * 0.9, `${first.points.length} -> ${second.points.length}`);
  assert.equal(second.generation, first.generation);
});

test('image patches start a new segment after forward gaps and reject backward jumps', () => {
  const tracker = new PatchTracker();
  const first = tracker.process(texturedImage(192, 144), 10, { patchSize: 32, patchSearchRadius: 16, threshold: 16 });
  const forward = tracker.process(texturedImage(192, 144, 5, 0), 15, { patchSize: 32, patchSearchRadius: 16, threshold: 16 });
  const backward = tracker.process(texturedImage(192, 144), 12, { patchSize: 32, patchSearchRadius: 16, threshold: 16 });
  const resumed = tracker.process(texturedImage(192, 144, 8, 0), 18, { patchSize: 32, patchSearchRadius: 16, threshold: 16 });
  assert.equal(forward.success, true, forward.reason);
  assert.equal(backward.interrupted, true);
  assert.equal(backward.points.length, 0);
  assert.equal(resumed.success, true, resumed.reason);
  assert.ok(forward.generation > first.generation);
  assert.ok(resumed.generation > forward.generation);
});

test('image patches can check the current frame repeatedly without changing settings', () => {
  const tracker = new PatchTracker();
  const image = texturedImage(192, 144);
  const first = tracker.process(image, 10, { patchSize: 32, patchSearchRadius: 16, threshold: 16 });
  const repeated = tracker.process(image, 10, { patchSize: 32, patchSearchRadius: 16, threshold: 16 });
  assert.equal(first.success, true, first.reason);
  assert.equal(repeated.success, true, repeated.reason);
  assert.ok(repeated.points.length >= first.points.length * 0.9);
  assert.equal(repeated.generation, first.generation);
  assert.equal(repeated.interrupted, undefined);
});