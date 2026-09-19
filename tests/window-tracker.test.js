import test from 'node:test';
import assert from 'node:assert/strict';
import { WindowTracker } from '../src/window-tracker.js';
import { ContextTracker, contextImage, registerOverlap, registerOverlapAsync, chooseRegistrationBackend, consensusPose, spatialConsensusPose, confirmSpatialClosure, cycleErrorMetrics, contextSearchRadii, selectContextReferences } from '../src/context-tracker.js';

test('GPU registration is selected only for matching results and lower measured latency', () => {
  const cpu = { accepted: true, pose: { x: 1, y: 2, rotation: 0 } };
  assert.equal(chooseRegistrationBackend(cpu, cpu, 30, 10, 4096, 3072).backend, 'WebGPU');
  assert.equal(chooseRegistrationBackend(cpu, cpu, 10, 30, 4096, 3072).backend, 'CPU');
  assert.equal(chooseRegistrationBackend(cpu, { accepted: false }, 30, 10, 4096, 3072).backend, 'CPU');
  assert.equal(chooseRegistrationBackend(cpu, { ...cpu, pose: { ...cpu.pose, rotation: 0.001 } }, 30, 10, 4096, 3072).backend, 'CPU');
});

test('async registration follows the identical CPU search and acceptance decisions', async () => {
  const source = contextImage(rigidImage(0, 0, 0)); const current = contextImage(rigidImage(4, -3, 0.02));
  const args = [current, source, { x: -2, y: 2, rotation: -0.01 }, { x: 0, y: 0, rotation: 0 }];
  const actual = await registerOverlapAsync({ execute: async step => step.cpu?.() }, ...args);
  assert.deepEqual(actual, registerOverlap(...args));
});

test('GPU errors fall back to CPU without losing references or retrying the broken backend', async () => {
  for (const failureStage of ['image', 'registration']) {
    let attempts = 0;
    const gpu = { reset() { this.failure = ''; this.cacheBytes = 0; }, disable(error) { this.failure = error.message; },
      async image(source, mask) { if (failureStage === 'image') { attempts++; throw new Error('GPU buffer limit'); } return contextImage(source, mask); },
      async execute() { attempts++; throw new Error('GPU device lost'); } };
    const tracker = new ContextTracker(1024 * 1024, gpu);
    const options = { useWebGpu: true, contextRecent: 2, contextSpatial: 2, contextRadius: 16, contextAngle: 1 };
    const source = rigidImage(0, 0, 0); const pose = { x: 0, y: 0, rotation: 0 };
    for (let frame = 0; frame < 3; frame++) {
      await tracker.begin(source, frame, { ...pose, x: frame === 2 ? 2 : 0 }, options);
      const result = tracker.finish();
      if (frame === 2) { assert.equal(result.applied, true); assert.ok(Math.abs(result.pose.x) < 0.2); assert.match(result.fallback, /GPU/); }
    }
    assert.equal(attempts, 1);
    tracker.reset(); assert.equal(gpu.failure, ''); assert.equal(tracker.cacheBytes, 0);
    await tracker.begin(source, 0, pose, { ...options, useWebGpu: false });
    assert.equal(tracker.finish().accelerator, 'CPU'); assert.equal(attempts, 1);
  }
});

test('overlap registration refines masked native image translation and rotation', () => {
  const source = contextImage(rigidImage(0, 0, 0));
  const angle = 1.5 * Math.PI / 180;
  const current = contextImage(rigidImage(34, -12, angle));
  const expected = { x: -Math.cos(angle) * 34 + Math.sin(angle) * 12,
    y: Math.sin(angle) * 34 + Math.cos(angle) * 12, rotation: -angle };
  const result = registerOverlap(current, source, { x: expected.x + 3, y: expected.y - 2, rotation: expected.rotation + 0.003 },
    { x: 0, y: 0, rotation: 0 }, { radius: 16, angle: 1 });
  assert.equal(result.accepted, true, JSON.stringify(result));
  assert.ok(Math.hypot(result.pose.x - expected.x, result.pose.y - expected.y) < 0.3, JSON.stringify(result));
  assert.ok(Math.abs(result.pose.rotation - expected.rotation) < 0.001);
  const unrelated = registerOverlap(contextImage(image(0, 0, 777)), contextImage(image()), { x: 0, y: 0, rotation: 0 }, { x: 0, y: 0, rotation: 0 });
  assert.equal(unrelated.accepted, false);
});

test('iterative coarse offsets recover a closure outside the local refinement basin', () => {
  const source = contextImage(sceneCrop());
  const current = contextImage(sceneCrop(50, -30));
  const result = registerOverlap(current, source, { x: 0, y: 0, rotation: 0 }, { x: 0, y: 0, rotation: 0 },
    { radius: 80, angle: 1, coarseStep: 8 });
  assert.equal(result.accepted, true, JSON.stringify(result));
  assert.ok(Math.hypot(result.pose.x - 50, result.pose.y + 30) < 1, JSON.stringify(result));
});

test('coarse offsets recover when the predicted pose has no initial overlap', () => {
  const source = contextImage(rigidImage(0, 0, 0));
  const current = contextImage(rigidImage(0, 0, 0));
  const result = registerOverlap(current, source, { x: 250, y: 0, rotation: 0 }, { x: 0, y: 0, rotation: 0 },
    { radius: 320, angle: 1, coarseStep: 32, coarseRadiusFactor: 1.05 });
  assert.equal(result.accepted, true, JSON.stringify(result));
  assert.ok(Math.hypot(result.pose.x, result.pose.y) < 1, JSON.stringify(result));
});

test('coarse closure search keeps its evaluation count bounded at full-frame radii', () => {
  const source = contextImage(sceneCrop());
  const current = contextImage(sceneCrop(50, -30));
  const result = registerOverlap(current, source, { x: 0, y: 0, rotation: 0 }, { x: 0, y: 0, rotation: 0 },
    { radius: 4096, angle: 1, coarseStep: 32 });
  assert.ok(result.evaluated < 500, JSON.stringify(result));
});

test('context consensus requires multiple consistent references and rejects split votes', () => {
  const prediction = { x: 4, y: 5, rotation: 0 };
  const match = (frame, x) => ({ accepted: true, frame, pose: { x, y: 0, rotation: 0 } });
  assert.equal(consensusPose(prediction, [match(1, 0)], 256, 192).applied, false);
  const result = consensusPose(prediction, [match(1, 0), match(2, 0.2), match(3, 40)], 256, 192);
  assert.equal(result.applied, true); assert.deepEqual(result.inliers, [1, 2]);
  assert.equal(consensusPose(prediction, [match(1, 0), match(2, 0.2), match(3, 40), match(4, 40.1)], 256, 192).applied, false);
});

test('cycle diagnostics separate translation, rotation and distributed image error', () => {
  const metrics = cycleErrorMetrics({ x: 0, y: 0, rotation: 0 }, { x: 3, y: 4, rotation: Math.PI / 180 },
    [{ x: 0, y: 0 }, { x: 100, y: 0 }, { x: -100, y: 0 }]);
  assert.equal(metrics.translation, 5);
  assert.equal(metrics.rotationDegrees, 1);
  assert.equal(metrics.samples, 3);
  assert.ok(metrics.p95 > metrics.median && metrics.rms >= metrics.median);
});

test('conditional spatial consensus requires a majority and reports independent reference groups', () => {
  const prediction = { x: 100, y: 20, rotation: 0 };
  const match = (frame, x, conditional = true) => ({ frame, kind: 'spatial', accepted: !conditional,
    conditionallyAccepted: conditional, pose: { x, y: 0, rotation: 0 } });
  assert.equal(spatialConsensusPose(prediction, [match(10, 0)], 256, 192).supported, false);
  const adjacent = spatialConsensusPose(prediction, [match(10, 0), match(11, 0.2), match(12, 40)], 256, 192);
  assert.equal(adjacent.supported, true); assert.deepEqual(adjacent.referenceGroups, [[10, 11]]);
  const independent = spatialConsensusPose(prediction, [match(10, 0), match(30, 0.2), match(50, 40)], 256, 192);
  assert.equal(independent.supported, true); assert.deepEqual(independent.referenceGroups, [[10], [30]]);
});

test('one reference group needs consistent evidence from consecutive current frames', () => {
  const spatial = frame => ({ supported: true, pose: { x: 90, y: 20, rotation: 0 }, inliers: [frame, frame + 1],
    referenceGroups: [[frame, frame + 1]] });
  const prediction = { x: 100, y: 20, rotation: 0 };
  const first = confirmSpatialClosure(null, 100, prediction, spatial(10), 256, 192);
  assert.equal(first.confirmed, false);
  const second = confirmSpatialClosure(first.evidence, 101, { x: 101, y: 20, rotation: 0 },
    { ...spatial(11), pose: { x: 91.2, y: 20, rotation: 0 } }, 256, 192);
  assert.equal(second.confirmed, true); assert.equal(second.diagnostics.consecutiveFrames, true);
  const unrelated = confirmSpatialClosure(first.evidence, 101, prediction, spatial(50), 256, 192);
  assert.equal(unrelated.confirmed, false); assert.equal(unrelated.diagnostics.relatedReferences, false);
});

test('tracker applies a conditional spatial closure only after consecutive-frame confirmation', () => {
  const tracker = new ContextTracker(); tracker.remember = () => {};
  const match = (frame, x) => ({ frame, kind: 'spatial', accepted: false, conditionallyAccepted: true,
    pose: { x, y: 20, rotation: 0 } });
  const finish = (frame, prediction, matches) => {
    tracker.pending = { prediction, matches, selected: matches, current: { width: 256, height: 192, bytes: 0 },
      metadata: { frame }, options: {}, milliseconds: 0, pyramidMs: 0, registrationMs: 0, cacheHits: 0, cacheMisses: 0,
      cacheEvictions: 0, pyramidProfile: {}, accelerators: new Set(['CPU']) };
    return tracker.finish();
  };
  const first = finish(100, { x: 100, y: 20, rotation: 0 }, [match(10, 90), match(11, 90.2)]);
  assert.equal(first.applied, false); assert.equal(first.spatialConfirmation.consecutiveFrames, false);
  const second = finish(101, { x: 101, y: 20, rotation: 0 }, [match(11, 91.1), match(12, 91.3)]);
  assert.equal(second.applied, true); assert.equal(second.confirmation, 'consecutive-frames');
  assert.deepEqual(second.inliers, [11, 12]); assert.equal(second.loopClosure.anchorFrame, 12);
});

test('context diagnostics retain each search and backward result independently of later poses', async () => {
  const tracker = new ContextTracker();
  const prediction = { x: 10, y: 0, rotation: 0 };
  const referencePose = { x: 0, y: 0, rotation: 0 };
  tracker.pending = { current: { width: 256, height: 192 }, prediction,
    options: { contextRadius: 16, contextAngle: 1 }, matches: [], accelerators: new Set(), milliseconds: 0, registrationMs: 0 };
  let calls = 0;
  tracker.register = async () => ++calls === 1 ? { accepted: false, reason: 'Korrelation', score: 0.3, accelerator: 'CPU' } :
    { accepted: true, pose: { x: calls === 2 ? 2 : 4, y: 0, rotation: 0 }, score: 0.99, accelerator: 'CPU' };
  await tracker.match({ reference: { frame: 1, pose: referencePose }, kind: 'spatial', cells: [1] }, {});
  const match = tracker.pending.matches[0];
  assert.deepEqual(match.attempts.map(attempt => attempt.searchRadius), [16, 32]);
  assert.equal(match.attempts[1].accepted, true);
  assert.equal(match.backward.accepted, true);
  assert.equal(match.reverseDistance, 4);
  assert.equal(match.reason, 'Rueckwaertspruefung bedingt');
  assert.equal(match.conditionallyAccepted, true);
  assert.equal(match.cycleQuality, 'conditional');
  assert.equal(match.cycleError.translation, 4);
  referencePose.x = 99; prediction.x = 99;
  assert.equal(match.referencePose.x, 0);
  assert.equal(match.prediction.x, 10);
  const saved = JSON.parse(JSON.stringify(match));
  assert.deepEqual(saved.attempts, match.attempts);
  assert.deepEqual(saved.backward, match.backward);
});

test('spatial closure search doubles its radius until it covers large drift', () => {
  const radii = contextSearchRadii(256, 4389, 7937, 'spatial');
  assert.deepEqual(radii.slice(0, 5), [256, 512, 1024, 2048, 4096]);
  assert.ok(radii.some(radius => radius >= Math.hypot(620, 2438)));
  assert.deepEqual(contextSearchRadii(256, 4389, 7937, 'recent'), [256]);
  const local = contextSearchRadii(128, 1920, 1080, 'spatial', 1.05);
  assert.ok(local.at(-1) >= Math.hypot(1920, 1080));
});

test('overlap registration excludes masked pixels and rejects absent support', () => {
  const source = rigidImage(0, 0, 0);
  const changed = rigidImage(0, 0, 0);
  const mask = { sourceWidth: source.width, sourceHeight: source.height, cellSize: 1, width: source.width, height: source.height,
    data: new Uint8Array(source.width * source.height) };
  for (let row = 0; row < source.height; row++) for (let column = 0; column < source.width; column++) {
    const index = row * source.width + column;
    if (column < source.width / 2) mask.data[index] = 1;
    else changed.data.fill((row * 173 + column * 29) % 256, index * 4, index * 4 + 3);
  }
  const pose = { x: 0, y: 0, rotation: 0 };
  const result = registerOverlap(contextImage(changed, mask), contextImage(source, mask), { ...pose, x: 2 }, pose);
  assert.equal(result.accepted, true, JSON.stringify(result));
  assert.ok(Math.abs(result.pose.x) < 0.2);
  mask.data.fill(0);
  assert.equal(registerOverlap(contextImage(source, mask), contextImage(source), pose, pose).accepted, false);
  const flat = { ...source, data: new Uint8ClampedArray(source.data.length).fill(255) };
  assert.equal(registerOverlap(contextImage(flat), contextImage(flat), pose, pose).accepted, false);
  assert.equal(registerOverlap(contextImage(source), contextImage(source), { ...pose, x: 1000 }, pose).accepted, false);
});

test('local overlap skips pyramid levels where the mask has too few samples', () => {
  const source = rigidImage(0, 0, 0);
  const cellSize = 4; const maskWidth = Math.ceil(source.width / cellSize); const maskHeight = Math.ceil(source.height / cellSize);
  const mask = { sourceWidth: source.width, sourceHeight: source.height, cellSize, width: maskWidth, height: maskHeight,
    data: new Uint8Array(maskWidth * maskHeight) };
  for (let row = 12; row < 36; row++) for (let column = 16; column < 48; column++) mask.data[row * maskWidth + column] = 1;
  const image = contextImage(source, mask);
  const result = registerOverlap(image, image, { x: 0, y: 0, rotation: 0 }, { x: 0, y: 0, rotation: 0 });
  assert.equal(result.accepted, true, JSON.stringify(result));
  assert.ok(result.support >= 128, JSON.stringify(result));
});

test('local overlap starts coarse recovery on the first usable masked level', () => {
  const source = rigidImage(0, 0, 0);
  const cellSize = 4; const maskWidth = Math.ceil(source.width / cellSize); const maskHeight = Math.ceil(source.height / cellSize);
  const mask = { sourceWidth: source.width, sourceHeight: source.height, cellSize, width: maskWidth, height: maskHeight,
    data: new Uint8Array(maskWidth * maskHeight) };
  for (let row = 12; row < 36; row++) for (let column = 16; column < 48; column++) mask.data[row * maskWidth + column] = 1;
  const image = contextImage(source, mask);
  const result = registerOverlap(image, image, { x: 72, y: 0, rotation: 0 }, { x: 0, y: 0, rotation: 0 },
    { radius: 96, angle: 1, coarseStep: 1, coarseRadiusFactor: 1.05 });
  assert.equal(result.accepted, true, JSON.stringify(result));
  assert.ok(Math.abs(result.pose.x) < 0.5, JSON.stringify(result));
});

test('partial overlap uses multiple matching segments without accepting the unrelated full image', () => {
  const source = image(0, 0, 123); const current = image(0, 0, 987);
  for (const [left, top] of [[32, 16], [96, 80]]) {
    for (let row = top; row < top + 64; row++) for (let column = left; column < left + 64; column++) {
      const offset = 4 * (row * source.width + column);
      current.data.set(source.data.subarray(offset, offset + 4), offset);
    }
  }
  const currentImage = contextImage(current); const sourceImage = contextImage(source);
  const prediction = { x: 0, y: 0, rotation: 0 };
  assert.equal(registerOverlap(currentImage, sourceImage, prediction, prediction).accepted, false);
  const partial = registerOverlap(currentImage, sourceImage, prediction, prediction, { partial: true });
  assert.equal(partial.accepted, true, JSON.stringify(partial));
  assert.ok(partial.support >= 128, JSON.stringify(partial));
});

test('context rejects missing references without changing the predicted pose', async () => {
  const tracker = new ContextTracker(1);
  const source = rigidImage(0, 0, 0); const pose = { x: 0, y: 0, rotation: 0 };
  const options = { contextRecent: 2, contextSpatial: 2, contextRadius: 16, contextAngle: 1 };
  await tracker.begin(source, 0, pose, options); tracker.finish();
  await tracker.begin(source, 1, pose, options);
  assert.throws(() => tracker.finish(), /fehlen/);
  await tracker.provide(0, null, 'Pausiert');
  const result = tracker.finish();
  assert.equal(result.applied, false); assert.deepEqual(result.pose, pose);
  assert.equal(result.matches[0].reason, 'Pausiert');
  await tracker.begin(source, 2, pose, options);
  await tracker.provide(0, { width: 1, height: 1 }); await tracker.provide(1, null, 'Dekodierung fehlgeschlagen');
  assert.equal(tracker.finish().applied, false);
  await assert.rejects(() => tracker.begin(source, 3, pose, { ...options, contextRecent: Infinity }), /Umfeldparameter/);
  assert.equal(tracker.pending, null);
});

test('context image cache evicts least recently used images within its byte budget', () => {
  const tracker = new ContextTracker(20);
  const pyramid = { bytes: 10 };
  tracker.remember(0, pyramid); tracker.remember(1, pyramid); tracker.remember(0, pyramid); tracker.remember(2, pyramid);
  assert.deepEqual([...tracker.cache.keys()], [0, 2]); assert.equal(tracker.cacheBytes, 20);
  tracker.remember(3, { bytes: 21 }); assert.equal(tracker.cacheBytes, 20);
  tracker.reset(); assert.equal(tracker.cacheBytes, 0);
});

test('automatic context cache fits the working set within a hard limit and respects fixed budgets', () => {
  const mebibyte = 1024 * 1024;
  const tracker = new ContextTracker();
  const options = { contextRecent: 2, contextSpatial: 4 };
  tracker.sizeCache(90 * mebibyte, options);
  assert.equal(tracker.cacheLimit, 630 * mebibyte);
  for (let frame = 0; frame < 7; frame++) tracker.remember(frame, { bytes: 90 * mebibyte });
  assert.equal(tracker.cache.size, 7);
  tracker.sizeCache(90 * mebibyte, { contextRecent: 8, contextSpatial: 8 });
  assert.equal(tracker.cacheLimit, 768 * mebibyte);
  tracker.sizeCache(90 * mebibyte, { contextRecent: 0, contextSpatial: 0 });
  assert.ok(tracker.cacheBytes <= 192 * mebibyte);
  tracker.reset(); assert.equal(tracker.cacheBytes, 0); assert.equal(tracker.cacheLimit, 192 * mebibyte);
  const fixed = new ContextTracker(20); fixed.sizeCache(90 * mebibyte, options);
  assert.equal(fixed.cacheLimit, 20);
});

test('corrected camera pose survives the next incremental step with a rotated noncentral window', () => {
  const tracker = new WindowTracker(); const source = rigidImage(0, 0, 0);
  const rectangle = { x: 20, y: 30, width: 128, height: 96 };
  tracker.process(source, 0, rectangle);
  const pose = { x: -900, y: 1300, rotation: 0.3 };
  tracker.setPose(pose, source.width, source.height);
  const result = tracker.process(source, 1, rectangle);
  assert.ok(Math.hypot(result.raw.x - pose.x, result.raw.y - pose.y) < 0.001);
  assert.ok(Math.abs(result.raw.rotation - pose.rotation) < 0.000001);
});

test('context selects sharp distributed references from full history independently of cache', async () => {
  const current = { width: 256, height: 192, pose: { x: 0, y: 0, rotation: 0 },
    coverage: new Float32Array(256).fill(1), sharpness: new Float32Array(256).fill(10) };
  const frame = (index, x, sharpness) => ({ ...current, frame: index, pose: { x, y: 0, rotation: 0 }, sharpness: new Float32Array(256).fill(sharpness) });
  const history = [frame(0, -90, 100), frame(1, 90, 80), frame(500, -88, 1), frame(1000, 0, 10), frame(1001, 1, 10)];
  const selected = selectContextReferences(current, history, 2, 2);
  assert.deepEqual(selected.map(candidate => candidate.reference.frame).sort((first, second) => first - second), [0, 1, 1000, 1001]);
  assert.equal(selectContextReferences(current, history, 0, 0).length, 0);
  const tracker = new ContextTracker(1);
  const options = { contextRecent: 2, contextSpatial: 2, contextRadius: 16, contextAngle: 1 };
  const source = rigidImage(0, 0, 0); const pose = { x: 0, y: 0, rotation: 0 };
  assert.deepEqual(await tracker.begin(source, 0, pose, options), []); tracker.finish();
  assert.equal(tracker.cache.size, 0);
  assert.deepEqual(await tracker.begin(source, 1, pose, options), [0]); await tracker.provide(0, source); tracker.finish();
  assert.deepEqual(await tracker.begin(source, 2, { ...pose, x: 2 }, options), [0, 1]);
  await tracker.provide(0, source); await tracker.provide(1, source);
  const result = tracker.finish();
  assert.equal(result.applied, true, JSON.stringify(result));
  assert.ok(Math.abs(result.pose.x) < 0.2);
  assert.equal(result.historyFrames, 3);
  tracker.reset(); assert.equal(tracker.history.length, 0);
});

test('spatial context selects the most distant frames with at least twenty percent overlap', () => {
  const base = { width: 256, height: 192, coverage: new Float32Array(256).fill(1), sharpness: new Float32Array(256).fill(10) };
  const current = { ...base, frame: 200, pose: { x: 0, y: 0, rotation: 0 } };
  const frame = (index, x, sharpness = 10) => ({ ...base, frame: index, pose: { x, y: 0, rotation: 0 },
    sharpness: new Float32Array(256).fill(sharpness) });
  const history = [frame(0, -200, 100), frame(100, -190, 1), frame(1, -180, 2), frame(2, 0, 1000), frame(199, 0, 10)];
  const selected = selectContextReferences(current, history, 1, 2);
  assert.deepEqual(selected.map(candidate => candidate.reference.frame), [199, 100, 1]);
  assert.deepEqual(selected.map(candidate => candidate.kind), ['recent', 'spatial', 'spatial']);
});

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

function sceneCrop(originX = 0, originY = 0) {
  const width = 192; const height = 160; const data = new Uint8ClampedArray(width * height * 4);
  for (let row = 0; row < height; row++) for (let column = 0; column < width; column++) {
    let value = 0;
    for (let sampleY = -2; sampleY <= 2; sampleY++) for (let sampleX = -2; sampleX <= 2; sampleX++) {
      const worldX = column + originX + sampleX; const worldY = row + originY + sampleY;
      let hash = Math.imul(worldX, 374761393) + Math.imul(worldY, 668265263);
      hash = Math.imul(hash ^ (hash >>> 13), 1274126177); hash ^= hash >>> 16;
      value += hash & 255;
    }
    value = Math.round(value / 25);
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

test('window tracking resumes from a seeded frame and absolute pose', () => {
  const tracker = new WindowTracker();
  const seed = image(5, -3);
  const pose = { x: -42.5, y: 18.25, rotation: 0.037 };
  assert.equal(tracker.process(seed, 120, rectangle).initial, true);
  tracker.setPose(pose, seed.width, seed.height);
  const resumed = tracker.process(image(5, -3), 121, rectangle);
  assert.equal(resumed.success, true, JSON.stringify(resumed));
  assert.equal(resumed.incrementalMatch.frame, 120);
  assert.ok(Math.abs(resumed.raw.x - pose.x) < 0.15, JSON.stringify(resumed.raw));
  assert.ok(Math.abs(resumed.raw.y - pose.y) < 0.15, JSON.stringify(resumed.raw));
  assert.ok(Math.abs(resumed.raw.rotation - pose.rotation) < 0.002, JSON.stringify(resumed.raw));
});

test('large window refinement evaluates a bounded seed when the FFT shift exceeds the search radius', () => {
  const tracker = new WindowTracker();
  const window = { x: 32, y: 32, width: 1024, height: 1024 };
  tracker.process(rigidImage(0, 0, 0, 1088, 1088), 0, window, 4);
  const result = tracker.process(rigidImage(8, 8, 0, 1088, 1088), 1, window, 4);
  assert.equal(result.success, false);
  assert.ok(result.score > -1, JSON.stringify(result));
  assert.ok(Math.abs(result.dx) <= 4 && Math.abs(result.dy) <= 4);
  const retried = tracker.process(rigidImage(8, 8, 0, 1088, 1088), 1, window, 16);
  assert.equal(retried.success, true, JSON.stringify(retried));
  assert.ok(Math.abs(retried.raw.x + 8) < 0.3 && Math.abs(retried.raw.y + 8) < 0.3);
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
  assert.equal(rejected.incrementalMatch.frame, 0);
  assert.equal(rejected.incrementalMatch.accepted, false);
  assert.equal(rejected.incrementalMatch.backward, null);
  const snapshot = JSON.stringify(rejected.incrementalMatch);
  assert.equal(tracker.process(image(), 1, rectangle).success, true);
  assert.equal(JSON.stringify(rejected.incrementalMatch), snapshot);
  tracker.reset(); assert.equal(tracker.process(image(), 0, rectangle).success, true);
});