import test from 'node:test';
import assert from 'node:assert/strict';
import { acceptPcbGpuOnly, acceptedPcbConstraints, composePose, invertPose, interpolateCorrection, interpolatePcbPoses,
  pcbRunFingerprint,
  optimizePcbComponents, planPcbPairs, planPcbTemporalBridges, planPcbTemporalSkips,
  polygonsOverlap, selectPcbIntermediateFrames,
  selectPcbKeyframes } from '../src/pcb-realignment.js';

const pose = (x, y = 0, rotation = 0) => ({ x, y, rotation });
const entry = (frame, x, y = 0, rotation = 0) => ({ frame, pose: pose(x, y, rotation), sharpness: { score: 1 } });
const field = { width: 100, height: 100 };
const maps = { outputWidth: 100, outputHeight: 100, origin: [0, 0], valid: new Uint8Array(10000).fill(1) };
const near = (actual, expected, tolerance = 1e-8) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);

test('SE2 inversion and short-angle correction interpolation preserve endpoints', () => {
  const original = pose(12, -9, 0.15);
  const identity = composePose(original, invertPose(original));
  near(identity.x, 0); near(identity.y, 0); near(identity.rotation, 0);
  const first = pose(3, 1, Math.PI - 0.1), second = pose(7, -2, -Math.PI + 0.1);
  const start = interpolateCorrection(first, second, 0), end = interpolateCorrection(first, second, 1);
  near(start.x, first.x); near(start.y, first.y); near(end.x, second.x); near(end.y, second.y);
  near(Math.atan2(Math.sin(end.rotation - second.rotation), Math.cos(end.rotation - second.rotation)), 0);
});

test('keyframes include row turns and spatial revisits without choosing every dense frame', () => {
  const path = [];
  for (let frame = 0; frame <= 100; frame++) path.push(entry(frame, frame, 0));
  for (let frame = 101; frame <= 200; frame++) path.push(entry(frame, 200 - frame, 50));
  const selected = selectPcbKeyframes(path, { width: 100, height: 100, spacing: 0.3, turnDegrees: 25, revisitGap: 1000 });
  assert.equal(selected[0].frame, 0); assert.equal(selected.at(-1).frame, 200);
  assert.ok(selected.length < 20);
  assert.ok(selected.some(item => item.frame >= 99 && item.frame <= 102));
});

test('local refinement budget prioritizes a displaced intermediate and retains spatial coverage', () => {
  const path = Array.from({ length: 21 }, (_, frame) => entry(frame, frame * 5));
  path[7].pose.y = 14;
  const selected = selectPcbIntermediateFrames(path, new Set([0, 10, 20]), 6);
  assert.equal(selected.length, 6);
  assert.ok(selected.some(item => item.frame === 7));
  assert.ok(selected.some(item => item.frame > 10));
  assert.equal(selectPcbIntermediateFrames(path, new Set([0, 10, 20]), 0).length, 0);
});

test('resume fingerprint changes with pose, calibration map, video, or mask', () => {
  const path = [entry(0, 0), entry(1, 5)];
  const calibration = { field, maps: { ...maps, inverseX: new Float32Array(10000),
    inverseY: new Float32Array(10000) } };
  const video = { name: 'scan.mp4', width: 100, height: 100, frameCount: 2 };
  const mask = { sourceWidth: 100, sourceHeight: 100, cellSize: 100, data: [1] };
  const base = pcbRunFingerprint(path, calibration, video, mask);
  assert.equal(base, pcbRunFingerprint(path, calibration, video, mask));
  path[1].pose.x = 6;
  assert.notEqual(base, pcbRunFingerprint(path, calibration, video, mask));
  path[1].pose.x = 5;
  calibration.maps.inverseX[0] = 1;
  assert.notEqual(base, pcbRunFingerprint(path, calibration, video, mask));
  calibration.maps.inverseX[0] = 0;
  assert.notEqual(base, pcbRunFingerprint(path, calibration, { ...video, name: 'other.mp4' }, mask));
  assert.notEqual(base, pcbRunFingerprint(path, calibration, video, { ...mask, data: [0] }));
});

test('pair planning sees cross-row overlap and rejects bounding-box-only polygon contact', () => {
  const path = [entry(0, 0, 0), entry(1, 70, 0), entry(2, 70, 60), entry(3, 0, 60)];
  const pairs = planPcbPairs(path, field, maps, { maxNeighbors: 3, maxPairs: 12 });
  assert.ok(pairs.some(pair => pair.reference === 0 && pair.current === 3));
  assert.ok(pairs.length <= 12);
  const square = [{ x: 0, y: 0 }, { x: 2, y: 0 }, { x: 2, y: 2 }, { x: 0, y: 2 }];
  const far = square.map(point => ({ x: point.x + 3, y: point.y + 3 }));
  assert.equal(polygonsOverlap(square, far), false);
});

test('failed temporal keyframe links receive complete reduced-frame bridge chains within budget', () => {
  const path = [0, 1, 2, 3, 10, 11, 20, 21, 22, 23, 24]
    .map(frame => entry(frame, frame));
  const pairs = [{ reference: 0, current: 3, kind: 'temporal' },
    { reference: 10, current: 11, kind: 'temporal' },
    { reference: 20, current: 24, kind: 'temporal' }];
  const matches = pairs.map(pair => ({ ...pair }));
  const accepted = [{ reference: 10, current: 11 }];
  const all = planPcbTemporalBridges(path, pairs, matches, accepted, 10);
  assert.deepEqual(all.map(pair => [pair.reference, pair.current]),
    [[0, 1], [1, 2], [2, 3], [20, 21], [21, 22], [22, 23], [23, 24]]);
  assert.deepEqual(planPcbTemporalBridges(path, pairs, matches, accepted, 3)
    .map(pair => [pair.reference, pair.current]), [[0, 1], [1, 2], [2, 3]]);
  assert.deepEqual(planPcbTemporalBridges(path, pairs, matches, accepted, 2), []);
});

test('a failed adjacent bridge pair gets a one-frame skip on either side', () => {
  const bridge = [[0, 1], [1, 2], [2, 3], [3, 4]].map(([reference, current]) =>
    ({ reference, current, kind: 'temporal', group: 'bridge:0:4' }));
  const accepted = bridge.slice(1).map(pair => ({ reference: pair.reference, current: pair.current }));
  assert.deepEqual(planPcbTemporalSkips(bridge, bridge, accepted, 10)
    .map(pair => [pair.reference, pair.current]), [[0, 2]]);
  assert.deepEqual(planPcbTemporalSkips(bridge, bridge, accepted, 0), []);
  assert.deepEqual(planPcbTemporalSkips(bridge, bridge, bridge, 10), []);
});

test('pair budget keeps temporal coverage and distributes spatial links across the route', () => {
  const path = Array.from({ length: 18 }, (_, frame) => entry(frame, frame * 30));
  const planned = planPcbPairs(path, field, maps, { maxNeighbors: 4, maxPairs: 24 });
  assert.equal(planned.length, 24);
  assert.equal(planned.filter(pair => pair.kind === 'temporal').length, 17);
  const spatial = planned.filter(pair => pair.kind === 'spatial');
  assert.ok(spatial.some(pair => pair.current < 9));
  assert.ok(spatial.some(pair => pair.current > 9));
});

test('pair planning rejects polygon overlap without common allowed image pixels', () => {
  const mask = { sourceWidth: 100, sourceHeight: 100, width: 10, height: 10,
    cellSize: 10, data: new Uint8Array(100) };
  for (let row = 0; row < 10; row++) mask.data[row * 10 + 5] = 1;
  const pairs = planPcbPairs([entry(0, 0), entry(1, 80)], field, maps,
    { maxNeighbors: 4, maxPairs: 8, mask });
  assert.equal(pairs.length, 0);
});

test('interpolation retains unequal local motion and hits optimized keyframes', () => {
  const path = [entry(0, 0), entry(1, 2), entry(2, 8), entry(3, 10)];
  const corrected = interpolatePcbPoses(path, new Map([[0, pose(3)], [3, pose(15)]]));
  near(corrected.get(0).x, 3); near(corrected.get(3).x, 15);
  near(corrected.get(1).x, 5.4); near(corrected.get(2).x, 12.6);
  assert.ok(corrected.get(2).x - corrected.get(1).x > corrected.get(1).x - corrected.get(0).x);
});

test('pair acceptance rejects an inconsistent return match and preserves measured direction', () => {
  const match = { reference: 1, current: 2, kind: 'spatial', referencePose: pose(10),
    forward: { accepted: true, pose: pose(13), score: 0.97, support: 300 },
    backward: { accepted: true }, reverseDistance: 0.5 };
  const good = acceptedPcbConstraints([match], { cycleLimit: 2, minimumScore: 0.9, minimumSupport: 128 });
  assert.equal(good.accepted.length, 1); near(good.accepted[0].measurement.x, 3);
  const bad = acceptedPcbConstraints([{ ...match, reverseDistance: 5 }], { cycleLimit: 2 });
  assert.equal(bad.accepted.length, 0); assert.equal(bad.rejected[0].reason, 'Rueckweg');
});

test('GPU and bidirectional FFT can use a measured residual tolerance without admitting an unverified cycle', () => {
  const match = { reference: 5688, current: 5946, kind: 'temporal',
    referencePose: pose(0), forward: { accepted: true, pose: pose(20), score: 0.936, support: 37994 },
    backward: { accepted: true }, reverseDistance: 4.47,
    fftGpuVerification: { accepted: true, cycleLimit: 5.44,
      fftReverse: { accepted: true, inlierCells: 13, uniqueSupportArea: 70718 } } };
  assert.equal(acceptedPcbConstraints([match], { cycleLimit: 2 }).accepted.length, 1);
  assert.equal(acceptedPcbConstraints([{ ...match, fftGpuVerification: null }],
    { cycleLimit: 2 }).accepted.length, 0);
  assert.equal(acceptedPcbConstraints([{ ...match, reverseDistance: 6 }],
    { cycleLimit: 2 }).accepted.length, 0);
});

test('GPU-only bridge requires a short nearby temporal pair and strong bidirectional evidence', () => {
  const pair = { reference: 4733, current: 4742, kind: 'temporal',
    currentPose: pose(100, 200), lever: 2000 };
  const candidate = { measured: { pose: pose(100, 200), score: 0.9998, support: 3000 },
    backward: { pose: pose(0), score: 0.9998, support: 3000 },
    correctedPose: pose(102, 199), reverseDistance: 0.4 };
  assert.equal(acceptPcbGpuOnly(pair, candidate).accepted, true);
  for (const bad of [
    { pair: { ...pair, kind: 'spatial' } },
    { pair: { ...pair, current: 4750 } },
    { candidate: { ...candidate, measured: { ...candidate.measured, score: 0.99 } } },
    { candidate: { ...candidate, backward: { ...candidate.backward, support: 200 } } },
    { candidate: { ...candidate, reverseDistance: 2 } },
    { candidate: { ...candidate, correctedPose: pose(150, 199) } }
  ]) assert.equal(acceptPcbGpuOnly(bad.pair ?? pair, bad.candidate ?? candidate).accepted, false);
});

test('longer temporal GPU bridge requires four agreeing independent regions', () => {
  const pair = { reference: 4743, current: 4826, kind: 'temporal',
    currentPose: pose(0), lever: 2000 };
  const regions = Array.from({ length: 4 }, () => ({ score: 0.994, support: 1100, distance: 5.6 }));
  const candidate = { measured: { pose: pose(50), score: 0.994, support: 2680 },
    backward: { pose: pose(0), score: 0.994, support: 2690 },
    correctedPose: pose(50), reverseDistance: 1.8, regions };
  assert.equal(acceptPcbGpuOnly(pair, candidate).method, 'independent-regions');
  for (const bad of [
    { pair: { ...pair, kind: 'spatial' } },
    { pair: { ...pair, current: 4900 } },
    { candidate: { ...candidate, regions: regions.slice(0, 3) } },
    { candidate: { ...candidate, regions: regions.map((region, index) =>
      index === 1 ? { ...region, distance: 9 } : region) } },
    { candidate: { ...candidate, regions: regions.map((region, index) =>
      index === 1 ? { ...region, score: 0.97 } : region) } },
    { candidate: { ...candidate, reverseDistance: 3 } },
    { candidate: { ...candidate, correctedPose: pose(80) } }
  ]) assert.equal(acceptPcbGpuOnly(bad.pair ?? pair, bad.candidate ?? candidate).accepted, false);
});

test('disconnected components optimize independently without invented links', () => {
  const graph = { nodes: [
    { frame: 0, pose: pose(0) }, { frame: 1, pose: pose(14) },
    { frame: 10, pose: pose(100) }, { frame: 11, pose: pose(114) }, { frame: 99, pose: pose(500) }],
  edges: [{ reference: 0, current: 1, measurement: pose(10), weight: 1, kind: 'pcb-temporal' },
    { reference: 10, current: 11, measurement: pose(10), weight: 1, kind: 'pcb-temporal' }] };
  const result = optimizePcbComponents(graph, { iterations: 8, huber: 20 });
  assert.equal(result.components.length, 3);
  assert.equal(result.components.filter(component => component.status === 'optimized').length, 2);
  assert.equal(result.components.find(component => component.anchor === 99).status, 'unconnected');
  near(result.corrections.find(item => item.frame === 1).pose.x, 10, 1e-3);
  near(result.corrections.find(item => item.frame === 11).pose.x, 110, 1e-3);
});
