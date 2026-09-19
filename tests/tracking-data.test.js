import test from 'node:test';
import assert from 'node:assert/strict';
import { applyLoopClosure, validateTracking, trackingReferenceCounts } from '../src/tracking-data.js';

test('reference counts include rejected requests, distinguish consensus and cover the complete path', () => {
  const path = Array.from({ length: 250 }, (_, frame) => ({ frame, context: { matches: [], inliers: [], applied: false } }));
  path[248].context = { matches: [{ frame: 0, accepted: true }, { frame: 1, accepted: false }, { frame: 0, accepted: true }], applied: true, inliers: [0] };
  path[249].context = { matches: [{ frame: 0 }, { frame: 248 }, { frame: 999 }, { frame: 249 }, null], applied: false, inliers: [0] };
  const original = JSON.stringify(path);
  const counts = trackingReferenceCounts(path);
  assert.deepEqual(counts.get(0), { referencedBy: 2, references: 0, consensusBy: 1, consensus: 0 });
  assert.equal(counts.get(1).referencedBy, 1);
  assert.deepEqual(counts.get(248), { referencedBy: 1, references: 2, consensusBy: 0, consensus: 1 });
  assert.equal(counts.get(249).references, 3);
  assert.equal(JSON.stringify(path), original);
  assert.deepEqual(trackingReferenceCounts(JSON.parse(original)), counts);
});

test('reference counts distinguish unavailable diagnostics from measured zero', () => {
  assert.equal(trackingReferenceCounts([]).size, 0);
  const legacy = trackingReferenceCounts([{ frame: 0 }]).get(0);
  assert.equal(legacy.references, null); assert.equal(legacy.referencedBy, null);
  const mixed = trackingReferenceCounts([{ frame: 0 }, { frame: 1, context: { matches: [{ frame: 0 }] } }]);
  assert.equal(mixed.get(0).references, null); assert.equal(mixed.get(0).referencedBy, 1);
  assert.equal(mixed.get(1).referencedBy, 0);
});

test('loop closure distributes error backward and propagates it forward while keeping its anchor fixed', () => {
  const path = [0, 5, 10, 15].map(frame => ({ frame, raw: { x: frame, y: frame * 2, rotation: frame / 100 },
    pose: { x: frame, y: frame * 2, rotation: frame / 100 } }));
  assert.equal(applyLoopClosure(path, { anchorFrame: 0, frame: 10, dx: -10, dy: -20, rotation: -0.1 }), 3);
  assert.deepEqual(path[0].pose, { x: 0, y: 0, rotation: 0 });
  assert.deepEqual(path[1].pose, { x: 0, y: 0, rotation: 0 });
  assert.deepEqual(path[1].raw, { x: 0, y: 0, rotation: 0 });
  assert.deepEqual(path[2].pose, { x: 0, y: 0, rotation: 0 });
  assert.deepEqual(path[3].pose, { x: 5, y: 10, rotation: 0.04999999999999999 });
});

const rescued = () => ({ format: 'rasterlabor-xyr-tracking', model_version: 1,
  source: 'live-browser-session', options: { trackingMode: 'window', trackingWindow: '15' },
  rectangle: { x: null, y: null, width: 3710, height: 3982 },
  path: [{ frame: 970, timestamp_us: 32410167, x_px: -10, y_px: 2, rotation_deg: 90,
    raw_x_px: -20, raw_y_px: 4, raw_rotation_deg: -45, patches: 1, rms_px: null }] });

test('rescued CSV tracking converts units, preserves source data and associates the video', () => {
  const original = rescued();
  const video = { name: 'test.mp4', width: 4320, height: 7680 };
  const result = validateTracking(original, video);
  assert.deepEqual(result.path[0].pose, { x: -10, y: 2, rotation: Math.PI / 2 });
  assert.equal(result.path[0].raw.rotation, -Math.PI / 4);
  assert.equal(result.path[0].timestamp, 32410167);
  assert.equal(result.path[0].raw.rms, null);
  assert.equal(result.path[0].mode, 'window');
  assert.equal(result.rectangle, null);
  assert.equal(result.path[0].rectangle, null);
  assert.deepEqual(result.video, video);
  assert.deepEqual(original, rescued());
  assert.deepEqual(validateTracking(JSON.parse(JSON.stringify(result))), result);
});

test('rescued tracking rejects missing coordinates, duplicate frames and invalid timestamps', () => {
  for (const field of ['x_px', 'rotation_deg', 'raw_rotation_deg', 'timestamp_us']) {
    const input = rescued(); input.path[0][field] = null;
    assert.throws(() => validateTracking(input), /Tracking/);
  }
  const input = rescued(); input.path.push({ ...input.path[0] });
  assert.throws(() => validateTracking(input), /Trackingframe/);
});
