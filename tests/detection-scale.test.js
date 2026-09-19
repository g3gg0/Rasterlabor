import test from 'node:test';
import assert from 'node:assert/strict';
import { detectionImageSize, restoreDetectionScale, scaleDetectionOptions } from '../src/detection-scale.js';

test('8K portrait detection is bounded to half resolution', () => {
  assert.deepEqual(detectionImageSize(4320, 7680), { width: 2160, height: 3840 });
  assert.deepEqual(detectionImageSize(1920, 1080), { width: 1920, height: 1080 });
});

test('detection options and results roundtrip between source and working pixels', () => {
  const options = scaleDetectionOptions({ approxStep: 290, roi: { x: 100, y: 200, width: 4000, height: 7000 } },
    4320, 7680, 2160, 3840);
  assert.deepEqual(options, { approxStep: 145, roi: { x: 50, y: 100, width: 2000, height: 3500 } });
  const restored = restoreDetectionScale({ points: [{ x: 25.25, y: 50.5 }], rejected: [{ x: 2, y: 3 }],
    lines: [{ points: [{ x: 4, y: 5 }] }], roi: options.roi, step: 145 }, 4320, 7680, 2160, 3840);
  assert.deepEqual(restored.points, [{ x: 50.5, y: 101 }]);
  assert.deepEqual(restored.rejected, [{ x: 4, y: 6 }]);
  assert.deepEqual(restored.lines[0].points, [{ x: 8, y: 10 }]);
  assert.deepEqual(restored.roi, { x: 100, y: 200, width: 4000, height: 7000 });
  assert.equal(restored.step, 290);
});