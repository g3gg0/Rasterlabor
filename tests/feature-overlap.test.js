import test from 'node:test';
import assert from 'node:assert/strict';
import { contextImage } from '../src/context-tracker.js';
import { registerFeatureOverlap } from '../src/feature-overlap.js';

function featureScene(dx = 0, dy = 0) {
  const width = 512; const height = 384; const data = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < width * height; index++) data.set([35, 35, 35, 255], index * 4);
  let state = 123456789;
  for (let feature = 0; feature < 180; feature++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0; const centerX = 24 + state % (width - 48) + dx;
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0; const centerY = 24 + state % (height - 48) + dy;
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0; const value = 80 + state % 176;
    const radius = 3 + feature % 7;
    for (let y = Math.max(0, centerY - radius); y <= Math.min(height - 1, centerY + radius); y++) {
      for (let x = Math.max(0, centerX - radius); x <= Math.min(width - 1, centerX + radius); x++) {
        if ((x - centerX) ** 2 + (y - centerY) ** 2 > radius ** 2) continue;
        data.set([value, value, value, 255], (y * width + x) * 4);
      }
    }
  }
  return { width, height, data };
}

test('ORB fallback recovers a translation from independent local features', () => {
  const reference = contextImage(featureScene());
  const current = contextImage(featureScene(73, -41));
  const result = registerFeatureOverlap(current, reference, { x: 0, y: 0, rotation: 0 });
  assert.equal(result.accepted, true, JSON.stringify(result));
  assert.ok(result.inliers >= 8, JSON.stringify(result));
  assert.ok(Math.hypot(result.pose.x + 73, result.pose.y - 41) < 5, JSON.stringify(result));
});
