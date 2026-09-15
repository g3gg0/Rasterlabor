import test from 'node:test';
import assert from 'node:assert/strict';
import { encoderSizes } from '../src/video-exporter.js';

test('encoder sizes preserve aspect ratio, use even dimensions and fit common H.264 limits', () => {
  const sizes = encoderSizes(2301, 4248);
  assert.deepEqual(sizes[0], { width: 2300, height: 4248 });
  assert.deepEqual(sizes[1], { width: 2218, height: 4096 });
  for (const size of sizes) {
    assert.equal(size.width % 2, 0);
    assert.equal(size.height % 2, 0);
    assert.ok(Math.abs(size.width / size.height - 2301 / 4248) < 0.001);
  }
});