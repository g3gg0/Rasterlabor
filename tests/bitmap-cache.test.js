import test from 'node:test';
import assert from 'node:assert/strict';
import { trimBitmapCache } from '../src/bitmap-cache.js';

test('eviction closes the oldest bitmap and retains the six recent frames', () => {
  const closed = [];
  const cache = new Map(Array.from({ length: 7 }, (_, frame) =>
    [frame, { close: () => closed.push(frame) }]));
  trimBitmapCache(cache, 6);
  assert.deepEqual(closed, [0]);
  assert.deepEqual([...cache.keys()], [1, 2, 3, 4, 5, 6]);
});
