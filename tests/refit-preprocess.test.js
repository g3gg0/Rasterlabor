import test from 'node:test';
import assert from 'node:assert/strict';
import { contextImage } from '../src/context-tracker.js';
import { normalizeRefitPreprocess, refitGray } from '../src/refit-preprocess.js';

test('default refit preprocessing preserves legacy luminance', () => {
  for (const color of [[0, 0, 0], [255, 255, 255], [255, 0, 0], [17, 93, 201]]) {
    const expected = Math.floor((299 * color[0] + 587 * color[1] + 114 * color[2] + 500) / 1000);
    assert.equal(refitGray(...color), expected);
  }
});

test('channel weights and tone controls alter matcher intensity predictably', () => {
  assert.equal(refitGray(240, 20, 10, { red: 1, green: 0, blue: 0 }), 240);
  assert.equal(refitGray(240, 20, 10, { red: 0, green: 1, blue: 0 }), 20);
  assert.ok(refitGray(96, 96, 96, { gamma: 2 }) > refitGray(96, 96, 96));
  assert.equal(refitGray(32, 32, 32, { brightness: -1 }), 0);
  assert.deepEqual(normalizeRefitPreprocess({ red: 2, green: 1, blue: 1 }),
    { brightness: 0, contrast: 1, gamma: 1, red: 0.5, green: 0.25, blue: 0.25 });
});

test('context images preprocess valid pixels and preserve excluded pixels', () => {
  const image = { width: 2, height: 1, data: new Uint8ClampedArray([200, 20, 10, 255, 10, 200, 20, 0]) };
  const result = contextImage(image, null, { red: 1, green: 0, blue: 0 });
  assert.deepEqual([...result.levels[0].gray], [200, -1]);
});
