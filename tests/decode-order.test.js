import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeFrontiers, decodeLookahead } from '../src/decode-order.js';

test('decode frontiers include future references required by presentation-ordered B-frames', () => {
  const presentation = [{ index: 0 }, { index: 2 }, { index: 3 }, { index: 1 }, { index: 5 }, { index: 6 }, { index: 4 }];
  assert.deepEqual(decodeFrontiers(presentation), [0, 2, 3, 3, 5, 6, 6]);
});

test('decode frontiers remain identical for videos without reordered frames', () => {
  assert.deepEqual(decodeFrontiers([{ index: 0 }, { index: 1 }, { index: 2 }]), [0, 1, 2]);
  assert.throws(() => decodeFrontiers([{ index: -1 }]), /Decode-Index/);
});

test('decode lookahead handles long videos without spreading the frame index onto the call stack', () => {
  const presentation = Array.from({ length: 100_000 }, (_, index) => ({ index }));
  presentation[50_000].index += 3;
  assert.equal(decodeLookahead(presentation), 5);
});