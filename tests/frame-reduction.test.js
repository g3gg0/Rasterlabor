import test from 'node:test';
import assert from 'node:assert/strict';
import { frameGeometry } from '../src/path-support.js';
import { fullyCoversBlock, reduceFramesByBlocks, usableImageBounds } from '../src/frame-reduction.js';

const field = { width: 20, height: 20 };
const maps = (valid = new Uint8Array(400).fill(1)) => ({ outputWidth: 20, outputHeight: 20, origin: [0, 0], valid });
const geometry = (frame, x, score, imageMaps = maps(), rotation = 0) => frameGeometry({ frame, mode: 'window',
  pose: { x, y: 0, rotation }, sharpness: { score } }, field, imageMaps);

test('usable rectangle follows the valid transformed image mask', () => {
  const valid = new Uint8Array(400);
  for (let y = 3; y < 17; y++) for (let x = 2; x < 18; x++) valid[y * 20 + x] = 1;
  assert.deepEqual(usableImageBounds(valid, 20, 20), { minX: 2, minY: 3, maxX: 18, maxY: 17 });
});

test('a block needs all valid pixels, not only its centre and corners', () => {
  const valid = new Uint8Array(400).fill(1);
  const block = { minX: -3, minY: -3, maxX: 3, maxY: 3 };
  assert.equal(fullyCoversBlock(geometry(1, 0, 1, maps(valid)), block), true);
  const withHole = valid.slice(); withHole[10 * 20 + 10] = 0;
  assert.equal(fullyCoversBlock(geometry(1, 0, 1, maps(withHole)), block), false);
  assert.equal(fullyCoversBlock(geometry(1, 0, 1, maps(), Math.PI / 4), block), true);
});

test('reduction keeps the sharpest M full contributors per square', () => {
  const candidates = [geometry(1, 0, 2), geometry(2, 0, 9), geometry(3, 0, 5), geometry(4, 30, 4)];
  const result = reduceFramesByBlocks(candidates, { divisions: 5, minimum: 2 });
  assert.deepEqual([...result.frames].sort((a, b) => a - b), [2, 3, 4]);
  assert.equal(result.blockSize, 4);
  assert.ok(result.blocks > 0);
  assert.ok(result.shortBlocks > 0);
});
