import test from 'node:test';
import assert from 'node:assert/strict';
import { createBigTiffLayout, encodeBigTiffHeader } from '../src/bigtiff-writer.js';

test('BigTIFF layout uses sparse 64-bit tiled offsets beyond four GiB', () => {
  const tiles = [];
  for (let y = 0; y < 65536; y += 2048) for (let x = 0; x < 65536; x += 2048) tiles.push({ x, y });
  const layout = createBigTiffLayout(65536, 65536, 2048, tiles);
  assert.equal(layout.tileCount, 1024);
  assert.ok(layout.fileBytes > 0xffffffff);
  assert.ok(layout.offsets.at(-1) > 0xffffffffn);
  const header = encodeBigTiffHeader(layout);
  const view = new DataView(header.buffer);
  assert.equal(view.getUint16(2, true), 43);
  assert.equal(view.getBigUint64(8, true), 16n);
});

test('BigTIFF layout leaves omitted tiles sparse', () => {
  const layout = createBigTiffLayout(5000, 3000, 2048, [{ x: 2048, y: 0 }]);
  assert.equal(layout.tileCount, 6);
  assert.equal(layout.offsets[0], 0n);
  assert.equal(layout.byteCounts[0], 0n);
  assert.ok(layout.offsets[1] > 0n);
});