import test from 'node:test';
import assert from 'node:assert/strict';
import { overlayTiles, overlayTileSize } from '../src/overlay-tiles.js';
import { frameGeometry } from '../src/path-support.js';

test('tile size uses larger power-of-two tiles within texture and memory limits', () => {
  assert.equal(overlayTileSize({ outputWidth: 4096, outputHeight: 3072 }, 8192), 4096);
  assert.equal(overlayTileSize({ outputWidth: 8192, outputHeight: 4320 }, 8192), 4096);
  assert.equal(overlayTileSize({ outputWidth: 4096, outputHeight: 3072 }, 2048), 2048);
  assert.equal(overlayTileSize({ outputWidth: 1024, outputHeight: 1024 }, 8192, 64 * 1024 * 1024), 1024);
});

test('tiles cover full-resolution output including partial edges without overlap', () => {
  const geometry = { corners: [{x:0,y:0},{x:9,y:0},{x:9,y:7},{x:0,y:7}] };
  const tiles = overlayTiles(9,7,0,0,[geometry],4);
  const counts = new Uint8Array(63);
  for (const tile of tiles) for(let y=tile.y;y<tile.y+tile.height;y++)for(let x=tile.x;x<tile.x+tile.width;x++)counts[y*9+x]++;
  assert.equal(tiles.length,6); assert.ok(counts.every(x=>x===1));
  assert.deepEqual([tiles.at(-1).width,tiles.at(-1).height],[1,3]);
});

test('tile selection includes bilinear edge footprints and skips distant frames', () => {
  const near={corners:[{x:4.25,y:1},{x:5,y:2}]};
  const far={corners:[{x:100,y:100},{x:101,y:101}]};
  const tiles=overlayTiles(8,4,0,0,[near,far],4);
  assert.equal(tiles.length,2);
  for(const tile of tiles)assert.deepEqual(tile.geometries,[near]);
});

test('merge tile selection keeps only locally useful sharp frames', () => {
  const maps = { outputWidth: 256, outputHeight: 256, valid: new Uint8Array(256 * 256).fill(255) };
  const geometry = (frame, score, left) => frameGeometry({ frame, mode: 'window', sharpness: { score },
    pose: { x: left + 128, y: 128, rotation: 0 } }, {}, maps);
  const geometries = [geometry(0, 50, 0), geometry(1, 40, 0), geometry(2, 30, 0),
    geometry(3, 20, 0), geometry(4, 10, 300)];
  const tiles = overlayTiles(512, 64, 64, 64, geometries, 64, { maxFrames: 2, edgeFeather: 0 });
  const interior = tiles.filter(tile => tile.x === 0 || tile.x === 320);
  assert.deepEqual(interior.map(tile => tile.geometries.map(item => item.entry.frame)), [[0, 1, 2], [4]]);
});

test('merge tile selection retains narrow coverage crossing a tile boundary', () => {
  const crossing = { width: 64, corners: [{ x: 0, y: 0 }, { x: 4.4, y: 0 }, { x: 4.4, y: 4 }, { x: 0, y: 4 }],
    entry: { frame: 1, sharpness: { score: 1 } }, supports: point => point.x >= 0 && point.x < 4.4 };
  const tiles = overlayTiles(8, 4, 0, 0, [crossing], 4, { maxFrames: 1, edgeFeather: 0.1 });
  assert.deepEqual(tiles.map(tile => tile.geometries.map(item => item.entry.frame)), [[1], [1]]);
});
