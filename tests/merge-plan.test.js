import test from 'node:test';
import assert from 'node:assert/strict';
import { blurriestFramesFirst, createMergeCoverage, mergeBounds, mergeEstimate, selectMergeFrames } from '../src/merge-plan.js';
import { frameGeometry } from '../src/path-support.js';

const geometry = (frame, score, corners) => ({ entry: { frame, sharpness: score === null ? null : { score } }, corners });

test('merge order puts unknown and blurry frames below sharper frames', () => {
  const frames = [geometry(8, 40, []), geometry(4, null, []), geometry(7, 12, []), geometry(3, 40, [])];
  assert.deepEqual(blurriestFramesFirst(frames).map(item => item.entry.frame), [4, 7, 3, 8]);
});

test('merge bounds preserve every tracked frame at native pixel scale', () => {
  const frames = [
    geometry(0, 1, [{ x: -2.4, y: 3.2 }, { x: 8.1, y: 11.01 }]),
    geometry(1, 2, [{ x: 4.5, y: -1.2 }, { x: 12.01, y: 9.5 }])
  ];
  assert.deepEqual(mergeBounds(frames), { minX: -3, minY: -2, maxX: 13, maxY: 12, width: 16, height: 14 });
  assert.deepEqual(mergeEstimate(mergeBounds(frames), 8), { columns: 2, rows: 2, tiles: 4, bytes: 896 });
});

test('empty merge plans have no bounds', () => {
  assert.equal(mergeBounds([]), null);
  assert.equal(mergeEstimate(null), null);
});

test('merge preselection retains only frames that add sharp masked coverage', () => {
  const maps = { outputWidth: 1000, outputHeight: 500, valid: new Uint8Array(500000).fill(255) };
  const rectangle = (frame, score, left) => frameGeometry({ frame, mode: 'window', sharpness: { score },
    pose: { x: left + 500, y: 250, rotation: 0 } }, {}, maps);
  const frames = [rectangle(0, 50, 0), rectangle(1, 40, 10), rectangle(2, 30, 20), rectangle(3, 20, 30),
    rectangle(4, 10, 900)];
  const bounds = { minX: 128, minY: 128, maxX: 256, maxY: 256 };
  assert.deepEqual(selectMergeFrames(frames, 2, null, 0, bounds).map(frame => frame.entry.frame), [0, 1, 2]);
  assert.ok(selectMergeFrames(frames, 2).includes(frames[4]));
  assert.deepEqual(selectMergeFrames(frames, 2, () => false), []);
});

test('fifty percent feathering still bounds overlapping merge candidates', () => {
  const maps = { outputWidth: 1000, outputHeight: 500, valid: new Uint8Array(500000).fill(255) };
  const frames = Array.from({ length: 100 }, (_, frame) => frameGeometry({ frame, mode: 'window',
    sharpness: { score: 100 - frame }, pose: { x: 500, y: 250, rotation: 0 } }, {}, maps));
  const bounds = { minX: 128, minY: 128, maxX: 256, maxY: 256 };
  assert.deepEqual(selectMergeFrames(frames, 2, null, 0.5, bounds).map(frame => frame.entry.frame), [0, 1, 2]);
});

test('partial mask coverage still bounds overlapping merge candidates', () => {
  const bounds = { minX: 0, minY: 0, maxX: 128, maxY: 128 };
  const frames = Array.from({ length: 100 }, (_, frame) => ({ entry: { frame, sharpness: { score: 100 - frame } },
    width: 256, corners: [{ x: 0, y: 0 }, { x: 256, y: 0 }, { x: 256, y: 256 }, { x: 0, y: 256 }],
    supports: () => true }));
  assert.deepEqual(selectMergeFrames(frames, 3, null, 0.1, bounds, () => 1).map(frame => frame.entry.frame), [0, 1, 2, 3]);
});

test('merge keeps a filler for a mask hole between coarse coverage probes', () => {
  const maps = { outputWidth: 256, outputHeight: 256, valid: new Uint8Array(256 * 256).fill(255) };
  const allowed = (x, y) => !(x >= 78 && x < 86 && y >= 78 && y < 86);
  const frames = [0, 0, 0, 32].map((offset, index) => frameGeometry({ frame: index, mode: 'window',
    sharpness: { score: 100 - index }, pose: { x: 128 + offset, y: 128, rotation: 0 } }, {}, maps));
  const bounds = { minX: 64, minY: 64, maxX: 128, maxY: 128 };
  const selected = selectMergeFrames(frames, 2, allowed, 0, bounds);
  assert.ok(selected.includes(frames[3]));
  for (let y = 64; y < 128; y++) for (let x = 64; x < 128; x++) {
    const point = { x: x + 0.5, y: y + 0.5 };
    assert.equal(selected.some(frame => frame.supports(point, allowed)), frames.some(frame => frame.supports(point, allowed)));
  }
});

test('merge certifies whole cells including invalid pixels and bilinear margins', () => {
  const maps = { outputWidth: 256, outputHeight: 256, valid: new Uint8Array(256 * 256).fill(255) };
  maps.valid[82 * 256 + 82] = 0;
  const frame = frameGeometry({ frame: 0, mode: 'window', pose: { x: 128, y: 128, rotation: 0 } }, {}, maps);
  const cell = { minX: 112, minY: 112, maxX: 144, maxY: 144 };
  assert.equal(createMergeCoverage(null, 0)(frame, cell), 2);
  assert.equal(createMergeCoverage(null, 0.1)(frame, cell), 2);
  assert.equal(createMergeCoverage(null, 0)(frame, { minX: 64, minY: 64, maxX: 128, maxY: 128 }), 1);
  assert.equal(createMergeCoverage(() => false)(frame, cell), 0);
});