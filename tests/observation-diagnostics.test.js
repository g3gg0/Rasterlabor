import test from 'node:test';
import assert from 'node:assert/strict';
import { analyzeObservations, frameGeometry, selectConsistentFrames } from '../src/observation-diagnostics.js';

const points = [{ col: 0, row: 0 }, { col: 10, row: 0 }, { col: 0, row: 10 }, { col: 10, row: 10 }];

function frame(id, scale, offsetX = 20, offsetY = 30) {
  return { id, enabled: true, points: points.map(point => ({
    ...point, x: offsetX + scale * point.col, y: offsetY + scale * point.row, confidence: 1
  })) };
}

test('raw diagnostics distinguish scale changes from rigid motion', () => {
  const result = analyzeObservations([frame(0, 1), frame(1, 1.1)], 100, 100, 1, 10, 10);
  assert.ok(result.metrics.rigidP95 > 0.4);
  assert.ok(result.metrics.similarityP95 < 1e-9);
  assert.equal(result.metrics.scaleP05, 1);
  assert.equal(result.metrics.scaleP95, 1.1);
});

test('raw diagnostics expose repeated frames and count each occupied cell once per frame', () => {
  const result = analyzeObservations([frame(0, 1), frame(1, 1), frame(2, 1, 30)], 100, 100, 1, 1, 1);
  assert.equal(result.frameCounts[0], 3);
  assert.equal(result.metrics.nearDuplicateFraction, 0.5);
  assert.equal(result.metrics.directionBins, 1);
});

test('affine geometry ignores translation and rotation while exposing scale and tilt', () => {
  const geometry = frameGeometry([
    { col: 0, row: 0, x: 50, y: 20 }, { col: 10, row: 0, x: 50, y: 30 },
    { col: 0, row: 10, x: 30, y: 20 }, { col: 10, row: 10, x: 30, y: 30 }
  ]);
  assert.ok(Math.abs(geometry.scale - Math.sqrt(2)) < 1e-12);
  assert.ok(Math.abs(geometry.anisotropy - 2) < 1e-12);
});

test('consistent selection retains the dominant geometry cluster', () => {
  const candidates = [frame(0, 1), frame(1, 1.004, 30), frame(2, 0.997, 40), frame(3, 0.91), frame(4, 1.08)];
  const result = selectConsistentFrames(candidates, 1, 0.01, 0.01);
  assert.deepEqual([...result.selectedIds], [0, 1, 2]);
  assert.deepEqual([...result.rejectedIds], [3, 4]);
});