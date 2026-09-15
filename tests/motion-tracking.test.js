import test from 'node:test';
import assert from 'node:assert/strict';
import { createSpline } from '../src/spline.js';
import { fitCameraPose, stabilizePose } from '../src/motion-tracking.js';

test('camera pose recovers inverse subpixel translation and rotation despite an outlier', () => {
  const field = createSpline(640, 480, 120);
  const theta = 0.018;
  const cosine = Math.cos(theta);
  const sine = Math.sin(theta);
  const tx = 3.25;
  const ty = -2.75;
  const points = [];
  for (let row = 0; row < 4; row++) for (let col = 0; col < 5; col++) {
    const sourceX = 90 + col * 105;
    const sourceY = 70 + row * 95;
    points.push({ col: sourceX, row: sourceY, x: cosine * sourceX - sine * sourceY + tx,
      y: sine * sourceX + cosine * sourceY + ty, confidence: 1 });
  }
  points.push({ col: 300, row: 200, x: 20, y: 450, confidence: 1 });
  const pose = fitCameraPose(points, field);
  const center = { x: 320, y: 240 };
  const expectedX = cosine * (center.x - tx) + sine * (center.y - ty) - center.x;
  const expectedY = -sine * (center.x - tx) + cosine * (center.y - ty) - center.y;
  assert.ok(Math.abs(pose.x - expectedX) < 1e-6);
  assert.ok(Math.abs(pose.y - expectedY) < 1e-6);
  assert.ok(Math.abs(pose.rotation + theta) < 1e-9);
  assert.equal(pose.points, 20);
});

test('pose stabilization uses the configured trailing frame window', () => {
  const path = [0, 2, 10].map((x, index) => ({ raw: { x, y: -x, rotation: index * 0.01, points: 10 } }));
  const pose = stabilizePose(path, 2);
  assert.equal(pose.x, 6);
  assert.equal(pose.y, -6);
  assert.ok(Math.abs(pose.rotation - 0.015) < 1e-9);
});

test('camera pose excludes replenished anchors outside the calibrated sensor', () => {
  const field = createSpline(640, 480, 120);
  const points = [[100, 100], [300, 100], [100, 300], [300, 300]].map(([col, row]) =>
    ({ col, row, x: col + 2, y: row - 3 }));
  points.push({ col: -500, row: -500, x: 20, y: 20 },
    { col: 300, row: 900, x: 300, y: 400 }, { col: NaN, row: 100, x: 100, y: 100 });
  const pose = fitCameraPose(points, field);
  assert.equal(pose.points, 4);
  assert.ok(Math.abs(pose.x + 2) < 1e-6);
  assert.ok(Math.abs(pose.y - 3) < 1e-6);
  assert.equal(fitCameraPose(points.slice(4), field), null);
});

test('missing poses stay missing and invalid history does not poison later poses', () => {
  const valid = { raw: { x: 2, y: 3, rotation: 0.1, points: 10 } };
  const invalid = { raw: { x: NaN, y: 0, rotation: NaN, points: 10 } };
  assert.equal(stabilizePose([valid, { raw: null }], 15), null);
  assert.equal(stabilizePose([valid, invalid], 15), null);
  assert.deepEqual(stabilizePose([invalid, { raw: null }, valid], 15), { x: 2, y: 3, rotation: 0.1 });
});