import test from 'node:test';
import assert from 'node:assert/strict';
import { contextImage } from '../src/context-tracker.js';
import { pcbVias, recoverPcbViaPair, recoverPcbViaPool } from '../src/pcb-vias.js';

function board(pose, spacing = 1, visible = [0, 1, 2]) {
  const width = 1600, height = 1600, data = new Uint8ClampedArray(width * height * 4);
  const vias = [[-550, -450], [530, -470], [120, 530]];
  const cosine = Math.cos(pose.rotation), sine = Math.sin(pose.rotation);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const wx = cosine * (x - width / 2) - sine * (y - height / 2) + pose.x;
    const wy = sine * (x - width / 2) + cosine * (y - height / 2) + pose.y;
    let value = 100 + wx * 0.002;
    if (Math.hypot(wx, wy) < 100) value = 30;
    if (Math.hypot(wx, wy) < 70) value = 220;
    if (Math.abs(wy + 650) < 2) value = 30;
    if (vias.some(([vx, vy], index) => visible.includes(index) && Math.hypot(wx - vx * spacing, wy - vy * spacing) < 12)) value = 25;
    data.set([value, value, value, 255], (y * width + x) * 4);
  }
  return contextImage({ width, height, data });
}

test('small via cores provide rotation and translation independently of a large round aperture', () => {
  const reference = { x: 0, y: 0, rotation: 0 }, truth = { x: 40, y: 80, rotation: 0.04 };
  const first = board(reference), second = board(truth);
  assert.equal(pcbVias(first).length, 3);
  const result = recoverPcbViaPair(first, second, reference, { x: 70, y: 110, rotation: -0.01 },
    { x: 40, y: 80, rotation: -0.01 }, { coarseRadius: 384 });
  assert.equal(result.accepted, true, JSON.stringify(result));
  assert.equal(result.inliers, 3);
  assert.ok(Math.hypot(result.pose.x - truth.x, result.pose.y - truth.y) < 2);
  assert.ok(Math.abs(result.pose.rotation - truth.rotation) < 0.002);
});

test('complementary frames provide two stable vias even when each reference sees only one', () => {
  const reference = { x: 0, y: 0, rotation: 0 }, truth = { x: 40, y: 80, rotation: 0.04 };
  const first = board(reference), second = board(truth), points = pcbVias(first);
  const frames = [0, 1, 2, 3].map(frame => ({ frame, center: reference, points: [points[frame < 2 ? 0 : 1]] }));
  const prediction = { x: 70, y: 110, rotation: -0.01 };
  const result = recoverPcbViaPool(second, frames, prediction, { ...truth, rotation: -0.01 }, { coarseRadius: 384 });
  assert.equal(result.accepted, true, JSON.stringify(result));
  assert.equal(result.inliers, 2);
  assert.ok(Math.hypot(result.pose.x - truth.x, result.pose.y - truth.y) < 2);
  assert.ok(Math.abs(result.pose.rotation - truth.rotation) < 0.002);
  assert.equal(recoverPcbViaPool(second, frames.filter(frame => frame.frame % 2 === 0), prediction,
    truth, { coarseRadius: 384 }).accepted, false);
});

test('one matched via fixes position while retaining the predicted angle', () => {
  const reference = { x: 0, y: 0, rotation: 0 }, truth = { x: 40, y: 80, rotation: 0.04 };
  const prediction = { x: 70, y: 110, rotation: -0.01 };
  const result = recoverPcbViaPair(board(reference, 1, [0]), board(truth, 1, [0]), reference, prediction,
    { ...truth, rotation: prediction.rotation }, { coarseRadius: 384 });
  assert.equal(result.accepted, true, JSON.stringify(result));
  assert.equal(result.translationOnly, true);
  assert.equal(result.pose.rotation, prediction.rotation);
});

test('via recovery rejects incompatible geometry and corrections outside the search bound', () => {
  const pose = { x: 0, y: 0, rotation: 0 }, first = board(pose);
  assert.equal(recoverPcbViaPair(first, board(pose, 1.1), pose, pose, pose, { coarseRadius: 384 }).accepted, false);
  const truth = { x: 40, y: 80, rotation: 0.04 };
  assert.equal(recoverPcbViaPair(first, board(truth), pose, pose, truth, { coarseRadius: 16 }).accepted, false);
});
