import test from 'node:test';
import assert from 'node:assert/strict';
import { basis, createSpline, evaluate, inversePoint, fieldColor } from '../src/spline.js';
import { lsqr, smoothRows, geometryCheck, poseFor, fitCalibration } from '../src/solver.js';
import { transposePartitionCount } from '../src/webgpu-lsqr.js';
import { remapChunkRows } from '../src/webgpu-remapper.js';

test('GPU transpose partitions cover uneven columns and partial blocks exactly once', () => {
  for (const [length, expected] of [[0, 1], [1, 1], [2048, 1], [2049, 2], [8197, 5], [131073, 64]]) {
    assert.equal(transposePartitionCount(new Uint32Array([0, length])), expected);
  }
  const lengths = [0, 1, 255, 256, 257, 2048, 2049, 8197, 131073];
  const offsets = new Uint32Array(lengths.length + 1);
  lengths.forEach((length, column) => { offsets[column + 1] = offsets[column] + length; });
  const parts = transposePartitionCount(offsets);
  for (let column = 0; column < lengths.length; column++) {
    const visits = new Uint8Array(lengths[column]);
    for (let part = 0; part < parts; part++) {
      for (let local = 0; local < 256; local++) {
        for (let cursor = offsets[column] + part * 256 + local; cursor < offsets[column + 1]; cursor += parts * 256) {
          visits[cursor - offsets[column]]++;
        }
      }
    }
    assert.ok(visits.every(count => count === 1));
  }
});

test('GPU remap chunks stay within binding and dispatch limits', () => {
  const width = 4381;
  const rows = remapChunkRows(width, 128 * 1024 * 1024, 65535);
  assert.equal(rows, 1024);
  assert.ok(width * rows * 4 <= 128 * 1024 * 1024);
  assert.ok(Math.ceil(width * rows / 256) <= 65535);
  assert.equal(Math.ceil(7852 / rows), 8);
});

test('cardinal basis partitions unity and analytic derivatives agree', () => {
  for (const value of [0, 0.13, 0.5, 0.99, 1]) {
    assert.ok(Math.abs(basis(value).reduce((sum, weight) => sum + weight, 0) - 1) < 1e-14);
    for (let derivative = 1; derivative <= 2; derivative++) {
      const before = basis(value - 1e-5, derivative - 1);
      const after = basis(value + 1e-5, derivative - 1);
      basis(value, derivative).forEach((weight, index) => {
        assert.ok(Math.abs(weight - (after[index] - before[index]) / 2e-5) < 1e-8);
      });
    }
  }
});

test('identity and affine fields include the padded image boundary', () => {
  const affine = [1.1, 0.14, -0.06, 0.93, -130, 23];
  const field = createSpline(641, 479, 93, affine);
  for (const [px, py] of [[0, 0], [640, 478], [221.4, 115.3]]) {
    const value = evaluate(field, px, py, true);
    assert.ok(Math.abs(value.x - (affine[0] * px + affine[1] * py + affine[4])) < 1e-10);
    assert.ok(Math.abs(value.y - (affine[2] * px + affine[3] * py + affine[5])) < 1e-10);
    assert.ok(Math.abs(value.j01 - affine[1]) < 1e-12);
    const inverted = inversePoint(field, value.x, value.y, 320, 239);
    assert.ok(inverted && Math.hypot(inverted.x - px, inverted.y - py) < 0.01);
  }
  const identity = evaluate(createSpline(50, 40, 16), 20.5, 10.5);
  assert.equal(identity.x, 20.5);
  assert.equal(identity.y, 10.5);
});

test('asymmetric smooth field has correct inverse and signed color channels', () => {
  const field = createSpline(320, 240, 50);
  for (let row = 0; row < field.ny; row++) {
    for (let col = 0; col < field.nx; col++) {
      const index = (row * field.nx + col) * 2;
      field.coefficients[index] = 6 * Math.sin(col * 0.6) * Math.cos(row * 0.4);
      field.coefficients[index + 1] = 4 * Math.sin(row * 0.7 + col * 0.2);
    }
  }
  for (let py = 10; py < 230; py += 23) {
    for (let px = 10; px < 310; px += 29) {
      const target = evaluate(field, px, py);
      const source = inversePoint(field, target.x, target.y, target.x, target.y);
      assert.ok(source && Math.hypot(source.x - px, source.y - py) < 0.01);
    }
  }
  assert.deepEqual(fieldColor(0, 0, 20), [128, 0, 128]);
  assert.deepEqual(fieldColor(20, -20, 20), [255, 0, 0]);
});

test('sparse LSQR solves an overdetermined system without normal matrix inversion', () => {
  const rows = [[1, 2], [3, 1], [2, -1]].map(weights => ({ indices: [0, 1], weights }));
  const solution = lsqr(rows, [8, 9, 1], 2);
  assert.ok(Math.abs(solution[0] - 2) < 1e-9);
  assert.ok(Math.abs(solution[1] - 3) < 1e-9);
});

test('bending energy is zero for affine fields and rejects folding', () => {
  const field = createSpline(120, 90, 40, [1.05, 0.1, 0.02, 0.95, -20, 5]);
  for (const row of smoothRows(field, 0.1)) {
    for (const component of [0, 1]) {
      const value = row.indices.reduce((sum, index, local) => sum + row.weights[local] * field.coefficients[2 * index + component], 0);
      assert.ok(Math.abs(value) < 1e-12);
    }
  }
  assert.equal(geometryCheck(field).valid, true);
  assert.equal(geometryCheck(createSpline(120, 90, 40, [-1, 0, 0, 1, 120, 0])).valid, false);
});

export function syntheticFrames(field, step = 20) {
  const frames = [];
  for (let frameId = 0; frameId < 14; frameId++) {
    const theta = frameId === 0 ? 0 : 0.055 * Math.sin(frameId * 1.7);
    const tx = frameId === 0 ? 0 : 7 * Math.sin(frameId * 2.3);
    const ty = frameId === 0 ? 0 : 7 * Math.cos(frameId * 1.3);
    const points = [];
    for (let row = 1; row <= 4; row++) {
      for (let col = 1; col <= 5; col++) {
        const qx = step * (Math.cos(theta) * col - Math.sin(theta) * row) + tx;
        const qy = step * (Math.sin(theta) * col + Math.cos(theta) * row) + ty;
        const source = inversePoint(field, qx, qy, qx, qy, 1e-9);
        if (source) points.push({ col, row, x: source.x, y: source.y, confidence: 1 });
      }
    }
    frames.push({ id: frameId, timestamp: frameId * 100000, points, enabled: true, role: frameId >= 11 ? 'validation' : 'train' });
  }
  return frames;
}

test('joint identity reconstruction holds the reference gauge and validates unseen frames', async () => {
  const field = createSpline(128, 100, 50);
  const frames = syntheticFrames(field);
  const pose = poseFor(frames[0].points, field, 20);
  assert.ok(Math.abs(pose.theta) < 1e-12);
  const progress = [];
  const result = await fitCalibration(frames, { width: 128, height: 100, spacing: 50,
    step: 20, sigma: 1, lambda: 0.01, delta: 1.5, tau: 0.12, iterations: 10 }, null, update => progress.push(update));
  assert.equal(progress.length, result.metrics.completedIterations);
  assert.ok(progress.length > 0);
  for (const update of progress) {
    assert.ok(Number.isFinite(update.iterationSeconds) && update.iterationSeconds >= 0);
    assert.ok(Number.isFinite(update.iterationsPerSecond) && update.iterationsPerSecond > 0);
    assert.equal(update.accelerator, 'CPU');
    const phases = Object.values(update.profile.phasesMs);
    assert.equal(phases.length, 5);
    assert.ok(phases.every(value => Number.isFinite(value) && value >= 0));
    assert.ok(Math.abs(phases.reduce((sum, value) => sum + value, 0) - update.iterationSeconds * 1000) < 1e-6);
  }
  assert.deepEqual(result.metrics.fitProfile, progress.at(-1).profile);
  assert.deepEqual(result.poses[result.referenceId], { theta: 0, tx: 0, ty: 0 });
  assert.ok(result.metrics.validation.rms < 1e-6);
  assert.ok(result.metrics.training.rms < 1e-6);
});

test('fit skips a mirrored high-point reference frame', async () => {
  const frames = syntheticFrames(createSpline(128, 100, 50));
  const mirroredPoints = frames[0].points.flatMap(point => [
    { ...point, row: -point.row },
    { ...point, row: -point.row, confidence: 0.9 }
  ]);
  const mirrored = { ...frames[0], id: 100, points: mirroredPoints };
  const result = await fitCalibration([mirrored, ...frames], { width: 128, height: 100, spacing: 50,
    step: 20, sigma: 1, lambda: 0.01, delta: 1.5, tau: 0.12, iterations: 2 });
  assert.notEqual(result.referenceId, mirrored.id);
  assert.equal(geometryCheck(result.field).valid, true);
});

test('joint fit reconstructs an asymmetric field beyond the training observations', async () => {
  const truth = createSpline(128, 100, 40);
  for (let row = 0; row < truth.ny; row++) {
    for (let col = 0; col < truth.nx; col++) {
      const index = (row * truth.nx + col) * 2;
      truth.coefficients[index] = 3 * Math.sin(col * 0.9) * Math.cos(row * 0.7);
      truth.coefficients[index + 1] = 2 * Math.sin(row * 1.1 + col * 0.4);
    }
  }
  const frames = syntheticFrames(truth);
  const result = await fitCalibration(frames, { width: 128, height: 100, spacing: 40,
    step: 20, sigma: 1, lambda: 0.000001, delta: 1.5, tau: 0.12, iterations: 100 });
  assert.deepEqual(result.poses[result.referenceId], { theta: 0, tx: 0, ty: 0 });
  assert.ok(result.metrics.validation.rms < 0.1, `Held-out RMS: ${result.metrics.validation.rms}`);
  let maximumError = 0;
  for (let py = 22; py <= 78; py += 7) {
    for (let px = 22; px <= 98; px += 7) {
      const expected = evaluate(truth, px, py);
      const actual = evaluate(result.field, px, py);
      maximumError = Math.max(maximumError, Math.hypot(actual.x - expected.x, actual.y - expected.y));
    }
  }
  assert.ok(maximumError < 0.2, `Independent field error: ${maximumError}`);
  assert.equal(geometryCheck(result.field).valid, true);
});