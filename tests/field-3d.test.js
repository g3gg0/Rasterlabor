import test from 'node:test';
import assert from 'node:assert/strict';
import { coveredTriangleIndices } from '../src/field-3d.js';

test('3D surface omits triangles touching uncovered vertices', () => {
  assert.deepEqual(coveredTriangleIndices(new Uint8Array([
    1, 1, 1,
    1, 0, 1,
    1, 1, 1
  ]), 3, 3), [
    0, 3, 1,
    5, 7, 8
  ]);
});

test('3D surface retains both triangles of fully covered cells', () => {
  assert.deepEqual(coveredTriangleIndices(new Uint8Array([1, 1, 1, 1]), 2, 2), [0, 2, 1, 1, 2, 3]);
  assert.deepEqual(coveredTriangleIndices(new Uint8Array([0, 0, 0, 0]), 2, 2), []);
});