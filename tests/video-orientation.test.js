import test from 'node:test';
import assert from 'node:assert/strict';
import { orientationFromMatrix } from '../src/video-orientation.js';

const fixed = value => value * 65536;
const matrix = (a, b, c, d) => [fixed(a), fixed(b), 0, fixed(c), fixed(d), 0, 0, 0, 1073741824];

test('track matrices produce bounded pixel orientations', () => {
  const expected = [
    [[1, 0, 0, 1], 6, 4], [[0, 1, -1, 0], 4, 6], [[-1, 0, 0, -1], 6, 4], [[0, -1, 1, 0], 4, 6],
    [[-1, 0, 0, 1], 6, 4], [[1, 0, 0, -1], 6, 4], [[0, 1, 1, 0], 4, 6], [[0, -1, -1, 0], 4, 6]
  ];
  for (const [[a, b, c, d], width, height] of expected) {
    const orientation = orientationFromMatrix(matrix(a, b, c, d), 6, 4);
    assert.equal(orientation.width, width);
    assert.equal(orientation.height, height);
    const corners = [[0, 0], [6, 0], [0, 4], [6, 4]].map(([x, y]) => [
      a * x + c * y + orientation.translateX, b * x + d * y + orientation.translateY
    ]);
    assert.equal(Math.min(...corners.map(([x]) => x)), 0);
    assert.equal(Math.max(...corners.map(([x]) => x)), width);
    assert.equal(Math.min(...corners.map(([, y]) => y)), 0);
    assert.equal(Math.max(...corners.map(([, y]) => y)), height);
  }
});

test('free track transformations remain unsupported', () => {
  assert.throws(() => orientationFromMatrix(matrix(1, 0, 1, 1), 6, 4), /nicht unterstuetzte freie Transformation/);
});