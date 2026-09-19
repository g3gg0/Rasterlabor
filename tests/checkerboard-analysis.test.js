import test from 'node:test';
import assert from 'node:assert/strict';
import { analyzeCheckerboard, checkerboardCells, refineCheckerboardCell, summarizeCheckerboard } from '../src/checkerboard-analysis.js';

test('field boundaries must follow image edges, not displaced corner estimates or texture', () => {
  const width = 160; const height = 160; const step = 32;
  const data = Float32Array.from({ length: width * height }, (_, index) => {
    const x = index % width; const y = Math.floor(index / width);
    return ((Math.floor(x / step) + Math.floor(y / step)) % 2 ? 220 : 30) + 8 * Math.sin(x * 1.7 + y * 2.3);
  });
  const points = [0, 1].flatMap(row => [0, 1].map(col => ({ col, row, x: 63.5 + col * step, y: 63.5 + row * step })));
  assert.ok(refineCheckerboardCell(checkerboardCells(points)[0], { width, height, data }));
  points[0].y += 7;
  const refined = refineCheckerboardCell(checkerboardCells(points)[0], { width, height, data });
  assert.ok(refined, 'Recover the displaced corner instead of discarding the field');
  assert.ok(Math.abs(refined.corners[0].y - 63.5) < 1);
  assert.ok(Math.abs(refined.area - step ** 2) < step ** 2 * 0.02);
  data.fill(120);
  assert.equal(refineCheckerboardCell(checkerboardCells(points)[0], { width, height, data }), null);
});

test('edge validation preserves genuine skew and unequal field dimensions', () => {
  const width = 220; const height = 180;
  const data = Float32Array.from({ length: width * height }, (_, index) => {
    const x = index % width; const y = Math.floor(index / width);
    return (Math.floor((x - y * 0.2) / 40) + Math.floor(y / 32)) % 2 ? 230 : 20;
  });
  const points = [0, 1].flatMap(row => [0, 1].map(col => {
    const y = 63.5 + row * 32;
    return { col, row, x: 79.5 + col * 40 + y * 0.2, y };
  }));
  const cell = checkerboardCells(points)[0];
  const refined = refineCheckerboardCell(cell, { width, height, data });
  assert.ok(refined);
  assert.ok(Math.abs(refined.area - 1280) < 1280 * 0.02);
  assert.ok(Math.abs(cell.area - 1280) < 1e-8);
});

test('blurred textured checkerboard retains broad coverage and measured areas', () => {
  const width = 640; const height = 480; const step = 48;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let row = 0; row < height; row++) for (let col = 0; col < width; col++) {
    const board = Math.tanh(Math.sin(Math.PI * (col + 0.5) / step) * 5) * Math.tanh(Math.sin(Math.PI * (row + 0.5) / step) * 5);
    const texture = 18 * Math.sin(col * 0.73) * Math.sin(row * 0.57) + 9 * Math.sin(col * 0.21 + row * 0.31);
    const value = Math.round(140 + 65 * board + texture);
    data.set([value, value, value, 255], (row * width + col) * 4);
  }
  const result = analyzeCheckerboard({ width, height, data }, { step });
  assert.ok(result.cells.length >= 60, `Only ${result.cells.length} blurred fields recovered`);
  assert.ok(Math.abs(result.statistics.mean - step ** 2) < step ** 2 * 0.03);
  for (const cell of result.cells) assert.ok(Math.abs(cell.area - step ** 2) < step ** 2 * 0.06);
});

test('checkerboard area uses measured quadrilaterals and maximum-relative statistics', () => {
  const points = [0, 1].flatMap(row => [0, 1, 2].map(col => ({ col, row, x: col === 2 ? 25 : col * 10, y: row * 10 })));
  const cells = checkerboardCells(points); const summary = summarizeCheckerboard(cells);
  assert.deepEqual(cells.map(cell => cell.area), [100, 150]);
  assert.deepEqual(cells.map(cell => [cell.col, cell.row]), [[0, 0], [1, 0]]);
  assert.equal(summary.mean, 125); assert.equal(summary.deviation, 25);
  assert.equal(cells[0].relativeDeviation, 1 / 3); assert.equal(cells[1].relativeDeviation, 0);
  assert.equal(checkerboardCells(points.slice(1)).length, 1);
  assert.equal(summarizeCheckerboard([]).maximum, null);
});

test('overlapping regional detection measures many cells without duplicates', () => {
  const width = 640; const height = 480; const step = 32;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let row = 0; row < height; row++) for (let col = 0; col < width; col++) {
    const value = (Math.floor(col / step) + Math.floor(row / step)) % 2 ? 230 : 20;
    data.set([value, value, value, 255], (row * width + col) * 4);
  }
  const result = analyzeCheckerboard({ width, height, data }, { step });
  assert.ok(result.cells.length > 180, `Only ${result.cells.length} cells`);
  assert.ok(Math.abs(result.statistics.mean - step ** 2) < 1);
  for (const [index, cell] of result.cells.entries()) assert.ok(result.cells.slice(index + 1).every(other => Math.hypot(cell.x - other.x, cell.y - other.y) > step / 2));
  data.fill(0);
  assert.equal(analyzeCheckerboard({ width, height, data }, { step }).cells.length, 0);
});