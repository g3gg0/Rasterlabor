import test from 'node:test';
import assert from 'node:assert/strict';
import { registerPcbFftPair, registerPcbRotationPair, coarsePcbFftSeeds, registerPcbCoarsePair } from '../src/pcb-fft-pair.js';

const image = () => {
  const width = 192, height = 192, data = new Uint8ClampedArray(width * height * 4);
  for (let row = 0; row < height; row++) for (let column = 0; column < width; column++) {
    const trace = Math.abs(row - 38 - 8 * Math.sin(column / 13)) < 2 ||
      Math.abs(column - 95 - 9 * Math.sin(row / 12)) < 2 ||
      Math.abs(row - 128 + 7 * Math.sin(column / 8)) < 2;
    const value = 70 + (trace ? 100 : 0) + 8 * Math.sin(column * 0.35 + row * 0.17);
    const offset = 4 * (row * width + column);
    data[offset] = data[offset + 1] = data[offset + 2] = value; data[offset + 3] = 255;
  }
  return { width, height, data };
};

test('distributed FFT cells correct a predicted translation from identical PCB frames', () => {
  const first = image(), second = image();
  const result = registerPcbFftPair(first, second, { x: 0, y: 0, rotation: 0 },
    { x: 3, y: -2, rotation: 0 }, [-96, -96], [-96, -96], { cellSize: 64, cellsPerAxis: 3, searchRadius: 12 });
  assert.equal(result.accepted, true, JSON.stringify(result));
  assert.ok(Math.hypot(result.pose.x, result.pose.y) < 0.8, JSON.stringify(result));
  assert.ok(result.inlierCells.length >= 3);
  assert.ok(result.uniqueSupportArea <= first.width * first.height);
});

test('adaptive FFT cells find distributed structure inside a circular local selection', () => {
  const first = image(), second = image();
  for (let y = 0; y < first.height; y++) for (let x = 0; x < first.width; x++) {
    if (Math.hypot(x - 96, y - 96) > 88) {
      first.data[(y * first.width + x) * 4 + 3] = 0;
      second.data[(y * second.width + x) * 4 + 3] = 0;
    }
  }
  const poses = [{ x: 0, y: 0, rotation: 0 }, { x: 3, y: -2, rotation: 0 }];
  const options = { cellSize: 64, cellsPerAxis: 3, searchRadius: 12 };
  const sparse = registerPcbFftPair(first, second, ...poses, [-96, -96], [-96, -96], options);
  assert.equal(sparse.accepted, false);
  assert.equal(sparse.reason, 'Zu wenig unabhaengige Strukturzellen');
  const adaptive = registerPcbFftPair(first, second, ...poses, [-96, -96], [-96, -96],
    { ...options, adaptiveCells: true });
  assert.equal(adaptive.accepted, true, JSON.stringify(adaptive));
  assert.ok(adaptive.inlierCells.length >= 3);
  assert.ok(Math.hypot(adaptive.pose.x, adaptive.pose.y) < 0.8);
});

test('FFT pair measurement respects the image inclusion mask', () => {
  const first = image(), second = image();
  const mask = { sourceWidth: 192, sourceHeight: 192, width: 1, height: 1,
    cellSize: 192, data: new Uint8Array([0]) };
  const result = registerPcbFftPair(first, second, { x: 0, y: 0, rotation: 0 },
    { x: 0, y: 0, rotation: 0 }, [-96, -96], [-96, -96], { mask });
  assert.equal(result.accepted, false);
  assert.ok(result.cells.every(cell => cell.reason === 'Maske oder Rand'));
});

test('FFT cells find a small usable area away from the full image center', () => {
  const content = image();
  const canvas = { width: 512, height: 512, data: new Uint8ClampedArray(512 * 512 * 4) };
  for (let row = 0; row < content.height; row++) for (let column = 0; column < content.width; column++) {
    const source = 4 * (row * content.width + column);
    const target = 4 * ((row + 20) * canvas.width + column + 290);
    canvas.data.set(content.data.subarray(source, source + 4), target);
  }
  const result = registerPcbFftPair(canvas, canvas, { x: 0, y: 0, rotation: 0 },
    { x: 20, y: -15, rotation: 0 }, [-256, -256], [-256, -256],
    { cellSize: 64, cellsPerAxis: 3, searchRadius: 30, residualLimit: 4 });
  assert.equal(result.accepted, true, JSON.stringify(result));
  assert.ok(Math.hypot(result.pose.x, result.pose.y) < 1, JSON.stringify(result));
});

test('distributed cells recover a small rotation and translation of analytic PCB texture', () => {
  const width = 192, height = 192;
  const render = pose => {
    const data = new Uint8ClampedArray(width * height * 4);
    const cosine = Math.cos(pose.rotation), sine = Math.sin(pose.rotation);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const localX = x - width / 2, localY = y - height / 2;
      const worldX = pose.x + cosine * localX - sine * localY;
      const worldY = pose.y + sine * localX + cosine * localY;
      const trace = Math.abs(worldY - 35 - 8 * Math.sin(worldX / 13)) < 2 ||
        Math.abs(worldX - 20 - 9 * Math.sin(worldY / 12)) < 2 ||
        Math.abs(worldY + 35 + 7 * Math.sin(worldX / 8)) < 2;
      const value = 70 + (trace ? 100 : 0) + 8 * Math.sin(worldX * 0.35 + worldY * 0.17);
      const offset = 4 * (y * width + x);
      data[offset] = data[offset + 1] = data[offset + 2] = value;
      data[offset + 3] = 255;
    }
    return { width, height, data };
  };
  const truth = { x: 1.5, y: -0.8, rotation: Math.PI / 180 };
  const measured = registerPcbFftPair(render({ x: 0, y: 0, rotation: 0 }), render(truth),
    { x: 0, y: 0, rotation: 0 }, { x: 2.7, y: -1.8, rotation: 1.5 * Math.PI / 180 },
    [-96, -96], [-96, -96], { cellSize: 64, cellsPerAxis: 3, searchRadius: 12 });
  assert.equal(measured.accepted, true, JSON.stringify(measured));
  assert.ok(Math.hypot(measured.pose.x - truth.x, measured.pose.y - truth.y) < 0.5);
  assert.ok(Math.abs(measured.pose.rotation - truth.rotation) < 0.005);
});

test('moving high contrast reflection patches do not outweigh distributed traces', () => {
  const first = image(), second = image();
  const addReflection = (target, centerX, centerY) => {
    for (let y = 0; y < target.height; y++) for (let x = 0; x < target.width; x++) {
      if (Math.hypot(x - centerX, y - centerY) > 24) continue;
      const value = 120 + 110 * Math.sin((x - centerX) * 1.6 + (y - centerY) * 1.1) *
        Math.cos((x - centerX) * 0.9 - (y - centerY) * 1.5);
      const offset = 4 * (y * target.width + x);
      target.data[offset] = target.data[offset + 1] = target.data[offset + 2] = value;
    }
  };
  addReflection(first, 48, 48);
  addReflection(second, 153, 150);
  const result = registerPcbFftPair(first, second, { x: 0, y: 0, rotation: 0 },
    { x: 3, y: -2, rotation: 0 }, [-96, -96], [-96, -96],
    { cellSize: 64, cellsPerAxis: 3, searchRadius: 12 });
  assert.equal(result.accepted, true, JSON.stringify(result));
  assert.ok(Math.hypot(result.pose.x, result.pose.y) < 0.5);
  assert.ok(result.inlierCells.length >= 5);
});

test('invalid overlap and flat content are rejected', () => {
  const first = image(), second = image();
  assert.equal(registerPcbFftPair(first, second, { x: 0, y: 0, rotation: 0 },
    { x: 170, y: 0, rotation: 0 }, [-96, -96], [-96, -96]).accepted, false);
  first.data.fill(100); second.data.fill(100);
  for (let offset = 3; offset < first.data.length; offset += 4) first.data[offset] = second.data[offset] = 255;
  assert.equal(registerPcbFftPair(first, second, { x: 0, y: 0, rotation: 0 },
    { x: 0, y: 0, rotation: 0 }, [-96, -96], [-96, -96]).accepted, false);
});

test('rotation search recovers a four degree seed error and obeys angular bounds', () => {
  const first = image(), second = image();
  const prediction = { x: 2, y: -1, rotation: 4 * Math.PI / 180 };
  const offsets = [-96, -96];
  const options = { cellSize: 64, cellsPerAxis: 3, searchRadius: 20,
    residualLimit: 2, angle: 5, radius: 20 };
  const result = registerPcbRotationPair(first, second, { x: 0, y: 0, rotation: 0 },
    prediction, offsets, offsets, options);
  assert.ok(result?.accepted, JSON.stringify(result));
  assert.ok(result.inlierCells.length >= 5);
  assert.ok(Math.hypot(result.pose.x, result.pose.y) < .8);
  assert.ok(Math.abs(result.pose.rotation) < .005);
  const bounded = registerPcbRotationPair(first, second, { x: 0, y: 0, rotation: 0 },
    prediction, offsets, offsets, { ...options, angle: .5 });
  assert.equal(bounded, null);
});

test('rotation search rejects flat or unrelated content', () => {
  const first = image(), second = image();
  for (let index = 0; index < second.data.length; index += 4)
    second.data[index] = second.data[index + 1] = second.data[index + 2] = 100;
  assert.equal(registerPcbRotationPair(first, second, { x: 0, y: 0, rotation: 0 },
    { x: 0, y: 0, rotation: .05 }, [-96, -96], [-96, -96],
    { cellSize: 64, angle: 5 }), null);
});

test('whole-image FFT seeds recover an offset outside the local cell search', () => {
  const first = image(), second = image();
  const reference = { x: 0, y: 0, rotation: 0 };
  const prediction = { x: 60, y: -8, rotation: 2 * Math.PI / 180 };
  const offsets = [-96, -96];
  const options = { cellSize: 64, cellsPerAxis: 3, searchRadius: 20,
    residualLimit: 3, angle: 5, radius: 90 };
  const initial = registerPcbFftPair(first, second, reference, prediction, offsets, offsets, options);
  assert.equal(initial.accepted, false);
  const coarse = coarsePcbFftSeeds(first, second, reference, prediction, offsets, offsets, options);
  assert.ok(coarse.seeds.some(seed => Math.hypot(seed.pose.x, seed.pose.y) < 8));
  const result = registerPcbCoarsePair(first, second, reference, prediction, offsets, offsets, options);
  assert.ok(result?.accepted, JSON.stringify(result));
  assert.ok(result.inlierCells.length >= 5);
  assert.ok(Math.hypot(result.pose.x, result.pose.y) < .8);
  assert.ok(Math.abs(result.pose.rotation) < .005);
});

test('whole-image FFT does not propose poses from masked or flat images', () => {
  const first = image(), second = image(), pose = { x: 0, y: 0, rotation: 0 };
  const mask = { sourceWidth: 192, sourceHeight: 192, width: 1, height: 1,
    cellSize: 192, data: new Uint8Array([0]) };
  assert.deepEqual(coarsePcbFftSeeds(first, second, pose, pose, [-96,-96], [-96,-96], {mask}).seeds, []);
  for(let i=0;i<first.data.length;i+=4) first.data[i]=first.data[i+1]=first.data[i+2]=100;
  assert.deepEqual(coarsePcbFftSeeds(first, second, pose, pose, [-96,-96], [-96,-96]).seeds, []);
});
