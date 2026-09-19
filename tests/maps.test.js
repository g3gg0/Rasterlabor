import test from 'node:test';
import assert from 'node:assert/strict';
import { createSpline, evaluate } from '../src/spline.js';
import { buildMaps, coverageGrid, heatmapGridSize, relativeCoverageScale, remapRGBA } from '../src/maps.js';

test('relative coverage stretches observed counts without coloring unknown areas as low coverage', () => {
  const counts = new Uint8Array([0, 40, 41, 42]);
  const scale = relativeCoverageScale(counts);
  assert.equal(scale.minimum, 40);
  assert.equal(scale.maximum, 42);
  assert.deepEqual(scale.colors[0], [135, 142, 138, 255]);
  assert.deepEqual(scale.colors[40], [235, 180, 60, 255]);
  assert.deepEqual(scale.colors[42], [25, 145, 165, 255]);
  assert.notDeepEqual(scale.colors[40], scale.colors[41]);
  assert.deepEqual(counts, new Uint8Array([0, 40, 41, 42]));
  for (const values of [[], [0, 0], [255, 255], [3, 3]]) {
    const uniform = relativeCoverageScale(new Uint8Array(values));
    assert.ok(uniform.colors.flat().every(Number.isFinite));
    assert.equal(uniform.minimum, uniform.maximum);
  }
});
import { exportCalibration, importCalibration } from '../src/format.js';
import { unzipSync, zipSync, strToU8 } from 'fflate';
import { createPatchMask, MASK_FORBIDDEN, MASK_SEARCH, paintPatchMask, remapInclusionMask } from '../src/patch-mask.js';

test('raw-frame inclusion masks follow inverse maps into rectified coordinates', () => {
  const source = createPatchMask(8, 4, 1);
  source.data[1 * source.width + 5] = MASK_SEARCH;
  source.revision = 7;
  const maps = { outputWidth: 2, outputHeight: 2,
    inverseX: new Float32Array([5, 1, 5, 5]), inverseY: new Float32Array([1, 1, 1, 1]),
    valid: new Uint8Array([1, 1, 0, 1]) };
  const target = remapInclusionMask(source, maps);
  assert.deepEqual([...target.data], [MASK_SEARCH, 0, 0, MASK_SEARCH]);
  assert.deepEqual([target.sourceWidth, target.sourceHeight, target.cellSize, target.revision], [2, 2, 1, 7]);
});

test('dense inverse, coverage mask, signed forward coordinates and lossless ZIP roundtrip', async () => {
  const field = createSpline(48, 40, 24, [1, 0.04, 0, 1, -10, -8]);
  const frames = Array.from({ length: 4 }, (_, id) => ({ id, timestamp: id * 100000,
    enabled: true, role: 'train', points: Array.from({ length: 30 }, (_, index) => ({
      x: 4 + index % 6 * 7, y: 4 + Math.floor(index / 6) * 7, col: index % 6 + 0.125, row: Math.floor(index / 6) + 0.25, confidence: 1
    })) }));
  const calibration = { field, referenceId: 0, step: 7, version: 1, quality: 'provisional', metrics: {}, poses: {}, parameters: { approxStep: 9 } };
  calibration.maps = await buildMaps(calibration, frames);
  assert.ok(calibration.maps.forward[0] < 0);
  assert.ok(calibration.maps.roundtrip.validCount > 300);
  assert.ok(calibration.maps.roundtrip.maximum < 0.01);
  const image = { width: 48, height: 40, data: new Uint8ClampedArray(48 * 40 * 4).fill(200) };
  const rectified = remapRGBA(image, calibration.maps);
  assert.ok(rectified.data.some(value => value === 200));
  const tracking = { format: 'rasterlabor-xyr-tracking', model_version: 1,
    path: [{ frame: 12, timestamp: 400000, pose: { x: 1.25, y: -2.5, rotation: 0.01 } }] };
  const brightness = { version: 2, width: calibration.maps.outputWidth, height: calibration.maps.outputHeight,
    gain: new Float32Array(calibration.maps.outputWidth * calibration.maps.outputHeight).fill(1.125),
    supported: new Uint8Array(calibration.maps.outputWidth * calibration.maps.outputHeight).fill(255),
    model: { representation: 'checkerboard-white-field' }, metrics: { validationRms: 0.03 } };
  const packed = exportCalibration(calibration, frames, { name: 'test.mp4' }, { gridMm: null }, 'synthetic', tracking, brightness);
  const restored = importCalibration(packed);
  assert.equal(restored.metadata.mm_per_pixel, null);
  assert.equal(restored.metadata.dpi, null);
  assert.deepEqual(restored.calibration.maps.forward, calibration.maps.forward);
  assert.deepEqual(restored.calibration.maps.inverseX, calibration.maps.inverseX);
  assert.deepEqual(restored.calibration.field.coefficients, field.coefficients);
  assert.deepEqual(restored.calibration.maps.coverageGrid,
    { cols: calibration.maps.coverageGrid.cols, rows: calibration.maps.coverageGrid.rows });
  assert.deepEqual(restored.observations, frames);
  assert.deepEqual(restored.tracking, tracking);
  assert.deepEqual(restored.brightness.gain, brightness.gain);
  assert.deepEqual(restored.brightness.supported, brightness.supported);
  assert.deepEqual(restored.brightness.model, brightness.model);
  assert.deepEqual(restored.brightness.metrics, brightness.metrics);
  const savedAgain = importCalibration(exportCalibration(restored.calibration, restored.observations,
    restored.video, restored.parameters, restored.opticalConfiguration, restored.tracking));
  assert.deepEqual(savedAgain.tracking, tracking);
  const withoutTracking = exportCalibration(calibration, frames, {}, { gridMm: null }, '');
  assert.equal(unzipSync(withoutTracking)['tracking.json'], undefined);
  assert.equal(unzipSync(withoutTracking)['brightness-gain.bin'], undefined);
  const emptyTracking = exportCalibration(calibration, frames, {}, { gridMm: null }, '', { ...tracking, path: [] });
  assert.equal(unzipSync(emptyTracking)['tracking.json'], undefined);
  const damaged = unzipSync(packed);
  damaged['tracking.json'] = strToU8(JSON.stringify({ ...tracking, path: [tracking.path[0], tracking.path[0]] }));
  assert.throws(() => importCalibration(zipSync(damaged)), /Trackingframe/);
  const damagedBrightness = unzipSync(packed);
  const damagedMetadata = JSON.parse(new TextDecoder().decode(damagedBrightness['metadata.json']));
  damagedMetadata.brightness.width++;
  damagedBrightness['metadata.json'] = strToU8(JSON.stringify(damagedMetadata));
  assert.throws(() => importCalibration(zipSync(damagedBrightness)), /Helligkeitsfeld/);
  for (let index = 0; index < calibration.maps.valid.length; index += 19) {
    if (!calibration.maps.valid[index]) continue;
    const point = evaluate(field, calibration.maps.inverseX[index], calibration.maps.inverseY[index]);
    const px = index % calibration.maps.outputWidth + calibration.maps.origin[0];
    const py = Math.floor(index / calibration.maps.outputWidth) + calibration.maps.origin[1];
    assert.ok(Math.hypot(point.x - px, point.y - py) < 0.01);
  }
});

test('calibration ZIP stores the patch mask as validated binary data', async () => {
  const field = createSpline(48, 40, 24, [1, 0, 0, 1, 0, 0]);
  const frames = [{ id: 0, timestamp: 0, enabled: true, role: 'train', points: [
    { x: 8, y: 8, col: 0, row: 0, confidence: 1 }
  ] }];
  const calibration = { field, referenceId: 0, step: 7, version: 1, quality: 'provisional', metrics: {}, poses: {}, maps: null };
  calibration.maps = await buildMaps(calibration, frames);
  const patchMask = createPatchMask(48, 40, 4);
  paintPatchMask(patchMask, 8, 8, 5, MASK_SEARCH);
  paintPatchMask(patchMask, 36, 28, 5, MASK_FORBIDDEN);
  const restored = importCalibration(exportCalibration(calibration, frames, {}, { gridMm: null, patchMask }, ''));
  assert.equal(restored.tracking, null);
  assert.deepEqual(restored.parameters.patchMask.data, patchMask.data);
  assert.equal(restored.parameters.patchMask.cellSize, 4);
  assert.equal(restored.metadata.parameters.patchMask, undefined);
});

test('red patch-mask border crops the rectified output and invalidates red source pixels', async () => {
  const field = createSpline(48, 40, 24, [1, 0, 0, 1, 0, 0]);
  const frames = Array.from({ length: 3 }, (_, id) => ({ id, enabled: true, role: 'train', points: [
    { x: 8, y: 8 }, { x: 40, y: 8 }, { x: 8, y: 32 }, { x: 40, y: 32 }
  ] }));
  const patchMask = createPatchMask(48, 40, 4);
  for (let row = 0; row < patchMask.height; row++) {
    for (let col = 0; col < patchMask.width; col++) {
      if (row === 0 || col === 0 || row === patchMask.height - 1 || col === patchMask.width - 1) {
        patchMask.data[row * patchMask.width + col] = MASK_FORBIDDEN;
      }
    }
  }
  const calibration = { field, step: 8, parameters: { pattern: 'patches', patchSize: 8, patchMask } };
  const maps = await buildMaps(calibration, frames);
  assert.deepEqual(maps.origin, [4, 4]);
  assert.equal(maps.outputWidth, 40);
  assert.equal(maps.outputHeight, 32);
  assert.equal(maps.numericalValid.length, 40 * 32);
  assert.ok(maps.numericalValid.every(Boolean));
});

test('patch coverage uses patch footprints instead of the unit coordinate step', () => {
  const frames = Array.from({ length: 3 }, (_, id) => ({ id, enabled: true, role: 'train', points: [
    { x: 50, y: 50 }, { x: 150, y: 100 }, { x: 250, y: 150 }
  ] }));
  const coverage = coverageGrid(frames, 300, 200, 1, 6, 4, 32);
  assert.ok([...coverage.counts].filter(count => count >= 3).length >= 3);
});

test('heatmap grid uses small adaptive cells with a bounded grid size', () => {
  assert.deepEqual(heatmapGridSize(2160, 3840), { cols: 90, rows: 160 });
  assert.deepEqual(heatmapGridSize(320, 240), { cols: 24, rows: 18 });
  assert.deepEqual(heatmapGridSize(8000, 6000), { cols: 160, rows: 160 });
});

test('map generation accepts an accelerated inverse builder and falls back when it fails', async () => {
  const field = createSpline(24, 20, 12);
  const frames = Array.from({ length: 3 }, () => ({ enabled: true, role: 'train', points: [
    { x: 2, y: 2 }, { x: 21, y: 2 }, { x: 2, y: 17 }, { x: 21, y: 17 }
  ] }));
  const calibration = { field, step: 8, parameters: { approxStep: 8 } };
  const cpu = await buildMaps(calibration, frames);
  let called = false;
  const accelerated = await buildMaps(calibration, frames, () => {}, () => false, async () => {
    called = true;
    return { inverseX: cpu.inverseX.slice(), inverseY: cpu.inverseY.slice(), valid: cpu.valid.slice(),
      numericalValid: cpu.numericalValid.slice(), numericalCount: cpu.roundtrip.numericalCount,
      validCount: cpu.roundtrip.validCount, maxRoundtrip: cpu.roundtrip.maximum, roundtrip: [],
      cropBounds: [0, 0, cpu.outputWidth - 1, cpu.outputHeight - 1], accelerator: 'WebGPU', timing: { totalMs: 1 } };
  });
  assert.equal(called, true);
  assert.equal(accelerated.accelerator, 'WebGPU');
  assert.deepEqual(accelerated.inverseX, cpu.inverseX);
  const fallback = await buildMaps(calibration, frames, () => {}, () => false, async () => { throw new Error('GPU failure'); });
  assert.equal(fallback.accelerator, 'CPU');
  assert.deepEqual(fallback.inverseX, cpu.inverseX);
});
