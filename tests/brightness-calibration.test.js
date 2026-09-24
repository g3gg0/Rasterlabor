import test from 'node:test';
import assert from 'node:assert/strict';
import { applyBrightnessCalibration, brightnessDisplayRange, brightnessFieldPixels,
  decodeBrightnessCalibration, encodeBrightnessCalibration } from '../src/brightness-calibration.js';

test('brightness application preserves alpha and leaves an identity field unchanged', () => {
  const image = { width: 2, height: 1, data: new Uint8ClampedArray([50, 100, 150, 255, 20, 30, 40, 0]) };
  applyBrightnessCalibration(image, { width: 2, height: 1, gain: new Float32Array([1, 2]) });
  assert.deepEqual([...image.data], [50, 100, 150, 255, 20, 30, 40, 0]);
});

test('brightness application multiplies all channels in linear light', () => {
  const image = { width: 1, height: 1, data: new Uint8ClampedArray([128, 64, 32, 255]) };
  applyBrightnessCalibration(image, { width: 1, height: 1, gain: new Float32Array([0.5]) });
  const linear = value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  const srgb = value => value <= 0.0031308 ? value * 12.92 : 1.055 * value ** (1 / 2.4) - 0.055;
  assert.deepEqual([...image.data], [128, 64, 32].map(value => Math.round(255 * srgb(linear(value / 255) * 0.5))).concat(255));
});

test('brightness field preview is grayscale and transparent outside the supported mask', () => {
  const calibration = { width: 2, height: 1, gain: new Float32Array([1, 0.8]), supported: new Uint8Array([255, 255]) };
  assert.deepEqual([...brightnessFieldPixels(calibration, 2, 1, x => x === 0)], [128, 128, 128, 255, 0, 0, 0, 0]);
});

test('brightness preview range is symmetric and ignores unsupported outliers', () => {
  const calibration = { width: 5, height: 1, gain: new Float32Array([1 / 0.8, 1 / 0.9, 1, 1 / 1.1, 0.1]),
    supported: new Uint8Array([255, 255, 255, 255, 0]) };
  const range = brightnessDisplayRange(calibration);
  assert.ok(Math.abs(range.low - 0.8) < 1e-6);
  assert.ok(Math.abs(range.high - 1.2) < 1e-6);
  assert.deepEqual([...brightnessFieldPixels(calibration, 5, 1, null, range).filter((_, index) => index % 4 === 0)], [0, 64, 128, 191, 0]);
});

test('brightness package roundtrips and rejects another geometric calibration', () => {
  const calibration = { version: 2, width: 2, height: 1, gain: new Float32Array([0.75, 1.25]),
    supported: new Uint8Array([255, 0]), model: { representation: 'checkerboard-white-native-block-field' }, metrics: { validationRms: 0.02 } };
  const bytes = encodeBrightnessCalibration(calibration, 'geometry-a');
  const decoded = decodeBrightnessCalibration(bytes, { width: 2, height: 1, geometryIdentity: 'geometry-a' });
  assert.deepEqual([...decoded.gain], [...calibration.gain]);
  assert.deepEqual([...decoded.supported], [255, 0]);
  assert.throws(() => decodeBrightnessCalibration(bytes, { width: 2, height: 1, geometryIdentity: 'geometry-b' }), /anderen Optik oder Entzerrung/);
});