import { detectGrid } from './detector.js';
import { fitCalibration, geometryCheck } from './solver.js';
import { buildMaps, remapRGBA } from './maps.js';
import { exportCalibration, importCalibration } from './format.js';
import { PatchTracker } from './patch-tracker.js';
import { WindowTracker } from './window-tracker.js';
import { patchGpuStatus } from './webgpu-patch-tracker.js';
import { remapGpuStatus, WebGpuRemapper } from './webgpu-remapper.js';
import { restoreDetectionScale, scaleDetectionOptions } from './detection-scale.js';
import { gpuGrayscaleBitmap, gpuImageStatus } from './webgpu-image.js';
import { buildInverseMapsGpu, mapBuilderGpuStatus } from './webgpu-map-builder.js';
import { createSpline } from './spline.js';

let calibration = null;
let cancelled = false;
let active = false;
const patchTracker = new PatchTracker();
const windowTracker = new WindowTracker();
const gpuRemapper = new WebGpuRemapper();
const progress = value => self.postMessage({ progress: value });

function calibrationTransferCopy(value) {
  const coefficients = value.field.coefficients.slice();
  const forward = value.maps.forward.slice();
  const inverseX = value.maps.inverseX.slice();
  const inverseY = value.maps.inverseY.slice();
  const sourceCoverage = value.maps.sourceCoverage.slice();
  const valid = value.maps.valid.slice();
  const numericalValid = value.maps.numericalValid.slice();
  return { value: { ...value, field: { ...value.field, coefficients }, maps: { ...value.maps,
    forward, inverseX, inverseY, sourceCoverage, valid, numericalValid } },
  transfer: [coefficients.buffer, forward.buffer, inverseX.buffer, inverseY.buffer,
    sourceCoverage.buffer, valid.buffer, numericalValid.buffer] };
}

function imageFromBitmap(bitmap) {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0);
  return context.getImageData(0, 0, bitmap.width, bitmap.height);
}

self.onmessage = async ({ data }) => {
  if (data.type === 'cancel') {
    cancelled = true;
    self.postMessage({ id: data.id, result: true });
    return;
  }
  if (active) { self.postMessage({ id: data.id, error: 'Rechen-Worker ist beschaeftigt.' }); return; }
  active = true;
  try {
    let result;
    let transfer = [];
    if (data.type === 'patch-gpu-status') result = await patchGpuStatus();
    else if (data.type === 'gpu-image-status') result = await gpuImageStatus();
    else if (data.type === 'map-builder-gpu-status') result = await mapBuilderGpuStatus();
    else if (data.type === 'map-builder-gpu-self-test') {
      const width = data.width ?? 96;
      const height = data.height ?? 72;
      const field = createSpline(width, height, Math.max(24, Math.round(Math.min(width, height) / 4)),
        [1.02, 0.035, -0.018, 0.98, -0.07 * width, 0.07 * height]);
      const frames = Array.from({ length: 4 }, (_, frame) => ({ enabled: true, role: 'train', points:
        Array.from({ length: 48 }, (_, index) => ({ x: width * (0.05 + index % 8 * 0.125),
          y: height * (0.05 + Math.floor(index / 8) * 0.18) })) }));
      const synthetic = { field, step: width / 8, parameters: { approxStep: width / 8 } };
      const cpuStarted = performance.now();
      const cpu = await buildMaps(synthetic, frames);
      const cpuMs = performance.now() - cpuStarted;
      const gpuStarted = performance.now();
      const gpu = await buildMaps(synthetic, frames, () => {}, () => false, buildInverseMapsGpu);
      const gpuMs = performance.now() - gpuStarted;
      let maskDifferences = 0;
      let maximumCoordinateDifference = 0;
      for (let index = 0; index < cpu.numericalValid.length; index++) {
        if (cpu.numericalValid[index] !== gpu.numericalValid[index]) maskDifferences++;
        if (cpu.numericalValid[index] && gpu.numericalValid[index]) maximumCoordinateDifference = Math.max(maximumCoordinateDifference,
          Math.abs(cpu.inverseX[index] - gpu.inverseX[index]), Math.abs(cpu.inverseY[index] - gpu.inverseY[index]));
      }
      result = { accelerator: gpu.accelerator, cpuMs, gpuMs, maskDifferences, maximumCoordinateDifference,
        cpuValid: cpu.roundtrip.numericalCount, gpuValid: gpu.roundtrip.numericalCount, maximumRoundtrip: gpu.roundtrip.maximum };
    }
    else if (data.type === 'remap-gpu-status') result = await remapGpuStatus();
    else if (data.type === 'remap-gpu-self-test') {
      const image = { width: 4, height: 4, data: new Uint8ClampedArray(4 * 4 * 4) };
      for (let index = 0; index < 16; index++) image.data.set([index * 11, index * 7, index * 3, 255], index * 4);
      const maps = { outputWidth: 2, outputHeight: 2, inverseX: new Float32Array([0.5, 1.25, 0.75, 1.5]),
        inverseY: new Float32Array([0.5, 0.75, 1.5, 1.25]), valid: new Uint8Array([1, 1, 1, 1]) };
      const cpu = remapRGBA(image, maps);
      const gpu = await gpuRemapper.remap(image, maps);
      result = { available: Boolean(gpu), maximumChannelDifference: gpu ? Math.max(...cpu.data.map((value, index) => Math.abs(value - gpu.data[index]))) : null,
        timing: gpu?.timing };
    }
    else if (data.type === 'track-window') {
      result = windowTracker.process(data.image, data.index, data.options.rectangle, data.options.patchSearchRadius, data.options.maxRotation);
    }
    else if (data.type === 'detect' || data.type === 'detect-bitmap') {
      const started = performance.now();
      if (data.type === 'detect-bitmap') {
        try {
          const reduced = await gpuGrayscaleBitmap(data.bitmap, 3840, data.options);
          if (reduced) {
            const detectStarted = performance.now();
            result = detectGrid(reduced, { ...reduced.options, precomputedAngle: reduced.angle,
              precomputedCandidates: reduced.cornerCandidates, precomputedCorners: reduced.refinedCorners });
            result = restoreDetectionScale(result, data.bitmap.width, data.bitmap.height, reduced.width, reduced.height);
            result.accelerator = 'WebGPU-Hybrid';
            result.timing = { ...reduced.timing, detectMs: performance.now() - detectStarted, scale: reduced.width / data.bitmap.width };
          } else result = detectGrid(imageFromBitmap(data.bitmap), data.options);
        } finally { data.bitmap.close(); }
      }
      else if (data.options.pattern === 'patches') {
        result = await patchTracker.processAsync(data.image, data.index, data.options);
        if (result.interrupted) {
          patchTracker.reset();
          result = await patchTracker.processAsync(data.image, data.index, data.options);
        }
      }
      else { patchTracker.reset(); result = detectGrid(data.image, data.options); }
      result.processingMs = performance.now() - started;
    }
    else if (data.type === 'fit') {
      cancelled = false;
      const fitted = await fitCalibration(data.frames, data.options, data.fresh ? null : calibration, progress, () => cancelled);
      fitted.parameters = data.options;
      fitted.maps = await buildMaps(fitted, data.frames, progress, () => cancelled,
        data.options.useWebGpu ? buildInverseMapsGpu : null);
      fitted.version = (calibration?.version ?? 0) + 1;
      fitted.quality = fitted.metrics.validation.count >= 20 && fitted.metrics.validation.frameCount >= 2 &&
        fitted.maps.roundtrip.validFraction >= 0.6 && fitted.metrics.validation.p95 <= data.options.acceptance &&
        !fitted.metrics.validation.inversionFailures && !fitted.metrics.stoppedByGeometry ? 'validated' : 'provisional';
      calibration = fitted;
      const response = calibrationTransferCopy(calibration);
      result = response.value;
      transfer = response.transfer;
    } else if (data.type === 'remap') {
      if (!calibration) throw new Error('Keine Kalibrierung geladen.');
      if (data.image.width !== calibration.field.width || data.image.height !== calibration.field.height) throw new Error('Abweichende Quellaufloesung.');
      const started = performance.now();
      result = null;
      if (data.useWebGpu) {
        try { result = await gpuRemapper.remap(data.image, calibration.maps); }
        catch { gpuRemapper.reset(); }
      }
      if (result) result.accelerator = 'WebGPU';
      else { result = remapRGBA(data.image, calibration.maps); result.accelerator = 'CPU'; }
      result.processingMs = performance.now() - started;
      transfer = [result.data.buffer];
    } else if (data.type === 'export') {
      result = exportCalibration(calibration, data.frames, data.video, data.parameters, data.opticalConfiguration, data.tracking);
      transfer = [result.buffer];
    } else if (data.type === 'import') {
      const imported = importCalibration(data.bytes);
      const geometry = geometryCheck(imported.calibration.field, imported.parameters?.tau ?? 0.12);
      if (!geometry.valid) throw new Error(`Geladenes Feld ungueltig: ${geometry.reason}`);
      patchTracker.reset();
      calibration = imported.calibration;
      const response = calibrationTransferCopy(calibration);
      result = { ...imported, calibration: response.value };
      transfer = response.transfer;
    } else if (data.type === 'reset') { calibration = null; patchTracker.reset(); windowTracker.reset(); gpuRemapper.reset(); result = true; }
    else throw new Error(`Unbekannter Workerauftrag: ${data.type}`);
    self.postMessage({ id: data.id, result }, transfer);
  } catch (error) { self.postMessage({ id: data.id, error: error.message }); }
  finally { active = false; }
};