import { detectGrid } from './detector.js';
import { analyzeCheckerboard } from './checkerboard-analysis.js';
import { fitCalibration, geometryCheck } from './solver.js';
import { buildMaps, remapRGBA } from './maps.js';
import { exportCalibration, importCalibration } from './format.js';
import { PatchTracker } from './patch-tracker.js';
import { WindowTracker } from './window-tracker.js';
import { ContextTracker, contextImage, contextSearchRadii, registerOverlap, shouldRunContext } from './context-tracker.js';
import { patchGpuStatus } from './webgpu-patch-tracker.js';
import { remapGpuStatus, WebGpuRemapper } from './webgpu-remapper.js';
import { restoreDetectionScale, scaleDetectionOptions } from './detection-scale.js';
import { gpuGrayscaleBitmap, gpuImageStatus } from './webgpu-image.js';
import { buildInverseMapsGpu, mapBuilderGpuStatus } from './webgpu-map-builder.js';
import { createSpline } from './spline.js';
import { optimizePoseGraph } from './pose-graph-refit.js';
import { registerFeatureOverlap } from './feature-overlap.js';
import { applyBrightnessCalibration } from './brightness-calibration.js';

let calibration = null;
let brightnessCalibration = null;
let cancelled = false;
let active = false;
const patchTracker = new PatchTracker();
const windowTracker = new WindowTracker();
const contextTracker = new ContextTracker();
let contextDetection = null;
let trackingMaps = null;
const gpuRemapper = new WebGpuRemapper();
const progress = value => self.postMessage({ progress: value });
const usableLocalMatch = match => Boolean(match?.accepted || (match?.conditionallyAccepted &&
  Number.isFinite(match.score) && match.score >= 0.93 && match.pose &&
  [match.pose.x, match.pose.y, match.pose.rotation].every(Number.isFinite)));
const wrapAngle = angle => Math.atan2(Math.sin(angle), Math.cos(angle));
function boundedFeatureMatch(match, prediction, limits, width, height) {
  if (!match.accepted) return match;
  const correction = Math.hypot(match.pose.x - prediction.x, match.pose.y - prediction.y);
  const rotation = Math.abs(wrapAngle(match.pose.rotation - prediction.rotation)) * 180 / Math.PI;
  return correction <= Math.hypot(width, height) * 1.05 && rotation <= limits.angle ? match :
    { ...match, accepted: false, reason: 'Feature-Treffer ausserhalb Suchbereich' };
}

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
  if (active) { data.native?.frame?.close(); self.postMessage({ id: data.id, error: 'Rechen-Worker ist beschaeftigt.' }); return; }
  active = true;
  try {
    let result;
    let transfer = [];
    if (data.type === 'patch-gpu-status') result = await patchGpuStatus();
    else if (data.type === 'pose-graph-refit') {
      result = optimizePoseGraph(data.graph, data.options);
    }
    else if (data.type === 'local-refit-register') {
      const images = new Map();
      try {
        for (const [index, item] of data.images.entries()) {
          progress({ operation: 'local-refit', stage: 'prepare', image: index + 1, images: data.images.length, frame: item.frame });
          images.set(item.frame, {
            ...item, image: contextImage(imageFromBitmap(item.bitmap), item.mask ?? data.mask, data.preprocessing)
          });
        }
        result = [];
        for (const [index, pair] of data.pairs.entries()) {
          if (cancelled) throw new Error('Lokaler Refit abgebrochen.');
          const current = images.get(pair.current); const reference = images.get(pair.reference);
          if (!current || !reference) continue;
          progress({ operation: 'local-refit', stage: 'pair', pair: index + 1, pairs: data.pairs.length,
            current: current.frame, reference: reference.frame });
          let forward;
          let forwardSeed = current.pose;
          const attempts = [];
          const radii = contextSearchRadii(data.limits.radius, current.image.width, current.image.height, 'recent');
          for (const [attempt, radius] of radii.entries()) {
            progress({ operation: 'local-refit', stage: 'forward', pair: index + 1, pairs: data.pairs.length,
              current: current.frame, reference: reference.frame, attempt: attempt + 1, attempts: radii.length, radius });
            forward = registerOverlap(current.image, reference.image, forwardSeed, reference.pose,
              { ...data.limits, radius, coarseStep: radius > data.limits.radius ? data.limits.radius : 0,
                coarseRadiusFactor: 1.05, partial: true });
            if (!forward.accepted && forward.reason === 'Mehrdeutig' && forward.score >= 0.93) {
              forward = { ...forward, conditionallyAccepted: true };
            }
            attempts.push({ accepted: forward.accepted, reason: forward.reason, score: forward.score,
              margin: forward.margin, support: forward.support, evaluated: forward.evaluated, radius });
            if (usableLocalMatch(forward) || !['Suchgrenze', 'Korrelation', 'Mehrdeutig',
              'Unzureichende Struktur oder Ueberlappung'].includes(forward.reason)) break;
            if (forward.reason === 'Suchgrenze' && Number.isFinite(forward.score) && forward.score >= 0.8 &&
              forward.pose && [forward.pose.x, forward.pose.y, forward.pose.rotation].every(Number.isFinite)) forwardSeed = forward.pose;
          }
          if (!forward.accepted) {
            progress({ operation: 'local-refit', stage: 'feature-forward', pair: index + 1, pairs: data.pairs.length,
              current: current.frame, reference: reference.frame });
            const feature = boundedFeatureMatch(registerFeatureOverlap(current.image, reference.image, reference.pose),
              current.pose, data.limits, current.image.width, current.image.height);
            if (feature.accepted) forward = { ...feature, attempts, fallbackFor: forward };
            else forward.featureFallback = feature;
          }
          const reverseRadius = Math.min(32, data.limits.reverseRadius ?? data.limits.radius);
          if (usableLocalMatch(forward)) progress({ operation: 'local-refit', stage: 'backward', pair: index + 1, pairs: data.pairs.length,
            current: current.frame, reference: reference.frame, radius: reverseRadius });
          let backward = usableLocalMatch(forward) ? registerOverlap(reference.image, current.image, reference.pose, forward.pose,
            { radius: reverseRadius, angle: data.limits.angle, coarseStep: 0, partial: true }) : null;
          if (!backward?.accepted && backward?.reason === 'Mehrdeutig' && backward.score >= 0.93) {
            backward = { ...backward, conditionallyAccepted: true };
          }
          if (usableLocalMatch(forward) && !backward?.accepted) {
            progress({ operation: 'local-refit', stage: 'feature-backward', pair: index + 1, pairs: data.pairs.length,
              current: current.frame, reference: reference.frame });
            const feature = boundedFeatureMatch(registerFeatureOverlap(reference.image, current.image, forward.pose),
              reference.pose, data.limits, current.image.width, current.image.height);
            if (feature.accepted) backward = { ...feature, fallbackFor: backward };
            else if (backward) backward.featureFallback = feature;
          }
          const rotation = backward?.pose ? backward.pose.rotation - reference.pose.rotation : 0;
          const reverseDistance = usableLocalMatch(backward) ? Math.hypot(backward.pose.x - reference.pose.x, backward.pose.y - reference.pose.y) +
            Math.abs(Math.atan2(Math.sin(rotation), Math.cos(rotation))) * Math.hypot(current.image.width, current.image.height) / 2 : null;
          result.push({ ...pair, currentPose: current.pose, referencePose: reference.pose,
            forward: { ...forward, attempts }, backward, reverseDistance });
        }
      } finally { for (const item of data.images) item.bitmap.close(); }
    }
    else if (data.type === 'checkerboard-analysis') {
      try { result = analyzeCheckerboard(imageFromBitmap(data.bitmap), data.options); }
      finally { data.bitmap.close(); }
    }
    else if (data.type === 'context-inspect') {
      try {
        const { radius, angle, coarseStep, reverseRadius = radius } = data.limits;
        if (![radius, angle, coarseStep, reverseRadius].every(Number.isFinite) || radius < 1 || radius > 10000 || angle < 0.1 || angle > 10 || coarseStep < 0 || reverseRadius < 1 || reverseRadius > 10000 ||
          [data.prediction, data.referencePose].some(pose => !pose || ![pose.x, pose.y, pose.rotation].every(Number.isFinite))) {
          throw new Error('Ungueltige Debug-Suchparameter.');
        }
        const current = contextImage(imageFromBitmap(data.current), data.mask);
        const reference = contextImage(imageFromBitmap(data.reference), data.mask);
        const forward = registerOverlap(current, reference, data.prediction, data.referencePose, data.limits);
        const backward = forward.accepted ? { ...registerOverlap(reference, current, data.referencePose, forward.pose, { radius: reverseRadius, angle }), searchRadius: reverseRadius, angle } : null;
        const rotation = backward?.pose ? backward.pose.rotation - data.referencePose.rotation : 0;
        const reverseDistance = backward?.accepted ? Math.hypot(backward.pose.x - data.referencePose.x, backward.pose.y - data.referencePose.y) +
          Math.abs(Math.atan2(Math.sin(rotation), Math.cos(rotation))) * Math.hypot(current.width, current.height) / 2 : null;
        result = { ...forward, prediction: data.prediction, referencePose: data.referencePose,
          attempts: [{ ...forward, searchRadius: radius, angle, coarseStep }], backward, reverseDistance,
          accepted: forward.accepted && reverseDistance !== null && reverseDistance <= 1.5,
          reason: forward.accepted && !(reverseDistance !== null && reverseDistance <= 1.5) ? 'Rueckwaertspruefung' : forward.reason };
      } finally { data.current.close(); data.reference.close(); }
    }
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
    else if (data.type === 'tracking-maps') {
      if (contextDetection) throw new Error('Umfeldregistrierung noch offen.');
      const maps = data.maps;
      if (!maps || !Number.isSafeInteger(maps.outputWidth) || !Number.isSafeInteger(maps.outputHeight) || maps.outputWidth < 1 || maps.outputHeight < 1 ||
        [maps.inverseX, maps.inverseY, maps.valid].some(values => values?.length !== maps.outputWidth * maps.outputHeight)) throw new Error('Ungueltige Tracking-Maps.');
      contextTracker.reset(); windowTracker.reset(); trackingMaps = maps; result = true;
    }
    else if (data.type === 'track-window' || data.type === 'track-window-native') {
      if (contextDetection) throw new Error('Umfeldregistrierung noch offen.');
      let image = data.image, prepared = null, preview = null;
      if (data.native) {
        if (!trackingMaps) throw new Error('Tracking-Maps fehlen.');
        const started = performance.now();
        prepared = await contextTracker.image({ ...data.native, maps: trackingMaps, width: trackingMaps.outputWidth,
          height: trackingMaps.outputHeight, readRgba: true }, data.options);
        prepared.preparationMs = performance.now() - started;
        image = prepared.rgba; delete prepared.rgba;
      }
      result = windowTracker.process(image, data.index, data.options.rectangle, data.options.patchSearchRadius, data.options.maxRotation);
      const previousContextFrame = contextTracker.history.at(-1)?.frame ?? null;
      if (result.success && (data.options.contextRecent > 0 || data.options.contextSpatial > 0) &&
        shouldRunContext(data.index, previousContextFrame, data.options.contextInterval, result.score)) {
        const references = await contextTracker.begin(image, data.index, result.raw, data.options, prepared);
        contextDetection = { result, width: image.width, height: image.height };
        result = { ...result, references, contextPending: true };
      }
      else if (prepared) result.timing = { ...result.timing, totalMs: result.timing.totalMs + prepared.preparationMs };
      if (data.native) {
        preview = await createImageBitmap(new ImageData(image.data, image.width, image.height));
        result = { ...result, preview, width: image.width, height: image.height };
        transfer.push(preview);
      }
    }
    else if (data.type === 'track-window-seed') {
      if (contextDetection) throw new Error('Umfeldregistrierung noch offen.');
      contextTracker.reset(); windowTracker.reset();
      const seeded = windowTracker.process(data.image, data.index, data.rectangle, data.searchRadius, data.maxRotation);
      if (!seeded.success) throw new Error(seeded.reason || 'Letzter Trackingframe konnte nicht als Referenz geladen werden.');
      windowTracker.setPose(data.pose, data.image.width, data.image.height);
      result = true;
    }
    else if (data.type === 'context-reference') {
      const image = data.native ? { ...data.native, maps: trackingMaps, width: trackingMaps.outputWidth, height: trackingMaps.outputHeight } : data.image;
      await contextTracker.provide(data.index, image, data.error);
      result = true;
    }
    else if (data.type === 'context-finish') {
      const context = contextTracker.finish();
      const original = contextDetection.result;
      if (context.applied) windowTracker.setPose(context.pose, contextDetection.width, contextDetection.height);
      result = { ...original, incremental: original.raw, raw: { ...context.pose, points: Math.max(1, context.inliers.length) }, context,
        accelerator: `${original.accelerator} / Umfeld ${context.accelerator}`,
        timing: { ...original.timing, contextMs: context.milliseconds, contextPyramidMs: context.pyramidMs,
          contextRegistrationMs: context.registrationMs, totalMs: original.timing.totalMs + context.milliseconds } };
      contextDetection = null;
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
      if (data.brightness !== false && brightnessCalibration) applyBrightnessCalibration(result, brightnessCalibration);
      result.processingMs = performance.now() - started;
      transfer = [result.data.buffer];
    } else if (data.type === 'brightness-set') {
      brightnessCalibration = data.calibration ?? null;
      result = true;
    } else if (data.type === 'export') {
      result = exportCalibration(calibration, data.frames, data.video, data.parameters, data.opticalConfiguration,
        data.tracking, Object.hasOwn(data, 'brightness') ? data.brightness : brightnessCalibration);
      transfer = [result.buffer];
    } else if (data.type === 'import') {
      const imported = importCalibration(data.bytes);
      const geometry = geometryCheck(imported.calibration.field, imported.parameters?.tau ?? 0.12);
      if (!geometry.valid) throw new Error(`Geladenes Feld ungueltig: ${geometry.reason}`);
      patchTracker.reset();
      calibration = imported.calibration;
      brightnessCalibration = imported.brightness;
      const response = calibrationTransferCopy(calibration);
      const brightness = brightnessCalibration ? { ...brightnessCalibration,
        gain: brightnessCalibration.gain.slice(), supported: brightnessCalibration.supported.slice() } : null;
      result = { ...imported, calibration: response.value, brightness };
      transfer = response.transfer;
      if (brightness) transfer.push(brightness.gain.buffer, brightness.supported.buffer);
    } else if (data.type === 'reset') { calibration = null; brightnessCalibration = null; trackingMaps = null; patchTracker.reset(); windowTracker.reset(); contextTracker.reset(); contextDetection = null; gpuRemapper.reset(); result = true; }
    else throw new Error(`Unbekannter Workerauftrag: ${data.type}`);
    self.postMessage({ id: data.id, result }, transfer);
  } catch (error) {
    self.postMessage({ id: data.id, error: error.message });
  }
  finally { data.native?.frame?.close(); active = false; }
};