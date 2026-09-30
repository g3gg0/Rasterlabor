import { measureEdgeConstraints } from './pcb-edge-constraints.js';
import { revalidateSavedCells, measureFinePair } from './fine-alignment.js';
import { detectGrid } from './detector.js';
import { analyzeCheckerboard } from './checkerboard-analysis.js';
import { fitCalibration, geometryCheck } from './solver.js';
import { buildMaps, remapRGBA } from './maps.js';
import { exportCalibration, importCalibration } from './format.js';
import { PatchTracker } from './patch-tracker.js';
import { WindowTracker } from './window-tracker.js';
import { ContextTracker, contextImage, contextSearchRadii, registerOverlap, registerOverlapAsync, shouldRunContext } from './context-tracker.js';
import { WebGpuContextTracker } from './webgpu-context-tracker.js';
import { patchGpuStatus } from './webgpu-patch-tracker.js';
import { remapGpuStatus, WebGpuRemapper } from './webgpu-remapper.js';
import { restoreDetectionScale, scaleDetectionOptions } from './detection-scale.js';
import { gpuGrayscaleBitmap, gpuImageStatus } from './webgpu-image.js';
import { buildInverseMapsGpu, mapBuilderGpuStatus } from './webgpu-map-builder.js';
import { createSpline } from './spline.js';
import { optimizePoseGraph } from './pose-graph-refit.js';
import { optimizePcbComponents } from './pcb-realignment.js';
import { acceptPcbGpuOnly, composePose, invertPose } from './pcb-realignment.js';
import { registerPcbFftPair, registerPcbRotationPair, registerPcbCoarsePair } from './pcb-fft-pair.js';
import { recoverPcbLandmarkPair } from './pcb-landmarks.js';
import { pcbVias } from './pcb-vias.js';
import { optimizePcbBundle } from './pcb-bundle-adjustment.js';
import { measureStructuralPair } from './pcb-structural-match.js';
import { recoverPcbViaConstellation } from './pcb-via-constellation.js';
import { maskIncludes } from './patch-mask.js';
import { registerFeatureOverlap } from './feature-overlap.js';
import { applyBrightnessCalibration } from './brightness-calibration.js';
import { setWebGpuSelection } from './webgpu-selection.js';

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
const pcbGpuContext = new WebGpuContextTracker();
let pcbNativeSetup = null;
const pcbNativeFrames = new Map();
let pcbNativeBytes = 0;
const pcbViaFrames = new Map();
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

function imageFromBitmap(bitmap, mask = null) {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0);
  const image = context.getImageData(0, 0, bitmap.width, bitmap.height);
  if (mask) for (let y = 0; y < image.height; y++) for (let x = 0; x < image.width; x++) {
    if (!maskIncludes(mask, x, y)) image.data[(y * image.width + x) * 4 + 3] = 0;
  }
  return image;
}

async function pcbGpuCoarseMatch(data, reference, current, referenceRgba, currentRgba,
  referenceImage = null, currentImage = null) {
  const started = performance.now();
  referenceImage ??= contextImage(referenceRgba, data.mask, data.preprocessing);
  currentImage ??= contextImage(currentRgba, data.mask, data.preprocessing);
  const centerOffset = (item, image) => ({ x: item.offset[0] + image.width / 2,
    y: item.offset[1] + image.height / 2, rotation: 0 });
  const referenceOffset = centerOffset(reference, referenceRgba);
  const currentOffset = centerOffset(current, currentRgba);
  const referencePose = composePose(reference.pose, referenceOffset);
  const currentPose = composePose(current.pose, currentOffset);
  const preparedMs = performance.now() - started;
  const gpuStarted = performance.now();
  const options = { radius: data.coarseRadius ?? data.limits.radius,
    angle: data.limits.angle, coarseStep: 1, partial: false,
    minimumOverlapFraction: data.minimumOverlapFraction ?? 0.05,
    stopAfterEmptyCoarseLevel: true };
  const measured = await registerOverlapAsync(pcbGpuContext, currentImage, referenceImage,
    currentPose, referencePose, options);
  const backward = measured.pose && measured.score >= 0.95 ? await registerOverlapAsync(pcbGpuContext,
    referenceImage, currentImage, referencePose, measured.pose,
    { radius: 32, angle: data.limits.angle, coarseStep: 0, partial: false,
      minimumOverlapFraction: options.minimumOverlapFraction }) : null;
  const reverseDistance = backward?.pose ? Math.hypot(backward.pose.x - referencePose.x,
    backward.pose.y - referencePose.y) + Math.abs(wrapAngle(backward.pose.rotation - referencePose.rotation)) *
    Math.hypot(currentRgba.width, currentRgba.height) / 2 : null;
  const correctedPose = measured.pose ? composePose(measured.pose, invertPose(currentOffset)) : null;
  const fftAfter = correctedPose ? registerPcbFftPair(referenceRgba, currentRgba,
    reference.pose, correctedPose, reference.offset, current.offset,
    { ...data.fft, searchRadius: Math.min(31, data.fft.searchRadius), mask: data.mask }) : null;
  const agreement = fftAfter?.pose ? Math.hypot(fftAfter.pose.x - correctedPose.x,
    fftAfter.pose.y - correctedPose.y) +
    Math.abs(wrapAngle(fftAfter.pose.rotation - correctedPose.rotation)) *
    Math.hypot(currentRgba.width, currentRgba.height) / 2 : null;
  let regions = null;
  if (!fftAfter?.accepted && data.pair?.kind === 'temporal' &&
      data.pair.current - data.pair.reference > 16 &&
      data.pair.current - data.pair.reference <= 128 &&
      measured.score >= 0.99 && backward?.score >= 0.99 && reverseDistance <= 3) {
    regions = [];
    for (const sampleRegion of [
      { xMin: -currentRgba.width / 2, xMax: 0, yMin: -currentRgba.height / 2, yMax: currentRgba.height / 2 },
      { xMin: 0, xMax: currentRgba.width / 2, yMin: -currentRgba.height / 2, yMax: currentRgba.height / 2 },
      { xMin: -currentRgba.width / 2, xMax: currentRgba.width / 2, yMin: -currentRgba.height / 2, yMax: 0 },
      { xMin: -currentRgba.width / 2, xMax: currentRgba.width / 2, yMin: 0, yMax: currentRgba.height / 2 }
    ]) {
      const region = await registerOverlapAsync(pcbGpuContext, currentImage, referenceImage,
        measured.pose, referencePose, { radius: 16, angle: 1, coarseStep: 0, partial: false,
          minimumOverlapFraction: options.minimumOverlapFraction, sampleRegion });
      const distance = region.pose ? Math.hypot(region.pose.x - measured.pose.x,
        region.pose.y - measured.pose.y) +
        Math.abs(wrapAngle(region.pose.rotation - measured.pose.rotation)) *
        Math.hypot(currentRgba.width, currentRgba.height) / 2 : null;
      regions.push({ score: region.score ?? null, support: region.support ?? 0,
        distance, reason: region.reason ?? null });
    }
  }
  return { measured, backward, reverseDistance, correctedPose, fftAfter, agreement, regions,
    preparedMs, gpuMs: performance.now() - gpuStarted };
}

async function pcbGpuVerifyFft(data, reference, current, referenceRgba, currentRgba, fft,
  referenceImage = null, currentImage = null) {
  referenceImage ??= contextImage(referenceRgba, data.mask, data.preprocessing);
  currentImage ??= contextImage(currentRgba, data.mask, data.preprocessing);
  const centerOffset = (item, image) => ({ x: item.offset[0] + image.width / 2,
    y: item.offset[1] + image.height / 2, rotation: 0 });
  const referenceOffset = centerOffset(reference, referenceRgba);
  const currentOffset = centerOffset(current, currentRgba);
  const referencePose = composePose(reference.pose, referenceOffset);
  const fftCenterPose = composePose(fft.pose, currentOffset);
  const options = { radius: 16, angle: Math.min(1, data.limits.angle), coarseStep: 0,
    partial: false, minimumOverlapFraction: data.minimumOverlapFraction ?? 0.05 };
  const forward = await registerOverlapAsync(pcbGpuContext, currentImage, referenceImage,
    fftCenterPose, referencePose, options);
  const backward = forward.pose && forward.score >= Math.max(0.9, data.limits.minimumScore ?? 0.9) ?
    await registerOverlapAsync(pcbGpuContext, referenceImage, currentImage,
      referencePose, forward.pose, options) : null;
  const lever = Math.hypot(currentRgba.width, currentRgba.height) / 2;
  const distance = (first, second) => first && second ? Math.hypot(first.x - second.x, first.y - second.y) +
    Math.abs(wrapAngle(first.rotation - second.rotation)) * lever : null;
  const agreement = distance(forward.pose, fftCenterPose);
  const reverseDistance = distance(backward?.pose, referencePose);
  const minimumScore = Math.max(0.9, data.limits.minimumScore ?? 0.9);
  let fftReverse = null;
  if (fft.inlierCells.length >= 4 && fft.uniqueSupportArea >= 8192 &&
      forward.score >= minimumScore && backward?.score >= minimumScore &&
      (!Number.isFinite(agreement) || agreement > 8 ||
        !Number.isFinite(reverseDistance) || reverseDistance > (data.limits.cycleLimit ?? 5))) {
    const reverse = registerPcbFftPair(currentRgba, referenceRgba,
      fft.pose, reference.pose, current.offset, reference.offset,
      { ...data.fft, searchRadius: Math.min(31, data.fft.searchRadius), mask: data.mask });
    fftReverse = { accepted: reverse.accepted, inlierCells: reverse.inlierCells?.length ?? 0,
      uniqueSupportArea: reverse.uniqueSupportArea ?? 0,
      residualRms: reverse.residualRms ?? null,
      distance: distance(reverse.pose, reference.pose), reason: reverse.reason ?? null };
  }
  const fftCycleAccepted = fftReverse?.accepted && fftReverse.inlierCells >= 4 &&
    fftReverse.uniqueSupportArea >= 1024 &&
    Number.isFinite(fftReverse.distance) &&
    fftReverse.distance <= Math.max(data.limits.cycleLimit ?? 5,
      Math.min(12, (data.limits.fftCycleFactor ?? 2) *
        Math.max(fft.residualRms ?? 0, fftReverse.residualRms ?? 0)));
  const fftCycleLimit = fftReverse ? Math.max(data.limits.cycleLimit ?? 5,
    Math.min(12, (data.limits.fftCycleFactor ?? 2) *
      Math.max(fft.residualRms ?? 0, fftReverse.residualRms ?? 0))) : data.limits.cycleLimit ?? 5;
  const accepted = fft.inlierCells.length >= 4 && fft.uniqueSupportArea >= 8192 &&
    forward.score >= minimumScore && backward?.score >= minimumScore &&
    forward.support >= 128 && backward?.support >= 128 &&
    Number.isFinite(agreement) && agreement <= 8 &&
    ((Number.isFinite(reverseDistance) && reverseDistance <= (data.limits.cycleLimit ?? 5)) ||
      fftCycleAccepted);
  return { accepted, score: Math.min(forward.score ?? 0, backward?.score ?? 0),
    agreement, reverseDistance: fftCycleAccepted ? fftReverse.distance : reverseDistance,
    gpuReverseDistance: reverseDistance, fftReverse,
    cycleLimit: fftCycleAccepted ? fftCycleLimit : data.limits.cycleLimit ?? 5,
    forwardScore: forward.score ?? null,
    backwardScore: backward?.score ?? null, forwardSupport: forward.support ?? 0,
    backwardSupport: backward?.support ?? 0,
    reason: accepted ? null : 'GPU-Pruefung an FFT-Pose widerspricht' };
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
    if (setWebGpuSelection(data.gpuSelection ?? 'default')) {
      patchTracker.reset(); windowTracker.reset(); contextTracker.reset();
      gpuRemapper.reset(); pcbGpuContext.reset(); contextDetection = null;
    }
    let result;
    let transfer = [];
    if (data.type === 'patch-gpu-status') result = await patchGpuStatus();
    else if (data.type === 'pose-graph-refit') {
      result = optimizePoseGraph(data.graph, data.options);
    }
    else if (data.type === 'pcb-graph-optimize') {
      result = data.network ? optimizePcbBundle(data.graph.nodes, data.network, data.options) : optimizePcbComponents(data.graph, data.options);
    }
    else if (data.type === 'pcb-pair-refine') {
      try {
        const [reference,current]=data.images;
        const first=imageFromBitmap(reference.bitmap,reference.mask),second=imageFromBitmap(current.bitmap,current.mask);
        result=measureFinePair(first,second,reference,current,{radius:8,mask:data.mask});
        if(!result.accepted) result=measureStructuralPair(first,second,reference,current,{radius:64,mask:data.mask});
        if(!result.accepted) result=measureEdgeConstraints(first,second,reference,current,{radius:64,mask:data.mask});
        result.kind=data.pair.kind;
        if(data.pair.cells?.length) result.anchorEvidence=revalidateSavedCells(first,second,reference,current,data.pair.cells);
      } finally { for(const item of data.images) item.bitmap.close(); }
    }
    else if (data.type === 'pcb-native-setup') {
      pcbNativeSetup = { maps: data.maps, sourceMask: data.sourceMask, brightness: data.brightness,
        cacheBudget: Math.max(0, data.cacheBudget ?? 512 * 1024 ** 2) };
      pcbNativeFrames.clear(); pcbNativeBytes = 0; pcbGpuContext.reset(); result = true;
    }
    else if (data.type === 'pcb-pair-coarse-probe') {
      try {
        const [reference, current] = data.images;
        const referenceRgba = imageFromBitmap(reference.bitmap), currentRgba = imageFromBitmap(current.bitmap);
        const { measured, backward, reverseDistance, fftAfter, agreement,
          preparedMs, gpuMs } = await pcbGpuCoarseMatch(data, reference, current, referenceRgba, currentRgba);
        result = { ...measured, preparedMs, gpuMs,
          backward: backward ? { accepted: backward.accepted, score: backward.score,
            reason: backward.reason, margin: backward.margin, support: backward.support } : null,
          reverseDistance,
          fftAfter: fftAfter ? { accepted: fftAfter.accepted, reason: fftAfter.reason,
            inlierCells: fftAfter.inlierCells?.length ?? 0,
            uniqueSupportArea: fftAfter.uniqueSupportArea ?? 0,
            residualRms: fftAfter.residualRms, agreement,
            cellReasons: fftAfter.cells.reduce((counts, cell) => {
              counts[cell.reason ?? 'accepted'] = (counts[cell.reason ?? 'accepted'] ?? 0) + 1;
              return counts;
            }, {}) } : null,
          accelerator: pcbGpuContext.failure ? 'WebGPU-Fehler' : 'WebGPU', failure: pcbGpuContext.failure };
      } finally { for (const item of data.images) item.bitmap.close(); }
    }
    else if (data.type === 'pcb-pair-register') {
      try {
        const [reference, current] = data.images;
        const profile = { readbackMs: 0, fftMs: 0, pyramidMs: 0, nccMs: 0, fallbackMs: 0 };
        let stageStarted = performance.now();
        let referenceImage = null, currentImage = null;
        const prepareNative = async item => {
          let cached = pcbNativeFrames.get(item.frame);
          if (cached) {
            pcbNativeFrames.delete(item.frame); pcbNativeFrames.set(item.frame, cached);
            item.native?.frame?.close(); if (item.native) item.native.frame = null;
            return cached.prepared;
          }
          if (!item.native?.frame) throw new Error(`GPU-Frame #${item.frame} fehlt im Cache.`);
          if (!pcbNativeSetup?.maps) throw new Error('Native PCB-GPU-Konfiguration fehlt.');
          const prepared = await pcbGpuContext.image({ frame: item.native.frame,
            orientation: item.native.orientation, maps: pcbNativeSetup.maps,
            width: pcbNativeSetup.maps.outputWidth, height: pcbNativeSetup.maps.outputHeight,
            sourceMask: pcbNativeSetup.sourceMask, brightness: pcbNativeSetup.brightness,
            readRgba: true }, data.mask);
          const bytes = (prepared.bytes ?? 0) + (prepared.rgba?.byteLength ?? 0);
          pcbNativeFrames.set(item.frame, { prepared, bytes }); pcbNativeBytes += bytes;
          while (pcbNativeBytes > pcbNativeSetup.cacheBudget && pcbNativeFrames.size) {
            const oldest = pcbNativeFrames.keys().next().value;
            const removed = pcbNativeFrames.get(oldest);
            pcbNativeFrames.delete(oldest); pcbNativeBytes -= removed.bytes;
          }
          item.native.frame.close(); item.native.frame = null;
          return prepared;
        };
        if (data.nativePath) {
          referenceImage = await prepareNative(reference);
          currentImage = await prepareNative(current);
        }
        const referenceRgba = referenceImage?.rgba ?? imageFromBitmap(reference.bitmap, reference.mask);
        const currentRgba = currentImage?.rgba ?? imageFromBitmap(current.bitmap, current.mask);
        if(data.pair.anchorsOnly){
          self.postMessage({id:data.id,result:{...data.pair,accepted:false,
            anchorEvidence:revalidateSavedCells(referenceRgba,currentRgba,reference,current,data.pair.cells??[]),
            nativeCachedFrames:data.nativePath?[...pcbNativeFrames.keys()]:undefined}});
          return;
        }
        if(data.localLandmarks||data.featureRecovery){
          let quick=measureFinePair(referenceRgba,currentRgba,reference,current,{radius:8,mask:data.mask});
          if(!quick.accepted)quick=measureEdgeConstraints(referenceRgba,currentRgba,reference,current,{radius:Math.min(64,data.limits.radius??64),mask:data.mask});
          if(quick.accepted){
            const anchorEvidence=data.pair.cells?.length?revalidateSavedCells(referenceRgba,currentRgba,reference,current,data.pair.cells):null;
            self.postMessage({id:data.id,result:{...data.pair,...quick,kind:data.pair.kind,anchorEvidence,
              accelerator:quick.partial?'Gerichtete Kantenprofile':'Mehrskalige Feinsuche',
              nativeCachedFrames:data.nativePath?[...pcbNativeFrames.keys()]:undefined}});
            return;
          }
        }
        const viaGroups=typeof data.pair.group==='string'?data.pair.group.split(':'):null;
        if (viaGroups?.length===2) {
          for (const [index, item] of [reference, current].entries()) {
            if (pcbViaFrames.has(item.frame)) continue;
            const rgba = index ? currentRgba : referenceRgba;
            const prepared = index ? (currentImage ??= contextImage(rgba,data.mask,data.preprocessing)) :
              (referenceImage ??= contextImage(rgba,data.mask,data.preprocessing));
            pcbViaFrames.set(item.frame,{frame:item.frame,group:viaGroups[index],points:pcbVias(prepared),
              center:composePose(item.pose,{x:item.offset[0]+rgba.width/2,y:item.offset[1]+rgba.height/2,rotation:0})});
            if(pcbViaFrames.size>512)pcbViaFrames.delete(pcbViaFrames.keys().next().value);
          }
        }
        profile.readbackMs = performance.now() - stageStarted;
        stageStarted = performance.now();
        let fft = registerPcbFftPair(referenceRgba, currentRgba, reference.pose, current.pose,
          reference.offset, current.offset, { ...data.fft, mask: data.mask });
        if (data.fft.adaptiveCells && !fft.accepted &&
            ['Zu wenig unabhaengige Strukturzellen', 'Zu wenig gemeinsamer Bildinhalt'].includes(fft.reason)) {
          for (const cellSize of [data.fft.cellSize / 2, data.fft.cellSize / 4]) {
            if (cellSize < 32) break;
            const retry = registerPcbFftPair(referenceRgba, currentRgba, reference.pose, current.pose,
              reference.offset, current.offset, { ...data.fft, cellSize,
                searchRadius: Math.min(data.fft.searchRadius, cellSize / 2 - 1), mask: data.mask });
            if (retry.accepted) { fft = retry; break; }
          }
        }
        if (!fft.accepted) {
          const coarseFft = registerPcbCoarsePair(referenceRgba, currentRgba, reference.pose, current.pose,
            reference.offset, current.offset, { ...data.fft, mask: data.mask,
              angle: data.limits.angle, radius: data.coarseRadius ?? data.limits.radius });
          if (coarseFft) fft = coarseFft;
        }
        if (!fft.accepted || fft.inlierCells.length < 5) {
          const rotated = registerPcbRotationPair(referenceRgba, currentRgba, reference.pose, current.pose,
            reference.offset, current.offset, { ...data.fft, mask: data.mask,
              angle: data.limits.angle, radius: data.limits.radius });
          if (rotated) fft = rotated;
        }
        let featureRecovery = null, landmarkRecovery = null;
        if (data.featureRecovery && !fft.accepted) {
          referenceImage ??= contextImage(referenceRgba, data.mask, data.preprocessing);
          currentImage ??= contextImage(currentRgba, data.mask, data.preprocessing);
          const referenceOffset = { x: reference.offset[0] + referenceRgba.width / 2,
            y: reference.offset[1] + referenceRgba.height / 2, rotation: 0 };
          const currentOffset = { x: current.offset[0] + currentRgba.width / 2,
            y: current.offset[1] + currentRgba.height / 2, rotation: 0 };
          const referenceCenter = composePose(reference.pose, referenceOffset);
          const featureAttempts = [];
          featureRecovery = {accepted:false, attempts:featureAttempts};
          for (const minimumFeatureEdge of [240, 480, 960]) {
            const forwardFeature = registerFeatureOverlap(currentImage, referenceImage, referenceCenter,
              {minimumFeatureEdge});
            const backwardFeature = forwardFeature.accepted ?
              registerFeatureOverlap(referenceImage, currentImage, forwardFeature.pose,
                {minimumFeatureEdge}) : null;
            const attempt = {minimumFeatureEdge, scale:forwardFeature.featureScale,
              currentFeatures:forwardFeature.currentFeatures, referenceFeatures:forwardFeature.referenceFeatures,
              matches:forwardFeature.matches, inliers:forwardFeature.inliers, score:forwardFeature.score,
              backwardInliers:backwardFeature?.inliers, backwardScore:backwardFeature?.score};
            featureAttempts.push(attempt);
            if (!forwardFeature.accepted || !backwardFeature?.accepted) continue;
            const reverseDistance = Math.hypot(backwardFeature.pose.x - referenceCenter.x,
              backwardFeature.pose.y - referenceCenter.y) +
              Math.abs(wrapAngle(backwardFeature.pose.rotation - referenceCenter.rotation)) *
                Math.hypot(currentRgba.width, currentRgba.height) / 2;
            const featurePose = composePose(forwardFeature.pose, invertPose(currentOffset));
            const correction = Math.hypot(featurePose.x - current.pose.x, featurePose.y - current.pose.y);
            const angle = Math.abs(wrapAngle(featurePose.rotation - current.pose.rotation)) * 180 / Math.PI;
            Object.assign(attempt,{reverseDistance,correction,angle,pose:featurePose,
              residual:forwardFeature.residual,backwardResidual:backwardFeature.residual});
            if (Math.min(forwardFeature.inliers,backwardFeature.inliers) < 30 ||
                Math.min(forwardFeature.score,backwardFeature.score) < 0.9 ||
                Math.max(forwardFeature.residual,backwardFeature.residual) > 8 ||
                reverseDistance > (data.limits.cycleLimit ?? 5) ||
                correction > data.coarseRadius || angle > Math.max(5,data.limits.angle)) continue;
            attempt.confirmations = [];
            for (const cellSize of [data.fft.cellSize, data.fft.cellSize / 2, data.fft.cellSize / 4]) {
              const confirmed = registerPcbFftPair(referenceRgba,currentRgba,reference.pose,featurePose,
                reference.offset,current.offset,{...data.fft,cellSize,candidateGrid:24,
                  searchRadius:Math.min(31,cellSize/2-1),mask:data.mask});
              const agreement = confirmed.pose ? Math.hypot(confirmed.pose.x-featurePose.x,
                confirmed.pose.y-featurePose.y) + Math.abs(wrapAngle(confirmed.pose.rotation-featurePose.rotation)) *
                  Math.hypot(currentRgba.width,currentRgba.height)/2 : Infinity;
              attempt.confirmations.push({cellSize,accepted:confirmed.accepted,reason:confirmed.reason,
                cells:confirmed.inlierCells?.length??0,support:confirmed.uniqueSupportArea??0,agreement,
                reasons:Object.fromEntries([...new Set(confirmed.cells.map(cell=>cell.reason))].map(reason=>
                  [reason??'accepted',confirmed.cells.filter(cell=>cell.reason===reason).length]))});
              if (confirmed.accepted && confirmed.inlierCells.length >= 5 &&
                  confirmed.uniqueSupportArea >= 1024 && agreement <= 8) {
                fft=confirmed;
                Object.assign(featureRecovery,{accepted:true,reverseDistance,correction,angle,agreement});
                break;
              }
            }
            if (featureRecovery.accepted) break;
          }
        }

        profile.fftMs = performance.now() - stageStarted;
        if (fft.accepted && (Math.hypot(fft.pose.x - current.pose.x, fft.pose.y - current.pose.y) >
              (featureRecovery?.accepted || fft.coarseSearch ? data.coarseRadius ?? data.limits.radius : data.limits.radius) ||
            Math.abs(wrapAngle(fft.pose.rotation - current.pose.rotation)) * 180 / Math.PI >
              (featureRecovery?.accepted ? Math.max(5, data.limits.angle) : data.limits.angle))) {
          fft.accepted = false; fft.reason = 'Korrektur ausserhalb Suchbereich';
        }
        let forward = { accepted: false, reason: fft.reason, pose: fft.pose ?? null, score: null,
          support: fft.uniqueSupportArea ?? 0 };
        let backward = null, reverseDistance = null, agreement = null;
        if (fft.accepted) {
          stageStarted = performance.now();
          referenceImage ??= contextImage(referenceRgba, data.mask, data.preprocessing);
          currentImage ??= contextImage(currentRgba, data.mask, data.preprocessing);
          profile.pyramidMs = performance.now() - stageStarted;
          stageStarted = performance.now();
          const centerOffset = (item, image) => ({ x: item.offset[0] + image.width / 2,
            y: item.offset[1] + image.height / 2, rotation: 0 });
          const referenceOffset = centerOffset(reference, referenceRgba);
          const currentOffset = centerOffset(current, currentRgba);
          const centerPose = (pose, offset) => composePose(pose, offset);
          const cameraPose = (pose, offset) => composePose(pose, invertPose(offset));
          let ncc = registerOverlap(currentImage, referenceImage,
            centerPose(fft.pose, currentOffset), centerPose(reference.pose, referenceOffset),
            { radius: Math.min(16, data.limits.radius), angle: data.limits.angle, coarseStep: 0, partial: true });
          // A good local FFT seed can still need more than 16 pixels of NCC refinement.
          // Retry the bounded search; acceptance and the symmetric check stay unchanged.
          if (data.fft.adaptiveCells && ncc.reason === 'Suchgrenze' && ncc.score >= 0.95) {
            const retry = registerOverlap(currentImage, referenceImage,
              centerPose(current.pose, currentOffset), centerPose(reference.pose, referenceOffset),
              { radius: Math.min(64, data.limits.radius), angle: data.limits.angle, coarseStep: 0, partial: true });
            if (retry.accepted || retry.score > ncc.score) ncc = retry;
          }
          const nccPose = ncc.pose ? cameraPose(ncc.pose, currentOffset) : null;
          agreement = nccPose ? Math.hypot(nccPose.x - fft.pose.x, nccPose.y - fft.pose.y) +
            Math.abs(wrapAngle(nccPose.rotation - fft.pose.rotation)) * Math.hypot(currentRgba.width, currentRgba.height) / 2 : null;
          const fftConfirmed = fft.inlierCells.length >= 5 && fft.uniqueSupportArea >= 1024;
          const conditionalForward = !ncc.accepted && ncc.reason === 'Mehrdeutig' &&
            ncc.score >= 0.95 && fftConfirmed && agreement <= 8;
          const backwardCenter = ncc.accepted || conditionalForward ? registerOverlap(referenceImage, currentImage,
            centerPose(reference.pose, referenceOffset), centerPose(nccPose, currentOffset),
            { radius: Math.min(32, data.limits.radius), angle: data.limits.angle, coarseStep: 0, partial: true }) : null;
          backward = backwardCenter ? { ...backwardCenter,
            pose: backwardCenter.pose ? cameraPose(backwardCenter.pose, referenceOffset) : null } : null;
          if (backward && !backward.accepted && backward.reason === 'Mehrdeutig' &&
              backward.score >= 0.95 && fftConfirmed) {
            backward.accepted = true; backward.conditional = true;
          }
          const cycle = backward?.pose ? Math.hypot(backward.pose.x - reference.pose.x, backward.pose.y - reference.pose.y) +
            Math.abs(wrapAngle(backward.pose.rotation - reference.pose.rotation)) * Math.hypot(currentRgba.width, currentRgba.height) / 2 : null;
          reverseDistance = Number.isFinite(cycle) ? cycle : null;
          if (data.fft.adaptiveCells && !backward?.accepted && (ncc.accepted || conditionalForward) &&
              ncc.score >= 0.995 && fftConfirmed && Number.isFinite(agreement) && agreement <= 8) {
            const reverseFft = registerPcbFftPair(currentRgba, referenceRgba,
              fft.pose, reference.pose, current.offset, reference.offset,
              { ...data.fft, searchRadius: Math.min(31, data.fft.searchRadius), mask: data.mask });
            const reverseCycle = reverseFft.pose ? Math.hypot(reverseFft.pose.x - reference.pose.x,
              reverseFft.pose.y - reference.pose.y) +
              Math.abs(wrapAngle(reverseFft.pose.rotation - reference.pose.rotation)) *
                Math.hypot(currentRgba.width, currentRgba.height) / 2 : null;
            if (reverseFft.accepted && reverseFft.inlierCells.length >= 5 &&
                reverseFft.uniqueSupportArea >= 1024 && Number.isFinite(reverseCycle) &&
                reverseCycle <= (data.limits.cycleLimit ?? 5)) {
              backward = { accepted: true, conditional: true, reason: null,
                pose: reverseFft.pose, score: ncc.score, support: reverseFft.uniqueSupportArea };
              reverseDistance = reverseCycle;
            }
          }
          forward = { accepted: Boolean((ncc.accepted || conditionalForward) && Number.isFinite(reverseDistance)),
            conditional: conditionalForward,
            reason: ncc.accepted || conditionalForward ? null : ncc.reason || 'NCC-Bestaetigung', pose: nccPose,
            score: ncc.score, support: fft.uniqueSupportArea };
          if ((featureRecovery?.accepted) && fftConfirmed &&
              (!forward.accepted || !backward?.accepted || reverseDistance > (data.limits.cycleLimit ?? 5))) {
            const reverseFft = registerPcbFftPair(currentRgba,referenceRgba,fft.pose,reference.pose,
              current.offset,reference.offset,{...data.fft,cellSize:fft.cellSize,candidateGrid:24,
                translationOnly:fft.translationOnly,searchRadius:31});
            const cycle = reverseFft.pose ? Math.hypot(reverseFft.pose.x-reference.pose.x,
              reverseFft.pose.y-reference.pose.y) + Math.abs(wrapAngle(reverseFft.pose.rotation-reference.pose.rotation)) *
                Math.hypot(currentRgba.width,currentRgba.height)/2 : Infinity;
            const atPose = registerOverlap(currentImage,referenceImage,
              centerPose(fft.pose,currentOffset),centerPose(reference.pose,referenceOffset),
              {radius:1,angle:0,coarseStep:0,partial:true});
            const verification = {accepted:false,cells:reverseFft.inlierCells?.length??0,
              support:reverseFft.uniqueSupportArea??0,reason:reverseFft.reason,cycle,score:atPose.score};
            featureRecovery.reverseVerification = verification;
            if (reverseFft.accepted && verification.cells >= 5 && verification.support >= 1024 &&
                cycle <= Math.min(5,data.limits.cycleLimit ?? 5) && atPose.score >= 0.95) {
              verification.accepted = true;
              forward = {accepted:true,conditional:true,pose:fft.pose,score:atPose.score,
                support:fft.uniqueSupportArea,reason:null};
              backward = {accepted:true,conditional:true,pose:reverseFft.pose,score:atPose.score,
                support:reverseFft.uniqueSupportArea,reason:null};
              reverseDistance = cycle;
              agreement = 0;
            }
          }
          profile.nccMs = performance.now() - stageStarted;
        }
        if((data.featureRecovery||data.localLandmarks)&&(!forward.accepted||!backward?.accepted)) {
          const edge=measureEdgeConstraints(referenceRgba,currentRgba,reference,current,{radius:Math.min(64,data.limits.radius??64),mask:data.mask});
          if(edge.accepted){fft=edge.fft;forward=edge.forward;backward=edge.backward;reverseDistance=edge.reverseDistance;agreement=0;}
        }
        // Generic gradients (board outlines, tracks, pads, markings) complement texture.
        if ((data.featureRecovery || data.localLandmarks) && (!forward.accepted || !backward?.accepted)) {
          const structural = measureStructuralPair(referenceRgba, currentRgba, reference, current,
            {mask:data.mask, radius:Math.min(100,data.limits.radius ?? 64)});
          if (structural.accepted) {
            fft=structural.fft;forward=structural.forward;backward=structural.backward;
            reverseDistance=structural.reverseDistance;agreement=0;
          }
        }
        if ((data.featureRecovery || data.localLandmarks) && (!forward.accepted || !backward?.accepted ||
            !Number.isFinite(reverseDistance) || reverseDistance > (data.limits.cycleLimit ?? 5))) {
          referenceImage ??= contextImage(referenceRgba,data.mask,data.preprocessing);
          currentImage ??= contextImage(currentRgba,data.mask,data.preprocessing);
          // Neutral images keep vias/trace gradients usable even when display contrast clips.
          const constellation = recoverPcbViaConstellation(contextImage(referenceRgba, data.mask),
            contextImage(currentRgba, data.mask), reference, current, data);
          landmarkRecovery = constellation;
          if (!constellation.accepted && data.featureRecovery && !constellation.reason?.startsWith('Mehrere passende')) {
            landmarkRecovery = recoverPcbLandmarkPair(referenceRgba,currentRgba,reference,current,
              referenceImage,currentImage,{...data,referenceVias:viaGroups&&viaGroups[0]!==viaGroups[1]?
                [...pcbViaFrames.values()].filter(item=>item.group===viaGroups[0]):[]});
            landmarkRecovery.constellation = constellation;
          }
          const recoveredFft=landmarkRecovery.fft;
          delete landmarkRecovery.fft;
          if (landmarkRecovery.accepted) {
            fft=recoveredFft;
            forward={accepted:true,conditional:true,pose:fft.pose,score:landmarkRecovery.score,
              support:fft.uniqueSupportArea,reason:null};
            backward={accepted:true,conditional:true,pose:landmarkRecovery.backwardPose,score:landmarkRecovery.score,
              support:landmarkRecovery.reverseVerification.support,reason:null};
            reverseDistance=landmarkRecovery.reverseVerification.cycle;
            agreement=0;
          } else if (data.localLandmarks) {
            forward.reason = `${forward.reason ?? "Paarpruefung"} | Vias: ${landmarkRecovery.reason ?? landmarkRecovery.constellation?.reason ?? "Nicht bestaetigt"}`;
          }
        }
        stageStarted = performance.now();
        let fftGpuVerification = null;
        if (data.useWebGpu !== false && fft.accepted &&
            (!forward.accepted || !backward?.accepted || reverseDistance > (data.limits.cycleLimit ?? 5)) &&
            fft.inlierCells.length >= 4 && fft.uniqueSupportArea >= 8192) {
          try {
            fftGpuVerification = await pcbGpuVerifyFft(data, reference, current,
              referenceRgba, currentRgba, fft, referenceImage, currentImage);
            if (fftGpuVerification.accepted) {
              forward = { accepted: true, conditional: true, reason: null,
                pose: fft.pose, score: fftGpuVerification.score, support: fft.uniqueSupportArea };
              backward = { accepted: true, conditional: true, reason: null,
                score: fftGpuVerification.backwardScore,
                support: fftGpuVerification.backwardSupport };
              reverseDistance = fftGpuVerification.reverseDistance;
              agreement = fftGpuVerification.agreement;
            }
          } catch (error) {
            fftGpuVerification = { accepted: false, reason: `GPU-FFT-Pruefung: ${error.message}` };
          }
        }
        let coarse = null, accelerator = fft.coarseSearch ? 'CPU FFT-Grobsuche + Zellmessung' : 'CPU-FFT/NCC';
        if (landmarkRecovery?.accepted) accelerator = fft.method?.startsWith('PCB-Via') ?
          'CPU-Via-Geometrie' : 'CPU FFT mit Bohrungsreferenz';
        if (fftGpuVerification?.accepted) accelerator = 'CPU-FFT + WebGPU-Bestaetigung';
        if (data.useWebGpu !== false && !fftGpuVerification?.accepted &&
            (!forward.accepted || !backward?.accepted ||
            reverseDistance === null || reverseDistance > (data.limits.cycleLimit ?? 5))) {
          try {
            const initialReason = fft.reason ?? forward.reason ?? backward?.reason ?? 'Rueckweg';
            const candidate = await pcbGpuCoarseMatch(data, reference, current, referenceRgba, currentRgba,
              referenceImage, currentImage);
            const { measured, backward: gpuBackward, fftAfter } = candidate;
            const accepted = measured.score >= 0.95 && gpuBackward?.score >= 0.95 &&
              Number.isFinite(candidate.reverseDistance) &&
              candidate.reverseDistance <= (data.limits.cycleLimit ?? 5) &&
              fftAfter?.accepted && fftAfter.inlierCells.length >= 3 &&
              fftAfter.uniqueSupportArea >= 128 && candidate.agreement <= 8;
            const gpuOnly = !accepted && !fftAfter?.accepted ? acceptPcbGpuOnly({
              ...data.pair, currentPose: current.pose,
              lever: Math.hypot(currentRgba.width, currentRgba.height) / 2 }, candidate) :
              { accepted: false, reason: candidate.agreement > 8 ? 'GPU-/FFT-Posen widersprechen' :
                candidate.reverseDistance > (data.limits.cycleLimit ?? 5) ? 'GPU-Rueckweg zu gross' :
                  measured.score < 0.95 || gpuBackward?.score < 0.95 ? 'GPU-Korrelation zu schwach' :
                    fftAfter?.reason ?? 'GPU-/FFT-Konsistenz' };
            coarse = { accepted: accepted || gpuOnly.accepted,
              method: accepted ? 'gpu-fft' : gpuOnly.accepted ? gpuOnly.method : null,
              initialReason, score: measured.score ?? null,
              backwardScore: gpuBackward?.score ?? null, reverseDistance: candidate.reverseDistance,
              fftAgreement: candidate.agreement, fftCells: fftAfter?.inlierCells?.length ?? 0,
              regions: candidate.regions, gpuMs: candidate.gpuMs, reason: accepted || gpuOnly.accepted ? null :
                gpuOnly.reason || measured.reason || gpuBackward?.reason || fftAfter?.reason || 'GPU-/FFT-Konsistenz' };
            if (accepted || gpuOnly.accepted) {
              if (accepted) fft = fftAfter;
              forward = { accepted: true, conditional: true, reason: null,
                pose: accepted ? fftAfter.pose : candidate.correctedPose,
                score: Math.min(measured.score, gpuBackward.score),
                support: accepted ? fftAfter.uniqueSupportArea : data.limits.minimumSupport ?? 128 };
              backward = { accepted: true, conditional: !gpuBackward.accepted,
                reason: gpuBackward.reason, score: gpuBackward.score };
              reverseDistance = candidate.reverseDistance;
              agreement = candidate.agreement;
              accelerator = accepted ? 'WebGPU-Grobsuche + FFT' : 'WebGPU-NCC (FFT strukturarm)';
            }
          } catch (error) {
            coarse = { accepted: false, reason: `WebGPU-Grobsuche: ${error.message}` };
          }
        }
        profile.fallbackMs = performance.now() - stageStarted;
        if (data.nativePath) accelerator = `WebGPU Direktpfad + ${accelerator}`;
        const anchorEvidence=data.pair.cells?.length?revalidateSavedCells(referenceRgba,currentRgba,reference,current,data.pair.cells):null;
        result = { ...data.pair, currentPose: current.pose, referencePose: reference.pose,
          anchorEvidence,
          forward, backward, reverseDistance, agreement, fft, featureRecovery, landmarkRecovery, coarse, fftGpuVerification, accelerator, profile,
          nativeCachedFrames: data.nativePath ? [...pcbNativeFrames.keys()] : undefined };
      } finally { for (const item of data.images) { item.bitmap?.close(); item.native?.frame?.close(); } }
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
      else { result = remapRGBA(data.image, calibration.maps, data.sourceMask); result.accelerator = 'CPU'; }
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
    } else if (data.type === 'reset-calibration') { calibration = null; patchTracker.reset(); gpuRemapper.reset(); result = true; }
    else if (data.type === 'reset') { calibration = null; brightnessCalibration = null; trackingMaps = null; patchTracker.reset(); windowTracker.reset(); contextTracker.reset(); contextDetection = null; gpuRemapper.reset(); result = true; }
    else throw new Error(`Unbekannter Workerauftrag: ${data.type}`);
    self.postMessage({ id: data.id, result }, transfer);
  } catch (error) {
    self.postMessage({ id: data.id, error: error.message });
  }
  finally { data.native?.frame?.close(); active = false; }
};
