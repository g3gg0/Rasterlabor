import { WebGpuContextTracker } from '../src/webgpu-context-tracker.js';
import { contextImage, registerOverlapAsync } from '../src/context-tracker.js';
import { composePose, invertPose } from '../src/pcb-realignment.js';

const gpu = new WebGpuContextTracker();

export async function probePcbGpuPair({ reference, current, mode, readFrame, maps, sourceMask,
  imageMask, brightness = null, radius = 384, angle = 2, minimumOverlapFraction = 0.05 }) {
  const profile = { readMs: 0, pyramidMs: 0, gpuPyramid: null, transferBytes: 0 };
  const prepare = async item => {
    const started = performance.now();
    const native = mode === 'native';
    const decoded = await readFrame(item.frame, native ?
      { output: 'native', gpu: true, measureSharpness: false } :
      { rectified: true, gpu: true, sourceMask, measureSharpness: false });
    profile.readMs += performance.now() - started;
    try {
      const pyramidStarted = performance.now();
      let image;
      if (native) {
        image = await gpu.image({ frame: decoded.frame, orientation: decoded.orientation,
          maps, width: maps.outputWidth, height: maps.outputHeight,
          sourceMask, brightness }, imageMask);
        profile.gpuPyramid = image.pyramidTiming;
        profile.transferBytes += image.pyramidTiming?.readbackBytes ?? 0;
      } else {
        const canvas = new OffscreenCanvas(decoded.bitmap.width, decoded.bitmap.height);
        const context = canvas.getContext('2d', { willReadFrequently: true });
        context.drawImage(decoded.bitmap, 0, 0);
        image = contextImage(context.getImageData(0, 0, canvas.width, canvas.height), imageMask);
        profile.transferBytes += canvas.width * canvas.height * 4;
      }
      profile.pyramidMs += performance.now() - pyramidStarted;
      return image;
    } finally { decoded.frame?.close(); decoded.bitmap?.close(); }
  };
  const referenceImage = await prepare(reference), currentImage = await prepare(current);
  const centered = item => composePose(item.pose,
    { x: item.offset[0] + maps.outputWidth / 2, y: item.offset[1] + maps.outputHeight / 2, rotation: 0 });
  const referencePose = centered(reference), currentPose = centered(current);
  const options = { radius, angle, coarseStep: 1, partial: false,
    minimumOverlapFraction, stopAfterEmptyCoarseLevel: true };
  const started = performance.now();
  const forward = await registerOverlapAsync(gpu, currentImage, referenceImage,
    currentPose, referencePose, options);
  const forwardMs = performance.now() - started;
  const backwardStarted = performance.now();
  const backward = forward.pose && forward.score >= 0.95 ?
    await registerOverlapAsync(gpu, referenceImage, currentImage,
      referencePose, forward.pose, { radius: 32, angle, coarseStep: 0, partial: false,
        minimumOverlapFraction }) : null;
  const backwardMs = performance.now() - backwardStarted;
  const corrected = forward.pose ? composePose(forward.pose, invertPose({
    x: current.offset[0] + maps.outputWidth / 2,
    y: current.offset[1] + maps.outputHeight / 2, rotation: 0 })) : null;
  const reverseDistance = backward?.pose ? Math.hypot(backward.pose.x - referencePose.x,
    backward.pose.y - referencePose.y) : null;
  return { mode, profile: { ...profile, forwardMs, backwardMs,
      totalMs: profile.readMs + profile.pyramidMs + forwardMs + backwardMs },
    forward: { accepted: forward.accepted, score: forward.score, support: forward.support,
      reason: forward.reason, evaluated: forward.evaluated },
    backward: backward ? { accepted: backward.accepted, score: backward.score,
      support: backward.support, reason: backward.reason } : null,
    reverseDistance, correction: corrected ? {
      x: corrected.x - current.pose.x, y: corrected.y - current.pose.y,
      rotation: corrected.rotation - current.pose.rotation } : null };
}
