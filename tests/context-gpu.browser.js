import { ContextTracker, contextImage, registerOverlap, registerOverlapAsync } from '../src/context-tracker.js';
import { WebGpuContextTracker } from '../src/webgpu-context-tracker.js';
import { WebGpuRemapper } from '../src/webgpu-native-remapper.js';
import { orientationFromMatrix } from '../src/video-orientation.js';
import { WorkerClient } from '../src/rpc.js';

function check(condition, message) { if (!condition) throw new Error(message); }

function texture(dx = 0, dy = 0, angle = 0, width = 256, height = 192) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let row = 0; row < height; row++) for (let column = 0; column < width; column++) {
    const shiftedX = column + 0.5 - width / 2 - dx; const shiftedY = row + 0.5 - height / 2 - dy;
    const sourceX = Math.cos(angle) * shiftedX + Math.sin(angle) * shiftedY;
    const sourceY = -Math.sin(angle) * shiftedX + Math.cos(angle) * shiftedY;
    const value = 125 + 28 * Math.sin(sourceX * 0.31 + sourceY * 0.19) + 32 * Math.cos(sourceX * 0.13 - sourceY * 0.37) +
      25 * Math.sin(sourceX * 0.051 + sourceY * 0.11) + 18 * Math.cos(sourceX * 0.43 + sourceY * 0.07);
    data.set([value, value, value, 255], 4 * (row * width + column));
  }
  return { width, height, data };
}

export async function runContextGpuChecks() {
  const gpu = new WebGpuContextTracker();
  try {
    check(await gpu.ready(), `WebGPU unavailable: ${gpu.failure}`);
    const colored = texture(0, 0, 0, 257, 193);
    for (let index = 0; index < colored.width * colored.height; index++) colored.data.set([index % 256, (index * 17) % 256, (index * 29) % 256, index % 31 ? 255 : 0], index * 4);
    const mask = { sourceWidth: colored.width, sourceHeight: colored.height, width: 52, height: 39, cellSize: 5,
      data: Uint8Array.from({ length: 52 * 39 }, (_, index) => index % 7 ? 1 : 2) };
    const cpuPyramid = contextImage(colored, mask); const gpuPyramid = await gpu.image(colored, mask);
    for (let index = 0; index < cpuPyramid.levels.length; index++) check(cpuPyramid.levels[index].gray.every((value, offset) => value === gpuPyramid.levels[index].gray[offset]), `Pyramid mismatch at level ${index}`);
    const origin = { x: 0, y: 0, rotation: 0 }; const angle = 0.026;
    const sourceImage = texture(); const currentImage = texture(34, -12, angle);
    const source = await gpu.image(sourceImage); const current = await gpu.image(currentImage);
    const expected = { x: -Math.cos(angle) * 34 + Math.sin(angle) * 12, y: Math.sin(angle) * 34 + Math.cos(angle) * 12, rotation: -angle };
    const prediction = { x: expected.x + 3, y: expected.y - 2, rotation: expected.rotation + 0.003 };
    const args = [current, source, prediction, origin, { radius: 16, angle: 1 }];
    const startedCpu = performance.now(); const cpu = registerOverlap(...args); const cpuMs = performance.now() - startedCpu;
    const startedGpu = performance.now(); const accelerated = await registerOverlapAsync(gpu, ...args); const gpuMs = performance.now() - startedGpu;
    check(cpu.accepted && accelerated.accepted, `Pair rejected: ${JSON.stringify({ cpu, accelerated })}`);
    const poseDifference = Math.hypot(cpu.pose.x - accelerated.pose.x, cpu.pose.y - accelerated.pose.y);
    check(poseDifference < 0.05 && Math.abs(cpu.pose.rotation - accelerated.pose.rotation) < 0.0002, `Pose mismatch: ${JSON.stringify({ cpu, accelerated })}`);
    const unsupported = contextImage({ ...sourceImage, data: new Uint8ClampedArray(sourceImage.data.length) });
    check(!(await registerOverlapAsync(gpu, unsupported, source, origin, origin)).accepted, 'Transparent image accepted');
    const rejected = await registerOverlapAsync(gpu, gpuPyramid, source, origin, origin);
    check(!rejected.accepted, 'Unrelated masked color texture accepted');
    const halfMask = { sourceWidth: 256, sourceHeight: 192, width: 32, height: 24, cellSize: 8,
      data: Uint8Array.from({ length: 32 * 24 }, (_, index) => index % 32 < 16 ? 1 : 0) };
    const corrupted = texture();
    for (let row = 0; row < 192; row++) for (let column = 128; column < 256; column++) corrupted.data.fill((row * 173 + column * 29) % 256, (row * 256 + column) * 4, (row * 256 + column) * 4 + 3);
    const maskedSource = await gpu.image(sourceImage, halfMask); const maskedCurrent = await gpu.image(corrupted, halfMask);
    const maskedArgs = [maskedCurrent, maskedSource, { ...origin, x: 2 }, origin];
    const maskedCpu = registerOverlap(...maskedArgs); const maskedGpu = await registerOverlapAsync(gpu, ...maskedArgs);
    check(maskedCpu.accepted && maskedGpu.accepted && Math.hypot(maskedCpu.pose.x - maskedGpu.pose.x, maskedCpu.pose.y - maskedGpu.pose.y) < 0.05, 'Masked exclusion parity failed');
    for (const maskValue of [0, 2]) {
      halfMask.data.fill(maskValue);
      const excluded = await gpu.image(sourceImage, halfMask);
      check(excluded.levels.every(level => level.gray.every(value => value === -1)), 'Excluded mask leaked pixels');
      check(!(await registerOverlapAsync(gpu, excluded, source, origin, origin)).accepted, 'Excluded image accepted');
    }
    const tracker = new ContextTracker(1024 * 1024, gpu);
    const options = { useWebGpu: true, contextRecent: 2, contextSpatial: 2, contextRadius: 16, contextAngle: 1 };
    const frames = [];
    for (let frame = 0; frame < 3; frame++) {
      await tracker.begin(sourceImage, frame, { ...origin, x: frame === 2 ? 2 : 0 }, options);
      frames.push(tracker.finish());
    }
    check(frames[2].applied && frames[2].accelerator.includes('WebGPU') && !gpu.failure, `GPU consensus failed: ${JSON.stringify(frames[2])}`);
    check(Boolean(frames[2].registrationChoice), 'Missing GPU/CPU runtime selection');
    const device = gpu.device; device.destroy(); await device.lost;
    await tracker.begin(sourceImage, 3, origin, options); const fallback = tracker.finish();
    check(fallback.applied && fallback.accelerator === 'CPU' && fallback.fallback && gpu.cacheBytes === 0, 'Device-loss fallback failed');
    tracker.reset();
    check(gpu.cacheBytes === 0 && !gpu.failure, 'GPU reset failed');
    const limited = new ContextTracker(1024 * 1024, new WebGpuContextTracker(1));
    try {
      await limited.begin(sourceImage, 0, origin, options);
      const result = limited.finish();
      check(result.accelerator === 'CPU' && result.fallback.includes('Cachebudget') && result.gpuCacheBytes === 0, 'GPU limit fallback failed');
    } finally { limited.reset(); }
    return { pyramidExact: true, poseDifference, cpuMs, gpuMs, cpu, accelerated, registrationChoice: frames[2].registrationChoice,
      maskParity: true, limitFallback: true, consensus: frames.map(frame => ({ applied: frame.applied, accelerator: frame.accelerator })), deviceLossFallback: fallback.fallback };
  } finally { gpu.reset(); }
}

export async function benchmarkContextGpu(width = 2048, height = 1536) {
  const gpu = new WebGpuContextTracker();
  try {
    check(await gpu.ready(), gpu.failure);
    const source = texture(0, 0, 0, width, height); const moved = texture(2, -1, 0.001, width, height);
    const startedCpu = performance.now(); const cpuSource = contextImage(source); const cpuMoved = contextImage(moved);
    const cpuPyramidMs = performance.now() - startedCpu;
    const startedGpu = performance.now(); const gpuSource = await gpu.image(source); const gpuMoved = await gpu.image(moved);
    const gpuPyramidMs = performance.now() - startedGpu;
    const prediction = { x: -1, y: 0, rotation: 0 }; const origin = { x: 0, y: 0, rotation: 0 };
    const cpuMatchStarted = performance.now(); const cpu = registerOverlap(cpuMoved, cpuSource, prediction, origin);
    const cpuRegistrationMs = performance.now() - cpuMatchStarted;
    const gpuMatchStarted = performance.now(); const accelerated = await registerOverlapAsync(gpu, gpuMoved, gpuSource, prediction, origin);
    const gpuRegistrationMs = performance.now() - gpuMatchStarted;
    check(cpu.accepted === accelerated.accepted, `Benchmark acceptance mismatch: ${JSON.stringify({ cpu, accelerated })}`);
    if (cpu.accepted) check(Math.hypot(cpu.pose.x - accelerated.pose.x, cpu.pose.y - accelerated.pose.y) < 0.1, 'Benchmark pose mismatch');
    const cache = new ContextTracker();
    cache.remember(0, gpuSource); cache.remember(1, gpuMoved);
    const cacheStarted = performance.now();
    for (let repeat = 0; repeat < 1000; repeat++) cache.remember(repeat % 2, cache.cache.get(repeat % 2));
    const cacheHitMs = (performance.now() - cacheStarted) / 1000;
    return { width, height, cpuPyramidMs, gpuPyramidMs, cpuRegistrationMs, gpuRegistrationMs,
      pyramids: [gpuSource.pyramidTiming, gpuMoved.pyramidTiming], pyramidBytes: gpuSource.bytes,
      cacheCapacity: Math.floor(cache.cacheLimit / gpuSource.bytes), cacheHitMs,
      storageLimit: gpu.device.limits.maxStorageBufferBindingSize, cpu, accelerated };
  } finally { gpu.reset(); }
}

function identityMaps(width, height) {
  return { outputWidth: width, outputHeight: height,
    inverseX: Float32Array.from({ length: width * height }, (_, index) => index % width),
    inverseY: Float32Array.from({ length: width * height }, (_, index) => Math.floor(index / width)),
    valid: new Uint8Array(width * height).fill(1) };
}

function videoFixture(width, height, axes = [1, 0, 0, 1]) {
  const image = texture(0, 0, 0, width, height);
  const canvas = new OffscreenCanvas(width, height);
  canvas.getContext('2d').putImageData(new ImageData(image.data, width, height), 0, 0);
  const [horizontalX, horizontalY, verticalX, verticalY] = axes;
  return { frame: new VideoFrame(canvas, { timestamp: 0 }), orientation: orientationFromMatrix([
    horizontalX * 65536, horizontalY * 65536, 0, verticalX * 65536, verticalY * 65536, 0, 0, 0, 1073741824], width, height) };
}

export async function runNativeContextChecks() {
  const gpu = new WebGpuContextTracker();
  const results = [];
  try {
    for (const axes of [[1, 0, 0, 1], [0, 1, -1, 0], [-1, 0, 0, 1]]) {
      const fixture = videoFixture(259, 197, axes);
      const width = fixture.orientation.width - 2, height = fixture.orientation.height - 2;
      const maps = identityMaps(width, height); maps.valid[9] = 0;
      for (let index = 0; index < maps.valid.length; index++) { maps.inverseX[index] += 0.25; maps.inverseY[index] += 0.375; }
      const mask = { sourceWidth: width, sourceHeight: height, width: Math.ceil(width / 8), height: Math.ceil(height / 8), cellSize: 8 };
      mask.data = Uint8Array.from({ length: mask.width * mask.height }, (_, index) => index % 7 ? 1 : 2);
      try {
        const native = await gpu.image({ ...fixture, width, height, maps, readRgba: true }, mask);
        const expected = contextImage(native.rgba, mask);
        for (const [index, level] of native.levels.entries()) check(level.gray.every((value, offset) => value === expected.levels[index].gray[offset]), `Native pyramid mismatch ${axes} level ${index}`);
        check(native.pyramidTiming.uploadBytes < width * height, 'Native path uploaded full RGBA');
        const reference = await gpu.image({ ...fixture, width, height, maps }, mask);
        check(!reference.rgba && reference.pyramidTiming.rgbaReadbackBytes === 0, 'Reference read RGBA back');
        check(reference.levels[0].gray.every((value, index) => value === expected.levels[0].gray[index]), 'Reference differs from current frame');
        results.push({ axes, exact: true, uploadBytes: reference.pyramidTiming.uploadBytes });
      } finally { fixture.frame.close(); }
    }
    const fixture = videoFixture(192, 144), maps = identityMaps(190, 142);
    const options = { useWebGpu: true, contextRecent: 2, contextSpatial: 2, contextRadius: 16, contextAngle: 1 };
    const pose = { x: 0, y: 0, rotation: 0 };
    const tracker = new ContextTracker(1, gpu);
    const image = { ...fixture, width: 190, height: 142, maps };
    try {
      await tracker.begin(image, 0, pose, options); tracker.finish();
      const needed = await tracker.begin(image, 1, pose, options);
      check(needed.length === 1 && needed[0] === 0, 'Expected reference cache miss');
      await tracker.provide(0, image); const result = tracker.finish();
      check(result.pyramidProfile.nativeImages === 2 && result.cacheMisses === 1, 'Native reference load/profile failed');
      gpu.device.destroy(); await gpu.device.lost;
      const fallback = await tracker.image({ ...image, readRgba: true }, options);
      check(fallback.accelerator === 'CPU' && fallback.rgba && fallback.levels[0].gray.some(value => value >= 0), 'Native CPU fallback failed');
      tracker.reset(); check(!gpu.nativeRemapper && gpu.cacheBytes === 0, 'Native reset leaked resources');
    } finally { fixture.frame.close(); }

    const worker = new WorkerClient('/compute-worker.js');
    const workerFixture = videoFixture(192, 144);
    try {
      await worker.call('tracking-maps', { maps });
      const workerOptions = { ...options, rectangle: { x: 20, y: 20, width: 128, height: 96 }, patchSearchRadius: 16, maxRotation: 3 };
      const diagnostics = [];
      for (let index = 0; index < 3; index++) {
        const frame = workerFixture.frame.clone();
        let result = await worker.call('track-window-native', { index, options: workerOptions, native: { frame, orientation: workerFixture.orientation } }, [frame]);
        check(result.preview?.width === 190 && result.success, 'Native worker preview/window failed');
        result.preview.close();
        check(result.contextPending && !result.references.length, 'Native worker unexpectedly missed cached reference');
        result = await worker.call('context-finish');
        diagnostics.push({ cacheHits: result.context.cacheHits, nativeImages: result.context.pyramidProfile.nativeImages, applied: result.context.applied });
      }
      check(diagnostics[2].applied && diagnostics[2].cacheHits === 2, 'Native worker consensus/cache failed');
      await worker.call('reset');
      return { pyramids: results, nativeReference: true, deviceLossFallback: true, worker: diagnostics };
    } finally { worker.terminate(); workerFixture.frame.close(); }
  } finally { gpu.reset(); }
}

export async function benchmarkNativeContext(width = 4096, height = 3072) {
  const fixture = videoFixture(width + 2, height + 2), maps = identityMaps(width, height);
  const remapper = await WebGpuRemapper.create(maps), gpu = new WebGpuContextTracker();
  const trials = [];
  try {
    await gpu.ready(); gpu.retainImages = false;
    const descriptor = { ...fixture, maps, width, height };
    await gpu.image(descriptor);
    for (let trial = 0; trial < 3; trial++) {
      const started = performance.now();
      const rgba = await remapper.render(fixture.frame, fixture.orientation, 'rgba');
      const remapMs = performance.now() - started;
      const uploaded = await gpu.image(rgba);
      const previousMs = performance.now() - started;
      const directStarted = performance.now();
      const native = await gpu.image(descriptor);
      const directMs = performance.now() - directStarted;
      for (const [index, level] of native.levels.entries()) check(level.gray.every((value, offset) => value === uploaded.levels[index].gray[offset]), 'Benchmark native pixel mismatch');
      trials.push({ remapMs, previousMs, directMs, previous: uploaded.pyramidTiming, direct: native.pyramidTiming });
    }
    return { width, height, trials };
  } finally { gpu.reset(); remapper.destroy(); fixture.frame.close(); }
}