let contextPromise = null;
let contextFailure = '';

const shader = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  featureCount: u32,
  searchRadius: u32,
  patchRadius: u32,
  sampleStride: u32,
  searchWidth: u32,
  padding: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<f32>;
@group(0) @binding(2) var<storage, read> targetImage: array<f32>;
@group(0) @binding(3) var<storage, read> features: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(128)
fn correlate(@builtin(global_invocation_id) id: vec3<u32>) {
  let candidates = params.searchWidth * params.searchWidth;
  let total = params.featureCount * candidates;
  if (id.x >= total) { return; }
  let featureIndex = id.x / candidates;
  let candidate = id.x % candidates;
  let dx = i32(candidate % params.searchWidth) - i32(params.searchRadius);
  let dy = i32(candidate / params.searchWidth) - i32(params.searchRadius);
  let origin = features[featureIndex];
  let fromX = i32(round(origin.x));
  let fromY = i32(round(origin.y));
  let toX = fromX + dx;
  let toY = fromY + dy;
  let radius = i32(params.patchRadius);
  if (toX - radius < 1 || toY - radius < 1 || toX + radius >= i32(params.width) - 1 || toY + radius >= i32(params.height) - 1) {
    scores[id.x] = -1.0;
    return;
  }
  var sumA = 0.0;
  var sumB = 0.0;
  var sumAA = 0.0;
  var sumBB = 0.0;
  var sumAB = 0.0;
  var count = 0.0;
  let stride = i32(params.sampleStride);
  var offsetY = -radius;
  loop {
    if (offsetY > radius) { break; }
    var offsetX = -radius;
    loop {
      if (offsetX > radius) { break; }
      let first = source[u32(fromY + offsetY) * params.width + u32(fromX + offsetX)];
      let second = targetImage[u32(toY + offsetY) * params.width + u32(toX + offsetX)];
      sumA += first; sumB += second; sumAA += first * first; sumBB += second * second; sumAB += first * second; count += 1.0;
      offsetX += stride;
    }
    offsetY += stride;
  }
  let covariance = sumAB - sumA * sumB / count;
  let variance = (sumAA - sumA * sumA / count) * (sumBB - sumB * sumB / count);
  scores[id.x] = select(-1.0, covariance * inverseSqrt(variance), variance > 1e-6);
}
`;

async function context() {
  if (!contextPromise) contextPromise = (async () => {
    if (!globalThis.navigator?.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    const module = device.createShaderModule({ code: shader });
    const compilation = await module.getCompilationInfo();
    if (compilation.messages.some(message => message.type === 'error')) throw new Error(compilation.messages.map(message => message.message).join('\n'));
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'correlate' } });
    return { device, pipeline };
  })().catch(error => { contextFailure = error.message; return null; });
  return contextPromise;
}

export async function patchGpuStatus() {
  const gpu = await context();
  return { available: Boolean(gpu), navigatorGpu: Boolean(globalThis.navigator?.gpu), failure: contextFailure };
}

function buffer(device, data, usage) {
  const result = device.createBuffer({ size: Math.max(4, Math.ceil(data.byteLength / 4) * 4), usage, mappedAtCreation: true });
  new Uint8Array(result.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  result.unmap();
  return result;
}

export function bestLocations(scores, features, searchRadius) {
  const searchWidth = 2 * searchRadius + 1;
  const candidates = searchWidth ** 2;
  return features.map((feature, featureIndex) => {
    const start = featureIndex * candidates;
    let best = -1;
    let bestScore = -1;
    for (let local = 0; local < candidates; local++) {
      if (scores[start + local] > bestScore) { bestScore = scores[start + local]; best = local; }
    }
    if (bestScore < 0.72) return null;
    const localX = best % searchWidth;
    const localY = Math.floor(best / searchWidth);
    const scoreAt = (x, y) => x >= 0 && y >= 0 && x < searchWidth && y < searchWidth ? scores[start + y * searchWidth + x] : bestScore;
    const refine = (low, middle, high) => {
      const curvature = low - 2 * middle + high;
      return Math.abs(curvature) > 1e-6 ? Math.max(-0.75, Math.min(0.75, 0.5 * (low - high) / curvature)) : 0;
    };
    return { x: Math.round(feature.x) + localX - searchRadius + refine(scoreAt(localX - 1, localY), bestScore, scoreAt(localX + 1, localY)),
      y: Math.round(feature.y) + localY - searchRadius + refine(scoreAt(localX, localY - 1), bestScore, scoreAt(localX, localY + 1)), score: bestScore };
  });
}

async function locateWithBuffers(gpu, sourceBuffer, targetBuffer, features, size, searchRadius) {
  const searchWidth = 2 * searchRadius + 1;
  const scoreCount = features.length * searchWidth ** 2;
  if (scoreCount * 4 > gpu.device.limits.maxStorageBufferBindingSize ||
    Math.ceil(scoreCount / 128) > gpu.device.limits.maxComputeWorkgroupsPerDimension) return null;
  const featureData = new Float32Array(features.length * 2);
  features.forEach((feature, index) => { featureData[2 * index] = feature.x; featureData[2 * index + 1] = feature.y; });
  const params = new Uint32Array([this.width, this.height, features.length, searchRadius, Math.floor(size / 2) - 1,
    Math.max(1, Math.floor(size / 12)), searchWidth, 0]);
  const owned = [buffer(gpu.device, params, GPUBufferUsage.UNIFORM), buffer(gpu.device, featureData, GPUBufferUsage.STORAGE)];
  const output = gpu.device.createBuffer({ size: scoreCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = gpu.device.createBuffer({ size: scoreCount * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const bindGroup = gpu.device.createBindGroup({ layout: gpu.pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: owned[0] } }, { binding: 1, resource: { buffer: sourceBuffer } },
    { binding: 2, resource: { buffer: targetBuffer } }, { binding: 3, resource: { buffer: owned[1] } },
    { binding: 4, resource: { buffer: output } }
  ] });
  const encoder = gpu.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(gpu.pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(Math.ceil(scoreCount / 128)); pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, scoreCount * 4);
  gpu.device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const scores = new Float32Array(readback.getMappedRange()).slice();
  readback.unmap();
  for (const item of [...owned, output, readback]) item.destroy();
  return bestLocations(scores, features, searchRadius);
}

export class WebGpuPatchLocator {
  reset() {
    this.previousBuffer?.destroy();
    this.previousBuffer = null;
    this.width = 0;
    this.height = 0;
  }

  async track(previous, current, width, height, features, size, searchRadius) {
    const started = performance.now();
    const gpu = await context();
    const contextMs = performance.now() - started;
    if (!gpu || !features.length || previous.byteLength > gpu.device.limits.maxStorageBufferBindingSize) return null;
    if (this.width !== width || this.height !== height) this.reset();
    this.width = width; this.height = height;
    const uploadStarted = performance.now();
    if (!this.previousBuffer) this.previousBuffer = buffer(gpu.device, previous, GPUBufferUsage.STORAGE);
    const currentBuffer = buffer(gpu.device, current, GPUBufferUsage.STORAGE);
    const uploadMs = performance.now() - uploadStarted;
    try {
      const forwardStarted = performance.now();
      const forward = await locateWithBuffers.call(this, gpu, this.previousBuffer, currentBuffer, features, size, searchRadius);
      const forwardMs = performance.now() - forwardStarted;
      if (!forward) return null;
      const pairs = features.map((feature, local) => forward[local] ? { feature, tracked: forward[local] } : null).filter(Boolean);
      const backwardStarted = performance.now();
      const backward = await locateWithBuffers.call(this, gpu, currentBuffer, this.previousBuffer, pairs.map(pair => pair.tracked), size, searchRadius);
      const backwardMs = performance.now() - backwardStarted;
      if (!backward) return null;
      return { pairs: pairs.map((pair, local) => ({ ...pair, backward: backward[local] })),
        profile: { contextMs, uploadMs, forwardMs, backwardMs } };
    } finally {
      this.previousBuffer?.destroy();
      this.previousBuffer = currentBuffer;
    }
  }
}