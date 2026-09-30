import { detectionImageSize, scaleDetectionOptions } from './detection-scale.js';
import { onWebGpuSelectionChange, requestSelectedGpuAdapter } from './webgpu-selection.js';

let contextPromise = null;
let contextFailure = '';
onWebGpuSelectionChange(() => {
  const previous = contextPromise; contextPromise = null; contextFailure = '';
  void previous?.then(value => value?.device?.destroy()).catch(() => {});
});

const shader = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  padding0: u32,
  padding1: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<u32>;
@group(0) @binding(2) var<storage, read_write> gray: array<f32>;

@compute @workgroup_size(16, 16)
fn grayscale(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.width || id.y >= params.height) { return; }
  let index = id.y * params.width + id.x;
  let pixel = source[index];
  let red = f32(pixel & 255u);
  let green = f32((pixel >> 8u) & 255u);
  let blue = f32((pixel >> 16u) & 255u);
  gray[index] = red * 0.299 + green * 0.587 + blue * 0.114;
}
`;

const angleShader = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  groupsX: u32,
  padding: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gray: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<vec2<f32>>;
var<workgroup> moments: array<vec2<f32>, 256>;

@compute @workgroup_size(16, 16)
fn angleMoments(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) local: u32,
  @builtin(workgroup_id) group: vec3<u32>) {
  let px = 2u + id.x * 2u;
  let py = 2u + id.y * 2u;
  var moment = vec2<f32>(0.0);
  if (px < params.width - 2u && py < params.height - 2u) {
    let index = py * params.width + px;
    let gx = gray[index + 1u] - gray[index - 1u];
    let gy = gray[index + params.width] - gray[index - params.width];
    let squared = gx * gx + gy * gy;
    if (squared >= 144.0) {
      let magnitude = sqrt(squared);
      let cosine2 = (gx * gx - gy * gy) / squared;
      let sine2 = 2.0 * gx * gy / squared;
      moment = magnitude * vec2<f32>(cosine2 * cosine2 - sine2 * sine2, 2.0 * cosine2 * sine2);
    }
  }
  moments[local] = moment;
  workgroupBarrier();
  var stride = 128u;
  loop {
    if (local < stride) { moments[local] += moments[local + stride]; }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride /= 2u;
  }
  if (local == 0u) { partials[group.y * params.groupsX + group.x] = moments[0]; }
}
`;

const cornerShader = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  columns: u32,
  rows: u32,
  originX: f32,
  originY: f32,
  radius: f32,
  scan: f32,
  cosine: f32,
  sine: f32,
  padding0: f32,
  padding1: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gray: array<f32>;
@group(0) @binding(2) var<storage, read_write> candidates: array<vec4<f32>>;

fn sampleGray(px: f32, py: f32) -> f32 {
  if (px < 0.0 || py < 0.0 || px >= f32(params.width - 1u) || py >= f32(params.height - 1u)) { return -1.0; }
  let col = u32(floor(px));
  let row = u32(floor(py));
  let localX = px - f32(col);
  let localY = py - f32(row);
  let index = row * params.width + col;
  return mix(mix(gray[index], gray[index + 1u], localX),
    mix(gray[index + params.width], gray[index + params.width + 1u], localX), localY);
}

fn cornerResponse(px: f32, py: f32) -> f32 {
  var values: array<f32, 4>;
  let cosine = params.cosine;
  let sine = params.sine;
  values[0] = sampleGray(px + params.radius * (-cosine + sine), py + params.radius * (-sine - cosine));
  values[1] = sampleGray(px + params.radius * (cosine + sine), py + params.radius * (sine - cosine));
  values[2] = sampleGray(px + params.radius * (cosine - sine), py + params.radius * (sine + cosine));
  values[3] = sampleGray(px + params.radius * (-cosine - sine), py + params.radius * (-sine + cosine));
  if (min(min(values[0], values[1]), min(values[2], values[3])) < 0.0) { return 0.0; }
  let contrast = abs(values[0] + values[2] - values[1] - values[3]) * 0.5;
  let imbalance = abs(values[0] - values[2]) + abs(values[1] - values[3]);
  return max(0.0, contrast - imbalance);
}

@compute @workgroup_size(16, 16)
fn cornerResponses(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.columns || id.y >= params.rows) { return; }
  let px = params.originX + 2.0 * params.radius + f32(id.x) * params.scan;
  let py = params.originY + 2.0 * params.radius + f32(id.y) * params.scan;
  candidates[id.y * params.columns + id.x] = vec4<f32>(px, py, cornerResponse(px, py), 0.0);
}
`;

const refineShader = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  count: u32,
  radius: u32,
  threshold: f32,
  cosine: f32,
  sine: f32,
  padding: f32,
  roiX: f32,
  roiY: f32,
  roiWidth: f32,
  roiHeight: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gray: array<f32>;
@group(0) @binding(2) var<storage, read> candidates: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> refined: array<vec4<f32>>;

fn inside(px: f32, py: f32) -> bool {
  return px >= params.roiX && py >= params.roiY && px < params.roiX + params.roiWidth &&
    py < params.roiY + params.roiHeight && px >= 0.0 && py >= 0.0 &&
    px < f32(params.width - 1u) && py < f32(params.height - 1u);
}

fn sampleGray(px: f32, py: f32) -> f32 {
  let col = u32(floor(px));
  let row = u32(floor(py));
  let localX = px - f32(col);
  let localY = py - f32(row);
  let index = row * params.width + col;
  return mix(mix(gray[index], gray[index + 1u], localX),
    mix(gray[index + params.width], gray[index + params.width + 1u], localX), localY);
}

fn cornerResponse(px: f32, py: f32) -> f32 {
  let radius = f32(params.radius);
  let cosine = params.cosine;
  let sine = params.sine;
  let positions = array<vec2<f32>, 4>(
    vec2(px + radius * (-cosine + sine), py + radius * (-sine - cosine)),
    vec2(px + radius * (cosine + sine), py + radius * (sine - cosine)),
    vec2(px + radius * (cosine - sine), py + radius * (sine + cosine)),
    vec2(px + radius * (-cosine - sine), py + radius * (-sine + cosine)));
  if (!inside(positions[0].x, positions[0].y) || !inside(positions[1].x, positions[1].y) ||
    !inside(positions[2].x, positions[2].y) || !inside(positions[3].x, positions[3].y)) { return 0.0; }
  let values = array<f32, 4>(sampleGray(positions[0].x, positions[0].y), sampleGray(positions[1].x, positions[1].y),
    sampleGray(positions[2].x, positions[2].y), sampleGray(positions[3].x, positions[3].y));
  let contrast = abs(values[0] + values[2] - values[1] - values[3]) * 0.5;
  let imbalance = abs(values[0] - values[2]) + abs(values[1] - values[3]);
  return max(0.0, contrast - imbalance);
}

@compute @workgroup_size(64)
fn refineCorners(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.count) { return; }
  let candidate = candidates[id.x];
  let radius = i32(params.radius);
  var px = candidate.x;
  var py = candidate.y;
  for (var iteration = 0u; iteration < 8u; iteration++) {
    var xx = 0.0; var xy = 0.0; var yy = 0.0; var bx = 0.0; var by = 0.0;
    for (var offsetY = -radius; offsetY <= radius; offsetY++) {
      for (var offsetX = -radius; offsetX <= radius; offsetX++) {
        let locationX = px + f32(offsetX);
        let locationY = py + f32(offsetY);
        if (!inside(locationX - 1.0, locationY) || !inside(locationX + 1.0, locationY) ||
          !inside(locationX, locationY - 1.0) || !inside(locationX, locationY + 1.0)) { continue; }
        let gx = (sampleGray(locationX + 1.0, locationY) - sampleGray(locationX - 1.0, locationY)) * 0.5;
        let gy = (sampleGray(locationX, locationY + 1.0) - sampleGray(locationX, locationY - 1.0)) * 0.5;
        let weight = exp(-f32(offsetX * offsetX + offsetY * offsetY) / f32(radius * radius));
        xx += weight * gx * gx; xy += weight * gx * gy; yy += weight * gy * gy;
        bx += weight * (gx * gx * locationX + gx * gy * locationY);
        by += weight * (gx * gy * locationX + gy * gy * locationY);
      }
    }
    let determinant = xx * yy - xy * xy;
    if (determinant < 0.000001) { break; }
    let nextX = (yy * bx - xy * by) / determinant;
    let nextY = (xx * by - xy * bx) / determinant;
    if (distance(vec2(nextX, nextY), candidate.xy) > f32(radius * 2)) { break; }
    let change = distance(vec2(nextX, nextY), vec2(px, py));
    px = nextX; py = nextY;
    if (change < 0.005) { break; }
  }
  let valid = select(0.0, 1.0, cornerResponse(px, py) > params.threshold);
  refined[id.x] = vec4(px, py, candidate.z, valid);
}
`;

async function context() {
  if (!contextPromise) contextPromise = (async () => {
    if (!globalThis.navigator?.gpu) return null;
    const adapter = await requestSelectedGpuAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    const module = device.createShaderModule({ code: shader });
    const angleModule = device.createShaderModule({ code: angleShader });
    const cornerModule = device.createShaderModule({ code: cornerShader });
    const refineModule = device.createShaderModule({ code: refineShader });
    const messages = (await Promise.all([module.getCompilationInfo(), angleModule.getCompilationInfo(), cornerModule.getCompilationInfo(),
      refineModule.getCompilationInfo()])).flatMap(info => info.messages);
    if (messages.some(message => message.type === 'error')) throw new Error(messages.map(message => message.message).join('\n'));
    return { device,
      pipeline: device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'grayscale' } }),
      anglePipeline: device.createComputePipeline({ layout: 'auto', compute: { module: angleModule, entryPoint: 'angleMoments' } }),
      cornerPipeline: device.createComputePipeline({ layout: 'auto', compute: { module: cornerModule, entryPoint: 'cornerResponses' } }),
      refinePipeline: device.createComputePipeline({ layout: 'auto', compute: { module: refineModule, entryPoint: 'refineCorners' } }) };
  })().catch(error => { contextFailure = error.message; return null; });
  return contextPromise;
}

function buffer(device, data, usage) {
  const result = device.createBuffer({ size: Math.max(4, Math.ceil(data.byteLength / 4) * 4), usage, mappedAtCreation: true });
  new Uint8Array(result.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  result.unmap();
  return result;
}

export async function gpuGrayscaleBitmap(bitmap, maximumEdge = 3840, options = {}) {
  const started = performance.now();
  const gpu = await context();
  const contextMs = performance.now() - started;
  if (!gpu) return null;
  const { width, height } = detectionImageSize(bitmap.width, bitmap.height, maximumEdge);
  const byteLength = width * height * Float32Array.BYTES_PER_ELEMENT;
  if (width > gpu.device.limits.maxTextureDimension2D || height > gpu.device.limits.maxTextureDimension2D ||
    byteLength > gpu.device.limits.maxStorageBufferBindingSize) return null;
  const resizeStarted = performance.now();
  const canvas = new OffscreenCanvas(width, height);
  const canvasContext = canvas.getContext('2d', { willReadFrequently: true });
  if (!canvasContext) return null;
  canvasContext.drawImage(bitmap, 0, 0, width, height);
  const pixels = canvasContext.getImageData(0, 0, width, height).data;
  const resizeMs = performance.now() - resizeStarted;
  const params = buffer(gpu.device, new Uint32Array([width, height, 0, 0]), GPUBufferUsage.UNIFORM);
  const output = gpu.device.createBuffer({ size: byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = gpu.device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const angleWidth = Math.ceil(Math.max(0, width - 4) / 2);
  const angleHeight = Math.ceil(Math.max(0, height - 4) / 2);
  const angleGroupsX = Math.ceil(angleWidth / 16);
  const angleGroupsY = Math.ceil(angleHeight / 16);
  const angleByteLength = angleGroupsX * angleGroupsY * 2 * Float32Array.BYTES_PER_ELEMENT;
  const angleParams = buffer(gpu.device, new Uint32Array([width, height, angleGroupsX, 0]), GPUBufferUsage.UNIFORM);
  const angleOutput = gpu.device.createBuffer({ size: angleByteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const angleReadback = gpu.device.createBuffer({ size: angleByteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const uploadStarted = performance.now();
  const source = buffer(gpu.device, pixels, GPUBufferUsage.STORAGE);
  const uploadMs = performance.now() - uploadStarted;
  try {
    const bindGroup = gpu.device.createBindGroup({ layout: gpu.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: source } },
      { binding: 2, resource: { buffer: output } }
    ] });
    const commandStarted = performance.now();
    const encoder = gpu.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(gpu.pipeline); pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16)); pass.end();
    const angleBindGroup = gpu.device.createBindGroup({ layout: gpu.anglePipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: angleParams } }, { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: angleOutput } }
    ] });
    const anglePass = encoder.beginComputePass();
    anglePass.setPipeline(gpu.anglePipeline); anglePass.setBindGroup(0, angleBindGroup);
    anglePass.dispatchWorkgroups(angleGroupsX, angleGroupsY); anglePass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, byteLength);
    encoder.copyBufferToBuffer(angleOutput, 0, angleReadback, 0, angleByteLength);
    gpu.device.queue.submit([encoder.finish()]);
    const commandMs = performance.now() - commandStarted;
    const completionStarted = performance.now();
    await Promise.all([readback.mapAsync(GPUMapMode.READ), angleReadback.mapAsync(GPUMapMode.READ)]);
    const gray = new Float32Array(readback.getMappedRange()).slice();
    const anglePartials = new Float32Array(angleReadback.getMappedRange());
    let angleX = 0;
    let angleY = 0;
    for (let index = 0; index < anglePartials.length; index += 2) {
      angleX += anglePartials[index]; angleY += anglePartials[index + 1];
    }
    const angle = Math.atan2(angleY, angleX) / 4;
    const completionMs = performance.now() - completionStarted;
    readback.unmap(); angleReadback.unmap();
    const detectionOptions = scaleDetectionOptions(options, bitmap.width, bitmap.height, width, height);
    let cornerCandidates = null;
    let refinedCorners = null;
    let cornerMs = 0;
    let refineMs = 0;
    if (detectionOptions.approxStep >= 10) {
      const cornerStarted = performance.now();
      const roi = detectionOptions.roi ?? { x: 0, y: 0, width, height };
      const radius = Math.max(2, Math.round(detectionOptions.approxStep * 0.18));
      const scan = Math.max(1, Math.floor(radius / 2));
      const columns = Math.max(0, Math.ceil((roi.width - 4 * radius) / scan));
      const rows = Math.max(0, Math.ceil((roi.height - 4 * radius) / scan));
      const count = columns * rows;
      if (count) {
        const values = new ArrayBuffer(48);
        const integers = new Uint32Array(values);
        const floats = new Float32Array(values);
        integers.set([width, height, columns, rows]);
        floats.set([roi.x, roi.y, radius, scan, Math.cos(angle), Math.sin(angle), 0, 0], 4);
        const cornerParams = buffer(gpu.device, new Uint8Array(values), GPUBufferUsage.UNIFORM);
        const cornerOutput = gpu.device.createBuffer({ size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const cornerReadback = gpu.device.createBuffer({ size: count * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        try {
          const cornerBindGroup = gpu.device.createBindGroup({ layout: gpu.cornerPipeline.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: cornerParams } }, { binding: 1, resource: { buffer: output } },
            { binding: 2, resource: { buffer: cornerOutput } }
          ] });
          const cornerEncoder = gpu.device.createCommandEncoder();
          const cornerPass = cornerEncoder.beginComputePass();
          cornerPass.setPipeline(gpu.cornerPipeline); cornerPass.setBindGroup(0, cornerBindGroup);
          cornerPass.dispatchWorkgroups(Math.ceil(columns / 16), Math.ceil(rows / 16)); cornerPass.end();
          cornerEncoder.copyBufferToBuffer(cornerOutput, 0, cornerReadback, 0, count * 16);
          gpu.device.queue.submit([cornerEncoder.finish()]);
          await cornerReadback.mapAsync(GPUMapMode.READ);
          const values = new Float32Array(cornerReadback.getMappedRange());
          const minimumResponse = Math.max(30, (detectionOptions.threshold ?? 16) * 3);
          cornerCandidates = [];
          for (let index = 0; index < count; index++) {
            if (values[4 * index + 2] > minimumResponse) cornerCandidates.push({
              x: values[4 * index], y: values[4 * index + 1], response: values[4 * index + 2]
            });
          }
          cornerReadback.unmap();
          cornerMs = performance.now() - cornerStarted;
          if (cornerCandidates.length) {
            const refineStarted = performance.now();
            cornerCandidates.sort((first, second) => second.response - first.response);
            const candidateValues = new Float32Array(cornerCandidates.length * 4);
            cornerCandidates.forEach((candidate, index) => candidateValues.set([candidate.x, candidate.y, candidate.response, 0], index * 4));
            const refineParamsValues = new ArrayBuffer(48);
            new Uint32Array(refineParamsValues).set([width, height, cornerCandidates.length, radius]);
            new Float32Array(refineParamsValues).set([minimumResponse, Math.cos(angle), Math.sin(angle), 0,
              roi.x, roi.y, roi.width, roi.height], 4);
            const refineParams = buffer(gpu.device, new Uint8Array(refineParamsValues), GPUBufferUsage.UNIFORM);
            const refineInput = buffer(gpu.device, candidateValues, GPUBufferUsage.STORAGE);
            const refineOutput = gpu.device.createBuffer({ size: candidateValues.byteLength,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
            const refineReadback = gpu.device.createBuffer({ size: candidateValues.byteLength,
              usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            try {
              const refineBindGroup = gpu.device.createBindGroup({ layout: gpu.refinePipeline.getBindGroupLayout(0), entries: [
                { binding: 0, resource: { buffer: refineParams } }, { binding: 1, resource: { buffer: output } },
                { binding: 2, resource: { buffer: refineInput } }, { binding: 3, resource: { buffer: refineOutput } }
              ] });
              const refineEncoder = gpu.device.createCommandEncoder();
              const refinePass = refineEncoder.beginComputePass();
              refinePass.setPipeline(gpu.refinePipeline); refinePass.setBindGroup(0, refineBindGroup);
              refinePass.dispatchWorkgroups(Math.ceil(cornerCandidates.length / 64)); refinePass.end();
              refineEncoder.copyBufferToBuffer(refineOutput, 0, refineReadback, 0, candidateValues.byteLength);
              gpu.device.queue.submit([refineEncoder.finish()]);
              await refineReadback.mapAsync(GPUMapMode.READ);
              const refinedValues = new Float32Array(refineReadback.getMappedRange());
              refinedCorners = [];
              for (let index = 0; index < cornerCandidates.length; index++) {
                const candidate = cornerCandidates[index];
                if (refinedCorners.some(point => Math.hypot(point.x - candidate.x, point.y - candidate.y) < detectionOptions.approxStep * 0.5)) continue;
                if (refinedValues[4 * index + 3] > 0.5) refinedCorners.push({
                  x: refinedValues[4 * index], y: refinedValues[4 * index + 1],
                  confidence: Math.min(1, Math.max(0.1, candidate.response / 255))
                });
                if (refinedCorners.length > 3000) break;
              }
              refineReadback.unmap();
            } finally { refineParams.destroy(); refineInput.destroy(); refineOutput.destroy(); refineReadback.destroy(); }
            refineMs = performance.now() - refineStarted;
          }
        } finally { cornerParams.destroy(); cornerOutput.destroy(); cornerReadback.destroy(); }
      }
      if (!cornerMs) cornerMs = performance.now() - cornerStarted;
    }
    let minimum = Infinity;
    let maximum = -Infinity;
    let sum = 0;
    let samples = 0;
    const stride = Math.max(1, Math.floor(gray.length / 4096));
    for (let index = 0; index < gray.length; index += stride) {
      minimum = Math.min(minimum, gray[index]); maximum = Math.max(maximum, gray[index]); sum += gray[index]; samples++;
    }
    return { width, height, data: gray, angle, cornerCandidates, refinedCorners, options: detectionOptions,
      timing: { contextMs, resizeMs, uploadMs, commandMs, completionMs, cornerMs, refineMs,
      grayMinimum: minimum, grayMaximum: maximum, grayMean: sum / samples } };
  } finally {
    source.destroy(); params.destroy(); output.destroy(); readback.destroy(); angleParams.destroy(); angleOutput.destroy(); angleReadback.destroy();
  }
}

export function gpuImageFailure() {
  return contextFailure;
}

export async function gpuImageStatus() {
  const gpu = await context();
  return { available: Boolean(gpu), failure: contextFailure, limits: gpu ? {
    maxTextureDimension2D: gpu.device.limits.maxTextureDimension2D,
    maxStorageBufferBindingSize: gpu.device.limits.maxStorageBufferBindingSize
  } : null };
}
