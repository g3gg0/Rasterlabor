let contextPromise = null;
let contextFailure = '';

const shader = /* wgsl */ `
struct Params {
  sourceWidth: u32,
  sourceHeight: u32,
  outputCount: u32,
  padding: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<u32>;
@group(0) @binding(2) var<storage, read> inverseX: array<f32>;
@group(0) @binding(3) var<storage, read> inverseY: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<u32>;

fn channel(pixel: u32, shift: u32) -> f32 {
  return f32((pixel >> shift) & 255u);
}

fn interpolate(first: u32, second: u32, third: u32, fourth: u32, localX: f32, localY: f32, shift: u32) -> u32 {
  let top = mix(channel(first, shift), channel(second, shift), localX);
  let bottom = mix(channel(third, shift), channel(fourth, shift), localX);
  return u32(clamp(round(mix(top, bottom, localY)), 0.0, 255.0));
}

@compute @workgroup_size(256)
fn remap(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.outputCount) { return; }
  let px = inverseX[index];
  let py = inverseY[index];
  if (px < 0.0 || py < 0.0) {
    output[index] = 0u;
    return;
  }
  let col = u32(floor(px));
  let row = u32(floor(py));
  if (col + 1u >= params.sourceWidth || row + 1u >= params.sourceHeight) {
    output[index] = 0u;
    return;
  }
  let firstIndex = row * params.sourceWidth + col;
  let first = source[firstIndex];
  let second = source[firstIndex + 1u];
  let third = source[firstIndex + params.sourceWidth];
  let fourth = source[firstIndex + params.sourceWidth + 1u];
  let localX = px - f32(col);
  let localY = py - f32(row);
  let red = interpolate(first, second, third, fourth, localX, localY, 0u);
  let green = interpolate(first, second, third, fourth, localX, localY, 8u);
  let blue = interpolate(first, second, third, fourth, localX, localY, 16u);
  output[index] = red | (green << 8u) | (blue << 16u) | 0xff000000u;
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
    const errors = compilation.messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(message => message.message).join('\n'));
    return { device, pipeline: device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'remap' } }) };
  })().catch(error => { contextFailure = error.message; return null; });
  return contextPromise;
}

export async function remapGpuStatus() {
  const gpu = await context();
  return { available: Boolean(gpu), navigatorGpu: Boolean(globalThis.navigator?.gpu), failure: contextFailure };
}

function buffer(device, data, usage) {
  const result = device.createBuffer({ size: Math.max(4, Math.ceil(data.byteLength / 4) * 4), usage, mappedAtCreation: true });
  new Uint8Array(result.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  result.unmap();
  return result;
}

export function remapChunkRows(outputWidth, maximumBindingSize, maximumWorkgroups) {
  const bindingRows = Math.floor(maximumBindingSize / (outputWidth * 4));
  const dispatchRows = Math.floor(maximumWorkgroups * 256 / outputWidth);
  return Math.max(1, Math.min(1024, bindingRows, dispatchRows));
}

export class WebGpuRemapper {
  reset() {
    for (const chunk of this.chunks ?? []) {
      chunk.inverseXBuffer.destroy();
      chunk.inverseYBuffer.destroy();
      chunk.paramsBuffer.destroy();
    }
    this.chunks = [];
    this.maps = null;
  }

  async configure(gpu, image, maps) {
    if (this.maps === maps) return;
    this.reset();
    const outputCount = maps.outputWidth * maps.outputHeight;
    const maskedX = maps.inverseX.slice();
    for (let index = 0; index < outputCount; index++) if (!maps.valid[index]) maskedX[index] = -1;
    const chunkRows = remapChunkRows(maps.outputWidth, gpu.device.limits.maxStorageBufferBindingSize,
      gpu.device.limits.maxComputeWorkgroupsPerDimension);
    for (let startRow = 0; startRow < maps.outputHeight; startRow += chunkRows) {
      const rows = Math.min(chunkRows, maps.outputHeight - startRow);
      const start = startRow * maps.outputWidth;
      const count = rows * maps.outputWidth;
      this.chunks.push({ start, count,
        inverseXBuffer: buffer(gpu.device, maskedX.subarray(start, start + count), GPUBufferUsage.STORAGE),
        inverseYBuffer: buffer(gpu.device, maps.inverseY.subarray(start, start + count), GPUBufferUsage.STORAGE),
        paramsBuffer: buffer(gpu.device, new Uint32Array([image.width, image.height, count, 0]), GPUBufferUsage.UNIFORM) });
    }
    this.maps = maps;
  }

  async remap(image, maps) {
    const started = performance.now();
    const gpu = await context();
    const contextMs = performance.now() - started;
    if (!gpu || image.data.byteLength % 4) return null;
    const outputBytes = maps.outputWidth * maps.outputHeight * 4;
    if (image.data.byteLength > gpu.device.limits.maxStorageBufferBindingSize ||
      image.data.byteLength > gpu.device.limits.maxBufferSize) return null;
    await this.configure(gpu, image, maps);
    const uploadStarted = performance.now();
    const sourceBuffer = buffer(gpu.device, image.data, GPUBufferUsage.STORAGE);
    const uploadMs = performance.now() - uploadStarted;
    const data = new Uint8ClampedArray(outputBytes);
    let commandMs = 0;
    let completionMs = 0;
    let readbackMs = 0;
    try {
      for (const chunk of this.chunks) {
        const chunkBytes = chunk.count * 4;
        const outputBuffer = gpu.device.createBuffer({ size: chunkBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const readbackBuffer = gpu.device.createBuffer({ size: chunkBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        try {
          const commandStarted = performance.now();
          const bindGroup = gpu.device.createBindGroup({ layout: gpu.pipeline.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: chunk.paramsBuffer } }, { binding: 1, resource: { buffer: sourceBuffer } },
            { binding: 2, resource: { buffer: chunk.inverseXBuffer } }, { binding: 3, resource: { buffer: chunk.inverseYBuffer } },
            { binding: 4, resource: { buffer: outputBuffer } }
          ] });
          const encoder = gpu.device.createCommandEncoder();
          const pass = encoder.beginComputePass();
          pass.setPipeline(gpu.pipeline); pass.setBindGroup(0, bindGroup);
          pass.dispatchWorkgroups(Math.ceil(chunk.count / 256)); pass.end();
          encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, chunkBytes);
          gpu.device.queue.submit([encoder.finish()]);
          commandMs += performance.now() - commandStarted;
          const completionStarted = performance.now();
          await readbackBuffer.mapAsync(GPUMapMode.READ);
          completionMs += performance.now() - completionStarted;
          const readbackStarted = performance.now();
          data.set(new Uint8ClampedArray(readbackBuffer.getMappedRange()), chunk.start * 4);
          readbackMs += performance.now() - readbackStarted;
          readbackBuffer.unmap();
        } finally { outputBuffer.destroy(); readbackBuffer.destroy(); }
      }
      return { width: maps.outputWidth, height: maps.outputHeight, data,
        timing: { contextMs, uploadMs, commandMs, completionMs, readbackMs, chunks: this.chunks.length,
          totalMs: performance.now() - started } };
    } finally {
      sourceBuffer.destroy();
    }
  }
}