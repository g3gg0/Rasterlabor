import { Matrix } from 'ml-matrix';
import { WebGpuRemapper } from './webgpu-native-remapper.js';
import { requestSelectedGpuAdapter } from './webgpu-selection.js';

const pyramidShader = /* wgsl */ `
struct Params { width: u32, height: u32, outputWidth: u32, outputHeight: u32, mode: u32, maskWidth: u32, cellSize: u32, masked: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<u32>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var rectified: texture_2d<f32>;
fn gray(index: u32) -> u32 { return (source[index / 2u] >> ((index % 2u) * 16u)) & 65535u; }
fn value(index: u32) -> u32 {
  if (index >= params.outputWidth * params.outputHeight) { return 65535u; }
  let column = index % params.outputWidth; let row = index / params.outputWidth;
  if (params.mode != 1u) {
    var rgba: u32;
    if (params.mode == 2u) {
      let channels = vec4u(round(textureLoad(rectified, vec2i(i32(column), i32(row)), 0) * 255.0));
      rgba = channels.r | (channels.g << 8u) | (channels.b << 16u) | (channels.a << 24u);
    } else { rgba = source[index]; }
    if ((rgba >> 24u) != 255u) { return 65535u; }
    if (params.masked != 0u && mask[(row / params.cellSize) * params.maskWidth + column / params.cellSize] != 1u) { return 65535u; }
    return (299u * (rgba & 255u) + 587u * ((rgba >> 8u) & 255u) + 114u * ((rgba >> 16u) & 255u) + 500u) / 1000u;
  }
  let offset = row * 2u * params.width + column * 2u;
  let first = gray(offset); let second = gray(offset + 1u);
  let third = gray(offset + params.width); let fourth = gray(offset + params.width + 1u);
  if (max(max(first, second), max(third, fourth)) > 255u) { return 65535u; }
  return (first + second + third + fourth + 2u) / 4u;
}
@compute @workgroup_size(256)
fn pyramid(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let index = id.x + id.y * groups.x * 256u;
  if (index * 2u >= params.outputWidth * params.outputHeight) { return; }
  output[index] = value(index * 2u) | (value(index * 2u + 1u) << 16u);
}
`;

const registrationShader = /* wgsl */ `
struct Params {
  count: u32, width: u32, height: u32, scale: f32,
  offsetX: f32, offsetY: f32, cosine: f32, sine: f32,
  gain: f32, bias: f32, lever: f32, mode: u32,
  halfWidth: f32, halfHeight: f32, padding: vec2<f32>
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> referencePixels: array<u32>;
@group(0) @binding(2) var<storage, read> samples: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> partial: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
var<workgroup> scratch: array<f32, 768>;
fn gray(index: u32) -> f32 {
  let value = (referencePixels[index / 2u] >> ((index % 2u) * 16u)) & 65535u;
  return select(f32(value), -1.0, value == 65535u);
}
fn pixel(point: vec2<f32>) -> f32 {
  if (point.x < 1.0 || point.y < 1.0 || point.x >= f32(params.width) - 2.0 || point.y >= f32(params.height) - 2.0) { return -1.0; }
  let offset = u32(floor(point.y)) * params.width + u32(floor(point.x));
  let first = gray(offset); let second = gray(offset + 1u);
  let third = gray(offset + params.width); let fourth = gray(offset + params.width + 1u);
  if (min(min(first, second), min(third, fourth)) < 0.0) { return -1.0; }
  let fraction = fract(point);
  return mix(mix(first, second, fraction.x), mix(third, fourth, fraction.x), fraction.y);
}
fn reduce(local: u32) {
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      for (var slot = 0u; slot < 12u; slot++) { scratch[local * 12u + slot] += scratch[(local + stride) * 12u + slot]; }
    }
    workgroupBarrier();
  }
}
@compute @workgroup_size(64)
fn measure(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3<u32>) {
  var values: array<f32, 12>;
  if (id.x < params.count) {
    let sample = samples[id.x];
    let rotated = vec2(params.cosine * sample.x - params.sine * sample.y, params.sine * sample.x + params.cosine * sample.y);
    let point = (rotated + vec2(params.offsetX + params.halfWidth, params.offsetY + params.halfHeight)) / params.scale - vec2(0.5);
    let value = pixel(point);
    if (value >= 0.0) {
      if (params.mode == 0u) {
        let first = sample.z - 128.0; let second = value - 128.0;
        values[0] = 1.0; values[1] = first; values[2] = second;
        values[3] = first * first; values[4] = second * second; values[5] = first * second;
      } else if (output[0] >= 128.0) {
        let left = pixel(point - vec2(1.0, 0.0)); let right = pixel(point + vec2(1.0, 0.0));
        let top = pixel(point - vec2(0.0, 1.0)); let bottom = pixel(point + vec2(0.0, 1.0));
        if (min(min(left, right), min(top, bottom)) >= 0.0) {
          let gx = (right - left) / (2.0 * params.scale); let gy = (bottom - top) / (2.0 * params.scale);
          let jacobian = vec3(gx, gy, (-gx * rotated.y + gy * rotated.x) / params.lever);
          let count = output[0];
          let variance = output[3] - output[1] * output[1] / count;
          let gain = (output[5] - output[1] * output[2] / count) / max(variance, 0.0001);
          let bias = (output[2] - gain * output[1]) / count + 128.0 * (1.0 - gain);
          let residual = value - gain * sample.z - bias;
          let weight = min(1.0, 15.0 / max(1.0, abs(residual)));
          values[0] = weight * jacobian.x * jacobian.x; values[1] = weight * jacobian.x * jacobian.y;
          values[2] = weight * jacobian.x * jacobian.z; values[3] = weight * jacobian.y * jacobian.y;
          values[4] = weight * jacobian.y * jacobian.z; values[5] = weight * jacobian.z * jacobian.z;
          values[6] = -weight * jacobian.x * residual; values[7] = -weight * jacobian.y * residual; values[8] = -weight * jacobian.z * residual;
        }
      }
    }
  }
  for (var slot = 0u; slot < 12u; slot++) { scratch[local * 12u + slot] = values[slot]; }
  reduce(local);
  if (local == 0u) { for (var slot = 0u; slot < 12u; slot++) { partial[group.x * 12u + slot] = scratch[slot]; } }
}
@compute @workgroup_size(64)
fn aggregate(@builtin(local_invocation_index) local: u32) {
  for (var slot = 0u; slot < 12u; slot++) {
    var sum = 0.0;
    for (var group = local; group < (params.count + 63u) / 64u; group += 64u) { sum += partial[group * 12u + slot]; }
    scratch[local * 12u + slot] = sum;
  }
  reduce(local);
  if (local == 0u) { for (var slot = 0u; slot < 12u; slot++) { output[params.mode * 12u + slot] = scratch[slot]; } }
}
`;

export class WebGpuContextTracker {
  constructor(cacheLimit = 192 * 1024 * 1024) { this.cacheLimit = cacheLimit; this.reset(); }

  reset() {
    this.releaseRemap();
    this.release();
    this.device?.destroy(); this.device = null; this.initializing = null; this.failure = ''; this.retainImages = true;
  }

  release() {
    for (const entry of this.cache?.values() ?? []) entry.buffer.destroy();
    this.cache = new Map(); this.levels = new WeakMap(); this.cacheBytes = 0;
    for (const buffer of this.workspace ?? []) buffer.destroy();
    this.workspace = null; this.stage = null;
  }

  disable(error) {
    this.failure = error?.message || String(error);
    this.releaseRemap();
    this.release();
  }

  releaseRemap() {
    for (const resource of this.nativeRemapper?.resources ?? []) resource.destroy();
    this.nativeRemapper = null; this.nativeMaps = null; this.nativeSourceMask = null;
    this.nativeSourceMaskRevision = null;
  }

  async ready() {
    if (this.failure) return false;
    if (!this.initializing) this.initializing = (async () => {
      if (!globalThis.navigator?.gpu) throw new Error('WebGPU nicht verfuegbar');
      const adapter = await requestSelectedGpuAdapter();
      if (!adapter) throw new Error('Kein WebGPU-Adapter');
      const storageLimit = Math.min(adapter.limits.maxStorageBufferBindingSize, 256 * 1024 * 1024);
      const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: storageLimit,
        maxTextureDimension2D: adapter.limits.maxTextureDimension2D,
        maxBufferSize: adapter.limits.maxBufferSize },
        requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : [] });
      this.device = device;
      this.emptyTexture = device.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING });
      device.lost.then(info => { if (this.device === device) this.disable(new Error(`WebGPU-Geraet verloren: ${info.message}`)); });
      device.addEventListener('uncapturederror', event => { if (this.device === device) this.disable(event.error); });
      const pyramidModule = device.createShaderModule({ code: pyramidShader });
      const registrationModule = device.createShaderModule({ code: registrationShader });
      for (const module of [pyramidModule, registrationModule]) {
        const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
        if (errors.length) throw new Error(errors.map(error => error.message).join('\n'));
      }
      this.pyramidPipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: pyramidModule, entryPoint: 'pyramid' } });
      const visibility = GPUShaderStage.COMPUTE;
      this.layout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility, buffer: { type: 'uniform' } },
        ...[1, 2].map(binding => ({ binding, visibility, buffer: { type: 'read-only-storage' } })),
        ...[3, 4].map(binding => ({ binding, visibility, buffer: { type: 'storage' } }))
      ] });
      const layout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
      this.measure = await device.createComputePipelineAsync({ layout, compute: { module: registrationModule, entryPoint: 'measure' } });
      this.aggregate = await device.createComputePipelineAsync({ layout, compute: { module: registrationModule, entryPoint: 'aggregate' } });
      return true;
    })().catch(error => { this.disable(error); return false; });
    return await this.initializing && !this.failure;
  }

  async checked(operation) {
    if (!await this.ready()) throw new Error(this.failure);
    const device = this.device;
    for (const filter of ['internal', 'out-of-memory', 'validation']) device.pushErrorScope(filter);
    try { return await operation(); }
    finally {
      const errors = [];
      for (let scope = 0; scope < 3; scope++) errors.push(await device.popErrorScope());
      const error = errors.find(Boolean);
      if (error) throw new Error(error.message);
      if (this.failure) throw new Error(this.failure);
    }
  }

  buffer(size, usage, data = null) {
    const bytes = Math.max(4, Math.ceil(size / 4) * 4);
    if (bytes > this.device.limits.maxBufferSize || ((usage & GPUBufferUsage.STORAGE) && bytes > this.device.limits.maxStorageBufferBindingSize)) {
      throw new Error(`Umfeldbild ueberschreitet WebGPU-Pufferlimit: ${(bytes / 1048576).toFixed(1)} MiB angefordert, ${(this.device.limits.maxStorageBufferBindingSize / 1048576).toFixed(1)} MiB Storage-Limit`);
    }
    const buffer = this.device.createBuffer({ size: bytes, usage, mappedAtCreation: Boolean(data) });
    if (data) { new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); buffer.unmap(); }
    return buffer;
  }

  retain(level, buffer) {
    const bytes = Math.ceil(level.gray.byteLength / 4) * 4;
    if (bytes > this.cacheLimit) throw new Error('Umfeldbild ueberschreitet GPU-Cachebudget');
    while (this.cacheBytes + bytes > this.cacheLimit) {
      const oldest = this.cache.keys().next().value; const entry = this.cache.get(oldest);
      entry.buffer.destroy(); this.cache.delete(oldest); this.cacheBytes -= entry.bytes;
    }
    const entry = { buffer, bytes };
    this.levels.set(level, entry); this.cache.set(entry, entry); this.cacheBytes += bytes;
  }

  async image(image, mask) {
    if (mask && (mask.sourceWidth !== image.width || mask.sourceHeight !== image.height)) throw new Error('Umfeldmaske passt nicht zum entzerrten Bild.');
    const setupStarted = performance.now();
    if (!await this.ready()) throw new Error(this.failure);
    const timing = { setupMs: performance.now() - setupStarted, uploadMs: 0, encodeMs: 0, waitMs: 0, readbackMs: 0,
      gpuComputeMs: 0, gpuTimedImages: 0, gpuImages: 1, cpuImages: 0, cpuMs: 0, failedGpuMs: 0, uploadBytes: 0, readbackBytes: 0,
      nativeImages: image.frame ? 1 : 0, nativeRemapSetupMs: 0, rgbaReadbackBytes: 0 };
    return this.checked(async () => {
      const temporary = [];
      const allocate = (size, usage, data) => { const buffer = this.buffer(size, usage, data); temporary.push(buffer); return buffer; };
      try {
        if (image.frame) {
          const remapStarted = performance.now();
          if (this.nativeMaps !== image.maps || this.nativeSourceMask !== image.sourceMask ||
              this.nativeSourceMaskRevision !== image.sourceMask?.revision) {
            this.releaseRemap();
            if (Math.max(image.width, image.height) > this.device.limits.maxTextureDimension2D) throw new Error('Umfeldbild ueberschreitet GPU-Texturlimit.');
            this.nativeRemapper = new WebGpuRemapper(this.device);
            await this.nativeRemapper.initializeRemap(image.maps, null, 0, image.brightness, image.sourceMask);
            this.nativeMaps = image.maps;
            this.nativeSourceMask = image.sourceMask;
            this.nativeSourceMaskRevision = image.sourceMask?.revision;
          }
          this.nativeRemapper.check();
          timing.nativeRemapSetupMs = performance.now() - remapStarted;
        }
        const uploadStarted = performance.now();
        const maskData = mask ? Uint32Array.from(mask.data) : new Uint32Array(1);
        const maskBuffer = allocate(maskData.byteLength, GPUBufferUsage.STORAGE, maskData);
        const sourceData = image.frame ? new Uint32Array(1) : image.data;
        let source = allocate(sourceData.byteLength, GPUBufferUsage.STORAGE, sourceData);
        timing.uploadMs = performance.now() - uploadStarted;
        timing.uploadBytes = sourceData.byteLength + maskData.byteLength;
        const encodeStarted = performance.now();
        const levels = []; const outputs = []; const copies = [];
        const queries = this.device.features.has('timestamp-query') ? this.device.createQuerySet({ type: 'timestamp', count: 12 }) : null;
        if (queries) temporary.push(queries);
        const queryBuffer = queries ? allocate(96, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC) : null;
        const queryReadback = queries ? allocate(96, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST) : null;
        const encoder = this.device.createCommandEncoder();
        if (image.frame) this.nativeRemapper.encodeRemap(encoder, image.frame, image.orientation);
        const texture = image.frame ? this.nativeRemapper.rectified : this.emptyTexture;
        const bytesPerRow = Math.ceil(image.width * 4 / 256) * 256;
        const rgbaReadback = image.readRgba ? allocate(bytesPerRow * image.height, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST) : null;
        if (rgbaReadback) encoder.copyTextureToBuffer({ texture }, { buffer: rgbaReadback, bytesPerRow }, [image.width, image.height]);
        let width = image.width; let height = image.height; let scale = 1;
        while (true) {
          const bytes = Math.ceil(width * height / 2) * 4;
          const output = allocate(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          const readback = allocate(bytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
          const previous = levels.at(-1);
          const params = new Uint32Array([previous?.width ?? width, previous?.height ?? height, width, height, previous ? 1 : image.frame ? 2 : 0, mask?.width ?? 1, mask?.cellSize ?? 1, mask ? 1 : 0]);
          const uniform = allocate(32, GPUBufferUsage.UNIFORM, params);
          const bindGroup = this.device.createBindGroup({ layout: this.pyramidPipeline.getBindGroupLayout(0), entries: [
            ...[uniform, source, maskBuffer, output].map((buffer, binding) => ({ binding, resource: { buffer } })),
            { binding: 4, resource: texture.createView() } ] });
          const pass = encoder.beginComputePass(queries ? { timestampWrites: { querySet: queries,
            beginningOfPassWriteIndex: levels.length * 2, endOfPassWriteIndex: levels.length * 2 + 1 } } : {});
          pass.setPipeline(this.pyramidPipeline); pass.setBindGroup(0, bindGroup);
          const groups = Math.ceil(width * height / 512); const columns = Math.min(groups, this.device.limits.maxComputeWorkgroupsPerDimension);
          pass.dispatchWorkgroups(columns, Math.ceil(groups / columns)); pass.end();
          encoder.copyBufferToBuffer(output, 0, readback, 0, bytes);
          levels.push({ width, height, scale }); outputs.push(output); copies.push(readback); source = output;
          if (levels.length >= 6 || Math.min(width, height) < 96) break;
          width = Math.floor(width / 2); height = Math.floor(height / 2); scale *= 2;
        }
        if (queries) { encoder.resolveQuerySet(queries, 0, levels.length * 2, queryBuffer, 0); encoder.copyBufferToBuffer(queryBuffer, 0, queryReadback, 0, levels.length * 16); }
        this.device.queue.submit([encoder.finish()]);
        timing.encodeMs = performance.now() - encodeStarted;
        for (let index = 0; index < levels.length; index++) {
          const level = levels[index]; const readback = copies[index];
          const waitStarted = performance.now();
          await readback.mapAsync(GPUMapMode.READ);
          timing.waitMs += performance.now() - waitStarted;
          const readStarted = performance.now();
          level.gray = new Int16Array(readback.getMappedRange(), 0, level.width * level.height).slice(); readback.unmap();
          timing.readbackMs += performance.now() - readStarted; timing.readbackBytes += level.gray.byteLength;
          if (this.retainImages) { this.retain(level, outputs[index]); temporary.splice(temporary.indexOf(outputs[index]), 1); }
        }
        if (queries) {
          const waitStarted = performance.now(); await queryReadback.mapAsync(GPUMapMode.READ); timing.waitMs += performance.now() - waitStarted;
          const stamps = new BigUint64Array(queryReadback.getMappedRange());
          for (let index = 0; index < levels.length; index++) timing.gpuComputeMs += Number(stamps[index * 2 + 1] - stamps[index * 2]) / 1e6;
          timing.gpuTimedImages = 1; queryReadback.unmap();
        }
        let rgba;
        if (rgbaReadback) {
          const waitStarted = performance.now(); await rgbaReadback.mapAsync(GPUMapMode.READ); timing.waitMs += performance.now() - waitStarted;
          const copyStarted = performance.now();
          const mapped = new Uint8Array(rgbaReadback.getMappedRange());
          const data = new Uint8ClampedArray(image.width * image.height * 4);
          for (let row = 0; row < image.height; row++) data.set(mapped.subarray(row * bytesPerRow, row * bytesPerRow + image.width * 4), row * image.width * 4);
          rgbaReadback.unmap(); timing.readbackMs += performance.now() - copyStarted;
          timing.rgbaReadbackBytes = data.byteLength;
          rgba = { width: image.width, height: image.height, data };
        }
        return { width: image.width, height: image.height, levels, bytes: levels.reduce((sum, level) => sum + level.gray.byteLength, 0), pyramidTiming: timing, rgba };
      } finally { for (const buffer of temporary) buffer.destroy(); }
    });
  }

  async execute(step) {
    if (step.type === 'normal') return step.best.normal;
    return this.checked(async () => {
      if (step.type === 'prepare') {
        if (!this.workspace) {
          this.workspace = [];
          for (const [size, usage] of [[64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST], [8192 * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST],
            [128 * 48, GPUBufferUsage.STORAGE], [96, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC], [96, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST],
            [64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST]]) this.workspace.push(this.buffer(size, usage));
        }
        if (step.samples.length > 8192) throw new Error('Zu viele GPU-Messpunkte');
        let entry = this.levels.get(step.target);
        if (entry && this.cache.has(entry)) { this.cache.delete(entry); this.cache.set(entry, entry); }
        else {
          const buffer = this.buffer(step.target.gray.byteLength, GPUBufferUsage.STORAGE, step.target.gray);
          try { this.retain(step.target, buffer); } catch (error) { buffer.destroy(); throw error; }
          entry = this.levels.get(step.target);
        }
        const samples = new Float32Array(step.samples.length * 4);
        step.samples.forEach((sample, index) => samples.set([sample.x, sample.y, sample.value, 0], index * 4));
        this.device.queue.writeBuffer(this.workspace[1], 0, samples);
        this.stage = step;
        this.bindGroup = this.device.createBindGroup({ layout: this.layout, entries: [this.workspace[0], entry.buffer, this.workspace[1], this.workspace[2], this.workspace[3]].map((buffer, binding) => ({ binding, resource: { buffer } })) });
        this.normalBindGroup = this.device.createBindGroup({ layout: this.layout, entries: [this.workspace[5], entry.buffer, this.workspace[1], this.workspace[2], this.workspace[3]].map((buffer, binding) => ({ binding, resource: { buffer } })) });
        return;
      }
      const { target, samples, reference } = this.stage;
      const params = new ArrayBuffer(64); const integers = new Uint32Array(params); const floats = new Float32Array(params);
      integers.set([samples.length, target.width, target.height]);
      floats.set([target.scale, step.pose.x, step.pose.y, Math.cos(step.pose.angle), Math.sin(step.pose.angle), 1, 0, this.stage.lever], 3);
      integers[11] = 0; floats[12] = reference.width / 2; floats[13] = reference.height / 2;
      this.device.queue.writeBuffer(this.workspace[0], 0, params);
      integers[11] = 1; this.device.queue.writeBuffer(this.workspace[5], 0, params);
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass(); pass.setBindGroup(0, this.bindGroup);
      pass.setPipeline(this.measure); pass.dispatchWorkgroups(Math.ceil(samples.length / 64));
      pass.setPipeline(this.aggregate); pass.dispatchWorkgroups(1);
      pass.setBindGroup(0, this.normalBindGroup); pass.setPipeline(this.measure); pass.dispatchWorkgroups(Math.ceil(samples.length / 64));
      pass.setPipeline(this.aggregate); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(this.workspace[3], 0, this.workspace[4], 0, 96);
      this.device.queue.submit([encoder.finish()]);
      const readback = this.workspace[4]; await readback.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(readback.getMappedRange()).slice(); readback.unmap();
      if (!values.every(Number.isFinite)) throw new Error('Ungueltige GPU-Messung');
      const [count, sumFirst, sumSecond, squareFirst, squareSecond, product] = values;
      if (count < 128 || count < samples.length * (step.minimumOverlapFraction ?? 0.2)) return null;
      const varianceFirst = squareFirst - sumFirst ** 2 / count; const varianceSecond = squareSecond - sumSecond ** 2 / count;
      if (Math.min(varianceFirst, varianceSecond) / count < 4) return null;
      const covariance = product - sumFirst * sumSecond / count; const gain = covariance / varianceFirst;
      return { count, score: covariance / Math.sqrt(varianceFirst * varianceSecond), gain, bias: (sumSecond - gain * sumFirst) / count + 128 * (1 - gain),
        normal: { normal: new Matrix([[values[12], values[13], values[14]], [values[13], values[15], values[16]], [values[14], values[16], values[17]]]), gradient: Matrix.columnVector(values.slice(18, 21)) } };
    });
  }
}
