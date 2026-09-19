import { edgeFeatherMask } from './path-support.js';

// Full-resolution remap and weighted accumulation. Pixel data stays on the GPU.
const vertex = /* wgsl */ `
@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(p[i], 0, 1);
}`;
const remapShader = vertex + /* wgsl */ `
struct Orientation { axes: vec4f, translationSize: vec4f }
@group(0) @binding(0) var source: texture_external;
@group(0) @binding(1) var nearestSampler: sampler;
@group(0) @binding(2) var map: texture_2d<f32>;
@group(0) @binding(3) var<uniform> orientation: Orientation;
@group(0) @binding(4) var edgeMask: texture_2d<f32>;
@group(0) @binding(5) var gainMap: texture_2d<f32>;
fn readSource(p: vec2f) -> vec3f {
  let v = p + vec2f(0.5) - orientation.translationSize.xy;
  let coded = vec2f(dot(orientation.axes.xy, v), dot(orientation.axes.zw, v));
  return textureSampleBaseClampToEdge(source, nearestSampler, coded / orientation.translationSize.zw).rgb;
}
fn toLinear(value: vec3f) -> vec3f {
  return select(value / 12.92, pow((value + 0.055) / 1.055, vec3f(2.4)), value > vec3f(0.04045));
}
fn toSrgb(value: vec3f) -> vec3f {
  return select(value * 12.92, 1.055 * pow(value, vec3f(1.0 / 2.4)) - 0.055, value > vec3f(0.0031308));
}
@fragment fn remap(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let p = textureLoad(map, vec2i(position.xy), 0).xy;
  if (p.x < 0 || p.y < 0) { return vec4f(0); }
  let q = floor(p); let f = fract(p);
  let rgb = mix(mix(readSource(q), readSource(q + vec2f(1, 0)), f.x),
    mix(readSource(q + vec2f(0, 1)), readSource(q + vec2f(1, 1)), f.x), f.y);
  let edge = textureLoad(edgeMask, vec2i(position.xy), 0).r;
  let gain = textureLoad(gainMap, vec2i(position.xy), 0).r;
  return vec4f(toSrgb(toLinear(rgb) * gain) * edge, edge);
}`;

// Shared native VideoFrame -> rectified full-resolution GPU surface.
export class WebGpuRemapper {
  static async create(maps, brightness = null) {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('Keine WebGPU-GPU verfuegbar.');
    if (Math.max(maps.outputWidth, maps.outputHeight) > adapter.limits.maxTextureDimension2D)
      throw new Error('Entzerrung ueberschreitet das GPU-Texturlimit.');
    const device = await adapter.requestDevice({ requiredLimits: {
      maxTextureDimension2D: adapter.limits.maxTextureDimension2D, maxBufferSize: adapter.limits.maxBufferSize } });
    const result = new WebGpuRemapper(device);
    try { await result.initializeRemap(maps, null, 0, brightness); return result; }
    catch (error) { result.destroy(); throw error; }
  }
  constructor(device) {
    this.device = device; this.resources = []; this.failure = null;
    device.addEventListener('uncapturederror', event => { this.failure = event.error; });
    device.lost.then(info => { this.failure = new Error(`GPU-Verbindung verloren: ${info.message}`); });
  }
  resource(value) { this.resources.push(value); return value; }
  buffer(size, usage) { return this.resource(this.device.createBuffer({ size, usage })); }
  check() { if (this.failure) throw new Error(this.failure.message); }
  async initializeRemap(maps, allowed = null, edgeFeather = 0, brightness = maps.brightness ?? null) {
    const d = this.device;
    this.width = maps.outputWidth; this.height = maps.outputHeight;
    const remapModule = d.createShaderModule({code: remapShader});
    const remapLayout = d.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, externalTexture: {} },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }
    ] });
    this.remapPipeline = await d.createRenderPipelineAsync({ layout: d.createPipelineLayout({ bindGroupLayouts: [remapLayout] }),
      vertex: { module: remapModule, entryPoint: 'vertex' }, fragment: { module: remapModule, entryPoint: 'remap', targets: [{ format: 'rgba8unorm' }] } });
    this.map = this.resource(d.createTexture({ size: [maps.outputWidth, maps.outputHeight], format: 'rg32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST }));
    for (let row = 0; row < maps.outputHeight; row += 128) {
      const rows = Math.min(128, maps.outputHeight - row);
      const values = new Float32Array(maps.outputWidth * rows * 2);
      for (let i = 0; i < values.length / 2; i++) {
        const index = row * maps.outputWidth + i;
        const valid = maps.valid[index] && (!allowed || allowed(i % maps.outputWidth, row + Math.floor(i / maps.outputWidth)));
        values[i * 2] = valid ? maps.inverseX[index] : -1; values[i * 2 + 1] = valid ? maps.inverseY[index] : -1;
      }
      d.queue.writeTexture({ texture: this.map, origin: [0, row] }, values, { bytesPerRow: maps.outputWidth * 8 }, [maps.outputWidth, rows]);
    }
    this.edgeMask = this.resource(d.createTexture({ size: [maps.outputWidth, maps.outputHeight], format: 'r8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST }));
    const weights = edgeFeatherMask(maps.outputWidth, maps.outputHeight, (x, y) => {
      const index = y * maps.outputWidth + x;
      return Boolean(maps.valid[index]) && (!allowed || allowed(x, y));
    }, edgeFeather);
    const bytesPerRow = Math.ceil(maps.outputWidth / 256) * 256;
    const upload = new Uint8Array(bytesPerRow * maps.outputHeight);
    for (let row = 0; row < maps.outputHeight; row++) upload.set(
      weights.subarray(row * maps.outputWidth, (row + 1) * maps.outputWidth), row * bytesPerRow);
    d.queue.writeTexture({ texture: this.edgeMask }, upload, { bytesPerRow }, [maps.outputWidth, maps.outputHeight]);
    this.gainMap = this.resource(d.createTexture({ size: [maps.outputWidth, maps.outputHeight], format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST }));
    const gain = brightness?.gain?.length === maps.outputWidth * maps.outputHeight ? brightness.gain :
      new Float32Array(maps.outputWidth * maps.outputHeight).fill(1);
    const gainBytesPerRow = Math.ceil(maps.outputWidth * 4 / 256) * 256;
    const gainUpload = new Float32Array(gainBytesPerRow / 4 * maps.outputHeight);
    for (let row = 0; row < maps.outputHeight; row++) gainUpload.set(
      gain.subarray(row * maps.outputWidth, (row + 1) * maps.outputWidth), row * gainBytesPerRow / 4);
    d.queue.writeTexture({ texture: this.gainMap }, gainUpload, { bytesPerRow: gainBytesPerRow }, [maps.outputWidth, maps.outputHeight]);
    this.rectified = this.resource(d.createTexture({ size: [maps.outputWidth, maps.outputHeight], format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC }));
    this.orientationBuffer = this.buffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.nearestSampler = d.createSampler({ minFilter: 'nearest', magFilter: 'nearest' });

    await d.queue.onSubmittedWorkDone(); this.check();
  }
  encodeRemap(command, frame, orientation) {
    const d = this.device;
    const o = orientation;
    d.queue.writeBuffer(this.orientationBuffer, 0, new Float32Array([o.a, o.b, o.c, o.d,
      o.translateX, o.translateY, frame.displayWidth, frame.displayHeight]));
    const external = d.importExternalTexture({ source: frame, colorSpace: 'srgb' });
    const bindGroup = d.createBindGroup({ layout: this.remapPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: external }, { binding: 1, resource: this.nearestSampler },
      { binding: 2, resource: this.map.createView() }, { binding: 3, resource: { buffer: this.orientationBuffer } },
      { binding: 4, resource: this.edgeMask.createView() }, { binding: 5, resource: this.gainMap.createView() }
    ] });
    const remap = command.beginRenderPass({ colorAttachments: [{ view: this.rectified.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }] });
    remap.setPipeline(this.remapPipeline); remap.setBindGroup(0, bindGroup); remap.draw(3); remap.end();

  }
  async render(frame, orientation, output = 'rgba') {
    this.check();
    const d = this.device, start = performance.now();
    const command = d.createCommandEncoder();
    this.encodeRemap(command, frame, orientation);
    const bytesPerRow = Math.ceil(this.width * 4 / 256) * 256;
    if (output === 'rgba') {
      this.readback ??= this.buffer(bytesPerRow * this.height, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      command.copyTextureToBuffer({texture: this.rectified}, {buffer: this.readback, bytesPerRow}, [this.width, this.height]);
    } else {
      if (!this.context) {
        this.canvas = new OffscreenCanvas(this.width, this.height);
        this.context = this.canvas.getContext('webgpu');
        this.context.configure({device: d, format: 'rgba8unorm', alphaMode: 'premultiplied',
          usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT});
      }
      command.copyTextureToTexture({texture: this.rectified}, {texture: this.context.getCurrentTexture()}, [this.width, this.height]);
    }
    d.queue.submit([command.finish()]);
    const submitted = performance.now();
    await d.queue.onSubmittedWorkDone(); this.check();
    const completed = performance.now();
    let result;
    if (output === 'rgba') {
      await this.readback.mapAsync(GPUMapMode.READ);
      try {
        const source = new Uint8Array(this.readback.getMappedRange());
        const data = new Uint8ClampedArray(this.width * this.height * 4);
        for (let row = 0; row < this.height; row++) data.set(source.subarray(row * bytesPerRow, row * bytesPerRow + this.width * 4), row * this.width * 4);
        result = {data};
      } finally { this.readback.unmap(); }
    } else result = {bitmap: this.canvas.transferToImageBitmap()};
    return {...result, width: this.width, height: this.height, accelerator: 'WebGPU native VideoFrame',
      processingMs: performance.now() - start, timing: {contextMs: 0, uploadMs: 0, commandMs: submitted-start,
        completionMs: completed-submitted, readbackMs: performance.now()-completed}};
  }
  destroy() { for (const resource of this.resources) resource.destroy(); this.resources = []; this.context?.unconfigure(); this.device.destroy(); }
}
