import { WebGpuRemapper } from './webgpu-native-remapper.js';
import { requestSelectedGpuAdapter } from './webgpu-selection.js';
const vertex = /* wgsl */ `
@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(p[i], 0, 1);
}`;
const accumulateShader = /* wgsl */ `
struct Params { sizeOffset: vec4u, rotationOrigin: vec4f, worldOrigin: vec4f, sourceScale: vec4f }
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> sum: array<vec4f>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> counts: array<u32>;
@compute @workgroup_size(16, 16)
fn accumulate(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.sizeOffset.x || id.y >= params.sizeOffset.y) { return; }
  let index = id.y * params.sizeOffset.x + id.x;
  let operation = u32(params.worldOrigin.z);
  if (operation == 0u && params.sizeOffset.w > 0u && sum[index].a >= f32(params.sizeOffset.w)) { return; }
  if (operation == 4u && params.sizeOffset.w > 0u && counts[index] >= params.sizeOffset.w && sum[index].a >= 0.999) { return; }
  if (operation == 5u && params.sizeOffset.w > 0u && sum[index].a >= f32(params.sizeOffset.w)) { return; }
  let world = vec2f(f32(id.x) + 0.5, f32(id.y + params.sizeOffset.z) + 0.5) * params.worldOrigin.w + params.worldOrigin.xy;
  let v = world - params.rotationOrigin.zw;
  let c = params.rotationOrigin.x; let s = params.rotationOrigin.y;
  let local = vec2f(c * v.x + s * v.y, -s * v.x + c * v.y) * params.sourceScale.xy;
  let size = vec2f(textureDimensions(source));
  // Transparent outside the image, including bilinear footprints at its edges.
  let q = local - vec2f(0.5); let base = vec2i(floor(q)); let f = fract(q);
  var value = vec4f(0);
  for (var y = 0; y < 2; y++) { for (var x = 0; x < 2; x++) {
    let p = base + vec2i(x, y);
    if (all(p >= vec2i(0)) && all(p < vec2i(size))) {
      let weight = select(1 - f.x, f.x, x == 1) * select(1 - f.y, f.y, y == 1);
      value += textureLoad(source, p, 0) * weight;
    }
  }}
  if (value.a > 0.0) {
    if (operation == 1u) { counts[index] += 1u; return; }
    if (operation >= 2u && operation <= 3u && params.sizeOffset.w > 0u) {
      let remaining = counts[index];
      if (remaining > 0u) { counts[index] = remaining - 1u; }
      if (remaining > params.sizeOffset.w) { return; }
    }
    if (operation == 3u) {
      sum[index] = vec4f(value.rgb + sum[index].rgb * (1.0 - value.a), value.a + sum[index].a * (1.0 - value.a));
    } else if (operation == 4u) {
      sum[index] = vec4f(sum[index].rgb + value.rgb * (1.0 - sum[index].a), sum[index].a + value.a * (1.0 - sum[index].a));
      counts[index] += 1u;
    } else if (operation == 5u) {
      if (params.sizeOffset.w > 0u) { value *= min(1.0, (f32(params.sizeOffset.w) - sum[index].a) / value.a); }
      sum[index] += value;
      counts[index] += 1u;
    } else {
      if (operation == 0u && params.sizeOffset.w > 0u) { value *= min(1.0, (f32(params.sizeOffset.w) - sum[index].a) / value.a); }
      sum[index] += value;
    }
    if (operation == 0u) { counts[index] += 1u; }
  }
}`;
const finishShader = vertex + /* wgsl */ `
@group(0) @binding(0) var<storage, read> sum: array<vec4f>;
@group(0) @binding(1) var<uniform> dimensions: vec4u;
@fragment fn finish(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let index = (u32(p.y) - dimensions.z) * dimensions.x + u32(p.x);
  let value = sum[index];
  if (value.a <= 0) { return vec4f(0); }
  let coverage = min(1.0, value.a);
  return vec4f(value.rgb / value.a * coverage, coverage);
}`;

export class WebGpuOverlay extends WebGpuRemapper {
  static async needsTiles(maps, width, height) {
    const adapter = await requestSelectedGpuAdapter();
    if (!adapter) throw new Error('Keine WebGPU-GPU verfuegbar.');
    return Math.max(width, height) > adapter.limits.maxTextureDimension2D ||
        width * height * 24 + maps.outputWidth * maps.outputHeight * 17 > 1536 * 1024 * 1024;
  }
      static async create(maps, width, height, minX, minY, pixelAllowed = null, maxFrames = 0, edgeFeather = 0.1, brightness = null) {
    if (!Number.isSafeInteger(maxFrames) || maxFrames < 0 || maxFrames > 65535) throw new Error('Ungueltige Framegrenze fuer die Ueberlagerung.');
    if (!navigator.gpu) throw new Error('WebGPU ist nicht verfuegbar.');
    const adapter = await requestSelectedGpuAdapter();
    if (!adapter) throw new Error('Keine WebGPU-GPU verfuegbar.');
    const maximum = adapter.limits.maxTextureDimension2D;
    if (Math.max(width, height, maps.outputWidth, maps.outputHeight) > maximum) {
      throw new Error(`Vollaufloesung ${width} x ${height} ueberschreitet das GPU-Texturlimit ${maximum}. Kleineren Bildbereich auswaehlen.`);
    }
    // Bound allocations before creating resources; never silently downscale.
    if (width * height * 24 + maps.outputWidth * maps.outputHeight * 17 > 1536 * 1024 * 1024) {
      throw new Error('Vollaufloesende Ueberlagerung ueberschreitet das GPU-Bildbudget von 1,5 GiB. Kleineren Bildbereich auswaehlen.');
    }
    const device = await adapter.requestDevice({ requiredLimits: { maxTextureDimension2D: maximum,
      maxBufferSize: adapter.limits.maxBufferSize } });
    const result = new WebGpuOverlay(device, width, height, minX, minY);
    result.maxFrames = maxFrames;
    result.edgeFeather = Math.max(0, Math.min(0.5, edgeFeather));
    try { await result.initialize(maps, pixelAllowed, brightness); return result; }
    catch (error) { result.destroy(); throw new Error(error.message || String(error)); }
  }

  constructor(device, width, height, minX, minY) {
    super(device);
    Object.assign(this, { width, height, minX, minY, chunks: [], frameTextures: new Map(), cacheBytes: 0, cacheBudget: 512 * 1024 * 1024, pixelScale: 1, cacheEnabled: false, remapCount: 0, cacheHits: 0, previewBytes: 0 });
  }

  async initialize(maps, allowed, brightness) {
    const d = this.device;
    const module = code => d.createShaderModule({ code });
    const finishModule = module(finishShader);
    const {width, height} = this;
    await this.initializeRemap(maps, allowed, this.edgeFeather, brightness);
    this.width = width; this.height = height;
    this.accumulatePipeline = await d.createComputePipelineAsync({ layout: 'auto', compute: { module: module(accumulateShader), entryPoint: 'accumulate' } });
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.finishPipeline = await d.createRenderPipelineAsync({ layout: 'auto', vertex: { module: finishModule, entryPoint: 'vertex' },
      fragment: { module: finishModule, entryPoint: 'finish', targets: [{ format: this.format }] } });
    this.sourceWidth = maps.outputWidth; this.sourceHeight = maps.outputHeight;
    this.canvas = new OffscreenCanvas(this.width, this.height);
    this.context = this.canvas.getContext('webgpu');
    this.allocateOutput();
    await d.queue.onSubmittedWorkDone(); this.check();
  }

  allocateOutput() {
    const d = this.device;
    const rowsPerChunk = Math.min(512, Math.floor(d.limits.maxStorageBufferBindingSize / (this.width * 16)));
    for (let row = 0; row < this.height; row += rowsPerChunk) {
      const rows = Math.min(rowsPerChunk, this.height - row);
      const sum = this.buffer(this.width * rows * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const counts = this.buffer(this.width * rows * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const params = this.buffer(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      const dimensions = this.buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      d.queue.writeBuffer(dimensions, 0, new Uint32Array([this.width, rows, row, 0]));
      const accumulate = d.createBindGroup({ layout: this.accumulatePipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: this.rectified.createView() }, { binding: 2, resource: { buffer: sum } }, { binding: 3, resource: { buffer: params } },
        { binding: 4, resource: { buffer: counts } }
      ] });
      const finish = d.createBindGroup({ layout: this.finishPipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: sum } }, { binding: 1, resource: { buffer: dimensions } }
      ] });
      this.chunks.push({ row, rows, sum, counts, params, dimensions, accumulate, finish });
    }
    this.canvas.width = this.width; this.canvas.height = this.height;
    this.context.configure({ device: d, format: this.format, alphaMode: 'premultiplied', colorSpace: 'srgb' });
  }

  check() { if (this.failure) throw new Error(this.failure.message); }

  async clear(minX, minY) {
    this.check();
    this.minX = minX; this.minY = minY;
    const command = this.device.createCommandEncoder();
    for (const chunk of this.chunks) { command.clearBuffer(chunk.sum); command.clearBuffer(chunk.counts); }
    this.device.queue.submit([command.finish()]);
    await this.device.queue.onSubmittedWorkDone(); this.check();
  }

  async resizeOutput(width, height, pixelScale = 1) {
    this.pixelScale = pixelScale;
    if (width === this.width && height === this.height) return;
    await this.device.queue.onSubmittedWorkDone();
    const obsolete = new Set(this.chunks.flatMap(c => [c.sum, c.counts, c.params, c.dimensions]));
    for (const resource of obsolete) resource.destroy();
    this.resources = this.resources.filter(r => !obsolete.has(r));
    this.chunks = []; this.width = width; this.height = height;
    this.allocateOutput();
  }

  setActiveFrames(frames) {
    const active = new Set(frames);
    for (const [id, entry] of this.frameTextures) if (!active.has(id)) {
      for (const texture of [entry.full, entry.preview]) texture?.destroy();
      this.cacheBytes -= entry.bytes; this.previewBytes -= entry.previewBytes;
      this.frameTextures.delete(id);
    }
  }

  accumulateTexture(command, texture, geometry, operation) {
    const d = this.device, origin = geometry.world(0, 0);
    const pass = command.beginComputePass(); pass.setPipeline(this.accumulatePipeline);
    for (const chunk of this.chunks) {
      const params = new ArrayBuffer(64);
      new Uint32Array(params).set([this.width, chunk.rows, chunk.row, this.maxFrames]);
      new Float32Array(params).set([geometry.c, geometry.s, origin.x, origin.y,
        this.minX, this.minY, operation, this.pixelScale,
        texture.width / this.sourceWidth, texture.height / this.sourceHeight, 0, 0], 4);
      d.queue.writeBuffer(chunk.params, 0, params);
      const bind = texture === this.rectified ? chunk.accumulate : d.createBindGroup({
        layout: this.accumulatePipeline.getBindGroupLayout(0), entries: [
          { binding: 0, resource: texture.createView() }, { binding: 2, resource: { buffer: chunk.sum } },
          { binding: 3, resource: { buffer: chunk.params } }, { binding: 4, resource: { buffer: chunk.counts } }] });
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(this.width / 16), Math.ceil(chunk.rows / 16));
    }
    pass.end();
  }

  async addCachedFrame(geometry, operation = 0, preview = false) {
    const entry = this.frameTextures.get(geometry.entry.frame);
    const texture = preview ? entry?.preview : entry?.full;
    if (!texture) return false;
    this.check(); this.cacheHits++;
    const command = this.device.createCommandEncoder();
    this.accumulateTexture(command, texture, geometry, operation);
    this.device.queue.submit([command.finish()]);
    // Queue writes and submissions stay ordered; finish() fences the entire batch.
    return true;
  }

  retainFrame(command, id) {
    if (!this.cacheEnabled) return;
    const existing = this.frameTextures.get(id);
    if (existing) {
      const bytes = this.sourceWidth * this.sourceHeight * 4;
      if (!existing.full && this.cacheBytes + bytes <= this.cacheBudget) {
        existing.full = this.device.createTexture({size:[this.sourceWidth,this.sourceHeight],format:'rgba8unorm',
          usage:GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING});
        command.copyTextureToTexture({texture:this.rectified},{texture:existing.full},[this.sourceWidth,this.sourceHeight]);
        existing.bytes = bytes; this.cacheBytes += bytes;
      }
      return;
    }
    const d = this.device, width = this.sourceWidth, height = this.sourceHeight;
    const scale = Math.min(1, 512 / Math.max(width, height));
    const pw = Math.max(1, Math.round(width * scale)), ph = Math.max(1, Math.round(height * scale));
    if (this.previewBytes + pw * ph * 4 > 128 * 1024 * 1024) return;
    const preview = d.createTexture({ size: [pw, ph], format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    if (!this.previewPipeline) {
      const module = d.createShaderModule({code: vertex + `
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var imageSampler: sampler;
@group(0) @binding(2) var<uniform> size: vec4f;
@fragment fn preview(@builtin(position) p: vec4f) -> @location(0) vec4f {
  return textureSample(image, imageSampler, p.xy / size.xy);
}`});
      this.previewPipeline = d.createRenderPipeline({layout:'auto', vertex:{module,entryPoint:'vertex'},
        fragment:{module,entryPoint:'preview',targets:[{format:'rgba8unorm'}]}});
      this.previewSampler = d.createSampler({minFilter:'linear',magFilter:'linear'});
      this.previewSize = this.buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    }
    d.queue.writeBuffer(this.previewSize, 0, new Float32Array([pw,ph,0,0]));
    const pass = command.beginRenderPass({colorAttachments:[{view:preview.createView(),loadOp:'clear',storeOp:'store'}]});
    pass.setPipeline(this.previewPipeline);
    pass.setBindGroup(0,d.createBindGroup({layout:this.previewPipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:this.rectified.createView()},{binding:1,resource:this.previewSampler},
      {binding:2,resource:{buffer:this.previewSize}}]}));
    pass.draw(3); pass.end();
    let full = null;
    const bytes = width * height * 4;
    if (this.cacheBytes + bytes <= this.cacheBudget) {
      full = d.createTexture({size:[width,height],format:'rgba8unorm',usage:GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING});
      command.copyTextureToTexture({texture:this.rectified},{texture:full},[width,height]);
      this.cacheBytes += bytes;
    }
    this.previewBytes += pw * ph * 4;
    this.frameTextures.set(id,{full,preview,bytes:full ? bytes : 0,previewBytes:pw * ph * 4});
  }

  async addFrame(frame, orientation, geometry, operation = 0) {
    this.check();
    const d = this.device;
    const command = d.createCommandEncoder();
    this.remapCount++;
    this.encodeRemap(command, frame, orientation);
    this.retainFrame(command, geometry.entry.frame);
    this.accumulateTexture(command, this.rectified, geometry, operation);
    d.queue.submit([command.finish()]);
    // One fence per frame, no GPU readback. Keeps decoded surfaces and queue bounded.
    await d.queue.onSubmittedWorkDone(); this.check();
  }

  async countFrame(frame, orientation, geometry) { return this.addFrame(frame, orientation, geometry, 1); }

  async finish() {
    this.check();
    const command = this.device.createCommandEncoder();
    const pass = command.beginRenderPass({ colorAttachments: [{ view: this.context.getCurrentTexture().createView(),
      loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }] });
    pass.setPipeline(this.finishPipeline);
    for (const chunk of this.chunks) {
      pass.setScissorRect(0, chunk.row, this.width, chunk.rows); pass.setBindGroup(0, chunk.finish); pass.draw(3);
    }
    pass.end(); this.device.queue.submit([command.finish()]);
    await this.device.queue.onSubmittedWorkDone(); this.check();
    return this.canvas.transferToImageBitmap();
  }

  destroy() { this.setActiveFrames([]); for (const r of this.resources) r.destroy(); this.resources = []; this.context?.unconfigure(); this.device.destroy(); }
}
