const vertex = /* wgsl */ `
@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(p[i], 0, 1);
}`;
const sampleShader = vertex + /* wgsl */ `
struct Size { value: vec2f, padding: vec2f }
@group(0) @binding(0) var source: texture_external;
@group(0) @binding(1) var sourceSampler: sampler;
@group(0) @binding(2) var<uniform> size: Size;
@fragment fn sample(@builtin(position) p: vec4f) -> @location(0) vec4f {
  return textureSampleBaseClampToEdge(source, sourceSampler, p.xy / size.value);
}`;
const measureShader = /* wgsl */ `
struct Params { imageSize: vec2u, origin: vec2u, regionSize: vec2u, groupsX: u32, partialOffset: u32 }
struct Partial { moments: vec4f, count: u32, padding0: u32, padding1: u32, padding2: u32 }
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> partials: array<Partial>;
var<workgroup> values: array<vec4f, 256>;
var<workgroup> counts: array<u32, 256>;
fn luma(p: vec2u) -> f32 { return dot(textureLoad(source, vec2i(p), 0).rgb, vec3f(0.2126, 0.7152, 0.0722)) * 255.0; }
@compute @workgroup_size(16, 16)
fn measure(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) local: u32,
  @builtin(workgroup_id) group: vec3u) {
  let p = params.origin + id.xy;
  let inside = all(id.xy > vec2u(0)) && all(id.xy + vec2u(1) < params.regionSize);
  var moments = vec4f(0); var count = 0u;
  if (inside) {
    let center = luma(p);
    let laplacian = 4.0 * center - luma(p - vec2u(1, 0)) - luma(p + vec2u(1, 0)) -
      luma(p - vec2u(0, 1)) - luma(p + vec2u(0, 1));
    moments = vec4f(laplacian, laplacian * laplacian, center, center * center); count = 1u;
  }
  values[local] = moments; counts[local] = count; workgroupBarrier();
  var stride = 128u;
  loop {
    if (local < stride) { values[local] += values[local + stride]; counts[local] += counts[local + stride]; }
    workgroupBarrier(); if (stride == 1u) { break; } stride /= 2u;
  }
  if (local == 0u) {
    let index = params.partialOffset + group.y * params.groupsX + group.x;
    partials[index].moments = values[0]; partials[index].count = counts[0];
  }
}`;

import { onWebGpuSelectionChange, requestSelectedGpuAdapter } from './webgpu-selection.js';

let contextPromise = null;
onWebGpuSelectionChange(() => {
  const previous = contextPromise; contextPromise = null;
  void previous?.then(value => value?.device?.destroy()).catch(() => {});
});
async function context() {
  if (!contextPromise) contextPromise = (async () => {
    const adapter = await requestSelectedGpuAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    const sampleModule = device.createShaderModule({code: sampleShader});
    const measureModule = device.createShaderModule({code: measureShader});
    const errors = (await Promise.all([sampleModule.getCompilationInfo(), measureModule.getCompilationInfo()]))
      .flatMap(info => info.messages).filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(error => error.message).join('\n'));
    const sampleLayout = device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.FRAGMENT,externalTexture:{}},
      {binding:1,visibility:GPUShaderStage.FRAGMENT,sampler:{}},
      {binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:'uniform'}}]});
    return {device, sampleLayout,
      samplePipeline:device.createRenderPipeline({layout:device.createPipelineLayout({bindGroupLayouts:[sampleLayout]}),
        vertex:{module:sampleModule,entryPoint:'vertex'},fragment:{module:sampleModule,entryPoint:'sample',targets:[{format:'rgba8unorm'}]}}),
      measurePipeline:device.createComputePipeline({layout:'auto',compute:{module:measureModule,entryPoint:'measure'}}),
      sampler:device.createSampler({minFilter:'linear',magFilter:'linear'})};
  })().catch(() => null);
  return contextPromise;
}

function dimensions(frame) {
  const scale=Math.min(1,1024/Math.max(frame.displayWidth,frame.displayHeight));
  return {width:Math.max(9,Math.floor(frame.displayWidth*scale/3)*3),
    height:Math.max(9,Math.floor(frame.displayHeight*scale/3)*3)};
}

export async function measureFrameSharpnessGpu(frame) {
  const gpu=await context(); if(!gpu)return null;
  const {device}=gpu,{width,height}=dimensions(frame);
  if(Math.max(width,height)>device.limits.maxTextureDimension2D)return null;
  const texture=device.createTexture({size:[width,height],format:'rgba8unorm',
    usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING});
  const size=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(size,0,new Float32Array([width,height,0,0]));
  const regions=[];let partialCount=0;
  for(let row=0;row<3;row++)for(let col=0;col<3;col++){
    const x=Math.floor(col*width/3),y=Math.floor(row*height/3);
    const regionWidth=Math.floor((col+1)*width/3)-x,regionHeight=Math.floor((row+1)*height/3)-y;
    const groupsX=Math.ceil(regionWidth/16),groupsY=Math.ceil(regionHeight/16);
    regions.push({x,y,width:regionWidth,height:regionHeight,groupsX,groupsY,offset:partialCount});
    partialCount+=groupsX*groupsY;
  }
  const partials=device.createBuffer({size:partialCount*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const readback=device.createBuffer({size:partialCount*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const uniforms=[],bindings=[];
  try {
    for(const region of regions){
      const uniform=device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});uniforms.push(uniform);
      device.queue.writeBuffer(uniform,0,new Uint32Array([width,height,region.x,region.y,region.width,region.height,region.groupsX,region.offset]));
      bindings.push(device.createBindGroup({layout:gpu.measurePipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:texture.createView()},{binding:1,resource:{buffer:uniform}},{binding:2,resource:{buffer:partials}}]}));
    }
    const sample=device.createBindGroup({layout:gpu.sampleLayout,entries:[
      {binding:0,resource:device.importExternalTexture({source:frame,colorSpace:'srgb'})},
      {binding:1,resource:gpu.sampler},{binding:2,resource:{buffer:size}}]});
    const encoder=device.createCommandEncoder();
    const render=encoder.beginRenderPass({colorAttachments:[{view:texture.createView(),loadOp:'clear',storeOp:'store',clearValue:[0,0,0,1]}]});
    render.setPipeline(gpu.samplePipeline);render.setBindGroup(0,sample);render.draw(3);render.end();
    const compute=encoder.beginComputePass();compute.setPipeline(gpu.measurePipeline);
    regions.forEach((region,index)=>{compute.setBindGroup(0,bindings[index]);compute.dispatchWorkgroups(region.groupsX,region.groupsY);});
    compute.end();encoder.copyBufferToBuffer(partials,0,readback,0,partialCount*32);device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const view=new DataView(readback.getMappedRange()),tiles=[];
    for(const region of regions){let count=0,sum=0,squared=0,luminance=0,luminanceSquared=0;
      for(let i=0;i<region.groupsX*region.groupsY;i++){
        const offset=(region.offset+i)*32;sum+=view.getFloat32(offset,true);squared+=view.getFloat32(offset+4,true);
        luminance+=view.getFloat32(offset+8,true);luminanceSquared+=view.getFloat32(offset+12,true);count+=view.getUint32(offset+16,true);
      }
      tiles.push({variance:Math.max(0,squared/count-(sum/count)**2),
        contrast:Math.sqrt(Math.max(0,luminanceSquared/count-(luminance/count)**2)),samples:count});
    }
    const variances=tiles.map(x=>x.variance).sort((a,b)=>a-b),contrasts=tiles.map(x=>x.contrast).sort((a,b)=>a-b);
    return {score:variances[4],mean:variances.reduce((a,b)=>a+b,0)/9,minimum:variances[0],maximum:variances[8],
      contrast:contrasts[4],samples:tiles.reduce((a,b)=>a+b.samples,0),sampleWidth:width,sampleHeight:height,
      method:'median-laplacian-variance-9-downsampled-regions',accelerator:'WebGPU'};
  } finally {
    if(readback.mapState==='mapped')readback.unmap();
    for(const uniform of uniforms)uniform.destroy();size.destroy();partials.destroy();readback.destroy();texture.destroy();
  }
}
