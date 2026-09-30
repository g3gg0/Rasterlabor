import test, { beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { FrameReader } from '../src/frame-reader.js';
import { WebGpuRemapper } from '../src/webgpu-native-remapper.js';

function mockGlobal(context, name, value) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  context.after(() => previous ? Object.defineProperty(globalThis, name, previous) : delete globalThis[name]);
}

beforeEach(context => {
  mockGlobal(context, 'ImageData', class {
    constructor(data, width, height) { Object.assign(this, { data, width, height }); }
  });
  mockGlobal(context, 'createImageBitmap', async source => ({ width: source.width, height: source.height, close() {} }));
});

test('queued requests never post to a decoder replaced during a video change', async () => {
  let oldCalls=0;
  let decoder={async call(){oldCalls++;throw new Error('terminated');}};
  const reader=new FrameReader({getDecoder:()=>decoder,getMaps:()=>null});
  const old=reader.read(0);
  decoder={async call(){return {index:7};}};
  const current=reader.read(7);
  await assert.rejects(old,/gewechselt/);
  assert.equal((await current).index,7);assert.equal(oldCalls,0);
  await reader.dispose();
});

test('queued requests reject a brightness field replaced before decoding starts', async () => {
  let brightness = { gain: new Float32Array([1]) }, calls = 0;
  const decoder = { async call() { calls++; return { index: 0 }; } };
  const reader = new FrameReader({ getDecoder: () => decoder, getMaps: () => null, getBrightness: () => brightness });
  const pending = reader.read(0);
  brightness = { gain: new Float32Array([1.1]) };
  await assert.rejects(pending, /Helligkeitsfeld.*gewechselt/);
  assert.equal(calls, 0);
  await reader.dispose();
});

test('CPU calibration samples are rectified without applying the existing brightness field', async () => {
  const maps = { outputWidth: 1, outputHeight: 1 };
  const brightness = { gain: new Float32Array([2]) };
  const source = { width: 2, height: 1, data: new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255]) };
  const flags = [];
  const decoder = { async call() { return { bitmap: { close() {} } }; } };
  const computer = { async call(type, options) {
    assert.equal(type, 'remap');
    assert.equal(options.image, source);
    flags.push(options.brightness);
    return { width: 1, height: 1, data: new Uint8ClampedArray([40, 50, 60, 255]) };
  } };
  const reader = new FrameReader({ getDecoder: () => decoder, getMaps: () => maps, getBrightness: () => brightness, computer });
  reader.toRgba = () => source;
  const sample = await reader.read(0, { gpu: false, rectified: true, output: 'rgba', brightness: false });
  assert.equal(sample.width, maps.outputWidth);
  await reader.read(1, { gpu: false, rectified: true, output: 'rgba' });
  assert.deepEqual(flags, [false, true]);
  await reader.dispose();
});

test('GPU calibration samples disable gain while normal rectified reads use the active field', async context => {
  const maps = {}, brightness = { gain: new Float32Array([2]) }, fields = [];
  const decoder = { async call() { return { frame: { close() {} }, orientation: {} }; } };
  context.mock.method(WebGpuRemapper, 'create', async (receivedMaps, field) => {
    assert.equal(receivedMaps, maps);
    fields.push(field);
    return { async render() { return { width: 1, height: 1, data: new Uint8ClampedArray(4) }; }, destroy() {} };
  });
  const reader = new FrameReader({ getDecoder: () => decoder, getMaps: () => maps, getBrightness: () => brightness });
  await reader.read(0, { gpu: true, rectified: true, output: 'rgba', brightness: false });
  await reader.read(1, { gpu: true, rectified: true, output: 'rgba' });
  assert.deepEqual(fields, [null, brightness]);
  await reader.dispose();
});

test('frame readers serialize decoder access and recover after a rejected read', async () => {
  let active=0, maximum=0;
  const decoder={async call(type,{index}) {
    active++;maximum=Math.max(maximum,active);
    await new Promise(r=>setTimeout(r,2));active--;
    if(index===1)throw new Error('decode failure');
    return {index,bitmap:{close(){}}};
  }};
  const reader=new FrameReader({getDecoder:()=>decoder,getMaps:()=>null});
  const results=await Promise.allSettled([reader.read(0),reader.read(1),reader.read(2)]);
  assert.equal(maximum,1);assert.equal(results[1].status,'rejected');assert.equal(results[2].value.index,2);
  await reader.dispose();
});

test('native GPU reads own intermediate frames and rebuild resources after map or decoder changes', async t => {
  let maps={}, closed=0, destroyed=0, created=0;
  const native=()=>({frame:{close(){closed++;}},orientation:{},timestamp:42});
  let decoder={call:async type=>{assert.equal(type,'native-frame');return native();}};
  t.mock.method(WebGpuRemapper,'create',async()=>{
    created++;return {render:async()=>({data:new Uint8ClampedArray(4),width:1,height:1}),destroy(){destroyed++;}};
  });
  const reader=new FrameReader({getDecoder:()=>decoder,getMaps:()=>maps});
  const options={gpu:true,rectified:true,output:'rgba'};
  assert.equal((await reader.read(0,options)).timestamp,42);
  await reader.read(1,options);assert.equal(created,1);assert.equal(closed,2);
  maps={};await reader.read(2,options);assert.equal(created,2);assert.equal(destroyed,1);
  decoder={call:async()=>native()};await reader.read(3,options);assert.equal(created,3);
  const returned=await reader.read(4,{gpu:true,output:'native'});
  assert.equal(destroyed,3);assert.equal(closed,4);returned.frame.close();assert.equal(closed,5);
  await reader.dispose();
});

test('failed GPU rendering closes the native frame, falls back and avoids repeated device creation', async t => {
  let closed=0, destroyed=0, attempts=0, fallbacks=0;
  t.mock.method(WebGpuRemapper,'create',async()=>{
    attempts++;return {render:async()=>{throw new Error('device lost');},destroy(){destroyed++;}};
  });
  const decoder={call:async type=>type==='native-frame'?{frame:{close(){closed++;}}}:{bitmap:{close(){closed++;}}}};
  const maps={};
  const computer={async call(type,{image,useWebGpu}){assert.equal(type,'remap');assert.equal(useWebGpu,false);return image;}};
  const reader=new FrameReader({getDecoder:()=>decoder,getMaps:()=>maps,computer,onFallback:()=>fallbacks++});
  reader.toRgba=()=>({width:1,height:1,data:new Uint8ClampedArray([1,2,3,255])});
  for(let i=0;i<2;i++)assert.deepEqual((await reader.read(i,{gpu:true,rectified:true,output:'rgba'})).data,new Uint8ClampedArray([1,2,3,255]));
  assert.equal(attempts,1);assert.equal(fallbacks,1);assert.equal(destroyed,1);assert.equal(closed,3);
  await reader.dispose();
});

test('rectified brightness-corrected bitmaps are reused and invalidated with calibration state', async t => {
  let maps = {}, decoded = 0, cacheClosed = 0;
  const originalCreateImageBitmap = globalThis.createImageBitmap;
  globalThis.createImageBitmap = async source => ({ width: source.width, height: source.height,
    close() { cacheClosed++; } });
  t.after(() => {
    if (originalCreateImageBitmap) globalThis.createImageBitmap = originalCreateImageBitmap;
    else delete globalThis.createImageBitmap;
  });
  const decoder = { async call() {
    decoded++;
    return { frame: { close() {} }, orientation: {}, timestamp: decoded };
  } };
  t.mock.method(WebGpuRemapper, 'create', async () => ({
    async render() { return { bitmap: { width: 4, height: 3, close() {} }, width: 4, height: 3 }; },
    destroy() {}
  }));
  const reader = new FrameReader({ getDecoder: () => decoder, getMaps: () => maps });
  const options = { gpu: true, rectified: true, output: 'bitmap', measureSharpness: false };
  const first = await reader.read(7, options);
  const second = await reader.read(7, options);
  assert.equal(decoded, 1);
  assert.equal(second.frameTiming.cacheHit, true);
  assert.deepEqual(reader.cacheStats().frames, 1);
  first.bitmap.close(); second.bitmap.close();
  maps = {};
  await reader.read(7, options);
  assert.equal(decoded, 2);
  assert.ok(cacheClosed >= 1);
  await reader.dispose();
});

test('native GPU frames reuse cached clones across renderer releases without proactive eviction', async () => {
  let decoded = 0, closed = 0, maps = {};
  const native = () => ({ codedWidth: 4, codedHeight: 4, displayWidth: 4, displayHeight: 4,
    clone() { return native(); }, close() { closed++; } });
  const decoder = { async call() { decoded++; return { frame: native(), orientation: {}, sharpness: { score: 1 } }; } };
  const reader = new FrameReader({ getDecoder: () => decoder, getMaps: () => maps });
  reader.copyNativeFrame = frame => frame.clone();
  reader.frameCacheBudget = () => 128;
  const first = await reader.read(0, { gpu: true, output: 'native' }); first.frame.close();
  await reader.releaseRenderer();
  const second = await reader.read(0, { gpu: true, output: 'native' }); second.frame.close();
  assert.equal(decoded, 1);
  assert.equal(second.frameTiming.cacheHit, true);
  assert.equal(reader.cacheStats().frames, 1);
  reader.frameCacheBudget = () => 64;
  const third = await reader.read(1, { gpu: true, output: 'native' }); third.frame.close();
  assert.equal(reader.cacheStats().frames, 2);
  maps = {};
  const fourth = await reader.read(1, { gpu: true, output: 'native' }); fourth.frame.close();
  assert.equal(decoded, 2);
  await reader.dispose();
  assert.equal(reader.cacheStats().frames, 0);
  assert.equal(closed, 6);
});

test('PCB source mask and brightness changes preserve cached native overlay frames', async () => {
  let decoded = 0, brightness = null;
  const native = () => ({ codedWidth: 4, codedHeight: 4, clone() { return native(); }, close() {} });
  const decoder = { async call() { decoded++; return { frame: native(), orientation: {}, sharpness: { score: 1 } }; } };
  const reader = new FrameReader({ getDecoder: () => decoder, getMaps: () => null, getBrightness: () => brightness });
  reader.copyNativeFrame = frame => frame.clone();
  const mask = { revision: 1 };
  (await reader.read(8, { output: 'native', sourceMask: mask })).frame.close();
  mask.revision = 2;
  (await reader.read(9, { output: 'native', sourceMask: mask })).frame.close();
  brightness = { gain: [1] };
  const overlay = await reader.read(8, { output: 'native' });
  overlay.frame.close();
  assert.equal(overlay.frameTiming.cacheHit, true);
  assert.equal(decoded, 2);
  assert.equal(reader.cacheStats().frames, 2);
  await reader.dispose();
});

test('allocation failure releases the oldest ten percent of cached frames and retries once', async () => {
  let decoded = 0, closed = 0, fail = false;
  const allocationError = () => Object.assign(new Error('Allocation failed'), { name: 'OutOfMemoryError' });
  const native = () => ({ codedWidth: 4, codedHeight: 4,
    clone() { if (fail) { fail = false; throw allocationError(); } return native(); },
    close() { closed++; } });
  const decoder = { async call() { decoded++; return { frame: native(), orientation: {}, sharpness: { score: 1 } }; } };
  const reader = new FrameReader({ getDecoder: () => decoder, getMaps: () => null });
  reader.copyNativeFrame = frame => frame.clone();
  for (let frame = 0; frame < 20; frame++) (await reader.read(frame, { output: 'native' })).frame.close();
  assert.equal(reader.cacheStats().frames, 20);
  fail = true;
  const reused = await reader.read(19, { output: 'native' }); reused.frame.close();
  assert.equal(reused.frameTiming.cacheHit, true);
  assert.equal(decoded, 20);
  assert.equal(reader.cacheStats().frames, 18);
  assert.equal(reader.nativeCache.has(0), false);
  assert.equal(reader.nativeCache.has(1), false);
  assert.equal(closed, 2 + 21);
  await reader.dispose();
});

test('decoder allocation errors retry after eviction but other errors retain cached frames', async () => {
  let decoded = 0, fail = false;
  const native = () => ({ codedWidth: 2, codedHeight: 2, clone() { return native(); }, close() {} });
  const decoder = { async call() {
    decoded++;
    if (fail) { fail = false; throw Object.assign(new Error('Out of memory'), { name: 'OutOfMemoryError' }); }
    return { frame: native(), orientation: {}, sharpness: { score: 1 } };
  } };
  const reader = new FrameReader({ getDecoder: () => decoder, getMaps: () => null });
  reader.copyNativeFrame = frame => frame.clone();
  for (let frame = 0; frame < 10; frame++) (await reader.read(frame, { output: 'native' })).frame.close();
  assert.equal(reader.releaseOnAllocationError(new Error('Decode failed')), false);
  assert.equal(reader.cacheStats().frames, 10);
  fail = true;
  (await reader.read(10, { output: 'native' })).frame.close();
  assert.equal(decoded, 12);
  assert.equal(reader.nativeCache.has(0), false);
  assert.equal(reader.cacheStats().frames, 10);
  await reader.dispose();
});

test('native cache snapshots release decoder surfaces while many cached frames remain available', async context => {
  let liveDecoderSurfaces = 0, decoded = 0;
  mockGlobal(context, 'OffscreenCanvas', class {
    constructor(width, height) { Object.assign(this, { width, height }); }
    getContext() { return { drawImage: frame => { this.pixels = frame.timestamp; } }; }
  });
  mockGlobal(context, 'VideoFrame', class {
    constructor(canvas, options) {
      this.pixels = canvas.pixels;
      Object.assign(this, options, { codedWidth: canvas.width, codedHeight: canvas.height });
    }
    clone() { return { pixels: this.pixels, close() {} }; }
    close() {}
  });
  const decoder = { async call(type, { index }) {
    assert.equal(type, 'native-frame'); decoded++; liveDecoderSurfaces++;
    if (liveDecoderSurfaces > 3) throw new Error('Decoder surface pool exhausted');
    return { frame: { codedWidth: 4, codedHeight: 4, displayWidth: 4, displayHeight: 4,
      timestamp: index, duration: 33333,
      clone() { throw new Error('A cached clone would retain the decoder surface'); },
      close() { liveDecoderSurfaces--; } }, orientation: {}, sharpness: { score: 1 } };
  } };
  const reader = new FrameReader({ getDecoder: () => decoder, getMaps: () => null });
  for (let index = 0; index < 20; index++) (await reader.read(index, { output: 'native' })).frame.close();
  assert.equal(liveDecoderSurfaces, 0);
  assert.equal(reader.nativeCache.size, 20);
  for (let index = 19; index >= 0; index--) {
    const cached = await reader.read(index, { output: 'native' });
    assert.equal(cached.frame.pixels, index);
    assert.equal(cached.frameTiming.cacheHit, true);
    cached.frame.close();
  }
  assert.equal(decoded, 20);
  await reader.dispose();
});

test('native reads and temporary mask or brightness variants retain other rectified cache entries', async context => {
  let decoded = 0;
  const maps = {}, brightness = {}, mask = { revision: 1 };
  const decoder = { async call() { decoded++; return { frame: { close() {} }, orientation: {} }; } };
  context.mock.method(WebGpuRemapper, 'create', async () => ({
    async render(frame, orientation, output) {
      return output === 'bitmap' ? { bitmap: { width: 4, height: 3, close() {} } } :
        { width: 4, height: 3, data: new Uint8ClampedArray(48) };
    }, destroy() {}
  }));
  const reader = new FrameReader({ getDecoder: () => decoder, getMaps: () => maps, getBrightness: () => brightness });
  const options = { gpu: true, rectified: true, measureSharpness: false };
  (await reader.read(1, options)).bitmap.close();
  (await reader.read(2, options)).bitmap.close();
  (await reader.read(3, { output: 'native', sourceMask: mask, measureSharpness: false })).frame.close();
  (await reader.read(4, { ...options, sourceMask: mask })).bitmap.close();
  await reader.read(5, { ...options, output: 'rgba', brightness: false });
  mask.revision++;
  (await reader.read(4, { ...options, sourceMask: mask })).bitmap.close();
  const calls = decoded;
  for (const index of [1, 2]) {
    const cached = await reader.read(index, options);
    assert.equal(cached.frameTiming.cacheHit, true);
    cached.bitmap.close();
  }
  assert.equal(decoded, calls);
  await reader.dispose();
});

test('rectification reuses a native overlay frame after brightness or maps change', async context => {
  let decoded = 0, rendered = 0, maps = {}, brightness = {};
  const native = () => ({ codedWidth: 4, codedHeight: 4, clone() { return native(); }, close() {} });
  const decoder = { async call() { decoded++; return { frame: native(), orientation: {}, sharpness: { score: 1 } }; } };
  context.mock.method(WebGpuRemapper, 'create', async () => ({
    async render() { rendered++; return { bitmap: { width: 4, height: 4, close() {} } }; }, destroy() {}
  }));
  const reader = new FrameReader({ getDecoder: () => decoder, getMaps: () => maps, getBrightness: () => brightness });
  reader.copyNativeFrame = frame => frame.clone();
  (await reader.read(7, { output: 'native' })).frame.close();
  (await reader.read(7, { gpu: true, rectified: true })).bitmap.close();
  brightness = {}; maps = {};
  (await reader.read(7, { gpu: true, rectified: true })).bitmap.close();
  assert.equal(decoded, 1);
  assert.equal(rendered, 2);
  await reader.dispose();
});

test('CPU RGBA overlays reuse rectified pixels without decoding again', async context => {
  let decoded = 0, remapped = 0;
  const pixels = { width: 2, height: 2, data: new Uint8ClampedArray(16).fill(127) };
  const decoder = { async call() { decoded++; return { bitmap: { close() {} } }; } };
  const computer = { async call() { remapped++; return pixels; } };
  const maps = {};
  const reader = new FrameReader({ getDecoder: () => decoder, getMaps: () => maps, computer });
  reader.toRgba = () => pixels;
  const first = await reader.read(7, { rectified: true, output: 'rgba', measureSharpness: false });
  const second = await reader.read(7, { rectified: true, output: 'rgba', measureSharpness: false });
  assert.deepEqual(second.data, first.data);
  assert.equal(second.frameTiming.cacheHit, true);
  assert.equal(decoded, 1);
  assert.equal(remapped, 1);
  assert.equal(reader.frameCache.get(7).metadata.data, undefined);
  await reader.dispose();
});


test('streaming merge reads do not retain decoded video surfaces', async context => {
  const frames = [];
  const decoder = { async call(type, options) {
    assert.equal(type, 'native-frame');
    const frame = { codedWidth: 2160, codedHeight: 3840, close() {} }; frames.push(frame);
    return { frame, index: options.index };
  } };
  const reader = new FrameReader({ getDecoder: () => decoder, getMaps: () => null });
  reader.rememberNativeFrame = () => assert.fail('Streaming frames must not enter the native cache');
  for (let i = 0; i < 20; i++) {
    const result = await reader.read(i, { output: 'native', cache: false, measureSharpness: false });
    assert.equal(result.frame, frames[i]);result.frame.close();
  }
  assert.equal(reader.cacheStats().bytes, 0);await reader.dispose();
});
