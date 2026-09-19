import test from 'node:test';
import assert from 'node:assert/strict';
import { FrameReader } from '../src/frame-reader.js';
import { WebGpuRemapper } from '../src/webgpu-native-remapper.js';

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
