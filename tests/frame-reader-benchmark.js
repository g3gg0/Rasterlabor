import { FrameReader } from '../src/frame-reader.js';
import { WorkerClient } from '../src/rpc.js';

const assert = (value, message) => { if (!value) throw new Error(message); };
export async function benchmarkFrameReader(decoder, computer, calibration) {
  const reader = new FrameReader({getDecoder: () => decoder, getMaps: () => calibration.maps, computer,
    onFallback: error => { throw error; }});
  const {outputWidth: width, outputHeight: height, valid} = calibration.maps;
  const rectangle = {x: Math.floor(width/2)-512, y: Math.floor(height/2)-512, width: 1024, height: 1024};
  for(let y=rectangle.y;y<rectangle.y+rectangle.height;y++) for(let x=rectangle.x;x<rectangle.x+rectangle.width;x++)
    assert(valid[y*width+x], 'Benchmark window must be entirely calibrated');
  const canvas = new OffscreenCanvas(1,1), context = canvas.getContext('2d', {willReadFrequently: true});
  async function legacy(index) {
    const start=performance.now(), decoded=await decoder.call('frame',{index}), at=performance.now();
    canvas.width=decoded.bitmap.width; canvas.height=decoded.bitmap.height;
    context.drawImage(decoded.bitmap,0,0);decoded.bitmap.close();
    const image=context.getImageData(0,0,canvas.width,canvas.height), rgbaAt=performance.now();
    const result=await computer.call('remap',{image,useWebGpu:true},[image.data.buffer]);
    return {...result,frameTiming:{decodeMs:at-start,rgbaMs:rgbaAt-at,remapMs:performance.now()-rgbaAt}};
  }
  const result={output:[width,height],rectangle,runs:[]};
  try {
    const before = await legacy(970);
    const after = await reader.read(970,{gpu:true,rectified:true,output:'rgba'});
    let maximum=0, alphaDifferences=0, count=0, sum=0;
    for(let i=0;i<before.data.length;i+=4*701) {
      if(before.data[i+3]!==after.data[i+3])alphaDifferences++;
      if(!before.data[i+3])continue;
      for(let k=0;k<3;k++){const d=Math.abs(before.data[i+k]-after.data[i+k]);maximum=Math.max(maximum,d);sum+=d;count++;}
    }
    result.parity={maximum,alphaDifferences,mean:sum/count,channels:count};
    assert(maximum<=1&&!alphaDifferences,JSON.stringify(result.parity));
    const bitmapResult=await reader.read(970,{gpu:true,rectified:true});
    assert(bitmapResult.bitmap.width===width&&bitmapResult.bitmap.height===height,'Bitmap resolution');
    const bitmapPixels=reader.toRgba(bitmapResult.bitmap);bitmapResult.bitmap.close();
    for(let i=0;i<after.data.length;i+=4*701)for(let k=0;k<4;k++)
      assert(Math.abs(bitmapPixels.data[i+k]-after.data[i+k])<=1,'GPU bitmap/readback mismatch');
    result.bitmapParity=true;
    for(let run=0;run<3;run++) for(const mode of ['before','after']) {
      const tracker=new WorkerClient('/compute-worker.js');
      const samples=[];
      const warm=await decoder.call('frame',{index:969});warm.bitmap.close();
      try {
        for(let index=970;index<982;index++) {
          const start=performance.now();
          const image=mode==='before'?await legacy(index):await reader.read(index,{gpu:true,rectified:true,output:'rgba'});
          const readMs=performance.now()-start;
          const detection=await tracker.call('track-window',{image,index,options:{rectangle,patchSearchRadius:128,maxRotation:5}},[image.data.buffer]);
          assert(detection.success,`${mode} frame ${index}: ${detection.reason}`);
          samples.push({index,...image.frameTiming,readMs,totalMs:performance.now()-start,pose:detection.raw});
        }
      } finally {tracker.terminate();}
      result.runs.push({run,mode,meanMs:samples.reduce((s,v)=>s+v.totalMs,0)/samples.length,
        meanReadMs:samples.reduce((s,v)=>s+v.readMs,0)/samples.length,samples});
    }
    // Explicit GPU-off request exercises the actual CPU fallback after a GPU read.
    const cpu=await reader.read(970,{gpu:false,rectified:true,output:'rgba'});
    assert(cpu.width===width&&cpu.height===height&&cpu.data.length===width*height*4,'CPU fallback resolution');
    let cpuMaximum=0;
    for(let i=0;i<cpu.data.length;i+=4*701) for(let k=0;k<4;k++)cpuMaximum=Math.max(cpuMaximum,Math.abs(cpu.data[i+k]-after.data[i+k]));
    assert(cpuMaximum<=1,`CPU fallback differs: ${cpuMaximum}`);result.cpuMaximum=cpuMaximum;
    const measured=[];
    for(let index=982;index<994;index++) {
      const started=performance.now();
      const frame=await reader.read(index,{gpu:true,output:'native'});
      try {measured.push({index,score:frame.sharpness.score,sharpnessMs:frame.sharpnessMs,totalMs:performance.now()-started});}
      finally {frame.frame.close();}
    }
    const cached=[];
    for(let index=982;index<994;index++) {
      const frame=await reader.read(index,{gpu:true,output:'native'});
      try {cached.push(frame.sharpnessMs);} finally {frame.frame.close();}
    }
    result.sharpness={method:measured[0].score>=0?'median-laplacian-variance-9-downsampled-regions':null,
      meanMs:measured.reduce((sum,item)=>sum+item.sharpnessMs,0)/measured.length,
      maximumMs:Math.max(...measured.map(item=>item.sharpnessMs)),cachedMaximumMs:Math.max(...cached),samples:measured};
    return result;
  } finally { await reader.dispose(); }
}
