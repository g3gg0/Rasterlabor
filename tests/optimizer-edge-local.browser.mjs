import http from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { unzipSync, strFromU8 } from 'fflate';

const root = resolve('dist');
const calibrationFile='20260918_004341___.zip';
const tracking=JSON.parse(await readFile('benchmarks/optimizer-input-tracking.json','utf8'));
const point={x:-274.1,y:5340.7};
const hook = `
const probeMatches=[];
let probeGraph;
const probeWorkerCall=WorkerClient.prototype.call;
WorkerClient.prototype.call=function(type,data,...rest) {
  if(type==='pose-graph-refit'&&data.options.includePassive)probeGraph=structuredClone(data);
  return probeWorkerCall.call(this,type,data,...rest);
};
const probeRegister=registerLocalPcbPairs;
registerLocalPcbPairs=async (...args)=>{
  const matches=await probeRegister(...args);
  probeMatches.push(...matches.filter(match=>match.pcbVerified).map(match=>({reference:match.reference,
    current:match.current,pose:match.forward.pose,referencePose:match.referencePose,
    method:match.landmarkRecovery?.viaPool?.accepted?'via-pool':match.fft?.method==='PCB-Vias'?'vias':match.landmarkRecovery?.accepted?'aperture':match.featureRecovery?.accepted?'feature':'fft'})));
  return matches;
};
globalThis.refitProbe = {
  ready: () => Boolean(videoInfo && calibration && !taskBusy && !trackingPreviewBusy),
  restore(value) {
    restoreTracking(value);
    trackingMosaicRequest++; clearTimeout(trackingMosaicTimer);
  },
  selection(point) {
    const entries = trackingPath.map(trackingGeometry).filter(g => g?.supports(point, trackingPixelAllowed)).map(g => g.entry);
    return {count: entries.length, parameters: pcbParameters(), preprocessing:refitPreprocessing(),
      field:{width:calibration.field.width,height:calibration.field.height}, maps:{width:calibration.maps.outputWidth,height:calibration.maps.outputHeight},
      groups:entries.map(e=>e.frame)};
  },
  async image(frame) {
    const decoded=await readTrackingFrame(frame,{rectified:true});
    try {
      const canvas=document.createElement('canvas');canvas.width=640;canvas.height=Math.round(640*decoded.bitmap.height/decoded.bitmap.width);
      canvas.getContext('2d').drawImage(decoded.bitmap,0,0,canvas.width,canvas.height);
      return canvas.toDataURL('image/png');
    } finally {decoded.bitmap.close();}
  },
  async pair(reference, current, point) {
    const worker = new WorkerClient('/compute-worker.js');
    if (${process.argv.includes('--gpu')}) {
      const call = worker.call.bind(worker);
      worker.call = (type,data,...rest) => call(type,type === 'pcb-pair-register' ? {...data,useWebGpu:true} : data,...rest);
    }
    const parameters = pcbParameters(), preprocessing = refitPreprocessing();
    const images = [];
    try {
      for(const frame of [reference,current]) {
        const entry = trackingPath.find(e => e.frame === frame);
        const decoded = await readTrackingFrame(frame,{rectified:true});
        const geometry = trackingGeometry(entry);
        images.push({frame,pose:{...(pathRefitProposal?.byFrame.get(frame)?.pose??entry.pose)},bitmap:decoded.bitmap,
          offset:entry.mode === 'window' ? [-decoded.bitmap.width/2,-decoded.bitmap.height/2] :
            [calibration.maps.origin[0]-calibration.field.width/2,calibration.maps.origin[1]-calibration.field.height/2],
          mask:localSelectionMask(geometry,point,preprocessing.region,(trackingRunOptions??trackingDataset?.options)?.imageMask??null)});
      }
      const [match] = await registerLocalPcbPairs(worker,images,[{reference,current,group:'0:1'}],
        {radius:parameters.radius,angle:parameters.angle},preprocessing,()=>{},parameters,true);
      return {reference,current,verified:match.pcbVerified,landmarkRecovery:match.landmarkRecovery,featureRecovery:match.featureRecovery,coarse:match.coarse,
        fft:{accepted:match.fft.accepted,reason:match.fft.reason,cells:match.fft.inlierCells?.length,
          support:match.fft.uniqueSupportArea,pose:match.fft.pose},
        forward:match.forward,backward:match.backward,cycle:match.reverseDistance,
        initialPose:match.currentPose,referencePose:match.referencePose};
    } finally {for(const image of images)image.bitmap.close();worker.terminate();}
  },
  async refit(point) {
    pathSelection=point;
    await refitTrackingPath();
    return {status:element('trackingPathRefitInfo').textContent,
      frames:pathRefitProposal?.byFrame.size,ready:pathRefitProposal?.overlayReady,
      aligned:pathRefitProposal?.alignedFrames,passive:pathRefitProposal?.passiveFrames,
      localEdges:pathRefitProposal?.localEdges,before:pathRefitProposal?.localBeforeRms,after:pathRefitProposal?.localAfterRms,
      graph:probeGraph,
      crossMatches:probeMatches.filter(match=>match.reference<30000&&match.current>30000),
      network:currentMatchNetwork(),corrections:pathRefitProposal?.corrections};
  },
  apply() {
    const expected=new Map(pathRefitProposal.byFrame);
    applyTrackingPathRefit();
    trackingMosaicRequest++; clearTimeout(trackingMosaicTimer);
    return trackingPath.filter(entry=>expected.has(entry.frame)&&
      ['x','y','rotation'].every(key=>entry.pose[key]===expected.get(entry.frame).pose[key])).length;
  }
};`;
const server = http.createServer(async (request,response) => {
  try {
    const path=resolve(root,'.'+(request.url==='/'?'/index.html':request.url));
    if(!path.startsWith(root+sep))throw new Error('path');
    let bytes=await readFile(path);
    if(request.url==='/app.js')bytes=Buffer.concat([bytes,Buffer.from(hook)]);
    response.setHeader('Content-Type',{'.html':'text/html','.js':'text/javascript','.css':'text/css'}[extname(path)]??'application/octet-stream');
    response.end(bytes);
  } catch {response.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
let browser;
try {
  browser=await chromium.launch({channel:'msedge',headless:true});
  const page=await browser.newPage();page.setDefaultTimeout(120000);
  page.on('pageerror',error=>console.error('pageerror:',error.message));
  await page.goto('http://127.0.0.1:'+server.address().port);
  await page.locator('#calibrationFile').setInputFiles(calibrationFile);
  await page.waitForFunction(()=>!document.querySelector('#exportButton').disabled);
  await page.locator('#videoFile').setInputFiles('20260918_004341.mp4');
  await page.waitForFunction(()=>globalThis.refitProbe.ready());
  await page.evaluate(value=>globalThis.refitProbe.restore(value),tracking);
  if(process.argv.includes('--proposal')) {
    const saved=JSON.parse(await readFile('benchmarks/local-refit-proposal-latest.json','utf8'));
    const byFrame=new Map(saved.refit.corrections.map(item=>[item.frame,item.pose]));
    for(const entry of tracking.frames??tracking.path??[])if(byFrame.has(entry.frame))entry.pose=byFrame.get(entry.frame);
    await page.evaluate(value=>globalThis.refitProbe.restore(value),tracking);
  }
  const selection=await page.evaluate(point=>globalThis.refitProbe.selection(point),point);
  console.log(JSON.stringify({selection}));
  const heartbeat=setInterval(()=>page.locator('#trackingPathRefitInfo').textContent().then(value=>console.log(value)).catch(()=>{}),20000);
  try{
    const refit=await page.evaluate(point=>globalThis.refitProbe.refit(point),point);
    await writeFile('benchmarks/optimizer-edge-local-proposal.json',JSON.stringify({selection,refit}));
    console.log(JSON.stringify({status:refit.status,frames:refit.frames,ready:refit.ready,before:refit.before,after:refit.after,localEdges:refit.localEdges}));
    assert.equal(refit.ready,true,refit.status);assert.equal(refit.frames,selection.count);
  }finally{clearInterval(heartbeat);}
} finally {await browser?.close();await new Promise(resolve=>server.close(resolve));}
