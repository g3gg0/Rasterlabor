import http from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { unzipSync, strFromU8 } from 'fflate';

const root = resolve('dist');
const calibrationFile=process.argv.includes('--latest')?'kalibrierung-v7-provisional______.zip':'kalibrierung-v8-provisional_________________.zip';
const tracking = process.argv.includes('--latest') ?
  JSON.parse(strFromU8(unzipSync(await readFile(calibrationFile),{filter:file=>file.name==='tracking.json'})['tracking.json'])) :
  JSON.parse(await readFile('tracking.json', 'utf8'));
const point = {x: -12025.7, y: 8951.2};
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
      corrections:pathRefitProposal?.corrections};
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
  console.log(JSON.stringify({selection:{count:selection.count,field:selection.field,maps:selection.maps}}));
  const results={selection,pairs:[]};
  if(process.argv.includes('--images')) {
    for(const frame of [27596,31344]) {
      const image=await page.evaluate(frame=>globalThis.refitProbe.image(frame),frame);
      await writeFile('benchmarks/refit-reference-'+frame+(process.argv.includes('--latest')?'-latest':'')+'.png',Buffer.from(image.split(',')[1],'base64'));
    }
  }
  if(process.argv.includes('--refit')) {
    const heartbeat=setInterval(()=>page.locator('#trackingPathRefitInfo').textContent().then(value=>console.log(value)).catch(()=>{}),30000);
    try {
      results.refit=await page.evaluate(point=>globalThis.refitProbe.refit(point),point);
      const {corrections,graph,...summary}=results.refit;
      console.log(JSON.stringify({refit:summary}));
      assert.equal(results.refit.ready,true,results.refit.status);
      assert.equal(results.refit.frames,selection.count);
      assert.ok(results.refit.crossMatches.length>0,'No measured connection between the two visits');
      assert.ok(results.refit.after<results.refit.before);
      for(const [reference,current] of [[27596,31344],[27597,31345],[27636,31365]]) {
        const pair=await page.evaluate(([reference,current,point])=>globalThis.refitProbe.pair(reference,current,point),[reference,current,point]);
        pair.remainingCorrection=pair.forward.pose ? Math.hypot(pair.forward.pose.x-pair.initialPose.x,pair.forward.pose.y-pair.initialPose.y) : null;
        results.pairs.push(pair);
        console.log(JSON.stringify({holdout:{reference,current,verified:pair.verified,remaining:pair.remainingCorrection,cycle:pair.cycle}}));
        assert.equal(pair.verified,true,'Reference pair was not confirmed after refitting');
        assert.ok(pair.remainingCorrection<=8,'Reference points still need more than 8 pixels of correction');
      }
      results.appliedFrames=await page.evaluate(()=>globalThis.refitProbe.apply());
      assert.equal(results.appliedFrames,selection.count);
      console.log(JSON.stringify({appliedFrames:results.appliedFrames}));
    }
    finally {clearInterval(heartbeat);}
  } else {
    for(const [reference,current] of (process.argv.includes('--third') ? [[27636,31365]] : process.argv.includes('--seed') ? [[27597,31345]] : [[27596,31344],[27597,31345],[27636,31365]])) {
      const pair=await page.evaluate(([reference,current,point])=>globalThis.refitProbe.pair(reference,current,point),[reference,current,point]);
      results.pairs.push(pair);
      console.log(JSON.stringify({pair:{reference,current,verified:pair.verified,cells:pair.fft.cells,
        cycle:pair.cycle,score:pair.forward.score,rotation:pair.forward.pose?.rotation,
        viaPool:pair.landmarkRecovery?.viaPool,vias:pair.landmarkRecovery?.vias&&{accepted:pair.landmarkRecovery.vias.accepted,inliers:pair.landmarkRecovery.vias.inliers,translationOnly:pair.landmarkRecovery.vias.translationOnly},
        correction:pair.forward.pose?Math.hypot(pair.forward.pose.x-pair.initialPose.x,pair.forward.pose.y-pair.initialPose.y):null}}));
      assert.equal(pair.verified,true,JSON.stringify(pair));
      assert.ok(pair.cycle<=5);
    }
  }
  await writeFile('benchmarks/local-refit-'+(process.argv.includes('--refit')?'proposal':'references')+
    (process.argv.includes('--latest')?'-latest':'')+'.json',JSON.stringify(results,null,2));
} finally {await browser?.close();await new Promise(resolve=>server.close(resolve));}
