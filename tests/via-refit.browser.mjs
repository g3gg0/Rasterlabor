import http from 'node:http';
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {resolve,extname,sep} from 'node:path';
import {chromium} from 'playwright';
const root=resolve('dist');
const hook=`
globalThis.viaProbe={
 ready:()=>Boolean(calibration&&videoInfo&&!taskBusy&&!trackingPreviewBusy),
 async run(){
  fineBusy=true;trackingMosaicRequest++;clearTimeout(trackingMosaicTimer);
  await frameReader.waitUntilIdle();
  const ids=[28590,28726].map(id=>trackingPath.reduce((best,item)=>Math.abs(item.frame-id)<Math.abs(best.frame-id)?item:best,trackingPath[0]).frame);
  const images=[];const worker=new WorkerClient('/compute-worker.js');
  try{
   for(const frame of ids){const entry=trackingPath.find(e=>e.frame===frame);const decoded=await readTrackingFrame(frame,{rectified:true,sourceMask:pcbSourceMask(),cache:false});
    images.push({frame,pose:entry.pose,offset:entry.mode==='window'?[-decoded.bitmap.width/2,-decoded.bitmap.height/2]:[calibration.maps.origin[0]-calibration.field.width/2,calibration.maps.origin[1]-calibration.field.height/2],bitmap:decoded.bitmap});}
   const results=[];
   for(const localLandmarks of [false,true]){
    const copies=await Promise.all(images.map(image=>createImageBitmap(image.bitmap)));
    const started=performance.now();
    const result=await worker.call('pcb-pair-register',{images:images.map((image,i)=>({...image,bitmap:copies[i]})),pair:{reference:ids[0],current:ids[1],kind:'spatial'},useWebGpu:false,featureRecovery:false,localLandmarks,
     coarseRadius:384,preprocessing:{brightness:0,contrast:1,gamma:1,red:.3,green:.59,blue:.11},
     fft:{cellSize:128,cellsPerAxis:3,searchRadius:63,minimumPsr:6,residualLimit:4,adaptiveCells:true},
     limits:{radius:128,angle:5,cycleLimit:5,minimumScore:.9,minimumSupport:128}},copies);
    const network=mergeNetworkMatches(emptyMatchNetwork(),[result]);
    results.push({verified:acceptedPcbConstraints([result]).accepted.length,networkEdges:networkEdges(network).length,networkCells:network.pairs[0]?.cells.length,localLandmarks,ms:performance.now()-started,forward:result.forward,backward:result.backward,reverseDistance:result.reverseDistance,method:result.fft?.method,landmarks:result.landmarkRecovery});
   }
   return {ids,poses:images.map(image=>image.pose),results};
  }finally{worker.terminate();for(const image of images)image.bitmap.close();fineBusy=false;}
 }
};`;
const server = http.createServer(async (req, res) => {
  try {
    const file = resolve(root, '.' + (req.url === '/' ? '/index.html' : req.url));
    if (!file.startsWith(root + sep)) throw Error('path');
    let bytes = await readFile(file);
    if (req.url === '/app.js') bytes = Buffer.concat([bytes, Buffer.from(hook)]);
    res.setHeader('Content-Type', { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' }[extname(file)] || 'application/octet-stream');
    res.end(bytes);
  } catch { res.writeHead(404).end(); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
let browser;
try{
 browser=await chromium.launch({channel:'msedge',headless:true});const page=await browser.newPage();page.setDefaultTimeout(120000);
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:'+server.address().port);
 await page.locator('#calibrationFile').setInputFiles('20260918_004341.zip');
 await page.waitForFunction(()=>!document.querySelector('#exportButton').disabled);
 await page.locator('#videoFile').setInputFiles('20260918_004341.mp4');await page.waitForFunction(()=>viaProbe.ready());
 const result=await page.evaluate(()=>viaProbe.run());
 assert.equal(result.results[0].verified,0);assert.equal(result.results[1].verified,1);assert.equal(result.results[1].networkEdges,1);assert.ok(result.results[1].networkCells>=5);assert.deepEqual(errors,[]);
 await writeFile('benchmarks/via-refit-browser.json',JSON.stringify({...result,errors},null,2));
 console.log(JSON.stringify({ids:result.ids,poses:result.poses,results:result.results.map(r=>({...r,landmarks:r.landmarks&&{accepted:r.landmarks.accepted,reason:r.landmarks.reason,referenceVias:r.landmarks.referenceVias,currentVias:r.landmarks.currentVias,attempts:r.landmarks.attempts}})),errors}));
}finally{await browser?.close();await new Promise(r=>server.close(r));}
