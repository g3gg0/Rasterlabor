import http from 'node:http';
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {resolve,extname,sep} from 'node:path';
import {chromium} from 'playwright';
const root=resolve('dist');
const hook=`
globalThis.optimizerProbe={
 ready:()=>Boolean(calibration&&videoInfo&&!taskBusy&&!trackingPreviewBusy),
 async run(corrections,pairs){
  fineBusy=true;trackingMosaicRequest++;clearTimeout(trackingMosaicTimer);await frameReader.waitUntilIdle();
  const poses=new Map(corrections.map(p=>[p.frame,p.pose]));const worker=new WorkerClient('/compute-worker.js');
  const mask=remapInclusionMask(pcbSourceMask(),calibration.maps);
  try{for(const [index,pair]of pairs.entries()){
   const images=[];
   try{for(const frame of [pair.reference,pair.current]){
    const decoded=await readTrackingFrame(frame,{rectified:true,sourceMask:pcbSourceMask(),cache:false,measureSharpness:false});
    images.push({frame,pose:poses.get(frame),offset:[-decoded.bitmap.width/2,-decoded.bitmap.height/2],bitmap:decoded.bitmap});
   }
   const started=performance.now();
   const match=await worker.call('pcb-pair-register',{images,pair,mask,localLandmarks:true,useWebGpu:false,limits:{radius:64,angle:2,cycleLimit:2},fft:{cellSize:128,cellsPerAxis:3,searchRadius:63}},images.map(i=>i.bitmap));
   await globalThis.auditResult({index,total:pairs.length,pair,match,ms:performance.now()-started});
   }finally{for(const image of images)image.bitmap.close();}
  }}finally{worker.terminate();fineBusy=false;}
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
const input=JSON.parse(await readFile('benchmarks/optimizer-input-tracking.json','utf8'));
const solved=JSON.parse(await readFile('benchmarks/optimizer-bundle-cycle4.json','utf8'));
const {planFinePairs}=await import('../src/fine-alignment.js');
const nodes=input.path.map(p=>({...p,pose:solved.corrections.find(c=>c.frame===p.frame).pose}));
const pairs=[[20474,24476]].map(([reference,current])=>({reference,current,kind:'overlap-check'}));
const filename='benchmarks/optimizer-edge-worker.jsonl';
const {appendFile}=await import('node:fs/promises');
let completed=[];try{completed=(await readFile(filename,'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);}catch{}
const done=new Set(completed.map(r=>`${r.pair.reference}:${r.pair.current}`));
const remaining=pairs;
let accepted=completed.filter(r=>r.match.accepted).length;
try{
 browser=await chromium.launch({channel:'msedge',headless:true});const page=await browser.newPage();page.setDefaultTimeout(180000);
 page.on('pageerror',e=>console.log('PAGEERROR '+e.message));
 await page.exposeFunction('dumpFrame',async(frame,bytes)=>writeFile('benchmarks/label-'+frame+'.png',Buffer.from(bytes,'base64')));
 await page.exposeFunction('auditResult',async r=>{await appendFile(filename,JSON.stringify(r)+'\n');if(r.match.accepted)accepted++;assert.equal(r.match.accepted,r.pair.reference!==27596);if(r.match.accepted){assert.ok(r.match.reverseDistance<=1);assert.ok(r.match.fft.pointPairs.every(c=>c.normal));assert.ok(r.ms<10000);}
 if(r.index%10===0)console.log(JSON.stringify({done:done.size+r.index+1,total:pairs.length,accepted,pair:[r.pair.reference,r.pair.current],reason:r.match.reason,ms:r.ms,decodeMs:r.decodeMs}));});
 await page.goto('http://127.0.0.1:'+server.address().port);
 await page.locator('#calibrationFile').setInputFiles('20260918_004341___.zip');await page.waitForFunction(()=>!document.querySelector('#exportButton').disabled);
 await page.locator('#videoFile').setInputFiles('20260918_004341.mp4');await page.waitForFunction(()=>optimizerProbe.ready());
 console.log(JSON.stringify({planned:pairs.length,remaining:remaining.length}));
 await page.evaluate(({corrections,pairs})=>optimizerProbe.run(corrections,pairs),{corrections:input.path,pairs:remaining});
 console.log(JSON.stringify({complete:true,accepted}));
}finally{await browser?.close();await new Promise(r=>server.close(r));}
