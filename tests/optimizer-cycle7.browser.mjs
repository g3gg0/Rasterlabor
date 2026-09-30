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
  const poses=new Map(corrections.map(p=>[p.frame,p.pose]));
  const worker=new WorkerClient('/fine-worker.js');let cached=new Set();const info=new Map();
  await worker.call('configure',{mask:remapInclusionMask(pcbSourceMask(),calibration.maps)});
  try{for(let i=0;i<pairs.length;i++){
   const started=performance.now(),pair=pairs[i];let decodeMs=0;
   for(const frame of [pair.reference,pair.current])if(!cached.has(frame)){
    const entry=trackingPath.find(p=>p.frame===frame);
    const decodeStarted=performance.now();const decoded=await readTrackingFrame(frame,{rectified:true,sourceMask:pcbSourceMask(),cache:false,measureSharpness:false});
    decodeMs+=performance.now()-decodeStarted;const bitmap=decoded.bitmap;
    info.set(frame,{frame,pose:poses.get(frame),offset:entry.mode==='window'?[-bitmap.width/2,-bitmap.height/2]:[calibration.maps.origin[0]-calibration.field.width/2,calibration.maps.origin[1]-calibration.field.height/2]});
    cached=new Set(await worker.call('store',{frame,bitmap,keep:[pair.reference,pair.current]},[bitmap]));
   }
   const args={reference:info.get(pair.reference),current:info.get(pair.current),options:{radius:8}};
   const anchorEvidence=pair.cells?.length?await worker.call('validate-anchors',{...args,cells:pair.cells}):null;
   const match={reference:pair.reference,current:pair.current,accepted:false,anchorEvidence};
   await globalThis.auditResult({index:i,total:pairs.length,pair,match:{...match,anchorEvidence},decodeMs,ms:performance.now()-started});
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
const solved=JSON.parse(await readFile('benchmarks/optimizer-bundle-cycle6.json','utf8'));
const network=JSON.parse(await readFile('benchmarks/optimizer-network-cycle6.json','utf8'));
const ids=new Set(solved.after.pairs.filter(p=>p.confirmedP90>4).map(p=>p.id));
const pairs=network.pairs.filter(p=>ids.has(p.id));
const filename='benchmarks/optimizer-anchor-audit-cycle7.jsonl';
const {appendFile}=await import('node:fs/promises');
let completed=[];try{completed=(await readFile(filename,'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);}catch{}
const done=new Set(completed.map(r=>`${r.pair.reference}:${r.pair.current}`));
const remaining=pairs.filter(p=>!done.has(`${p.reference}:${p.current}`));
let accepted=completed.filter(r=>r.match.accepted).length;
try{
 browser=await chromium.launch({channel:'msedge',headless:true});const page=await browser.newPage();page.setDefaultTimeout(180000);
 page.on('pageerror',e=>console.log('PAGEERROR '+e.message));
 await page.exposeFunction('auditResult',async r=>{await appendFile(filename,JSON.stringify(r)+'\n');if(r.match.accepted)accepted++;
 if(r.index%10===0)console.log(JSON.stringify({done:done.size+r.index+1,total:pairs.length,accepted,pair:[r.pair.reference,r.pair.current],reason:r.match.reason,ms:r.ms,decodeMs:r.decodeMs}));});
 await page.goto('http://127.0.0.1:'+server.address().port);
 await page.locator('#calibrationFile').setInputFiles('20260918_004341___.zip');await page.waitForFunction(()=>!document.querySelector('#exportButton').disabled);
 await page.locator('#videoFile').setInputFiles('20260918_004341.mp4');await page.waitForFunction(()=>optimizerProbe.ready());
 console.log(JSON.stringify({planned:pairs.length,remaining:remaining.length}));
 await page.evaluate(({corrections,pairs})=>optimizerProbe.run(corrections,pairs),{corrections:solved.corrections,pairs:remaining});
 console.log(JSON.stringify({complete:true,accepted}));
}finally{await browser?.close();await new Promise(r=>server.close(r));}
