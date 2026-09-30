import assert from 'node:assert/strict';
import http from 'node:http';
import {readFile,writeFile} from 'node:fs/promises';
import {resolve,extname,sep} from 'node:path';
import {chromium} from 'playwright';
const root=resolve('dist');
const hook=`globalThis.rotationProbe={ready:()=>Boolean(videoInfo&&calibration&&!taskBusy&&!trackingPreviewBusy),
async pair(){const worker=new WorkerClient('/compute-worker.js');const images=[];const raw=[];
try{for(const frame of [13263,13264]){const entry=trackingPath.find(e=>e.frame===frame);const decoded=await readTrackingFrame(frame,{rectified:true});
const canvas=new OffscreenCanvas(decoded.bitmap.width,decoded.bitmap.height);const ctx=canvas.getContext('2d');ctx.drawImage(decoded.bitmap,0,0);
const bytes=ctx.getImageData(0,0,canvas.width,canvas.height).data;
let str='';for(let i=0;i<bytes.length;i+=32768)str+=String.fromCharCode(...bytes.subarray(i,i+32768));
raw.push({frame,width:canvas.width,height:canvas.height,pose:entry.pose,data:btoa(str)});
images.push({frame,pose:{...entry.pose,rotation:entry.pose.rotation+(frame===13264?${process.argv.includes('--rotated')?'4*Math.PI/180':'0'}:0)},offset:[-canvas.width/2,-canvas.height/2],bitmap:decoded.bitmap});}
const p=pcbParameters();const result=await worker.call('pcb-pair-register',{images,pair:{reference:13263,current:13264,kind:'temporal'},useWebGpu:true,coarseRadius:p.coarseRadius,preprocessing:refitPreprocessing(),fft:{cellSize:256,cellsPerAxis:3,searchRadius:127,minimumPsr:p.minimumPsr,residualLimit:p.residualLimit},limits:{radius:p.radius,angle:5,cycleLimit:45,fftCycleFactor:p.fftCycleFactor,minimumScore:p.minimumScore,minimumSupport:p.minimumSupport}} ,images.map(i=>i.bitmap));return{raw,result};
}finally{worker.terminate();}}};`;
const server=http.createServer(async(req,res)=>{try{const file=resolve(root,'.'+(req.url==='/'?'/index.html':req.url));if(!file.startsWith(root+sep))throw Error();let b=await readFile(file);if(req.url==='/app.js')b=Buffer.concat([b,Buffer.from(hook)]);res.setHeader('Content-Type',{'.html':'text/html','.js':'text/javascript','.css':'text/css'}[extname(file)]??'application/octet-stream');res.end(b);}catch{res.writeHead(404).end();}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try{browser=await chromium.launch({channel:'msedge',headless:true});const page=await browser.newPage();page.setDefaultTimeout(180000);await page.goto('http://127.0.0.1:'+server.address().port);await page.locator('#calibrationFile').setInputFiles('kalibrierung-v7-provisional______.zip');await page.waitForFunction(()=>!document.querySelector('#exportButton').disabled);await page.locator('#videoFile').setInputFiles('20260918_004341.mp4');await page.waitForFunction(()=>rotationProbe.ready());const {raw,result}=await page.evaluate(()=>rotationProbe.pair());for(const item of raw){await writeFile('benchmarks/rotation-'+item.frame+'.rgba',Buffer.from(item.data,'base64'));delete item.data;}await writeFile('benchmarks/rotation-pair.json',JSON.stringify({raw,result},null,2));console.log(JSON.stringify({raw,forward:result.forward,fft:{reason:result.fft?.reason,method:result.fft?.method,cells:result.fft?.inlierCells?.length},cycle:result.reverseDistance,coarse:result.coarse}));assert.equal(result.forward.accepted,true);assert.equal(result.backward.accepted,true);assert.ok(result.reverseDistance<5);assert.ok(Math.abs(result.forward.pose.rotation-raw[1].pose.rotation)<.1*Math.PI/180);}finally{await browser?.close();await new Promise(r=>server.close(r));}
