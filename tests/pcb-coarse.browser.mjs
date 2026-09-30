import assert from 'node:assert/strict';
import http from 'node:http';
import {readFile,writeFile} from 'node:fs/promises';
import {resolve,extname,sep} from 'node:path';
import {chromium} from 'playwright';
const root=resolve('dist');
const hook=`globalThis.coarseProbe={ready:()=>Boolean(videoInfo&&calibration&&!taskBusy&&!trackingPreviewBusy),
async pair(){const worker=new WorkerClient('/compute-worker.js');const images=[];const raw=[];
try{for(const frame of [13343,13512]){const entry=trackingPath.find(e=>e.frame===frame);const decoded=await readTrackingFrame(frame,{rectified:true});
const canvas=new OffscreenCanvas(decoded.bitmap.width,decoded.bitmap.height);const ctx=canvas.getContext('2d');ctx.drawImage(decoded.bitmap,0,0);
const bytes=ctx.getImageData(0,0,canvas.width,canvas.height).data;
let str='';for(let i=0;i<bytes.length;i+=32768)str+=String.fromCharCode(...bytes.subarray(i,i+32768));
raw.push({frame,width:canvas.width,height:canvas.height,pose:entry.pose,data:btoa(str)});
images.push({frame,pose:{...entry.pose,x:entry.pose.x+(frame===13512&&${process.argv.includes('--drift')}?250:0),rotation:entry.pose.rotation+(frame===13512&&${process.argv.includes('--drift')}?3*Math.PI/180:0)},offset:[-canvas.width/2,-canvas.height/2],bitmap:decoded.bitmap});}
const p=pcbParameters();const started=performance.now();const result=await worker.call('pcb-pair-register',{images,pair:{reference:13343,current:13512,kind:'temporal'},useWebGpu:true,coarseRadius:p.coarseRadius,preprocessing:refitPreprocessing(),fft:{cellSize:256,cellsPerAxis:3,searchRadius:127,minimumPsr:p.minimumPsr,residualLimit:p.residualLimit},limits:{radius:p.radius,angle:5,cycleLimit:45,fftCycleFactor:p.fftCycleFactor,minimumScore:p.minimumScore,minimumSupport:p.minimumSupport}} ,images.map(i=>i.bitmap));return{raw,result,ms:performance.now()-started};
}finally{worker.terminate();}}};`;
const server=http.createServer(async(req,res)=>{try{const file=resolve(root,'.'+(req.url==='/'?'/index.html':req.url));if(!file.startsWith(root+sep))throw Error();let b=await readFile(file);if(req.url==='/app.js')b=Buffer.concat([b,Buffer.from(hook)]);res.setHeader('Content-Type',{'.html':'text/html','.js':'text/javascript','.css':'text/css'}[extname(file)]??'application/octet-stream');res.end(b);}catch{res.writeHead(404).end();}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try{browser=await chromium.launch({channel:'msedge',headless:true});const page=await browser.newPage();page.setDefaultTimeout(180000);await page.goto('http://127.0.0.1:'+server.address().port);await page.locator('#calibrationFile').setInputFiles('kalibrierung-v7-provisional______.zip');await page.waitForFunction(()=>!document.querySelector('#exportButton').disabled);await page.locator('#videoFile').setInputFiles('20260918_004341.mp4');await page.waitForFunction(()=>coarseProbe.ready());const {raw,result,ms}=await page.evaluate(()=>coarseProbe.pair());for(const item of raw){await writeFile('benchmarks/coarse-'+item.frame+'.rgba',Buffer.from(item.data,'base64'));delete item.data;}await writeFile('benchmarks/coarse-pair.json',JSON.stringify({raw,result,ms},null,2));console.log(JSON.stringify({ms,coarseMs:result.fft?.coarseSearch?.ms,raw,forward:result.forward,fft:{reason:result.fft?.reason,method:result.fft?.method,cells:result.fft?.inlierCells?.length},cycle:result.reverseDistance,coarse:result.coarse}));assert.equal(result.forward.accepted,true);assert.equal(result.backward.accepted,true);assert.ok(result.reverseDistance<5);if(process.argv.includes('--drift'))assert.equal(result.fft.method,'FFT-Grobsuche');}finally{await browser?.close();await new Promise(r=>server.close(r));}
