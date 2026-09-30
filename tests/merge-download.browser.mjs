import assert from 'node:assert/strict';
import http from 'node:http';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {resolve,extname,sep} from 'node:path';
import {chromium} from 'playwright';
const root=resolve('dist'),output=resolve('benchmarks/merge-browser-fixture');await mkdir(output,{recursive:true});
const hook=`
globalThis.downloadProbe={
 prepare(){
  const size=128,n=size*size;
  calibration={field:{width:size,height:size},maps:{outputWidth:size,outputHeight:size,origin:[-64,-64],
   valid:new Uint8Array(n).fill(1),inverseX:Float32Array.from({length:n},(_,i)=>i%size),inverseY:Float32Array.from({length:n},(_,i)=>Math.floor(i/size))}};
  videoInfo={name:'fixture.mp4',width:size,height:size,frameCount:2,fps:30,duration:1,firstTimestamp:0,timestamps:[0,33333]};
  trackingPath=[{frame:0,mode:'window',pose:{x:64,y:64,rotation:0}},{frame:1,mode:'window',pose:{x:64,y:2112,rotation:0}}];
  trackingDataset={path:trackingPath,video:{name:'fixture.mp4',width:size,height:size}};
  const canvas=new OffscreenCanvas(size,size),ctx=canvas.getContext('2d');
  frameReader.read=async index=>{ctx.fillStyle=index?'#2030e0':'#e02030';ctx.fillRect(0,0,size,size);
   return {frame:new VideoFrame(canvas,{timestamp:index*33333}),orientation:{a:1,b:0,c:0,d:1,translateX:0,translateY:0}};};
  element('mergeSplit').value='two';element('mergeEdgeFeather').value='0';setWebGpuSelection('default');updateMergeControls();
 },state:()=>({busy:taskBusy,mergeRunning,status:element('mergeStatus').textContent,
  modes:mergeDownloads.map(d=>d.handle.mode),files:[...element('mergeDownloads').querySelectorAll('a')].map(a=>a.download),
  join:element('mergeJoinCommand').textContent})};`;
const server=http.createServer(async(req,res)=>{try{
 if(req.url==='/embed'){res.setHeader('Content-Type','text/html');res.end('<iframe src="/" style="width:1400px;height:1000px"></iframe>');return;}
 const file=resolve(root,'.'+(req.url==='/'?'/index.html':req.url));if(!file.startsWith(root+sep))throw Error();
 let b=await readFile(file);if(req.url==='/app.js')b=Buffer.concat([b,Buffer.from(hook)]);
 res.setHeader('Content-Type',{'.html':'text/html','.js':'text/javascript','.css':'text/css'}[extname(file)]||'application/octet-stream');res.end(b);
}catch{res.writeHead(404).end();}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try{
 browser=await chromium.launch({channel:'msedge',headless:true});const page=await browser.newPage({acceptDownloads:true,viewport:{width:1440,height:1080}});
 const errors=[],downloads=[];page.on('pageerror',e=>errors.push(e.message));
 page.on('download',d=>downloads.push(d.saveAs(resolve(output,d.suggestedFilename()))));
 await page.addInitScript(()=>{Object.defineProperty(window,'showSaveFilePicker',{value:undefined});Object.defineProperty(navigator.storage,'getDirectory',{value:undefined});});
 await page.goto('http://127.0.0.1:'+server.address().port+'/embed');
 const frame=page.frames().find(f=>f!==page.mainFrame());await frame.waitForFunction(()=>globalThis.downloadProbe);
 await frame.evaluate(()=>downloadProbe.prepare());await frame.locator('[data-workflow="merge"]').click();await frame.locator('#mergeStart').click();
 await frame.waitForFunction(()=>!downloadProbe.state().busy&&/^Fertig:/.test(downloadProbe.state().status),null,{timeout:60000});
 const state=await frame.evaluate(()=>downloadProbe.state());assert.equal(state.files.length,2);assert.deepEqual(state.modes,['Browser-Download','Browser-Download']);
 assert.match(state.join,/vips join/);const pngDownload=page.waitForEvent('download',{timeout:15000});await frame.locator('#mergeSavePreview').click();await pngDownload;await Promise.all(downloads);
 assert.equal(downloads.length,3);assert.deepEqual(errors,[]);await writeFile(resolve(output,'result.json'),JSON.stringify(state,null,2));console.log(JSON.stringify(state));
}finally{await browser?.close();await new Promise(r=>server.close(r));}
