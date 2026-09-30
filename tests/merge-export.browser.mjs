import assert from 'node:assert/strict';
import http from 'node:http';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { chromium } from 'playwright';
const root = resolve('dist'), output = resolve('exports/20260918_004341-browser');
await mkdir(output, {recursive:true});
const hook = `
globalThis.mergeProbe = {
 ready:()=>Boolean(calibration && videoInfo && !taskBusy && !trackingPreviewBusy),
 state:()=>({busy:taskBusy,mergeRunning,status:element('mergeStatus').textContent,preview:element('mergePreviewState').textContent,
   cache:frameReader.cacheStats(),heap:performance.memory?.usedJSHeapSize,
   downloads:[...element('mergeDownloads').querySelectorAll('a')].map(a=>a.download),
   modes:mergeDownloads.map(d=>d.handle.mode),join:element('mergeJoinCommand').textContent}),
 stopMosaic(){trackingMosaicRequest++;clearTimeout(trackingMosaicTimer);},
};`;
const server = http.createServer(async(req,res)=>{
 try {
  const file=resolve(root,'.'+(req.url==='/'?'/index.html':req.url));
  if(!file.startsWith(root+sep))throw Error('path');
  let bytes=await readFile(file);if(req.url==='/app.js')bytes=Buffer.concat([bytes,Buffer.from(hook)]);
  res.setHeader('Content-Type',{'.html':'text/html','.js':'text/javascript','.css':'text/css'}[extname(file)]||'application/octet-stream');res.end(bytes);
 }catch{res.writeHead(404).end();}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
let browser;
try {
 browser=await chromium.launch({channel:'msedge',headless:true});
 const context=await browser.newContext({acceptDownloads:true,viewport:{width:1440,height:1000}});
 const page=await context.newPage();page.setDefaultTimeout(180000);
 const errors=[],downloads=[];let crashed=false;
 page.on('pageerror',e=>errors.push(e.message));page.on('crash',()=>{crashed=true;console.log('BROWSER CRASH');});
 page.on('download',d=>{console.log('DOWNLOAD '+d.suggestedFilename());downloads.push(d.saveAs(resolve(output,d.suggestedFilename())));});
 if(process.argv.includes('--blob')) await page.addInitScript(()=>Object.defineProperty(navigator.storage,'getDirectory',{value:undefined}));
 await page.addInitScript(()=>Object.defineProperty(window,'showSaveFilePicker',{value:undefined}));
 await page.goto('http://127.0.0.1:'+server.address().port);
 await page.locator('#calibrationFile').setInputFiles('20260918_004341.zip');
 await page.waitForFunction(()=>document.querySelector('#exportButton').disabled===false);
 await page.locator('#videoFile').setInputFiles('20260918_004341.mp4');
 await page.waitForFunction(()=>mergeProbe.ready());
 await page.evaluate(()=>mergeProbe.stopMosaic());
 await page.locator('[data-workflow="merge"]').click();
 if(process.argv.includes('--two'))await page.locator('#mergeSplit').selectOption('two');
 await page.locator('#mergeStart').click();
 let maximumCache=0,maximumHeap=0;
 for(let i=0;i<1800;i++) {
   await page.waitForTimeout(1000);
   const state=await page.evaluate(()=>mergeProbe.state());
   maximumCache=Math.max(maximumCache,state.cache.bytes);maximumHeap=Math.max(maximumHeap,state.heap||0);
   if(i%10===0)console.log(JSON.stringify(state));
   if(!state.busy&&!state.mergeRunning){
     assert.match(state.status,/^Fertig:/);assert.ok(state.downloads.length>=1);
     await writeFile(resolve(output,'export-result.json'),JSON.stringify({...state,maximumCache,maximumHeap,errors,crashed},null,2));
     await writeFile(resolve(output,'zusammenfuegen.sh'),state.join+'\n');
     break;
   }
   if(i===1799)throw Error('Export timeout');
 }
 const pngDownload=page.waitForEvent('download',{timeout:30000});await page.locator('#mergeSavePreview').click();
 await pngDownload;await Promise.all(downloads);
 await page.locator('[data-workflow-panel="merge"]').screenshot({path:resolve(output,'export-ui.png')});
 assert.deepEqual(errors,[]);assert.equal(crashed,false);assert.equal(maximumCache,0);
 console.log(JSON.stringify({output,maximumCache,maximumHeap,files:downloads.length,errors}));
} finally {await browser?.close();await new Promise(r=>server.close(r));}
