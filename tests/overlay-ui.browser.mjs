import assert from 'node:assert/strict';
import http from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { chromium } from 'playwright';

const root=resolve('dist');
// Only appended to the isolated test server's response; never written to the application bundle.
const hook=`
globalThis.overlaySmoke = {
  ready: () => Boolean(videoInfo && calibration && !taskBusy && !trackingPreviewBusy),
  unmaskedSupport() {
    const mask=ensureTrackingImageMask(calibration.maps.outputWidth,calibration.maps.outputHeight);
    mask.data.fill(MASK_NEUTRAL);trackingMaskHasSelection=false;
    const pose=trackingPath.find(entry=>entry.pose)?.pose;
    const point={x:pose.x,y:pose.y};
    pathHover=point;drawTrackingPath();
    const info=element('trackingPathInfo').textContent;
    return {supporting:Number(info.split('| ')[1]?.split(' ')[0]||0),info};
  },
  prepare() {
    trackingPath = trackingPath.filter(e => e.frame >= 970 && e.frame < 982);
    ensureTrackingImageMask(calibration.maps.outputWidth, calibration.maps.outputHeight).data.fill(MASK_SEARCH);
    trackingMaskHasSelection = true;
    element('trackingUseWebGpu').checked = true;
    drawTrackingPath();
    const pose = trackingPath[0].raw ?? trackingPath[0].pose;
    return {x:pose.x,y:pose.y};
  },
  async run(point) {
    await loadPathOverlay(point);
    return { info:element('trackingOverlayInfo').textContent,width:pathOverlay.width,height:pathOverlay.height };
  },
  async selectContributor() {
    const selector=element('trackingOverlayFrame');
    const frame=selector.options[1]?.value;
    selector.value=frame;await selectOverlayContributor();
    return {frame:Number(frame),options:selector.options.length,selected:overlaySelectedImage?.geometry.entry.frame,
      inspectDisabled:element('trackingOverlayInspect').disabled};
  },
  restoreComposite() {
    const selector=element('trackingOverlayFrame');selector.value='';selector.dispatchEvent(new Event('change'));
    return {selected:overlaySelectedImage,inspectDisabled:element('trackingOverlayInspect').disabled};
  },
  async cancelThenRun(point) {
    const cancelled=loadPathOverlay(point);
    await new Promise(r=>setTimeout(r,20));
    const current=loadPathOverlay(point);
    await Promise.all([cancelled,current]);
    return {info:element('trackingOverlayInfo').textContent,width:pathOverlay.width,height:pathOverlay.height};
  },
  async readerCheck() {
    const cpuRaw=await readFrame(970,{gpu:false});
    const sharpness=cpuRaw.sharpness;
    const cpuPixels=frameReader.toRgba(cpuRaw.bitmap);cpuRaw.bitmap.close();
    const gpuRaw=await readFrame(970,{gpu:true});
    const gpuPixels=frameReader.toRgba(gpuRaw.bitmap);gpuRaw.bitmap.close();
    let rawMaximum=0;
    for(let i=0;i<cpuPixels.data.length;i+=4*701)for(let k=0;k<4;k++)
      rawMaximum=Math.max(rawMaximum,Math.abs(cpuPixels.data[i+k]-gpuPixels.data[i+k]));
    await showFrame(970);
    element('useWebGpu').checked=true;
    await updateRectified();
    const preview=[rectifiedImage.width,rectifiedImage.height];
    element('trackingMode').value='window';
    element('trackingStart').value='970';element('trackingEnd').value='972';
    element('trackingSearchRadius').value='128';
    trackingRectangle={x:1682,y:3456,width:1024,height:1024};
    trackingImageMask=null;
    await runTracking(true);
    const tracking={frames:trackingProfile.frames,rgbaMs:trackingProfile.rgbaMs,
      meanMs:trackingProfile.totalMs/trackingProfile.frames,accelerators:[...trackingProfile.accelerators]};
    let bytes=0;
    const writable=new WritableStream({write(chunk){bytes+=chunk.data.byteLength;}});
    const encoder=await VideoExporter.create({width:calibration.maps.outputWidth,height:calibration.maps.outputHeight,fps:30,writable});
    try {
      const result=await readFrame(970,{rectified:true});
      try {await encoder.addFrame(result.bitmap,result.width,result.height,0,33333,0);}
      finally {result.bitmap.close();}
      await encoder.finish();
    } catch(error){await encoder.cancel();throw error;}
    return {preview,tracking,exportBytes:bytes,rawMaximum,sharpness,
      trackedSharpness:trackingPath.at(-1).sharpness};
  },
  async tiled(point) {
    const previous=WebGpuOverlay.needsTiles;
    WebGpuOverlay.needsTiles=async()=>true;
    try {await loadPathOverlay(point);return {info:element('trackingOverlayInfo').textContent,width:pathOverlay.width,height:pathOverlay.height,tiles:pathOverlay.tiles?.length};}
    finally {WebGpuOverlay.needsTiles=previous;}
  }
};`;
const server=http.createServer(async(req,res)=>{
  try {
    const path=resolve(root,'.'+(req.url==='/'?'/index.html':req.url));
    if(!path.startsWith(root+sep))throw new Error('path');
    let bytes=await readFile(path);
    if(req.url==='/app.js')bytes=Buffer.concat([bytes,Buffer.from(hook)]);
    res.setHeader('Content-Type',{'.html':'text/html','.js':'text/javascript','.css':'text/css'}[extname(path)]||'application/octet-stream');res.end(bytes);
  }catch{res.writeHead(404).end();}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
let browser;
try {
  browser=await chromium.launch({channel:'msedge',headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  page.setDefaultTimeout(120000);
  const errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('dialog',d=>d.accept());
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.locator('#calibrationFile').setInputFiles('../kalibrierung-v9-provisional-mit-xyr.zip');
  await page.waitForFunction(()=>document.querySelector('#exportButton').disabled===false);
  await page.locator('#videoFile').setInputFiles('../20260914_214049.mp4');
  await page.waitForFunction(()=>globalThis.overlaySmoke?.ready());
  await page.locator('[data-workflow="tracking"]').click();
  await page.waitForFunction(()=>globalThis.overlaySmoke.ready());
  const unmasked=await page.evaluate(()=>globalThis.overlaySmoke.unmaskedSupport());
  console.log(JSON.stringify({unmasked}));
  assert.ok(unmasked.supporting>0);assert.match(unmasked.info,/\| [1-9][0-9]* Bilder:/);
  const point=await page.evaluate(()=>globalThis.overlaySmoke.prepare());
  const first=await page.evaluate(p=>globalThis.overlaySmoke.run(p),point);
  assert.match(first.info,/WebGPU.*ms\/Frame/);assert.equal(first.width,5044);assert.equal(first.height,7973);
  const contributor=await page.evaluate(()=>globalThis.overlaySmoke.selectContributor());
  assert.ok(contributor.options>1);assert.equal(contributor.selected,contributor.frame);assert.equal(contributor.inspectDisabled,false);
  await page.locator('#trackingOverlayInspect').click();
  await page.waitForFunction(()=>document.querySelector('#trackingInspector').hidden===false);
  await page.waitForFunction(()=>trackingPreviewBusy===false);
  const restored=await page.evaluate(()=>globalThis.overlaySmoke.restoreComposite());
  assert.equal(restored.selected,null);assert.equal(restored.inspectDisabled,true);
  const restarted=await page.evaluate(p=>globalThis.overlaySmoke.cancelThenRun(p),point);
  assert.match(restarted.info,/WebGPU.*ms\/Frame/);assert.equal(restarted.width,5044);assert.equal(restarted.height,7973);
  const tiled=await page.evaluate(p=>globalThis.overlaySmoke.tiled(p),point);
  assert.equal(tiled.width,5044);assert.equal(tiled.height,7973);assert.ok(tiled.tiles>1);assert.match(tiled.info,/Kacheln/);
  await page.locator('#trackingOverlayPane').screenshot({path:'benchmarks/overlay-ui-tiles.png'});
  const reader=await page.evaluate(()=>globalThis.overlaySmoke.readerCheck());
  assert.deepEqual(reader.preview,[4389,7937]);assert.equal(reader.tracking.frames,3);
  assert.equal(reader.tracking.rgbaMs,0);assert.ok(reader.tracking.accelerators.includes('WebGPU native VideoFrame'));
  assert.ok(reader.exportBytes>0);assert.ok(reader.rawMaximum<=1);
  assert.ok(Number.isFinite(reader.sharpness.score)&&reader.sharpness.score>0);
  assert.ok(Number.isFinite(reader.trackedSharpness.score)&&reader.trackedSharpness.score>0);
  assert.equal(reader.trackedSharpness.method,reader.sharpness.method);
  console.log(JSON.stringify({reader}));
  assert.deepEqual(errors,[]);
  await writeFile('benchmarks/overlay-ui-tiles.json',JSON.stringify({first,restarted,tiled,reader,errors},null,2));
  console.log(JSON.stringify({first,restarted,tiled,reader,errors},null,2));
}finally{await browser?.close();await new Promise(r=>server.close(r));}
