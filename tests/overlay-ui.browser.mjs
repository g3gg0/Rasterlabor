import assert from 'node:assert/strict';
import http from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { chromium } from 'playwright';

const root=resolve('dist');
// Only appended to the isolated test server's response; never written to the application bundle.
const hook=`
globalThis.overlaySmoke = {
  async verifyCachePixels() {
    const edge = 64, n = edge * edge;
    const maps = {outputWidth:edge,outputHeight:edge,valid:new Uint8Array(n).fill(1),
      inverseX:Float32Array.from({length:n},(_,i)=>i%edge),inverseY:Float32Array.from({length:n},(_,i)=>Math.floor(i/edge))};
    const source = new OffscreenCanvas(edge,edge), ctx = source.getContext('2d');
    ctx.fillStyle='#124b26';ctx.fillRect(0,0,edge,edge);
    ctx.fillStyle='#ef5231';ctx.fillRect(5,8,23,17);ctx.fillStyle='#728ae8';ctx.fillRect(35,32,19,25);
    const frame = new VideoFrame(source,{timestamp:0});
    const orientation = {a:1,b:0,c:0,d:1,translateX:0,translateY:0};
    const geometry = (x,y,angle) => ({entry:{frame:1},c:Math.cos(angle),s:Math.sin(angle),world:()=>({x,y})});
    const renderer = await WebGpuOverlay.create(maps,96,96,0,0,null,0,0);
    renderer.cacheEnabled=true;
    const read = async () => {
      const bitmap=await renderer.finish(), canvas=new OffscreenCanvas(96,96), context=canvas.getContext('2d');
      context.drawImage(bitmap,0,0);bitmap.close();return context.getImageData(0,0,96,96).data;
    };
    try {
      await renderer.addFrame(frame,orientation,geometry(0,0,0));
      await renderer.clear(0,0);
      const moved=geometry(19,7,0.13);
      await renderer.addCachedFrame(moved);const cached=await read();
      await renderer.clear(0,0);await renderer.addFrame(frame,orientation,moved);const fresh=await read();
      await renderer.clear(0,0);await renderer.addCachedFrame(moved,0,true);const preview=await read();
      const difference=(a,b)=>a.reduce((maximum,value,i)=>Math.max(maximum,Math.abs(value-b[i])),0);
      const result={cachedError:difference(cached,fresh),previewError:difference(preview,fresh),remaps:renderer.remapCount};
      renderer.setActiveFrames([]);result.released=renderer.frameTextures.size===0 && renderer.cacheBytes===0 && renderer.previewBytes===0;
      return result;
    } finally {frame.close();renderer.destroy();}
  },
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
    trackingMosaicRequest++;clearTimeout(trackingMosaicTimer);
    const originalCall=decoder.call.bind(decoder);
    this.decoderReads=0;
    decoder.call=(type,...args)=>{
      if(type==='frame'||type==='native-frame')this.decoderReads++;
      return originalCall(type,...args);
    };
    trackingPath = trackingPath.filter(e => e.frame >= 970 && e.frame < ${process.argv.includes('--gpu-cache') ? 972 : 982});
    ensureTrackingImageMask(calibration.maps.outputWidth, calibration.maps.outputHeight).data.fill(MASK_SEARCH);
    trackingMaskHasSelection = true;
    setWebGpuSelection('default'); element('globalGpuAdapter').value = 'default';
    drawTrackingPath();
    const pose = trackingPath[0].raw ?? trackingPath[0].pose;
    return {x:pose.x,y:pose.y};
  },
  async run(point) {
    await frameReader.waitUntilIdle();
    await loadPathOverlay(point);
    clearTimeout(trackingMosaicTimer);
    await renderTrackingMosaic();
    // A pose change can select a previously hidden contributor. Warm this small
    // fixture so the test measures reuse rather than a legitimate first decode.
    for(const entry of trackingPath){
      const image=await readTrackingFrame(entry.frame,{output:'native'});image.frame.close();
    }
    return { info:element('trackingOverlayInfo').textContent,width:pathOverlay.width,height:pathOverlay.height };
  },
  async selectContributor() {
    const selector=element('trackingOverlayFrame');
    const frame=selector.options[1]?.value;
    selector.value=frame;await selectOverlayContributor();
    return {frame:Number(frame),options:selector.options.length,selected:overlaySelectedImage?.geometry.entry.frame,
      inspectDisabled:element('trackingOverlayInspect').disabled};
  },
  hit() {
    const bounds=element('trackingOverlayCanvas').getBoundingClientRect();
    const scale=Math.min(bounds.width/pathOverlay.width,bounds.height/pathOverlay.height)*overlayZoom;
    for(let row=1;row<10;row++)for(let column=1;column<10;column++){
      const x=bounds.width*column/10,y=bounds.height*row/10;
      const world={x:overlayBounds.minX+(x-(bounds.width-pathOverlay.width*scale)/2-overlayPan.x)/scale,
        y:overlayBounds.minY+(y-(bounds.height-pathOverlay.height*scale)/2-overlayPan.y)/scale};
      const covering=overlayContributors.filter(g=>g.supports(world,trackingPixelAllowed));
      if(covering.length)return {...this.screen(world),world,covering:covering.length};
    }
    throw new Error('No visible frame in overlay');
  },
  screen(world) {
    const bounds=element('trackingOverlayCanvas').getBoundingClientRect();
    const scale=Math.min(bounds.width/pathOverlay.width,bounds.height/pathOverlay.height)*overlayZoom;
    return {x:bounds.left+(bounds.width-pathOverlay.width*scale)/2+overlayPan.x+(world.x-overlayBounds.minX)*scale,
      y:bounds.top+(bounds.height-pathOverlay.height*scale)/2+overlayPan.y+(world.y-overlayBounds.minY)*scale,scale};
  },
  pointerTarget(world) {
    const position=this.screen(world);
    const geometry=overlayPoseDraft&&overlaySelectedImage&&trackingGeometry(
      {...overlaySelectedImage.geometry.entry,pose:overlayPoseDraft.pose});
    return {selected:geometry?.supports(world,trackingPixelAllowed),
      original:overlaySelectedImage?.geometry.supports(world,trackingPixelAllowed),
      target:document.elementFromPoint(position.x,position.y)?.id,position};
  },
  poseState() {
    const frame=overlayPoseDraft?.frame;
    const entry=trackingPath.find(e=>e.frame===frame);
    return {frame,pose:entry?.pose&&{...entry.pose},raw:entry?.raw&&{...entry.raw},
      draft:overlayPoseDraft?.pose&&{...overlayPoseDraft.pose},pan:{...overlayPan},
      applyDisabled:element('trackingOverlayPoseApply').disabled,
      discardDisabled:element('trackingOverlayPoseDiscard').disabled};
  },
  committedPose(frame) {
    const entry=trackingPath.find(e=>e.frame===frame);
    return {pose:entry.pose,raw:entry.raw};
  },
  bounds() { return {width:pathOverlay.width,height:pathOverlay.height}; },
  cacheState() {
    return {...frameReader.cacheStats(),decoderReads:this.decoderReads,
      thumbnails:trackingMosaicFrames.size,
      gpu:overlayGpuSession ? {remaps:overlayGpuSession.renderer.remapCount,
        hits:overlayGpuSession.renderer.cacheHits, frames:overlayGpuSession.renderer.frameTextures.size,
        bytes:overlayGpuSession.renderer.cacheBytes, previewBytes:overlayGpuSession.renderer.previewBytes,
        liveMs:overlayGpuSession.renderer.lastLiveMs, failure:overlayGpuSession.renderer.failure?.message} : null};
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
    setWebGpuSelection('default'); element('globalGpuAdapter').value='default';
    await updateRectified();
    const preview=[rectifiedImage.width,rectifiedImage.height];
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
  assert.ok(unmasked.supporting>0);assert.match(unmasked.info,/\| [1-9][0-9]* Bilder \| Anzeige:/);
  const point=await page.evaluate(()=>globalThis.overlaySmoke.prepare());
  const first=await page.evaluate(p=>globalThis.overlaySmoke.run(p),point);
  assert.match(first.info,/WebGPU.*ms\/Frame/);assert.ok(first.width>0&&first.height>0);
  await page.locator('#trackingTabOverlay').click();
  await page.locator('#trackingOverlayCanvas').scrollIntoViewIfNeeded();
  const hit=await page.evaluate(()=>globalThis.overlaySmoke.hit());
  const panBefore=await page.evaluate(()=>globalThis.overlaySmoke.poseState());
  await page.mouse.move(hit.x,hit.y);await page.mouse.down();await page.mouse.move(hit.x+15,hit.y+8);await page.mouse.up();
  const panned=await page.evaluate(()=>globalThis.overlaySmoke.poseState());
  assert.deepEqual(panned.pan,{x:panBefore.pan.x+15,y:panBefore.pan.y+8});
  assert.equal(panned.frame,undefined);
  await page.locator('#trackingOverlayFit').click();
  await page.mouse.click(hit.x,hit.y);
  await page.waitForFunction(()=>globalThis.overlaySmoke.poseState().frame!==undefined);
  const selected=await page.evaluate(()=>globalThis.overlaySmoke.poseState());
  assert.equal(selected.applyDisabled,true);assert.equal(selected.discardDisabled,true);
  if(hit.covering>1){
    const pointOnCanvas=await page.evaluate(world=>globalThis.overlaySmoke.screen(world),hit.world);
    await page.mouse.click(pointOnCanvas.x,pointOnCanvas.y);
    await page.waitForFunction(frame=>{
      const current=globalThis.overlaySmoke.poseState().frame;
      return current!==undefined&&current!==frame;
    },selected.frame);
  }
  const original=await page.evaluate(()=>globalThis.overlaySmoke.poseState());
  const cacheBeforeEditing=await page.evaluate(()=>globalThis.overlaySmoke.cacheState());
  const target=await page.evaluate(world=>globalThis.overlaySmoke.pointerTarget(world),hit.world);
  assert.equal(target.selected,true,JSON.stringify(target));
  assert.equal(target.target,'trackingOverlayCanvas',JSON.stringify(target));
  const drag=async()=>{
    const position=await page.evaluate(world=>globalThis.overlaySmoke.screen(world),hit.world);
    await page.mouse.move(position.x,position.y);await page.mouse.down();
    await page.mouse.move(position.x+12,position.y+7,{steps:2});await page.mouse.up();
    return {state:await page.evaluate(()=>globalThis.overlaySmoke.poseState()),scale:position.scale};
  };
  const {state:preview,scale:dragScale}=await drag();
  assert.equal(preview.frame,original.frame);assert.deepEqual(preview.pose,original.pose);
  assert.deepEqual(preview.pan,original.pan);
  assert.ok(Math.abs(preview.draft.x-original.pose.x-12/dragScale)<0.01);
  assert.ok(Math.abs(preview.draft.y-original.pose.y-7/dragScale)<0.01);
  assert.equal(preview.applyDisabled,false);assert.equal(preview.discardDisabled,false);
  await page.waitForFunction(()=>Boolean(overlaySmoke.cacheState().gpu?.hits) &&
    /GPU-Livevorschau/.test(document.querySelector('#trackingOverlayInfo').textContent));
  await page.locator('#trackingOverlayPoseDiscard').click();
  const discarded=await page.evaluate(()=>globalThis.overlaySmoke.poseState());
  assert.deepEqual(discarded.draft,original.pose);assert.deepEqual(discarded.pose,original.pose);
  assert.equal(discarded.applyDisabled,true);
  await drag();await page.locator('#trackingOverlayPoseApply').click();
  await page.waitForFunction(()=>!globalThis.overlaySmoke.poseState().frame);
  const applied=await page.evaluate(frame=>globalThis.overlaySmoke.committedPose(frame),original.frame);
  assert.ok(Math.abs(applied.pose.x-original.pose.x-12/dragScale)<0.01);
  assert.ok(Math.abs(applied.pose.y-original.pose.y-7/dragScale)<0.01);
  if(original.raw){assert.ok(Math.abs(applied.raw.x-original.raw.x-12/dragScale)<0.01);}
  await page.waitForFunction(()=>/Mischbild aktualisiert/.test(document.querySelector('#trackingOverlayInfo').textContent));
  const cacheAfterEditing=await page.evaluate(()=>globalThis.overlaySmoke.cacheState());
  assert.equal(cacheAfterEditing.decoderReads,cacheBeforeEditing.decoderReads,
    JSON.stringify({cacheBeforeEditing,cacheAfterEditing}));
  const contributor=await page.evaluate(()=>globalThis.overlaySmoke.selectContributor());
  assert.ok(contributor.options>1);assert.equal(contributor.selected,contributor.frame);assert.equal(contributor.inspectDisabled,false);
  const rotationStart=await page.evaluate(()=>globalThis.overlaySmoke.poseState());
  const wheelPoint=await page.evaluate(world=>globalThis.overlaySmoke.screen(world),hit.world);
  await page.mouse.move(wheelPoint.x,wheelPoint.y);
  await page.keyboard.down('Shift');await page.mouse.wheel(0,100);await page.keyboard.up('Shift');
  await page.waitForFunction(rotation=>globalThis.overlaySmoke.poseState().draft.rotation!==rotation,rotationStart.draft.rotation);
  const fine=await page.evaluate(()=>globalThis.overlaySmoke.poseState());
  assert.ok(Math.abs((fine.draft.rotation-rotationStart.pose.rotation)*180/Math.PI-0.1)<0.001);
  assert.deepEqual(fine.pan,rotationStart.pan);assert.equal(fine.applyDisabled,false);
  await page.keyboard.down('Alt');await page.mouse.wheel(0,100);await page.keyboard.up('Alt');
  await page.waitForFunction(rotation=>globalThis.overlaySmoke.poseState().draft.rotation!==rotation,fine.draft.rotation);
  const coarse=await page.evaluate(()=>globalThis.overlaySmoke.poseState());
  assert.ok(Math.abs((coarse.draft.rotation-fine.draft.rotation)*180/Math.PI-1)<0.001);
  await page.locator('#trackingOverlayPoseDiscard').click();
  assert.deepEqual((await page.evaluate(()=>globalThis.overlaySmoke.poseState())).draft,rotationStart.pose);
  await page.mouse.move(wheelPoint.x,wheelPoint.y);
  await page.keyboard.down('Shift');await page.mouse.wheel(0,-100);await page.keyboard.up('Shift');
  await page.waitForFunction(rotation=>globalThis.overlaySmoke.poseState().draft.rotation!==rotation,rotationStart.draft.rotation);
  await page.locator('#trackingOverlayPoseApply').click();
  const rotated=await page.evaluate(frame=>globalThis.overlaySmoke.committedPose(frame),rotationStart.frame);
  assert.ok(Math.abs((rotated.pose.rotation-rotationStart.pose.rotation)*180/Math.PI+0.1)<0.001);
  if(rotationStart.raw)assert.ok(Math.abs((rotated.raw.rotation-rotationStart.raw.rotation)*180/Math.PI+0.1)<0.001);
  await page.waitForFunction(()=>/Mischbild aktualisiert/.test(document.querySelector('#trackingOverlayInfo').textContent));
  const cacheAfterRotation=await page.evaluate(()=>globalThis.overlaySmoke.cacheState());
  assert.equal(cacheAfterRotation.decoderReads,cacheAfterEditing.decoderReads,
    JSON.stringify({cacheAfterEditing,cacheAfterRotation}));
  assert.ok(cacheBeforeEditing.gpu?.frames > 0);
  if (process.argv.includes('--gpu-cache')) assert.equal(cacheAfterRotation.gpu.remaps,cacheBeforeEditing.gpu.remaps);
  assert.ok(cacheAfterRotation.gpu.hits > cacheBeforeEditing.gpu.hits);
  assert.equal(cacheAfterRotation.gpu.failure,undefined);
  console.log(JSON.stringify({cacheBeforeEditing,cacheAfterEditing,cacheAfterRotation}));
  console.log(JSON.stringify({poseEditing:'passed',fineDegrees:0.1,coarseDegrees:1}));
  const pixels=await page.evaluate(()=>globalThis.overlaySmoke.verifyCachePixels());
  assert.equal(pixels.cachedError,0);assert.equal(pixels.previewError,0);assert.equal(pixels.released,true);
  console.log(JSON.stringify({gpuCachePixels:pixels}));
  if (!process.argv.includes('--pose-only')) {
  await page.evaluate(()=>globalThis.overlaySmoke.selectContributor());
  await page.locator('#trackingOverlayInspect').click();
  await page.waitForFunction(()=>document.querySelector('#trackingInspector').hidden===false);
  await page.waitForFunction(()=>globalThis.overlaySmoke.ready());
  const restored=await page.evaluate(()=>globalThis.overlaySmoke.restoreComposite());
  assert.equal(restored.selected,null);assert.equal(restored.inspectDisabled,true);
  const currentBounds=await page.evaluate(()=>({width:overlaySmoke.bounds().width,height:overlaySmoke.bounds().height}));
  const restarted=await page.evaluate(p=>globalThis.overlaySmoke.cancelThenRun(p),point);
  assert.match(restarted.info,/WebGPU.*ms\/Frame/);assert.equal(restarted.width,currentBounds.width);assert.equal(restarted.height,currentBounds.height);
  const tiled=await page.evaluate(p=>globalThis.overlaySmoke.tiled(p),point);
  assert.equal(tiled.width,restarted.width);assert.equal(tiled.height,restarted.height);assert.ok(tiled.tiles>1);assert.match(tiled.info,/Kacheln/);
  await page.locator('#trackingOverlayPane').screenshot({path:'benchmarks/overlay-ui-tiles.png'});
  console.log(JSON.stringify({overlayRestartAndTiles:{restarted,tiled}}));
  if (!process.argv.includes('--overlay-only')) {
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
  }
  }
  assert.deepEqual(errors,[]);
}finally{await browser?.close();await new Promise(r=>server.close(r));}
