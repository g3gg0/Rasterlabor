import assert from 'node:assert/strict';
import http from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { chromium } from 'playwright';
const root = resolve('dist');
const hook = `
globalThis.fineProbe = {
 ready:()=>Boolean(calibration && videoInfo && !taskBusy && !trackingPreviewBusy),
 prepare(){
  trackingMosaicRequest++;clearTimeout(trackingMosaicTimer);
  const valid=trackingPath.filter(entry=>entry.pose && entry.success!==false);
  trackingPath=valid.slice(Math.floor(valid.length/2),Math.floor(valid.length/2)+8);
  trackingDataset={...trackingDataset,matchNetwork:emptyMatchNetwork(currentNetworkGeometryKey())};
  element('trackingOverlayMaxFrames').value='1';
  updateControls();activateTrackingTab('optimization');
  return trackingPath.map(entry=>entry.frame);
 },
 report:()=>trackingDataset.fineAlignment,
 state:()=>({busy:fineBusy,status:element('fineStatus').textContent,selected:overlayPoseDraft?.frame,
   contributors:overlayContributors.map(item=>item.entry.frame),cache:frameReader.cacheStats(),
   poses:trackingPath.map(entry=>({...entry.pose})),anchors:currentMatchNetwork().pairs.length}),
 async controlled(){
  trackingMosaicRequest++;clearTimeout(trackingMosaicTimer);await frameReader.waitUntilIdle();
  const entry=trackingPath[0];
  const canvas=new OffscreenCanvas(calibration.maps.outputWidth,calibration.maps.outputHeight),ctx=canvas.getContext('2d');
  const pixels=ctx.createImageData(canvas.width,canvas.height);
  for(let y=0;y<canvas.height;y++)for(let x=0;x<canvas.width;x++){
    const v=110+24*Math.sin(x*.24+Math.sin(y*.1)*2)+27*Math.cos(y*.32+Math.sin(x*.07))+20*Math.sin(x*.4+y*.27);
    const i=4*(y*canvas.width+x);pixels.data[i]=pixels.data[i+1]=pixels.data[i+2]=v;pixels.data[i+3]=255;
  }
  ctx.putImageData(pixels,0,0);const decoded={bitmap:await createImageBitmap(canvas)};
  const original=frameReader.read.bind(frameReader);
  const ids=trackingPath.map(item=>item.frame), base={...entry.pose};
  trackingPath=ids.map((frame,i)=>({...entry,frame,sharpness:{score:i?0:100},
    pose:{x:base.x+(i?2+(i%3):0),y:base.y+(i?-.8:0),rotation:base.rotation}}));
  trackingDataset={...trackingDataset,matchNetwork:emptyMatchNetwork(currentNetworkGeometryKey())};
  frameReader.read=async(frame,options={})=>options.output==='rgba'?{width:canvas.width,height:canvas.height,data:new Uint8ClampedArray(pixels.data)}:{bitmap:await createImageBitmap(decoded.bitmap)};
  this.cleanup=()=>{frameReader.read=original;decoded.bitmap.close();};
  setWebGpuSelection('none');fineUi.restore(null);updateControls();activateTrackingTab('optimization');
  return ids;
 },
 stopMosaic(){trackingMosaicRequest++;clearTimeout(trackingMosaicTimer);}
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
try {
  browser = await chromium.launch({ channel: 'msedge', headless: true });
  const page = await browser.newPage({ viewport: { width: 1450, height: 1100 } });
  page.setDefaultTimeout(60000);
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('http://127.0.0.1:' + server.address().port);
  await page.locator('#calibrationFile').setInputFiles('20260918_004341.zip');
  await page.waitForFunction(() => !document.querySelector('#exportButton').disabled);
  await page.locator('#videoFile').setInputFiles('20260918_004341.mp4');
  await page.waitForFunction(() => fineProbe.ready());
  await page.locator('[data-workflow="tracking"]').click();
  const frames = await page.evaluate(() => fineProbe.prepare());
  await page.locator('#fineStart').click();
  await page.waitForFunction(() => !fineProbe.state().busy);
  const real = await page.evaluate(() => ({ report: fineProbe.report(), state: fineProbe.state() }));
  assert.equal(real.report.rows.length, frames.length);
  assert.ok(real.report.rows.every(row => row.attempts > 0));
  console.log(JSON.stringify({ realFrames: frames, status: real.state.status,
    pairs: real.report.pairs, results: real.report.rows.map(row => ({ frame: row.frame, attempts: row.attempts, accepted: row.accepted, score: row.score })) }));
  const controlledFrames = await page.evaluate(() => fineProbe.controlled());
  console.log('Controlled prepared');
  const before = await page.evaluate(() => fineProbe.state());
  await page.locator('#fineStart').click();
  await page.waitForFunction(() => !fineProbe.state().busy);
  console.log('Controlled measured');
  const measured = await page.evaluate(() => ({ report: fineProbe.report(), state: fineProbe.state() }));
  assert.ok(measured.report.rows.every(row => row.accepted > 0), JSON.stringify(measured));
  assert.equal(await page.locator('#fineApply').isEnabled(), true);
  await page.locator('#fineApply').click();
  console.log('Applied');
  const applied = await page.evaluate(() => fineProbe.state());
  assert.ok(applied.anchors > 0);
  assert.ok(applied.poses.some((pose, i) => Math.hypot(pose.x - before.poses[i].x, pose.y - before.poses[i].y) > 1));
  await page.evaluate(() => fineProbe.stopMosaic());
  await page.locator('#fineFrames').click();
  assert.equal(await page.locator('#fineList [role="option"]').count(), controlledFrames.length);
  assert.equal(await page.locator('#fineList .fine-bar>span').count(), controlledFrames.length);
  const target = controlledFrames.at(-1);
  await page.locator('#fineFilter').fill(String(target));
  assert.equal(await page.locator('#fineList [role="option"]').count(), 1);
  await page.locator(`#fineList [data-frame="${target}"]`).click();
  console.log('Selecting '+target);
  await page.waitForFunction(frame => fineProbe.state().selected === frame, target);
  assert.equal(await page.locator('#trackingOverlayPane').isVisible(), true);
  assert.equal(await page.locator('#trackingOverlayFrame').inputValue(), String(target));
  const selected = await page.evaluate(() => fineProbe.state());
  assert.ok(selected.contributors.includes(target));
  await page.locator('#trackingTabOptimization').click();
  await page.locator('#fineUndo').click();
  assert.deepEqual((await page.evaluate(() => fineProbe.state())).poses, before.poses);
  assert.deepEqual(errors, []);
  await writeFile('benchmarks/fine-alignment-browser.json', JSON.stringify({ real, controlled: measured, selected, errors }, null, 2));
  console.log(JSON.stringify({ controlledStatus: measured.state.status, forcedSelection: target, errors }));
} finally { await browser?.close(); await new Promise(r => server.close(r)); }
