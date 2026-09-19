import { chromium } from 'playwright';
import { readFile } from 'node:fs/promises';
import http from 'node:http';
import assert from 'node:assert/strict';

const server = http.createServer(async (req, res) => {
  try {
    const path = req.url === '/' ? '/index.html' : req.url;
    let body = await readFile(new URL(`../dist${path}`, import.meta.url));
    if (path === '/app.js') body = Buffer.concat([body, Buffer.from(`
      window.pathFixture = () => {
        calibration = { field: { width: 100, height: 80 }, maps: { outputWidth: 100, outputHeight: 80, origin: [0,0], valid: new Uint8Array(8000).fill(1) } };
        trackingPath = [0,1].map(frame => ({frame, mode:'window', pose:{x:frame*10,y:0,rotation:0},raw:{x:frame*10,y:0,rotation:0}}));
        videoInfo = {name:'fixture'};
        trackingImageMask = createPatchMask(100,80); trackingImageMask.data.fill(MASK_SEARCH);
        decoder.call = async () => ({bitmap: await createImageBitmap(new OffscreenCanvas(100,80))});
        computer.call = async () => ({width:100,height:80,data:new Uint8ClampedArray(32000).fill(255)});
        drawTrackingPath();
        return pathProject({ x: 5, y: 25 });
      };
      window.overlayState = () => ({zoom:overlayZoom,pan:overlayPan,width:pathOverlay.width});
    `)]);
    res.setHeader('Content-Type', path.endsWith('.js') ? 'text/javascript' : path.endsWith('.css') ? 'text/css' : path.endsWith('.html') ? 'text/html' : 'application/octet-stream');
    res.end(body);
  } catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({headless:true, args:['--no-sandbox', '--single-process', '--no-zygote', '--use-gl=angle', '--use-angle=swiftshader']});
  const page = await browser.newPage();
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.locator('[data-workflow="tracking"]').click();
  const point = await page.evaluate(() => window.pathFixture());
  const bounds = await page.locator('#trackingPathCanvas').boundingBox();
  await page.mouse.move(bounds.x + point.x, bounds.y + point.y);
  assert.match(await page.locator('#trackingPathInfo').textContent(), /2 Bilder/);
  await page.mouse.click(bounds.x + point.x, bounds.y + point.y);
  await page.waitForFunction(() => document.querySelector('#trackingOverlayInfo').textContent.includes('Gleich gewichtet, deckend'));
  assert.ok((await page.evaluate(() => window.overlayState())).width > 0);
  const overlay = page.locator('#trackingOverlayCanvas'); await overlay.scrollIntoViewIfNeeded();
  const box = await overlay.boundingBox();
  await page.mouse.move(box.x+100,box.y+100); await page.mouse.wheel(0,-500);
  await page.waitForFunction(() => window.overlayState().zoom > 1);
  await page.mouse.down(); await page.mouse.move(box.x+140,box.y+130); await page.mouse.up();
  assert.notDeepEqual((await page.evaluate(() => window.overlayState())).pan,{x:0,y:0});
  await page.locator('#trackingOverlayFit').click();
  assert.equal((await page.evaluate(() => window.overlayState())).zoom,1);
  assert.deepEqual(errors,[]);
  console.log('Browser: hover, overlay loading, zoom, pan, fit passed.');
} finally { await browser?.close(); server.close(); }
