import assert from 'node:assert/strict';
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { chromium } from 'playwright';

// Run after building, with a video containing the reported frame range:
// node tests/frame-cache.browser.mjs [video.mp4] [firstFrame] [count] [calibration.zip]
const videoFile = resolve(process.argv[2] ?? '20260918_004341.mp4');
const calibrationFile = process.argv[5] ? resolve(process.argv[5]) : null;
const firstFrame = Number(process.argv[3] ?? 27990), count = Number(process.argv[4] ?? 51);
const root = resolve('dist');
const hook = `
globalThis.frameCacheSmoke = {
  ready: () => Boolean(videoInfo && !taskBusy && !trackingPreviewBusy),
  async run(first, count) {
    await frameReader.dispose();
    const originalCall = decoder.call.bind(decoder);
    let reads = 0;
    decoder.call = (type, ...args) => {
      if (type === 'frame' || type === 'native-frame') reads++;
      return originalCall(type, ...args);
    };
    const pixelSample = frame => {
      const canvas = new OffscreenCanvas(32, 32);
      const context = canvas.getContext('2d', {willReadFrequently: true});
      context.drawImage(frame, 100, 100, 32, 32, 0, 0, 32, 32);
      return context.getImageData(0, 0, 32, 32).data;
    };
    let originalPixels;
    const started = performance.now();
    try {
      for (let index = first; index < first + count; index++) {
        const result = await frameReader.read(index, {output: 'native', measureSharpness: false});
        try { if (index === 28016) originalPixels = pixelSample(result.frame); }
        finally { result.frame.close(); }
      }
      const filled = frameReader.cacheStats();
      const decodeReads = reads;
      let cacheHits = 0, pixelDifference = 0;
      for (let index = first + count - 1; index >= first; index--) {
        const result = await frameReader.read(index, {output: 'native', measureSharpness: false});
        try {
          if (result.frameTiming.cacheHit) cacheHits++;
          if (index === 28016 && originalPixels) {
            const cachedPixels = pixelSample(result.frame);
            for (let i = 0; i < cachedPixels.length; i++) {
              pixelDifference = Math.max(pixelDifference, Math.abs(cachedPixels[i] - originalPixels[i]));
            }
          }
        } finally { result.frame.close(); }
      }
      let rgba;
      if (calibration) {
        const firstImage = await frameReader.read(first, {gpu: true, rectified: true, output: 'rgba', measureSharpness: false});
        const cachedImage = await frameReader.read(first, {gpu: false, rectified: true, output: 'rgba', measureSharpness: false});
        let maximumDifference = 0;
        for (let i = 0; i < firstImage.data.length; i += 701) {
          maximumDifference = Math.max(maximumDifference, Math.abs(firstImage.data[i] - cachedImage.data[i]));
        }
        rgba = {width: cachedImage.width, height: cachedImage.height,
          bytes: cachedImage.data?.length, cacheHit: cachedImage.frameTiming.cacheHit, maximumDifference};
      }
      return {first, count, filled, decodeReads, readsAfterReuse: reads, cacheHits, rgba,
        pixelDifference, seconds: (performance.now() - started) / 1000};
    } finally { decoder.call = originalCall; await frameReader.dispose(); }
  }
};`;
const server = http.createServer(async (request, response) => {
  try {
    const path = resolve(root, '.' + (request.url === '/' ? '/index.html' : request.url));
    if (!path.startsWith(root + sep)) throw new Error('path');
    let bytes = await readFile(path);
    if (request.url === '/app.js') bytes = Buffer.concat([bytes, Buffer.from(hook)]);
    response.setHeader('Content-Type', {'.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css'}[extname(path)] ?? 'application/octet-stream');
    response.end(bytes);
  } catch { response.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({channel: 'msedge', headless: true});
  const page = await browser.newPage();
  page.setDefaultTimeout(120000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('http://127.0.0.1:' + server.address().port);
  if (calibrationFile) {
    await page.locator('#calibrationFile').setInputFiles(calibrationFile);
    await page.waitForFunction(() => !document.querySelector('#exportButton').disabled);
  }
  await page.locator('#videoFile').setInputFiles(videoFile);
  await page.waitForFunction(() => globalThis.frameCacheSmoke.ready());
  const result = await page.evaluate(([first, count]) => globalThis.frameCacheSmoke.run(first, count), [firstFrame, count]);
  console.log(JSON.stringify(result));
  assert.equal(result.decodeReads, count);
  assert.equal(result.readsAfterReuse, count);
  assert.equal(result.cacheHits, count);
  assert.equal(result.filled.frames, count);
  assert.ok(result.pixelDifference <= 1, JSON.stringify(result));
  if (calibrationFile) {
    assert.equal(result.rgba.bytes, result.rgba.width * result.rgba.height * 4);
    assert.equal(result.rgba.cacheHit, true);
    assert.ok(result.rgba.maximumDifference <= 1, JSON.stringify(result.rgba));
  }
  assert.deepEqual(errors, []);
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
