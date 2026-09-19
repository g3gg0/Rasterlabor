import assert from 'node:assert/strict';
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { chromium } from 'playwright';
import { createSpline } from '../src/spline.js';
import { buildMaps } from '../src/maps.js';
import { exportCalibration, importCalibration } from '../src/format.js';
import { unzipSync, zipSync, strToU8 } from 'fflate';
import { validateTracking } from '../src/tracking-data.js';

const root = resolve('dist');
const server = http.createServer(async (req, res) => {
  try {
    const path = resolve(root, '.' + (req.url === '/' ? '/index.html' : req.url));
    if (!path.startsWith(root + sep)) throw new Error('Invalid path');
    const data = await readFile(path);
    res.setHeader('Content-Type', { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' }[extname(path)] || 'application/octet-stream');
    res.end(data);
  } catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ channel: 'msedge', headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('dialog', dialog => dialog.accept());
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const sharpness = { score: 123.5, mean: 140, minimum: 80, maximum: 220, contrast: 35,
    samples: 1000, sampleWidth: 192, sampleHeight: 108,
    method: 'median-laplacian-variance-9-downsampled-regions', accelerator: 'WebGPU' };
  const field = createSpline(48, 40, 24, [1, 0, 0, 1, 0, 0]);
  const frames = [{ id: 0, timestamp: 0, enabled: true, accepted: true, role: 'train', sharpness, points: [
    { x: 8, y: 8, col: 0, row: 0, confidence: 1 }
  ] }];
  const calibration = { field, referenceId: 0, step: 7, version: 1, quality: 'provisional',
    metrics: { validation: { count: 0 }, train: { count: 1 } }, poses: {}, parameters: { approxStep: 7 } };
  calibration.maps = await buildMaps(calibration, frames);
  const tracking = { format: 'rasterlabor-xyr-tracking', model_version: 1,
    video: { name: 'test.mp4', width: 48, height: 40 },
    path: [{ frame: 12, timestamp: 400000, sharpness, pose: { x: 1.25, y: -2.5, rotation: 0.01 } }] };
  const legacy = { format: tracking.format, model_version: 1, options: { trackingMode: 'window' },
    rectangle: { x: null, y: null, width: 10, height: 10 },
    path: [{ frame: 12, timestamp_us: 400000, x_px: 1.25, y_px: -2.5, rotation_deg: 90,
      raw_x_px: 2, raw_y_px: -3, raw_rotation_deg: 45, patches: 1, rms_px: null }] };
  for (const dataset of [tracking, legacy, null]) {
    const files = unzipSync(exportCalibration(calibration, frames, { name: 'test.mp4' }, { gridMm: null, approxStep: 7 }, ''));
    if (dataset) files['tracking.json'] = strToU8(JSON.stringify(dataset));
    const bytes = zipSync(files);
    await page.locator('#calibrationFile').setInputFiles({ name: 'test.zip', mimeType: 'application/zip', buffer: Buffer.from(bytes) });
    await page.waitForFunction(() => document.querySelector('#exportButton').disabled === false);
    assert.equal(await page.locator('#lensDataStatus').evaluate(el => el.classList.contains('present')), true);
    assert.equal(await page.locator('#trackingDataStatus').evaluate(el => el.classList.contains('present')), Boolean(dataset));
    assert.equal(await page.locator('#brightnessDataStatus').evaluate(el => el.classList.contains('present')), false);
    const downloadPromise = page.waitForEvent('download');
    await page.locator('#exportButton').click();
    const download = await downloadPromise;
    const restored = importCalibration(new Uint8Array(await readFile(await download.path())));
    assert.deepEqual(restored.observations[0].sharpness, sharpness);
    assert.deepEqual(restored.tracking?.path ?? null, validateTracking(dataset)?.path ?? null);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  if (process.argv[2]) {
    page.setDefaultTimeout(180000);
    await page.locator('#calibrationFile').setInputFiles(resolve(process.argv[2]));
    await page.waitForFunction(() => document.querySelector('#exportButton').disabled === false &&
      document.querySelector('#fieldVersion').textContent.startsWith('v9'));
    assert.equal(await page.locator('#trackingDataStatus').evaluate(el => el.classList.contains('present')), true);
    assert.match(await page.locator('#trackingSummary').textContent(), /6509 Frames/);
    console.log('Actual v9 rescue ZIP: calibration and all 6509 tracking frames loaded in browser.');
  }
  assert.deepEqual(errors, []);
  console.log('Browser: ZIP import/export with and without tracking, LEDs and 390px layout passed.');
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
