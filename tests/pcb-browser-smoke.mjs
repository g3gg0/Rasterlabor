// Run after `npm run build` and `npm start` on http://127.0.0.1:4173.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { chromium } from '@playwright/test';

const browser = await chromium.launch({
  executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  headless: true
});
try {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const bundle = await readFile(new URL('../dist/app.js', import.meta.url), 'utf8');
  await page.route('**/app.js', route => route.fulfill({
    contentType: 'text/javascript', body: `${bundle}\nwindow.__pcbFixture = {
      async gpuStatus() { return { selection: getWebGpuSelection(),
        worker: await computer.call('patch-gpu-status') }; },
      async nativeCache() {
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = 16;
        let decoded = 0;
        const decoder = { async call() {
          decoded++;
          return { frame: new VideoFrame(canvas, { timestamp: 0 }), orientation: {
            a: 1, b: 0, c: 0, d: 1, translateX: 0, translateY: 0 } };
        } };
        const reader = new FrameReader({ getDecoder: () => decoder, getMaps: () => null });
        try {
          for (let index = 0; index < 2; index++) {
            const result = await reader.read(0, { gpu: true, output: 'native', measureSharpness: false });
            result.frame.close();
            if (!index) await reader.releaseRenderer();
          }
          return { decoded, cached: reader.cacheStats().frames };
        } finally { await reader.dispose(); }
      },
      prepare() {
        const size = 192;
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = size;
        const context = canvas.getContext('2d');
        const image = context.createImageData(size, size);
        for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
          const trace = Math.abs(y - 38 - 8 * Math.sin(x / 13)) < 2 ||
            Math.abs(x - 95 - 9 * Math.sin(y / 12)) < 2 ||
            Math.abs(y - 128 + 7 * Math.sin(x / 8)) < 2;
          const value = 70 + (trace ? 100 : 0) + 8 * Math.sin(x * .35 + y * .17);
          const offset = 4 * (y * size + x);
          image.data[offset] = image.data[offset + 1] = image.data[offset + 2] = value;
          image.data[offset + 3] = 255;
        }
        context.putImageData(image, 0, 0);
        frameReader.read = async () => {
          await new Promise(resolve => setTimeout(resolve, 20));
          return { bitmap: await createImageBitmap(canvas) };
        };
        videoInfo = { width: size, height: size, frameCount: 3 };
        calibration = { field: { width: size, height: size },
          maps: { outputWidth: size, outputHeight: size, origin: [0, 0],
            valid: new Uint8Array(size * size).fill(1) } };
        trackingPath = [0, 1, 2].map(frame => ({ frame, mode: 'window', success: true,
          pose: { x: frame * 2, y: 0, rotation: 0 } }));
        trackingDataset = { path: trackingPath, video: videoInfo };
        updatePcbControls();
      },
      poses() { return trackingPath.map(entry => ({ ...entry.pose })); },
      pathView() { drawTrackingPath(); return { zoom: pathZoom, pan: { ...pathPan } }; },
      pcbAfterTask() {
        taskBusy = true; updatePcbControls();
        const blocked = document.querySelector('#pcbRealignStart').disabled;
        taskBusy = false; updateControls();
        return { blocked, enabled: !document.querySelector('#pcbRealignStart').disabled };
      },
      pairView() { return { zoom: pcbPairView.zoom, panX: pcbPairView.panX,
        panY: pcbPairView.panY, images: pcbPairView.images.length }; },
      firstCell() { return pcbCellTargets[0]; },
      selectedCell() { return pcbCellTargets.find(target => target.cellId === pcbSelectedCellId); },
      cellView() { return { ...pcbCellView }; },
      draftCount() { return trackingDataset?.pcbRealignmentDraft?.matches?.length ?? 0; },
      roundtrip() { restoreTracking(JSON.parse(JSON.stringify(trackingForExport()))); },
      async focusedRefit() {
        this.prepare();
        pathSelection = { x: 0, y: 0 };
        await refitTrackingPath();
        return { status: element('trackingPathRefitInfo').textContent,
          matches: pathRefitProposal?.refitMatches, corrected: pathRefitProposal?.byFrame.size,
          ready: pathRefitProposal?.overlayReady, poses: this.poses() };
      },
      disconnected() {
        this.prepare();
        const read = frameReader.read;
        const blank = document.createElement('canvas');
        blank.width = blank.height = 192;
        const context = blank.getContext('2d');
        context.fillStyle = '#646464'; context.fillRect(0, 0, 192, 192);
        frameReader.read = async frame => frame === 2 ? { bitmap: await createImageBitmap(blank) } : read(frame);
        trackingPath.forEach((entry, index) => { entry.pose.x = index * 5; });
      }
    };`
  }));
  await page.goto('http://127.0.0.1:4173/');
  await page.waitForFunction(() => !document.querySelector('#globalGpuAdapter').disabled);
  assert.deepEqual(await page.evaluate(() => window.__pcbFixture.nativeCache()), { decoded: 1, cached: 1 });
  await page.locator('.image-adjustments > summary').click();
  const selectedGpu = await page.locator('#globalGpuAdapter').inputValue();
  assert.notEqual(selectedGpu, 'none');
  assert.match(await page.locator('#globalGpuAdapter').textContent(), /NVIDIA|AMD|Intel|WebGPU-Adapter/i);
  await page.locator('#globalGpuAdapter').selectOption('none');
  assert.equal((await page.evaluate(() => window.__pcbFixture.gpuStatus())).worker.available, false);
  await page.locator('#globalGpuAdapter').selectOption(selectedGpu);
  assert.equal((await page.evaluate(() => window.__pcbFixture.gpuStatus())).worker.available, true);
  await page.locator('.image-adjustments > summary').click();
  await page.locator('[data-workflow="tracking"]').click();
  await page.locator('#trackingTabOptimization').click();
  assert.equal(await page.locator('#trackingOptimizationPanel').isVisible(), true);
  assert.equal(await page.locator('#pcbRealignStart').isVisible(), true);
  await page.evaluate(() => window.__pcbFixture.prepare());
  assert.deepEqual(await page.evaluate(() => window.__pcbFixture.pcbAfterTask()), { blocked: true, enabled: true });
  assert.equal((await page.evaluate(() => window.__pcbFixture.pathView())).zoom, 1);
  const pathBox = await page.locator('#trackingPathCanvas').boundingBox();
  await page.mouse.move(pathBox.x + pathBox.width / 2, pathBox.y + pathBox.height / 2);
  await page.mouse.wheel(0, -600);
  assert.ok((await page.evaluate(() => window.__pcbFixture.pathView())).zoom > 1);
  await page.mouse.down();
  await page.mouse.move(pathBox.x + pathBox.width * 0.9, pathBox.y + pathBox.height * 0.9);
  await page.mouse.up();
  await page.mouse.wheel(0, 10000);
  assert.deepEqual(await page.evaluate(() => window.__pcbFixture.pathView()),
    { zoom: 1, pan: { x: 0, y: 0 } });
  const posesBeforeGpuSwitch = await page.evaluate(() => window.__pcbFixture.poses());
  await page.locator('.image-adjustments > summary').click();
  await page.locator('#globalGpuAdapter').selectOption('none');
  await page.locator('#globalGpuAdapter').selectOption(selectedGpu);
  await page.locator('.image-adjustments > summary').click();
  assert.deepEqual(await page.evaluate(() => window.__pcbFixture.poses()), posesBeforeGpuSwitch);
  await page.locator('#pcbFftCellSize').fill('64');
  await page.locator('#pcbSearchRadius').fill('12');
  await page.locator('#pcbRealignStart').click();
  await page.waitForFunction(() => !document.querySelector('#pcbRealignCancel').disabled, null, { timeout: 5000 });
  await page.locator('#pcbRealignPause').click();
  await page.waitForFunction(() => document.querySelector('#pcbRealignStatus').textContent.includes('Pausiert.'),
    null, { timeout: 30000 });
  if (await page.evaluate(() => window.__pcbFixture.draftCount()) === 0) {
    await page.locator('#pcbRealignStep').click();
    await page.waitForFunction(() => window.__pcbFixture.draftCount() >= 1, null, { timeout: 30000 });
  }
  assert.equal(await page.locator('#pcbRealignResume').isEnabled(), true);
  assert.equal(await page.evaluate(() => window.__pcbFixture.draftCount()), 1);
  assert.equal(await page.locator('#pcbPairSelect option').count(), 1);
  await page.locator('#pcbRealignDetails > summary').click();
  assert.match(await page.locator('#pcbPairSummary').textContent(), /Paarmessung/);
  await page.waitForFunction(() => window.__pcbFixture.pairView().images === 2, null, { timeout: 5000 });
  await page.locator('#pcbRealignDetails > summary').click();
  await page.locator('#pcbRealignCancel').click();
  await page.waitForFunction(() => document.querySelector('#pcbRealignCancel').disabled, null, { timeout: 5000 });
  await page.evaluate(() => window.__pcbFixture.roundtrip());
  assert.equal(await page.evaluate(() => window.__pcbFixture.draftCount()), 1);
  await page.locator('#pcbRealignStart').click();
  await page.waitForFunction(() => !document.querySelector('#pcbRealignCancel').disabled, null, { timeout: 5000 });
  await page.waitForFunction(() => document.querySelector('#pcbRealignCancel').disabled, null, { timeout: 30000 });
  const status = await page.locator('#pcbRealignStatus').textContent();
  assert.equal(errors.length, 0, errors.join('\n'));
  assert.equal(await page.locator('#pcbPairSelect option').count(), 2);
  await page.locator('#pcbRealignDetails > summary').click();
  await page.locator('#pcbPairSelect').selectOption('1');
  assert.match(await page.locator('#pcbPairSummary').textContent(), /Paarmessung angenommen/);
  await page.waitForFunction(() => {
    const canvas = document.querySelector('#pcbPairAfter');
    const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
    for (let index = 0; index < pixels.length; index += 400)
      if (pixels[index] < 190) return true;
    return false;
  }, null, { timeout: 5000 });
  const cellCanvas = page.locator('#pcbCellCanvas');
  const cellBox = await cellCanvas.boundingBox();
  await page.mouse.move(cellBox.x + cellBox.width / 2, cellBox.y + cellBox.height / 2);
  await page.mouse.wheel(0, -350);
  assert.ok((await page.evaluate(() => window.__pcbFixture.cellView())).zoom > 1);
  await cellCanvas.dblclick();
  assert.equal((await page.evaluate(() => window.__pcbFixture.cellView())).zoom, 1);
  await page.mouse.move(cellBox.x + cellBox.width / 2, cellBox.y + cellBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(cellBox.x + cellBox.width * 0.8, cellBox.y + cellBox.height * 0.8);
  await page.mouse.up();
  assert.deepEqual(await page.evaluate(() => {
    const { panX, panY } = window.__pcbFixture.cellView(); return { panX, panY };
  }), { panX: 0, panY: 0 });
  const cellTarget = await page.evaluate(() => window.__pcbFixture.firstCell());
  const cellX = cellBox.x + cellTarget.x * cellBox.width / 512;
  const cellY = cellBox.y + cellTarget.y * cellBox.height / 320;
  await page.mouse.click(cellX, cellY);
  assert.equal(await page.locator('#pcbCellDetails').isVisible(), true);
  assert.match(await page.locator('#pcbCellDetailsStatus').textContent(), /Zelle/);
  await page.locator('#pcbCellDetails').screenshot({ path: 'benchmarks/pcb-cell-details.png' });
  const cellBoxAfter = await cellCanvas.boundingBox();
  const deselectTarget = await page.evaluate(() => window.__pcbFixture.selectedCell());
  await cellCanvas.click({ position: { x: deselectTarget.x * cellBoxAfter.width / 512,
    y: deselectTarget.y * cellBoxAfter.height / 320 } });
  assert.equal(await page.locator('#pcbCellDetails').isVisible(), false);
  const pairCanvas = page.locator('#pcbPairAfter');
  const pairBox = await pairCanvas.boundingBox();
  await page.mouse.move(pairBox.x + pairBox.width * 0.7, pairBox.y + pairBox.height * 0.4);
  await page.mouse.wheel(0, -350);
  assert.ok((await page.evaluate(() => window.__pcbFixture.pairView())).zoom > 1);
  await pairCanvas.dblclick();
  assert.equal((await page.evaluate(() => window.__pcbFixture.pairView())).zoom, 1);
  await page.locator('#pcbRealignDetails').screenshot({ path: 'benchmarks/pcb-diagnostics.png' });
  assert.equal(await page.locator('#pcbRealignApply').isEnabled(), true, status);
  await page.locator('#pcbRealignApply').click();
  const applied = await page.evaluate(() => window.__pcbFixture.poses());
  assert.equal(await page.locator('#pcbRealignStart').isEnabled(), true);
  assert.ok(Math.abs(applied[1].x) < 1, JSON.stringify({ status, applied }));
  assert.ok(Math.abs(applied[2].x) < 1, JSON.stringify({ status, applied }));
  assert.match(status, /1\/1 Zwischenframes lokal bestaetigt/);
  await page.locator('#pcbRealignUndo').click();
  const undone = await page.evaluate(() => window.__pcbFixture.poses());
  assert.equal(undone[1].x, 2);
  assert.equal(undone[2].x, 4);
  await page.evaluate(() => window.__pcbFixture.disconnected());
  await page.locator('#pcbSpacing').fill('0.02');
  await page.locator('#pcbRealignStart').click();
  await page.waitForFunction(() => document.querySelector('#pcbRealignStatus').textContent.includes('Komponenten;'),
    null, { timeout: 30000 });
  assert.equal(await page.locator('#pcbRealignApply').isEnabled(), true);
  assert.ok(await page.locator('#pcbPairSelect option').count() >= 2);
  assert.match(await page.locator('#pcbRealignDiagnostics').textContent(), /"components"/);
  assert.deepEqual(await page.evaluate(() => window.__pcbFixture.poses().map(pose => pose.x)), [0, 5, 10]);
  await page.locator('#pcbRealignApply').click();
  assert.ok((await page.evaluate(() => window.__pcbFixture.poses()))[2].x < 10);
  await page.locator('#globalGpuAdapter').selectOption('none');
  const refit = await page.evaluate(() => window.__pcbFixture.focusedRefit());
  assert.equal(refit.matches, 3, refit.status);
  assert.equal(refit.corrected, 3, refit.status);
  assert.equal(refit.ready, true, refit.status);
  assert.deepEqual(refit.poses.map(pose => pose.x), [0, 2, 4]);
  console.log(JSON.stringify({ status, applied, undone, errors }));
} finally {
  await browser.close();
}
