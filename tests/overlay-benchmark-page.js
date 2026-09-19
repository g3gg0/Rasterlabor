import { WorkerClient } from '../src/rpc.js';
import { frameGeometry, accumulateFrame, averagedFrames } from '../src/path-support.js';

export async function runBenchmark(mode, video, archive) {
  const adapter = await navigator.gpu?.requestAdapter();
  const info = adapter?.info;
  const hardware = info ? { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description,
    fallback: info.isFallbackAdapter, maxTextureDimension2D: adapter.limits.maxTextureDimension2D } : null;
  const decoder = new WorkerClient(mode === 'before' ? '/benchmarks/baseline-decoder-worker.js' : '/decoder-worker.js');
  const computer = new WorkerClient('/compute-worker.js');
  try {
    const videoInfo = await decoder.call('open', { file: video });
    const bytes = new Uint8Array(await archive.arrayBuffer());
    const imported = await computer.call('import', { bytes }, [bytes.buffer]);
    const { calibration, tracking } = imported;
    if (mode === 'reader') {
      const {benchmarkFrameReader} = await import('./frame-reader-benchmark.js');
      return {hardware, ...await benchmarkFrameReader(decoder,computer,calibration)};
    }

    if (mode === 'verify') {
      const { runGpuChecks } = await import('./overlay-gpu-checks.js');
      return { hardware, checks: await runGpuChecks(decoder, computer, calibration) };
    }
    const entries = tracking.path.filter(entry => entry.frame >= 970 && entry.frame < 982);
    const geometries = entries.map(entry => frameGeometry(entry, calibration.field, calibration.maps));
    const corners = geometries.flatMap(g => g.corners);
    const minX = Math.floor(Math.min(...corners.map(p => p.x))), minY = Math.floor(Math.min(...corners.map(p => p.y)));
    const width = Math.ceil(Math.max(...corners.map(p => p.x))) - minX;
    const height = Math.ceil(Math.max(...corners.map(p => p.y))) - minY;
    if (mode === 'tiled') {
      const { WebGpuOverlay } = await import('../src/webgpu-overlay.js');
      const { renderTiledOverlay, closeOverlayTiles } = await import('../src/overlay-tiles.js');
      const largeWidth=9000,largeHeight=11000;
      if (!await WebGpuOverlay.needsTiles(calibration.maps,largeWidth,largeHeight)) throw new Error('Test must exceed the old budget');
      const start=performance.now();
      const tiled=await renderTiledOverlay({maps:calibration.maps,width:largeWidth,height:largeHeight,minX,minY,geometries,
        decode:index=>decoder.call('native-frame',{index})});
      try {return {hardware,output:[tiled.width,tiled.height],tiles:tiled.tiles.length,framePasses:tiled.framePasses,
        totalMs:performance.now()-start,oldEstimatedMiB:(largeWidth*largeHeight*20+calibration.maps.outputWidth*calibration.maps.outputHeight*12)/1024**2};}
      finally {closeOverlayTiles(tiled);}
    }
    const result = { mode, hardware, video: videoInfo.name, source: [videoInfo.width, videoInfo.height],
      output: [width, height], frames: entries.map(e => e.frame), runs: [] };
    const conversion = new OffscreenCanvas(videoInfo.width, videoInfo.height);
    const conversionContext = conversion.getContext('2d', { willReadFrequently: true });
    for (let run = 0; run < 3; run++) {
      const warm = await decoder.call('frame', { index: 969 }); warm.bitmap.close();
      if (mode === 'after') {
        const { WebGpuOverlay } = await import('../src/webgpu-overlay.js');
        const setup = performance.now();
        const overlay = await WebGpuOverlay.create(calibration.maps, width, height, minX, minY);
        const setupMs = performance.now() - setup;
        const samples = [];
        try {
          for (const geometry of geometries) {
            const start = performance.now();
            const decoded = await decoder.call('native-frame', { index: geometry.entry.frame });
            const decodedAt = performance.now();
            try { await overlay.addFrame(decoded.frame, decoded.orientation, geometry); }
            finally { decoded.frame.close(); }
            const end = performance.now();
            samples.push({ frame: geometry.entry.frame, decodeMs: decodedAt - start,
              gpuMs: end - decodedAt, totalMs: end - start });
          }
          const start = performance.now();
          const bitmap = await overlay.finish();
          const display = new OffscreenCanvas(width, height);
          display.getContext('2d').drawImage(bitmap, 0, 0); bitmap.close();
          const finalizeMs = performance.now() - start;
          const totals = samples.map(s => s.totalMs).sort((a,b) => a-b);
          result.runs.push({ setupMs, finalizeMs, meanMs: totals.reduce((a,b) => a+b, 0)/totals.length,
            medianMs: totals[Math.floor(totals.length/2)], samples });
        } finally { overlay.destroy(); }
        continue;
      }
      const setup = performance.now();
      const composite = new OffscreenCanvas(width, height);
      const context = composite.getContext('2d', { willReadFrequently: true });
      const sum = new Float32Array(width * height * 4);
      const scratch = new OffscreenCanvas(1, 1);
      const samples = [];
      const setupMs = performance.now() - setup;
      for (const geometry of geometries) {
        const start = performance.now();
        const decoded = await decoder.call('frame', { index: geometry.entry.frame });
        const decodedAt = performance.now();
        conversionContext.drawImage(decoded.bitmap, 0, 0); decoded.bitmap.close();
        const image = conversionContext.getImageData(0, 0, videoInfo.width, videoInfo.height);
        const rgbaAt = performance.now();
        const rectified = await computer.call('remap', { image, useWebGpu: true }, [image.data.buffer]);
        const remapAt = performance.now();
        scratch.width = rectified.width; scratch.height = rectified.height;
        scratch.getContext('2d').putImageData(new ImageData(rectified.data, rectified.width, rectified.height), 0, 0);
        const origin = geometry.world(0, 0);
        context.setTransform(1, 0, 0, 1, 0, 0); context.clearRect(0, 0, width, height);
        context.setTransform(geometry.c, geometry.s, -geometry.s, geometry.c, origin.x - minX, origin.y - minY);
        context.drawImage(scratch, 0, 0);
        accumulateFrame(sum, context.getImageData(0, 0, width, height).data);
        const end = performance.now();
        samples.push({ frame: geometry.entry.frame, decodeMs: decodedAt - start, rgbaMs: rgbaAt - decodedAt,
          remapMs: remapAt - rgbaAt, compositeMs: end - remapAt, totalMs: end - start,
          accelerator: rectified.accelerator, gpu: rectified.timing });
      }
      const finalStart = performance.now();
      const pixels = averagedFrames(sum);
      context.setTransform(1, 0, 0, 1, 0, 0); context.putImageData(new ImageData(pixels, width, height), 0, 0);
      const finalizeMs = performance.now() - finalStart;
      const totals = samples.map(s => s.totalMs).sort((a,b) => a-b);
      result.runs.push({ setupMs, finalizeMs, meanMs: totals.reduce((a,b) => a+b, 0)/totals.length,
        medianMs: totals[Math.floor(totals.length/2)], samples });
    }
    return result;
  } finally { decoder.terminate(); computer.terminate(); }
}
