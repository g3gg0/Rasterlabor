import { WebGpuRemapper } from './webgpu-native-remapper.js';

// One entry point for decoded frames. Returned frames/bitmaps belong to the caller;
// intermediate native surfaces and GPU allocations belong to this reader.
export class FrameReader {
  constructor({ getDecoder, getMaps, getBrightness = () => null, computer, onFallback = () => {} }) {
    Object.assign(this, { getDecoder, getMaps, getBrightness, computer, onFallback });
    this.pending = Promise.resolve();
  }

  read(index, options = {}) {
    // Serialize access to the decoder and reusable render/readback surfaces.
    const decoder = this.getDecoder(), maps = this.getMaps(), brightness = options.brightness === false ? null : this.getBrightness();
    const run = this.pending.then(() => {
      if (decoder !== this.getDecoder() || maps !== this.getMaps() || brightness !== (options.brightness === false ? null : this.getBrightness())) throw new Error('Video, Kalibrierung oder Helligkeitsfeld wurde waehrend des Frameabrufs gewechselt.');
      return this.readFrame(decoder, maps, brightness, index, options);
    });
    this.pending = run.catch(() => {});
    return run;
  }

  toRgba(bitmap) {
    this.conversion ??= new OffscreenCanvas(1, 1);
    if (this.conversion.width !== bitmap.width) this.conversion.width = bitmap.width;
    if (this.conversion.height !== bitmap.height) this.conversion.height = bitmap.height;
    const context = this.conversion.getContext('2d', { willReadFrequently: true });
    context.clearRect(0, 0, bitmap.width, bitmap.height);
    context.drawImage(bitmap, 0, 0);
    return context.getImageData(0, 0, bitmap.width, bitmap.height);
  }

  async readFrame(decoder, maps, brightness, index, { gpu = false, rectified = false, output = 'bitmap', brightness: applyBrightness = true } = {}) {
    if (!['bitmap', 'rgba', 'native'].includes(output)) throw new Error(`Unbekanntes Frameformat: ${output}`);
    if (rectified && (!maps || output === 'native')) throw new Error('Entzerrte Frames benoetigen Maps und Bitmap- oder RGBA-Ausgabe.');
    if (this.decoder !== decoder || this.maps !== maps || this.brightness !== brightness) {
      this.renderer?.destroy(); this.renderer = null; this.gpuFailed = false;
      this.decoder = decoder; this.maps = maps; this.brightness = brightness;
    }
    if (!gpu || output === 'native') {
      this.renderer?.destroy(); this.renderer = null;
      if (!gpu) this.gpuFailed = false;
    }
    const timing = { decodeMs: 0, sharpnessMs: 0, rgbaMs: 0, remapMs: 0 };
    const recordDecodeTiming = (decoded, elapsed) => {
      timing.sharpnessMs = decoded.sharpnessMs || 0;
      timing.decodeMs = Math.max(0, elapsed - timing.sharpnessMs);
    };
    let started = performance.now();
    if (output === 'native') {
      const decoded = await decoder.call('native-frame', { index, useWebGpu: gpu });
      recordDecodeTiming(decoded, performance.now() - started);
      return { ...decoded, frameTiming: timing };
    }
    if (gpu && !rectified) {
      const decoded = await decoder.call('native-frame', { index, useWebGpu: gpu });
      recordDecodeTiming(decoded, performance.now() - started);
      try {
        const o = decoded.orientation;
        let bitmap;
        if (!o.transformed) bitmap = await createImageBitmap(decoded.frame);
        else {
          this.oriented ??= new OffscreenCanvas(o.width, o.height);
          if (this.oriented.width !== o.width) this.oriented.width = o.width;
          if (this.oriented.height !== o.height) this.oriented.height = o.height;
          const context = this.oriented.getContext('2d');
          context.setTransform(1, 0, 0, 1, 0, 0);
          context.clearRect(0, 0, o.width, o.height);
          context.setTransform(o.a, o.b, o.c, o.d, o.translateX, o.translateY);
          context.drawImage(decoded.frame, 0, 0);
          bitmap = this.oriented.transferToImageBitmap();
        }
        const elapsed = performance.now() - started;
        timing.decodeMs += Math.max(0, elapsed - timing.decodeMs - timing.sharpnessMs);
        const metadata = { index, timestamp: decoded.timestamp, color: decoded.color,
          sharpness: decoded.sharpness, frameTiming: timing };
        if (output === 'bitmap') return { bitmap, ...metadata };
        started = performance.now();
        try {
          const image = this.toRgba(bitmap);
          timing.rgbaMs = performance.now() - started;
          return { width: image.width, height: image.height, data: image.data, ...metadata };
        } finally { bitmap.close(); }
      } finally { decoded.frame.close(); }
    }
    if (gpu && rectified && !this.gpuFailed) {
      try {
        this.renderer ??= await WebGpuRemapper.create(maps, brightness);
      } catch (error) {
        this.gpuFailed = true; this.onFallback(error);
      }
      // Include first-use initialization in the reported remap time.
      timing.remapMs = performance.now() - started;
      if (this.renderer) {
        const decoded = await decoder.call('native-frame', { index, useWebGpu: gpu });
        recordDecodeTiming(decoded, performance.now() - started - timing.remapMs);
        try {
          const renderStarted = performance.now();
          const result = await this.renderer.render(decoded.frame, decoded.orientation, output);
          timing.remapMs += performance.now() - renderStarted;
          return { ...result, index, timestamp: decoded.timestamp, color: decoded.color,
            sharpness: decoded.sharpness, frameTiming: timing };
        } catch (error) {
          this.renderer.destroy(); this.renderer = null; this.gpuFailed = true; this.onFallback(error);
        } finally { decoded.frame.close(); }
      }
    }
    // CPU mode retains the worker's bounded bitmap cache.
    started = performance.now();
    const decoded = await decoder.call('frame', { index, useWebGpu: gpu });
    recordDecodeTiming(decoded, performance.now() - started);
    if (!rectified && output === 'bitmap') return { ...decoded, frameTiming: timing };
    started = performance.now();
    let image;
    try { image = this.toRgba(decoded.bitmap); } finally { decoded.bitmap.close(); }
    timing.rgbaMs = performance.now() - started;
    if (rectified) {
      started = performance.now();
      image = await this.computer.call('remap', { image, useWebGpu: false, brightness: applyBrightness }, [image.data.buffer]);
      timing.remapMs += performance.now() - started;
    }
    const metadata = { index, timestamp: decoded.timestamp, color: decoded.color,
      sharpness: decoded.sharpness, frameTiming: timing };
    if (output === 'rgba') return { ...image, width: image.width, height: image.height, data: image.data, ...metadata };
    const bitmap = await createImageBitmap(new ImageData(image.data, image.width, image.height));
    return { bitmap, width: image.width, height: image.height, accelerator: image.accelerator || 'CPU', ...metadata };
  }

  async dispose() {
    await this.pending;
    this.renderer?.destroy(); this.renderer = null;
  }
}
