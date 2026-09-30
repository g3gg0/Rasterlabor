import { WebGpuRemapper } from './webgpu-native-remapper.js';

// One entry point for decoded frames. Returned frames/bitmaps belong to the caller;
// intermediate native surfaces and GPU allocations belong to this reader.
export class FrameReader {
  constructor({ getDecoder, getMaps, getBrightness = () => null, computer, onFallback = () => {} }) {
    Object.assign(this, { getDecoder, getMaps, getBrightness, computer, onFallback });
    this.pending = Promise.resolve();
    this.frameCache = new Map();
    this.frameCacheBytes = 0;
    this.nativeCache = new Map();
    this.nativeCacheBytes = 0;
    this.cacheAge = 0;
  }

  read(index, options = {}) {
    // Serialize access to the decoder and reusable render/readback surfaces.
    const decoder = this.getDecoder(), maps = this.getMaps(), brightness = options.brightness === false ? null : this.getBrightness();
    const run = this.pending.then(async () => {
      if (decoder !== this.getDecoder() || maps !== this.getMaps() || brightness !== (options.brightness === false ? null : this.getBrightness())) throw new Error('Video, Kalibrierung oder Helligkeitsfeld wurde waehrend des Frameabrufs gewechselt.');
      try { return await this.readFrame(decoder, maps, brightness, index, options); }
      catch (error) {
        if (!this.releaseOnAllocationError(error)) throw error;
        return this.readFrame(decoder, maps, brightness, index, options);
      }
    });
    this.pending = run.catch(() => {});
    return run;
  }

  async waitUntilIdle() {
    await this.pending;
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

  clearFrameCache() {
    this.clearRectifiedCache();
    this.clearNativeCache();
  }

  clearRectifiedCache() {
    for (const entry of this.frameCache.values()) entry.bitmap.close();
    this.frameCache.clear();
    this.frameCacheBytes = 0;
  }

  clearNativeCache() {
    for (const entry of this.nativeCache.values()) entry.frame.close();
    this.nativeCache.clear();
    this.nativeCacheBytes = 0;
  }

  frameCacheBudget() {
    const heap = globalThis.performance?.memory;
    if (!heap?.jsHeapSizeLimit || !Number.isFinite(heap.usedJSHeapSize)) return 512 * 1024 ** 2;
    return Math.max(0, heap.jsHeapSizeLimit * 2 / 3 - heap.usedJSHeapSize);
  }

  evictOldestFrames() {
    const count = Math.ceil((this.frameCache.size + this.nativeCache.size) / 10);
    const oldest = [...this.frameCache.entries()].map(([key, entry]) => ({ key, entry, cache: this.frameCache }))
      .concat([...this.nativeCache.entries()].map(([key, entry]) => ({ key, entry, cache: this.nativeCache })))
      .sort((first, second) => first.entry.age - second.entry.age);
    for (const { key, entry, cache } of oldest.slice(0, count)) {
      (entry.bitmap ?? entry.frame).close();
      cache.delete(key);
      if (cache === this.frameCache) this.frameCacheBytes -= entry.bytes;
      else this.nativeCacheBytes -= entry.bytes;
    }
    return count;
  }

  releaseOnAllocationError(error) {
    if (!/^(?:OutOfMemoryError|QuotaExceededError)$/.test(error?.name ?? '') &&
        !/(?:out of memory|allocation failed|not enough memory|insufficient memory)/i.test(error?.message ?? '')) return false;
    return this.evictOldestFrames() > 0;
  }

  async cachedNativeFrame(index, measureSharpness) {
    const entry = this.nativeCache.get(index);
    if (!entry || (measureSharpness && !entry.metadata.sharpness)) return null;
    entry.age = ++this.cacheAge;
    return { ...entry.metadata, frame: entry.frame.clone(),
      frameTiming: { decodeMs: 0, sharpnessMs: 0, rgbaMs: 0, remapMs: 0, cacheHit: true } };
  }

  copyNativeFrame(frame) {
    // clone() retains the hardware decoder's output surface. Snapshot through a
    // canvas so the cache owns its pixels and the decoder can reuse its buffers.
    const width = frame.displayWidth, height = frame.displayHeight;
    this.nativeSnapshot ??= new OffscreenCanvas(width, height);
    if (this.nativeSnapshot.width !== width) this.nativeSnapshot.width = width;
    if (this.nativeSnapshot.height !== height) this.nativeSnapshot.height = height;
    const context = this.nativeSnapshot.getContext('2d', { alpha: false });
    context.drawImage(frame, 0, 0, width, height);
    return new VideoFrame(this.nativeSnapshot, { timestamp: frame.timestamp,
      duration: frame.duration ?? undefined });
  }

  rememberNativeFrame(index, result) {
    const bytes = (result.frame.codedWidth ?? result.frame.displayWidth) *
      (result.frame.codedHeight ?? result.frame.displayHeight) * 4;
    if (!Number.isFinite(bytes) || bytes <= 0) return;
    let frame;
    try { frame = this.copyNativeFrame(result.frame); }
    catch (error) {
      if (!this.releaseOnAllocationError(error)) return;
      try { frame = this.copyNativeFrame(result.frame); } catch { return; }
    }
    const previous = this.nativeCache.get(index);
    if (previous) { previous.frame.close(); this.nativeCacheBytes -= previous.bytes; this.nativeCache.delete(index); }
    const { frame: ignored, ...metadata } = result;
    this.nativeCache.set(index, { frame, bytes, metadata, age: ++this.cacheAge });
    this.nativeCacheBytes += bytes;
  }

  async cachedFrame(index, measureSharpness, { maps, brightness, sourceMask }) {
    const entry = this.frameCache.get(index);
    if (!entry || (measureSharpness && !entry.sharpness) || entry.maps !== maps ||
        entry.brightness !== brightness || entry.sourceMask !== sourceMask ||
        entry.sourceMaskRevision !== sourceMask?.revision) return null;
    entry.age = ++this.cacheAge;
    return { ...entry.metadata, bitmap: await createImageBitmap(entry.bitmap),
      frameTiming: { decodeMs: 0, sharpnessMs: 0, rgbaMs: 0, remapMs: 0, cacheHit: true } };
  }

  async rememberFrame(index, result, { maps, brightness, sourceMask }) {
    const width = result.bitmap?.width ?? result.width, height = result.bitmap?.height ?? result.height;
    const bytes = width * height * 4;
    if (!Number.isFinite(bytes) || bytes <= 0) return;
    const source = result.bitmap ?? new ImageData(result.data, width, height);
    let bitmap;
    try { bitmap = await createImageBitmap(source); }
    catch (error) {
      if (!this.releaseOnAllocationError(error)) return;
      try { bitmap = await createImageBitmap(source); } catch { return; }
    }
    const previous = this.frameCache.get(index);
    if (previous) { previous.bitmap.close(); this.frameCacheBytes -= previous.bytes; this.frameCache.delete(index); }
    const { bitmap: ignored, data: ignoredData, ...metadata } = result;
    this.frameCache.set(index, { bitmap, bytes, sharpness: result.sharpness, metadata,
      maps, brightness, sourceMask, sourceMaskRevision: sourceMask?.revision, age: ++this.cacheAge });
    this.frameCacheBytes += bytes;
  }

  cacheStats() {
    return { frames: this.frameCache.size + this.nativeCache.size,
      bytes: this.frameCacheBytes + this.nativeCacheBytes, budget: this.frameCacheBudget() };
  }

  async readFrame(decoder, maps, brightness, index, { gpu = false, rectified = false, output = 'bitmap', brightness: applyBrightness = true, measureSharpness = true, sourceMask = null, cache = true } = {}) {
    if (!['bitmap', 'rgba', 'native'].includes(output)) throw new Error(`Unbekanntes Frameformat: ${output}`);
    if (rectified && (!maps || output === 'native')) throw new Error('Entzerrte Frames benoetigen Maps und Bitmap- oder RGBA-Ausgabe.');
    if (this.decoder !== decoder) {
      this.renderer?.destroy(); this.renderer = null; this.gpuFailed = false;
      this.clearFrameCache();
      this.decoder = decoder;
    }
    if (rectified && gpu && (this.maps !== maps || this.brightness !== brightness ||
        this.sourceMask !== sourceMask || this.sourceMaskRevision !== sourceMask?.revision)) {
      this.renderer?.destroy(); this.renderer = null; this.gpuFailed = false;
      this.maps = maps; this.brightness = brightness;
      this.sourceMask = sourceMask; this.sourceMaskRevision = sourceMask?.revision;
    }
    if (!gpu || output === 'native') {
      this.renderer?.destroy(); this.renderer = null;
      if (!gpu) this.gpuFailed = false;
    }
    if (output === 'native') {
      const cached = await this.cachedNativeFrame(index, measureSharpness);
      if (cached) return cached;
    }
    const cacheable = cache && rectified && applyBrightness;
    const cacheParameters = { maps, brightness, sourceMask };
    if (cacheable) {
      const cached = await this.cachedFrame(index, measureSharpness, cacheParameters);
      if (cached) {
        if (output === 'bitmap') return cached;
        try {
          const image = this.toRgba(cached.bitmap);
          const { bitmap: ignored, ...metadata } = cached;
          return { ...metadata, width: image.width, height: image.height, data: image.data };
        } finally { cached.bitmap.close(); }
      }
    }
    const result = await this.readUncachedFrame(decoder, maps, brightness, index,
      { gpu, rectified, output, brightness: applyBrightness, measureSharpness, sourceMask, cache });
    if (cacheable) await this.rememberFrame(index, result, cacheParameters);
    return result;
  }

  async decodeNativeFrame(decoder, index, gpu, measureSharpness, retain = true) {
    const cached = await this.cachedNativeFrame(index, measureSharpness);
    if (cached) return cached;
    const result = await decoder.call('native-frame', { index, useWebGpu: gpu, measureSharpness });
    if (retain) this.rememberNativeFrame(index, result);
    return result;
  }

  async nativeBitmap(decoded) {
    const o = decoded.orientation;
    if (!o.transformed) return createImageBitmap(decoded.frame);
    this.oriented ??= new OffscreenCanvas(o.width, o.height);
    if (this.oriented.width !== o.width) this.oriented.width = o.width;
    if (this.oriented.height !== o.height) this.oriented.height = o.height;
    const context = this.oriented.getContext('2d');
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, o.width, o.height);
    context.setTransform(o.a, o.b, o.c, o.d, o.translateX, o.translateY);
    context.drawImage(decoded.frame, 0, 0);
    return this.oriented.transferToImageBitmap();
  }

  async readUncachedFrame(decoder, maps, brightness, index,
    { gpu = false, rectified = false, output = 'bitmap', brightness: applyBrightness = true,
      measureSharpness = true, sourceMask = null, cache = true } = {}) {
    const timing = { decodeMs: 0, sharpnessMs: 0, rgbaMs: 0, remapMs: 0 };
    const recordDecodeTiming = (decoded, elapsed) => {
      timing.sharpnessMs = decoded.sharpnessMs || 0;
      timing.decodeMs = Math.max(0, elapsed - timing.sharpnessMs);
      if (decoded.frameTiming?.cacheHit) { timing.decodeMs = 0; timing.sharpnessMs = 0; timing.sourceCacheHit = true; }
    };
    let started = performance.now();
    if (output === 'native') {
      const decoded = await this.decodeNativeFrame(decoder, index, gpu, measureSharpness, cache);
      recordDecodeTiming(decoded, performance.now() - started);
      return { ...decoded, frameTiming: timing };
    }
    if (gpu && !rectified) {
      const decoded = await this.decodeNativeFrame(decoder, index, gpu, measureSharpness, false);
      recordDecodeTiming(decoded, performance.now() - started);
      try {
        const bitmap = await this.nativeBitmap(decoded);
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
        this.renderer ??= await WebGpuRemapper.create(maps, brightness, sourceMask);
      } catch (error) {
        this.gpuFailed = true; this.onFallback(error);
      }
      // Include first-use initialization in the reported remap time.
      timing.remapMs = performance.now() - started;
      if (this.renderer) {
        const decoded = await this.decodeNativeFrame(decoder, index, gpu, measureSharpness, false);
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
    const cachedNative = await this.cachedNativeFrame(index, measureSharpness);
    let decoded;
    if (cachedNative) {
      const { frame: ignored, ...metadata } = cachedNative;
      try { decoded = { ...metadata, bitmap: await this.nativeBitmap(cachedNative) }; }
      finally { cachedNative.frame.close(); }
    } else decoded = await decoder.call('frame', { index, useWebGpu: gpu, measureSharpness });
    recordDecodeTiming(decoded, performance.now() - started);
    if (!rectified && output === 'bitmap') return { ...decoded, frameTiming: timing };
    started = performance.now();
    let image;
    try { image = this.toRgba(decoded.bitmap); } finally { decoded.bitmap.close(); }
    timing.rgbaMs = performance.now() - started;
    if (rectified) {
      started = performance.now();
      image = await this.computer.call('remap', { image, useWebGpu: false, brightness: applyBrightness, sourceMask }, [image.data.buffer]);
      timing.remapMs += performance.now() - started;
    }
    const metadata = { index, timestamp: decoded.timestamp, color: decoded.color,
      sharpness: decoded.sharpness, frameTiming: timing };
    if (output === 'rgba') return { ...image, width: image.width, height: image.height, data: image.data, ...metadata };
    const bitmap = await createImageBitmap(new ImageData(image.data, image.width, image.height));
    return { bitmap, width: image.width, height: image.height, accelerator: image.accelerator || 'CPU', ...metadata };
  }

  async releaseRenderer() {
    await this.pending;
    this.renderer?.destroy(); this.renderer = null;
  }

  async dispose() {
    await this.releaseRenderer();
    this.clearFrameCache();
    this.nativeSnapshot = null;
  }
}
