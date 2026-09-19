import { EncodedPacket, EncodedVideoPacketSource, Mp4OutputFormat, Output, StreamTarget } from 'mediabunny';

export function encoderSizes(width, height) {
  const sizes = [];
  for (const maximum of [Infinity, 4096, 3840, 2160]) {
    const scale = Math.min(1, maximum / Math.max(width, height));
    const candidate = { width: Math.max(2, Math.floor(width * scale / 2) * 2),
      height: Math.max(2, Math.floor(height * scale / 2) * 2) };
    if (!sizes.some(size => size.width === candidate.width && size.height === candidate.height)) sizes.push(candidate);
  }
  return sizes;
}

export class VideoExporter {
  static async create({ width, height, fps, writable }) {
    if (!globalThis.VideoEncoder || !globalThis.VideoFrame) throw new Error('Videoexport benoetigt WebCodecs VideoEncoder in Microsoft Edge.');
    if (!writable) throw new Error('Kein beschreibbares Ausgabeziel gewaehlt.');
    for (const size of encoderSizes(width, height)) {
      const bitrate = Math.max(4_000_000, Math.min(80_000_000, Math.round(size.width * size.height * fps * 0.7)));
      for (const codec of ['avc1.640034', 'avc1.640032', 'avc1.4d0032']) {
        const config = { codec, ...size, bitrate, framerate: fps, hardwareAcceleration: 'no-preference',
          latencyMode: 'quality', avc: { format: 'avc' } };
        const support = await VideoEncoder.isConfigSupported(config);
        if (support.supported) {
          const exporter = new VideoExporter(support.config, fps, width, height, writable);
          await exporter.start();
          return exporter;
        }
      }
    }
    throw new Error(`Kein kompatibler H.264-Encoder fuer ${width} x ${height} px oder eine skalierte Ausgabe verfuegbar.`);
  }

  constructor(config, fps, sourceWidth, sourceHeight, writable) {
    this.config = config;
    this.fps = fps;
    this.sourceWidth = sourceWidth;
    this.sourceHeight = sourceHeight;
    this.scaled = config.width !== sourceWidth || config.height !== sourceHeight;
    this.frameCanvas = new OffscreenCanvas(sourceWidth, sourceHeight);
    this.frameContext = this.frameCanvas.getContext('2d', { alpha: false });
    this.packetSource = new EncodedVideoPacketSource('avc');
    this.output = new Output({ format: new Mp4OutputFormat({ fastStart: 'fragmented', minimumFragmentDuration: 1 }),
      target: new StreamTarget(writable, { chunked: true, chunkSize: 4 * 1024 * 1024 }) });
    this.output.addVideoTrack(this.packetSource, { frameRate: fps });
    this.packetWrites = Promise.resolve();
    this.pendingPacketCount = 0;
    this.failure = null;
    this.encoder = new VideoEncoder({
      output: (chunk, metadata) => {
        this.pendingPacketCount++;
        this.packetWrites = this.packetWrites
          .then(() => this.packetSource.add(EncodedPacket.fromEncodedChunk(chunk), metadata))
          .finally(() => { this.pendingPacketCount--; });
        this.packetWrites.catch(error => { this.failure = error; });
      },
      error: error => { this.failure = error; }
    });
    this.encoder.configure(config);
  }

  async start() {
    await this.output.start();
  }

  async addFrame(data, width, height, timestamp, duration, index) {
    const started = performance.now();
    if (this.failure) throw this.failure;
    if (width !== this.sourceWidth || height !== this.sourceHeight) throw new Error('Frameaufloesung weicht von der Exportmap ab.');
    const bitmap = typeof ImageBitmap !== 'undefined' && data instanceof ImageBitmap;
    const pixels = bitmap ? null : data instanceof Uint8ClampedArray ? data : new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength);
    let frame;
    let direct = true;
    try {
      frame = bitmap ? new VideoFrame(data, {timestamp, duration}) : new VideoFrame(pixels, { format: 'RGBA', codedWidth: width, codedHeight: height, timestamp, duration });
    } catch {
      direct = false;
      if (bitmap) this.frameContext.drawImage(data, 0, 0);
      else this.frameContext.putImageData(new ImageData(pixels, width, height), 0, 0);
      frame = new VideoFrame(this.frameCanvas, { timestamp, duration });
    }
    const frameMs = performance.now() - started;
    const submitStarted = performance.now();
    try { this.encoder.encode(frame, { keyFrame: index % Math.max(1, Math.round(this.fps)) === 0 }); }
    finally { frame.close(); }
    const submitMs = performance.now() - submitStarted;
    const submitted = performance.now();
    if (this.encoder.encodeQueueSize > 4 || this.pendingPacketCount > 4) {
      if (this.encoder.encodeQueueSize > 4) await new Promise(resolve => this.encoder.addEventListener('dequeue', resolve, { once: true }));
      await this.packetWrites;
    }
    if (this.failure) throw this.failure;
    return { frameMs, submitMs, backpressureMs: performance.now() - submitted, totalMs: performance.now() - started, direct };
  }

  async finish() {
    await this.encoder.flush();
    this.encoder.close();
    await this.packetWrites;
    if (this.failure) throw this.failure;
    this.packetSource.close();
    await this.output.finalize();
  }

  async cancel() {
    try { this.encoder.close(); } catch {}
    await this.output.cancel();
  }
}