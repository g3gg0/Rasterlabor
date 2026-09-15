import { createFile, DataStream } from 'mp4box';
import { orientationFromMatrix } from './video-orientation.js';

let file = null;
let samples = [];
let presentation = [];
let configuration = null;
const cache = new Map();
let cacheLimit = 3;
let decoder = null;
let decodedSampleIndex = -1;
let requestedTimestamp = null;
let selectedFrame = null;
let decoderError = null;
let sequentialSafe = false;
let resolveSelected = null;
let rejectSelected = null;
let orientation = null;
let decodedColor = null;

function closeDecoder() {
  selectedFrame?.close();
  selectedFrame = null;
  if (decoder && decoder.state !== 'closed') decoder.close();
  decoder = null;
  decodedSampleIndex = -1;
  requestedTimestamp = null;
  decoderError = null;
  resolveSelected = null;
  rejectSelected = null;
}

function startDecoder(sampleIndex) {
  closeDecoder();
  decoder = new VideoDecoder({
    output: frame => {
      if (frame.timestamp === requestedTimestamp) {
        selectedFrame?.close();
        selectedFrame = frame;
        resolveSelected?.();
      } else frame.close();
    },
    error: error => {
      decoderError = error;
      rejectSelected?.(error);
    }
  });
  decoder.configure(configuration);
  decodedSampleIndex = sampleIndex - 1;
}

async function createOrientedBitmap(frame) {
  if (!orientation.transformed) return createImageBitmap(frame);
  const canvas = new OffscreenCanvas(orientation.width, orientation.height);
  const context = canvas.getContext('2d');
  if (!context) throw new Error('2D-Canvas fuer die Videoorientierung ist nicht verfuegbar.');
  context.setTransform(orientation.a, orientation.b, orientation.c, orientation.d,
    orientation.translateX, orientation.translateY);
  context.drawImage(frame, 0, 0);
  return canvas.transferToImageBitmap();
}

async function openVideo(source) {
  if (!self.VideoDecoder) throw new Error('WebCodecs ist nicht verfuegbar. Aktuelles Microsoft Edge ueber http://localhost verwenden.');
  closeDecoder();
  for (const bitmap of cache.values()) bitmap.close();
  cache.clear();
  decodedColor = null;
  file = source;
  const parser = createFile(false);
  let information = null;
  let parseError = null;
  parser.onReady = value => { information = value; };
  parser.onError = value => { parseError = String(value); };
  let offset = 0;
  while (offset < file.size) {
    const buffer = await file.slice(offset, Math.min(file.size, offset + 1024 * 1024)).arrayBuffer();
    buffer.fileStart = offset;
    const next = parser.appendBuffer(buffer);
    offset = Math.max(offset + buffer.byteLength, next || 0);
    if (parseError) throw new Error(`MP4-Struktur ungueltig: ${parseError}`);
    self.postMessage({ progress: { stage: 'index', done: Math.min(offset, file.size), total: file.size } });
  }
  parser.flush();
  const track = information?.videoTracks?.[0];
  if (!track) throw new Error('Die Datei enthaelt keine lesbare MP4-Videospur.');
  const trackBox = parser.getTrackById(track.id);
  orientation = orientationFromMatrix(trackBox.tkhd.matrix, track.video.width, track.video.height);
  const entries = trackBox.mdia.minf.stbl.stsd.entries;
  if (entries.length !== 1) throw new Error('Wechselnde Sample-Konfigurationen werden nicht unterstuetzt.');
  const entry = entries[0];
  if (entry.type === 'encv') throw new Error('Verschluesselte Videospuren werden nicht unterstuetzt.');
  configuration = { codec: track.codec, codedWidth: track.video.width, codedHeight: track.video.height,
    hardwareAcceleration: 'no-preference' };
  const descriptionBox = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;
  if (descriptionBox) {
    const stream = new DataStream(undefined, 0, DataStream.BIG_ENDIAN);
    descriptionBox.write(stream);
    configuration.description = stream.buffer.slice(8);
  }
  let supported;
  try { supported = await VideoDecoder.isConfigSupported(configuration); }
  catch (error) { throw new Error(`Codec ${track.codec} nicht konfigurierbar: ${error.message}`); }
  if (!supported.supported) throw new Error(`MP4 erkannt, aber Codec ${track.codec} wird von diesem Edge/WebCodecs nicht dekodiert. H.264/AVC ist eine geeignete Alternative.`);
  samples = parser.getTrackSamplesInfo(track.id).map((sample, index) => ({
    index, offset: sample.offset, size: sample.size, key: sample.is_sync,
    timestamp: Math.round(sample.cts / sample.timescale * 1e6), duration: Math.round(sample.duration / sample.timescale * 1e6)
  }));
  if (!samples.length || samples.some(sample => !Number.isSafeInteger(sample.offset) || sample.offset < 0 || sample.offset + sample.size > file.size)) {
    throw new Error('Ungueltiger MP4-Sampleindex.');
  }
  presentation = [...samples].sort((first, second) => first.timestamp - second.timestamp);
  if (new Set(presentation.map(sample => sample.timestamp)).size !== presentation.length) throw new Error('Mehrdeutige Praesentationszeitstempel in der Videospur.');
  sequentialSafe = presentation.every((sample, index) => sample.index === index);
  const editEntries = trackBox.edts?.elst?.entries;
  if (editEntries && (editEntries.length > 1 || editEntries.some(edit => edit.media_rate_integer !== 1 || edit.media_rate_fraction !== 0 || edit.media_time < 0))) {
    throw new Error('Komplexe MP4-Editlisten werden fuer framegenaue Navigation nicht unterstuetzt.');
  }
  const firstTimestamp = presentation[0].timestamp;
  const duration = (presentation.at(-1).timestamp + presentation.at(-1).duration - firstTimestamp) / 1e6;
  const durations = presentation.slice(1).map((sample, index) => sample.timestamp - presentation[index].timestamp);
  const averageDuration = durations.reduce((sum, value) => sum + value, 0) / (durations.length || 1);
  cacheLimit = Math.max(1, Math.min(4, Math.floor(48 * 1024 * 1024 / (orientation.width * orientation.height * 4))));
  return { name: file.name, size: file.size, lastModified: file.lastModified,
    width: orientation.width, height: orientation.height, duration, codec: track.codec, frameCount: presentation.length,
    fps: averageDuration > 0 ? 1e6 / averageDuration : null,
    variableFrameRate: durations.some(value => Math.abs(value - averageDuration) > Math.max(100, averageDuration * 0.02)),
    firstTimestamp, timestamps: presentation.map(sample => sample.timestamp), cacheLimit };
}

async function decodeFrame(index) {
  if (!file || !presentation[index]) throw new Error('Frameindex ausserhalb der Videospur.');
  const target = presentation[index];
  if (cache.has(target.timestamp)) return { bitmap: await createImageBitmap(cache.get(target.timestamp)), index,
    timestamp: target.timestamp, color: decodedColor };
  let keyIndex = target.index;
  while (keyIndex > 0 && !samples[keyIndex].key) keyIndex--;
  if (!samples[keyIndex].key) throw new Error('Kein vorausgehender Keyframe fuer dieses Sample vorhanden.');
  const continueCost = sequentialSafe && decoder && target.index > decodedSampleIndex ? target.index - decodedSampleIndex : Infinity;
  const restartCost = target.index - keyIndex + 1;
  if (!decoder || !sequentialSafe || restartCost < continueCost) startDecoder(keyIndex);
  requestedTimestamp = target.timestamp;
  selectedFrame?.close();
  selectedFrame = null;
  const selected = new Promise((resolve, reject) => {
    resolveSelected = resolve;
    rejectSelected = reject;
  });
  try {
    for (let sampleIndex = decodedSampleIndex + 1; sampleIndex <= target.index; sampleIndex++) {
      const sample = samples[sampleIndex];
      const bytes = await file.slice(sample.offset, sample.offset + sample.size).arrayBuffer();
      if (decoderError) throw decoderError;
      decoder.decode(new EncodedVideoChunk({ type: sample.key ? 'key' : 'delta', timestamp: sample.timestamp,
        duration: sample.duration, data: bytes }));
      decodedSampleIndex = sampleIndex;
      if (decoder.decodeQueueSize > 12) await new Promise(resolve => { decoder.addEventListener('dequeue', resolve, { once: true }); });
    }
    if (sequentialSafe) await selected;
    else await decoder.flush();
    if (decoderError) throw decoderError;
    if (!selectedFrame) throw new Error(`Decoder lieferte Frame ${index} mit PTS ${target.timestamp} nicht.`);
    if (selectedFrame.displayWidth !== configuration.codedWidth || selectedFrame.displayHeight !== configuration.codedHeight) {
      throw new Error('Anzeigecrop oder nichtquadratische Pixel stimmen nicht mit der indexierten Quellgeometrie ueberein.');
    }
    decodedColor ||= { format: selectedFrame.format, primaries: selectedFrame.colorSpace?.primaries,
      transfer: selectedFrame.colorSpace?.transfer, matrix: selectedFrame.colorSpace?.matrix,
      fullRange: selectedFrame.colorSpace?.fullRange };
    const bitmap = await createOrientedBitmap(selectedFrame);
    cache.set(target.timestamp, bitmap);
    while (cache.size > cacheLimit) {
      const oldest = cache.keys().next().value;
      cache.get(oldest).close();
      cache.delete(oldest);
    }
    return { bitmap: await createImageBitmap(bitmap), index, timestamp: target.timestamp, color: decodedColor };
  } catch (error) {
    closeDecoder();
    throw error;
  } finally {
    selectedFrame?.close();
    selectedFrame = null;
    resolveSelected = null;
    rejectSelected = null;
  }
}

let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try {
      const result = data.type === 'open' ? await openVideo(data.file) : await decodeFrame(data.index);
      self.postMessage({ id: data.id, result }, result.bitmap ? [result.bitmap] : []);
    } catch (error) { self.postMessage({ id: data.id, error: error.message }); }
  });
};