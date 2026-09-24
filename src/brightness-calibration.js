const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
const toLinear = value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
const toSrgb = value => value <= 0.0031308 ? value * 12.92 : 1.055 * value ** (1 / 2.4) - 0.055;

export function encodeBrightnessCalibration(calibration, geometryIdentity) {
  if (!(calibration?.gain instanceof Float32Array) || !(calibration.supported instanceof Uint8Array) ||
      calibration.gain.length !== calibration.width * calibration.height || calibration.supported.length !== calibration.gain.length) {
    throw new Error('Ungueltiges Helligkeitsfeld.');
  }
  const metadata = new TextEncoder().encode(`${JSON.stringify({ format: 'rasterlabor-brightness', modelVersion: 2,
    width: calibration.width, height: calibration.height, model: calibration.model, metrics: calibration.metrics,
    gainType: 'float32-le', supportType: 'uint8', normalization: 'geometric-mean-1',
    colorSpace: 'srgb-decoded-linear-light', geometryIdentity })}\n`);
  const pixels = calibration.gain.length, result = new Uint8Array(metadata.length + pixels * 5);
  result.set(metadata);
  const view = new DataView(result.buffer);
  for (let index = 0; index < pixels; index++) view.setFloat32(metadata.length + index * 4, calibration.gain[index], true);
  result.set(calibration.supported, metadata.length + pixels * 4);
  return result;
}

export function decodeBrightnessCalibration(bytes, { width, height, geometryIdentity }) {
  if (!(bytes instanceof Uint8Array)) bytes = new Uint8Array(bytes);
  const lineEnd = bytes.indexOf(10);
  if (lineEnd < 1 || lineEnd > 65536) throw new Error('Ungueltiges Helligkeitsfeldpaket.');
  let metadata;
  try { metadata = JSON.parse(new TextDecoder().decode(bytes.subarray(0, lineEnd))); }
  catch { throw new Error('Ungueltige Metadaten im Helligkeitsfeldpaket.'); }
  if (metadata.format !== 'rasterlabor-brightness' || metadata.modelVersion !== 2 || metadata.gainType !== 'float32-le' ||
      metadata.colorSpace !== 'srgb-decoded-linear-light' || metadata.normalization !== 'geometric-mean-1' ||
      metadata.width !== width || metadata.height !== height) throw new Error('Helligkeitsfeld passt nicht zur geladenen geometrischen Entzerrung.');
  if (!geometryIdentity || metadata.geometryIdentity !== geometryIdentity) throw new Error('Helligkeitsfeld wurde mit einer anderen Optik oder Entzerrung erstellt.');
  const pixels = width * height, start = lineEnd + 1;
  if (bytes.length !== start + pixels * 5) throw new Error('Helligkeitsfelddaten sind unvollstaendig.');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), gain = new Float32Array(pixels);
  for (let index = 0; index < pixels; index++) gain[index] = view.getFloat32(start + index * 4, true);
  if (gain.some(value => !Number.isFinite(value) || value <= 0)) throw new Error('Helligkeitsfeld enthaelt ungueltige Gain-Werte.');
  return { version: 2, width, height, model: metadata.model, metrics: metadata.metrics,
    gain, supported: bytes.slice(start + pixels * 4) };
}

export function applyBrightnessCalibration(image, calibration) {
  if (!calibration || image.width !== calibration.width || image.height !== calibration.height) return image;
  for (let pixel = 0; pixel < calibration.gain.length; pixel++) {
    const alpha = image.data[pixel * 4 + 3];
    if (!alpha) continue;
    const gain = calibration.gain[pixel];
    for (let channel = 0; channel < 3; channel++) image.data[pixel * 4 + channel] =
      Math.round(255 * clamp(toSrgb(toLinear(image.data[pixel * 4 + channel] / 255) * gain), 0, 1));
  }
  return image;
}

export function brightnessDisplayRange(calibration, allowed = null) {
  const values = [];
  const stride = Math.max(1, Math.ceil(Math.sqrt(calibration.gain.length / 100000)));
  for (let y = 0; y < calibration.height; y += stride) for (let x = 0; x < calibration.width; x += stride) {
    const index = y * calibration.width + x;
    if ((calibration.supported[index] || calibration.model?.completion) &&
      (!allowed || allowed(x, y))) values.push(1 / calibration.gain[index]);
  }
  if (!values.length) return { low: 0.6, high: 1.4 };
  values.sort((a, b) => a - b);
  const percentile = fraction => values[Math.floor((values.length - 1) * fraction)];
  const span = Math.max(0.02, 1 - percentile(0.02), percentile(0.98) - 1);
  return { low: 1 - span, high: 1 + span };
}

export function brightnessFieldPixels(calibration, outputWidth, outputHeight, allowed = null, range = { low: 0.6, high: 1.4 }) {
  if (!calibration || !Number.isInteger(outputWidth) || !Number.isInteger(outputHeight) || outputWidth < 1 || outputHeight < 1) {
    throw new Error('Ungueltige Helligkeitsfeldansicht.');
  }
  const data = new Uint8ClampedArray(outputWidth * outputHeight * 4);
  const scaleX = calibration.width / outputWidth, scaleY = calibration.height / outputHeight;
  for (let y = 0; y < outputHeight; y++) for (let x = 0; x < outputWidth; x++) {
    const sourceX = Math.min(calibration.width - 1, Math.floor((x + 0.5) * scaleX));
    const sourceY = Math.min(calibration.height - 1, Math.floor((y + 0.5) * scaleY));
    const source = sourceY * calibration.width + sourceX;
    if ((!calibration.supported[source] && !calibration.model?.completion) ||
      (allowed && !allowed(sourceX, sourceY))) continue;
    const relativeTransmission = clamp((1 / calibration.gain[source] - range.low) / (range.high - range.low), 0, 1);
    const gray = Math.round(relativeTransmission * 255), target = (y * outputWidth + x) * 4;
    data[target] = gray; data[target + 1] = gray; data[target + 2] = gray; data[target + 3] = 255;
  }
  return data;
}