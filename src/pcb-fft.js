import { transform2d } from './window-tracker.js';

const nextPowerOfTwo = value => 2 ** Math.ceil(Math.log2(value));

function highPass(values, width, height, radius = 5) {
  const stride = width + 1, integral = new Float64Array(stride * (height + 1));
  for (let row = 0; row < height; row++) for (let column = 0; column < width; column++) {
    const index = (row + 1) * stride + column + 1;
    integral[index] = values[row * width + column] + integral[index - 1] + integral[index - stride] - integral[index - stride - 1];
  }
  const output = new Float64Array(values.length);
  for (let row = 0; row < height; row++) for (let column = 0; column < width; column++) {
    const left = Math.max(0, column - radius), right = Math.min(width, column + radius + 1);
    const top = Math.max(0, row - radius), bottom = Math.min(height, row + radius + 1);
    const sum = integral[bottom * stride + right] - integral[top * stride + right] -
      integral[bottom * stride + left] + integral[top * stride + left];
    output[row * width + column] = values[row * width + column] - sum / ((right - left) * (bottom - top));
  }
  return output;
}

export function fftPatchShift(first, second, width, height, { searchRadius = 16, minimumStructure = 2,
  minimumPsr = 6, ownerBounds = null } = {}) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 16 || height < 16 ||
      first?.length !== width * height || second?.length !== width * height ||
      !Number.isFinite(searchRadius) || searchRadius < 1 || searchRadius >= Math.min(width, height) / 2) {
    throw new Error('Ungueltiges FFT-Messfenster.');
  }
  const highFirst = highPass(first, width, height), highSecond = highPass(second, width, height);
  const energy = values => Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length);
  const firstEnergy = energy(highFirst), secondEnergy = energy(highSecond);
  if (firstEnergy < minimumStructure || secondEnergy < minimumStructure)
    return { accepted: false, reason: 'Unzureichende Struktur', dx: null, dy: null };
  let supportPixels = 0, ownedSupportPixels = 0;
  for (let index = 0; index < highFirst.length; index++) {
    if (Math.abs(highFirst[index]) <= Math.max(minimumStructure * 2, firstEnergy * 0.5) ||
        Math.abs(highSecond[index]) <= Math.max(minimumStructure * 2, secondEnergy * 0.5)) continue;
    supportPixels++;
    const x = index % width, y = Math.floor(index / width);
    if (!ownerBounds || (x >= ownerBounds.minX && x < ownerBounds.maxX &&
        y >= ownerBounds.minY && y < ownerBounds.maxY)) ownedSupportPixels++;
  }
  const fftWidth = nextPowerOfTwo(width * 2), fftHeight = nextPowerOfTwo(height * 2);
  const firstSpectrum = new Float64Array(2 * fftWidth * fftHeight);
  const secondSpectrum = new Float64Array(firstSpectrum.length);
  for (let row = 0; row < height; row++) for (let column = 0; column < width; column++) {
    const taper = (0.5 - 0.5 * Math.cos(2 * Math.PI * column / (width - 1))) *
      (0.5 - 0.5 * Math.cos(2 * Math.PI * row / (height - 1)));
    const index = 2 * (row * fftWidth + column);
    firstSpectrum[index] = highFirst[row * width + column] * taper;
    secondSpectrum[index] = highSecond[row * width + column] * taper;
  }
  const firstFft = transform2d(firstSpectrum, fftWidth, fftHeight);
  const secondFft = transform2d(secondSpectrum, fftWidth, fftHeight);
  const cross = new Float64Array(firstFft.length);
  let maximumMagnitude = 0;
  for (let index = 0; index < cross.length; index += 2) {
    const real = secondFft[index] * firstFft[index] + secondFft[index + 1] * firstFft[index + 1];
    const imaginary = secondFft[index + 1] * firstFft[index] - secondFft[index] * firstFft[index + 1];
    const magnitude = Math.hypot(real, imaginary);
    maximumMagnitude = Math.max(maximumMagnitude, magnitude);
    cross[index] = real; cross[index + 1] = imaginary;
  }
  const regularization = maximumMagnitude * 1e-4;
  for (let index = 0; index < cross.length; index += 2) {
    const denominator = Math.hypot(cross[index], cross[index + 1]) + regularization;
    cross[index] /= denominator; cross[index + 1] /= denominator;
  }
  const response = transform2d(cross, fftWidth, fftHeight, true);
  const at = (x, y) => response[2 * (((y + fftHeight) % fftHeight) * fftWidth + (x + fftWidth) % fftWidth)];
  let peak = -Infinity, peakX = 0, peakY = 0, secondPeak = -Infinity;
  const radius = Math.floor(searchRadius);
  for (let y = -radius; y <= radius; y++) for (let x = -radius; x <= radius; x++) {
    const value = at(x, y);
    if (value > peak) { secondPeak = peak; peak = value; peakX = x; peakY = y; }
    else secondPeak = Math.max(secondPeak, value);
  }
  let sum = 0, squares = 0, count = 0;
  for (let y = -radius; y <= radius; y++) for (let x = -radius; x <= radius; x++) {
    if (Math.abs(x - peakX) <= 2 && Math.abs(y - peakY) <= 2) continue;
    const value = at(x, y); sum += value; squares += value * value; count++;
  }
  const mean = sum / count, deviation = Math.sqrt(Math.max(1e-12, squares / count - mean * mean));
  const psr = (peak - mean) / deviation;
  const subpixel = (lower, center, upper) => {
    const curvature = lower - 2 * center + upper;
    return curvature < -1e-9 ? Math.max(-0.5, Math.min(0.5, 0.5 * (lower - upper) / curvature)) : 0;
  };
  const dx = peakX + subpixel(at(peakX - 1, peakY), peak, at(peakX + 1, peakY));
  const dy = peakY + subpixel(at(peakX, peakY - 1), peak, at(peakX, peakY + 1));
  const accepted = psr >= minimumPsr && Math.abs(dx) < searchRadius - 0.5 && Math.abs(dy) < searchRadius - 0.5;
  return { accepted, reason: accepted ? null : psr < minimumPsr ? 'Mehrdeutig oder Rauschen' : 'Suchgrenze',
    dx, dy, psr, peak, secondPeak, peakRatio: peak / Math.max(1e-12, secondPeak),
    structureRms: Math.min(firstEnergy, secondEnergy), supportPixels, ownedSupportPixels };
}
