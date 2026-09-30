import FFT from 'fft.js';

export function transform2d(values, width, height, inverse = false) {
  const output = values.slice();
  for (const vertical of [false, true]) {
    const length = vertical ? height : width;
    const count = vertical ? width : height;
    const fft = new FFT(length);
    const input = fft.createComplexArray();
    const result = fft.createComplexArray();
    for (let line = 0; line < count; line++) {
      for (let offset = 0; offset < length; offset++) {
        const index = 2 * (vertical ? offset * width + line : line * width + offset);
        input[2 * offset] = output[index]; input[2 * offset + 1] = output[index + 1];
      }
      if (inverse) fft.inverseTransform(result, input);
      else fft.transform(result, input);
      for (let offset = 0; offset < length; offset++) {
        const index = 2 * (vertical ? offset * width + line : line * width + offset);
        output[index] = result[2 * offset]; output[index + 1] = result[2 * offset + 1];
      }
    }
  }
  return output;
}

function sampleWindow(image, rectangle) {
  const { x, y, width, height } = rectangle;
  if (![x, y, width, height].every(Number.isFinite) || width < 32 || height < 32 ||
    x < 0 || y < 0 || x + width > image.width || y + height > image.height) throw new Error('Fenster muss mindestens 32 x 32 px gross sein und im Bild liegen.');
  const columns = 2 ** Math.floor(Math.log2(Math.min(256, width)));
  const rows = 2 ** Math.floor(Math.log2(Math.min(256, height)));
  const gray = new Float64Array(columns * rows);
  let sum = 0; let squares = 0;
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const left = Math.floor(x + column * width / columns);
      const right = Math.floor(x + (column + 1) * width / columns);
      const top = Math.floor(y + row * height / rows);
      const bottom = Math.floor(y + (row + 1) * height / rows);
      let value = 0;
      for (let py = top; py < bottom; py++) {
        for (let px = left; px < right; px++) {
          const offset = 4 * (py * image.width + px);
          if (image.data[offset + 3] < 255) throw new Error('Fenster enthaelt ungueltige Entzerrungspixel.');
          value += 0.299 * image.data[offset] + 0.587 * image.data[offset + 1] + 0.114 * image.data[offset + 2];
        }
      }
      value /= (right - left) * (bottom - top);
      gray[row * columns + column] = value;
      sum += value; squares += value * value;
    }
  }
  const mean = sum / gray.length;
  const deviation = Math.sqrt(Math.max(0, squares / gray.length - mean ** 2));
  const complex = new Float64Array(gray.length * 2);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      complex[2 * (row * columns + column)] = (gray[row * columns + column] - mean) *
        (0.5 - 0.5 * Math.cos(2 * Math.PI * column / (columns - 1))) *
        (0.5 - 0.5 * Math.cos(2 * Math.PI * row / (rows - 1)));
    }
  }
  return { gray, columns, rows, deviation, spectrum: transform2d(complex, columns, rows) };
}

function refineRigid(previous, current, rectangle, initialX, initialY, searchRadius, angleLimit) {
  const { columns, rows } = current;
  const scaleX = rectangle.width / columns; const scaleY = rectangle.height / rows;
  const centerX = (columns - 1) / 2; const centerY = (rows - 1) / 2;
  const stride = Math.max(1, Math.ceil(Math.sqrt(columns * rows / 4096)));
  const samples = [];
  for (let row = 2; row < rows - 2; row += stride) for (let column = 2; column < columns - 2; column += stride) {
    samples.push({ x: (column - centerX) * scaleX, y: (row - centerY) * scaleY, value: previous.gray[row * columns + column] });
  }
  const scoreAt = (dx, dy, angle) => {
    if (Math.abs(dx) > searchRadius || Math.abs(dy) > searchRadius || Math.abs(angle) > angleLimit) return -1;
    const cosine = Math.cos(angle); const sine = Math.sin(angle);
    let firstSum = 0; let secondSum = 0; let firstSquare = 0; let secondSquare = 0; let product = 0; let count = 0;
    for (const sample of samples) {
      const px = (cosine * sample.x - sine * sample.y + dx) / scaleX + centerX;
      const py = (sine * sample.x + cosine * sample.y + dy) / scaleY + centerY;
      if (px < 0 || py < 0 || px >= columns - 1 || py >= rows - 1) continue;
      const left = Math.floor(px); const top = Math.floor(py);
      const fractionX = px - left; const fractionY = py - top;
      const offset = top * columns + left;
      const second = (1 - fractionY) * ((1 - fractionX) * current.gray[offset] + fractionX * current.gray[offset + 1]) +
        fractionY * ((1 - fractionX) * current.gray[offset + columns] + fractionX * current.gray[offset + columns + 1]);
      const first = sample.value;
      firstSum += first; secondSum += second; firstSquare += first ** 2; secondSquare += second ** 2; product += first * second; count++;
    }
    if (count < Math.max(1, samples.length * 0.65)) return -1;
    return (product - firstSum * secondSum / count) /
      Math.max(1e-9, Math.sqrt(Math.max(0, (firstSquare - firstSum ** 2 / count) * (secondSquare - secondSum ** 2 / count))));
  };
  const seedX = Math.max(-searchRadius, Math.min(searchRadius, initialX));
  const seedY = Math.max(-searchRadius, Math.min(searchRadius, initialY));
  let best = { dx: seedX, dy: seedY, angle: 0, score: scoreAt(seedX, seedY, 0) };
  const consider = (dx, dy, angle) => {
    const score = scoreAt(dx, dy, angle);
    if (score > best.score) best = { dx, dy, angle, score };
  };
  for (let degrees = -angleLimit * 180 / Math.PI; degrees <= angleLimit * 180 / Math.PI; degrees += 1) {
    consider(seedX, seedY, degrees * Math.PI / 180);
  }
  let iterations = 0;
  for (let level = 0; level < 7; level++) {
    const translationStep = Math.max(scaleX, scaleY) * 2 / 2 ** level;
    const angleStep = Math.PI / 180 / 2 ** level;
    for (let iteration = 0; iteration < 16; iteration++) {
      const before = best;
      consider(before.dx - translationStep, before.dy, before.angle);
      consider(before.dx + translationStep, before.dy, before.angle);
      const horizontal = best;
      consider(horizontal.dx, horizontal.dy - translationStep, horizontal.angle);
      consider(horizontal.dx, horizontal.dy + translationStep, horizontal.angle);
      const translated = best;
      consider(translated.dx, translated.dy, translated.angle - angleStep);
      consider(translated.dx, translated.dy, translated.angle + angleStep);
      iterations++;
      if (before === best) break;
    }
  }
  return { ...best, iterations };
}

export class WindowTracker {
  reset() {
    this.previous = null; this.index = null; this.signature = null; this.x = 0; this.y = 0; this.angle = 0;
  }
  constructor() { this.reset(); }

  setPose(pose, width, height) {
    this.angle = -pose.rotation;
    const cosine = Math.cos(this.angle); const sine = Math.sin(this.angle);
    this.x = width / 2 - cosine * (width / 2 + pose.x) + sine * (height / 2 + pose.y);
    this.y = height / 2 - sine * (width / 2 + pose.x) - cosine * (height / 2 + pose.y);
  }

  process(image, index, rectangle, searchRadius = 32, maxRotation = 5) {
    const started = performance.now();
    const signature = JSON.stringify([image.width, image.height, rectangle]);
    if (this.signature !== null && (signature !== this.signature || index !== this.index + 1)) throw new Error('Fenster oder Framefolge geaendert: Tracking neu starten.');
    const current = sampleWindow(image, rectangle);
    const sampleMs = performance.now() - started;
    let refinementMs = 0;
    const referenceFrame = this.index;
    const shiftedReferenceX = image.width / 2 - this.x; const shiftedReferenceY = image.height / 2 - this.y;
    const referencePose = { x: Math.cos(this.angle) * shiftedReferenceX + Math.sin(this.angle) * shiftedReferenceY - image.width / 2,
      y: -Math.sin(this.angle) * shiftedReferenceX + Math.cos(this.angle) * shiftedReferenceY - image.height / 2, rotation: -this.angle };
    const finish = result => ({ ...result, incrementalMatch: referenceFrame === null ? null : {
      frame: referenceFrame, kind: 'incremental', accepted: result.success, reason: result.reason ?? '',
      referencePose: { ...referencePose }, prediction: { ...referencePose }, pose: result.raw ? { ...result.raw } : null,
      score: result.score, psr: result.psr, dx: result.dx, dy: result.dy, angle: result.angle, iterations: result.iterations,
      searchRadius, maxRotation, rectangle: { ...rectangle }, backward: null, milliseconds: performance.now() - started },
      points: [], accelerator: 'CPU FFT + Rigid', window: rectangle,
      resolution: `${current.columns} x ${current.rows}`, timing: { sampleMs, refinementMs, matchMs: performance.now() - started - sampleMs - refinementMs, totalMs: performance.now() - started } });
    if (current.deviation < 2) return finish({ success: false, reason: 'Zu wenig Struktur im Fenster.' });
    if (!this.previous) {
      this.previous = current; this.index = index; this.signature = signature;
      return finish({ success: true, initial: true, raw: { x: 0, y: 0, rotation: 0, points: 1 }, dx: 0, dy: 0 });
    }
    const { columns, rows } = current;
    const cross = new Float64Array(current.spectrum.length);
    for (let offset = 0; offset < cross.length; offset += 2) {
      const real = current.spectrum[offset] * this.previous.spectrum[offset] + current.spectrum[offset + 1] * this.previous.spectrum[offset + 1];
      const imaginary = current.spectrum[offset + 1] * this.previous.spectrum[offset] - current.spectrum[offset] * this.previous.spectrum[offset + 1];
      const magnitude = Math.hypot(real, imaginary);
      if (magnitude > 1e-9) { cross[offset] = real / magnitude; cross[offset + 1] = imaginary / magnitude; }
    }
    const correlation = transform2d(cross, columns, rows, true);
    let peak = 0;
    for (let index = 1; index < columns * rows; index++) if (correlation[2 * index] > correlation[2 * peak]) peak = index;
    const peakX = peak % columns; const peakY = Math.floor(peak / columns);
    const at = (px, py) => correlation[2 * (((py + rows) % rows) * columns + (px + columns) % columns)];
    const refine = (low, center, high) => {
      const curvature = low - 2 * center + high;
      return curvature < -1e-9 ? Math.max(-0.5, Math.min(0.5, 0.5 * (low - high) / curvature)) : 0;
    };
    const shiftX = peakX > columns / 2 ? peakX - columns : peakX;
    const shiftY = peakY > rows / 2 ? peakY - rows : peakY;
    const dx = (shiftX + refine(at(peakX - 1, peakY), at(peakX, peakY), at(peakX + 1, peakY))) * rectangle.width / columns;
    const dy = (shiftY + refine(at(peakX, peakY - 1), at(peakX, peakY), at(peakX, peakY + 1))) * rectangle.height / rows;
    let sum = 0; let squares = 0; let count = 0;
    for (let row = 0; row < rows; row++) for (let column = 0; column < columns; column++) {
      if (Math.min(Math.abs(column - peakX), columns - Math.abs(column - peakX)) <= 3 &&
        Math.min(Math.abs(row - peakY), rows - Math.abs(row - peakY)) <= 3) continue;
      const value = at(column, row); sum += value; squares += value * value; count++;
    }
    const psr = (at(peakX, peakY) - sum / count) / Math.max(1e-9, Math.sqrt(Math.max(0, squares / count - (sum / count) ** 2)));
    const refinementStarted = performance.now();
    const angleLimit = maxRotation * Math.PI / 180;
    const rigid = refineRigid(this.previous, current, rectangle, dx, dy, searchRadius, angleLimit);
    refinementMs = performance.now() - refinementStarted;
    const limitX = Math.min(searchRadius, rectangle.width / 2);
    const limitY = Math.min(searchRadius, rectangle.height / 2);
    const rejected = [];
    if (!(rigid.score >= 0.65)) rejected.push('Korrelation zu niedrig');
    if (!(Math.abs(rigid.dx) < limitX)) rejected.push('X-Suchgrenze erreicht');
    if (!(Math.abs(rigid.dy) < limitY)) rejected.push('Y-Suchgrenze erreicht');
    if (!(Math.abs(rigid.angle) < angleLimit)) rejected.push('Winkel-Suchgrenze erreicht');
    const success = rejected.length === 0;
    if (!success) {
      return finish({ success, psr, ...rigid, reason: `Fenster-Match verworfen: ${rejected.join(', ')}. ` +
        `NCC ${rigid.score.toFixed(3)} (min. 0.650), X ${rigid.dx.toFixed(2)} / +/-${limitX.toFixed(2)} px, ` +
        `Y ${rigid.dy.toFixed(2)} / +/-${limitY.toFixed(2)} px, Winkel ${(rigid.angle * 180 / Math.PI).toFixed(3)} / +/-${maxRotation} deg. ` +
        `FFT-Start: ${dx.toFixed(2)}, ${dy.toFixed(2)} px; PSR ${psr.toFixed(1)}.` });
    }
    const cosine = Math.cos(rigid.angle); const sine = Math.sin(rigid.angle);
    const centerX = rectangle.x + rectangle.width / 2; const centerY = rectangle.y + rectangle.height / 2;
    const tx = centerX + rigid.dx - cosine * centerX + sine * centerY;
    const ty = centerY + rigid.dy - sine * centerX - cosine * centerY;
    const nextX = cosine * this.x - sine * this.y + tx;
    this.y = sine * this.x + cosine * this.y + ty; this.x = nextX; this.angle += rigid.angle;
    this.previous = current; this.index = index;
    const shiftedX = image.width / 2 - this.x; const shiftedY = image.height / 2 - this.y;
    return finish({ success, psr, ...rigid, raw: {
      x: Math.cos(this.angle) * shiftedX + Math.sin(this.angle) * shiftedY - image.width / 2,
      y: -Math.sin(this.angle) * shiftedX + Math.cos(this.angle) * shiftedY - image.height / 2,
      rotation: -this.angle, points: 1 } });
  }
}
