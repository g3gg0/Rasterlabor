import { Matrix, QrDecomposition } from 'ml-matrix';
import { createSpline, evaluate, stencil, inversePoint } from './spline.js';
import { webGpuLsqrPair } from './webgpu-lsqr.js';

export const tick = () => new Promise(resolve => setTimeout(resolve, 0));
const norm = values => Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));

export function lsqr(rows, right, count, initial = new Float64Array(count), iterations = 350) {
  const multiply = vector => Float64Array.from(rows, row => row.indices.reduce(
    (sum, index, local) => sum + row.weights[local] * vector[index], 0));
  const transpose = vector => {
    const result = new Float64Array(count);
    rows.forEach((row, index) => row.indices.forEach((column, local) => {
      result[column] += row.weights[local] * vector[index];
    }));
    return result;
  };
  const scale = new Float64Array(count);
  rows.forEach(row => row.indices.forEach((index, local) => { scale[index] += row.weights[local] ** 2; }));
  for (let index = 0; index < count; index++) scale[index] = 1 / Math.sqrt(scale[index] || 1);
  const apply = vector => multiply(vector.map((value, index) => value * scale[index]));
  const applyTranspose = vector => transpose(vector).map((value, index) => value * scale[index]);
  const initialProduct = multiply(initial);
  let left = Float64Array.from(right, (value, index) => value - initialProduct[index]);
  let beta = norm(left);
  if (beta < 1e-14) return initial.slice();
  left = left.map(value => value / beta);
  let rightVector = applyTranspose(left);
  let alpha = norm(rightVector);
  if (alpha < 1e-14) return initial.slice();
  rightVector = rightVector.map(value => value / alpha);
  let direction = rightVector.slice();
  const solution = new Float64Array(count);
  let phiBar = beta;
  let rhoBar = alpha;
  const initialNormal = alpha * beta;
  for (let iteration = 0; iteration < iterations; iteration++) {
    const nextLeft = apply(rightVector).map((value, index) => value - alpha * left[index]);
    beta = norm(nextLeft);
    left = nextLeft.map(value => beta ? value / beta : 0);
    const nextRight = applyTranspose(left).map((value, index) => value - beta * rightVector[index]);
    alpha = norm(nextRight);
    rightVector = nextRight.map(value => alpha ? value / alpha : 0);
    const rho = Math.hypot(rhoBar, beta);
    if (rho < 1e-20) break;
    const cosine = rhoBar / rho;
    const sine = beta / rho;
    const theta = sine * alpha;
    rhoBar = -cosine * alpha;
    const phi = cosine * phiBar;
    phiBar *= sine;
    for (let index = 0; index < count; index++) {
      solution[index] += phi / rho * direction[index];
      direction[index] = rightVector[index] - theta / rho * direction[index];
    }
    if (Math.abs(alpha * sine * phi) < 1e-9 * initialNormal) break;
  }
  return solution.map((value, index) => initial[index] + scale[index] * value);
}

export function poseFor(points, field, step, weights = null) {
  let sum = 0;
  let meanQx = 0;
  let meanQy = 0;
  let meanUx = 0;
  let meanUy = 0;
  const transformed = points.map(point => evaluate(field, point.x, point.y));
  points.forEach((point, index) => {
    const weight = weights?.[index] ?? point.confidence ?? 1;
    sum += weight;
    meanQx += weight * step * point.col;
    meanQy += weight * step * point.row;
    meanUx += weight * transformed[index].x;
    meanUy += weight * transformed[index].y;
  });
  if (!(sum > 0)) throw new Error('No positive observation weights');
  meanQx /= sum; meanQy /= sum; meanUx /= sum; meanUy /= sum;
  let cosineSum = 0;
  let sineSum = 0;
  points.forEach((point, index) => {
    const weight = weights?.[index] ?? point.confidence ?? 1;
    const qx = step * point.col - meanQx;
    const qy = step * point.row - meanQy;
    const ux = transformed[index].x - meanUx;
    const uy = transformed[index].y - meanUy;
    cosineSum += weight * (qx * ux + qy * uy);
    sineSum += weight * (qx * uy - qy * ux);
  });
  const theta = Math.atan2(sineSum, cosineSum);
  const cosine = Math.cos(theta);
  const sine = Math.sin(theta);
  return { theta, tx: meanUx - cosine * meanQx + sine * meanQy,
    ty: meanUy - sine * meanQx - cosine * meanQy };
}

export function target(point, pose, step) {
  const cosine = Math.cos(pose.theta);
  const sine = Math.sin(pose.theta);
  return { x: step * (cosine * point.col - sine * point.row) + pose.tx,
    y: step * (sine * point.col + cosine * point.row) + pose.ty };
}

function affineStart(frame, width, height, spacing, step) {
  const design = new Matrix(frame.points.map(point => [point.x / width, point.y / height, 1]));
  const desired = new Matrix(frame.points.map(point => [step * point.col, step * point.row]));
  const affine = new QrDecomposition(design).solve(desired);
  return createSpline(width, height, spacing, [affine.get(0, 0) / width, affine.get(1, 0) / height,
    affine.get(0, 1) / width, affine.get(1, 1) / height, affine.get(2, 0), affine.get(2, 1)]);
}

export function smoothRows(field, lambda) {
  if (!lambda) return [];
  const gauss = [[0.06943184420297371, 0.17392742256872693], [0.33000947820757187, 0.32607257743127307],
    [0.6699905217924281, 0.32607257743127307], [0.9305681557970262, 0.17392742256872693]];
  const rows = [];
  const width = field.width - 1;
  const height = field.height - 1;
  const length = Math.max(field.width, field.height);
  for (let top = 0; top < height; top += field.spacing) {
    for (let left = 0; left < width; left += field.spacing) {
      const cellWidth = Math.min(field.spacing, width - left);
      const cellHeight = Math.min(field.spacing, height - top);
      for (const [localX, weightX] of gauss) {
        for (const [localY, weightY] of gauss) {
          const area = cellWidth * cellHeight / (width * height) * weightX * weightY;
          for (const [orderX, orderY, factor] of [[2, 0, 1], [1, 1, 2], [0, 2, 1]]) {
            const row = stencil(field, left + localX * cellWidth, top + localY * cellHeight, orderX, orderY);
            const scale = Math.sqrt(2 * lambda * area * factor) * width ** orderX * height ** orderY / length;
            row.weights = row.weights.map(weight => weight * scale);
            rows.push(row);
          }
        }
      }
    }
  }
  return rows;
}

const cross = (first, second, third) => (second.x - first.x) * (third.y - first.y) -
  (second.y - first.y) * (third.x - first.x);

function properIntersection(first, second, third, fourth) {
  return cross(first, second, third) * cross(first, second, fourth) < -1e-10 &&
    cross(third, fourth, first) * cross(third, fourth, second) < -1e-10;
}

export function geometryCheck(field, tau = 0.12, resolution = 40) {
  const countX = Math.max(resolution, Math.ceil(field.width / field.spacing * 4));
  const countY = Math.max(12, Math.ceil(countX * field.height / field.width));
  if (countX * countY > 100000) return { valid: false, reason: 'Pruefgitter zu gross; groebere Splineweite verwenden.' };
  const vertices = [];
  let minSingular = Infinity;
  let minDeterminant = Infinity;
  for (let row = 0; row <= countY; row++) {
    for (let col = 0; col <= countX; col++) {
      const px = col / countX * (field.width - 1);
      const py = row / countY * (field.height - 1);
      const value = evaluate(field, px, py, true);
      const determinant = value.j00 * value.j11 - value.j01 * value.j10;
      const trace = value.j00 ** 2 + value.j01 ** 2 + value.j10 ** 2 + value.j11 ** 2;
      const maxSingularSquared = (trace + Math.sqrt(Math.max(0, trace ** 2 - 4 * determinant ** 2))) / 2;
      const singular = Math.abs(determinant) / Math.sqrt(maxSingularSquared);
      minSingular = Math.min(minSingular, singular);
      minDeterminant = Math.min(minDeterminant, determinant);
      if (!(determinant > 0 && singular > tau && Number.isFinite(trace))) {
        return { valid: false, reason: 'Faltung oder fast singulaere Abbildung', minSingular, minDeterminant };
      }
      vertices.push({ x: value.x, y: value.y, px, py, id: vertices.length });
    }
  }
  const triangles = [];
  for (let row = 0; row < countY; row++) {
    for (let col = 0; col < countX; col++) {
      const index = row * (countX + 1) + col;
      triangles.push([vertices[index], vertices[index + 1], vertices[index + countX + 2]],
        [vertices[index], vertices[index + countX + 2], vertices[index + countX + 1]]);
    }
  }
  const minX = Math.min(...vertices.map(value => value.x));
  const minY = Math.min(...vertices.map(value => value.y));
  const maxX = Math.max(...vertices.map(value => value.x));
  const maxY = Math.max(...vertices.map(value => value.y));
  const binSize = Math.max(maxX - minX, maxY - minY) / 32 || 1;
  const bins = new Map();
  for (let index = 0; index < triangles.length; index++) {
    const triangle = triangles[index];
    if (cross(...triangle) <= 0) return { valid: false, reason: 'Umgeschlagene transformierte Rasterzelle' };
    const checked = new Set();
    const startCol = Math.floor((Math.min(...triangle.map(value => value.x)) - minX) / binSize);
    const endCol = Math.floor((Math.max(...triangle.map(value => value.x)) - minX) / binSize);
    const startRow = Math.floor((Math.min(...triangle.map(value => value.y)) - minY) / binSize);
    const endRow = Math.floor((Math.max(...triangle.map(value => value.y)) - minY) / binSize);
    for (let row = startRow; row <= endRow; row++) {
      for (let col = startCol; col <= endCol; col++) {
        const key = `${col},${row}`;
        const bucket = bins.get(key) ?? [];
        for (const otherIndex of bucket) {
          if (checked.has(otherIndex)) continue;
          checked.add(otherIndex);
          const other = triangles[otherIndex];
          if (triangle.some(vertex => other.some(otherVertex => vertex.id === otherVertex.id))) continue;
          const inside = (point, shape) => shape.every((vertex, edge) => cross(vertex, shape[(edge + 1) % 3], point) > 1e-8);
          if (triangle.some(vertex => inside(vertex, other)) || other.some(vertex => inside(vertex, triangle)) ||
              triangle.some((vertex, edge) => other.some((otherVertex, otherEdge) =>
                properIntersection(vertex, triangle[(edge + 1) % 3], otherVertex, other[(otherEdge + 1) % 3])))) {
            return { valid: false, reason: 'Globale Selbstueberlappung des transformierten Bildgitters' };
          }
        }
        bucket.push(index);
        bins.set(key, bucket);
      }
    }
  }
  const boundary = vertices.filter((_, index) => {
    const row = Math.floor(index / (countX + 1));
    const col = index % (countX + 1);
    return row === 0 || row === countY || col === 0 || col === countX;
  }).sort((first, second) => {
    const order = vertex => vertex.py === 0 ? vertex.px : vertex.px === field.width - 1 ? field.width + vertex.py :
      vertex.py === field.height - 1 ? 2 * field.width + field.height - vertex.px : 2 * (field.width + field.height) - vertex.py;
    return order(first) - order(second);
  });
  for (let first = 0; first < boundary.length; first++) {
    for (let second = first + 2; second < boundary.length; second++) {
      if (first === 0 && second === boundary.length - 1) continue;
      if (properIntersection(boundary[first], boundary[(first + 1) % boundary.length],
        boundary[second], boundary[(second + 1) % boundary.length])) return { valid: false, reason: 'Selbstschnitt des Bildrandes' };
    }
  }
  if (minSingular < tau * 2 && resolution < 80) return geometryCheck(field, tau, resolution * 2);
  return { valid: true, minSingular, minDeterminant, triangles, bounds: [minX, minY, maxX, maxY],
    samples: vertices.length, discrete: true };
}

export function statistics(errors) {
  if (!errors.length) return { count: 0, median: null, rms: null, p95: null };
  const sorted = [...errors].sort((first, second) => first - second);
  return { count: errors.length, median: sorted[Math.floor(sorted.length / 2)],
    rms: Math.sqrt(errors.reduce((sum, value) => sum + value * value, 0) / errors.length),
    p95: sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] };
}

export function validateFrames(field, frames, step, sigma = 1, delta = 1.5) {
  const errors = [];
  const sourceErrors = [];
  const spatial = [];
  let inversionFailures = 0;
  for (const frame of frames) {
    const fitting = frame.points.filter((_, index) => index % 2 === 0);
    const scoring = frame.points.filter((_, index) => index % 2 === 1);
    if (fitting.length < 3 || scoring.length < 3) continue;
    let weights = fitting.map(point => point.confidence);
    let pose;
    for (let iteration = 0; iteration < 8; iteration++) {
      pose = poseFor(fitting, field, step, weights);
      weights = fitting.map(point => {
        const expected = target(point, pose, step);
        const actual = evaluate(field, point.x, point.y);
        return point.confidence * Math.min(1, delta * sigma / (Math.hypot(expected.x - actual.x, expected.y - actual.y) || 1e-20));
      });
    }
    for (const point of scoring) {
      const expected = target(point, pose, step);
      const actual = evaluate(field, point.x, point.y);
      const error = Math.hypot(expected.x - actual.x, expected.y - actual.y);
      errors.push(error);
      const inverse = inversePoint(field, expected.x, expected.y, point.x, point.y);
      if (inverse) sourceErrors.push(Math.hypot(inverse.x - point.x, inverse.y - point.y));
      else inversionFailures++;
      spatial.push({ x: point.x, y: point.y, error, frameId: frame.id });
    }
  }
  return { ...statistics(errors), sourcePixels: statistics(sourceErrors), inversionFailures, spatial, frameCount: frames.length };
}

export async function fitCalibration(frames, options, previous = null, notify = () => {}, cancelled = () => false) {
  const training = frames.filter(frame => frame.enabled && frame.role === 'train' && frame.points.length >= 6);
  if (training.length < 2) throw new Error('Mindestens zwei aktive Trainingsframes mit 2D-Raster erforderlich.');
  if (!(options.step > 0 && options.sigma > 0 && options.lambda >= 0 && options.spacing > 0)) throw new Error('Ungueltige Fitparameter.');
  const preferred = training.find(frame => frame.id === previous?.referenceId);
  const candidates = [...training].sort((first, second) => second.points.length - first.points.length);
  if (preferred) candidates.splice(candidates.indexOf(preferred), 1), candidates.unshift(preferred);
  let reference = null;
  let field = null;
  let geometry = null;
  let bestDeterminant = -Infinity;
  for (const candidate of candidates) {
    const candidateField = previous?.field && previous.referenceId === candidate.id && previous.field.spacing === options.spacing &&
      previous.step === options.step ? { ...previous.field, coefficients: previous.field.coefficients.slice() } :
      affineStart(candidate, options.width, options.height, options.spacing, options.step);
    const candidateGeometry = geometryCheck(candidateField, options.tau);
    bestDeterminant = Math.max(bestDeterminant, candidateGeometry.minDeterminant ?? -Infinity);
    if (!candidateGeometry.valid) continue;
    reference = candidate;
    field = candidateField;
    geometry = candidateGeometry;
    break;
  }
  if (!reference) throw new Error(`Initialisierung ungueltig: Alle ${candidates.length} Trainingsframes sind gefaltet, fast singulaer oder gespiegelt (beste Determinante ${bestDeterminant.toPrecision(4)}). Rasterhaendigkeit und Punktverteilung pruefen.`);
  const observations = training.flatMap(frame => frame.points.map(point => ({ ...point, frameId: frame.id })));
  const totalWeight = observations.reduce((sum, point) => sum + point.confidence, 0);
  const dataRows = observations.map(point => stencil(field, point.x, point.y));
  const regularization = smoothRows(field, options.lambda);
  let robust = observations.map(() => 1);
  const poses = new Map();
  let stoppedByGeometry = false;
  let completedIterations = 0;
  let finalChange = Infinity;
  let useWebGpu = Boolean(options.useWebGpu);
  let accelerationError = null;
  let fitProfile = null;
  const iterations = options.iterations ?? 35;
  const fitStarted = performance.now();
  for (let iteration = 0; iteration < iterations; iteration++) {
    if (cancelled()) throw new Error('Berechnung abgebrochen.');
    const iterationStarted = performance.now();
    let offset = 0;
    for (const frame of training) {
      const weights = frame.points.map((point, index) => point.confidence * robust[offset + index]);
      poses.set(frame.id, frame.id === reference.id ? { theta: 0, tx: 0, ty: 0 } : poseFor(frame.points, field, options.step, weights));
      offset += frame.points.length;
    }
    const posesFinished = performance.now();
    const desired = observations.map(point => target(point, poses.get(point.frameId), options.step));
    const factors = observations.map((point, index) => Math.sqrt(point.confidence * robust[index] / totalWeight) / options.sigma);
    const rows = dataRows.map((row, index) => ({ indices: row.indices, weights: row.weights.map(weight => weight * factors[index]) })).concat(regularization);
    const initials = [0, 1].map(component => Float64Array.from({ length: field.nx * field.ny },
      (_, index) => field.coefficients[2 * index + component]));
    const rights = [0, 1].map(component => {
      const axis = component ? 'y' : 'x';
      const right = new Float64Array(rows.length);
      observations.forEach((point, index) => { right[index] = (desired[index][axis] - point[axis]) * factors[index]; });
      return right;
    });
    const systemFinished = performance.now();
    let solutions;
    let gpuProfile = null;
    if (useWebGpu) {
      try {
        solutions = await webGpuLsqrPair(rows, rights, initials[0].length, initials, options.linearIterations ?? 220,
          profile => { gpuProfile = profile; });
      } catch (error) {
        accelerationError = error.message;
        useWebGpu = false;
      }
    }
    if (!solutions) solutions = rights.map((right, component) =>
      lsqr(rows, right, initials[component].length, initials[component], options.linearIterations ?? 220));
    const solveFinished = performance.now();
    const candidate = new Float64Array(field.coefficients.length);
    let accepted = false;
    let change = 0;
    for (let damping = 1; damping >= 1 / 64; damping /= 2) {
      change = 0;
      for (let index = 0; index < candidate.length; index++) {
        const difference = solutions[index % 2][Math.floor(index / 2)] - field.coefficients[index];
        candidate[index] = field.coefficients[index] + damping * difference;
        change = Math.max(change, Math.abs(damping * difference));
      }
      geometry = geometryCheck({ ...field, coefficients: candidate }, options.tau);
      if (geometry.valid) { accepted = true; break; }
    }
    if (!accepted) { stoppedByGeometry = true; break; }
    const geometryFinished = performance.now();
    field = { ...field, coefficients: candidate.slice() };
    const errors = observations.map((point, index) => {
      const value = evaluate(field, point.x, point.y);
      return Math.hypot(value.x - desired[index].x, value.y - desired[index].y);
    });
    robust = errors.map(error => Math.min(1, options.delta * options.sigma / (error || 1e-20)));
    completedIterations = iteration + 1;
    finalChange = change;
    const rms = statistics(errors).rms;
    const iterationFinished = performance.now();
    fitProfile = { phasesMs: { poses: posesFinished - iterationStarted, system: systemFinished - posesFinished,
      solve: solveFinished - systemFinished, geometry: geometryFinished - solveFinished,
      residuals: iterationFinished - geometryFinished }, gpu: gpuProfile };
    notify({ stage: 'fit', iteration: iteration + 1, iterations, rms,
      profile: fitProfile,
      iterationSeconds: (iterationFinished - iterationStarted) / 1000,
      iterationsPerSecond: completedIterations * 1000 / Math.max(iterationFinished - fitStarted, 0.001),
      accelerator: useWebGpu ? 'WebGPU' : accelerationError ? 'CPU (WebGPU-Fallback)' : 'CPU' });
    await tick();
    if (iteration > 3 && change < 1e-4) break;
  }
  const trainingErrors = [];
  const spatial = [];
  for (const frame of training) {
    const pose = frame.id === reference.id ? { theta: 0, tx: 0, ty: 0 } : poseFor(frame.points, field, options.step);
    poses.set(frame.id, pose);
    for (const point of frame.points) {
      const expected = target(point, pose, options.step);
      const value = evaluate(field, point.x, point.y);
      const error = Math.hypot(value.x - expected.x, value.y - expected.y);
      trainingErrors.push(error);
      spatial.push({ x: point.x, y: point.y, error, frameId: frame.id });
    }
  }
  const validation = validateFrames(field, frames.filter(frame => frame.enabled && frame.role === 'validation'), options.step, options.sigma, options.delta);
  geometry = geometryCheck(field, options.tau);
  return { field, step: options.step, referenceId: reference.id, poses: Object.fromEntries(poses),
    metrics: { training: { ...statistics(trainingErrors), spatial }, validation,
      geometry: { valid: geometry.valid, minSingular: geometry.minSingular, minDeterminant: geometry.minDeterminant,
        samples: geometry.samples, discrete: true }, stoppedByGeometry, completedIterations, finalChange, fitProfile },
    quality: 'provisional', acceleration: { requested: Boolean(options.useWebGpu), used: useWebGpu, error: accelerationError } };
}