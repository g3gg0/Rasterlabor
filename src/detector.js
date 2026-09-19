const median = values => [...values].sort((first, second) => first - second)[Math.floor(values.length / 2)] ?? 0;
const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));

function imageSampler(gray, width, height) {
  return (px, py) => {
    if (px < 0 || py < 0 || px >= width - 1 || py >= height - 1) return NaN;
    const col = Math.floor(px);
    const row = Math.floor(py);
    const localX = px - col;
    const localY = py - row;
    const index = row * width + col;
    return (1 - localY) * ((1 - localX) * gray[index] + localX * gray[index + 1]) +
      localY * ((1 - localX) * gray[index + width] + localX * gray[index + width + 1]);
  };
}

function dominantAngle(gray, width, height, roi) {
  const histogram = new Float64Array(180);
  for (let py = Math.max(2, roi.y); py < Math.min(height - 2, roi.y + roi.height); py += 2) {
    for (let px = Math.max(2, roi.x); px < Math.min(width - 2, roi.x + roi.width); px += 2) {
      const index = py * width + px;
      const gradientX = gray[index + 1] - gray[index - 1];
      const gradientY = gray[index + width] - gray[index - width];
      const magnitude = Math.hypot(gradientX, gradientY);
      if (magnitude < 12) continue;
      const angle = ((Math.atan2(gradientY, gradientX) * 180 / Math.PI) % 90 + 90) % 90;
      const bin = Math.round(angle * 2) % 180;
      for (let offset = -8; offset <= 8; offset++) histogram[(bin + offset + 180) % 180] += magnitude * (9 - Math.abs(offset));
    }
  }
  let best = 0;
  histogram.forEach((value, index) => { if (value > histogram[best]) best = index; });
  const degrees = best / 2;
  return (degrees > 45 ? degrees - 90 : degrees) * Math.PI / 180;
}

function peaks(profile, threshold, radius) {
  const candidates = [];
  for (let index = 2; index < profile.length - 2; index++) {
    if (!(profile[index] >= threshold && profile[index] >= profile[index - 1] && profile[index] > profile[index + 1])) continue;
    let sum = 0;
    let moment = 0;
    const half = Math.max(1, Math.floor(radius / 2));
    for (let offset = -half; offset <= half; offset++) {
      const value = Math.max(0, profile[index + offset] || 0);
      sum += value;
      moment += (index + offset) * value;
    }
    const candidate = { position: moment / sum, strength: profile[index] };
    const previous = candidates.at(-1);
    if (previous && candidate.position - previous.position < radius * 1.6) {
      if (candidate.strength > previous.strength) candidates[candidates.length - 1] = candidate;
    } else candidates.push(candidate);
  }
  return candidates;
}

function checkTopology(points) {
  const byRow = new Map();
  const byCol = new Map();
  for (const point of points) {
    if (!byRow.has(point.row)) byRow.set(point.row, []);
    if (!byCol.has(point.col)) byCol.set(point.col, []);
    byRow.get(point.row).push(point);
    byCol.get(point.col).push(point);
  }
  if (byRow.size < 3 || byCol.size < 3 || points.length < 9) return 'Weniger als 3 Zeilen, 3 Spalten oder 9 verlaessliche Kreuzungen.';
  const distances = [];
  for (const [groups, axis] of [[byRow, 'col'], [byCol, 'row']]) {
    for (const group of groups.values()) {
      group.sort((first, second) => first[axis] - second[axis]);
      const gaps = [];
      for (let index = 1; index < group.length; index++) {
        const before = group[index - 1];
        const point = group[index];
        gaps.push(Math.hypot(point.x - before.x, point.y - before.y) / (point[axis] - before[axis]));
      }
      const typical = median(gaps);
      if (gaps.some(gap => gap < typical * 0.55 || gap > typical * 1.6)) return 'Inkonsistente Rasterabstaende: fehlende Linie, Doppelkante oder zu starke lokale Verzerrung.';
      distances.push(...gaps);
    }
  }
  const lookup = new Map(points.map(point => [`${point.col},${point.row}`, point]));
  let cells = 0;
  for (const point of points) {
    const right = lookup.get(`${point.col + 1},${point.row}`);
    const below = lookup.get(`${point.col},${point.row + 1}`);
    if (!right || !below) continue;
    const area = (right.x - point.x) * (below.y - point.y) - (right.y - point.y) * (below.x - point.x);
    if (area <= 0) return 'Checkerboard-Zellen sind inkonsistent angeordnet oder die Haendigkeit ist gespiegelt.';
    cells++;
  }
  if (cells < 4) return 'Keine ausreichend zusammenhaengende zweidimensionale Rastertopologie.';
  return { step: median(distances), rows: byRow.size, cols: byCol.size };
}

function chessboardPoints(sample, width, height, roi, angle, expectedStep, threshold, preparedCandidates = null, preparedCorners = null) {
  const started = performance.now();
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const radius = Math.max(2, Math.round(expectedStep * 0.18));
  const responseAt = (px, py) => {
    const quadrants = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([across, along]) =>
      sample(px + radius * (across * cosine - along * sine), py + radius * (across * sine + along * cosine)));
    if (!quadrants.every(Number.isFinite)) return 0;
    const contrast = Math.abs(quadrants[0] + quadrants[2] - quadrants[1] - quadrants[3]) / 2;
    const imbalance = Math.abs(quadrants[0] - quadrants[2]) + Math.abs(quadrants[1] - quadrants[3]);
    return Math.max(0, contrast - imbalance);
  };
  const candidates = preparedCandidates ? [...preparedCandidates] : [];
  const scan = Math.max(1, Math.floor(radius / 2));
  if (!preparedCandidates) {
    for (let py = roi.y + radius * 2; py < roi.y + roi.height - radius * 2; py += scan) {
      for (let px = roi.x + radius * 2; px < roi.x + roi.width - radius * 2; px += scan) {
        const response = responseAt(px, py);
        if (response > Math.max(30, threshold * 3)) candidates.push({ x: px, y: py, response });
      }
    }
  }
  if (!preparedCorners) candidates.sort((first, second) => second.response - first.response);
  const candidatesFinished = performance.now();
  const corners = preparedCorners ? [...preparedCorners] : [];
  if (!preparedCorners) {
    for (const candidate of candidates) {
      if (corners.some(point => Math.hypot(point.x - candidate.x, point.y - candidate.y) < expectedStep * 0.5)) continue;
      let px = candidate.x;
      let py = candidate.y;
      for (let iteration = 0; iteration < 8; iteration++) {
        let xx = 0; let xy = 0; let yy = 0; let bx = 0; let by = 0;
        for (let offsetY = -radius; offsetY <= radius; offsetY++) {
          for (let offsetX = -radius; offsetX <= radius; offsetX++) {
            const locationX = px + offsetX;
            const locationY = py + offsetY;
            const gx = (sample(locationX + 1, locationY) - sample(locationX - 1, locationY)) / 2;
            const gy = (sample(locationX, locationY + 1) - sample(locationX, locationY - 1)) / 2;
            if (!Number.isFinite(gx + gy)) continue;
            const weight = Math.exp(-(offsetX ** 2 + offsetY ** 2) / radius ** 2);
            xx += weight * gx * gx; xy += weight * gx * gy; yy += weight * gy * gy;
            bx += weight * (gx * gx * locationX + gx * gy * locationY);
            by += weight * (gx * gy * locationX + gy * gy * locationY);
          }
        }
        const determinant = xx * yy - xy * xy;
        if (determinant < 1e-6) break;
        const nextX = (yy * bx - xy * by) / determinant;
        const nextY = (xx * by - xy * bx) / determinant;
        if (Math.hypot(nextX - candidate.x, nextY - candidate.y) > radius * 2) break;
        const change = Math.hypot(nextX - px, nextY - py);
        px = nextX; py = nextY;
        if (change < 0.005) break;
      }
      if (responseAt(px, py) > Math.max(30, threshold * 3)) corners.push({ x: px, y: py, confidence: clamp(candidate.response / 255, 0.1, 1) });
      if (corners.length > 3000) break;
    }
  }
  const refinementFinished = performance.now();
  const timing = { candidatesMs: candidatesFinished - started, refinementMs: refinementFinished - candidatesFinished,
    candidateCount: candidates.length, cornerCount: corners.length };
  if (!corners.length) return { points: [], timing: { ...timing, topologyMs: 0 } };
  const assigned = new Map([[0, { col: 0, row: 0 }]]);
  const queue = [0];
  for (let cursor = 0; cursor < queue.length; cursor++) {
    const index = queue[cursor];
    const point = corners[index];
    for (const [directionCol, directionRow] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      let best = -1;
      let distance = Infinity;
      corners.forEach((other, otherIndex) => {
        if (otherIndex === index) return;
        const deltaX = other.x - point.x;
        const deltaY = other.y - point.y;
        const across = deltaX * cosine + deltaY * sine;
        const along = -deltaX * sine + deltaY * cosine;
        const forward = directionCol * across + directionRow * along;
        const side = -directionRow * across + directionCol * along;
        if (forward > expectedStep * 0.55 && forward < expectedStep * 1.55 && Math.abs(side) < forward * 0.4 && forward < distance) {
          best = otherIndex; distance = forward;
        }
      });
      if (best < 0) continue;
      const coordinate = { col: assigned.get(index).col + directionCol, row: assigned.get(index).row + directionRow };
      if (assigned.has(best)) {
        if (assigned.get(best).col !== coordinate.col || assigned.get(best).row !== coordinate.row) {
          return { points: [], timing: { ...timing, topologyMs: performance.now() - refinementFinished } };
        }
      } else { assigned.set(best, coordinate); queue.push(best); }
    }
  }
  return { points: [...assigned].map(([index, coordinate]) => ({ ...corners[index], ...coordinate })),
    timing: { ...timing, topologyMs: performance.now() - refinementFinished } };
}

export function detectGrid(image, options = {}) {
  const { width, height, data } = image;
  const gray = data instanceof Float32Array && data.length === width * height ? data : new Float32Array(width * height);
  if (gray !== data) for (let index = 0; index < gray.length; index++) gray[index] = data.length === gray.length ? data[index] :
    0.299 * data[4 * index] + 0.587 * data[4 * index + 1] + 0.114 * data[4 * index + 2];
  const roi = options.roi ?? { x: 0, y: 0, width, height };
  const sampleRaw = imageSampler(gray, width, height);
  const sample = (px, py) => px >= roi.x && py >= roi.y && px < roi.x + roi.width && py < roi.y + roi.height ? sampleRaw(px, py) : NaN;
  const angle = Number.isFinite(options.precomputedAngle) ? options.precomputedAngle : dominantAngle(gray, width, height, roi);
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const corners = [[roi.x, roi.y], [roi.x + roi.width - 1, roi.y], [roi.x, roi.y + roi.height - 1], [roi.x + roi.width - 1, roi.y + roi.height - 1]];
  const us = corners.map(([px, py]) => cosine * px + sine * py);
  const vs = corners.map(([px, py]) => -sine * px + cosine * py);
  const rangeU = [Math.min(...us), Math.max(...us)];
  const rangeV = [Math.min(...vs), Math.max(...vs)];
  const rotated = (across, along) => sample(cosine * across - sine * along, sine * across + cosine * along);
  const threshold = options.threshold ?? 16;
  let expectedStep = options.approxStep || 0;
  if (!expectedStep) {
    const gaps = [];
    for (const fraction of [0.2, 0.4, 0.6, 0.8]) {
      const profile = new Float32Array(Math.ceil(rangeU[1] - rangeU[0]));
      const along = rangeV[0] + fraction * (rangeV[1] - rangeV[0]);
      for (let index = 0; index < profile.length; index++) {
        const across = rangeU[0] + index;
        const center = rotated(across, along);
        const low = rotated(across - 6, along);
        const high = rotated(across + 6, along);
        profile[index] = Math.abs(high - low);
      }
      const candidates = peaks(profile, threshold, 4);
      for (let index = 1; index < candidates.length; index++) gaps.push(candidates[index].position - candidates[index - 1].position);
    }
    expectedStep = median(gaps.filter(gap => gap > 10));
  }
  const result = { points: [], lines: [], rejected: [], roi, angle, success: false, reason: '', confidence: 0, coverage: 0 };
  if (!(expectedStep >= 10)) return { ...result, reason: 'Kein periodisches Raster gefunden. Rasterabstand in Pixeln angeben oder Kontrast pruefen.' };
  const chessboard = chessboardPoints(sample, width, height, roi, angle, expectedStep, threshold,
    options.precomputedCandidates, options.precomputedCorners);
  result.points = chessboard.points;
  result.detectorTiming = chessboard.timing;
  const topology = checkTopology(result.points);
  if (typeof topology === 'string') return { ...result, reason: topology };
  if (options.columns && topology.cols > options.columns || options.rows && topology.rows > options.rows) {
    return { ...result, ...topology, reason: `Erkannt: ${topology.cols} Spalten und ${topology.rows} Zeilen von Kreuzungspunkten; angegeben: ${options.columns || 'unbekannt'} Spalten und ${options.rows || 'unbekannt'} Zeilen. Mustergroesse und Linienzuordnung pruefen (Punkte, nicht Zellen).` };
  }
  const occupied = new Set(result.points.map(point => `${Math.floor(point.x / width * 12)},${Math.floor(point.y / height * 9)}`));
  return { ...result, success: true, ...topology, coverage: occupied.size / 108,
    confidence: result.points.reduce((sum, point) => sum + point.confidence, 0) / result.points.length };
}