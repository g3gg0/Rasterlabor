import { detectGrid } from './detector.js';

export function checkerboardCells(points) {
  const indexed = new Map(points.map(point => [`${point.col},${point.row}`, point]));
  const cells = [];
  for (const point of points) {
    const corners = [point, indexed.get(`${point.col + 1},${point.row}`),
      indexed.get(`${point.col + 1},${point.row + 1}`), indexed.get(`${point.col},${point.row + 1}`)];
    if (corners.some(corner => !corner || !Number.isFinite(corner.x + corner.y))) continue;
    const turns = corners.map((corner, index) => {
      const next = corners[(index + 1) % 4]; const after = corners[(index + 2) % 4];
      return (next.x - corner.x) * (after.y - next.y) - (next.y - corner.y) * (after.x - next.x);
    });
    if (!turns.every(value => value > 0) && !turns.every(value => value < 0)) continue;
    const area = Math.abs(corners.reduce((sum, corner, index) => {
      const next = corners[(index + 1) % 4]; return sum + corner.x * next.y - next.x * corner.y;
    }, 0)) / 2;
    cells.push({ col: point.col, row: point.row, corners: corners.map(({ x, y }) => ({ x, y })), area,
      x: corners.reduce((sum, corner) => sum + corner.x, 0) / 4,
      y: corners.reduce((sum, corner) => sum + corner.y, 0) / 4 });
  }
  return cells;
}

export function summarizeCheckerboard(cells) {
  if (!cells.length) return { count: 0, minimum: null, maximum: null, mean: null, median: null, deviation: null, maxDeviation: null };
  const areas = cells.map(cell => cell.area).sort((first, second) => first - second);
  const maximum = areas.at(-1); const minimum = areas[0];
  const mean = areas.reduce((sum, value) => sum + value, 0) / areas.length;
  const deviation = Math.sqrt(areas.reduce((sum, value) => sum + (value - mean) ** 2, 0) / areas.length);
  for (const cell of cells) cell.relativeDeviation = (maximum - cell.area) / maximum;
  return { count: cells.length, minimum, maximum, mean, median: (areas[Math.floor(areas.length / 2)] + areas[Math.floor((areas.length - 1) / 2)]) / 2,
    deviation, maxDeviation: (maximum - minimum) / maximum };
}

export function checkerboardColor(relativeDeviation) {
  const fraction = Math.max(0, Math.min(1, relativeDeviation));
  return `hsl(${120 * (1 - fraction)} 75% 45%)`;
}

export function refineCheckerboardCell(cell, image, threshold = 10) {
  const { width, height, data } = image;
  const sample = (x, y) => {
    const left = Math.floor(x); const top = Math.floor(y);
    if (left < 0 || top < 0 || left + 1 >= width || top + 1 >= height) return NaN;
    const across = x - left; const down = y - top;
    return (1 - down) * ((1 - across) * data[top * width + left] + across * data[top * width + left + 1]) +
      down * ((1 - across) * data[(top + 1) * width + left] + across * data[(top + 1) * width + left + 1]);
  };
  const lines = []; let polarity = 0;
  for (const [index, start] of cell.corners.entries()) {
    const end = cell.corners[(index + 1) % 4];
    const length = Math.hypot(end.x - start.x, end.y - start.y);
    if (length < 4) return null;
    const tangentX = (end.x - start.x) / length; const tangentY = (end.y - start.y) / length;
    const normalX = -tangentY; const normalY = tangentX;
    const radius = Math.max(2, Math.min(12, length * 0.035));
    const tolerance = Math.max(1, length * 0.012);
    const search = Math.ceil(Math.max(4, length * 0.22));
    const observations = [];
    for (let position = 0; position < 13; position++) {
      const fraction = 0.12 + position * 0.76 / 12;
      const x = start.x + fraction * (end.x - start.x); const y = start.y + fraction * (end.y - start.y);
      const profile = []; let strongest = 0;
      for (let offset = -search; offset <= search; offset++) {
        let contrast = 0;
        for (let along = -2; along <= 2; along++) {
          const centerX = x + along * radius * 0.5 * tangentX + offset * normalX;
          const centerY = y + along * radius * 0.5 * tangentY + offset * normalY;
          contrast += sample(centerX + radius * normalX, centerY + radius * normalY) -
            sample(centerX - radius * normalX, centerY - radius * normalY);
        }
        contrast /= 5; profile.push(contrast);
        if (Math.abs(contrast) > Math.abs(strongest)) strongest = contrast;
      }
      if (Math.abs(strongest) < Math.max(15, threshold * 2)) continue;
      const peak = profile.indexOf(strongest);
      if (peak === 0 || peak === profile.length - 1) continue;
      let lower = peak; let upper = peak;
      while (lower > 0 && profile[lower - 1] / strongest >= 0.85) lower--;
      while (upper + 1 < profile.length && profile[upper + 1] / strongest >= 0.85) upper++;
      let weight = 0; let offsetSum = 0;
      for (let location = lower; location <= upper; location++) {
        const amount = Math.abs(profile[location]); weight += amount; offsetSum += amount * (location - search);
      }
      observations.push({ along: (fraction - 0.5) * length, offset: offsetSum / weight, sign: Math.sign(strongest) });
    }
    const sign = Math.sign(observations.reduce((sum, observation) => sum + observation.sign, 0));
    if (!sign || (polarity && sign !== polarity)) return null;
    polarity = sign;
    const candidates = observations.filter(observation => observation.sign === sign);
    let inliers = []; let bestError = Infinity;
    for (const [firstIndex, first] of candidates.entries()) for (const second of candidates.slice(firstIndex + 1)) {
      if (second.along - first.along < length * 0.25) continue;
      const slope = (second.offset - first.offset) / (second.along - first.along);
      const intercept = first.offset - slope * first.along;
      const matches = candidates.filter(point => Math.abs(point.offset - slope * point.along - intercept) <= tolerance);
      const error = matches.reduce((sum, point) => sum + (point.offset - slope * point.along - intercept) ** 2, 0);
      if (matches.length > inliers.length || (matches.length === inliers.length && error < bestError)) { inliers = matches; bestError = error; }
    }
    if (inliers.length < 8 || inliers.at(-1).along - inliers[0].along < length * 0.5) return null;
    const meanAlong = inliers.reduce((sum, point) => sum + point.along, 0) / inliers.length;
    const meanOffset = inliers.reduce((sum, point) => sum + point.offset, 0) / inliers.length;
    const slope = inliers.reduce((sum, point) => sum + (point.along - meanAlong) * (point.offset - meanOffset), 0) /
      inliers.reduce((sum, point) => sum + (point.along - meanAlong) ** 2, 0);
    const fittedX = normalX - slope * tangentX; const fittedY = normalY - slope * tangentY;
    lines.push({ x: fittedX, y: fittedY, distance: fittedX * (start.x + end.x) / 2 + fittedY * (start.y + end.y) / 2 + meanOffset - slope * meanAlong, length });
  }
  const corners = [];
  for (const [index, line] of lines.entries()) {
    const previous = lines[(index + 3) % 4];
    const determinant = previous.x * line.y - line.x * previous.y;
    if (Math.abs(determinant) < 0.2) return null;
    const x = (previous.distance * line.y - line.distance * previous.y) / determinant;
    const y = (previous.x * line.distance - line.x * previous.distance) / determinant;
    if (Math.hypot(x - cell.corners[index].x, y - cell.corners[index].y) > Math.min(line.length, previous.length) * 0.3) return null;
    corners.push({ x, y, col: index === 1 || index === 2 ? 1 : 0, row: index >= 2 ? 1 : 0 });
  }
  return checkerboardCells(corners)[0] ?? null;
}

export function analyzeCheckerboard(image, { step, threshold = 10 } = {}) {
  if (!Number.isFinite(step) || step < 10 || step > Math.min(image.width, image.height) / 2) throw new Error('Feldkante muss zwischen 10 px und der halben Bildgroesse liegen.');
  const started = performance.now();
  const gray = new Float32Array(image.width * image.height);
  for (let index = 0; index < gray.length; index++) gray[index] = image.data[index * 4 + 3] < 255 ? NaN :
    0.299 * image.data[index * 4] + 0.587 * image.data[index * 4 + 1] + 0.114 * image.data[index * 4 + 2];
  const tileSize = Math.ceil(Math.max(256, step * 12));
  const stride = Math.max(1, tileSize - Math.ceil(step * 4));
  const cells = []; const buckets = new Map(); let regions = 0;
  const bucketSize = step / 2;
  for (let top = 0; top < image.height - step * 2; top += stride) {
    for (let left = 0; left < image.width - step * 2; left += stride) {
      const roi = { x: left, y: top, width: Math.min(tileSize, image.width - left), height: Math.min(tileSize, image.height - top) };
      const detection = detectGrid({ width: image.width, height: image.height, data: gray },
        { approxStep: step, threshold, roi });
      regions++;
      for (const candidate of checkerboardCells(detection.points)) {
        const cell = refineCheckerboardCell(candidate, { width: image.width, height: image.height, data: gray }, threshold);
        if (!cell) continue;
        let valid = true;
        for (let row = 0; row <= 4 && valid; row++) for (let column = 0; column <= 4; column++) {
          const across = column / 4; const down = row / 4; const corners = cell.corners;
          const x = (1 - down) * ((1 - across) * corners[0].x + across * corners[1].x) + down * ((1 - across) * corners[3].x + across * corners[2].x);
          const y = (1 - down) * ((1 - across) * corners[0].y + across * corners[1].y) + down * ((1 - across) * corners[3].y + across * corners[2].y);
          if (x < 0 || y < 0 || x >= image.width || y >= image.height || !Number.isFinite(gray[Math.floor(y) * image.width + Math.floor(x)])) { valid = false; break; }
        }
        if (!valid) continue;
        const col = Math.floor(cell.x / bucketSize); const row = Math.floor(cell.y / bucketSize);
        let duplicate = false;
        for (let offsetY = -1; offsetY <= 1; offsetY++) for (let offsetX = -1; offsetX <= 1; offsetX++) {
          if ((buckets.get(`${col + offsetX},${row + offsetY}`) ?? []).some(other => Math.hypot(other.x - cell.x, other.y - cell.y) < step * 0.3)) duplicate = true;
        }
        if (duplicate) continue;
        const key = `${col},${row}`;
        if (!buckets.has(key)) buckets.set(key, []);
        buckets.get(key).push(cell); cells.push(cell);
      }
    }
  }
  return { cells, statistics: summarizeCheckerboard(cells), regions, milliseconds: performance.now() - started };
}

export function drawCheckerboardAnalysis(context, analysis, scale) {
  context.save(); context.lineWidth = 1 / scale; context.textAlign = 'center'; context.textBaseline = 'middle';
  for (const cell of analysis.cells) {
    context.beginPath(); cell.corners.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y)); context.closePath();
    context.fillStyle = checkerboardColor(analysis.statistics.maxDeviation ? cell.relativeDeviation / analysis.statistics.maxDeviation : 0); context.globalAlpha = 0.42; context.fill();
    context.globalAlpha = 1; context.strokeStyle = '#ffffff'; context.stroke();
    const label = cell.area.toFixed(0);
    const edge = Math.min(...cell.corners.map((point, index) => Math.hypot(point.x - cell.corners[(index + 1) % 4].x, point.y - cell.corners[(index + 1) % 4].y)));
    context.font = `${Math.min(12 / scale, edge / (label.length * 0.7 + 2))}px 'IBM Plex Sans'`;
    context.lineWidth = 3 / scale; context.strokeStyle = '#18221f'; context.strokeText(label, cell.x, cell.y);
    context.fillStyle = '#ffffff'; context.fillText(label, cell.x, cell.y); context.lineWidth = 1 / scale;
  }
  context.restore();
}