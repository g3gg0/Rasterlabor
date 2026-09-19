import { createPatchMask, MASK_SEARCH, maskIncludes } from './patch-mask.js';

// Coordinates are relative to the camera centre; camera poses map image to world.
export function frameGeometry(entry, field, maps) {
  const pose = entry.pose ?? entry.raw;
  if (!pose || entry.success === false || !maps) return null;
  const width = maps.outputWidth, height = maps.outputHeight;
  const offsetX = entry.mode === 'window' ? -width / 2 : maps.origin[0] - field.width / 2;
  const offsetY = entry.mode === 'window' ? -height / 2 : maps.origin[1] - field.height / 2;
  const c = Math.cos(pose.rotation), s = Math.sin(pose.rotation);
  const world = (x, y) => ({ x: pose.x + c * (x + offsetX) - s * (y + offsetY),
    y: pose.y + s * (x + offsetX) + c * (y + offsetY) });
  const local = point => {
    const dx = point.x - pose.x, dy = point.y - pose.y;
    return { x: c * dx + s * dy - offsetX, y: -s * dx + c * dy - offsetY };
  };
  return { entry, width, height, c, s, world, local, valid: maps.valid,
    corners: [[0, 0], [width, 0], [width, height], [0, height]].map(([x, y]) => world(x, y)),
    supports(point, pixelAllowed = null) {
      const { x, y } = local(point);
      return x >= 0 && y >= 0 && x < width && y < height && Boolean(maps.valid[Math.floor(y) * width + Math.floor(x)]) &&
        (!pixelAllowed || pixelAllowed(x, y));
    } };
}

export function localSelectionMask(geometry, point, diameterFraction = 0.4, baseMask = null) {
  const mask = createPatchMask(geometry.width, geometry.height, baseMask?.cellSize);
  const center = geometry.local(point);
  const radius = Math.min(geometry.width, geometry.height) * Math.max(0.1, Math.min(1, diameterFraction)) / 2;
  for (let row = 0; row < mask.height; row++) for (let column = 0; column < mask.width; column++) {
    const x = Math.min(mask.sourceWidth - 1, (column + 0.5) * mask.cellSize);
    const y = Math.min(mask.sourceHeight - 1, (row + 0.5) * mask.cellSize);
    if ((x - center.x) ** 2 + (y - center.y) ** 2 <= radius ** 2 && (!baseMask || maskIncludes(baseMask, x, y))) {
      mask.data[row * mask.width + column] = MASK_SEARCH;
    }
  }
  return mask;
}

export function localSelectionDistance(geometry, point) {
  const local = geometry.local(point);
  const dx = Math.max(0, -local.x, local.x - geometry.width);
  const dy = Math.max(0, -local.y, local.y - geometry.height);
  return Math.hypot(dx, dy);
}

export function localSelectionSupport(geometry, point, diameterFraction = 0.4, baseMask = null, pixelAllowed = null) {
  const { width, height } = geometry;
  const center = geometry.local(point);
  const radius = Math.min(width, height) * Math.max(0.1, Math.min(1, diameterFraction)) / 2;
  const cellSize = baseMask?.cellSize ?? Math.max(4, Math.ceil(Math.max(width, height) / 512));
  const stride = Math.max(1, Math.ceil(Math.sqrt(width * height / 6000)));
  let support = 0;
  for (let row = 2; row < height - 2; row += stride) for (let column = 2; column < width - 2; column += stride) {
    const x = Math.min(width - 1, (Math.floor(column / cellSize) + 0.5) * cellSize);
    const y = Math.min(height - 1, (Math.floor(row / cellSize) + 0.5) * cellSize);
    if ((x - center.x) ** 2 + (y - center.y) ** 2 <= radius ** 2 &&
        (!baseMask || maskIncludes(baseMask, x, y)) && (!pixelAllowed || pixelAllowed(column, row))) support++;
  }
  return support;
}

export function supportColor(frame) { return `hsl(${(frame * 137.508) % 360} 70% 38%)`; }

export function applyPixelMask(rgba, width, allowed) {
  for (let pixel = 0; pixel < rgba.length / 4; pixel++) {
    if (!allowed(pixel % width, Math.floor(pixel / width))) rgba[pixel * 4 + 3] = 0;
  }
  return rgba;
}

export function sharpestFramesFirst(geometries) {
  const score = geometry => Number.isFinite(geometry.entry.sharpness?.score) ? geometry.entry.sharpness.score : -Infinity;
  return [...geometries].sort((first, second) => {
    const firstScore = score(first), secondScore = score(second);
    return firstScore === secondScore ? first.entry.frame - second.entry.frame : firstScore > secondScore ? -1 : 1;
  });
}

export function approximateTopFrames(geometries, maxFrames, pixelAllowed = null, anchor = null, cellSize = null, sampleBounds = null) {
  if (!geometries.length || maxFrames < 1) return [];
  const corners = geometries.flatMap(geometry => geometry.corners);
  const minX = Math.floor(Math.max(Math.min(...corners.map(point => point.x)), sampleBounds?.minX ?? -Infinity));
  const minY = Math.floor(Math.max(Math.min(...corners.map(point => point.y)), sampleBounds?.minY ?? -Infinity));
  const maxX = Math.ceil(Math.min(Math.max(...corners.map(point => point.x)), sampleBounds?.maxX ?? Infinity));
  const maxY = Math.ceil(Math.min(Math.max(...corners.map(point => point.y)), sampleBounds?.maxY ?? Infinity));
  if (maxX <= minX || maxY <= minY) return [];
  const width = Math.max(1, maxX - minX), height = Math.max(1, maxY - minY);
  cellSize ??= Math.max(64, Math.ceil(geometries[0].width * 0.1));
  const columns = Math.ceil(width / cellSize), rows = Math.ceil(height / cellSize);
  const counts = new Uint8Array(columns * rows);
  const selected = [];
  let anchorCount = 0;
  for (const geometry of sharpestFramesFirst(geometries)) {
    const left = Math.max(0, Math.floor((Math.min(...geometry.corners.map(point => point.x)) - minX) / cellSize));
    const right = Math.min(columns - 1, Math.floor((Math.max(...geometry.corners.map(point => point.x)) - minX) / cellSize));
    const top = Math.max(0, Math.floor((Math.min(...geometry.corners.map(point => point.y)) - minY) / cellSize));
    const bottom = Math.min(rows - 1, Math.floor((Math.max(...geometry.corners.map(point => point.y)) - minY) / cellSize));
    const covered = [];
    let needed = Boolean(anchor && anchorCount < maxFrames && geometry.supports(anchor, pixelAllowed));
    for (let row = top; row <= bottom; row++) for (let col = left; col <= right; col++) {
      const point = sampleBounds ?
        { x: (minX + col * cellSize + Math.min(maxX, minX + (col + 1) * cellSize)) / 2,
          y: (minY + row * cellSize + Math.min(maxY, minY + (row + 1) * cellSize)) / 2 } :
        { x: minX + (col + 0.5) * cellSize, y: minY + (row + 0.5) * cellSize };
      if (!geometry.supports(point, pixelAllowed)) continue;
      const index = row * columns + col;
      covered.push(index);
      if (counts[index] < maxFrames) needed = true;
    }
    if (!needed) continue;
    selected.push(geometry);
    if (anchor && anchorCount < maxFrames && geometry.supports(anchor, pixelAllowed)) anchorCount++;
    for (const index of covered) if (counts[index] < maxFrames) counts[index]++;
  }
  return selected;
}

function edgeFeatherPixels(width, height, fraction) {
  return Math.min(width * Math.max(0, fraction), Math.max(0, (Math.min(width, height) - 1) / 2));
}

export function edgeFeatherWeight(x, y, width, height, fraction = 0.1) {
  const feather = edgeFeatherPixels(width, height, fraction);
  if (!(feather > 0)) return 1;
  const distance = Math.min(x + 0.5, width - x - 0.5, y + 0.5, height - y - 0.5);
  const t = Math.max(0, Math.min(1, distance / feather));
  return t * t * (3 - 2 * t);
}

export function edgeFeatherMask(width, height, allowed = () => true, fraction = 0.1) {
  const distance = new Uint16Array(width * height);
  const maximum = 0xffff;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const index = y * width + x;
    distance[index] = allowed(x, y) ? Math.min(maximum, 3 * Math.min(x + 1, y + 1, width - x, height - y)) : 0;
  }
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const index = y * width + x;
    if (x) distance[index] = Math.min(distance[index], distance[index - 1] + 3);
    if (y) distance[index] = Math.min(distance[index], distance[index - width] + 3);
    if (x && y) distance[index] = Math.min(distance[index], distance[index - width - 1] + 4);
    if (x + 1 < width && y) distance[index] = Math.min(distance[index], distance[index - width + 1] + 4);
  }
  for (let y = height - 1; y >= 0; y--) for (let x = width - 1; x >= 0; x--) {
    const index = y * width + x;
    if (x + 1 < width) distance[index] = Math.min(distance[index], distance[index + 1] + 3);
    if (y + 1 < height) distance[index] = Math.min(distance[index], distance[index + width] + 3);
    if (x + 1 < width && y + 1 < height) distance[index] = Math.min(distance[index], distance[index + width + 1] + 4);
    if (x && y + 1 < height) distance[index] = Math.min(distance[index], distance[index + width - 1] + 4);
  }
  const feather = edgeFeatherPixels(width, height, fraction); const weights = new Uint8Array(distance.length);
  for (let index = 0; index < weights.length; index++) {
    if (!distance[index]) continue;
    if (!feather) { weights[index] = 255; continue; }
    const t = Math.max(0, Math.min(1, (distance[index] / 3 - 0.5) / feather));
    weights[index] = Math.round(255 * t * t * (3 - 2 * t));
  }
  return weights;
}

export function applyEdgeFeather(rgba, width, height, fraction = 0.1, weights = null) {
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const alpha = y * width * 4 + x * 4 + 3;
    const weight = weights ? weights[y * width + x] / 255 : edgeFeatherWeight(x, y, width, height, fraction);
    rgba[alpha] = Math.round(rgba[alpha] * weight);
  }
  return rgba;
}

export function accumulateFrame(sum, rgba, counts = null, maxFrames = Infinity) {
  for (let i = 0; i < rgba.length; i += 4) {
    if (maxFrames > 0 && sum[i + 3] >= maxFrames) continue;
    const alpha = maxFrames > 0 ? Math.min(rgba[i + 3] / 255, maxFrames - sum[i + 3]) : rgba[i + 3] / 255;
    if (!alpha) continue;
    sum[i] += rgba[i] * alpha; sum[i + 1] += rgba[i + 1] * alpha;
    sum[i + 2] += rgba[i + 2] * alpha; sum[i + 3] += alpha;
    if (counts) counts[i / 4]++;
  }
}

export function averagedFrames(sum, preserveCoverage = false) {
  const rgba = new Uint8ClampedArray(sum.length);
  for (let i = 0; i < sum.length; i += 4) {
    if (!sum[i + 3]) continue;
    for (let channel = 0; channel < 3; channel++) rgba[i + channel] = sum[i + channel] / sum[i + 3];
    // Normalize each covered pixel to opaque; uncovered pixels stay transparent.
    rgba[i + 3] = preserveCoverage ? Math.min(255, sum[i + 3] * 255) : 255;
  }
  return rgba;
}

export function sparsePathFrames(geometries, limit = 12) {
  if (!geometries.length || limit < 1) return [];
  if (geometries.length <= limit) return geometries.slice();
  const centers = geometries.map(geometry => geometry.world(geometry.width / 2, geometry.height / 2));
  const selected = new Set([geometries.length - 1]);
  if (limit > 1) selected.add(0);
  while (selected.size < limit) {
    let bestIndex = -1, bestDistance = -1;
    for (let index = 0; index < geometries.length; index++) {
      if (selected.has(index)) continue;
      let nearest = Infinity;
      for (const selectedIndex of selected) {
        const dx = centers[index].x - centers[selectedIndex].x;
        const dy = centers[index].y - centers[selectedIndex].y;
        nearest = Math.min(nearest, dx * dx + dy * dy);
      }
      if (nearest > bestDistance) { bestIndex = index; bestDistance = nearest; }
    }
    selected.add(bestIndex);
  }
  return [...selected].sort((first, second) => first - second).map(index => geometries[index]);
}

export function nearestPathEntry(entries, project, point, radius = 12) {
  let nearest = null, distance = radius;
  for (const entry of entries) {
    if (!entry.pose) continue;
    const p = project(entry.pose), d = Math.hypot(p.x - point.x, p.y - point.y);
    if (d <= distance) { nearest = entry; distance = d; }
  }
  if (nearest) return nearest;
  for (let index = 1; index < entries.length; index++) {
    const first = entries[index - 1], last = entries[index];
    if (!first.pose || !last.pose) continue;
    const a = project(first.pose), b = project(last.pose);
    const dx = b.x - a.x, dy = b.y - a.y, length = dx * dx + dy * dy;
    if (!length) continue;
    const t = Math.max(0, Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / length));
    const d = Math.hypot(point.x - a.x - t * dx, point.y - a.y - t * dy);
    if (d <= distance) { nearest = t < 0.5 ? first : last; distance = d; }
  }
  return nearest;
}
