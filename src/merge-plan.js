import { sharpestFramesFirst } from './path-support.js';

export function blurriestFramesFirst(geometries) {
  const score = geometry => Number.isFinite(geometry.entry.sharpness?.score) ? geometry.entry.sharpness.score : -Infinity;
  return [...geometries].sort((first, second) => {
    const difference = score(first) - score(second);
    return difference || first.entry.frame - second.entry.frame;
  });
}

export function createMergeCoverage(pixelAllowed = null) {
  const masks = new WeakMap();
  const blockSize = 16;
  return (geometry, bounds) => {
    if (!geometry.local || !geometry.valid) return 1;
    const { width, height, valid } = geometry;
    let mask = masks.get(valid);
    if (!mask) {
      const columns = Math.ceil(width / blockSize), rows = Math.ceil(height / blockSize), stride = columns + 1;
      const occupied = new Uint32Array(stride * (rows + 1)), incomplete = new Uint32Array(occupied.length);
      for (let row = 0; row < rows; row++) for (let column = 0; column < columns; column++) {
        let any = false, all = true;
        for (let y = row * blockSize; y < Math.min(height, (row + 1) * blockSize); y++) {
          for (let x = column * blockSize; x < Math.min(width, (column + 1) * blockSize); x++) {
            const allowed = Boolean(valid[y * width + x]) && (!pixelAllowed || pixelAllowed(x, y));
            any ||= allowed; all &&= allowed;
          }
        }
        const index = (row + 1) * stride + column + 1;
        occupied[index] = Number(any) + occupied[index - 1] + occupied[index - stride] - occupied[index - stride - 1];
        incomplete[index] = Number(!all) + incomplete[index - 1] + incomplete[index - stride] - incomplete[index - stride - 1];
      }
      mask = { occupied, incomplete, stride, columns, rows };
      masks.set(valid, mask);
    }
    const corners = [[bounds.minX, bounds.minY], [bounds.maxX, bounds.minY],
      [bounds.maxX, bounds.maxY], [bounds.minX, bounds.maxY]].map(([x, y]) => geometry.local({ x, y }));
    const left = Math.min(...corners.map(point => point.x)) - 1;
    const right = Math.max(...corners.map(point => point.x)) + 1;
    const top = Math.min(...corners.map(point => point.y)) - 1;
    const bottom = Math.max(...corners.map(point => point.y)) + 1;
    if (right <= 0 || bottom <= 0 || left >= width || top >= height) return 0;
    const sum = (data, margin = 0) => {
      const firstColumn = Math.max(0, Math.floor((left - margin) / blockSize));
      const lastColumn = Math.min(mask.columns, Math.ceil((right + margin) / blockSize));
      const firstRow = Math.max(0, Math.floor((top - margin) / blockSize));
      const lastRow = Math.min(mask.rows, Math.ceil((bottom + margin) / blockSize));
      return data[lastRow * mask.stride + lastColumn] - data[firstRow * mask.stride + lastColumn] -
        data[lastRow * mask.stride + firstColumn] + data[firstRow * mask.stride + firstColumn];
    };
    if (!sum(mask.occupied)) return 0;
    const margin = 1;
    return left >= margin && top >= margin && right <= width - margin && bottom <= height - margin &&
      !sum(mask.incomplete, margin) ? 2 : 1;
  };
}

export function selectMergeFrames(geometries, maxFrames, pixelAllowed = null, edgeFeather = 0.1, bounds = null,
  coverage = createMergeCoverage(pixelAllowed)) {
  if (!geometries.length || !Number.isInteger(maxFrames) || maxFrames < 1) return [];
  const frameWidth = Math.max(1, geometries[0].width || 1);
  const cellSize = Math.max(64, Math.ceil(frameWidth * Math.max(0.025, edgeFeather)));
  bounds ??= mergeBounds(geometries);
  const columns = Math.ceil((bounds.maxX - bounds.minX) / cellSize), rows = Math.ceil((bounds.maxY - bounds.minY) / cellSize);
  if (columns <= 0 || rows <= 0) return [];
  const probeColumns = 8, probesPerCell = probeColumns * probeColumns, limit = maxFrames + 1;
  const counts = new Uint16Array(columns * rows * probesPerCell), selected = [];
  for (const geometry of sharpestFramesFirst(geometries)) {
    const extent = mergeBounds([geometry]);
    const firstColumn = Math.max(0, Math.floor((extent.minX - 1 - bounds.minX) / cellSize));
    const lastColumn = Math.min(columns - 1, Math.floor((extent.maxX + 1 - bounds.minX) / cellSize));
    const firstRow = Math.max(0, Math.floor((extent.minY - 1 - bounds.minY) / cellSize));
    const lastRow = Math.min(rows - 1, Math.floor((extent.maxY + 1 - bounds.minY) / cellSize));
    const covered = [];
    for (let row = firstRow; row <= lastRow; row++) for (let column = firstColumn; column <= lastColumn; column++) {
      const index = row * columns + column;
      const cell = { minX: bounds.minX + column * cellSize, minY: bounds.minY + row * cellSize,
        maxX: Math.min(bounds.maxX, bounds.minX + (column + 1) * cellSize),
        maxY: Math.min(bounds.maxY, bounds.minY + (row + 1) * cellSize) };
      const support = coverage(geometry, cell);
      if (!support) continue;
      const base = index * probesPerCell;
      for (let probeRow = 0; probeRow < probeColumns; probeRow++) for (let probeColumn = 0; probeColumn < probeColumns; probeColumn++) {
        const probe = base + probeRow * probeColumns + probeColumn;
        if (counts[probe] >= limit) continue;
        const point = { x: cell.minX + (probeColumn + 0.5) / probeColumns * (cell.maxX - cell.minX),
          y: cell.minY + (probeRow + 0.5) / probeColumns * (cell.maxY - cell.minY) };
        if (support === 2 || geometry.supports(point, pixelAllowed)) covered.push(probe);
      }
    }
    if (!covered.length) continue;
    selected.push(geometry);
    for (const probe of covered) counts[probe]++;
  }
  return selected;
}

export function mergeBounds(geometries) {
  if (!geometries.length) return null;
  const corners = geometries.flatMap(geometry => geometry.corners);
  const minX = Math.floor(Math.min(...corners.map(point => point.x)));
  const minY = Math.floor(Math.min(...corners.map(point => point.y)));
  const maxX = Math.ceil(Math.max(...corners.map(point => point.x)));
  const maxY = Math.ceil(Math.max(...corners.map(point => point.y)));
  return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY };
}

export function mergeEstimate(bounds, tileSize = 2048, bytesPerPixel = 4) {
  if (!bounds || !Number.isSafeInteger(tileSize) || tileSize < 1) return null;
  const columns = Math.ceil(bounds.width / tileSize);
  const rows = Math.ceil(bounds.height / tileSize);
  return { columns, rows, tiles: columns * rows, bytes: bounds.width * bounds.height * bytesPerPixel };
}