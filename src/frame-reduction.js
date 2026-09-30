import { sharpestFramesFirst } from './path-support.js';

const validityTiles = new WeakMap();
function invalidTileCount(valid, width, height, minX, minY, maxX, maxY) {
  let tiles = validityTiles.get(valid);
  if (!tiles || tiles.width !== width || tiles.height !== height) {
    const size = 32, columns = Math.ceil(width / size), rows = Math.ceil(height / size), stride = columns + 1;
    const flags = new Uint8Array(columns * rows);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++)
      if (!valid[y * width + x]) flags[Math.floor(y / size) * columns + Math.floor(x / size)] = 1;
    const prefix = new Uint32Array(stride * (rows + 1));
    for (let y = 0; y < rows; y++) for (let x = 0; x < columns; x++) {
      const index = (y + 1) * stride + x + 1;
      prefix[index] = flags[y * columns + x] + prefix[index - 1] + prefix[index - stride] - prefix[index - stride - 1];
    }
    tiles = { width, height, size, stride, prefix }; validityTiles.set(valid, tiles);
  }
  const x0 = Math.max(0, Math.floor(minX / tiles.size));
  const y0 = Math.max(0, Math.floor(minY / tiles.size));
  const x1 = Math.min(Math.ceil(width / tiles.size), Math.ceil(maxX / tiles.size));
  const y1 = Math.min(Math.ceil(height / tiles.size), Math.ceil(maxY / tiles.size));
  const { prefix, stride } = tiles;
  return prefix[y1 * stride + x1] - prefix[y0 * stride + x1] - prefix[y1 * stride + x0] + prefix[y0 * stride + x0];
}

export function usableImageBounds(valid, width, height) {
  if (!valid || valid.length !== width * height) throw new Error('Gueltige Bildmaske fehlt.');
  let minX = width, minY = height, maxX = -1, maxY = -1;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) if (valid[y * width + x]) {
    minX = Math.min(minX, x); minY = Math.min(minY, y);
    maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
  }
  if (maxX < minX) throw new Error('Die Bildmaske enthaelt keine gueltigen Pixel.');
  return { minX, minY, maxX: maxX + 1, maxY: maxY + 1 };
}

function transformedBounds(geometry, usable) {
  const corners = [[usable.minX, usable.minY], [usable.maxX, usable.minY],
    [usable.maxX, usable.maxY], [usable.minX, usable.maxY]].map(([x, y]) => geometry.world(x, y));
  return { minX: Math.min(...corners.map(point => point.x)), minY: Math.min(...corners.map(point => point.y)),
    maxX: Math.max(...corners.map(point => point.x)), maxY: Math.max(...corners.map(point => point.y)) };
}

function overlaps(a, b) {
  return a.minX < b.maxX && a.maxX > b.minX && a.minY < b.maxY && a.maxY > b.minY;
}

// A world-aligned square is mapped back into the rectified pixel grid. Every
// pixel centre inside that polygon must be valid.
export function fullyCoversBlock(geometry, block) {
  const corners = [[block.minX, block.minY], [block.maxX, block.minY],
    [block.maxX, block.maxY], [block.minX, block.maxY]].map(([x, y]) => geometry.local({ x, y }));
  const minX = Math.min(...corners.map(point => point.x));
  const maxX = Math.max(...corners.map(point => point.x));
  const minY = Math.min(...corners.map(point => point.y));
  const maxY = Math.max(...corners.map(point => point.y));
  if (minX < -1e-7 || minY < -1e-7 || maxX > geometry.width + 1e-7 || maxY > geometry.height + 1e-7) return false;
  if (!geometry.valid) return true;
  const { valid, width } = geometry;
  if (!invalidTileCount(valid, width, geometry.height, minX, minY, maxX, maxY)) return true;
  for (let y = Math.max(0, Math.floor(minY)); y < Math.min(geometry.height, Math.ceil(maxY)); y++) {
    const intersections = [];
    for (let edge = 0; edge < 4; edge++) {
      const a = corners[edge], b = corners[(edge + 1) % 4];
      if ((a.y <= y + 0.5 && b.y > y + 0.5) || (b.y <= y + 0.5 && a.y > y + 0.5))
        intersections.push(a.x + (y + 0.5 - a.y) * (b.x - a.x) / (b.y - a.y));
    }
    if (!intersections.length) continue;
    const left = Math.max(0, Math.floor(Math.min(...intersections)));
    const right = Math.min(width - 1, Math.ceil(Math.max(...intersections)) - 1);
    for (let x = left; x <= right; x++) if (!valid[y * width + x]) return false;
  }
  return true;
}

export function reduceFramesByBlocks(geometries, { divisions = 5, minimum = 3 } = {}) {
  if (!Number.isInteger(divisions) || divisions < 1 || divisions > 50) throw new Error('N muss zwischen 1 und 50 liegen.');
  if (!Number.isInteger(minimum) || minimum < 1 || minimum > 100) throw new Error('M muss zwischen 1 und 100 liegen.');
  if (!geometries.length) throw new Error('Keine Tracking-Posen vorhanden.');
  const { width, height, valid } = geometries[0];
  const usable = usableImageBounds(valid, width, height);
  const blockSize = Math.max(1, Math.floor(Math.min(usable.maxX - usable.minX, usable.maxY - usable.minY) / divisions));
  const ranked = sharpestFramesFirst(geometries);
  const items = ranked.map(geometry => ({ geometry, bounds: transformedBounds(geometry, usable) }));
  const extent = { minX: Math.floor(Math.min(...items.map(item => item.bounds.minX))),
    minY: Math.floor(Math.min(...items.map(item => item.bounds.minY))),
    maxX: Math.ceil(Math.max(...items.map(item => item.bounds.maxX))),
    maxY: Math.ceil(Math.max(...items.map(item => item.bounds.maxY))) };
  const columns = Math.ceil((extent.maxX - extent.minX) / blockSize);
  const rows = Math.ceil((extent.maxY - extent.minY) / blockSize);
  if (columns * rows > 200000) throw new Error('Zu viele Bloecke. N kleiner waehlen.');
  const selected = new Set();
  let blocks = 0, shortBlocks = 0, uncoveredBlocks = 0;
  for (let row = 0; row < rows; row++) for (let column = 0; column < columns; column++) {
    const block = { minX: extent.minX + column * blockSize, minY: extent.minY + row * blockSize,
      maxX: Math.min(extent.maxX, extent.minX + (column + 1) * blockSize),
      maxY: Math.min(extent.maxY, extent.minY + (row + 1) * blockSize) };
    const candidates = items.filter(item => overlaps(item.bounds, block));
    if (!candidates.length) continue;
    blocks++;
    let count = 0;
    for (const { geometry } of candidates) {
      if (!fullyCoversBlock(geometry, block)) continue;
      selected.add(geometry.entry.frame);
      if (++count === minimum) break;
    }
    if (count < minimum) shortBlocks++;
    if (!count) uncoveredBlocks++;
  }
  return { frames: selected, blocks, shortBlocks, uncoveredBlocks, blockSize,
    totalFrames: geometries.length, selectedFrames: selected.size };
}
