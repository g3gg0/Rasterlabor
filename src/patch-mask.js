export const MASK_NEUTRAL = 0;
export const MASK_SEARCH = 1;
export const MASK_FORBIDDEN = 2;

export function createPatchMask(width, height, cellSize = Math.max(4, Math.ceil(Math.max(width, height) / 512))) {
  return { width: Math.ceil(width / cellSize), height: Math.ceil(height / cellSize), cellSize,
    sourceWidth: width, sourceHeight: height, data: new Uint8Array(Math.ceil(width / cellSize) * Math.ceil(height / cellSize)), revision: 0 };
}

export function patchTouches(mask, x, y, size, value) {
  if (!mask?.data) return false;
  const radius = size / 2;
  const minCol = Math.max(0, Math.floor((x - radius) / mask.cellSize));
  const maxCol = Math.min(mask.width - 1, Math.floor((x + radius) / mask.cellSize));
  const minRow = Math.max(0, Math.floor((y - radius) / mask.cellSize));
  const maxRow = Math.min(mask.height - 1, Math.floor((y + radius) / mask.cellSize));
  for (let row = minRow; row <= maxRow; row++) {
    for (let col = minCol; col <= maxCol; col++) if (mask.data[row * mask.width + col] === value) return true;
  }
  return false;
}

export function patchAllowed(mask, x, y, size, requireSearch = false) {
  if (!mask?.data) return true;
  if (patchTouches(mask, x, y, size, MASK_FORBIDDEN)) return false;
  if (!requireSearch) return true;
  const col = Math.max(0, Math.min(mask.width - 1, Math.floor(x / mask.cellSize)));
  const row = Math.max(0, Math.min(mask.height - 1, Math.floor(y / mask.cellSize)));
  return mask.data[row * mask.width + col] === MASK_SEARCH;
}

export function paintPatchMask(mask, x, y, radius, value) {
  const minCol = Math.max(0, Math.floor((x - radius) / mask.cellSize));
  const maxCol = Math.min(mask.width - 1, Math.floor((x + radius) / mask.cellSize));
  const minRow = Math.max(0, Math.floor((y - radius) / mask.cellSize));
  const maxRow = Math.min(mask.height - 1, Math.floor((y + radius) / mask.cellSize));
  const radiusSquared = radius ** 2;
  for (let row = minRow; row <= maxRow; row++) {
    for (let col = minCol; col <= maxCol; col++) {
      const nearestX = Math.max(col * mask.cellSize, Math.min(x, (col + 1) * mask.cellSize));
      const nearestY = Math.max(row * mask.cellSize, Math.min(y, (row + 1) * mask.cellSize));
      if ((nearestX - x) ** 2 + (nearestY - y) ** 2 <= radiusSquared) mask.data[row * mask.width + col] = value;
    }
  }
  mask.revision = (mask.revision || 0) + 1;
}

export function validatePointsAgainstMask(points, mask, patchSize) {
  let invalid = 0;
  for (const point of points) {
    point.maskValid = !patchTouches(mask, point.x, point.y, patchSize, MASK_FORBIDDEN);
    if (!point.maskValid) invalid++;
  }
  return invalid;
}