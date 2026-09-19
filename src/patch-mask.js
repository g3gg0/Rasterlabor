export const MASK_NEUTRAL = 0;
export const MASK_SEARCH = 1;
export const MASK_FORBIDDEN = 2;

export function createPatchMask(width, height, cellSize = Math.max(4, Math.ceil(Math.max(width, height) / 512))) {
  return { width: Math.ceil(width / cellSize), height: Math.ceil(height / cellSize), cellSize,
    sourceWidth: width, sourceHeight: height, data: new Uint8Array(Math.ceil(width / cellSize) * Math.ceil(height / cellSize)), revision: 0 };
}

export function maskIncludes(mask, x, y) {
  if (!mask?.data || x < 0 || y < 0 || x >= mask.sourceWidth || y >= mask.sourceHeight) return false;
  const col = Math.min(mask.width - 1, Math.floor(x / mask.cellSize));
  const row = Math.min(mask.height - 1, Math.floor(y / mask.cellSize));
  return mask.data[row * mask.width + col] === MASK_SEARCH;
}

export function remapInclusionMask(mask, maps) {
  if (!mask?.data || !maps || maps.inverseX?.length !== maps.outputWidth * maps.outputHeight ||
      maps.inverseY?.length !== maps.outputWidth * maps.outputHeight || maps.valid?.length !== maps.outputWidth * maps.outputHeight) return null;
  const defaultCellSize = Math.max(4, Math.ceil(Math.max(maps.outputWidth, maps.outputHeight) / 512));
  const result = createPatchMask(maps.outputWidth, maps.outputHeight, Math.min(mask.cellSize, defaultCellSize));
  for (let row = 0; row < result.height; row++) for (let col = 0; col < result.width; col++) {
    const x = Math.min(maps.outputWidth - 1, Math.floor((col + 0.5) * result.cellSize));
    const y = Math.min(maps.outputHeight - 1, Math.floor((row + 0.5) * result.cellSize));
    const index = y * maps.outputWidth + x;
    if (maps.valid[index] && maskIncludes(mask, maps.inverseX[index], maps.inverseY[index])) {
      result.data[row * result.width + col] = MASK_SEARCH;
    }
  }
  result.revision = mask.revision || 0;
  return result;
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