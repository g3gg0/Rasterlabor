import { WebGpuOverlay } from './webgpu-overlay.js';
import { sharpestFramesFirst } from './path-support.js';
import { blurriestFramesFirst, selectMergeFrames } from './merge-plan.js';

export function overlayTileSize(maps, maximumTextureSize, budget = 1536 * 1024 * 1024) {
  const remaining = budget - maps.outputWidth * maps.outputHeight * 17;
  if (remaining <= 0) throw new Error('Entzerrungs-Maps ueberschreiten das GPU-Bildbudget.');
  const maximumEdge = Math.min(4096, maximumTextureSize, Math.floor(Math.sqrt(remaining / 24)));
  if (maximumEdge < 512) throw new Error('Zu wenig GPU-Bildbudget fuer eine Ueberlagerungskachel.');
  return 2 ** Math.floor(Math.log2(maximumEdge));
}

export function overlayTiles(width, height, minX, minY, geometries, tileSize = 2048, selection = null) {
  if (![width, height, tileSize].every(x => Number.isSafeInteger(x) && x > 0)) throw new Error('Ungueltige Kachelgroesse.');
  const bounds = geometries.map(geometry => ({ geometry,
    left: Math.min(...geometry.corners.map(p => p.x)) - 1,
    right: Math.max(...geometry.corners.map(p => p.x)) + 1,
    top: Math.min(...geometry.corners.map(p => p.y)) - 1,
    bottom: Math.max(...geometry.corners.map(p => p.y)) + 1 }));
  const tiles = [];
  for (let y = 0; y < height; y += tileSize) for (let x = 0; x < width; x += tileSize) {
    const w = Math.min(tileSize, width - x), h = Math.min(tileSize, height - y);
    const margin = selection ? Math.max(2, Math.ceil((geometries[0]?.width || 1) * selection.edgeFeather)) : 0;
    const selectionBounds = { minX: Math.max(minX, minX + x - margin), minY: Math.max(minY, minY + y - margin),
      maxX: Math.min(minX + width, minX + x + w + margin), maxY: Math.min(minY + height, minY + y + h + margin) };
    let contributing = bounds.filter(b => b.left < selectionBounds.maxX && b.right > selectionBounds.minX &&
      b.top < selectionBounds.maxY && b.bottom > selectionBounds.minY).map(b => b.geometry);
    if (selection && contributing.length && contributing.every(geometry => typeof geometry.supports === 'function')) {
      contributing = selectMergeFrames(contributing, selection.maxFrames,
        selection.pixelAllowed, selection.edgeFeather, selectionBounds);
    }
    if (contributing.length) tiles.push({ x, y, width: w, height: h, geometries: contributing });
  }
  return tiles;
}

export function closeOverlayTiles(overlay) {
  for (const tile of overlay?.tiles ?? []) tile.bitmap.close();
}

export async function renderTiledOverlay({ maps, width, height, minX, minY, geometries, pixelAllowed,
  decode, cancelled = () => false, progress = () => {}, tileSize = null, maxFrames = 0, edgeFeather = 0.1,
  brightness = null, frameOrder = 'existing', blend = 'average', tileReady = null, retainTiles = true }) {
  if (tileSize === null) {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('Keine WebGPU-GPU verfuegbar.');
    tileSize = overlayTileSize(maps, adapter.limits.maxTextureDimension2D);
  }
  const mergeSelection = frameOrder === 'blurriest';
  const ordered = mergeSelection ? sharpestFramesFirst(geometries) :
    frameOrder === 'sharpest' || (frameOrder === 'existing' && maxFrames) ? sharpestFramesFirst(geometries) : geometries;
  const plan = overlayTiles(width, height, minX, minY, ordered, tileSize,
    mergeSelection ? { maxFrames, pixelAllowed, edgeFeather } : null);
  const plannedFramePasses = plan.reduce((sum, tile) => sum + tile.geometries.length, 0);
  const output = { width, height, tileSize, tiles: [], plannedFramePasses, framePasses: 0, frameMs: 0, setupMs: 0, decodeMs: 0, gpuMs: 0, finishMs: 0 };
  let renderer;
  let completed = false;
  try {
    if (!plan.length || cancelled()) return null;
    const setup = performance.now();
    // One full-resolution remap and one tile accumulator, reused for the entire mosaic.
    renderer = await WebGpuOverlay.create(maps, Math.min(tileSize, width), Math.min(tileSize, height), minX, minY,
      pixelAllowed, maxFrames, edgeFeather, brightness);
    output.setupMs = performance.now() - setup;
    for (const [tileIndex, tile] of plan.entries()) {
      if (cancelled()) return null;
      await renderer.clear(minX + tile.x, minY + tile.y);
      for (const [frameIndex, geometry] of tile.geometries.entries()) {
        if (cancelled()) return null;
        progress({ tileIndex, tileCount: plan.length, frameIndex, frameCount: tile.geometries.length,
          framePass: output.framePasses + 1, plannedFramePasses, frame: geometry.entry.frame, tileSize });
        const start = performance.now();
        const decoded = await decode(geometry.entry.frame);
        output.decodeMs += performance.now() - start;
        try {
          if (cancelled()) return null;
          const gpuStarted = performance.now();
          await renderer.addFrame(decoded.frame, decoded.orientation, geometry,
            mergeSelection ? blend === 'sharp-over' ? 4 : 5 : 0);
          output.gpuMs += performance.now() - gpuStarted;
        } finally { decoded.frame.close(); }
        output.frameMs += performance.now() - start; output.framePasses++;
      }
      if (cancelled()) return null;
      const finishStarted = performance.now();
      const bitmap = await renderer.finish();
      output.finishMs += performance.now() - finishStarted;
      const completedTile = { x: tile.x, y: tile.y, width: tile.width, height: tile.height, bitmap };
      try {
        if (tileReady) await tileReady(completedTile);
        if (retainTiles) output.tiles.push(completedTile);
      } finally { if (!retainTiles) bitmap.close(); }
    }
    if (cancelled()) return null;
    completed = true;
    return output;
  } finally {
    renderer?.destroy();
    if (!completed) closeOverlayTiles(output);
  }
}
