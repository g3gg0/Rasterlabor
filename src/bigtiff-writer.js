const TYPE_SHORT = 3;
const TYPE_LONG = 4;
const TYPE_LONG8 = 16;

function align8(value) { return Math.ceil(value / 8) * 8; }

export function createBigTiffLayout(width, height, tileSize, tiles) {
  if (![width, height, tileSize].every(value => Number.isSafeInteger(value) && value > 0)) throw new Error('Ungueltige BigTIFF-Abmessungen.');
  const columns = Math.ceil(width / tileSize), rows = Math.ceil(height / tileSize);
  const tileCount = columns * rows;
  if (!Number.isSafeInteger(tileCount)) throw new Error('Zu viele BigTIFF-Kacheln.');
  const entries = 12;
  const ifdBytes = 8 + entries * 20 + 8;
  const offsetsOffset = align8(16 + ifdBytes);
  const byteCountsOffset = offsetsOffset + tileCount * 8;
  const pixelOffset = align8(byteCountsOffset + tileCount * 8);
  const tileBytes = tileSize * tileSize * 4;
  if (!Number.isSafeInteger(tileBytes)) throw new Error('BigTIFF-Kachel ist zu gross.');
  const offsets = new BigUint64Array(tileCount);
  const byteCounts = new BigUint64Array(tileCount);
  let nextOffset = pixelOffset;
  const indexedTiles = new Map();
  for (const tile of tiles) {
    const column = tile.x / tileSize, row = tile.y / tileSize;
    if (!Number.isInteger(column) || !Number.isInteger(row) || column < 0 || column >= columns || row < 0 || row >= rows)
      throw new Error('BigTIFF-Kachel liegt nicht auf dem Ausgaberaster.');
    const index = row * columns + column;
    if (indexedTiles.has(index)) throw new Error('Doppelte BigTIFF-Kachel.');
    indexedTiles.set(index, tile);
    offsets[index] = BigInt(nextOffset); byteCounts[index] = BigInt(tileBytes);
    nextOffset += tileBytes;
    if (!Number.isSafeInteger(nextOffset)) throw new Error('BigTIFF-Datei ueberschreitet den sicheren Browser-Adressraum.');
  }
  return { width, height, tileSize, columns, rows, tileCount, tileBytes, offsets, byteCounts,
    indexedTiles, offsetsOffset, byteCountsOffset, pixelOffset, fileBytes: nextOffset, entries };
}

export function encodeBigTiffHeader(layout) {
  const header = new ArrayBuffer(layout.pixelOffset);
  const view = new DataView(header);
  view.setUint16(0, 0x4949, true); view.setUint16(2, 43, true);
  view.setUint16(4, 8, true); view.setUint16(6, 0, true); view.setBigUint64(8, 16n, true);
  view.setBigUint64(16, BigInt(layout.entries), true);
  let position = 24;
  const entry = (tag, type, count, value) => {
    view.setUint16(position, tag, true); view.setUint16(position + 2, type, true);
    view.setBigUint64(position + 4, BigInt(count), true);
    if (Array.isArray(value)) value.forEach((item, index) => view.setUint16(position + 12 + index * 2, item, true));
    else if (type === TYPE_SHORT && count === 1) view.setUint16(position + 12, value, true);
    else if (type === TYPE_LONG && count === 1) view.setUint32(position + 12, value, true);
    else view.setBigUint64(position + 12, BigInt(value), true);
    position += 20;
  };
  entry(256, TYPE_LONG8, 1, layout.width);
  entry(257, TYPE_LONG8, 1, layout.height);
  entry(258, TYPE_SHORT, 4, [8, 8, 8, 8]);
  entry(259, TYPE_SHORT, 1, 1);
  entry(262, TYPE_SHORT, 1, 2);
  entry(277, TYPE_SHORT, 1, 4);
  entry(284, TYPE_SHORT, 1, 1);
  entry(322, TYPE_LONG, 1, layout.tileSize);
  entry(323, TYPE_LONG, 1, layout.tileSize);
  entry(324, TYPE_LONG8, layout.tileCount, layout.tileCount === 1 ? layout.offsets[0] : layout.offsetsOffset);
  entry(325, TYPE_LONG8, layout.tileCount, layout.tileCount === 1 ? layout.byteCounts[0] : layout.byteCountsOffset);
  entry(338, TYPE_SHORT, 1, 2);
  view.setBigUint64(position, 0n, true);
  if (layout.tileCount > 1) {
    for (let index = 0; index < layout.tileCount; index++) {
      view.setBigUint64(layout.offsetsOffset + index * 8, layout.offsets[index], true);
      view.setBigUint64(layout.byteCountsOffset + index * 8, layout.byteCounts[index], true);
    }
  }
  return new Uint8Array(header);
}

export async function beginBigTiff(writable, mosaic, progress = () => {}) {
  const layout = createBigTiffLayout(mosaic.width, mosaic.height, mosaic.tileSize, mosaic.tiles);
  await writable.write({ type: 'write', position: 0, data: encodeBigTiffHeader(layout) });
  const canvas = new OffscreenCanvas(layout.tileSize, layout.tileSize);
  const context = canvas.getContext('2d', { willReadFrequently: true });
  let completed = 0;
  return { layout, async writeTile(tile) {
    const column = tile.x / layout.tileSize, row = tile.y / layout.tileSize;
    const index = row * layout.columns + column;
    if (!Number.isInteger(index) || layout.indexedTiles.get(index)?.x !== tile.x) throw new Error('Unerwartete BigTIFF-Kachel.');
    context.clearRect(0, 0, layout.tileSize, layout.tileSize);
    context.drawImage(tile.bitmap, 0, 0);
    const pixels = context.getImageData(0, 0, layout.tileSize, layout.tileSize).data;
    await writable.write({ type: 'write', position: Number(layout.offsets[index]), data: pixels });
    progress({ completed: ++completed, total: mosaic.tiles.length, bytes: Number(layout.offsets[index]) + pixels.byteLength,
      fileBytes: layout.fileBytes });
  }, async finish() { await writable.truncate(layout.fileBytes); return layout; } };
}

export async function writeBigTiff(writable, mosaic, progress = () => {}) {
  const writer = await beginBigTiff(writable, mosaic, progress);
  for (const tile of mosaic.tiles) await writer.writeTile(tile);
  return writer.finish();
}