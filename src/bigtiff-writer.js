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
  entry(259, TYPE_SHORT, 1, layout.compression ?? 1);
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

// Independent zlib streams per tile keep compression and memory bounded.
export async function deflateTiffTile(pixels) {
  const compressed = new Blob([pixels]).stream().pipeThrough(new CompressionStream('deflate'));
  return new Uint8Array(await new Response(compressed).arrayBuffer());
}

export async function beginBigTiff(writable, mosaic, progress = () => {}) {
  const layout = createBigTiffLayout(mosaic.width, mosaic.height, mosaic.tileSize, mosaic.tiles);
  layout.compression = mosaic.compression === false ? 1 : 8;
  if (layout.compression === 8) { layout.offsets.fill(0n); layout.byteCounts.fill(0n); layout.fileBytes = layout.pixelOffset; }
  await writable.write({ type: 'write', position: 0, data: encodeBigTiffHeader(layout) });
  let canvas, context;
  const completed = new Set();
  let closed = false;
  const writePixels = async (tile, pixels) => {
    if (closed) throw new Error('BigTIFF ist bereits abgeschlossen.');
    const column = tile.x / layout.tileSize, row = tile.y / layout.tileSize;
    const index = row * layout.columns + column;
    if (!Number.isInteger(column) || !Number.isInteger(row) || layout.indexedTiles.get(index)?.x !== tile.x ||
        layout.indexedTiles.get(index)?.y !== tile.y) throw new Error('Unerwartete BigTIFF-Kachel.');
    if (completed.has(index)) throw new Error('BigTIFF-Kachel wurde bereits geschrieben.');
    if (pixels.byteLength !== layout.tileBytes) throw new Error('Unvollstaendige BigTIFF-Pixeldaten.');
    const bytes = layout.compression === 8 ? await deflateTiffTile(pixels) : pixels;
    if (layout.compression === 8) {
      layout.offsets[index] = BigInt(align8(layout.fileBytes)); layout.byteCounts[index] = BigInt(bytes.byteLength);
    }
    const position = Number(layout.offsets[index]);
    await writable.write({ type: 'write', position, data: bytes });
    if (layout.compression === 8) layout.fileBytes = position + bytes.byteLength;
    completed.add(index);
    progress({ completed: completed.size, total: mosaic.tiles.length, bytes: layout.fileBytes, fileBytes: layout.fileBytes });
  };
  return { layout, writePixels, async writeTile(tile) {
    canvas ??= new OffscreenCanvas(layout.tileSize, layout.tileSize);
    context ??= canvas.getContext('2d', { willReadFrequently: true });
    context.clearRect(0, 0, layout.tileSize, layout.tileSize);
    // Edge tiles include accumulator padding; never write pixels outside the mosaic.
    context.drawImage(tile.bitmap, 0, 0, tile.width, tile.height, 0, 0, tile.width, tile.height);
    await writePixels(tile, context.getImageData(0, 0, layout.tileSize, layout.tileSize).data);
  }, async finish() {
    if (completed.size !== layout.indexedTiles.size) throw new Error('BigTIFF ist unvollstaendig: Es fehlen Kacheln.');
    // A zero offset/count is a GDAL sparse-TIFF extension. GIMP/libtiff can
    // stop at the first such tile, so encode transparent pixels explicitly.
    let empty;
    for (let index = 0; index < layout.tileCount; index++) {
      if (layout.byteCounts[index] !== 0n) continue;
      empty ??= layout.compression === 8 ? await deflateTiffTile(new Uint8Array(layout.tileBytes)) : new Uint8Array(layout.tileBytes);
      const position = align8(layout.fileBytes);
      await writable.write({ type: 'write', position, data: empty });
      layout.offsets[index] = BigInt(position); layout.byteCounts[index] = BigInt(empty.byteLength);
      layout.fileBytes = position + empty.byteLength;
    }
    await writable.write({ type: 'write', position: 0, data: encodeBigTiffHeader(layout) });
    await writable.truncate(layout.fileBytes); closed = true;
    if (canvas) canvas.width = canvas.height = 1;
    return layout;
  } };
}

export async function writeBigTiff(writable, mosaic, progress = () => {}) {
  const writer = await beginBigTiff(writable, mosaic, progress);
  for (const tile of mosaic.tiles) await writer.writeTile(tile);
  return writer.finish();
}
