import { beginBigTiff } from './bigtiff-writer.js';

export function planTiffParts({ width, height, tileSize, tiles }, name, split = '0') {
  const rows = Math.ceil(height / tileSize), columns = Math.ceil(width / tileSize);
  let rowsPerPart = rows;
  if (split === 'two') rowsPerPart = Math.ceil(rows / 2);
  else if (Number(split) > 0) {
    // Includes TIFF metadata and worst-case deflate overhead. Compression can
    // only make the chosen parts smaller; the estimate never needs image pixels.
    rowsPerPart = Math.floor((Number(split) * 1024 ** 2 - 1024 ** 2) / (columns * tileSize ** 2 * 4 * 1.001));
    if (rowsPerPart < 1) throw new Error('Eine Kachelzeile passt nicht in die gewaehlte TIFF-Dateigroesse. Groesseres Limit waehlen.');
  }
  const parts = [];
  for (let y = 0; y < height; y += rowsPerPart * tileSize) {
    const partHeight = Math.min(rowsPerPart * tileSize, height - y);
    parts.push({ width, height: partHeight, y, tileSize,
      tiles: tiles.filter(tile => tile.y >= y && tile.y < y + partHeight).map(tile => ({ ...tile, y: tile.y - y })) });
  }
  return parts.map((part, i) => ({ ...part, name: parts.length === 1 ? name :
    name.replace(/\.tiff?$/i, '') + `-teil-${String(i + 1).padStart(3, '0')}.tif` }));
}

export async function beginTiffParts(parts, createHandle, onPart = () => {}, progress = () => {}) {
  let index = -1, handle, writable, writer;
  const completed = [];
  const finishPart = async () => {
    if (!writer) return;
    const layout = await writer.finish(); await writable.close();
    const saved = await handle.getFile();
    if (saved.size !== layout.fileBytes) throw new Error('Gespeicherte TIFF-Datei ist unvollstaendig.');
    const part = { ...parts[index], handle, saved, layout };
    completed.push(part); await onPart(part, index, parts.length);
    writer = null; writable = null; handle = null;
  };
  const next = async () => {
    await finishPart(); index++;
    handle = await createHandle(parts[index], index, parts.length);
    writable = await handle.createWritable();
    writer = await beginBigTiff(writable, parts[index], state => progress({ ...state, part: index + 1, parts: parts.length }));
  };
  return {
    async writeTile(tile) {
      while (index < 0 || tile.y >= parts[index].y + parts[index].height) await next();
      if (tile.y < parts[index].y) throw new Error('TIFF-Kacheln muessen zeilenweise geschrieben werden.');
      await writer.writeTile({ ...tile, y: tile.y - parts[index].y });
    },
    async finish() {
      while (index < parts.length - 1) await next();
      await finishPart(); return completed;
    },
    async abort() { try { await writable?.abort(); } finally { await handle?.dispose?.(); } }
  };
}

export function tiffJoinCommand(parts, width, height) {
  if (parts.length < 2) return '';
  const quote = value => "'" + value.replaceAll("'", "'\\''") + "'";
  // Individual join calls preserve unequal heights without arrayjoin padding.
  const lines = ['# Debian / Ubuntu: sudo apt install libvips-tools'];
  let previous = parts[0].name;
  for (let i = 1; i < parts.length; i++) {
    const output = i === parts.length - 1 ? 'merge-gesamt.tif[tile,compression=deflate,bigtiff]' : `merge-zwischen-${i}.tif[tile,compression=deflate,bigtiff]`;
    lines.push(`vips join ${quote(previous)} ${quote(parts[i].name)} ${quote(output)} vertical`);
    previous = output.split('[')[0];
  }
  lines.push(`# Ergebnis: ${width} x ${height} Pixel, ohne Skalierung oder Ueberlappung.`);
  return lines.join('\n');
}
