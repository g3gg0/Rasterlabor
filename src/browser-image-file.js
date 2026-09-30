// Random-access output without assembling the whole image in a Uint8Array.
// OPFS keeps large downloads on disk; Blob segments cover embedded browsers
// which expose neither a file picker nor origin-private storage.
export async function createBrowserImageFile(name, type = 'image/tiff', { storage = globalThis.navigator?.storage } = {}) {
  let directory, fileHandle, stream, storageName;
  try {
    directory = await (await storage.getDirectory()).getDirectoryHandle('rasterlabor-exports', { create: true });
    storageName = `${Date.now()}-${crypto.randomUUID()}`;
    fileHandle = await directory.getFileHandle(storageName, { create: true });
    stream = await fileHandle.createWritable();
  } catch {
    if (directory && storageName) await directory.removeEntry(storageName).catch(() => {});
    directory = null;
  }
  let size = 0, segments = [], state = 'open', result = null;
  const assertOpen = () => { if (state !== 'open') throw new Error('Die Bilddatei ist bereits geschlossen.'); };
  const replace = (position, data) => {
    const end = position + data.size, next = [];
    for (const segment of segments) {
      if (segment.end <= position || segment.start >= end) { next.push(segment); continue; }
      if (segment.start < position) next.push({ start: segment.start, end: position, blob: segment.blob.slice(0, position - segment.start) });
      if (segment.end > end) next.push({ start: end, end: segment.end, blob: segment.blob.slice(end - segment.start) });
    }
    next.push({ start: position, end, blob: data });
    segments = next.sort((a, b) => a.start - b.start);
  };
  const cleanup = async () => { if (directory) await directory.removeEntry(storageName).catch(() => {}); segments = []; };
  const writable = {
    async write({ position, data }) {
      assertOpen();
      const bytes = data.byteLength ?? data.size;
      if (!Number.isSafeInteger(position) || position < 0 || !Number.isSafeInteger(bytes)) throw new Error('Ungueltiger Dateibereich.');
      if (stream) await stream.write({ type: 'write', position, data });
      else replace(position, new Blob([data]));
      size = Math.max(size, position + bytes);
    },
    async truncate(length) {
      assertOpen();
      if (!Number.isSafeInteger(length) || length < 0) throw new Error('Ungueltige Dateilaenge.');
      if (stream) await stream.truncate(length);
      else segments = segments.filter(s => s.start < length).map(s => s.end <= length ? s :
        { start: s.start, end: length, blob: s.blob.slice(0, length - s.start) });
      size = length;
    },
    async close() {
      assertOpen();
      if (stream) { await stream.close(); result = await fileHandle.getFile(); }
      else {
        const parts = []; let position = 0;
        for (const segment of segments) {
          if (segment.start > position) parts.push(new Uint8Array(segment.start - position));
          parts.push(segment.blob); position = segment.end;
        }
        if (size > position) parts.push(new Uint8Array(size - position));
        result = new Blob(parts, { type }); segments = [];
      }
      if (result.size !== size) throw new Error('Die gespeicherte Bilddatei ist unvollstaendig.');
      state = 'closed';
    },
    async abort() { if (state === 'open') { await stream?.abort().catch(() => {}); state = 'aborted'; } await cleanup(); }
  };
  return { name, mode: directory ? 'Browser-Dateispeicher' : 'Browser-Download', createWritable: async () => writable,
    getFile() { if (!result || state !== 'closed') throw new Error('Bilddatei noch nicht fertig.'); return result; },
    async dispose() { result = null; if (state === 'open') await writable.abort(); else await cleanup(); } };
}
