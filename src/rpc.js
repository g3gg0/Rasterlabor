import { getWebGpuSelection } from './webgpu-selection.js';

export class WorkerClient {
  constructor(url, onProgress = () => {}) {
    this.worker = new Worker(url, { type: 'module' });
    this.pending = new Map();
    this.sequence = 0;
    this.worker.onmessage = ({ data }) => {
      if (data.progress) { onProgress(data.progress); return; }
      const entry = this.pending.get(data.id);
      if (!entry) return;
      this.pending.delete(data.id);
      if (data.error) entry.reject(new Error(data.error));
      else entry.resolve(data.result);
    };
    this.worker.onerror = event => {
      for (const entry of this.pending.values()) entry.reject(new Error(event.message || 'Worker konnte nicht geladen werden.'));
      this.pending.clear();
    };
  }
  call(type, payload = {}, transfer = []) {
    return new Promise((resolve, reject) => {
      const id = ++this.sequence;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, type, ...payload, gpuSelection: getWebGpuSelection() }, transfer);
    });
  }
  terminate() {
    this.worker.terminate();
    for (const entry of this.pending.values()) entry.reject(new Error('Worker beendet.'));
    this.pending.clear();
  }
}
