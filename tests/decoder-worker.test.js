import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { transform } from 'esbuild';
import * as decodeOrder from '../src/decode-order.js';

test('decoder supplies more input when codec output latency exceeds reorder lookahead', async () => {
  const source = await readFile(new URL('../src/decoder-worker.js', import.meta.url), 'utf8');
  const { code } = await transform(source, { format: 'cjs' });
  let submitted = 0;
  let staleClosed = false;
  class DelayedDecoder extends EventTarget {
    constructor(callbacks) { super(); this.callbacks = callbacks; this.state = 'configured'; this.decodeQueueSize = 0; }
    configure() {}
    close() { this.state = 'closed'; }
    decode(chunk) {
      submitted++;
      if (chunk.timestamp === 8) {
        assert.equal(staleClosed, true, 'obsolete buffered frames must release decoder surfaces before more input');
        this.callbacks.output({ timestamp: 8, displayWidth: 2, displayHeight: 2, close() {}, colorSpace: {} });
      }
      if (submitted === 8) this.callbacks.output({ timestamp: 0, displayWidth: 2, displayHeight: 2,
        close() {}, colorSpace: {} });
    }
    async flush() {}
  }
  const context = vm.createContext({
    require: name => name.endsWith('decode-order.js') ? decodeOrder : {},
    self: {}, VideoDecoder: DelayedDecoder, EncodedVideoChunk: class { constructor(value) { Object.assign(this, value); } },
    setTimeout, clearTimeout, performance,
    closeStale: () => { staleClosed = true; }
  });
  vm.runInContext(code, context);
  vm.runInContext(`
    file = { slice: () => ({ arrayBuffer: async () => new ArrayBuffer(1) }) };
    configuration = { codedWidth: 2, codedHeight: 2 };
    samples = Array.from({ length: 20 }, (_, index) => ({ index, timestamp: index, offset: index, size: 1, key: index === 0 }));
    presentation = samples;
    requiredDecodeIndices = samples.map(sample => sample.index);
  `, context);
  let timer;
  try {
    const result = await Promise.race([
      vm.runInContext('decodeFrame(0, true, false, false)', context),
      new Promise((resolve, reject) => { timer = setTimeout(() => reject(new Error(`Decoder stalled after ${submitted} samples`)), 1000); })
    ]);
    assert.equal(result.timestamp, 0);
    assert.equal(submitted, 8);
    result.frame.close();
    vm.runInContext('pendingFrames.set(2, { close: closeStale })', context);
    const next = await vm.runInContext('decodeFrame(8, true, false, false)', context);
    assert.equal(next.timestamp, 8);
    next.frame.close();
  } finally { clearTimeout(timer); }
});