import test from 'node:test';
import assert from 'node:assert/strict';
import { discoverWebGpuAdapters, getWebGpuSelection, onWebGpuSelectionChange,
  requestSelectedGpuAdapter, setWebGpuSelection } from '../src/webgpu-selection.js';

test('adapter list shows only usable distinct browser choices and Keine disables requests', async () => {
  const requests = [];
  const adapter = (vendor, architecture) => ({
    info: { vendor, architecture, device: '', description: '', isFallbackAdapter: false },
    limits: { maxTextureDimension2D: 8192, maxStorageBufferBindingSize: 128 * 1024 * 1024 },
    async requestDevice() { return { destroy() {} }; }
  });
  const gpu = { async requestAdapter(options) {
    requests.push(options);
    return options.forceFallbackAdapter ? null : options.powerPreference === 'low-power' ?
      adapter('intel', 'xe') : adapter('nvidia', 'lovelace');
  } };
  const choices = await discoverWebGpuAdapters(gpu);
  assert.deepEqual(choices.map(choice => choice.value), ['default', 'low-power']);
  assert.match(choices[0].label, /nvidia lovelace/i);
  assert.match(choices[1].label, /intel xe/i);
  assert.equal(requests.length, 4);
  const previousNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { gpu } });
  let changes = 0;
  const unsubscribe = onWebGpuSelectionChange(() => { changes++; });
  try {
    setWebGpuSelection('none');
    assert.equal(await requestSelectedGpuAdapter(), null);
    assert.equal(requests.length, 4);
    setWebGpuSelection('low-power');
    assert.equal((await requestSelectedGpuAdapter()).info.vendor, 'intel');
    assert.deepEqual(requests.at(-1), { powerPreference: 'low-power' });
    assert.equal(changes, 2);
  } finally {
    unsubscribe(); setWebGpuSelection('default');
    if (previousNavigator) Object.defineProperty(globalThis, 'navigator', previousNavigator);
    else delete globalThis.navigator;
  }
  assert.equal(getWebGpuSelection(), 'default');
});
