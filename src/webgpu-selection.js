const options = {
  default: {},
  'high-performance': { powerPreference: 'high-performance' },
  'low-power': { powerPreference: 'low-power' },
  fallback: { forceFallbackAdapter: true }
};

let selection = 'default';
const resetHandlers = new Set();

export function getWebGpuSelection() { return selection; }
export function setWebGpuSelection(value) {
  if (value !== 'none' && !Object.hasOwn(options, value)) throw new Error('Unbekannte WebGPU-Auswahl.');
  if (selection === value) return false;
  selection = value;
  for (const reset of resetHandlers) reset();
  return true;
}
export function onWebGpuSelectionChange(reset) {
  resetHandlers.add(reset);
  return () => resetHandlers.delete(reset);
}
export async function requestSelectedGpuAdapter() {
  if (selection === 'none') return null;
  return globalThis.navigator?.gpu?.requestAdapter(options[selection]) ?? null;
}

function adapterName(info) {
  const description = info?.description?.trim();
  if (description) return description;
  const vendor = info?.vendor?.trim();
  const brand = { nvidia: 'NVIDIA', amd: 'AMD', intel: 'Intel', apple: 'Apple' }[vendor?.toLowerCase()] ?? vendor;
  const architecture = info?.architecture?.trim();
  const model = architecture ? `${architecture[0].toUpperCase()}${architecture.slice(1)}` : info?.device?.trim();
  const name = [brand, model].filter(Boolean).join(' ');
  return name || 'WebGPU-Adapter (Name vom Browser verborgen)';
}

export async function discoverWebGpuAdapters(gpu = globalThis.navigator?.gpu) {
  if (!gpu) return [];
  const found = [], seen = new Set();
  for (const [value, request] of Object.entries(options)) {
    try {
      const adapter = await gpu.requestAdapter(request);
      if (!adapter) continue;
      const device = await adapter.requestDevice();
      device.destroy();
      const info = adapter.info ?? {};
      const name = adapterName(info);
      const signature = [info.vendor, info.architecture, info.device, info.description,
        info.isFallbackAdapter, adapter.limits.maxTextureDimension2D,
        adapter.limits.maxStorageBufferBindingSize].join('|');
      // If the browser hides all identifiers, keep the preferences distinct.
      const identifiable = Boolean(info.vendor || info.architecture || info.device || info.description);
      if (identifiable && seen.has(signature)) continue;
      seen.add(signature);
      const prefix = value === 'default' ? 'Automatisch' : value === 'high-performance' ?
        'Leistung bevorzugt' : value === 'low-power' ? 'Energiesparen bevorzugt' : 'Softwareadapter';
      found.push({ value, label: `${prefix}: ${name}`,
        name, limits: { maxTextureDimension2D: adapter.limits.maxTextureDimension2D,
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
    } catch { /* An adapter is listed only when a device can actually be opened. */ }
  }
  return found;
}
