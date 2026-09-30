import { onWebGpuSelectionChange, requestSelectedGpuAdapter } from './webgpu-selection.js';

let contextPromise = null;
onWebGpuSelectionChange(() => {
  const previous = contextPromise; contextPromise = null;
  void previous?.then(value => value?.device?.destroy()).catch(() => {});
});

const shader = /* wgsl */ `
struct Params {
  rows: u32,
  columns: u32,
  rightOffset: u32,
  initialOffset: u32,
  uOffset: u32,
  vOffset: u32,
  wOffset: u32,
  xOffset: u32,
  tempMOffset: u32,
  tempNOffset: u32,
  stateOffset: u32,
  transposeOffset: u32,
  transposeParts: u32,
  padding2: u32,
  padding3: u32,
  padding4: u32,
}
struct Entry { index: u32, weight: f32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> rowOffsets: array<u32>;
@group(0) @binding(2) var<storage, read> rowEntries: array<Entry>;
@group(0) @binding(3) var<storage, read> columnOffsets: array<u32>;
@group(0) @binding(4) var<storage, read> columnEntries: array<Entry>;
@group(0) @binding(5) var<storage, read> scales: array<f32>;
@group(0) @binding(6) var<storage, read_write> work: array<f32>;
var<workgroup> sums: array<f32, 256>;

fn state(lane: u32, slot: u32) -> u32 { return params.stateOffset + lane * 8u + slot; }

@compute @workgroup_size(64)
fn initialResidual(@builtin(global_invocation_id) id: vec3<u32>) {
  let item = id.x;
  if (item >= 2u * params.rows) { return; }
  let lane = item / params.rows;
  let row = item % params.rows;
  var value = 0.0;
  for (var cursor = rowOffsets[row]; cursor < rowOffsets[row + 1u]; cursor++) {
    let entry = rowEntries[cursor];
    value += entry.weight * work[params.initialOffset + lane * params.columns + entry.index];
  }
  work[params.uOffset + item] = work[params.rightOffset + item] - value;
}

fn reduceSum(value: f32, local: u32) -> f32 {
  sums[local] = value;
  workgroupBarrier();
  var stride = 128u;
  while (stride > 0u) {
    if (local < stride) { sums[local] += sums[local + stride]; }
    workgroupBarrier();
    stride /= 2u;
  }
  return sums[0];
}

fn reduceNorm(offset: u32, length: u32, lane: u32, local: u32) -> f32 {
  var sum = 0.0;
  for (var index = local; index < length; index += 256u) {
    let value = work[offset + lane * length + index];
    sum += value * value;
  }
  return sqrt(reduceSum(sum, local));
}

@compute @workgroup_size(256)
fn initialNormU(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let lane = group.x;
  let value = reduceNorm(params.uOffset, params.rows, lane, localId.x);
  if (localId.x == 0u) {
    work[state(lane, 0u)] = value;
    work[state(lane, 2u)] = value;
  }
}

@compute @workgroup_size(64)
fn normalizeInitialU(@builtin(global_invocation_id) id: vec3<u32>) {
  let item = id.x;
  if (item >= 2u * params.rows) { return; }
  let lane = item / params.rows;
  work[params.uOffset + item] /= max(work[state(lane, 0u)], 1e-20);
}

@compute @workgroup_size(256)
fn transposePartial(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let part = group.x;
  let column = group.y;
  let lane = group.z;
  var value = 0.0;
  for (var cursor = columnOffsets[column] + part * 256u + localId.x;
       cursor < columnOffsets[column + 1u]; cursor += params.transposeParts * 256u) {
    let entry = columnEntries[cursor];
    value += entry.weight * work[params.uOffset + lane * params.rows + entry.index];
  }
  let partial = reduceSum(value, localId.x);
  if (localId.x == 0u) {
    work[params.transposeOffset + (lane * params.columns + column) * params.transposeParts + part] = partial;
  }
}

fn transposeSum(item: u32) -> f32 {
  var value = 0.0;
  for (var part = 0u; part < params.transposeParts; part++) {
    value += work[params.transposeOffset + item * params.transposeParts + part];
  }
  return value;
}

@compute @workgroup_size(64)
fn initialTranspose(@builtin(global_invocation_id) id: vec3<u32>) {
  let item = id.x;
  if (item >= 2u * params.columns) { return; }
  let column = item % params.columns;
  let value = transposeSum(item);
  work[params.tempNOffset + item] = scales[column] * value;
}

@compute @workgroup_size(256)
fn initialNormV(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let lane = group.x;
  let value = reduceNorm(params.tempNOffset, params.columns, lane, localId.x);
  if (localId.x == 0u) {
    work[state(lane, 1u)] = value;
    work[state(lane, 3u)] = value;
    work[state(lane, 7u)] = value * work[state(lane, 0u)];
  }
}

@compute @workgroup_size(64)
fn initializeVectors(@builtin(global_invocation_id) id: vec3<u32>) {
  let item = id.x;
  if (item >= 2u * params.columns) { return; }
  let lane = item / params.columns;
  let value = work[params.tempNOffset + item] / max(work[state(lane, 1u)], 1e-20);
  work[params.vOffset + item] = value;
  work[params.wOffset + item] = value;
  work[params.xOffset + item] = 0.0;
}

@compute @workgroup_size(64)
fn multiplyIteration(@builtin(global_invocation_id) id: vec3<u32>) {
  let item = id.x;
  if (item >= 2u * params.rows) { return; }
  let lane = item / params.rows;
  let row = item % params.rows;
  var value = 0.0;
  for (var cursor = rowOffsets[row]; cursor < rowOffsets[row + 1u]; cursor++) {
    let entry = rowEntries[cursor];
    value += entry.weight * scales[entry.index] * work[params.vOffset + lane * params.columns + entry.index];
  }
  work[params.tempMOffset + item] = value - work[state(lane, 1u)] * work[params.uOffset + item];
}

@compute @workgroup_size(256)
fn normBeta(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let lane = group.x;
  let value = reduceNorm(params.tempMOffset, params.rows, lane, localId.x);
  if (localId.x == 0u) { work[state(lane, 0u)] = value; }
}

@compute @workgroup_size(64)
fn normalizeU(@builtin(global_invocation_id) id: vec3<u32>) {
  let item = id.x;
  if (item >= 2u * params.rows) { return; }
  let lane = item / params.rows;
  work[params.uOffset + item] = work[params.tempMOffset + item] / max(work[state(lane, 0u)], 1e-20);
}

@compute @workgroup_size(64)
fn transposeIteration(@builtin(global_invocation_id) id: vec3<u32>) {
  let item = id.x;
  if (item >= 2u * params.columns) { return; }
  let lane = item / params.columns;
  let column = item % params.columns;
  let value = transposeSum(item);
  work[params.tempNOffset + item] = scales[column] * value - work[state(lane, 0u)] * work[params.vOffset + item];
}

@compute @workgroup_size(256)
fn normAlpha(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
  let lane = group.x;
  let value = reduceNorm(params.tempNOffset, params.columns, lane, localId.x);
  if (localId.x == 0u) { work[state(lane, 1u)] = value; }
}

@compute @workgroup_size(64)
fn normalizeV(@builtin(global_invocation_id) id: vec3<u32>) {
  let item = id.x;
  if (item >= 2u * params.columns) { return; }
  let lane = item / params.columns;
  work[params.vOffset + item] = work[params.tempNOffset + item] / max(work[state(lane, 1u)], 1e-20);
}

@compute @workgroup_size(1)
fn updateScalars(@builtin(global_invocation_id) id: vec3<u32>) {
  let lane = id.x;
  if (lane >= 2u) { return; }
  let beta = work[state(lane, 0u)];
  let alpha = work[state(lane, 1u)];
  let phiBar = work[state(lane, 2u)];
  let rhoBar = work[state(lane, 3u)];
  let rhoScale = max(max(abs(rhoBar), abs(beta)), 1e-20);
  let rho = max(rhoScale * length(vec2<f32>(rhoBar, beta) / rhoScale), 1e-20);
  let cosine = rhoBar / rho;
  let sine = beta / rho;
  work[state(lane, 4u)] = cosine * phiBar;
  work[state(lane, 5u)] = rho;
  work[state(lane, 6u)] = sine * alpha;
  work[state(lane, 2u)] = phiBar * sine;
  work[state(lane, 3u)] = -cosine * alpha;
}

@compute @workgroup_size(64)
fn updateSolution(@builtin(global_invocation_id) id: vec3<u32>) {
  let item = id.x;
  if (item >= 2u * params.columns) { return; }
  let lane = item / params.columns;
  let oldW = work[params.wOffset + item];
  work[params.xOffset + item] += work[state(lane, 4u)] / work[state(lane, 5u)] * oldW;
  work[params.wOffset + item] = work[params.vOffset + item] - work[state(lane, 6u)] / work[state(lane, 5u)] * oldW;
}

@compute @workgroup_size(64)
fn finishSolution(@builtin(global_invocation_id) id: vec3<u32>) {
  let item = id.x;
  if (item >= 2u * params.columns) { return; }
  let column = item % params.columns;
  work[params.initialOffset + item] += scales[column] * work[params.xOffset + item];
}
`;

async function context() {
  if (!navigator.gpu) throw new Error('WebGPU wird von diesem Browser nicht bereitgestellt.');
  if (!contextPromise) contextPromise = (async () => {
    const adapter = await requestSelectedGpuAdapter();
    if (!adapter) throw new Error('Kein WebGPU-Adapter verfuegbar.');
    let device;
    try {
      device = await adapter.requestDevice({ requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : [] });
    } catch {
      device = await adapter.requestDevice();
    }
    const module = device.createShaderModule({ code: shader });
    const bindGroupLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ...[1, 2, 3, 4, 5].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } })),
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
    ] });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    const names = ['initialResidual', 'initialNormU', 'normalizeInitialU', 'transposePartial', 'initialTranspose', 'initialNormV',
      'initializeVectors', 'multiplyIteration', 'normBeta', 'normalizeU', 'transposeIteration', 'normAlpha',
      'normalizeV', 'updateScalars', 'updateSolution', 'finishSolution'];
    const pipelines = Object.fromEntries(await Promise.all(names.map(async name => [name,
      await device.createComputePipelineAsync({ layout, compute: { module, entryPoint: name } })])));
    return { device, pipelines, bindGroupLayout };
  })();
  return contextPromise;
}

function buffer(device, data, usage) {
  const result = device.createBuffer({ size: Math.max(4, Math.ceil(data.byteLength / 4) * 4), usage, mappedAtCreation: true });
  new Uint8Array(result.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  result.unmap();
  return result;
}

function entriesBuffer(device, indices, weights) {
  const bytes = new ArrayBuffer(indices.length * 8);
  const view = new DataView(bytes);
  for (let index = 0; index < indices.length; index++) {
    view.setUint32(index * 8, indices[index], true);
    view.setFloat32(index * 8 + 4, weights[index], true);
  }
  return buffer(device, new Uint8Array(bytes), GPUBufferUsage.STORAGE);
}

function sparseArrays(rows, columns) {
  const rowOffsets = new Uint32Array(rows.length + 1);
  let nonzeros = 0;
  rows.forEach((row, index) => { nonzeros += row.indices.length; rowOffsets[index + 1] = nonzeros; });
  const rowIndices = new Uint32Array(nonzeros);
  const rowWeights = new Float32Array(nonzeros);
  const counts = new Uint32Array(columns);
  let cursor = 0;
  rows.forEach(row => row.indices.forEach((column, local) => {
    rowIndices[cursor] = column;
    rowWeights[cursor++] = row.weights[local];
    counts[column]++;
  }));
  const columnOffsets = new Uint32Array(columns + 1);
  for (let index = 0; index < columns; index++) columnOffsets[index + 1] = columnOffsets[index] + counts[index];
  const positions = columnOffsets.slice(0, columns);
  const columnIndices = new Uint32Array(nonzeros);
  const columnWeights = new Float32Array(nonzeros);
  rows.forEach((row, rowIndex) => row.indices.forEach((column, local) => {
    const position = positions[column]++;
    columnIndices[position] = rowIndex;
    columnWeights[position] = row.weights[local];
  }));
  const scales = new Float32Array(columns);
  for (let index = 0; index < nonzeros; index++) scales[rowIndices[index]] += rowWeights[index] ** 2;
  for (let index = 0; index < columns; index++) scales[index] = 1 / Math.sqrt(scales[index] || 1);
  return { rowOffsets, rowIndices, rowWeights, columnOffsets, columnIndices, columnWeights, scales };
}

export function transposePartitionCount(columnOffsets) {
  let longestColumn = 0;
  for (let column = 0; column + 1 < columnOffsets.length; column++) {
    longestColumn = Math.max(longestColumn, columnOffsets[column + 1] - columnOffsets[column]);
  }
  return Math.max(1, Math.min(64, Math.ceil(longestColumn / 2048)));
}

export async function webGpuLsqrPair(rows, rights, columns, initials, iterations = 350, reportProfile = null) {
  const started = performance.now();
  const { device, pipelines, bindGroupLayout } = await context();
  const contextFinished = performance.now();
  const matrix = sparseArrays(rows, columns);
  if (columns > device.limits.maxComputeWorkgroupsPerDimension) throw new Error('Zu viele Spalten fuer das WebGPU-Transposegitter.');
  const transposeParts = transposePartitionCount(matrix.columnOffsets);
  const sparseFinished = performance.now();
  const rowCount = rows.length;
  const offsets = {};
  let length = 0;
  for (const [name, size] of [['right', 2 * rowCount], ['initial', 2 * columns], ['u', 2 * rowCount],
    ['v', 2 * columns], ['w', 2 * columns], ['x', 2 * columns], ['tempM', 2 * rowCount], ['tempN', 2 * columns],
    ['state', 16], ['transpose', 2 * columns * transposeParts]]) {
    offsets[name] = length;
    length += size;
  }
  const workspace = new Float32Array(length);
  rights.forEach((values, lane) => workspace.set(values, offsets.right + lane * rowCount));
  initials.forEach((values, lane) => workspace.set(values, offsets.initial + lane * columns));
  const params = new Uint32Array([rowCount, columns, offsets.right, offsets.initial, offsets.u, offsets.v, offsets.w,
    offsets.x, offsets.tempM, offsets.tempN, offsets.state, offsets.transpose, transposeParts, 0, 0, 0]);
  const resources = [];
  try {
    const paramsBuffer = buffer(device, params, GPUBufferUsage.UNIFORM);
    const rowOffsetBuffer = buffer(device, matrix.rowOffsets, GPUBufferUsage.STORAGE);
    const rowEntryBuffer = entriesBuffer(device, matrix.rowIndices, matrix.rowWeights);
    const columnOffsetBuffer = buffer(device, matrix.columnOffsets, GPUBufferUsage.STORAGE);
    const columnEntryBuffer = entriesBuffer(device, matrix.columnIndices, matrix.columnWeights);
    const scaleBuffer = buffer(device, matrix.scales, GPUBufferUsage.STORAGE);
    const workspaceBuffer = buffer(device, workspace, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const readback = device.createBuffer({ size: 2 * columns * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    resources.push(paramsBuffer, rowOffsetBuffer, rowEntryBuffer, columnOffsetBuffer, columnEntryBuffer, scaleBuffer, workspaceBuffer, readback);
    const entries = [paramsBuffer, rowOffsetBuffer, rowEntryBuffer, columnOffsetBuffer, columnEntryBuffer, scaleBuffer, workspaceBuffer]
      .map((resource, binding) => ({ binding, resource: { buffer: resource } }));
    const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries });
    const buffersFinished = performance.now();
    const queryCount = 2 * (8 + 9 * Math.max(0, Math.ceil(iterations)));
    const timestamps = Boolean(reportProfile && device.features.has('timestamp-query') && queryCount <= 4096);
    let querySet;
    let queryResolve;
    let queryReadback;
    const kernelNames = [];
    if (timestamps) {
      querySet = device.createQuerySet({ type: 'timestamp', count: queryCount });
      resources.push(querySet);
      queryResolve = device.createBuffer({ size: queryCount * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      resources.push(queryResolve);
      queryReadback = device.createBuffer({ size: queryCount * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      resources.push(queryReadback);
    }
    const encoder = device.createCommandEncoder();
    const sharedPass = timestamps ? null : encoder.beginComputePass();
    const dispatch = (name, count, countY = 1, countZ = 1) => {
      const queryIndex = kernelNames.length * 2;
      const pass = sharedPass ?? encoder.beginComputePass({ timestampWrites: {
        querySet, beginningOfPassWriteIndex: queryIndex, endOfPassWriteIndex: queryIndex + 1
      } });
      pass.setPipeline(pipelines[name]);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(count, countY, countZ);
      if (timestamps) { pass.end(); kernelNames.push(name); }
    };
    dispatch('initialResidual', Math.ceil(2 * rowCount / 64));
    dispatch('initialNormU', 2);
    dispatch('normalizeInitialU', Math.ceil(2 * rowCount / 64));
    dispatch('transposePartial', transposeParts, columns, 2);
    dispatch('initialTranspose', Math.ceil(2 * columns / 64));
    dispatch('initialNormV', 2);
    dispatch('initializeVectors', Math.ceil(2 * columns / 64));
    for (let iteration = 0; iteration < iterations; iteration++) {
      dispatch('multiplyIteration', Math.ceil(2 * rowCount / 64));
      dispatch('normBeta', 2);
      dispatch('normalizeU', Math.ceil(2 * rowCount / 64));
      dispatch('transposePartial', transposeParts, columns, 2);
      dispatch('transposeIteration', Math.ceil(2 * columns / 64));
      dispatch('normAlpha', 2);
      dispatch('normalizeV', Math.ceil(2 * columns / 64));
      dispatch('updateScalars', 2);
      dispatch('updateSolution', Math.ceil(2 * columns / 64));
    }
    dispatch('finishSolution', Math.ceil(2 * columns / 64));
    sharedPass?.end();
    encoder.copyBufferToBuffer(workspaceBuffer, offsets.initial * 4, readback, 0, 2 * columns * 4);
    if (timestamps) {
      encoder.resolveQuerySet(querySet, 0, kernelNames.length * 2, queryResolve, 0);
      encoder.copyBufferToBuffer(queryResolve, 0, queryReadback, 0, kernelNames.length * 16);
    }
    const commands = encoder.finish();
    const commandsFinished = performance.now();
    device.queue.submit([commands]);
    await Promise.all([readback.mapAsync(GPUMapMode.READ), queryReadback?.mapAsync(GPUMapMode.READ)]);
    const completionFinished = performance.now();
    const values = new Float32Array(readback.getMappedRange());
    const result = [Float64Array.from(values.subarray(0, columns)), Float64Array.from(values.subarray(columns))];
    readback.unmap();
    const kernelsMs = {};
    if (timestamps) {
      const values = new BigUint64Array(queryReadback.getMappedRange());
      kernelNames.forEach((name, index) => {
        kernelsMs[name] = (kernelsMs[name] ?? 0) + Number(values[index * 2 + 1] - values[index * 2]) / 1e6;
      });
      queryReadback.unmap();
    }
    reportProfile?.({ timestampStatus: timestamps ? 'available' : device.features.has('timestamp-query') ? 'query-limit' : 'unsupported',
      contextMs: contextFinished - started, sparseMs: sparseFinished - contextFinished,
      buffersMs: buffersFinished - sparseFinished, commandsMs: commandsFinished - buffersFinished,
      completionMs: completionFinished - commandsFinished, readbackMs: performance.now() - completionFinished,
      kernelsMs, rows: rowCount, columns, nonzeros: matrix.rowIndices.length, iterations, transposeParts });
    return result;
  } finally {
    resources.forEach(resource => resource.destroy());
  }
}
