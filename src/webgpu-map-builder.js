import { MASK_FORBIDDEN } from './patch-mask.js';

let contextPromise = null;
let contextFailure = '';

const shader = /* wgsl */ `
struct Params {
  sourceWidth: u32,
  sourceHeight: u32,
  outputWidth: u32,
  chunkRows: u32,
  originX: i32,
  originY: i32,
  chunkStart: u32,
  nx: u32,
  spacing: f32,
  tileSize: u32,
  tileColumns: u32,
  padding: u32,
}

struct Triangle {
  target0: vec2<f32>,
  target1: vec2<f32>,
  target2: vec2<f32>,
  source0: vec2<f32>,
  source1: vec2<f32>,
  source2: vec2<f32>,
}

struct Evaluation {
  position: vec2<f32>,
  row0: vec2<f32>,
  row1: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> coefficients: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(3) var<storage, read> binOffsets: array<u32>;
@group(0) @binding(4) var<storage, read> binTriangles: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;

fn basis(value: f32) -> vec4<f32> {
  let inverse = 1.0 - value;
  return vec4(inverse * inverse * inverse / 6.0,
    (3.0 * value * value * value - 6.0 * value * value + 4.0) / 6.0,
    (-3.0 * value * value * value + 3.0 * value * value + 3.0 * value + 1.0) / 6.0,
    value * value * value / 6.0);
}

fn derivative(value: f32) -> vec4<f32> {
  return vec4(-0.5 * (1.0 - value) * (1.0 - value), 1.5 * value * value - 2.0 * value,
    -1.5 * value * value + value + 0.5, 0.5 * value * value);
}

fn evaluate(px: f32, py: f32, jacobian: bool) -> Evaluation {
  let cellX = i32(floor(px / params.spacing));
  let cellY = i32(floor(py / params.spacing));
  let localX = px / params.spacing - f32(cellX);
  let localY = py / params.spacing - f32(cellY);
  let weightsX = basis(localX);
  let weightsY = basis(localY);
  let derivativesX = derivative(localX) / params.spacing;
  let derivativesY = derivative(localY) / params.spacing;
  var result = Evaluation(vec2(px, py), vec2(1.0, 0.0), vec2(0.0, 1.0));
  for (var row = 0; row < 4; row++) {
    for (var col = 0; col < 4; col++) {
      let coefficient = coefficients[u32((cellY + row) * i32(params.nx) + cellX + col)];
      result.position += weightsX[col] * weightsY[row] * coefficient;
      if (jacobian) {
        result.row0 += vec2(derivativesX[col] * weightsY[row] * coefficient.x,
          weightsX[col] * derivativesY[row] * coefficient.x);
        result.row1 += vec2(derivativesX[col] * weightsY[row] * coefficient.y,
          weightsX[col] * derivativesY[row] * coefficient.y);
      }
    }
  }
  return result;
}

fn seedFor(outputX: u32, outputY: u32, destination: vec2<f32>) -> vec2<f32> {
  let tileColumn = outputX / params.tileSize;
  let tileRow = outputY / params.tileSize;
  let bin = tileRow * params.tileColumns + tileColumn;
  let start = binOffsets[bin];
  let end = binOffsets[bin + 1u];
  for (var offset = start; offset < end; offset++) {
    let triangle = triangles[binTriangles[offset]];
    let determinant = (triangle.target1.y - triangle.target2.y) * (triangle.target0.x - triangle.target2.x) +
      (triangle.target2.x - triangle.target1.x) * (triangle.target0.y - triangle.target2.y);
    let first = ((triangle.target1.y - triangle.target2.y) * (destination.x - triangle.target2.x) +
      (triangle.target2.x - triangle.target1.x) * (destination.y - triangle.target2.y)) / determinant;
    let second = ((triangle.target2.y - triangle.target0.y) * (destination.x - triangle.target2.x) +
      (triangle.target0.x - triangle.target2.x) * (destination.y - triangle.target2.y)) / determinant;
    let third = 1.0 - first - second;
    if (min(first, min(second, third)) >= -0.0000001) {
      return first * triangle.source0 + second * triangle.source1 + third * triangle.source2;
    }
  }
  return vec2(-1.0);
}

@compute @workgroup_size(16, 16)
fn inverseMap(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.outputWidth || id.y >= params.chunkRows) { return; }
  let outputY = params.chunkStart + id.y;
  let destination = vec2(f32(params.originX) + f32(id.x), f32(params.originY) + f32(outputY));
  let seed = seedFor(id.x, outputY, destination);
  let outputIndex = id.y * params.outputWidth + id.x;
  if (seed.x < 0.0) { output[outputIndex] = vec4(-1.0, -1.0, 0.0, 0.0); return; }
  var point = clamp(seed, vec2(0.0), vec2(f32(params.sourceWidth - 1u), f32(params.sourceHeight - 1u)));
  var converged = false;
  for (var iteration = 0; iteration < 25; iteration++) {
    let value = evaluate(point.x, point.y, true);
    let error = value.position - destination;
    let magnitude = length(error);
    if (magnitude < 0.002) { converged = true; break; }
    let determinant = value.row0.x * value.row1.y - value.row0.y * value.row1.x;
    if (determinant <= 0.0000000001) { break; }
    let step = vec2(value.row1.y * error.x - value.row0.y * error.y,
      -value.row1.x * error.x + value.row0.x * error.y) / determinant;
    var accepted = false;
    var alpha = 1.0;
    for (var attempt = 0; attempt < 8; attempt++) {
      let next = point - alpha * step;
      if (all(next >= vec2(0.0)) && next.x <= f32(params.sourceWidth - 1u) && next.y <= f32(params.sourceHeight - 1u)) {
        let nextValue = evaluate(next.x, next.y, false);
        if (distance(nextValue.position, destination) < magnitude) { point = next; accepted = true; break; }
      }
      alpha *= 0.5;
    }
    if (!accepted) { break; }
  }
  if (!converged || point.x >= f32(params.sourceWidth - 1u) || point.y >= f32(params.sourceHeight - 1u)) {
    output[outputIndex] = vec4(-1.0, -1.0, 0.0, 0.0); return;
  }
  let storedError = distance(evaluate(point.x, point.y, false).position, destination);
  output[outputIndex] = select(vec4(-1.0, -1.0, storedError, 0.0), vec4(point, storedError, 1.0), storedError <= 0.01);
}
`;

async function context() {
  if (!contextPromise) contextPromise = (async () => {
    if (!globalThis.navigator?.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    const module = device.createShaderModule({ code: shader });
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(errors.map(message => message.message).join('\n'));
    return { device, pipeline: device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'inverseMap' } }) };
  })().catch(error => { contextFailure = error.message; return null; });
  return contextPromise;
}

function buffer(device, data, usage) {
  const result = device.createBuffer({ size: Math.max(4, Math.ceil(data.byteLength / 4) * 4), usage, mappedAtCreation: true });
  new Uint8Array(result.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  result.unmap();
  return result;
}

function triangleBins(geometry, origin, outputWidth, outputHeight, tileSize = 64) {
  const columns = Math.ceil(outputWidth / tileSize);
  const rows = Math.ceil(outputHeight / tileSize);
  const bins = Array.from({ length: columns * rows }, () => []);
  const values = new Float32Array(geometry.triangles.length * 12);
  geometry.triangles.forEach((triangle, triangleIndex) => {
    const coordinates = triangle.flatMap(point => [point.x, point.y, point.px, point.py]);
    values.set([coordinates[0], coordinates[1], coordinates[4], coordinates[5], coordinates[8], coordinates[9],
      coordinates[2], coordinates[3], coordinates[6], coordinates[7], coordinates[10], coordinates[11]], triangleIndex * 12);
    const minimumX = Math.max(0, Math.floor((Math.min(...triangle.map(point => point.x)) - origin[0]) / tileSize));
    const maximumX = Math.min(columns - 1, Math.floor((Math.max(...triangle.map(point => point.x)) - origin[0]) / tileSize));
    const minimumY = Math.max(0, Math.floor((Math.min(...triangle.map(point => point.y)) - origin[1]) / tileSize));
    const maximumY = Math.min(rows - 1, Math.floor((Math.max(...triangle.map(point => point.y)) - origin[1]) / tileSize));
    for (let row = minimumY; row <= maximumY; row++) {
      for (let col = minimumX; col <= maximumX; col++) bins[row * columns + col].push(triangleIndex);
    }
  });
  const offsets = new Uint32Array(bins.length + 1);
  for (let index = 0; index < bins.length; index++) offsets[index + 1] = offsets[index] + bins[index].length;
  const indices = new Uint32Array(offsets[offsets.length - 1]);
  bins.forEach((entries, index) => indices.set(entries, offsets[index]));
  return { tileSize, columns, values, offsets, indices };
}

function forbidden(mask, x, y) {
  if (!mask?.data || x < 0 || y < 0 || x >= mask.sourceWidth || y >= mask.sourceHeight) return false;
  const col = Math.min(mask.width - 1, Math.floor((x + 1e-4) / mask.cellSize));
  const row = Math.min(mask.height - 1, Math.floor((y + 1e-4) / mask.cellSize));
  return mask.data[row * mask.width + col] === MASK_FORBIDDEN;
}

export async function buildInverseMapsGpu(field, geometry, origin, outputWidth, outputHeight, sourceCoverage,
  patchMask, notify = () => {}, cancelled = () => false) {
  const started = performance.now();
  const gpu = await context();
  if (!gpu) return null;
  const outputCount = outputWidth * outputHeight;
  const inverseX = new Float32Array(outputCount).fill(-1);
  const inverseY = new Float32Array(outputCount).fill(-1);
  const valid = new Uint8Array(outputCount);
  const numericalValid = new Uint8Array(outputCount);
  const bins = triangleBins(geometry, origin, outputWidth, outputHeight);
  const coefficientValues = Float32Array.from(field.coefficients);
  const coefficientBuffer = buffer(gpu.device, coefficientValues, GPUBufferUsage.STORAGE);
  const triangleBuffer = buffer(gpu.device, bins.values, GPUBufferUsage.STORAGE);
  const offsetBuffer = buffer(gpu.device, bins.offsets, GPUBufferUsage.STORAGE);
  const indexBuffer = buffer(gpu.device, bins.indices, GPUBufferUsage.STORAGE);
  const maximumRows = Math.max(1, Math.floor(gpu.device.limits.maxStorageBufferBindingSize / (outputWidth * 16)));
  const chunkSize = Math.min(256, maximumRows);
  const roundtrip = [];
  let numericalCount = 0;
  let validCount = 0;
  let maxRoundtrip = 0;
  let cropMinimumX = outputWidth;
  let cropMinimumY = outputHeight;
  let cropMaximumX = -1;
  let cropMaximumY = -1;
  try {
    for (let chunkStart = 0; chunkStart < outputHeight; chunkStart += chunkSize) {
      const chunkRows = Math.min(chunkSize, outputHeight - chunkStart);
      const count = outputWidth * chunkRows;
      const values = new ArrayBuffer(48);
      const integers = new Uint32Array(values);
      integers.set([field.width, field.height, outputWidth, chunkRows]);
      new Int32Array(values).set([origin[0], origin[1]], 4);
      integers.set([chunkStart, field.nx], 6);
      new Float32Array(values)[8] = field.spacing;
      integers.set([bins.tileSize, bins.columns, 0], 9);
      const paramsBuffer = buffer(gpu.device, new Uint8Array(values), GPUBufferUsage.UNIFORM);
      const outputBuffer = gpu.device.createBuffer({ size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readbackBuffer = gpu.device.createBuffer({ size: count * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        const bindGroup = gpu.device.createBindGroup({ layout: gpu.pipeline.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: paramsBuffer } }, { binding: 1, resource: { buffer: coefficientBuffer } },
          { binding: 2, resource: { buffer: triangleBuffer } }, { binding: 3, resource: { buffer: offsetBuffer } },
          { binding: 4, resource: { buffer: indexBuffer } }, { binding: 5, resource: { buffer: outputBuffer } }
        ] });
        const encoder = gpu.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(gpu.pipeline); pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(outputWidth / 16), Math.ceil(chunkRows / 16)); pass.end();
        encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, count * 16);
        gpu.device.queue.submit([encoder.finish()]);
        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const output = new Float32Array(readbackBuffer.getMappedRange());
        for (let localIndex = 0; localIndex < count; localIndex++) {
          if (output[localIndex * 4 + 3] < 0.5) continue;
          const index = chunkStart * outputWidth + localIndex;
          const px = output[localIndex * 4];
          const py = output[localIndex * 4 + 1];
          const error = output[localIndex * 4 + 2];
          if (forbidden(patchMask, px, py)) continue;
          inverseX[index] = px; inverseY[index] = py; numericalValid[index] = 1;
          numericalCount++;
          maxRoundtrip = Math.max(maxRoundtrip, error);
          if (index % 101 === 0) roundtrip.push(error);
          const sourceIndex = Math.floor(py) * field.width + Math.floor(px);
          if (sourceCoverage[sourceIndex] >= 3 && sourceCoverage[sourceIndex + 1] >= 3 &&
            sourceCoverage[sourceIndex + field.width] >= 3 && sourceCoverage[sourceIndex + field.width + 1] >= 3) {
            valid[index] = 1; validCount++;
          }
          cropMinimumX = Math.min(cropMinimumX, index % outputWidth);
          cropMinimumY = Math.min(cropMinimumY, Math.floor(index / outputWidth));
          cropMaximumX = Math.max(cropMaximumX, index % outputWidth);
          cropMaximumY = Math.max(cropMaximumY, Math.floor(index / outputWidth));
        }
        readbackBuffer.unmap();
      } finally { paramsBuffer.destroy(); outputBuffer.destroy(); readbackBuffer.destroy(); }
      notify({ stage: 'inverse', done: Math.min(outputHeight, chunkStart + chunkRows), total: outputHeight, accelerator: 'WebGPU' });
      if (cancelled()) throw new Error('Berechnung abgebrochen.');
    }
  } finally { coefficientBuffer.destroy(); triangleBuffer.destroy(); offsetBuffer.destroy(); indexBuffer.destroy(); }
  return { inverseX, inverseY, valid, numericalValid, numericalCount, validCount, maxRoundtrip, roundtrip,
    cropBounds: [cropMinimumX, cropMinimumY, cropMaximumX, cropMaximumY],
    timing: { totalMs: performance.now() - started }, accelerator: 'WebGPU' };
}

export async function mapBuilderGpuStatus() {
  const gpu = await context();
  return { available: Boolean(gpu), failure: contextFailure };
}