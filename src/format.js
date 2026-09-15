import { zipSync, unzipSync, strToU8, strFromU8 } from 'fflate';

const limit = 768 * 1024 * 1024;
const littleEndian = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
const definitions = {
  coefficients: ['float64', 8], forward: ['float32', 4], inverseX: ['float32', 4], inverseY: ['float32', 4],
  sourceCoverage: ['uint8', 1], valid: ['uint8', 1], numericalValid: ['uint8', 1]
};

function bytesFor(array) {
  const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength).slice();
  if (!littleEndian && array.BYTES_PER_ELEMENT > 1) {
    for (let offset = 0; offset < bytes.length; offset += array.BYTES_PER_ELEMENT) bytes.subarray(offset, offset + array.BYTES_PER_ELEMENT).reverse();
  }
  return bytes;
}

export function exportCalibration(calibration, observations, video, parameters, opticalConfiguration) {
  if (!calibration?.maps) throw new Error('Keine konsistente Kalibrierung zum Speichern vorhanden.');
  const { field, maps } = calibration;
  const arrays = { coefficients: field.coefficients, forward: maps.forward, inverseX: maps.inverseX, inverseY: maps.inverseY,
    sourceCoverage: maps.sourceCoverage, valid: maps.valid, numericalValid: maps.numericalValid };
  const mm = parameters.gridMm > 0 ? parameters.gridMm : null;
  const { patchMask, ...storedParameters } = parameters;
  const metadata = {
    format: 'microscope-grid-calibration', model_version: 1, byte_order: 'little-endian',
    array_order: '[row, col, component]', coordinate_convention: 'pixel centers; x right; y down; right-handed grid numbering',
    fixed_image_flip_or_rotation: 'none; unrotated coded pixels',
    source_width: field.width, source_height: field.height, pixels_per_grid_step: calibration.step,
    grid_step_mm_or_null: mm, mm_per_pixel: mm === null ? null : mm / calibration.step,
    dpi: mm === null ? null : 25.4 * calibration.step / mm,
    reference_frame_id: calibration.referenceId, reference_pose: { theta: 0, tx: 0, ty: 0 },
    spline: { basis: 'uniform-cardinal-cubic', spacing_xy: [field.spacing, field.spacing], index_origin: [-1, -1],
      nx: field.nx, ny: field.ny, coefficient_components: ['dx', 'dy'], boundary: 'optimized padded coefficients; no zero boundary' },
    output_origin_xy: maps.origin, output_width: maps.outputWidth, output_height: maps.outputHeight,
    coverage_grid: maps.coverageGrid ? { cols: maps.coverageGrid.cols, rows: maps.coverageGrid.rows } : null,
    version: calibration.version, quality: calibration.quality, validation_metrics: calibration.metrics,
    roundtrip: maps.roundtrip, parameters: storedParameters, optical_configuration: opticalConfiguration,
    video, observations, poses: calibration.poses, created_at: new Date().toISOString(),
    coverage_definition: `distinct accepted training frames per ${maps.coverageGrid?.cols ?? 24}x${maps.coverageGrid?.rows ?? 18} source heatmap cell within the observed feature footprint; >=3 for valid output; not statistical confidence`,
    inverse_invalid_value: -1, interpolation: 'bilinear; valid mask includes source support', arrays: {}
  };
  const files = {};
  if (patchMask?.data) {
    files['patch-mask.bin'] = [patchMask.data.slice(), { level: 1 }];
    metadata.patch_mask = { file: 'patch-mask.bin', width: patchMask.width, height: patchMask.height,
      cell_size: patchMask.cellSize, source_width: patchMask.sourceWidth, source_height: patchMask.sourceHeight };
  }
  for (const [name, array] of Object.entries(arrays)) {
    const filename = `${name}.bin`;
    files[filename] = [bytesFor(array), { level: 0 }];
    metadata.arrays[name] = { file: filename, dtype: definitions[name][0], length: array.length,
      shape: name === 'coefficients' ? [field.ny, field.nx, 2] : name === 'forward' ? [field.height, field.width, 2] :
        name === 'sourceCoverage' ? [field.height, field.width] : [maps.outputHeight, maps.outputWidth] };
  }
  files['metadata.json'] = strToU8(JSON.stringify(metadata));
  return zipSync(files, { level: 1 });
}

export function importCalibration(bytes) {
  let expanded = 0;
  const files = unzipSync(bytes, { filter: entry => {
    expanded += entry.originalSize;
    if (expanded > limit || entry.originalSize > limit) throw new Error('Kalibrierpaket ueberschreitet das 768-MiB-Limit.');
    return true;
  } });
  if (!files['metadata.json']) throw new Error('Metadaten fehlen.');
  const metadata = JSON.parse(strFromU8(files['metadata.json']));
  if (metadata.format !== 'microscope-grid-calibration' || metadata.model_version !== 1 || metadata.byte_order !== 'little-endian' ||
    metadata.spline?.basis !== 'uniform-cardinal-cubic' || metadata.spline.index_origin?.join(',') !== '-1,-1') {
    throw new Error('Unbekanntes Kalibrierformat oder inkompatible Splinebasis.');
  }
  const positiveInteger = value => Number.isSafeInteger(value) && value > 1 && value < 100000;
  if (![metadata.source_width, metadata.source_height, metadata.output_width, metadata.output_height].every(positiveInteger) ||
    !(metadata.pixels_per_grid_step > 0 && Number.isFinite(metadata.pixels_per_grid_step)) ||
    !metadata.output_origin_xy?.every(Number.isFinite)) throw new Error('Ungueltige Geometrie im Paket.');
  const spacing = metadata.spline.spacing_xy?.[0];
  if (!(spacing > 0 && Number.isFinite(spacing)) || metadata.spline.spacing_xy[1] !== spacing ||
    metadata.spline.nx !== Math.floor((metadata.source_width - 1) / spacing) + 4 ||
    metadata.spline.ny !== Math.floor((metadata.source_height - 1) / spacing) + 4) throw new Error('Ungueltige Kontrollgitterdimensionen.');
  const arrays = {};
  for (const [name, [dtype, byteWidth]] of Object.entries(definitions)) {
    const description = metadata.arrays?.[name];
    const raw = files[`${name}.bin`];
    const expectedLength = name === 'coefficients' ? metadata.spline.nx * metadata.spline.ny * 2 :
      name === 'forward' ? metadata.source_width * metadata.source_height * 2 :
        name === 'sourceCoverage' ? metadata.source_width * metadata.source_height : metadata.output_width * metadata.output_height;
    if (description?.dtype !== dtype || description.length !== expectedLength || raw?.length !== expectedLength * byteWidth) {
      throw new Error(`Ungueltiges Array: ${name}`);
    }
    const copy = raw.slice();
    if (!littleEndian && byteWidth > 1) for (let offset = 0; offset < copy.length; offset += byteWidth) copy.subarray(offset, offset + byteWidth).reverse();
    arrays[name] = dtype === 'float64' ? new Float64Array(copy.buffer) : dtype === 'float32' ? new Float32Array(copy.buffer) : copy;
    if (byteWidth > 1 && !arrays[name].every(Number.isFinite)) throw new Error(`Nichtendliche Koordinaten in ${name}`);
  }
  const observations = metadata.observations;
  if (!Array.isArray(observations) || new Set(observations.map(frame => frame.id)).size !== observations.length) throw new Error('Ungueltige oder doppelte Beobachtungsframes.');
  for (const frame of observations) {
    if (!Number.isSafeInteger(frame.id) || !Number.isFinite(frame.timestamp) || !['train', 'validation'].includes(frame.role) || !Array.isArray(frame.points)) {
      throw new Error('Ungueltiger Beobachtungsframe.');
    }
    for (const point of frame.points) {
      if (!(Number.isFinite(point.x) && Number.isFinite(point.y) && point.x >= 0 && point.y >= 0 &&
        point.x < metadata.source_width && point.y < metadata.source_height && Number.isFinite(point.col) && Number.isFinite(point.row) &&
        point.confidence > 0 && point.confidence <= 1)) throw new Error('Ungueltige Rasterbeobachtung.');
    }
  }
  if (!observations.some(frame => frame.id === metadata.reference_frame_id)) throw new Error('Referenzframe fehlt.');
  for (let index = 0; index < arrays.valid.length; index++) {
    if (arrays.valid[index] > 1 || arrays.numericalValid[index] > 1 || arrays.valid[index] > arrays.numericalValid[index]) throw new Error('Ungueltige Maske.');
    if (arrays.numericalValid[index] && !(arrays.inverseX[index] >= 0 && arrays.inverseY[index] >= 0 &&
      arrays.inverseX[index] < metadata.source_width - 1 && arrays.inverseY[index] < metadata.source_height - 1)) throw new Error('Inverse Map ausserhalb des Bildes.');
  }
  const gridMm = metadata.grid_step_mm_or_null;
  if (gridMm !== null && !(gridMm > 0 && Number.isFinite(gridMm))) throw new Error('Ungueltiger metrischer Massstab.');
  if (metadata.mm_per_pixel !== (gridMm === null ? null : gridMm / metadata.pixels_per_grid_step) ||
    metadata.dpi !== (gridMm === null ? null : 25.4 * metadata.pixels_per_grid_step / gridMm)) throw new Error('Widerspruechliche metrische Metadaten.');
  const coverageGrid = metadata.coverage_grid;
  if (coverageGrid !== null && coverageGrid !== undefined &&
    (![coverageGrid.cols, coverageGrid.rows].every(value => Number.isSafeInteger(value) && value > 0 && value <= 160))) {
    throw new Error('Ungueltige Heatmap-Dimensionen.');
  }
  let patchMask = null;
  if (metadata.patch_mask) {
    const description = metadata.patch_mask;
    const raw = files[description.file];
    if (![description.width, description.height, description.cell_size].every(value => Number.isSafeInteger(value) && value > 0) ||
      description.source_width !== metadata.source_width || description.source_height !== metadata.source_height ||
      description.width !== Math.ceil(metadata.source_width / description.cell_size) ||
      description.height !== Math.ceil(metadata.source_height / description.cell_size) || raw?.length !== description.width * description.height ||
      raw.some(value => value > 2)) throw new Error('Ungueltige Patch-Maske.');
    patchMask = { width: description.width, height: description.height, cellSize: description.cell_size,
      sourceWidth: description.source_width, sourceHeight: description.source_height, data: raw.slice(), revision: 0 };
  }
  const calibration = { field: { width: metadata.source_width, height: metadata.source_height, spacing,
    nx: metadata.spline.nx, ny: metadata.spline.ny, coefficients: arrays.coefficients },
    step: metadata.pixels_per_grid_step, referenceId: metadata.reference_frame_id, version: metadata.version,
    quality: metadata.quality, metrics: metadata.validation_metrics, poses: metadata.poses,
    parameters: metadata.parameters,
    maps: { forward: arrays.forward, inverseX: arrays.inverseX, inverseY: arrays.inverseY,
      sourceCoverage: arrays.sourceCoverage, valid: arrays.valid, numericalValid: arrays.numericalValid,
      origin: metadata.output_origin_xy, outputWidth: metadata.output_width, outputHeight: metadata.output_height,
      coverageGrid: coverageGrid ? { cols: coverageGrid.cols, rows: coverageGrid.rows } : undefined,
      roundtrip: metadata.roundtrip } };
  const parameters = { ...metadata.parameters, ...(patchMask ? { patchMask } : {}) };
  calibration.parameters = parameters;
  return { calibration, observations, video: metadata.video, parameters,
    opticalConfiguration: metadata.optical_configuration, metadata };
}