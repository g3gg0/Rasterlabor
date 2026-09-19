import { applyLoopClosure, trackingReferenceCounts } from './tracking-data.js';
import './style.css';
import { createIcons, icons } from 'lucide';
import { WorkerClient } from './rpc.js';
import { evaluate, fieldColor } from './spline.js';
import { poseFor } from './solver.js';
import { Field3DView } from './field-3d.js';
import { VideoExporter } from './video-exporter.js';
import { relativeCoverageScale } from './maps.js';
import { createPatchMask, MASK_FORBIDDEN, MASK_NEUTRAL, MASK_SEARCH, maskIncludes, paintPatchMask, patchAllowed, remapInclusionMask, validatePointsAgainstMask } from './patch-mask.js';
import { analyzeObservations, selectConsistentFrames } from './observation-diagnostics.js';
import { fitCameraPose, stabilizePose } from './motion-tracking.js';
import { frameGeometry, localSelectionMask, localSelectionDistance, localSelectionSupport, supportColor, applyPixelMask, edgeFeatherMask, applyEdgeFeather, accumulateFrame, approximateTopFrames, averagedFrames, sharpestFramesFirst } from './path-support.js';
import { FrameReader } from './frame-reader.js';
import { installTrackingInspector } from './tracking-inspector.js';
import { installCheckerboardView } from './checkerboard-view.js';
import { WebGpuOverlay } from './webgpu-overlay.js';
import { renderTiledOverlay, closeOverlayTiles } from './overlay-tiles.js';
import { applyPoseCorrection, buildPoseGraph, localRefitGroups, searchLocalRefit } from './pose-graph-refit.js';
import { refitColorMatrix } from './refit-preprocess.js';

createIcons({ icons });
const element = id => document.getElementById(id);
const video = element('video');
const rawCanvas = element('rawCanvas');
const resultCanvas = element('resultCanvas');
const field3dCanvas = element('field3dCanvas');
const field3d = new Field3DView(field3dCanvas);
const rawImage = document.createElement('canvas');
const fieldImage = document.createElement('canvas');
const coverageImage = document.createElement('canvas');
const residualImage = document.createElement('canvas');
const rectifiedImage = document.createElement('canvas');
const observationImage = document.createElement('canvas');
let decoder = new WorkerClient('/decoder-worker.js', showProgress);
const computer = new WorkerClient('/compute-worker.js', showProgress);
let trackingComputer = new WorkerClient('/compute-worker.js', showProgress);
let videoInfo = null;
let expectedVideo = null;
let videoUrl = null;
let currentIndex = 0;
let currentSharpness = null;
let calibration = null;
const frameReader = new FrameReader({ getDecoder: () => decoder, getMaps: () => calibration?.maps, computer,
  onFallback: error => console.warn('FrameReader: CPU-Fallback:', error.message) });
const readFrame = (index, options = {}) => frameReader.read(index, { gpu: element('useWebGpu').checked, ...options });
const readTrackingFrame = (index, options = {}) => frameReader.read(index, { gpu: element('trackingUseWebGpu').checked, ...options });
let snapshotFrames = [];
let snapshotParameters = null;
let frames = new Map();
const detections = new Map();
let currentDetection = null;
let phaseChecked = false;
let phaseStarted = false;
let continuous = false;
let cancelRequested = false;
let revalidating = false;
let taskBusy = false;
let navigationBusy = false;
let stale = false;
let detectionsStale = false;
let detectedOnce = false;
let view = 'field';
let zoom = 1;
let pan = { x: 0, y: 0 };
let nextProcessingIndex = 0;
let acceptedSinceFit = 0;
let displayRevision = 0;
let parameterRevision = 0;
let timer = null;
let measurements = [];
let rawReady = false;
let rectifiedReady = false;
let rectifiedPlayback = false;
let patchMask = null;
let observationDiagnostics = null;
let observationDiagnosticsDirty = true;
let geometrySelection = null;
let maskTool = 'pan';
let maskImageRevision = -1;
const maskImage = document.createElement('canvas');
const transforms = new Map();
const previewFilter = 'url(#imageAdjustmentFilter)';
const refitPreviewFilter = 'url(#refitAdjustmentFilter)';
function drawAdjustedImage(context, image, ...coordinates) {
  context.save();
  try {
    context.filter = previewFilter;
    context.drawImage(image, ...coordinates);
  } finally { context.restore(); }
}

function drawRefitPreviewImage(context, image, ...coordinates) {
  context.save();
  try {
    context.filter = element('refitPreprocessPreview')?.checked ? refitPreviewFilter : previewFilter;
    context.drawImage(image, ...coordinates);
  } finally { context.restore(); }
}

function refitPreprocessing() {
  return { brightness: number('refitBrightness', 0) / 100, contrast: number('refitContrast', 100) / 100,
    gamma: number('refitGamma', 1), red: number('refitRed', 30), green: number('refitGreen', 59), blue: number('refitBlue', 11),
    region: number('refitRegion', 40) / 100, edgeFeather: number('overlayEdgeFeather', 10) / 100 };
}

function updateRefitPreprocessing() {
  const options = refitPreprocessing();
  element('refitBrightnessValue').value = `${Math.round(options.brightness * 100)}`;
  element('refitContrastValue').value = `${Math.round(options.contrast * 100)}%`;
  element('refitGammaValue').value = options.gamma.toFixed(2);
  element('refitRegionValue').value = `${Math.round(options.region * 100)}%`;
  element('overlayEdgeFeatherValue').value = `${Math.round(options.edgeFeather * 100)}%`;
  for (const channel of ['Red', 'Green', 'Blue']) element(`refit${channel}Value`).value = `${number(`refit${channel}`, 0)}`;
  element('refitColorMatrix').setAttribute('values', refitColorMatrix(options));
  const intercept = 0.5 - options.contrast * 0.5 + options.brightness;
  for (const channel of ['Red', 'Green', 'Blue']) {
    element(`refit${channel}Level`).setAttribute('slope', `${options.contrast}`);
    element(`refit${channel}Level`).setAttribute('intercept', `${intercept}`);
    element(`refit${channel}Gamma`).setAttribute('exponent', `${1 / options.gamma}`);
  }
  renderPathOverlay();
}
const trackingPreviewImage = document.createElement('canvas');
let workflow = 'calibration';
let trackingRunning = false;
let trackingPaused = false;
let trackingNextIndex = 0;
let trackingPath = [];
let trackingFailures = [];
let trackingDataset = null;
let trackingPreviewTransform = null;
let trackingPreviewPoints = [];
let trackingPreviewRequest = 0;
let trackingProfile = null;
let trackingZoom = 1;
let trackingPan = { x: 0, y: 0 };
let selectedTrackingPatch = null;
let trackingRectangle = null;
let selectingTrackingWindow = false;
let trackingPreviewBusy = false;
let trackingRunOptions = null;
let trackingLost = false;
let pathProject = null;
let pathHover = null;
let pathSelection = null;
let overlayRequest = 0;
let overlayGpuIdle = Promise.resolve();
let pathOverlay = document.createElement('canvas');
let overlayContributors = [];
let overlayBounds = null;
let overlaySelectedImage = null;
let overlaySelectionRequest = 0;
function clearPathOverlay() {
  closeOverlayTiles(pathOverlay);
  if (!pathOverlay.tiles) pathOverlay.width = pathOverlay.height = 0;
  pathOverlay = document.createElement('canvas');
  overlaySelectedImage?.bitmap.close();
  overlaySelectedImage = null;
  overlayContributors = [];
  overlayBounds = null;
  overlaySelectionRequest++;
  const selector = element('trackingOverlayFrame');
  selector.replaceChildren(new Option('Mischbild', ''));
  selector.disabled = true;
  element('trackingOverlayInspect').disabled = true;
}
let overlayZoom = 1;
let overlayPan = { x: 0, y: 0 };
let pathRefitProposal = null;
let pathRefitBusy = false;
let trackingImageMask = null;
let trackingMaskHasSelection = false;
let trackingMaskTool = 'include';
let trackingMaskPreviewRequest = 0;
const trackingMaskPreview = document.createElement('canvas');
const trackingMaskPaintLayer = document.createElement('canvas');
let trackingMaskTransform = null;
const windowTracking = () => element('trackingMode').value === 'window';
const checkerboardView = installCheckerboardView({ canvas: resultCanvas, redraw: () => draw(),
  getState: () => ({ visible: view === 'rectified', ready: rectifiedReady && Boolean(calibration) && video.paused && !rectifiedPlayback && !taskBusy && !navigationBusy && !trackingRunning,
    key: `${currentIndex}:${displayRevision}:${calibration?.version}`, image: rectifiedImage, step: calibration?.step }) });
const trackingInspector = installTrackingInspector({ readFrame: readTrackingFrame,
  getMask: () => (trackingRunOptions ?? trackingDataset?.options)?.imageMask ?? null,
  busy: () => trackingRunning || trackingPreviewBusy || taskBusy,
  drawImage: drawAdjustedImage });
createIcons({ icons });
const number = (id, fallback) => element(id).value === '' ? fallback : Number(element(id).value);
const text = (id, value) => { element(id).textContent = value; };
const fixed = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : '-';
const megabytes = bytes => `${(bytes / 1024 / 1024).toFixed(bytes >= 100 * 1024 * 1024 ? 0 : 1)} MB`;
const APP_SETTINGS_KEY = 'app-settings';
const videoSettingIds = ['previewBrightness', 'previewContrast', 'previewGamma', 'trackingMaskBrush', 'speed', 'overlay'];

function storedAppSettings() {
  try {
    const settings = JSON.parse(localStorage.getItem(APP_SETTINGS_KEY) || '{}');
    return settings && typeof settings === 'object' && !Array.isArray(settings) ? settings : {};
  } catch { return {}; }
}

function restoreVideoOptions() {
  const options = storedAppSettings().videoOptions;
  if (!options || typeof options !== 'object' || Array.isArray(options)) return;
  for (const id of videoSettingIds) {
    const control = element(id), value = options[id];
    if (control.type === 'checkbox') {
      if (typeof value === 'boolean') control.checked = value;
      continue;
    }
    if (typeof value !== 'string' && typeof value !== 'number') continue;
    const candidate = String(value);
    if (control instanceof HTMLSelectElement) {
      if ([...control.options].some(option => option.value === candidate)) control.value = candidate;
      continue;
    }
    const numeric = Number(candidate), minimum = Number(control.min), maximum = Number(control.max);
    if (Number.isFinite(numeric) && (!control.min || numeric >= minimum) && (!control.max || numeric <= maximum)) control.value = candidate;
  }
}

function saveVideoOptions() {
  const settings = storedAppSettings();
  settings.videoOptions = Object.fromEntries(videoSettingIds.map(id => {
    const control = element(id);
    return [id, control.type === 'checkbox' ? control.checked : control.value];
  }));
  try { localStorage.setItem(APP_SETTINGS_KEY, JSON.stringify(settings)); } catch {}
}

function updateMemoryStats() {
  const footer = element('memoryStats');
  if (!footer) return;
  const framePoints = [...frames.values()].reduce((sum, frame) => sum + frame.points.length, 0);
  const snapshotPoints = snapshotFrames.reduce((sum, frame) => sum + frame.points.length, 0);
  const detectionPoints = [...detections.values()].reduce((sum, detection) => sum + (detection.points?.length || 0), 0);
  const uniquePoints = new Set();
  for (const frame of frames.values()) for (const point of frame.points) uniquePoints.add(point);
  for (const frame of snapshotFrames) for (const point of frame.points) uniquePoints.add(point);
  for (const detection of detections.values()) for (const point of detection.points || []) uniquePoints.add(point);
  const pointCount = uniquePoints.size;
  const pointBytes = pointCount * 5 * Float64Array.BYTES_PER_ELEMENT;
  const arrays = calibration ? [calibration.field?.coefficients, calibration.maps?.forward, calibration.maps?.inverseX,
    calibration.maps?.inverseY, calibration.maps?.sourceCoverage, calibration.maps?.valid, calibration.maps?.numericalValid] : [];
  const arrayBytes = arrays.reduce((sum, array) => sum + (array?.byteLength || 0), 0);
  const canvases = [rawImage, fieldImage, coverageImage, residualImage, rectifiedImage, rawCanvas, resultCanvas];
  const canvasBytes = canvases.reduce((sum, canvas) => sum + canvas.width * canvas.height * 4, 0);
  const heap = performance.memory;
  footer.replaceChildren();
  const entries = [
    ['Beobachtungspunkte', framePoints.toLocaleString('de-DE')],
    ['2D-Vektoren', (2 * pointCount).toLocaleString('de-DE')],
    ['Vektor-Nutzlast (geschaetzt)', megabytes(pointBytes)],
    ['Kalibrierungsarrays', megabytes(arrayBytes)],
    ['Bildpuffer', megabytes(canvasBytes)],
    ['JS-Heap (Hauptseite)', heap ? `${megabytes(heap.usedJSHeapSize)} / ${megabytes(heap.jsHeapSizeLimit)}` : 'nicht verfuegbar']
  ];
  for (const [label, value] of entries) {
    const item = document.createElement('span');
    item.append(`${label}: `);
    const strong = document.createElement('strong'); strong.textContent = value; item.append(strong); footer.append(item);
  }
  footer.title = `Punktkopien: Frames ${framePoints.toLocaleString('de-DE')}, Fit-Snapshot ${snapshotPoints.toLocaleString('de-DE')}, Erkennungscache ${detectionPoints.toLocaleString('de-DE')}. Worker-Heap ist separat und nicht enthalten.`;
}

function message(value = '', error = false) {
  element('message').hidden = !value;
  element('message').textContent = value;
  element('message').className = error ? 'error' : '';
}

function showProgress(progress) {
  const names = { index: 'MP4-Index', forward: 'Dichte Vorwaertsmap', inverse: 'Inverse Map', fit: 'Gemeinsamer Fit' };
  const accelerator = progress.accelerator ? ` | ${progress.accelerator}` : '';
  const speed = Number.isFinite(progress.iterationsPerSecond) && Number.isFinite(progress.iterationSeconds) ?
    ` | Mittel ${progress.iterationsPerSecond.toPrecision(3)} Iter./s | letzte ${fixed(progress.iterationSeconds, 2)} s` : '';
  const indexing = progress.stage === 'index' && Number.isFinite(progress.done) && Number.isFinite(progress.total) ?
    ` ${fixed(100 * progress.done / Math.max(1, progress.total), 1)}% | ${megabytes(progress.done)} / ${megabytes(progress.total)}` : '';
  text('processingStatus', progress.stage === 'fit' ? `Fit ${progress.iteration}/${progress.iterations} | RMS ${fixed(progress.rms)} px${accelerator}${speed}` :
    `${names[progress.stage] || progress.stage}${indexing}${accelerator}`);
  element('progress').value = progress.stage === 'fit' ? progress.iteration / progress.iterations : (progress.done || 0) / (progress.total || 1);
  if (progress.stage === 'fit') renderFitProfile(progress.profile);
}

function updateTimeProfiles() {
  const summaries = [];
  const exportSection = element('exportProfileSection');
  const patchSection = element('patchProfileSection');
  const fitSection = element('fitProfileSection');
  const trackingSection = element('trackingProfileSection');
  if (!exportSection.hidden) summaries.push(exportSection.dataset.summary);
  if (!patchSection.hidden) summaries.push(patchSection.dataset.summary);
  if (!fitSection.hidden) summaries.push(fitSection.dataset.summary);
  if (!trackingSection.hidden) summaries.push(trackingSection.dataset.summary);
  text('timeProfilesSummary', summaries.length ? `Zeitprofile | ${summaries.join(' | ')}` : 'Zeitprofile');
}

function renderTrackingProfile(profile) {
  const section = element('trackingProfileSection');
  section.hidden = !profile?.frames;
  if (!profile?.frames) { updateTimeProfiles(); return; }
  const average = value => value / profile.frames;
  const rows = [
    ['Frame dekodieren', average(profile.decodeMs)], ['Bildschaerfe messen', average(profile.sharpnessMs)],
    ['Bitmap nach RGBA', average(profile.rgbaMs)],
    ['Entzerrung inkl. Transfer', average(profile.remapMs)],
    ['Umfeldframes laden / entzerren', average(profile.referenceMs || 0)],
    ['Worker-Transfer / Warteschlange', average(profile.workerTransferMs)], ['Tracking im Worker', average(profile.trackMs)],
    ['Pose / Stabilisierung', average(profile.poseMs)], ['Vorschau / Tabelle / Pfad', average(profile.renderMs)],
    ['Gesamt', average(profile.totalMs)]
  ];
  const tracker = profile.tracker;
  if (profile.mode === 'window') rows.splice(5, 0,
    ['  Fenster / Grauwert / FFT', average(tracker.sampleMs)], ['  FFT-Translationssuche', average(tracker.matchMs)],
    ['  Iterative X/Y- und Rotationssuche', average(tracker.refinementMs)],
    ['  Umfeld gesamt', average(tracker.contextMs || 0)],
    ['    Pyramiden inkl. nativer Entzerrung/Transfer', average(tracker.contextPyramidMs || 0)],
    ['    Paarregistrierung vor / zurueck', average(tracker.contextRegistrationMs || 0)]);
  if (tracker.accelerator === 'WebGPU') rows.splice(4, 0,
    ['  GPU Grauwert', average(tracker.grayscaleMs)], ['  GPU Kontext / Shader', average(tracker.contextMs)],
    ['  GPU Upload', average(tracker.uploadMs)], ['  GPU Vorwaertssuche', average(tracker.forwardMs)],
    ['  GPU Rueckwaertssuche', average(tracker.backwardMs)], ['  CPU Nachfilter / Wiedererkennung', average(tracker.postprocessMs)]);
  const tbody = element('trackingProfileRows');
  const pyramid = profile.pyramidProfile;
  if (pyramid) {
    const insertion = rows.findIndex(row => row[0].includes('Pyramiden inkl.')) + 1;
    rows.splice(insertion, 0, ...[
      ['      CPU-Pyramiden', 'cpuMs'], ['      GPU-Fehlversuch', 'failedGpuMs'],
      ['      Native Entzerrung initialisieren', 'nativeRemapSetupMs'],
      ['      GPU-Initialisierung', 'setupMs'], ['      GPU-Upload vorbereiten/kopieren', 'uploadMs'],
      ['      GPU-Puffer / Befehle', 'encodeMs'], ['      GPU/Transfer warten', 'waitMs'],
      ['      Grauwert/RGBA-Rueckkopie CPU', 'readbackMs'],
      ...(pyramid.gpuTimedImages ? [['      GPU-Kernel (Teil der Wartezeit)', 'gpuComputeMs']] : [])
    ].map(([label, key]) => [label, average(pyramid[key] || 0)]));
    rows.splice(rows.findIndex(row => row[0] === 'Umfeldframes laden / entzerren') + 1, 0,
      ['  Referenzen dekodieren', average(profile.referenceDecodeMs || 0)],
      ['  Referenzen Schaerfe', average(profile.referenceSharpnessMs || 0)],
      ['  Referenzen RGBA', average(profile.referenceRgbaMs || 0)],
      ['  Referenzen entzerren', average(profile.referenceRemapMs || 0)]);
  }
  tbody.replaceChildren();
  for (const [label, milliseconds] of rows) {
    if (!Number.isFinite(milliseconds)) continue;
    const row = tbody.insertRow();
    row.insertCell().textContent = label;
    row.insertCell().textContent = `${fixed(milliseconds, 2)} ms`;
  }
  const slowest = rows.slice(0, -1).reduce((largest, row) => row[1] > largest[1] ? row : largest);
  section.dataset.summary = `Tracking ${fixed(1000 / average(profile.totalMs), 2)} fps`;
  text('trackingProfileNote', `${profile.frames} Frames | ${[...profile.accelerators].join(' / ')} | langsamster Schritt: ${slowest[0].trim()} ${fixed(slowest[1], 2)} ms / Frame | letzte Frame ${fixed(profile.lastTotalMs, 2)} ms`);
  if (pyramid) element('trackingProfileNote').textContent += ` | Pyramiden CPU/GPU ${pyramid.cpuImages || 0}/${pyramid.gpuImages || 0} | Upload/Ruecklesen ${fixed((pyramid.uploadBytes || 0) / 1048576 / profile.frames, 1)}/${fixed((pyramid.readbackBytes || 0) / 1048576 / profile.frames, 1)} MiB/Frame | Cache ${profile.cacheHits || 0} Treffer / ${profile.cacheMisses || 0} angefordert / ${profile.cacheEvictions || 0} verdraengt`;
  if (pyramid?.nativeImages) element('trackingProfileNote').textContent += ` | Direkt GPU ${pyramid.nativeImages} Bilder | RGBA fuer Fenster/Vorschau ${fixed((pyramid.rgbaReadbackBytes || 0) / 1048576 / profile.frames, 1)} MiB/Frame`;
  updateTimeProfiles();
}

function renderExportProfile(profile) {
  const section = element('exportProfileSection');
  section.hidden = !profile?.frames;
  if (!profile?.frames) { updateTimeProfiles(); return; }
  const average = value => value / profile.frames;
  const rows = [
    ['Quelldecodierung', average(profile.decodeMs)], ['Bildschaerfe messen', average(profile.sharpnessMs)],
    ['Bitmap nach RGBA', average(profile.rgbaMs)],
    ['Entzerrung im Worker', average(profile.remapComputeMs)], ['Worker-Transfer / Nachrichtenlaufzeit', average(profile.remapTransferMs)],
    ['VideoFrame vorbereiten', average(profile.frameMs)], ['Encoder-Einreichung', average(profile.submitMs)],
    ['Encoder- / Schreib-Wartezeit', average(profile.backpressureMs)], ['Gesamt', average(profile.totalMs)]
  ];
  if (profile.remapAccelerator === 'WebGPU') rows.splice(3, 0,
    ['GPU-Kontext / Map-Konfiguration', average(profile.gpuContextMs)], ['GPU-Quellupload / Puffer', average(profile.gpuUploadMs)],
    ['GPU-Befehle aufzeichnen', average(profile.gpuCommandMs)], ['GPU-Ausfuehrung + Transfer', average(profile.gpuCompletionMs)],
    ['GPU-Readback kopieren', average(profile.gpuReadbackMs)]);
  const tbody = element('exportProfileRows');
  tbody.replaceChildren();
  for (const [label, milliseconds] of rows) {
    const row = tbody.insertRow();
    row.insertCell().textContent = label;
    row.insertCell().textContent = `${fixed(milliseconds, 2)} ms`;
  }
  const fps = 1000 / average(profile.totalMs);
  section.dataset.summary = `Export ${fixed(fps, 2)} fps`;
  text('exportProfileNote', `${profile.frames} Frames gemessen | ${profile.width} x ${profile.height} px | Entzerrung ${profile.remapAccelerator || 'CPU'} | VideoFrame ${profile.directFrames ? 'direkt aus RGBA' : 'ueber Canvas'}` +
    (Number.isFinite(profile.finalizeMs) ? ` | Abschluss ${fixed(profile.finalizeMs / 1000, 2)} s` : ''));
  updateTimeProfiles();
}

function renderFitProfile(profile) {
  const section = element('fitProfileSection');
  section.hidden = !profile;
  if (!profile) { updateTimeProfiles(); return; }
  const labels = { poses: 'Posen', system: 'Systemaufbau', solve: 'LSQR gesamt', geometry: 'Geometriepruefung', residuals: 'Restfehler / Statistik' };
  const phases = Object.entries(profile.phasesMs);
  const slowest = phases.reduce((largest, entry) => entry[1] > largest[1] ? entry : largest);
  const total = phases.reduce((sum, entry) => sum + entry[1], 0);
  section.dataset.summary = `Fit ${fixed(total / 1000, 2)} s`;
  const rows = phases.map(([name, milliseconds]) => [labels[name], milliseconds]);
  const gpu = profile.gpu;
  if (gpu) {
    rows.push(['GPU-Kontext / Shader (Wartezeit)', gpu.contextMs], ['Sparse-Matrix aufbauen (CPU)', gpu.sparseMs],
      ['Puffer anlegen / befuellen (CPU)', gpu.buffersMs], ['GPU-Befehle aufzeichnen (CPU)', gpu.commandsMs],
      ['Submit bis Readback bereit (Wartezeit)', gpu.completionMs], ['Ergebnis / Zeitstempel auslesen (CPU)', gpu.readbackMs]);
    if (gpu.timestampStatus === 'available') {
      rows.push(['GPU-Kernel gesamt (Zeitstempel)', Object.values(gpu.kernelsMs).reduce((sum, value) => sum + value, 0)]);
      rows.push(...Object.entries(gpu.kernelsMs).sort((first, second) => second[1] - first[1])
        .map(([name, milliseconds]) => [`GPU: ${name}`, milliseconds]));
    }
  }
  const tbody = element('fitProfileRows');
  tbody.replaceChildren();
  for (const [label, milliseconds] of rows) {
    const row = tbody.insertRow();
    row.insertCell().textContent = label;
    row.insertCell().textContent = `${fixed(milliseconds, 2)} ms`;
  }
  text('fitProfileNote', gpu ? `${gpu.rows} Zeilen | ${gpu.columns} Unbekannte je Achse | ${gpu.nonzeros} Matrixeintraege | ${gpu.iterations} innere Iterationen. ` +
    (gpu.timestampStatus === 'available' ? 'GPU-Zeiten: Summe je Kernel ueber beide Achsen und alle inneren Iterationen. Diagnose mit zusaetzlichen Passgrenzen; Wartezeit enthaelt auch Queue, Transfers und Treiber.' :
      'Keine GPU-Kernelzeitstempel verfuegbar; Wartezeit enthaelt GPU-Rechnung, Queue, Transfers und Treiber.') : 'CPU-Zeiten der letzten abgeschlossenen Iteration.');
  updateTimeProfiles();
}

function renderPatchProfile(detection) {
  const section = element('patchProfileSection');
  const timing = detection?.timing;
  section.hidden = !timing;
  if (!timing) { updateTimeProfiles(); return; }
  let rows;
  if (detection.accelerator === 'WebGPU-Hybrid') rows = [
    ['GPU-Kontext / Shader', timing.contextMs], ['Bitmap skalieren / RGBA lesen', timing.resizeMs],
    ['GPU-Puffer hochladen', timing.uploadMs], ['GPU-Befehle aufzeichnen', timing.commandMs],
    ['GPU-Grauwert / Winkel / Readback', timing.completionMs], ['GPU-Corner-Responses', timing.cornerMs],
    ['GPU-Subpixel-Verfeinerung', timing.refineMs], ['CPU-Topologie', timing.detectMs],
    ['Worker gesamt', detection.processingMs]
  ];
  else if (detection.accelerator === 'WebGPU') rows = [
    ['Grauwertkonvertierung', timing.grayscaleMs], ['GPU-Kontext / Shader', timing.contextMs],
    ['Bildupload', timing.uploadMs], ['Vorwaertssuche', timing.forwardMs],
    ['Rueckwaertssuche', timing.backwardMs], ['CPU-Nachfilter / Nachfuellen', timing.postprocessMs],
    ['Gesamt', timing.totalMs]
  ];
  else rows = [['CPU-Erkennung gesamt', timing.cpuMs], ['Gesamt', timing.totalMs]];
  const tbody = element('patchProfileRows');
  tbody.replaceChildren();
  for (const [label, milliseconds] of rows) {
    if (!Number.isFinite(milliseconds)) continue;
    const row = tbody.insertRow();
    row.insertCell().textContent = label;
    row.insertCell().textContent = `${fixed(milliseconds, 2)} ms`;
  }
  const total = detection.processingMs ?? timing.totalMs;
  section.dataset.summary = `Erkennung ${fixed(total, 1)} ms`;
  text('patchProfileNote', detection.accelerator === 'WebGPU-Hybrid' ?
    `${detection.accelerator} | Frame ${currentIndex} | ${detection.points.length} Punkte | Arbeitsaufloesung ${fixed(timing.scale * 100, 0)}% je Achse` :
    `${detection.accelerator || 'CPU'} | Frame ${currentIndex} | ${detection.points.length} Patches | ${detection.patchSize} px`);
  updateTimeProfiles();
}

function parameters() {
  const width = videoInfo?.width ?? calibration?.field.width;
  const height = videoInfo?.height ?? calibration?.field.height;
  const settings = { width, height, pattern: element('pattern').value, gridMm: number('gridMm', null), step: number('step', 0),
    approxStep: number('approxStep', 0), columns: number('columns', null), rows: number('rows', null),
    threshold: number('threshold', 16), lineRadius: number('lineRadius', 0), spacing: number('spacing', Math.round(Math.max(width || 640, height || 480) / 4)),
    patchSize: number('patchSize', 32), patchSearchRadius: number('patchSearchRadius', 16),
    lambda: number('lambda', 0.01), sigma: number('sigma', 1), delta: number('delta', 1.5), tau: number('tau', 0.12),
    iterations: number('iterations', 35), acceptance: number('acceptance', 1), useWebGpu: element('useWebGpu').checked, interval: 0,
    minMotion: number('minMotion', 0), updateEvery: number('updateEvery', 4), validationFrom: number('validationFrom', 80),
    startPercent: number('startPercent', 0), endPercent: number('endPercent', 100) };
  if (settings.pattern === 'patches') {
    settings.step = 1; settings.gridMm = null; settings.approxStep = 0; settings.columns = null; settings.rows = null; settings.lineRadius = 0;
    if (width && height) ensurePatchMask(width, height);
    if (patchMask) settings.patchMask = { ...patchMask, data: patchMask.data.slice() };
  }
  if (!(settings.spacing >= 16 && settings.minMotion >= 0 && settings.endPercent > settings.startPercent &&
    settings.sigma > 0 && settings.delta > 0 && settings.lambda >= 0 && settings.tau > 0 && settings.iterations >= 5 && settings.iterations <= 300 &&
    settings.threshold > 0 && settings.patchSize >= 16 && settings.patchSize <= 256 && settings.patchSearchRadius >= 4 && settings.patchSearchRadius <= 128 && settings.acceptance > 0 && settings.updateEvery >= 2 &&
    (settings.gridMm === null || settings.gridMm > 0))) throw new Error('Ungueltige Parameter. Positive Werte und gueltigen Zeitbereich verwenden.');
  if (Math.ceil(width / settings.spacing) * Math.ceil(height / settings.spacing) > 700) throw new Error('Kontrollgitter zu fein fuer den lokalen Solver. Groebere Splineweite waehlen.');
  return settings;
}

function applyParameters(settings) {
  for (const [key, value] of Object.entries(settings || {})) {
    if (element(key) && typeof value !== 'object') {
      if (element(key).type === 'checkbox') element(key).checked = Boolean(value);
      else element(key).value = value ?? '';
    }
  }
  for (const key of ['gridMm', 'columns', 'rows']) if (settings?.[key] == null) element(key).value = '';
  patchMask = settings?.patchMask || null;
  maskImageRevision = -1;
  updatePatternControls();
}

function updatePatternControls() {
  const patches = element('pattern').value === 'patches';
  element('patchControls').hidden = !patches;
  element('gridGeometryControls').hidden = patches;
  element('lineRadiusControl').hidden = patches;
  if (patches) {
    element('step').value = '1';
    const width = videoInfo?.width ?? calibration?.field.width;
    const height = videoInfo?.height ?? calibration?.field.height;
    if (width && height) ensurePatchMask(width, height);
  }
}

function ensurePatchMask(width, height) {
  if (!patchMask || patchMask.sourceWidth !== width || patchMask.sourceHeight !== height) {
    patchMask = createPatchMask(width, height);
    maskImageRevision = -1;
  }
  return patchMask;
}

function revalidatePatchFrames(frameList, mask = patchMask) {
  let invalid = 0;
  for (const frame of frameList) {
    if (!frame.patchSize) continue;
    invalid += validatePointsAgainstMask(frame.points, mask, frame.patchSize);
  const trackingSection = element('trackingProfileSection');
    if (frame.points.length < 6) {
      frame.enabled = false;
      frame.accepted = false;
  if (!trackingSection.hidden) summaries.push(trackingSection.dataset.summary);
    }
  }
  return invalid;
}

function commitMaskEdit() {
  const invalid = revalidatePatchFrames(frames.values());
  parameterRevision++;
  stale = Boolean(calibration);
  detectionsStale = false;
  phaseChecked = [...frames.values()].some(frame => frame.accepted);
  detections.clear();
  currentDetection = detectionFromFrame(frames.get(currentIndex));
  updateTable(); updateMetrics(); updateControls(); draw();
  message(invalid ? `${invalid} Patch-Vektoren beruehrten die rote Maske und wurden verworfen. Nur neu fitten erforderlich.` :
    'Patch-Maske geaendert. Vorhandene Vektoren wurden direkt geprueft; nur neu fitten erforderlich.');
}

function updateCorrectionDataStatus() {
  for (const [id, label, present] of [
    ['lensDataStatus', 'Linsenkalibrierung', Boolean(calibration?.maps)],
    ['trackingDataStatus', 'XYR-Tracking', trackingPath.length > 0],
    ['brightnessDataStatus', 'Helligkeitskorrektur', false]
  ]) {
    const indicator = element(id);
    indicator.classList.toggle('present', present);
    const state = `${label}: ${present ? 'vorhanden' : 'nicht vorhanden'}`;
    indicator.title = state;
    indicator.setAttribute('aria-label', state);
  }
}

function trackingForExport() {
  if (!trackingPath.length) return null;
  return {
    ...trackingDataset,
    format: 'rasterlabor-xyr-tracking', model_version: 1,
    created_at: trackingDataset?.created_at ?? new Date().toISOString(),
    video: trackingDataset?.video ?? (videoInfo ? { name: videoInfo.name, width: videoInfo.width, height: videoInfo.height } : null),
    options: trackingRunOptions,
    coordinate_system: 'rectified pixels; x right; y down; rotation radians; camera pose relative to first tracked frame',
    path: trackingPath.map(({ patchPoints, ...entry }) => entry), failures: trackingFailures
  };
}

function restoreTracking(tracking) {
  if (!tracking?.path?.length) return;
  trackingImageMask = null;
  trackingMaskHasSelection = false;
  trackingDataset = tracking;
  trackingPath = tracking.path;
  trackingFailures = Array.isArray(tracking.failures) ? tracking.failures : [];
  trackingRunOptions = tracking.options ?? null;
    const storedMask = tracking.options?.sourceImageMask ?? tracking.options?.patchMask;
    const sourceWidth = videoInfo?.width ?? calibration?.field.width;
    const sourceHeight = videoInfo?.height ?? calibration?.field.height;
  if (storedMask?.data && Number.isInteger(storedMask.width) && Number.isInteger(storedMask.height) &&
      storedMask.width > 0 && storedMask.height > 0 && Number.isFinite(storedMask.cellSize) && storedMask.cellSize > 0 &&
      storedMask.data.length === storedMask.width * storedMask.height &&
      storedMask.sourceWidth === sourceWidth && storedMask.sourceHeight === sourceHeight) {
    const data = Uint8Array.from(storedMask.data);
    if (data.every(value => value === MASK_NEUTRAL || value === MASK_SEARCH || value === MASK_FORBIDDEN)) {
      trackingImageMask = { ...storedMask, data };
      trackingMaskHasSelection = data.includes(MASK_SEARCH);
    }
  }
  // A stored path can be inspected/exported; decoder and tracker history are not stored.
  trackingPaused = false;
  renderTrackingResults(trackingPath.at(-1));
  updateTrackingControls();
  updateCorrectionDataStatus();
}

function updateControls() {
  checkerboardView.refresh();
  updateCorrectionDataStatus();
  const hasVideo = Boolean(videoInfo);
  for (const id of ['firstFrame', 'previousFrame', 'nextFrame', 'timeline']) element(id).disabled = !hasVideo || navigationBusy || rectifiedPlayback || trackingRunning;
  element('playButton').disabled = !hasVideo || trackingRunning || (navigationBusy && !rectifiedPlayback);
  element('detectButton').disabled = !hasVideo || taskBusy || continuous || rectifiedPlayback || trackingRunning;
  element('startButton').disabled = !hasVideo || !phaseChecked || taskBusy || phaseStarted || rectifiedPlayback || trackingRunning;
  element('pauseButton').disabled = !continuous && !revalidating;
  element('pauseButton').querySelector('span').textContent = revalidating ? 'Abbrechen' : 'Pause';
  element('resumeButton').disabled = !hasVideo || !phaseStarted || continuous || taskBusy || nextProcessingIndex >= (videoInfo?.frameCount || 0);
  element('evaluateButton').disabled = !hasVideo || !phaseChecked || taskBusy || continuous || rectifiedPlayback || trackingRunning;
  element('evaluateNextButton').disabled = !hasVideo || !phaseChecked || taskBusy || continuous || rectifiedPlayback || trackingRunning || currentIndex >= (videoInfo?.frameCount || 0) - 1;
  element('refitButton').disabled = taskBusy || continuous || rectifiedPlayback || trackingRunning || [...frames.values()].filter(frame => frame.enabled && frame.role === 'train').length < 2 || (detectionsStale && !hasVideo);
  element('exportButton').disabled = !calibration || taskBusy || continuous || rectifiedPlayback || trackingRunning;
  element('correctedVideoButton').disabled = !calibration || !hasVideo || taskBusy || continuous || rectifiedPlayback || trackingRunning;
  element('importButton').disabled = taskBusy || continuous || rectifiedPlayback || trackingRunning;
  element('openVideo').disabled = taskBusy || continuous || rectifiedPlayback || trackingRunning;
  element('resetButton').disabled = taskBusy || continuous || rectifiedPlayback || trackingRunning;
  for (const control of document.querySelectorAll('.settings input, .settings select, .settings textarea')) {
    if (control.id !== 'follow' && control.id !== 'opticalConfiguration') control.disabled = taskBusy || continuous || trackingRunning;
  }
  for (const control of element('frameTable').querySelectorAll('input, select')) {
    control.disabled = taskBusy || continuous || (control.type === 'checkbox' && !frames.get(Number(control.dataset.frameId))?.accepted);
  }
  const label = calibration ? `v${calibration.version}${stale ? ' / veraltet' : ''}` : 'Kein Feld';
  text('fieldVersion', label);
  element('fieldVersion').className = `tag ${stale ? 'stale' : calibration ? 'good' : ''}`;
  text('phaseOneState', phaseChecked ? 'Geprueft' : 'Ungeprueft');
  element('phaseOneState').className = `tag ${phaseChecked ? 'good' : ''}`;
  text('resultState', calibration ? `Feld v${calibration.version} | ${stale ? 'veraltet' : calibration.quality === 'validated' ? 'validiert' : 'vorlaeufig'}` : 'Keine Kalibrierung');
  updateTrackingControls();
}

async function task(operation) {
  if (taskBusy) return;
  taskBusy = true;
  updateControls();
  try { await operation(); }
  catch (error) { message(error.message, true); continuous = false; }
  finally { taskBusy = false; updateControls(); if (!continuous) text('processingStatus', 'Bereit'); }
}

function closestFrame(timestamp) {
  let low = 0;
  let high = videoInfo.timestamps.length - 1;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (videoInfo.timestamps[middle] < timestamp) low = middle + 1;
    else high = middle;
  }
  return low > 0 && Math.abs(videoInfo.timestamps[low - 1] - timestamp) < Math.abs(videoInfo.timestamps[low] - timestamp) ? low - 1 : low;
}

function timeLabel(seconds) {
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, '0')}:${(seconds % 60).toFixed(3).padStart(6, '0')}`;
}

function updateTime() {
  if (!videoInfo) return;
  const timestamp = videoInfo.timestamps[currentIndex];
  text('frameTime', `Frame ${currentIndex} / PTS ${(timestamp / 1e6).toFixed(6)} s`);
  const sharpness = Number.isFinite(currentSharpness?.score) ? ` | Schaerfe ${fixed(currentSharpness.score, 2)}` : '';
  text('rawState', `${video.paused ? 'Framegenau' : 'Wiedergabe'} | #${currentIndex}${sharpness}`);
  element('timeline').value = currentIndex;
  text('elapsed', timeLabel((timestamp - videoInfo.firstTimestamp) / 1e6));
}

function renderVideoMetadata() {
  if (!videoInfo) return;
  const color = videoInfo.color;
  const hdr = color?.transfer === 'smpteSt2084' || color?.transfer === 'hlg';
  const colorLabel = color?.transfer ? ` | ${color.primaries || 'Farbraum'} / ${color.transfer}${hdr ? ' (HDR -> SDR-Canvas)' : ''}` : '';
  text('videoMetadata', `${videoInfo.width} x ${videoInfo.height} | ${videoInfo.duration.toFixed(2)} s | ${videoInfo.frameCount} Frames | ${fixed(videoInfo.fps, 3)} fps${videoInfo.variableFrameRate ? ' (variabel, Mittel)' : ''} | ${videoInfo.codec}${colorLabel}`);
}


function setRaw(bitmap, index, sharpness = null) {
  rawImage.width = bitmap.width;
  rawImage.height = bitmap.height;
  rawImage.getContext('2d').drawImage(bitmap, 0, 0);
  rawReady = true;
  if (element('pattern').value === 'patches') ensurePatchMask(bitmap.width, bitmap.height);
  currentIndex = index;
  currentSharpness = sharpness;
  currentDetection = detections.get(index) || detectionFromFrame(frames.get(index));
  element('rawEmpty').hidden = true;
  updateTime();
  draw();
}

function detectionFromFrame(frame) {
  if (!frame) return null;
  const lines = [];
  for (const family of ['row', 'col']) {
    const groups = new Map();
    for (const point of frame.points) {
      if (!groups.has(point[family])) groups.set(point[family], []);
      groups.get(point[family]).push(point);
    }
    for (const [index, points] of groups) lines.push({ family, index, points: points.sort((first, second) => first[family === 'row' ? 'col' : 'row'] - second[family === 'row' ? 'col' : 'row']) });
  }
  return { points: frame.points, lines: frame.patchSize ? [] : lines, rejected: [], roi: frame.roi, patchSize: frame.patchSize,
    success: frame.accepted, reason: frame.reason, accelerator: frame.accelerator,
    sharpness: frame.sharpness, timing: frame.timing };
}

async function showFrame(index) {
  if (!videoInfo || navigationBusy) return;
  navigationBusy = true;
  video.pause();
  updatePlayIcon();
  updateControls();
  try {
    const result = await readFrame(Math.max(0, Math.min(videoInfo.frameCount - 1, index)));
    setRaw(result.bitmap, result.index, result.sharpness);
    result.bitmap.close();
    rectifiedReady = false;
    if (view === 'rectified' && calibration && !taskBusy) await task(updateRectified);
  } finally { navigationBusy = false; updateControls(); draw(); }
}

async function playRectified() {
  if (!videoInfo || !calibration || rectifiedPlayback) return;
  video.pause();
  rectifiedPlayback = true;
  updatePlayIcon();
  updateControls();
  message();
  let index = currentIndex >= videoInfo.frameCount - 1 ? 0 : currentIndex;
  try {
    while (rectifiedPlayback && index < videoInfo.frameCount) {
      const started = performance.now();
      navigationBusy = true;
      updateControls();
      const result = await readFrame(index);
      setRaw(result.bitmap, result.index, result.sharpness);
      result.bitmap.close();
      await updateRectified();
      navigationBusy = false;
      text('processingStatus', `Entzerrte Wiedergabe | Frame ${index + 1}/${videoInfo.frameCount}`);
      element('progress').value = (index + 1) / videoInfo.frameCount;
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      if (!rectifiedPlayback) break;
      const nextIndex = index + 1;
      if (nextIndex >= videoInfo.frameCount) break;
      const frameDuration = (videoInfo.timestamps[nextIndex] - videoInfo.timestamps[index]) / 1000 / Number(element('speed').value);
      const remaining = frameDuration - (performance.now() - started);
      if (remaining > 0) await new Promise(resolve => setTimeout(resolve, remaining));
      index = nextIndex;
    }
  } catch (error) {
    message(`Entzerrte Wiedergabe abgebrochen: ${error.message}`, true);
  } finally {
    rectifiedPlayback = false;
    navigationBusy = false;
    updatePlayIcon();
    updateControls();
    text('processingStatus', 'Bereit');
    text('pendingStatus', currentIndex >= videoInfo.frameCount - 1 ? 'Videoende' : `Pausiert bei Frame ${currentIndex}`);
  }
}

async function detectIndex(index, display = true) {
  const revision = parameterRevision;
  const result = await readFrame(index);
  if (display) setRaw(result.bitmap, index, result.sharpness);
  const settings = parameters();
  let detection;
  if (settings.pattern === 'chessboard' && settings.useWebGpu) {
    detection = await computer.call('detect-bitmap', { bitmap: result.bitmap, index, options: settings }, [result.bitmap]);
  } else {
    const image = frameReader.toRgba(result.bitmap);
    result.bitmap.close();
    detection = await computer.call('detect', { image, index, options: settings }, [image.data.buffer]);
  }
  if (revision !== parameterRevision) throw new Error('Parameter wurden waehrend der Erkennung geaendert.');
  detection.sharpness = result.sharpness;
  detections.set(index, detection);
  while (detections.size > 30) detections.delete(detections.keys().next().value);
  if (currentIndex === index) currentDetection = detection;
  detectedOnce = true;
  if (detection.success) {
    if (detection.patchSize) element('step').value = '1';
    else {
      if (!number('step', 0)) element('step').value = detection.step.toFixed(4);
      if (!number('approxStep', 0)) element('approxStep').value = detection.step.toFixed(2);
    }
    if (!number('spacing', 0)) element('spacing').value = Math.round(Math.max(videoInfo.width, videoInfo.height) / 4);
    phaseChecked = true;
  } else {
    if (!phaseStarted) phaseChecked = false;
  }
  updateMetrics();
  draw();
  return detection;
}

function movementDistance(first, second) {
  if (!first.length || !second.length) return Infinity;
  const distances = first.map(point => Math.min(...second.map(other => Math.hypot(point.x - other.x, point.y - other.y))));
  distances.sort((left, right) => left - right);
  return distances[Math.floor(distances.length / 2)];
}

async function evaluateIndex(index, display = true, force = false, status = '', refreshUi = true) {
  text('processingStatus', status || `Erkennung Frame ${index} | PTS ${(videoInfo.timestamps[index] / 1e6).toFixed(3)} s`);
  const settings = parameters();
  const previous = frames.get(index);
  const retryMotionRejection = settings.minMotion === 0 && previous && !previous.accepted && previous.points.length > 0;
  const retryInterruptedPatch = previous && !previous.accepted && previous.reason.startsWith('Image-Patch-Sequenz');
  if (frames.has(index) && !detectionsStale && !force && !retryMotionRejection && !retryInterruptedPatch) {
    if (display) {
      const result = await readFrame(index);
      setRaw(result.bitmap, index, result.sharpness);
      result.bitmap.close();
    }
    return false;
  }
  const detection = await detectIndex(index, display);
  const elapsed = (videoInfo.timestamps[index] - videoInfo.firstTimestamp) / 1e6;
  const role = elapsed >= videoInfo.duration * settings.validationFrom / 100 ? 'validation' : 'train';
  let reason = detection.success ? '' : detection.reason;
  if (detection.success && !force && settings.minMotion > 0) {
    const similar = [...frames.values()].filter(frame => frame.enabled && frame.role === role);
    if (similar.some(frame => movementDistance(detection.points, frame.points) < settings.minMotion)) reason = 'Zu aehnliche Musterposition (Mindestbewegung).';
  }
  const oldFrame = frames.get(index);
  const frame = { id: index, timestamp: videoInfo.timestamps[index], role: oldFrame?.role || role,
    enabled: !reason, accepted: !reason, points: detection.success ? detection.points : [], roi: settings.roi || null,
    patchSize: detection.patchSize || null, confidence: detection.confidence, accelerator: detection.accelerator,
    sharpness: detection.sharpness, timing: detection.timing,
    reason: reason || (detection.patchSize ? 'Brauchbare Image-Patches' : 'Brauchbares Raster') };
  frames.set(index, frame);
  geometrySelection = null;
  observationDiagnosticsDirty = true;
  if (!reason) acceptedSinceFit++;
  stale = Boolean(calibration);
  if (refreshUi) {
    updateTable();
    updateMetrics();
    updateControls();
  }
  return !reason;
}

async function refit() {
  if (detectionsStale) {
    const previous = [...frames.values()];
    revalidating = true;
    updateControls();
    try {
      for (let position = 0; position < previous.length; position++) {
        if (cancelRequested) {
          text('pendingStatus', `Revalidierung abgebrochen nach ${position}/${previous.length} Messframes`);
          updateTable(); updateMetrics();
          return;
        }
        const frame = previous[position];
        const status = `Messframes revalidieren ${position + 1}/${previous.length} | Frame ${frame.id}`;
        element('progress').value = position / previous.length;
        text('pendingStatus', `Noch ${previous.length - position} Messframes`);
        await evaluateIndex(frame.id, false, true, status, false);
        const replacement = frames.get(frame.id);
        replacement.enabled = frame.enabled && replacement.accepted;
        replacement.role = frame.role;
      }
      element('progress').value = 1;
      detectionsStale = false;
      updateTable(); updateMetrics();
    } finally {
      revalidating = false;
      updateControls();
    }
  }
  const settings = parameters();
  const fittingFrames = [...frames.values()].map(frame => ({ ...frame, points: frame.points.map(point => ({ ...point })) }));
  if (fittingFrames.filter(frame => frame.enabled && frame.role === 'train').length < 2) return;
  const result = await computer.call('fit', { frames: fittingFrames, options: settings, fresh: stale && Boolean(snapshotParameters &&
    (snapshotParameters.step !== settings.step || snapshotParameters.spacing !== settings.spacing)) });
  calibration = result;
  snapshotFrames = fittingFrames;
  snapshotParameters = settings;
  acceptedSinceFit = 0;
  stale = false;
  rectifiedReady = false;
  renderFieldImages();
  updateMetrics();
  updateControls();
  if (view === 'rectified' && rawReady) await updateRectified();
  draw();
  if (videoInfo) setTimeout(() => { void loadTrackingMaskPreview(); }, 0);
}

async function runContinuous() {
  if (continuous || taskBusy) return;
  cancelRequested = false;
  continuous = true;
  phaseStarted = true;
  video.pause();
  updatePlayIcon();
  await task(async () => {
    const settings = parameters();
    const endTime = videoInfo.firstTimestamp + videoInfo.duration * 1e6 * settings.endPercent / 100;
    const startTime = videoInfo.firstTimestamp + videoInfo.duration * 1e6 * settings.startPercent / 100;
    const endIndex = videoInfo.timestamps.findLastIndex(timestamp => timestamp <= endTime);
    nextProcessingIndex = Math.max(nextProcessingIndex, closestFrame(startTime));
    while (continuous && nextProcessingIndex < videoInfo.frameCount && videoInfo.timestamps[nextProcessingIndex] <= endTime) {
      const index = nextProcessingIndex;
      const remaining = Math.max(0, endIndex - index);
      text('pendingStatus', `PTS ${(videoInfo.timestamps[index] / 1e6).toFixed(3)} s | noch ${remaining} Frames`);
      await evaluateIndex(index, element('follow').checked);
      nextProcessingIndex = index + 1;
      if (element('follow').checked && !document.hidden) {
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      }
      if (acceptedSinceFit >= settings.updateEvery) await refit();
    }
    if (acceptedSinceFit && [...frames.values()].filter(frame => frame.enabled && frame.role === 'train').length >= 2) await refit();
    continuous = false;
    text('pendingStatus', nextProcessingIndex >= videoInfo.frameCount || videoInfo.timestamps[nextProcessingIndex] > endTime ? 'Abschnitt abgeschlossen' : `Pausiert vor Frame ${nextProcessingIndex}`);
  });
}

function updateMetrics() {
  const all = [...frames.values()];
  text('trainCount', all.filter(frame => frame.enabled && frame.role === 'train').length);
  text('validationCount', all.filter(frame => frame.enabled && frame.role === 'validation').length);
  text('pointCount', all.filter(frame => frame.enabled).reduce((sum, frame) => sum + frame.points.length, 0));
  text('observationSummary', `${all.filter(frame => !frame.enabled).length} verworfen / deaktiviert`);
  text('trainRms', calibration ? `${fixed(calibration.metrics.training.rms)} px` : '-');
  text('validationP95', calibration?.metrics.validation.count ? `${fixed(calibration.metrics.validation.p95)} px` : '-');
  text('validCoverage', calibration ? `${fixed(calibration.maps.roundtrip.validFraction * 100, 0)}%` : '-');
  const step = calibration?.step || number('step', 0);
  const mm = calibration && !stale ? snapshotParameters?.gridMm : number('gridMm', null);
  text('metricScale', mm > 0 && step > 0 ? `${(mm / step).toPrecision(5)} mm/px | ${(25.4 * step / mm).toFixed(1)} DPI` : 'Massstab in Pixeln / Rastereinheiten');
  if (calibration) text('qualityDetails', JSON.stringify({ quality: calibration.quality, version: calibration.version,
    stale, training: { ...calibration.metrics.training, spatial: undefined }, validation: { ...calibration.metrics.validation, spatial: undefined },
    geometry: calibration.metrics.geometry, roundtrip: calibration.maps.roundtrip,
    iterations: calibration.metrics.completedIterations, finalCoefficientChange: calibration.metrics.finalChange,
    video: expectedVideo ? { ...expectedVideo, timestamps: undefined } : null }, null, 2));
  updateMemoryStats();
}

function updateTable() {
  const tbody = element('frameTable');
  tbody.replaceChildren();
  const visibleFrames = [...frames.values()].filter(frame => view !== 'data' || frame.enabled);
  for (const frame of visibleFrames.sort((first, second) => Number(second.enabled) - Number(first.enabled) || first.id - second.id)) {
    const row = document.createElement('tr');
    const activeCell = row.insertCell();
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox'; checkbox.checked = frame.enabled;
    checkbox.dataset.frameId = frame.id;
    checkbox.disabled = !frame.accepted || taskBusy || continuous;
    checkbox.setAttribute('aria-label', `Frame ${frame.id} aktiv`);
    checkbox.onchange = () => { frame.enabled = checkbox.checked; geometrySelection = null; stale = Boolean(calibration); observationDiagnosticsDirty = true; updateMetrics(); updateControls(); void task(refit); };
    activeCell.append(checkbox);
    const frameButton = document.createElement('button');
    frameButton.textContent = `#${frame.id} / ${(frame.timestamp / 1e6).toFixed(3)} s`;
    frameButton.disabled = !videoInfo;
    frameButton.onclick = () => void showFrame(frame.id).catch(error => message(error.message, true));
    row.insertCell().append(frameButton);
    const role = document.createElement('select');
    role.setAttribute('aria-label', `Rolle Frame ${frame.id}`);
    for (const [value, label] of [['train', 'Training'], ['validation', 'Validierung']]) role.add(new Option(label, value));
    role.value = frame.role; role.disabled = taskBusy || continuous;
    role.onchange = () => { frame.role = role.value; stale = Boolean(calibration); updateMetrics(); updateControls(); void task(refit); };
    row.insertCell().append(role);
    row.insertCell().textContent = frame.points.length;
    const geometry = geometrySelection?.candidates.get(frame.id);
    const sharpness = Number.isFinite(frame.sharpness?.score) ? ` | Schaerfe ${fixed(frame.sharpness.score, 2)}` : '';
    row.insertCell().textContent = (geometry ? `${geometry.selected ? 'Geometrie passend' : 'Geometrie abweichend'} | Skala ${fixed(geometry.scale, 4)} | Anisotropie ${fixed(geometry.anisotropy, 4)}` : frame.enabled && calibration && !stale ?
      (frame.role === 'train' ? 'Im Fit verwendet' : 'Zur Validierung verwendet') : frame.reason) + sharpness;
    tbody.append(row);
  }
  if (!visibleFrames.length) { const row = tbody.insertRow(); const cell = row.insertCell(); cell.colSpan = 5; cell.textContent = view === 'data' ? 'Keine selektierten Messframes' : 'Keine Messframes'; cell.className = 'muted'; }
}

function renderObservationDiagnostics(force = false) {
  const width = videoInfo?.width ?? calibration?.field.width;
  const height = videoInfo?.height ?? calibration?.field.height;
  if (!width || !height || !frames.size) { observationDiagnostics = null; return; }
  if (force || observationDiagnosticsDirty || !observationDiagnostics) {
    const maximum = 160;
    const scale = maximum / Math.max(width, height);
    const cols = Math.max(24, Math.ceil(width * scale));
    const rows = Math.max(18, Math.ceil(height * scale));
    observationDiagnostics = analyzeObservations(frames.values(), width, height, number('step', 1) || 1, cols, rows);
    observationDiagnosticsDirty = false;
  }
  const diagnostic = observationDiagnostics;
  const mode = element('dataMode').value;
  const pixels = new ImageData(diagnostic.cols, diagnostic.rows);
  if (mode === 'diversity') {
    const nonzero = [...diagnostic.frameCounts].filter(Boolean);
    const maximum = Math.max(1, ...nonzero);
    for (let index = 0; index < diagnostic.frameCounts.length; index++) {
      const count = diagnostic.frameCounts[index];
      if (!count) pixels.data.set([135, 142, 138, 255], index * 4);
      else {
        const ratio = Math.log1p(count) / Math.log1p(maximum);
        pixels.data.set([Math.round(235 - 210 * ratio), Math.round(180 - 35 * ratio), Math.round(60 + 105 * ratio), 255], index * 4);
      }
    }
  } else {
    const cellErrors = [];
    for (let index = 0; index < diagnostic.residualCounts.length; index++) {
      if (diagnostic.residualCounts[index]) cellErrors.push(diagnostic.residualSums[index] / diagnostic.residualCounts[index]);
    }
    cellErrors.sort((first, second) => first - second);
    const quantile = fraction => cellErrors[Math.min(cellErrors.length - 1, Math.round((cellErrors.length - 1) * fraction))] || 0;
    const low = quantile(0.05);
    const middle = quantile(0.5);
    const high = quantile(0.95);
    const span = Math.max(1e-6, high - low);
    for (let index = 0; index < diagnostic.residualCounts.length; index++) {
      const count = diagnostic.residualCounts[index];
      if (!count) pixels.data.set([135, 142, 138, 255], index * 4);
      else {
        const ratio = Math.max(0, Math.min(1, (diagnostic.residualSums[index] / count - low) / span));
        const firstHalf = Math.min(1, ratio * 2);
        const secondHalf = Math.max(0, ratio * 2 - 1);
        pixels.data.set([
          Math.round(28 + 214 * firstHalf - 25 * secondHalf),
          Math.round(137 + 61 * firstHalf - 143 * secondHalf),
          Math.round(132 - 72 * firstHalf + 5 * secondHalf), 255
        ], index * 4);
      }
    }
    diagnostic.residualScale = { low, middle, high };
  }
  observationImage.width = diagnostic.cols;
  observationImage.height = diagnostic.rows;
  observationImage.getContext('2d').putImageData(pixels, 0, 0);
  const metrics = diagnostic.metrics;
  text('otherLegend', mode === 'diversity' ?
    `Framevielfalt relativ | Grau: keine Daten | ${metrics.frames} Frames, ${metrics.points.toLocaleString('de-DE')} Punkte | nahezu identisch ${fixed(100 * metrics.nearDuplicateFraction, 0)}% | Bewegungsrichtungen ${metrics.directionBins}/8 | typischer Schritt ${fixed(metrics.movementMedian, 1)} px` :
    `Abweichung vom starren Frame-Modell | Tuerkis P05 ${fixed(diagnostic.residualScale.low, 2)} px | Gelb Median ${fixed(diagnostic.residualScale.middle, 2)} px | Rot P95 ${fixed(diagnostic.residualScale.high, 2)} px | Frame-RMS Median ${fixed(metrics.rigidMedian, 2)}, P95 ${fixed(metrics.rigidP95, 2)} px | mit Skalierung Median ${fixed(metrics.similarityMedian, 2)} px | Skalierung P05-P95 ${fixed(metrics.scaleP05, 5)}-${fixed(metrics.scaleP95, 5)}`);
}

function renderCoverageImage() {
  if (!calibration) return;
  const { field, maps } = calibration;
  const relative = element('relativeCoverage').checked;
  const scale = relative ? relativeCoverageScale(maps.sourceCoverage) : null;
  const pixels = new ImageData(field.width, field.height);
  for (let index = 0; index < maps.sourceCoverage.length; index++) {
    const count = maps.sourceCoverage[index];
    const hatch = (index % field.width + Math.floor(index / field.width)) % 12 < 3;
    pixels.data.set(relative ? scale.colors[count] : count >= 3 ? [25, Math.min(190, 100 + count * 8), 122, 255] :
      count ? [203, 155, 59, hatch ? 240 : 145] : [135, 142, 138, hatch ? 180 : 70], index * 4);
  }
  coverageImage.width = field.width;
  coverageImage.height = field.height;
  coverageImage.getContext('2d').putImageData(pixels, 0, 0);
  const maximum = scale?.maximum === 255 ? '255+' : scale?.maximum;
  element('coverageLegend').textContent = relative ? !scale.maximum ? 'Grau: unbeobachtet' :
    scale.minimum === scale.maximum ? `Einheitlich ${maximum} Messframes | Grau: unbeobachtet` :
      `Relativ: Gelb ${scale.minimum} bis Tuerkis ${maximum} Messframes | Grau: unbeobachtet | Keine Fehlerbewertung` :
    'Gruen: >=3 Messframes | Gelb: 1-2 | Grau: unbeobachtet';
}

function renderFieldImages() {
  if (!calibration) return;
  renderFitProfile(calibration.metrics.fitProfile);
  const { field, maps } = calibration;
  const range = Math.max(0.1, number('colorRange', 100));
  let rigid = null;
  if (element('hideRigid').checked) {
    const points = [];
    for (let row = 0; row < 8; row++) for (let col = 0; col < 8; col++) points.push({
      x: col / 7 * (field.width - 1), y: row / 7 * (field.height - 1), col: col / 7 * (field.width - 1), row: row / 7 * (field.height - 1), confidence: 1
    });
    rigid = poseFor(points, field, 1);
  }
  const fieldPixels = new ImageData(field.width, field.height);
  let clipped = 0;
  for (let index = 0; index < field.width * field.height; index++) {
    const px = index % field.width;
    const py = Math.floor(index / field.width);
    let ux = maps.forward[index * 2];
    let uy = maps.forward[index * 2 + 1];
    if (rigid) {
      const shiftedX = ux - rigid.tx;
      const shiftedY = uy - rigid.ty;
      ux = Math.cos(rigid.theta) * shiftedX + Math.sin(rigid.theta) * shiftedY;
      uy = -Math.sin(rigid.theta) * shiftedX + Math.cos(rigid.theta) * shiftedY;
    }
    const dx = ux - px;
    const dy = uy - py;
    if (Math.abs(dx) > range || Math.abs(dy) > range) clipped++;
    const rgb = fieldColor(dx, dy, range);
    const covered = maps.sourceCoverage[index] >= 3;
    const hatch = !covered && (px + py) % 12 < 3;
    fieldPixels.data.set([rgb[0], hatch ? 85 : 0, rgb[2], covered ? 255 : hatch ? 220 : 95], index * 4);
  }
  for (const [canvas, pixels] of [[fieldImage, fieldPixels]]) {
    canvas.width = pixels.width; canvas.height = pixels.height; canvas.getContext('2d').putImageData(pixels, 0, 0);
  }
  renderCoverageImage();
  residualImage.width = field.width; residualImage.height = field.height;
  const context = residualImage.getContext('2d');
  context.fillStyle = 'rgb(105, 112, 109)';
  context.fillRect(0, 0, field.width, field.height);
  const points = calibration.metrics.validation.spatial.length ? calibration.metrics.validation.spatial : calibration.metrics.training.spatial;
  const cols = maps.coverageGrid?.cols ?? 24;
  const rows = maps.coverageGrid?.rows ?? 18;
  const bins = new Map();
  for (const point of points) {
    const col = Math.min(cols - 1, Math.max(0, Math.floor(point.x / field.width * cols)));
    const row = Math.min(rows - 1, Math.max(0, Math.floor(point.y / field.height * rows)));
    const key = row * cols + col;
    const entry = bins.get(key) || { sum: 0, count: 0, col, row };
    entry.sum += point.error; entry.count++; bins.set(key, entry);
  }
  const errorScale = snapshotParameters?.acceptance || 1;
  for (const entry of bins.values()) {
    const ratio = Math.min(1, entry.sum / entry.count / errorScale);
    context.fillStyle = `rgb(${Math.round(35 + ratio * 195)}, ${Math.round(174 - ratio * 110)}, ${Math.round(124 - ratio * 55)})`;
    context.fillRect(entry.col / cols * field.width, entry.row / rows * field.height,
      field.width / cols + 1, field.height / rows + 1);
  }
  text('clipping', `Clipping ${(100 * clipped / (field.width * field.height)).toFixed(1)}% | Schraffur: <3 Messframes`);
  field3d.update(calibration, range, number('height3d', 1), rigid);
}

async function updateRectified() {
  if (!calibration || !rawReady || !video.paused) return;
  const revision = displayRevision;
  const frameIndex = currentIndex;
  const result = await readFrame(frameIndex, { rectified: true });
  try {
    if (frameIndex !== currentIndex || revision !== displayRevision) return;
    rectifiedImage.width = result.bitmap.width; rectifiedImage.height = result.bitmap.height;
    rectifiedImage.getContext('2d').drawImage(result.bitmap, 0, 0);
  } finally { result.bitmap.close(); }
  rectifiedReady = true;
  draw();
}

function drawCanvas(canvas, image, width, height, overlay = null, adjustImage = false) {
  const bounds = canvas.getBoundingClientRect();
  const pixelRatio = Math.min(devicePixelRatio || 1, 2);
  const screenWidth = Math.round(bounds.width * pixelRatio);
  const screenHeight = Math.round(bounds.height * pixelRatio);
  if (canvas.width !== screenWidth || canvas.height !== screenHeight) { canvas.width = screenWidth; canvas.height = screenHeight; }
  const context = canvas.getContext('2d');
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!width || !height) return;
  const scale = Math.min(bounds.width / width, bounds.height / height) * zoom;
  const left = (bounds.width - width * scale) / 2 + pan.x * bounds.width;
  const top = (bounds.height - height * scale) / 2 + pan.y * bounds.height;
  transforms.set(canvas, { scale, left, top, width, height });
  context.setTransform(scale * pixelRatio, 0, 0, scale * pixelRatio, left * pixelRatio, top * pixelRatio);
  context.imageSmoothingEnabled = zoom < 2;
  if (image) {
    if (adjustImage) drawAdjustedImage(context, image, 0, 0, width, height);
    else context.drawImage(image, 0, 0, width, height);
  }
  if (overlay) overlay(context, scale);
}

function drawPatchMask(context) {
  if (!patchMask || element('pattern').value !== 'patches') return;
  if (maskImageRevision !== patchMask.revision) {
    maskImage.width = patchMask.width;
    maskImage.height = patchMask.height;
    const pixels = maskImage.getContext('2d').createImageData(patchMask.width, patchMask.height);
    for (let index = 0; index < patchMask.data.length; index++) {
      const offset = 4 * index;
      if (patchMask.data[index] === MASK_SEARCH) pixels.data.set([20, 184, 110, 82], offset);
      else if (patchMask.data[index] === MASK_FORBIDDEN) pixels.data.set([223, 53, 69, 105], offset);
    }
    maskImage.getContext('2d').putImageData(pixels, 0, 0);
    maskImageRevision = patchMask.revision;
  }
  context.save();
  context.imageSmoothingEnabled = false;
  context.drawImage(maskImage, 0, 0, patchMask.sourceWidth, patchMask.sourceHeight);
  context.restore();
}

function drawOverlay(context, scale) {
  drawPatchMask(context);
  if (!element('overlay').checked || !currentDetection) return;
  const detection = currentDetection;
  if (detection.patchSize) {
    context.lineWidth = 1 / scale;
    for (const point of detection.points || []) {
      const size = detection.patchSize;
      if (!patchAllowed(patchMask, point.x, point.y, size)) continue;
      context.strokeStyle = `rgba(39, 234, 182, ${0.35 + 0.65 * point.confidence})`;
      context.strokeRect(point.x - size / 2, point.y - size / 2, size, size);
      context.fillStyle = '#ffd047';
      context.fillRect(point.x - 1.5 / scale, point.y - 1.5 / scale, 3 / scale, 3 / scale);
    }
    return;
  }
  context.lineWidth = 1 / scale;
  for (const line of detection.lines || []) {
    context.strokeStyle = line.family === 'col' ? '#27eab6' : '#ffd047';
    context.beginPath();
    line.points.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
    context.stroke();
  }
  context.font = `${10 / scale}px 'IBM Plex Sans'`;
  for (const point of detection.points || []) {
    context.fillStyle = '#f9fffd'; context.beginPath(); context.arc(point.x, point.y, 2.4 / scale, 0, Math.PI * 2); context.fill();
    context.fillStyle = '#003e31'; context.fillRect(point.x + 4 / scale, point.y - 12 / scale, 33 / scale, 11 / scale);
    context.fillStyle = '#ffffff'; context.fillText(`${point.col},${point.row}`, point.x + 5 / scale, point.y - 3 / scale);
  }
  context.strokeStyle = '#ff334b';
  for (const point of detection.rejected || []) {
    context.beginPath(); context.moveTo(point.x - 4 / scale, point.y - 4 / scale); context.lineTo(point.x + 4 / scale, point.y + 4 / scale);
    context.moveTo(point.x + 4 / scale, point.y - 4 / scale); context.lineTo(point.x - 4 / scale, point.y + 4 / scale); context.stroke();
  }
}

function draw() {
  checkerboardView.refresh();
  const detection = currentDetection;
  if (!rawReady) text('detectionStatus', 'Kein Frame geladen.');
  else if (!detection) text('detectionStatus', `Frame #${currentIndex}: noch nicht geprueft.`);
  else if (detectionsStale && !detections.has(currentIndex)) text('detectionStatus', `Frame #${currentIndex}: gespeicherte Erkennung veraltet.`);
  else if (!detection.success) text('detectionStatus', detection.reason);
  else if (detection.patchSize) text('detectionStatus', `${detection.points.length} zweidimensionale Patches | Groesse ${detection.patchSize} px | Abdeckung ${(100 * detection.coverage).toFixed(0)}%` +
    (detection.accelerator ? ` | ${detection.accelerator}${Number.isFinite(detection.processingMs) ? ` ${fixed(detection.processingMs, 1)} ms` : ''}` : ''));
  else {
    const cols = detection.cols ?? new Set(detection.points.map(point => point.col)).size;
    const rows = detection.rows ?? new Set(detection.points.map(point => point.row)).size;
    const details = Number.isFinite(detection.coverage) && Number.isFinite(detection.confidence) ?
      ` | Abdeckung ${(100 * detection.coverage).toFixed(0)}% | Sicherheit ${(100 * detection.confidence).toFixed(0)}%` : ' | gespeicherte Beobachtung';
    const accelerator = detection.accelerator ? ` | ${detection.accelerator}${Number.isFinite(detection.processingMs) ? ` ${fixed(detection.processingMs, 1)} ms` : ''}` : '';
    text('detectionStatus', `${detection.points.length} Punkte | ${cols} x ${rows}${details}${accelerator}`);
  }
  renderPatchProfile(detection);
  if (rawReady) drawCanvas(rawCanvas, video.paused ? rawImage : video, rawImage.width, rawImage.height, drawOverlay, true);
  else drawCanvas(rawCanvas, null, 0, 0);
  element('rawEmpty').hidden = rawReady;
  const is3d = view === 'field3d';
  resultCanvas.hidden = is3d;
  field3dCanvas.hidden = !is3d;
  const image = view === 'field' ? fieldImage : view === 'coverage' ? coverageImage : view === 'residual' ? residualImage : view === 'data' ? observationImage : rectifiedImage;
  const showResult = view === 'data' ? Boolean(observationDiagnostics?.metrics.frames) : Boolean(calibration) && (view !== 'rectified' || rectifiedReady);
  element('resultEmpty').hidden = showResult;
  element('resultEmpty').querySelector('span').textContent = view === 'data' ? 'Keine Messdaten vorhanden' : calibration && view === 'rectified' ? videoInfo ? 'Vorschau fuer pausierten Frame' : 'Zugehoeriges Video nicht geladen' : 'Kein Feld berechnet';
  if (showResult && is3d) field3d.resize();
  else if (showResult) drawCanvas(resultCanvas, image, image.width, image.height, view === 'rectified' ? (context, scale) => {
    checkerboardView.draw(context, scale);
    context.strokeStyle = '#ffd74b'; context.fillStyle = '#ffd74b'; context.lineWidth = 1.5 / scale;
    context.beginPath(); measurements.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y)); context.stroke();
    for (const point of measurements) { context.beginPath(); context.arc(point.x, point.y, 3 / scale, 0, Math.PI * 2); context.fill(); }
  } : null, view === 'rectified');
  else drawCanvas(resultCanvas, null, 0, 0);
  text('zoomValue', `${Math.round(zoom * 100)}%`);
}

function updateTrackingControls() {
  element('trackingSearchRadius').max = windowTracking() ? '1024' : '128';
  const ready = Boolean(videoInfo && calibration && videoInfo.width === calibration.field.width && videoInfo.height === calibration.field.height && !taskBusy && !continuous && !rectifiedPlayback);
  element('trackingStartButton').disabled = !ready || trackingRunning || trackingPreviewBusy || (windowTracking() && !trackingRectangle);
  element('trackingPauseButton').disabled = !trackingRunning;
  element('trackingResumeButton').disabled = !ready || trackingRunning || trackingPreviewBusy || trackingLost || !trackingPaused || trackingNextIndex > number('trackingEnd', 0);
  element('trackingResetButton').disabled = trackingRunning || trackingPreviewBusy;
  element('trackingExportButton').disabled = !trackingPath.some(entry => entry.pose);
  for (const control of document.querySelectorAll('.tracking-settings input, .tracking-settings select')) control.disabled = trackingRunning || trackingPreviewBusy;
  element('trackingSelectWindow').disabled = trackingRunning || trackingPreviewBusy || !trackingPreviewImage.width;
  text('trackingState', trackingRunning ? 'Laeuft' : trackingPaused ? 'Pausiert' : trackingPath.length ? 'Erfasst' : 'Bereit');
  element('trackingState').className = `tag ${trackingRunning || trackingPath.length ? 'good' : ''}`;
  if (!videoInfo) text('trackingStatus', 'Video und Kalibrierung laden.');
  else if (!calibration) text('trackingStatus', 'Zuerst eine Linsenkalibrierung laden oder erstellen.');
  else if (videoInfo.width !== calibration.field.width || videoInfo.height !== calibration.field.height) text('trackingStatus', 'Videoaufloesung und Kalibrierung passen nicht zusammen.');
  else text('trackingStatus', trackingRunning ? 'Tracking laeuft.' : ready ? 'Video und Kalibrierung bereit.' : 'Andere Verarbeitung aktiv.');
}

function resetTrackingWorker() {
  trackingComputer.terminate();
  trackingComputer = new WorkerClient('/compute-worker.js', showProgress);
}

function clearTrackingResults(preserveRectangle = false) {
  trackingInspector.clear();
  overlayRequest++;
  pathHover = null; pathSelection = null; pathProject = null;
  pathRefitProposal = null; pathRefitBusy = false;
  clearPathOverlay();
  element('trackingOverlayPane').hidden = true;
  trackingPreviewRequest++;
  trackingRunning = false;
  trackingPaused = false;
  trackingNextIndex = number('trackingStart', 0);
  trackingPath = [];
  trackingFailures = [];
  trackingDataset = null;
  updateCorrectionDataStatus();
  trackingProfile = null;
  trackingRunOptions = null;
  trackingLost = false;
  selectingTrackingWindow = false;
  element('trackingSelectWindow').setAttribute('aria-pressed', 'false');
  if (!preserveRectangle) trackingRectangle = null;
  text('trackingWindowInfo', trackingRectangle ? `${trackingRectangle.width} x ${trackingRectangle.height} px` : 'Kein Fenster ausgewaehlt');
  text('trackingMatchQuality', 'Korrelation: -');
  element('trackingMatchQuality').title = '';
  trackingPreviewTransform = null;
  trackingPreviewPoints = [];
  trackingZoom = 1;
  trackingPan = { x: 0, y: 0 };
  selectedTrackingPatch = null;
  trackingPreviewImage.width = 0;
  trackingPreviewImage.height = 0;
  resetTrackingWorker();
  for (const id of ['trackingX', 'trackingY', 'trackingRotation', 'trackingRms']) text(id, '-');
  text('trackingPatchCount', '0');
  text('trackingFrameState', 'Frame -');
  text('trackingPreviewState', 'Noch nicht gestartet');
  text('trackingPathState', '0 Posen');
  text('trackingSummary', '0 Frames');
  renderTrackingProfile(null);
  element('trackingPreviewEmpty').hidden = false;
  element('trackingPathEmpty').hidden = false;
  const row = document.createElement('tr');
  const cell = row.insertCell(); cell.colSpan = 8; cell.className = 'muted'; cell.textContent = 'Noch keine Posen';
  element('trackingTable').replaceChildren(row);
  for (const id of ['trackingPreviewCanvas', 'trackingPathCanvas']) {
    const canvas = element(id); canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  }
  updateTrackingControls();
}

function ensureTrackingImageMask(width, height) {
  if (!trackingImageMask || trackingImageMask.sourceWidth !== width || trackingImageMask.sourceHeight !== height) {
    trackingImageMask = createPatchMask(width, height);
    trackingMaskHasSelection = false;
  }
  return trackingImageMask;
}

function updateTrackingMaskBrushLimit(width, height) {
  const control = element('trackingMaskBrush');
  const step = Math.max(1, Number(control.step) || 1);
  const maximum = Math.max(Number(control.min) || step, Math.floor(Math.min(width, height) * 0.5 / step) * step);
  control.max = String(maximum);
  if (Number(control.value) > maximum) control.value = String(maximum);
  text('trackingMaskBrushValue', `${control.value} px`);
}

function trackingPixelAllowed(x, y) {
  // An untouched mask is optional. As soon as one green cell exists it becomes
  // an inclusion mask and only green cells contribute to footprints/overlays.
  if (!trackingMaskHasSelection) return true;
  if (!trackingImageMask?.data) return true;
  const maps = calibration?.maps;
  if (!maps) return maskIncludes(trackingImageMask, x, y);
  const px = Math.floor(x), py = Math.floor(y);
  if (px < 0 || py < 0 || px >= maps.outputWidth || py >= maps.outputHeight) return false;
  const index = py * maps.outputWidth + px;
  return Boolean(maps.valid[index]) && maskIncludes(trackingImageMask, maps.inverseX[index], maps.inverseY[index]);
}

function updateTrackingMaskInfo(prefix = '') {
  text('trackingMaskInfo', `${prefix}${prefix ? ' | ' : ''}${trackingMaskHasSelection ?
    'nur gruene Bereiche werden genutzt' : 'keine Auswahl: gesamter gueltiger Bildbereich'}`);
}

function renderTrackingMask() {
  const canvas = element('trackingMaskCanvas');
  const { context, width, height } = trackingCanvasContext('trackingMaskCanvas');
  context.clearRect(0, 0, width, height);
  if (!trackingMaskPreview.width) { trackingMaskTransform = null; return; }
  const scale = Math.min(width / trackingMaskPreview.width, height / trackingMaskPreview.height);
  const left = (width - trackingMaskPreview.width * scale) / 2;
  const top = (height - trackingMaskPreview.height * scale) / 2;
  drawAdjustedImage(context, trackingMaskPreview, left, top, trackingMaskPreview.width * scale, trackingMaskPreview.height * scale);
  const mask = trackingImageMask;
  if (mask) {
    trackingMaskPaintLayer.width = mask.width; trackingMaskPaintLayer.height = mask.height;
    const pixels = trackingMaskPaintLayer.getContext('2d').createImageData(mask.width, mask.height);
    for (let index = 0; index < mask.data.length; index++) {
      if (mask.data[index] === MASK_SEARCH) pixels.data.set([20, 184, 110, 110], index * 4);
      else pixels.data.set([25, 34, 30, 135], index * 4);
    }
    trackingMaskPaintLayer.getContext('2d').putImageData(pixels, 0, 0);
    context.imageSmoothingEnabled = false;
    context.drawImage(trackingMaskPaintLayer, left, top, trackingMaskPreview.width * scale, trackingMaskPreview.height * scale);
  }
  trackingMaskTransform = { left, top, scale };
  canvas.style.cursor = trackingMaskTool === 'erase' ? 'cell' : 'crosshair';
}

async function loadTrackingMaskPreview() {
  const request = ++trackingMaskPreviewRequest;
  if (!videoInfo) { text('trackingMaskInfo', 'Zuerst ein Video laden.'); return; }
  const index = Math.max(0, Math.min(videoInfo.frameCount - 1, Math.round(number('trackingStart', 0))));
  text('trackingMaskInfo', `Lade Rohframe #${index}...`);
  try {
    const image = await readTrackingFrame(index, { rectified: false, output: 'rgba' });
    if (request !== trackingMaskPreviewRequest) return;
    updateTrackingMaskBrushLimit(image.width, image.height);
    trackingMaskPreview.width = image.width; trackingMaskPreview.height = image.height;
    trackingMaskPreview.getContext('2d').putImageData(new ImageData(image.data, image.width, image.height), 0, 0);
    ensureTrackingImageMask(image.width, image.height);
    updateTrackingMaskInfo(`Rohframe #${index}`);
    renderTrackingMask(); drawTrackingPath();
  } catch (error) { if (request === trackingMaskPreviewRequest) text('trackingMaskInfo', error.message); }
}

function trackingCanvasContext(id) {
  const canvas = element(id);
  const bounds = canvas.getBoundingClientRect();
  const ratio = Math.min(devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(bounds.width * ratio));
  const height = Math.max(1, Math.round(bounds.height * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width: bounds.width, height: bounds.height };
}

function renderTrackingPreview() {
  if (!trackingPreviewImage.width || !trackingPreviewImage.height) return;
  const { context, width, height } = trackingCanvasContext('trackingPreviewCanvas');
  context.clearRect(0, 0, width, height);
  const scale = Math.min(width / trackingPreviewImage.width, height / trackingPreviewImage.height) * trackingZoom;
  const left = (width - trackingPreviewImage.width * scale) / 2 + trackingPan.x * width;
  const top = (height - trackingPreviewImage.height * scale) / 2 + trackingPan.y * height;
  drawAdjustedImage(context, trackingPreviewImage, left, top, trackingPreviewImage.width * scale, trackingPreviewImage.height * scale);
  trackingPreviewTransform = { scale, left, top };
  element('trackingPreviewEmpty').hidden = true;
  context.fillStyle = '#19d6a0'; context.strokeStyle = '#063d34'; context.lineWidth = 1;
  for (const point of trackingPreviewPoints) {
    const px = left + point.x * scale;
    const py = top + point.y * scale;
    const selected = point === selectedTrackingPatch;
    context.beginPath(); context.arc(px, py, selected ? 5 : 2.5, 0, Math.PI * 2); context.fill(); context.stroke();
    if (selected) { context.strokeStyle = '#ffd74b'; context.lineWidth = 2; context.beginPath(); context.arc(px, py, 8, 0, Math.PI * 2); context.stroke(); context.strokeStyle = '#063d34'; context.lineWidth = 1; }
  }
  if (windowTracking() && trackingRectangle) {
    context.strokeStyle = '#ffd74b'; context.lineWidth = 2;
    context.strokeRect(left + trackingRectangle.x * scale, top + trackingRectangle.y * scale,
      trackingRectangle.width * scale, trackingRectangle.height * scale);
  }
  text('trackingZoomValue', `${Math.round(trackingZoom * 100)}%`);
}


function drawTrackingFrame(bitmap) {
  trackingPreviewImage.width = bitmap.width;
  trackingPreviewImage.height = bitmap.height;
  trackingPreviewImage.getContext('2d').drawImage(bitmap, 0, 0);
  trackingPreviewPoints = [];
  renderTrackingPreview();
}

function drawTrackingPatches(points) {
  trackingPreviewPoints = points;
  if (!points.includes(selectedTrackingPatch)) selectedTrackingPatch = null;
  renderTrackingPreview();
}

function renderTrackingPatchDetail() {
  const point = selectedTrackingPatch;
  if (!point || !trackingPreviewImage.width) return;
  const patchSize = number('trackingPatchSize', 64);
  const cropSize = Math.max(128, patchSize * 2);
  const canvas = element('trackingPatchCanvas');
  canvas.width = cropSize; canvas.height = cropSize;
  const context = canvas.getContext('2d');
  context.imageSmoothingEnabled = false;
  drawAdjustedImage(context, trackingPreviewImage, point.x - cropSize / 2, point.y - cropSize / 2, cropSize, cropSize, 0, 0, cropSize, cropSize);
  context.strokeStyle = '#ffd74b'; context.lineWidth = Math.max(1, cropSize / 192);
  context.strokeRect(cropSize / 2 - patchSize / 2, cropSize / 2 - patchSize / 2, patchSize, patchSize);
  context.strokeStyle = '#c6404a'; context.beginPath(); context.moveTo(cropSize / 2 - 7, cropSize / 2); context.lineTo(cropSize / 2 + 7, cropSize / 2); context.moveTo(cropSize / 2, cropSize / 2 - 7); context.lineTo(cropSize / 2, cropSize / 2 + 7); context.stroke();
  text('trackingPatchInfo', `Frame-Patch bei (${fixed(point.x, 1)}, ${fixed(point.y, 1)}) | gelb: ${patchSize} x ${patchSize} px`);
}

function selectTrackingPatch(point) {
  selectedTrackingPatch = point;
  renderTrackingPreview(); renderTrackingPatchDetail();
  element('trackingPatchDialog').showModal();
}

async function showTrackedFrame(entry) {
  if (trackingRunning || trackingPreviewBusy || taskBusy) return;
  trackingInspector.show(entry);
  const request = ++trackingPreviewRequest;
  trackingPreviewBusy = true; updateTrackingControls();
  try {
    const decoded = await readTrackingFrame(entry.frame, { rectified: entry.mode === 'window' });
    if (request !== trackingPreviewRequest) { decoded.bitmap.close(); return; }
    if (entry.mode === 'window') {
      trackingRectangle = entry.rectangle; drawTrackingFrame(decoded.bitmap); decoded.bitmap.close();
    } else { drawTrackingFrame(decoded.bitmap); decoded.bitmap.close(); drawTrackingPatches(entry.patchPoints ?? []); }
    renderTrackingResults(entry);
  } finally { trackingPreviewBusy = false; updateTrackingControls(); }
}

function installTrackingPreviewInteraction() {
  const canvas = element('trackingPreviewCanvas');
  let drag = null;
  const pointAt = event => {
    const transform = trackingPreviewTransform;
    if (!transform) return null;
    const bounds = canvas.getBoundingClientRect();
    const x = (event.clientX - bounds.left - transform.left) / transform.scale;
    const y = (event.clientY - bounds.top - transform.top) / transform.scale;
    return x >= 0 && y >= 0 && x < trackingPreviewImage.width && y < trackingPreviewImage.height ? { x, y } : null;
  };
  canvas.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    drag = { x: event.clientX, y: event.clientY, pan: { ...trackingPan }, moved: false,
      selection: selectingTrackingWindow && !trackingRunning && !trackingPreviewBusy ? pointAt(event) : null };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    if (!drag) return;
    if (drag.selection) {
      const point = pointAt(event);
      if (point) {
        const x = Math.floor(Math.min(point.x, drag.selection.x)); const y = Math.floor(Math.min(point.y, drag.selection.y));
        trackingRectangle = { x, y, width: Math.floor(Math.max(point.x, drag.selection.x)) - x, height: Math.floor(Math.max(point.y, drag.selection.y)) - y };
        renderTrackingPreview();
      }
      return;
    }
    const deltaX = event.clientX - drag.x;
    const deltaY = event.clientY - drag.y;
    if (Math.hypot(deltaX, deltaY) > 4) drag.moved = true;
    if (drag.moved) {
      const bounds = canvas.getBoundingClientRect();
      trackingPan = { x: drag.pan.x + deltaX / bounds.width, y: drag.pan.y + deltaY / bounds.height };
      renderTrackingPreview();
    }
  });
  canvas.addEventListener('pointerup', event => {
    if (drag?.selection) {
      if (!trackingRectangle || trackingRectangle.width < 32 || trackingRectangle.height < 32) trackingRectangle = null;
      selectingTrackingWindow = false;
      element('trackingSelectWindow').setAttribute('aria-pressed', 'false');
      trackingPaused = false;
      text('trackingWindowInfo', trackingRectangle ? `${trackingRectangle.width} x ${trackingRectangle.height} px | (${trackingRectangle.x}, ${trackingRectangle.y})` : 'Fenster mindestens 32 x 32 px');
      updateTrackingControls(); renderTrackingPreview();
    } else if (drag && !drag.moved) {
      const sourcePoint = pointAt(event);
      if (sourcePoint && trackingPreviewPoints.length) {
        const closest = trackingPreviewPoints.reduce((nearest, point) =>
          Math.hypot(point.x - sourcePoint.x, point.y - sourcePoint.y) < Math.hypot(nearest.x - sourcePoint.x, nearest.y - sourcePoint.y) ? point : nearest);
        if (Math.hypot(closest.x - sourcePoint.x, closest.y - sourcePoint.y) <= 14 / trackingPreviewTransform.scale) selectTrackingPatch(closest);
      }
    }
    drag = null;
  });
  canvas.addEventListener('pointercancel', () => { drag = null; });
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    trackingZoom = Math.max(0.5, Math.min(20, trackingZoom * Math.exp(-event.deltaY * 0.001)));
    renderTrackingPreview();
  }, { passive: false });
}

async function previewTrackingStartFrame() {
  if (!videoInfo || trackingRunning || trackingPreviewBusy || taskBusy || (windowTracking() && !calibration)) return;
  const index = Math.max(0, Math.min(videoInfo.frameCount - 1, Math.round(number('trackingStart', 0))));
  const request = ++trackingPreviewRequest;
  trackingPreviewBusy = true; updateTrackingControls();
  text('trackingPreviewState', `Lade Frame ${index}...`);
  try {
    const decoded = await readTrackingFrame(index, { rectified: windowTracking() });
    if (request !== trackingPreviewRequest || trackingRunning) { decoded.bitmap.close(); return; }
    if (windowTracking()) {
      drawTrackingFrame(decoded.bitmap); decoded.bitmap.close();
    } else { drawTrackingFrame(decoded.bitmap); decoded.bitmap.close(); }
    text('trackingFrameState', `Vorschau Frame ${index} / PTS ${(decoded.timestamp / 1e6).toFixed(6)} s`);
    text('trackingPreviewState', windowTracking() ? 'Entzerrter Startframe' : 'Startframe-Vorschau');
  } catch (error) {
    if (request === trackingPreviewRequest) text('trackingPreviewState', `Vorschau fehlgeschlagen: ${error.message}`);
  } finally { trackingPreviewBusy = false; updateTrackingControls(); }
}

function drawTrackingPath() {
  const poses = trackingPath.filter(entry => entry.pose);
  const { context, width, height } = trackingCanvasContext('trackingPathCanvas');
  context.clearRect(0, 0, width, height);
  if (!poses.length) { pathProject = null; element('trackingPathEmpty').hidden = false; return; }
  element('trackingPathEmpty').hidden = true;
  const geometries = trackingPath.map(entry => frameGeometry(entry, calibration?.field, calibration?.maps)).filter(Boolean);
  const corners = geometries.flatMap(geometry => geometry.corners);
  const proposed = pathRefitProposal ? [...pathRefitProposal.byFrame.values()].map(item => item.pose) : [];
  const xs = [...poses.map(entry => entry.pose.x), ...proposed.map(pose => pose.x), ...corners.map(point => point.x)];
  const ys = [...poses.map(entry => entry.pose.y), ...proposed.map(pose => pose.y), ...corners.map(point => point.y)];
  const minimumX = Math.min(0, ...xs); const maximumX = Math.max(0, ...xs);
  const minimumY = Math.min(0, ...ys); const maximumY = Math.max(0, ...ys);
  const rangeX = Math.max(1, maximumX - minimumX);
  const rangeY = Math.max(1, maximumY - minimumY);
  const padding = 28;
  const scale = Math.min((width - 2 * padding) / rangeX, (height - 2 * padding) / rangeY);
  const project = pose => ({ x: padding + (pose.x - minimumX) * scale, y: height - padding - (pose.y - minimumY) * scale });
  pathProject = project;
  pathProject.invert = point => ({ x: minimumX + (point.x - padding) / scale,
    y: minimumY + (height - padding - point.y) / scale });
  const selected = pathHover ?? pathSelection;
  const supporting = selected ? geometries.filter(geometry => geometry.supports(selected, trackingPixelAllowed)) : [];
  for (const geometry of supporting) {
    const points = geometry.corners.map(project);
    context.strokeStyle = supportColor(geometry.entry.frame); context.lineWidth = 1;
    context.beginPath(); points.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
    context.closePath(); context.stroke();
    context.fillStyle = context.strokeStyle; context.font = '11px sans-serif';
    context.fillText(`#${geometry.entry.frame}`, points[0].x + 3, points[0].y - 3);
  }
  text('trackingPathInfo', selected ? `Position (${fixed(selected.x, 1)}, ${fixed(selected.y, 1)}) | ${supporting.length} Bilder: ${supporting.map(item => `#${item.entry.frame}`).join(', ')}` : 'Maus im Canvas: Bildrechtecke anzeigen. Klicken: Frames an dieser Position ueberlagern.');
  const origin = project({ x: 0, y: 0 });
  context.strokeStyle = '#aebbb5'; context.lineWidth = 1; context.beginPath();
  context.moveTo(padding, origin.y); context.lineTo(width - padding, origin.y);
  context.moveTo(origin.x, padding); context.lineTo(origin.x, height - padding); context.stroke();
  context.strokeStyle = '#067566'; context.lineWidth = 2; context.beginPath();
  poses.forEach((entry, index) => { const point = project(entry.pose); index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y); });
  context.stroke();
  if (pathRefitProposal) {
    context.strokeStyle = '#c6404a'; context.lineWidth = 2; context.setLineDash([6, 4]); context.beginPath();
    poses.forEach((entry, index) => {
      const point = project(pathRefitProposal.byFrame.get(entry.frame)?.pose ?? entry.pose);
      index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y);
    });
    context.stroke(); context.setLineDash([]);
  }
  const latest = project(poses.at(-1).pose);
  context.fillStyle = '#c6404a'; context.beginPath(); context.arc(latest.x, latest.y, 4, 0, Math.PI * 2); context.fill();
  if (selected) {
    const point = project(selected);
    context.strokeStyle = '#111'; context.beginPath(); context.arc(point.x, point.y, 6, 0, Math.PI * 2); context.stroke();
  }
}

function updatePathRefitControls() {
  element('trackingPathRefit').disabled = pathRefitBusy || Boolean(pathRefitProposal) || !pathSelection || trackingPath.length < 2;
  element('trackingPathRefitApply').disabled = pathRefitBusy || !pathRefitProposal?.overlayReady;
  element('trackingPathRefitDiscard').disabled = pathRefitBusy || !pathRefitProposal;
  for (const control of document.querySelectorAll('.refit-adjustments input, .refit-adjustments button')) control.disabled = pathRefitBusy;
}

function trackingGeometry(entry) {
  const proposed = pathRefitProposal?.byFrame.get(entry.frame)?.pose;
  return frameGeometry(proposed ? { ...entry, pose: proposed } : entry, calibration?.field, calibration?.maps);
}

function localRefitFailure(matches, conditionalLimit) {
  const usable = match => match?.accepted || match?.conditionallyAccepted;
  const forward = matches.filter(match => usable(match.forward)).length;
  const backward = matches.filter(match => usable(match.backward)).length;
  const cycle = matches.filter(match => usable(match.backward) && Number.isFinite(match.reverseDistance) &&
    match.reverseDistance <= conditionalLimit).length;
  const reasons = new Map();
  for (const match of matches) {
    const reason = usable(match.forward) ? (usable(match.backward) ?
      (Number.isFinite(match.reverseDistance) ? (match.reverseDistance <= conditionalLimit ? 'Kein Mehrheitskonsens' :
        `Zyklus > ${fixed(conditionalLimit, 1)} px`) : 'Zyklus ungueltig') : 'Rueckwaertssuche') :
      (match.forward?.reason || 'Vorwaertssuche');
    reasons.set(reason, (reasons.get(reason) ?? 0) + 1);
  }
  const breakdown = [...reasons].sort((first, second) => second[1] - first[1]).map(([reason, count]) => `${reason}: ${count}`).join(', ');
  return `${matches.length} Paare versucht; vorwaerts ${forward}, rueckwaerts ${backward}, innerhalb Zyklusgrenze ${cycle}. ${breakdown}`;
}

async function localGroupComposite(entries, bounds, maxFrames, edgeFeather, featherMask, baseMask, region) {
  const candidates = sharpestFramesFirst(entries.map(entry => frameGeometry(entry, calibration?.field, calibration?.maps)).filter(Boolean));
  const supporting = approximateTopFrames(candidates, maxFrames, trackingPixelAllowed, pathSelection).slice(0, 16);
  const composite = document.createElement('canvas');
  composite.width = bounds.width; composite.height = bounds.height;
  const context = composite.getContext('2d', { willReadFrequently: true });
  const scratch = document.createElement('canvas');
  const sum = new Float32Array(bounds.width * bounds.height * 4);
  const counts = new Uint32Array(bounds.width * bounds.height);
  for (const [index, geometry] of supporting.entries()) {
    text('trackingPathRefitInfo', `Gruppenmatch: Lade ${index + 1}/${supporting.length}, Frame #${geometry.entry.frame}...`);
    const image = await readTrackingFrame(geometry.entry.frame, { rectified: true, output: 'rgba' });
    const mask = localSelectionMask(geometry, pathSelection, region, baseMask);
    const allowed = (x, y) => trackingPixelAllowed(x, y) && maskIncludes(mask, x, y);
    applyPixelMask(image.data, image.width, allowed);
    applyEdgeFeather(image.data, image.width, image.height, edgeFeather, featherMask);
    scratch.width = image.width; scratch.height = image.height;
    scratch.getContext('2d').putImageData(new ImageData(image.data, image.width, image.height), 0, 0);
    const origin = geometry.world(0, 0);
    context.setTransform(1, 0, 0, 1, 0, 0); context.clearRect(0, 0, bounds.width, bounds.height);
    context.setTransform(geometry.c, geometry.s, -geometry.s, geometry.c, origin.x - bounds.minX, origin.y - bounds.minY);
    context.drawImage(scratch, 0, 0);
    accumulateFrame(sum, context.getImageData(0, 0, bounds.width, bounds.height).data, counts, maxFrames);
  }
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.putImageData(new ImageData(averagedFrames(sum), bounds.width, bounds.height), 0, 0);
  return { bitmap: await createImageBitmap(composite), frames: supporting.map(geometry => geometry.entry.frame) };
}

async function seedFromLocalGroupMatch(entries, worker, limits, preprocessing, baseMask, selectedFrame, localLimit) {
  const groups = localRefitGroups(entries);
  if (groups.length < 2) return { entries, attempts: 0, accepted: 0 };
  const radius = Math.min(calibration.maps.outputWidth, calibration.maps.outputHeight) * preprocessing.region / 2;
  const bounds = { minX: Math.floor(pathSelection.x - radius), minY: Math.floor(pathSelection.y - radius),
    width: Math.max(128, Math.ceil(radius * 2)), height: Math.max(128, Math.ceil(radius * 2)) };
  const maxFrames = Math.min(8, Math.max(1, number('trackingOverlayMaxFrames', 3)));
  const edgeFeather = number('overlayEdgeFeather', 10) / 100;
  const anchor = Math.max(0, groups.findIndex(group => group.some(entry => entry.frame === Number(selectedFrame))));
  const centerPose = { x: bounds.minX + bounds.width / 2, y: bounds.minY + bounds.height / 2, rotation: 0 };
  const featherMask = edgeFeatherMask(calibration.maps.outputWidth, calibration.maps.outputHeight, trackingPixelAllowed, edgeFeather);
  const composites = [];
  try {
    for (const [index, group] of groups.entries()) {
      const composite = await localGroupComposite(group, bounds, maxFrames, edgeFeather, featherMask, baseMask, preprocessing.region);
      composites.push({ frame: index, pose: centerPose,
        bitmap: composite.bitmap, sourceFrames: composite.frames });
    }
    const pairs = [];
    for (let reference = 0; reference < groups.length; reference++) for (let current = reference + 1; current < groups.length; current++) {
      pairs.push({ reference, current, group: `${reference}:${current}` });
    }
    text('trackingPathRefitInfo', `Pruefe ${pairs.length} Gruppenueberlappungen vor dem Einzelbild-Refinement...`);
    const matches = await worker.call('local-refit-register', { images: composites, pairs,
      limits: { ...limits, radius: Math.max(limits.radius, radius), reverseRadius: Math.min(32, limits.reverseRadius) },
      preprocessing }, composites.map(item => item.bitmap));
    const accepted = matches.filter(match => match.forward?.accepted && match.backward?.accepted &&
      Number.isFinite(match.reverseDistance) && match.reverseDistance <= localLimit);
    const centers = new Map([[anchor, centerPose]]);
    let changed = true;
    while (changed) {
      changed = false;
      for (const match of accepted) {
        if (centers.has(match.reference) && !centers.has(match.current)) {
          centers.set(match.current, applyPoseCorrection(match.forward.pose, match.referencePose, centers.get(match.reference)));
          changed = true;
        } else if (centers.has(match.current) && !centers.has(match.reference)) {
          centers.set(match.reference, applyPoseCorrection(match.backward.pose, match.forward.pose, centers.get(match.current)));
          changed = true;
        }
      }
    }
    return { entries: entries.map(entry => {
      const group = groups.findIndex(items => items.some(item => item.frame === entry.frame));
      return centers.has(group) ? { ...entry, pose: applyPoseCorrection(entry.pose, centerPose, centers.get(group)) } : entry;
    }), attempts: matches.length, accepted: accepted.length, aligned: centers.size };
  } finally { for (const composite of composites) composite.bitmap.close(); }
}

async function refitTrackingPath() {
  if (!pathSelection || pathRefitBusy || trackingRunning || taskBusy) return;
  const geometries = trackingPath.map(entry => frameGeometry(entry, calibration?.field, calibration?.maps)).filter(Boolean);
  const seeds = geometries.filter(geometry => geometry.supports(pathSelection, trackingPixelAllowed)).map(geometry => geometry.entry.frame);
  if (!seeds.length) { text('trackingPathRefitInfo', 'Am Auswahlpunkt liegen keine Frames.'); return; }
  const conditionalLimit = trackingRunOptions?.contextCycleConditional ?? number('trackingContextCycleConditional', 7.5);
  const preprocessing = refitPreprocessing();
  const baseMask = (trackingRunOptions ?? trackingDataset?.options)?.imageMask ?? null;
  const regionRadius = Math.min(calibration.maps.outputWidth, calibration.maps.outputHeight) * preprocessing.region / 2;
  const localCandidates = path => path.map(entry => frameGeometry(entry, calibration?.field, calibration?.maps)).filter(Boolean)
    .map(geometry => ({ geometry, distance: localSelectionDistance(geometry, pathSelection) }))
    .filter(item => item.distance <= regionRadius)
    .filter(item => localSelectionSupport(item.geometry, pathSelection, preprocessing.region, baseMask,
      (x, y) => Boolean(calibration.maps.valid[y * calibration.maps.outputWidth + x])) >= 128)
    .map(item => ({ ...item.geometry.entry, localSelectionDistance: item.distance }));
  let localEntries = localCandidates(trackingPath);
  const selectedFrame = element('trackingOverlayFrame').value;
  if (localEntries.length < 2) { text('trackingPathRefitInfo', 'Die Auswahl enthaelt zu wenige auswertbare Frames.'); return; }
  pathRefitBusy = true; pathRefitProposal = null; updatePathRefitControls();
  let searchProgress = { pass: 1, round: 1, attempted: 0, limit: 512 };
  const worker = new WorkerClient('/compute-worker.js', status => {
    if (status.operation === 'local-refit') text('trackingPathRefitInfo', searchProgress.stage === 'group' ?
      `Gruppenmatch ${status.pair}/${status.pairs}: Gruppe ${status.current} gegen ${status.reference}...` :
      `Refit-Iteration ${searchProgress.pass}, Suchrunde ${searchProgress.round}, Paar ${searchProgress.attempted + status.pair}/${searchProgress.limit}: #${status.current} gegen #${status.reference}...`);
  });
  try {
    const limits = { radius: Math.min(256, Math.max(64, number('trackingContextRadius', 32) * 4)),
      reverseRadius: Math.min(256, Math.max(64, number('trackingContextRadius', 32) * 4)),
      angle: Math.min(5, Math.max(2, number('trackingContextAngle', 1) * 4)), coarseStep: 1 };
    let graph = buildPoseGraph(trackingPath, { conditionalLimit });
    const localLimit = Math.min(50, Math.max(25, conditionalLimit * 3));
    const graphOptions = { conditionalLimit, cycleLimit: localLimit, consensusLimit: localLimit,
      lever: Math.hypot(calibration.maps.outputWidth, calibration.maps.outputHeight) / 2 };
    searchProgress = { stage: 'group', pass: 0, round: 0, attempted: 0, limit: 0 };
    const groupSeed = await seedFromLocalGroupMatch(localEntries, worker, limits, preprocessing, baseMask, selectedFrame, localLimit);
    localEntries = groupSeed.entries;
    text('trackingPathRefitInfo', `Gruppenmatch: ${groupSeed.accepted}/${groupSeed.attempts} Ueberlappungen bestaetigt, ${groupSeed.aligned} Gruppen ausgerichtet; starte Einzelbild-Refinement...`);
    let result = null; let totalMatches = 0; let completedPasses = 0;
    for (let pass = 1; pass <= 3 && localEntries.length >= 2; pass++) {
      const entries = new Map(localEntries.map(entry => [entry.frame, entry]));
      const search = await searchLocalRefit(localEntries, async (pairs, progress) => {
        searchProgress = { ...progress, pass };
        const requiredFrames = [...new Set(pairs.flatMap(pair => [pair.current, pair.reference]))];
        const images = [];
        try {
          for (const [index, frame] of requiredFrames.entries()) {
            text('trackingPathRefitInfo', `Refit-Iteration ${pass}: Lade Frame ${index + 1}/${requiredFrames.length}: #${frame}...`);
            const decoded = await readTrackingFrame(frame, { rectified: true });
            const entry = entries.get(frame); const geometry = frameGeometry(entry, calibration?.field, calibration?.maps);
            images.push({ frame, pose: { ...entry.pose }, bitmap: decoded.bitmap,
              mask: localSelectionMask(geometry, pathSelection, preprocessing.region, baseMask) });
          }
          return await worker.call('local-refit-register', { images, pairs, limits, preprocessing }, images.map(item => item.bitmap));
        } finally { for (const item of images) item.bitmap.close(); }
      }, { baseGraph: graph, preferredFrames: selectedFrame ? [Number(selectedFrame)] : [],
        maxPairs: pass === 1 ? 512 : 256, graphOptions });
      totalMatches += search.matches.length;
      if (search.graph.localEdges < 1) {
        if (!result) throw new Error(`Kein bestaetigter lokaler Match nach ${search.rounds} Runden (${search.limited ? 'Suchbudget erreicht' : 'Kandidaten ausgeschoepft'}). ${localRefitFailure(search.matches, localLimit)}`);
        break;
      }
      graph = search.graph; completedPasses = pass;
      text('trackingPathRefitInfo', `Refit-Iteration ${pass}: ${graph.localEdges} neue lokale Kanten; verteile Korrektur durch das Pose-Netz...`);
      result = await worker.call('pose-graph-refit', { graph, options: { seedFrames: seeds, iterations: 10, huber: 20 } });
      const corrected = new Map(result.corrections.map(item => [item.frame, item.pose]));
      localEntries = localCandidates(trackingPath.map(entry => corrected.has(entry.frame) ?
        { ...entry, pose: corrected.get(entry.frame) } : entry));
    }
    pathRefitProposal = { ...result, byFrame: new Map(result.corrections.map(item => [item.frame, item])),
      overlayFrames: localEntries.map(entry => entry.frame), preprocessing, refitPasses: completedPasses,
      refitMatches: totalMatches, groupMatches: groupSeed.accepted, groupAttempts: groupSeed.attempts, overlayReady: false };
    text('trackingPathRefitInfo', `${groupSeed.accepted}/${groupSeed.attempts} Gruppenmatches | ${completedPasses} Refit-Iterationen, ${totalMatches} Paarversuche | ${result.nodes} Frames, ${result.edges} Kanten (${result.localEdges} lokal, ${result.spatialEdges} raeumlich) | Lokal RMS ${fixed(result.localBeforeRms, 2)} -> ${fixed(result.localAfterRms, 2)} px | Netz RMS ${fixed(result.beforeRms, 2)} -> ${fixed(result.afterRms, 2)} px | Rot gestrichelt: Vorschlag.`);
    drawTrackingPath();
    await loadPathOverlay(pathSelection);
    if (!pathOverlay.width) throw new Error('Das Mischbild der Refit-Vorschau konnte nicht aufgebaut werden.');
    pathRefitProposal.overlayReady = true;
    text('trackingPathRefitInfo', `${groupSeed.accepted}/${groupSeed.attempts} Gruppenmatches | ${completedPasses} Refit-Iterationen, ${totalMatches} Paarversuche | ${result.nodes} Frames, ${result.edges} Kanten (${result.localEdges} lokal, ${result.spatialEdges} raeumlich) | Lokal RMS ${fixed(result.localBeforeRms, 2)} -> ${fixed(result.localAfterRms, 2)} px | Netz RMS ${fixed(result.beforeRms, 2)} -> ${fixed(result.afterRms, 2)} px | Mischbild zeigt die Refit-Vorschau.`);
  } catch (error) { text('trackingPathRefitInfo', `Refit fehlgeschlagen: ${error.message}`); }
  finally { worker.terminate(); pathRefitBusy = false; updatePathRefitControls(); }
}

async function discardTrackingPathRefit() {
  if (pathRefitBusy) return;
  pathRefitBusy = true;
  pathRefitProposal = null; text('trackingPathRefitInfo', 'Kein Refit-Vorschlag.');
  updatePathRefitControls(); drawTrackingPath();
  try { if (pathSelection) await loadPathOverlay(pathSelection); }
  finally { pathRefitBusy = false; updatePathRefitControls(); }
}

function applyTrackingPathRefit() {
  if (!pathRefitProposal || pathRefitBusy) return;
  for (const entry of trackingPath) {
    const correction = pathRefitProposal.byFrame.get(entry.frame);
    if (!correction) continue;
    entry.pose = { ...correction.pose };
    if (entry.raw) entry.raw = { ...entry.raw, x: entry.raw.x + correction.dx, y: entry.raw.y + correction.dy,
      rotation: Math.atan2(Math.sin(entry.raw.rotation + correction.rotation), Math.cos(entry.raw.rotation + correction.rotation)) };
  }
  const summary = { created_at: new Date().toISOString(), nodes: pathRefitProposal.nodes, edges: pathRefitProposal.edges,
    spatialEdges: pathRefitProposal.spatialEdges, localEdges: pathRefitProposal.localEdges,
    refitPasses: pathRefitProposal.refitPasses, refitMatches: pathRefitProposal.refitMatches,
    groupMatches: pathRefitProposal.groupMatches, groupAttempts: pathRefitProposal.groupAttempts,
    localBeforeRms: pathRefitProposal.localBeforeRms, localAfterRms: pathRefitProposal.localAfterRms,
    beforeRms: pathRefitProposal.beforeRms, afterRms: pathRefitProposal.afterRms,
    preprocessing: pathRefitProposal.preprocessing };
  if (trackingDataset) (trackingDataset.refits ??= []).push(summary);
  pathRefitProposal = null; clearPathOverlay();
  text('trackingPathRefitInfo', `Refit uebernommen | RMS ${fixed(summary.beforeRms, 2)} -> ${fixed(summary.afterRms, 2)} px.`);
  updatePathRefitControls(); renderTrackingResults(trackingPath.at(-1));
  if (pathSelection) void loadPathOverlay(pathSelection);
}

function renderPathOverlay() {
  if (element('trackingOverlayPane').hidden) return;
  const { context, width, height } = trackingCanvasContext('trackingOverlayCanvas');
  context.clearRect(0, 0, width, height);
  if (!pathOverlay.width) return;
  const scale = Math.min(width / pathOverlay.width, height / pathOverlay.height) * overlayZoom;
  context.imageSmoothingEnabled = false;
  if (overlaySelectedImage && overlayBounds) {
    const left = (width - pathOverlay.width * scale) / 2 + overlayPan.x;
    const top = (height - pathOverlay.height * scale) / 2 + overlayPan.y;
    const { geometry, bitmap } = overlaySelectedImage;
    const origin = geometry.world(0, 0);
    context.save();
    context.translate(left + (origin.x - overlayBounds.minX) * scale, top + (origin.y - overlayBounds.minY) * scale);
    context.transform(geometry.c * scale, geometry.s * scale, -geometry.s * scale, geometry.c * scale, 0, 0);
    drawRefitPreviewImage(context, bitmap, 0, 0);
    context.restore();
    return;
  }
  if (pathOverlay.tiles) {
    const left = (width - pathOverlay.width * scale) / 2 + overlayPan.x;
    const top = (height - pathOverlay.height * scale) / 2 + overlayPan.y;
    const ratio = context.getTransform().a;
    const snap = value => Math.round(value * ratio) / ratio;
    for (const tile of pathOverlay.tiles) {
      const x = snap(left + tile.x * scale), y = snap(top + tile.y * scale);
      const w = snap(left + (tile.x + tile.width) * scale) - x;
      const h = snap(top + (tile.y + tile.height) * scale) - y;
      if (w <= 0 || h <= 0 || x >= width || y >= height || x + w <= 0 || y + h <= 0) continue;
      drawRefitPreviewImage(context, tile.bitmap, 0, 0, tile.width, tile.height, x, y, w, h);
    }
    return;
  }
  drawRefitPreviewImage(context, pathOverlay, (width - pathOverlay.width * scale) / 2 + overlayPan.x,
    (height - pathOverlay.height * scale) / 2 + overlayPan.y, pathOverlay.width * scale, pathOverlay.height * scale);
}

function publishOverlayContributors(supporting, bounds) {
  overlayContributors = supporting;
  overlayBounds = bounds;
  const selector = element('trackingOverlayFrame');
  selector.replaceChildren(new Option('Mischbild', ''));
  for (const geometry of supporting) {
    const entry = geometry.entry;
    const score = Number.isFinite(entry.sharpness?.score) ? ` | Schaerfe ${fixed(entry.sharpness.score, 1)}` : '';
    const matches = (entry.incrementalMatch ? 1 : 0) + (entry.context?.matches?.length ?? 0);
    selector.add(new Option(`#${entry.frame}${score} | ${matches} Matches`, String(entry.frame)));
  }
  selector.disabled = false;
}

async function selectOverlayContributor() {
  const request = ++overlaySelectionRequest;
  overlaySelectedImage?.bitmap.close();
  overlaySelectedImage = null;
  const frame = Number(element('trackingOverlayFrame').value);
  const geometry = overlayContributors.find(item => item.entry.frame === frame);
  element('trackingOverlayInspect').disabled = !geometry;
  if (!geometry) { renderPathOverlay(); return; }
  text('trackingOverlayInfo', `Lade Frame #${frame} einzeln...`);
  const decoded = await readTrackingFrame(frame, { rectified: true });
  if (request !== overlaySelectionRequest) { decoded.bitmap.close(); return; }
  overlaySelectedImage = { geometry, bitmap: decoded.bitmap };
  const pose = geometry.entry.pose ?? geometry.entry.raw;
  text('trackingOverlayInfo', `Frame #${frame} einzeln | Pose (${fixed(pose.x, 2)}, ${fixed(pose.y, 2)}, ${fixed(pose.rotation * 180 / Math.PI, 3)} Grad) | Pruefen zeigt die zugehoerigen Referenzmatches.`);
  renderPathOverlay();
}

async function loadPathOverlay(point) {
  const request = ++overlayRequest;
  pathSelection = point;
  updatePathRefitControls();
  const maxFrames = number('trackingOverlayMaxFrames', 3);
  const edgeFeather = number('overlayEdgeFeather', 10) / 100;
  if (!Number.isInteger(maxFrames) || maxFrames < 1 || maxFrames > 64) {
    text('trackingOverlayInfo', 'Framegrenze muss zwischen 1 und 64 liegen.'); return;
  }
  const positionLabel = `${pathRefitProposal ? 'Refit-Vorschau | ' : ''}Position (${fixed(point.x, 1)}, ${fixed(point.y, 1)}) | Top ${maxFrames}/Bereich`;
  element('trackingOverlayPane').hidden = false;
  clearPathOverlay();
  overlayZoom = 1; overlayPan = { x: 0, y: 0 }; renderPathOverlay(); drawTrackingPath();
  if (!videoInfo || !calibration || (trackingDataset?.video && trackingDataset.video.name !== videoInfo.name)) {
    text('trackingOverlayInfo', 'Bitte das zum Tracking gehoerende Video und die Kalibrierung laden.'); return;
  }
  if (trackingRunning || taskBusy) { text('trackingOverlayInfo', 'Verarbeitung pausieren und die Position erneut anklicken.'); return; }
  const previewFrames = pathRefitProposal?.overlayFrames ? new Set(pathRefitProposal.overlayFrames) : null;
  const candidates = sharpestFramesFirst(trackingPath.filter(item => !previewFrames || previewFrames.has(item.frame))
    .map(item => trackingGeometry(item)).filter(geometry => geometry && (previewFrames || geometry.supports(point, trackingPixelAllowed))));
  if (!candidates.length) { text('trackingOverlayInfo', 'Keine Bilddaten an dieser Position.'); return; }
  const supporting = previewFrames ? candidates : approximateTopFrames(candidates, maxFrames, trackingPixelAllowed, point);
  const corners = supporting.flatMap(geometry => geometry.corners);
  const minX = Math.floor(Math.min(...corners.map(p => p.x))), minY = Math.floor(Math.min(...corners.map(p => p.y)));
  const width = Math.ceil(Math.max(...corners.map(p => p.x))) - minX;
  const height = Math.ceil(Math.max(...corners.map(p => p.y))) - minY;
  const bounds = { minX, minY, width, height };
  if (element('trackingUseWebGpu').checked && navigator.gpu) {
    const previous = overlayGpuIdle;
    let release;
    overlayGpuIdle = new Promise(resolve => { release = resolve; });
    await previous; // Cancelled requests release large buffers before the next allocation.
    let overlay = null;
    const started = performance.now();
    try {
      await frameReader.dispose(); // Reserve the GPU budget for the overlay's accumulation buffers.
      if (request !== overlayRequest) return;
      text('trackingOverlayInfo', `${positionLabel} | Bereite Vollaufloesung ${width} x ${height} vor...`);
      if (await WebGpuOverlay.needsTiles(calibration.maps, width, height)) {
        const tiled = await renderTiledOverlay({ maps: calibration.maps, width, height, minX, minY,
          geometries: supporting, pixelAllowed: trackingPixelAllowed, maxFrames, edgeFeather,
          decode: index => readTrackingFrame(index, { output: 'native' }), cancelled: () => request !== overlayRequest,
          progress: p => text('trackingOverlayInfo', `${positionLabel} | Kachel ${p.tileIndex + 1}/${p.tileCount} bis ${p.tileSize} px | Kandidat ${p.frameIndex + 1}/${p.frameCount}: #${p.frame} | Durchlauf ${p.framePass}/${p.plannedFramePasses} | WebGPU, Vollaufloesung`) });
        if (!tiled) return;
        if (request !== overlayRequest) { closeOverlayTiles(tiled); return; }
        pathOverlay = tiled;
        publishOverlayContributors(supporting, bounds);
        text('trackingOverlayInfo', `${positionLabel} | ${supporting.length} von ${candidates.length} Kandidaten | ${width} x ${height} px, Vollaufloesung | WebGPU | ${tiled.tiles.length} Kacheln bis ${tiled.tileSize} px | ${fixed(tiled.frameMs / Math.max(1, tiled.framePasses), 1)} ms/Frame-Durchlauf | ${tiled.framePasses} Durchlaeufe | Gesamt ${fixed((performance.now() - started) / 1000, 2)} s`);
        element('trackingOverlayInfo').textContent += ` | Frameabruf ${fixed(tiled.decodeMs / 1000, 2)} s | GPU inkl. Entzerrung/Warten ${fixed(tiled.gpuMs / 1000, 2)} s | Ausgabe ${fixed(tiled.finishMs / 1000, 2)} s | Vorbereitung ${fixed(tiled.setupMs, 0)} ms`;
        renderPathOverlay();
        return;
      }
      overlay = await WebGpuOverlay.create(calibration.maps, width, height, minX, minY, trackingPixelAllowed, maxFrames, edgeFeather);
      if (request !== overlayRequest) return;
      const setupMs = performance.now() - started;
      let frameMs = 0, decodeMs = 0, gpuMs = 0;
      for (const [index, geometry] of supporting.entries()) {
        text('trackingOverlayInfo', `${positionLabel} | Lade ${index + 1}/${supporting.length}: Frame #${geometry.entry.frame} | WebGPU, Vollaufloesung`);
        const frameStarted = performance.now();
        const decoded = await readTrackingFrame(geometry.entry.frame, { output: 'native' });
        decodeMs += performance.now() - frameStarted;
        try {
          if (request !== overlayRequest) return;
          const gpuStarted = performance.now();
          await overlay.addFrame(decoded.frame, decoded.orientation, geometry);
          gpuMs += performance.now() - gpuStarted;
        } finally { decoded.frame.close(); }
        if (request !== overlayRequest) return;
        frameMs += performance.now() - frameStarted;
      }
      const bitmap = await overlay.finish();
      try {
        if (request !== overlayRequest) return;
        pathOverlay.width = width; pathOverlay.height = height;
        pathOverlay.getContext('2d').drawImage(bitmap, 0, 0);
      } finally { bitmap.close(); }
      text('trackingOverlayInfo', `${positionLabel} | ${supporting.length} von ${candidates.length} Kandidaten | ${width} x ${height} px, Vollaufloesung | WebGPU | ${fixed(frameMs / supporting.length, 1)} ms/Frame | Vorbereitung ${fixed(setupMs, 0)} ms | Gesamt ${fixed((performance.now() - started) / 1000, 2)} s`);
      element('trackingOverlayInfo').textContent += ` | Frameabruf ${fixed(decodeMs / 1000, 2)} s | GPU inkl. Entzerrung/Warten ${fixed(gpuMs / 1000, 2)} s`;
      publishOverlayContributors(supporting, bounds);
      renderPathOverlay();
    } catch (error) {
      if (request === overlayRequest) text('trackingOverlayInfo', `Ueberlagerung fehlgeschlagen: ${error.message}`);
    } finally { overlay?.destroy(); release(); }
    return;
  }
  // Keep the native pixel scale. Larger mosaics need tiling, never silent downscaling.
  if (width * height > 64e6 || Math.max(width, height) > 16384) {
    text('trackingOverlayInfo', 'Vollaufloesende Ueberlagerung ist zu gross fuer den CPU-Puffer. Kleineren Bildbereich auswaehlen.'); return;
  }
  const resolution = 1;
  const composite = document.createElement('canvas');
  composite.width = Math.ceil(width * resolution); composite.height = Math.ceil(height * resolution);
  const context = composite.getContext('2d', { willReadFrequently: true });
  // Sum before dividing to avoid rounding alpha 1/N to zero for long paths.
  const sum = new Float32Array(composite.width * composite.height * 4);
  const counts = new Uint32Array(composite.width * composite.height);
  const scratch = document.createElement('canvas');
  const featherMask = edgeFeatherMask(calibration.maps.outputWidth, calibration.maps.outputHeight, trackingPixelAllowed, edgeFeather);
  try {
    for (const [index, geometry] of supporting.entries()) {
      text('trackingOverlayInfo', `${positionLabel} | Lade ${index + 1}/${supporting.length}: Frame #${geometry.entry.frame}`);
      const image = await readTrackingFrame(geometry.entry.frame, { rectified: true, output: 'rgba' });
      if (request !== overlayRequest) return;
      applyPixelMask(image.data, image.width, trackingPixelAllowed);
      applyEdgeFeather(image.data, image.width, image.height, edgeFeather, featherMask);
      scratch.width = image.width; scratch.height = image.height;
      scratch.getContext('2d').putImageData(new ImageData(image.data, image.width, image.height), 0, 0);
      const origin = geometry.world(0, 0);
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, composite.width, composite.height);
      context.setTransform(resolution * geometry.c, resolution * geometry.s, -resolution * geometry.s, resolution * geometry.c,
        resolution * (origin.x - minX), resolution * (origin.y - minY));
      context.drawImage(scratch, 0, 0);
      accumulateFrame(sum, context.getImageData(0, 0, composite.width, composite.height).data, counts, maxFrames);
    }
    pathOverlay.width = composite.width; pathOverlay.height = composite.height;
    pathOverlay.getContext('2d').putImageData(new ImageData(averagedFrames(sum), composite.width, composite.height), 0, 0);
    text('trackingOverlayInfo', `${positionLabel} | ${supporting.length} von ${candidates.length} Kandidaten | ${width} x ${height} px, Vollaufloesung | Gleich gewichtet, deckend | ${supporting.map(g => `#${g.entry.frame}`).join(', ')}`);
    publishOverlayContributors(supporting, bounds);
    renderPathOverlay();
  } catch (error) { if (request === overlayRequest) text('trackingOverlayInfo', `Ueberlagerung fehlgeschlagen: ${error.message}`); }
}

function installPathInteraction() {
  const path = element('trackingPathCanvas');
  const hit = event => {
    if (!pathProject) return null;
    const bounds = path.getBoundingClientRect();
    return pathProject.invert({ x: event.clientX - bounds.left, y: event.clientY - bounds.top });
  };
  path.addEventListener('pointermove', event => { pathHover = hit(event); drawTrackingPath(); });
  path.addEventListener('pointerleave', () => { pathHover = null; drawTrackingPath(); });
  path.addEventListener('click', event => { const point = hit(event); if (point) void loadPathOverlay(point); });
  const canvas = element('trackingOverlayCanvas');
  let drag = null;
  canvas.addEventListener('pointerdown', event => { if (event.button !== 0) return; drag = { clientX: event.clientX, clientY: event.clientY, pan: { ...overlayPan } }; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener('pointermove', event => { if (!drag) return; overlayPan = { x: drag.pan.x + event.clientX - drag.clientX, y: drag.pan.y + event.clientY - drag.clientY }; renderPathOverlay(); });
  for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(name, () => { drag = null; });
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    const bounds = canvas.getBoundingClientRect();
    const next = Math.max(0.25, Math.min(64, overlayZoom * Math.exp(-event.deltaY * 0.001)));
    const factor = next / overlayZoom;
    const x = event.clientX - bounds.left - bounds.width / 2, y = event.clientY - bounds.top - bounds.height / 2;
    overlayPan = { x: x - (x - overlayPan.x) * factor, y: y - (y - overlayPan.y) * factor };
    overlayZoom = next; renderPathOverlay();
  }, { passive: false });
  element('trackingOverlayFit').onclick = () => { overlayZoom = 1; overlayPan = { x: 0, y: 0 }; renderPathOverlay(); };
  element('trackingPathRefit').onclick = () => void refitTrackingPath();
  element('trackingPathRefitApply').onclick = applyTrackingPathRefit;
  element('trackingPathRefitDiscard').onclick = () => void discardTrackingPathRefit();
  for (const id of ['refitBrightness', 'refitContrast', 'refitGamma', 'refitRed', 'refitGreen', 'refitBlue', 'refitRegion', 'overlayEdgeFeather']) {
    element(id).addEventListener('input', updateRefitPreprocessing);
  }
  element('overlayEdgeFeather').addEventListener('change', () => { if (pathSelection) void loadPathOverlay(pathSelection); });
  element('refitPreprocessPreview').onchange = renderPathOverlay;
  element('refitPreprocessReset').onclick = () => {
    const defaults = { refitBrightness: 0, refitContrast: 100, refitGamma: 1, refitRed: 30, refitGreen: 59, refitBlue: 11,
      refitRegion: 40, overlayEdgeFeather: 10 };
    for (const [id, value] of Object.entries(defaults)) element(id).value = value;
    updateRefitPreprocessing();
    if (pathSelection) void loadPathOverlay(pathSelection);
  };
  element('trackingOverlayFrame').onchange = () => void selectOverlayContributor().catch(error => message(error.message, true));
  element('trackingOverlayInspect').onclick = () => {
    const frame = Number(element('trackingOverlayFrame').value);
    const entry = overlayContributors.find(item => item.entry.frame === frame)?.entry;
    if (!entry) return;
    void showTrackedFrame(entry).then(() => element('trackingInspector').scrollIntoView({ behavior: 'smooth', block: 'start' })).catch(error => message(error.message, true));
  };
  element('trackingOverlayMaxFrames').onchange = () => { if (pathSelection) void loadPathOverlay(pathSelection); };
  updateRefitPreprocessing();
  updatePathRefitControls();
  new ResizeObserver(drawTrackingPath).observe(path.parentElement);
  new ResizeObserver(renderPathOverlay).observe(canvas.parentElement);
}

function installTrackingMaskInteraction() {
  const canvas = element('trackingMaskCanvas');
  let painting = false;
  const paint = event => {
    if (!trackingMaskTransform || !trackingImageMask) return;
    const bounds = canvas.getBoundingClientRect();
    const x = (event.clientX - bounds.left - trackingMaskTransform.left) / trackingMaskTransform.scale;
    const y = (event.clientY - bounds.top - trackingMaskTransform.top) / trackingMaskTransform.scale;
    if (x < 0 || y < 0 || x >= trackingImageMask.sourceWidth || y >= trackingImageMask.sourceHeight) return;
    paintPatchMask(trackingImageMask, x, y, number('trackingMaskBrush', 96) / 2,
      trackingMaskTool === 'include' ? MASK_SEARCH : MASK_NEUTRAL);
    if (trackingMaskTool === 'include') trackingMaskHasSelection = true;
    renderTrackingMask(); drawTrackingPath();
  };
  canvas.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    painting = true; canvas.setPointerCapture(event.pointerId); paint(event);
  });
  canvas.addEventListener('pointermove', event => { if (painting) paint(event); });
  const finish = () => {
    if (!painting) return;
    painting = false;
    trackingMaskHasSelection = trackingImageMask?.data.includes(MASK_SEARCH) ?? false;
    updateTrackingMaskInfo();
    if (pathSelection) void loadPathOverlay(pathSelection);
  };
  canvas.addEventListener('pointerup', finish);
  canvas.addEventListener('pointercancel', finish);
  for (const button of document.querySelectorAll('[data-tracking-mask-tool]')) button.onclick = () => {
    trackingMaskTool = button.dataset.trackingMaskTool;
    for (const tool of document.querySelectorAll('[data-tracking-mask-tool]')) tool.setAttribute('aria-pressed', String(tool === button));
    renderTrackingMask();
  };
  element('trackingMaskBrush').oninput = () => {
    text('trackingMaskBrushValue', `${element('trackingMaskBrush').value} px`);
    saveVideoOptions();
  };
  element('reloadTrackingMask').onclick = () => { void loadTrackingMaskPreview(); };
  element('clearTrackingMask').onclick = () => {
    if (!trackingImageMask) return;
    trackingImageMask.data.fill(MASK_NEUTRAL); trackingImageMask.revision++;
    trackingMaskHasSelection = false;
    updateTrackingMaskInfo();
    renderTrackingMask(); drawTrackingPath();
    if (pathSelection) void loadPathOverlay(pathSelection);
  };
  new ResizeObserver(renderTrackingMask).observe(canvas.parentElement);
}

function renderTrackingResults(entry) {
  updateCorrectionDataStatus();
  const pose = entry.pose;
  text('trackingFrameState', `Frame ${entry.frame} / PTS ${(entry.timestamp / 1e6).toFixed(6)} s`);
  const sharpnessLabel = Number.isFinite(entry.sharpness?.score) ? ` | Schaerfe ${fixed(entry.sharpness.score, 2)}` : '';
  text('trackingPreviewState', entry.success ? `${entry.points} Patches | ${entry.rediscovered} wiedererkannt | ${entry.accelerator}${sharpnessLabel}` : entry.reason);
  if (entry.mode === 'window') {
    text('trackingPreviewState', entry.success ? `Window | ${entry.resolution} | ${entry.accelerator}${sharpnessLabel}` : entry.reason);
    const context = entry.context;
    const status = context ? ` | Umfeld ${context.inliers.length}/${context.selected}: ${context.applied ? 'korrigiert' : 'inkrementell'} | ${context.accelerator || 'CPU'}${context.fallback ? ' (Fallback)' : ''}` : '';
    text('trackingMatchQuality', `Korrelation: ${fixed(entry.score, 3)} | Iterationen: ${entry.iterations ?? 0}${status}`);
    element('trackingMatchQuality').title = context?.matches.map(match => {
      const diagnostics = `${Number.isFinite(match.margin) ? ` | Marge ${fixed(match.margin, 5)}` : ''}${Number.isFinite(match.reverseDistance) ? ` | Rueckweg ${fixed(match.reverseDistance, 2)} px` : ''}${Number.isFinite(match.searchRadius) ? ` | Radius ${match.searchRadius} px` : ''}`;
      return `#${match.frame} ${match.kind || ''}: NCC ${fixed(match.score, 3)}${diagnostics} ${match.reason || (context.inliers.includes(match.frame) ? 'Konsens' : 'kein Konsens')}`;
    }).join('\n') || '';
    if (context) element('trackingMatchQuality').title += `\nBildcache: ${context.cacheHits ?? 0} Treffer / ${context.cacheMisses ?? 0} angeforderte Nachladungen\nGPU-Cache: ${fixed((context.gpuCacheBytes || 0) / 1048576, 1)} MiB${context.fallback ? `\nCPU-Fallback/Backendwahl: ${context.fallback}` : ''}`;
    if (context?.pyramidBytes) element('trackingMatchQuality').title += `\nPyramide ${context.imageWidth} x ${context.imageHeight}: ${fixed(context.pyramidBytes / 1048576, 1)} MiB | Cache ${context.cacheEntries} Bilder, ${fixed(context.cacheBytes / 1048576, 1)}/${fixed(context.cacheLimit / 1048576, 0)} MiB, Kapazitaet ${Math.floor(context.cacheLimit / context.pyramidBytes)} Bilder | ${context.cacheEvictions} Verdraengungen`;
  }
  if (pose) {
    text('trackingX', `${fixed(pose.x, 3)} px`); text('trackingY', `${fixed(pose.y, 3)} px`);
    text('trackingRotation', `${fixed(pose.rotation * 180 / Math.PI, 4)} deg`);
    text('trackingRms', `${fixed(entry.raw?.rms, 3)} px`);
  } else {
    for (const id of ['trackingX', 'trackingY', 'trackingRotation', 'trackingRms']) text(id, '-');
  }
  text('trackingPatchCount', String(entry.points));
  text('trackingPathState', `${trackingPath.filter(item => item.pose).length} Posen`);
  text('trackingSummary', `${trackingPath.length} Frames | Fenster ${number('trackingWindow', 1)}`);
  const tbody = element('trackingTable'); tbody.replaceChildren();
  const referenceCounts = trackingReferenceCounts(trackingPath);
  for (const item of [...trackingPath, ...trackingFailures].sort((first, second) => first.frame - second.frame).slice(-200).reverse()) {
    const row = tbody.insertRow();
    row.className = 'tracking-row'; row.tabIndex = 0;
    row.insertCell().textContent = `#${item.frame} / ${(item.timestamp / 1e6).toFixed(3)} s`;
    row.insertCell().textContent = item.pose ? fixed(item.pose.x, 3) : '-';
    row.insertCell().textContent = item.pose ? fixed(item.pose.y, 3) : '-';
    row.insertCell().textContent = item.pose ? `${fixed(item.pose.rotation * 180 / Math.PI, 4)} deg` : '-';
    row.insertCell().textContent = (item.mode === 'window' ? (item.success ? `NCC ${fixed(item.score, 3)}` : item.reason) :
      item.raw ? `${item.raw.points} / ${fixed(item.raw.rms, 3)} px` : item.reason);
    row.insertCell().textContent = Number.isFinite(item.sharpness?.score) ? fixed(item.sharpness.score, 2) : '-';
    const stats = referenceCounts.get(item.frame) ?? { referencedBy: null, references: null };
    const referencedBy = row.insertCell(); referencedBy.textContent = stats.referencedBy ?? '-';
    referencedBy.title = stats.referencedBy === null ? 'Keine Umfeldreferenzdiagnose gespeichert' : `Von anderen Frames angefordert: ${stats.referencedBy}; davon im Konsens: ${stats.consensusBy}`;
    const references = row.insertCell(); references.textContent = stats.references ?? '-';
    references.title = stats.references === null ? 'Keine Umfeldreferenzdiagnose gespeichert' : `Eigene angeforderte Umfeldreferenzen: ${stats.references}; davon im Konsens: ${stats.consensus}`;
    row.onclick = () => void showTrackedFrame(item).catch(error => message(error.message, true));
    row.onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); void showTrackedFrame(item).catch(error => message(error.message, true)); } };
  }
  drawTrackingPath();
}

function trackingOptions() {
  const patchSize = number('trackingPatchSize', 64);
  const patchSearchRadius = number('trackingSearchRadius', 32);
  const threshold = number('trackingThreshold', 16);
  const windowSize = number('trackingWindow', 1);
  const maximumSearchRadius = windowTracking() ? 1024 : 128;
  if (!Number.isFinite(patchSearchRadius) || patchSearchRadius < 4 || patchSearchRadius > maximumSearchRadius) {
    throw new Error(`Suchradius muss zwischen 4 und ${maximumSearchRadius} px liegen.`);
  }
  if (!(windowSize >= 1 && windowSize <= 240)) throw new Error('Stabilisierungsfenster muss zwischen 1 und 240 Frames liegen.');
  if (!windowTracking() && !(patchSize >= 16 && patchSize <= 256 && Number.isFinite(threshold) && threshold > 0)) {
    throw new Error('Patchgroesse muss zwischen 16 und 256 px liegen; Mindestkontrast muss positiv und endlich sein.');
  }
  const maxRotation = number('trackingMaxRotation', 5);
  const contextRecent = number('trackingContextRecent', 2); const contextSpatial = number('trackingContextSpatial', 2);
  const contextRadius = number('trackingContextRadius', 32); const contextAngle = number('trackingContextAngle', 1);
  const contextCycleStrict = number('trackingContextCycleStrict', 1.5);
  const contextCycleConditional = number('trackingContextCycleConditional', 7.5);
  if (windowTracking() && (![contextRecent, contextSpatial].every(value => Number.isInteger(value) && value >= 0 && value <= 8) ||
    !Number.isFinite(contextRadius) || contextRadius < 4 || contextRadius > 256 || !Number.isFinite(contextAngle) || contextAngle < 0.1 || contextAngle > 5 ||
    !Number.isFinite(contextCycleStrict) || contextCycleStrict <= 0 || !Number.isFinite(contextCycleConditional) ||
    contextCycleConditional < contextCycleStrict || contextCycleConditional > 50)) {
    throw new Error('Umfeld: n/m 0 bis 8, Radius 4 bis 256 px, Winkel 0.1 bis 5 Grad; Zyklusgrenzen positiv, aufsteigend und maximal 50 px.');
  }
  if (windowTracking() && (!trackingRectangle || !Number.isFinite(maxRotation) || maxRotation < 1 || maxRotation > 10)) throw new Error('Fenster auswaehlen und Rotationsgrenze zwischen 1 und 10 Grad setzen.');
  const sourceImageMask = trackingMaskHasSelection && trackingImageMask ?
    { ...trackingImageMask, coordinateSystem: 'oriented source pixels', data: Array.from(trackingImageMask.data) } : null;
  const imageMask = windowTracking() && sourceImageMask ? remapInclusionMask(trackingImageMask, calibration?.maps) : null;
  return { mode: element('trackingMode').value, rectangle: trackingRectangle ? { ...trackingRectangle } : null, maxRotation,
    contextRecent, contextSpatial, contextRadius, contextAngle, contextCycleStrict, contextCycleConditional,
    sourceImageMask, imageMask: imageMask ? { ...imageMask, data: Array.from(imageMask.data) } : null,
    patchMask: !windowTracking() ? sourceImageMask : null,
    pattern: 'patches', patchSize, patchSearchRadius, threshold, rediscoverPatches: true,
    useWebGpu: element('trackingUseWebGpu').checked };
}

async function runTracking(restart = false) {
  if (!videoInfo || !calibration) throw new Error('Video und passende Linsenkalibrierung werden fuer Tracking benoetigt.');
  if (videoInfo.width !== calibration.field.width || videoInfo.height !== calibration.field.height) throw new Error('Videoaufloesung passt nicht zur Kalibrierung.');
  const start = Math.max(0, Math.min(videoInfo.frameCount - 1, Math.round(number('trackingStart', 0))));
  const end = Math.max(start, Math.min(videoInfo.frameCount - 1, Math.round(number('trackingEnd', videoInfo.frameCount - 1))));
  if (trackingPreviewBusy) return;
  let options = !restart && trackingRunOptions ? trackingRunOptions : trackingOptions();
  if (!restart && trackingRunOptions?.mode === 'window') {
    const changed = trackingOptions();
    options = { ...trackingRunOptions, patchSearchRadius: changed.patchSearchRadius, maxRotation: changed.maxRotation,
      contextRadius: changed.contextRadius, contextAngle: changed.contextAngle,
      contextCycleStrict: changed.contextCycleStrict, contextCycleConditional: changed.contextCycleConditional };
    trackingRunOptions = options;
  }
  message();
  if (restart) {
    clearTrackingResults(true); trackingNextIndex = start;
    trackingRunOptions = options;
    trackingProfile = { mode: options.mode, frames: 0, decodeMs: 0, sharpnessMs: 0, rgbaMs: 0, remapMs: 0, referenceMs: 0, workerTransferMs: 0, trackMs: 0, poseMs: 0,
      renderMs: 0, totalMs: 0, lastTotalMs: 0, accelerators: new Set(), tracker: { accelerator: 'CPU',
        grayscaleMs: 0, contextMs: 0, uploadMs: 0, forwardMs: 0, backwardMs: 0, postprocessMs: 0, sampleMs: 0, matchMs: 0, refinementMs: 0 } };
  }
  trackingInspector.clear();
  video.pause(); trackingRunning = true; trackingPaused = false; updateControls();
  try {
    const nativeTracking = options.mode === 'window' && options.useWebGpu && (options.contextRecent > 0 || options.contextSpatial > 0);
    if (nativeTracking && trackingComputer.nativeMaps !== calibration.maps) {
      const { outputWidth, outputHeight, inverseX, inverseY, valid } = calibration.maps;
      await trackingComputer.call('tracking-maps', { maps: { outputWidth, outputHeight, inverseX, inverseY, valid } });
      trackingComputer.nativeMaps = calibration.maps;
    }
    while (trackingRunning && trackingNextIndex <= end) {
      const frameStarted = performance.now();
      const index = trackingNextIndex;
      text('processingStatus', `Tracking Frame ${index}/${end}`);
      element('progress').value = (index - start + 1) / (end - start + 1);
      const image = await readTrackingFrame(index, { gpu: options.useWebGpu,
        rectified: !nativeTracking && options.mode === 'window', output: nativeTracking ? 'native' : 'rgba' });
      const { decodeMs, sharpnessMs, rgbaMs, remapMs } = image.frameTiming;
      const renderStarted = performance.now();
      if (!nativeTracking) {
        trackingPreviewImage.width = image.width; trackingPreviewImage.height = image.height;
        trackingPreviewImage.getContext('2d').putImageData(new ImageData(image.data, image.width, image.height), 0, 0);
      }
      let renderMs = performance.now() - renderStarted;
      const workerStarted = performance.now();
      let detection, nativeRenderMs = 0;
      if (nativeTracking) {
        try {
          detection = await trackingComputer.call('track-window-native', { native: { frame: image.frame, orientation: image.orientation }, index, options }, [image.frame]);
        } finally { image.frame.close(); }
        const previewStarted = performance.now();
        try {
          trackingPreviewImage.width = detection.width; trackingPreviewImage.height = detection.height;
          trackingPreviewImage.getContext('2d').drawImage(detection.preview, 0, 0);
        } finally { detection.preview?.close(); delete detection.preview; }
        nativeRenderMs = performance.now() - previewStarted; renderMs += nativeRenderMs;
      } else detection = await trackingComputer.call(options.mode === 'window' ? 'track-window' : 'detect', { image, index, options }, [image.data.buffer]);
      let referenceMs = 0;
      if (detection.contextPending) {
        for (const referenceIndex of detection.references) {
          let reference = null; let failure = null;
          const referenceStarted = performance.now();
          try {
            if (!trackingRunning) throw new Error('Pausiert');
            text('processingStatus', `Tracking #${index} | Umfeld #${referenceIndex}`);
            reference = await readTrackingFrame(referenceIndex, { gpu: options.useWebGpu, rectified: !nativeTracking, output: nativeTracking ? 'native' : 'rgba' });
            for (const [destination, source] of [['referenceDecodeMs', 'decodeMs'], ['referenceSharpnessMs', 'sharpnessMs'], ['referenceRgbaMs', 'rgbaMs'], ['referenceRemapMs', 'remapMs']]) {
              trackingProfile[destination] = (trackingProfile[destination] || 0) + (reference.frameTiming?.[source] || 0);
            }
          } catch (error) { failure = error.message; }
          referenceMs += performance.now() - referenceStarted;
          try {
            await trackingComputer.call('context-reference', { index: referenceIndex,
              ...(nativeTracking && reference ? { native: { frame: reference.frame, orientation: reference.orientation } } : { image: reference }),
              error: failure }, reference ? [nativeTracking ? reference.frame : reference.data.buffer] : []);
          } finally { reference?.frame?.close(); }
        }
        detection = await trackingComputer.call('context-finish');
      }
      const workerRoundtripMs = performance.now() - workerStarted - referenceMs - nativeRenderMs;
      const poseStarted = performance.now();
      const raw = options.mode === 'window' ? detection.raw ?? null : detection.success && detection.points.length >= 3 ? fitCameraPose(detection.points, calibration.field) : null;
      const entry = { frame: index, timestamp: videoInfo.timestamps[index], sharpness: image.sharpness, raw, pose: null,
        mode: options.mode, rectangle: options.rectangle, score: detection.score, iterations: detection.iterations, resolution: detection.resolution,
        incremental: detection.incremental, incrementalMatch: detection.incrementalMatch, context: detection.context,
        points: options.mode === 'window' ? (detection.success ? 1 : 0) : detection.points.length, patchPoints: detection.points,
        rediscovered: detection.rediscovered || 0,
        success: detection.success && Boolean(raw), reason: detection.reason || (raw ? '' : 'Zu wenige gueltige Patch-Paare im Kalibrierbereich.'),
        accelerator: detection.accelerator || 'CPU' };
      if (options.mode === 'window' && !detection.success) {
        trackingFailures.push(entry);
        renderTrackingResults(entry); trackingInspector.show(entry);
        trackingPaused = true;
        message(`${detection.reason} Suchgrenze anpassen und denselben Frame mit Fortsetzen erneut versuchen.`, true);
        break;
      }
      if (detection.context?.loopClosure) applyLoopClosure(trackingPath, detection.context.loopClosure);
      trackingPath.push(entry);
      entry.pose = stabilizePose(trackingPath, number('trackingWindow', 1));
      const poseMs = performance.now() - poseStarted;
      const overlayStarted = performance.now();
      drawTrackingPatches(detection.points); renderTrackingResults(entry);
      renderMs += performance.now() - overlayStarted;
      const workerMs = detection.timing?.totalMs ?? workerRoundtripMs;
      const profile = trackingProfile;
      profile.frames++;
      profile.referenceMs = (profile.referenceMs || 0) + referenceMs;
      if (detection.context?.pyramidProfile) {
        profile.pyramidProfile ??= {};
        for (const [key, value] of Object.entries(detection.context.pyramidProfile)) profile.pyramidProfile[key] = (profile.pyramidProfile[key] || 0) + value;
        for (const key of ['cacheHits', 'cacheMisses', 'cacheEvictions']) profile[key] = (profile[key] || 0) + (detection.context[key] || 0);
      }
      profile.decodeMs += decodeMs; profile.sharpnessMs += sharpnessMs; profile.rgbaMs += rgbaMs; profile.remapMs += remapMs; profile.workerTransferMs += Math.max(0, workerRoundtripMs - workerMs);
      profile.trackMs += workerMs; profile.poseMs += poseMs; profile.renderMs += Math.max(0, renderMs);
      profile.lastTotalMs = performance.now() - frameStarted; profile.totalMs += profile.lastTotalMs;
      profile.accelerators.add(image.accelerator || 'CPU');
      profile.accelerators.add(detection.accelerator || 'CPU');
      profile.tracker.accelerator = detection.accelerator || 'CPU';
      for (const key of ['grayscaleMs', 'contextMs', 'contextPyramidMs', 'contextRegistrationMs', 'uploadMs', 'forwardMs', 'backwardMs', 'postprocessMs', 'sampleMs', 'matchMs', 'refinementMs']) {
        profile.tracker[key] = (profile.tracker[key] || 0) + (detection.timing?.[key] || 0);
      }
      renderTrackingProfile(profile);
      trackingNextIndex = index + 1;
      text('pendingStatus', `${end - index} Tracking-Frames verbleiben`);
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  } catch (error) {
    if (options.mode === 'window') trackingLost = true;
    throw error;
  } finally {
    const completed = trackingNextIndex > end;
    trackingRunning = false; trackingPaused = !completed && !trackingLost;
    text('processingStatus', 'Bereit');
    text('pendingStatus', trackingLost ? 'Window Tracking verloren | Neustart erforderlich' : completed ? `Tracking abgeschlossen | ${trackingPath.length} Frames` : `Tracking pausiert vor Frame ${trackingNextIndex}`);
    updateControls();
  }
}

function canvasPoint(event, canvas) {
  const transform = transforms.get(canvas);
  if (!transform) return null;
  const bounds = canvas.getBoundingClientRect();
  const px = (event.clientX - bounds.left - transform.left) / transform.scale;
  const py = (event.clientY - bounds.top - transform.top) / transform.scale;
  return px >= 0 && py >= 0 && px < transform.width && py < transform.height ? { x: px, y: py } : null;
}

function inspect(event, canvas) {
  const point = canvasPoint(event, canvas);
  if (!point || !calibration) return;
  let px = point.x;
  let py = point.y;
  if (canvas === resultCanvas && view === 'rectified') {
    const index = Math.floor(point.y) * calibration.maps.outputWidth + Math.floor(point.x);
    if (!calibration.maps.valid[index]) { text('pixelInfo', 'Ungueltiger Ausgabepixel'); return; }
    px = calibration.maps.inverseX[index]; py = calibration.maps.inverseY[index];
  }
  if (px >= calibration.field.width || py >= calibration.field.height) return;
  const transformed = evaluate(calibration.field, px, py);
  const dx = transformed.x - px;
  const dy = transformed.y - py;
  const count = calibration.maps.sourceCoverage[Math.floor(py) * calibration.field.width + Math.floor(px)];
  text('pixelInfo', `Quelle (${fixed(px, 1)}, ${fixed(py, 1)}) | Ebene (${fixed(transformed.x)}, ${fixed(transformed.y)}) | dx ${fixed(dx)}, dy ${fixed(dy)} | Betrag ${fixed(Math.hypot(dx, dy))} px | ${count} Messframes${count < 3 ? ' / interpoliert' : ''}`);
}

function installPan(canvas) {
  let drag = null;
  const painting = () => canvas === rawCanvas && element('pattern').value === 'patches' && maskTool !== 'pan';
  const paint = event => {
    const point = canvasPoint(event, canvas);
    if (!point) return;
    ensurePatchMask(rawImage.width, rawImage.height);
    const value = maskTool === 'search' ? MASK_SEARCH : maskTool === 'forbidden' ? MASK_FORBIDDEN : MASK_NEUTRAL;
    paintPatchMask(patchMask, point.x, point.y, number('maskBrushSize', 48) / 2, value);
    maskImageRevision = -1;
    draw();
  };
  canvas.addEventListener('pointerdown', event => {
    drag = { x: event.clientX, y: event.clientY, pan: { ...pan }, moved: false, painting: painting() };
    canvas.setPointerCapture(event.pointerId);
    if (drag.painting) paint(event);
  });
  canvas.addEventListener('pointermove', event => {
    inspect(event, canvas);
    if (!drag) return;
    if (drag.painting) { paint(event); return; }
    const deltaX = event.clientX - drag.x;
    const deltaY = event.clientY - drag.y;
    if (Math.hypot(deltaX, deltaY) > 4) drag.moved = true;
    if (drag.moved) {
      const bounds = canvas.getBoundingClientRect();
      pan = { x: drag.pan.x + deltaX / bounds.width, y: drag.pan.y + deltaY / bounds.height };
      draw();
    }
  });
  canvas.addEventListener('pointerup', event => {
    if (drag?.painting) {
      commitMaskEdit();
      drag = null;
      return;
    }
    if (drag && !drag.moved && canvas === resultCanvas && view === 'rectified' && rectifiedReady) {
      const point = canvasPoint(event, canvas);
      if (point && calibration.maps.valid[Math.floor(point.y) * calibration.maps.outputWidth + Math.floor(point.x)]) {
        if (measurements.length === 2) measurements = [];
        measurements.push(point);
        if (measurements.length === 2) {
          const distance = Math.hypot(measurements[1].x - measurements[0].x, measurements[1].y - measurements[0].y);
          const mm = snapshotParameters?.gridMm;
          text('measurement', `${distance.toFixed(2)} px${mm > 0 ? ` / ${(distance * mm / calibration.step).toFixed(4)} mm` : ` / ${(distance / calibration.step).toFixed(3)} Raster`}`);
        }
        draw();
      }
    }
    drag = null;
  });
  canvas.addEventListener('pointercancel', () => { if (drag?.painting) commitMaskEdit(); drag = null; });
  canvas.addEventListener('wheel', event => { event.preventDefault(); zoom = Math.max(0.5, Math.min(20, zoom * Math.exp(-event.deltaY * 0.001))); draw(); }, { passive: false });
}

function updatePlayIcon() {
  const playing = rectifiedPlayback || !video.paused;
  element('playButton').innerHTML = `<i data-lucide="${playing ? 'pause' : 'play'}"></i>`;
  createIcons({ icons, root: element('playButton') });
  element('playButton').setAttribute('aria-label', playing ? 'Pause' : 'Wiedergabe');
}

function playbackFrame(_, metadata) {
  if (!video.paused && videoInfo) {
    currentIndex = closestFrame(videoInfo.firstTimestamp + metadata.mediaTime * 1e6);
    currentDetection = detections.get(currentIndex) || detectionFromFrame(frames.get(currentIndex));
    rectifiedReady = false;
    updateTime(); draw();
  }
  video.requestVideoFrameCallback(playbackFrame);
}
video.requestVideoFrameCallback(playbackFrame);
video.addEventListener('ended', () => { updatePlayIcon(); void showFrame(currentIndex); });
video.addEventListener('error', () => message(`Videovorschau fehlgeschlagen (Medienfehler ${video.error?.code}). Framegenauer WebCodecs-Zugriff kann davon unabhaengig verfuegbar sein.`, true));

async function resetCalibration() {
  await computer.call('reset');
  clearTrackingResults();
  calibration = null; frames = new Map(); snapshotFrames = []; snapshotParameters = null;
  observationDiagnostics = null; observationDiagnosticsDirty = true; geometrySelection = null;
  detections.clear(); currentDetection = null; phaseChecked = false; phaseStarted = false; stale = false; detectionsStale = false;
  detectedOnce = false; acceptedSinceFit = 0; nextProcessingIndex = 0; measurements = []; rectifiedReady = false;
  patchMask = null; maskImageRevision = -1;
  text('measurement', 'Abstand -'); text('detectionStatus', 'Noch nicht geprueft.'); text('qualityDetails', 'Noch keine Kalibrierung.');
  renderExportProfile(null); renderPatchProfile(null); renderFitProfile(null);
  updateMetrics(); updateTable(); updateControls(); draw();
}

element('openVideo').onclick = () => {
  element('videoFile').value = '';
  element('videoFile').click();
};
element('videoFile').onchange = () => void task(async () => {
  const file = element('videoFile').files[0];
  if (!file) return;
  message();
  const candidate = new WorkerClient('/decoder-worker.js', showProgress);
  let metadata;
  try { metadata = await candidate.call('open', { file }); }
  catch (error) { candidate.terminate(); throw error; }
  const videoNameMismatch = expectedVideo && expectedVideo.name !== metadata.name && (calibration || frames.size);
  if (calibration && (metadata.width !== calibration.field.width || metadata.height !== calibration.field.height)) {
    candidate.terminate(); throw new Error('Videoaufloesung passt nicht zur geladenen Kalibrierung.');
  }
  decoder.terminate(); decoder = candidate;
  video.pause(); if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = URL.createObjectURL(file); video.src = videoUrl;
  const retainedTracking = trackingForExport();
  videoInfo = metadata; expectedVideo = { ...metadata, timestamps: undefined };
  trackingImageMask = null; trackingMaskHasSelection = false; trackingMaskPreview.width = trackingMaskPreview.height = 0;
  element('trackingStart').max = metadata.frameCount - 1; element('trackingEnd').max = metadata.frameCount - 1;
  element('trackingEnd').value = metadata.frameCount - 1; text('trackingVideoName', metadata.name);
  clearTrackingResults();
  if (retainedTracking?.video?.name === metadata.name && retainedTracking.video.width === metadata.width && retainedTracking.video.height === metadata.height) {
    restoreTracking(retainedTracking);
  }
  rawReady = false; currentIndex = 0; zoom = 1; pan = { x: 0, y: 0 };
  element('timeline').max = metadata.frameCount - 1;
  text('videoName', metadata.name);
  renderVideoMetadata();
  const first = await readFrame(0);
  videoInfo.color = first.color; renderVideoMetadata(); setRaw(first.bitmap, 0, first.sharpness); first.bitmap.close();
  if (frames.size) phaseChecked = [...frames.values()].some(frame => frame.accepted);
  updateTable(); text('processingStatus', videoNameMismatch ?
    'Video geladen; Dateiname weicht von der Kalibrierung ab, vorhandene Daten bleiben erhalten' :
    'Video geladen; keine automatische Kalibrierung');
  setTimeout(() => { void loadTrackingMaskPreview(); }, 0);
});

element('detectButton').onclick = () => void task(async () => { message(); await detectIndex(currentIndex); });
element('startButton').onclick = () => { nextProcessingIndex = currentIndex; void runContinuous(); };
element('pauseButton').onclick = () => {
  cancelRequested = true;
  continuous = false;
  text('pendingStatus', 'Abbruch angefordert; aktueller Rechenschritt wird abgeschlossen');
  void computer.call('cancel').catch(() => {});
  updateControls();
};
element('resumeButton').onclick = () => { nextProcessingIndex = Math.max(nextProcessingIndex, currentIndex); void runContinuous(); };
element('evaluateButton').onclick = () => void task(async () => { cancelRequested = false; phaseStarted = true; await evaluateIndex(currentIndex); await refit(); });
element('evaluateNextButton').onclick = () => void task(async () => { cancelRequested = false; phaseStarted = true; await evaluateIndex(Math.min(videoInfo.frameCount - 1, currentIndex + 1)); await refit(); });
element('refitButton').onclick = () => { cancelRequested = false; void task(refit); };
element('resetButton').onclick = () => { if (confirm('Kalibrierung und alle Beobachtungen zuruecksetzen? Das geladene Video bleibt erhalten.')) void task(resetCalibration); };
element('firstFrame').onclick = () => void showFrame(0).catch(error => message(error.message, true));
element('previousFrame').onclick = () => void showFrame(currentIndex - 1).catch(error => message(error.message, true));
element('nextFrame').onclick = () => void showFrame(currentIndex + 1).catch(error => message(error.message, true));
element('timeline').oninput = () => { if (!navigationBusy) void showFrame(Number(element('timeline').value)).catch(error => message(error.message, true)); };
element('playButton').onclick = async () => {
  if (rectifiedPlayback) {
    rectifiedPlayback = false;
    updatePlayIcon();
    return;
  }
  if (view === 'rectified' && calibration) {
    void playRectified();
    return;
  }
  if (video.paused) {
    video.currentTime = (videoInfo.timestamps[currentIndex] - videoInfo.firstTimestamp) / 1e6;
    try { await video.play(); } catch (error) { message(`Wiedergabe nicht moeglich: ${error.message}`, true); }
  } else { video.pause(); await showFrame(currentIndex); }
  updatePlayIcon();
};
element('speed').onchange = () => { video.playbackRate = Number(element('speed').value); saveVideoOptions(); };
element('overlay').onchange = () => { saveVideoOptions(); draw(); };
element('trackingStart').onchange = () => { void previewTrackingStartFrame(); void loadTrackingMaskPreview(); };
element('trackingMode').onchange = () => {
  clearTrackingResults();
  element('trackingWindowControls').hidden = !windowTracking();
  for (const id of ['trackingPatchSize', 'trackingThreshold']) element(id).closest('label').hidden = windowTracking();
  void previewTrackingStartFrame();
  void loadTrackingMaskPreview();
};
element('previewBrightness').closest('details').addEventListener('toggle', event => {
  if (event.currentTarget.open) void loadTrackingMaskPreview();
});
element('trackingSelectWindow').onclick = () => {
  selectingTrackingWindow = !selectingTrackingWindow;
  element('trackingSelectWindow').setAttribute('aria-pressed', String(selectingTrackingWindow));
};
function updatePreviewAdjustments(save = true) {
  const brightness = number('previewBrightness', 0) / 100;
  const contrast = number('previewContrast', 100) / 100;
  const gamma = number('previewGamma', 1);
  const intercept = 0.5 * (1 - contrast) + brightness;
  for (const channel of ['Red', 'Green', 'Blue']) {
    element(`preview${channel}Level`).setAttribute('slope', contrast);
    element(`preview${channel}Level`).setAttribute('intercept', intercept);
    element(`preview${channel}Gamma`).setAttribute('exponent', 1 / gamma);
  }
  text('previewBrightnessValue', `${brightness > 0 ? '+' : ''}${element('previewBrightness').value}`);
  text('previewContrastValue', `${element('previewContrast').value}%`);
  text('previewGammaValue', gamma.toFixed(2));
  draw();
  renderTrackingPreview();
  renderPathOverlay();
  renderTrackingMask();
  if (element('trackingPatchDialog').open) renderTrackingPatchDetail();
  if (!trackingPreviewImage.width && videoInfo && !taskBusy && !continuous && !rectifiedPlayback && !trackingRunning) {
    void previewTrackingStartFrame();
  }
  if (save) saveVideoOptions();
}
for (const id of ['previewBrightness', 'previewContrast', 'previewGamma']) element(id).oninput = updatePreviewAdjustments;
element('resetPreviewAdjustments').onclick = () => {
  element('previewBrightness').value = 0;
  element('previewContrast').value = 100;
  element('previewGamma').value = 1;
  updatePreviewAdjustments();
};
for (const button of document.querySelectorAll('[data-mask-tool]')) button.onclick = () => {
  maskTool = button.dataset.maskTool;
  for (const tool of document.querySelectorAll('[data-mask-tool]')) tool.setAttribute('aria-pressed', String(tool === button));
  rawCanvas.style.cursor = maskTool === 'pan' ? 'grab' : 'crosshair';
};
element('maskBrushSize').oninput = () => text('maskBrushValue', `${element('maskBrushSize').value} px`);
element('clearPatchMask').onclick = () => {
  const width = videoInfo?.width ?? calibration?.field.width;
  const height = videoInfo?.height ?? calibration?.field.height;
  if (!width || !height) return;
  ensurePatchMask(width, height).data.fill(MASK_NEUTRAL);
  patchMask.revision++;
  maskImageRevision = -1;
  commitMaskEdit();
};
element('zoomIn').onclick = () => { zoom = Math.min(20, zoom * 1.25); draw(); };
element('zoomOut').onclick = () => { zoom = Math.max(0.5, zoom / 1.25); draw(); };
element('fitView').onclick = () => { zoom = 1; pan = { x: 0, y: 0 }; draw(); };
element('trackingZoomIn').onclick = () => { trackingZoom = Math.min(20, trackingZoom * 1.25); renderTrackingPreview(); };
element('trackingZoomOut').onclick = () => { trackingZoom = Math.max(0.5, trackingZoom / 1.25); renderTrackingPreview(); };
element('trackingFitView').onclick = () => { trackingZoom = 1; trackingPan = { x: 0, y: 0 }; renderTrackingPreview(); };
element('closeTrackingPatch').onclick = () => element('trackingPatchDialog').close();
element('colorRange').oninput = element('hideRigid').onchange = () => { renderFieldImages(); draw(); };
element('relativeCoverage').onchange = () => { renderCoverageImage(); draw(); };
element('dataMode').onchange = () => { renderObservationDiagnostics(); draw(); };
element('refreshData').onclick = () => { renderObservationDiagnostics(true); draw(); };
element('selectConsistent').onclick = () => {
  const candidates = [...frames.values()].filter(frame => frame.accepted && frame.points?.length >= 3);
  const result = selectConsistentFrames(candidates, number('step', 1) || 1,
    number('dataScaleTolerance', 1) / 100, number('dataTiltTolerance', 1) / 100);
  if (!result.center) { message('Keine auswertbaren Frames fuer die Geometrieauswahl.', true); return; }
  for (const frame of frames.values()) frame.enabled = result.selectedIds.has(frame.id);
  geometrySelection = { ...result, candidates: new Map(result.candidates.map(candidate => [candidate.id, candidate])) };
  stale = Boolean(calibration); observationDiagnosticsDirty = true;
  updateTable(); updateMetrics(); updateControls(); renderObservationDiagnostics(true); draw();
  message(`${result.selectedIds.size} von ${candidates.length} Frames gewaehlt | Clusterskala ${fixed(result.center.scale, 5)} | Anisotropie ${fixed(result.center.anisotropy, 5)}. Zum Anwenden neu fitten.`);
};
element('enableAccepted').onclick = () => {
  for (const frame of frames.values()) frame.enabled = frame.accepted;
  geometrySelection = null; stale = Boolean(calibration); observationDiagnosticsDirty = true;
  updateTable(); updateMetrics(); updateControls(); renderObservationDiagnostics(true); draw();
  message('Alle brauchbaren Frames sind wieder aktiv. Zum Anwenden neu fitten.');
};
element('height3d').oninput = () => { text('height3dValue', Number(element('height3d').value).toFixed(1)); renderFieldImages(); draw(); };
element('clearMeasurement').onclick = () => { measurements = []; text('measurement', 'Abstand -'); draw(); };
for (const button of document.querySelectorAll('[data-workflow]')) button.onclick = () => {
  workflow = button.dataset.workflow;
  for (const tab of document.querySelectorAll('[data-workflow]')) tab.setAttribute('aria-selected', String(tab === button));
  for (const panel of document.querySelectorAll('[data-workflow-panel]')) panel.hidden = panel.dataset.workflowPanel !== workflow;
  element('timeProfiles').hidden = false;
  if (workflow === 'tracking') {
    drawTrackingPath(); updateTrackingControls();
    if (!trackingPreviewImage.width) void previewTrackingStartFrame();
    if (!trackingMaskPreview.width) void loadTrackingMaskPreview();
  }
};
element('trackingStartButton').onclick = () => void runTracking(true).catch(error => {
  message(error.message, true); trackingRunning = false; updateControls();
});
element('trackingPauseButton').onclick = () => { trackingRunning = false; trackingPaused = true; updateControls(); };
element('trackingResumeButton').onclick = () => void runTracking(false).catch(error => {
  message(error.message, true); trackingRunning = false; updateControls();
});
element('trackingResetButton').onclick = () => { clearTrackingResults(); void previewTrackingStartFrame(); void loadTrackingMaskPreview(); };
element('trackingExportButton').onclick = () => {
  const header = 'frame,timestamp_us,x_px,y_px,rotation_deg,raw_x_px,raw_y_px,raw_rotation_deg,patches,rms_px';
  const rows = trackingPath.filter(entry => entry.pose).map(entry => [entry.frame, entry.timestamp, entry.pose.x, entry.pose.y,
    entry.pose.rotation * 180 / Math.PI, entry.raw?.x ?? '', entry.raw?.y ?? '', entry.raw ? entry.raw.rotation * 180 / Math.PI : '', entry.raw?.points ?? '', entry.raw?.rms ?? ''].join(','));
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([[header, ...rows].join('\n')], { type: 'text/csv' }));
  link.download = `${(videoInfo?.name || 'tracking').replace(/\.[^.]+$/, '')}-movement.csv`; link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 0);
};
for (const button of document.querySelectorAll('[data-view]')) button.onclick = () => {
  if (rectifiedPlayback && button.dataset.view !== 'rectified') rectifiedPlayback = false;
  view = button.dataset.view; displayRevision++;
  for (const tab of document.querySelectorAll('[data-view]')) tab.setAttribute('aria-selected', String(tab === button));
  text('resultTitle', { field: 'Dichtes Offsetfeld', field3d: '3D-Verschiebungsfeld', rectified: 'Entzerrtes Bild', data: 'Datenlage vor dem Fit', coverage: 'Messabdeckung', residual: 'Raeumliche Restfehler' }[view]);
  element('fieldLegend').hidden = view !== 'field'; element('rigidControl').hidden = !['field', 'field3d'].includes(view); element('height3dControl').hidden = view !== 'field3d'; element('otherLegend').hidden = view === 'field';
  element('dataControls').hidden = view !== 'data';
  element('coverageControl').hidden = view !== 'coverage';
  element('coverageLegend').hidden = view !== 'coverage';
  if (view === 'coverage') element('otherLegend').hidden = true;
  text('otherLegend', view === 'coverage' ? `Heatmap ${calibration?.maps.coverageGrid?.cols ?? 24} x ${calibration?.maps.coverageGrid?.rows ?? 18} | Gruen: >=3 Messframes | Gelb: 1-2 | Grau: unbeobachtet` : view === 'residual' ?
    `Mittlerer Punktfehler je Heatmap-Zelle | Rot: >=${snapshotParameters?.acceptance || 1} px | Grau: keine Messwerte | ${calibration?.metrics.validation.count ? 'zurueckgehaltene Punkte' : 'nur Trainingsdaten'}` : view === 'field3d' ? 'Hoehe: |d| relativ zum R/B-Bereich | Farbe: dx Rot, dy Blau | Luecken: unzureichend abgedeckt' : 'Maskierte bilineare Rueckabbildung | Abstand in Ausgabepixeln');
  if (view === 'data') renderObservationDiagnostics();
  if (view === 'rectified' && calibration && rawReady && !taskBusy) void task(updateRectified);
  updateTable();
  draw();
};

const detectionParameters = new Set(['pattern', 'patchSize', 'patchSearchRadius', 'approxStep', 'columns', 'rows', 'threshold', 'lineRadius']);
const nonGeometric = new Set(['follow', 'opticalConfiguration', 'minMotion', 'updateEvery', 'startPercent', 'endPercent', 'maskBrushSize']);
for (const control of document.querySelectorAll('.settings input, .settings select')) control.addEventListener('change', () => {
  if (control.id === 'pattern') {
    updatePatternControls();
  }
  if (nonGeometric.has(control.id)) return;
  parameterRevision++; stale = Boolean(calibration); observationDiagnosticsDirty = true;
  if (detectionParameters.has(control.id)) { detectionsStale = frames.size > 0; phaseChecked = false; detections.clear(); }
  if (control.id === 'validationFrom' && videoInfo) {
    const boundary = videoInfo.firstTimestamp + videoInfo.duration * 1e6 * number('validationFrom', 80) / 100;
    for (const frame of frames.values()) frame.role = frame.timestamp >= boundary ? 'validation' : 'train';
    updateTable();
  }
  updateMetrics(); updateControls();
  if (calibration) message('Parameter geaendert. Sichtbar bleibt die letzte konsistente Feldversion; Neu fitten aktualisiert die Kalibrierung.');
  clearTimeout(timer);
  if (detectedOnce && videoInfo && detectionParameters.has(control.id)) timer = setTimeout(() => { if (!taskBusy && !continuous) void task(() => detectIndex(currentIndex)); }, 250);
});

element('exportButton').onclick = () => void task(async () => {
  if (stale && !confirm(`Die aktuellen Eingaben sind noch nicht eingerechnet. Konsistente Feldversion ${calibration.version} mit ihren damaligen Parametern und Beobachtungen speichern?`)) return;
  const tracking = trackingForExport();
  const bytes = await computer.call('export', { frames: snapshotFrames, video: expectedVideo, parameters: snapshotParameters,
    opticalConfiguration: element('opticalConfiguration').value, tracking });
  const url = URL.createObjectURL(new Blob([bytes], { type: 'application/zip' }));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = `kalibrierung-v${calibration.version}-${calibration.quality}.zip`; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
});
element('correctedVideoButton').onclick = () => void task(async () => {
  message();
  video.pause();
  updatePlayIcon();
  const restoreIndex = currentIndex;
  const fps = videoInfo.fps || 30;
  if (!globalThis.showSaveFilePicker) throw new Error('Blockweises Speichern benoetigt die File System Access API in Edge.');
  let handle;
  try {
    handle = await showSaveFilePicker({ suggestedName: `${videoInfo.name.replace(/\.[^.]+$/, '')}-korrigiert.mp4`,
      types: [{ description: 'H.264/MP4-Video', accept: { 'video/mp4': ['.mp4'] } }] });
  } catch (error) {
    if (error.name === 'AbortError') return;
    throw error;
  }
  const writable = await handle.createWritable();
  let exporter;
  try {
    exporter = await VideoExporter.create({ width: calibration.maps.outputWidth, height: calibration.maps.outputHeight, fps, writable });
  } catch (error) {
    await writable.abort();
    throw error;
  }
  if (exporter.scaled) text('pendingStatus', `H.264-Ausgabe proportional skaliert: ${exporter.sourceWidth} x ${exporter.sourceHeight} -> ${exporter.config.width} x ${exporter.config.height} px`);
  const profile = { frames: 0, decodeMs: 0, sharpnessMs: 0, rgbaMs: 0, remapComputeMs: 0, remapTransferMs: 0,
    gpuContextMs: 0, gpuUploadMs: 0, gpuCommandMs: 0, gpuCompletionMs: 0, gpuReadbackMs: 0,
    frameMs: 0, submitMs: 0, backpressureMs: 0, totalMs: 0,
    width: exporter.config.width, height: exporter.config.height };
  renderExportProfile(null);
  try {
    for (let index = 0; index < videoInfo.frameCount; index++) {
      const frameStarted = performance.now();
      text('processingStatus', `Videoexport | Frame ${index + 1}/${videoInfo.frameCount}`);
      element('progress').value = index / videoInfo.frameCount;
      const remapped = await readFrame(index, { rectified: true });
      profile.decodeMs += remapped.frameTiming.decodeMs;
      profile.sharpnessMs += remapped.frameTiming.sharpnessMs;
      profile.rgbaMs += remapped.frameTiming.rgbaMs;
      profile.remapComputeMs += remapped.frameTiming.remapMs;
      profile.remapAccelerator = remapped.accelerator;
      if (remapped.timing) {
        profile.gpuContextMs += remapped.timing.contextMs;
        profile.gpuUploadMs += remapped.timing.uploadMs;
        profile.gpuCommandMs += remapped.timing.commandMs;
        profile.gpuCompletionMs += remapped.timing.completionMs;
        profile.gpuReadbackMs += remapped.timing.readbackMs;
      }
      const timestamp = videoInfo.timestamps[index] - videoInfo.firstTimestamp;
      const nextTimestamp = index + 1 < videoInfo.frameCount ? videoInfo.timestamps[index + 1] - videoInfo.firstTimestamp : timestamp + Math.round(1e6 / fps);
      let encoderTiming;
      try { encoderTiming = await exporter.addFrame(remapped.bitmap, remapped.width, remapped.height, timestamp, nextTimestamp - timestamp, index); }
      finally { remapped.bitmap.close(); }
      profile.frameMs += encoderTiming.frameMs;
      profile.submitMs += encoderTiming.submitMs;
      profile.backpressureMs += encoderTiming.backpressureMs;
      profile.directFrames = encoderTiming.direct;
      profile.frames++;
      profile.totalMs += performance.now() - frameStarted;
      if (profile.frames === 1 || profile.frames % 10 === 0) renderExportProfile(profile);
    }
    const finalizeStarted = performance.now();
    await exporter.finish();
    profile.finalizeMs = performance.now() - finalizeStarted;
    renderExportProfile(profile);
  } catch (error) {
    await exporter.cancel();
    throw error;
  }
  element('progress').value = 1;
  text('pendingStatus', `${videoInfo.frameCount} Frames blockweise als H.264/MP4 gespeichert`);
  await showFrame(restoreIndex);
});
element('importButton').onclick = () => {
  element('calibrationFile').value = '';
  element('calibrationFile').click();
};
element('calibrationFile').onchange = () => void task(async () => {
  const file = element('calibrationFile').files[0]; if (!file) return;
  if (calibration && !confirm('Aktuelle Kalibrierung durch das ausgewaehlte Paket ersetzen?')) return;
  if (file.size > 768 * 1024 * 1024) throw new Error('Paket groesser als 768 MiB.');
  const bytes = new Uint8Array(await file.arrayBuffer());
  const imported = await computer.call('import', { bytes }, [bytes.buffer]);
  const videoNameMismatch = videoInfo && imported.video?.name !== videoInfo.name;
  calibration = imported.calibration;
  clearTrackingResults();
  restoreTracking(imported.tracking);
  applyParameters(imported.parameters);
  const invalidMaskPoints = revalidatePatchFrames(imported.observations, patchMask);
  frames = new Map(imported.observations.map(frame => [frame.id, frame]));
  observationDiagnostics = null; observationDiagnosticsDirty = true; geometrySelection = null;
  snapshotFrames = imported.observations; snapshotParameters = imported.parameters; expectedVideo = imported.video;
  element('opticalConfiguration').value = imported.opticalConfiguration || '';
  phaseStarted = true; phaseChecked = [...frames.values()].some(frame => frame.accepted);
  stale = invalidMaskPoints > 0; detectionsStale = false; rectifiedReady = false; detections.clear(); currentDetection = null;
  nextProcessingIndex = 0; measurements = [];
  renderFieldImages(); updateTable(); updateMetrics(); updateControls(); draw();
  message(invalidMaskPoints ? `Kalibrierung geladen. ${invalidMaskPoints} gespeicherte Patch-Vektoren beruehrten die rote Maske und wurden verworfen; neu fitten erforderlich.` :
    videoNameMismatch ? 'Kalibrierung geladen. Der Videodateiname weicht ab; Video und Kalibrierungsdaten bleiben erhalten.' :
    'Kalibrierung geladen. Feld und Metadaten sind lokal verfuegbar.');
  setTimeout(() => { void loadTrackingMaskPreview(); }, 0);
});
element('infoButton').onclick = () => element('infoDialog').showModal();
element('closeInfo').onclick = () => element('infoDialog').close();
installPan(rawCanvas); installPan(resultCanvas);
installTrackingPreviewInteraction();
installPathInteraction();
installTrackingMaskInteraction();
new ResizeObserver(draw).observe(element('rawCanvas').parentElement);
new ResizeObserver(draw).observe(element('resultCanvas').parentElement);
new ResizeObserver(renderTrackingPreview).observe(element('trackingPreviewCanvas').parentElement);
window.addEventListener('beforeunload', event => { if (frames.size && !calibration) { event.preventDefault(); event.returnValue = ''; } });
restoreVideoOptions();
video.playbackRate = Number(element('speed').value);
text('trackingMaskBrushValue', `${element('trackingMaskBrush').value} px`);
updatePreviewAdjustments(false);
updatePatternControls();
updateControls(); updateMetrics(); draw();
setInterval(updateMemoryStats, 2000);
