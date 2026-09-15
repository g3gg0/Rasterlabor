import './style.css';
import { createIcons, icons } from 'lucide';
import { WorkerClient } from './rpc.js';
import { evaluate, fieldColor } from './spline.js';
import { poseFor } from './solver.js';
import { Field3DView } from './field-3d.js';
import { VideoExporter } from './video-exporter.js';
import { relativeCoverageScale } from './maps.js';
import { createPatchMask, MASK_FORBIDDEN, MASK_NEUTRAL, MASK_SEARCH, paintPatchMask, patchAllowed, validatePointsAgainstMask } from './patch-mask.js';
import { analyzeObservations, selectConsistentFrames } from './observation-diagnostics.js';

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
const conversionCanvas = new OffscreenCanvas(1, 1);
const conversionContext = conversionCanvas.getContext('2d', { willReadFrequently: true });
let decoder = new WorkerClient('/decoder-worker.js', showProgress);
const computer = new WorkerClient('/compute-worker.js', showProgress);
let videoInfo = null;
let expectedVideo = null;
let videoUrl = null;
let currentIndex = 0;
let calibration = null;
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
const number = (id, fallback) => element(id).value === '' ? fallback : Number(element(id).value);
const text = (id, value) => { element(id).textContent = value; };
const fixed = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : '-';
const megabytes = bytes => `${(bytes / 1024 / 1024).toFixed(bytes >= 100 * 1024 * 1024 ? 0 : 1)} MB`;

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
  text('processingStatus', progress.stage === 'fit' ? `Fit ${progress.iteration}/${progress.iterations} | RMS ${fixed(progress.rms)} px${accelerator}${speed}` :
    `${names[progress.stage] || progress.stage}${accelerator}`);
  element('progress').value = progress.stage === 'fit' ? progress.iteration / progress.iterations : (progress.done || 0) / (progress.total || 1);
  if (progress.stage === 'fit') renderFitProfile(progress.profile);
}

function updateTimeProfiles() {
  const summaries = [];
  const exportSection = element('exportProfileSection');
  const patchSection = element('patchProfileSection');
  const fitSection = element('fitProfileSection');
  if (!exportSection.hidden) summaries.push(exportSection.dataset.summary);
  if (!patchSection.hidden) summaries.push(patchSection.dataset.summary);
  if (!fitSection.hidden) summaries.push(fitSection.dataset.summary);
  text('timeProfilesSummary', summaries.length ? `Zeitprofile | ${summaries.join(' | ')}` : 'Zeitprofile');
}

function renderExportProfile(profile) {
  const section = element('exportProfileSection');
  section.hidden = !profile?.frames;
  if (!profile?.frames) { updateTimeProfiles(); return; }
  const average = value => value / profile.frames;
  const rows = [
    ['Quelldecodierung', average(profile.decodeMs)], ['Bitmap nach RGBA', average(profile.rgbaMs)],
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
    frame.points = frame.points.filter(point => point.maskValid !== false);
    if (frame.points.length < 6) {
      frame.enabled = false;
      frame.accepted = false;
      frame.reason = 'Zu wenige Patches ausserhalb der verbotenen Maskenflaeche';
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

function updateControls() {
  const hasVideo = Boolean(videoInfo);
  for (const id of ['firstFrame', 'previousFrame', 'nextFrame', 'timeline']) element(id).disabled = !hasVideo || navigationBusy || rectifiedPlayback;
  element('playButton').disabled = !hasVideo || (navigationBusy && !rectifiedPlayback);
  element('detectButton').disabled = !hasVideo || taskBusy || continuous || rectifiedPlayback;
  element('startButton').disabled = !hasVideo || !phaseChecked || taskBusy || phaseStarted || rectifiedPlayback;
  element('pauseButton').disabled = !continuous && !revalidating;
  element('pauseButton').querySelector('span').textContent = revalidating ? 'Abbrechen' : 'Pause';
  element('resumeButton').disabled = !hasVideo || !phaseStarted || continuous || taskBusy || nextProcessingIndex >= (videoInfo?.frameCount || 0);
  element('evaluateButton').disabled = !hasVideo || !phaseChecked || taskBusy || continuous || rectifiedPlayback;
  element('evaluateNextButton').disabled = !hasVideo || !phaseChecked || taskBusy || continuous || rectifiedPlayback || currentIndex >= (videoInfo?.frameCount || 0) - 1;
  element('refitButton').disabled = taskBusy || continuous || rectifiedPlayback || [...frames.values()].filter(frame => frame.enabled && frame.role === 'train').length < 2 || (detectionsStale && !hasVideo);
  element('exportButton').disabled = !calibration || taskBusy || continuous || rectifiedPlayback;
  element('correctedVideoButton').disabled = !calibration || !hasVideo || taskBusy || continuous || rectifiedPlayback;
  element('importButton').disabled = taskBusy || continuous || rectifiedPlayback;
  element('openVideo').disabled = taskBusy || continuous || rectifiedPlayback;
  element('resetButton').disabled = taskBusy || continuous || rectifiedPlayback;
  for (const control of document.querySelectorAll('.settings input, .settings select, .settings textarea')) {
    if (control.id !== 'follow' && control.id !== 'opticalConfiguration') control.disabled = taskBusy || continuous;
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
  text('rawState', `${video.paused ? 'Framegenau' : 'Wiedergabe'} | #${currentIndex}`);
  element('timeline').value = currentIndex;
  text('elapsed', timeLabel((timestamp - videoInfo.firstTimestamp) / 1e6));
}

function imageFromBitmap(bitmap) {
  if (conversionCanvas.width !== bitmap.width || conversionCanvas.height !== bitmap.height) {
    conversionCanvas.width = bitmap.width;
    conversionCanvas.height = bitmap.height;
  }
  conversionContext.drawImage(bitmap, 0, 0);
  return conversionContext.getImageData(0, 0, bitmap.width, bitmap.height);
}

function setRaw(bitmap, index) {
  rawImage.width = bitmap.width;
  rawImage.height = bitmap.height;
  rawImage.getContext('2d').drawImage(bitmap, 0, 0);
  rawReady = true;
  if (element('pattern').value === 'patches') ensurePatchMask(bitmap.width, bitmap.height);
  currentIndex = index;
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
    success: frame.accepted, reason: frame.reason, accelerator: frame.accelerator, timing: frame.timing };
}

async function showFrame(index) {
  if (!videoInfo || navigationBusy) return;
  navigationBusy = true;
  video.pause();
  updatePlayIcon();
  updateControls();
  try {
    const result = await decoder.call('frame', { index: Math.max(0, Math.min(videoInfo.frameCount - 1, index)) });
    setRaw(result.bitmap, result.index);
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
      const result = await decoder.call('frame', { index });
      setRaw(result.bitmap, result.index);
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
  const result = await decoder.call('frame', { index });
  if (display) setRaw(result.bitmap, index);
  const settings = parameters();
  let detection;
  if (settings.pattern === 'chessboard' && settings.useWebGpu) {
    detection = await computer.call('detect-bitmap', { bitmap: result.bitmap, index, options: settings }, [result.bitmap]);
  } else {
    const image = imageFromBitmap(result.bitmap);
    result.bitmap.close();
    detection = await computer.call('detect', { image, index, options: settings }, [image.data.buffer]);
  }
  if (revision !== parameterRevision) throw new Error('Parameter wurden waehrend der Erkennung geaendert.');
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
      const result = await decoder.call('frame', { index });
      setRaw(result.bitmap, index);
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
    timing: detection.timing,
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
    row.insertCell().textContent = geometry ? `${geometry.selected ? 'Geometrie passend' : 'Geometrie abweichend'} | Skala ${fixed(geometry.scale, 4)} | Anisotropie ${fixed(geometry.anisotropy, 4)}` : frame.enabled && calibration && !stale ?
      (frame.role === 'train' ? 'Im Fit verwendet' : 'Zur Validierung verwendet') : frame.reason;
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
  const image = rawImage.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, rawImage.width, rawImage.height);
  const result = await computer.call('remap', { image, useWebGpu: element('useWebGpu').checked }, [image.data.buffer]);
  if (frameIndex !== currentIndex || revision !== displayRevision) return;
  rectifiedImage.width = result.width; rectifiedImage.height = result.height;
  rectifiedImage.getContext('2d').putImageData(new ImageData(result.data, result.width, result.height), 0, 0);
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
    context.save();
    if (adjustImage) context.filter = previewFilter;
    context.drawImage(image, 0, 0, width, height);
    context.restore();
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
    context.strokeStyle = '#ffd74b'; context.fillStyle = '#ffd74b'; context.lineWidth = 1.5 / scale;
    context.beginPath(); measurements.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y)); context.stroke();
    for (const point of measurements) { context.beginPath(); context.arc(point.x, point.y, 3 / scale, 0, Math.PI * 2); context.fill(); }
  } : null, view === 'rectified');
  else drawCanvas(resultCanvas, null, 0, 0);
  text('zoomValue', `${Math.round(zoom * 100)}%`);
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
  calibration = null; frames = new Map(); snapshotFrames = []; snapshotParameters = null;
  observationDiagnostics = null; observationDiagnosticsDirty = true; geometrySelection = null;
  detections.clear(); currentDetection = null; phaseChecked = false; phaseStarted = false; stale = false; detectionsStale = false;
  detectedOnce = false; acceptedSinceFit = 0; nextProcessingIndex = 0; measurements = []; rectifiedReady = false;
  patchMask = null; maskImageRevision = -1;
  text('measurement', 'Abstand -'); text('detectionStatus', 'Noch nicht geprueft.'); text('qualityDetails', 'Noch keine Kalibrierung.');
  renderExportProfile(null); renderPatchProfile(null); renderFitProfile(null);
  updateMetrics(); updateTable(); updateControls(); draw();
}

element('openVideo').onclick = () => element('videoFile').click();
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
  videoInfo = metadata; expectedVideo = { ...metadata, timestamps: undefined };
  rawReady = false; currentIndex = 0; zoom = 1; pan = { x: 0, y: 0 };
  element('timeline').max = metadata.frameCount - 1;
  text('videoName', metadata.name);
  text('videoMetadata', `${metadata.width} x ${metadata.height} | ${metadata.duration.toFixed(2)} s | ${metadata.frameCount} Frames | ${fixed(metadata.fps, 3)} fps${metadata.variableFrameRate ? ' (variabel, Mittel)' : ''} | ${metadata.codec}`);
  const first = await decoder.call('frame', { index: 0 }); setRaw(first.bitmap, 0); first.bitmap.close();
  if (frames.size) phaseChecked = [...frames.values()].some(frame => frame.accepted);
  updateTable(); text('processingStatus', videoNameMismatch ?
    'Video geladen; Dateiname weicht von der Kalibrierung ab, vorhandene Daten bleiben erhalten' :
    'Video geladen; keine automatische Kalibrierung');
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
element('speed').onchange = () => { video.playbackRate = Number(element('speed').value); };
element('overlay').onchange = draw;
function updatePreviewAdjustments() {
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
  const bytes = await computer.call('export', { frames: snapshotFrames, video: expectedVideo, parameters: snapshotParameters,
    opticalConfiguration: element('opticalConfiguration').value });
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
  const profile = { frames: 0, decodeMs: 0, rgbaMs: 0, remapComputeMs: 0, remapTransferMs: 0,
    gpuContextMs: 0, gpuUploadMs: 0, gpuCommandMs: 0, gpuCompletionMs: 0, gpuReadbackMs: 0,
    frameMs: 0, submitMs: 0, backpressureMs: 0, totalMs: 0,
    width: exporter.config.width, height: exporter.config.height };
  renderExportProfile(null);
  try {
    for (let index = 0; index < videoInfo.frameCount; index++) {
      const frameStarted = performance.now();
      text('processingStatus', `Videoexport | Frame ${index + 1}/${videoInfo.frameCount}`);
      element('progress').value = index / videoInfo.frameCount;
      let phaseStarted = performance.now();
      const decoded = await decoder.call('frame', { index });
      profile.decodeMs += performance.now() - phaseStarted;
      phaseStarted = performance.now();
      const image = imageFromBitmap(decoded.bitmap);
      decoded.bitmap.close();
      profile.rgbaMs += performance.now() - phaseStarted;
      phaseStarted = performance.now();
      const remapped = await computer.call('remap', { image, useWebGpu: element('useWebGpu').checked }, [image.data.buffer]);
      const remapRoundtripMs = performance.now() - phaseStarted;
      profile.remapComputeMs += remapped.processingMs;
      profile.remapTransferMs += Math.max(0, remapRoundtripMs - remapped.processingMs);
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
      const encoderTiming = await exporter.addFrame(remapped.data, remapped.width, remapped.height, timestamp, nextTimestamp - timestamp, index);
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
});
element('infoButton').onclick = () => element('infoDialog').showModal();
element('closeInfo').onclick = () => element('infoDialog').close();
installPan(rawCanvas); installPan(resultCanvas);
new ResizeObserver(draw).observe(element('rawCanvas').parentElement);
new ResizeObserver(draw).observe(element('resultCanvas').parentElement);
window.addEventListener('beforeunload', event => { if (frames.size && !calibration) { event.preventDefault(); event.returnValue = ''; } });
updatePatternControls();
updateControls(); updateMetrics(); draw();
setInterval(updateMemoryStats, 2000);