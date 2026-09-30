import { planPcbOverlapChecks } from './pcb-overlap-checks.js';
import { importOptimizedProject } from './optimized-project.js';
import { networkGeometryKey, emptyMatchNetwork, migrateMatchNetwork, mergeNetworkMatches,
  deleteNetworkMatch, networkEdges, applyNetworkToGraph, projectNetworkCells } from './match-network.js';
import { installFineAlignment } from './fine-ui.js';
import { pairStiffness } from './pair-stiffness.js';
import { applyLoopClosure, removeUnselectedTrackingFrames, trackingReferenceCounts, trackingVideoCompatible } from './tracking-data.js';
import './style.css';
import { createIcons, icons } from 'lucide';
import { WorkerClient } from './rpc.js';
import { evaluate, fieldColor } from './spline.js';
import { poseFor } from './solver.js';
import { Field3DView } from './field-3d.js';
import { VideoExporter } from './video-exporter.js';
import { relativeCoverageScale } from './maps.js';
import { createPatchMask, MASK_FORBIDDEN, MASK_NEUTRAL, MASK_SEARCH, maskIncludes, paintPatchMask, remapInclusionMask } from './patch-mask.js';
import { analyzeObservations, selectConsistentFrames } from './observation-diagnostics.js';
import { fitCameraPose, stabilizePose } from './motion-tracking.js';
import { frameGeometry, localSelectionMask, supportColor, applyPixelMask, edgeFeatherMask, edgeFeatherWeight, applyEdgeFeather, accumulateFrame, approximateTopFrames, averagedFrames, sharpestFramesFirst, sparsePathFrames, pointBounds, evenlySpaced } from './path-support.js';
import { FrameReader } from './frame-reader.js';
import { installTrackingInspector } from './tracking-inspector.js';
import { installCheckerboardView } from './checkerboard-view.js';
import { checkerboardCells } from './checkerboard-analysis.js';
import { WebGpuOverlay } from './webgpu-overlay.js';
import { discoverWebGpuAdapters, getWebGpuSelection, requestSelectedGpuAdapter,
  setWebGpuSelection } from './webgpu-selection.js';
import { renderTiledOverlay, closeOverlayTiles, overlayTiles, overlayTileSize } from './overlay-tiles.js';
import { measuredRefitComponent, addLocalRefitEdges, buildPoseGraph, planFocusedRefitPairs, planRefitBridgePairs } from './pose-graph-refit.js';
import { refitColorMatrix } from './refit-preprocess.js';
import { brightnessDisplayRange, brightnessFieldPixels, decodeBrightnessCalibration, encodeBrightnessCalibration } from './brightness-calibration.js';
import { fitCheckerboardBrightness, sampleCheckerboardBrightness } from './checkerboard-brightness.js';
import { mergeBounds, mergeEstimate, selectMergeFrames } from './merge-plan.js';
import { reduceFramesByBlocks } from './frame-reduction.js';
import { acceptedPcbConstraints, composePose, interpolatePcbPoses, pcbRunFingerprint, planPcbPairs, selectPcbIntermediateFrames,
  selectPcbKeyframes, planPcbTemporalBridges, planPcbTemporalSkips } from './pcb-realignment.js';
import { planTiffParts, beginTiffParts, tiffJoinCommand } from './tiff-parts.js';
import { createBrowserImageFile } from './browser-image-file.js';

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
let decoder = new WorkerClient('./decoder-worker.js', showProgress);
const computer = new WorkerClient('./compute-worker.js', showProgress);
let trackingComputer = new WorkerClient('./compute-worker.js', showProgress);
let videoInfo = null;
let expectedVideo = null;
let videoUrl = null;
let currentIndex = 0;
let currentSharpness = null;
let calibration = null;
let brightnessCalibration = null;
const frameReader = new FrameReader({ getDecoder: () => decoder, getMaps: () => calibration?.maps, getBrightness: () => brightnessCalibration, computer,
  onFallback: error => console.warn('FrameReader: CPU-Fallback:', error.message) });
const useWebGpu = () => getWebGpuSelection() !== 'none';
let gpuAdapterReady = false;
function updateGpuAdapterControl() {
  element('globalGpuAdapter').disabled = !gpuAdapterReady || taskBusy || continuous || rectifiedPlayback ||
    navigationBusy || trackingRunning || trackingPreviewBusy || pcbBusy || pathRefitBusy ||
    mergeRunning || overlayGpuInFlight > 0;
}
const readFrame = (index, options = {}) => frameReader.read(index, { gpu: useWebGpu(), ...options });
const readTrackingFrame = (index, options = {}) => frameReader.read(index, { gpu: useWebGpu(), ...options });
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
let fineUi = null;
let fineBusy = false;
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
let observationDiagnostics = null;
let observationDiagnosticsDirty = true;
let geometrySelection = null;
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
let brightnessRunning = false;
let brightnessPaused = false;
let brightnessSession = null;
let trackingNextIndex = 0;
let trackingNeedsSeed = false;
let trackingPath = [];
let pcbBusy = false;
let pcbCancelled = false;
let pcbPauseRequested = false;
let pcbStepRemaining = 0;
let pcbResume = null;
let pcbWorker = null;
let pcbProposal = null;
let pcbUndo = null;
let pcbLastDiagnostics = null;
function pcbParameters() {
  const parameters = { spacing: number('pcbSpacing', 0.3), turnDegrees: number('pcbTurnDegrees', 25),
    revisitGap: number('pcbRevisitGap', 256), maxNeighbors: number('pcbMaxNeighbors', 4),
    maxPairs: number('pcbMaxPairs', 2048), bridgeBudget: number('pcbBridgeBudget', 2048),
    radius: number('pcbSearchRadius', 128),
    coarseRadius: number('pcbCoarseRadius', 384),
    coarseOverlap: number('pcbCoarseOverlap', 5) / 100,
    angle: number('pcbSearchAngle', 2), cycleLimit: number('pcbCycleLimit', 5),
    fftCycleFactor: number('pcbFftCycleFactor', 2),
    cellSize: number('pcbFftCellSize', 256), cellsPerAxis: number('pcbFftCells', 3),
    minimumPsr: number('pcbFftPsr', 6), residualLimit: number('pcbFftResidual', 6),
    minimumScore: number('pcbMinScore', 0.9), minimumSupport: number('pcbMinSupport', 128),
    iterations: number('pcbIterations', 8), localBudget: number('pcbLocalBudget', 64),
    huber: number('pcbHuber', 20),
    brightness: number('pcbBrightness', 0) / 100, contrast: number('pcbContrast', 100) / 100,
    gamma: number('pcbGamma', 1) };
  if (!(parameters.radius >= 4 && parameters.radius <= 256 && parameters.angle >= 0.1 && parameters.angle <= 10 &&
      parameters.coarseRadius >= 128 && parameters.coarseRadius <= 2048 &&
      parameters.coarseOverlap >= 0.01 && parameters.coarseOverlap <= 0.5 &&
      parameters.cycleLimit > 0 && parameters.cycleLimit <= 50 && parameters.minimumScore >= 0 && parameters.minimumScore <= 1 &&
      parameters.fftCycleFactor >= 1 && parameters.fftCycleFactor <= 3 &&
      Number.isInteger(parameters.minimumSupport) && parameters.minimumSupport >= 16 && parameters.minimumSupport <= 1000000 &&
      Number.isInteger(parameters.iterations) && parameters.iterations >= 1 && parameters.iterations <= 50 &&
      Number.isInteger(parameters.localBudget) && parameters.localBudget >= 0 && parameters.localBudget <= 100000 &&
      Number.isInteger(parameters.bridgeBudget) && parameters.bridgeBudget >= 0 && parameters.bridgeBudget <= 100000 &&
      parameters.huber > 0 && parameters.huber <= 1000 &&
      Number.isInteger(parameters.cellSize) && parameters.cellSize >= 16 && parameters.cellSize <= 256 &&
      Number.isInteger(parameters.cellsPerAxis) && parameters.cellsPerAxis >= 2 && parameters.cellsPerAxis <= 8 &&
      parameters.minimumPsr >= 2 && parameters.minimumPsr <= 30 &&
      parameters.residualLimit >= 1 && parameters.residualLimit <= 20 &&
      parameters.brightness >= -1 && parameters.brightness <= 1 &&
      parameters.contrast >= 0.25 && parameters.contrast <= 3 && parameters.gamma >= 0.2 && parameters.gamma <= 3))
    throw new Error('Ungueltige PCB-Realignment-Parameter.');
  return parameters;
}
function updatePcbControls() {
  updateGpuAdapterControl();
  element('pcbRealignStart').disabled = pcbBusy || trackingRunning || taskBusy || pathRefitBusy;
  element('pcbRealignPause').disabled = !pcbBusy || pcbPauseRequested;
  element('pcbRealignResume').disabled = !pcbBusy || !pcbPauseRequested;
  element('pcbRealignStep').disabled = !pcbBusy || !pcbPauseRequested;
  element('pcbRealignCancel').disabled = !pcbBusy;
  element('pcbRealignApply').disabled = pcbBusy || !pcbProposal;
  element('pcbRealignDiscard').disabled = pcbBusy || !pcbProposal;
  element('pcbRealignUndo').disabled = pcbBusy || !pcbUndo;
}
async function waitForPcbResume() {
  if (!pcbPauseRequested) return;
  text('pcbRealignStatus', 'Pausiert. Fortsetzen oder ein Paar messen.');
  await new Promise(resolve => { pcbResume = resolve; });
  pcbResume = null;
  if (pcbCancelled) throw new Error('PCB-Realignment abgebrochen.');
}
function invalidatePcbProposal() {
  pcbProposal = null;
  updatePcbControls(); drawTrackingPath();
}
let pcbDiagnosticRevision = 0;
let pcbSelectedMatch = null;
const pcbPairView = { images: [], match: null, offsets: [], zoom: 1, panX: 0, panY: 0 };
let pcbCellMatch = null;
let pcbSelectedCellId = null;
let pcbCellTargets = [];
const pcbCellView = { zoom: 1, panX: 0, panY: 0 };
function clampPcbCellView(canvas) {
  const maxPanX = Math.max(0, (canvas.width - 64) * (pcbCellView.zoom - 1) / 2);
  const maxPanY = Math.max(0, (canvas.height - 64) * (pcbCellView.zoom - 1) / 2);
  pcbCellView.panX = Math.max(-maxPanX, Math.min(maxPanX, pcbCellView.panX));
  pcbCellView.panY = Math.max(-maxPanY, Math.min(maxPanY, pcbCellView.panY));
}
function selectedPcbCell() {
  return pcbCellMatch?.fft?.cells?.find(cell => cell.cellId === pcbSelectedCellId) ?? null;
}
function pcbCellSize() {
  const source = pcbProposal ?? pcbLastDiagnostics ?? trackingDataset?.pcbRealignment;
  return pcbCellMatch?.fft?.cellSize ?? source?.parameters?.cellSize ?? 256;
}
function releasePcbPairImages() {
  pcbDiagnosticRevision++;
  for (const bitmap of pcbPairView.images) bitmap.close();
  pcbPairView.images = [];
  pcbPairView.offsets = [];
}
function pcbDiagnosticPairs() {
  const source = pcbProposal ?? pcbLastDiagnostics ?? trackingDataset?.pcbRealignment;
  return [...(source?.pairDiagnostics ?? []), ...(source?.localDiagnostics ?? [])];
}
function drawPcbCells(match) {
  if (pcbCellMatch !== match) {
    pcbSelectedCellId = null;
    Object.assign(pcbCellView, { zoom: 1, panX: 0, panY: 0 });
  }
  pcbCellMatch = match;
  pcbCellTargets = [];
  const canvas = element('pcbCellCanvas'), context = canvas.getContext('2d');
  clampPcbCellView(canvas);
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = '#f7faf8'; context.fillRect(0, 0, canvas.width, canvas.height);
  const cells = match.fft?.cells ?? [];
  if (!cells.length) { context.fillStyle = '#52635b'; context.fillText(match.fft?.reason ?? 'Keine Messzellen', 18, 25); return; }
  const minX = Math.min(...cells.map(cell => cell.center.x)), maxX = Math.max(...cells.map(cell => cell.center.x));
  const minY = Math.min(...cells.map(cell => cell.center.y)), maxY = Math.max(...cells.map(cell => cell.center.y));
  const scaleX = (canvas.width - 64) / Math.max(1, maxX - minX);
  const scaleY = (canvas.height - 64) / Math.max(1, maxY - minY);
  const px = value => canvas.width / 2 + (value - (minX + maxX) / 2) * scaleX * pcbCellView.zoom + pcbCellView.panX;
  const py = value => canvas.height / 2 + (value - (minY + maxY) / 2) * scaleY * pcbCellView.zoom + pcbCellView.panY;
  const inliers = new Set(match.fft?.inlierCells ?? []);
  for (const cell of cells) {
    const x = px(cell.center.x), y = py(cell.center.y);
    const structure = Math.max(1, Math.min(15, cell.structureRms ?? cell.texture ?? 1));
    const diameter = 3 + (structure - 1) / 14 * 17;
    const radius = diameter / 2;
    pcbCellTargets.push({ x, y, radius, cellId: cell.cellId });
    const color = inliers.has(cell.cellId) ? '#08765f' :
      cell.reason === 'Maske oder Rand' ? '#89938f' :
        cell.reason === 'Nicht ausgewaehlt' ? '#5796c8' :
          cell.accepted ? '#ba7619' : '#b04949';
    context.fillStyle = color; context.beginPath(); context.arc(x, y, radius, 0, Math.PI * 2); context.fill();
    if (cell.cellId === pcbSelectedCellId) {
      context.strokeStyle = '#143bcb'; context.lineWidth = 3;
      context.beginPath(); context.arc(x, y, radius + 4, 0, Math.PI * 2); context.stroke();
    }
    if (Number.isFinite(cell.dx) && Number.isFinite(cell.dy)) {
      const arrowScale = Math.min(8, scaleX, scaleY) * pcbCellView.zoom;
      const endX = x + cell.dx * arrowScale, endY = y + cell.dy * arrowScale;
      context.strokeStyle = color; context.lineWidth = 2; context.beginPath();
      context.moveTo(x, y); context.lineTo(endX, endY); context.stroke();
    }
  }
  renderPcbCellDetails();
}
function renderPcbCellDetails() {
  const panel = element('pcbCellDetails'), cell = selectedPcbCell();
  panel.hidden = !cell;
  if (!cell) return;
  const reason = cell.reason === 'Nicht ausgewaehlt' ? 'Nicht fuer FFT ausgewaehlt' :
    cell.reason ?? (cell.accepted ? 'Messung verwendbar' : 'Nicht verwendbar');
  const structure = cell.structureRms ?? cell.texture;
  text('pcbCellDetailsStatus', `Zelle ${cell.cellId} | ${reason} | Struktur ${fixed(structure, 2)} | ` +
    `PSR ${fixed(cell.psr, 2)} | eigene Strukturpixel ${Math.round(cell.ownedSupportPixels ?? 0)} | ` +
    `Verschiebung (${fixed(cell.dx, 2)}, ${fixed(cell.dy, 2)}) px`);
  const match = pcbCellMatch;
  text('pcbCellReferenceLabel', `Referenz #${match.reference}`);
  text('pcbCellCurrentLabel', `Frame #${match.current} vor Korrektur`);
  const size = pcbCellSize();
  for (const [index, id, pose] of [[0, 'pcbCellReference', match.referencePose],
    [1, 'pcbCellCurrent', match.currentPose]]) {
    const canvas = element(id), context = canvas.getContext('2d');
    context.fillStyle = '#f7faf8'; context.fillRect(0, 0, canvas.width, canvas.height);
    if (pcbPairView.images.length !== 2 || !pose) continue;
    const scale = Math.min(canvas.width, canvas.height) / (size * 1.15);
    context.save();
    context.translate(canvas.width / 2, canvas.height / 2);
    context.scale(scale, scale);
    context.translate(-cell.center.x, -cell.center.y);
    context.translate(pose.x, pose.y); context.rotate(pose.rotation);
    drawAdjustedImage(context, pcbPairView.images[index],
      pcbPairView.offsets[index][0], pcbPairView.offsets[index][1]);
    context.restore();
    context.strokeStyle = '#143bcb'; context.lineWidth = 1;
    context.strokeRect((canvas.width - size * scale) / 2,
      (canvas.height - size * scale) / 2, size * scale, size * scale);
  }
}
function renderPcbPairImages() {
  const { images, match, offsets, zoom, panX, panY } = pcbPairView;
  const before = element('pcbPairBefore'), after = element('pcbPairAfter');
  for (const [canvas, currentPose] of [[before, match?.currentPose], [after, match?.forward?.pose ?? match?.currentPose]]) {
    const context = canvas.getContext('2d');
    context.fillStyle = '#f7faf8'; context.fillRect(0, 0, canvas.width, canvas.height);
    if (images.length !== 2 || !match?.referencePose || !currentPose) continue;
    const scale = Math.min(canvas.width / (images[0].width * 1.4),
      canvas.height / (images[0].height * 1.4)) * zoom;
    context.save();
    context.translate(canvas.width / 2 + panX, canvas.height / 2 + panY);
    context.scale(scale, scale);
    context.translate(-match.referencePose.x, -match.referencePose.y);
    const drawImageAtPose = (bitmap, pose, imageOffset, alpha) => {
      context.save(); context.globalAlpha = alpha;
      context.translate(pose.x, pose.y); context.rotate(pose.rotation);
      drawAdjustedImage(context, bitmap, imageOffset[0], imageOffset[1]);
      context.restore();
    };
    drawImageAtPose(images[0], match.referencePose, offsets[0], 1);
    drawImageAtPose(images[1], currentPose, offsets[1], 0.55);
    const cell = selectedPcbCell();
    if (cell && match === pcbCellMatch) {
      const size = pcbCellSize();
      context.strokeStyle = '#143bcb'; context.lineWidth = 2 / scale;
      context.strokeRect(cell.center.x - size / 2, cell.center.y - size / 2, size, size);
    }
    context.restore();
  }
}
function centerPcbPairOnSelectedCell() {
  const cell = selectedPcbCell();
  const { images, match, zoom } = pcbPairView;
  if (!cell || images.length !== 2 || !match?.referencePose) return;
  const canvas = element('pcbPairBefore');
  const scale = Math.min(canvas.width / (images[0].width * 1.4),
    canvas.height / (images[0].height * 1.4)) * zoom;
  pcbPairView.panX = -(cell.center.x - match.referencePose.x) * scale;
  pcbPairView.panY = -(cell.center.y - match.referencePose.y) * scale;
}
async function drawPcbPairImages(match) {
  releasePcbPairImages();
  const revision = pcbDiagnosticRevision;
  Object.assign(pcbPairView, { images: [], match, offsets: [], zoom: 1, panX: 0, panY: 0 });
  renderPcbPairImages();
  if (!videoInfo || !calibration?.maps || !match.referencePose || !match.currentPose) return;
  const images = [];
  try {
    for (const frame of [match.reference, match.current]) images.push((await readTrackingFrame(frame,
      { rectified: true, sourceMask: pcbSourceMask() })).bitmap);
    if (revision !== pcbDiagnosticRevision) return;
    const offset = (frame, bitmap) => {
      const entry = trackingPath.find(item => item.frame === frame);
      return entry?.mode === 'window' ? [-bitmap.width / 2, -bitmap.height / 2] :
        [calibration.maps.origin[0] - calibration.field.width / 2,
          calibration.maps.origin[1] - calibration.field.height / 2];
    };
    pcbPairView.images = images.splice(0);
    pcbPairView.offsets = [offset(match.reference, pcbPairView.images[0]),
      offset(match.current, pcbPairView.images[1])];
    centerPcbPairOnSelectedCell();
    renderPcbPairImages();
    renderPcbCellDetails();
  } catch (error) {
    if (revision === pcbDiagnosticRevision) text('pcbPairSummary', `Bildpaar konnte nicht geladen werden: ${error.message}`);
  } finally { for (const bitmap of images) bitmap.close(); }
}
function installPcbPairZoom() {
  const cellPanel = document.createElement('div');
  cellPanel.id = 'pcbCellDetails'; cellPanel.hidden = true;
  cellPanel.className = 'pcb-cell-details';
  cellPanel.innerHTML = '<p id="pcbCellDetailsStatus" class="status-line"></p>' +
    '<div class="pcb-cell-images"><div><strong id="pcbCellReferenceLabel"></strong>' +
    '<canvas id="pcbCellReference" width="320" height="320" aria-label="Ausschnitt der Referenzzelle"></canvas></div>' +
    '<div><strong id="pcbCellCurrentLabel"></strong>' +
    '<canvas id="pcbCellCurrent" width="320" height="320" aria-label="Ausschnitt der aktuellen Zelle"></canvas></div></div>';
  element('pcbRealignDetails').querySelector('.pcb-diagnostic-grid').after(cellPanel);
  const cellCanvas = element('pcbCellCanvas');
  cellCanvas.title = 'Mausrad: Zoom; ziehen: verschieben; Doppelklick: einpassen; Zelle anklicken: Bildausschnitte';
  let cellDrag = null, suppressCellClick = false;
  cellCanvas.addEventListener('wheel', event => {
    event.preventDefault();
    if (!pcbCellMatch) return;
    const rect = cellCanvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * cellCanvas.width / rect.width;
    const y = (event.clientY - rect.top) * cellCanvas.height / rect.height;
    const next = Math.max(1, Math.min(32, pcbCellView.zoom * Math.exp(-event.deltaY * 0.001)));
    const factor = next / pcbCellView.zoom;
    pcbCellView.panX = x - cellCanvas.width / 2 - (x - cellCanvas.width / 2 - pcbCellView.panX) * factor;
    pcbCellView.panY = y - cellCanvas.height / 2 - (y - cellCanvas.height / 2 - pcbCellView.panY) * factor;
    pcbCellView.zoom = next;
    clampPcbCellView(cellCanvas);
    drawPcbCells(pcbCellMatch);
  }, { passive: false });
  cellCanvas.addEventListener('pointerdown', event => {
    suppressCellClick = false;
    cellDrag = { x: event.clientX, y: event.clientY, moved: false };
    cellCanvas.setPointerCapture(event.pointerId);
  });
  cellCanvas.addEventListener('pointermove', event => {
    if (!cellDrag) return;
    const dx = event.clientX - cellDrag.x, dy = event.clientY - cellDrag.y;
    if (Math.abs(dx) + Math.abs(dy) > 2) cellDrag.moved = true;
    const rect = cellCanvas.getBoundingClientRect();
    pcbCellView.panX += dx * cellCanvas.width / rect.width;
    pcbCellView.panY += dy * cellCanvas.height / rect.height;
    clampPcbCellView(cellCanvas);
    cellDrag.x = event.clientX; cellDrag.y = event.clientY;
    if (cellDrag.moved && pcbCellMatch) drawPcbCells(pcbCellMatch);
  });
  cellCanvas.addEventListener('pointerup', () => {
    suppressCellClick = Boolean(cellDrag?.moved); cellDrag = null;
  });
  cellCanvas.addEventListener('pointercancel', () => { cellDrag = null; });
  cellCanvas.addEventListener('dblclick', () => {
    Object.assign(pcbCellView, { zoom: 1, panX: 0, panY: 0 });
    if (pcbCellMatch) drawPcbCells(pcbCellMatch);
  });
  cellCanvas.addEventListener('click', event => {
    if (suppressCellClick) { suppressCellClick = false; return; }
    const rect = cellCanvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * cellCanvas.width / rect.width;
    const y = (event.clientY - rect.top) * cellCanvas.height / rect.height;
    const nearest = pcbCellTargets.map(target => ({ target,
      distance: Math.hypot(target.x - x, target.y - y) }))
      .sort((first, second) => first.distance - second.distance)[0];
    pcbSelectedCellId = nearest?.distance <= Math.max(12, nearest.target.radius + 4) &&
      nearest.target.cellId !== pcbSelectedCellId ?
      nearest.target.cellId : null;
    if (pcbCellMatch) drawPcbCells(pcbCellMatch);
    if (pcbSelectedCellId !== null) centerPcbPairOnSelectedCell();
    renderPcbPairImages();
  });
  for (const canvas of [element('pcbPairBefore'), element('pcbPairAfter')]) {
    canvas.title = 'Mausrad: Zoom; ziehen: verschieben; Doppelklick: einpassen';
    canvas.addEventListener('wheel', event => {
      event.preventDefault();
      if (pcbPairView.images.length !== 2) return;
      const rect = canvas.getBoundingClientRect();
      const x = (event.clientX - rect.left) * canvas.width / rect.width - canvas.width / 2;
      const y = (event.clientY - rect.top) * canvas.height / rect.height - canvas.height / 2;
      const next = Math.max(0.5, Math.min(64, pcbPairView.zoom * Math.exp(-event.deltaY * 0.001)));
      const factor = next / pcbPairView.zoom;
      pcbPairView.panX = x - (x - pcbPairView.panX) * factor;
      pcbPairView.panY = y - (y - pcbPairView.panY) * factor;
      pcbPairView.zoom = next;
      renderPcbPairImages();
    }, { passive: false });
    let dragging = null;
    canvas.addEventListener('pointerdown', event => {
      if (pcbPairView.images.length !== 2) return;
      dragging = { x: event.clientX, y: event.clientY };
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', event => {
      if (!dragging) return;
      const rect = canvas.getBoundingClientRect();
      pcbPairView.panX += (event.clientX - dragging.x) * canvas.width / rect.width;
      pcbPairView.panY += (event.clientY - dragging.y) * canvas.height / rect.height;
      dragging = { x: event.clientX, y: event.clientY };
      renderPcbPairImages();
    });
    canvas.addEventListener('pointerup', () => { dragging = null; });
    canvas.addEventListener('pointercancel', () => { dragging = null; });
    canvas.addEventListener('dblclick', () => {
      pcbPairView.zoom = 1; pcbPairView.panX = 0; pcbPairView.panY = 0;
      renderPcbPairImages();
    });
  }
}
function refreshPcbPairDiagnostics() {
  const pairs = pcbDiagnosticPairs(), select = element('pcbPairSelect');
  const selected = pcbSelectedMatch;
  select.replaceChildren();
  for (const [index, pair] of pairs.entries()) {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = `#${pair.current} gegen #${pair.reference} (${pair.kind ?? 'lokal'})`;
    select.append(option);
  }
  element('pcbRealignDetails').hidden = pairs.length === 0;
  if (!pairs.length) {
    pcbSelectedMatch = null;
    releasePcbPairImages();
    return;
  }
  const selectedIndex = pairs.findIndex(pair => pair === selected ||
    (selected && pair.reference === selected.reference && pair.current === selected.current &&
      pair.kind === selected.kind));
  select.value = String(Math.max(0, selectedIndex));
  const show = () => {
    const match = pairs[Number(select.value)] ?? pairs[0];
    pcbSelectedMatch = match;
    const effectiveCycleLimit = match.fftGpuVerification?.accepted ?
      Math.max((pcbProposal ?? pcbLastDiagnostics ?? trackingDataset?.pcbRealignment)?.parameters?.cycleLimit ?? 5,
        match.fftGpuVerification.cycleLimit ?? 0) :
      (pcbProposal ?? pcbLastDiagnostics ?? trackingDataset?.pcbRealignment)?.parameters?.cycleLimit;
    const accepted = Boolean(match.forward?.accepted && match.backward?.accepted &&
      Number.isFinite(match.reverseDistance) &&
      match.reverseDistance <= effectiveCycleLimit);
    const cells = match.fft?.cells ?? [];
    text('pcbPairSummary', `${accepted ? 'Paarmessung angenommen' : 'Paarmessung verworfen'} | ` +
      `${match.fft?.inlierCells?.length ?? 0}/${cells.length} Zellen | ` +
      `${match.fft?.uniqueSupportArea ?? 0} eindeutige Strukturpixel | ` +
      `Residuum ${fixed(match.fft?.residualRms, 2)} px | ` +
      `Rueckweg ${fixed(match.reverseDistance, 2)} / ${fixed(effectiveCycleLimit, 2)} px | ` +
      `${accepted ? `Feder Position ${fixed(pairStiffness(match).weight, 1)}, Rotation ${pairStiffness(match).rotationWeight ? fixed(pairStiffness(match).weight, 1) : 'frei'} | ` : ''}` +
      `${match.accelerator ?? 'CPU-FFT/NCC'} | ` +
      `${match.fftGpuVerification?.accepted ? 'FFT-Pose durch WebGPU bestaetigt' :
        match.coarse?.accepted ? 'GPU-Grobsuche bestaetigt' : match.fftGpuVerification?.reason ?? match.coarse?.reason ??
        match.fft?.reason ?? match.forward?.reason ?? ''}`);
    drawPcbCells(match);
    if (element('pcbRealignDetails').open) void drawPcbPairImages(match);
    else releasePcbPairImages();
  };
  select.onchange = show;
  if (selected !== pairs[Number(select.value)]) show();
}
function pcbSourceMask() {
  if (trackingImageMask) return trackingMaskHasSelection ? trackingImageMask : null;
  return (trackingRunOptions ?? trackingDataset?.options)?.sourceImageMask ?? null;
}
async function runPcbRealignment() {
  if (pcbBusy || trackingRunning || taskBusy || pathRefitBusy) return;
  if (!videoInfo || !calibration?.maps || !trackingVideoCompatible(trackingDataset, videoInfo)) {
    text('pcbRealignStatus', 'Passendes Video, Kalibrierung und Tracking-Posen laden.'); return;
  }
  let parameters;
  try { parameters = pcbParameters(); }
  catch (error) { text('pcbRealignStatus', error.message); return; }
  const sourcePath = trackingPath;
  const sourcePoses = new Map(sourcePath.map(entry => [entry.frame, { ...entry.pose }]));
  const sourceMask = pcbSourceMask();
  const mask = sourceMask ? remapInclusionMask(sourceMask, calibration.maps) :
    (trackingRunOptions ?? trackingDataset?.options)?.imageMask ?? null;
  let keyframes, pairs;
  try {
    keyframes = selectPcbKeyframes(sourcePath, { width: calibration.maps.outputWidth,
      height: calibration.maps.outputHeight, ...parameters });
    pairs = planPcbPairs(keyframes, calibration.field, calibration.maps, { ...parameters, mask });
  } catch (error) { text('pcbRealignStatus', error.message); return; }
  if (keyframes.length < 2) { text('pcbRealignStatus', 'Zu wenige Keyframes.'); return; }
  if (parameters.maxPairs < keyframes.length - 1) {
    text('pcbRealignStatus', `Paarbudget ${parameters.maxPairs} reicht fuer ${keyframes.length} Keyframes nicht. Mindestens ${keyframes.length - 1} einstellen.`); return;
  }
  if (!pairs.length) { text('pcbRealignStatus', 'Keine ueberlappenden Keyframe-Paare.'); return; }
  invalidatePcbProposal(); pcbLastDiagnostics = null; pcbBusy = true; pcbCancelled = false;
  pcbPauseRequested = false; pcbStepRemaining = 0; updatePcbControls();
  refreshPcbPairDiagnostics();
  const worker = new WorkerClient('./compute-worker.js'); pcbWorker = worker;
  const started = performance.now();
  const entries = new Map(sourcePath.map(entry => [entry.frame, entry]));
  const nativeFrameCache = new Map();
  const profile = { pairs: 0, decoded: 0, reused: 0, decodeMs: 0, remapMs: 0,
    copyMs: 0, workerMs: 0, readbackMs: 0, fftMs: 0, pyramidMs: 0, nccMs: 0, fallbackMs: 0 };
  const pairBitmap = async frame => {
    const decoded = await readTrackingFrame(frame, { gpu: useWebGpu(), rectified: true,
      sourceMask, measureSharpness: false });
    if (decoded.frameTiming?.cacheHit) profile.reused++;
    else profile.decoded++;
    profile.decodeMs += decoded.frameTiming?.decodeMs ?? 0;
    profile.remapMs += decoded.frameTiming?.remapMs ?? 0;
    return decoded.bitmap;
  };
  const preprocessing = { brightness: parameters.brightness, contrast: parameters.contrast,
    gamma: parameters.gamma, red: 0.3, green: 0.59, blue: 0.11, region: 0.4, edgeFeather: 0.1 };
  let nativeGpuPath = useWebGpu() && parameters.brightness === 0 &&
    parameters.contrast === 1 && parameters.gamma === 1;
  const fingerprint = pcbRunFingerprint(sourcePath, calibration, videoInfo, mask);
  const previousDraft = trackingDataset?.pcbRealignmentDraft;
  const sameParameters = previousDraft?.parameters && Object.entries(parameters).every(([key, value]) =>
    (previousDraft.parameters[key] ?? (key === 'fftCycleFactor' ? 2 : undefined)) === value);
  const resume = previousDraft?.schemaVersion === 1 && previousDraft.viaRecoveryVersion === 4 && previousDraft.fingerprint === fingerprint &&
    sameParameters &&
    Array.isArray(previousDraft.matches) && previousDraft.matches.length <= pairs.length &&
    previousDraft.matches.every((match, index) => match.reference === pairs[index].reference &&
      match.current === pairs[index].current);
  const draft = resume ? previousDraft : { schemaVersion: 1, viaRecoveryVersion: 4, fingerprint, parameters,
    matches: [], bridgeMatches: [], skipMatches: [], localMatches: [] };
  draft.parameters = parameters;
  const matches = draft.matches;
  trackingDataset = { ...trackingDataset, pcbRealignmentDraft: draft };
  const publishPcbPairs = () => {
    storeNetworkMatches([...matches, ...(draft.bridgeMatches ?? []), ...(draft.skipMatches ?? []),
      ...(draft.localMatches ?? [])], parameters);
    drawTrackingPath();
    pcbLastDiagnostics = { parameters,
      pairDiagnostics: [...matches, ...(draft.bridgeMatches ?? []), ...(draft.skipMatches ?? [])],
      localDiagnostics: [...(draft.localMatches ?? [])] };
    refreshPcbPairDiagnostics();
  };
  if (matches.length || draft.bridgeMatches?.length || draft.skipMatches?.length || draft.localMatches?.length)
    publishPcbPairs();
  if (resume && matches.length) text('pcbRealignStatus', `${matches.length} gespeicherte Paarmessungen werden fortgesetzt.`);
  const measurePair = async (pair, poses = null) => {
    const images = [];
    try {
      for (const frame of [pair.reference, pair.current]) {
        const entry = entries.get(frame);
        let bitmap = null, native = null;
        if (nativeGpuPath) {
          if (nativeFrameCache.has(frame)) {
            nativeFrameCache.delete(frame); nativeFrameCache.set(frame, true); profile.reused++;
          } else {
            const decoded = await readTrackingFrame(frame, { gpu: true, output: 'native', measureSharpness: false });
            native = { frame: decoded.frame, orientation: decoded.orientation };
            nativeFrameCache.set(frame, true);
            profile.decoded++; profile.decodeMs += decoded.frameTiming?.decodeMs ?? 0;
          }
        } else bitmap = await pairBitmap(frame);
        const width = nativeGpuPath ? calibration.maps.outputWidth : bitmap.width;
        const height = nativeGpuPath ? calibration.maps.outputHeight : bitmap.height;
        const offset = entry.mode === 'window' ? [-width / 2, -height / 2] :
          [calibration.maps.origin[0] - calibration.field.width / 2,
            calibration.maps.origin[1] - calibration.field.height / 2];
        images.push({ frame, pose: { ...(poses?.get(frame) ?? entry.pose) }, offset, bitmap, native });
      }
      if (pcbCancelled) throw new Error('PCB-Realignment abgebrochen.');
      const workerStarted = performance.now();
      const result = await worker.call('pcb-pair-register', { images, pair:{...pair,cells:pair.cells??currentMatchNetwork().pairs.find(p=>p.reference===pair.reference&&p.current===pair.current)?.cells??[]}, mask, preprocessing,
        nativePath: nativeGpuPath,
        localLandmarks: true, useWebGpu: useWebGpu(), coarseRadius: parameters.coarseRadius,
        minimumOverlapFraction: parameters.coarseOverlap,
        fft: { cellSize: parameters.cellSize, cellsPerAxis: parameters.cellsPerAxis,
          searchRadius: Math.min(parameters.radius, parameters.cellSize / 2 - 1), minimumPsr: parameters.minimumPsr,
          residualLimit: parameters.residualLimit },
        limits: { radius: parameters.radius, angle: parameters.angle,
          cycleLimit: parameters.cycleLimit,
          fftCycleFactor: parameters.fftCycleFactor,
          minimumScore: parameters.minimumScore,
          minimumSupport: parameters.minimumSupport } }, images.map(item => item.native?.frame ?? item.bitmap).filter(Boolean));
      profile.workerMs += performance.now() - workerStarted;
      profile.pairs++;
      if (nativeGpuPath && Array.isArray(result.nativeCachedFrames)) {
        nativeFrameCache.clear();
        for (const frame of result.nativeCachedFrames) nativeFrameCache.set(frame, true);
        delete result.nativeCachedFrames;
      }
      for (const key of ['readbackMs', 'fftMs', 'pyramidMs', 'nccMs', 'fallbackMs'])
        profile[key] += result.profile?.[key] ?? 0;
      return result;
    } catch (error) {
      if (!nativeGpuPath) throw error;
      nativeGpuPath = false;
      nativeFrameCache.clear();
      console.warn('PCB: nativer WebGPU-Pfad fehlgeschlagen, Bitmap-Fallback:', error.message);
      return measurePair(pair, poses);
    } finally { for (const item of images) { item.bitmap?.close(); item.native?.frame?.close(); } }
  };
  try {
    await frameReader.waitUntilIdle();
    if (nativeGpuPath) {
      try {
        const cache = frameReader.cacheStats();
        await worker.call('pcb-native-setup', { maps: calibration.maps, sourceMask,
          brightness: brightnessCalibration, cacheBudget: Math.max(0, cache.budget - cache.bytes) });
      } catch (error) {
        console.warn('PCB: nativer WebGPU-Pfad nicht verfuegbar, Bitmap-Fallback:', error.message);
        nativeGpuPath = false;
      }
    }
    for (let pairIndex = matches.length; pairIndex < pairs.length; pairIndex++) {
      const pair = pairs[pairIndex];
      await waitForPcbResume();
      if (pcbCancelled) throw new Error('PCB-Realignment abgebrochen.');
      text('pcbRealignStatus', `Paar ${pairIndex + 1}/${pairs.length}: #${pair.current} gegen #${pair.reference} | ${Math.round((performance.now() - started) / 1000)} s`);
      matches.push(await measurePair(pair));
      draft.phase = 'keyframes'; draft.nextPair = matches.length;
      publishPcbPairs();
      if (pcbStepRemaining > 0 && --pcbStepRemaining === 0) { pcbPauseRequested = true; updatePcbControls(); }
      if (pairIndex % 4 === 3) await new Promise(resolve => setTimeout(resolve, 0));
    }
    await waitForPcbResume();
    const initialMeasurements = acceptedPcbConstraints(matches, parameters);
    const bridgePairs = planPcbTemporalBridges(sourcePath, pairs, matches,
      initialMeasurements.accepted, parameters.bridgeBudget);
    const bridgeMatches = draft.bridgeMatches ?? (draft.bridgeMatches = []);
    if (bridgeMatches.length > bridgePairs.length || bridgeMatches.some((match, index) =>
      match.reference !== bridgePairs[index].reference || match.current !== bridgePairs[index].current))
      bridgeMatches.length = 0;
    for (let index = bridgeMatches.length; index < bridgePairs.length; index++) {
      await waitForPcbResume();
      if (pcbCancelled) throw new Error('PCB-Realignment abgebrochen.');
      const pair = bridgePairs[index];
      text('pcbRealignStatus', `Brueckenpaar ${index + 1}/${bridgePairs.length}: #${pair.current} gegen #${pair.reference}`);
      bridgeMatches.push(await measurePair(pair));
      draft.phase = 'bridges'; draft.nextBridge = bridgeMatches.length;
      publishPcbPairs();
      if (pcbStepRemaining > 0 && --pcbStepRemaining === 0) { pcbPauseRequested = true; updatePcbControls(); }
      if (index % 4 === 3) await new Promise(resolve => setTimeout(resolve, 0));
    }
    await waitForPcbResume();
    const bridgeAccepted = acceptedPcbConstraints(bridgeMatches, parameters);
    const skipPairs = planPcbTemporalSkips(bridgePairs, bridgeMatches,
      bridgeAccepted.accepted, Math.max(0, parameters.bridgeBudget - bridgePairs.length));
    const skipMatches = draft.skipMatches ?? (draft.skipMatches = []);
    if (skipMatches.length > skipPairs.length || skipMatches.some((match, index) =>
      match.reference !== skipPairs[index].reference || match.current !== skipPairs[index].current))
      skipMatches.length = 0;
    for (let index = skipMatches.length; index < skipPairs.length; index++) {
      await waitForPcbResume();
      if (pcbCancelled) throw new Error('PCB-Realignment abgebrochen.');
      const pair = skipPairs[index];
      text('pcbRealignStatus', `Bruecken-Ersatzpaar ${index + 1}/${skipPairs.length}: #${pair.current} gegen #${pair.reference}`);
      skipMatches.push(await measurePair(pair));
      draft.phase = 'bridge-skips'; draft.nextSkip = skipMatches.length;
      publishPcbPairs();
      if (pcbStepRemaining > 0 && --pcbStepRemaining === 0) { pcbPauseRequested = true; updatePcbControls(); }
      if (index % 4 === 3) await new Promise(resolve => setTimeout(resolve, 0));
    }
    await waitForPcbResume();
    if (useWebGpu() && (draft.matchVersion ?? 1) < 2) {
      const retry = [...matches, ...bridgeMatches, ...skipMatches].filter(match =>
        match.fft?.accepted && match.fft.inlierCells?.length >= 4 &&
        match.fft.uniqueSupportArea >= 8192 &&
        !acceptedPcbConstraints([match], parameters).accepted.length);
      for (const [index, match] of retry.entries()) {
        await waitForPcbResume();
        if (pcbCancelled) throw new Error('PCB-Realignment abgebrochen.');
        text('pcbRealignStatus', `FFT-Pose auf WebGPU pruefen ${index + 1}/${retry.length}: #${match.current} gegen #${match.reference}`);
        const updated = await measurePair(match);
        for (const group of [matches, bridgeMatches, skipMatches]) {
          const position = group.indexOf(match);
          if (position >= 0) { group[position] = updated; break; }
        }
        publishPcbPairs();
        if (pcbStepRemaining > 0 && --pcbStepRemaining === 0) { pcbPauseRequested = true; updatePcbControls(); }
        if (index % 4 === 3) await new Promise(resolve => setTimeout(resolve, 0));
      }
      draft.matchVersion = 2;
    }
    const allMatches = [...matches, ...bridgeMatches, ...skipMatches];
    storeNetworkMatches(allMatches, parameters);
    const measurements = acceptedPcbConstraints(allMatches, parameters);
    measurements.accepted = networkEdges(currentMatchNetwork(), new Set(sourcePath.map(entry => entry.frame)));
    const graphFrames = new Set(keyframes.map(entry => entry.frame));
    let expanded;
    do {
      expanded = false;
      for (const edge of measurements.accepted) {
        if (!graphFrames.has(edge.reference) && !graphFrames.has(edge.current)) continue;
        for (const frame of [edge.reference, edge.current]) if (!graphFrames.has(frame)) {
          graphFrames.add(frame); expanded = true;
        }
      }
    } while (expanded);
    const graphEdges = measurements.accepted.filter(edge =>
      graphFrames.has(edge.reference) && graphFrames.has(edge.current));
    pcbLastDiagnostics = { parameters, pairDiagnostics: allMatches.map(match => ({ reference: match.reference,
      current: match.current, kind: match.kind, referencePose: match.referencePose,
      currentPose: match.currentPose, fft: match.fft, forward: match.forward, backward: match.backward,
      reverseDistance: match.reverseDistance, agreement: match.agreement,
      coarse: match.coarse, fftGpuVerification: match.fftGpuVerification,
      accelerator: match.accelerator })) };
    text('pcbRealignDiagnostics', JSON.stringify({ keyframes: [...graphFrames],
      accepted: measurements.accepted, rejected: measurements.rejected,
      pairs: pcbLastDiagnostics.pairDiagnostics }, null, 2));
    refreshPcbPairDiagnostics();
    if (!measurements.accepted.length && !currentMatchNetwork().pairs.some(pair => pair.cells.length >= (pair.minimumCells ?? 3))) throw new Error(`Kein belastbares Keyframe-Paar: ${measurements.rejected.length} verworfen.`);
    const graph = { nodes: sourcePath.map(entry => entry.frame).sort((a, b) => a - b).map(frame =>
      ({ frame, pose: { ...entries.get(frame).pose } })),
      edges: graphEdges };
    text('pcbRealignStatus', `Optimiere ${graph.nodes.length} Frames und ${graph.edges.length} gemessene Kanten...`);
    let solved = await worker.call('pcb-graph-optimize', { graph, network: currentMatchNetwork(),
      options: { iterations: Math.max(20, parameters.iterations), huber: Math.min(4, parameters.huber),
        lever: Math.hypot(calibration.maps.outputWidth, calibration.maps.outputHeight) / 2 } });
    if (pcbCancelled) throw new Error('PCB-Realignment abgebrochen.');
    text('pcbRealignDiagnostics', JSON.stringify({ keyframes: [...graphFrames],
      components: solved.components, accepted: measurements.accepted,
      rejected: measurements.rejected, pairs: pcbLastDiagnostics.pairDiagnostics }, null, 2));
    const optimized = new Map(solved.corrections.map(item => [item.frame, item.pose]));
    if (!optimized.size) throw new Error('Kein belastbarer Teilgraph konnte verbessert werden.');
    const byFrame = interpolatePcbPoses(sourcePath, optimized);
    const localMatches = draft.localMatches;
    const localCandidates = selectPcbIntermediateFrames(sourcePath, graphFrames,
      Math.max(parameters.localBudget, sourcePath.length));
    const optimizedFrames = [...graphFrames].filter(frame => optimized.has(frame)).sort((first, second) => first - second);
    let referenceIndex = 0;
    if (localMatches.length > localCandidates.length || localMatches.some((match, index) =>
      match.current !== localCandidates[index].frame)) localMatches.length = 0;
    for (const match of localMatches) {
      const edge = networkEdges(currentMatchNetwork(), new Set([match.reference,match.current])).find(edge=>
        edge.reference===match.reference&&edge.current===match.current);
      if(edge&&byFrame.has(edge.reference)&&!edge.currentPivot&&!edge.pointCells)byFrame.set(edge.current,composePose(byFrame.get(edge.reference),edge.measurement));
    }
    for (let index = localMatches.length; index < localCandidates.length; index++) {
      const entry = localCandidates[index];
      await waitForPcbResume();
      if (pcbCancelled) throw new Error('PCB-Realignment abgebrochen.');
      while (referenceIndex + 1 < optimizedFrames.length && optimizedFrames[referenceIndex + 1] <= entry.frame)
        referenceIndex++;
      const references = [optimizedFrames[referenceIndex], optimizedFrames[referenceIndex + 1]]
        .filter(frame => frame !== undefined && frame !== entry.frame);
      const reference = references.sort((first, second) => {
        const distance = frame => Math.hypot(byFrame.get(frame).x - byFrame.get(entry.frame).x,
          byFrame.get(frame).y - byFrame.get(entry.frame).y);
        return distance(first) - distance(second);
      })[0];
      if (reference === undefined || !byFrame.has(entry.frame)) continue;
      text('pcbRealignStatus', `Zwischenframe ${index + 1}/${localCandidates.length}: #${entry.frame} gegen #${reference}`);
      const match = await measurePair({ reference, current: entry.frame, kind: 'local' }, byFrame);
      localMatches.push(match);
      draft.phase = 'intermediate'; draft.nextLocal = localMatches.length;
      publishPcbPairs();
      const verified = acceptedPcbConstraints([match], parameters);
      const retained = networkEdges(currentMatchNetwork(), new Set([match.reference,match.current])).find(edge=>edge.current===entry.frame);
      if (verified.accepted.length && retained && !retained.currentPivot && !retained.pointCells)
        byFrame.set(entry.frame, composePose(byFrame.get(retained.reference),retained.measurement));
      if (pcbStepRemaining > 0 && --pcbStepRemaining === 0) { pcbPauseRequested = true; updatePcbControls(); }
      if (index % 4 === 3) await new Promise(resolve => setTimeout(resolve, 0));
    }
    await waitForPcbResume();
    const localVerification = acceptedPcbConstraints(localMatches, parameters);
    // Image evidence, not graph disagreement alone, controls anchor confidence.
    const residualChecks = solved.after.pairs.filter(pair => pair.confirmedP90 > 8);
    const residualMatches = [];
    for (const [index, pair] of residualChecks.entries()) {
      await waitForPcbResume();
      if (pcbCancelled) throw new Error('PCB-Realignment abgebrochen.');
      const saved = currentMatchNetwork().pairs.find(item => item.id === pair.id);
      text('pcbRealignStatus', `Pruefe Widerspruch ${index+1}/${residualChecks.length}: #${pair.reference} / #${pair.current}`);
      const measured = await measurePair({reference:pair.reference,current:pair.current,kind:'residual-check',cells:saved?.cells??[]},byFrame);
      residualMatches.push(measured);
      storeNetworkMatches([measured],parameters);
    }
    // A connected graph can still contain displaced visits: check real overlap,
    // independently of graph residuals, using the same bounded refinement as the UI.
    const overlapChecks=planPcbOverlapChecks(sourcePath.map(entry=>({...entry,pose:byFrame.get(entry.frame)})),
      currentMatchNetwork(),Math.min(calibration.maps.outputWidth,calibration.maps.outputHeight)*.6);
    const overlapMatches=[];
    for(const [index,pair] of overlapChecks.entries()) {
      await waitForPcbResume();
      if(pcbCancelled) throw new Error('PCB-Realignment abgebrochen.');
      text('pcbRealignStatus',`Pruefe ueberlappende Bildstreifen ${index+1}/${overlapChecks.length}: #${pair.reference} / #${pair.current}`);
      const images=[];
      try {
        for(const frame of [pair.reference,pair.current]) {
          const entry=entries.get(frame),bitmap=await pairBitmap(frame);
          const offset=entry.mode==='window'?[-bitmap.width/2,-bitmap.height/2]:
            [calibration.maps.origin[0]-calibration.field.width/2,calibration.maps.origin[1]-calibration.field.height/2];
          images.push({frame,pose:byFrame.get(frame),offset,bitmap});
        }
        const saved=currentMatchNetwork().pairs.find(p=>p.id===`${pair.reference}:${pair.current}`);
        const match=await worker.call('pcb-pair-refine',{images,pair:{...pair,cells:saved?.cells??[]},mask},images.map(i=>i.bitmap));
        overlapMatches.push(match);storeNetworkMatches([match],parameters);
      } finally { for(const item of images) item.bitmap.close(); }
      if(pcbStepRemaining>0 && --pcbStepRemaining===0){pcbPauseRequested=true;updatePcbControls();}
    }
    // Jointly solve again after intermediate measurements, so a local correction
    // propagates through every retained anchor instead of moving one frame alone.
    solved = await worker.call('pcb-graph-optimize', {graph, network:currentMatchNetwork(),
      options:{iterations:Math.max(20,parameters.iterations),huber:Math.min(4,parameters.huber),
        lever:Math.hypot(calibration.maps.outputWidth,calibration.maps.outputHeight)/2}});
    for (const item of solved.corrections) byFrame.set(item.frame,item.pose);
    const anchorAudits=[];
    const conflicts=solved.after.pairs.filter(pair=>pair.confirmedP90>4);
    for(const [index,pair]of conflicts.entries()){
      await waitForPcbResume();
      if(pcbCancelled)throw new Error('PCB-Realignment abgebrochen.');
      text('pcbRealignStatus',`Pruefe gespeicherte Anker ${index+1}/${conflicts.length}: #${pair.reference} / #${pair.current}`);
      const cells=currentMatchNetwork().pairs.find(p=>p.id===pair.id)?.cells??[];
      const audit=await measurePair({...pair,cells,anchorsOnly:true},byFrame);
      anchorAudits.push(audit);storeNetworkMatches([audit],parameters);
      if(pcbStepRemaining>0 && --pcbStepRemaining===0){pcbPauseRequested=true;updatePcbControls();}
    }
    if(anchorAudits.length){
      solved=await worker.call('pcb-graph-optimize',{graph,network:currentMatchNetwork(),
        options:{iterations:Math.max(20,parameters.iterations),huber:Math.min(4,parameters.huber),lever:Math.hypot(calibration.maps.outputWidth,calibration.maps.outputHeight)/2}});
      for(const item of solved.corrections)byFrame.set(item.frame,item.pose);
    }

    if (pcbCancelled) throw new Error('PCB-Realignment abgebrochen.');
    trackingDataset = { ...trackingDataset, pcbRealignmentDraft: undefined };
    pcbProposal = { byFrame, sourcePath, sourcePoses, maps: calibration.maps, parameters,
      keyframes: [...graphFrames],
      constraints: graphEdges, rejected: measurements.rejected, components: solved.components,
      residualChecks: residualMatches, overlapChecks:overlapMatches, anchorAudits, pointResiduals: solved.after,
      localVerification: { attempted: localMatches.length, accepted: localVerification.accepted.length,
        rejected: localVerification.rejected.length },
      pairDiagnostics: allMatches.map(match => ({ reference: match.reference, current: match.current,
        kind: match.kind, referencePose: match.referencePose, currentPose: match.currentPose,
        fft: match.fft, forward: match.forward, backward: match.backward,
        reverseDistance: match.reverseDistance, agreement: match.agreement,
        coarse: match.coarse, fftGpuVerification: match.fftGpuVerification,
        accelerator: match.accelerator })),
      localDiagnostics: [...localMatches, ...residualMatches, ...overlapMatches].map(match => ({ reference: match.reference, current: match.current,
        referencePose: match.referencePose, currentPose: match.currentPose,
        fft: match.fft, forward: match.forward, backward: match.backward,
        reverseDistance: match.reverseDistance, coarse: match.coarse,
        fftGpuVerification: match.fftGpuVerification,
        accelerator: match.accelerator })),
      beforeRms: solved.beforeRms, afterRms: solved.afterRms,
      elapsedMs: performance.now() - started };
    pcbLastDiagnostics = { parameters, pairDiagnostics: pcbProposal.pairDiagnostics, localDiagnostics: pcbProposal.localDiagnostics };
    text('pcbRealignStatus', `${graphFrames.size} Graphframes | ${allMatches.length} Paare: ${measurements.accepted.length} angenommen, ${measurements.rejected.length} verworfen | ` +
      `${solved.components.length} Komponenten; ${sourcePath.length - solved.unmeasured.length}/${sourcePath.length} Frames mit Messpunkten, ${solved.unmeasured.length} interpoliert | ` +
      `${solved.directionOnlyFrames?.length ?? 0} Frames nur quer zu Kanten gestuetzt | ` +
      `${localVerification.accepted.length}/${localMatches.length} Zwischenframes lokal bestaetigt | ` +
      `RMS ${fixed(solved.beforeRms, 2)} Ã¢â€ â€™ ${fixed(solved.afterRms, 2)} px | ` +
      `${profile.reused} Frameabrufe aus Cache, ${fixed(profile.workerMs / Math.max(1, profile.pairs) / 1000, 2)} s Rechenzeit/Paar | ` +
      `Vorschlag im XY-Pfad rot gestrichelt.`);
    text('pcbRealignDiagnostics', JSON.stringify({ components: solved.components, pairs: pcbProposal.pairDiagnostics,
      intermediatePairs: pcbProposal.localDiagnostics,
      keyframes: pcbProposal.keyframes, durationMs: Math.round(pcbProposal.elapsedMs) }, null, 2));
    refreshPcbPairDiagnostics();
    drawTrackingPath();
  } catch (error) { text('pcbRealignStatus', pcbCancelled ? 'PCB-Realignment abgebrochen.' : error.message); }
  finally { console.info('PCB-Paarmessung Laufzeitprofil', profile);
    worker.terminate(); if (pcbWorker === worker) pcbWorker = null;
    pcbResume = null; pcbBusy = false; pcbPauseRequested = false; pcbStepRemaining = 0; updatePcbControls(); }
}

function fineSourceSnapshot() {
  const maps = calibration.maps, sourceMask = pcbSourceMask();
  return { path: trackingPath, maps, video: videoInfo, brightness: brightnessCalibration,
    networkState: trackingDataset?.matchNetwork, network: currentMatchNetwork(), sourceMask,
    sourceMaskRevision: sourceMask?.revision,
    mask: sourceMask ? remapInclusionMask(sourceMask, maps) : (trackingRunOptions ?? trackingDataset?.options)?.imageMask ?? null,
    reach: Math.hypot(maps.outputWidth, maps.outputHeight),
    frames: trackingPath.map(entry => ({ frame: entry.frame, pose: entry.pose ? { ...entry.pose } : null,
      offset: entry.mode === 'window' ? [-maps.outputWidth / 2, -maps.outputHeight / 2] :
        [maps.origin[0] - calibration.field.width / 2, maps.origin[1] - calibration.field.height / 2] })) };
}
function fineSourceCurrent(source) {
  const mask = pcbSourceMask();
  return source.path === trackingPath && source.maps === calibration?.maps && source.video === videoInfo &&
    source.brightness === brightnessCalibration && source.networkState === trackingDataset?.matchNetwork &&
    source.sourceMask === mask && source.sourceMaskRevision === mask?.revision &&
    source.frames.length === trackingPath.length && source.frames.every((item, index) => {
      const entry = trackingPath[index];
      return item.frame === entry.frame && ['x','y','rotation'].every(key => item.pose?.[key] === entry.pose?.[key]);
    });
}
function refreshFinePoses() {
  invalidatePcbProposal(); pathRefitProposal = null;
  clearPathOverlay(true); renderTrackingResults(trackingPath.at(-1));
  scheduleTrackingMosaic(); updateControls();
}
function installFineControls() {
  fineUi = installFineAlignment({
    available: () => Boolean(videoInfo && calibration?.maps && trackingPath.length > 1 &&
      trackingVideoCompatible(trackingDataset, videoInfo) && !taskBusy && !trackingRunning &&
      !trackingPreviewBusy && !pcbBusy && !pathRefitBusy && !mergeRunning && !continuous && !rectifiedPlayback),
    snapshot: fineSourceSnapshot, current: fineSourceCurrent,
    busy(value) {
      fineBusy = value; taskBusy = value;
      if (value) { invalidatePcbProposal(); pathRefitProposal = null;
        trackingMosaicRequest++; clearTimeout(trackingMosaicTimer); }
      updateControls();
    },
    async bitmap(frame, source) {
      const decoded = await readTrackingFrame(frame, { rectified: true, sourceMask: source.sourceMask,
        cache: false, measureSharpness: false });
      return decoded.bitmap;
    },
    report(report) { trackingDataset = { ...trackingDataset, fineAlignment: report }; },
    apply(proposal) {
      const undo = { poses: new Map(trackingPath.map(entry => [entry.frame, entry.pose ? {...entry.pose} : null])), dataset: trackingDataset };
      for (const entry of trackingPath) if (proposal.byFrame.get(entry.frame)) entry.pose = { ...proposal.byFrame.get(entry.frame) };
      trackingDataset = { ...trackingDataset, matchNetwork: proposal.network,
        fineAlignment: { ...proposal.report, applied: true } };
      networkDeleteUndo = null; networkViewKey = '';
      refreshFinePoses(); undo.after = fineSourceSnapshot(); return undo;
    },
    undo(saved) {
      if (!fineSourceCurrent(saved.after)) return false;
      for (const entry of trackingPath) entry.pose = saved.poses.get(entry.frame);
      trackingDataset = saved.dataset; networkDeleteUndo = null; networkViewKey = '';
      refreshFinePoses(); return true;
    },
    async openFrame(frame) {
      if (taskBusy || trackingRunning || pcbBusy || pathRefitBusy || mergeRunning) throw new Error('Laufende Verarbeitung zuerst beenden.');
      const entry = trackingPath.find(item => item.frame === frame);
      const geometry = entry && frameGeometry(entry, calibration?.field, calibration?.maps);
      if (!geometry) throw new Error(`Frame #${frame} hat keine gueltige Pose.`);
      invalidatePcbProposal(); pathRefitProposal = null;
      await loadPathOverlay(geometry.world(geometry.width / 2, geometry.height / 2), { forceFrame: frame });
      if (!overlayContributors.some(item => item.entry.frame === frame)) throw new Error(`Frame #${frame} konnte nicht geladen werden.`);
      element('trackingOverlayFrame').value = String(frame);
      await selectOverlayContributor();
    }
  });
}
function installPcbRealignment() {
  installPcbPairZoom();
  element('pcbRealignDetails').addEventListener('toggle', () => {
    if (element('pcbRealignDetails').open) element('pcbPairSelect').onchange?.();
    else releasePcbPairImages();
  });
  element('pcbRealignStart').onclick = () => void runPcbRealignment();
  element('pcbRealignPause').onclick = () => { pcbPauseRequested = true; pcbStepRemaining = 0;
    text('pcbRealignStatus', 'Pause nach dem aktuellen Paar angefordert.'); updatePcbControls(); };
  element('pcbRealignResume').onclick = () => { pcbPauseRequested = false; pcbStepRemaining = 0;
    pcbResume?.(); updatePcbControls(); };
  element('pcbRealignStep').onclick = () => { pcbPauseRequested = false; pcbStepRemaining = 1;
    pcbResume?.(); updatePcbControls(); };
  element('pcbRealignCancel').onclick = () => { pcbCancelled = true; pcbResume?.(); pcbWorker?.terminate();
    text('pcbRealignStatus', 'Abbruch angefordert.'); };
  element('pcbRealignDiscard').onclick = () => { invalidatePcbProposal(); text('pcbRealignStatus', 'Vorschlag verworfen.'); };
  element('pcbRealignApply').onclick = () => {
    if (!pcbProposal || pcbBusy) return;
    const current = pcbProposal.sourcePath === trackingPath && pcbProposal.maps === calibration?.maps &&
      trackingPath.length === pcbProposal.sourcePoses.size && trackingPath.every(entry => {
        const pose = pcbProposal.sourcePoses.get(entry.frame);
        return pose && pose.x === entry.pose.x && pose.y === entry.pose.y && pose.rotation === entry.pose.rotation;
      });
    if (!current) { invalidatePcbProposal();
      text('pcbRealignStatus', 'Tracking-Posen oder Kalibrierung wurden seit der Berechnung geaendert. Erneut berechnen.'); return; }
    pcbUndo = { poses: new Map(trackingPath.map(entry => [entry.frame, { ...entry.pose }])), dataset: trackingDataset };
    const optimizedComponents = pcbProposal.components.filter(component => component.status === 'optimized').length;
    for (const entry of trackingPath) if (pcbProposal.byFrame.has(entry.frame)) entry.pose = { ...pcbProposal.byFrame.get(entry.frame) };
    trackingDataset = { ...trackingDataset, pcbRealignment: { schemaVersion: 1,
      parameters: pcbProposal.parameters, keyframes: pcbProposal.keyframes,
      constraints: pcbProposal.constraints, rejected: pcbProposal.rejected,
      components: pcbProposal.components, residualChecks: pcbProposal.residualChecks, pointResiduals: pcbProposal.pointResiduals, pairDiagnostics: pcbProposal.pairDiagnostics,
      localVerification: pcbProposal.localVerification, localDiagnostics: pcbProposal.localDiagnostics,
      originalPoses: [...pcbUndo.poses],
      beforeRms: pcbProposal.beforeRms, afterRms: pcbProposal.afterRms } };
    invalidatePcbProposal();
    text('pcbRealignStatus', `Posen fuer ${trackingPath.length} Frames uebernommen (${optimizedComponents} optimierte Teilgraphen und interpolierte Zwischenframes). Originalposen im Projektzustand gespeichert.`);
    renderTrackingResults(trackingPath.at(-1)); scheduleTrackingMosaic(); updatePcbControls();
  };
  element('pcbRealignUndo').onclick = () => {
    if (!pcbUndo || pcbBusy) return;
    for (const entry of trackingPath) if (pcbUndo.poses.has(entry.frame)) entry.pose = pcbUndo.poses.get(entry.frame);
    trackingDataset = pcbUndo.dataset; pcbUndo = null;
    text('pcbRealignStatus', 'Urspruengliche Posen wiederhergestellt.');
    renderTrackingResults(trackingPath.at(-1)); scheduleTrackingMosaic(); updatePcbControls();
  };
  for (const control of element('trackingOptimizationPanel').querySelectorAll('.pcb-realignment-card input')) control.oninput = () => {
    if (pcbProposal) { invalidatePcbProposal(); text('pcbRealignStatus', 'Parameter geaendert. Erneut berechnen.'); }
  };
  updatePcbControls();
}
let frameReductionPreview = null;
let frameReductionUndo = null;
function resetFrameReduction() {
  frameReductionPreview = null;
  frameReductionUndo = null;
  element('frameReductionApply').disabled = true;
  element('frameReductionUndo').disabled = true;
  element('frameReductionDetails').hidden = true;
  text('frameReductionStatus', 'Noch keine Auswahl berechnet.');
}
function refreshReducedTrackingPath() {
  clearPathOverlay(); pathSelection = null; pathRefitProposal = null;
  renderTrackingResults(trackingPath.at(-1));
  updateTrackingControls();
  scheduleTrackingMosaic();
}
function installFrameReduction() {
  for (const id of ['frameReductionDivisions', 'frameReductionMinimum']) element(id).oninput = () => {
    frameReductionPreview = null;
    element('frameReductionApply').disabled = true;
    text('frameReductionStatus', 'Parameter geaendert. Auswahl erneut berechnen.');
  };
  element('frameReductionCalculate').onclick = () => {
    if (trackingRunning || taskBusy || pathRefitBusy) {
      text('frameReductionStatus', 'Warten, bis die laufende Verarbeitung abgeschlossen ist.'); return;
    }
    try {
      const geometries = trackingPath.map(entry => frameGeometry(entry, calibration?.field, calibration?.maps)).filter(Boolean);
      const result = reduceFramesByBlocks(geometries, {
        divisions: number('frameReductionDivisions', 5), minimum: number('frameReductionMinimum', 3) });
      frameReductionPreview = { result, path: trackingPath, maps: calibration?.maps };
      const ids = [...result.frames].sort((a, b) => a - b);
      text('frameReductionFrames', ids.map(frame => `#${frame}`).join(', ') || 'Keine vollstaendig abdeckbaren Bloecke.');
      element('frameReductionDetails').hidden = false;
      element('frameReductionApply').disabled = !ids.length || Boolean(frameReductionUndo);
      text('frameReductionStatus', `${result.selectedFrames} von ${result.totalFrames} Frames ausgewÃƒÂ¤hlt | Blockkante ${result.blockSize} px | ${result.blocks} betroffene BlÃƒÂ¶cke` +
        (result.shortBlocks ? ` | ${result.shortBlocks} BlÃƒÂ¶cke mit weniger als M vollstÃƒÂ¤ndigen Bildern, davon ${result.uncoveredBlocks} ohne vollstÃƒÂ¤ndige Abdeckung. In diesen Bereichen kann die Reduktion Bildinhalt verlieren.` :
          ' | Alle betroffenen BlÃƒÂ¶cke erreichen M vollstÃƒÂ¤ndige Bilder.'));
    } catch (error) {
      frameReductionPreview = null; element('frameReductionApply').disabled = true;
      text('frameReductionStatus', error.message);
    }
  };
  element('frameReductionApply').onclick = () => {
    if (!frameReductionPreview || frameReductionPreview.path !== trackingPath || frameReductionPreview.maps !== calibration?.maps ||
        frameReductionUndo || trackingRunning || taskBusy || pathRefitBusy) return;
    const { result } = frameReductionPreview;
    const reduction = { created_at: new Date().toISOString(), divisions: number('frameReductionDivisions', 5),
      minimum: number('frameReductionMinimum', 3), block_size_px: result.blockSize,
      source_frames: trackingPath.length, selected_frames: result.selectedFrames,
      short_blocks: result.shortBlocks, uncovered_blocks: result.uncoveredBlocks };
    const reduced = removeUnselectedTrackingFrames({ ...trackingDataset, path: trackingPath, failures: trackingFailures }, result.frames, reduction);
    frameReductionUndo = { path: trackingPath, failures: trackingFailures, dataset: trackingDataset };
    trackingDataset = reduced;
    trackingPath = reduced.path;
    trackingFailures = reduced.failures;
    pcbProposal = null; pcbUndo = null; updatePcbControls();
    frameReductionPreview = null;
    element('frameReductionApply').disabled = true;
    element('frameReductionUndo').disabled = false;
    text('frameReductionStatus', `${reduction.source_frames - trackingPath.length} Frames aus der Registrierung entfernt; ${trackingPath.length} bleiben. RÃƒÂ¼ckgÃƒÂ¤ngig stellt den ursprÃƒÂ¼nglichen Pfad wieder her.`);
    refreshReducedTrackingPath();
  };
  element('frameReductionUndo').onclick = () => {
    if (!frameReductionUndo || trackingRunning || taskBusy || pathRefitBusy) return;
    trackingPath = frameReductionUndo.path;
    trackingFailures = frameReductionUndo.failures;
    trackingDataset = frameReductionUndo.dataset;
    pcbProposal = null; updatePcbControls();
    resetFrameReduction();
    text('frameReductionStatus', `${trackingPath.length} registrierte Frames wiederhergestellt.`);
    refreshReducedTrackingPath();
  };
}
let trackingTab = 'poses';
function activateTrackingTab(tab, { focus = false } = {}) {
  const panels = { poses: 'trackingPosesPanel', overlay: 'trackingOverlayPane', match: 'trackingMatchPanel', optimization: 'trackingOptimizationPanel' };
  if (!Object.hasOwn(panels, tab)) return;
  trackingTab = tab;
  for (const [name, panel] of Object.entries(panels)) {
    element(panel).hidden = name !== tab;
    const button = document.querySelector(`[data-tracking-tab="${name}"]`);
    button.setAttribute('aria-selected', String(name === tab));
    button.tabIndex = name === tab ? 0 : -1;
  }
  if (focus) document.querySelector(`[data-tracking-tab="${tab}"]`).focus();
  if (tab === 'overlay') requestAnimationFrame(renderPathOverlay);
}
function installTrackingTabs() {
  const tabs = [...document.querySelectorAll('[data-tracking-tab]')];
  for (const button of tabs) {
    button.onclick = () => activateTrackingTab(button.dataset.trackingTab);
    button.onkeydown = event => {
      const index = tabs.indexOf(button);
      const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length :
        event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : -1;
      if (next < 0) return;
      event.preventDefault(); activateTrackingTab(tabs[next].dataset.trackingTab, { focus: true });
    };
  }
}
let trackingFailures = [];
let trackingDataset = null;
let trackingPreviewTransform = null;
let trackingPreviewRequest = 0;
let trackingProfile = null;
let trackingZoom = 1;
let trackingPan = { x: 0, y: 0 };
let trackingRectangle = null;
let selectingTrackingWindow = false;
let trackingPreviewBusy = false;
let trackingRunOptions = null;
let trackingLost = false;
let pathProject = null;
let pathHover = null;
let pathSelection = null;
let pathZoom = 1;
let pathPan = { x: 0, y: 0 };
let networkSelectedPair = null;
let networkSelectedCell = null;
let networkHitTargets = [];
let networkDeleteUndo = null;
let networkVisiblePairs = [];
let networkViewKey = '';
const networkMeasuredMatches = new WeakSet();
const networkCalibrationKeys = new WeakMap();
function currentNetworkGeometryKey() {
  if (!calibration) return null;
  if (!networkCalibrationKeys.has(calibration)) networkCalibrationKeys.set(calibration, networkGeometryKey(calibration));
  return networkCalibrationKeys.get(calibration);
}
function currentMatchNetwork() {
  const key = currentNetworkGeometryKey();
  const network = trackingDataset?.matchNetwork;
  return network?.schemaVersion === 1 && (!network.geometryKey || network.geometryKey === key) ? network : emptyMatchNetwork(key);
}
function storeNetworkMatches(matches, parameters) {
  const pending = matches.filter(match => !networkMeasuredMatches.has(match));
  if (!pending.length) return;
  pending.forEach(match => networkMeasuredMatches.add(match));
  const previous = currentMatchNetwork();
  const network = mergeNetworkMatches(previous, pending, parameters);
  if (network === previous) return;
  const old = trackingDataset?.matchNetwork;
  if (old && old !== previous && old.geometryKey && old.geometryKey !== network.geometryKey)
    trackingDataset = {...trackingDataset, matchNetworkArchive:[...(trackingDataset.matchNetworkArchive ?? []),old]};
  trackingDataset = { ...trackingDataset, matchNetwork: network };
  networkDeleteUndo = null;
}

let trackingMosaicImage = document.createElement('canvas');
let trackingMosaicBounds = null;
let trackingMosaicRequest = 0;
let trackingMosaicTimer = 0;
let trackingMosaicBusy = false;
const trackingMosaicFrames = new Map();
let overlayRequest = 0;
let overlayGpuIdle = Promise.resolve();
let overlayGpuSession = null;
let overlayLiveRequest = 0;
let overlayLiveBitmap = null;
let overlayLiveRunning = false;
let overlayGpuInFlight = 0;
let pathOverlay = document.createElement('canvas');
let overlayContributors = [];
let overlayBounds = null;
let overlaySelectedImage = null;
let overlaySelectionRequest = 0;
let overlayPoseDraft = null;
function releaseOverlayGpuSession() {
  overlayLiveRequest++;
  const session = overlayGpuSession; overlayGpuSession = null;
  if (session) void overlayGpuIdle.then(() => session.renderer.destroy());
}
function clearPathOverlay(preserveGpu = false) {
  if (!preserveGpu) releaseOverlayGpuSession();
  overlayLiveRequest++; overlayLiveBitmap?.close(); overlayLiveBitmap = null;
  closeOverlayTiles(pathOverlay);
  if (!pathOverlay.tiles) pathOverlay.width = pathOverlay.height = 0;
  pathOverlay = document.createElement('canvas');
  overlaySelectedImage?.bitmap.close();
  overlaySelectedImage = null;
  overlayPoseDraft = null;
  overlayContributors = [];
  overlayBounds = null;
  overlaySelectionRequest++;
  const selector = element('trackingOverlayFrame');
  selector.replaceChildren(new Option('Mischbild', ''));
  selector.disabled = true;
  element('trackingOverlayInspect').disabled = true;
  updateOverlayPoseControls();
}
function updateOverlayPoseControls() {
  const changed = overlayPoseDraft && (overlayPoseDraft.pose.x !== overlayPoseDraft.originalPose.x ||
    overlayPoseDraft.pose.y !== overlayPoseDraft.originalPose.y ||
    overlayPoseDraft.pose.rotation !== overlayPoseDraft.originalPose.rotation);
  const busy = trackingRunning || taskBusy || pathRefitBusy || pcbBusy || Boolean(pathRefitProposal || pcbProposal);
  element('trackingOverlayPoseApply').disabled = !changed || busy;
  element('trackingOverlayPoseDiscard').disabled = !changed;
}
let overlayZoom = 1;
let overlayPan = { x: 0, y: 0 };
const mergePreviewImage = document.createElement('canvas');
let mergeZoom = 1;
let mergePan = { x: 0, y: 0 };
let pathRefitProposal = null;
let pathRefitBusy = false;
let pathRefitWorker = null;
let pathRefitCancelRequested = false;
let trackingImageMask = null;
let trackingMaskHasSelection = false;
let trackingMaskTool = 'include';
let trackingMaskPreviewRequest = 0;
const trackingMaskPreview = document.createElement('canvas');
const trackingMaskPaintLayer = document.createElement('canvas');
let trackingMaskTransform = null;
let mergeRunning = false;
let mergeCancelled = false;
let mergeResultStatus = null;
let mergeDownloads = [];
let mergeTilePlan = null;
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
  const frameCacheStats = frameReader.cacheStats();
  footer.replaceChildren();
  const entries = [
    ['Beobachtungspunkte', framePoints.toLocaleString('de-DE')],
    ['2D-Vektoren', (2 * pointCount).toLocaleString('de-DE')],
    ['Vektor-Nutzlast (geschaetzt)', megabytes(pointBytes)],
    ['Kalibrierungsarrays', megabytes(arrayBytes)],
    ['Bildpuffer', megabytes(canvasBytes)],
    ['Entzerrter Framecache', `${frameCacheStats.frames.toLocaleString('de-DE')} Frames / ${megabytes(frameCacheStats.bytes)}`],
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

let operationTiming = null;

function etaLabel(seconds) {
  const rounded = Math.max(1, Math.round(seconds));
  if (rounded < 60) return `${rounded} s`;
  if (rounded < 3600) return `${Math.floor(rounded / 60)} min ${rounded % 60} s`;
  return `${Math.floor(rounded / 3600)} h ${Math.floor(rounded % 3600 / 60)} min`;
}

function operationProgress(status, value, key = status) {
  const now = performance.now(), fraction = Math.max(0, Math.min(1, value));
  if (!operationTiming || operationTiming.key !== key || fraction < operationTiming.value) {
    operationTiming = { key, started: now, value: fraction, sampled: now, rate: 0 };
  } else if (fraction > operationTiming.value && now > operationTiming.sampled) {
    const rate = (fraction - operationTiming.value) / ((now - operationTiming.sampled) / 1000);
    operationTiming.rate = operationTiming.rate ? operationTiming.rate * 0.8 + rate * 0.2 : rate;
    operationTiming.value = fraction; operationTiming.sampled = now;
  }
  const elapsed = (now - operationTiming.started) / 1000;
  const remaining = operationTiming.rate > 0 ? (1 - fraction) / operationTiming.rate : NaN;
  const eta = elapsed >= 3 && remaining >= 2 && Number.isFinite(remaining) ? ` | ETA ${etaLabel(remaining)}` : '';
  text('processingStatus', `${status}${eta}`);
  element('progress').value = fraction;
}

function showProgress(progress) {
  const names = { index: 'MP4-Index', forward: 'Dichte Vorwaertsmap', inverse: 'Inverse Map', fit: 'Gemeinsamer Fit' };
  const accelerator = progress.accelerator ? ` | ${progress.accelerator}` : '';
  const speed = Number.isFinite(progress.iterationsPerSecond) && Number.isFinite(progress.iterationSeconds) ?
    ` | Mittel ${progress.iterationsPerSecond.toPrecision(3)} Iter./s | letzte ${fixed(progress.iterationSeconds, 2)} s` : '';
  const indexing = progress.stage === 'index' && Number.isFinite(progress.done) && Number.isFinite(progress.total) ?
    ` ${fixed(100 * progress.done / Math.max(1, progress.total), 1)}% | ${megabytes(progress.done)} / ${megabytes(progress.total)}` : '';
  const status = progress.stage === 'fit' ? `Fit ${progress.iteration}/${progress.iterations} | RMS ${fixed(progress.rms)} px${accelerator}${speed}` :
    `${names[progress.stage] || progress.stage}${indexing}${accelerator}`;
  operationProgress(status, progress.stage === 'fit' ? progress.iteration / progress.iterations : (progress.done || 0) / (progress.total || 1),
    `worker-${progress.stage}`);
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
    `${detection.accelerator || 'CPU'} | Frame ${currentIndex} | ${detection.points.length} Checkerboard-Ecken`);
  updateTimeProfiles();
}

function parameters() {
  const width = videoInfo?.width ?? calibration?.field.width;
  const height = videoInfo?.height ?? calibration?.field.height;
  const settings = { width, height, gridMm: number('gridMm', null), step: number('step', 0),
    approxStep: number('approxStep', 0), columns: number('columns', null), rows: number('rows', null),
    threshold: number('threshold', 16), spacing: number('spacing', Math.round(Math.max(width || 640, height || 480) / 4)),
    lambda: number('lambda', 0.01), sigma: number('sigma', 1), delta: number('delta', 1.5), tau: number('tau', 0.12),
    iterations: number('iterations', 35), acceptance: number('acceptance', 1), useWebGpu: useWebGpu(), interval: 0,
    minMotion: number('minMotion', 0), updateEvery: number('updateEvery', 4), validationFrom: number('validationFrom', 80),
    startPercent: number('startPercent', 0), endPercent: number('endPercent', 100) };
  if (!(settings.spacing >= 16 && settings.minMotion >= 0 && settings.endPercent > settings.startPercent &&
    settings.sigma > 0 && settings.delta > 0 && settings.lambda >= 0 && settings.tau > 0 && settings.iterations >= 5 && settings.iterations <= 300 &&
    settings.threshold > 0 && settings.acceptance > 0 && settings.updateEvery >= 2 &&
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
}

function updateCorrectionDataStatus() {
  for (const [id, label, present] of [
    ['lensDataStatus', 'Linsenkalibrierung', Boolean(calibration?.maps)],
    ['trackingDataStatus', 'XYR-Tracking', trackingPath.length > 0],
    ['brightnessDataStatus', 'Helligkeitskorrektur', Boolean(brightnessCalibration)]
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
  pathZoom = 1; pathPan = { x: 0, y: 0 };
  trackingImageMask = null;
  trackingMaskHasSelection = false;
  trackingDataset = tracking;
  trackingPath = tracking.path;
  fineUi?.restore(tracking.fineAlignment);
  trackingDataset.matchNetwork = migrateMatchNetwork(tracking, currentNetworkGeometryKey());
  networkSelectedPair = null; networkSelectedCell = null; networkDeleteUndo = null; networkViewKey = '';

  trackingFailures = Array.isArray(tracking.failures) ? tracking.failures : [];
  pcbUndo = null;
  if (tracking.pcbRealignmentDraft?.schemaVersion === 1)
    text('pcbRealignStatus', `PCB-Realignment unterbrochen: ${tracking.pcbRealignmentDraft.matches?.length ?? 0} Paare gespeichert. Nach Zuordnung des passenden Videos erneut Berechnen waehlen.`);
  if (Array.isArray(tracking.pcbRealignment?.originalPoses)) {
    const original = new Map(tracking.pcbRealignment.originalPoses);
    pcbUndo = { poses: original, dataset: { ...tracking, pcbRealignment: undefined } };
    updatePcbControls();
    text('pcbRealignStatus', `PCB-Realignment geladen: ${tracking.pcbRealignment.keyframes?.length ?? 0} Keyframes; Posekorrektur kann rueckgaengig gemacht werden.`);
    refreshPcbPairDiagnostics();
  }
  if (tracking.reduction) text('frameReductionStatus', `${trackingPath.length} registrierte Frames nach Framereduzierung. Die entfernten Frames sind nicht im geladenen Datensatz.`);
  const last = [...trackingPath].reverse().find(entry => entry.pose);
  if (!last) return;
  trackingRectangle = last.rectangle ?? tracking.rectangle ?? tracking.options?.rectangle ?? trackingRectangle;
  trackingRunOptions = { ...(tracking.options ?? {}), mode: 'window', rectangle: trackingRectangle };
  trackingNextIndex = last.frame + 1;
  trackingNeedsSeed = true;
  trackingLost = false;
  if (videoInfo && number('trackingEnd', 0) < trackingNextIndex) element('trackingEnd').value = videoInfo.frameCount - 1;
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
  // Legacy packages need only their final pose and frame; context history is rebuilt after resuming.
  trackingPaused = true;
  renderTrackingResults(trackingPath.at(-1));
  scheduleTrackingMosaic();
  updateTrackingControls();
  updateCorrectionDataStatus();
}

function updateControls() {
  checkerboardView.refresh();
  updateCorrectionDataStatus();
  const hasVideo = Boolean(videoInfo);
  updateGpuAdapterControl();
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
  const brightnessFrames = [...frames.values()].filter(frame =>
    frame.accepted && frame.enabled && !frame.patchSize && frame.points.length >= 4).length;
  const brightnessReady = Boolean(calibration && hasVideo && brightnessFrames >= 5 &&
    !continuous && !rectifiedPlayback && !trackingRunning);
  element('brightnessFit').disabled = !brightnessReady || taskBusy || brightnessRunning;
  element('brightnessPause').disabled = !brightnessRunning;
  element('brightnessResume').disabled = !brightnessReady || taskBusy || brightnessRunning || !brightnessPaused || !brightnessSession;
  element('brightnessReset').disabled = taskBusy || brightnessRunning || (!brightnessSession && !brightnessCalibration);
  element('brightnessDisable').disabled = !brightnessCalibration || taskBusy || trackingRunning || brightnessRunning;
  element('brightnessSave').disabled = !brightnessCalibration || taskBusy || trackingRunning || brightnessRunning;
  element('brightnessLoad').disabled = !calibration || taskBusy || trackingRunning || brightnessRunning;
  element('brightnessPasses').disabled = taskBusy || Boolean(brightnessSession);
  element('brightnessMaxGain').disabled = taskBusy || Boolean(brightnessSession);
  for (const control of document.querySelectorAll('.settings input, .settings select, .settings textarea')) {
    if (control.id !== 'follow' && control.id !== 'opticalConfiguration') control.disabled = taskBusy || continuous || trackingRunning ||
      (Boolean(brightnessSession) && ['brightnessPasses', 'brightnessMaxGain'].includes(control.id));
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
  updateMergeControls();
  updateTrackingControls();
  updatePcbControls();
}

async function task(operation) {
  if (taskBusy) return;
  taskBusy = true;
  updateControls();
  try { await operation(); }
  catch (error) { message(error.message, true); continuous = false; }
  finally { taskBusy = false; operationTiming = null; updateControls(); if (!continuous) text('processingStatus', 'Bereit'); }
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
  if (settings.useWebGpu) {
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
  if (frames.has(index) && !detectionsStale && !force && !retryMotionRejection) {
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
    confidence: detection.confidence, accelerator: detection.accelerator,
    sharpness: detection.sharpness, timing: detection.timing,
    reason: reason || 'Brauchbares Checkerboard' };
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

function drawOverlay(context, scale) {
  if (!element('overlay').checked || !currentDetection) return;
  const detection = currentDetection;
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
  fineUi?.refresh();
  updateGpuAdapterControl();
  element('trackingSearchRadius').max = '1024';
  const ready = Boolean(videoInfo && calibration && videoInfo.width === calibration.field.width && videoInfo.height === calibration.field.height && !taskBusy && !continuous && !rectifiedPlayback);
  const canContinue = trackingPath.some(entry => entry.pose) && trackingNextIndex <= number('trackingEnd', 0);
  element('trackingStartButton').disabled = !ready || trackingRunning || trackingPreviewBusy || !trackingRectangle;
  element('trackingPauseButton').disabled = !trackingRunning;
  element('trackingResumeButton').disabled = !ready || trackingRunning || trackingPreviewBusy || !trackingRectangle || Boolean(trackingDataset?.reduction) ||
    trackingLost || (!trackingPaused && !canContinue) || trackingNextIndex > number('trackingEnd', 0);
  element('trackingResetButton').disabled = taskBusy || trackingRunning || trackingPreviewBusy;
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
  trackingComputer = new WorkerClient('./compute-worker.js', showProgress);
}

function clearTrackingResults(preserveRectangle = false) {
  fineUi?.restore(null);
  pcbCancelled = true; pcbWorker?.terminate(); pcbProposal = null; pcbUndo = null;
  resetFrameReduction();
  trackingInspector.clear();
  overlayRequest++;
  trackingMosaicRequest++; clearTimeout(trackingMosaicTimer);
  trackingMosaicImage.width = trackingMosaicImage.height = 0; trackingMosaicBounds = null;
  trackingMosaicFrames.clear();
  pathHover = null; pathSelection = null; pathProject = null;
  pathRefitProposal = null; pathRefitBusy = false;
  clearPathOverlay();
  element('trackingOverlayEmpty').hidden = false;
  text('trackingMatchTabFrame', '');
  element('trackingMatchEmpty').hidden = false;
  activateTrackingTab('poses');
  trackingPreviewRequest++;
  trackingRunning = false;
  trackingPaused = false;
  trackingNextIndex = number('trackingStart', 0);
  trackingNeedsSeed = false;
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
  trackingZoom = 1;
  trackingPan = { x: 0, y: 0 };
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
  if (trackingRectangle) {
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
  renderTrackingPreview();
}

async function showTrackedFrame(entry) {
  if (trackingRunning || trackingPreviewBusy || taskBusy) return;
  trackingInspector.show(entry);
  text('trackingMatchTabFrame', `(#${entry.frame})`);
  element('trackingMatchEmpty').hidden = true;
  activateTrackingTab('match');
  const request = ++trackingPreviewRequest;
  trackingPreviewBusy = true; updateTrackingControls();
  try {
    const decoded = await readTrackingFrame(entry.frame, { rectified: true });
    if (request !== trackingPreviewRequest) { decoded.bitmap.close(); return; }
    trackingRectangle = entry.rectangle ?? trackingRectangle; drawTrackingFrame(decoded.bitmap); decoded.bitmap.close();
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
  if (!videoInfo || !calibration || trackingRunning || trackingPreviewBusy || taskBusy) return;
  const index = Math.max(0, Math.min(videoInfo.frameCount - 1, Math.round(number('trackingStart', 0))));
  const request = ++trackingPreviewRequest;
  trackingPreviewBusy = true; updateTrackingControls();
  text('trackingPreviewState', `Lade Frame ${index}...`);
  try {
    const decoded = await readTrackingFrame(index, { rectified: true });
    if (request !== trackingPreviewRequest || trackingRunning) { decoded.bitmap.close(); return; }
    drawTrackingFrame(decoded.bitmap); decoded.bitmap.close();
    text('trackingFrameState', `Vorschau Frame ${index} / PTS ${(decoded.timestamp / 1e6).toFixed(6)} s`);
    text('trackingPreviewState', 'Entzerrter Startframe');
  } catch (error) {
    if (request === trackingPreviewRequest) text('trackingPreviewState', `Vorschau fehlgeschlagen: ${error.message}`);
  } finally { trackingPreviewBusy = false; updateTrackingControls(); }
}

async function renderTrackingMosaic() {
  if (fineBusy || mergeRunning || trackingRunning || pathRefitBusy || overlayGpuInFlight || trackingMosaicBusy || !videoInfo || !calibration?.maps || !trackingPath.some(entry => entry.pose)) return;
    trackingMosaicBusy = true;
    const request = trackingMosaicRequest;
    const geometries = trackingPath.map(trackingGeometry).filter(Boolean);
    const bounds = mergeBounds(geometries);
    const selected = sparsePathFrames(geometries, 96);
    try {
      if (!bounds || !selected.length) return;
      const scale = Math.min(1, 640 / Math.max(bounds.width, bounds.height));
      const composite = document.createElement('canvas');
      composite.width = Math.max(1, Math.ceil(bounds.width * scale));
      composite.height = Math.max(1, Math.ceil(bounds.height * scale));
      const context = composite.getContext('2d');
      const local = document.createElement('canvas');
      for (const geometry of selected) {
        if (fineBusy || mergeRunning || trackingRunning || request !== trackingMosaicRequest) return;
        const maps = calibration.maps, brightness = brightnessCalibration;
        const localScale = Math.min(1, 640 / Math.max(geometry.width, geometry.height));
        local.width = Math.max(1, Math.ceil(geometry.width * localScale));
        local.height = Math.max(1, Math.ceil(geometry.height * localScale));
        const localContext = local.getContext('2d', { willReadFrequently: true });
        let thumbnail = trackingMosaicFrames.get(geometry.entry.frame);
        if (!thumbnail || thumbnail.decoder !== decoder || thumbnail.maps !== maps ||
            thumbnail.brightness !== brightness || thumbnail.image.width !== local.width ||
            thumbnail.image.height !== local.height) {
          const decoded = await readTrackingFrame(geometry.entry.frame, { rectified: true });
          try {
            if (fineBusy || mergeRunning || trackingRunning || request !== trackingMosaicRequest) return;
            localContext.clearRect(0, 0, local.width, local.height);
            localContext.drawImage(decoded.bitmap, 0, 0, local.width, local.height);
            thumbnail = { decoder, maps, brightness, image: localContext.getImageData(0, 0, local.width, local.height) };
            trackingMosaicFrames.set(geometry.entry.frame, thumbnail);
          } finally { decoded.bitmap.close(); }
        }
        const image = new ImageData(new Uint8ClampedArray(thumbnail.image.data), local.width, local.height);
        for (let y = 0; y < local.height; y++) for (let x = 0; x < local.width; x++) {
          const sourceX = (x + 0.5) / localScale;
          const sourceY = (y + 0.5) / localScale;
          const alpha = (y * local.width + x) * 4 + 3;
          const weight = trackingPixelAllowed(sourceX, sourceY) ?
            edgeFeatherWeight(sourceX, sourceY, geometry.width, geometry.height, 0.1) : 0;
          image.data[alpha] = Math.round(image.data[alpha] * weight);
        }
        localContext.putImageData(image, 0, 0);
        const origin = geometry.world(0, 0);
        const factor = scale / localScale;
        context.setTransform(factor * geometry.c, factor * geometry.s, -factor * geometry.s, factor * geometry.c,
          scale * (origin.x - bounds.minX), scale * (origin.y - bounds.minY));
        context.drawImage(local, 0, 0);
      }
      if (!trackingPath.length) return;
      trackingMosaicImage = composite; trackingMosaicBounds = bounds;
      drawTrackingPath();
    } catch (error) {
      if (request === trackingMosaicRequest) console.warn('Tracking-Mosaik:', error.message);
    } finally {
      trackingMosaicBusy = false;
      if (request !== trackingMosaicRequest && trackingPath.length && !fineBusy && !mergeRunning && !trackingRunning && !pathRefitBusy && !overlayGpuInFlight) {
        clearTimeout(trackingMosaicTimer);
        trackingMosaicTimer = setTimeout(() => void renderTrackingMosaic(), 100);
      }
    }
}

function scheduleTrackingMosaic() {
  trackingMosaicRequest++;
  clearTimeout(trackingMosaicTimer);
  if (fineBusy || mergeRunning || trackingRunning || pathRefitBusy) return;
  trackingMosaicTimer = setTimeout(() => void renderTrackingMosaic(), 250);
}

function displayedNetworkStrength(pair) {
  let quality=0,confirmed=0;
  for(const cell of pair.cells){quality+=cell.quality??1;confirmed+=(cell.quality??1)*(cell.confidence??1);}
  return pair.weight*(quality?confirmed/quality:1);
}
function refreshMatchNetworkControls(network, pairs, poses) {
  const shown = element('trackingNetworkShow').checked;
  const key = `${network.revision}|${pairs.map(pair=>pair.id).join(',')}|${networkSelectedPair}|${networkSelectedCell}|${shown}`;
  if (key !== networkViewKey) {
    networkViewKey = key;
    const selector = element('trackingNetworkPair');
    selector.replaceChildren(new Option('Verbindung auswaehlen', ''));
    for (const pair of pairs) selector.add(new Option(`#${pair.reference} \u2194 #${pair.current} | ${pair.cells.length} Messpunkte | Staerke ${fixed(displayedNetworkStrength(pair),1)}`,pair.id));
    selector.value = networkSelectedPair ?? '';
    const cells = element('trackingNetworkCell');
    cells.replaceChildren(new Option('Alle Messpunkte', ''));
    const pair = pairs.find(pair => pair.id === networkSelectedPair);
    for (const cell of pair?.cells ?? []) cells.add(new Option(`${cell.normal?'Kante':'Zelle'} ${cell.sourceCellId ?? cell.id} | ${cell.psr === null ? 'Geometrie' : `PSR ${fixed(cell.psr,1)}`} | Vertrauen ${fixed(100*(cell.confidence??1),0)}%`, cell.id));
    cells.value = networkSelectedCell ?? '';
  }
  const pair = pairs.find(pair => pair.id === networkSelectedPair);
  const busy = trackingRunning || taskBusy || pathRefitBusy || pcbBusy;
  element('trackingNetworkPair').disabled = !shown || !pairs.length;
  element('trackingNetworkCell').disabled = !shown || !pair?.cells.length;
  element('trackingNetworkDeleteCell').disabled = !shown || busy || !pair?.cells.some(cell => cell.id === networkSelectedCell);
  element('trackingNetworkDeletePair').disabled = !shown || busy || !pair;
  element('trackingNetworkUndo').disabled = busy || !networkDeleteUndo;
  const cells = pair ? projectNetworkCells(pair,poses) : [];
  const cell = cells.find(cell => cell.id === networkSelectedCell);
  const normalAngle=pair?poses.get(pair.reference)?.rotation??0:0;
  const normal=cell?.normal?{x:Math.cos(normalAngle)*cell.normal.x-Math.sin(normalAngle)*cell.normal.y,
    y:Math.sin(normalAngle)*cell.normal.x+Math.cos(normalAngle)*cell.normal.y}:null;
  const error = cell ? normal?Math.abs((cell.referenceWorld.x-cell.currentWorld.x)*normal.x+(cell.referenceWorld.y-cell.currentWorld.y)*normal.y):
    Math.hypot(cell.referenceWorld.x-cell.currentWorld.x,cell.referenceWorld.y-cell.currentWorld.y) : null;
  const count = pairs.reduce((sum,pair)=>sum+pair.cells.length,0);
  const inactive = pair && !networkEdges({...network,pairs:[pair]}).length;
  text('trackingNetworkInfo', !shown ? 'Match-Netz ausgeblendet.' :
    pair ? `#${pair.reference} \u2194 #${pair.current} | ${pair.cells.length} Messpunkte | Staerke ${fixed(displayedNetworkStrength(pair),1)} | ${pair.method}` +
      `${pair.approximate ? ' | Punktlagen aus gespeicherter Bildpose rekonstruiert' : ''}` +
      `${inactive ? ' | Zu wenige verteilte Messpunkte: Verbindung inaktiv' : ''}` +
      `${error !== null ? ` | ${normal?'Kantenabstand quer':'Punktabstand'} ${fixed(error,2)} px` : ''}` +
      `${normal?' | Wirkt nur quer zur Kante; entlang bleibt frei':''}` +
      `${cell?.validationReason ? ` | Nachpruefung: ${cell.validationReason} (Vertrauen ${fixed(100*(cell.confidence??1),0)}%)` : ''}` :
    `${pairs.length}/${network.pairs.length} Bildverbindungen | ${count} Messpunkte der aktiven Bilder. Punkte oder Linien anklicken; Gruen: stark, Orange: schwach.`);
}
function drawMatchNetwork(context, project, entries, geometries, width, height) {
  const network = currentMatchNetwork();
  const point = pathSelection ?? pathHover;
  const active = overlayContributors.length ? new Set(overlayContributors.map(g=>g.entry.frame)) :
    new Set(point ? geometries.filter(g=>g.supports(point,trackingPixelAllowed)).map(g=>g.entry.frame) : []);
  const poses = new Map(entries.map(entry=>[entry.frame,entry.pose]));
  if (overlayPoseDraft) poses.set(overlayPoseDraft.frame,overlayPoseDraft.pose);
  const all = element('trackingNetworkAll').checked;
  networkVisiblePairs = network.pairs.filter(pair=>poses.has(pair.reference)&&poses.has(pair.current)&&
    (all||active.has(pair.reference)||active.has(pair.current)));
  if (!networkVisiblePairs.some(pair=>pair.id === networkSelectedPair)) {
    networkSelectedPair = null; networkSelectedCell = null;
  }
  refreshMatchNetworkControls(network,networkVisiblePairs,poses);
  networkHitTargets = [];
  if (!element('trackingNetworkShow').checked) return;
  context.save();
  const visible = (a,b) => Math.max(a.x,b.x)>=-8 && Math.min(a.x,b.x)<=width+8 &&
    Math.max(a.y,b.y)>=-8 && Math.min(a.y,b.y)<=height+8;
  for (const pair of networkVisiblePairs) {
    const selected = pair.id === networkSelectedPair;
    const strength = Math.min(1,Math.log1p(displayedNetworkStrength(pair))/Math.log(257));
    const color = selected ? '#663bb2' : strength>.6 ? '#087d63' : '#b87912';
    const a = project(poses.get(pair.reference)), b = project(poses.get(pair.current));
    context.strokeStyle=color; context.globalAlpha=selected?.8:.25; context.lineWidth=selected?2:1;
    if (visible(a,b)) { context.beginPath();context.moveTo(a.x,a.y);context.lineTo(b.x,b.y);context.stroke();
      networkHitTargets.push({pairId:pair.id,cellId:null,a,b}); }
    for (const cell of projectNetworkCells(pair,poses)) {
      const a=project(cell.referenceWorld),b=project(cell.currentWorld);
      if (!visible(a,b)) continue;
      const highlighted=selected&&cell.id===networkSelectedCell;
      context.globalAlpha=highlighted?1:selected?.75:.3+.4*strength;
      context.strokeStyle=highlighted?'#cb294b':(cell.confidence??1)<.5?'#b87912':color;context.fillStyle=context.strokeStyle;
      context.lineWidth=highlighted?2.5:.75+strength;
      context.beginPath();context.moveTo(a.x,a.y);context.lineTo(b.x,b.y);context.stroke();
      for (const p of [a,b]) {context.beginPath();context.arc(p.x,p.y,highlighted?5:selected?3:2,0,2*Math.PI);context.stroke();}
      networkHitTargets.push({pairId:pair.id,cellId:cell.id,a,b});
    }
  }
  context.restore();
}
function hitMatchNetwork(point) {
  let nearest=null,best=7;
  for (const target of networkHitTargets) {
    const dx=target.b.x-target.a.x,dy=target.b.y-target.a.y;
    const denominator=dx*dx+dy*dy;
    const t=denominator?Math.max(0,Math.min(1,((point.x-target.a.x)*dx+(point.y-target.a.y)*dy)/denominator)):0;
    const distance=Math.hypot(point.x-target.a.x-t*dx,point.y-target.a.y-t*dy)+(target.cellId===null?1:0);
    if(distance<best){best=distance;nearest=target;}
  }
  return nearest;
}
function installMatchNetworkControls() {
  for(const id of ['trackingNetworkShow','trackingNetworkAll'])element(id).onchange=()=>{networkViewKey='';drawTrackingPath();};
  element('trackingNetworkPair').onchange=event=>{
    networkSelectedPair=event.target.value||null;networkSelectedCell=null;networkViewKey='';drawTrackingPath();
  };
  element('trackingNetworkCell').onchange=event=>{
    networkSelectedCell=event.target.value||null;networkViewKey='';drawTrackingPath();
  };
  const remove=cell=>{
    if(trackingRunning||taskBusy||pathRefitBusy||pcbBusy||!networkSelectedPair)return;
    const network=currentMatchNetwork();
    const next=deleteNetworkMatch(network,networkSelectedPair,cell?networkSelectedCell:null);
    if(next===network)return;
    networkDeleteUndo=network;
    trackingDataset={...trackingDataset,matchNetwork:next};
    if(pcbUndo)pcbUndo.dataset={...pcbUndo.dataset,matchNetwork:next};
    pcbProposal=null;pathRefitProposal=null;updatePcbControls();updatePathRefitControls();
    text('trackingPathRefitInfo','Match-Netz geaendert. Positionsvorschlag erneut berechnen.');
    if(!cell)networkSelectedPair=null;
    networkSelectedCell=null;networkViewKey='';drawTrackingPath();
    if(pathSelection)void loadPathOverlay(pathSelection);
  };
  element('trackingNetworkDeleteCell').onclick=()=>remove(true);
  element('trackingNetworkDeletePair').onclick=()=>remove(false);
  element('trackingNetworkUndo').onclick=()=>{
    if(!networkDeleteUndo||trackingRunning||taskBusy||pathRefitBusy||pcbBusy)return;
    trackingDataset={...trackingDataset,matchNetwork:networkDeleteUndo};
    if(pcbUndo)pcbUndo.dataset={...pcbUndo.dataset,matchNetwork:networkDeleteUndo};
    networkDeleteUndo=null;pcbProposal=null;pathRefitProposal=null;
    updatePcbControls();updatePathRefitControls();networkViewKey='';drawTrackingPath();
    if(pathSelection)void loadPathOverlay(pathSelection);
  };
}

function drawTrackingPath() {
  const poses = trackingPath.filter(entry => entry.pose);
  const { context, width, height } = trackingCanvasContext('trackingPathCanvas');
  context.clearRect(0, 0, width, height);
  if (!poses.length) { networkHitTargets=[];networkVisiblePairs=[];refreshMatchNetworkControls(currentMatchNetwork(),[],new Map());pathProject = null; pathZoom = 1; pathPan = { x: 0, y: 0 };
    element('trackingPathEmpty').hidden = false; return; }
  element('trackingPathEmpty').hidden = true;
  const geometries = trackingPath.map(entry => frameGeometry(entry, calibration?.field, calibration?.maps)).filter(Boolean);
  const corners = geometries.flatMap(geometry => geometry.corners);
  const activeProposal = pathRefitProposal?.byFrame ?? pcbProposal?.byFrame;
  const proposed = activeProposal ? [...activeProposal.values()].map(item => item.pose ?? item) : [];
  const bounds = pointBounds([poses.map(entry => entry.pose), proposed, corners], { x: 0, y: 0 });
  const minimumX = bounds.minX; const maximumX = bounds.maxX;
  const minimumY = bounds.minY; const maximumY = bounds.maxY;
  const rangeX = Math.max(1, maximumX - minimumX);
  const rangeY = Math.max(1, maximumY - minimumY);
  const padding = 28;
  const scale = Math.min((width - 2 * padding) / rangeX, (height - 2 * padding) / rangeY);
  const centerX = (minimumX + maximumX) / 2, centerY = (minimumY + maximumY) / 2;
  const clampPan = (pan, extent, size) => extent <= size - 2 * padding ? 0 :
    Math.max(size - padding - (size + extent) / 2, Math.min(padding - (size - extent) / 2, pan));
  pathPan.x = clampPan(pathPan.x, rangeX * scale * pathZoom, width);
  pathPan.y = clampPan(pathPan.y, rangeY * scale * pathZoom, height);
  const project = pose => ({ x: width / 2 + (pose.x - centerX) * scale * pathZoom + pathPan.x,
    y: height / 2 + (pose.y - centerY) * scale * pathZoom + pathPan.y });
  pathProject = project;
  pathProject.invert = point => ({ x: centerX + (point.x - width / 2 - pathPan.x) / (scale * pathZoom),
    y: centerY + (point.y - height / 2 - pathPan.y) / (scale * pathZoom) });
  if (trackingMosaicImage.width && trackingMosaicBounds) {
    const topLeft = project({ x: trackingMosaicBounds.minX, y: trackingMosaicBounds.minY });
    const bottomRight = project({ x: trackingMosaicBounds.maxX, y: trackingMosaicBounds.maxY });
    context.save(); context.globalAlpha = 0.82;
    context.drawImage(trackingMosaicImage, topLeft.x, topLeft.y,
      bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
    context.restore();
  }
  const selected = pathHover ?? pathSelection;
  const supporting = selected ? geometries.filter(geometry => geometry.supports(selected, trackingPixelAllowed)) : [];
  const visibleSupporting = evenlySpaced(supporting, 96);
  for (const geometry of visibleSupporting) {
    const points = geometry.corners.map(project);
    context.strokeStyle = supportColor(geometry.entry.frame); context.lineWidth = 1;
    context.beginPath(); points.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
    context.closePath(); context.stroke();
    context.fillStyle = context.strokeStyle; context.font = '11px sans-serif';
    context.fillText(`#${geometry.entry.frame}`, points[0].x + 3, points[0].y - 3);
  }
  const frameLabels = visibleSupporting.slice(0, 12).map(item => `#${item.entry.frame}`).join(', ');
  text('trackingPathInfo', selected ? `Position (${fixed(selected.x, 1)}, ${fixed(selected.y, 1)}) | ${supporting.length} Bilder` +
    (frameLabels ? ` | Anzeige: ${frameLabels}${visibleSupporting.length > 12 ? ', ...' : ''}` : '') :
    'Maus im Canvas: Bildrechtecke anzeigen. Klicken: Frames an dieser Position ueberlagern.');
  const origin = project({ x: 0, y: 0 });
  context.strokeStyle = '#aebbb5'; context.lineWidth = 1; context.beginPath();
  context.moveTo(padding, origin.y); context.lineTo(width - padding, origin.y);
  context.moveTo(origin.x, padding); context.lineTo(origin.x, height - padding); context.stroke();
  context.strokeStyle = '#067566'; context.lineWidth = 2; context.beginPath();
  poses.forEach((entry, index) => { const point = project(entry.pose); index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y); });
  context.stroke();
  if (activeProposal) {
    context.strokeStyle = '#c6404a'; context.lineWidth = 2; context.setLineDash([6, 4]); context.beginPath();
    poses.forEach((entry, index) => {
      const proposedPose = activeProposal.get(entry.frame);
      const point = project(proposedPose?.pose ?? proposedPose ?? entry.pose);
      index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y);
    });
    context.stroke(); context.setLineDash([]);
  }
  drawMatchNetwork(context, project, poses, geometries, width, height);
  const latest = project(poses.at(-1).pose);
  context.fillStyle = '#c6404a'; context.beginPath(); context.arc(latest.x, latest.y, 4, 0, Math.PI * 2); context.fill();
  if (selected) {
    const point = project(selected);
    context.strokeStyle = '#111'; context.beginPath(); context.arc(point.x, point.y, 6, 0, Math.PI * 2); context.stroke();
  }
}

function updatePathRefitControls() {
  updateGpuAdapterControl();
  element('trackingPathRefit').disabled = pathRefitBusy || Boolean(pathRefitProposal) || !pathSelection || trackingPath.length < 2;
  element('trackingPathRefitApply').disabled = pathRefitBusy || !pathRefitProposal?.overlayReady;
  const discard = element('trackingPathRefitDiscard');
  discard.disabled = !pathRefitBusy && !pathRefitProposal;
  discard.querySelector('span').textContent = pathRefitBusy ? 'Abbrechen' : 'Verwerfen';
  discard.title = pathRefitBusy ? 'Laufenden lokalen Refit abbrechen' : 'Refit-Vorschlag verwerfen';
  for (const control of document.querySelectorAll('.refit-adjustments input, .refit-adjustments button')) control.disabled = pathRefitBusy;
}

function trackingGeometry(entry) {
  const proposed = pathRefitProposal?.byFrame.get(entry.frame)?.pose ?? pcbProposal?.byFrame.get(entry.frame);
  return frameGeometry(proposed ? { ...entry, pose: proposed } : entry, calibration?.field, calibration?.maps);
}

function mergePlan() {
  const geometries = trackingPath.map(trackingGeometry).filter(Boolean);
  const bounds = mergeBounds(geometries);
  return { geometries, bounds, estimate: mergeEstimate(bounds) };
}

function updateMergeControls() {
  updateGpuAdapterControl();
  const ready = Boolean(useWebGpu() && calibration && trackingPath.some(entry => entry.pose) &&
    trackingVideoCompatible(trackingDataset, videoInfo));
  element('mergeStart').disabled = !ready || taskBusy || mergeRunning;
  element('mergeCancel').disabled = !mergeRunning;
  text('mergeState', mergeRunning ? 'Laeuft' : ready ? 'Bereit' : useWebGpu() ? 'Nicht bereit' : 'WebGPU aus');
  element('mergeState').className = `tag ${ready ? 'good' : ''}`;
  if (workflow !== 'merge' || mergeRunning) return;
  if (mergeResultStatus) { text('mergeStatus', mergeResultStatus); return; }
  if (!useWebGpu()) { text('mergeStatus', 'FÃƒÂ¼r Merge ein WebGPU-GerÃƒÂ¤t auswÃƒÂ¤hlen.'); return; }
  const plan = mergePlan();
  if (!plan.bounds) {
    text('mergeMetrics', 'Keine Ausgabe geplant');
    text('mergeStatus', 'Tracking und zugehoeriges Video erforderlich.');
    return;
  }
  const key = `${calibration.maps.outputWidth}x${calibration.maps.outputHeight}`;
  const tileSize = mergeTilePlan?.key === key ? mergeTilePlan.tileSize : 2048;
  const estimate = mergeEstimate(plan.bounds, tileSize);
  const files = planTiffParts({ ...plan.bounds, tileSize, tiles: [] }, 'merge.tif', element('mergeSplit').value);
  text('mergeMetrics', `${plan.bounds.width} x ${plan.bounds.height} px | ${plan.geometries.length} Frames | ` +
    `${trackingMaskHasSelection ? 'Rohmaske aktiv' : 'gesamter Bildbereich'}`);
  text('mergeStatus', `${files.length} TIFF-Datei(en) | ${estimate.tiles} Kacheln bei ${tileSize} px | BigTIFF verlustfrei komprimiert | unkomprimiert bis ca. ${megabytes(estimate.tiles * tileSize * tileSize * 4)}`);
}

async function updateMergeTilePlan() {
  if (!calibration?.maps || !useWebGpu()) return;
  const adapter = await requestSelectedGpuAdapter();
  if (!adapter) return;
  mergeTilePlan = { key: `${calibration.maps.outputWidth}x${calibration.maps.outputHeight}`,
    tileSize: Math.min(2048, overlayTileSize(calibration.maps, adapter.limits.maxTextureDimension2D)) };
  updateMergeControls();
}

function prepareMergePreview(bounds) {
  const scale = Math.min(1, 4096 / Math.max(bounds.width, bounds.height));
  mergePreviewImage.width = Math.max(1, Math.ceil(bounds.width * scale));
  mergePreviewImage.height = Math.max(1, Math.ceil(bounds.height * scale));
  const context = mergePreviewImage.getContext('2d');
  context.clearRect(0, 0, mergePreviewImage.width, mergePreviewImage.height);
  context.imageSmoothingEnabled = true;
  mergeZoom = 1; mergePan = { x: 0, y: 0 };
  element('mergeEmpty').hidden = true;
  renderMergePreview();
  return { context, scale };
}

function renderMergePreview() {
  const { context, width, height } = trackingCanvasContext('mergeCanvas');
  context.clearRect(0, 0, width, height);
  if (!mergePreviewImage.width || !mergePreviewImage.height) return;
  const scale = Math.min(width / mergePreviewImage.width, height / mergePreviewImage.height) * mergeZoom;
  const left = (width - mergePreviewImage.width * scale) / 2 + mergePan.x;
  const top = (height - mergePreviewImage.height * scale) / 2 + mergePan.y;
  context.imageSmoothingEnabled = mergeZoom < 2;
  context.drawImage(mergePreviewImage, left, top, mergePreviewImage.width * scale, mergePreviewImage.height * scale);
  text('mergeZoomValue', `${Math.round(mergeZoom * 100)}%`);
}

function installMergePreviewInteraction() {
  const canvas = element('mergeCanvas');
  let drag = null;
  canvas.addEventListener('pointerdown', event => {
    if (event.button !== 0 || !mergePreviewImage.width) return;
    drag = { x: event.clientX, y: event.clientY, pan: { ...mergePan } };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    if (!drag) return;
    mergePan = { x: drag.pan.x + event.clientX - drag.x, y: drag.pan.y + event.clientY - drag.y };
    renderMergePreview();
  });
  for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(name, () => { drag = null; });
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    const bounds = canvas.getBoundingClientRect();
    const next = Math.max(0.25, Math.min(64, mergeZoom * Math.exp(-event.deltaY * 0.001)));
    const factor = next / mergeZoom;
    const x = event.clientX - bounds.left - bounds.width / 2, y = event.clientY - bounds.top - bounds.height / 2;
    mergePan = { x: x - (x - mergePan.x) * factor, y: y - (y - mergePan.y) * factor };
    mergeZoom = next; renderMergePreview();
  }, { passive: false });
  element('mergeZoomIn').onclick = () => { mergeZoom = Math.min(64, mergeZoom * 1.25); renderMergePreview(); };
  element('mergeZoomOut').onclick = () => { mergeZoom = Math.max(0.25, mergeZoom / 1.25); renderMergePreview(); };
  element('mergeFitView').onclick = () => { mergeZoom = 1; mergePan = { x: 0, y: 0 }; renderMergePreview(); };
  new ResizeObserver(renderMergePreview).observe(canvas.parentElement);
}

async function renderMergeToBigTiff(createHandle, onPart = () => {}) {
  const maxFrames = number('mergeMaxFrames', 3);
  const edgeFeather = number('mergeEdgeFeather', 10) / 100;
  if (!Number.isInteger(maxFrames) || maxFrames < 1 || maxFrames > 64) throw new Error('Scharfe Frames je Pixel muss zwischen 1 und 64 liegen.');
  if (!Number.isFinite(edgeFeather) || edgeFeather < 0 || edgeFeather > 0.5) throw new Error('Randueberblendung muss zwischen 0 und 50 Prozent liegen.');
  const { geometries, bounds } = mergePlan();
  if (!bounds || !geometries.length) throw new Error('Keine gueltigen Tracking-Posen fuer den Merge.');
  const selectedGeometries = selectMergeFrames(geometries, maxFrames, trackingPixelAllowed, edgeFeather);
  if (!selectedGeometries.length) throw new Error('Die Trackingmaske enthaelt keine verwendbaren Merge-Bereiche.');
  const adapter = await requestSelectedGpuAdapter();
  if (!adapter) throw new Error('Merge benoetigt WebGPU.');
  const tileSize = Math.min(2048, overlayTileSize(calibration.maps, adapter.limits.maxTextureDimension2D));
  mergeTilePlan = { key: `${calibration.maps.outputWidth}x${calibration.maps.outputHeight}`, tileSize };
  const tiles = overlayTiles(bounds.width, bounds.height, bounds.minX, bounds.minY, selectedGeometries, tileSize,
    { maxFrames, pixelAllowed: trackingPixelAllowed, edgeFeather });
  const name = `${(videoInfo?.name || 'mosaic').replace(/\.[^.]+$/, '')}-merge.tif`;
  const parts = planTiffParts({ ...bounds, tileSize, tiles }, name, element('mergeSplit').value);
  let writer;
  mergeResultStatus = null;
  mergeRunning = true; mergeCancelled = false; updateMergeControls();
  element('mergeSavePreview').disabled = true;
  const preview = prepareMergePreview(bounds);
  try {
    writer = await beginTiffParts(parts, typeof createHandle === 'function' ? createHandle : () => createHandle, onPart, state => {
      text('mergePreviewState', `TIFF ${state.part}/${state.parts} | Kachel ${state.completed}/${state.total} | ${megabytes(state.bytes)} komprimiert`);
    });
    trackingMosaicRequest++; clearTimeout(trackingMosaicTimer);
    releaseOverlayGpuSession(); await overlayGpuIdle;
    await frameReader.dispose();
    const result = await renderTiledOverlay({ maps: calibration.maps, ...bounds, geometries: selectedGeometries, pixelAllowed: trackingPixelAllowed,
      maxFrames, edgeFeather, brightness: brightnessCalibration, frameOrder: 'blurriest', blend: element('mergeBlend').value,
      tileSize, retainTiles: false, decode: index => readTrackingFrame(index, { output: 'native', cache: false, measureSharpness: false }),
      cancelled: () => mergeCancelled,
      progress: state => {
        const status = `${state.stage === 'coverage' ? 'Abdeckung' : 'Mischen'} | Kachel ${state.tileIndex + 1}/${state.tileCount} | Frame ${state.frameIndex + 1}/${state.frameCount} (#${state.frame}) | GPU-Durchlauf ${state.framePass}/${state.plannedFramePasses}`;
        text('mergeStatus', status);
        operationProgress(status, state.framePass / state.plannedFramePasses, 'merge');
      },
      tileReady: async tile => {
        await writer.writeTile(tile);
        preview.context.drawImage(tile.bitmap, 0, 0, tile.width, tile.height,
          tile.x * preview.scale, tile.y * preview.scale, tile.width * preview.scale, tile.height * preview.scale);
        renderMergePreview();
      } });
    if (!result || mergeCancelled) { await writer.abort(); mergeResultStatus = 'Merge abgebrochen. Fertige Teile bleiben verfuegbar; der unvollstaendige Teil wurde verworfen.'; return null; }
    text('mergePreviewState', 'Schliesse Bilddatei ab...');
    const files = await writer.finish();
    const totalBytes = files.reduce((sum, file) => sum + file.layout.fileBytes, 0);
    mergeResultStatus = `Fertig: ${bounds.width} x ${bounds.height} px | ${tiles.length} Kacheln | ${files.length} TIFF-Datei(en), ${megabytes(totalBytes)} gesamt | ` +
      `${selectedGeometries.length}/${geometries.length} Frames nach Pose-, Masken- und Schaerfeauswahl.`;
    text('mergePreviewState', 'BigTIFF vollstaendig erstellt');
    element('mergeSavePreview').disabled = false;
    const command = tiffJoinCommand(files, bounds.width, bounds.height);
    element('mergeJoin').hidden = !command; element('mergeJoinCommand').textContent = command;
    return { files, bounds };
  } catch (error) {
    try { await writer?.abort(); } catch {}
    mergeResultStatus = `Merge fehlgeschlagen: ${error.message}`;
    throw error;
  } finally { mergeRunning = false; updateMergeControls(); }
}

async function startMerge() {
  if (taskBusy || mergeRunning) return;
  const baseName = (videoInfo?.name || 'mosaic').replace(/\.[^.]+$/, '');
  let chosenHandle;
  if (element('mergeSaveMethod').value === 'picker' && element('mergeSplit').value === '0' && globalThis.showSaveFilePicker) {
    try { chosenHandle = await showSaveFilePicker({ suggestedName: `${baseName}-merge.tif`,
      types: [{ description: 'BigTIFF (verlustfrei komprimiert)', accept: { 'image/tiff': ['.tif', '.tiff'] } }] }); }
    catch (error) { if (error.name === 'AbortError') return;
      if (!['SecurityError', 'NotAllowedError'].includes(error.name)) throw error; }
  }
  await task(async () => {
    for (const download of mergeDownloads) { URL.revokeObjectURL(download.url); await download.handle.dispose?.(); }
    mergeDownloads = []; element('mergeDownloads').replaceChildren(); element('mergeJoin').hidden = true;
    const result = await renderMergeToBigTiff(part => chosenHandle ?? createBrowserImageFile(part.name), (part, index, total) => {
      if (!part.handle.dispose) return;
      const url = URL.createObjectURL(part.saved), link = document.createElement('a');
      mergeDownloads.push({ url, handle: part.handle });
      link.href = url; link.download = part.name; link.style.display = 'block';
      link.textContent = `${part.name} (${megabytes(part.saved.size)})`;
      element('mergeDownloads').append(link); link.click();
      text('mergePreviewState', `TIFF ${index + 1}/${total} fertig. Download gestartet; Links bleiben zum erneuten Speichern verfuegbar.`);
    });
    if (result) text('mergePreviewState', chosenHandle ? 'TIFF gespeichert und Dateigroesse geprueft' :
      `${result.files.length} TIFF-Datei(en) fertig. Falls ein Download fehlt, den Dateilink anklicken.`);
  });
}

async function saveMergePreview() {
  const blob = await new Promise(resolve => mergePreviewImage.toBlob(resolve, 'image/png'));
  if (!blob) throw new Error('PNG-Uebersicht konnte nicht erstellt werden.');
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = `${(videoInfo?.name || 'mosaic').replace(/\.[^.]+$/, '')}-merge-uebersicht.png`;
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 60000);
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
      (Number.isFinite(match.reverseDistance) ? (match.reverseDistance <= conditionalLimit ? 'PCB-Zellenpruefung' :
        `Zyklus > ${fixed(conditionalLimit, 1)} px`) : 'Zyklus ungueltig') : 'Rueckwaertssuche') :
      (match.forward?.reason || 'Vorwaertssuche');
    reasons.set(reason, (reasons.get(reason) ?? 0) + 1);
  }
  const breakdown = [...reasons].sort((first, second) => second[1] - first[1]).map(([reason, count]) => `${reason}: ${count}`).join(', ');
  return `${matches.length} Paare versucht; vorwaerts ${forward}, rueckwaerts ${backward}, innerhalb Zyklusgrenze ${cycle}. ${breakdown}`;
}

function drawBrightnessCalibration() {
  const canvas = element('brightnessCanvas');
  const empty = element('brightnessEmpty');
  if (!brightnessCalibration) { canvas.width = canvas.height = 0; empty.hidden = false; return; }
  const scale = Math.max(1, Math.ceil(Math.max(brightnessCalibration.width, brightnessCalibration.height) / 640));
  canvas.width = Math.ceil(brightnessCalibration.width / scale); canvas.height = Math.ceil(brightnessCalibration.height / scale);
  const range = brightnessDisplayRange(brightnessCalibration, trackingPixelAllowed);
  const image = new ImageData(brightnessFieldPixels(brightnessCalibration, canvas.width, canvas.height, trackingPixelAllowed, range),
    canvas.width, canvas.height);
  canvas.getContext('2d').putImageData(image, 0, 0); empty.hidden = true;
  text('brightnessLegend', `Schwarz: ${Math.round((range.low - 1) * 100)}% | Grau: 0% | Weiss: +${Math.round((range.high - 1) * 100)}%`);
  text('brightnessState', `v${brightnessCalibration.version}`); element('brightnessState').className = 'tag good';
  const metrics = brightnessCalibration.metrics;
  const frameRange = Number.isSafeInteger(metrics.firstFrame) && Number.isSafeInteger(metrics.lastFrame) ?
    `${metrics.frames} Frames #${metrics.firstFrame} bis #${metrics.lastFrame} | ` : '';
  text('brightnessMetrics', `${metrics.accelerator ?? 'CPU'} | ` + frameRange + `${metrics.trainingFrames} Training / ${metrics.validationFrames} Validierung | ` +
    `${metrics.equations} weisse Bloecke | Validierung ${fixed(metrics.baselineValidationRms, 3)} -> ${fixed(metrics.validationRms, 3)} log | ` +
    `${(metrics.covered * 100).toFixed(0)}% gestuetzt` +
    (brightnessCalibration.model?.completion ? ' | Rest geglaettet ergaenzt' : ''));
}

async function discardBrightnessSession() {
  brightnessSession = null;
  brightnessRunning = false;
  brightnessPaused = false;
}

async function fitBrightnessCalibration(restart = false) {
  const from = element('brightnessFrom'), to = element('brightnessTo');
  if (!from.validity.valid || !to.validity.valid) throw new Error('Bitte gueltige ganze Frame-Nummern ab 0 eingeben.');
  if (!element('brightnessPasses').validity.valid || !element('brightnessMaxGain').validity.valid) {
    throw new Error('Bitte eine gueltige Zielzahl an Frames und maximale Verstaerkung eingeben.');
  }
  const maximumFrame = to.value === '' ? Infinity : to.valueAsNumber;
  const available = [...frames.values()].filter(frame => frame.accepted && frame.enabled && !frame.patchSize &&
    frame.points.length >= 4 && frame.id >= from.valueAsNumber && frame.id <= maximumFrame).sort((first, second) => first.id - second.id);
  if (available.length < 5) throw new Error(`Im gewaehlten Bereich sind nur ${available.length} akzeptierte Checkerboard-Frames vorhanden. Mindestens fuenf werden benoetigt.`);
  if (restart) await discardBrightnessSession();
  if (brightnessSession && brightnessSession.backend !== 'checkerboard-white') await discardBrightnessSession();
  const target = Math.min(available.length, element('brightnessPasses').valueAsNumber);
  const selected = Array.from({ length: target }, (_, index) => available[Math.round(index * (available.length - 1) / (target - 1))]);
  if (!brightnessSession) brightnessSession = { backend: 'checkerboard-white', width: calibration.maps.outputWidth,
    height: calibration.maps.outputHeight, maxGain: element('brightnessMaxGain').valueAsNumber,
    observations: new Map(), samples: new Map(), timings: { readMs: 0, sampleMs: 0, totalMs: 0 } };
  const session = brightnessSession;
  session.observations = new Map(selected.map(frame => [frame.id, frame]));
  for (const frame of session.samples.keys()) if (!session.observations.has(frame)) session.samples.delete(frame);
  const runStarted = performance.now();
  brightnessRunning = true;
  brightnessPaused = false;
  updateControls();
  try {
    for (let index = 0; index < selected.length; index++) {
      const observation = selected[index];
      if (session.samples.has(observation.id)) continue;
      text('brightnessStatus', `Weisse Checkerboard-Flaechen extrahieren | Frame #${observation.id} | ${index + 1}/${selected.length}`);
      operationProgress('Checkerboard-Helligkeitsflaechen', index / selected.length, 'brightness-checkerboard');
      let started = performance.now();
      const image = await readTrackingFrame(observation.id, { gpu: false, rectified: true, output: 'rgba', brightness: false });
      session.timings.readMs += performance.now() - started;
      started = performance.now();
      const mappedCells = checkerboardCells(observation.points).map(cell => {
        const corners = cell.corners.map(point => {
          const x = Math.max(0, Math.min(calibration.field.width - 1, point.x));
          const y = Math.max(0, Math.min(calibration.field.height - 1, point.y));
          const left = Math.min(calibration.field.width - 2, Math.floor(x)), top = Math.min(calibration.field.height - 2, Math.floor(y));
          const across = x - left, down = y - top, width = calibration.field.width, forward = calibration.maps.forward;
          const component = axis => (1 - down) * ((1 - across) * forward[(top * width + left) * 2 + axis] +
            across * forward[(top * width + left + 1) * 2 + axis]) + down * ((1 - across) * forward[((top + 1) * width + left) * 2 + axis] +
            across * forward[((top + 1) * width + left + 1) * 2 + axis]);
          return { x: component(0) - calibration.maps.origin[0], y: component(1) - calibration.maps.origin[1] };
        });
        return { col: cell.col, row: cell.row, corners,
          x: corners.reduce((sum, point) => sum + point.x, 0) / 4,
          y: corners.reduce((sum, point) => sum + point.y, 0) / 4 };
      });
      session.samples.set(observation.id, sampleCheckerboardBrightness(image, mappedCells, observation.id, 8, trackingPixelAllowed));
      session.timings.sampleMs += performance.now() - started;
      if (!brightnessRunning) {
        brightnessPaused = true;
        text('brightnessStatus', `Pausiert | ${session.samples.size}/${selected.length} Checkerboard-Frames analysiert.`);
        return;
      }
    }
    const sampledFrames = selected.map(observation => session.samples.get(observation.id));
    const heldOut = new Set(sampledFrames.map((_, index) => index % 5 === 4 ? index : -1).filter(index => index >= 0));
    text('brightnessStatus', `${sampledFrames.length} Checkerboard-Frames | direkte WeiÃƒÅ¸referenz | lokales Modell`);
    const result = fitCheckerboardBrightness(sampledFrames, heldOut, { outputWidth: session.width, outputHeight: session.height,
      maxGain: session.maxGain, allowed: trackingPixelAllowed });
    const frameNumbers = selected.map(frame => frame.id);
    result.metrics.frames = sampledFrames.length; result.metrics.firstFrame = Math.min(...frameNumbers); result.metrics.lastFrame = Math.max(...frameNumbers);
    result.metrics.acceptedBlocks = sampledFrames.reduce((sum, frame) => sum + frame.uniform.reduce((count, value) => count + Number(Boolean(value)), 0), 0);
    result.metrics.acceptedCells = sampledFrames.reduce((sum, frame) => sum + frame.cells, 0);
    const totalMs = session.timings.totalMs + performance.now() - runStarted;
    result.metrics.timings = { ...session.timings, totalMs };
    brightnessCalibration = result;
    const workerCalibration = { ...result, gain: result.gain.slice(), supported: result.supported.slice() };
    await computer.call('brightness-set', { calibration: workerCalibration }, [workerCalibration.gain.buffer, workerCalibration.supported.buffer]);
    await frameReader.dispose(); trackingComputer.nativeMaps = null;
    scheduleTrackingMosaic();
    text('brightnessStatus', `Weisses Checkerboard-Papier: ${result.metrics.acceptedCells.toLocaleString('de-DE')} Zellen / ` +
      `${result.metrics.acceptedBlocks.toLocaleString('de-DE')} Bloecke | ${sampledFrames.length} Frames | ` +
      `Validierung ${fixed(result.metrics.baselineValidationRms, 3)} -> ${fixed(result.metrics.validationRms, 3)} log | ` +
      `Lesen/Entzerren ${fixed(session.timings.readMs / 1000, 1)} s, Analyse und Fit ${fixed((totalMs - session.timings.readMs) / 1000, 1)} s.`);
    operationProgress('Helligkeitskarte aus weissen Checkerboard-Flaechen berechnet', 1, 'brightness-checkerboard');
    drawBrightnessCalibration();
  } finally {
    session.timings.totalMs += performance.now() - runStarted;
    brightnessRunning = false;
    brightnessPaused = Boolean(brightnessSession);
    updateControls();
  }
}

async function resetBrightnessCalibration() {
  await discardBrightnessSession();
  await disableBrightnessCalibration();
  text('brightnessStatus', 'Helligkeitssitzung zurueckgesetzt.');
}

async function disableBrightnessCalibration() {
  brightnessCalibration = null;
  await computer.call('brightness-set', { calibration: null });
  await frameReader.dispose(); trackingComputer.nativeMaps = null;
  scheduleTrackingMosaic();
  text('brightnessStatus', 'Helligkeitskorrektur deaktiviert.');
  text('brightnessState', 'Kein Feld'); element('brightnessState').className = 'tag';
  text('brightnessMetrics', 'Keine Messung'); drawBrightnessCalibration(); updateControls();
}

async function brightnessGeometryIdentity() {
  if (!calibration) throw new Error('Keine geometrische Kalibrierung geladen.');
  const description = JSON.stringify({ version: calibration.version, sourceWidth: calibration.field.width,
    sourceHeight: calibration.field.height, spacing: calibration.field.spacing, nx: calibration.field.nx,
    ny: calibration.field.ny, coefficients: Array.from(calibration.field.coefficients),
    outputWidth: calibration.maps.outputWidth, outputHeight: calibration.maps.outputHeight,
    origin: calibration.maps.origin, opticalConfiguration: element('opticalConfiguration').value.trim() });
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(description));
  return Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('');
}

async function downloadBrightnessCalibration() {
  if (!brightnessCalibration) return;
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([encodeBrightnessCalibration(brightnessCalibration,
    await brightnessGeometryIdentity())], { type: 'application/octet-stream' }));
  link.download = `${(videoInfo?.name || 'brightness').replace(/\.[^.]+$/, '')}-brightness.rbright`; link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 0);
}

async function loadBrightnessCalibration(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const loaded = decodeBrightnessCalibration(bytes, { width: calibration.maps.outputWidth,
    height: calibration.maps.outputHeight, geometryIdentity: await brightnessGeometryIdentity() });
  await discardBrightnessSession();
  brightnessCalibration = loaded;
  const workerCalibration = { ...brightnessCalibration, gain: brightnessCalibration.gain.slice(), supported: brightnessCalibration.supported.slice() };
  await computer.call('brightness-set', { calibration: workerCalibration }, [workerCalibration.gain.buffer, workerCalibration.supported.buffer]);
  await frameReader.dispose(); trackingComputer.nativeMaps = null; drawBrightnessCalibration();
  scheduleTrackingMosaic();
  text('brightnessStatus', 'Geladenes Helligkeitsfeld ist aktiv.'); updateControls();
}

async function registerLocalPcbPairs(worker, images, pairs, limits, preprocessing, activity, parameters, featureRecovery = false) {
  const byFrame = new Map(images.map(image => [image.frame, image]));
  const matches = [];
  for (const [index, pair] of pairs.entries()) {
    if (pathRefitCancelRequested) throw new Error('Lokaler Refit abgebrochen.');
    activity(`Pruefe PCB-Zellen ${index + 1}/${pairs.length}: #${pair.current} gegen #${pair.reference}`);
    const selected = [byFrame.get(pair.reference), byFrame.get(pair.current)];
    const diameter = Math.min(...selected.map(image => Math.min(image.bitmap.width, image.bitmap.height))) *
      (selected.some(image => image.mask) ? preprocessing.region : 1);
    const cellSize = Math.min(parameters.cellSize, 2 ** Math.floor(Math.log2(diameter / 2)));
    const bitmaps = await Promise.all(selected.map(image => createImageBitmap(image.bitmap)));
    try {
      const match = await worker.call('pcb-pair-register', { images: selected.map((image, item) => ({
        frame: image.frame, pose: image.pose, offset: image.offset, mask: image.mask, bitmap: bitmaps[item] })),
      pair:{...pair,cells:currentMatchNetwork().pairs.find(p=>p.reference===pair.reference&&p.current===pair.current)?.cells??[]}, preprocessing, useWebGpu: false, featureRecovery, localLandmarks: true, coarseRadius: parameters.coarseRadius,
      minimumOverlapFraction: parameters.coarseOverlap,
      fft: { cellSize: Math.max(16, cellSize), cellsPerAxis: parameters.cellsPerAxis,
        searchRadius: Math.min(limits.radius, Math.max(16, cellSize) / 2 - 1),
        minimumPsr: parameters.minimumPsr, residualLimit: parameters.residualLimit,
        adaptiveCells: true },
      limits: { radius: limits.radius, angle: limits.angle, cycleLimit: parameters.cycleLimit,
        fftCycleFactor: parameters.fftCycleFactor, minimumScore: parameters.minimumScore,
        minimumSupport: parameters.minimumSupport } }, bitmaps);
      storeNetworkMatches([match], parameters);
      const verified = acceptedPcbConstraints([match], parameters).accepted.length > 0;
      matches.push({ ...match, pcbVerified: verified,
        forward: verified ? match.forward : { ...match.forward, accepted: false,
          reason: match.forward?.reason ?? 'PCB-Paarpruefung' } });
    } finally { for (const bitmap of bitmaps) bitmap.close(); }
  }
  return matches;
}

async function refitTrackingPath() {
  if (!pathSelection || pathRefitBusy || trackingRunning || taskBusy) return;
  const geometries = trackingPath.map(entry => frameGeometry(entry, calibration?.field, calibration?.maps)).filter(Boolean);
  const overlayFrames = new Set(overlayContributors.map(geometry => geometry.entry.frame));
  const localEntries = geometries.filter(geometry => overlayFrames.has(geometry.entry.frame) ||
    geometry.supports(pathSelection, trackingPixelAllowed))
    .map(geometry => geometry.entry);
  const seeds = localEntries.map(entry => entry.frame);
  if (!seeds.length) { text('trackingPathRefitInfo', 'Am Auswahlpunkt liegen keine Frames.'); return; }
  const preprocessing = refitPreprocessing();
  const baseMask = (trackingRunOptions ?? trackingDataset?.options)?.imageMask ?? null;
  const selectedFrame = element('trackingOverlayFrame').value;
  if (localEntries.length < 2) { text('trackingPathRefitInfo', 'Die Auswahl enthaelt zu wenige auswertbare Frames.'); return; }
  pathRefitBusy = true; pathRefitProposal = null; updatePathRefitControls();
  pathRefitCancelRequested = false;
  let searchProgress = { attempted: 0, limit: 0 };
  const refitStarted = performance.now();
  let refitActivity = 'Lokaler Refit wird vorbereitet';
  const showRefitActivity = activity => {
    refitActivity = activity;
    const elapsed = Math.floor((performance.now() - refitStarted) / 1000);
    text('trackingPathRefitInfo', `${refitActivity} | aktiv seit ${elapsed} s`);
    text('processingStatus', `${refitActivity} | ${elapsed} s`);
  };
  const refitHeartbeat = setInterval(() => showRefitActivity(refitActivity), 1000);
  const worker = new WorkerClient('./compute-worker.js', status => {
    if (status.operation !== 'local-refit') return;
    const scope = 'Lokaler Refit';
    const pair = `${searchProgress.attempted + (status.pair || 0)}/${searchProgress.limit}`;
    const activity = status.stage === 'prepare' ? `${scope}: bereite Matcherbild ${status.image}/${status.images} vor${Number.isFinite(status.frame) ? ` (#${status.frame})` : ''}` :
      status.stage === 'forward' ? `${scope} ${pair}: Vorwaertssuche ${status.attempt}/${status.attempts}, Radius ${status.radius} px` :
      status.stage === 'feature-forward' ? `${scope} ${pair}: Feature-Fallback vorwaerts` :
      status.stage === 'backward' ? `${scope} ${pair}: Rueckpruefung, Radius ${status.radius} px` :
      status.stage === 'feature-backward' ? `${scope} ${pair}: Feature-Fallback rueckwaerts` :
      `${scope} ${pair}: #${status.current} gegen #${status.reference}`;
    showRefitActivity(activity);
    const fraction = status.stage === 'prepare' ? 0.05 * status.image / status.images :
      ((status.pair || 1) - 1 + (status.stage === 'backward' || status.stage === 'feature-backward' ? 0.8 : 0.3)) / Math.max(1, status.pairs || 1);
    element('progress').value = Math.min(0.99, fraction);
  });
  pathRefitWorker = worker;
  try {
    overlayRequest++; trackingMosaicRequest++; clearTimeout(trackingMosaicTimer);
    showRefitActivity('Lokaler Refit wartet auf laufende Bildabrufe');
    await frameReader.waitUntilIdle();
    const pcb = pcbParameters();
    const limits = { radius: pcb.radius, angle: pcb.angle };
    const pairs = planFocusedRefitPairs(localEntries, selectedFrame ? [Number(selectedFrame)] : []);
    const entries = new Map(localEntries.map(entry => [entry.frame, entry]));
    const matches = [];
    const registerPairs = async (planned, featureRecovery = false) => {
      const offset = matches.length;
      searchProgress.limit = offset + planned.length;
      for (let start = 0; start < planned.length; start += 2) {
        if (pathRefitCancelRequested) throw new Error('Lokaler Refit abgebrochen.');
        const batch = planned.slice(start, start + 2);
        const requiredFrames = [...new Set(batch.flatMap(pair => [pair.reference, pair.current]))];
        const images = [];
        try {
          for (const frame of requiredFrames) {
            showRefitActivity(`Lade Frame #${frame} | Paar ${offset + start + 1}/${searchProgress.limit}`);
            const decoded = await readTrackingFrame(frame, { rectified: true, sourceMask: pcbSourceMask() });
            const entry = entries.get(frame);
            const geometry = frameGeometry(entry, calibration?.field, calibration?.maps);
            images.push({ frame, pose: { ...entry.pose }, bitmap: decoded.bitmap,
              offset: entry.mode === 'window' ? [-decoded.bitmap.width / 2, -decoded.bitmap.height / 2] :
                [calibration.maps.origin[0] - calibration.field.width / 2,
                  calibration.maps.origin[1] - calibration.field.height / 2],
              mask: localSelectionMask(geometry, pathSelection, preprocessing.region, baseMask) });
          }
          searchProgress.attempted = offset + start;
          matches.push(...await registerLocalPcbPairs(worker, images, batch, limits, preprocessing, showRefitActivity, pcb, featureRecovery));
        } finally { for (const image of images) image.bitmap.close(); }
      }
    };
    const baseGraph = buildPoseGraph(localEntries, { conditionalLimit: pcb.cycleLimit });
    const graphForMatches = () => applyNetworkToGraph(addLocalRefitEdges(baseGraph, matches,
      { cycleLimit: pcb.cycleLimit, acceptVerifiedPairs: true }), currentMatchNetwork());
    await registerPairs(pairs);
    let graph = graphForMatches();
    let refitPasses = 1;
    if (pcb.localBudget) {
      let measuredFrames = measuredRefitComponent(graph);
      let remaining = pcb.localBudget;
      while (remaining && measuredFrames.length < localEntries.length) {
        const bridges = planRefitBridgePairs(localEntries, measuredFrames, matches, Math.min(32, remaining));
        if (!bridges.length) break;
        showRefitActivity(`Pruefe ${bridges.length} weitere Paare fuer unverbundene Frames`);
        await registerPairs(bridges, true);
        remaining -= bridges.length;
        graph = graphForMatches();
        measuredFrames = measuredRefitComponent(graph);
        refitPasses++;
      }
    }
    if (!graph.localEdges) throw new Error(`Kein bestaetigter lokaler Match (${matches.length} Paare). ${localRefitFailure(matches, pcb.cycleLimit)}`);
    showRefitActivity(`${graph.localEdges} bestaetigte lokale Kanten; optimiere Posen`);
    const result = await worker.call('pose-graph-refit', { graph,
      options: { seedFrames: seeds, iterations: pcb.iterations, huber: pcb.huber, includePassive: true } });
    pathRefitProposal = { ...result, measuredFrames: measuredRefitComponent(graph).length, networkRevision:currentMatchNetwork().revision, byFrame: new Map(result.corrections.map(item => [item.frame, item])),
      overlayFrames: localEntries.map(entry => entry.frame),
      preprocessing, refitPasses,
      refitMatches: matches.length, groupMatches: 0, groupAttempts: 0, overlayReady: false };
    text('trackingPathRefitInfo', `${matches.length} Paarpruefungen | ${graph.localEdges} bestaetigte Netzverbindungen | ${pathRefitProposal.byFrame.size} korrigierbare Frames (${result.nodes} im Netz), ${result.edges} Kanten | Lokal RMS ${fixed(result.localBeforeRms, 2)} -> ${fixed(result.localAfterRms, 2)} px | Rot gestrichelt: Vorschlag.`);
    drawTrackingPath();
    await loadPathOverlay(pathSelection);
    if (!pathOverlay.width) throw new Error('Das Mischbild der Refit-Vorschau konnte nicht aufgebaut werden.');
    pathRefitProposal.overlayReady = true;
    const bridgeWarning = pathRefitProposal.measuredFrames < localEntries.length ? " | Achtung: Noch getrennte Bildmatch-Gruppen; deren gegenseitiger Offset ist nicht bestaetigt." : "";
    text('trackingPathRefitInfo', `${matches.length} Paarpruefungen | ${graph.localEdges} bestaetigte Netzverbindungen | ${pathRefitProposal.measuredFrames}/${localEntries.length} Frames durch Bildmatches verbunden, ${result.passiveFrames.length} passiv interpoliert | Lokal RMS ${fixed(result.localBeforeRms, 2)} -> ${fixed(result.localAfterRms, 2)} px | Mischbild zeigt alle ${pathRefitProposal.overlayFrames.length} Frames des Vorschlags.${bridgeWarning}`);
  } catch (error) {
    text('trackingPathRefitInfo', pathRefitCancelRequested ? 'Lokaler Refit abgebrochen.' : `Refit fehlgeschlagen: ${error.message}`);
  }
  finally {
    clearInterval(refitHeartbeat);
    worker.terminate();
    if (pathRefitWorker === worker) pathRefitWorker = null;
    const cancelled = pathRefitCancelRequested;
    pathRefitCancelRequested = false; pathRefitBusy = false; updatePathRefitControls();
    text('processingStatus', 'Bereit');
    if (cancelled && pathSelection) void loadPathOverlay(pathSelection).finally(scheduleTrackingMosaic);
  }
}

async function discardTrackingPathRefit() {
  if (pathRefitBusy) {
    pathRefitCancelRequested = true;
    text('trackingPathRefitInfo', 'Lokaler Refit wird abgebrochen...');
    pathRefitWorker?.terminate();
    return;
  }
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
  const summary = { created_at: new Date().toISOString(), nodes: pathRefitProposal.nodes,
    correctedFrames: pathRefitProposal.byFrame.size, edges: pathRefitProposal.edges,
    passiveFrames: pathRefitProposal.passiveFrames,
    spatialEdges: pathRefitProposal.spatialEdges, localEdges: pathRefitProposal.localEdges,
    refitPasses: pathRefitProposal.refitPasses, refitMatches: pathRefitProposal.refitMatches,
    groupMatches: pathRefitProposal.groupMatches, groupAttempts: pathRefitProposal.groupAttempts,
    localBeforeRms: pathRefitProposal.localBeforeRms, localAfterRms: pathRefitProposal.localAfterRms,
    beforeRms: pathRefitProposal.beforeRms, afterRms: pathRefitProposal.afterRms,
    preprocessing: pathRefitProposal.preprocessing };
  if (trackingDataset) (trackingDataset.refits ??= []).push(summary);
  pathRefitProposal = null; clearPathOverlay(true);
  pcbProposal = null; updatePcbControls();
  frameReductionPreview = null;
  element('frameReductionApply').disabled = true;
  text('trackingPathRefitInfo', `Refit uebernommen | RMS ${fixed(summary.beforeRms, 2)} -> ${fixed(summary.afterRms, 2)} px.`);
  updatePathRefitControls(); renderTrackingResults(trackingPath.at(-1));
  scheduleTrackingMosaic();
  if (pathSelection) void loadPathOverlay(pathSelection);
}

function scheduleOverlayLiveRender() {
  overlayLiveRequest++;
  if (overlayLiveRunning || !overlayGpuSession || !overlayPoseDraft || !overlayBounds) return;
  overlayLiveRunning = true;
  overlayGpuInFlight++; updateGpuAdapterControl();
  requestAnimationFrame(async () => {
    const previous = overlayGpuIdle;
    let release;
    overlayGpuIdle = new Promise(resolve => { release = resolve; });
    await previous;
    const session = overlayGpuSession, request = overlayRequest, revision = overlayLiveRequest;
    try {
      if (!session || !overlayPoseDraft || !overlayBounds) return;
      const renderer = session.renderer, bounds = { ...overlayBounds };
      const scale = Math.max(1, Math.max(bounds.width, bounds.height) / 1024);
      const geometries = overlayContributors.map(g => g.entry.frame === overlayPoseDraft.frame ?
        frameGeometry({ ...g.entry, pose: { ...overlayPoseDraft.pose } }, calibration.field, calibration.maps) : g);
      if (geometries.some(g => !renderer.frameTextures.get(g.entry.frame)?.preview)) return;
      const started = performance.now();
      await renderer.resizeOutput(Math.ceil(bounds.width / scale), Math.ceil(bounds.height / scale), scale);
      await renderer.clear(bounds.minX, bounds.minY);
      for (const geometry of geometries) await renderer.addCachedFrame(geometry, 0, true);
      const bitmap = await renderer.finish();
      if (request !== overlayRequest || revision !== overlayLiveRequest || session !== overlayGpuSession) { bitmap.close(); return; }
      renderer.lastLiveMs = performance.now() - started;
      overlayLiveBitmap?.close(); overlayLiveBitmap = bitmap;
      renderPathOverlay();
      text('trackingOverlayInfo', `GPU-Livevorschau | ${geometries.length} Frames aus GPU-Texturen | ${fixed(renderer.lastLiveMs, 1)} ms | Uebernehmen rendert Vollaufloesung`);
    } catch (error) {
      if (request === overlayRequest) text('trackingOverlayInfo', `GPU-Livevorschau fehlgeschlagen: ${error.message}`);
    } finally {
      release(); overlayLiveRunning = false;
      overlayGpuInFlight--; updateGpuAdapterControl();
      if (revision !== overlayLiveRequest && overlayPoseDraft && overlayGpuSession) scheduleOverlayLiveRender();
    }
  });
}

function renderPathOverlay() {
  if (trackingTab !== 'overlay') return;
  const { context, width, height } = trackingCanvasContext('trackingOverlayCanvas');
  context.clearRect(0, 0, width, height);
  if (!pathOverlay.width) return;
  const scale = Math.min(width / pathOverlay.width, height / pathOverlay.height) * overlayZoom;
  context.imageSmoothingEnabled = false;
  if (overlaySelectedImage && overlayBounds && !overlayPoseDraft) {
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
  if (overlayLiveBitmap && overlayPoseDraft) {
    drawRefitPreviewImage(context, overlayLiveBitmap, (width - pathOverlay.width * scale) / 2 + overlayPan.x,
      (height - pathOverlay.height * scale) / 2 + overlayPan.y, pathOverlay.width * scale, pathOverlay.height * scale);
  } else if (pathOverlay.tiles) {
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
  } else {
    drawRefitPreviewImage(context, pathOverlay, (width - pathOverlay.width * scale) / 2 + overlayPan.x,
      (height - pathOverlay.height * scale) / 2 + overlayPan.y, pathOverlay.width * scale, pathOverlay.height * scale);
  }
  if (!overlayPoseDraft || !overlaySelectedImage || !overlayBounds) return;
  const geometry = frameGeometry({ ...overlaySelectedImage.geometry.entry, pose: overlayPoseDraft.pose },
    calibration?.field, calibration?.maps);
  if (!geometry) return;
  const left = (width - pathOverlay.width * scale) / 2 + overlayPan.x;
  const top = (height - pathOverlay.height * scale) / 2 + overlayPan.y;
  const project = point => ({ x: left + (point.x - overlayBounds.minX) * scale,
    y: top + (point.y - overlayBounds.minY) * scale });
  const origin = project(geometry.world(0, 0));
  context.save();
  // Keep the editable frame above both the cached mosaic and GPU live preview.
  context.globalAlpha = 0.5;
  context.translate(origin.x, origin.y);
  context.transform(geometry.c * scale, geometry.s * scale, -geometry.s * scale, geometry.c * scale, 0, 0);
  drawRefitPreviewImage(context, overlaySelectedImage.bitmap, 0, 0);
  context.restore();
  context.strokeStyle = '#f5d647'; context.lineWidth = 2; context.setLineDash([6, 4]);
  context.beginPath(); geometry.corners.map(project).forEach((corner, index) => {
    if (index) context.lineTo(corner.x, corner.y);
    else context.moveTo(corner.x, corner.y);
  });
  context.closePath(); context.stroke(); context.setLineDash([]);
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
    const passive = pathRefitProposal?.passiveFrames?.includes(entry.frame) ? ' | passiv interpoliert' : '';
    selector.add(new Option(`#${entry.frame}${score} | ${matches} Matches${passive}`, String(entry.frame)));
  }
  selector.disabled = false;
}

async function selectOverlayContributor() {
  overlayLiveRequest++; overlayLiveBitmap?.close(); overlayLiveBitmap = null;
  const request = ++overlaySelectionRequest;
  overlaySelectedImage?.bitmap.close();
  overlaySelectedImage = null;
  overlayPoseDraft = null;
  updateOverlayPoseControls();
  const value = element('trackingOverlayFrame').value;
  const frame = value === '' ? null : Number(value);
  const geometry = overlayContributors.find(item => item.entry.frame === frame);
  element('trackingOverlayInspect').disabled = !geometry;
  if (!geometry) { renderPathOverlay(); return; }
  text('trackingOverlayInfo', `Lade Frame #${frame} einzeln...`);
  const decoded = await readTrackingFrame(frame, { rectified: true });
  if (request !== overlaySelectionRequest) { decoded.bitmap.close(); return; }
  overlaySelectedImage = { geometry, bitmap: decoded.bitmap };
  const pose = geometry.entry.pose ?? geometry.entry.raw;
  overlayPoseDraft = { frame, originalPose: { ...pose }, pose: { ...pose } };
  updateOverlayPoseControls();
  text('trackingOverlayInfo', `Frame #${frame} ueber Mischbild | Pose (${fixed(pose.x, 2)}, ${fixed(pose.y, 2)}, ${fixed(pose.rotation * 180 / Math.PI, 3)} Grad) | Shift+Mausrad: fein drehen; Alt+Mausrad: grob drehen`);
  renderPathOverlay();
}

async function loadPathOverlay(point, { forceFrame = null } = {}) {
  const request = ++overlayRequest;
  trackingMosaicRequest++; clearTimeout(trackingMosaicTimer);
  pathSelection = point;
  updatePathRefitControls();
  const maxFrames = number('trackingOverlayMaxFrames', 3);
  const edgeFeather = number('overlayEdgeFeather', 10) / 100;
  if (!Number.isInteger(maxFrames) || maxFrames < 1 || maxFrames > 64) {
    text('trackingOverlayInfo', 'Framegrenze muss zwischen 1 und 64 liegen.'); return;
  }
  const positionLabel = `${pathRefitProposal ? 'Refit-Vorschau | ' : ''}Position (${fixed(point.x, 1)}, ${fixed(point.y, 1)}) | Top ${maxFrames}/Bereich`;
  element('trackingOverlayEmpty').hidden = true;
  activateTrackingTab('overlay');
  clearPathOverlay(true);
  overlayZoom = 1; overlayPan = { x: 0, y: 0 };
  text('trackingOverlayInfo', `${positionLabel} | Suche passende Frames...`);
  renderPathOverlay(); drawTrackingPath();
  if (!calibration || !trackingVideoCompatible(trackingDataset, videoInfo)) {
    text('trackingOverlayInfo', 'Bitte das zum Tracking gehoerende Video und die Kalibrierung laden.'); return;
  }
  if (trackingRunning) {
    trackingRunning = false; trackingPaused = true; updateControls();
    text('trackingOverlayInfo', `${positionLabel} | Tracking wird pausiert, danach werden die Frames automatisch geladen...`);
    return;
  }
  if (taskBusy) { text('trackingOverlayInfo', 'Andere Verarbeitung zuerst abschliessen.'); return; }
  await new Promise(resolve => requestAnimationFrame(resolve));
  if (request !== overlayRequest) return;
  const previewFrames = pathRefitProposal?.overlayFrames ? new Set(pathRefitProposal.overlayFrames) : null;
  const candidates = sharpestFramesFirst(trackingPath.filter(item => !previewFrames || previewFrames.has(item.frame))
    .map(item => trackingGeometry(item)).filter(geometry => geometry &&
      (geometry.entry.frame === forceFrame || previewFrames || geometry.supports(point, trackingPixelAllowed))));
  if (!candidates.length) { text('trackingOverlayInfo', 'Keine Bilddaten an dieser Position.'); return; }
  const supporting = previewFrames ? candidates : approximateTopFrames(candidates, maxFrames, trackingPixelAllowed, point);
  const forced = candidates.find(item => item.entry.frame === forceFrame);
  if (forced && !supporting.includes(forced)) supporting.push(forced);
  const corners = supporting.flatMap(geometry => geometry.corners);
  const minX = Math.floor(Math.min(...corners.map(p => p.x))), minY = Math.floor(Math.min(...corners.map(p => p.y)));
  const width = Math.ceil(Math.max(...corners.map(p => p.x))) - minX;
  const height = Math.ceil(Math.max(...corners.map(p => p.y))) - minY;
  const bounds = { minX, minY, width, height };
  if (useWebGpu() && navigator.gpu) {
    const retryAllocation = async create => {
      try { return await create(); }
      catch (error) {
        if (!frameReader.releaseOnAllocationError(error)) throw error;
        return create();
      }
    };
    overlayGpuInFlight++; updateGpuAdapterControl();
    const previous = overlayGpuIdle;
    let release;
    overlayGpuIdle = new Promise(resolve => { release = resolve; });
    await previous; // Cancelled requests release large buffers before the next allocation.
    let overlay = null;
    const started = performance.now();
    try {
      const key = [calibration.maps, videoInfo, brightnessCalibration, trackingImageMask,
        trackingImageMask?.revision, trackingMaskHasSelection, edgeFeather, element('globalGpuAdapter')?.value];
      if (overlayGpuSession && key.some((value, i) => value !== overlayGpuSession.key[i])) {
        overlayGpuSession.renderer.destroy(); overlayGpuSession = null;
      }
      const tiledNeeded = await WebGpuOverlay.needsTiles(calibration.maps, width, height);
      const tileEdge = tiledNeeded ? overlayTileSize(calibration.maps, overlayGpuSession?.renderer.device.limits.maxTextureDimension2D || 8192) : null;
      if (!overlayGpuSession) {
        await frameReader.releaseRenderer();
        const renderer = await retryAllocation(() => WebGpuOverlay.create(calibration.maps,
          tiledNeeded ? Math.min(tileEdge, width) : width, tiledNeeded ? Math.min(tileEdge, height) : height,
          minX, minY, trackingPixelAllowed, maxFrames, edgeFeather, brightnessCalibration));
        renderer.cacheEnabled = true;
        overlayGpuSession = { key, renderer };
      }
      overlay = overlayGpuSession.renderer;
      overlay.setActiveFrames(supporting.map(g => g.entry.frame));
      overlay.maxFrames = maxFrames;
      if (request !== overlayRequest) return;
      text('trackingOverlayInfo', `${positionLabel} | Bereite Vollaufloesung ${width} x ${height} vor...`);
      if (tiledNeeded) {
        const tiled = await retryAllocation(() => renderTiledOverlay({ maps: calibration.maps, width, height, minX, minY,
          geometries: supporting, pixelAllowed: trackingPixelAllowed, maxFrames, edgeFeather, renderer: overlay, tileSize: tileEdge,
          brightness: brightnessCalibration,
          decode: index => readTrackingFrame(index, { output: 'native' }), cancelled: () => request !== overlayRequest,
          progress: p => text('trackingOverlayInfo', `${positionLabel} | Kachel ${p.tileIndex + 1}/${p.tileCount} bis ${p.tileSize} px | Kandidat ${p.frameIndex + 1}/${p.frameCount}: #${p.frame} | Durchlauf ${p.framePass}/${p.plannedFramePasses} | WebGPU, Vollaufloesung`) }));
        if (!tiled) return;
        if (request !== overlayRequest) { closeOverlayTiles(tiled); return; }
        pathOverlay = tiled;
        publishOverlayContributors(supporting, bounds);
        text('trackingOverlayInfo', `${positionLabel} | ${supporting.length} von ${candidates.length} Kandidaten | ${width} x ${height} px, Vollaufloesung | WebGPU | ${tiled.tiles.length} Kacheln bis ${tiled.tileSize} px | ${fixed(tiled.frameMs / Math.max(1, tiled.framePasses), 1)} ms/Frame-Durchlauf | ${tiled.framePasses} Durchlaeufe | Gesamt ${fixed((performance.now() - started) / 1000, 2)} s`);
        element('trackingOverlayInfo').textContent += ` | Frameabruf ${fixed(tiled.decodeMs / 1000, 2)} s | GPU inkl. Entzerrung/Warten ${fixed(tiled.gpuMs / 1000, 2)} s | Ausgabe ${fixed(tiled.finishMs / 1000, 2)} s | Vorbereitung ${fixed(tiled.setupMs, 0)} ms`;
        renderPathOverlay();
        return;
      }
      await overlay.resizeOutput(width, height);
      await overlay.clear(minX, minY);
      if (request !== overlayRequest) return;
      const setupMs = performance.now() - started, previousGpuHits = overlay.cacheHits;
      let frameMs = 0, decodeMs = 0, gpuMs = 0, cacheHits = 0;
      for (const [index, geometry] of supporting.entries()) {
        text('trackingOverlayInfo', `${positionLabel} | Lade ${index + 1}/${supporting.length}: Frame #${geometry.entry.frame} | WebGPU, Vollaufloesung`);
        const frameStarted = performance.now();
        if (await overlay.addCachedFrame(geometry)) {
          cacheHits++; gpuMs += performance.now() - frameStarted;
          frameMs += performance.now() - frameStarted; continue;
        }
        const decoded = await readTrackingFrame(geometry.entry.frame, { output: 'native' });
        decodeMs += performance.now() - frameStarted;
        if (decoded.frameTiming?.cacheHit) {
          cacheHits++;
          text('trackingOverlayInfo', `${positionLabel} | Cache ${index + 1}/${supporting.length}: Frame #${geometry.entry.frame} | WebGPU, Vollaufloesung`);
        }
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
      element('trackingOverlayInfo').textContent += ` | ${cacheHits}/${supporting.length} Frames aus Cache | Frameabruf ${fixed(decodeMs / 1000, 2)} s | GPU inkl. Entzerrung/Warten ${fixed(gpuMs / 1000, 2)} s`;
      element('trackingOverlayInfo').textContent += ` | ${overlay.cacheHits - previousGpuHits}/${supporting.length} direkt aus GPU-Texturen | GPU-Bildcache ${fixed(overlay.cacheBytes / 1048576, 0)} MiB`;
      publishOverlayContributors(supporting, bounds);
      renderPathOverlay();
    } catch (error) {
      overlayGpuSession?.renderer.destroy(); overlayGpuSession = null;
      frameReader.releaseOnAllocationError(error);
      if (request === overlayRequest) text('trackingOverlayInfo', `Ueberlagerung fehlgeschlagen: ${error.message}`);
    } finally {
      release(); overlayGpuInFlight--; updateGpuAdapterControl();
      if (request === overlayRequest) scheduleTrackingMosaic();
    }
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
  installMatchNetworkControls();
  const path = element('trackingPathCanvas');
  let pathDrag = null;
  let ignorePathClick = false;
  const hit = event => {
    if (!pathProject) return null;
    const bounds = path.getBoundingClientRect();
    return pathProject.invert({ x: event.clientX - bounds.left, y: event.clientY - bounds.top });
  };
  path.addEventListener('pointerdown', event => { if (event.button !== 0 || !pathProject || pathZoom === 1) return;
    ignorePathClick = false;
    pathDrag = { x: event.clientX, y: event.clientY, pan: { ...pathPan }, moved: false };
    path.setPointerCapture(event.pointerId);
  });
  path.addEventListener('pointermove', event => {
    if (pathDrag) {
      const dx = event.clientX - pathDrag.x, dy = event.clientY - pathDrag.y;
      if (Math.hypot(dx, dy) > 3) pathDrag.moved = true;
      if (pathDrag.moved) pathPan = { x: pathDrag.pan.x + dx, y: pathDrag.pan.y + dy };
    }
    pathHover = hit(event); drawTrackingPath();
  });
  for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) path.addEventListener(name, () => {
    if (pathDrag?.moved) {
      ignorePathClick = true;
      setTimeout(() => { ignorePathClick = false; }, 0);
    }
    pathDrag = null;
  });
  path.addEventListener('pointerleave', () => { pathHover = null; drawTrackingPath(); });
  path.addEventListener('click', event => {
    if (ignorePathClick) return;
    const bounds = path.getBoundingClientRect();
    const target = hitMatchNetwork({x:event.clientX-bounds.left,y:event.clientY-bounds.top});
    if (target) { networkSelectedPair = target.pairId; networkSelectedCell = target.cellId;
      networkViewKey = ''; drawTrackingPath(); return; }
    const point = hit(event); if (point) void loadPathOverlay(point);
  });
  path.addEventListener('wheel', event => {
    if (!pathProject) return;
    event.preventDefault();
    const bounds = path.getBoundingClientRect();
    const x = event.clientX - bounds.left - bounds.width / 2;
    const y = event.clientY - bounds.top - bounds.height / 2;
    const next = Math.max(1, Math.min(32, pathZoom * Math.exp(-event.deltaY * 0.001)));
    const factor = next / pathZoom;
    pathPan = next === 1 ? { x: 0, y: 0 } :
      { x: x - (x - pathPan.x) * factor, y: y - (y - pathPan.y) * factor };
    pathZoom = next;
    drawTrackingPath();
  }, { passive: false });
  const canvas = element('trackingOverlayCanvas');
  let drag = null;
  const overlayPoint = event => {
    if (!overlayBounds || !pathOverlay.width) return null;
    const bounds = canvas.getBoundingClientRect();
    const scale = Math.min(bounds.width / pathOverlay.width, bounds.height / pathOverlay.height) * overlayZoom;
    const x = (event.clientX - bounds.left - (bounds.width - pathOverlay.width * scale) / 2 - overlayPan.x) / scale;
    const y = (event.clientY - bounds.top - (bounds.height - pathOverlay.height * scale) / 2 - overlayPan.y) / scale;
    return x >= 0 && y >= 0 && x < pathOverlay.width && y < pathOverlay.height ?
      { x: overlayBounds.minX + x, y: overlayBounds.minY + y, scale } : null;
  };
  canvas.addEventListener('pointerdown', event => {
    if (event.button !== 0 || !pathOverlay.width) return;
    const point = overlayPoint(event);
    const selected = overlayPoseDraft && overlaySelectedImage &&
      frameGeometry({ ...overlaySelectedImage.geometry.entry, pose: overlayPoseDraft.pose },
        calibration?.field, calibration?.maps);
    drag = { x: event.clientX, y: event.clientY, pan: { ...overlayPan },
      pose: overlayPoseDraft ? { ...overlayPoseDraft.pose } : null,
      scale: point?.scale, frame: Boolean(point && selected?.supports(point, trackingPixelAllowed)), moved: false };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    if (!drag) {
      const point = overlayPoint(event);
      const selected = overlayPoseDraft && overlaySelectedImage &&
        frameGeometry({ ...overlaySelectedImage.geometry.entry, pose: overlayPoseDraft.pose },
          calibration?.field, calibration?.maps);
      canvas.style.cursor = point && selected?.supports(point, trackingPixelAllowed) ? 'move' : 'grab';
      return;
    }
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    if (Math.hypot(dx, dy) > 3) drag.moved = true;
    if (!drag.moved) return;
    if (drag.frame && overlayPoseDraft && drag.scale) {
      canvas.style.cursor = 'move';
      overlayPoseDraft.pose = { ...drag.pose, x: drag.pose.x + dx / drag.scale, y: drag.pose.y + dy / drag.scale };
      scheduleOverlayLiveRender();
      updateOverlayPoseControls();
      drawTrackingPath();
      text('trackingOverlayInfo', `Frame #${overlayPoseDraft.frame} | Versatz (${fixed(dx / drag.scale, 1)}, ${fixed(dy / drag.scale, 1)}) px | Vorschau`);
    } else { canvas.style.cursor = 'grabbing'; overlayPan = { x: drag.pan.x + dx, y: drag.pan.y + dy }; }
    renderPathOverlay();
  });
  canvas.addEventListener('pointerup', event => {
    if (!drag) return;
    const wasClick = !drag.moved;
    drag = null;
    canvas.style.cursor = 'grab';
    if (!wasClick || !overlayContributors.length) return;
    const point = overlayPoint(event);
    if (!point) return;
    const candidates = overlayContributors.filter(geometry => geometry.supports(point, trackingPixelAllowed));
    if (!candidates.length) return;
    const current = candidates.findIndex(geometry => geometry.entry.frame === overlayPoseDraft?.frame);
    const next = candidates[(current + 1) % candidates.length].entry.frame;
    if (next === overlayPoseDraft?.frame) return;
    element('trackingOverlayFrame').value = String(next);
    void selectOverlayContributor().catch(error => message(error.message, true));
  });
  for (const name of ['pointercancel', 'lostpointercapture']) canvas.addEventListener(name, () => {
    drag = null; canvas.style.cursor = 'grab';
  });
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    if (overlayPoseDraft && (event.shiftKey || event.altKey)) {
      const degrees = event.altKey ? 1 : 0.1;
      const delta = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? event.deltaY * 16 :
        event.deltaMode === WheelEvent.DOM_DELTA_PAGE ? event.deltaY * canvas.clientHeight : event.deltaY;
      overlayPoseDraft.pose.rotation += Math.max(-3, Math.min(3, delta / 100)) * degrees * Math.PI / 180;
      scheduleOverlayLiveRender();
      updateOverlayPoseControls(); renderPathOverlay(); drawTrackingPath();
      text('trackingOverlayInfo', `Frame #${overlayPoseDraft.frame} | Rotation ${fixed(overlayPoseDraft.pose.rotation * 180 / Math.PI, 3)} Grad | Vorschau`);
      return;
    }
    const bounds = canvas.getBoundingClientRect();
    const next = Math.max(0.25, Math.min(64, overlayZoom * Math.exp(-event.deltaY * 0.001)));
    const factor = next / overlayZoom;
    const x = event.clientX - bounds.left - bounds.width / 2, y = event.clientY - bounds.top - bounds.height / 2;
    overlayPan = { x: x - (x - overlayPan.x) * factor, y: y - (y - overlayPan.y) * factor };
    overlayZoom = next; renderPathOverlay();
  }, { passive: false });
  element('trackingOverlayFit').onclick = () => { overlayZoom = 1; overlayPan = { x: 0, y: 0 }; renderPathOverlay(); };
  element('trackingOverlayPoseDiscard').onclick = () => {
    if (!overlayPoseDraft) return;
    overlayPoseDraft.pose = { ...overlayPoseDraft.originalPose };
    scheduleOverlayLiveRender();
    updateOverlayPoseControls(); renderPathOverlay(); drawTrackingPath();
    text('trackingOverlayInfo', `Frame #${overlayPoseDraft.frame} | PosenÃƒÂ¤nderung verworfen.`);
  };
  element('trackingOverlayPoseApply').onclick = () => {
    updateOverlayPoseControls();
    if (!overlayPoseDraft || element('trackingOverlayPoseApply').disabled) return;
    const { frame, pose, originalPose } = overlayPoseDraft;
    const entry = trackingPath.find(item => item.frame === frame);
    if (!entry || !entry.pose || entry.pose.x !== originalPose.x || entry.pose.y !== originalPose.y ||
        entry.pose.rotation !== originalPose.rotation) {
      text('trackingOverlayInfo', 'Frame-Pose wurde zwischenzeitlich geaendert. Ueberlagerung erneut laden.');
      return;
    }
    const dx = pose.x - originalPose.x, dy = pose.y - originalPose.y;
    const rotationDelta = pose.rotation - originalPose.rotation;
    entry.pose = { ...entry.pose, x: pose.x, y: pose.y, rotation: pose.rotation };
    if (entry.raw) entry.raw = { ...entry.raw, x: entry.raw.x + dx, y: entry.raw.y + dy,
      rotation: entry.raw.rotation + rotationDelta };
    overlayPoseDraft = null;
    frameReductionPreview = null;
    element('frameReductionApply').disabled = true;
    updateOverlayPoseControls();
    renderTrackingResults(trackingPath.at(-1));
    const refreshRequest = overlayRequest + 1;
    if (pathSelection) void loadPathOverlay(pathSelection).then(() => {
      if (refreshRequest === overlayRequest && pathOverlay.width) {
        text('trackingOverlayInfo', `Frame #${frame} | Versatz (${fixed(dx, 1)}, ${fixed(dy, 1)}) px | Rotation ${fixed(rotationDelta * 180 / Math.PI, 3)} Grad | Mischbild aktualisiert.`);
      }
    }).catch(error => message(error.message, true)).finally(scheduleTrackingMosaic);
    else scheduleTrackingMosaic();
  };
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
      refitRegion: 100, overlayEdgeFeather: 10 };
    for (const [id, value] of Object.entries(defaults)) element(id).value = value;
    updateRefitPreprocessing();
    if (pathSelection) void loadPathOverlay(pathSelection);
  };
  element('trackingOverlayFrame').onchange = () => void selectOverlayContributor().catch(error => message(error.message, true));
  element('trackingOverlayInspect').onclick = () => {
    const frame = Number(element('trackingOverlayFrame').value);
    const entry = overlayContributors.find(item => item.entry.frame === frame)?.entry;
    if (!entry) return;
    void showTrackedFrame(entry).then(() => element('trackingMatchPanel').scrollIntoView({ behavior: 'smooth', block: 'start' })).catch(error => message(error.message, true));
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
    scheduleTrackingMosaic();
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
    scheduleTrackingMosaic();
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
  for (const item of [...trackingPath, ...trackingFailures].sort((first, second) => first.frame - second.frame).reverse()) {
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
  const patchSearchRadius = number('trackingSearchRadius', 32);
  const windowSize = number('trackingWindow', 1);
  if (!Number.isFinite(patchSearchRadius) || patchSearchRadius < 4 || patchSearchRadius > 1024) {
    throw new Error('Suchradius muss zwischen 4 und 1024 px liegen.');
  }
  if (!(windowSize >= 1 && windowSize <= 240)) throw new Error('Stabilisierungsfenster muss zwischen 1 und 240 Frames liegen.');
  const maxRotation = number('trackingMaxRotation', 5);
  const contextRecent = number('trackingContextRecent', 2); const contextSpatial = number('trackingContextSpatial', 2);
  const contextInterval = 8;
  const contextRadius = number('trackingContextRadius', 32); const contextAngle = number('trackingContextAngle', 1);
  const contextCycleStrict = number('trackingContextCycleStrict', 1.5);
  const contextCycleConditional = number('trackingContextCycleConditional', 7.5);
  if (![contextRecent, contextSpatial].every(value => Number.isInteger(value) && value >= 0 && value <= 8) ||
    !Number.isFinite(contextRadius) || contextRadius < 4 || contextRadius > 256 || !Number.isFinite(contextAngle) || contextAngle < 0.1 || contextAngle > 5 ||
    !Number.isFinite(contextCycleStrict) || contextCycleStrict <= 0 || !Number.isFinite(contextCycleConditional) ||
    contextCycleConditional < contextCycleStrict || contextCycleConditional > 50) {
    throw new Error('Umfeld: n/m 0 bis 8, Radius 4 bis 256 px, Winkel 0.1 bis 5 Grad; Zyklusgrenzen positiv, aufsteigend und maximal 50 px.');
  }
  if (!trackingRectangle || !Number.isFinite(maxRotation) || maxRotation < 1 || maxRotation > 10) throw new Error('Fenster auswaehlen und Rotationsgrenze zwischen 1 und 10 Grad setzen.');
  const sourceImageMask = trackingMaskHasSelection && trackingImageMask ?
    { ...trackingImageMask, coordinateSystem: 'oriented source pixels', data: Array.from(trackingImageMask.data) } : null;
  const imageMask = sourceImageMask ? remapInclusionMask(trackingImageMask, calibration?.maps) : null;
  return { mode: 'window', rectangle: { ...trackingRectangle }, maxRotation,
    contextRecent, contextSpatial, contextInterval, contextRadius, contextAngle, contextCycleStrict, contextCycleConditional,
    sourceImageMask, imageMask: imageMask ? { ...imageMask, data: Array.from(imageMask.data) } : null,
    patchSearchRadius,
    useWebGpu: useWebGpu() };
}

async function runTracking(restart = false) {
  if (!videoInfo || !calibration) throw new Error('Video und passende Linsenkalibrierung werden fuer Tracking benoetigt.');
  if (videoInfo.width !== calibration.field.width || videoInfo.height !== calibration.field.height) throw new Error('Videoaufloesung passt nicht zur Kalibrierung.');
  const start = Math.max(0, Math.min(videoInfo.frameCount - 1, Math.round(number('trackingStart', 0))));
  const end = Math.max(start, Math.min(videoInfo.frameCount - 1, Math.round(number('trackingEnd', videoInfo.frameCount - 1))));
  if (trackingPreviewBusy) return;
  let options = !restart && trackingRunOptions ? trackingRunOptions : trackingOptions();
  if (!restart && trackingRunOptions) {
    const changed = trackingOptions();
    options = { ...trackingRunOptions, ...changed, mode: 'window' };
    trackingRunOptions = options;
  }
  message();
  if (restart) {
    await discardBrightnessSession();
    clearTrackingResults(true); trackingNextIndex = start;
    trackingRunOptions = options;
  }
  if (restart || !trackingProfile) {
    trackingProfile = { mode: options.mode, frames: 0, decodeMs: 0, sharpnessMs: 0, rgbaMs: 0, remapMs: 0, referenceMs: 0, workerTransferMs: 0, trackMs: 0, poseMs: 0,
      renderMs: 0, totalMs: 0, lastTotalMs: 0, accelerators: new Set(), tracker: { accelerator: 'CPU',
        grayscaleMs: 0, contextMs: 0, uploadMs: 0, forwardMs: 0, backwardMs: 0, postprocessMs: 0, sampleMs: 0, matchMs: 0, refinementMs: 0 } };
  }
  trackingInspector.clear();
  video.pause(); trackingRunning = true; trackingPaused = false; updateControls();
  pcbProposal = null; updatePcbControls();
  frameReductionPreview = null;
  element('frameReductionApply').disabled = true;
  text('trackingMatchTabFrame', '');
  element('trackingMatchEmpty').hidden = false;
  activateTrackingTab('poses');
  trackingMosaicRequest++; clearTimeout(trackingMosaicTimer);
  let lastRenderedFrame = null;
  try {
    let nativeTracking = options.useWebGpu && (options.contextRecent > 0 || options.contextSpatial > 0);
    if (nativeTracking && trackingComputer.nativeMaps !== calibration.maps) {
      const { outputWidth, outputHeight, inverseX, inverseY, valid } = calibration.maps;
      await trackingComputer.call('tracking-maps', { maps: { outputWidth, outputHeight, inverseX, inverseY, valid,
        brightness: brightnessCalibration ? { width: brightnessCalibration.width, height: brightnessCalibration.height,
          gain: brightnessCalibration.gain } : null } });
      trackingComputer.nativeMaps = calibration.maps;
    }
    if (!restart && trackingNeedsSeed) {
      const reference = [...trackingPath].reverse().find(entry => entry.pose && entry.frame < trackingNextIndex);
      if (!reference) throw new Error('Legacy-Pfad enthaelt keine letzte gueltige Pose zum Fortsetzen.');
      text('processingStatus', `Letzten Trackingframe #${reference.frame} als Referenz laden`);
      const seeded = await readTrackingFrame(reference.frame, { gpu: options.useWebGpu, rectified: true, output: 'rgba' });
      await trackingComputer.call('track-window-seed', { image: seeded, index: reference.frame,
        rectangle: options.rectangle, searchRadius: options.patchSearchRadius, maxRotation: options.maxRotation,
        pose: reference.pose }, [seeded.data.buffer]);
      trackingNextIndex = reference.frame + 1;
      trackingNeedsSeed = false;
      trackingLost = false;
    }
    while (trackingRunning && trackingNextIndex <= end) {
      const frameStarted = performance.now();
      const index = trackingNextIndex;
      text('processingStatus', `Tracking Frame ${index}/${end}`);
      element('progress').value = (index - start + 1) / (end - start + 1);
      const image = await readTrackingFrame(index, { gpu: options.useWebGpu,
        rectified: !nativeTracking, output: nativeTracking ? 'native' : 'rgba', measureSharpness: false });
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
      } else detection = await trackingComputer.call('track-window', { image, index, options }, [image.data.buffer]);
      let referenceMs = 0;
      if (detection.contextPending) {
        for (const referenceIndex of detection.references) {
          let reference = null; let failure = null;
          const referenceStarted = performance.now();
          try {
            if (!trackingRunning) throw new Error('Pausiert');
            text('processingStatus', `Tracking #${index} | Umfeld #${referenceIndex}`);
            reference = await readTrackingFrame(referenceIndex, { gpu: options.useWebGpu, rectified: !nativeTracking,
              output: nativeTracking ? 'native' : 'rgba', measureSharpness: false });
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
      if (detection.context?.registrationChoice?.backend === 'CPU') nativeTracking = false;
      const workerRoundtripMs = performance.now() - workerStarted - referenceMs - nativeRenderMs;
      const poseStarted = performance.now();
      const raw = detection.raw ?? null;
      const entry = { frame: index, timestamp: videoInfo.timestamps[index], sharpness: image.sharpness, raw, pose: null,
        mode: 'window', rectangle: options.rectangle, score: detection.score, iterations: detection.iterations, resolution: detection.resolution,
        incremental: detection.incremental, incrementalMatch: detection.incrementalMatch, context: detection.context,
        points: detection.success ? 1 : 0,
        success: detection.success && Boolean(raw), reason: detection.reason || (raw ? '' : 'Fensterregistrierung lieferte keine Pose.'),
        accelerator: detection.accelerator || 'CPU' };
      if (!detection.success) {
        trackingFailures.push(entry);
        renderTrackingResults(entry); trackingInspector.show(entry);
        text('trackingMatchTabFrame', `(#${entry.frame})`);
        element('trackingMatchEmpty').hidden = true;
        activateTrackingTab('match');
        trackingPaused = true;
        message(`${detection.reason} Suchgrenze anpassen und denselben Frame mit Fortsetzen erneut versuchen.`, true);
        break;
      }
      if (detection.context?.loopClosure) applyLoopClosure(trackingPath, detection.context.loopClosure);
      trackingPath.push(entry);
      entry.pose = stabilizePose(trackingPath, number('trackingWindow', 1));
      const poseMs = performance.now() - poseStarted;
      const overlayStarted = performance.now();
      if (trackingPath.length % 8 === 0 || index === end) {
        renderTrackingPreview(); renderTrackingResults(entry); lastRenderedFrame = entry.frame;
      }
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
    trackingLost = !trackingNeedsSeed;
    throw error;
  } finally {
    const completed = trackingNextIndex > end;
    trackingRunning = false; trackingPaused = !completed && !trackingLost;
    const latest = trackingPath.at(-1);
    const failedCurrent = trackingFailures.at(-1)?.frame === trackingNextIndex;
    if (latest && !failedCurrent && latest.frame !== lastRenderedFrame) {
      renderTrackingPreview(); renderTrackingResults(latest);
    }
    if (pathSelection) void loadPathOverlay(pathSelection).finally(scheduleTrackingMosaic);
    else scheduleTrackingMosaic();
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
  canvas.addEventListener('pointerdown', event => {
    drag = { x: event.clientX, y: event.clientY, pan: { ...pan }, moved: false };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    inspect(event, canvas);
    if (!drag) return;
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
  canvas.addEventListener('pointercancel', () => { drag = null; });
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
  await computer.call('reset-calibration');
  calibration = null; frames = new Map(); snapshotFrames = []; snapshotParameters = null;
  observationDiagnostics = null; observationDiagnosticsDirty = true; geometrySelection = null;
  detections.clear(); currentDetection = null; phaseChecked = false; phaseStarted = false; stale = false; detectionsStale = false;
  detectedOnce = false; acceptedSinceFit = 0; nextProcessingIndex = 0; measurements = []; rectifiedReady = false;
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
  const candidate = new WorkerClient('./decoder-worker.js', showProgress);
  let metadata;
  try { metadata = await candidate.call('open', { file }); }
  catch (error) { candidate.terminate(); throw error; }
  const videoNameMismatch = expectedVideo && expectedVideo.name !== metadata.name && (calibration || frames.size);
  if (calibration && (metadata.width !== calibration.field.width || metadata.height !== calibration.field.height)) {
    candidate.terminate(); throw new Error('Videoaufloesung passt nicht zur geladenen Kalibrierung.');
  }
  await discardBrightnessSession();
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
element('resetButton').onclick = () => { if (confirm('Linsenkalibrierung und Checkerboard-Beobachtungen zuruecksetzen? Video, Helligkeitskorrektur und Tracking bleiben erhalten.')) void task(resetCalibration); };
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
  if (!trackingPreviewImage.width && videoInfo && !taskBusy && !continuous && !rectifiedPlayback && !trackingRunning) {
    void previewTrackingStartFrame();
  }
  if (save) saveVideoOptions();
}
for (const id of ['previewBrightness', 'previewContrast', 'previewGamma']) element(id).oninput = updatePreviewAdjustments;
async function initializeGpuAdapterSelector() {
  const select = element('globalGpuAdapter');
  const adapters = await discoverWebGpuAdapters();
  select.replaceChildren(new Option('Keine', 'none'),
    ...adapters.map(adapter => new Option(adapter.label, adapter.value)));
  let saved = null;
  try { saved = localStorage.getItem('rasterlabor-webgpu-adapter'); } catch {}
  const selected = adapters.find(adapter => adapter.value === saved) ?? adapters[0] ?? null;
  const value = saved === 'none' ? 'none' : selected?.value ?? 'none';
  const unavailable = saved && saved !== 'none' && saved !== value;
  setWebGpuSelection(value);
  select.value = value;
  gpuAdapterReady = true;
  const adapter = adapters.find(item => item.value === value);
  text('globalGpuStatus', adapter ? `${unavailable ? 'Gespeicherte Auswahl nicht mehr angeboten. ' : ''}${adapter.name} Ã‚Â· ` +
    `max. Textur ${adapter.limits.maxTextureDimension2D} px. Der Browser bestimmt die Hardware anhand der Auswahl.` :
    adapters.length ? 'GPU aus: CPU wird verwendet.' : 'Kein WebGPU-GerÃƒÂ¤t verwendbar; CPU wird verwendet.');
  updateControls(); updateMergeControls();
}
element('globalGpuAdapter').onchange = async () => {
  if (taskBusy || continuous || rectifiedPlayback || navigationBusy || trackingRunning || trackingPreviewBusy ||
      pcbBusy || pathRefitBusy || mergeRunning || overlayGpuInFlight > 0) {
    element('globalGpuAdapter').value = getWebGpuSelection(); return;
  }
  releaseOverlayGpuSession();
  setWebGpuSelection(element('globalGpuAdapter').value);
  try { localStorage.setItem('rasterlabor-webgpu-adapter', getWebGpuSelection()); } catch {}
  await frameReader.dispose();
  trackingComputer.nativeMaps = null;
  mergeTilePlan = null;
  const option = element('globalGpuAdapter').selectedOptions[0];
  text('globalGpuStatus', useWebGpu() ? `${option.textContent}. Der Browser bestimmt die Hardware anhand der Auswahl.` :
    'GPU aus: CPU wird verwendet.');
  updateMergeControls();
  if (useWebGpu()) void updateMergeTilePlan();
};
element('resetPreviewAdjustments').onclick = () => {
  element('previewBrightness').value = 0;
  element('previewContrast').value = 100;
  element('previewGamma').value = 1;
  updatePreviewAdjustments();
};
element('zoomIn').onclick = () => { zoom = Math.min(20, zoom * 1.25); draw(); };
element('zoomOut').onclick = () => { zoom = Math.max(0.5, zoom / 1.25); draw(); };
element('fitView').onclick = () => { zoom = 1; pan = { x: 0, y: 0 }; draw(); };
element('trackingZoomIn').onclick = () => { trackingZoom = Math.min(20, trackingZoom * 1.25); renderTrackingPreview(); };
element('trackingZoomOut').onclick = () => { trackingZoom = Math.max(0.5, trackingZoom / 1.25); renderTrackingPreview(); };
element('trackingFitView').onclick = () => { trackingZoom = 1; trackingPan = { x: 0, y: 0 }; renderTrackingPreview(); };
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
  if (workflow === 'brightness') drawBrightnessCalibration();
  if (workflow === 'merge') { updateMergeControls(); void updateMergeTilePlan(); }
};
element('mergeStart').onclick = () => void startMerge().catch(error => message(error.message, true));
element('mergeCancel').onclick = () => { mergeCancelled = true; text('mergeStatus', 'Breche nach dem laufenden GPU-Durchlauf ab...'); };
element('mergeSavePreview').onclick = () => void saveMergePreview().catch(error => message(error.message, true));
element('mergeSaveMethod').onchange = () => {
  if (element('mergeSaveMethod').value === 'picker') element('mergeSplit').value = '0';
  mergeResultStatus = null; updateMergeControls();
};
for (const id of ['mergeBlend', 'mergeMaxFrames', 'mergeEdgeFeather', 'mergeSplit']) element(id).onchange = () => {
  if (element('mergeSplit').value !== '0') element('mergeSaveMethod').value = 'download';
  mergeResultStatus = null; updateMergeControls();
};
element('brightnessFit').onclick = () => void task(() => fitBrightnessCalibration(true)).catch(error => message(error.message, true));
element('brightnessPause').onclick = () => { brightnessRunning = false; brightnessPaused = true; updateControls(); };
element('brightnessResume').onclick = () => void task(() => fitBrightnessCalibration(false)).catch(error => message(error.message, true));
element('brightnessReset').onclick = () => void task(resetBrightnessCalibration).catch(error => message(error.message, true));
element('brightnessDisable').onclick = () => void task(disableBrightnessCalibration).catch(error => message(error.message, true));
element('brightnessSave').onclick = () => void task(downloadBrightnessCalibration).catch(error => message(error.message, true));
element('brightnessLoad').onclick = () => element('brightnessFile').click();
element('brightnessFile').onchange = event => {
  const [file] = event.target.files;
  if (file) void task(() => loadBrightnessCalibration(file)).catch(error => message(error.message, true));
  event.target.value = '';
};
element('trackingStartButton').onclick = () => void runTracking(true).catch(error => {
  message(error.message, true); trackingRunning = false; updateControls();
});
element('trackingPauseButton').onclick = () => { trackingRunning = false; trackingPaused = true; updateControls(); };
element('trackingResumeButton').onclick = () => void runTracking(false).catch(error => {
  message(error.message, true); trackingRunning = false; updateControls();
});
element('trackingResetButton').onclick = () => void (async () => {
  clearTrackingResults();
  void previewTrackingStartFrame();
  void loadTrackingMaskPreview();
})().catch(error => message(error.message, true));
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

const detectionParameters = new Set(['approxStep', 'columns', 'rows', 'threshold']);
const nonGeometric = new Set(['follow', 'opticalConfiguration', 'minMotion', 'updateEvery', 'startPercent', 'endPercent']);
for (const control of document.querySelectorAll('.settings input, .settings select')) control.addEventListener('change', () => {
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
  const brightness = brightnessCalibration ? { ...brightnessCalibration,
    gain: brightnessCalibration.gain.slice(), supported: brightnessCalibration.supported.slice() } : null;
  const bytes = await computer.call('export', { frames: snapshotFrames, video: expectedVideo, parameters: snapshotParameters,
    opticalConfiguration: element('opticalConfiguration').value, tracking, brightness },
  brightness ? [brightness.gain.buffer, brightness.supported.buffer] : []);
  const url = URL.createObjectURL(new Blob([bytes], { type: 'application/zip' }));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = `kalibrierung-v${calibration.version}-${calibration.quality}.zip`; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  message(brightnessCalibration ? 'Gesamtkalibrierung inklusive Helligkeitskorrektur gespeichert.' : 'Gesamtkalibrierung ohne Helligkeitskorrektur gespeichert.');
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
  if (file.name.toLowerCase().endsWith('.json')) {
    const imported = importOptimizedProject(JSON.parse(await file.text()),calibration,videoInfo);
    clearTrackingResults();
    restoreTracking(imported);
    updateControls(); drawTrackingPath();
    message('Optimierte Posen und Match-Netz geladen. Linsen- und Helligkeitskalibrierung bleiben erhalten.');
    return;
  }
  if (calibration && !confirm('Aktuelle Kalibrierung durch das ausgewaehlte Paket ersetzen?')) return;
  await discardBrightnessSession();
  if (file.size > 768 * 1024 * 1024) throw new Error('Paket groesser als 768 MiB.');
  const bytes = new Uint8Array(await file.arrayBuffer());
  const imported = await computer.call('import', { bytes }, [bytes.buffer]);
  const videoNameMismatch = videoInfo && imported.video?.name !== videoInfo.name;
  calibration = imported.calibration;
  brightnessCalibration = imported.brightness;
  await frameReader.dispose(); trackingComputer.nativeMaps = null;
  clearTrackingResults();
  restoreTracking(imported.tracking);
  applyParameters(imported.parameters);
  frames = new Map(imported.observations.map(frame => [frame.id, frame]));
  observationDiagnostics = null; observationDiagnosticsDirty = true; geometrySelection = null;
  snapshotFrames = imported.observations; snapshotParameters = imported.parameters; expectedVideo = imported.video;
  element('opticalConfiguration').value = imported.opticalConfiguration || '';
  phaseStarted = true; phaseChecked = [...frames.values()].some(frame => frame.accepted);
  stale = false; detectionsStale = false; rectifiedReady = false; detections.clear(); currentDetection = null;
  nextProcessingIndex = 0; measurements = [];
  if (brightnessCalibration) drawBrightnessCalibration();
  else {
    text('brightnessState', 'Kein Feld'); element('brightnessState').className = 'tag';
    text('brightnessMetrics', 'Keine Messung'); drawBrightnessCalibration();
  }
  renderFieldImages(); updateTable(); updateMetrics(); updateControls(); draw();
  message(videoNameMismatch ? 'Kalibrierung geladen. Der Videodateiname weicht ab; Video und Kalibrierungsdaten bleiben erhalten.' :
    brightnessCalibration ? 'Kalibrierung geladen. Geometrie und Helligkeitskorrektur sind aktiv.' :
    'Kalibrierung geladen. Feld und Metadaten sind lokal verfuegbar.');
  setTimeout(() => { void loadTrackingMaskPreview(); }, 0);
});
element('infoButton').onclick = () => element('infoDialog').showModal();
element('closeInfo').onclick = () => element('infoDialog').close();
installPan(rawCanvas); installPan(resultCanvas);
installMergePreviewInteraction();
installTrackingPreviewInteraction();
installTrackingTabs();
installPcbRealignment();
installFineControls();
installFrameReduction();
installPathInteraction();
installTrackingMaskInteraction();
new ResizeObserver(draw).observe(element('rawCanvas').parentElement);
new ResizeObserver(draw).observe(element('resultCanvas').parentElement);
new ResizeObserver(renderTrackingPreview).observe(element('trackingPreviewCanvas').parentElement);
window.addEventListener('beforeunload', event => { if (mergeRunning || (frames.size && !calibration)) { event.preventDefault(); event.returnValue = ''; } });
restoreVideoOptions();
video.playbackRate = Number(element('speed').value);
text('trackingMaskBrushValue', `${element('trackingMaskBrush').value} px`);
updatePreviewAdjustments(false);
updateControls(); updateMetrics(); draw();
void initializeGpuAdapterSelector();
setInterval(updateMemoryStats, 2000);
