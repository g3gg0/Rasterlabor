import { installTrackingInspector } from '../src/tracking-inspector.js';
import { WorkerClient } from '../src/rpc.js';
import { contextImage, registerOverlap } from '../src/context-tracker.js';
import { validateTracking } from '../src/tracking-data.js';

function check(condition, message) { if (!condition) throw new Error(message); }

function texture() {
  const image = new ImageData(256, 192);
  for (let row = 0; row < image.height; row++) for (let column = 0; column < image.width; column++) {
    const value = 125 + 28 * Math.sin(column * 0.31 + row * 0.19) + 32 * Math.cos(column * 0.13 - row * 0.37) +
      25 * Math.sin(column * 0.051 + row * 0.11) + 18 * Math.cos(column * 0.43 + row * 0.07);
    image.data.set([value, value, value, 255], 4 * (row * image.width + column));
  }
  return image;
}

export async function runInspectorWorkerChecks() {
  const client = new WorkerClient('/compute-worker.js');
  const image = texture(); const origin = { x: 0, y: 0, rotation: 0 }; const prediction = { ...origin, x: 2 };
  const limits = { radius: 16, angle: 1, coarseStep: 0, reverseRadius: 8 };
  const register = async (currentImage, mask = null) => {
    const current = await createImageBitmap(currentImage); const reference = await createImageBitmap(image);
    return client.call('context-inspect', { current, reference, prediction, referencePose: origin, limits, mask }, [current, reference]);
  };
  try {
    const options = { rectangle: { x: 16, y: 16, width: 224, height: 160 }, patchSearchRadius: 16, maxRotation: 5 };
    const initial = await client.call('track-window', { image, index: 0, options });
    check(initial.initial, 'Window tracker not initialized');
    const result = await register(image);
    check(result.accepted && result.backward.accepted, `Identity registration failed: ${JSON.stringify(result)}`);
    check(result.backward.searchRadius === 8 && result.attempts.length === 1, 'Search limits missing');
    const pyramid = contextImage(image);
    const expected = registerOverlap(pyramid, pyramid, prediction, origin, limits);
    check(Math.hypot(result.pose.x - expected.pose.x, result.pose.y - expected.pose.y) < 1e-8, 'Worker registration differs from direct CPU');
    const rejected = await register(new ImageData(256, 192));
    check(!rejected.accepted && rejected.backward === null && rejected.reason, 'Transparent pair accepted');
    const masked = await register(image, { width: 1, height: 1, sourceWidth: 256, sourceHeight: 192, cellSize: 256, data: [0] });
    check(!masked.accepted, 'Saved inclusion mask ignored');
    const next = await client.call('track-window', { image, index: 1, options });
    check(next.success && next.incrementalMatch.frame === 0, 'Debug call changed incremental tracker state');
    const dataset = validateTracking(JSON.parse(JSON.stringify({ format: 'rasterlabor-xyr-tracking', model_version: 1,
      path: [{ frame: 1, timestamp: 1, pose: origin, raw: origin, context: { matches: [result] }, incrementalMatch: next.incrementalMatch }],
      failures: [{ frame: 2, timestamp: 2, incrementalMatch: { ...next.incrementalMatch, accepted: false } }] })));
    check(dataset.path[0].context.matches[0].attempts.length === 1 && dataset.failures.length === 1, 'Diagnostic roundtrip lost attempts');
    return { accepted: result.accepted, reverseDistance: result.reverseDistance, rejected: rejected.reason, masked: masked.reason, trackerUnchanged: true, roundtrip: true };
  } finally { client.terminate(); }
}

export function mountInspectorFixture() {
  document.getElementById('trackingInspector')?.remove();
  const image = texture(); const origin = { x: 0, y: 0, rotation: 0 };
  const pair = { frame: 1400, kind: 'spatial', accepted: false, reason: 'Rueckwaertspruefung', score: 0.99, margin: 0.03,
    searchRadius: 32, prediction: { ...origin, x: 2 }, referencePose: origin, pose: { ...origin, x: 0.01 },
    attempts: [{ accepted: false, reason: 'Korrelation', score: 0.5, searchRadius: 16, angle: 1, coarseStep: 0, pose: { ...origin, x: 3 } },
      { accepted: true, score: 0.99, searchRadius: 32, angle: 1, coarseStep: 16, pose: { ...origin, x: 0.01 } }],
    backward: { accepted: true, score: 0.98, pose: { ...origin, x: 2 }, searchRadius: 16, angle: 1 }, reverseDistance: 2 };
  const entry = { frame: 1555, timestamp: 1, raw: { ...origin }, pose: { ...origin }, context: { matches: [pair,
    { frame: 1401, kind: 'spatial', accepted: false, reason: 'Decode fehlgeschlagen', prediction: origin, referencePose: origin, attempts: [], backward: null },
    { frame: 1554, kind: 'recent', accepted: false, reason: 'Altbestand ohne Startpose' }], inliers: [], applied: false } };
  const inspector = installTrackingInspector({ readFrame: async () => ({ bitmap: await createImageBitmap(image) }), getMask: () => null, busy: () => false });
  inspector.show(entry);
  const root = document.getElementById('trackingInspector'); root.closest('main').hidden = false;
  return { entry, inspector };
}

export async function runInspectorDisplayChecks() {
  document.getElementById('trackingInspector')?.remove();
  const image = texture(); const origin = { x: 0, y: 0, rotation: 0 };
  const match = { frame: 1401, kind: 'spatial', accepted: true, score: 0.99, searchRadius: 32,
    prediction: origin, referencePose: origin, pose: origin, attempts: [], backward: null };
  const adjusted = [];
  const inspector = installTrackingInspector({ readFrame: async () => ({ bitmap: await createImageBitmap(image) }),
    getMask: () => ({ width: 2, height: 2, sourceWidth: 256, sourceHeight: 192, cellSize: 128, data: [0, 1, 1, 1] }),
    busy: () => false, drawImage: (context, bitmap, ...coordinates) => {
      adjusted.push({ alpha: context.globalAlpha, filter: 'preview', image: bitmap }); context.drawImage(bitmap, ...coordinates);
    } });
  inspector.show({ frame: 1555, context: { matches: [match], inliers: [], applied: false } });
  const root = document.getElementById('trackingInspector'); root.closest('main').hidden = false;
  root.querySelector('tbody button').click();
  await new Promise(resolve => setTimeout(resolve));
  const canvas = root.querySelector('canvas'); const context = canvas.getContext('2d'); const original = context.drawImage;
  const calls = [];
  context.drawImage = function(source, ...coordinates) { calls.push({ source, alpha: this.globalAlpha }); return original.call(this, source, ...coordinates); };
  adjusted.length = 0;
  root.querySelector('[data-role=alpha]').value = '0'; root.querySelector('[data-role=alpha]').dispatchEvent(new Event('input'));
  context.drawImage = original;
  check(adjusted.length === 1 && adjusted[0].alpha === 1, 'Hidden current image was adjusted or drawn');
  check(calls.length === 2 && calls[0].source === adjusted[0].image && calls[1].source !== adjusted[0].image, 'Hidden current mask was drawn');
  check(context.imageSmoothingEnabled && context.imageSmoothingQuality === 'high', 'High-quality scaling is disabled');
  return { adjustedImages: adjusted.length, totalDrawCalls: calls.length, smoothing: context.imageSmoothingQuality };
}