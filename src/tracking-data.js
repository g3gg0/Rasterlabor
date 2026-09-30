import { validSharpness } from './sharpness.js';

const wrap = angle => Math.atan2(Math.sin(angle), Math.cos(angle));

export function applyLoopClosure(path, correction) {
  const span = correction.frame - correction.anchorFrame;
  if (!(span > 0)) return 0;
  let corrected = 0;
  for (const entry of path) {
    if (entry.frame <= correction.anchorFrame) continue;
    const weight = Math.min(1, (entry.frame - correction.anchorFrame) / span);
    for (const key of ['raw', 'pose']) {
      const pose = entry[key];
      if (!pose || ![pose.x, pose.y, pose.rotation].every(Number.isFinite)) continue;
      pose.x += correction.dx * weight;
      pose.y += correction.dy * weight;
      pose.rotation = wrap(pose.rotation + correction.rotation * weight);
    }
    corrected++;
  }
  return corrected;
}

export function trackingReferenceCounts(path) {
  const counts = new Map(path.map(entry => [entry.frame, { referencedBy: 0, references: null, consensusBy: 0, consensus: 0 }]));
  const hasDiagnostics = path.some(entry => Array.isArray(entry.context?.matches));
  for (const entry of path) {
    const stats = counts.get(entry.frame);
    if (!hasDiagnostics) stats.referencedBy = null;
    if (!Array.isArray(entry.context?.matches)) continue;
    const references = new Set(entry.context.matches.map(match => match?.frame).filter(frame => Number.isSafeInteger(frame) && frame >= 0 && frame !== entry.frame));
    const inliers = new Set(entry.context.applied && Array.isArray(entry.context.inliers) ? entry.context.inliers : []);
    stats.references = references.size;
    for (const frame of references) {
      const reference = counts.get(frame);
      if (reference) reference.referencedBy++;
      if (inliers.has(frame)) {
        stats.consensus++;
        if (reference) reference.consensusBy++;
      }
    }
  }
  return counts;
}

export function trackingVideoCompatible(tracking, video) {
  if (!video) return false;
  const source = tracking?.video;
  if (source && Number.isFinite(source.width) && Number.isFinite(source.height) &&
      (source.width !== video.width || source.height !== video.height)) return false;
  if (Number.isSafeInteger(video.frameCount) && tracking?.path?.some(entry => entry.frame >= video.frameCount)) return false;
  return true;
}

export function removeUnselectedTrackingFrames(tracking, selectedFrames, reduction) {
  if (!tracking?.path?.length || !(selectedFrames instanceof Set) || !selectedFrames.size) {
    throw new Error('Keine registrierten Frames zum Entfernen ausgewaehlt.');
  }
  const path = tracking.path.filter(entry => selectedFrames.has(entry.frame)).map(entry => {
    const kept = { ...entry };
    if (entry.incrementalMatch && !selectedFrames.has(entry.incrementalMatch.frame)) delete kept.incrementalMatch;
    if (entry.context) {
      const context = { ...entry.context };
      if (Array.isArray(context.matches)) context.matches = context.matches.filter(match => selectedFrames.has(match?.frame));
      if (Array.isArray(context.inliers)) context.inliers = context.inliers.filter(frame => selectedFrames.has(frame));
      if (Array.isArray(context.matches)) context.selected = context.matches.length;
      if (context.loopClosure && !selectedFrames.has(context.loopClosure.anchorFrame)) delete context.loopClosure;
      delete context.spatialConfirmation;
      kept.context = context;
    }
    return kept;
  });
  if (path.length !== selectedFrames.size) throw new Error('Die Auswahl enthaelt unbekannte Trackingframes.');
  return { ...tracking, path, failures: [], reduction };
}

export function validateTracking(tracking, fallbackVideo = null) {
  if (tracking == null) return null;
  if (tracking.format !== 'rasterlabor-xyr-tracking' || tracking.model_version !== 1 || !Array.isArray(tracking.path)) {
    throw new Error('Unbekanntes Trackingformat.');
  }
  // The live-session rescue ZIP used CSV column names and degrees, also at version 1.
  const legacy = tracking.path.length > 0 && tracking.path.every(entry => entry &&
    Object.hasOwn(entry, 'timestamp_us') && !Object.hasOwn(entry, 'timestamp'));
  if (legacy) {
    const rectangle = tracking.rectangle;
    const validRectangle = rectangle && [rectangle.x, rectangle.y, rectangle.width, rectangle.height].every(Number.isFinite) &&
      rectangle.width > 0 && rectangle.height > 0 ? rectangle : null;
    tracking = { ...tracking, video: tracking.video ?? fallbackVideo,
      coordinate_system: 'rectified pixels; x right; y down; rotation radians; camera pose relative to first tracked frame',
      rectangle: validRectangle,
      path: tracking.path.map(entry => {
        if (![entry.x_px, entry.y_px, entry.rotation_deg, entry.raw_x_px, entry.raw_y_px, entry.raw_rotation_deg].every(Number.isFinite)) {
          throw new Error('Ungueltige Trackingpose im CSV-Datensatz.');
        }
        return { ...entry, timestamp: entry.timestamp_us,
          pose: { x: entry.x_px, y: entry.y_px, rotation: entry.rotation_deg * Math.PI / 180 },
          raw: { x: entry.raw_x_px, y: entry.raw_y_px, rotation: entry.raw_rotation_deg * Math.PI / 180,
            points: entry.patches, rms: entry.rms_px },
          mode: tracking.options?.trackingMode ?? 'patches', rectangle: validRectangle,
          points: entry.patches, success: true, reason: '', rediscovered: 0, accelerator: 'Import' };
      }) };
  }
  const ids = new Set();
  for (const entry of tracking.path) {
    if (!entry || !Number.isSafeInteger(entry.frame) || entry.frame < 0 || ids.has(entry.frame) || !Number.isFinite(entry.timestamp)) {
      throw new Error('Ungueltiger oder doppelter Trackingframe.');
    }
    ids.add(entry.frame);
    if (entry.sharpness != null && !validSharpness(entry.sharpness)) throw new Error('Ungueltige Schaerfemessung im Trackingframe.');
    for (const pose of [entry.pose, entry.raw]) {
      if (pose != null && ![pose.x, pose.y, pose.rotation].every(Number.isFinite)) throw new Error('Ungueltige Trackingpose.');
    }
  }
  return tracking.path.length ? tracking : null;
}
