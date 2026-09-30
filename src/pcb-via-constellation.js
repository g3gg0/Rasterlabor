import { composePose, invertPose } from './pcb-realignment.js';
import { pcbVias, recoverPcbViaPair } from './pcb-vias.js';

const angle = r => Math.atan2(Math.sin(r), Math.cos(r));
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

// Verify gradients rather than the large, almost uniform solder-mask areas.
// This gives traces and nonperiodic boundaries a vote alongside the via pattern.
export function pcbTraceAgreement(first, second, referencePose, currentPose) {
  const a = first.levels.findLast(level => level.scale <= 4) ?? first.levels[0];
  const b = second.levels.findLast(level => level.scale <= 4) ?? second.levels[0];
  const mapping = composePose(invertPose(currentPose), referencePose);
  const c = Math.cos(mapping.rotation), s = Math.sin(mapping.rotation);
  const gradient = (level, x, y) => {
    x = Math.round(x); y = Math.round(y);
    if (x < 1 || y < 1 || x + 1 >= level.width || y + 1 >= level.height) return null;
    const i = y * level.width + x, d = level.gray;
    if ([d[i - 1], d[i + 1], d[i - level.width], d[i + level.width]].some(value => value < 0)) return null;
    return { x: d[i + 1] - d[i - 1], y: d[i + level.width] - d[i - level.width] };
  };
  let dot = 0, aa = 0, bb = 0, support = 0;
  for (let y = 2; y < a.height - 2; y += 2) for (let x = 2; x < a.width - 2; x += 2) {
    const p = gradient(a, x, y);
    if (!p) continue;
    const local = composePose(mapping, { x: (x + .5) * a.scale - first.width / 2,
      y: (y + .5) * a.scale - first.height / 2, rotation: 0 });
    const q = gradient(b, (local.x + second.width / 2) / b.scale - .5,
      (local.y + second.height / 2) / b.scale - .5);
    if (!q) continue;
    const qx = c * q.x + s * q.y, qy = -s * q.x + c * q.y;
    const pa = p.x * p.x + p.y * p.y, qa = qx * qx + qy * qy;
    if (Math.max(pa, qa) < 64) continue;
    // Cap reflection edges so a single shiny border cannot dominate all traces.
    const weight = 1 / Math.max(1, Math.sqrt(Math.max(pa, qa)) / 40);
    dot += weight * (p.x * qx + p.y * qy); aa += weight * pa; bb += weight * qa; support++;
  }
  return { score: aa && bb ? dot / Math.sqrt(aa * bb) : 0, support };
}

export function recoverPcbViaConstellation(first, second, reference, current, data) {
  const refOffset = { x: reference.offset[0] + first.width / 2, y: reference.offset[1] + first.height / 2, rotation: 0 };
  const curOffset = { x: current.offset[0] + second.width / 2, y: current.offset[1] + second.height / 2, rotation: 0 };
  const refCenter = composePose(reference.pose, refOffset), curCenter = composePose(current.pose, curOffset);
  const refPoints = pcbVias(first), curPoints = pcbVias(second);
  const result = { accepted: false, method: 'Via-Konstellation + Leiterbahnen',
    referenceVias: refPoints.length, currentVias: curPoints.length, seedChecks: [], attempts: [] };
  if (refPoints.length < 3 || curPoints.length < 3) return { ...result, reason: 'Weniger als drei Vias' };
  // Dense texture is the ordinary cell matcher's job; keep fallback work bounded.
  if (refPoints.length > 160 || curPoints.length > 160) return { ...result, reason: 'Zu viele Via-Kandidaten' };
  const radius = data.coarseRadius ?? data.limits?.radius ?? 192;
  const maxAngle = Math.min(5, data.limits?.angle ?? 5) * Math.PI / 180;
  const bins = new Map();
  for (const fraction of [0, -.5, .5, -1, 1]) {
    const rotation = curCenter.rotation + fraction * maxAngle;
    for (const target of refPoints) for (const source of curPoints) {
      if (Math.max(target.width / source.width, source.width / target.width,
        target.height / source.height, source.height / target.height) > 1.6) continue;
      const a = composePose(refCenter, { ...target, rotation: 0 });
      const b = composePose({ ...curCenter, rotation }, { ...source, rotation: 0 });
      const dx = a.x - b.x, dy = a.y - b.y;
      if (Math.hypot(dx, dy) > radius) continue;
      const key = `${fraction}:${Math.round(dx / 24)}:${Math.round(dy / 24)}`;
      const bin = bins.get(key) ?? { dx: 0, dy: 0, rotation, votes: 0 };
      bin.dx += dx; bin.dy += dy; bin.votes++; bins.set(key, bin);
    }
  }
  const candidates = [], checked = [];
  for (const bin of [...bins.values()].sort((a, b) => b.votes - a.votes).slice(0, 32)) {
    const seed = { x: curCenter.x + bin.dx / bin.votes, y: curCenter.y + bin.dy / bin.votes, rotation: bin.rotation };
    const vias = recoverPcbViaPair(first, second, refCenter, curCenter, seed,
      { coarseRadius: radius, referencePoints: refPoints, currentPoints: curPoints, minimumDescriptor: .7, minimumConsensus: .35, minimumPatchScore: .6 });
    result.seedChecks.push({ votes: bin.votes, matches: vias.matches, inliers: vias.inliers,
      accepted: vias.accepted, residual: vias.residual, patchScores: vias.patchScores });
    if (!vias.accepted || vias.inliers < 3 || Math.abs(angle(vias.pose.rotation - curCenter.rotation)) > maxAngle + 1e-6) continue;
    const cameraPose = composePose(vias.pose, invertPose(curOffset));
    if (distance(cameraPose, current.pose) > radius) continue;
    // Collinear vias can constrain angle; repeated rows are disambiguated by
    // independent traces and competing pose hypotheses below.
    if (checked.some(pose => distance(pose, vias.pose) < 8 && Math.abs(angle(pose.rotation - vias.pose.rotation)) < .002)) continue;
    checked.push(vias.pose);
    const reverse = recoverPcbViaPair(second, first, vias.pose, refCenter, refCenter,
      { coarseRadius: radius, referencePoints: curPoints, currentPoints: refPoints, minimumDescriptor: .7, minimumConsensus: .35, minimumPatchScore: .6 });
    if (!reverse.accepted || reverse.inliers < 3) continue;
    const cycle = distance(reverse.pose, refCenter) + Math.abs(angle(reverse.pose.rotation - refCenter.rotation)) * Math.hypot(first.width, first.height) / 2;
    if (cycle > Math.min(3, data.limits?.cycleLimit ?? 3)) continue;
    const traces = pcbTraceAgreement(first, second, refCenter, vias.pose);
    const patchScore = Math.min(...vias.patchScores, ...reverse.patchScores);
    result.attempts.push({ pose: cameraPose, inliers: vias.inliers, cycle, traces, patchScore });
    const strongGeometry = vias.inliers >= 5 && reverse.inliers >= 5 &&
      Math.max(vias.residual, reverse.residual) <= 3 && cycle <= 1.5;
    if (traces.support < 64 || patchScore < .6) continue;
    const eligible = traces.score >= (strongGeometry ? .6 : .65) && patchScore >= (strongGeometry ? .6 : .9);
    candidates.push({ vias, reverse, cycle, cameraPose, traces, patchScore, eligible });
  }
  candidates.sort((a, b) => b.traces.score - a.traces.score || b.vias.inliers - a.vias.inliers);
  const best = candidates.find(candidate => candidate.eligible);
  const next = candidates.find(candidate => candidate !== best);
  if (!best) return { ...result, reason: 'Via-Geometrie oder Leiterbahnen nicht bestaetigt' };
  if (next && best.traces.score - next.traces.score < .08)
    return { ...result, reason: 'Mehrere passende Via-Reihen / Y-Versatz mehrdeutig' };
  const { vias, reverse, cycle, cameraPose, patchScore } = best;
  const geometricEvidence = { viaCount: Math.min(vias.inliers, reverse.inliers),
    residual: Math.max(vias.residual, reverse.residual), cycle, patchScore,
    traceScore: best.traces.score, traceSupport: best.traces.support,
    ambiguityMargin: next ? best.traces.score - next.traces.score : 1 };
  return { ...result, accepted: true, score: patchScore, geometricEvidence,
    backwardPose: composePose(reverse.pose, invertPose(refOffset)),
    reverseVerification: { accepted: true, cycle, score: patchScore, cells: reverse.inliers, support: reverse.inliers * 128 },
    fft: { accepted: true, pose: cameraPose, cellSize: 32, method: 'PCB-Via-Konstellation', translationOnly: false,
      residualRms: vias.residual, uniqueSupportArea: vias.inliers * 128,
      inlierCells: vias.pairs.map((_, i) => i),
      cells: vias.pairs.map((pair, cellId) => ({ cellId, accepted: true, center: composePose(refCenter, { ...pair.reference, rotation: 0 }) })),
      pointPairs: vias.pairs.map((pair, cellId) => ({ cellId, support: 128,
        reference: { x: refOffset.x + pair.reference.x, y: refOffset.y + pair.reference.y },
        current: { x: curOffset.x + pair.source.x, y: curOffset.y + pair.source.y } })) } };
}
