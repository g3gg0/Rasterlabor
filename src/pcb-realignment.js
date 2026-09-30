import { pairStiffness } from './pair-stiffness.js';
import { frameGeometry } from './path-support.js';
import { optimizePoseGraph } from './pose-graph-refit.js';
import { maskIncludes } from './patch-mask.js';

const wrap = angle => Math.atan2(Math.sin(angle), Math.cos(angle));
const validPose = pose => pose && [pose.x, pose.y, pose.rotation].every(Number.isFinite);
const rotate = (x, y, angle) => ({ x: Math.cos(angle) * x - Math.sin(angle) * y,
  y: Math.sin(angle) * x + Math.cos(angle) * y });
export function composePose(first, second) {
  const offset = rotate(second.x, second.y, first.rotation);
  return { x: first.x + offset.x, y: first.y + offset.y, rotation: wrap(first.rotation + second.rotation) };
}
export function invertPose(pose) {
  const offset = rotate(-pose.x, -pose.y, -pose.rotation);
  return { ...offset, rotation: wrap(-pose.rotation) };
}
export const relativePose = (current, reference) => composePose(invertPose(reference), current);
function expPose({ x, y, rotation }) {
  const angle = rotation, a = Math.abs(angle) < 1e-8 ? 1 - angle * angle / 6 : Math.sin(angle) / angle;
  const b = Math.abs(angle) < 1e-8 ? angle / 2 : (1 - Math.cos(angle)) / angle;
  return { x: a * x - b * y, y: b * x + a * y, rotation: angle };
}
function logPose(pose) {
  const angle = wrap(pose.rotation), a = Math.abs(angle) < 1e-8 ? 1 - angle * angle / 6 : Math.sin(angle) / angle;
  const b = Math.abs(angle) < 1e-8 ? angle / 2 : (1 - Math.cos(angle)) / angle;
  const denominator = a * a + b * b;
  return { x: (a * pose.x + b * pose.y) / denominator, y: (-b * pose.x + a * pose.y) / denominator,
    rotation: angle };
}
export function interpolateCorrection(first, second, alpha) {
  const relative = logPose(composePose(invertPose(first), second));
  return composePose(first, expPose({ x: alpha * relative.x, y: alpha * relative.y,
    rotation: alpha * relative.rotation }));
}

export function selectPcbKeyframes(path, { width, height, spacing = 0.3, turnDegrees = 25, revisitGap = 256 } = {}) {
  if (!(width > 0 && height > 0) || !(spacing >= 0.02 && spacing <= 2) ||
      !(turnDegrees >= 1 && turnDegrees <= 180) || !Number.isInteger(revisitGap) || revisitGap < 1) {
    throw new Error('Ungueltige Keyframe-Parameter.');
  }
  const entries = path.filter(entry => validPose(entry.pose)).sort((first, second) => first.frame - second.frame);
  if (!entries.length) return [];
  const selected = [entries[0]];
  const distanceLimit = Math.min(width, height) * spacing;
  for (let index = 1; index < entries.length - 1; index++) {
    const entry = entries[index], previous = entries[index - 1], next = entries[index + 1], last = selected.at(-1);
    const distance = Math.hypot(entry.pose.x - last.pose.x, entry.pose.y - last.pose.y);
    const angle = Math.abs(wrap(entry.pose.rotation - last.pose.rotation)) * 180 / Math.PI;
    const incoming = Math.atan2(entry.pose.y - previous.pose.y, entry.pose.x - previous.pose.x);
    const outgoing = Math.atan2(next.pose.y - entry.pose.y, next.pose.x - entry.pose.x);
    const turn = Math.hypot(entry.pose.x - previous.pose.x, entry.pose.y - previous.pose.y) > 1 &&
      Math.hypot(next.pose.x - entry.pose.x, next.pose.y - entry.pose.y) > 1 &&
      Math.abs(wrap(outgoing - incoming)) * 180 / Math.PI >= turnDegrees;
    const revisit = entry.frame - last.frame >= revisitGap;
    if (distance >= distanceLimit || angle >= turnDegrees || turn || revisit) selected.push(entry);
  }
  if (selected.at(-1).frame !== entries.at(-1).frame) selected.push(entries.at(-1));
  return selected;
}

function bounds(geometry) {
  const xs = geometry.corners.map(point => point.x), ys = geometry.corners.map(point => point.y);
  return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
}
export function polygonsOverlap(first, second) {
  for (const polygon of [first, second]) for (let edge = 0; edge < polygon.length; edge++) {
    const a = polygon[edge], b = polygon[(edge + 1) % polygon.length];
    const axis = { x: a.y - b.y, y: b.x - a.x };
    const project = points => points.map(point => point.x * axis.x + point.y * axis.y);
    const left = project(first), right = project(second);
    if (Math.max(...left) <= Math.min(...right) || Math.max(...right) <= Math.min(...left)) return false;
  }
  return true;
}

export function planPcbPairs(keyframes, field, maps, { maxNeighbors = 4, maxPairs = 512,
  mask = null } = {}) {
  if (!Number.isInteger(maxNeighbors) || maxNeighbors < 1 || maxNeighbors > 16 ||
      !Number.isInteger(maxPairs) || maxPairs < 1 || maxPairs > 100000) throw new Error('Ungueltiges Paarbudget.');
  const geometries = keyframes.map(entry => frameGeometry(entry, field, maps)).filter(Boolean);
  const items = geometries.map(geometry => ({ geometry, bounds: bounds(geometry) }));
  const gridSize = Math.max(maps.outputWidth, maps.outputHeight);
  const buckets = new Map(), temporal = [], spatial = [], seen = new Set();
  const overlapScore = (first, second) => {
    const minX = Math.max(first.bounds.minX, second.bounds.minX);
    const maxX = Math.min(first.bounds.maxX, second.bounds.maxX);
    const minY = Math.max(first.bounds.minY, second.bounds.minY);
    const maxY = Math.min(first.bounds.maxY, second.bounds.maxY);
    if (!(maxX > minX && maxY > minY)) return 0;
    const allowed = mask ? (x, y) => maskIncludes(mask, x, y) : null;
    let support = 0;
    for (let row = 0; row < 16; row++) for (let column = 0; column < 16; column++) {
      const point = { x: minX + (column + 0.5) * (maxX - minX) / 16,
        y: minY + (row + 0.5) * (maxY - minY) / 16 };
      if (first.geometry.supports(point, allowed) && second.geometry.supports(point, allowed)) support++;
    }
    return support;
  };
  const add = (first, second, kind, overlapSupport) => {
    if (first === second) return;
    const reference = Math.min(first, second), current = Math.max(first, second), key = `${reference}:${current}`;
    if (seen.has(key)) return;
    seen.add(key); (kind === 'temporal' ? temporal : spatial).push({
      reference, current, kind, group: key, overlapSupport });
  };
  for (const [index, item] of items.entries()) {
    const { geometry, bounds: box } = item;
    if (index && polygonsOverlap(geometry.corners, items[index - 1].geometry.corners)) {
      const support = overlapScore(item, items[index - 1]);
      if (support >= 4) add(items[index - 1].geometry.entry.frame, geometry.entry.frame, 'temporal', support);
    }
    const candidates = new Map();
    for (let y = Math.floor(box.minY / gridSize); y <= Math.floor(box.maxY / gridSize); y++)
      for (let x = Math.floor(box.minX / gridSize); x <= Math.floor(box.maxX / gridSize); x++) {
        for (const other of buckets.get(`${x}:${y}`) ?? []) candidates.set(other.geometry.entry.frame, other);
      }
    const ranked = [...candidates.values()].filter(other =>
      geometry.entry.frame !== other.geometry.entry.frame && polygonsOverlap(geometry.corners, other.geometry.corners))
      .map(other => ({ other, support: overlapScore(item, other) }))
      .filter(candidate => candidate.support >= 4)
      .sort((first, second) => second.support - first.support ||
        first.other.geometry.entry.frame - second.other.geometry.entry.frame);
    for (const { other, support } of ranked.slice(0, maxNeighbors))
      add(other.geometry.entry.frame, geometry.entry.frame, 'spatial', support);
    for (let y = Math.floor(box.minY / gridSize); y <= Math.floor(box.maxY / gridSize); y++)
      for (let x = Math.floor(box.minX / gridSize); x <= Math.floor(box.maxX / gridSize); x++) {
        const key = `${x}:${y}`;
        if (!buckets.has(key)) buckets.set(key, []);
        const bucket = buckets.get(key);
        bucket.push(item);
        if (bucket.length > 32) bucket.splice(1, bucket.length - 32);
      }
  }
  if (temporal.length >= maxPairs) {
    if (temporal.length === maxPairs) return temporal;
    return Array.from({ length: maxPairs }, (_, index) =>
      temporal[Math.floor((index + 0.5) * temporal.length / maxPairs)]);
  }
  const remaining = maxPairs - temporal.length;
  if (spatial.length <= remaining) return [...temporal, ...spatial];
  return [...temporal, ...Array.from({ length: remaining }, (_, index) =>
    spatial[Math.floor((index + 0.5) * spatial.length / remaining)])];
}

export function planPcbTemporalBridges(path, pairs, matches, accepted, budget = 512) {
  if (!Number.isInteger(budget) || budget < 0 || budget > 100000)
    throw new Error('Ungueltiges Brueckenbudget.');
  const ordered = path.filter(entry => validPose(entry.pose)).sort((a, b) => a.frame - b.frame);
  const acceptedKeys = new Set(accepted.map(edge => `${edge.reference}:${edge.current}`));
  const attempted = new Set(matches.map(match => `${match.reference}:${match.current}`));
  const chains = [];
  for (const pair of pairs) {
    if (pair.kind !== 'temporal' || !attempted.has(`${pair.reference}:${pair.current}`) ||
        acceptedKeys.has(`${pair.reference}:${pair.current}`)) continue;
    const between = ordered.filter(entry => entry.frame >= pair.reference && entry.frame <= pair.current);
    if (between.length < 3) continue;
    const chain = [];
    for (let index = 1; index < between.length; index++) chain.push({
      reference: between[index - 1].frame, current: between[index].frame,
      kind: 'temporal', group: `bridge:${pair.reference}:${pair.current}` });
    chains.push(chain);
  }
  chains.sort((a, b) => a.length - b.length || a[0].reference - b[0].reference);
  const selected = []; let remaining = budget;
  for (const chain of chains) if (chain.length <= remaining) {
    selected.push(...chain); remaining -= chain.length;
  }
  return selected.sort((a, b) => a.reference - b.reference || a.current - b.current);
}

export function planPcbTemporalSkips(bridgePairs, bridgeMatches, accepted, budget = 512) {
  if (!Number.isInteger(budget) || budget < 0 || budget > 100000)
    throw new Error('Ungueltiges Brueckenbudget.');
  const acceptedKeys = new Set(accepted.map(edge => `${edge.reference}:${edge.current}`));
  const attempted = new Set(bridgeMatches.map(match => `${match.reference}:${match.current}`));
  const groups = new Map(), output = [], seen = new Set();
  for (const pair of bridgePairs) {
    if (!groups.has(pair.group)) groups.set(pair.group, []);
    groups.get(pair.group).push(pair);
  }
  const add = (first, second, group) => {
    if (!first || !second || first.reference >= second.current) return;
    const key = `${first.reference}:${second.current}`;
    if (seen.has(key) || attempted.has(key)) return;
    seen.add(key);
    output.push({ reference: first.reference, current: second.current,
      kind: 'temporal', group: `bridge-skip:${group}` });
  };
  for (const [group, chain] of groups) for (let index = 0; index < chain.length; index++) {
    const pair = chain[index], key = `${pair.reference}:${pair.current}`;
    if (!attempted.has(key) || acceptedKeys.has(key)) continue;
    if (index > 0) add(chain[index - 1], pair, group);
    if (index + 1 < chain.length) add(pair, chain[index + 1], group);
  }
  return output.slice(0, budget);
}

export function interpolatePcbPoses(path, optimizedKeyframes) {
  const ordered = path.filter(entry => validPose(entry.pose)).sort((first, second) => first.frame - second.frame);
  const anchors = ordered.filter(entry => optimizedKeyframes.has(entry.frame));
  if (!anchors.length) return new Map();
  const correction = entry => composePose(optimizedKeyframes.get(entry.frame), invertPose(entry.pose));
  const output = new Map();
  const distanceAt = new Map([[ordered[0].frame, 0]]);
  let travelled = 0;
  for (let index = 1; index < ordered.length; index++) {
    travelled += Math.hypot(ordered[index].pose.x - ordered[index - 1].pose.x,
      ordered[index].pose.y - ordered[index - 1].pose.y);
    distanceAt.set(ordered[index].frame, travelled);
  }
  let anchorIndex = 0;
  for (const entry of ordered) {
    while (anchorIndex + 1 < anchors.length && entry.frame > anchors[anchorIndex + 1].frame) anchorIndex++;
    const first = anchors[anchorIndex], second = anchors[Math.min(anchors.length - 1, anchorIndex + 1)];
    if (entry.frame === first.frame) { output.set(entry.frame, optimizedKeyframes.get(entry.frame)); continue; }
    if (entry.frame === second.frame) { output.set(entry.frame, optimizedKeyframes.get(entry.frame)); continue; }
    let alpha = 0;
    if (first !== second && entry.frame > first.frame && entry.frame < second.frame) {
      const total = distanceAt.get(second.frame) - distanceAt.get(first.frame);
      alpha = total > 1e-8 ? (distanceAt.get(entry.frame) - distanceAt.get(first.frame)) / total :
        (entry.frame - first.frame) / (second.frame - first.frame);
    }
    const transform = first === second || entry.frame < first.frame ? correction(first) :
      interpolateCorrection(correction(first), correction(second), alpha);
    output.set(entry.frame, composePose(transform, entry.pose));
  }
  return output;
}

export function selectPcbIntermediateFrames(path, keyframeFrames, budget) {
  if (!Number.isInteger(budget) || budget < 0 || budget > 100000)
    throw new Error('Ungueltiges Budget fuer Zwischenframes.');
  const ordered = path.filter(entry => validPose(entry.pose)).sort((first, second) => first.frame - second.frame);
  const candidates = ordered.filter(entry => !keyframeFrames.has(entry.frame));
  if (budget >= candidates.length) return candidates;
  if (!budget) return [];
  const indexByFrame = new Map(ordered.map((entry, index) => [entry.frame, index]));
  const anomaly = entry => {
    const index = indexByFrame.get(entry.frame);
    if (!index || index + 1 >= ordered.length) return 0;
    const before = ordered[index - 1], after = ordered[index + 1];
    const fraction = (entry.frame - before.frame) / Math.max(1, after.frame - before.frame);
    const x = before.pose.x + fraction * (after.pose.x - before.pose.x);
    const y = before.pose.y + fraction * (after.pose.y - before.pose.y);
    const angle = wrap(after.pose.rotation - before.pose.rotation);
    return Math.hypot(entry.pose.x - x, entry.pose.y - y,
      wrap(entry.pose.rotation - before.pose.rotation - fraction * angle) * 100);
  };
  const selected = new Map();
  const anomalyBudget = Math.min(budget, Math.ceil(budget / 2));
  const suspicious = [...candidates].sort((first, second) => anomaly(second) - anomaly(first) || first.frame - second.frame);
  for (const entry of suspicious.slice(0, anomalyBudget)) selected.set(entry.frame, entry);
  for (let index = 0; index < candidates.length && selected.size < budget; index++) {
    const position = Math.floor((index + 0.5) * candidates.length / Math.max(1, budget - anomalyBudget));
    const entry = candidates[Math.min(candidates.length - 1, position)];
    selected.set(entry.frame, entry);
  }
  for (const entry of candidates) { if (selected.size >= budget) break; selected.set(entry.frame, entry); }
  return [...selected.values()].sort((first, second) => first.frame - second.frame);
}

export function pcbRunFingerprint(path, calibration, video, mask) {
  const sampled = values => {
    if (!values?.length) return [];
    const samples = [];
    for (let index = 0; index < 257; index++)
      samples.push(values[Math.floor(index * (values.length - 1) / 256)]);
    return samples;
  };
  const maps = calibration?.maps;
  const payload = JSON.stringify({
    video: [video?.name, video?.width, video?.height, video?.frameCount],
    field: [calibration?.field?.width, calibration?.field?.height],
    maps: [maps?.outputWidth, maps?.outputHeight, maps?.origin, sampled(maps?.valid),
      sampled(maps?.inverseX), sampled(maps?.inverseY)],
    mask: mask ? [mask.sourceWidth, mask.sourceHeight, mask.cellSize, [...mask.data]] : null,
    path: path.map(entry => [entry.frame, entry.mode, entry.success, entry.pose?.x,
      entry.pose?.y, entry.pose?.rotation])
  });
  let hash = 2166136261;
  for (let index = 0; index < payload.length; index++) {
    hash ^= payload.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return `${payload.length}:${(hash >>> 0).toString(16)}`;
}

export function acceptedPcbConstraints(matches, { cycleLimit = 2, minimumScore = 0.9,
  minimumSupport = 128 } = {}) {
  const accepted = [], rejected = [];
  for (const match of matches) {
    const forward = match.forward, backward = match.backward;
    const evidence = match.landmarkRecovery?.geometricEvidence;
    // Via geometry has a separate acceptance model; keep the actual patch NCC
    // in diagnostics instead of pretending it is a near-perfect image match.
    const geometric = match.fft?.method === 'PCB-Via-Konstellation' && evidence &&
      evidence.viaCount >= 5 && evidence.residual <= 3 && evidence.cycle <= 1.5 &&
      evidence.patchScore >= .6 && evidence.traceScore >= .6 && evidence.traceSupport >= 64 &&
      evidence.ambiguityMargin >= .08;
    const effectiveCycleLimit = match.fftGpuVerification?.accepted &&
      match.fftGpuVerification.fftReverse?.accepted ?
      Math.max(cycleLimit, match.fftGpuVerification.cycleLimit ?? 0) : cycleLimit;
    const reason = !forward?.accepted || !backward?.accepted ? 'Vorwaerts-/Rueckwaertsmatch' :
      !Number.isFinite(match.reverseDistance) || match.reverseDistance > effectiveCycleLimit ? 'Rueckweg' :
      !Number.isFinite(forward.score) || (!geometric && forward.score < minimumScore) ? 'Korrelation' :
      !Number.isFinite(forward.support) || forward.support < minimumSupport ? 'Strukturflaeche' :
      !validPose(forward.pose) || !validPose(match.referencePose) ? 'Pose' : null;
    if (reason) { rejected.push({ reference: match.reference, current: match.current, reason }); continue; }
    accepted.push({ reference: match.reference, current: match.current,
      measurement: relativePose(forward.pose, match.referencePose), kind: match.kind === 'temporal' ? 'pcb-temporal' : 'pcb-spatial',
      ...pairStiffness(match), verified: true,
      score: forward.score, reverseDistance: match.reverseDistance });
  }
  return { accepted, rejected };
}

export function acceptPcbGpuOnly(pair, candidate, { minimumScore = 0.9995,
  maximumCycle = 1, maximumCorrection = 32, minimumSupport = 1024 } = {}) {
  const { measured, backward, reverseDistance, correctedPose } = candidate;
  const gap = pair.current - pair.reference;
  if (pair.kind !== 'temporal' || gap <= 0 || gap > 128)
    return { accepted: false, reason: 'Kein pruefbarer zeitlicher Nachbar' };
  if (!validPose(pair.currentPose) || !validPose(correctedPose) ||
      !validPose(measured?.pose) || !validPose(backward?.pose) ||
      !Number.isFinite(pair.lever) ||
      !Number.isFinite(measured.score) || !Number.isFinite(backward.score))
    return { accepted: false, reason: 'GPU-Korrelation nicht eindeutig genug' };
  if (Math.min(measured.support ?? 0, backward.support ?? 0) < minimumSupport)
    return { accepted: false, reason: 'Zu kleine GPU-Strukturflaeche' };
  const original = pair.currentPose;
  const correction = Math.hypot(correctedPose.x - original.x, correctedPose.y - original.y) +
    Math.abs(wrap(correctedPose.rotation - original.rotation)) * pair.lever;
  if (gap <= 16 && measured.score >= minimumScore && backward.score >= minimumScore &&
      Number.isFinite(reverseDistance) && reverseDistance <= maximumCycle &&
      correction <= maximumCorrection) return { accepted: true, reason: null, method: 'near-duplicate' };
  const regions = candidate.regions;
  if (measured.score >= 0.99 && backward.score >= 0.99 &&
      Number.isFinite(reverseDistance) && reverseDistance <= 2 && correction <= 64 &&
      Array.isArray(regions) && regions.length === 4 &&
      regions.every(region => Number.isFinite(region.score) && region.score >= 0.99 &&
        region.support >= 512 && Number.isFinite(region.distance) && region.distance <= 6))
    return { accepted: true, reason: null, method: 'independent-regions' };
  return { accepted: false, reason: 'GPU-Teilbereiche oder Rueckweg widersprechen' };
}

export function optimizePcbComponents(graph, { iterations = 8, huber = 20, lever = 1000 } = {}) {
  const nodes = new Map(graph.nodes.map(node => [node.frame, node]));
  const neighbors = new Map(graph.nodes.map(node => [node.frame, []]));
  for (const edge of graph.edges) {
    if (!nodes.has(edge.reference) || !nodes.has(edge.current)) continue;
    neighbors.get(edge.reference).push({ frame: edge.current, edge });
    neighbors.get(edge.current).push({ frame: edge.reference, edge });
  }
  const visited = new Set(), components = [], corrections = [];
  let beforeSquared = 0, afterSquared = 0, edgeCount = 0;
  for (const frame of [...nodes.keys()].sort((first, second) => first - second)) {
    if (visited.has(frame)) continue;
    const queue = [frame], frames = new Set([frame]), componentEdges = new Set(); visited.add(frame);
    for (let cursor = 0; cursor < queue.length; cursor++) for (const link of neighbors.get(queue[cursor])) {
      componentEdges.add(link.edge);
      if (visited.has(link.frame)) continue;
      visited.add(link.frame); frames.add(link.frame); queue.push(link.frame);
    }
    const part = { nodes: [...frames].map(id => nodes.get(id)),
      edges: [...componentEdges] };
    if (part.nodes.length < 2 || !part.edges.length) {
      components.push({ anchor: frame, nodes: part.nodes.length, edges: 0, status: 'unconnected' });
      continue;
    }
    const result = optimizePoseGraph(part, { seedFrames: [frame], iterations, huber, lever });
    if (!Number.isFinite(result.afterRms) || result.afterRms > result.beforeRms + 1e-6) {
      components.push({ anchor: frame, nodes: part.nodes.length, edges: part.edges.length, status: 'rejected',
        beforeRms: result.beforeRms, afterRms: result.afterRms });
      continue;
    }
    corrections.push(...result.corrections);
    beforeSquared += result.beforeRms ** 2 * result.edges;
    afterSquared += result.afterRms ** 2 * result.edges;
    edgeCount += result.edges;
    components.push({ anchor: frame, nodes: result.nodes, edges: result.edges, status: 'optimized',
      beforeRms: result.beforeRms, afterRms: result.afterRms });
  }
  return { corrections, components, nodes: graph.nodes.length, edges: graph.edges.length,
    beforeRms: edgeCount ? Math.sqrt(beforeSquared / edgeCount) : null,
    afterRms: edgeCount ? Math.sqrt(afterSquared / edgeCount) : null };
}
