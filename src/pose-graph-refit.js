const wrap = angle => Math.atan2(Math.sin(angle), Math.cos(angle));
const rotate = (x, y, angle) => ({ x: Math.cos(angle) * x - Math.sin(angle) * y, y: Math.sin(angle) * x + Math.cos(angle) * y });

function relative(current, reference) {
  return { ...rotate(current.x - reference.x, current.y - reference.y, -reference.rotation),
    rotation: wrap(current.rotation - reference.rotation) };
}

function compose(reference, measurement) {
  const translation = rotate(measurement.x, measurement.y, reference.rotation);
  return { x: reference.x + translation.x, y: reference.y + translation.y,
    rotation: wrap(reference.rotation + measurement.rotation) };
}

function validPose(pose) {
  return pose && [pose.x, pose.y, pose.rotation].every(Number.isFinite);
}

function sharpness(entry) {
  return Number.isFinite(entry.sharpness?.score) ? entry.sharpness.score : -Infinity;
}

function relevance(entry) {
  return Number.isFinite(entry.localSelectionDistance) ? entry.localSelectionDistance : Infinity;
}

export function localRefitGroups(entries, temporalGap = 256) {
  const ordered = [...new Map(entries.filter(entry => validPose(entry.pose)).map(entry => [entry.frame, entry])).values()]
    .sort((first, second) => first.frame - second.frame);
  const groups = [];
  for (const entry of ordered) {
    if (!groups.length || entry.frame - groups.at(-1).at(-1).frame > temporalGap) groups.push([]);
    groups.at(-1).push(entry);
  }
  return groups;
}

export function applyPoseCorrection(pose, from, to) {
  const local = rotate(pose.x - from.x, pose.y - from.y, -from.rotation);
  const world = rotate(local.x, local.y, to.rotation);
  return { x: to.x + world.x, y: to.y + world.y,
    rotation: wrap(pose.rotation + to.rotation - from.rotation) };
}

export function planLocalRefitPairs(entries, { temporalGap = 256, perGroup = 8, maxPairs = perGroup * perGroup,
  maxFrames = 32, preferredFrames = [], matches = [] } = {}) {
  const preferred = new Set(preferredFrames);
  const groups = localRefitGroups(entries, temporalGap);
  const representatives = groups.map(group => {
    const count = Math.min(perGroup, group.length);
    const strata = Array.from({ length: count }, (_, index) => {
      const start = Math.floor(index * group.length / count);
      const end = Math.floor((index + 1) * group.length / count);
      return [...group.slice(start, end)].sort((first, second) =>
        Number(preferred.has(second.frame)) - Number(preferred.has(first.frame)) || relevance(first) - relevance(second) ||
        sharpness(second) - sharpness(first));
    });
    return Array.from({ length: Math.max(...strata.map(stratum => stratum.length)) }, (_, rank) =>
      strata.flatMap(stratum => stratum[rank] ? [stratum[rank]] : [])).flat();
  });
  const pairs = [];
  const seen = new Set(matches.map(match => `${match.reference}:${match.current}`));
  const frames = new Set();
  const append = pair => {
    const key = `${pair.reference}:${pair.current}`;
    if (pair.reference === pair.current || seen.has(key) || pairs.length >= maxPairs) return;
    if (frames.size + Number(!frames.has(pair.reference)) + Number(!frames.has(pair.current)) > maxFrames) return;
    frames.add(pair.reference); frames.add(pair.current);
    seen.add(key); pairs.push(pair);
  };
  const promising = matches.filter(match => validPose(match.forward?.pose) && match.forward.score >= 0.9)
    .sort((first, second) => second.forward.score - first.forward.score);
  const neighbors = (group, frame) => {
    const index = group.findIndex(entry => entry.frame === frame);
    return index < 0 ? [] : group.slice(Math.max(0, index - 2), index + 3);
  };
  for (const match of promising) {
    const [first, second] = match.group.split(':').map(Number);
    if (!groups[first] || !groups[second]) continue;
    const nearby = [];
    for (const reference of neighbors(groups[first], match.reference)) for (const current of neighbors(groups[second], match.current)) {
      if (first === second && reference.frame >= current.frame) continue;
      nearby.push({ reference: reference.frame, current: current.frame, group: match.group });
    }
    nearby.sort((left, right) =>
      Number(right.reference !== match.reference && right.current !== match.current) -
      Number(left.reference !== match.reference && left.current !== match.current));
    for (const pair of nearby) {
      if (pairs.length >= Math.floor(maxPairs / 2)) break;
      append(pair);
    }
    if (pairs.length >= Math.floor(maxPairs / 2)) break;
  }
  function* visitPairs(first, second) {
    const references = representatives[first], currents = representatives[second];
    const referenceBlocks = Math.ceil(references.length / perGroup), currentBlocks = Math.ceil(currents.length / perGroup);
    for (let diagonal = 0; diagonal < referenceBlocks + currentBlocks - 1; diagonal++) {
      for (let referenceBlock = 0; referenceBlock < referenceBlocks; referenceBlock++) {
        const currentBlock = diagonal - referenceBlock;
        if (currentBlock < 0 || currentBlock >= currentBlocks) continue;
        for (const reference of references.slice(referenceBlock * perGroup, (referenceBlock + 1) * perGroup)) {
          for (const current of currents.slice(currentBlock * perGroup, (currentBlock + 1) * perGroup)) {
            if (first === second && reference.frame >= current.frame) continue;
            yield { reference: reference.frame, current: current.frame, group: `${first}:${second}` };
          }
        }
      }
    }
  }
  const queues = [];
  for (let first = 0; first < groups.length; first++) for (let second = first + (groups.length > 1 ? 1 : 0); second < groups.length; second++) {
    queues.push(visitPairs(first, second));
  }
  while (queues.length && pairs.length < maxPairs) {
    for (let index = 0; index < queues.length && pairs.length < maxPairs;) {
      let next = queues[index].next();
      while (!next.done && seen.has(`${next.value.reference}:${next.value.current}`)) next = queues[index].next();
      if (next.done) queues.splice(index, 1);
      else { append(next.value); index++; }
    }
  }
  return pairs;
}

export async function searchLocalRefit(entries, register, { baseGraph, graphOptions = {}, preferredFrames = [],
  maxPairs = 512, batchSize = 64 } = {}) {
  const matches = [];
  let graph = addLocalRefitEdges(baseGraph, matches, graphOptions);
  let rounds = 0;
  while (matches.length < maxPairs) {
    const pairs = planLocalRefitPairs(entries, { matches, preferredFrames, maxPairs: Math.min(batchSize, maxPairs - matches.length) });
    if (!pairs.length) break;
    const batch = await register(pairs, { round: ++rounds, attempted: matches.length, limit: maxPairs });
    if (batch.length !== pairs.length || batch.some((match, index) =>
      match.reference !== pairs[index].reference || match.current !== pairs[index].current)) {
      throw new Error('Unvollstaendige Ergebnisse der lokalen Registrierung.');
    }
    matches.push(...batch);
    graph = addLocalRefitEdges(baseGraph, matches, graphOptions);
    if (graph.localEdges) break;
  }
  return { graph, matches, rounds, limited: !graph.localEdges && matches.length >= maxPairs };
}

export function addLocalRefitEdges(graph, matches, { conditionalLimit = 7.5, cycleLimit = conditionalLimit,
  consensusLimit = conditionalLimit, lever = 1000 } = {}) {
  const usable = match => match?.accepted || match?.conditionallyAccepted;
  const candidates = matches.filter(match => usable(match.forward) && usable(match.backward) &&
    Number.isFinite(match.reverseDistance) && match.reverseDistance <= cycleLimit &&
    validPose(match.currentPose) && validPose(match.referencePose) && validPose(match.forward.pose));
  const accepted = [];
  for (const group of new Set(matches.map(match => match.group))) {
    const members = candidates.filter(match => match.group === group);
    const correction = match => ({ x: match.forward.pose.x - match.currentPose.x, y: match.forward.pose.y - match.currentPose.y,
      rotation: wrap(match.forward.pose.rotation - match.currentPose.rotation) });
    const distance = (first, second) => Math.hypot(first.x - second.x, first.y - second.y,
      wrap(first.rotation - second.rotation) * lever);
    const neighborhoods = members.map(match => {
      const center = correction(match);
      return members.filter(other => distance(center, correction(other)) <= consensusLimit);
    }).sort((first, second) => second.length - first.length);
    const inliers = neighborhoods[0] ?? [];
    const independent = new Set(inliers.map(match => match.current)).size >= 2 &&
      new Set(inliers.map(match => match.reference)).size >= 2;
    if (independent && inliers.length > members.length / 2) {
      accepted.push(...inliers);
      continue;
    }
    const strict = members.filter(match => match.forward.accepted && match.backward.accepted && match.reverseDistance <= 1.5 &&
      match.forward.score >= 0.98 && match.backward.score >= 0.98 && match.forward.margin >= 0.002 &&
      match.backward.margin >= 0.002 && match.forward.support >= 128 && match.backward.support >= 128)
      .sort((first, second) => second.forward.score + second.backward.score - first.forward.score - first.backward.score);
    if (strict.length) accepted.push(strict[0]);
  }
  const existing = new Set(graph.edges.map(edge => `${edge.reference}:${edge.current}:${edge.kind}`));
  const edges = [...graph.edges];
  for (const match of accepted) {
    const key = `${match.reference}:${match.current}:local-refit`;
    if (existing.has(key)) continue;
    existing.add(key);
    edges.push({ reference: match.reference, current: match.current,
      measurement: relative(match.forward.pose, match.referencePose), kind: 'local-refit',
      weight: match.reverseDistance <= 1.5 ? 10 : 6, score: match.forward.score ?? null,
      reverseDistance: match.reverseDistance });
  }
  return { ...graph, edges, localEdges: edges.length - graph.edges.length, candidates: candidates.length };
}

function spatialCandidates(entry, conditionalLimit) {
  const matches = (entry.context?.matches ?? []).filter(match => match.kind === 'spatial' && validPose(match.pose) && validPose(match.referencePose));
  const confirmed = Boolean(entry.context?.confirmation);
  const inliers = new Set(entry.context?.inliers ?? []);
  return matches.filter(match => match.accepted || (confirmed && inliers.has(match.frame) && match.backward?.accepted &&
    Number.isFinite(match.reverseDistance) && match.reverseDistance <= conditionalLimit));
}

export function buildPoseGraph(path, { conditionalLimit = 7.5 } = {}) {
  const nodes = path.filter(entry => validPose(entry.pose)).map(entry => ({ frame: entry.frame, pose: { ...entry.pose } }));
  const nodeFrames = new Set(nodes.map(node => node.frame));
  const edges = [];
  const add = (entry, match, kind, weight) => {
    if (!nodeFrames.has(match.frame) || !validPose(match.pose) || !validPose(match.referencePose)) return;
    edges.push({ reference: match.frame, current: entry.frame, measurement: relative(match.pose, match.referencePose),
      kind, weight, score: match.score ?? null, reverseDistance: match.reverseDistance ?? null });
  };
  for (const entry of path) {
    if (!nodeFrames.has(entry.frame)) continue;
    if (entry.incrementalMatch?.accepted) add(entry, entry.incrementalMatch, 'incremental', 1);
    const spatial = spatialCandidates(entry, conditionalLimit);
    for (const match of spatial) {
      const strict = match.accepted && (!Number.isFinite(match.reverseDistance) || match.reverseDistance <= 1.5);
      add(entry, match, strict ? 'spatial-strict' : 'spatial-conditional', strict ? 8 : 4);
    }
  }
  return { nodes, edges };
}

function connectedComponent(nodes, edges, seeds) {
  const adjacency = new Map(nodes.map(node => [node.frame, []]));
  for (const edge of edges) {
    adjacency.get(edge.reference)?.push(edge.current);
    adjacency.get(edge.current)?.push(edge.reference);
  }
  const pending = seeds.filter(frame => adjacency.has(frame)); const reached = new Set(pending);
  while (pending.length) for (const frame of adjacency.get(pending.shift()) ?? []) if (!reached.has(frame)) { reached.add(frame); pending.push(frame); }
  return reached;
}

function multiplyLaplacian(values, edges, weights, anchor, damping) {
  const output = new Float64Array(values.length);
  for (let index = 0; index < values.length; index++) output[index] = damping * values[index];
  for (let index = 0; index < edges.length; index++) {
    const edge = edges[index]; const weight = weights[index];
    const difference = values[edge.currentIndex] - values[edge.referenceIndex];
    output[edge.referenceIndex] -= weight * difference;
    output[edge.currentIndex] += weight * difference;
  }
  output[anchor] += 1e9 * values[anchor];
  return output;
}

function conjugateGradient(edges, weights, right, anchor, iterations = 400) {
  const values = new Float64Array(right.length); const damping = 1e-6;
  const residual = Float64Array.from(right); residual[anchor] = 0;
  const direction = Float64Array.from(residual);
  let squared = residual.reduce((sum, value) => sum + value * value, 0);
  for (let iteration = 0; iteration < iterations && squared > 1e-12; iteration++) {
    const product = multiplyLaplacian(direction, edges, weights, anchor, damping);
    const denominator = direction.reduce((sum, value, index) => sum + value * product[index], 0);
    if (!(denominator > 1e-20)) break;
    const step = squared / denominator;
    for (let index = 0; index < values.length; index++) { values[index] += step * direction[index]; residual[index] -= step * product[index]; }
    residual[anchor] = 0;
    const nextSquared = residual.reduce((sum, value) => sum + value * value, 0);
    const beta = nextSquared / squared;
    for (let index = 0; index < direction.length; index++) direction[index] = residual[index] + beta * direction[index];
    direction[anchor] = 0; squared = nextSquared;
  }
  return values;
}

export function optimizePoseGraph(graph, { seedFrames = [], iterations = 8, huber = 20 } = {}) {
  if (!graph.nodes.length) throw new Error('Pose-Graph enthaelt keine Posen.');
  const requested = seedFrames.length ? seedFrames : [graph.nodes[0].frame];
  const component = connectedComponent(graph.nodes, graph.edges, requested);
  const nodes = graph.nodes.filter(node => component.has(node.frame));
  const indexByFrame = new Map(nodes.map((node, index) => [node.frame, index]));
  const edges = graph.edges.filter(edge => component.has(edge.reference) && component.has(edge.current))
    .map(edge => ({ ...edge, referenceIndex: indexByFrame.get(edge.reference), currentIndex: indexByFrame.get(edge.current) }));
  if (nodes.length < 2 || !edges.length) throw new Error('Auswahl ist nicht mit einem messbaren Pose-Netz verbunden.');
  const poses = nodes.map(node => ({ ...node.pose })); const initial = nodes.map(node => ({ ...node.pose }));
  const anchor = 0; const lever = 1000;
  const initialMagnitudes = edges.map(edge => {
    const target = compose(poses[edge.referenceIndex], edge.measurement); const current = poses[edge.currentIndex];
    return Math.hypot(target.x - current.x, target.y - current.y, wrap(target.rotation - current.rotation) * lever);
  });
  let beforeRms = 0; let afterRms = 0;
  for (let outer = 0; outer < iterations; outer++) {
    const residuals = edges.map(edge => {
      const target = compose(poses[edge.referenceIndex], edge.measurement); const current = poses[edge.currentIndex];
      return { x: target.x - current.x, y: target.y - current.y, rotation: wrap(target.rotation - current.rotation) };
    });
    const magnitudes = residuals.map(value => Math.hypot(value.x, value.y, value.rotation * lever));
    if (outer === 0) beforeRms = Math.sqrt(magnitudes.reduce((sum, value) => sum + value ** 2, 0) / magnitudes.length);
    const weights = edges.map((edge, index) => {
      const normalized = edge.kind === 'local-refit' ? 0 : Math.max(magnitudes[index], initialMagnitudes[index]) / huber;
      return edge.weight / (1 + normalized ** 2) ** 2;
    });
    for (const key of ['x', 'y', 'rotation']) {
      const right = new Float64Array(nodes.length);
      for (let index = 0; index < edges.length; index++) {
        const edge = edges[index]; const residual = residuals[index][key]; const weight = weights[index];
        right[edge.referenceIndex] -= weight * residual; right[edge.currentIndex] += weight * residual;
      }
      const correction = conjugateGradient(edges, weights, right, anchor);
      for (let index = 0; index < poses.length; index++) poses[index][key] = key === 'rotation' ? wrap(poses[index][key] + correction[index]) : poses[index][key] + correction[index];
    }
  }
  const finalResiduals = edges.map(edge => {
    const target = compose(poses[edge.referenceIndex], edge.measurement); const current = poses[edge.currentIndex];
    return Math.hypot(target.x - current.x, target.y - current.y, wrap(target.rotation - current.rotation) * lever);
  });
  afterRms = Math.sqrt(finalResiduals.reduce((sum, value) => sum + value ** 2, 0) / finalResiduals.length);
  const localIndexes = edges.map((edge, index) => edge.kind === 'local-refit' ? index : -1).filter(index => index >= 0);
  const rmsAt = (values, indexes) => indexes.length ?
    Math.sqrt(indexes.reduce((sum, index) => sum + values[index] ** 2, 0) / indexes.length) : 0;
  const corrections = nodes.map((node, index) => ({ frame: node.frame, pose: poses[index],
    dx: poses[index].x - initial[index].x, dy: poses[index].y - initial[index].y,
    rotation: wrap(poses[index].rotation - initial[index].rotation) }));
  return { corrections, nodes: nodes.length, edges: edges.length,
    spatialEdges: edges.filter(edge => edge.kind.startsWith('spatial') || edge.kind === 'local-refit').length,
    localEdges: localIndexes.length, localBeforeRms: rmsAt(initialMagnitudes, localIndexes),
    localAfterRms: rmsAt(finalResiduals, localIndexes), beforeRms, afterRms, iterations };
}
