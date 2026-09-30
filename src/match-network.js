import { acceptedPcbConstraints, composePose, invertPose } from './pcb-realignment.js';

const pointValid = p => p && Number.isFinite(p.x) && Number.isFinite(p.y);
const poseValid = p => pointValid(p) && Number.isFinite(p.rotation);
const local = (pose, point) => composePose(invertPose(pose), { ...point, rotation: 0 });
export const networkPairId = (reference, current) => `${Math.min(reference, current)}:${Math.max(reference, current)}`;
const cellId = cell => [cell.reference.x, cell.reference.y, cell.current.x, cell.current.y]
  .map(value => Math.round(value / 8)).join(':') + (cell.normal ? ':normal' : '');
export function networkGeometryKey(calibration) {
  return JSON.stringify({ width: calibration?.field?.width, height: calibration?.field?.height,
    nx: calibration?.field?.nx, ny: calibration?.field?.ny,
    coefficients: Array.from(calibration?.field?.coefficients ?? []),
    outputWidth: calibration?.maps?.outputWidth, outputHeight: calibration?.maps?.outputHeight,
    origin: calibration?.maps?.origin });
}
export function emptyMatchNetwork(geometryKey = null) {
  return { schemaVersion: 1, geometryKey, revision: 0, pairs: [], deletedPairs: [], deletedCells: [] };
}
export function normalizeMatchNetwork(value, geometryKey = null) {
  if (value?.schemaVersion !== 1) return emptyMatchNetwork(geometryKey);
  const validFrames = pair => pair && Number.isSafeInteger(pair.reference) && Number.isSafeInteger(pair.current) &&
    pair.reference >= 0 && pair.current >= 0 && pair.reference !== pair.current;
  return { ...emptyMatchNetwork(geometryKey), ...value,
    pairs: (Array.isArray(value.pairs) ? value.pairs : []).filter(pair => validFrames(pair) &&
      poseValid(pair.measurement) && Number.isFinite(pair.weight) && pair.weight > 0).map(pair => ({...pair,
        id: networkPairId(pair.reference, pair.current),
        cells: (Array.isArray(pair.cells) ? pair.cells : []).filter(cell => cell && pointValid(cell.reference) && pointValid(cell.current) && (!cell.normal || pointValid(cell.normal) && Math.abs(Math.hypot(cell.normal.x,cell.normal.y)-1)<.001)).map(cell=>({...cell,id:cell.id??cellId(cell),quality:Number.isFinite(cell.quality)&&cell.quality>0?Math.min(1,cell.quality):1})) })),
    deletedPairs: (Array.isArray(value.deletedPairs) ? value.deletedPairs : []).filter(id => typeof id === 'string'),
    deletedCells: (Array.isArray(value.deletedCells) ? value.deletedCells : []).filter(cell =>
      cell && typeof cell.pairId === 'string' && pointValid(cell.reference) && pointValid(cell.current) && Number.isFinite(cell.radius)) };
}
function suppressed(network, pairId, cell) {
  return network.deletedCells.some(deleted => deleted.pairId === pairId &&
    Math.hypot(deleted.reference.x - cell.reference.x, deleted.reference.y - cell.reference.y) <= deleted.radius &&
    Math.hypot(deleted.current.x - cell.current.x, deleted.current.y - cell.current.y) <= deleted.radius);
}
function pairFromMatch(match, edge) {
  const flip = match.reference > match.current;
  let points = match.fft?.pointPairs ?? [];
  let approximate = false;
  if (!points.length && poseValid(match.referencePose) && poseValid(match.forward?.pose)) {
    // Old projects did not save the seed used to measure each cell. Keep their
    // fitted locations as an explicitly labelled approximation, never invent dx.
    const inliers = new Set(match.fft?.inlierCells ?? []);
    points = (match.fft?.cells ?? []).filter(cell => inliers.has(cell.cellId) && pointValid(cell.center)).map(cell => ({
      cellId: cell.cellId, reference: local(match.referencePose, cell.center),
      current: local(match.forward.pose, cell.center), psr: cell.psr, support: cell.ownedSupportPixels }));
    approximate = true;
  }
  const cells = points.filter(cell => pointValid(cell.reference) && pointValid(cell.current)).map(cell => {
    const item = { reference: flip ? cell.current : cell.reference, current: flip ? cell.reference : cell.current,
      ...(cell.normal ? {normal:flip?{
        x:Math.cos(match.referencePose.rotation-match.currentPose.rotation)*cell.normal.x-Math.sin(match.referencePose.rotation-match.currentPose.rotation)*cell.normal.y,
        y:Math.sin(match.referencePose.rotation-match.currentPose.rotation)*cell.normal.x+Math.cos(match.referencePose.rotation-match.currentPose.rotation)*cell.normal.y
      }:cell.normal}:{}),
      sourceCellId: cell.cellId, psr: Number.isFinite(cell.psr) ? cell.psr : null,
      support: Number.isFinite(cell.support) ? cell.support : null,
      quality: Number.isFinite(cell.psr) ? Math.max(.2, Math.min(1, cell.psr / 20)) : 1 };
    return { ...item, id: cellId(item) };
  });
  return { id: networkPairId(match.reference, match.current), reference: Math.min(match.reference, match.current),
    current: Math.max(match.reference, match.current), measurement: flip ? invertPose(edge.measurement) : edge.measurement,
    weight: edge.weight, rotationWeight: edge.rotationWeight ?? 1, score: edge.score,
    reverseDistance: edge.reverseDistance, cellSize: match.fft?.cellSize ?? 32,
    method: match.fft?.method ?? 'FFT-Zellen', approximate, hasCells: cells.length > 0,
    minimumCells: match.fft?.method === 'PCB-Vias' ? 1 :
      match.fft?.method === 'PCB-Via-Konstellation' && match.forward.score < .9 ? 5 : 3, cells };
}
export function mergeNetworkMatches(network, matches, parameters = {}) {
  const pairs = new Map(network.pairs.map(pair => [pair.id, pair]));
  let changed = false;
  for (const match of matches) {
    const pairId=networkPairId(match.reference,match.current), previous=pairs.get(pairId);
    if(previous&&match.anchorEvidence?.length){
      const evidence=new Map(match.anchorEvidence.map(item=>[item.id,item]));
      pairs.set(pairId,{...previous,cells:previous.cells.map(cell=>{
        const checked=evidence.get(cell.id);
        return checked?{...cell,confidence:Math.max(.05,Math.min(1,checked.confidence)),validationScore:checked.score,validationReason:checked.reason}:cell;
      })});changed=true;
    }
    const [edge] = acceptedPcbConstraints([match], parameters).accepted;
    if (!edge) continue;
    const incoming = pairFromMatch(match, edge);
    if (network.deletedPairs.includes(incoming.id)) continue;
    incoming.cells = incoming.cells.filter(cell => !suppressed(network, incoming.id, cell));
    const old = pairs.get(incoming.id);
    const cells = new Map((old?.approximate && !incoming.approximate ? [] : old?.cells ?? []).map(cell => [cell.id, cell]));
    for (const cell of incoming.cells) cells.set(cell.id, { ...cells.get(cell.id), ...cell });
    // Preserve precise measurements when migrating older approximate diagnostics.
    if (old && !old.approximate && incoming.approximate) continue;
    pairs.set(incoming.id, { ...incoming,
      weight: parameters.preserveStrength ? Math.max(old?.weight ?? 0, incoming.weight) : incoming.weight,
      cells: [...cells.values()] }); changed = true;
  }
  return changed ? { ...network, revision: network.revision + 1, pairs: [...pairs.values()] } : network;
}
export function migrateMatchNetwork(tracking, geometryKey) {
  let network = normalizeMatchNetwork(tracking?.matchNetwork, geometryKey);
  if (tracking?.matchNetwork?.schemaVersion === 1) return network;
  const state = tracking?.pcbRealignment;
  network = mergeNetworkMatches(network, [...(state?.pairDiagnostics ?? []), ...(state?.localDiagnostics ?? [])], state?.parameters);
  const pairs = new Map(network.pairs.map(pair => [pair.id, pair]));
  for (const edge of state?.constraints ?? []) {
    if (!poseValid(edge.measurement) || !Number.isSafeInteger(edge.reference) || !Number.isSafeInteger(edge.current)) continue;
    const id = networkPairId(edge.reference, edge.current);
    if (pairs.has(id)) continue;
    pairs.set(id, { ...edge, id, reference: Math.min(edge.reference, edge.current), current: Math.max(edge.reference, edge.current),
      measurement: edge.reference > edge.current ? invertPose(edge.measurement) : edge.measurement,
      method: 'Gespeicherte Bildpose', cells: [], hasCells: false, rotationWeight: edge.rotationWeight ?? 1 });
  }
  const draft = tracking?.pcbRealignmentDraft;
  network = { ...network, pairs: [...pairs.values()] };
  return mergeNetworkMatches(network, [...(draft?.matches ?? []), ...(draft?.bridgeMatches ?? []),
    ...(draft?.skipMatches ?? []), ...(draft?.localMatches ?? [])], draft?.parameters);
}
export function deleteNetworkMatch(network, pairId, selectedCellId = null) {
  const pair = network.pairs.find(item => item.id === pairId);
  if (!pair) return network;
  if (selectedCellId === null) return { ...network, revision: network.revision + 1,
    pairs: network.pairs.filter(item => item.id !== pairId), deletedPairs: [...new Set([...network.deletedPairs, pairId])] };
  const cell = pair.cells.find(item => item.id === selectedCellId);
  if (!cell) return network;
  return { ...network, revision: network.revision + 1,
    pairs: network.pairs.map(item => item.id === pairId ? { ...item, cells: item.cells.filter(c => c.id !== selectedCellId) } : item),
    deletedCells: [...network.deletedCells, { pairId, reference: cell.reference, current: cell.current,
      radius: Math.max(8, (pair.cellSize ?? 32) / 2) }] };
}
function edgeFromPair(pair, kind) {
  if (!pair.hasCells) return { ...pair, kind, verified: true };
  if (pair.cells.length < pair.minimumCells) return null;
  if(pair.cells.some(cell=>cell.normal)) return {reference:pair.reference,current:pair.current,kind,verified:true,
    measurement:pair.measurement,weight:pair.weight,rotationWeight:0,pointCells:pair.cells,minimumCells:pair.minimumCells,
    normalOnly:pair.cells.every(cell=>cell.normal)};
  const total = pair.cells.reduce((sum, cell) => sum + (cell.quality * (cell.confidence ?? 1)), 0);
  const mean = key => ({ x: pair.cells.reduce((sum, cell) => sum + (cell.quality * (cell.confidence ?? 1)) * cell[key].x, 0) / total,
    y: pair.cells.reduce((sum, cell) => sum + (cell.quality * (cell.confidence ?? 1)) * cell[key].y, 0) / total });
  const source = mean('current'), target = mean('reference');
  if (!pair.rotationWeight) return { reference: pair.reference, current: pair.current, kind, verified: true,
    measurement: { ...target, rotation: 0 }, currentPivot: source, weight: pair.weight, rotationWeight: 0 };
  let dot = 0, cross = 0, spread = 0;
  for (const cell of pair.cells) {
    const x = cell.current.x - source.x, y = cell.current.y - source.y;
    const qx = cell.reference.x - target.x, qy = cell.reference.y - target.y;
    dot += (cell.quality * (cell.confidence ?? 1)) * (x * qx + y * qy); cross += (cell.quality * (cell.confidence ?? 1)) * (x * qy - y * qx);
    spread += (cell.quality * (cell.confidence ?? 1)) * (qx * qx + qy * qy);
  }
  if (spread / total < (pair.cellSize ?? 32) ** 2 / 16 && pair.minimumCells !== 1) return null;
  const rotation = spread / total < (pair.cellSize ?? 32) ** 2 / 16 ? pair.measurement.rotation : Math.atan2(cross, dot), c = Math.cos(rotation), s = Math.sin(rotation);
  return { reference: pair.reference, current: pair.current, kind, verified: true,
    measurement: { x: target.x - c * source.x + s * source.y, y: target.y - s * source.x - c * source.y, rotation },
    weight: pair.weight * total / Math.max(3, pair.minimumCells, pair.cells.reduce((sum,cell)=>sum+cell.quality,0)), rotationWeight: 1 };
}
export function networkEdges(network, frames = null, kind = 'match-network') {
  const excluded = new Set(network.deletedPairs);
  return network.pairs.filter(pair => !excluded.has(pair.id) && (!frames ||
    (frames.has(pair.reference) && frames.has(pair.current)))).map(pair => edgeFromPair(pair, kind)).filter(Boolean);
}
export function applyNetworkToGraph(graph, network) {
  const frames = new Set(graph.nodes.map(node => node.frame));
  const known = new Set([...network.pairs.map(pair => pair.id), ...network.deletedPairs]);
  const edges = [...graph.edges.filter(edge => !known.has(networkPairId(edge.reference, edge.current))),
    ...networkEdges(network, frames, 'local-refit')];
  return { ...graph, network, edges, localEdges: edges.filter(edge => edge.kind === 'local-refit').length };
}
export function projectNetworkCells(pair, poses) {
  const reference = poses.get(pair.reference), current = poses.get(pair.current);
  if (!poseValid(reference) || !poseValid(current)) return [];
  return pair.cells.map(cell => ({ ...cell, referenceWorld: composePose(reference, {...cell.reference, rotation: 0}),
    currentWorld: composePose(current, {...cell.current, rotation: 0}) }));
}
