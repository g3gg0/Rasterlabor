import { revalidateEdgeCell } from './pcb-edge-constraints.js';
import { registerPcbFftPair } from './pcb-fft-pair.js';
import { composePose, invertPose, optimizePcbComponents } from './pcb-realignment.js';
import { networkPairId } from './match-network.js';

const angle = value => Math.atan2(Math.sin(value), Math.cos(value));
const valid = pose => pose && [pose.x, pose.y, pose.rotation].every(Number.isFinite);
const transform = (pose, point) => composePose(pose, { ...point, rotation: 0 });

// Direct NCC is inexpensive in a 6-pixel neighbourhood and avoids the windowing
// bias of phase correlation when the remaining error is already subpixel-sized.
export function finePatchShift(first, second, width, height, { searchRadius = 6, ownerBounds = null } = {}) {
  const radius = Math.floor(searchRadius), margin = radius + 2, samples = [];
  let sum = 0, squared = 0, support = 0, owned = 0;
  for (let y = margin; y < height - margin; y += 2) for (let x = margin; x < width - margin; x += 2) {
    const i = y * width + x, value = first[i];
    samples.push({ i, value }); sum += value; squared += value * value;
    if (Math.abs(value - first[i - 1]) + Math.abs(value - first[i - width]) > 4) {
      support += 4;
      if (!ownerBounds || (x >= ownerBounds.minX && x < ownerBounds.maxX && y >= ownerBounds.minY && y < ownerBounds.maxY)) owned += 4;
    }
  }
  const n = samples.length, variance = squared - sum * sum / n;
  if (variance < n * 4 || owned < 8) return { accepted: false, reason: 'Unzureichende Struktur' };
  const scores = new Map(); let best = { score: -1, x: 0, y: 0 };
  const key = (x, y) => `${x}:${y}`;
  for (let y = -radius; y <= radius; y++) for (let x = -radius; x <= radius; x++) {
    let b = 0, bb = 0, ab = 0;
    for (const sample of samples) {
      const value = second[sample.i + y * width + x];
      b += value; bb += value * value; ab += sample.value * value;
    }
    const denominator = Math.sqrt(variance * Math.max(0, bb - b * b / n));
    const score = denominator > 0 ? (ab - sum * b / n) / denominator : -1;
    scores.set(key(x, y), score);
    if (score > best.score) best = { score, x, y };
  }
  let runnerUp = -1;
  for (let y = -radius; y <= radius; y++) for (let x = -radius; x <= radius; x++)
    if (Math.hypot(x - best.x, y - best.y) >= 2) runnerUp = Math.max(runnerUp, scores.get(key(x, y)));
  const boundary = Math.abs(best.x) === radius || Math.abs(best.y) === radius;
  const accepted = !boundary && best.score >= .9 && best.score - runnerUp >= .003;
  const subpixel = (a, b, c) => Number.isFinite(a + b + c) && a - 2 * b + c < -1e-8 ?
    Math.max(-.5, Math.min(.5, .5 * (a - c) / (a - 2 * b + c))) : 0;
  return { accepted, reason: accepted ? null : boundary ? 'Suchgrenze' : 'Mehrdeutige Korrelation',
    dx: best.x + subpixel(scores.get(key(best.x - 1, best.y)), best.score, scores.get(key(best.x + 1, best.y))),
    dy: best.y + subpixel(scores.get(key(best.x, best.y - 1)), best.score, scores.get(key(best.x, best.y + 1))),
    score: best.score, psr: Math.max(0, Math.min(30, (best.score - runnerUp) * 200)),
    supportPixels: support, ownedSupportPixels: owned };
}

// Every usable frame gets its own neighbours, including frames with no old anchors.
export function planFinePairs(frames, network, reach, neighbours = 3) {
  const pairs = new Map(), deleted = new Set(network.deletedPairs ?? []);
  const known = new Map((network.pairs ?? []).map(pair => [pair.id, pair.weight]));
  for (const frame of frames) {
    if (!valid(frame.pose)) continue;
    const candidates = frames.filter(other => other.frame !== frame.frame && valid(other.pose))
      .map(other => ({ other, id: networkPairId(frame.frame, other.frame),
        distance: Math.hypot(other.pose.x - frame.pose.x, other.pose.y - frame.pose.y) }))
      .filter(item => item.distance < reach && !deleted.has(item.id))
      .sort((a, b) => a.distance - b.distance || a.other.frame - b.other.frame);
    const anchored = candidates.filter(item => known.has(item.id)).sort((a, b) => known.get(b.id) - known.get(a.id))[0];
    const chosen = [...candidates.slice(0, neighbours), ...(anchored ? [anchored] : [])];
    for (const { other, id } of chosen) pairs.set(id, {
      reference: Math.min(frame.frame, other.frame), current: Math.max(frame.frame, other.frame), kind: 'fine' });
  }
  return [...pairs.values()].sort((a, b) => a.reference - b.reference || a.current - b.current);
}

function gray(image, x, y) {
  const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
  if (ix < 0 || iy < 0 || ix + 1 >= image.width || iy + 1 >= image.height) return null;
  const at = (xx, yy) => {
    const i = 4 * (yy * image.width + xx), d = image.data;
    return d[i + 3] === 255 ? .299 * d[i] + .587 * d[i + 1] + .114 * d[i + 2] : null;
  };
  const a = at(ix, iy), b = at(ix + 1, iy), c = at(ix, iy + 1), d = at(ix + 1, iy + 1);
  return [a, b, c, d].includes(null) ? null : (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

// Independent photometric verification at the measured cell correspondences.
export function cellNcc(first, second, cell, firstOffset, secondOffset, rotation, radius = 24) {
  let n = 0, a = 0, b = 0, aa = 0, bb = 0, ab = 0;
  const c = Math.cos(rotation), s = Math.sin(rotation);
  const step = Math.max(2, Math.ceil(radius / 12));
  for (let y = -radius; y <= radius; y += step) for (let x = -radius; x <= radius; x += step) {
    const p = gray(first, cell.reference.x - firstOffset[0] + x, cell.reference.y - firstOffset[1] + y);
    const q = gray(second, cell.current.x - secondOffset[0] + c * x - s * y,
      cell.current.y - secondOffset[1] + s * x + c * y);
    if (p === null || q === null) continue;
    n++; a += p; b += q; aa += p * p; bb += q * q; ab += p * q;
  }
  const va = aa - a * a / n, vb = bb - b * b / n;
  return n >= 100 && va > n * 4 && vb > n * 4 ? Math.max(-1, Math.min(1, (ab - a * b / n) / Math.sqrt(va * vb))) : 0;
}

export function measureFinePair(first, second, reference, current, options = {}) {
  const small = measureFineScale(first, second, reference, current, {...options, cellSize:64});
  if (small.accepted || options.singleScale) return small;
  // Larger context disambiguates a straight track or a small repeated marking.
  // The search radius and bidirectional acceptance checks remain unchanged.
  const large = measureFineScale(first, second, reference, current, {...options, cellSize:128});
  large.scaleAttempts = [64,128];
  if(large.accepted) return large;
  // Permit small local non-rigidity only with at least six spatial cells in each
  // direction. This cannot relax the one-pixel cycle or photometric thresholds.
  const context = measureFineScale(first, second, reference, current, {...options,cellSize:128,residualLimit:2});
  context.scaleAttempts = [64,128];
  if(context.accepted) return context;
  return [small,large,context].sort((a,b)=>(b.fft.inlierCells?.length??0)-(a.fft.inlierCells?.length??0))[0];
}

function measureFineScale(first, second, reference, current, { radius = 6, mask = null, cellSize = 64, residualLimit = .8 } = {}) {
  const settings = { cellSize, cellsPerAxis: 4, candidateGrid: 12,
    adaptiveCells: true, searchRadius: radius, minimumPsr: 7, residualLimit, mask, measureShift: finePatchShift };
  const fft = registerPcbFftPair(first, second, reference.pose, current.pose, reference.offset, current.offset, settings);
  const result = { reference: reference.frame, current: current.frame, kind: 'fine',
    referencePose: reference.pose, currentPose: current.pose, fft: { ...fft, method: 'Feinsuche' },
    accepted: false, score: 0, reason: fft.reason ?? null };
  if (!fft.accepted) {
    const observed = (fft.cells ?? []).map(cell => cell.score).filter(Number.isFinite);
    result.score = observed.length ? observed.reduce((sum, score) => sum + score, 0) / observed.length : 0;
    const failures = new Map();
    for (const cell of fft.cells ?? []) if (cell.reason && cell.reason !== 'Nicht ausgewaehlt')
      failures.set(cell.reason, (failures.get(cell.reason) ?? 0) + 1);
    result.cellFailures = Object.fromEntries(failures);
    // A circular valid area normally excludes many grid candidates. Report
    // the failure of actually sampled patches instead of hiding it behind that.
    const measuredFailures = [...failures].filter(([reason]) => reason !== 'Maske oder Rand');
    const dominant = (measuredFailures.length ? measuredFailures : [...failures]).sort((a,b)=>b[1]-a[1])[0];
    if (dominant) result.reason += ` (${dominant[1]} Zellen: ${dominant[0]})`;
    return result;
  }
  const scores = fft.pointPairs.map(cell => cellNcc(first, second, cell, reference.offset, current.offset,
    reference.pose.rotation - fft.pose.rotation));
  result.score = scores.reduce((sum, score) => sum + score, 0) / scores.length;
  result.worstCellScore = Math.min(...scores);
  const reverse = registerPcbFftPair(second, first, current.pose, reference.pose, current.offset, reference.offset, settings);
  if (!reverse.accepted) { result.reason = `Rueckpruefung: ${reverse.reason}`; return result; }
  const forwardRelative = composePose(invertPose(reference.pose), fft.pose);
  const reverseRelative = composePose(invertPose(current.pose), reverse.pose);
  const cycle = composePose(forwardRelative, reverseRelative);
  const lever = Math.hypot(first.width, first.height) / 2;
  result.reverseDistance = Math.hypot(cycle.x, cycle.y) + Math.abs(angle(cycle.rotation)) * lever;
  const movement = Math.hypot(fft.pose.x - current.pose.x, fft.pose.y - current.pose.y) +
    Math.abs(angle(fft.pose.rotation - current.pose.rotation)) * lever;
  const reverseScores = reverse.pointPairs.map(cell => cellNcc(second, first, cell, current.offset, reference.offset,
    current.pose.rotation - reverse.pose.rotation));
  const reverseScore = reverseScores.reduce((sum, score) => sum + score, 0) / reverseScores.length;
  const enoughCells = residualLimit <= .8 || (fft.inlierCells.length >= 6 && reverse.inlierCells.length >= 6);
  result.accepted = enoughCells && result.score >= .93 && reverseScore >= .93 && result.worstCellScore >= .8 &&
    result.reverseDistance <= 1 && movement <= radius * 1.5;
  result.reason = result.accepted ? null : !enoughCells ? 'Zu wenig verteilte Kontextzellen' : result.reverseDistance > 1 ? 'Rueckweg > 1 px' :
    movement > radius * 1.5 ? 'Korrektur ausserhalb Feinbereich' : 'Korrelation nicht eindeutig';
  result.forward = { accepted: result.accepted, pose: fft.pose, score: result.score, support: fft.uniqueSupportArea };
  result.backward = { accepted: result.accepted, pose: reverse.pose, score: reverseScore, support: reverse.uniqueSupportArea };
  return result;
}

// Recheck the actual saved correspondences, independently of their graph error.
// Flat or ambiguous patches retain only a weak pull; their points remain visible.
export function revalidateSavedCells(first, second, reference, current, cells) {
  const rotation=reference.pose.rotation-current.pose.rotation,c=Math.cos(rotation),s=Math.sin(rotation);
  return cells.map(cell=>{
    if(cell.normal)return revalidateEdgeCell(first,second,reference,current,cell);
    let best=null;
    for(const size of [64,128]) {
      const a=new Float64Array(size*size),b=new Float64Array(size*size);let valid=true;
      for(let y=0;y<size&&valid;y++)for(let x=0;x<size;x++){
        const dx=x-size/2,dy=y-size/2;
        const av=gray(first,cell.reference.x-reference.offset[0]+dx,cell.reference.y-reference.offset[1]+dy);
        const bv=gray(second,cell.current.x-current.offset[0]+c*dx-s*dy,cell.current.y-current.offset[1]+s*dx+c*dy);
        if(av===null||bv===null){valid=false;break;}
        a[y*size+x]=av;b[y*size+x]=bv;
      }
      if(!valid)continue;
      const measured=finePatchShift(a,b,size,size,{searchRadius:3});
      if(!best||(measured.score??-1)>(best.score??-1))best=measured;
      if(measured.accepted){best=measured;break;}
    }
    return {id:cell.id,score:best?.score??null,confidence:best?.accepted?1:.05,
      reason:best?.accepted?null:best?.reason??'Messpunkt derzeit nicht beobachtbar'};
  });
}

export function solveFineGraph(graph, { radius = 6, lever = 1000 } = {}) {
  const solved = optimizePcbComponents(graph, { iterations: 16, huber: 2, lever });
  const source = new Map(graph.nodes.map(node => [node.frame, node.pose]));
  let maximum = 0;
  for (const item of solved.corrections) {
    const old = source.get(item.frame);
    maximum = Math.max(maximum, Math.hypot(item.pose.x - old.x, item.pose.y - old.y) +
      Math.abs(angle(item.pose.rotation - old.rotation)) * lever);
  }
  // A common step size preserves coupled movement; never clip individual frames.
  const scale = maximum > radius ? radius / maximum : 1;
  const corrections = solved.corrections.map(item => {
    const old = source.get(item.frame);
    return { ...item, pose: { x: old.x + (item.pose.x - old.x) * scale,
      y: old.y + (item.pose.y - old.y) * scale,
      rotation: old.rotation + angle(item.pose.rotation - old.rotation) * scale } };
  });
  return { corrections, scale, components: solved.components };
}

export function fineFrameReport(frames, matches, edges = [], poses = null) {
  const rows = new Map(frames.map(frame => [frame.frame, { frame: frame.frame, attempts: 0, accepted: 0,
    score: null, ncc: null, worstPartner: null, reason: 'Noch nicht gemessen', residual: null }]));
  for (const match of matches) for (const [id, partner] of [[match.reference, match.current], [match.current, match.reference]]) {
    const row = rows.get(id); if (!row) continue;
    row.attempts++; if (match.accepted) row.accepted++;
    // Failed geometric checks stay below confirmed measurements, but retain
    // their measured correlation so weak failures can still be ranked.
    const score = match.accepted ? match.worstCellScore ?? match.score : Math.max(0, Math.min(1, match.score ?? 0)) * .75;
    if (row.score === null || score < row.score) {
      row.score = score; row.ncc = match.score ?? null; row.worstPartner = partner;
      row.reason = match.reason ?? `NCC ${(100 * score).toFixed(1)}%`;
    }
  }
  if (poses) for (const edge of edges) {
    const a = poses.get(edge.reference), b = poses.get(edge.current);
    if (!a || !b) continue;
    const expected = transform(a, edge.measurement);
    const actual = edge.currentPivot ? transform(b, edge.currentPivot) : b;
    const error = Math.hypot(expected.x - actual.x, expected.y - actual.y);
    for (const id of [edge.reference, edge.current]) {
      const row = rows.get(id); if (row) row.residual = Math.max(row.residual ?? 0, error);
    }
  }
  return [...rows.values()].sort((a, b) => (a.score ?? -1) - (b.score ?? -1) ||
    (b.residual ?? 0) - (a.residual ?? 0) || a.frame - b.frame);
}
