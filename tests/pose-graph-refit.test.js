import test from 'node:test';
import assert from 'node:assert/strict';
import { addLocalRefitEdges, applyPoseCorrection, buildPoseGraph, interpolateRefitPoses, localRefitGroups, optimizePoseGraph, planFocusedRefitPairs, planLocalRefitPairs, planRefitBridgePairs, searchLocalRefit } from '../src/pose-graph-refit.js';

test('bridge planning gives every unaligned frame a new nearby anchor pair before repeating frames', () => {
  const entries = [0, 1, 2, 3, 4].map(frame => ({ frame,
    pose: { x: frame * 10, y: 0, rotation: 0 } }));
  const attempted = [{ reference: 1, current: 2 }];
  const pairs = planRefitBridgePairs(entries, [0, 1], attempted, 3);
  assert.equal(pairs.length, 3);
  assert.deepEqual(new Set(pairs.map(pair => pair.current)), new Set([2, 3, 4]));
  assert.ok(pairs.every(pair => !attempted.some(match => match.reference === pair.reference && match.current === pair.current)));
});

test('bridge planning prefers the same temporal visit when saved positions are far apart', () => {
  const entries = [{ frame: 1, pose: { x: 0, y: 0, rotation: 0 } },
    { frame: 2, pose: { x: 500, y: 0, rotation: 0 } },
    { frame: 1000, pose: { x: 490, y: 0, rotation: 0 } }];
  const [pair] = planRefitBridgePairs(entries, [1, 1000], [], 1);
  assert.deepEqual([pair.reference, pair.current], [1, 2]);
});

test('bridge recovery reserves a bounded budget for another visit despite many local failures', () => {
  const entries = [...Array.from({length:100}, (_, frame)=>({frame,pose:{x:frame,y:0,rotation:0}})),
    ...Array.from({length:20}, (_, index)=>({frame:1000+index,pose:{x:1000+index,y:0,rotation:0}}))];
  const attempted = entries.slice(1).map(entry=>({reference:0,current:entry.frame,fft:{accepted:false}}));
  const pairs = planRefitBridgePairs(entries,[0],attempted,16);
  const cross = pairs.filter(pair=>pair.current>=1000);
  assert.equal(pairs.length,16);
  assert.equal(cross.length,8);
  assert.ok(cross[0].current < 1003 && cross.at(-1).current > 1017);
  assert.equal(new Set(pairs.map(pair=>`${pair.reference}:${pair.current}`)).size,pairs.length);
  const retried = [...attempted,...pairs.map(pair=>({...pair,featureRecovery:{accepted:false}}))];
  const again = planRefitBridgePairs(entries,[0],retried,16);
  assert.ok(again.every(pair=>!pairs.some(previous=>previous.reference===pair.reference&&previous.current===pair.current)));
});

test('bridge planning retries a failed FFT pair before trying a new anchor', () => {
  const entries = [0, 1, 2].map(frame => ({ frame,
    pose: { x: frame * 10, y: 0, rotation: 0 } }));
  const attempted = [{ reference: 1, current: 2, fft: { accepted: false } }];
  const [pair] = planRefitBridgePairs(entries, [0, 1], attempted, 1);
  assert.deepEqual([pair.reference, pair.current], [1, 2]);
  attempted[0].featureRecovery = { accepted: false };
  const [next] = planRefitBridgePairs(entries, [0, 1], attempted, 1);
  assert.deepEqual([next.reference, next.current], [0, 2]);
  attempted.unshift({ reference: 1, current: 2, fft: { accepted: false } });
  const [afterRetry] = planRefitBridgePairs(entries, [0, 1], attempted, 1);
  assert.deepEqual([afterRetry.reference, afterRetry.current], [0, 2]);
});

test('a plausible FFT rejected by the pair check still gets one recovery attempt', () => {
  const entries=[0,1,2].map(frame=>({frame,pose:{x:frame*10,y:0,rotation:0}}));
  const failed={reference:1,current:2,fft:{accepted:true},pcbVerified:false};
  const [pair]=planRefitBridgePairs(entries,[0,1],[failed],1);
  assert.deepEqual([pair.reference,pair.current],[1,2]);
  const [next]=planRefitBridgePairs(entries,[0,1],[failed,{...failed,landmarkRecovery:{accepted:false}}],1);
  assert.deepEqual([next.reference,next.current],[0,2]);
});

const pose = x => ({ x, y: 0, rotation: 0 });
const incremental = (frame, referenceX, currentX) => ({ frame, accepted: true, referencePose: pose(referenceX), pose: pose(currentX), score: 0.99 });

test('passive frames interpolate corrections by frame distance and follow endpoints', () => {
  const nodes = [0, 10, 15, 30, 40].map(frame => ({ frame, pose: pose(frame * 10) }));
  const corrected = nodes.map(node => pose(node.pose.x + (node.frame === 10 ? 20 : 60)));
  const passive = interpolateRefitPoses(nodes, corrected, [10, 30]);
  assert.equal(passive.get(0).x, 20);
  assert.equal(passive.get(15).x, 180);
  assert.equal(passive.get(40).x, 460);
  assert.equal(passive.has(10), false);
});

test('passive rotations preserve rigid pivots and interpolate across the angle wrap', () => {
  const nodes = [0, 1, 2].map(frame => ({ frame, pose: pose(frame * 10) }));
  const transform = { x: 50, y: 100, rotation: Math.PI / 2 };
  const rigid = nodes.map(node => applyPoseCorrection(node.pose, pose(0), transform));
  const passive = interpolateRefitPoses(nodes, rigid, [0, 2]).get(1);
  assert.ok(Math.abs(passive.x - 50) < 1e-9);
  assert.ok(Math.abs(passive.y - 110) < 1e-9);
  assert.ok(Math.abs(passive.rotation - Math.PI / 2) < 1e-9);
  const wrapped = interpolateRefitPoses(nodes,
    [{ ...pose(0), rotation: 179 * Math.PI / 180 }, pose(10),
      { ...pose(20), rotation: -179 * Math.PI / 180 }], [0, 2]).get(1);
  assert.ok(Math.abs(Math.abs(wrapped.rotation) - Math.PI) < 1e-9);
});

test('passive interpolation stays within a visit and uses spatial proximity for unconnected visits', () => {
  const nodes = [{ frame: 0, pose: pose(0) }, { frame: 1, pose: pose(10) },
    { frame: 1000, pose: pose(100) }, { frame: 1001, pose: pose(110) },
    { frame: 2000, pose: pose(101) }];
  const passive = interpolateRefitPoses(nodes, [pose(20), pose(10), pose(60), pose(110), pose(101)], [0, 1000]);
  assert.equal(passive.get(1).x, 30);
  assert.equal(passive.get(1001).x, 70);
  assert.equal(passive.get(2000).x, 61);
});

test('optimization includes every passive frame and keeps improvements from separate measured components', () => {
  const graph = { nodes: [0, 10, 20, 30, 40, 50].map(frame => ({ frame, pose: pose(frame * 10) })), edges: [
    { reference: 0, current: 20, measurement: pose(180), kind: 'local-refit', weight: 10 },
    { reference: 30, current: 40, measurement: pose(90), kind: 'local-refit', weight: 10 }
  ] };
  const result = optimizePoseGraph(graph, { seedFrames: [0], includePassive: true });
  const corrections = new Map(result.corrections.map(item => [item.frame, item]));
  assert.equal(corrections.size, 6);
  assert.deepEqual(result.passiveFrames, [10, 50]);
  assert.ok(Math.abs(corrections.get(10).pose.x - 90) < 0.01);
  assert.ok(Math.abs(corrections.get(30).pose.x - 280) < 0.01);
  assert.ok(Math.abs(corrections.get(40).pose.x - 370) < 0.01);
  assert.ok(Math.abs(corrections.get(50).pose.x - 470) < 0.01);
  assert.ok(result.localAfterRms < 0.01);
});

test('pose graph spreads a loop correction through the connected path', () => {
  const path = Array.from({ length: 6 }, (_, frame) => ({ frame, pose: pose(frame * 11),
    incrementalMatch: frame ? incremental(frame - 1, (frame - 1) * 10, frame * 10) : null, context: { matches: [] } }));
  path[5].context.matches = [0, 1].map(frame => ({ frame, kind: 'spatial', accepted: true, referencePose: pose(frame * 10),
    pose: pose(50 + frame * 0.05), score: 0.98, reverseDistance: 1 }));
  const graph = buildPoseGraph(path);
  const result = optimizePoseGraph(graph, { seedFrames: [5], iterations: 10 });
  assert.equal(result.nodes, 6); assert.equal(result.spatialEdges, 2);
  assert.ok(result.afterRms < result.beforeRms * 0.1, JSON.stringify(result));
  const corrections = new Map(result.corrections.map(item => [item.frame, item]));
  assert.equal(corrections.get(0).dx, 0);
  assert.ok(corrections.get(3).dx < -1 && corrections.get(3).dx > -5);
  assert.ok(corrections.get(5).dx < -4);
});

test('conditional spatial edges need a consistent majority', () => {
  const path = [0, 1, 2, 3].map(frame => ({ frame, pose: pose(frame * 10),
    incrementalMatch: frame ? incremental(frame - 1, (frame - 1) * 10, frame * 10) : null, context: { matches: [] } }));
  path[3].context = { confirmation: 'independent-reference-groups', inliers: [0, 1], matches: [
    { frame: 0, kind: 'spatial', accepted: false, referencePose: pose(0), pose: pose(28), score: 0.97, reverseDistance: 5, backward: { accepted: true } },
    { frame: 1, kind: 'spatial', accepted: false, referencePose: pose(10), pose: pose(28.2), score: 0.97, reverseDistance: 5, backward: { accepted: true } },
    { frame: 2, kind: 'spatial', accepted: false, referencePose: pose(20), pose: pose(80), score: 0.97, reverseDistance: 5, backward: { accepted: true } }
  ] };
  const graph = buildPoseGraph(path);
  assert.equal(graph.edges.filter(edge => edge.kind === 'spatial-conditional').length, 2);
});

test('an unconfirmed conditional pair is not added to the graph', () => {
  const path = [{ frame: 0, pose: pose(0), context: { matches: [] } },
    { frame: 1, pose: pose(12), incrementalMatch: incremental(0, 0, 10), context: { matches: [
      { frame: 0, kind: 'spatial', accepted: false, referencePose: pose(0), pose: pose(10), reverseDistance: 5, backward: { accepted: true } }
    ] } }];
  assert.equal(buildPoseGraph(path).edges.filter(edge => edge.kind.startsWith('spatial')).length, 0);
});

test('robust weighting limits a grossly incorrect loop edge', () => {
  const graph = { nodes: [0, 1, 2].map(frame => ({ frame, pose: pose(frame * 10) })), edges: [
    { reference: 0, current: 1, measurement: pose(10), kind: 'incremental', weight: 1 },
    { reference: 1, current: 2, measurement: pose(10), kind: 'incremental', weight: 1 },
    { reference: 0, current: 2, measurement: pose(100), kind: 'spatial-strict', weight: 8 }
  ] };
  const result = optimizePoseGraph(graph, { seedFrames: [1], iterations: 10, huber: 10 });
  const end = result.corrections.find(item => item.frame === 2).pose.x;
  assert.ok(end < 25, `outlier moved final pose to ${end}`);
});

test('disconnected frames outside the selected network remain absent', () => {
  const graph = { nodes: [{ frame: 0, pose: pose(0) }, { frame: 1, pose: pose(12) }, { frame: 9, pose: pose(90) }],
    edges: [{ reference: 0, current: 1, measurement: { x: 10, y: 0, rotation: 0 }, kind: 'incremental', weight: 1 }] };
  const result = optimizePoseGraph(graph, { seedFrames: [1] });
  assert.deepEqual(result.corrections.map(item => item.frame), [0, 1]);
});

test('local pair planning compares sharp representatives across separate visits', () => {
  const entries = [0, 1, 2, 1000, 1001, 1002].map(frame => ({ frame, pose: pose(frame), sharpness: { score: frame % 1000 } }));
  const pairs = planLocalRefitPairs(entries, { perGroup: 2 });
  assert.equal(pairs.length, 4);
  assert.ok(pairs.every(pair => pair.reference < 1000 && pair.current >= 1000));
});

test('local refit groups split temporal visits and rigid corrections preserve relative poses', () => {
  const entries = [2, 3, 400, 401].map(frame => ({ frame, pose: pose(frame) }));
  assert.deepEqual(localRefitGroups(entries).map(group => group.map(entry => entry.frame)), [[2, 3], [400, 401]]);
  const corrected = applyPoseCorrection({ x: 12, y: 20, rotation: 0.3 }, { x: 10, y: 20, rotation: 0 },
    { x: 100, y: 50, rotation: Math.PI / 2 });
  assert.ok(Math.abs(corrected.x - 100) < 1e-9);
  assert.ok(Math.abs(corrected.y - 52) < 1e-9);
  assert.ok(Math.abs(corrected.rotation - (0.3 + Math.PI / 2)) < 1e-9);
});

test('focused refit tests all small-group pairs and bounds large-group pairs while covering every frame', () => {
  const small = [0, 1, 2, 1000, 1001].map(frame => ({ frame, pose: pose(frame) }));
  assert.equal(planFocusedRefitPairs(small).length, 10);
  const large = [0, 1000, 2000].flatMap(start => Array.from({ length: 30 }, (_, index) =>
    ({ frame: start + index, pose: pose(start + index), sharpness: { score: index } })));
  const pairs = planFocusedRefitPairs(large, [1005]);
  assert.ok(pairs.length < large.length * 3);
  assert.equal(new Set(pairs.flatMap(pair => [pair.reference, pair.current])).size, large.length);
  assert.ok(pairs.some(pair => pair.reference === 1005 || pair.current === 1005));
  assert.ok(pairs.some(pair => pair.group === '0:2'));
  assert.equal(new Set(pairs.map(pair => `${pair.reference}:${pair.current}`)).size, pairs.length);
});

test('focused refit keeps each independently verified cell match and excludes rejected pairs', () => {
  const entries = [0, 1, 2, 3].map(frame => ({ frame, pose: pose(frame * 10) }));
  const matches = planFocusedRefitPairs(entries).map((pair, index) => ({ ...pair,
    currentPose: pose(pair.current * 10), referencePose: pose(pair.reference * 10),
    forward: { accepted: true, pose: pose(pair.current * 10 + 1), score: 0.99, support: 1024 },
    backward: { accepted: true }, reverseDistance: 1, pcbVerified: index !== 2 }));
  const graph = addLocalRefitEdges({ nodes: entries, edges: [] }, matches, { acceptVerifiedPairs: true });
  assert.equal(graph.localEdges, matches.length - 1);
  assert.ok(!graph.edges.some(edge => edge.reference === matches[2].reference && edge.current === matches[2].current));
});

test('local pair planning uses up to 64 pairs from two visits by default', () => {
  const entries = [...Array(8).keys(), ...Array.from({ length: 8 }, (_, index) => 1000 + index)]
    .map(frame => ({ frame, pose: pose(frame), sharpness: { score: frame } }));
  const pairs = planLocalRefitPairs(entries);
  assert.equal(pairs.length, 64);
  assert.equal(new Set(pairs.map(pair => pair.reference)).size, 8);
  assert.equal(new Set(pairs.map(pair => pair.current)).size, 8);
});

test('local pair planning distributes representatives across each visit', () => {
  const entries = [...Array.from({ length: 80 }, (_, frame) => frame),
    ...Array.from({ length: 80 }, (_, index) => 1000 + index)]
    .map(frame => ({ frame, pose: pose(frame), sharpness: { score: -(frame % 1000) } }));
  const pairs = planLocalRefitPairs(entries);
  const references = [...new Set(pairs.map(pair => pair.reference))];
  const currents = [...new Set(pairs.map(pair => pair.current))];
  assert.equal(references.length, 8); assert.equal(currents.length, 8);
  assert.ok(Math.max(...references) >= 70, JSON.stringify(references));
  assert.ok(Math.max(...currents) >= 1070, JSON.stringify(currents));
});

test('local pair planning prioritizes a selected frame within its temporal stratum', () => {
  const entries = [...Array.from({ length: 16 }, (_, frame) => frame),
    ...Array.from({ length: 16 }, (_, index) => 1000 + index)]
    .map(frame => ({ frame, pose: pose(frame), sharpness: { score: frame === 1006 ? 99 : frame === 1007 ? 100 : 1 } }));
  const pairs = planLocalRefitPairs(entries, { preferredFrames: [1006] });
  assert.ok(pairs.some(pair => pair.current === 1006), JSON.stringify(pairs));
  assert.ok(!pairs.some(pair => pair.current === 1007), JSON.stringify(pairs));
});

test('local pair planning prefers the representative nearest the selected region', () => {
  const entries = [...Array.from({ length: 16 }, (_, frame) => frame),
    ...Array.from({ length: 16 }, (_, index) => 1000 + index)]
    .map(frame => ({ frame, pose: pose(frame), localSelectionDistance: frame === 1006 ? 2 : 20,
      sharpness: { score: frame === 1007 ? 100 : 1 } }));
  const pairs = planLocalRefitPairs(entries);
  assert.ok(pairs.some(pair => pair.current === 1006), JSON.stringify(pairs));
  assert.ok(!pairs.some(pair => pair.current === 1007), JSON.stringify(pairs));
});

test('adaptive planning exhausts candidates without repeating failed pairs', () => {
  const entries = [0, 1, 2, 3, 1000, 1001, 1002, 1003].map(frame => ({ frame, pose: pose(frame) }));
  const matches = [];
  for (let round = 0; round < 4; round++) {
    const pairs = planLocalRefitPairs(entries, { perGroup: 2, matches });
    assert.equal(pairs.length, 4);
    matches.push(...pairs.map(pair => ({ ...pair, forward: { accepted: false } })));
  }
  assert.equal(new Set(matches.map(match => `${match.reference}:${match.current}`)).size, 16);
  assert.equal(planLocalRefitPairs(entries, { perGroup: 2, matches }).length, 0);
});

test('adaptive planning confirms promising matches with new frames on both sides', () => {
  const entries = [...Array.from({ length: 16 }, (_, frame) => frame),
    ...Array.from({ length: 16 }, (_, index) => 1000 + index)].map(frame => ({ frame, pose: pose(frame) }));
  const matches = planLocalRefitPairs(entries).map(pair => ({ ...pair, forward: { accepted: false } }));
  const anchor = matches.find(match => match.reference === 6 && match.current === 1006);
  anchor.forward = { accepted: true, score: 0.96, pose: pose(1004) };
  const pairs = planLocalRefitPairs(entries, { matches });
  assert.ok(pairs[0].reference !== anchor.reference && pairs[0].current !== anchor.current);
  assert.ok(Math.abs(pairs[0].reference - anchor.reference) <= 2);
  assert.ok(Math.abs(pairs[0].current - anchor.current) <= 2);
  assert.ok(pairs.every(pair => !matches.some(match => pair.reference === match.reference && pair.current === match.current)));
  assert.equal(pairs.length, 64);
});

test('adaptive planning gives every visit pair a share of the budget', () => {
  const entries = [0, 1000, 2000].flatMap(start => Array.from({ length: 10 }, (_, index) =>
    ({ frame: start + index, pose: pose(start + index) })));
  const pairs = planLocalRefitPairs(entries, { maxPairs: 6 });
  assert.deepEqual(pairs.map(pair => pair.group), ['0:1', '0:2', '1:2', '0:1', '0:2', '1:2']);
});

test('adaptive search combines confirmations across rounds and stops at consensus', async () => {
  const entries = [0, 1000].flatMap(start => Array.from({ length: 16 }, (_, index) =>
    ({ frame: start + index, pose: pose(start + index) })));
  let calls = 0;
  const result = await searchLocalRefit(entries, async (pairs, progress) => {
    calls++;
    assert.equal(progress.round, calls);
    assert.equal(progress.attempted, (calls - 1) * 64);
    return pairs.map((pair, index) => ({ ...pair, currentPose: pose(pair.current), referencePose: pose(pair.reference),
      forward: { accepted: index === 0, score: index === 0 ? 0.96 : 0.1, pose: pose(pair.current - 3) },
      backward: index === 0 ? { accepted: true, score: 0.96 } : null, reverseDistance: index === 0 ? 1 : null }));
  }, { baseGraph: { nodes: entries, edges: [] } });
  assert.equal(calls, 2);
  assert.equal(result.graph.localEdges, 2);
  assert.equal(result.matches.length, 128);
  assert.equal(result.limited, false);
});

test('adaptive search moves beyond the original representatives with bounded work and memory', async () => {
  const entries = [0, 1000].flatMap(start => Array.from({ length: 80 }, (_, index) =>
    ({ frame: start + index, pose: pose(start + index) })));
  const result = await searchLocalRefit(entries, async pairs => {
    assert.ok(new Set(pairs.flatMap(pair => [pair.reference, pair.current])).size <= 32);
    return pairs.map(pair => ({ ...pair, forward: { accepted: false } }));
  }, { baseGraph: { nodes: entries, edges: [] } });
  assert.equal(result.matches.length, 512);
  assert.equal(result.rounds, 8);
  assert.equal(result.limited, true);
  assert.equal(new Set(result.matches.map(match => `${match.reference}:${match.current}`)).size, 512);
  assert.ok(new Set(result.matches.flatMap(match => [match.reference, match.current])).size > 16);
});

test('adaptive search reports exhausted candidates separately from its budget', async () => {
  const entries = [0, 1, 1000, 1001].map(frame => ({ frame, pose: pose(frame) }));
  const result = await searchLocalRefit(entries, async pairs => pairs.map(pair => ({ ...pair, forward: { accepted: false } })),
    { baseGraph: { nodes: entries, edges: [] } });
  assert.equal(result.matches.length, 4);
  assert.equal(result.rounds, 1);
  assert.equal(result.limited, false);
  await assert.rejects(searchLocalRefit(entries, async () => [], { baseGraph: { nodes: entries, edges: [] } }), /Unvollstaendige/);
  await assert.rejects(searchLocalRefit(entries, async () => { throw new Error('decode failed'); },
    { baseGraph: { nodes: entries, edges: [] } }), /decode failed/);
});

test('local measurements require majority consensus before becoming graph edges', () => {
  const graph = { nodes: [0, 1, 10, 11].map(frame => ({ frame, pose: pose(frame * 10) })), edges: [] };
  const match = (current, reference, measured, group = '0:1') => ({ current, reference, group,
    currentPose: pose(current * 10), referencePose: pose(reference * 10), forward: { accepted: true, pose: pose(measured), score: 0.96 },
    backward: { accepted: true }, reverseDistance: 1 });
  const result = addLocalRefitEdges(graph, [match(10, 0, 97), match(11, 1, 107.2), match(10, 1, 160)], { conditionalLimit: 7.5 });
  assert.equal(result.localEdges, 2);
});

test('local consensus needs independent frames on both sides', () => {
  const graph = { nodes: [0, 1, 10].map(frame => ({ frame, pose: pose(frame * 10) })), edges: [] };
  const match = reference => ({ current: 10, reference, group: '0:1', currentPose: pose(100), referencePose: pose(reference * 10),
    forward: { accepted: true, pose: pose(97 + reference * 0.1) }, backward: { accepted: true }, reverseDistance: 1 });
  const result = addLocalRefitEdges(graph, [match(0), match(1)]);
  assert.equal(result.localEdges, 0);
});

test('one strict symmetric local match can become a graph edge', () => {
  const graph = { nodes: [0, 10].map(frame => ({ frame, pose: pose(frame * 10) })), edges: [] };
  const result = addLocalRefitEdges(graph, [{ reference: 0, current: 10, group: '0:1', currentPose: pose(100),
    referencePose: pose(0), forward: { accepted: true, pose: pose(97), score: 0.995, margin: 0.01, support: 512 },
    backward: { accepted: true, score: 0.993, margin: 0.008, support: 480 }, reverseDistance: 0.4 }]);
  assert.equal(result.localEdges, 1);
});

test('one PCB-verified symmetric match can become a graph edge without NCC margins', () => {
  const graph = { nodes: [0, 10].map(frame => ({ frame, pose: pose(frame * 10) })), edges: [] };
  const match = { reference: 0, current: 10, group: '0:1', currentPose: pose(100),
    referencePose: pose(0), forward: { accepted: true, pose: pose(97), score: 0.995, support: 512 },
    backward: { accepted: true, score: 0.993 }, reverseDistance: 0.4 };
  assert.equal(addLocalRefitEdges(graph, [match]).localEdges, 0);
  assert.equal(addLocalRefitEdges(graph, [{ ...match, pcbVerified: true }]).localEdges, 1);
  assert.equal(addLocalRefitEdges(graph, [{ ...match, pcbVerified: true, reverseDistance: 8 }]).localEdges, 0);
});

test('one merely accepted local match still requires independent consensus', () => {
  const graph = { nodes: [0, 10].map(frame => ({ frame, pose: pose(frame * 10) })), edges: [] };
  const result = addLocalRefitEdges(graph, [{ reference: 0, current: 10, group: '0:1', currentPose: pose(100),
    referencePose: pose(0), forward: { accepted: true, pose: pose(97), score: 0.95, margin: 0.01, support: 512 },
    backward: { accepted: true, score: 0.95, margin: 0.008, support: 480 }, reverseDistance: 0.4 }]);
  assert.equal(result.localEdges, 0);
});

test('disconnected visits each keep their own anchor and only matched visits enter the preview', () => {
  const graph = { nodes: [0, 1, 10, 11, 20, 21].map(frame => ({ frame, pose: pose(frame * 10) })), edges: [
    { reference: 0, current: 1, measurement: pose(10), kind: 'incremental', weight: 1 },
    { reference: 10, current: 11, measurement: pose(10), kind: 'incremental', weight: 1 },
    { reference: 10, current: 11, measurement: pose(5), kind: 'local-refit', weight: 10 },
    { reference: 20, current: 21, measurement: pose(10), kind: 'incremental', weight: 1 }
  ] };
  const result = optimizePoseGraph(graph, { seedFrames: [0, 10, 20] });
  assert.deepEqual(result.alignedFrames, [10, 11]);
  assert.equal(result.corrections.find(item => item.frame === 10).pose.x, 100);
  assert.ok(result.corrections.find(item => item.frame === 11).pose.x < 110);
  assert.equal(result.corrections.find(item => item.frame === 21).pose.x, 210);
});

test('disconnected matched components are not mixed into one preview', () => {
  const graph = { nodes: [0, 1, 10, 11, 20, 21].map(frame => ({ frame, pose: pose(frame * 10) })), edges: [
    { reference: 0, current: 1, measurement: pose(5), kind: 'local-refit', weight: 10 },
    { reference: 10, current: 11, measurement: pose(5), kind: 'local-refit', weight: 10 },
    { reference: 20, current: 21, measurement: pose(5), kind: 'local-refit', weight: 10 },
    { reference: 20, current: 21, measurement: pose(6), kind: 'local-refit', weight: 10 }
  ] };
  const result = optimizePoseGraph(graph, { seedFrames: [0, 10, 20] });
  assert.deepEqual(result.alignedFrames, [20, 21]);
  assert.equal(result.corrections.find(item => item.frame === 20).pose.x, 200);
});

test('ambiguous local matches can become edges only through independent consensus', () => {
  const graph = { nodes: [0, 1, 10, 11].map(frame => ({ frame, pose: pose(frame * 10) })), edges: [] };
  const match = (current, reference, measured) => ({ current, reference, group: '0:1',
    currentPose: pose(current * 10), referencePose: pose(reference * 10),
    forward: { accepted: false, conditionallyAccepted: true, pose: pose(measured), score: 0.95 },
    backward: { accepted: false, conditionallyAccepted: true }, reverseDistance: 2 });
  const result = addLocalRefitEdges(graph, [match(10, 0, 96), match(11, 1, 106.5)]);
  assert.equal(result.localEdges, 2);
});

test('confirmed local edges can carry a large correction through the path', () => {
  const graph = { nodes: [0, 1, 2].map(frame => ({ frame, pose: pose(frame * 200) })), edges: [
    { reference: 0, current: 1, measurement: pose(200), kind: 'incremental', weight: 1 },
    { reference: 1, current: 2, measurement: pose(200), kind: 'incremental', weight: 1 },
    { reference: 0, current: 2, measurement: pose(20), kind: 'local-refit', weight: 6 }
  ] };
  const result = optimizePoseGraph(graph, { seedFrames: [2], iterations: 12, huber: 20 });
  assert.ok(result.corrections.find(item => item.frame === 2).pose.x < 40);
  assert.ok(result.localAfterRms < result.localBeforeRms * 0.1);
});


test('a translation-only aperture follows measured rotation without pinning the old camera angle', () => {
  const truth = { x: 40, y: 260, rotation: 0.05 }, pivot = { x: 0, y: -400 };
  const prediction = { x: 100, y: 420, rotation: 0 };
  const graph = { nodes: [{ frame: 0, pose: pose(0) }, { frame: 1, pose: prediction }], edges: [
    { reference: 0, current: 1, measurement: truth, kind: 'local-refit', weight: 10 }
  ] };
  const measured = { x: truth.x - Math.sin(truth.rotation) * pivot.y,
    y: truth.y + Math.cos(truth.rotation) * pivot.y - pivot.y, rotation: 0 };
  const match = { reference: 0, current: 1, group: '0:1', pcbVerified: true,
    referencePose: pose(0), currentPose: prediction, fft: { translationOnly: true },
    landmarkRecovery: { currentPivot: pivot },
    forward: { accepted: true, pose: measured, score: 0.99 },
    backward: { accepted: true }, reverseDistance: 0.5 };
  // Use a distinct kind for the initial angular constraint so both are retained.
  graph.edges[0].kind = 'spatial';
  const augmented = addLocalRefitEdges(graph, [match], { acceptVerifiedPairs: true });
  assert.equal(augmented.edges[1].rotationWeight, 0);
  const result = optimizePoseGraph(augmented, { iterations: 12, huber: 1000 });
  const actual = result.corrections.find(item => item.frame === 1).pose;
  assert.ok(Math.abs(actual.rotation - truth.rotation) < 1e-5);
  assert.ok(Math.hypot(actual.x - truth.x, actual.y - truth.y) < 0.05);
});
