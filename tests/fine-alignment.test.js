import test from 'node:test';
import assert from 'node:assert/strict';
import { measureFinePair, planFinePairs, solveFineGraph, fineFrameReport, revalidateSavedCells } from '../src/fine-alignment.js';
import { emptyMatchNetwork, mergeNetworkMatches, networkEdges, deleteNetworkMatch } from '../src/match-network.js';

const pose = (x = 0, y = 0, rotation = 0) => ({ x, y, rotation });
function texture(actual = pose(), blank = false) {
  const width = 256, height = 256, data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const px = actual.x + Math.cos(actual.rotation) * (x - 128) - Math.sin(actual.rotation) * (y - 128);
    const py = actual.y + Math.sin(actual.rotation) * (x - 128) + Math.cos(actual.rotation) * (y - 128);
    const value = blank ? 90 : 110 + 24 * Math.sin(px * .24 + Math.sin(py * .1) * 2) +
      27 * Math.cos(py * .32 + Math.sin(px * .07)) + 20 * Math.sin(px * .4 + py * .27);
    const i = 4 * (y * width + x); data[i] = data[i + 1] = data[i + 2] = value; data[i + 3] = 255;
  }
  return { width, height, data };
}
const item = (frame, p = pose()) => ({ frame, pose: p, offset: [-128, -128] });

test('saved-anchor revalidation distinguishes texture from flat ambiguity without deleting points',()=>{
  const cells=[{id:'center',reference:{x:0,y:0},current:{x:0,y:0}}];
  const good=revalidateSavedCells(texture(),texture(),item(1),item(2),cells);
  const flat=revalidateSavedCells(texture(pose(),true),texture(pose(),true),item(1),item(2),cells);
  assert.equal(good[0].confidence,1);
  assert.equal(flat[0].confidence,.05);
  assert.equal(flat[0].id,'center');
});

test('fine pass recovers 1-5 pixel offsets and rotation, with measured subpixel anchors', () => {
  const truth = pose(1.3, -.7, .006);
  const match = measureFinePair(texture(), texture(truth), item(1), item(2, pose(4.5, -2, .009)));
  assert.ok(match.accepted, JSON.stringify({ reason: match.reason, score: match.score, cycle: match.reverseDistance, cells: match.fft.cells.map(c => ({reason:c.reason, psr:c.psr, owned:c.ownedSupportPixels,dx:c.dx})) }));
  assert.ok(Math.hypot(match.forward.pose.x - truth.x, match.forward.pose.y - truth.y) < .5);
  assert.ok(Math.abs(match.forward.pose.rotation - truth.rotation) < .003);
  const network = mergeNetworkMatches(emptyMatchNetwork(), [match], { cycleLimit: 1, minimumScore: .93 });
  assert.equal(network.pairs.length, 1);
  assert.ok(network.pairs[0].cells.length >= 3);
  assert.equal(networkEdges(network).length, 1);
  const strong = { ...network, pairs: network.pairs.map(pair => ({ ...pair, weight: 256 })) };
  assert.equal(mergeNetworkMatches(strong, [match], { preserveStrength: true }).pairs[0].weight, 256);
  const deleted = deleteNetworkMatch(network, network.pairs[0].id);
  assert.equal(mergeNetworkMatches(deleted, [match]).pairs.length, 0);
});

test('fine search rejects featureless images and corrections beyond its local radius', () => {
  const blank = measureFinePair(texture(), texture(pose(), true), item(1), item(2));
  assert.equal(blank.accepted, false);
  const distant = measureFinePair(texture(), texture(), item(1), item(2, pose(30, -20)));
  assert.equal(distant.accepted, false);
});

test('pair planning includes every unanchored frame and respects deleted pairs', () => {
  const frames = Array.from({ length: 12 }, (_, i) => item(i, pose(i * 40)));
  const network = { ...emptyMatchNetwork(), deletedPairs: ['0:1'] };
  const pairs = planFinePairs(frames, network, 110, 2);
  const covered = new Set(pairs.flatMap(pair => [pair.reference, pair.current]));
  assert.equal(covered.size, frames.length);
  assert.ok(!pairs.some(pair => pair.reference === 0 && pair.current === 1));
});

test('strong anchors couple translation and rotation while the joint fine step stays bounded', () => {
  const nodes = [item(0), item(1, pose(13)), item(2, pose(23)), item(3, pose(1000))];
  const edges = [
    { reference: 0, current: 1, measurement: pose(10, 0, .001), kind: 'match-network', weight: 80, rotationWeight: 1, verified: true },
    { reference: 1, current: 2, measurement: pose(10), kind: 'match-network', weight: 256, rotationWeight: 1, verified: true }
  ];
  const result = solveFineGraph({ nodes, edges }, { radius: 6, lever: 100 });
  const poses = new Map(result.corrections.map(row => [row.frame, row.pose]));
  assert.ok(Math.abs(poses.get(1).x - 10) < .1);
  assert.ok(Math.abs(poses.get(2).x - poses.get(1).x - 10) < .1);
  assert.ok(Math.abs(poses.get(2).rotation - poses.get(1).rotation) < .0001);
  assert.ok(!poses.has(3));
  const bounded = solveFineGraph({ nodes, edges: [{ ...edges[0], measurement: pose(-30) }, edges[1]] }, { radius: 6, lever: 100 });
  assert.ok(bounded.scale < 1);
  for (const row of bounded.corrections) assert.ok(Math.abs(row.pose.x - nodes[row.frame].pose.x) <= 6.001);
});

test('report retains all frames, worst failed partner and unmeasured frames ahead of good matches', () => {
  const report = fineFrameReport([item(1), item(2), item(3), item(4)], [
    { reference: 1, current: 2, accepted: true, score: .99 },
    { reference: 2, current: 3, accepted: false, score: .6, reason: 'ambiguous' }
  ]);
  assert.deepEqual(report.map(row => row.frame), [4, 2, 3, 1]);
  assert.equal(report.find(row => row.frame === 2).worstPartner, 3);
  assert.equal(report.find(row => row.frame === 2).accepted, 1);
});

function contextTexture(scale){const width=512,height=512,data=new Uint8ClampedArray(width*height*4);for(let y=0;y<height;y++)for(let x=0;x<width;x++){const px=(x-256)*scale,py=(y-256)*scale;const v=110+24*Math.sin(px*.08+Math.sin(py*.033)*2)+27*Math.cos(py*.107+Math.sin(px*.023))+20*Math.sin(px*.133+py*.09);const i=(y*width+x)*4;data[i]=data[i+1]=data[i+2]=v;data[i+3]=255;}return {width,height,data};}
const contextItem=frame=>({frame,pose:{x:0,y:0,rotation:0},offset:[-256,-256]});

test('larger context tolerates small local model errors only with distributed two-way evidence',()=>{
 const match=measureFinePair(contextTexture(1),contextTexture(1.008),contextItem(1),contextItem(2),{radius:8});
 assert.equal(match.accepted,true,match.reason);
 assert.equal(match.fft.cellSize,128);
 assert.ok(match.fft.inlierCells.length>=6);
 assert.ok(match.fft.residualRms<=2);
 assert.ok(match.reverseDistance<=1);
 const incompatible=measureFinePair(contextTexture(1),contextTexture(1.03),contextItem(1),contextItem(2),{radius:8});
 assert.equal(incompatible.accepted,false);
});
