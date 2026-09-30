import test from 'node:test';
import assert from 'node:assert/strict';
import { contextImage } from '../src/context-tracker.js';
import { recoverPcbViaConstellation } from '../src/pcb-via-constellation.js';
import { measuredRefitComponent, planRefitBridgePairs } from '../src/pose-graph-refit.js';
import { acceptedPcbConstraints } from '../src/pcb-realignment.js';
const zero = { x: 0, y: 0, rotation: 0 };
function board(pose, periodic = false) {
  const width = 1200, height = 1200, data = new Uint8ClampedArray(width * height * 4);
  const vias = periodic ? [-400, 0, 400].flatMap(x => Array.from({ length: 21 }, (_, i) => [x, (i - 10) * 140])) :
    [[-440,-420],[420,-410],[-420,350],[260,320],[0,80]];
  const c = Math.cos(pose.rotation), s = Math.sin(pose.rotation);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const wx = c * (x - width/2) - s * (y - height/2) + pose.x;
    const wy = s * (x - width/2) + c * (y - height/2) + pose.y;
    let value = 110;
    if (vias.some(([vx,vy]) => Math.hypot(wx-vx,wy-vy) < 11)) value = 25;
    if (!periodic && (Math.abs(wy - 470) < 3 || Math.abs(wx + 510 + 12 * Math.sin(wy / 80)) < 3)) value = 40;
    const i = 4 * (y * width + x); data.set([value,value,value,255],i);
  }
  return contextImage({width,height,data});
}
test('via constellation recovers Y drift without any bright aperture and verifies traces', () => {
  const truth = {x:12,y:40,rotation:.02};
  const reference = {pose:zero,offset:[-600,-600]}, current = {pose:{x:20,y:300,rotation:0},offset:[-600,-600]};
  const result = recoverPcbViaConstellation(board(zero),board(truth),reference,current,
    {coarseRadius:384,limits:{angle:3,cycleLimit:5}});
  assert.equal(result.accepted,true,JSON.stringify(result));
  assert.ok(Math.hypot(result.fft.pose.x-truth.x,result.fft.pose.y-truth.y)<2);
  assert.ok(Math.abs(result.fft.pose.rotation-truth.rotation)<.002);
  assert.ok(result.fft.pointPairs.length>=3);
});
test('periodic via rows cannot authorize an ambiguous Y offset', () => {
  const image = board(zero,true), frame = {pose:zero,offset:[-600,-600]};
  const result = recoverPcbViaConstellation(image,image,frame,frame,{coarseRadius:300,limits:{angle:1,cycleLimit:5}});
  assert.equal(result.accepted,false,JSON.stringify(result));
});
test('tracking-only links do not hide missing matches between two measured groups', () => {
  const nodes = [1,2,3,4].map(frame=>({frame,pose:zero}));
  const edges = [{reference:1,current:2,kind:'local-refit'}, {reference:2,current:3,kind:'temporal'},
    {reference:3,current:4,kind:'local-refit'}];
  assert.deepEqual(measuredRefitComponent({nodes,edges}),[1,2]);
  edges.push({reference:2,current:4,kind:'local-refit'});
  assert.equal(measuredRefitComponent({nodes,edges}).length,4);
});

test('low photometric NCC requires strong via geometry AND independent unambiguous traces', () => {
  const evidence = {viaCount:5,residual:2.7,cycle:.1,patchScore:.68,traceScore:.63,traceSupport:978,ambiguityMargin:1};
  const match = {reference:1,current:2,referencePose:zero,
    forward:{accepted:true,score:.68,support:640,pose:zero},backward:{accepted:true,score:.68},reverseDistance:.1,
    fft:{method:'PCB-Via-Konstellation',residualRms:2.7,inlierCells:[0,1,2,3,4],uniqueSupportArea:640},
    landmarkRecovery:{accepted:true,geometricEvidence:evidence}};
  assert.equal(acceptedPcbConstraints([match]).accepted.length,1);
  for(const weaker of [{viaCount:4},{residual:3.1},{cycle:1.6},{patchScore:.59},{traceScore:.59},{traceSupport:63},{ambiguityMargin:.07}]) {
    assert.equal(acceptedPcbConstraints([{...match,landmarkRecovery:{accepted:true,geometricEvidence:{...evidence,...weaker}}}]).accepted.length,0);
  }
});

test('failed early via search still allows the later feature/aperture recovery', () => {
  const entries = [1,2].map(frame=>({frame,pose:zero}));
  const attempted = [{reference:1,current:2,pcbVerified:false,
    landmarkRecovery:{method:'Via-Konstellation + Leiterbahnen',accepted:false}}];
  assert.equal(planRefitBridgePairs(entries,[1],attempted,1).length,1);
  attempted[0].featureRecovery={accepted:false};
  assert.equal(planRefitBridgePairs(entries,[1],attempted,1).length,0);
});
