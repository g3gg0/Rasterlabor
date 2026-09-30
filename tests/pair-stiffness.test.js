import test from 'node:test';
import assert from 'node:assert/strict';
import { pairStiffness } from '../src/pair-stiffness.js';
import { acceptedPcbConstraints, composePose, invertPose } from '../src/pcb-realignment.js';
import { optimizePoseGraph } from '../src/pose-graph-refit.js';
const pose = (x, y = 0, rotation = 0) => ({ x, y, rotation });
const measurement = {
  reference: 1, current: 2, kind: 'temporal', referencePose: pose(0),
  forward: { accepted: true, pose: pose(100), score: .99, support: 156173 },
  backward: { accepted: true, score: .99, support: 156173 }, reverseDistance: 4.84,
  fft: { accepted: true, inlierCells: Array.from({length:18}, (_, i) => i),
    uniqueSupportArea: 156173, residualRms: 2.07 }
};
test('distributed low-error screenshot measurement is much stiffer than marginal evidence', () => {
  const strong = pairStiffness(measurement);
  const weak = pairStiffness({...measurement, forward:{...measurement.forward,score:.91,support:128},
    backward:{...measurement.backward,score:.91},
    fft:{inlierCells:[0,1,2],uniqueSupportArea:128,residualRms:12}});
  assert.ok(strong.weight > 200);
  assert.ok(strong.weight > weak.weight * 30);
  assert.equal(strong.rotationWeight, 1);
  assert.equal(pairStiffness({...measurement,fft:{...measurement.fft,translationOnly:true}}).rotationWeight,0);
  assert.ok(pairStiffness({...measurement,fft:{...measurement.fft,residualRms:20}}).weight < strong.weight);
});
test('verified stiff pair follows a large translation and rotation together despite weaker conflicting links', () => {
  const [edge] = acceptedPcbConstraints([measurement],{cycleLimit:5}).accepted;
  assert.equal(edge.verified,true);
  const result = optimizePoseGraph({nodes:[{frame:0,pose:pose(0)},
    {frame:1,pose:pose(0)}, {frame:2,pose:pose(100)}], edges:[
      {reference:0,current:1,measurement:pose(500,200,.2),weight:20,verified:true,kind:'pcb-spatial'},
      {...edge},
      {reference:0,current:2,measurement:pose(100),weight:1,kind:'incremental'}
    ]},{iterations:20});
  const first = result.corrections.find(item=>item.frame===1).pose;
  const second = result.corrections.find(item=>item.frame===2).pose;
  const relative = composePose(invertPose(first),second);
  assert.ok(first.x>450 && second.x>540);
  assert.ok(first.rotation>.18 && second.rotation>.18);
  assert.ok(Math.hypot(relative.x-100,relative.y)<1);
  assert.ok(Math.abs(relative.rotation)<.001);
});
