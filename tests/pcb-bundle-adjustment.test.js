import test from 'node:test';
import assert from 'node:assert/strict';
import {optimizePcbBundle} from '../src/pcb-bundle-adjustment.js';
import {composePose,invertPose} from '../src/pcb-realignment.js';
const pose=(x=0,y=0,rotation=0)=>({x,y,rotation});
test('directional edges rotate both free endpoints consistently',()=>{
  const truth=[pose(),pose(80,20,.04),pose(160,40,.08)];
  const edges=[pair(0,1,truth),pair(1,2,truth),pair(0,2,truth)];
  for(const edge of edges)for(const cell of edge.cells){
    const angle=.7-truth[edge.reference].rotation;
    cell.normal={x:Math.cos(angle),y:Math.sin(angle)};
  }
  const nodes=truth.map((p,frame)=>({frame,pose:frame?pose(p.x+12,p.y-9,p.rotation+(frame===1?.06:-.04)):p}));
  const result=optimizePcbBundle(nodes,{pairs:edges},{lever:250,temporalWeight:.00001,iterations:30});
  assert.ok(result.afterRms<.01,`Directional residual ${result.beforeRms} -> ${result.afterRms}`);
  for(const i of [1,2])assert.ok(Math.abs(result.corrections[i].pose.rotation-truth[i].rotation)<.0001);
  assert.deepEqual(result.directionOnlyFrames,[0,1,2]);
});
function pair(a,b,truth,weight=50){
  return {id:`${a}:${b}`,reference:a,current:b,weight,cells:[[-200,-160],[140,-200],[220,180],[-180,240],[0,0]].map(([x,y])=>({
    reference:composePose(invertPose(truth[a]),pose(x,y)),current:composePose(invertPose(truth[b]),pose(x,y)),quality:1}))};
}
test('joint point solve recovers rotations and translations; unmeasured frames remain labelled',()=>{
  const truth=[pose(),pose(80,20,.04),pose(160,40,.08),pose(240,60,.12)];
  const nodes=truth.map((p,frame)=>({frame,pose:frame?pose(p.x+12,p.y-9,p.rotation+.025):p}));
  const network={pairs:[pair(0,1,truth),pair(1,3,truth),pair(0,3,truth)]};
  const result=optimizePcbBundle(nodes,network,{lever:250,temporalWeight:.001});
  assert.ok(result.afterRms<.03,JSON.stringify(result.after));
  for(const i of [1,3])assert.ok(Math.hypot(result.corrections[i].pose.x-truth[i].x,result.corrections[i].pose.y-truth[i].y)<.1);
  assert.deepEqual(result.unmeasured,[2]);
  assert.ok(Math.abs(result.corrections[2].pose.x-truth[2].x)<1);
});
test('deleted pairs cannot pull a frame and disconnected data stays finite',()=>{
  const truth=[pose(),pose(20),pose(40)],nodes=truth.map((p,frame)=>({frame,pose:p}));
  const wrong=pair(0,2,[pose(),pose(20),pose(400)]);
  const result=optimizePcbBundle(nodes,{pairs:[pair(0,1,truth),wrong],deletedPairs:[wrong.id]});
  assert.ok(Math.abs(result.corrections[2].pose.x-40)<.001);
  assert.deepEqual(result.unmeasured,[2]);
});
test('an outlying saved point cannot drag an otherwise consistent rigid pair',()=>{
  const truth=[pose(),pose(20,10,.02)],edge=pair(0,1,truth);
  edge.cells.push({reference:pose(800,500),current:pose(-400,100),quality:1});
  const result=optimizePcbBundle([{frame:0,pose:truth[0]},{frame:1,pose:pose(28,4,.03)}],{pairs:[edge]},
    {huber:1,lever:250,temporalWeight:.001,iterations:30});
  const p=result.corrections[1].pose;
  assert.ok(Math.hypot(p.x-20,p.y-10)<2,JSON.stringify(p));
  assert.ok(Math.abs(p.rotation-.02)<.005,JSON.stringify(p));
});
