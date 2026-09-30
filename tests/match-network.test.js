import test from 'node:test';
import assert from 'node:assert/strict';
import { emptyMatchNetwork, mergeNetworkMatches, migrateMatchNetwork, deleteNetworkMatch,
  networkEdges, applyNetworkToGraph, projectNetworkCells, networkGeometryKey } from '../src/match-network.js';
import { composePose, invertPose } from '../src/pcb-realignment.js';
import { validateTracking } from '../src/tracking-data.js';
const pose = (x=0,y=0,rotation=0)=>({x,y,rotation});
function measurement(reference=1,current=2) {
  const points=[[-100,-100],[100,-100],[-100,100],[100,100]].map(([x,y],cellId)=>({cellId,
    reference:{x,y},current:{x:x-10,y},psr:24,support:2048}));
  return { reference,current,referencePose:pose(),currentPose:pose(12),
    forward:{accepted:true,pose:pose(10),score:.99,support:8192},
    backward:{accepted:true,pose:pose(),score:.99,support:8192},reverseDistance:.5,
    fft:{accepted:true,cellSize:64,inlierCells:[0,1,2,3],uniqueSupportArea:8192,residualRms:.5,pointPairs:points} };
}
test('local point coordinates survive pose edits and a project save/load roundtrip',()=>{
  const network=mergeNetworkMatches(emptyMatchNetwork('lens'),[measurement()]);
  const tracking={format:'rasterlabor-xyr-tracking',model_version:1,path:[1,2].map(frame=>({frame,timestamp:frame,pose:pose()})),matchNetwork:network};
  const loaded=validateTracking(JSON.parse(JSON.stringify(tracking)));
  const restored=migrateMatchNetwork(loaded,'lens');
  assert.deepEqual(restored,network);
  const poses=new Map([[1,pose(100,200,.3)],[2,composePose(pose(100,200,.3),pose(10))]]);
  for(const cell of projectNetworkCells(restored.pairs[0],poses))
    assert.ok(Math.hypot(cell.referenceWorld.x-cell.currentWorld.x,cell.referenceWorld.y-cell.currentWorld.y)<1e-8);
});

test('rechecked ambiguous anchors remain visible, weaker and cannot be revived by replaying an old match',()=>{
  const match=measurement(),initial=mergeNetworkMatches(emptyMatchNetwork(),[match]);
  const evidence={reference:1,current:2,anchorEvidence:initial.pairs[0].cells.map(cell=>({id:cell.id,confidence:.05,score:null,reason:'Flat'}))};
  const checked=mergeNetworkMatches(initial,[evidence]);
  const replay=mergeNetworkMatches(checked,[match]);
  assert.equal(replay.pairs[0].cells.length,4);
  assert.ok(replay.pairs[0].cells.every(cell=>cell.confidence===.05));
  assert.ok(networkEdges(replay)[0].weight<networkEdges(initial)[0].weight*.1);
  assert.ok(initial.pairs[0].cells.every(cell=>cell.confidence===undefined));
});
test('deleting a false cell refits the connection from the remaining points',()=>{
  const match=measurement();match.fft.pointPairs[3].current.x+=40;
  const network=mergeNetworkMatches(emptyMatchNetwork(),[match]);
  assert.ok(Math.abs(networkEdges(network)[0].measurement.x-10)>5);
  const falseCell=network.pairs[0].cells.find(cell=>cell.sourceCellId===3);
  const edited=deleteNetworkMatch(network,'1:2',falseCell.id);
  const edge=networkEdges(edited)[0];
  assert.ok(Math.abs(edge.measurement.x-10)<1e-8);
  assert.ok(Math.abs(edge.measurement.rotation)<1e-8);
  assert.equal(network.pairs[0].cells.length,4,'undo source remains intact');
  assert.equal(edited.pairs[0].cells.length,3);
});
test('deleted cells stay excluded when a new run remeasures them nearby',()=>{
  const original=mergeNetworkMatches(emptyMatchNetwork(),[measurement()]);
  const edited=deleteNetworkMatch(original,'1:2',original.pairs[0].cells[0].id);
  const match=measurement();match.fft.pointPairs[0].reference.x+=1;match.fft.pointPairs[0].current.x+=1;
  const restored=migrateMatchNetwork({matchNetwork:JSON.parse(JSON.stringify(edited))},null);
  const repeated=mergeNetworkMatches(restored,[match]);
  assert.equal(repeated.pairs[0].cells.length,3);
  assert.equal(repeated.deletedCells.length,1);
});
test('deleted pairs cannot return through old diagnostics, incremental edges or remeasurement',()=>{
  const original=mergeNetworkMatches(emptyMatchNetwork(),[measurement()]);
  const deleted=deleteNetworkMatch(original,'1:2');
  const loaded=migrateMatchNetwork({matchNetwork:JSON.parse(JSON.stringify(deleted)),
    pcbRealignment:{pairDiagnostics:[measurement()]}},null);
  assert.equal(mergeNetworkMatches(loaded,[measurement()]).pairs.length,0);
  const graph={nodes:[1,2,3].map(frame=>({frame,pose:pose()})),edges:[
    {reference:1,current:2,kind:'incremental',measurement:pose(10),weight:1},
    {reference:2,current:3,kind:'incremental',measurement:pose(10),weight:1}]};
  const filtered=applyNetworkToGraph(graph,loaded);
  assert.equal(filtered.edges.length,1);assert.equal(filtered.edges[0].current,3);
});
test('too few remaining FFT cells deactivate the connection instead of using its stale pose',()=>{
  let network=mergeNetworkMatches(emptyMatchNetwork(),[measurement()]);
  for(const cell of network.pairs[0].cells.slice(0,2))network=deleteNetworkMatch(network,'1:2',cell.id);
  assert.equal(networkEdges(network).length,0);
  const graph=applyNetworkToGraph({nodes:[1,2].map(frame=>({frame,pose:pose()})),
    edges:[{reference:1,current:2,kind:'local-refit',measurement:pose(10),weight:10}]},network);
  assert.equal(graph.edges.length,0);
});
test('reverse-direction pairs share one identity and preserve the inverse transform',()=>{
  const match=measurement(2,1);
  const network=mergeNetworkMatches(emptyMatchNetwork(),[match]);
  const edge=networkEdges(network)[0];
  assert.equal(edge.reference,1);assert.equal(edge.current,2);
  assert.ok(Math.abs(edge.measurement.x+10)<1e-8);
});
test('older projects retain pose constraints and label reconstructed point positions',()=>{
  const match=measurement();delete match.fft.pointPairs;
  match.fft.cells=[[-100,-100],[100,-100],[-100,100]].map(([x,y],cellId)=>({cellId,center:{x,y}}));
  const network=migrateMatchNetwork({pcbRealignment:{pairDiagnostics:[match],constraints:[
    {reference:2,current:3,measurement:pose(20),weight:4}]}},'lens');
  assert.equal(network.pairs.length,2);
  assert.equal(network.pairs[0].approximate,true);
  assert.equal(network.pairs[0].cells.length,3);
  assert.equal(networkEdges(network).length,2);
});
test('calibration identity changes with distortion and stays independent of tracking poses',()=>{
  const calibration={field:{width:192,height:192,nx:2,ny:2,coefficients:new Float64Array([1,2])},
    maps:{outputWidth:192,outputHeight:192,origin:[0,0]}};
  const key=networkGeometryKey(calibration);
  assert.equal(networkGeometryKey(JSON.parse(JSON.stringify({...calibration,
    field:{...calibration.field,coefficients:[1,2]}}))),key);
  calibration.field.coefficients[0]=3;
  assert.notEqual(networkGeometryKey(calibration),key);
});
