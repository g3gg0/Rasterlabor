import test from 'node:test';
import assert from 'node:assert/strict';
import {planPcbOverlapChecks} from '../src/pcb-overlap-checks.js';
const frame=(frame,x,y=0)=>({frame,pose:{x,y,rotation:0}});
test('audits revisited regions despite an already connected graph and ignores FPS',()=>{
  const nodes=[frame(1,0),frame(2,500),frame(3,1000),frame(4,500,50),frame(5,0,50)];
  const network={pairs:[{reference:1,current:5}],deletedPairs:[]};
  const a=planPcbOverlapChecks(nodes,network,300);
  assert.ok(a.some(p=>p.reference===1&&p.current===5));
  assert.deepEqual(planPcbOverlapChecks(nodes.map(n=>({...n,frame:n.frame*1000})),network,300),
    a.map(p=>({...p,reference:p.reference*1000,current:p.current*1000})));
});
test('does not turn stationary captures into visits; respects explicit deletion',()=>{
  assert.deepEqual(planPcbOverlapChecks([frame(1,0),frame(10000,10)],{},300),[]);
  const nodes=[frame(1,0),frame(2,1000),frame(3,5)];
  assert.deepEqual(planPcbOverlapChecks(nodes,{deletedPairs:['1:3']},300),[]);
});
