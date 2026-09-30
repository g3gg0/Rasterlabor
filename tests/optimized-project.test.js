import test from 'node:test';
import assert from 'node:assert/strict';
import {importOptimizedProject} from '../src/optimized-project.js';
import {networkGeometryKey} from '../src/match-network.js';
const calibration={field:{width:100,height:100,nx:2,ny:2,coefficients:new Float64Array(8)},maps:{outputWidth:100,outputHeight:100,origin:[0,0]}};
const tracking={format:'rasterlabor-xyr-tracking',model_version:1,video:{width:100,height:100},path:[{frame:1,timestamp:1000,pose:{x:0,y:0,rotation:0}}],matchNetwork:{pairs:[],deletedPairs:['1:2']}};
const value={format:'rasterlabor-pose-refinement',version:1,geometryKey:networkGeometryKey(calibration),tracking};
test('optimized JSON restores complete tracking and deletion history without replacing lens maps',()=>{
  assert.equal(importOptimizedProject(value,calibration,{width:100,height:100,frameCount:2}),tracking);
  assert.deepEqual(tracking.matchNetwork.deletedPairs,['1:2']);
});
test('optimized JSON rejects mismatched calibration, malformed poses and wrong video',()=>{
  assert.throws(()=>importOptimizedProject(value,null),/ZIP/);
  assert.throws(()=>importOptimizedProject({...value,geometryKey:'wrong'},calibration),/Linsenkalibrierung/);
  assert.throws(()=>importOptimizedProject(value,calibration,{width:200,height:100}),/Video/);
  assert.throws(()=>importOptimizedProject({...value,tracking:{...tracking,path:[{frame:1,timestamp:0,pose:{x:NaN,y:0,rotation:0}}]}},calibration),/Trackingpose/);
});
