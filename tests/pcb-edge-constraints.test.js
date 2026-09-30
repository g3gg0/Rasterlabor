import test from 'node:test';
import assert from 'node:assert/strict';
import {measureEdgeConstraints} from '../src/pcb-edge-constraints.js';
import {optimizePcbBundle} from '../src/pcb-bundle-adjustment.js';
import {optimizePoseGraph} from '../src/pose-graph-refit.js';
import {mergeNetworkMatches,emptyMatchNetwork,networkEdges,normalizeMatchNetwork} from '../src/match-network.js';
function im(offset=0,slope=0,flat=false){const width=512,height=768,data=new Uint8ClampedArray(width*height*4);for(let y=0;y<height;y++)for(let x=0;x<width;x++){const a=flat?0:1/(1+Math.exp(-(x-256-offset-slope*(y-384))/2));const i=(y*width+x)*4;data[i]=40+160*a;data[i+1]=110+45*a;data[i+2]=60-20*a;data[i+3]=255;}return {width,height,data};}
const ref={frame:1,pose:{x:0,y:0,rotation:0},offset:[-256,-384]},cur={frame:2,pose:{x:0,y:23,rotation:0},offset:[-256,-384]};
test('soft color edge constrains its normal and rotation while leaving tangent motion free',()=>{
 const match=measureEdgeConstraints(im(),im(6,.004),ref,cur);
 assert.equal(match.accepted,true,match.reason);assert.equal(match.partial,true);
 assert.ok(match.fft.pointPairs.every(c=>Math.abs(Math.hypot(c.normal.x,c.normal.y)-1)<1e-6));
 const net=mergeNetworkMatches(emptyMatchNetwork(),[match]);assert.equal(net.pairs.length,1);
 const result=optimizePcbBundle([ref,cur],net,{iterations:30});const p=result.corrections[1].pose;
 assert.ok(Math.abs(p.x+6)<.2,JSON.stringify(p));assert.ok(Math.abs(p.rotation-.004)<.0005);
 assert.ok(Math.abs(p.y-23)<.1,'Unobservable tangent must not be snapped to the sampled point');
 const edges=networkEdges(net);assert.equal(edges[0].normalOnly,true);
 const local=optimizePoseGraph({nodes:[ref,cur],edges},{includePassive:true,iterations:20});
 assert.ok(Math.abs(local.corrections[1].pose.y-23)<.1);
 assert.ok(local.localAfterRms<local.localBeforeRms);
 assert.ok(normalizeMatchNetwork(JSON.parse(JSON.stringify(net))).pairs[0].cells[0].normal);
});
test('mask boundaries and featureless color fields cannot create edge anchors',()=>{
 const a=im(0,0,true),b=im(0,0,true);
 for(let y=0;y<a.height;y++)for(let x=0;x<256;x++)a.data[(y*a.width+x)*4+3]=0;
 assert.equal(measureEdgeConstraints(a,b,ref,cur).accepted,false);
});

test('repeated parallel edges remain ambiguous instead of becoming normal anchors',()=>{
 const stripe=shift=>{const image=im();for(let y=0;y<image.height;y++)for(let x=0;x<image.width;x++){
  const i=(y*image.width+x)*4,v=110+70*Math.sin((x+shift)*Math.PI/16);image.data[i]=image.data[i+1]=image.data[i+2]=v;
 }return image;};
 assert.equal(measureEdgeConstraints(stripe(0),stripe(4),ref,cur).accepted,false);
});
