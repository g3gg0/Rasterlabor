import test from 'node:test';
import assert from 'node:assert/strict';
import { contextImage } from '../src/context-tracker.js';
import { pcbApertures, recoverPcbLandmarkPair } from '../src/pcb-landmarks.js';

function board(pose, {flat=false, clipped=false}={}) {
  const width=768,height=1024,data=new Uint8ClampedArray(width*height*4);
  const vias=[[-260,-120],[-200,180],[230,150],[250,-170],[-100,300],[80,320],[-240,40],[270,30]];
  for(let y=0;y<height;y++) for(let x=0;x<width;x++) {
    const lx=x-width/2,ly=y-height/2,
      wx=Math.cos(pose.rotation)*lx-Math.sin(pose.rotation)*ly+pose.x,
      wy=Math.sin(pose.rotation)*lx+Math.cos(pose.rotation)*ly+pose.y;
    const distance=Math.hypot(wx,wy+50);
    let value=flat?80:80+wx*0.003+wy*0.005;
    if(!flat) {
      if(distance<140)value=35;
      if(distance<110)value=215;
      if(vias.some(([vx,vy])=>Math.hypot(wx-vx,wy-vy)<7))value=15;
      if(Math.abs(wx+290+15*Math.sin(wy/50))<2)value=140;
    }
    const index=(y*width+x)*4;
    data.set([value,value,value,(lx/360)**2+(ly/475)**2<=1&&(!clipped||ly>-400)?255:0],index);
  }
  return {width,height,data};
}

const data={coarseRadius:384,fft:{cellSize:256,cellsPerAxis:3,searchRadius:127,minimumPsr:6,residualLimit:15,adaptiveCells:true},limits:{cycleLimit:5}};

test('aperture seed recovers drift beyond the local FFT radius with independently checked cells', () => {
  const refPose={x:0,y:0,rotation:0},truth={x:40,y:260,rotation:0};
  const first=board(refPose),second=board(truth);
  const result=recoverPcbLandmarkPair(first,second,{pose:refPose,offset:[-384,-512]},
    {pose:{x:100,y:420,rotation:0},offset:[-384,-512]},contextImage(first),contextImage(second),data);
  assert.equal(result.accepted,true,JSON.stringify({attempts:result.attempts,pose:result.fft?.pose}));
  assert.ok(Math.hypot(result.fft.pose.x-truth.x,result.fft.pose.y-truth.y)<3,JSON.stringify({attempts:result.attempts,pose:result.fft?.pose}));
  assert.ok(result.reverseVerification.cycle<=5);
  assert.ok(result.fft.inlierCells.length>=3&&result.reverseVerification.cells>=3);
  assert.ok(Math.abs(result.fft.pose.rotation)<0.002);
});

test('distributed aperture checks also recover an angular drift between visits', () => {
  const refPose={x:0,y:0,rotation:0},truth={x:40,y:260,rotation:0.025};
  const first=board(refPose),second=board(truth);
  const result=recoverPcbLandmarkPair(first,second,{pose:refPose,offset:[-384,-512]},
    {pose:{x:100,y:420,rotation:-0.028},offset:[-384,-512]},contextImage(first),contextImage(second),data);
  assert.equal(result.accepted,true,JSON.stringify({attempts:result.attempts,pose:result.fft?.pose}));
  assert.ok(Math.hypot(result.fft.pose.x-truth.x,result.fft.pose.y-truth.y)<3,JSON.stringify({attempts:result.attempts,pose:result.fft?.pose}));
  assert.ok(Math.abs(result.fft.pose.rotation-truth.rotation)<0.005,JSON.stringify({attempts:result.attempts,pose:result.fft?.pose}));
  assert.ok(result.reverseVerification.cycle<=5);
});

test('a partly masked aperture remains a search seed and flat boards provide no constraint', () => {
  const pose={x:40,y:260,rotation:0},partial=contextImage(board(pose,{clipped:true}));
  assert.ok(pcbApertures(partial).some(aperture=>aperture.touchesMask));
  const flat=board(pose,{flat:true});
  const result=recoverPcbLandmarkPair(flat,flat,{pose,offset:[-384,-512]},{pose,offset:[-384,-512]},
    contextImage(flat),contextImage(flat),data);
  assert.equal(result.accepted,false);
});

test('an aperture cannot authorize a correction beyond the configured search bound', () => {
  const refPose={x:0,y:0,rotation:0},truth={x:40,y:260,rotation:0};
  const first=board(refPose),second=board(truth);
  const result=recoverPcbLandmarkPair(first,second,{pose:refPose,offset:[-384,-512]},
    {pose:{x:100,y:420,rotation:0},offset:[-384,-512]},contextImage(first),contextImage(second),{...data,coarseRadius:64});
  assert.equal(result.accepted,false);
});
