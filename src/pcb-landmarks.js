import { composePose, invertPose } from './pcb-realignment.js';
import { registerPcbFftPair } from './pcb-fft-pair.js';
import { registerOverlap } from './context-tracker.js';
import { recoverPcbViaPair, recoverPcbViaPool } from './pcb-vias.js';

// Bright apertures provide search seeds on smooth boards, including partly
// masked holes. A seed is accepted only after independent masked FFT checks.
export function pcbApertures(image) {
  const level=image.levels.findLast(level=>Math.max(level.width,level.height)>=240)??image.levels[0];
  const {width,height,gray,scale}=level,visited=new Uint8Array(gray.length),result=[];
  for(let index=0;index<gray.length;index++) {
    if(visited[index]||gray[index]<160) continue;
    const queue=[index];visited[index]=1;
    let minX=width,minY=height,maxX=0,maxY=0,touches=false;
    for(let position=0;position<queue.length;position++) {
      const current=queue[position],x=current%width,y=Math.floor(current/width);
      minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);
      for(const [dx,dy] of [[-1,0],[1,0],[0,-1],[0,1]]) {
        const nx=x+dx,ny=y+dy;
        if(nx<0||ny<0||nx>=width||ny>=height){touches=true;continue;}
        const next=ny*width+nx;
        if(gray[next]<0){touches=true;continue;}
        if(visited[next]||gray[next]<160) continue;
        visited[next]=1;queue.push(next);
      }
    }
    const w=(maxX-minX+1)*scale,h=(maxY-minY+1)*scale;
    if(Math.min(w,h)<100||Math.max(w,h)>1200||Math.max(w,h)/Math.min(w,h)>2||
        queue.length/((maxX-minX+1)*(maxY-minY+1))<0.55) continue;
    result.push({x:(minX+maxX+1)*scale/2-image.width/2,
      y:(minY+maxY+1)*scale/2-image.height/2,width:w,height:h,touchesMask:touches});
  }
  return result;
}

export function recoverPcbLandmarkPair(referenceRgba,currentRgba,reference,current,referenceImage,currentImage,data) {
  const a=pcbApertures(referenceImage),b=pcbApertures(currentImage);
  const result={reference:a,current:b,attempts:[],accepted:false};
  const refOffset={x:reference.offset[0]+referenceRgba.width/2,y:reference.offset[1]+referenceRgba.height/2,rotation:0};
  const curOffset={x:current.offset[0]+currentRgba.width/2,y:current.offset[1]+currentRgba.height/2,rotation:0};
  const refCenter=composePose(reference.pose,refOffset),curCenter=composePose(current.pose,curOffset);
  candidates: for(const first of a) for(const second of b) {
    if(Math.max(first.width/second.width,second.width/first.width,first.height/second.height,second.height/first.height)>1.5) continue;
    const refWorld=composePose(refCenter,{...first,rotation:0}),curWorld=composePose(curCenter,{...second,rotation:0});
    let pose={...current.pose,x:current.pose.x+refWorld.x-curWorld.x,y:current.pose.y+refWorld.y-curWorld.y};
    if(Math.hypot(pose.x-current.pose.x,pose.y-current.pose.y)>data.coarseRadius) continue;
    const aperturePose=pose;
    if(data.referenceVias?.length>1) {
      const pool=recoverPcbViaPool(currentImage,data.referenceVias,curCenter,composePose(pose,curOffset),data);
      result.viaPool=pool;
      if(pool.accepted) {
        const cameraPose=composePose(pool.pose,invertPose(curOffset));
        const score=registerOverlap(currentImage,referenceImage,pool.pose,refCenter,
          {radius:1,angle:0,coarseStep:0,partial:false}).score;
        pool.score=score;
        const reverseErrors=pool.pairs.flatMap(pair=>pair.observations.filter(item=>item.frame===reference.frame)
          .map(item=>{const world=composePose(pool.pose,pair.source);return{x:world.x-item.world.x,y:world.y-item.world.y};}));
        const cycle=reverseErrors.length?Math.sqrt(reverseErrors.reduce((sum,error)=>sum+error.x**2+error.y**2,0)/reverseErrors.length):Infinity;
        pool.cycle=cycle;
        if(score>=Math.max(0.9,data.limits.minimumScore??0.9)&&cycle<=Math.min(5,data.limits.cycleLimit??5)&&
            Math.hypot(cameraPose.x-current.pose.x,cameraPose.y-current.pose.y)<=data.coarseRadius) {
          result.fft={accepted:true,pose:cameraPose,translationOnly:false,cellSize:32,
            cells:pool.pairs.map((pair,cellId)=>({cellId,accepted:true,center:pair.target})),
            inlierCells:pool.pairs.map((_,index)=>index),uniqueSupportArea:pool.inliers*128,
            residualRms:pool.residual,method:'PCB-Vias',
            pointPairs:pool.pairs.map((pair,cellId)=>({cellId,
              reference:composePose(invertPose(reference.pose),{...pair.target,rotation:0}),
              current:{x:curOffset.x+pair.source.x,y:curOffset.y+pair.source.y},support:128}))};
          const backwardPose={...reference.pose,
            x:reference.pose.x+reverseErrors.reduce((sum,error)=>sum+error.x,0)/reverseErrors.length,
            y:reference.pose.y+reverseErrors.reduce((sum,error)=>sum+error.y,0)/reverseErrors.length};
          Object.assign(result,{accepted:true,backwardPose,score,
            reverseVerification:{accepted:true,cycle,score,cells:pool.inliers,support:pool.inliers*128,method:'mutual-via-pool'}});
          return result;
        }
      }
    }
    const vias=recoverPcbViaPair(referenceImage,currentImage,refCenter,curCenter,
      composePose(pose,curOffset),data);
    result.vias=vias;
    if(vias.accepted) {
      const reverse=recoverPcbViaPair(currentImage,referenceImage,vias.pose,refCenter,refCenter,data);
      const angle=reverse.pose ? reverse.pose.rotation-refCenter.rotation : Infinity;
      const cycle=reverse.pose ? Math.hypot(reverse.pose.x-refCenter.x,reverse.pose.y-refCenter.y)+
        Math.abs(Math.atan2(Math.sin(angle),Math.cos(angle)))*Math.hypot(currentRgba.width,currentRgba.height)/2 : Infinity;
      const score=registerOverlap(currentImage,referenceImage,vias.pose,refCenter,
        {radius:1,angle:0,coarseStep:0,partial:false}).score;
      vias.verification={accepted:reverse.accepted,cycle,score,inliers:reverse.inliers};
      if(reverse.accepted&&cycle<=Math.min(5,data.limits.cycleLimit??5)&&score>=0.95) {
        const cameraPose=composePose(vias.pose,invertPose(curOffset));
        if(Math.hypot(cameraPose.x-current.pose.x,cameraPose.y-current.pose.y)<=data.coarseRadius) {
          result.fft={accepted:true,pose:cameraPose,translationOnly:Boolean(vias.translationOnly),cellSize:32,
            cells:vias.pairs.map((pair,cellId)=>({cellId,accepted:true,center:composePose(refCenter,pair.reference)})),
            inlierCells:vias.pairs.map((_,index)=>index),uniqueSupportArea:vias.inliers*128,
            residualRms:vias.residual,method:'PCB-Vias',
            pointPairs:vias.pairs.map((pair,cellId)=>({cellId,
              reference:{x:refOffset.x+pair.reference.x,y:refOffset.y+pair.reference.y},
              current:{x:curOffset.x+pair.source.x,y:curOffset.y+pair.source.y},support:128}))};
          Object.assign(result,{accepted:true,backwardPose:composePose(reverse.pose,invertPose(refOffset)),
            currentPivot:vias.translationOnly?{x:curOffset.x+vias.pairs[0].source.x,y:curOffset.y+vias.pairs[0].source.y}:null,
            reverseVerification:{accepted:true,cycle,score,cells:reverse.inliers,support:reverse.inliers*128},score});
          return result;
        }
      }
    }
    const refined=registerOverlap(currentImage,referenceImage,composePose(pose,curOffset),refCenter,
      {radius:96,angle:5,coarseStep:0,partial:false});
    if(refined.pose&&refined.score>=0.95) {
      pose=composePose(refined.pose,{x:-curOffset.x,y:-curOffset.y,rotation:0});
      result.refinement={pose,score:refined.score};
    }
    // A round hole locates translation but cannot determine rotation. Measure
    // angles from vias/traces outside its ring before considering a fixed angle.
    measurements: for(const mode of ['outer','seeded','rigid','translation']) for(const cellSize of [64,128,256]) {
      const translationOnly=mode==='translation';
      const measurementPose=mode==='rigid'||translationOnly?aperturePose:pose;
      const excludedRegions=mode!=='outer'?[]:[{center:refWorld,radius:Math.max(first.width,first.height)*0.55}];
      const confirmed=registerPcbFftPair(referenceRgba,currentRgba,reference.pose,measurementPose,reference.offset,current.offset,
        {...data.fft,cellSize,candidateGrid:24,translationOnly,excludedRegions,searchRadius:Math.min(63,cellSize/2-1)});
      const attempt={seed:measurementPose,cellSize,translationOnly,accepted:confirmed.accepted,reason:confirmed.reason,
        cells:confirmed.inlierCells?.length,support:confirmed.uniqueSupportArea,pose:confirmed.pose};
      result.attempts.push(attempt);
      if(confirmed.accepted&&confirmed.inlierCells.length>=3&&confirmed.uniqueSupportArea>=128) {
        const reverseOptions={...data.fft,cellSize,candidateGrid:24,translationOnly,excludedRegions,
          residualLimit:translationOnly?data.fft.residualLimit:4};
        let reverse=registerPcbFftPair(currentRgba,referenceRgba,confirmed.pose,reference.pose,
          current.offset,reference.offset,{...reverseOptions,searchRadius:translationOnly?31:8});
        if(!translationOnly&&!reverse.accepted) reverse=registerPcbFftPair(currentRgba,referenceRgba,
          confirmed.pose,reference.pose,current.offset,reference.offset,{...reverseOptions,searchRadius:31});
        const angle=confirmed.pose.rotation-current.pose.rotation;
        const rotationCorrection=Math.abs(Math.atan2(Math.sin(angle),Math.cos(angle)))*180/Math.PI;
        const reverseAngle=reverse.pose ? reverse.pose.rotation-reference.pose.rotation : Infinity;
        const cycle=reverse.pose ? Math.hypot(reverse.pose.x-reference.pose.x,reverse.pose.y-reference.pose.y)+
          Math.abs(Math.atan2(Math.sin(reverseAngle),Math.cos(reverseAngle)))*Math.hypot(currentRgba.width,currentRgba.height)/2 : Infinity;
        const atPose=registerOverlap(currentImage,referenceImage,composePose(confirmed.pose,curOffset),refCenter,
          {radius:1,angle:0,coarseStep:0,partial:true});
        attempt.verification={cells:reverse.inlierCells?.length??0,support:reverse.uniqueSupportArea??0,
          cycle,score:atPose.score};
        if(reverse.accepted&&reverse.inlierCells.length>=3&&reverse.uniqueSupportArea>=128&&
            cycle<=Math.min(5,data.limits.cycleLimit??5)&&atPose.score>=0.95&&
            rotationCorrection<=Math.max(5,data.limits.angle??0)&&
            Math.hypot(confirmed.pose.x-current.pose.x,confirmed.pose.y-current.pose.y)<=data.coarseRadius) {
          // Without structure outside the aperture, its fitted angle remains
          // ambiguous even if the circular patches are mutually consistent.
          const anglePassive=translationOnly||mode!=='outer';
          result.fft={...confirmed,translationOnly:anglePassive};
          Object.assign(result,{accepted:true,reverseVerification:{...attempt.verification,accepted:true},
            backwardPose:reverse.pose,score:atPose.score,
            currentPivot:anglePassive?{x:curOffset.x+second.x,y:curOffset.y+second.y}:null});
          break measurements;
        }
      }
    }
    if(result.accepted) break candidates;
  }
  return result;
}
