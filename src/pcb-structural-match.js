import { registerPcbFftPair } from './pcb-fft-pair.js';
import { measureFinePair } from './fine-alignment.js';

// Real image gradients only: invalid/masked boundaries must never become edges.
export function structuralImage(image) {
  const {width,height,data}=image, gray=new Float32Array(width*height), out=new Uint8ClampedArray(data.length);
  for(let i=0;i<gray.length;i++)gray[i]=.299*data[i*4]+.587*data[i*4+1]+.114*data[i*4+2];
  for(let y=1;y<height-1;y++)for(let x=1;x<width-1;x++){
    const i=y*width+x;
    if([i-width-1,i-width,i-width+1,i-1,i,i+1,i+width-1,i+width,i+width+1].some(j=>data[j*4+3]!==255))continue;
    const gx=gray[i-width+1]+2*gray[i+1]+gray[i+width+1]-gray[i-width-1]-2*gray[i-1]-gray[i+width-1];
    const gy=gray[i+width-1]+2*gray[i+width]+gray[i+width+1]-gray[i-width-1]-2*gray[i-width]-gray[i-width+1];
    const v=Math.min(255,Math.max(0,Math.hypot(gx,gy)-8)*1.5);
    out[i*4]=out[i*4+1]=out[i*4+2]=v;out[i*4+3]=255;
  }
  return {width,height,data:out};
}

export function measureStructuralPair(first,second,reference,current,{radius=64,mask=null}={}) {
  const a=structuralImage(first),b=structuralImage(second);
  let match=measureFinePair(a,b,reference,current,{radius:12,mask});
  if(!match.accepted&&radius>12){
    const seed=registerPcbFftPair(a,b,reference.pose,current.pose,reference.offset,current.offset,
      {cellSize:128,cellsPerAxis:4,candidateGrid:12,adaptiveCells:true,searchRadius:Math.min(63,radius),minimumPsr:6,residualLimit:3,mask});
    if(seed.accepted&&seed.inlierCells.length>=5) {
      match=measureFinePair(a,b,reference,{...current,pose:seed.pose},{radius:12,mask});
      match.currentPose=current.pose;
    }
  }
  if(match.accepted){
    const p=match.forward.pose,lever=Math.hypot(first.width,first.height)/2;
    const delta=Math.hypot(p.x-current.pose.x,p.y-current.pose.y)+Math.abs(Math.atan2(Math.sin(p.rotation-current.pose.rotation),Math.cos(p.rotation-current.pose.rotation)))*lever;
    if(delta>radius||match.fft.inlierCells.length<5){match.accepted=false;match.forward.accepted=false;match.backward.accepted=false;match.reason='Kantenkorrektur ausserhalb Bereich oder zu wenig verteilte Zellen';}
    match.fft.method='Kantenmuster';
  }
  return match;
}
