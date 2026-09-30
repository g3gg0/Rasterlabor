import {composePose,invertPose} from './pcb-realignment.js';
import {maskIncludes} from './patch-mask.js';

const rot=(p,a)=>({x:Math.cos(a)*p.x-Math.sin(a)*p.y,y:Math.sin(a)*p.x+Math.cos(a)*p.y});
const world=(item,x,y)=>composePose(item.pose,{x:x+item.offset[0],y:y+item.offset[1],rotation:0});
const local=(item,p)=>{const q=composePose(invertPose(item.pose),{...p,rotation:0});return {x:q.x-item.offset[0],y:q.y-item.offset[1]};};
function pixel(image,item,p,mask) {
  const q=local(item,p),x=Math.round(q.x),y=Math.round(q.y);
  if(x<2||y<2||x>=image.width-2||y>=image.height-2||mask&&!maskIncludes(mask,x,y))return null;
  const i=(y*image.width+x)*4,d=image.data;
  if(d[i+3]!==255)return null;
  return [d[i],d[i+1],d[i+2]];
}
function profile(image,item,p,n,color,lo,hi,mask){
  const values=[];let support=0;
  for(let t=lo-3;t<=hi+3;t++){
    let sum=0;
    for(let u=-24;u<=24;u+=4){
      const v=pixel(image,item,{x:p.x+t*n.x-u*n.y,y:p.y+t*n.y+u*n.x},mask);
      if(!v)return null;
      sum+=v[0]*color[0]+v[1]*color[1]+v[2]*color[2];
    }
    values.push(sum/13);
  }
  const gradient=values.slice(3,-3).map((_,i)=>(values[i+6]-values[i])/6);
  for(const v of gradient)if(Math.abs(v)>2)support+=13;
  return {gradient,support};
}
function shiftProfile(a,b,radius){
  const n=a.length,mean=a.reduce((s,v)=>s+v,0)/n,av=a.map(v=>v-mean),aa=av.reduce((s,v)=>s+v*v,0);
  if(aa<n*.1)return null;
  const scores=[];let best=-1,at=0;
  for(let d=-radius;d<=radius;d++){
    let sum=0,sq=0,dot=0;
    for(let i=0;i<n;i++){const v=b[i+d+radius];sum+=v;sq+=v*v;dot+=av[i]*v;}
    const score=dot/Math.sqrt(aa*Math.max(1e-9,sq-sum*sum/n));scores.push(score);
    if(score>best){best=score;at=d;}
  }
  const other=Math.max(...scores.filter((_,i)=>Math.abs(i-radius-at)>=4));
  if(best<.95||best-other<.035||Math.abs(at)===radius)return {accepted:false,score:best,margin:best-other,shift:at};
  const i=at+radius,den=scores[i-1]-2*scores[i]+scores[i+1];
  const sub=den<0?Math.max(-.5,Math.min(.5,.5*(scores[i-1]-scores[i+1])/den)):0;
  return {accepted:true,shift:at+sub,score:best,margin:best-other};
}
function solve(cells,current,lever){
  const A=Array.from({length:3},()=>[0,0,0]),b=[0,0,0];
  for(const c of cells){
    const q={x:c.center.x+c.shift*c.normal.x-current.pose.x,y:c.center.y+c.shift*c.normal.y-current.pose.y};
    const row=[c.normal.x,c.normal.y,(-q.y*c.normal.x+q.x*c.normal.y)/lever];
    for(let i=0;i<3;i++){b[i]-=row[i]*c.shift;for(let j=0;j<3;j++)A[i][j]+=row[i]*row[j];}
  }
  // Null directions retain their current value; a line never invents tangent motion.
  for(let i=0;i<3;i++)A[i][i]+=1e-6;
  for(let k=0;k<3;k++){
    let pivot=k;for(let i=k+1;i<3;i++)if(Math.abs(A[i][k])>Math.abs(A[pivot][k]))pivot=i;
    [A[k],A[pivot]]=[A[pivot],A[k]];[b[k],b[pivot]]=[b[pivot],b[k]];
    const v=A[k][k];for(let j=k;j<3;j++)A[k][j]/=v;b[k]/=v;
    for(let i=0;i<3;i++)if(i!==k){const f=A[i][k];for(let j=k;j<3;j++)A[i][j]-=f*A[k][j];b[i]-=f*b[k];}
  }
  return {x:b[0],y:b[1],rotation:b[2]/lever};
}
const residual=(cell,delta,current)=>{
  const q={x:cell.center.x+cell.shift*cell.normal.x-current.pose.x,y:cell.center.y+cell.shift*cell.normal.y-current.pose.y};
  const r=rot(q,delta.rotation);
  return Math.abs((r.x-q.x+delta.x)*cell.normal.x+(r.y-q.y+delta.y)*cell.normal.y+cell.shift);
};

export function measureEdgeConstraints(first,second,reference,current,{radius=64,mask=null}={}){
  radius=Math.max(2,Math.min(64,Math.floor(radius)));
  const candidates=[],tile=96,lever=Math.hypot(first.width,first.height)/2;
  // One strongest interior color edge per tile. No alpha/mask boundary is evidence.
  for(let top=24;top<first.height-24;top+=tile)for(let left=24;left<first.width-24;left+=tile){
    let best=null;
    for(let y=top;y<Math.min(top+tile,first.height-24);y+=6)for(let x=left;x<Math.min(left+tile,first.width-24);x+=6){
      const p=world(reference,x,y);
      if(!pixel(second,current,p,mask))continue;
      const a=pixel(first,reference,world(reference,x-2,y),mask),b=pixel(first,reference,world(reference,x+2,y),mask);
      const c=pixel(first,reference,world(reference,x,y-2),mask),d=pixel(first,reference,world(reference,x,y+2),mask);
      if(!a||!b||!c||!d)continue;
      const gx=b.map((v,i)=>v-a[i]),gy=d.map((v,i)=>v-c[i]);
      let channel=0;for(let i=1;i<3;i++)if(gx[i]**2+gy[i]**2>gx[channel]**2+gy[channel]**2)channel=i;
      const strength=Math.hypot(gx[channel],gy[channel]);
      if(strength<6||best&&strength<=best.strength)continue;
      const n={x:gx[channel]/strength,y:gy[channel]/strength},color=gx.map((v,i)=>v*n.x+gy[i]*n.y),length=Math.hypot(...color);
      best={center:p,normal:rot(n,reference.pose.rotation),color:color.map(v=>v/length),strength};
    }
    if(best)candidates.push(best);
  }
  const cells=[], failures={profiles:0,forward:0,backward:0,peaks:[]};
  for(const c of candidates.sort((a,b)=>b.strength-a.strength).slice(0,120)){
    const a=profile(first,reference,c.center,c.normal,c.color,-40,40,mask);
    const b=profile(second,current,c.center,c.normal,c.color,-40-radius,40+radius,mask);
    if(!a||!b){failures.profiles++;continue;}
    const m=shiftProfile(a.gradient,b.gradient,radius);if(!m?.accepted){failures.forward++;failures.peaks.push(m);continue;}
    const q={x:c.center.x+m.shift*c.normal.x,y:c.center.y+m.shift*c.normal.y};
    const backA=profile(second,current,q,c.normal,c.color,-40,40,mask);
    const backB=profile(first,reference,q,c.normal,c.color,-40-radius,40+radius,mask);
    const back=backA&&backB?shiftProfile(backA.gradient,backB.gradient,radius):null;
    if(!back?.accepted||Math.abs(back.shift+m.shift)>1){failures.backward++;continue;}
    cells.push({...c,...m,cycle:Math.abs(back.shift+m.shift),support:Math.min(a.support,backA.support)});
  }
  const rejected=reason=>({reference:reference.frame,current:current.frame,referencePose:reference.pose,currentPose:current.pose,accepted:false,reason,
    fft:{accepted:false,method:'Gerichtete Kanten',cells:[]},edgeDiagnostics:{candidates:candidates.length,measured:cells.length,failures}});
  if(cells.length<6)return rejected('Zu wenig eindeutige Kantenprofile');
  let best=[];
  for(let i=0;i<Math.min(200,cells.length*8);i++){
    const picks=[cells[i%cells.length],cells[(i*7+1)%cells.length],cells[(i*13+3)%cells.length]];
    const delta=solve(picks,current,lever);
    if(Math.hypot(delta.x,delta.y)+Math.abs(delta.rotation)*lever>radius*1.5)continue;
    const inliers=cells.filter(c=>residual(c,delta,current)<=1.5);
    if(inliers.length>best.length)best=inliers;
  }
  if(best.length<6||best.length<cells.length*.6)return rejected('Kein gemeinsamer Kantenversatz');
  const delta=solve(best,current,lever),rms=Math.sqrt(best.reduce((s,c)=>s+residual(c,delta,current)**2,0)/best.length);
  const span=Math.hypot(Math.max(...best.map(c=>c.center.x))-Math.min(...best.map(c=>c.center.x)),Math.max(...best.map(c=>c.center.y))-Math.min(...best.map(c=>c.center.y)));
  if(span<300||rms>1.5)return rejected('Kantenbedingung raeumlich nicht ausreichend bestaetigt');
  const pose={x:current.pose.x+delta.x,y:current.pose.y+delta.y,rotation:current.pose.rotation+delta.rotation};
  const score=best.reduce((s,c)=>s+c.score,0)/best.length,support=best.reduce((s,c)=>s+c.support,0);
  const pointPairs=best.map((c,i)=>({cellId:i,reference:composePose(invertPose(reference.pose),{...c.center,rotation:0}),
    current:composePose(invertPose(current.pose),{x:c.center.x+c.shift*c.normal.x,y:c.center.y+c.shift*c.normal.y,rotation:0}),
    normal:rot(c.normal,-reference.pose.rotation),psr:20*c.score,support:c.support}));
  return {reference:reference.frame,current:current.frame,referencePose:reference.pose,currentPose:current.pose,accepted:true,partial:true,score,
    reason:null,reverseDistance:Math.max(...best.map(c=>c.cycle)),forward:{accepted:true,pose,score,support},backward:{accepted:true,score,support},
    fft:{accepted:true,pose,method:'Gerichtete Kanten',cellSize:96,residualRms:rms,uniqueSupportArea:support,
      pointPairs,inlierCells:best.map((_,i)=>i),cells:best.map((c,i)=>({cellId:i,center:c.center,dx:c.shift*c.normal.x,dy:c.shift*c.normal.y,accepted:true,normal:c.normal}))},
    edgeDiagnostics:{candidates:candidates.length,measured:cells.length,inliers:best.length,span}};
}

export function revalidateEdgeCell(first,second,reference,current,cell){
  const p=composePose(reference.pose,{...cell.reference,rotation:0}),q=composePose(current.pose,{...cell.current,rotation:0});
  const n=rot(cell.normal,reference.pose.rotation);
  const a=pixel(first,reference,{x:p.x-12*n.x,y:p.y-12*n.y},null),b=pixel(first,reference,{x:p.x+12*n.x,y:p.y+12*n.y},null);
  let measured=null;
  if(a&&b){const delta=b.map((v,i)=>v-a[i]),length=Math.hypot(...delta);
    if(length>8){const color=delta.map(v=>v/length),pa=profile(first,reference,p,n,color,-40,40,null),pb=profile(second,current,q,n,color,-46,46,null);
      if(pa&&pb)measured=shiftProfile(pa.gradient,pb.gradient,6);}
  }
  return {id:cell.id,score:measured?.score??null,confidence:measured?.accepted?1:.05,
    reason:measured?.accepted?null:'Kantenprofil derzeit nicht eindeutig bestaetigt'};
}
