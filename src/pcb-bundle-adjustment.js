import { composePose, invertPose } from './pcb-realignment.js';

const wrap = x => Math.atan2(Math.sin(x), Math.cos(x));
const quantile = (values, fraction) => values.length ? values[Math.min(values.length - 1, Math.floor(fraction * values.length))] : 0;

export function pcbPointResiduals(nodes, network) {
  const poses = new Map(nodes.map(node => [node.frame, node.pose]));
  const fullFrames=new Set();
  const pairs = [], all = [], reliable = [], degree = new Map(nodes.map(node => [node.frame, 0]));
  for (const pair of network.pairs) {
    if (network.deletedPairs?.includes(pair.id)) continue;
    const a = poses.get(pair.reference), b = poses.get(pair.current);
    if (!a || !b || pair.cells.length < (pair.minimumCells ?? 3)) continue;
    const confirmed=[];
    const residuals = pair.cells.map(cell => {
      const p = composePose(a, {...cell.reference,rotation:0}), q = composePose(b,{...cell.current,rotation:0});
      const n=cell.normal;
      const nx=n?Math.cos(a.rotation)*n.x-Math.sin(a.rotation)*n.y:0,ny=n?Math.sin(a.rotation)*n.x+Math.cos(a.rotation)*n.y:0;
      const error=n?Math.abs((p.x-q.x)*nx+(p.y-q.y)*ny):Math.hypot(p.x-q.x,p.y-q.y);
      if((cell.confidence??1)>=.5)confirmed.push(error);
      return error;
    }).sort((a,b)=>a-b);
    all.push(...residuals);reliable.push(...confirmed);confirmed.sort((a,b)=>a-b);
    if(confirmed.length >= (pair.minimumCells??3)) { degree.set(pair.reference,degree.get(pair.reference)+1);degree.set(pair.current,degree.get(pair.current)+1); }
    if(pair.cells.filter(c=>!c.normal&&(c.confidence??1)>=.5).length>=(pair.minimumCells??3)){fullFrames.add(pair.reference);fullFrames.add(pair.current);}
    pairs.push({reference:pair.reference,current:pair.current,id:pair.id,median:quantile(residuals,.5),p90:quantile(residuals,.9),confirmedP90:quantile(confirmed,.9),cells:residuals.length,confirmedCells:confirmed.length});
  }
  all.sort((a,b)=>a-b);
  return {cells:all.length,confirmedCells:reliable.length,confirmedRms:Math.sqrt(reliable.reduce((sum,x)=>sum+x*x,0)/Math.max(1,reliable.length)),median:quantile(all,.5),p90:quantile(all,.9),p95:quantile(all,.95),p99:quantile(all,.99),
    rms:Math.sqrt(all.reduce((sum,x)=>sum+x*x,0)/Math.max(1,all.length)),
    unmeasured:[...degree].filter(([,count])=>!count).map(([frame])=>frame),directionOnlyFrames:[...degree].filter(([frame,count])=>count&&!fullFrames.has(frame)).map(([frame])=>frame),pairs:pairs.sort((a,b)=>b.p90-a.p90)};
}

// Joint SE(2) bundle adjustment of actual camera-local correspondences.
// The angular Jacobian couples rotations to point displacement, unlike three
// independent Laplacian solves for x, y and angle.
export function optimizePcbBundle(nodes, network, {iterations=20,huber=4,lever=1500,temporalWeight=.05}={}) {
  nodes=nodes.filter(node=>node.pose&&[node.pose.x,node.pose.y,node.pose.rotation].every(Number.isFinite)).sort((a,b)=>a.frame-b.frame);
  const indices=new Map(nodes.map((node,index)=>[node.frame,index])), poses=nodes.map(node=>({...node.pose}));
  const observations=[], parent=nodes.map((_,i)=>i);
  const root=i=>parent[i]===i?i:(parent[i]=root(parent[i]));
  for(const pair of network.pairs) {
    const a=indices.get(pair.reference),b=indices.get(pair.current);
    if(a===undefined||b===undefined||network.deletedPairs?.includes(pair.id)||pair.cells.length<(pair.minimumCells??3))continue;
    parent[root(b)]=root(a);
    const total=pair.cells.reduce((sum,cell)=>sum+(cell.quality??1),0);
    for(const cell of pair.cells)observations.push({a,b,p:cell.reference,q:cell.current,
      normal:cell.normal,weight:Math.min(256,pair.weight)*(cell.quality??1)*(cell.confidence??1)/total,measured:true});
  }
  // Weak relative-motion priors interpolate unmeasured frames and retain the
  // gauge of disconnected visits. They never count as image evidence.
  for(let b=1;b<nodes.length;b++) {
    const a=b-1, relative=composePose(invertPose(poses[a]),poses[b]);
    for(const q of [{x:0,y:0},{x:lever,y:0},{x:-lever,y:0},{x:0,y:lever},{x:0,y:-lever}]) {
      observations.push({a,b,p:composePose(relative,{...q,rotation:0}),q,weight:temporalWeight/5/Math.max(1,(nodes[b].frame-nodes[a].frame)/32),measured:false});
    }
  }
  if(!nodes.length)return {corrections:[],components:[],beforeRms:0,afterRms:0};
  const before=pcbPointResiduals(nodes,network), size=nodes.length*3;
  const evaluate=(values,threshold,linearize=false)=>{
    const rows=[],right=new Float64Array(size),diagonal=new Float64Array(size).fill(1e-5);let cost=0;
    for(const o of observations) {
      const a=values[o.a],b=values[o.b],ca=Math.cos(a.rotation),sa=Math.sin(a.rotation),cb=Math.cos(b.rotation),sb=Math.sin(b.rotation);
      const ax=ca*o.p.x-sa*o.p.y,ay=sa*o.p.x+ca*o.p.y,bx=cb*o.q.x-sb*o.q.y,by=sb*o.q.x+cb*o.q.y;
      const rx=b.x+bx-a.x-ax,ry=b.y+by-a.y-ay;
      if(o.normal){
        const nx=ca*o.normal.x-sa*o.normal.y,ny=sa*o.normal.x+ca*o.normal.y;
        const signed=nx*rx+ny*ry,r=Math.abs(signed),robust=r>threshold?threshold/r:1;
        cost+=o.weight*(r<=threshold?.5*r*r:threshold*(r-.5*threshold));
        if(!linearize)continue;
        const weight=o.weight*robust,ia=o.a*3,ib=o.b*3;
        const entries=[[ia,-nx],[ia+1,-ny],[ia+2,((b.y+by-a.y)*nx-(b.x+bx-a.x)*ny)/lever],
          [ib,nx],[ib+1,ny],[ib+2,(-by*nx+bx*ny)/lever]];
        rows.push({entries,weight});
        for(const [index,j]of entries){right[index]-=weight*j*signed;diagonal[index]+=weight*j*j;}
        continue;
      }
      const r=Math.hypot(rx,ry);
      const limit=o.measured?threshold:Math.max(threshold,20), robust=r>limit?limit/r:1;
      cost+=o.weight*(r<=limit?.5*r*r:limit*(r-.5*limit));
      if(!linearize)continue;
      const weight=o.weight*robust,ia=o.a*3,ib=o.b*3;
      const row={ia,ib,ax:ay/lever,ay:-ax/lever,bx:-by/lever,by:bx/lever,weight};rows.push(row);
      const entries=[[ia,-1,0],[ia+1,0,-1],[ia+2,row.ax,row.ay],[ib,1,0],[ib+1,0,1],[ib+2,row.bx,row.by]];
      for(const [index,jx,jy]of entries){right[index]-=weight*(jx*rx+jy*ry);diagonal[index]+=weight*(jx*jx+jy*jy);}
    }
    return {cost,rows,right,diagonal};
  };
  let completed=0;
  for(let iteration=0;iteration<iterations;iteration++) {
    const threshold=Math.max(huber,24*Math.pow(.65,iteration));
    const {rows,right,diagonal,cost}=evaluate(poses,threshold,true);
    for(let i=0;i<3;i++)right[i]=0;
    const multiply=v=>{
      const out=Float64Array.from(v,x=>x*1e-5);
      for(const r of rows){
        if(r.entries){const projection=r.entries.reduce((sum,[i,j])=>sum+j*v[i],0)*r.weight;
          for(const [i,j]of r.entries)out[i]+=j*projection;continue;}
        const a=r.ia,b=r.ib;
        const x=(-v[a]+v[b]+r.ax*v[a+2]+r.bx*v[b+2])*r.weight;
        const y=(-v[a+1]+v[b+1]+r.ay*v[a+2]+r.by*v[b+2])*r.weight;
        out[a]-=x;out[a+1]-=y;out[a+2]+=r.ax*x+r.ay*y;
        out[b]+=x;out[b+1]+=y;out[b+2]+=r.bx*x+r.by*y;
      }
      for(let i=0;i<3;i++)out[i]=v[i];return out;
    };
    const delta=new Float64Array(size),residual=right.slice(),z=Float64Array.from(right,(x,i)=>x/diagonal[i]),direction=z.slice();
    let rz=residual.reduce((sum,x,i)=>sum+x*z[i],0),initial=rz;
    for(let step=0;step<600&&rz>Math.max(1e-12,initial*1e-10);step++){
      const product=multiply(direction),den=direction.reduce((sum,x,i)=>sum+x*product[i],0);if(!(den>0))break;
      const alpha=rz/den;
      for(let i=3;i<size;i++){delta[i]+=alpha*direction[i];residual[i]-=alpha*product[i];z[i]=residual[i]/diagonal[i];}
      const next=residual.reduce((sum,x,i)=>sum+x*z[i],0),beta=next/rz;
      for(let i=3;i<size;i++)direction[i]=z[i]+beta*direction[i];rz=next;
    }
    let accepted=false,maximum=0;
    for(let scale=1;scale>=1/64;scale/=2){
      const next=poses.map((p,i)=>({x:p.x+scale*delta[i*3],y:p.y+scale*delta[i*3+1],rotation:wrap(p.rotation+scale*delta[i*3+2]/lever)}));
      if(evaluate(next,threshold).cost<=cost+1e-9){
        maximum=Math.max(...delta.map(x=>Math.abs(x)*scale));poses.splice(0,poses.length,...next);accepted=true;break;
      }
    }
    completed++;if(!accepted||(iteration>6&&maximum<.001))break;
  }
  const corrections=nodes.map((node,i)=>({frame:node.frame,pose:poses[i],dx:poses[i].x-node.pose.x,dy:poses[i].y-node.pose.y,rotation:wrap(poses[i].rotation-node.pose.rotation)}));
  const after=pcbPointResiduals(corrections,network);
  const groups=new Map();
  for(let i=0;i<nodes.length;i++){
    const key=root(i);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(nodes[i].frame);
  }
  const components=[...groups.values()].map(frames=>{
    const members=new Set(frames), pairs=after.pairs.filter(p=>members.has(p.reference)&&members.has(p.current));
    return {anchor:frames[0],nodes:frames.length,edges:pairs.length,
      status:pairs.length?'optimized':'interpolated',frames};
  });
  return {corrections,before,after,beforeRms:before.confirmedRms,afterRms:after.confirmedRms,iterations:completed,
    components,unmeasured:after.unmeasured,directionOnlyFrames:after.directionOnlyFrames};
}

// Preserve directional point evidence in local graph refinement as well.
export function optimizePcbPointGraph(graph,options={}) {
  const measured=graph.edges.filter(e=>e.verified||e.kind==='local-refit');
  const lever=options.lever??1000;
  const pairs=measured.map((e,i)=>{
    const points=e.currentPivot?[e.currentPivot]:[{x:0,y:0},{x:lever,y:0},{x:-lever,y:0},{x:0,y:lever},{x:0,y:-lever}];
    const cells=e.pointCells??points.map(q=>({reference:e.currentPivot?e.measurement:composePose(e.measurement,{...q,rotation:0}),current:q,quality:1}));
    return {id:`graph:${i}`,reference:e.reference,current:e.current,weight:e.weight,cells,minimumCells:e.minimumCells??1};
  });
  const actual=graph.network?{...graph.network,pairs:[...graph.network.pairs,...pairs.filter(p=>
    !graph.network.pairs.some(q=>q.cells?.length&&q.reference===Math.min(p.reference,p.current)&&q.current===Math.max(p.reference,p.current)))]}:{pairs,deletedPairs:[]};
  const result=optimizePcbBundle(graph.nodes,actual,{...options,iterations:Math.max(20,options.iterations??20),huber:Math.min(4,options.huber??4)});
  const alignedFrames=[...new Set(measured.flatMap(e=>[e.reference,e.current]))];
  return {...result,nodes:graph.nodes.length,edges:measured.length,alignedFrames,
    passiveFrames:graph.nodes.filter(n=>!alignedFrames.includes(n.frame)).map(n=>n.frame),
    localEdges:measured.length,localBeforeRms:result.beforeRms,localAfterRms:result.afterRms,
    directionalEdges:measured.filter(e=>e.pointCells?.some(c=>c.normal)).length};
}
