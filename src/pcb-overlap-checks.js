// Audit overlapping visits even when they already belong to one graph component.
// Distance travelled separates visits without depending on video FPS or frame IDs.
export function planPcbOverlapChecks(frames, network, reach) {
  if (!(reach > 0)) return [];
  const nodes = frames.filter(n => n.pose && [n.pose.x,n.pose.y,n.pose.rotation].every(Number.isFinite))
    .slice().sort((a,b) => a.frame-b.frame);
  const distance = (a,b) => Math.hypot(a.pose.x-b.pose.x,a.pose.y-b.pose.y);
  const travelled = [0], deleted = new Set(network.deletedPairs ?? []);
  for(let i=1;i<nodes.length;i++) travelled[i]=travelled[i-1]+distance(nodes[i-1],nodes[i]);
  const regions = new Map(), spacing = reach/2;
  for(let i=0;i<nodes.length;i++) {
    const a=nodes[i]; let best=null;
    for(let j=i+1;j<nodes.length;j++) {
      const b=nodes[j],id=`${a.frame}:${b.frame}`;
      if(travelled[j]-travelled[i]<reach*2 || deleted.has(id)) continue;
      const separation=distance(a,b);
      if(separation>=reach || (best && separation>=best.separation)) continue;
      best={reference:a.frame,current:b.frame,kind:'overlap-check',separation,
        x:(a.pose.x+b.pose.x)/2,y:(a.pose.y+b.pose.y)/2};
    }
    if(!best) continue;
    const region=`${Math.floor(best.x/spacing)}:${Math.floor(best.y/spacing)}`;
    if(!regions.has(region) || best.separation<regions.get(region).separation) regions.set(region,best);
  }
  return [...regions.values()].sort((a,b)=>a.reference-b.reference||a.current-b.current)
    .map(({reference,current,kind})=>({reference,current,kind}));
}
