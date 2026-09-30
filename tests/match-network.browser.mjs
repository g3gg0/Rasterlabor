import assert from 'node:assert/strict';
import http from 'node:http';
import {readFile,writeFile} from 'node:fs/promises';
import {resolve,extname,sep} from 'node:path';
import {chromium} from 'playwright';
const root=resolve('dist');
const hook=`
globalThis.networkProbe={
 prepare(){
  calibration={field:{width:240,height:240,nx:2,ny:2,coefficients:new Float64Array([1,2])},
    maps:{outputWidth:240,outputHeight:240,origin:[-120,-120],valid:new Uint8Array(240*240).fill(1)}};
  trackingPath=[1,2,3,4].map((frame,index)=>({frame,timestamp:frame,mode:'window',success:true,
    pose:{x:index<2?index*10:2000+(index-2)*10,y:0,rotation:0},rectangle:{x:0,y:0,width:240,height:240}}));
  trackingDataset={format:'rasterlabor-xyr-tracking',model_version:1,path:trackingPath};
  this.matches=[this.match(1,2),this.match(3,4)];
  storeNetworkMatches(this.matches,{cycleLimit:5});
  pathSelection={x:0,y:0};pathHover=null;drawTrackingPath();
 },
 match(reference,current){
  const referencePose=trackingPath.find(e=>e.frame===reference).pose;
  const currentPose=trackingPath.find(e=>e.frame===current).pose;
  return {reference,current,referencePose,currentPose,forward:{accepted:true,pose:currentPose,score:.99,support:8192},
    backward:{accepted:true,pose:referencePose,score:.99,support:8192},reverseDistance:.5,
    fft:{accepted:true,cellSize:32,inlierCells:[0,1,2,3],uniqueSupportArea:8192,residualRms:.5,
      pointPairs:[[-60,-60],[60,-60],[-60,60],[60,60]].map(([x,y],cellId)=>({cellId,reference:{x,y},current:{x:x-10,y},psr:24,support:2048}))}};
 },
 state(){return {hitTargets:networkHitTargets.length,pairs:currentMatchNetwork().pairs.map(p=>({id:p.id,cells:p.cells.length})),
  visible:networkVisiblePairs.map(p=>p.id),edges:networkEdges(currentMatchNetwork()).map(e=>[e.reference,e.current]),
  deletedPairs:currentMatchNetwork().deletedPairs,deletedCells:currentMatchNetwork().deletedCells.length,
  selected:{pair:networkSelectedPair,cell:networkSelectedCell},status:element('trackingNetworkInfo').textContent,
  poses:trackingPath.map(e=>e.pose)};},
 target(){return networkHitTargets.find(t=>t.cellId!==null);},
 reload(){const value=JSON.parse(JSON.stringify(trackingForExport()));restoreTracking(value);
  trackingMosaicRequest++;clearTimeout(trackingMosaicTimer);pathSelection={x:0,y:0};drawTrackingPath();return this.state();},
 repeat(){storeNetworkMatches([this.match(1,2),this.match(3,4)],{cycleLimit:5});drawTrackingPath();return this.state();},
 async globalRun(){
  setWebGpuSelection('none');
  const size=192,canvas=new OffscreenCanvas(size,size),ctx=canvas.getContext('2d');const data=ctx.createImageData(size,size);
  for(let y=0;y<size;y++)for(let x=0;x<size;x++){
    const trace=Math.abs(y-38-8*Math.sin(x/13))<2||Math.abs(x-95-9*Math.sin(y/12))<2||Math.abs(y-128+7*Math.sin(x/8))<2;
    const value=70+(trace?100:0)+8*Math.sin(x*.35+y*.17);const i=4*(y*size+x);
    data.data[i]=data.data[i+1]=data.data[i+2]=value;data.data[i+3]=255;
  }ctx.putImageData(data,0,0);
  frameReader.read=async()=>({bitmap:await createImageBitmap(canvas)});
  videoInfo={name:'fixture.mp4',width:size,height:size,frameCount:20};
  calibration={field:{width:size,height:size},maps:{outputWidth:size,outputHeight:size,origin:[-96,-96],valid:new Uint8Array(size*size).fill(1)}};
  trackingPath=[10,11,12].map((frame,index)=>({frame,timestamp:frame,mode:'window',success:true,pose:{x:index*6,y:0,rotation:0}}));
  trackingDataset={format:'rasterlabor-xyr-tracking',model_version:1,path:trackingPath};
  pathSelection=null;pathHover=null;overlayContributors=[];pcbProposal=null;pathRefitProposal=null;
  for(const[id,value]of Object.entries({pcbSpacing:.02,pcbFftCellSize:64,pcbBridgeBudget:0,pcbLocalBudget:0,pcbMaxPairs:16,pcbSearchRadius:20}))element(id).value=value;
  await runPcbRealignment();
  trackingMosaicRequest++;clearTimeout(trackingMosaicTimer);
  return {proposal:Boolean(pcbProposal),network:this.state(),status:element('pcbRealignStatus').textContent};
 },
 realView(){element('trackingNetworkAll').checked=true;pathSelection={x:-12025.7,y:8951.2};drawTrackingPath();return this.state();},
 move(){trackingPath[0].pose.x+=100;trackingPath[0].pose.rotation=.1;drawTrackingPath();return this.state();}
};`;
const server=http.createServer(async(req,res)=>{try{const file=resolve(root,'.'+(req.url==='/'?'/index.html':req.url));
 if(!file.startsWith(root+sep))throw Error();let b=await readFile(file);if(req.url==='/app.js')b=Buffer.concat([b,Buffer.from(hook)]);
 res.setHeader('Content-Type',{'.html':'text/html','.js':'text/javascript','.css':'text/css'}[extname(file)]??'application/octet-stream');res.end(b);
}catch{res.writeHead(404).end();}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try{
 browser=await chromium.launch({channel:'msedge',headless:true});const page=await browser.newPage({viewport:{width:1440,height:1080}});
 const errors=[];page.on('pageerror',error=>errors.push(error.message));await page.goto('http://127.0.0.1:'+server.address().port);
 await page.locator('[data-workflow="tracking"]').click();await page.evaluate(()=>networkProbe.prepare());
 let state=await page.evaluate(()=>networkProbe.state());assert.deepEqual(state.visible,['1:2']);assert.equal(state.pairs.length,2);
 await page.locator('#trackingNetworkAll').check();assert.equal((await page.evaluate(()=>networkProbe.state())).visible.length,2);
 await page.locator('#trackingNetworkAll').uncheck();
 const target=await page.evaluate(()=>networkProbe.target());const box=await page.locator('#trackingPathCanvas').boundingBox();
 await page.mouse.click(box.x+target.a.x,box.y+target.a.y);
 state=await page.evaluate(()=>networkProbe.state());assert.equal(state.selected.pair,'1:2');assert.ok(state.selected.cell);
 await page.screenshot({path:'benchmarks/match-network-ui.png',fullPage:true});
 const poses=state.poses;
 await page.locator('#trackingNetworkDeleteCell').click();state=await page.evaluate(()=>networkProbe.state());
 assert.equal(state.pairs[0].cells,3);assert.equal(state.deletedCells,1);assert.deepEqual(state.poses,poses);
 await page.locator('#trackingNetworkUndo').click();state=await page.evaluate(()=>networkProbe.state());assert.equal(state.pairs[0].cells,4);
 await page.locator('#trackingNetworkPair').selectOption('1:2');await page.locator('#trackingNetworkCell').selectOption({index:1});
 await page.locator('#trackingNetworkDeleteCell').click();state=await page.evaluate(()=>networkProbe.reload());assert.equal(state.pairs[0].cells,3);
 state=await page.evaluate(()=>networkProbe.repeat());assert.equal(state.pairs[0].cells,3);
 await page.locator('#trackingNetworkPair').selectOption('1:2');await page.locator('#trackingNetworkDeletePair').click();
 state=await page.evaluate(()=>networkProbe.reload());assert.deepEqual(state.deletedPairs,['1:2']);assert.equal(state.pairs.length,1);
 state=await page.evaluate(()=>networkProbe.repeat());assert.equal(state.pairs.length,1);assert.deepEqual(state.edges,[[3,4]]);
 await page.locator('#trackingNetworkAll').check();await page.locator('#trackingNetworkShow').uncheck();
 assert.equal(await page.evaluate(()=>networkProbe.state().hitTargets),0);
 await page.locator('#trackingNetworkShow').check();await page.screenshot({path:'benchmarks/match-network-ui-deleted.png',fullPage:true});
 assert.deepEqual(errors,[]);await writeFile('benchmarks/match-network-ui.json',JSON.stringify({state,errors},null,2));
 console.log(JSON.stringify({state,errors}));
 const global=await page.evaluate(()=>networkProbe.globalRun());
 assert.equal(global.proposal,true,global.status);assert.ok(global.network.pairs.length>=2);
 await page.evaluate(()=>document.querySelector('#pcbRealignDiscard').click());
 let retained=await page.evaluate(()=>networkProbe.state());assert.equal(retained.pairs.length,global.network.pairs.length);
 retained=await page.evaluate(()=>networkProbe.reload());assert.equal(retained.pairs.length,global.network.pairs.length);
 console.log(JSON.stringify({global:{pairs:global.network.pairs,status:global.status,retained:retained.pairs.length}}));
 if(process.argv.includes('--real')){
  const realPage=await browser.newPage({viewport:{width:1440,height:1080}});realPage.on('pageerror',error=>errors.push(error.message));
  await realPage.goto('http://127.0.0.1:'+server.address().port);
  const archiveIndex=process.argv.indexOf('--archive');
  await realPage.locator('#calibrationFile').setInputFiles(archiveIndex>=0?process.argv[archiveIndex+1]:'kalibrierung-v7-provisional______.zip');
  await realPage.waitForFunction(()=>!document.querySelector('#exportButton').disabled);
  await realPage.locator('[data-workflow="tracking"]').click();
  const real=await realPage.evaluate(()=>networkProbe.realView());assert.ok(real.pairs.length>0);assert.ok(real.hitTargets>0);
  await realPage.screenshot({path:'benchmarks/match-network-real.png',fullPage:true});
  console.log(JSON.stringify({real:{pairs:real.pairs.length,points:real.pairs.reduce((sum,p)=>sum+p.cells,0),hitTargets:real.hitTargets}}));
 }
 assert.deepEqual(errors,[]);
}finally{await browser?.close();await new Promise(r=>server.close(r));}
