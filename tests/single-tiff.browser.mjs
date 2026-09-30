import http from 'node:http';
import {createReadStream} from 'node:fs';
import {stat,mkdir,writeFile} from 'node:fs/promises';
import {resolve,sep,extname} from 'node:path';
import {chromium} from 'playwright';
import assert from 'node:assert/strict';
const root=resolve('.'),dir=resolve('exports/20260918_004341-browser/gimp');await mkdir(dir,{recursive:true});
const server=http.createServer(async(req,res)=>{try{
 if(req.url==='/'){res.setHeader('Content-Type','text/html');res.end('<input id="parts" type="file" multiple>');return;}
 const file=resolve(root,'.'+req.url);if(!file.startsWith(root+sep)||!(await stat(file)).isFile())throw Error();
 res.setHeader('Content-Type',extname(file)==='.js'?'text/javascript':'application/octet-stream');createReadStream(file).pipe(res);
}catch{res.writeHead(404).end();}});
await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try {
 browser=await chromium.launch({channel:'msedge',headless:true});const page=await browser.newPage({acceptDownloads:true});
 page.on('console',m=>console.log(m.text()));await page.goto('http://127.0.0.1:'+server.address().port);
 await page.locator('#parts').setInputFiles([resolve(dir,'20260918_004341-merge-teil-001.tif'),resolve(dir,'20260918_004341-merge-teil-002.tif')]);
 const download=page.waitForEvent('download',{timeout:180000});
 const result=await page.evaluate(async()=>{
  const {beginBigTiff}=await import('/src/bigtiff-writer.js');const {createBrowserImageFile}=await import('/src/browser-image-file.js');
  const sources=[];
  for(const file of document.querySelector('#parts').files){
   const head=new DataView(await file.slice(0,65536).arrayBuffer()),entries=Number(head.getBigUint64(16,true)),tags=new Map();
   for(let i=0;i<entries;i++){const p=24+i*20;tags.set(head.getUint16(p,true),{n:Number(head.getBigUint64(p+4,true)),v:Number(head.getBigUint64(p+12,true))});}
   const array=tag=>{const {n,v}=tags.get(tag);return n===1?[v]:Array.from({length:n},(_,i)=>Number(head.getBigUint64(v+i*8,true)));};
   sources.push({file,width:tags.get(256).v,height:tags.get(257).v,tileSize:tags.get(322).v,offsets:array(324),counts:array(325)});
  }
  const width=sources[0].width,height=sources.reduce((s,p)=>s+p.height,0),tileSize=sources[0].tileSize,tiles=[];
  for(let y=0;y<height;y+=tileSize)for(let x=0;x<width;x+=tileSize)tiles.push({x,y});
  const handle=await createBrowserImageFile('20260918_004341-merge-gesamt.tif'),out=await handle.createWritable();
  const writer=await beginBigTiff(out,{width,height,tileSize,tiles});let yOffset=0,completed=0,peakHeap=0;
  for(const source of sources){
   const columns=Math.ceil(width/tileSize);
   for(let i=0;i<source.offsets.length;i++){
    const pixels=new Uint8Array(await new Response(source.file.slice(source.offsets[i],source.offsets[i]+source.counts[i]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer());
    await writer.writePixels({x:(i%columns)*tileSize,y:yOffset+Math.floor(i/columns)*tileSize},pixels);
    completed++;peakHeap=Math.max(peakHeap,performance.memory?.usedJSHeapSize||0);
    if(completed%12===0)console.log('Kachel '+completed+'/'+tiles.length);
   }yOffset+=source.height;
  }
  const layout=await writer.finish();await out.close();const file=handle.getFile();
  const a=document.createElement('a');a.href=URL.createObjectURL(file);a.download=handle.name;document.body.append(a);a.click();
  return {width,height,tiles:completed,bytes:file.size,expected:layout.fileBytes,mode:handle.mode,peakHeap};
 });
 const saved=await download;await saved.saveAs(resolve(dir,saved.suggestedFilename()));assert.equal(result.bytes,result.expected);
 await writeFile(resolve(dir,'single-browser-validation.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}finally{await browser?.close();await new Promise(r=>server.close(r));}
