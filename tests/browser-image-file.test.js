import test from 'node:test';
import assert from 'node:assert/strict';
import { inflateSync } from 'node:zlib';
import { createBrowserImageFile } from '../src/browser-image-file.js';
import { beginBigTiff } from '../src/bigtiff-writer.js';
import { planTiffParts, tiffJoinCommand } from '../src/tiff-parts.js';

test('embedded-browser download supports appended pixels and rewritten header without a full image buffer', async()=>{
 const file=await createBrowserImageFile('a.tif','image/tiff',{storage:null});const out=await file.createWritable();
 await out.write({position:0,data:new Uint8Array([1,2,3,4])});
 await out.write({position:8,data:new Uint8Array([9,10])});
 await out.write({position:1,data:new Uint8Array([20,30])});
 await out.truncate(9);await out.close();
 assert.deepEqual([...new Uint8Array(await file.getFile().arrayBuffer())],[1,20,30,4,0,0,0,0,9]);
 await assert.rejects(out.write({position:0,data:new Uint8Array(1)}),/geschlossen/);await file.dispose();
});

test('compressed TIFF decodes every byte including alpha and refuses incomplete or duplicate tiles',async()=>{
 const file=await createBrowserImageFile('a.tif','image/tiff',{storage:null}),out=await file.createWritable();
 const writer=await beginBigTiff(out,{width:32,height:16,tileSize:16,tiles:[{x:0,y:0},{x:16,y:0}]});
 const pixels=Uint8Array.from({length:16*16*4},(_,i)=>(i*7)%256);
 await writer.writePixels({x:0,y:0},pixels);
 await assert.rejects(writer.finish(),/unvollstaendig/);
 await assert.rejects(writer.writePixels({x:0,y:0},pixels),/bereits/);
 await writer.writePixels({x:16,y:0},pixels);const layout=await writer.finish();await out.close();
 const bytes=new Uint8Array(await file.getFile().arrayBuffer());
 assert.equal(bytes.length,layout.fileBytes);assert.ok(bytes.length<2048);
 for(let i=0;i<2;i++) {
   const start=Number(layout.offsets[i]),length=Number(layout.byteCounts[i]);
   assert.equal(start%8,0);assert.deepEqual(new Uint8Array(inflateSync(bytes.subarray(start,start+length))),pixels);
 }
 assert.equal(layout.compression,8);await file.dispose();
});

test('TIFF strips cover the original image exactly and respect the conservative file limit',()=>{
 const tiles=[];for(let y=0;y<15145;y+=2048)for(let x=0;x<16521;x+=2048)tiles.push({x,y});
 const mosaic={width:16521,height:15145,tileSize:2048,tiles};
 const two=planTiffParts(mosaic,'board.tif','two');assert.equal(two.length,2);
 assert.equal(two[0].height+two[1].height,15145);assert.equal(two[1].y,two[0].height);
 assert.equal(two[1].tiles[0].y,0);assert.equal(two[0].width,two[1].width);
 const limited=planTiffParts(mosaic,'board.tif','512');
 for(const part of limited) assert.ok(part.tiles.length*2048**2*4*1.001+1048576<=512*1024**2);
 const command=tiffJoinCommand(two,16521,15145);assert.match(command,/vips join 'board-teil-001.tif' 'board-teil-002.tif'/);
 assert.match(command,/vertical/);assert.equal(planTiffParts(mosaic,'board.tif','0').length,1);
});


test('finished TIFF contains explicit transparent tiles for GIMP, including empty tile rows', async()=>{
 const file=await createBrowserImageFile('gimp.tif','image/tiff',{storage:null}),out=await file.createWritable();
 const writer=await beginBigTiff(out,{width:32,height:48,tileSize:16,tiles:[{x:0,y:0},{x:0,y:32}]});
 const pixels=new Uint8Array(1024).fill(255);
 await writer.writePixels({x:0,y:0},pixels);await writer.writePixels({x:0,y:32},pixels);
 const layout=await writer.finish();await out.close();const bytes=new Uint8Array(await file.getFile().arrayBuffer());
 for(let i=0;i<6;i++){
  assert.ok(layout.offsets[i]>0n);assert.ok(layout.byteCounts[i]>0n);
  const start=Number(layout.offsets[i]),end=start+Number(layout.byteCounts[i]);
  const decoded=new Uint8Array(inflateSync(bytes.subarray(start,end)));
  assert.deepEqual(decoded,i===0||i===4?pixels:new Uint8Array(1024));
 }
 await file.dispose();
});
