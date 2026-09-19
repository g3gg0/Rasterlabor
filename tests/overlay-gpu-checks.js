import { WebGpuOverlay } from '../src/webgpu-overlay.js';
import { orientationFromMatrix } from '../src/video-orientation.js';
import { edgeFeatherMask, applyEdgeFeather, accumulateFrame, averagedFrames, sharpestFramesFirst } from '../src/path-support.js';
import { renderTiledOverlay, closeOverlayTiles } from '../src/overlay-tiles.js';
import { measureFrameSharpness } from '../src/sharpness.js';
import { measureFrameSharpnessGpu } from '../src/webgpu-sharpness.js';

function assert(value, message) { if (!value) throw new Error(message); }
async function pixels(overlay) {
  const bitmap = await overlay.finish();
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  canvas.getContext('2d').drawImage(bitmap, 0, 0); bitmap.close();
  return canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
}
const identity = { c: 1, s: 0, world: (x, y) => ({ x, y }) };
function compare(a, b) {
  let maximum = 0, sum = 0, count = 0, alphaDifferences = 0;
  for (let i = 0; i < a.length; i += 4) {
    if (a[i + 3] !== b[i + 3]) alphaDifferences++;
    if (!a[i + 3] || !b[i + 3]) continue;
    for (let k = 0; k < 3; k++) { const delta = Math.abs(a[i + k] - b[i + k]); maximum = Math.max(maximum, delta); sum += delta; count++; }
  }
  return { maximum, mean: sum / (count || 1), alphaDifferences, comparedChannels: count };
}

export async function runTopNOverlayChecks() {
  const maps = { outputWidth: 4, outputHeight: 3, inverseX: Float32Array.from({ length: 12 }, (_, index) => index % 4),
    inverseY: Float32Array.from({ length: 12 }, (_, index) => Math.floor(index / 4)), valid: new Uint8Array(12).fill(1) };
  maps.valid[5] = 0;
  const orientation = orientationFromMatrix([65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824], 5, 4);
  const geometries = [0, 1, 2].map(frame => ({ c: 1, s: 0, entry: { frame, sharpness: { score: frame * 10 } },
    world: (x, y) => ({ x: x + frame, y }), corners: [{ x: frame, y: 0 }, { x: frame + 4, y: 0 }, { x: frame + 4, y: 3 }, { x: frame, y: 3 }] }));
  const frames = geometries.map(geometry => {
    const canvas = new OffscreenCanvas(5, 4), context = canvas.getContext('2d');
    context.fillStyle = `rgb(${60 + geometry.entry.frame * 70}, 40, 90)`; context.fillRect(0, 0, 5, 4);
    return new VideoFrame(canvas, { timestamp: geometry.entry.frame });
  });
  const results = [];
  try {
    for (const maxFrames of [1, 2, 0]) {
      const sum = new Float32Array(6 * 3 * 4), counts = new Uint32Array(18);
      for (const geometry of sharpestFramesFirst(geometries)) {
        const rgba = new Uint8ClampedArray(sum.length);
        for (let row = 0; row < 3; row++) for (let column = 0; column < 4; column++) {
          if (maps.valid[row * 4 + column]) rgba.set([60 + geometry.entry.frame * 70, 40, 90, 255], (row * 6 + column + geometry.entry.frame) * 4);
        }
        accumulateFrame(sum, rgba, counts, maxFrames || Infinity);
      }
      const expected = averagedFrames(sum);
      const overlay = await WebGpuOverlay.create(maps, 6, 3, 0, 0, null, maxFrames);
      try {
        for (let repeat = 0; repeat < 2; repeat++) {
          await overlay.clear(0, 0);
          for (const geometry of sharpestFramesFirst(geometries)) await overlay.addFrame(frames[geometry.entry.frame], orientation, geometry);
          const diff = compare(await pixels(overlay), expected);
          assert(diff.maximum <= 1 && !diff.alphaDifferences, `Top ${maxFrames} / clear ${repeat}: ${JSON.stringify(diff)}`);
        }
      } finally { overlay.destroy(); }
      const tiled = await renderTiledOverlay({ maps, width: 6, height: 3, minX: 0, minY: 0, geometries, tileSize: 2, maxFrames,
        decode: async index => ({ frame: frames[index].clone(), orientation }) });
      try {
        const canvas = new OffscreenCanvas(6, 3), context = canvas.getContext('2d');
        for (const tile of tiled.tiles) context.drawImage(tile.bitmap, 0, 0, tile.width, tile.height, tile.x, tile.y, tile.width, tile.height);
        const diff = compare(context.getImageData(0, 0, 6, 3).data, expected);
        assert(diff.maximum <= 1 && !diff.alphaDifferences, `Tiled top ${maxFrames}: ${JSON.stringify(diff)}`);
        results.push({ maxFrames, ...diff });
      } finally { closeOverlayTiles(tiled); }
    }
    const width = 20, height = 20;
    const featherMaps = { outputWidth: width, outputHeight: height,
      inverseX: Float32Array.from({ length: width * height }, (_, index) => index % width),
      inverseY: Float32Array.from({ length: width * height }, (_, index) => Math.floor(index / width)),
      valid: new Uint8Array(width * height).fill(1) };
    for (let y = 10; y < height; y++) for (let x = 10; x < width; x++) featherMaps.valid[y * width + x] = 0;
    const canvas = new OffscreenCanvas(width, height), context = canvas.getContext('2d');
    context.fillStyle = 'rgb(120, 80, 40)'; context.fillRect(0, 0, width, height);
    const frame = new VideoFrame(canvas, { timestamp: 0 });
    const feathered = new Uint8ClampedArray(width * height * 4);
    for (let index = 0; index < width * height; index++) feathered.set([120, 80, 40, 255], index * 4);
    const featherFraction = 0.25;
    applyEdgeFeather(feathered, width, height, featherFraction,
      edgeFeatherMask(width, height, (x, y) => Boolean(featherMaps.valid[y * width + x]), featherFraction));
    const expectedSum = new Float32Array(feathered.length);
    accumulateFrame(expectedSum, feathered);
    const featherExpected = averagedFrames(expectedSum);
    const featherOrientation = orientationFromMatrix([65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824], width, height);
    const overlay = await WebGpuOverlay.create(featherMaps, width, height, 0, 0, null, 0, featherFraction);
    try {
      await overlay.addFrame(frame, featherOrientation, identity);
      const actual = await pixels(overlay);
      const diff = compare(actual, featherExpected);
      const maximumAlpha = Math.max(...actual.filter((_, index) => index % 4 === 3).map((value, index) => Math.abs(value - featherExpected[index * 4 + 3])));
      assert(diff.maximum <= 3 && maximumAlpha <= 1, `Edge feather GPU parity: ${JSON.stringify({ diff, maximumAlpha })}`);
      results.push({ edgeFeather: true, maximumAlpha, ...diff });
    } finally { overlay.destroy(); frame.close(); }
    return results;
  } finally { for (const frame of frames) frame.close(); }
}

export async function runGpuChecks(decoder, computer, calibration) {
  const result = { orientations: [] };
  const sharpCanvas=new OffscreenCanvas(192,108),sharpContext=sharpCanvas.getContext('2d');
  const sharpPixels=sharpContext.createImageData(192,108);
  for(let y=0;y<108;y++)for(let x=0;x<192;x++){
    const value=(Math.floor(x/3)+Math.floor(y/3))%2?230:20;
    sharpPixels.data.set([value,value,value,255],(y*192+x)*4);
  }
  sharpContext.putImageData(sharpPixels,0,0);
  const sharpFrame=new VideoFrame(sharpCanvas,{timestamp:0});
  try {
    const cpu=await measureFrameSharpness(sharpFrame,false),gpu=await measureFrameSharpnessGpu(sharpFrame);
    assert(gpu&&Math.abs(gpu.score-cpu.score)/cpu.score<0.02,`Sharpness GPU parity: ${cpu.score} / ${gpu?.score}`);
    result.sharpness={cpu:cpu.score,gpu:gpu.score,relativeDifference:Math.abs(gpu.score-cpu.score)/cpu.score};
  } finally {sharpFrame.close();}
  for (const axes of [[1,0,0,1], [0,1,-1,0], [-1,0,0,-1], [0,-1,1,0], [-1,0,0,1], [1,0,0,-1], [0,1,1,0], [0,-1,-1,0]]) {
    const [a,b,c,d] = axes;
    const orientation = orientationFromMatrix([a*65536,b*65536,0,c*65536,d*65536,0,0,0,1073741824],9,7);
    const canvas = new OffscreenCanvas(9, 7), ctx = canvas.getContext('2d');
    const input = ctx.createImageData(9, 7);
    for (let y=0;y<7;y++) for(let x=0;x<9;x++) input.data.set([x*23,y*31,(x*13+y*17)%256,255],(y*9+x)*4);
    ctx.putImageData(input,0,0);
    const frame = new VideoFrame(canvas,{timestamp:0});
    const rotated = new OffscreenCanvas(orientation.width,orientation.height), rc = rotated.getContext('2d');
    rc.setTransform(a,b,c,d,orientation.translateX,orientation.translateY);rc.drawImage(frame,0,0);
    const rgb = rc.getImageData(0,0,rotated.width,rotated.height).data;
    const maps = {outputWidth:4,outputHeight:3,inverseX:new Float32Array(12),inverseY:new Float32Array(12),valid:new Uint8Array(12).fill(1)};
    maps.valid[2]=0;
    const expected = new Uint8ClampedArray(48);
    for(let i=0;i<12;i++) {
      const x=i%4+0.25,y=Math.floor(i/4)+0.375; maps.inverseX[i]=x;maps.inverseY[i]=y;
      if(!maps.valid[i]||i===9)continue;
      const index=(Math.floor(y)*rotated.width+Math.floor(x))*4;
      for(let k=0;k<3;k++)expected[i*4+k]=0.625*(rgb[index+k]*0.75+rgb[index+4+k]*0.25)+0.375*(rgb[index+rotated.width*4+k]*0.75+rgb[index+rotated.width*4+4+k]*0.25);
      expected[i*4+3]=255;
    }
    const overlay=await WebGpuOverlay.create(maps,4,3,0,0,(x,y)=>y*4+x!==9);
    try {
      await overlay.addFrame(frame,orientation,identity);
      const diff=compare(await pixels(overlay),expected);
      assert(diff.maximum<=1&&diff.alphaDifferences===0,`Orientation ${axes}: ${JSON.stringify(diff)}`);
      result.orientations.push({axes,...diff});
      if(a===1&&d===1) {
        for(let i=0;i<300;i++)await overlay.addFrame(frame,orientation,identity);
        const repeated=compare(await pixels(overlay),expected);
        assert(repeated.maximum<=1&&repeated.alphaDifferences===0,'Float accumulation changed repeated frame');
        result.accumulatedFrames=301;
        const geometry={...identity,entry:{frame:0},corners:[{x:0,y:0},{x:4,y:0},{x:4,y:3},{x:0,y:3}]};
        const tiled=await renderTiledOverlay({maps,width:4,height:3,minX:0,minY:0,geometries:[geometry],tileSize:2,
          pixelAllowed:(x,y)=>y*4+x!==9,decode:async()=>({frame:frame.clone(),orientation})});
        try {
          const assembled=new OffscreenCanvas(4,3),ac=assembled.getContext('2d');
          for(const tile of tiled.tiles)ac.drawImage(tile.bitmap,0,0,tile.width,tile.height,tile.x,tile.y,tile.width,tile.height);
          const diff=compare(ac.getImageData(0,0,4,3).data,expected);
          assert(diff.maximum<=1&&diff.alphaDifferences===0,`Tile seams: ${JSON.stringify(diff)}`);
          result.tileSeams=diff;
        } finally {closeOverlayTiles(tiled);}
        let cancel=false;
        const aborted=await renderTiledOverlay({maps,width:4,height:3,minX:0,minY:0,geometries:[geometry],tileSize:2,
          decode:async()=>({frame:frame.clone(),orientation}),cancelled:()=>cancel,
          progress:p=>{if(p.tileIndex===1)cancel=true;}});
        assert(aborted===null,'Cancelled tile output was published');
        result.tileCancellation=true;
        const placed=await WebGpuOverlay.create(maps,12,12,0,0,(x,y)=>y*4+x!==9);
        const expectedCanvas=new OffscreenCanvas(4,3);
        expectedCanvas.getContext('2d').putImageData(new ImageData(expected,4,3),0,0);
        const target=new OffscreenCanvas(12,12),tc=target.getContext('2d',{willReadFrequently:true});
        const sum=new Float32Array(12*12*4);
        try {
          for(const [c,s,ox,oy] of [[1,0,4,3],[0,1,8,3]]) {
            const geometry={c,s,world:(x,y)=>({x:ox+c*x-s*y,y:oy+s*x+c*y})};
            await placed.addFrame(frame,orientation,geometry);
            tc.setTransform(1,0,0,1,0,0);tc.clearRect(0,0,12,12);
            tc.setTransform(c,s,-s,c,ox,oy);tc.drawImage(expectedCanvas,0,0);
            accumulateFrame(sum,tc.getImageData(0,0,12,12).data);
          }
          const diff=compare(await pixels(placed),averagedFrames(sum));
          assert(diff.maximum<=1&&diff.alphaDifferences===0,`World pose/mask blend: ${JSON.stringify(diff)}`);
          result.worldTransform=diff;
        } finally {placed.destroy();}
      }
    } finally {frame.close();overlay.destroy();}
  }
  // Compare a real HEVC frame through the old Canvas path and the new native path.
  const decoded=await decoder.call('frame',{index:970});
  const source=new OffscreenCanvas(decoded.bitmap.width,decoded.bitmap.height);
  const context=source.getContext('2d',{willReadFrequently:true});context.drawImage(decoded.bitmap,0,0);decoded.bitmap.close();
  const image=context.getImageData(0,0,source.width,source.height);
  const old=await computer.call('remap',{image,useWebGpu:true},[image.data.buffer]);
  const full=calibration.maps, width=256,height=256;
  const maps={outputWidth:width,outputHeight:height,inverseX:new Float32Array(width*height),inverseY:new Float32Array(width*height),valid:new Uint8Array(width*height)};
  const expected=new Uint8ClampedArray(width*height*4);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++) {
    const i=y*width+x,j=Math.floor((y+0.5)*full.outputHeight/height)*full.outputWidth+Math.floor((x+0.5)*full.outputWidth/width);
    maps.inverseX[i]=full.inverseX[j];maps.inverseY[i]=full.inverseY[j];maps.valid[i]=full.valid[j];
    expected.set(old.data.subarray(j*4,j*4+4),i*4);
  }
  const native=await decoder.call('native-frame',{index:970});
  const overlay=await WebGpuOverlay.create(maps,width,height,0,0);
  try {
    await overlay.addFrame(native.frame,native.orientation,identity);
    result.realFrame=compare(await pixels(overlay),expected);
    assert(result.realFrame.comparedChannels>1000,'Real comparison region contains no valid pixels');
    assert(result.realFrame.mean<2&&result.realFrame.maximum<12&&result.realFrame.alphaDifferences===0,`Real-frame difference: ${JSON.stringify(result.realFrame)}`);
  }finally{native.frame.close();overlay.destroy();}
  return result;
}
