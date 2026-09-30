import test from 'node:test';
import assert from 'node:assert/strict';
import {structuralImage,measureStructuralPair} from '../src/pcb-structural-match.js';
const pose = (x = 0, y = 0, rotation = 0) => ({ x, y, rotation });
function texture(actual = pose(), blank = false) {
  const width = 256, height = 256, data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const px = actual.x + Math.cos(actual.rotation) * (x - 128) - Math.sin(actual.rotation) * (y - 128);
    const py = actual.y + Math.sin(actual.rotation) * (x - 128) + Math.cos(actual.rotation) * (y - 128);
    const value = blank ? 90 : 110 + 24 * Math.sin(px * .24 + Math.sin(py * .1) * 2) +
      27 * Math.cos(py * .32 + Math.sin(px * .07)) + 20 * Math.sin(px * .4 + py * .27);
    const i = 4 * (y * width + x); data[i] = data[i + 1] = data[i + 2] = value; data[i + 3] = 255;
  }
  return { width, height, data };
}
const item = (frame, p = pose()) => ({ frame, pose: p, offset: [-128, -128] });


test('generic gradients refine a textured surface without vias',()=>{
 const truth=pose(1.3,-.7,.004);
 const m=measureStructuralPair(texture(),texture(truth),item(1),item(2,pose(4,-2,.005)),{radius:12});
 assert.ok(m.accepted,JSON.stringify({reason:m.reason,score:m.score,cycle:m.reverseDistance}));
 assert.ok(Math.hypot(m.forward.pose.x-truth.x,m.forward.pose.y-truth.y)<.6);
});
test('masked borders never become structural features',()=>{
 const image=texture(pose(),true);
 for(let y=0;y<256;y++)for(let x=0;x<120;x++)image.data[(y*256+x)*4+3]=0;
 const edge=structuralImage(image);
 assert.equal(edge.data.filter((v,i)=>i%4!==3&&v!==0).length,0);
 const m=measureStructuralPair(image,image,item(1),item(2),{radius:12});
 assert.equal(m.accepted,false);
});
