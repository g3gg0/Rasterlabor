import { measureEdgeConstraints } from './pcb-edge-constraints.js';
import { measureFinePair, solveFineGraph, cellNcc, revalidateSavedCells } from './fine-alignment.js';
import { measureStructuralPair } from './pcb-structural-match.js';
const cache = new Map();
let bytes = 0;
let mask = null;
self.onmessage = ({ data }) => {
  try {
    let result;
    if (data.type === 'configure') { mask = data.mask; result = true; }
    else if (data.type === 'store') {
      const bitmap = data.bitmap;
      try {
        const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        const context = canvas.getContext('2d', { willReadFrequently: true });
        context.drawImage(bitmap, 0, 0);
        const image = context.getImageData(0, 0, bitmap.width, bitmap.height);
        if (cache.has(data.frame)) bytes -= cache.get(data.frame).data.byteLength;
        cache.delete(data.frame); cache.set(data.frame, image); bytes += image.data.byteLength;
        while (bytes > 160 * 1024 ** 2 && cache.size > 2) {
          const first = [...cache.keys()].find(frame => !data.keep?.includes(frame));
          if (first === undefined) break;
          bytes -= cache.get(first).data.byteLength; cache.delete(first);
        }
        result = [...cache.keys()];
      } finally { bitmap.close(); }
    } else if (data.type === 'measure' || data.type === 'structural' || data.type === 'edges') {
      const images = [data.reference, data.current].map(item => {
        const image = cache.get(item.frame);
        if (!image) throw new Error(`Frame #${item.frame} fehlt im Fein-Cache.`);
        cache.delete(item.frame); cache.set(item.frame, image); return image;
      });
      result = (data.type === 'edges' ? measureEdgeConstraints : data.type === 'structural' ? measureStructuralPair : measureFinePair)(...images, data.reference, data.current, { ...data.options, mask });
      if(!result.accepted && data.type === 'structural') result=measureEdgeConstraints(...images,data.reference,data.current,{...data.options,mask});
    } else if (data.type === 'validate-anchors') {
      result=revalidateSavedCells(cache.get(data.reference.frame),cache.get(data.current.frame),data.reference,data.current,data.cells);
    } else if (data.type === 'anchors') {
      result=data.cells.map(cell=>cellNcc(cache.get(data.reference.frame),cache.get(data.current.frame),cell,data.reference.offset,data.current.offset,data.reference.pose.rotation-data.current.pose.rotation,Math.max(24,(data.cellSize??64)/2-2)));
    } else if (data.type === 'solve') result = solveFineGraph(data.graph, data.options);
    else throw new Error('Unbekannter Feinauftrag');
    self.postMessage({ id: data.id, result });
  } catch (error) { self.postMessage({ id: data.id, error: error.message }); }
};
