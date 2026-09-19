import { installCheckerboardView } from '../src/checkerboard-view.js';

export function mountCheckerboardFixture() {
  document.getElementById('checkerboardAnalysis')?.remove();
  const image = document.createElement('canvas'); image.width = 768; image.height = 576;
  const context = image.getContext('2d'); context.fillStyle = '#eeeeee'; context.fillRect(0, 0, image.width, image.height);
  let top = 0;
  for (let row = 0; row < 16; row++) {
    const height = 30 + row;
    for (let col = 0; col < 24; col++) {
      context.fillStyle = (row + col) % 2 ? '#eeeeee' : '#222222';
      context.fillRect(col * 32, top, 32, height);
    }
    top += height;
  }
  const canvas = document.getElementById('resultCanvas');
  const state = { visible: true, ready: true, key: 'fixture-1', image, step: 34 };
  const redraw = () => {
    canvas.width = Math.max(1, canvas.clientWidth); canvas.height = Math.max(1, canvas.clientHeight);
    const drawing = canvas.getContext('2d'); const scale = Math.min(canvas.width / image.width, canvas.height / image.height);
    drawing.translate((canvas.width - image.width * scale) / 2, (canvas.height - image.height * scale) / 2);
    drawing.scale(scale, scale); drawing.drawImage(image, 0, 0); view.draw(drawing, scale);
  };
  const view = installCheckerboardView({ canvas, getState: () => state, redraw });
  document.getElementById('resultEmpty').hidden = true;
  view.refresh(); redraw();
  new ResizeObserver(redraw).observe(canvas);
  return { state, view, redraw };
}