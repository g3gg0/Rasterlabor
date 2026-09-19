import { createIcons, icons } from 'lucide';
import { WorkerClient } from './rpc.js';
import { drawCheckerboardAnalysis } from './checkerboard-analysis.js';

export function installCheckerboardView({ canvas, getState, redraw }) {
  const root = document.createElement('div'); root.className = 'checkerboard-analysis'; root.id = 'checkerboardAnalysis'; root.hidden = true;
  root.innerHTML = `<div class="checkerboard-controls"><button data-role="run" title="Schachbrettflaechen analysieren"><i data-lucide="scan-search"></i>Flaechen analysieren</button>
    <button data-role="cancel" title="Analyse abbrechen" aria-label="Analyse abbrechen" hidden><i data-lucide="square"></i></button>
    <label>Feldkante, px<input data-role="step" type="number" min="10" step="1" value="100"></label>
    <label class="check"><input data-role="visible" type="checkbox" checked>Flaechen</label></div>
    <div data-role="status" role="status">Noch keine Analyse</div>
    <div data-role="statistics"></div>
    <div data-role="legend" class="checkerboard-legend" hidden><span>Maximum: 0%</span><span class="checkerboard-gradient"></span><span data-role="maximumDeviation"></span></div>`;
  canvas.closest('section').append(root); createIcons({ icons, root });
  const field = name => root.querySelector(`[data-role="${name}"]`);
  let key = null; let analysis = null; let worker = null; let revision = 0;
  const numeric = value => value.toLocaleString('de-DE', { maximumFractionDigits: 2 });
  function cancel() { revision++; worker?.terminate(); worker = null; field('cancel').hidden = true; }
  function refresh() {
    const state = getState(); root.hidden = !state.visible;
    if (key !== state.key || !state.ready) {
      cancel(); analysis = null; key = state.key;
      field('statistics').textContent = ''; field('legend').hidden = true;
      field('status').textContent = state.ready ? 'Noch keine Analyse' : 'Kein pausiertes entzerrtes Bild';
    }
    if (state.step && !field('step').dataset.edited) field('step').value = String(Math.round(state.step));
    field('run').disabled = !state.ready || Boolean(worker); field('step').disabled = Boolean(worker);
  }
  field('step').oninput = () => { field('step').dataset.edited = 'true'; };
  field('visible').onchange = redraw;
  field('cancel').onclick = () => { cancel(); field('status').textContent = 'Analyse abgebrochen'; refresh(); };
  field('run').onclick = async () => {
    const state = getState(); const step = Number(field('step').value);
    if (!state.ready || worker) return;
    if (!Number.isFinite(step) || step < 10 || step > Math.min(state.image.width, state.image.height) / 2) {
      field('status').textContent = 'Ungueltige Feldkante'; return;
    }
    const request = ++revision; const client = new WorkerClient('/compute-worker.js'); worker = client;
    field('run').disabled = true; field('step').disabled = true; field('cancel').hidden = false;
    field('status').textContent = 'Schachbrettsuche laeuft ...';
    try {
      const bitmap = await createImageBitmap(state.image);
      if (request !== revision) { bitmap.close(); return; }
      const result = await client.call('checkerboard-analysis', { bitmap, options: { step } }, [bitmap]);
      if (request !== revision || getState().key !== state.key || !getState().ready) return;
      analysis = result;
      const stats = result.statistics;
      field('status').textContent = `${stats.count} vollstaendige Felder | ${result.regions} Suchbereiche | ${numeric(result.milliseconds / 1000)} s`;
      field('statistics').textContent = stats.count ? `Flaeche in px²: Min ${numeric(stats.minimum)} | Max ${numeric(stats.maximum)} | Mittel ${numeric(stats.mean)} | Median ${numeric(stats.median)} | Standardabw. ${numeric(stats.deviation)}` : 'Keine vollstaendigen Schachbrettfelder erkannt';
      field('legend').hidden = !stats.count;
      field('maximumDeviation').textContent = `${numeric(stats.maxDeviation * 100)}% kleiner`;
      field('legend').title = 'Abweichung = (maximale Flaeche - Feldflaeche) / maximale Flaeche. Farbskala: 0 bis zur groessten gemessenen Abweichung.';
      redraw();
    } catch (error) { if (request === revision) field('status').textContent = error.message; }
    finally { client.terminate(); if (worker === client) { worker = null; field('cancel').hidden = true; } refresh(); }
  };
  return { refresh, draw(context, scale) { if (analysis && getState().key === key && getState().ready && field('visible').checked) drawCheckerboardAnalysis(context, analysis, scale); } };
}