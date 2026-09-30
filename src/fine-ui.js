import { WorkerClient } from './rpc.js';
import { planFinePairs, fineFrameReport } from './fine-alignment.js';
import { mergeNetworkMatches, networkEdges } from './match-network.js';

export function installFineAlignment(api) {
  const el = id => document.getElementById(id);
  let busy = false, cancelled = false, proposal = null, undo = null, rows = [], selected = null;
  const status = message => { el('fineStatus').textContent = message; };
  const close = () => { el('finePopup').hidden = true; el('fineFrames').setAttribute('aria-expanded', 'false'); };
  function refresh() {
    el('fineStart').disabled = busy || !api.available();
    el('fineRadius').disabled = busy;
    el('fineCancel').disabled = !busy;
    el('fineApply').disabled = busy || !proposal || !api.available();
    el('fineDiscard').disabled = busy || !proposal;
    el('fineUndo').disabled = busy || !undo || !api.available();
    el('fineFrames').disabled = busy || !rows.length;
  }
  function render() {
    const query = el('fineFilter').value.trim().replace(/^#/, '');
    const list = el('fineList'); list.replaceChildren();
    for (const row of rows.filter(row => String(row.frame).includes(query))) {
      const option = document.createElement('button'); option.type = 'button'; option.role = 'option';
      option.dataset.frame = row.frame; option.setAttribute('aria-selected', String(row.frame === selected));
      option.className = 'fine-option';
      const label = document.createElement('strong'); label.textContent = `#${row.frame}`;
      const bar = document.createElement('span'); bar.className = 'fine-bar';
      const fill = document.createElement('span');
      const quality = Math.max(0, Math.min(1, ((row.score ?? 0) - .8) / .2));
      fill.style.width = `${Math.max(3, (row.score ?? 0) * 100)}%`; fill.style.backgroundColor = `hsl(${quality * 120} 75% 38%)`;
      bar.append(fill); bar.setAttribute('aria-label', row.score === null ? 'Ungemessen' : `Match ${(row.score * 100).toFixed(1)} Prozent`);
      const detail = document.createElement('span'); detail.className = 'fine-detail';
      detail.textContent = `${row.accepted}/${row.attempts} bestaetigt | ${row.reason}` +
        (row.ncc === null || row.ncc === undefined ? '' : ` | Paar-NCC ${(row.ncc * 100).toFixed(1)}%`) +
        (row.worstPartner === null ? '' : ` gegen #${row.worstPartner}`) +
        (row.residual === null ? '' : ` | Ankerrest ${row.residual.toFixed(2)} px`) +
        (!row.correction ? '' : ` | Korrektur ${row.correction.distance.toFixed(2)} px / ${row.correction.degrees.toFixed(3)} Grad`);
      option.append(label, bar, detail);
      option.onclick = async () => {
        selected = row.frame; close(); el('fineFrames').textContent = `#${row.frame} - ${row.reason}`;
        try { await api.openFrame(row.frame); } catch (error) { status(error.message); }
      };
      list.append(option);
    }
    if (selected === null) el('fineFrames').textContent = `${rows.length} Frames - schlechteste Matches zuerst`;
    const worst = rows[0];
    el('fineWorst').textContent = worst ? `Schwaechster Befund: #${worst.frame}` +
      (worst.worstPartner === null ? '' : ` gegen #${worst.worstPartner}`) + ` - ${worst.reason}.` : '';
    refresh();
  }
  el('fineFrames').onclick = () => {
    const open = el('finePopup').hidden;
    el('finePopup').hidden = !open; el('fineFrames').setAttribute('aria-expanded', String(open));
    if (open) { render(); el('fineFilter').focus(); }
  };
  el('fineFilter').oninput = render;
  el('finePopup').onkeydown = event => {
    if (event.key === 'Escape') { close(); el('fineFrames').focus(); return; }
    const buttons = [...el('fineList').children], index = buttons.indexOf(document.activeElement);
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault(); buttons[Math.max(0, Math.min(buttons.length - 1, index + (event.key === 'ArrowDown' ? 1 : -1)))]?.focus();
    }
    if (event.key === 'Home' || event.key === 'End') {
      event.preventDefault(); buttons[event.key === 'Home' ? 0 : buttons.length - 1]?.focus();
    }
  };
  document.addEventListener('click', event => { if (!el('fineDropdown').contains(event.target)) close(); });
  el('fineCancel').onclick = () => { cancelled = true; status('Stop nach dem aktuellen Paar; gefundene Verbesserungen bleiben als Vorschlag erhalten.'); };
  el('fineDiscard').onclick = () => { proposal = null; status('Vorschlag verworfen. Diagnose bleibt sichtbar.'); refresh(); };
  el('fineApply').onclick = () => {
    if (!proposal || busy || !api.available()) return;
    if (!api.current(proposal.source)) { proposal = null; status('Posen, Anker oder Kalibrierung wurden geaendert. Bitte erneut berechnen.'); refresh(); return; }
    undo = api.apply(proposal);
    proposal = null; status('Feinkorrekturen und bestaetigte Anker uebernommen.'); refresh();
  };
  el('fineUndo').onclick = () => {
    if (!undo || busy || !api.available()) return;
    if (!api.undo(undo)) { status('Seit der Uebernahme wurde weitergearbeitet. Rueckgaengig wuerde diese Aenderungen ueberschreiben.'); return; }
    undo = null; proposal = null; status('Feinkorrekturen rueckgaengig gemacht.'); refresh();
  };
  el('fineStart').onclick = async () => {
    if (busy || !api.available()) return;
    const radius = Number(el('fineRadius').value);
    if (!Number.isInteger(radius) || radius < 2 || radius > 12) { status('Suchradius: ganze Pixel zwischen 2 und 12.'); return; }
    const source = api.snapshot(), entries = new Map(source.frames.map(frame => [frame.frame, frame]));
    const pairs = planFinePairs(source.frames, source.network, source.reach);
    proposal = null; cancelled = false; selected = null; busy = true; close(); api.busy(true); refresh();
    const worker = new WorkerClient('./fine-worker.js'), matches = [];
    let cached = new Set(), network = source.network;
    rows = fineFrameReport(source.frames, matches); render();
    try {
      await worker.call('configure', { mask: source.mask });
      for (const [index, pair] of pairs.entries()) {
        if (cancelled) break;
        status(`Feinsuche ${index + 1}/${pairs.length}: #${pair.current} gegen #${pair.reference} | Radius ${radius} px`);
        try {
          for (const frame of [pair.reference, pair.current]) if (!cached.has(frame)) {
            const bitmap = await api.bitmap(frame, source);
            cached = new Set(await worker.call('store', { frame, bitmap, keep: [pair.reference, pair.current] }, [bitmap]));
          }
          const match = await worker.call('measure', { reference: entries.get(pair.reference), current: entries.get(pair.current),
            options: { radius } });
          matches.push(match);
          network = mergeNetworkMatches(network, [match], { minimumScore: .93, cycleLimit: 1, preserveStrength: true });
        } catch (error) {
          matches.push({ ...pair, accepted: false, score: 0, reason: `Messfehler: ${error.message}` });
        }
        rows = fineFrameReport(source.frames, matches);
        // Keep the live summary cheap; the complete list is built on opening it.
        el('fineWorst').textContent = `Gemessen: ${rows.filter(row => row.attempts).length}/${source.frames.length} Frames; ${matches.filter(match => match.accepted).length} bestaetigte Paare.`;
      }
      const nodes = source.frames.filter(frame => frame.pose &&
        [frame.pose.x, frame.pose.y, frame.pose.rotation].every(Number.isFinite));
      const edges = networkEdges(network, new Set(nodes.map(frame => frame.frame)));
      status('Gemeinsame Feinausrichtung mit gespeicherten Ankern...');
      const solved = await worker.call('solve', { graph: { nodes, edges }, options: { radius, lever: source.reach / 2 } });
      const byFrame = new Map(source.frames.map(frame => [frame.frame, frame.pose]));
      for (const correction of solved.corrections) byFrame.set(correction.frame, correction.pose);
      rows = fineFrameReport(source.frames, matches, edges, byFrame);
      for (const row of rows) {
        const previous = entries.get(row.frame)?.pose, corrected = byFrame.get(row.frame);
        if (previous && corrected) row.correction = {
          distance: Math.hypot(corrected.x - previous.x, corrected.y - previous.y),
          degrees: Math.atan2(Math.sin(corrected.rotation - previous.rotation), Math.cos(corrected.rotation - previous.rotation)) * 180 / Math.PI };
      }
      for (const row of rows) if (!row.attempts) row.reason = cancelled ? 'Durchlauf gestoppt; nicht gemessen' : 'Kein geeigneter Nachbar / keine gueltige Pose';
      const accepted = matches.filter(match => match.accepted).length;
      const report = { schemaVersion: 1, radius, createdAt: new Date().toISOString(), rows,
        pairs: matches.map(match => ({ reference: match.reference, current: match.current,
          accepted: match.accepted, score: match.score, worstCellScore: match.worstCellScore,
          reason: match.reason, cellFailures: match.cellFailures,
          residual: match.fft?.residualRms, reverseDistance: match.reverseDistance })) };
      if (accepted) proposal = { source, byFrame, network, report };
      api.report(report);
      status(`${cancelled ? 'Gestoppt. ' : ''}${rows.filter(row => row.attempts).length}/${source.frames.length} Frames geprueft; ${accepted}/${matches.length} Paare bestaetigt. ` +
        `${rows.filter(row => !row.accepted).length} Frames ohne bestaetigte Feinmessung. ` +
        (proposal ? `Vorschlag bereit (max. ${radius} px inkl. Rotation). Erst uebernehmen, um korrigierte Posen manuell zu pruefen.` : 'Kein belastbarer neuer Vorschlag.') +
        (solved.scale < 1 ? ' Gemeinsamer Korrekturschritt begrenzt.' : ''));
    } catch (error) { status(`Feinoptimierung fehlgeschlagen: ${error.message}`); }
    finally { worker.terminate(); busy = false; api.busy(false); render(); }
  };
  refresh();
  return { refresh, restore(report) {
    proposal = null; undo = null; selected = null; rows = report?.rows ?? []; el('fineFilter').value = '';
    status(report ? `Gespeicherte Feindiagnose: ${rows.length} Frames. ${report.applied ? 'Korrekturen wurden uebernommen.' : 'Keine uebernommenen Feinkorrekturen.'}` : 'Noch kein Feindurchlauf.');
    render();
  } };
}
