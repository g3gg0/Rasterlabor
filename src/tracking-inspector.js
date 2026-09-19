import { WorkerClient } from './rpc.js';

const validPose = pose => pose && [pose.x, pose.y, pose.rotation].every(Number.isFinite);
const numeric = value => Number.isFinite(value) ? value.toFixed(3) : '-';
const poseText = pose => validPose(pose) ? `${numeric(pose.x)}, ${numeric(pose.y)} px / ${numeric(pose.rotation * 180 / Math.PI)} deg` : 'Nicht gespeichert';

export function installTrackingInspector({ readFrame, getMask, busy, drawImage = (context, image, ...coordinates) => context.drawImage(image, ...coordinates) }) {
  const root = document.createElement('section');
  root.className = 'tracking-inspector'; root.id = 'trackingInspector'; root.hidden = true;
  root.innerHTML = `<div class="section-heading"><h2>Matchdiagnose</h2><span data-role="frame"></span></div>
    <p data-role="chain"></p><div class="table-scroll"><table><thead><tr><th>Referenz</th><th>Typ</th><th>Ergebnis</th><th>NCC</th><th>Radius</th><th>Rueckwaerts</th></tr></thead><tbody data-role="references"></tbody></table></div>
    <p data-role="status" aria-live="polite"></p>
    <div data-role="pair" hidden>
      <div class="inspector-controls"><label>Suchstufe<select data-role="stage"></select></label><label>Ausrichtung<select data-role="alignment"><option value="result">Matchpose</option><option value="seed">Startpose</option></select></label>
      <label>Ansicht<select data-role="view"><option value="overlay">Ueberlagerung</option><option value="difference">Differenz</option><option value="reference">Nur Referenz</option><option value="current">Nur aktueller Frame</option></select></label>
      <label>Mischung<input data-role="alpha" type="range" min="0" max="1" step="0.05" value="0.5"></label>
      <label>Zoom<input data-role="zoom" type="range" min="1" max="8" step="0.1" value="1"></label>
      <label class="check"><input data-role="mask" type="checkbox" checked>Auswertebereich</label>
      <button data-role="fit" title="Ansicht einpassen" aria-label="Ansicht einpassen"><i data-lucide="maximize"></i></button></div>
      <canvas data-role="canvas" aria-label="Matchdiagnose: Referenz und aktueller Frame in gespeicherter Ausrichtung"></canvas>
      <pre data-role="details"></pre>
      <form data-role="retry" class="inspector-controls"><label>Radius, px<input name="radius" type="number" min="1" max="10000" value="32" required></label>
      <label>Winkel, Grad<input name="angle" type="number" min="0.1" max="10" step="0.1" value="1" required></label>
      <label>Rueckradius, px<input name="reverseRadius" type="number" min="1" max="10000" value="32" required></label>
      <label class="check"><input name="coarseStep" type="checkbox">Grobsuche</label>
      <label>Seed<select name="seed"><option value="prediction">Gespeicherte Startpose</option><option value="result">Matchpose</option></select></label>
      <button type="submit"><i data-lucide="rotate-cw"></i> Paar erneut pruefen</button></form>
      <label>Ergebnis<select data-role="experiment"></select></label>
    </div>`;
  document.getElementById('trackingTable').closest('section').after(root);
  const field = name => root.querySelector(`[data-role="${name}"]`);
  const canvas = field('canvas');
  let selectedEntry = null; let selectedMatch = null; let currentImage = null; let referenceImage = null;
  let revision = 0; let worker = null; let experiments = []; let pan = { x: 0, y: 0 }; let drag = null;
  let maskLayer = null;
  const active = () => experiments[Number(field('experiment').value)] ?? selectedMatch;
  function stages() {
    const match = active();
    field('stage').replaceChildren(new Option('Endergebnis', 'final'));
    for (const [index, attempt] of (match.attempts ?? []).entries()) {
      field('stage').add(new Option(`Versuch ${index + 1}: ${attempt.searchRadius} px`, String(index)));
    }
    if (match.backward) field('stage').add(new Option('Rueckwaertspruefung', 'backward'));
  }
  function release() {
    revision++; worker?.terminate(); worker = null;
    currentImage?.close(); referenceImage?.close(); currentImage = referenceImage = null;
    maskLayer = null;
    field('retry').querySelector('button').disabled = false;
  }
  function details() {
    const match = active();
    if (!match) return;
    const attempts = match.attempts;
    field('details').textContent = [
      `#${selectedEntry.frame} -> #${match.frame}: ${match.reason || (match.accepted ? 'Paar akzeptiert' : 'Verworfen')}`,
      `Start: ${poseText(match.prediction)} | Referenz: ${poseText(match.referencePose)}`,
      `Match: ${poseText(match.pose)} | NCC ${numeric(match.score)} | Marge ${numeric(match.margin)} | Pixel ${match.support ?? '-'} | Auswertungen ${match.evaluated ?? '-'}`,
      `Ueberdeckung (Auswahl): ${Number.isFinite(match.overlap) ? (match.overlap * 100).toFixed(1) + '%' : '-'} | Backend ${match.accelerator ?? '-'} | ${numeric(match.milliseconds)} ms`,
      ...(attempts ? attempts.map((attempt, index) => `Versuch ${index + 1}: Radius ${attempt.searchRadius} px, Winkel ${attempt.angle} deg, Grobsuche ${attempt.coarseStep > 0 ? 'an' : 'aus'} | NCC ${numeric(attempt.score)} | Marge ${numeric(attempt.margin)} | Pixel ${attempt.support ?? '-'} | Auswertungen ${attempt.evaluated ?? '-'} | ${attempt.reason || 'akzeptiert'} | ${poseText(attempt.pose)}`) : ['Suchverlauf: nicht gespeichert']),
      match.backward ? `Rueckwaerts: ${match.backward.reason || (match.backward.accepted ? 'akzeptiert' : 'verworfen')} | NCC ${numeric(match.backward.score)} | Abstand ${numeric(match.reverseDistance)} px | ${poseText(match.backward.pose)}` :
        `Rueckwaerts: ${Object.hasOwn(match, 'backward') ? 'nicht ausgefuehrt' : 'nicht gespeichert'}`,
      match.cycleError ? `Zyklus: ${match.cycleQuality ?? '-'} | Translation ${numeric(match.cycleError.translation)} px | Rotation ${numeric(match.cycleError.rotationDegrees)} deg | Bildfehler Median ${numeric(match.cycleError.median)}, RMS ${numeric(match.cycleError.rms)}, P95 ${numeric(match.cycleError.p95)} px (${match.cycleError.samples} Stuetzpunkte) | Grenzen ${numeric(match.cycleLimits?.strict)} / ${numeric(match.cycleLimits?.conditional)} px` :
        'Zyklusmetriken: nicht gespeichert',
      match.kind === 'incremental' ? `Verfahren: Fenster FFT + Rigid | PSR ${numeric(match.psr)} | dx ${numeric(match.dx)}, dy ${numeric(match.dy)} px` :
        'Verfahren: Grauwert-NCC, Bildpyramide, X/Y/Rotation; keine Feature-Keypoints.',
    ].join('\n');
  }
  function draw() {
    if (!currentImage || !referenceImage) return;
    const match = active(); const stage = field('stage').value;
    const backward = stage === 'backward';
    const result = backward ? match.backward : stage === 'final' ? match : match.attempts[Number(stage)];
    const referencePose = backward ? match.attempts?.at(-1)?.pose ?? match.pose : match.referencePose;
    const seed = backward ? match.referencePose : match.prediction;
    const pose = field('alignment').value === 'seed' ? seed : result?.pose ?? seed;
    const source = backward ? referenceImage : currentImage; const target = backward ? currentImage : referenceImage;
    const context = canvas.getContext('2d');
    canvas.width = Math.max(320, Math.round(canvas.clientWidth)); canvas.height = Math.round(canvas.clientHeight || 420);
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!validPose(pose) || !validPose(referencePose)) {
      field('status').textContent = 'Historische Ausrichtung nicht gespeichert. Keine nachtraeglich korrigierte Pose eingesetzt.'; return;
    }
    const cosine = Math.cos(-referencePose.rotation); const sine = Math.sin(-referencePose.rotation);
    const offsetX = cosine * (pose.x - referencePose.x) - sine * (pose.y - referencePose.y);
    const offsetY = sine * (pose.x - referencePose.x) + cosine * (pose.y - referencePose.y);
    const angle = pose.rotation - referencePose.rotation;
    const extent = Math.max(Math.hypot(source.width, source.height) + 2 * Math.hypot(offsetX, offsetY), Math.hypot(target.width, target.height));
    const scale = Math.min(canvas.width, canvas.height) / extent * Number(field('zoom').value);
    context.translate(canvas.width / 2 + pan.x, canvas.height / 2 + pan.y); context.scale(scale, scale);
    const view = field('view').value;
    context.imageSmoothingEnabled = true; context.imageSmoothingQuality = 'high';
    const drawLayer = (image, opacity) => {
      if (opacity <= 0) return;
      context.save(); context.globalAlpha = opacity;
      drawImage(context, image, -image.width / 2, -image.height / 2);
      context.restore();
    };
    const drawMask = (image, opacity) => {
      if (opacity <= 0 || !field('mask').checked || view === 'difference') return;
      context.save(); context.globalAlpha = 1;
      context.globalAlpha = opacity;
      if (match.kind === 'incremental' && match.rectangle) {
        const rectangle = match.rectangle;
        context.strokeStyle = '#e4c549'; context.lineWidth = 2 / scale;
        context.strokeRect(rectangle.x - image.width / 2, rectangle.y - image.height / 2, rectangle.width, rectangle.height);
      } else if (maskLayer) {
        context.drawImage(maskLayer.canvas, -image.width / 2, -image.height / 2, maskLayer.width, maskLayer.height);
      }
      context.restore();
    };
    const alpha = Number(field('alpha').value);
    if (view !== 'current') drawLayer(target, 1);
    if (view !== 'reference') {
      context.translate(offsetX, offsetY); context.rotate(angle);
      context.globalAlpha = 1;
      if (view === 'difference') context.globalCompositeOperation = 'difference';
      drawLayer(source, view === 'overlay' ? alpha : 1);
    }
    context.globalCompositeOperation = 'source-over';
    context.setTransform(scale, 0, 0, scale, canvas.width / 2 + pan.x, canvas.height / 2 + pan.y);
    if (view !== 'current') drawMask(target, view === 'overlay' ? 1 - alpha : 1);
    if (view !== 'reference') {
      context.translate(offsetX, offsetY); context.rotate(angle);
      drawMask(source, view === 'overlay' ? alpha : 1);
    }
  }
  async function select(match) {
    release(); const request = revision;
    selectedMatch = match; experiments = [match, ...(match.debugAttempts ?? [])];
    field('experiment').replaceChildren(new Option('Gespeicherter Lauf', '0'));
    experiments.slice(1).forEach((experiment, index) => field('experiment').add(new Option(`Debug ${index + 1}: ${experiment.reason || 'akzeptiert'}`, String(index + 1))));
    stages();
    field('pair').hidden = false; field('status').textContent = `Lade #${selectedEntry.frame} und #${match.frame} ...`;
    field('retry').elements.radius.value = match.searchRadius ?? 32;
    field('retry').elements.angle.value = match.attempts?.at(-1)?.angle ?? 1;
    field('retry').elements.coarseStep.checked = (match.attempts?.at(-1)?.coarseStep ?? 0) > 0;
    field('retry').elements.reverseRadius.value = match.backward?.searchRadius ?? 32;
    field('retry').querySelector('button').disabled = true;
    pan = { x: 0, y: 0 }; field('zoom').value = '1'; details();
    try {
      const current = await readFrame(selectedEntry.frame, { rectified: true });
      if (request !== revision) { current.bitmap.close(); return; }
      currentImage = current.bitmap;
      const reference = await readFrame(match.frame, { rectified: true });
      if (request !== revision) { reference.bitmap.close(); return; }
      referenceImage = reference.bitmap;
      const mask = getMask();
      if (mask?.data) {
        const layer = document.createElement('canvas'); layer.width = mask.width; layer.height = mask.height;
        const pixels = new ImageData(mask.width, mask.height);
        for (let index = 0; index < mask.data.length; index++) if (mask.data[index] !== 1) pixels.data.set([198, 64, 74, 170], index * 4);
        layer.getContext('2d').putImageData(pixels, 0, 0);
        maskLayer = { canvas: layer, width: mask.width * mask.cellSize, height: mask.height * mask.cellSize };
      }
      field('status').textContent = `#${selectedEntry.frame} / #${match.frame} | Entzerrte Vollbilder | ${match.pose ? 'Matchpose vorhanden' : 'Keine Matchpose: Anzeige der Startpose'} | Debug: Umfeldregistrierung, gespeicherte Laufmaske`;
      field('retry').querySelector('button').disabled = !validPose(match.referencePose);
      draw();
    } catch (error) { if (request === revision) { release(); field('status').textContent = error.message; } }
  }
  field('retry').onsubmit = async event => {
    event.preventDefault();
    if (busy() || !currentImage || !referenceImage || worker) return;
    const base = active(); const form = event.currentTarget;
    const prediction = form.elements.seed.value === 'result' ? base.pose : base.prediction;
    if (!validPose(prediction) || !validPose(base.referencePose)) { field('status').textContent = 'Seed oder Referenzpose nicht gespeichert.'; return; }
    const request = revision;
    const limits = Object.fromEntries(['radius', 'angle', 'reverseRadius'].map(key => [key, Number(form.elements[key].value)]));
    limits.coarseStep = form.elements.coarseStep.checked ? 1 : 0;
    worker = new WorkerClient('/compute-worker.js'); const client = worker;
    form.querySelector('button').disabled = true; field('status').textContent = 'Debugversuch laeuft (CPU, gespeicherte Laufmaske) ...';
    try {
      const current = await createImageBitmap(currentImage); const reference = await createImageBitmap(referenceImage);
      if (request !== revision) { current.close(); reference.close(); return; }
      const started = performance.now();
      const result = await client.call('context-inspect', { current, reference, prediction, referencePose: base.referencePose, limits, mask: getMask() }, [current, reference]);
      if (request !== revision) return;
      const experiment = { ...result, frame: selectedMatch.frame, accelerator: 'CPU Debug', milliseconds: performance.now() - started };
      (selectedMatch.debugAttempts ??= []).push(structuredClone(experiment));
      experiments.push(experiment);
      field('experiment').add(new Option(`Debug ${experiments.length - 1}: ${result.reason || 'akzeptiert'}`, String(experiments.length - 1)));
      field('experiment').value = String(experiments.length - 1);
      field('status').textContent = 'Debugversuch gespeichert. Trackingpfad unveraendert.'; stages(); details(); draw();
    } catch (error) { if (request === revision) field('status').textContent = error.message; }
    finally { client.terminate(); if (worker === client) worker = null; if (request === revision) form.querySelector('button').disabled = false; }
  };
  for (const control of ['stage', 'alignment', 'view', 'alpha', 'zoom', 'mask']) field(control).oninput = draw;
  field('experiment').onchange = () => { stages(); details(); draw(); };
  field('fit').onclick = () => { pan = { x: 0, y: 0 }; field('zoom').value = '1'; draw(); };
  canvas.onpointerdown = event => { drag = { x: event.clientX, y: event.clientY }; canvas.setPointerCapture(event.pointerId); };
  canvas.onpointermove = event => { if (!drag) return; pan.x += event.clientX - drag.x; pan.y += event.clientY - drag.y; drag = { x: event.clientX, y: event.clientY }; draw(); };
  canvas.onpointerup = canvas.onpointercancel = () => { drag = null; };
  new ResizeObserver(draw).observe(canvas);
  return {
    clear() { release(); selectedEntry = null; root.hidden = true; },
    show(entry) {
      release(); selectedEntry = entry; root.hidden = false; field('pair').hidden = true;
      field('frame').textContent = `Frame #${entry.frame}`;
      const closure = entry.context?.loopClosure;
      const confirmation = entry.context?.spatialConfirmation;
      const confirmationText = confirmation ? ` | Raeumlicher Vorschlag: ${confirmation.referenceGroups.length} Referenzgruppe(n), ${confirmation.independentGroups ? 'unabhaengige Gruppen bestaetigt' : confirmation.consecutiveFrames ? `durch Frame #${confirmation.previousFrame} bestaetigt` : 'wartet auf weiteren Frame'}` : '';
      field('chain').textContent = closure ? `Pfadkorrektur: Anker #${closure.anchorFrame}, dx ${numeric(closure.dx)}, dy ${numeric(closure.dy)} px, Rotation ${numeric(closure.rotation * 180 / Math.PI)} deg; ${entry.context?.confirmation === 'consecutive-frames' ? 'Mehrframe-Konsens' : 'Referenzgruppen-Konsens'}; rueckverteilt und vorwaerts uebernommen.` :
        `Pfadkorrektur: ${entry.context ? 'keine Loop-Closure' : 'nicht dokumentiert'}${entry.context?.applied ? ' | Umfeldkonsens angewendet' : ''}${confirmationText}`;
      const matches = entry.context?.matches || entry.incrementalMatch ? [...(entry.incrementalMatch ? [entry.incrementalMatch] : []), ...(entry.context?.matches ?? [])] : null;
      field('status').textContent = !Array.isArray(matches) ? 'Keine gespeicherte Referenzdiagnose fuer diesen Frame.' : matches.length ? `${matches.length} Referenzversuche` : 'Keine Umfeldreferenzen angefordert.';
      const rows = field('references'); rows.replaceChildren();
      for (const match of matches ?? []) {
        const row = rows.insertRow(); const button = document.createElement('button'); button.textContent = `#${match.frame}`;
        button.onclick = () => { if (!busy()) void select(match); };
        row.insertCell().append(button);
        for (const value of [match.kind === 'spatial' ? 'Raeumlich' : match.kind === 'incremental' ? 'Inkrementell' : 'Aktuell', match.reason || (entry.context?.inliers?.includes(match.frame) && match.kind !== 'incremental' ? 'Konsens' : match.accepted ? 'Paar akzeptiert' : 'Verworfen'), numeric(match.score), match.searchRadius ?? '-',
          match.backward ? `${match.backward.accepted ? 'berechnet' : 'verworfen'} / ${numeric(match.reverseDistance)} px` : Object.hasOwn(match, 'backward') ? 'nicht ausgefuehrt' : 'nicht gespeichert']) row.insertCell().textContent = value;
      }
    },
  };
}