// Confidence is based on independent structure and geometric agreement, not NCC alone.
export function pairStiffness(match) {
  const fft = match.fft;
  const cells = Array.isArray(fft?.inlierCells) ? fft.inlierCells.length : 0;
  const support = Math.max(0, Math.min(match.forward?.support ?? 0,
    fft?.uniqueSupportArea ?? match.forward?.support ?? 0));
  const score = Math.min(match.forward?.score ?? 0, match.backward?.score ?? 0);
  const correlation = Math.max(0, Math.min(1, (score - 0.9) / 0.1));
  const residual = Number.isFinite(fft?.residualRms) ? Math.max(0, fft.residualRms) : 8;
  const cycle = Number.isFinite(match.reverseDistance) ? Math.max(0, match.reverseDistance) : 20;
  // Saturate evidence so dense sampling or image size cannot make an infinite spring.
  const area = Math.min(16, Math.sqrt(support / 128));
  const distributed = Math.min(4, Math.sqrt(cells / 3));
  const agreement = 1 / (1 + (residual / 4) ** 2 + (cycle / 10) ** 2);
  let weight = Math.max(1, Math.min(256, 4 + 12 * correlation * area * distributed * agreement));
  if(fft?.pointPairs?.some(cell=>cell.normal)) {
    // Scalar profile errors are measured in pixels along the observable normal.
    // A sharply verified edge can be stiff there without constraining its tangent.
    weight=Math.min(256,weight/Math.max(.2,residual*residual+cycle*cycle));
  }
  const evidence = match.landmarkRecovery?.geometricEvidence;
  if (fft?.method === 'PCB-Via-Konstellation' && evidence) {
    weight = Math.max(weight, Math.min(64, 8 * Math.sqrt(evidence.viaCount) *
      Math.max(0, evidence.traceScore) / (1 + (evidence.residual / 3) ** 2)));
  }
  // A single circular landmark has no angular evidence and only weak positional evidence.
  if (fft?.translationOnly && fft.method !== 'PCB-Vias') weight = 1;
  return { weight, rotationWeight: fft?.translationOnly ? 0 : 1 };
}
