export function detectionImageSize(width, height, maximumEdge = 3840) {
  const scale = Math.min(1, maximumEdge / Math.max(width, height));
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}

export function scaleDetectionOptions(options, sourceWidth, sourceHeight, targetWidth, targetHeight) {
  const scaleX = targetWidth / sourceWidth;
  const scaleY = targetHeight / sourceHeight;
  const scale = (scaleX + scaleY) / 2;
  return { ...options,
    approxStep: options.approxStep ? options.approxStep * scale : options.approxStep,
    lineRadius: options.lineRadius ? options.lineRadius * scale : options.lineRadius,
    roi: options.roi ? { x: options.roi.x * scaleX, y: options.roi.y * scaleY,
      width: options.roi.width * scaleX, height: options.roi.height * scaleY } : undefined };
}

export function restoreDetectionScale(detection, sourceWidth, sourceHeight, targetWidth, targetHeight) {
  const scaleX = sourceWidth / targetWidth;
  const scaleY = sourceHeight / targetHeight;
  const scale = (scaleX + scaleY) / 2;
  const point = value => ({ ...value, x: value.x * scaleX, y: value.y * scaleY });
  return { ...detection,
    points: detection.points.map(point),
    rejected: detection.rejected.map(point),
    lines: detection.lines.map(line => ({ ...line, points: line.points.map(point) })),
    roi: detection.roi ? { x: detection.roi.x * scaleX, y: detection.roi.y * scaleY,
      width: detection.roi.width * scaleX, height: detection.roi.height * scaleY } : detection.roi,
    step: Number.isFinite(detection.step) ? detection.step * scale : detection.step };
}