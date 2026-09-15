const FIXED_POINT_ONE = 65536;

function axisValue(value) {
  const normalized = value / FIXED_POINT_ONE;
  const rounded = Math.round(normalized);
  return Math.abs(normalized - rounded) <= 1 / FIXED_POINT_ONE && Math.abs(rounded) <= 1 ? rounded : null;
}

export function orientationFromMatrix(matrix, sourceWidth, sourceHeight) {
  const values = matrix ? [matrix[0], matrix[1], matrix[3], matrix[4]].map(axisValue) : [1, 0, 0, 1];
  const [a, b, c, d] = values;
  const orthogonal = a * a + b * b === 1 && c * c + d * d === 1 && a * c + b * d === 0;
  if (values.some(value => value === null) || !orthogonal || Math.abs(a * d - b * c) !== 1) {
    throw new Error('Die MP4-Spur verwendet eine nicht unterstuetzte freie Transformation. Nur 90-Grad-Rotation und Spiegelung sind moeglich.');
  }
  const corners = [[0, 0], [sourceWidth, 0], [0, sourceHeight], [sourceWidth, sourceHeight]]
    .map(([x, y]) => [a * x + c * y, b * x + d * y]);
  const xs = corners.map(([x]) => x);
  const ys = corners.map(([, y]) => y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  return {
    a, b, c, d, translateX: -minX, translateY: -minY,
    width: Math.max(...xs) - minX, height: Math.max(...ys) - minY,
    transformed: a !== 1 || b !== 0 || c !== 0 || d !== 1
  };
}