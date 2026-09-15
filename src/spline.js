export function basis(value, derivative = 0) {
  if (derivative === 1) return [-((1 - value) ** 2) / 2, 1.5 * value ** 2 - 2 * value,
    -1.5 * value ** 2 + value + 0.5, value ** 2 / 2];
  if (derivative === 2) return [1 - value, 3 * value - 2, 1 - 3 * value, value];
  return [(1 - value) ** 3 / 6, (3 * value ** 3 - 6 * value ** 2 + 4) / 6,
    (-3 * value ** 3 + 3 * value ** 2 + 3 * value + 1) / 6, value ** 3 / 6];
}

export function createSpline(width, height, spacing, affine = [1, 0, 0, 1, 0, 0]) {
  if (!(width > 1 && height > 1 && spacing > 0)) throw new Error('Invalid spline geometry');
  const nx = Math.floor((width - 1) / spacing) + 4;
  const ny = Math.floor((height - 1) / spacing) + 4;
  const coefficients = new Float64Array(nx * ny * 2);
  for (let row = 0; row < ny; row++) {
    for (let col = 0; col < nx; col++) {
      const px = (col - 1) * spacing;
      const py = (row - 1) * spacing;
      const index = 2 * (row * nx + col);
      coefficients[index] = (affine[0] - 1) * px + affine[1] * py + affine[4];
      coefficients[index + 1] = affine[2] * px + (affine[3] - 1) * py + affine[5];
    }
  }
  return { width, height, spacing, nx, ny, coefficients };
}

export function stencil(field, px, py, derivativeX = 0, derivativeY = 0) {
  const cellX = Math.floor(px / field.spacing);
  const cellY = Math.floor(py / field.spacing);
  const weightsX = basis(px / field.spacing - cellX, derivativeX);
  const weightsY = basis(py / field.spacing - cellY, derivativeY);
  const scale = field.spacing ** (-derivativeX - derivativeY);
  const indices = new Int32Array(16);
  const weights = new Float64Array(16);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      const index = row * 4 + col;
      if (cellX + col < 0 || cellX + col >= field.nx || cellY + row < 0 || cellY + row >= field.ny) {
        throw new Error('Spline evaluation outside padded domain');
      }
      indices[index] = (cellY + row) * field.nx + cellX + col;
      weights[index] = weightsX[col] * weightsY[row] * scale;
    }
  }
  return { indices, weights };
}

export function evaluate(field, px, py, jacobian = false) {
  const cellX = Math.floor(px / field.spacing);
  const cellY = Math.floor(py / field.spacing);
  const localX = px / field.spacing - cellX;
  const localY = py / field.spacing - cellY;
  const weightsX = basis(localX);
  const weightsY = basis(localY);
  const derivativesX = jacobian ? basis(localX, 1) : null;
  const derivativesY = jacobian ? basis(localY, 1) : null;
  const result = { x: px, y: py, j00: 1, j01: 0, j10: 0, j11: 1 };
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      const index = ((cellY + row) * field.nx + cellX + col) * 2;
      const dx = field.coefficients[index];
      const dy = field.coefficients[index + 1];
      const weight = weightsX[col] * weightsY[row];
      result.x += weight * dx;
      result.y += weight * dy;
      if (jacobian) {
        const weightX = derivativesX[col] * weightsY[row] / field.spacing;
        const weightY = weightsX[col] * derivativesY[row] / field.spacing;
        result.j00 += weightX * dx;
        result.j10 += weightX * dy;
        result.j01 += weightY * dx;
        result.j11 += weightY * dy;
      }
    }
  }
  return result;
}

export function inversePoint(field, targetX, targetY, startX, startY, tolerance = 0.005) {
  let px = Math.max(0, Math.min(field.width - 1, startX));
  let py = Math.max(0, Math.min(field.height - 1, startY));
  for (let iteration = 0; iteration < 25; iteration++) {
    const value = evaluate(field, px, py, true);
    const errorX = value.x - targetX;
    const errorY = value.y - targetY;
    const error = Math.hypot(errorX, errorY);
    if (error < tolerance) return { x: px, y: py, error };
    const determinant = value.j00 * value.j11 - value.j01 * value.j10;
    if (!(determinant > 1e-10)) return null;
    const stepX = (value.j11 * errorX - value.j01 * errorY) / determinant;
    const stepY = (-value.j10 * errorX + value.j00 * errorY) / determinant;
    let accepted = false;
    for (let alpha = 1; alpha >= 1 / 128; alpha /= 2) {
      const nextX = px - alpha * stepX;
      const nextY = py - alpha * stepY;
      if (nextX < 0 || nextY < 0 || nextX > field.width - 1 || nextY > field.height - 1) continue;
      const next = evaluate(field, nextX, nextY);
      if (Math.hypot(next.x - targetX, next.y - targetY) < error) {
        px = nextX;
        py = nextY;
        accepted = true;
        break;
      }
    }
    if (!accepted) return null;
  }
  return null;
}

export function fieldColor(dx, dy, range) {
  return [Math.round(127.5 * (1 + Math.max(-1, Math.min(1, dx / range)))), 0,
    Math.round(127.5 * (1 + Math.max(-1, Math.min(1, dy / range))))];
}