import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const squares = 100;
const printerDpi = 300;
const pixelsPerSquare = 12;
const millimetresPerInch = 25.4;
const squareMillimetres = pixelsPerSquare * millimetresPerInch / printerDpi;
const pointsPerMillimetre = 72 / millimetresPerInch;
const squareSize = squareMillimetres * pointsPerMillimetre;
const pageSize = squares * squareSize;
const output = resolve(process.cwd(), process.argv[2] ?? 'checkerboard-100x100-1.016mm-300dpi.pdf');

const content = ['0 g'];
for (let row = 0; row < squares; row++) for (let column = 0; column < squares; column++) {
  if ((row + column) % 2) continue;
  const x = column * squareSize;
  const y = pageSize - (row + 1) * squareSize;
  content.push(`${x.toFixed(6)} ${y.toFixed(6)} ${squareSize.toFixed(6)} ${squareSize.toFixed(6)} re f`);
}
const stream = content.join('\n') + '\n';
const objects = [
  '<< /Type /Catalog /Pages 2 0 R >>',
  '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
  `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${pageSize.toFixed(6)} ${pageSize.toFixed(6)}] /Resources << >> /Contents 4 0 R >>`,
  `<< /Length ${Buffer.byteLength(stream, 'ascii')} >>\nstream\n${stream}endstream`
];
let pdf = '%PDF-1.4\n%\xFF\xFF\xFF\xFF\n';
const offsets = [0];
for (const [index, object] of objects.entries()) {
  offsets.push(Buffer.byteLength(pdf, 'binary'));
  pdf += `${index + 1} 0 obj\n${object}\nendobj\n`;
}
const xref = Buffer.byteLength(pdf, 'binary');
pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
for (const offset of offsets.slice(1)) pdf += `${String(offset).padStart(10, '0')} 00000 n \n`;
pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;

await writeFile(output, Buffer.from(pdf, 'binary'));
console.log(`Checkerboard geschrieben: ${output}`);
console.log(`Seite: ${(squares * squareMillimetres).toFixed(3)} x ${(squares * squareMillimetres).toFixed(3)} mm | Raster: ${squares} x ${squares} Felder | Feldkante: ${squareMillimetres.toFixed(6)} mm = ${pixelsPerSquare} Pixel bei ${printerDpi} dpi`);