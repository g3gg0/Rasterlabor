import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawn } from 'node:child_process';

test('pattern generator writes a 300 dpi aligned vector PDF with 5,000 black 12-pixel squares', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'checkerboard-'));
  const output = join(directory, 'pattern.pdf');
  try {
    await new Promise((resolve, reject) => {
      const child = spawn(process.execPath, ['scripts/generate-pattern.js', output], { cwd: process.cwd() });
      child.on('error', reject); child.on('exit', code => code === 0 ? resolve() : reject(new Error(`Generator exited ${code}`)));
    });
    const pdf = await readFile(output, 'latin1');
    assert.match(pdf, /\/MediaBox \[0 0 288\.000000 288\.000000\]/);
    assert.equal((pdf.match(/ re f\n/g) ?? []).length, 5000);
    assert.match(pdf, /2\.880000 2\.880000 re f/);
    assert.match(pdf, /startxref\n\d+\n%%EOF/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});