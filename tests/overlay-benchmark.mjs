import http from 'node:http';
import { readFile, mkdir, writeFile, copyFile } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { chromium } from 'playwright';

const mode = process.argv[2] || 'before';
const root = resolve('.');
await mkdir('benchmarks', { recursive: true });
if (mode === 'before') await copyFile('dist/decoder-worker.js', 'benchmarks/baseline-decoder-worker.js');
const server = http.createServer(async (req, res) => {
  try {
    const name = new URL(req.url, 'http://localhost').pathname;
    if (name === '/') { res.setHeader('Content-Type', 'text/html'); res.end('<input id="video" type="file"><input id="calib" type="file">'); return; }
    const path = resolve(root, '.' + (/^\/(decoder|compute)-worker.js$/.test(name) ? '/dist' + name : name));
    if (!path.startsWith(root + sep)) throw new Error('Invalid path');
    const bytes = await readFile(path);
    res.setHeader('Content-Type', extname(path) === '.js' ? 'text/javascript' : 'application/octet-stream'); res.end(bytes);
  } catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ channel: 'msedge', headless: true });
  const page = await browser.newPage();
  page.on('console', event => { if (event.type() === 'error') console.error(event.text()); });
  page.on('pageerror', error => console.error(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.locator('#video').setInputFiles('../20260914_214049.mp4');
  await page.locator('#calib').setInputFiles('../kalibrierung-v9-provisional-mit-xyr.zip');
  const result = await page.evaluate(async mode => {
    const { runBenchmark } = await import('/tests/overlay-benchmark-page.js');
    return runBenchmark(mode, document.querySelector('#video').files[0], document.querySelector('#calib').files[0]);
  }, mode);
  result.browser = browser.version(); result.date = new Date().toISOString();
  await writeFile(`benchmarks/overlay-${mode}.json`, JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result.runs ? { file:`benchmarks/overlay-${mode}.json`, hardware:result.hardware,
    output:result.output, runs:result.runs.map(({samples,...summary})=>summary) } : result, null, 2));
} finally { await browser?.close(); await new Promise(resolve => server.close(resolve)); }
