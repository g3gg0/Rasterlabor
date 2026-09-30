import { build } from 'esbuild';
import { mkdir, copyFile } from 'node:fs/promises';
const pages = process.argv.includes('--pages');
const outdir = pages ? 'dist-pages' : 'dist';
await mkdir(outdir, { recursive: true });
await build({ entryPoints: { app: 'src/app.js', 'compute-worker': 'src/compute-worker.js',
  'decoder-worker': 'src/decoder-worker.js',
  'fine-worker': 'src/fine-worker.js',
  ...(!pages ? { 'tests/overlay-gpu-checks': 'tests/overlay-gpu-checks.js',
  'benchmarks/pcb-native-gpu-probe': 'benchmarks/pcb-native-gpu-probe.js' } : {}) },
  bundle: true, outdir, format: 'esm', platform: 'browser', target: 'es2022',
  loader: { '.woff2': 'file', '.woff': 'file' }, assetNames: 'assets/[name]-[hash]', sourcemap: !pages, minify: pages });
await copyFile('calib.html', `${outdir}/index.html`);
console.log(`Bundle ready in ${outdir}/`);
