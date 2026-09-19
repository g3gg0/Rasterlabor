import { build } from 'esbuild';
import { mkdir, copyFile } from 'node:fs/promises';
await mkdir('dist', { recursive: true });
await build({ entryPoints: { app: 'src/app.js', 'compute-worker': 'src/compute-worker.js',
  'decoder-worker': 'src/decoder-worker.js', 'brightness-gpu-check': 'src/brightness-gpu-check.js',
  'tests/overlay-gpu-checks': 'tests/overlay-gpu-checks.js' },
  bundle: true, outdir: 'dist', format: 'esm', platform: 'browser', target: 'es2022',
  loader: { '.woff2': 'file', '.woff': 'file' }, assetNames: 'assets/[name]-[hash]', sourcemap: true });
await copyFile('calib.html', 'dist/index.html');
console.log('Local bundle ready in dist/');