import { build } from 'esbuild';
import { mkdir, copyFile } from 'node:fs/promises';
await mkdir('dist', { recursive: true });
await build({ entryPoints: ['src/app.js', 'src/compute-worker.js', 'src/decoder-worker.js'],
  bundle: true, outdir: 'dist', format: 'esm', platform: 'browser', target: 'es2022',
  loader: { '.woff2': 'file', '.woff': 'file' }, assetNames: 'assets/[name]-[hash]', sourcemap: true });
await copyFile('calib.html', 'dist/index.html');
console.log('Local bundle ready in dist/');