import http from 'node:http';
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {resolve,extname,sep} from 'node:path';
import {chromium} from 'playwright';
const root=resolve('dist');
const hook=`globalThis.workerProbe={async run(nodes,network){const worker=new WorkerClient('/compute-worker.js');try{return await worker.call('pcb-graph-optimize',{graph:{nodes,edges:[]},network,options:{iterations:20,huber:4}});}finally{worker.terminate();}}};`;
const server = http.createServer(async (req, res) => {
  try {
    const file = resolve(root, '.' + (req.url === '/' ? '/index.html' : req.url));
    if (!file.startsWith(root + sep)) throw Error('path');
    let bytes = await readFile(file);
    if (req.url === '/app.js') bytes = Buffer.concat([bytes, Buffer.from(hook)]);
    res.setHeader('Content-Type', { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' }[extname(file)] || 'application/octet-stream');
    res.end(bytes);
  } catch { res.writeHead(404).end(); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
let browser;
try{
 browser=await chromium.launch({channel:'msedge',headless:true});const page=await browser.newPage();
 await page.goto('http://127.0.0.1:'+server.address().port);
 const source=JSON.parse(await readFile('benchmarks/optimizer-input-tracking.json','utf8'));
 const network=JSON.parse(await readFile('benchmarks/optimizer-network-cycle3.json','utf8'));
 const nodes=source.path.map(p=>({frame:p.frame,pose:p.pose}));
 const result=await page.evaluate(({nodes,network})=>workerProbe.run(nodes,network),{nodes,network});
 assert.equal(result.corrections.length,nodes.length);assert.ok(result.afterRms<result.beforeRms);
 assert.ok(result.corrections.every(p=>[p.pose.x,p.pose.y,p.pose.rotation].every(Number.isFinite)));
 await writeFile('benchmarks/optimizer-worker-browser.json',JSON.stringify({frames:result.corrections.length,beforeRms:result.beforeRms,afterRms:result.afterRms,unmeasured:result.unmeasured},null,2));
 console.log(JSON.stringify({frames:result.corrections.length,beforeRms:result.beforeRms,afterRms:result.afterRms}));
}finally{await browser?.close();await new Promise(r=>server.close(r));}
