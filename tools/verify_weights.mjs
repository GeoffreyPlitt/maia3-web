// Checks weights/ against weights/manifest.json: file size and SHA-256.
// Usage: node tools/verify_weights.mjs [weight file to compare instead]
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const manifest = JSON.parse(readFileSync(new URL('../weights/manifest.json', import.meta.url), 'utf8'));
const path = process.argv[2] ?? fileURLToPath(new URL(`../weights/${manifest.file}`, import.meta.url));
const data = readFileSync(path), sha256 = createHash('sha256').update(data).digest('hex');
const problems = [];
if (data.length !== manifest.bytes) problems.push(`size ${data.length}, manifest says ${manifest.bytes}`);
if (sha256 !== manifest.sha256) problems.push(`SHA-256 ${sha256}, manifest says ${manifest.sha256}`);
const header = JSON.parse(data.subarray(4, 4 + data.readUInt32LE(0)).toString('utf8'));
if (header.format !== 'int8') problems.push(`format ${header.format}, expected int8`);
if (problems.length) { console.error(`${path}: ${problems.join('; ')}`); process.exit(1); }
console.log(`${path}: ${data.length} bytes, SHA-256 ${sha256} matches the manifest`);
