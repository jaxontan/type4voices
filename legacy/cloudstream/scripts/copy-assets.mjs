// Copies the files tsc does not emit: renderer assets and provider protocol configs.
import { cp, mkdir, readdir } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');

async function copyRenderer() {
  const from = join(root, 'src', 'renderer');
  const to = join(root, 'dist', 'renderer');
  await mkdir(to, { recursive: true });
  await cp(from, to, { recursive: true });
  console.log('copied renderer -> dist/renderer');
}

async function copyProtocols() {
  const from = join(root, 'src', 'main', 'providers');
  const to = join(root, 'dist', 'main', 'providers');
  await walk(from, async (file) => {
    if (!file.endsWith('.json')) return;
    const dest = join(to, file.slice(from.length).replace(/^[\\/]/, ''));
    await mkdir(dirname(dest), { recursive: true });
    await cp(file, dest);
  });
  console.log('copied protocol configs -> dist/main/providers');
}

async function walk(dir, fn) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) await walk(full, fn);
    else await fn(full);
  }
}

await copyRenderer();
await copyProtocols();
