import { cp, mkdir, readdir, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const webRoot = path.resolve(__dirname, '..');
const repoRoot = path.resolve(webRoot, '..');

const pyFiles = [
  'art.py',
  'art_class.py',
  'art_render.py',
  'art_scheme.py',
  'art_constants.txt',
  'art_settings.txt',
];

function shouldSkip(name) {
  return name.toLowerCase().endsWith('.zip');
}

async function copyRecursive(src, dest) {
  const srcStat = await stat(src);

  if (srcStat.isDirectory()) {
    await mkdir(dest, { recursive: true });
    const entries = await readdir(src, { withFileTypes: true });
    for (const entry of entries) {
      if (shouldSkip(entry.name)) continue;
      await copyRecursive(path.join(src, entry.name), path.join(dest, entry.name));
    }
    return;
  }

  if (shouldSkip(path.basename(src))) return;
  await mkdir(path.dirname(dest), { recursive: true });
  await cp(src, dest, { force: true });
}

async function main() {
  for (const file of pyFiles) {
    await copyRecursive(path.join(repoRoot, file), path.join(webRoot, 'public', 'py', file));
  }

  const dataSrc = path.join(repoRoot, 'data');
  try {
    const dataStat = await stat(dataSrc);
    if (dataStat.isDirectory()) {
      await copyRecursive(dataSrc, path.join(webRoot, 'public', 'data'));
    }
  } catch {
    // data directory is optional
  }
}

await main();
