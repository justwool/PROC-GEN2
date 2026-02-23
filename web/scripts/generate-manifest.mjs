import { mkdir, readdir, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const webRoot = path.resolve(__dirname, '..');
const publicRoot = path.join(webRoot, 'public');

function shouldSkip(name) {
  return name.toLowerCase().endsWith('.zip');
}

async function listFilesRecursive(rootDir, relDir = '') {
  const dirPath = path.join(rootDir, relDir);
  let entries;
  try {
    entries = await readdir(dirPath, { withFileTypes: true });
  } catch {
    return [];
  }

  const out = [];
  for (const entry of entries) {
    if (shouldSkip(entry.name)) continue;
    const relPath = path.posix.join(relDir, entry.name);
    const absPath = path.join(rootDir, relPath);

    if (entry.isDirectory()) {
      out.push(...(await listFilesRecursive(rootDir, relPath)));
      continue;
    }

    const st = await stat(absPath);
    if (st.isFile()) {
      out.push(relPath);
    }
  }

  return out;
}

async function main() {
  const pyDir = path.join(publicRoot, 'py');
  const dataDir = path.join(publicRoot, 'data');

  const manifest = {
    py: (await listFilesRecursive(pyDir)).sort(),
    data: (await listFilesRecursive(dataDir)).sort(),
  };

  await mkdir(publicRoot, { recursive: true });
  await writeFile(path.join(publicRoot, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
}

await main();
