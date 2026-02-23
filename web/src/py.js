import { snapshotDir, writeBinary, writeText } from './fs.js';

const PYODIDE_URL = 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js';

let pyodide;
let stdinQueue = [];
let stdinResolvers = [];
let interruptBuffer;

function normalizeLine(line) {
  return line.replace(/\r\n/g, '\n').replace(/\r/g, '\n').replace(/\n$/, '');
}

async function loadScript(src) {
  await new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-src="${src}"]`);
    if (existing) return resolve();
    const script = document.createElement('script');
    script.src = src;
    script.dataset.src = src;
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

export async function bootPyodide({ onWrite }) {
  stdinQueue = [];
  stdinResolvers = [];
  await loadScript(PYODIDE_URL);
  pyodide = await globalThis.loadPyodide({
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/',
  });

  if (typeof SharedArrayBuffer !== 'undefined' && pyodide.setInterruptBuffer) {
    interruptBuffer = new Uint8Array(new SharedArrayBuffer(1));
    pyodide.setInterruptBuffer(interruptBuffer);
  } else {
    interruptBuffer = null;
  }

  globalThis.__stdin_readline__ = () => {
    if (stdinQueue.length > 0) return Promise.resolve(stdinQueue.shift());
    return new Promise((resolve) => stdinResolvers.push(resolve));
  };

  globalThis.__term_write__ = (text) => onWrite(text);

  await pyodide.runPythonAsync(`
import sys
import builtins
import js
import pyodide

class _JSWriter:
    def write(self, text):
        js.__term_write__(text)
        return len(text)
    def flush(self):
        return None

sys.stdout = _JSWriter()
sys.stderr = _JSWriter()

async def _ainput(prompt=""):
    if prompt:
        print(prompt, end="")
    return await js.__stdin_readline__()

def input(prompt=""):
    return pyodide.ffi.run_sync(_ainput(prompt))

builtins.input = input
`);

  await pyodide.runPythonAsync(`
import os, errno
try:
    st = os.stat("/data")
    # if it exists but is not a dir, remove it
    if not os.path.isdir("/data"):
        os.remove("/data")
        os.mkdir("/data")
except FileNotFoundError:
    os.mkdir("/data")
`);

  await loadProjectFiles();
  return pyodide;
}

export function enqueueInput(line) {
  const normalized = normalizeLine(line);
  if (stdinResolvers.length > 0) {
    stdinResolvers.shift()(normalized);
  } else {
    stdinQueue.push(normalized);
  }
}

export function interruptExecution() {
  if (!interruptBuffer) return false;
  interruptBuffer[0] = 2;
  return true;
}

export async function runArtMain() {
  await pyodide.runPythonAsync(`
import importlib
import art
importlib.reload(art)
art.main()
`);
}

function isTextFile(path) {
  return /\.(txt|py|json|gitkeep)$/i.test(path);
}

export async function loadProjectFiles() {
  const manifestRes = await fetch('/manifest.json');
  if (!manifestRes.ok) {
    throw new Error(`fetch /manifest.json -> ${manifestRes.status}`);
  }

  const manifest = await manifestRes.json();
  const pyFiles = Array.isArray(manifest?.py) ? manifest.py : [];
  const dataFiles = Array.isArray(manifest?.data) ? manifest.data : [];

  for (const name of pyFiles) {
    const url = `/py/${name}`;
    const fsPath = `/${name}`;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`fetch ${url} -> ${res.status} (write ${fsPath})`);
      if (isTextFile(name)) writeText(pyodide, fsPath, await res.text());
      else writeBinary(pyodide, fsPath, new Uint8Array(await res.arrayBuffer()));
    } catch (e) {
      throw new Error(`loadProjectFiles failed on ${name} -> ${fsPath}: ${e?.message || e}`);
    }
  }

  for (const name of dataFiles) {
    const url = `/data/${name}`;
    const fsPath = `/data/${name}`;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`fetch ${url} -> ${res.status} (write ${fsPath})`);
      if (isTextFile(name)) writeText(pyodide, fsPath, await res.text());
      else writeBinary(pyodide, fsPath, new Uint8Array(await res.arrayBuffer()));
    } catch (e) {
      throw new Error(`loadProjectFiles failed on data/${name} -> ${fsPath}: ${e?.message || e}`);
    }
  }
}

export function snapshotFS(root = '/') {
  return snapshotDir(pyodide, root);
}

export function readBinary(path) {
  return pyodide.FS.readFile(path, { encoding: 'binary' });
}
