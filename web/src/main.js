import { createTerminal } from './term.js';
import { bootPyodide, enqueueInput, interruptExecution, readBinary, runArtMain, snapshotFS } from './py.js';

const terminal = createTerminal(document.getElementById('terminal'), {
  onLine: handleLine,
  onInterrupt: handleInterrupt,
});

const galleryList = document.getElementById('gallery-list');
const preview = document.getElementById('preview');
const resetBtn = document.getElementById('reset-btn');
const stopBtn = document.getElementById('stop-btn');
const downloadLatestBtn = document.getElementById('download-latest');
const downloadSelectedBtn = document.getElementById('download-selected');

let pyReady = false;
let pythonRunning = false;
let commandPrompt = 'proc-gen2> ';
let baselineSnapshot = [];
let gallery = [];
let selectedIndex = -1;

async function init() {
  try {
    terminal.writeln('Booting Pyodide runtime...');
    await bootPyodide({ onWrite: (txt) => terminal.write(txt.replace(/\n/g, '\r\n')) });
    pyReady = true;
    baselineSnapshot = snapshotFS('/');
    terminal.writeln('Ready. Type help for commands.');
    terminal.setPrompt(commandPrompt);
    terminal.printPrompt();
  } catch (err) {
    pyReady = false;
    const msg =
      err?.stack ||
      err?.message ||
      (typeof err === 'object' ? JSON.stringify(err, null, 2) : String(err));

    terminal.writeln(`\r\n[boot error] ${msg}`);
  }
}

async function handleLine(line) {
  if (!pyReady) return;

  if (pythonRunning) {
    enqueueInput(line);
    await refreshGallery();
    return;
  }

  const cmd = line.trim();
  if (!cmd) {
    terminal.printPrompt();
    return;
  }

  if (cmd === 'help') {
    terminal.writeln('Commands: help, run, reset, ls, gallery, download latest, download <index>, clear');
  } else if (cmd === 'run') {
    await runPythonApp();
  } else if (cmd === 'reset') {
    await resetRuntime();
  } else if (cmd === 'ls') {
    lsFiles();
  } else if (cmd === 'gallery') {
    printGallery();
  } else if (cmd === 'download latest') {
    downloadLatest();
  } else if (cmd.startsWith('download ')) {
    const idx = Number(cmd.slice('download '.length));
    downloadIndex(idx);
  } else if (cmd === 'clear') {
    terminal.clear();
  } else {
    terminal.writeln(`Unknown command: ${cmd}`);
  }
  terminal.printPrompt();
}

async function runPythonApp() {
  pythonRunning = true;
  terminal.setInputEnabled(true);
  stopBtn.disabled = false;
  try {
    await runArtMain();
  } catch (err) {
    terminal.writeln(`\r\n[python error] ${String(err)}`);
  } finally {
    pythonRunning = false;
    stopBtn.disabled = true;
    await refreshGallery();
  }
}

function lsFiles() {
  const files = snapshotFS('/').map((x) => x.path).sort();
  files.forEach((path) => terminal.writeln(path));
}

function detectChangedPngs(oldSnap, newSnap) {
  const oldMap = new Map(oldSnap.map((f) => [f.path, `${f.size}-${f.mtime}`]));
  return newSnap.filter((f) => f.path.toLowerCase().endsWith('.png') && oldMap.get(f.path) !== `${f.size}-${f.mtime}`);
}

async function refreshGallery() {
  const next = snapshotFS('/');
  const changed = detectChangedPngs(baselineSnapshot, next);
  const now = new Date().toISOString();
  changed.forEach((item) => {
    if (!gallery.some((g) => g.path === item.path && g.mtime === item.mtime && g.size === item.size)) {
      gallery.push({ ...item, discoveredAt: now });
    }
  });
  baselineSnapshot = next;
  renderGallery();
}

function renderGallery() {
  galleryList.innerHTML = '';
  gallery.forEach((item, idx) => {
    const li = document.createElement('li');
    li.textContent = `[${idx}] ${item.path} @ ${item.discoveredAt}`;
    li.onclick = () => selectImage(idx);
    if (idx === selectedIndex) li.classList.add('selected');
    galleryList.appendChild(li);
  });
}

function selectImage(idx) {
  selectedIndex = idx;
  renderGallery();
  const item = gallery[idx];
  if (!item) return;
  const bytes = readBinary(item.path);
  const blob = new Blob([bytes], { type: 'image/png' });
  preview.src = URL.createObjectURL(blob);
}

function printGallery() {
  if (!gallery.length) {
    terminal.writeln('No generated PNG files detected yet.');
    return;
  }
  gallery.forEach((item, idx) => terminal.writeln(`[${idx}] ${item.path} (${item.size} bytes)`));
}

function triggerDownload(item) {
  if (!item) {
    terminal.writeln('No image selected.');
    return;
  }
  const bytes = readBinary(item.path);
  const blob = new Blob([bytes], { type: 'image/png' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = item.path.split('/').pop();
  a.click();
}

function downloadLatest() {
  triggerDownload(gallery[gallery.length - 1]);
}

function downloadIndex(idx) {
  if (!Number.isInteger(idx) || idx < 0 || idx >= gallery.length) {
    terminal.writeln('Invalid gallery index.');
    return;
  }
  triggerDownload(gallery[idx]);
}

function handleInterrupt() {
  if (!pythonRunning) return;
  interruptExecution();
  terminal.writeln('^C');
}

async function resetRuntime() {
  pyReady = false;
  pythonRunning = false;
  gallery = [];
  selectedIndex = -1;
  preview.src = '';
  terminal.clear();
  await init();
}

resetBtn.onclick = () => resetRuntime();
stopBtn.onclick = () => handleInterrupt();
downloadLatestBtn.onclick = () => downloadLatest();
downloadSelectedBtn.onclick = () => downloadIndex(selectedIndex);
stopBtn.disabled = true;

init();
