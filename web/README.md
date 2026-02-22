# PROC-GEN2 browser terminal bridge

This Vite + vanilla JS app runs the existing Python CLI (`art.py`) in-browser using Pyodide and xterm.js, preserving `input()`/`print()` semantics.

## Install / run

```bash
cd web
npm install
npm run dev
```

Open the local URL from Vite.

## Terminal commands

- `help`
- `run` (start CLI app)
- `reset` (full runtime reset)
- `ls` (list in-memory filesystem)
- `gallery` (list detected PNG outputs)
- `download latest`
- `download <index>`

Buttons:
- **Reset runtime**: reboots Pyodide and reloads files.
- **Stop**: sends KeyboardInterrupt (`Ctrl+C`) using interrupt buffer.

## Downloads / gallery

After each interaction/run step, the app snapshots Pyodide FS and detects new/changed `.png` files by path + mtime + size. The Gallery tracks those files regardless of naming scheme.

Click a gallery entry to preview, then use download controls.

## Notes / limitations

- First load is slower because Pyodide runtime is downloaded.
- The app is fully client-side (no backend).
- Files are loaded from the checked-out repo into Pyodide FS at startup.
