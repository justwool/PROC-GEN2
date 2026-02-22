# PROC-GEN2 React + Three.js Recreation

This web app recreates the spirit of `art.py` as an interactive browser renderer.

## Implemented parallels from `art.py`

- Seeded pseudo-random generation for reproducible runs
- Palette synthesis in HSV then conversion to RGB
- Border generation with height/amplitude/frequency/exponent
- Glow layer generation
- Ptych layout logic (`[2,3,2]`, horizontal/vertical variants)
- Multi-panel rendering on a single canvas
- Export current render to PNG

## Run

```bash
npm install
npm run dev
```

Visit `http://localhost:4173`.
