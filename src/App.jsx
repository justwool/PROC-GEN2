import { useMemo, useState } from 'react';
import ThreeArtwork from './ThreeArtwork.jsx';
import { PRESETS, generateSpec } from './procgen.js';

const DEFAULT_WIDTH = 1280;
const DEFAULT_HEIGHT = 900;

export default function App() {
  const [seedInput, setSeedInput] = useState(() => `${Math.floor(Math.random() * 1_000_000)}`);
  const [seed, setSeed] = useState(() => Number(seedInput));
  const [preset, setPreset] = useState('random');
  const [resolution, setResolution] = useState(DEFAULT_WIDTH);

  const spec = useMemo(() => {
    const width = resolution;
    const ratio = DEFAULT_WIDTH / DEFAULT_HEIGHT;
    const height = Math.round(width / ratio);
    return generateSpec(seed, width, height, preset);
  }, [seed, preset, resolution]);

  const regenerate = () => {
    const parsed = Number(seedInput);
    if (!Number.isFinite(parsed)) return;
    setSeed(Math.floor(parsed));
  };

  const randomize = () => {
    const next = `${Math.floor(Math.random() * 1_000_000)}`;
    setSeedInput(next);
    setSeed(Number(next));
  };

  const exportPng = () => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return;
    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = `proc-gen2-${preset}-${seed}.png`;
    link.click();
  };

  return (
    <main className="app-shell">
      <header>
        <h1>PROC-GEN2 React + Three.js</h1>
        <p>
          Recreation of the Python CLI art generator as a browser-first renderer with seed-based reproducibility,
          palette synthesis, glow/border layers, and ptych layouts.
        </p>
      </header>

      <section className="controls">
        <label>
          Seed
          <input value={seedInput} onChange={(e) => setSeedInput(e.target.value)} />
        </label>

        <label>
          Preset
          <select value={preset} onChange={(e) => setPreset(e.target.value)}>
            {Object.entries(PRESETS).map(([key, config]) => (
              <option key={key} value={key}>
                {config.name}
              </option>
            ))}
          </select>
        </label>

        <label>
          Resolution
          <select value={resolution} onChange={(e) => setResolution(Number(e.target.value))}>
            <option value={960}>960</option>
            <option value={1280}>1280</option>
            <option value={1600}>1600</option>
            <option value={2048}>2048</option>
          </select>
        </label>

        <div className="button-row">
          <button onClick={regenerate}>Regenerate</button>
          <button onClick={randomize}>Random Seed</button>
          <button onClick={exportPng}>Export PNG</button>
        </div>
      </section>

      <section>
        <ThreeArtwork spec={spec} />
        <pre className="meta">{JSON.stringify({ layout: spec.layout, seed: spec.seed, preset: spec.presetKey }, null, 2)}</pre>
      </section>
    </main>
  );
}
