export const PRESETS = {
  random: { name: 'Random', paletteSize: [4, 11], panelChance: 0.45, intricacy: 0.55 },
  single: { name: 'Single Painting', paletteSize: [4, 11], panelChance: 0.0, intricacy: 0.55 },
  ptych: { name: 'Ptych Focused', paletteSize: [6, 14], panelChance: 1.0, intricacy: 0.72 },
  soft: { name: 'Soft/Pastel', paletteSize: [4, 9], panelChance: 0.4, intricacy: 0.35 },
};

export function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function randRange(rng, min, max) {
  return min + (max - min) * rng();
}

function randomHSV(rng, paletteSize, pastel = 0) {
  const hues = [];
  const sats = [];
  const vals = [];

  const blocks = Math.max(3, Math.floor(randRange(rng, paletteSize * 0.8, paletteSize * 1.7)));
  const hueShift = rng();

  for (let i = 0; i < paletteSize; i += 1) {
    const hPart = Math.floor(rng() * blocks) / blocks;
    hues.push(((hPart + hueShift) % 1) * 360);
    sats.push(Math.pow(rng(), 1.2 + pastel) * (0.8 - pastel * 0.35));
    vals.push(0.3 + Math.pow(rng(), 0.5) * 0.7);
  }

  return hues.map((h, i) => [h, sats[i], vals[i]]);
}

function hsvToRgb(h, s, v) {
  const c = v * s;
  const hh = (h / 60) % 6;
  const x = c * (1 - Math.abs((hh % 2) - 1));
  let [r, g, b] = [0, 0, 0];

  if (hh >= 0 && hh < 1) [r, g, b] = [c, x, 0];
  else if (hh < 2) [r, g, b] = [x, c, 0];
  else if (hh < 3) [r, g, b] = [0, c, x];
  else if (hh < 4) [r, g, b] = [0, x, c];
  else if (hh < 5) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];

  const m = v - c;
  return [r + m, g + m, b + m];
}

function makeLayout(rng, width, height, panelChance) {
  if (rng() > panelChance) return [1];
  const options = [2, 3, 4, 6, 7];
  const n = options[Math.floor(rng() * options.length)];

  if (n === 7) return [2, 3, 2];
  if (width > height) return n === 6 ? [2, 2, 2] : Array.from({ length: n }, () => 1);
  return n === 6 ? [3, 3] : [n];
}

function makeBorders(rng, paletteRgb, countBoost = 0) {
  const count = Math.max(1, Math.floor(randRange(rng, 2, 6 + countBoost)));
  const borders = [];

  for (let i = 0; i < count; i += 1) {
    borders.push({
      height: randRange(rng, 0.1, 0.95),
      amplitude: randRange(rng, 0.0, 0.17),
      frequency: randRange(rng, 0.4, 4.4),
      exponent: Math.pow(2, randRange(rng, -1, 2.8)),
      color: paletteRgb[Math.floor(rng() * paletteRgb.length)],
    });
  }

  borders.sort((a, b) => b.height - a.height);
  return borders;
}

function makeGlow(rng, paletteRgb) {
  return {
    circular: rng() > 0.45,
    height: randRange(rng, 0.15, 0.8),
    x: randRange(rng, 0.2, 0.8),
    y: randRange(rng, 0.2, 0.8),
    color: paletteRgb[Math.floor(rng() * paletteRgb.length)],
  };
}

function makePainting(rng, preset) {
  const size = Math.floor(randRange(rng, preset.paletteSize[0], preset.paletteSize[1]));
  const hsv = randomHSV(rng, size, 1 - preset.intricacy);
  const paletteRgb = hsv.map(([h, s, v]) => hsvToRgb(h, s, v));

  return {
    paletteRgb,
    glow: makeGlow(rng, paletteRgb),
    borders: makeBorders(rng, paletteRgb, preset.intricacy * 2),
    params: {
      noiseScale: randRange(rng, 0.4, 4.0),
      flowA: randRange(rng, 0.5, 5.2) * (0.5 + preset.intricacy),
      flowB: randRange(rng, 0.5, 4.7),
      mixBias: randRange(rng, -0.4, 0.4),
      grain: randRange(rng, 0.01, 0.07),
    },
  };
}

export function generateSpec(seed, width, height, presetKey) {
  const preset = PRESETS[presetKey] ?? PRESETS.random;
  const rng = mulberry32(seed);
  const layout = makeLayout(rng, width, height, preset.panelChance);
  const count = layout.reduce((a, b) => a + b, 0);

  return {
    seed,
    width,
    height,
    presetKey,
    layout,
    paintings: Array.from({ length: count }, () => makePainting(rng, preset)),
  };
}
