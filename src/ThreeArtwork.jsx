import { useEffect, useRef } from 'react';
import * as THREE from 'three';

function colorArrayToVector3(color) {
  return new THREE.Vector3(color[0], color[1], color[2]);
}

function createShaderMaterial(painting, index) {
  const palette = painting.paletteRgb.slice(0, 8);
  while (palette.length < 8) palette.push(palette[palette.length - 1] ?? [1, 1, 1]);

  const paletteVec = palette.map(colorArrayToVector3);
  const borderColors = painting.borders.slice(0, 6).map((b) => colorArrayToVector3(b.color));
  while (borderColors.length < 6) borderColors.push(new THREE.Vector3(0, 0, 0));

  const borderData = painting.borders.slice(0, 6).map((b) => [b.height, b.amplitude, b.frequency, b.exponent]);
  while (borderData.length < 6) borderData.push([0, 0, 1, 1]);

  return new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uPalette: { value: paletteVec },
      uBorderColors: { value: borderColors },
      uBorders: { value: borderData },
      uBorderCount: { value: Math.min(painting.borders.length, 6) },
      uNoiseScale: { value: painting.params.noiseScale },
      uFlowA: { value: painting.params.flowA },
      uFlowB: { value: painting.params.flowB },
      uMixBias: { value: painting.params.mixBias },
      uGrain: { value: painting.params.grain },
      uGlowPos: { value: new THREE.Vector2(painting.glow.x, painting.glow.y) },
      uGlowHeight: { value: painting.glow.height },
      uGlowCircular: { value: painting.glow.circular ? 1 : 0 },
      uGlowColor: { value: colorArrayToVector3(painting.glow.color) },
      uIndex: { value: index },
    },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      varying vec2 vUv;
      uniform float uTime;
      uniform vec3 uPalette[8];
      uniform vec3 uBorderColors[6];
      uniform vec4 uBorders[6];
      uniform int uBorderCount;
      uniform float uNoiseScale;
      uniform float uFlowA;
      uniform float uFlowB;
      uniform float uMixBias;
      uniform float uGrain;
      uniform vec2 uGlowPos;
      uniform float uGlowHeight;
      uniform int uGlowCircular;
      uniform vec3 uGlowColor;
      uniform float uIndex;

      float hash(vec2 p) {
        p = fract(p * vec2(234.34, 546.21));
        p += dot(p, p + 23.45);
        return fract(p.x * p.y);
      }

      float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        float a = hash(i);
        float b = hash(i + vec2(1.0, 0.0));
        float c = hash(i + vec2(0.0, 1.0));
        float d = hash(i + vec2(1.0, 1.0));
        vec2 u = f * f * (3.0 - 2.0 * f);
        return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
      }

      vec3 paletteMix(float t) {
        float idx = clamp(t * 7.0, 0.0, 7.0);
        float i0 = floor(idx);
        float i1 = min(7.0, i0 + 1.0);
        float m = fract(idx);
        return mix(uPalette[int(i0)], uPalette[int(i1)], smoothstep(0.0, 1.0, m));
      }

      float borderSignal(vec2 uv, vec4 b) {
        float y = b.x + sin((uv.x + uTime * 0.03) * b.z * 6.2831) * b.y;
        float d = abs(uv.y - y);
        return 1.0 - smoothstep(0.0, 0.025 + 0.02 / max(0.5, b.w), d);
      }

      void main() {
        vec2 uv = vUv;
        float t = uTime * 0.12 + uIndex * 2.0;

        float f1 = noise(uv * (4.0 * uNoiseScale) + vec2(t * uFlowA, -t * 0.4));
        float f2 = noise((uv.yx + 0.13) * (2.0 * uNoiseScale) + vec2(-t * 0.3, t * uFlowB));
        float f3 = sin((uv.x + f2) * 8.0 + t * 1.2) * 0.5 + 0.5;

        float mixer = clamp((f1 * 0.55 + f2 * 0.35 + f3 * 0.2) + uMixBias, 0.0, 1.0);
        vec3 color = paletteMix(mixer);

        for (int i = 0; i < 6; i++) {
          if (i >= uBorderCount) break;
          float s = borderSignal(uv, uBorders[i]);
          color = mix(color, uBorderColors[i], s * 0.55);
        }

        vec2 glowUv = uv - uGlowPos;
        float gd = uGlowCircular == 1 ? length(glowUv) : abs(glowUv.y) * 0.75 + abs(glowUv.x) * 0.25;
        float glow = exp(-gd * 4.0) * uGlowHeight;
        color += uGlowColor * glow * 0.35;

        float grain = (hash(uv * 550.0 + t) - 0.5) * uGrain;
        color += grain;

        gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
      }
    `,
  });
}

export default function ThreeArtwork({ spec }) {
  const mountRef = useRef(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return undefined;

    const width = container.clientWidth;
    const height = Math.floor(width * (spec.height / spec.width));

    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
    camera.position.z = 1;

    const rowCount = spec.layout.length;
    const meshes = [];
    let paintIndex = 0;

    spec.layout.forEach((colsInRow, rowIdx) => {
      for (let col = 0; col < colsInRow; col += 1) {
        const panelW = 2 / colsInRow;
        const panelH = 2 / rowCount;
        const x = -1 + panelW * (col + 0.5);
        const y = 1 - panelH * (rowIdx + 0.5);

        const geo = new THREE.PlaneGeometry(panelW * 0.97, panelH * 0.97, 1, 1);
        const mat = createShaderMaterial(spec.paintings[paintIndex], paintIndex);
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(x, y, 0);
        scene.add(mesh);
        meshes.push(mesh);
        paintIndex += 1;
      }
    });

    let raf;
    const clock = new THREE.Clock();

    const render = () => {
      const t = clock.getElapsedTime();
      meshes.forEach((m) => {
        m.material.uniforms.uTime.value = t;
      });
      renderer.render(scene, camera);
      raf = requestAnimationFrame(render);
    };

    render();

    const onResize = () => {
      const w = container.clientWidth;
      const h = Math.floor(w * (spec.height / spec.width));
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      meshes.forEach((m) => {
        m.geometry.dispose();
        m.material.dispose();
      });
      renderer.dispose();
    };
  }, [spec]);

  return <div className="art-canvas" ref={mountRef} />;
}
