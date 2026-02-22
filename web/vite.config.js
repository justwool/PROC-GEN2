import path from 'node:path';
import { defineConfig } from 'vite';

const repoRoot = path.resolve(__dirname, '..');

export default defineConfig({
  server: {
    fs: {
      allow: ['..'],
    },
  },
  define: {
    __REPO_ROOT__: JSON.stringify(repoRoot.replace(/\\/g, '/')),
  },
});
