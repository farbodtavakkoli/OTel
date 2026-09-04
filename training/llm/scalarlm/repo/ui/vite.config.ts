/// <reference types="vitest" />
import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The production bundle is served by FastAPI at /app/. Hashed assets live under
// /app/assets/ and are cached indefinitely; index.html is served with no-cache.
// In dev, Vite runs on :5173 and proxies API traffic to a running ScalarLM pod.
//
// Under vitest the `/app/` base breaks internal module-URL resolution
// (node_modules paths get the prefix stripped), so tests fall back to the
// default "/" base. VITEST=true is set automatically by the vitest runner.
export default defineConfig(({ mode }) => ({
  base: process.env.VITEST ? "/" : "/app/",
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  build: {
    outDir: "dist",
    assetsDir: "assets",
    emptyOutDir: true,
    sourcemap: mode === "development",
  },
  server: {
    port: 5173,
    proxy: {
      "/v1": {
        target: process.env.SCALARLM_API_URL ?? "http://localhost:8000",
        changeOrigin: true,
      },
      "/app/api-config.json": {
        target: process.env.SCALARLM_API_URL ?? "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  // Vitest lives alongside Vite config so the same alias/resolver setup
  // applies to unit tests. See docs/test-plan.md §5.14 for the vocabulary.
  test: {
    include: ["test/**/*.test.ts", "test/**/*.test.tsx"],
    // Default: plain Node environment. Component tests that need DOM opt
    // into jsdom explicitly via a `// @vitest-environment jsdom` comment
    // at the top of the file to keep the default-environment tests fast.
    environment: "node",
    globals: false,
    reporters: ["default"],
  },
}));
