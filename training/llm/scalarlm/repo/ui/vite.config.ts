import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

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
  test: {
    include: ["test/**/*.test.ts", "test/**/*.test.tsx"],
    environment: "node",
    globals: false,
    reporters: ["default"],
  },
}));
