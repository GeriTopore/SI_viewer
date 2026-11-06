import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the backend runs separately (`si-viewer --no-browser`) and
// API calls are proxied to it. `npm run build` writes into the Python package
// so the backend serves the app from the same origin.
const BACKEND = "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: BACKEND, ws: true },
    },
  },
  build: {
    outDir: "../backend/si_viewer/static",
    emptyOutDir: true,
  },
});
