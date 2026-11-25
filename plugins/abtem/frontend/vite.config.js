import { defineConfig } from "vite";

// Builds into the Python package, which serves it at /plugins/abtem/. React
// comes from the app (``host.React``); Three.js is a chunk loaded on first use.
export default defineConfig({
  build: {
    outDir: "../src/si_viewer_abtem/frontend",
    emptyOutDir: true,
    target: "es2022",
    lib: { entry: "src/index.js", formats: ["es"], fileName: () => "index.js" },
    rollupOptions: { output: { chunkFileNames: "[name]-[hash].js" } },
  },
});
