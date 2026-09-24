import path from "node:path"

import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// Relative asset paths let the same build run from a subpath (GitHub Pages) or the local server.
// In development, /api goes to `groundwork serve` on its default port.
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "./src") } },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  // One bundle of about 680 kB (about 200 kB compressed, measured 24 Sep 2026). Splitting it
  // gains little: every view but Method draws charts, so the chart library loads on arrival anyway.
  build: { chunkSizeWarningLimit: 800 },
})
