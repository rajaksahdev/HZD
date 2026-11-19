import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the map runs on :5173 and forwards /v1 to the API (API_URL, default :8000).
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/v1": process.env.API_URL || "http://localhost:8000" } },
  build: { chunkSizeWarningLimit: 1500 }, // MapLibre alone is about 800 kB
});
