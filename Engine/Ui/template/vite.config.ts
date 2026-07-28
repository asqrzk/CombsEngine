import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

// server/dev.mjs launches the permission proxy on a free port and hands it
// over via COMBS_PROXY_PORT (8787 for `npm start` / standalone runs).
const proxyTarget = `http://localhost:${process.env.COMBS_PROXY_PORT ?? 8787}`;

export default defineConfig({
  plugins: [svelte(), tailwindcss()],
  server: {
    port: 5173,
    watch: {
      // The proxy writes runtime state under server/ (data/, logs, keys,
      // grants) — these are NOT source changes; without this, vite
      // full-reloads the page mid-session and wipes all UI state.
      ignored: [
        "**/server/data/**",
        "**/server/*.log",
        "**/server/master.key",
        "**/server/permissions.json",
        "**/server/manifest.json",
        "**/server/authn.json",
      ],
    },
    proxy: {
      // Every backend call goes same-origin → Vite → the permission proxy
      // (server/proxy.mjs). The browser never talks to the internet or the
      // inference server directly.
      "/api": {
        target: proxyTarget,
        changeOrigin: true,
        ws: true, // /api/observe/ws (Control Tower realtime bus)
      },
    },
  },
});
