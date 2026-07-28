/**
 * Dev launcher: starts the permission proxy AND the Vite dev server.
 * The UI calls same-origin /api/* which Vite forwards to the proxy
 * (see vite.config.ts), so even in dev every byte crosses the proxy.
 *
 * The proxy port is picked free at launch (COMBS_PROXY_PORT) so multiple
 * chew apps can run side by side without colliding on 8787.
 */

import { spawn } from "node:child_process";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, "..");
const viteBin = path.join(ROOT, "node_modules", "vite", "bin", "vite.js");

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

const proxyPort = process.env.COMBS_PROXY_PORT
  ? Number(process.env.COMBS_PROXY_PORT)
  : await freePort();

const children = [
  spawn(process.execPath, [path.join(HERE, "proxy.mjs"), "--port", String(proxyPort)], {
    stdio: "inherit",
  }),
  spawn(process.execPath, [viteBin, ...process.argv.slice(2)], {
    cwd: ROOT,
    stdio: "inherit",
    env: { ...process.env, COMBS_PROXY_PORT: String(proxyPort) },
  }),
];

function shutdown(code = 0) {
  for (const c of children) c.kill("SIGTERM");
  process.exit(code);
}
process.on("SIGINT", () => shutdown());
process.on("SIGTERM", () => shutdown());
for (const c of children) c.on("exit", (code) => shutdown(code ?? 0));
