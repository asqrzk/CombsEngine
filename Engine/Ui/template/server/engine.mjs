/**
 * Engine spawner — starts additional `combs serve` subprocesses on free
 * ports (roleplay's second role engine, agent engines, ...). Each engine
 * is its own PROCESS with its own GPU device (the one-device-per-adapter
 * limit is per-process), so engines on different ports run fully
 * isolated inference contexts.
 *
 * Endpoints (wired in proxy.mjs):
 *   POST /api/engine/spawn  {model}  → {port, url} (idempotent per model)
 *   GET  /api/engine               → [{model, port, url}]
 *   POST /api/engine/stop   {port} → {ok}
 *
 * The combs binary is resolved via COMBS_BIN, then PATH. Spawn requires
 * the `system:subprocess` permission (passkey dialog on the main tab).
 */

import { spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { bus as obsBus } from "./observe.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.join(HERE, "data");

/** model -> {model, port, url, child} */
const engines = new Map();

function combsBin() {
  return process.env.COMBS_BIN || "combs";
}

function resolveModelPath(model) {
  if (fs.existsSync(model)) return path.resolve(model);
  const home = process.env.COMBS_HOME ||
    path.join(process.env.HOME || process.env.USERPROFILE || "", ".cache", "combs");
  const preset = path.join(home, "models", model);
  if (fs.existsSync(preset)) return preset;
  return null;
}

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

async function waitHealthy(port, tries = 240) {
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/health`);
      if (res.ok) return true;
    } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

export async function handleEngine(req, res, url, readBody, send, requirePermission) {
  const action = url.pathname === "/api/engine"
    ? "list"
    : url.pathname.slice("/api/engine/".length);

  if (action === "list" && req.method === "GET") {
    return send(res, 200, {
      engines: [...engines.values()].map((e) => ({ model: e.model, port: e.port, url: e.url })),
    });
  }

  const body = JSON.parse((await readBody(req)).toString("utf8") || "{}");

  if (action === "stop" && req.method === "POST") {
    for (const [model, e] of engines) {
      if (e.port === body.port) {
        try { e.child.kill("SIGTERM"); } catch { /* dead */ }
        engines.delete(model);
        obsBus.event("proxy", "proxy.engine.stop", { attrs: { model, port: e.port } });
        return send(res, 200, { ok: true });
      }
    }
    return send(res, 404, { error: "no engine on that port" });
  }

  if (action !== "spawn" || req.method !== "POST") {
    return send(res, 404, { error: "unknown engine endpoint" });
  }

  const { model } = body;
  if (!model) return send(res, 400, { error: "need {model}" });

  // Idempotent: one engine per model per proxy.
  const existing = engines.get(model);
  if (existing) {
    if (await waitHealthy(existing.port, 1)) {
      return send(res, 200, { port: existing.port, url: existing.url, reused: true });
    }
    engines.delete(model);
  }

  // Spawning a subprocess is a privileged action — permission-checked
  // (passkey dialog on the main tab).
  if (!(await requirePermission(res, "system:subprocess", `start a second engine for ${model} on a new port`))) {
    return;
  }

  const modelPath = resolveModelPath(model);
  if (!modelPath) {
    return send(res, 400, {
      error: `model '${model}' not found locally — run: combs pull ${model}`,
    });
  }

  const port = await freePort();
  fs.mkdirSync(DATA_DIR, { recursive: true });
  const logFd = fs.openSync(path.join(DATA_DIR, `engine-${port}.log`), "a");
  const child = spawn(combsBin(), ["serve", "--model", modelPath, "--port", String(port)], {
    stdio: ["ignore", logFd, logFd],
  });
  child.on("exit", () => engines.delete(model));

  // A missing/misresolved binary fires an async 'error' (ENOENT) — without a
  // handler it is an UNHANDLED 'error' event that crashes the whole proxy.
  // Surface it as a clean 502 instead.
  const spawnError = await new Promise((resolve) => {
    child.once("error", (err) => resolve(err));
    // give the spawn a tick to fail before assuming it started
    setTimeout(() => resolve(null), 300);
  });
  if (spawnError) {
    try { child.kill("SIGTERM"); } catch { /* dead */ }
    return send(res, 502, {
      error: `could not start engine: ${spawnError.code === "ENOENT"
        ? `combs binary not found (tried '${combsBin()}') — put combs on PATH or set COMBS_BIN`
        : spawnError.message}`,
    });
  }

  const engineUrl = `http://localhost:${port}`;
  const entry = { model, port, url: engineUrl, child };
  engines.set(model, entry);
  obsBus.event("proxy", "proxy.engine.spawn", { attrs: { model, port, url: engineUrl, pid: child.pid } });
  obsBus.context("proxy", "proxy.engines", [...engines.values()].map((e) => ({ model: e.model, port: e.port, url: e.url })));

  if (!(await waitHealthy(port))) {
    try { child.kill("SIGTERM"); } catch { /* dead */ }
    engines.delete(model);
    obsBus.event("proxy", "proxy.engine.spawn_failed", { status: "error", attrs: { model, port } });
    return send(res, 502, { error: `engine did not come up — see server/data/engine-${port}.log` });
  }
  obsBus.event("proxy", "proxy.engine.healthy", { attrs: { model, port, url: engineUrl } });
  send(res, 200, { port, url: engineUrl });
}

/** Kills all spawned engines (called on proxy shutdown). */
export function stopAllEngines() {
  for (const e of engines.values()) {
    try { e.child.kill("SIGTERM"); } catch { /* dead */ }
  }
}
