/**
 * Combs UI proxy — the single network & file gatekeeper for the app.
 *
 * The browser NEVER talks to the internet or the disk directly:
 *   - every outbound request goes through  POST /api/relay
 *   - every file write/read goes through   /api/files/<name>
 * and both are permission-checked HERE, on the backend. The frontend's
 * only job is to show the permission dialog and forward the decision.
 *
 * Endpoints:
 *   POST /api/relay                {url, method?, headers?, body?, scope, detail?}
 *                                  → relays bytes upstream (SSE-safe streaming)
 *                                  → 428 {permissionRequired:{scope,detail}} when ungranted
 *   GET  /api/permissions          → current grants
 *   POST /api/permissions/decide   {scope, grant: once|session|always|deny|reset}
 *   POST /api/files/<name>?scope=storage:chats   write body to data/<name> (gated)
 *   GET  /api/files/<name>         read it back
 *   DELETE /api/files/<name>?scope=…             delete (gated)
 *   GET  /api/monitor              {upBytes, downBytes, dataBytes, sys}
 *
 * Options: --port <n> (default 8787)   --serve <dir> (also serve static files)
 * Zero dependencies; Node >= 18.
 */

import http from "node:http";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";
import { fileURLToPath } from "node:url";
import { forgetStorage, initZeroTrust, openFromStorage, sealForStorage } from "./zerotrust.mjs";
import { handleAuthn } from "./authn.mjs";
import { handleSandbox } from "./sandbox.mjs";
import { handleEngine, stopAllEngines } from "./engine.mjs";
import { bus as obsBus, handleObserve, handleObserveUpgrade, initObserve } from "./observe.mjs";
import { getSysStats, startSysStats } from "./sysstats.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.join(HERE, "data");
const GRANTS_FILE = path.join(HERE, "permissions.json");

const args = process.argv.slice(2);
const PORT = Number(args[args.indexOf("--port") + 1] || 8787);
const SERVE_DIR = args.includes("--serve") ? path.resolve(args[args.indexOf("--serve") + 1]) : null;

const VALID_GRANTS = new Set(["once", "session", "always", "deny", "reset"]);

// ---------------------------------------------------------------------------
// Permission store (backend-owned). "always"/"deny" persist to disk;
// "session" grants live in memory; "once" allows exactly one attempt.
// ---------------------------------------------------------------------------
let persisted = {};
try {
  persisted = JSON.parse(fs.readFileSync(GRANTS_FILE, "utf8"));
} catch { /* first run */ }
const session = new Map(); // scope -> "session" | "deny"
const onceAllowed = new Set(); // scopes approved for a single attempt

function checkScope(scope) {
  if (persisted[scope] === "always") return "allow";
  if (persisted[scope] === "deny") return "deny";
  if (session.get(scope) === "session") return "allow";
  if (session.get(scope) === "deny") return "deny";
  if (onceAllowed.has(scope)) {
    onceAllowed.delete(scope); // consumed
    return "allow";
  }
  return "ask";
}

async function decide(scope, grant) {
  if (!VALID_GRANTS.has(grant)) return;
  if (grant === "reset") {
    delete persisted[scope];
    session.delete(scope);
    onceAllowed.delete(scope);
  } else if (grant === "always" || grant === "deny") {
    persisted[scope] = grant;
  } else if (grant === "session") {
    session.set(scope, "session");
  } else if (grant === "once") {
    onceAllowed.add(scope);
  }
  await fsp.mkdir(DATA_DIR, { recursive: true });
  await fsp.writeFile(GRANTS_FILE, JSON.stringify(persisted, null, 2));
}

// ---------------------------------------------------------------------------
// Monitor counters (the backend sees every byte, so these are the truth).
// ---------------------------------------------------------------------------
let upBytes = 0;
let downBytes = 0;

async function dataBytes() {
  let total = 0;
  try {
    for (const f of await fsp.readdir(DATA_DIR)) {
      total += (await fsp.stat(path.join(DATA_DIR, f))).size;
    }
  } catch { /* empty */ }
  return total;
}

// Cache the disk scan briefly: the monitor polls every ~2s, and during
// roleplay server/data/ holds continuously-growing engine logs — re-stat'ing
// the whole directory on every poll needlessly slows the proxy.
let dataBytesCache = { at: 0, value: 0 };
async function dataBytesCached() {
  const now = Date.now();
  if (now - dataBytesCache.at > 2000) {
    dataBytesCache = { at: now, value: await dataBytes() };
  }
  return dataBytesCache.value;
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------
function send(res, status, body, headers = {}) {
  const isObj = typeof body === "object" && body !== null && !Buffer.isBuffer(body);
  const payload = isObj ? JSON.stringify(body) : body;
  res.writeHead(status, {
    "content-type": isObj ? "application/json" : "application/octet-stream",
    "access-control-allow-origin": "*",
    ...headers,
  });
  res.end(payload);
}

async function readBody(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  return Buffer.concat(chunks);
}

async function requirePermission(res, scope, detail) {
  const verdict = checkScope(scope);
  if (verdict === "allow") return true;
  if (verdict === "deny") {
    send(res, 403, { error: `permission denied: ${scope}` });
    return false;
  }
  // 428 Precondition Required — the frontend shows the dialog, POSTs the
  // decision to /api/permissions/decide, then retries.
  send(res, 428, { permissionRequired: { scope, detail } });
  return false;
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------
async function handleRelay(req, res) {
  const { url, method = "GET", headers = {}, body = null, scope, detail } = JSON.parse(
    (await readBody(req)).toString("utf8") || "{}",
  );
  if (!url || !scope) return send(res, 400, { error: "relay needs {url, scope}" });
  if (!(await requirePermission(res, scope, detail ?? `${method} ${url}`))) return;

  const spanId = `r${Date.now().toString(36)}${Math.floor(Math.random() * 1e6).toString(36)}`;
  const traceId = `t${spanId}`;
  const start = Date.now();
  obsBus.spanStart("proxy", "proxy.relay", {
    traceId, spanId,
    input: { method, url, body: body && body.length < 2000 ? tryJson(body) : undefined },
    attrs: { method, url, scope },
  });

  upBytes += body ? Buffer.byteLength(body) : 0;
  let upstream;
  try {
    upstream = await fetch(url, {
      method,
      headers,
      body: body ?? undefined,
      cache: "no-store", // SSE: no buffering
    });
  } catch (e) {
    obsBus.spanEnd("proxy", "proxy.relay", { traceId, spanId, status: "error", error: e.message, attrs: { durationMs: Date.now() - start } });
    return send(res, 502, { error: `upstream unreachable: ${e.message}` });
  }

  const pass = {};
  for (const h of ["content-type", "cache-control"]) {
    const v = upstream.headers.get(h);
    if (v) pass[h] = v;
  }
  res.writeHead(upstream.status, { ...pass, "access-control-allow-origin": "*" });
  if (!upstream.body) {
    obsBus.spanEnd("proxy", "proxy.relay", { traceId, spanId, status: upstream.status < 400 ? "ok" : "error", attrs: { status: upstream.status, durationMs: Date.now() - start } });
    return res.end();
  }
  let bytes = 0;
  try {
    for await (const chunk of Readable.fromWeb(upstream.body)) {
      downBytes += chunk.length;
      bytes += chunk.length;
      res.write(chunk);
    }
  } catch { /* client aborted */ }
  res.end();
  obsBus.spanEnd("proxy", "proxy.relay", { traceId, spanId, status: upstream.status < 400 ? "ok" : "error", attrs: { status: upstream.status, bytes, durationMs: Date.now() - start } });
}

function tryJson(s) {
  try { return JSON.parse(s); } catch { return s; }
}

function safeName(name) {
  const clean = path.basename(name); // no traversal
  if (!clean || clean.startsWith(".")) throw new Error("bad file name");
  return clean;
}

async function handleFiles(req, res, url) {
  const name = safeName(decodeURIComponent(url.pathname.slice("/api/files/".length)));
  const file = path.join(DATA_DIR, name);
  const scope = url.searchParams.get("scope") || "storage:cache";

  if (req.method === "GET") {
    try {
      const blob = await fsp.readFile(file, "utf8");
      // zero-trust: decrypt + verify integrity hash before anything leaves
      const data = await openFromStorage(name, blob);
      downBytes += data.length;
      send(res, 200, Buffer.from(data), { "content-type": "application/json" });
    } catch (e) {
      if (e.code === "ENOENT") return send(res, 404, { error: "not found" });
      send(res, 409, { error: `integrity check failed: ${e.message}` });
    }
    return;
  }
  if (!(await requirePermission(res, scope, `${req.method} ${name} on this device`))) return;

  if (req.method === "POST") {
    const body = await readBody(req);
    // zero-trust: encrypt at rest (per-file key, wrapped; hash in manifest)
    const { blob } = await sealForStorage(name, body);
    await fsp.mkdir(DATA_DIR, { recursive: true });
    await fsp.writeFile(file, blob, "utf8");
    upBytes += body.length;
    return send(res, 200, { ok: true, bytes: body.length });
  }
  if (req.method === "DELETE") {
    await fsp.rm(file, { force: true });
    await forgetStorage(name);
    return send(res, 200, { ok: true });
  }
  send(res, 405, { error: "method not allowed" });
}

const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png",
  ".ico": "image/x-icon", ".woff2": "font/woff2",
};

async function serveStatic(res, pathname) {
  const rel = pathname === "/" ? "index.html" : pathname.slice(1);
  const file = path.join(SERVE_DIR, path.normalize(rel));
  if (!file.startsWith(SERVE_DIR)) return send(res, 403, { error: "forbidden" });
  try {
    const data = await fsp.readFile(file);
    send(res, 200, data, { "content-type": MIME[path.extname(file)] ?? "application/octet-stream" });
  } catch {
    // SPA fallback
    try {
      const data = await fsp.readFile(path.join(SERVE_DIR, "index.html"));
      send(res, 200, data, { "content-type": "text/html" });
    } catch {
      send(res, 404, { error: "not found" });
    }
  }
}

// ---------------------------------------------------------------------------
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  try {
    if (req.method === "OPTIONS") {
      res.writeHead(204, {
        "access-control-allow-origin": "*",
        "access-control-allow-methods": "GET,POST,DELETE,OPTIONS",
        "access-control-allow-headers": "content-type",
      });
      return res.end();
    }
    if (url.pathname.startsWith("/api/auth/passkey/")) {
      return await handleAuthn(req, res, url, readBody, send);
    }
    if (url.pathname === "/api/relay" && req.method === "POST") return await handleRelay(req, res);
    if (url.pathname === "/api/permissions" && req.method === "GET") {
      return send(res, 200, { persisted, session: Object.fromEntries(session) });
    }
    if (url.pathname === "/api/permissions/decide" && req.method === "POST") {
      const { scope, grant } = JSON.parse((await readBody(req)).toString("utf8") || "{}");
      if (!scope || !VALID_GRANTS.has(grant)) return send(res, 400, { error: "need {scope, grant}" });
      await decide(scope, grant);
      obsBus.event("proxy", "proxy.permissions.decide", { attrs: { scope, grant } });
      return send(res, 200, { ok: true });
    }
    if (url.pathname.startsWith("/api/observe")) {
      return await handleObserve(req, res, url, send, readBody);
    }
    if (url.pathname.startsWith("/api/sandbox/") && req.method === "POST") {
      return await handleSandbox(req, res, url, readBody, send, requirePermission);
    }
    if (url.pathname.startsWith("/api/engine")) {
      return await handleEngine(req, res, url, readBody, send, requirePermission);
    }
    if (url.pathname.startsWith("/api/files/")) return await handleFiles(req, res, url);
    if (url.pathname === "/api/monitor" && req.method === "GET") {
      return send(res, 200, { upBytes, downBytes, dataBytes: await dataBytesCached(), sys: getSysStats() });
    }
    if (SERVE_DIR) return await serveStatic(res, url.pathname);
    send(res, 404, { error: "not found" });
  } catch (e) {
    send(res, 500, { error: e.message });
  }
});

await initZeroTrust();
await initObserve();
startSysStats();
process.on("exit", stopAllEngines);
process.on("SIGINT", () => { stopAllEngines(); process.exit(0); });
process.on("SIGTERM", () => { stopAllEngines(); process.exit(0); });
server.on("upgrade", (req, socket) => {
  if (!handleObserveUpgrade(req, socket)) socket.destroy();
});
server.listen(PORT, () => {
  console.log(`[combs-proxy] listening on http://localhost:${PORT}`);
  console.log(`[combs-proxy] data dir: ${DATA_DIR}`);
  if (SERVE_DIR) console.log(`[combs-proxy] serving static app from ${SERVE_DIR}`);
});
