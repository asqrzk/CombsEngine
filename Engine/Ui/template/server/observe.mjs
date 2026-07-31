/**
 * Observe — the proxy's runtime adapter for @combs/observe.
 *
 * The isomorphic core (Engine/Js/observe) is dependency-free TypeScript. This
 * module is the Node runtime binding used INSIDE the zero-dependency proxy: a
 * plain-JS EventBus with the same event contract, plus the sinks the proxy
 * owns — a WebSocket broadcast (Control Tower), an NDJSON file, and a REST
 * snapshot. Keeping it as .mjs avoids forcing an npm/TS dependency into the
 * proxy while preserving the exact ObsEvent shape the tower consumes.
 *
 * Endpoints (wired in proxy.mjs):
 *   GET  /api/observe       → { events: [...recent], sources: {...} }
 *   POST /api/observe       → publish a client event (orchestration spans)
 *   GET  /api/observe/ws    → WebSocket upgrade; every event broadcast live
 */

import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.join(HERE, "data");
const LOG_FILE = path.join(DATA_DIR, "observe.ndjson");

const SECRET_RE =
  /(api[_-]?key|authorization|bearer|password|secret|token|sk-[a-z0-9]{8,})/gi;
// Keys that match SECRET_RE but are benign sampling parameters.
const SAFE_KEYS = new Set(["max_tokens", "stop_token_ids"]);

function scrub(v) {
  if (typeof v === "string") return v.replace(SECRET_RE, (m) => "•".repeat(Math.min(m.length, 12)));
  if (Array.isArray(v)) return v.map(scrub);
  if (v && typeof v === "object") {
    const out = {};
    for (const [k, val] of Object.entries(v)) {
      out[k] = SECRET_RE.test(k) && !SAFE_KEYS.has(k) ? "***" : scrub(val);
      SECRET_RE.lastIndex = 0;
    }
    return out;
  }
  return v;
}

let counter = 0;
function id() {
  counter = (counter + 1) & 0xffffff;
  return `${Date.now().toString(36)}-${counter.toString(36)}-${Math.floor(Math.random() * 0xffffffff).toString(36)}`;
}

class Bus {
  constructor(capacity = 2000) {
    this.capacity = capacity;
    this.ring = [];
    this.subs = new Set();
  }
  publish(partial) {
    let event = { id: id(), ts: Date.now(), ...partial };
    try {
      event = { ...event,
        input: event.input !== undefined ? scrub(event.input) : undefined,
        output: event.output !== undefined ? scrub(event.output) : undefined,
        context: event.context !== undefined ? scrub(event.context) : undefined,
      };
    } catch { /* redaction never breaks */ }
    this.ring.push(event);
    if (this.ring.length > this.capacity) this.ring.splice(0, this.ring.length - this.capacity);
    for (const fn of this.subs) { try { fn(event); } catch { /* subscriber never breaks */ } }
    return event;
  }
  subscribe(fn) { this.subs.add(fn); return () => this.subs.delete(fn); }
  snapshot() { return [...this.ring]; }
  spanStart(source, name, extra = {}) { return this.publish({ source, kind: "span.start", name, ...extra }); }
  spanEnd(source, name, extra = {}) { return this.publish({ source, kind: "span.end", name, ...extra }); }
  event(source, name, extra = {}) { return this.publish({ source, kind: "event", name, ...extra }); }
  metric(source, name, value, extra = {}) { return this.publish({ ...extra, source, kind: "metric", name, attrs: { value, ...extra.attrs } }); }
  context(source, name, context, extra = {}) { return this.publish({ source, kind: "context", name, context, ...extra }); }
  log(source, name, message, extra = {}) { return this.publish({ source, kind: "log", name, output: message, ...extra }); }
}

export const bus = new Bus();

// --- sinks -----------------------------------------------------------------
const wsClients = new Set();
bus.subscribe((e) => {
  const frame = JSON.stringify(e);
  for (const ws of wsClients) { try { ws.write(encodeWs(frame)); } catch { /* dropped */ } }
});
let logReady = false;
bus.subscribe((e) => {
  if (!logReady) return;
  fsp.appendFile(LOG_FILE, JSON.stringify(e) + "\n").catch(() => {});
});

export async function initObserve() {
  await fsp.mkdir(DATA_DIR, { recursive: true });
  logReady = true;
}

/** Wrap a handler unit of work in a span (input/context in, output/status out). */
export async function span(source, name, opts, fn) {
  const spanId = id();
  const traceId = opts.traceId ?? `t${spanId}`;
  const start = Date.now();
  bus.spanStart(source, name, { traceId, spanId, input: opts.input, context: opts.context, attrs: opts.attrs });
  try {
    const result = await fn({ spanId, traceId });
    bus.spanEnd(source, name, { traceId, spanId, status: "ok", output: result?.output, attrs: { durationMs: Date.now() - start, ...result?.attrs } });
    return result?.value !== undefined ? result.value : result;
  } catch (err) {
    bus.spanEnd(source, name, { traceId, spanId, status: "error", error: err instanceof Error ? err.message : String(err), attrs: { durationMs: Date.now() - start } });
    throw err;
  }
}

// --- minimal RFC6455 server (text frames only; no external deps) -----------
const WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
import crypto from "node:crypto";

function encodeWs(str) {
  const payload = Buffer.from(str, "utf8");
  const len = payload.length;
  let header;
  if (len < 126) {
    header = Buffer.from([0x81, len]);
  } else if (len < 65536) {
    header = Buffer.alloc(4);
    header[0] = 0x81; header[1] = 126; header.writeUInt16BE(len, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = 0x81; header[1] = 127; header.writeBigUInt64BE(BigInt(len), 2);
  }
  return Buffer.concat([header, payload]);
}

const CLIENT_KINDS = new Set(["span.start", "span.end", "event", "metric", "log", "context"]);
const MAX_FIELD = 8192; // clamp input/output/context JSON

function clamp(v) {
  if (v === undefined) return undefined;
  try {
    const s = JSON.stringify(v);
    if (s.length <= MAX_FIELD) return v;
    return { truncated: true, bytes: s.length };
  } catch {
    return { unserializable: true };
  }
}

export async function handleObserve(req, res, url, send, readBody) {
  if (url.pathname === "/api/observe" && req.method === "GET") {
    return send(res, 200, { events: bus.snapshot().slice(-500) });
  }
  // Frontend-originated events (e.g. orchestration per-turn spans). The bus
  // applies secret scrubbing, the ring, NDJSON and the WS broadcast.
  if (url.pathname === "/api/observe" && req.method === "POST") {
    let body;
    try {
      body = JSON.parse((await readBody(req)).toString("utf8") || "{}");
    } catch {
      return send(res, 400, { error: "invalid JSON" });
    }
    const { source, kind, name } = body;
    if (typeof source !== "string" || !source.trim() || source.length > 120)
      return send(res, 400, { error: "need source (1..120 chars)" });
    if (typeof name !== "string" || !name.trim() || name.length > 120)
      return send(res, 400, { error: "need name (1..120 chars)" });
    if (!CLIENT_KINDS.has(kind))
      return send(res, 400, { error: `kind must be one of ${[...CLIENT_KINDS].join(", ")}` });
    const attrs = body.attrs && typeof body.attrs === "object" ? body.attrs : {};
    const event = bus.publish({
      source: source.trim(),
      kind,
      name: name.trim(),
      traceId: typeof body.traceId === "string" ? body.traceId.slice(0, 120) : undefined,
      spanId: typeof body.spanId === "string" ? body.spanId.slice(0, 120) : undefined,
      parentSpanId: typeof body.parentSpanId === "string" ? body.parentSpanId.slice(0, 120) : undefined,
      status: body.status === "ok" || body.status === "error" ? body.status : undefined,
      error: typeof body.error === "string" ? body.error.slice(0, 500) : undefined,
      attrs: { ...attrs, client: true },
      input: clamp(body.input),
      output: clamp(body.output),
      context: clamp(body.context),
    });
    return send(res, 200, { ok: true, id: event.id });
  }
  return send(res, 404, { error: "unknown observe endpoint" });
}

/** Call from proxy's server.on("upgrade"). Returns true if handled. */
export function handleObserveUpgrade(req, socket) {
  if (!req.url.startsWith("/api/observe/ws")) return false;
  const key = req.headers["sec-websocket-key"];
  if (!key) { socket.destroy(); return true; }
  const accept = crypto.createHash("sha1").update(key + WS_GUID).digest("base64");
  socket.write(
    "HTTP/1.1 101 Switching Protocols\r\n" +
    "Upgrade: websocket\r\nConnection: Upgrade\r\n" +
    `Sec-WebSocket-Accept: ${accept}\r\n\r\n`,
  );
  socket.setNoDelay(true);
  wsClients.add(socket);
  // Send the recent backlog so a freshly opened tower catches up.
  for (const e of bus.snapshot().slice(-200)) {
    try { socket.write(encodeWs(JSON.stringify(e))); } catch { /* ignore */ }
  }
  socket.on("close", () => wsClients.delete(socket));
  socket.on("error", () => wsClients.delete(socket));
  socket.on("data", () => { /* tower is receive-only; ignore client frames */ });
  return true;
}
