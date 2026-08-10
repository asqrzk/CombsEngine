/**
 * Sandboxed internet proxy — the token1/token2 zero-trust flow for agents
 * (MCP / tools) that need web access.
 *
 * Flow:
 *  1. Agent asks the main tab for internet access.
 *  2. Main tab calls POST /api/sandbox/request {agentId, allowlist}.
 *     The `network:internet` scope is permission-checked (passkey dialog).
 *  3. On approval the main proxy mints token1 (agent→sandbox key) and
 *     token2 (sandbox→main key), spawns a dedicated sandbox proxy, and
 *     answers {port, token1, token2}. The main tab SAVES token2 and hands
 *     only {port, token1} to the agent — the agent adopts token1 as its
 *     credential for that channel (replacing its public key).
 *  4. Agent → sandbox: POST /fetch {agentId, sealed} where sealed =
 *     token1-encrypted {url, method?, headers?, body?}. The sandbox
 *     decrypts, guardrail-checks (allowlist, size caps, secret denylist),
 *     fetches upstream, guardrail-checks the response, then re-encrypts
 *     with token2 → the main tab opens it with the token2 it saved.
 *
 * Tampered/forged messages fail AES-GCM and are dropped. Tokens expire
 * (default 15 min). Every request/response is appended to an encrypted
 * audit log via the zero-trust storage middleware.
 */

import http from "node:http";
import fsp from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  mintToken,
  tokenDecrypt,
  tokenEncrypt,
  tokenValid,
} from "@combs-edge/combs-zerotrust";
import { openFromStorage, sealForStorage } from "./zerotrust.mjs";

const te = new TextEncoder();
const td = new TextDecoder();
const HERE = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.join(HERE, "data");

/** Guardrail limits. */
const MAX_REQ_BODY = 1 * 1024 * 1024; // 1 MB out
const MAX_RES_BODY = 10 * 1024 * 1024; // 10 MB in
const ALLOWED_METHODS = new Set(["GET", "POST"]);
/** Things that must never leave the machine. */
const SECRET_PATTERNS = [
  /sk-[A-Za-z0-9_-]{20,}/, // OpenAI-style keys
  /-----BEGIN [A-Z ]*PRIVATE KEY-----/,
  /api[_-]?key["'\s:=]{1,10}[A-Za-z0-9_-]{16,}/i,
];

/** agentId -> {token1, token2, allowlist, server, port} */
const sandboxes = new Map();

async function audit(agentId, line) {
  const name = `sandbox-audit-${agentId.replace(/[^a-z0-9_-]/gi, "_")}.log`;
  const file = path.join(DATA_DIR, name);
  let prior = "";
  try {
    prior = td.decode(await openFromStorage(name, await fsp.readFile(file, "utf8")));
  } catch { /* first entry */ }
  const { blob } = await sealForStorage(name, te.encode(prior + line + "\n"));
  await fsp.mkdir(DATA_DIR, { recursive: true });
  await fsp.writeFile(file, blob, "utf8");
}

function guardrailRequest(agentId, payload) {
  let url;
  try {
    url = new URL(payload.url);
  } catch {
    return `bad url`;
  }
  if (!["http:", "https:"].includes(url.protocol)) return "protocol not allowed";
  const sb = sandboxes.get(agentId);
  if (!sb.allowlist.includes(url.host)) return `host ${url.host} not in allowlist`;
  const method = (payload.method ?? "GET").toUpperCase();
  if (!ALLOWED_METHODS.has(method)) return `method ${method} not allowed`;
  const body = payload.body ?? "";
  if (body.length > MAX_REQ_BODY) return "request body too large";
  for (const re of SECRET_PATTERNS) {
    if (re.test(body)) return "request body matches secret pattern — blocked";
  }
  return null;
}

/** Starts the sandbox HTTP server for one agent. */
async function startSandbox(agentId, allowlist, token1, token2) {
  const server = http.createServer(async (req, res) => {
    const send = (status, obj) => {
      res.writeHead(status, { "content-type": "application/json" });
      res.end(JSON.stringify(obj));
    };
    if (req.method !== "POST" || req.url !== "/fetch") return send(404, { error: "not found" });

    const chunks = [];
    for await (const c of req) chunks.push(c);
    let msg;
    try {
      msg = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    } catch {
      return send(400, { error: "bad json" });
    }
    if (msg.agentId !== agentId) return send(403, { error: "agent mismatch" });
    if (!tokenValid(token1, agentId)) return send(403, { error: "token1 expired" });

    // token1 = the agent's credential for this channel: decrypt the request
    let payload;
    try {
      payload = JSON.parse(td.decode(await tokenDecrypt(token1.token, msg.sealed)));
    } catch {
      return send(403, { error: "cannot decrypt request (wrong token or tampered)" });
    }

    const blocked = guardrailRequest(agentId, payload);
    if (blocked) {
      await audit(agentId, `${new Date().toISOString()} BLOCKED ${payload.url} — ${blocked}`);
      return send(403, { error: `guardrail: ${blocked}` });
    }

    let upstream;
    try {
      upstream = await fetch(payload.url, {
        method: (payload.method ?? "GET").toUpperCase(),
        headers: payload.headers ?? {},
        body: payload.body ?? undefined,
      });
    } catch (e) {
      return send(502, { error: `upstream unreachable: ${e.message}` });
    }
    const buf = Buffer.from(await upstream.arrayBuffer());
    if (buf.length > MAX_RES_BODY) {
      await audit(agentId, `${new Date().toISOString()} BLOCKED-RESPONSE ${payload.url} — too large`);
      return send(502, { error: "guardrail: response too large" });
    }

    const result = JSON.stringify({
      status: upstream.status,
      headers: Object.fromEntries(
        ["content-type"].map((h) => [h, upstream.headers.get(h) ?? ""]),
      ),
      body: buf.toString("base64"),
    });
    await audit(agentId, `${new Date().toISOString()} OK ${payload.url} -> ${upstream.status} (${buf.length}B)`);

    // token2 = response encryption for the main tab (it saved token2)
    const sealed = await tokenEncrypt(token2.token, te.encode(result));
    send(200, { sealed });
  });

  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  return { server, port };
}

/**
 * Handles POST /api/sandbox/request {agentId, allowlist} and
 * POST /api/sandbox/close {agentId}. `requirePermission` gates the request.
 */
export async function handleSandbox(req, res, url, readBody, send, requirePermission) {
  const action = url.pathname.slice("/api/sandbox/".length);
  const body = JSON.parse((await readBody(req)).toString("utf8") || "{}");
  const { agentId } = body;
  if (!agentId || typeof agentId !== "string") return send(res, 400, { error: "need {agentId}" });

  if (action === "close") {
    const sb = sandboxes.get(agentId);
    if (sb) {
      sb.server.close();
      sandboxes.delete(agentId);
    }
    return send(res, 200, { ok: true });
  }

  if (action !== "request") return send(res, 404, { error: "unknown sandbox endpoint" });

  const allowlist = Array.isArray(body.allowlist) && body.allowlist.length > 0
    ? body.allowlist.map(String)
    : null;
  if (!allowlist) return send(res, 400, { error: "need non-empty {allowlist: [hosts]}" });

  // Permission (passkey dialog on the main tab) — 428 when ungranted.
  if (!(await requirePermission(res, "network:internet", `agent "${agentId}" wants internet access (sandboxed to: ${allowlist.join(", ")})`))) {
    return;
  }

  // Reuse a live sandbox for the same agent; otherwise mint + spawn.
  let sb = sandboxes.get(agentId);
  if (!sb || !tokenValid(sb.token1, agentId)) {
    if (sb) sb.server.close();
    const token1 = mintToken(agentId);
    const token2 = mintToken(agentId);
    const { server, port } = await startSandbox(agentId, allowlist, token1, token2);
    sb = { token1, token2, allowlist, server, port };
    sandboxes.set(agentId, sb);
  }
  // The main tab saves token2 and forwards only {port, token1} to the agent.
  send(res, 200, {
    port: sb.port,
    token1: sb.token1.token,
    token2: sb.token2.token,
    expires: sb.token1.expires,
    allowlist: sb.allowlist,
  });
}
