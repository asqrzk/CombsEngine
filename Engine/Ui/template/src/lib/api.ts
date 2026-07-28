/**
 * Engine API access — thin adapter over the official `combs-client` npm
 * package. Everything goes through the backend permission proxy:
 *
 *   browser ──/api/relay──▶ proxy (permission check) ──▶ combs serve
 *   browser ──/api/files───▶ proxy (permission check) ──▶ disk
 *
 * The proxy owns and enforces all grants; this layer only asks the user
 * (via the permission dialog) when the proxy answers 428 and then retries.
 */

import { CombsClient } from "@combs-edge/combs-client";
import { permissions, type PermissionScope } from "./permissions.svelte";

export interface ChatMessage {
  role: string;
  content: string;
}

export interface StreamCallbacks {
  onDelta: (text: string) => void;
  onDone: (finishReason: string) => void;
  onError: (err: Error) => void;
}

/** Runs a request; on 428 from the proxy, asks the user and retries. */
async function withPermission(
  scope: PermissionScope,
  detail: string,
  fn: () => Promise<Response>,
): Promise<Response> {
  for (;;) {
    const res = await fn();
    if (res.status !== 428) return res;
    const ok = await permissions.ask(scope, detail);
    if (!ok) throw new Error(`permission denied: ${detail}`);
  }
}

/** fetchImpl handed to CombsClient: relays every request via the proxy. */
function relayFetch(scope: PermissionScope): typeof fetch {
  return (async (url: string | URL | Request, init?: RequestInit) => {
    const target = String(url);
    const method = init?.method ?? "GET";
    const res = await withPermission(scope, `${method} ${target}`, () =>
      fetch("/api/relay", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          url: target,
          method,
          headers: init?.headers ?? {},
          body: init?.body ?? null,
          scope,
        }),
        signal: init?.signal,
      }),
    );
    // Surface the proxy's upstream error (e.g. "combs serve not running")
    // instead of a bare status code.
    if (res.status >= 500) {
      const body = await res.json().catch(() => null);
      throw new Error(body?.error ?? `proxy error: HTTP ${res.status}`);
    }
    return res;
  }) as typeof fetch;
}

/** One CombsClient per server URL; all traffic relayed + gated. */
const clients = new Map<string, CombsClient>();

function clientFor(server: string): CombsClient {
  let client = clients.get(server);
  if (!client) {
    client = new CombsClient({
      baseUrl: server,
      fetchImpl: relayFetch("network:inference"),
    });
    clients.set(server, client);
  }
  return client;
}

/** Lists models on the server. */
export function listModels(server: string): Promise<string[]> {
  return clientFor(server).listModels();
}

/** Streams a chat completion (SSE), calling back per delta. */
export function streamChat(
  server: string,
  messages: ChatMessage[],
  opts: {
    model?: string;
    maxTokens?: number;
    temperature?: number;
    topK?: number;
    topP?: number;
    repetitionPenalty?: number;
    frequencyPenalty?: number;
    presencePenalty?: number;
  },
  cb: StreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  return clientFor(server).streamChatCompletion({ ...opts, messages, signal }, cb);
}

// ---------------------------------------------------------------------------
// Engine spawning (roleplay's second engine; agent engines) — via proxy.
// ---------------------------------------------------------------------------

/** Asks the proxy to start another `combs serve` on a free port. */
export async function spawnEngine(model: string): Promise<{ port: number; url: string }> {
  // The proxy health-waits up to ~120s for the engine; bound the client wait
  // just past that so a dead engine surfaces as an error instead of a hang.
  const ctrl = new AbortController();
  const timeout = setTimeout(() => ctrl.abort(), 130_000);
  try {
    const res = await withPermission(
      "system:subprocess",
      `start a second engine for ${model}`,
      () =>
        fetch("/api/engine/spawn", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ model }),
          signal: ctrl.signal,
        }),
    );
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      throw new Error(body?.error ?? `engine spawn failed: HTTP ${res.status}`);
    }
    return res.json();
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new Error("engine spawn timed out — the model may still be loading; try again");
    }
    throw e;
  } finally {
    clearTimeout(timeout);
  }
}

/** Stops a spawned engine (best effort). */
export async function stopEngine(port: number): Promise<void> {
  try {
    await fetch("/api/engine/stop", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ port }),
    });
  } catch {
    // proxy unreachable
  }
}

// ---------------------------------------------------------------------------
// File persistence — via the proxy too (gated by storage scopes).
// ---------------------------------------------------------------------------

export async function readFile(name: string): Promise<string | null> {
  try {
    const res = await fetch(`/api/files/${encodeURIComponent(name)}`);
    return res.ok ? await res.text() : null;
  } catch {
    return null;
  }
}

export async function writeFile(
  name: string,
  contents: string,
  scope: PermissionScope,
): Promise<boolean> {
  try {
    const res = await withPermission(scope, `save ${name} on this device`, () =>
      fetch(`/api/files/${encodeURIComponent(name)}?scope=${scope}`, {
        method: "POST",
        body: contents,
      }),
    );
    return res.ok;
  } catch {
    return false;
  }
}

export async function deleteFile(name: string, scope: PermissionScope): Promise<void> {
  await withPermission(scope, `delete ${name} from this device`, () =>
    fetch(`/api/files/${encodeURIComponent(name)}?scope=${scope}`, { method: "DELETE" }),
  );
}
