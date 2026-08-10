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
export async function spawnEngine(model: string, tag?: string): Promise<{ port: number; url: string }> {
  // The proxy health-waits up to ~120s for the engine; bound the client wait
  // just past that so a dead engine surfaces as an error instead of a hang.
  const ctrl = new AbortController();
  const timeout = setTimeout(() => ctrl.abort(), 130_000);
  try {
    const res = await withPermission(
      "system:subprocess",
      `start ${tag ? `engine '${tag}'` : "a second engine"} for ${model}`,
      () =>
        fetch("/api/engine/spawn", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ model, tag }),
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
// Observe bus — publish client-side events (orchestration spans) so they
// show up in the Control Tower next to the proxy's own spans.
// ---------------------------------------------------------------------------

export interface ObserveEvent {
  source: string;
  kind: "span.start" | "span.end" | "event" | "metric" | "log" | "context";
  name: string;
  traceId?: string;
  spanId?: string;
  parentSpanId?: string;
  status?: "ok" | "error";
  error?: string;
  attrs?: Record<string, unknown>;
  input?: unknown;
  output?: unknown;
  context?: unknown;
}

/** Fire-and-forget publish to the proxy's observe bus (never throws). */
export async function postObserve(event: ObserveEvent): Promise<void> {
  try {
    await fetch("/api/observe", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(event),
    });
  } catch {
    // observability must never break the app
  }
}

// ---------------------------------------------------------------------------
// Streaming with usage — own SSE parser over the relay (the combs-client
// package drops the final `usage`; kv-cache-ui needs `cached_tokens`).
// ---------------------------------------------------------------------------

export interface CompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: { cached_tokens?: number };
}

export interface UsageStreamCallbacks {
  onDelta: (text: string) => void;
  onUsage?: (usage: CompletionUsage) => void;
  onDone: (finishReason: string) => void;
  onError: (err: Error) => void;
}

/** Like streamChat, but surfaces the final chunk's `usage` (KV cache hits). */
export async function streamChatWithUsage(
  server: string,
  messages: ChatMessage[],
  opts: {
    model?: string;
    maxTokens?: number;
    temperature?: number;
    frequencyPenalty?: number;
    presencePenalty?: number;
    sessionId?: string;
  },
  cb: UsageStreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  try {
    const url = `${server}/v1/chat/completions`;
    const res = await withPermission("network:inference", `POST ${url}`, () =>
      fetch("/api/relay", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          url,
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            messages,
            model: opts.model,
            max_tokens: opts.maxTokens ?? 256,
            temperature: opts.temperature ?? 0.7,
            frequency_penalty: opts.frequencyPenalty,
            presence_penalty: opts.presencePenalty,
            stream: true,
            session_id: opts.sessionId,
          }),
          scope: "network:inference",
        }),
        signal,
      }),
    );
    if (res.status >= 500) {
      const body = await res.json().catch(() => null);
      throw new Error(body?.error ?? `proxy error: HTTP ${res.status}`);
    }
    if (!res.ok || !res.body) throw new Error(`stream failed: HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";
      for (const raw of events) {
        const line = raw.trim();
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") {
          cb.onDone("stop");
          return;
        }
        const parsed = JSON.parse(data);
        if (parsed?.error) throw new Error(parsed.error.message ?? "engine error");
        const delta = parsed?.choices?.[0]?.delta?.content;
        const finish = parsed?.choices?.[0]?.finish_reason;
        if (delta) cb.onDelta(delta);
        if (parsed?.usage) cb.onUsage?.(parsed.usage as CompletionUsage);
        if (finish) {
          cb.onDone(finish);
          return;
        }
      }
    }
    cb.onDone("stop");
  } catch (err) {
    cb.onError(err instanceof Error ? err : new Error(String(err)));
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
