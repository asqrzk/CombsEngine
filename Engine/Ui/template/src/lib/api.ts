/**
 * API client for `combs serve` (OpenAI-compatible), with:
 * - permission gates checked before ANY network activity
 * - byte counting feeding the realtime network monitor
 */

import { monitor } from "./monitor";
import { permissions } from "./permissions";

export interface ChatMessage {
  role: string;
  content: string;
}

export interface StreamCallbacks {
  onDelta: (text: string) => void;
  onDone: (finishReason: string) => void;
  onError: (err: Error) => void;
}

/** Permission-checked fetch wrapper with byte counting. */
export async function guardedFetch(
  url: string,
  init: RequestInit,
  scope: "network:download" | "network:inference",
  detail: string,
): Promise<Response> {
  const allowed = await permissions.require(scope, detail);
  if (!allowed) throw new Error(`permission denied: ${detail}`);
  monitor.netUp(init.body ? String(init.body).length : 0);
  const res = await fetch(url, init);
  return res;
}

/** Lists models on the server. */
export async function listModels(server: string): Promise<string[]> {
  const res = await guardedFetch(
    `${server}/v1/models`,
    {},
    "network:inference",
    `list models on ${server}`,
  );
  const body = await res.json();
  monitor.netDown(JSON.stringify(body).length);
  return (body?.data ?? []).map((m: { id: string }) => m.id);
}

/** Streams a chat completion (SSE), calling back per delta. */
export async function streamChat(
  server: string,
  messages: ChatMessage[],
  opts: { model?: string; maxTokens?: number; temperature?: number },
  cb: StreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  try {
    const res = await guardedFetch(
      `${server}/v1/chat/completions`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          messages,
          model: opts.model,
          max_tokens: opts.maxTokens ?? 256,
          temperature: opts.temperature ?? 0.7,
          stream: true,
        }),
        signal,
      },
      "network:inference",
      `chat completion via ${server}`,
    );
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      monitor.netDown(value.byteLength);
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
        const delta = parsed?.choices?.[0]?.delta?.content;
        const finish = parsed?.choices?.[0]?.finish_reason;
        if (delta) cb.onDelta(delta);
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
