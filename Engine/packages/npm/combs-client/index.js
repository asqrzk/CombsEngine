/**
 * combs-client — official JavaScript client for Combs Engine
 * (`combs serve`, OpenAI-compatible HTTP/SSE API).
 *
 * Zero dependencies; runs in browsers, Node 18+, Deno and Bun.
 *
 * All network I/O goes through the injectable `fetchImpl`, so hosts can
 * layer auth headers, permission gates, caching or mocks without patching
 * globals. `onDownload` reports received bytes (per SSE chunk and per JSON
 * body) for bandwidth monitors.
 *
 * @example
 * import { CombsClient } from "combs-client";
 * const client = new CombsClient({ baseUrl: "http://localhost:8080" });
 * await client.streamChatCompletion(
 *   { messages: [{ role: "user", content: "Hello" }] },
 *   { onDelta: (t) => process.stdout.write(t) },
 * );
 */

export class CombsClient {
  /** @type {string} */ #baseUrl;
  /** @type {typeof fetch} */ #fetch;
  /** @type {((bytes: number) => void) | null} */ #onDownload;
  /** @type {string | null} */ #defaultModel;

  /**
   * @param {object} options
   * @param {string} options.baseUrl `combs serve` URL (required — no default,
   *   the host application always decides which server to talk to).
   * @param {typeof fetch} [options.fetchImpl] Custom fetch (auth, permission
   *   gates, retries). Defaults to global `fetch`.
   * @param {(bytes: number) => void} [options.onDownload] Received-byte hook.
   * @param {string} [options.model] Default model id for requests.
   */
  constructor(options = {}) {
    const { baseUrl, fetchImpl, onDownload, model } = options;
    if (!baseUrl) throw new Error("combs-client: options.baseUrl is required");
    this.#baseUrl = String(baseUrl).replace(/\/+$/, "");
    const f = fetchImpl ?? globalThis.fetch?.bind(globalThis);
    if (!f) throw new Error("combs-client: no fetch implementation available");
    this.#fetch = f;
    this.#onDownload = onDownload ?? null;
    this.#defaultModel = model ?? null;
  }

  get baseUrl() {
    return this.#baseUrl;
  }

  /** Server health check. @returns {Promise<boolean>} */
  async health() {
    try {
      const res = await this.#fetch(`${this.#baseUrl}/health`);
      return res.ok;
    } catch {
      return false;
    }
  }

  /** Lists model ids served by the engine. @returns {Promise<string[]>} */
  async listModels() {
    const res = await this.#fetch(`${this.#baseUrl}/v1/models`);
    if (!res.ok) throw new Error(`combs-client: HTTP ${res.status}`);
    const body = await res.json();
    this.#onDownload?.(JSON.stringify(body).length);
    return (body?.data ?? []).map((m) => m.id);
  }

  /**
   * Non-streaming chat completion.
   * @param {object} request
   * @param {{ role: string, content: string }[]} request.messages
   * @param {string} [request.model]
   * @param {number} [request.maxTokens=256]
   * @param {number} [request.temperature=0.7]
   * @param {AbortSignal} [request.signal]
   * @returns {Promise<string>} the assistant message content
   */
  async chatCompletion(request = {}) {
    const res = await this.#fetch(`${this.#baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(this.#payload(request, false)),
      signal: request.signal,
    });
    if (!res.ok) throw new Error(`combs-client: HTTP ${res.status}`);
    const body = await res.json();
    this.#onDownload?.(JSON.stringify(body).length);
    return body?.choices?.[0]?.message?.content ?? "";
  }

  /**
   * Streaming chat completion (SSE). Calls `onDelta` per token chunk and
   * exactly one of `onDone` / `onError` at the end.
   * @param {object} request Same shape as {@link chatCompletion}.
   * @param {object} callbacks
   * @param {(text: string) => void} [callbacks.onDelta]
   * @param {(finishReason: string) => void} [callbacks.onDone]
   * @param {(err: Error) => void} [callbacks.onError]
   */
  async streamChatCompletion(request = {}, callbacks = {}) {
    const { onDelta, onDone, onError } = callbacks;
    try {
      const res = await this.#fetch(`${this.#baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(this.#payload(request, true)),
        signal: request.signal,
      });
      if (!res.ok || !res.body) throw new Error(`combs-client: HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        this.#onDownload?.(value.byteLength);
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";
        for (const raw of events) {
          const line = raw.trim();
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") {
            onDone?.("stop");
            return;
          }
          const parsed = JSON.parse(data);
          const delta = parsed?.choices?.[0]?.delta?.content;
          const finish = parsed?.choices?.[0]?.finish_reason;
          if (delta) onDelta?.(delta);
          if (finish) {
            onDone?.(finish);
            return;
          }
        }
      }
      onDone?.("stop");
    } catch (err) {
      onError?.(err instanceof Error ? err : new Error(String(err)));
    }
  }

  #payload(request, stream) {
    return {
      messages: request.messages,
      model: request.model ?? this.#defaultModel ?? undefined,
      max_tokens: request.maxTokens ?? 256,
      temperature: request.temperature ?? 0.7,
      top_k: request.topK ?? undefined,
      top_p: request.topP ?? undefined,
      repetition_penalty: request.repetitionPenalty ?? undefined,
      frequency_penalty: request.frequencyPenalty ?? undefined,
      presence_penalty: request.presencePenalty ?? undefined,
      stream,
    };
  }
}

export default CombsClient;
