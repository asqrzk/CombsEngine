/**
 * RemoteEngine: EngineClient over the OpenAI-compatible HTTP/SSE API
 * (`combs serve`). Same contract as FfiEngine, so app code is
 * transport-agnostic.
 */
import type {
  ChatRequest,
  EngineClient,
  EngineMetadata,
  GenerationStats,
  StreamEvent,
} from "./types.ts";

export class RemoteEngine implements EngineClient {
  readonly kind = "remote" as const;

  constructor(readonly baseUrl: string) {}

  async metadata(): Promise<EngineMetadata> {
    const res = await fetch(`${this.baseUrl}/v1/models`);
    if (!res.ok) throw new Error(`GET /v1/models failed: HTTP ${res.status}`);
    const body = await res.json();
    const id = body?.data?.[0]?.id ?? "unknown";
    // The OpenAI surface does not expose full metadata; report what we know.
    return {
      architecture: "remote",
      vocab_size: 0,
      max_position_embeddings: 0,
      max_seq_len: 0,
      page_size: 0,
      eos_token_ids: [],
      im_end_id: null,
      ...({ model_id: id } as object),
    } as EngineMetadata;
  }

  async *stream(request: ChatRequest, _requestId?: string): AsyncIterable<StreamEvent> {
    const body = {
      model: "combs",
      messages: request.messages,
      max_tokens: request.max_tokens,
      temperature: request.temperature,
      top_p: request.top_p,
      frequency_penalty: request.frequency_penalty,
      presence_penalty: request.presence_penalty,
      seed: request.seed,
      stop: request.stop,
      stream: true,
    };
    const res = await fetch(`${this.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) {
      const text = await res.text();
      throw new Error(`chat completion failed: HTTP ${res.status}: ${text}`);
    }

    const decoder = new TextDecoder();
    let buffer = "";
    const started = performance.now();
    let generated = 0;
    for await (const chunk of res.body) {
      buffer += decoder.decode(chunk, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";
      for (const raw of events) {
        const line = raw.trim();
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") return;
        const parsed = JSON.parse(data);
        const delta = parsed?.choices?.[0]?.delta;
        const finish = parsed?.choices?.[0]?.finish_reason;
        if (delta?.content) {
          generated += 1;
          yield { type: "delta", text: delta.content, token_id: 0 };
        }
        if (finish) {
          const elapsed = (performance.now() - started) / 1000;
          const stats: GenerationStats = {
            prompt_tokens: 0,
            generated_tokens: generated,
            ttft_ms: 0,
            decode_tokens_per_second: generated / Math.max(elapsed, 1e-6),
            prefill_tokens_per_second: 0,
            cache_pages_used: 0,
          };
          yield { type: "done", finish_reason: finish, stats };
          return;
        }
      }
    }
  }

  async complete(
    request: ChatRequest,
  ): Promise<{ text: string; stats: GenerationStats; finishReason: string }> {
    let text = "";
    for await (const event of this.stream(request)) {
      if (event.type === "delta") text += event.text;
      if (event.type === "done") {
        return { text, stats: event.stats, finishReason: event.finish_reason };
      }
      if (event.type === "error") throw new Error(event.message);
    }
    throw new Error("stream ended without a done event");
  }

  cancel(_requestId: string): void {
    // HTTP cancellation is client-side (AbortController); nothing to send.
  }

  close(): void {}
}
