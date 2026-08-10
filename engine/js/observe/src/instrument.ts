/**
 * @combs/observe — instrument wrappers (the middleware).
 *
 * Wrap a component ONCE at instantiation and every operation emits spans with
 * input / output / context — no per-call-site logging. These are deliberately
 * generic: they depend only on structural types (an object with stream(), a
 * fetch-like function), never on a concrete engine or runtime.
 */

import type { EventBus } from "./bus.ts";
import type { ObsEvent } from "./types.ts";

/** Structural shape of a streaming engine (matches @combs/core EngineClient
 *  without importing it — keeps @combs/observe dependency-free). */
export interface StreamLike {
  delta?: string;
  type?: string;
}
export interface EngineLike {
  readonly kind?: string;
  stream(request: unknown, requestId?: string): AsyncIterable<unknown>;
  complete?(request: unknown): Promise<unknown>;
  metadata?(): Promise<unknown>;
  cancel?(requestId: string): void;
  close?(): void;
}

let spanCounter = 0;
function newSpanId(): string {
  spanCounter = (spanCounter + 1) & 0xffffff;
  return `s${Date.now().toString(36)}${spanCounter.toString(36)}`;
}

/**
 * Runs `fn` inside a span on the bus, emitting span.start (with input/context)
 * and span.end (with output/status/duration). Errors are captured onto the
 * span and rethrown.
 */
export async function span<T>(
  bus: EventBus,
  source: string,
  name: string,
  opts: { input?: unknown; context?: unknown; attrs?: ObsEvent["attrs"]; traceId?: string; parentSpanId?: string },
  fn: (s: { spanId: string; traceId: string; event: (n: string, a?: ObsEvent["attrs"]) => void }) => Promise<T> | T,
): Promise<T> {
  const spanId = newSpanId();
  const traceId = opts.traceId ?? `t${spanId}`;
  const start = Date.now();
  bus.spanStart(source, name, {
    traceId,
    spanId,
    parentSpanId: opts.parentSpanId,
    input: opts.input,
    context: opts.context,
    attrs: opts.attrs,
  });
  const api = {
    spanId,
    traceId,
    event: (n: string, a?: ObsEvent["attrs"]) =>
      bus.event(source, n, { traceId, spanId, attrs: a }),
  };
  try {
    const result = await fn(api);
    bus.spanEnd(source, name, {
      traceId,
      spanId,
      status: "ok",
      output: summarize(result),
      attrs: { durationMs: Date.now() - start },
    });
    return result;
  } catch (err) {
    bus.spanEnd(source, name, {
      traceId,
      spanId,
      status: "error",
      error: err instanceof Error ? err.message : String(err),
      attrs: { durationMs: Date.now() - start },
    });
    throw err;
  }
}

/** Keep span outputs small: truncate long strings/streams. */
function summarize(v: unknown): unknown {
  if (typeof v === "string") return v.length > 4000 ? v.slice(0, 4000) + "…" : v;
  return v;
}

/**
 * Wraps a streaming engine so each request emits:
 *   span.start (input=request, context=request.messages)
 *   metric engine.tokens (on done, with generation stats)
 *   span.end (output=accumulated text + stats)
 * Works with any EngineLike (FfiEngine / RemoteEngine / WorkerEngine).
 */
export function wrapEngine<T extends EngineLike>(engine: T, bus: EventBus, source: string): T {
  const wrapped = Object.create(Object.getPrototypeOf(engine));
  Object.assign(wrapped, engine);

  wrapped.stream = async function* (request: unknown, requestId?: string): AsyncIterable<unknown> {
    const spanId = newSpanId();
    const traceId = `t${spanId}`;
    const start = Date.now();
    const messages = (request as { messages?: unknown })?.messages;
    bus.spanStart(source, "engine.stream", {
      traceId,
      spanId,
      input: redactRequest(request),
      context: messages,
      attrs: { kind: engine.kind, requestId },
    });
    let text = "";
    let stats: unknown;
    let finishReason: string | undefined;
    try {
      for await (const ev of engine.stream(request, requestId)) {
        const e = ev as { type?: string; text?: string; stats?: unknown; finish_reason?: string; message?: string };
        if (e?.type === "delta" && typeof e.text === "string") text += e.text;
        if (e?.type === "done") {
          stats = e.stats;
          finishReason = e.finish_reason;
        }
        if (e?.type === "error") throw new Error(e.message ?? "engine error");
        yield ev;
      }
      if (stats && typeof stats === "object") {
        const s = stats as Record<string, number>;
        bus.metric(source, "engine.tokens", s.generated_tokens ?? 0, {
          traceId,
          spanId,
          attrs: {
            promptTokens: s.prompt_tokens,
            decodeTps: s.decode_tokens_per_second,
            prefillTps: s.prefill_tokens_per_second,
            ttftMs: s.ttft_ms,
            kvPages: s.cache_pages_used,
          },
        });
        // KV / cache state is part of the observable context.
        bus.context(source, "engine.kv", { cache_pages_used: s.cache_pages_used }, { traceId, spanId });
      }
      bus.spanEnd(source, "engine.stream", {
        traceId,
        spanId,
        status: "ok",
        output: { text: summarize(text), finishReason, stats },
        attrs: { durationMs: Date.now() - start, finishReason },
      });
    } catch (err) {
      bus.spanEnd(source, "engine.stream", {
        traceId,
        spanId,
        status: "error",
        error: err instanceof Error ? err.message : String(err),
        attrs: { durationMs: Date.now() - start },
      });
      throw err;
    }
  };

  if (typeof engine.complete === "function") {
    const origComplete = engine.complete.bind(engine);
    wrapped.complete = (request: unknown) =>
      span(bus, source, "engine.complete", { input: redactRequest(request), context: (request as { messages?: unknown })?.messages }, () =>
        origComplete(request),
      );
  }
  return wrapped as T;
}

/** Drop nothing structural but keep the request small for the event. */
function redactRequest(request: unknown): unknown {
  if (!request || typeof request !== "object") return request;
  const r = { ...(request as Record<string, unknown>) };
  // messages are carried separately in `context`; avoid duplicating large prompt strings
  if (typeof r.prompt === "string" && r.prompt.length > 2000) r.prompt = r.prompt.slice(0, 2000) + "…";
  return r;
}

/** A fetch-like function (browser fetch, Node fetch, undici, ...). */
export type FetchLike = (url: unknown, init?: unknown) => Promise<unknown>;

/**
 * Wraps a fetch implementation so every call emits an http.fetch span with
 * method/url/status/duration. Response is passed through untouched (streaming
 * bodies are NOT consumed — we only record status + timing).
 */
export function instrumentFetch(bus: EventBus, source: string, fetchImpl: FetchLike): FetchLike {
  return async (url: unknown, init?: unknown) => {
    const method = (init as { method?: string })?.method ?? "GET";
    const target = String((url as { url?: string })?.url ?? url);
    const spanId = newSpanId();
    const traceId = `t${spanId}`;
    const start = Date.now();
    bus.spanStart(source, "http.fetch", { traceId, spanId, attrs: { method, url: target } });
    try {
      const res = await fetchImpl(url, init);
      const status = (res as { status?: number })?.status;
      bus.spanEnd(source, "http.fetch", {
        traceId,
        spanId,
        status: status && status >= 400 ? "error" : "ok",
        attrs: { method, url: target, status, durationMs: Date.now() - start },
      });
      return res;
    } catch (err) {
      bus.spanEnd(source, "http.fetch", {
        traceId,
        spanId,
        status: "error",
        error: err instanceof Error ? err.message : String(err),
        attrs: { method, url: target, durationMs: Date.now() - start },
      });
      throw err;
    }
  };
}
