/**
 * Tests for @combs/observe: bus fan-out + ring, redaction, instrument wrappers.
 */
import { assert, assertEquals } from "jsr:@std/assert";
import {
  defaultRedactor,
  EventBus,
  instrumentFetch,
  MemorySink,
  NdjsonSink,
  span,
  wrapEngine,
} from "../mod.ts";
import type { EngineLike } from "../mod.ts";

Deno.test("bus: publish fills id/ts and fans out to subscribers + sinks", () => {
  const sink = new MemorySink();
  const bus = new EventBus({ sinks: [sink] });
  const seen: string[] = [];
  bus.subscribe((e) => seen.push(e.name));
  const ev = bus.event("proxy", "proxy.relay", { attrs: { status: 200 } });
  assert(ev.id.length > 0);
  assert(typeof ev.ts === "number");
  assertEquals(seen, ["proxy.relay"]);
  assertEquals(sink.events.length, 1);
  assertEquals(sink.events[0].attrs?.status, 200);
});

Deno.test("bus: ring buffer evicts oldest beyond capacity", () => {
  const bus = new EventBus({ capacity: 5 });
  for (let i = 0; i < 10; i++) bus.event("s", `e${i}`);
  const names = bus.snapshot().map((e) => e.name);
  assertEquals(names, ["e5", "e6", "e7", "e8", "e9"]);
});

Deno.test("bus: a throwing subscriber/sink does not break publishing", () => {
  const bus = new EventBus({ sinks: [{ write: () => { throw new Error("boom"); } }] });
  bus.subscribe(() => { throw new Error("boom"); });
  const ev = bus.event("s", "ok");
  assertEquals(ev.name, "ok");
});

Deno.test("redactor: masks secret-looking keys and strings", () => {
  const ev = defaultRedactor({
    id: "x",
    ts: 0,
    source: "s",
    kind: "event",
    name: "n",
    input: { authorization: "Bearer abcdefgh", note: "sk-1234567890abcdef", safe: 1 },
  });
  const input = ev.input as Record<string, unknown>;
  assertEquals(input.authorization, "***");
  assertEquals(typeof input.note, "string");
  assert(!(input.note as string).includes("sk-12345678"));
  assertEquals(input.safe, 1);
});

Deno.test("span: emits start + ok end with duration and output", async () => {
  const sink = new MemorySink();
  const bus = new EventBus({ sinks: [sink] });
  const out = await span(bus, "agent:coder", "agent.delegate", { input: { task: "x" } }, async (s) => {
    s.event("agent.progress", { pct: 50 });
    return { result: "done" };
  });
  assertEquals(out, { result: "done" });
  const kinds = sink.events.map((e) => e.kind);
  assertEquals(kinds[0], "span.start");
  assert(kinds.includes("event"));
  assertEquals(kinds[kinds.length - 1], "span.end");
  const end = sink.events[sink.events.length - 1];
  assertEquals(end.status, "ok");
  assert(typeof end.attrs?.durationMs === "number");
  // traceId correlates all events in the run
  const traces = new Set(sink.events.map((e) => e.traceId));
  assertEquals(traces.size, 1);
});

Deno.test("span: error is captured on the span and rethrown", async () => {
  const sink = new MemorySink();
  const bus = new EventBus({ sinks: [sink] });
  let threw = false;
  try {
    await span(bus, "s", "op", {}, () => {
      throw new Error("kaboom");
    });
  } catch {
    threw = true;
  }
  assert(threw);
  const end = sink.events[sink.events.length - 1];
  assertEquals(end.status, "error");
  assertEquals(end.error, "kaboom");
});

/** Minimal fake streaming engine for wrapEngine tests. */
function fakeEngine(chunks: string[]): EngineLike {
  return {
    kind: "ffi",
    async *stream() {
      for (const c of chunks) yield { type: "delta", text: c, token_id: 1 };
      yield {
        type: "done",
        finish_reason: "stop",
        stats: {
          prompt_tokens: 10,
          generated_tokens: chunks.length,
          ttft_ms: 5,
          decode_tokens_per_second: 30,
          prefill_tokens_per_second: 600,
          cache_pages_used: 3,
        },
      };
    },
  };
}

Deno.test("wrapEngine: emits stream span, token metric, and KV context", async () => {
  const sink = new MemorySink();
  const bus = new EventBus({ sinks: [sink] });
  const engine = wrapEngine(fakeEngine(["hello", " ", "world"]), bus, "engine:A");
  let text = "";
  for await (const ev of engine.stream({ messages: [{ role: "user", content: "hi" }] })) {
    const e = ev as { type?: string; text?: string };
    if (e.type === "delta") text += e.text;
  }
  assertEquals(text, "hello world");
  const names = sink.events.map((e) => e.name);
  assert(names.includes("engine.stream"));
  assert(names.includes("engine.tokens"));
  assert(names.includes("engine.kv"));
  const metric = sink.events.find((e) => e.name === "engine.tokens");
  assertEquals(metric?.attrs?.value, 3);
  assertEquals(metric?.attrs?.kvPages, 3);
  const ctx = sink.events.find((e) => e.name === "engine.kv");
  assertEquals((ctx?.context as Record<string, number>).cache_pages_used, 3);
});

Deno.test("instrumentFetch: records method/url/status/duration", async () => {
  const sink = new MemorySink();
  const bus = new EventBus({ sinks: [sink] });
  const fetchImpl = instrumentFetch(bus, "proxy", () =>
    Promise.resolve({ status: 200, json: () => Promise.resolve({}) }),
  );
  await fetchImpl("http://localhost:8080/v1/models", { method: "GET" });
  const end = sink.events.find((e) => e.kind === "span.end");
  assertEquals(end?.name, "http.fetch");
  assertEquals(end?.attrs?.status, 200);
  assertEquals(end?.status, "ok");
});

Deno.test("ndjson sink: serializes one line per event via injected append", () => {
  const lines: string[] = [];
  const sink = NdjsonSink.toArray(lines);
  const bus = new EventBus({ sinks: [sink] });
  bus.event("s", "a");
  bus.event("s", "b");
  assertEquals(lines.length, 2);
  const parsed = JSON.parse(lines[0]);
  assertEquals(parsed.name, "a");
});
