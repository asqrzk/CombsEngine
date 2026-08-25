/**
 * The worker transport, tested against a scripted worker.
 *
 * What is being checked is the protocol, not the engine: that a stream of
 * `event` replies becomes an async iterable of `StreamEvent`, that the
 * terminal event ends the iteration, that `complete()` concatenates deltas,
 * and that a cancel mid-stream produces a normal completion rather than an
 * error. All of that is transport logic, and none of it needs a GPU.
 */

import { assertEquals } from "jsr:@std/assert@1";
import { WorkerEngine } from "../src/worker.ts";

const WORKER = new URL("./fake.worker.js", import.meta.url);

Deno.test("worker: load resolves with the engine's metadata", async () => {
  const engine = await WorkerEngine.load(WORKER, {
    max_seq_len: 4096,
    modelUrl: "https://example.invalid/model.gguf",
  });
  const md = await engine.metadata();
  assertEquals(md.architecture, "llama");
  assertEquals(md.max_seq_len, 4096);
  engine.close();
});

Deno.test("worker: model bytes are accepted in place of a URL", async () => {
  // The path a page takes when it already holds the weights: hand over the
  // buffer rather than a URL the worker would have to resolve itself. (The
  // platform's own backend posts this directly so the buffer can be
  // transferred; through `WorkerEngine.load` it is structure-cloned, which
  // costs a copy but is the same contract.)
  const bytes = new Uint8Array([0x47, 0x47, 0x55, 0x46]).buffer;
  const engine = await WorkerEngine.load(WORKER, {
    max_seq_len: 2048,
    modelBytes: bytes,
  });
  assertEquals((await engine.metadata()).max_seq_len, 2048);
  engine.close();
});

Deno.test("worker: a load with no model at all is refused", async () => {
  let failed = false;
  try {
    await WorkerEngine.load(WORKER, { max_seq_len: 2048 });
  } catch (e) {
    failed = true;
    assertEquals(String(e).includes("modelBytes"), true);
  }
  assertEquals(failed, true, "a load without a model must not silently succeed");
});

Deno.test("worker: stream yields every delta then the terminal event", async () => {
  const engine = await WorkerEngine.load(WORKER, { modelUrl: "https://example.invalid/m.gguf" });
  const kinds: string[] = [];
  let text = "";
  for await (const event of engine.stream({ prompt: "The capital of France is" })) {
    kinds.push(event.type);
    if (event.type === "delta") text += event.text;
  }
  assertEquals(text, "Paris is the capital");
  assertEquals(kinds.at(-1), "done");
  assertEquals(kinds.filter((k) => k === "delta").length, 5);
  engine.close();
});

Deno.test("worker: complete concatenates the stream", async () => {
  const engine = await WorkerEngine.load(WORKER, { modelUrl: "https://example.invalid/m.gguf" });
  const { text, finishReason, stats } = await engine.complete({ prompt: "x" });
  assertEquals(text, "Paris is the capital");
  assertEquals(finishReason, "stop");
  assertEquals(stats.generated_tokens, 5);
  engine.close();
});

Deno.test("worker: a cancel mid-stream keeps what arrived", async () => {
  const engine = await WorkerEngine.load(WORKER, { modelUrl: "https://example.invalid/m.gguf" });
  const id = crypto.randomUUID();
  let text = "";
  let reason = "";
  for await (const event of engine.stream({ prompt: "x" }, id)) {
    if (event.type === "delta") {
      text += event.text;
      // Stop after the first token, the way a Stop button does.
      if (text.length > 0) engine.cancel(id);
    }
    if (event.type === "done") reason = event.finish_reason;
  }
  assertEquals(reason, "cancelled", "a stopped turn completes, it does not error");
  assertEquals(text.length > 0, true, "the text already streamed is kept");
  assertEquals(text.length < "Paris is the capital".length, true, "and it is partial");
  engine.close();
});

Deno.test("worker: the engine's KV state is reachable", async () => {
  // A browser engine has no port to poll, so its cache is only visible if
  // the host can ask for it. Nothing else in the app can see it.
  const engine = await WorkerEngine.load(WORKER, {
    modelUrl: "https://example.invalid/m.gguf",
  });
  const stats = await (engine as unknown as {
    rpc: (r: { kind: string; id: string }) => Promise<unknown>;
  }).rpc({ kind: "stats", id: crypto.randomUUID() }) as {
    kind: string;
    sessions: { history_tokens: number }[];
  };
  assertEquals(stats.kind, "paged");
  assertEquals(stats.sessions[0].history_tokens, 42);
  engine.close();
});

Deno.test("worker: loading twice frees the engine it replaces", async () => {
  // A worker hosts one engine. Loading a second model into the same worker
  // must free the first, or hundreds of megabytes of weights stay
  // reachable only through an id nobody holds.
  const engine = await WorkerEngine.load(WORKER, {
    modelUrl: "https://example.invalid/first.gguf",
  });
  const rpc = (engine as unknown as {
    rpc: (r: { kind: string; id: string; payload?: unknown }) => Promise<unknown>;
  }).rpc.bind(engine);
  await rpc({
    kind: "load",
    id: crypto.randomUUID(),
    payload: { modelUrl: "https://example.invalid/second.gguf" },
  });
  const { live } = await rpc({ kind: "live", id: crypto.randomUUID() }) as { live: number };
  assertEquals(live, 1, "a second load left the first engine alive");
  engine.close();
});

Deno.test("worker: a refused load rejects rather than hanging", async () => {
  let failed = false;
  try {
    await WorkerEngine.load(WORKER, { fail: true });
  } catch (e) {
    failed = true;
    assertEquals(String(e).includes("load refused"), true);
  }
  assertEquals(failed, true);
});
