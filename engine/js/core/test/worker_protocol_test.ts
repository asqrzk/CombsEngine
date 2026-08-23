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
  const engine = await WorkerEngine.load(WORKER, { max_seq_len: 4096 });
  const md = await engine.metadata();
  assertEquals(md.architecture, "llama");
  assertEquals(md.max_seq_len, 4096);
  engine.close();
});

Deno.test("worker: stream yields every delta then the terminal event", async () => {
  const engine = await WorkerEngine.load(WORKER, {});
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
  const engine = await WorkerEngine.load(WORKER, {});
  const { text, finishReason, stats } = await engine.complete({ prompt: "x" });
  assertEquals(text, "Paris is the capital");
  assertEquals(finishReason, "stop");
  assertEquals(stats.generated_tokens, 5);
  engine.close();
});

Deno.test("worker: a cancel mid-stream keeps what arrived", async () => {
  const engine = await WorkerEngine.load(WORKER, {});
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
