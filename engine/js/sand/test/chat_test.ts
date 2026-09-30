/**
 * The chat wire, pinned: SSE frames split across chunk boundaries on purpose,
 * keepalive comments, a tool call whole in one chunk and the same call in
 * fragments, usage from whichever frame carries it, `[DONE]`, an error frame,
 * an abort that cancels the reader, the assistant's tool loop and its turn
 * bound, and a session that round-trips. No network: every wire is a fake
 * fetch, and every stream is a ReadableStream this file builds.
 */
import { assert, assertEquals, assertRejects } from "jsr:@std/assert";
import { assistant, chat, emit, kind, run, SandError, shape, text, tool, unsure } from "../mod.ts";

const enc = new TextEncoder();
const env = (vars: Record<string, string> = {}) => (n: string) => vars[n];
const CFG = { models: { gpt: { kind: "chat", url: "https://chat.example/v1", model: "m" } }, default: { decide: "gpt", ask: "gpt" } };

/** One SSE body, handed out in `chunk`-byte pieces so no frame is whole in one read. */
function sseBody(frames: string[], opts: { chunk?: number; stall?: boolean; onCancel?: () => void } = {}) {
  const bytes = enc.encode(frames.join(""));
  let i = 0;
  return new ReadableStream<Uint8Array>({
    pull(c) {
      if (i < bytes.length) { const n = opts.chunk ?? 7; c.enqueue(bytes.slice(i, i + n)); i += n; return; }
      if (opts.stall) return new Promise<void>(() => {}); // the model went quiet; only an abort ends this
      c.close();
    },
    cancel() { opts.onCancel?.(); },
  });
}
const sse = (frames: string[], opts?: Parameters<typeof sseBody>[1]) =>
  new Response(sseBody(frames, opts), { status: 200, headers: { "content-type": "text/event-stream" } });
const chunk = (delta: unknown, extra: Record<string, unknown> = {}) =>
  `data: ${JSON.stringify({ id: "c1", object: "chat.completion.chunk", choices: [{ index: 0, delta, finish_reason: null }], ...extra })}\n\n`;
const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
const said = (content: string, usage: unknown = { total_tokens: 12 }) => json({ choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }], usage });

/** A fake chat host: each call is answered by the next script entry. */
function host(script: ((body: any) => Response)[]) {
  const calls: any[] = [];
  const f = (async (_url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body));
    calls.push(body);
    const step = script[Math.min(calls.length - 1, script.length - 1)];
    return step(body);
  }) as unknown as typeof fetch;
  return { fetch: f, calls };
}

Deno.test("stream: frames split across chunks, a keepalive comment, a whole tool call, usage, [DONE]", async () => {
  const round1 = [
    ": keepalive\n\n",
    chunk({ role: "assistant" }),
    chunk({ content: "Hello" }),
    ": keepalive\n\n",
    chunk({ content: ", desk." }),
    chunk({ tool_calls: [{ index: 0, id: "call_a", type: "function", function: { name: "file_a_ticket", arguments: '{"summary":"printer is down"}' } }] }),
    `data: ${JSON.stringify({ id: "c1", choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }], usage: { total_tokens: 41 } })}\n\n`,
    "data: [DONE]\n\n",
  ];
  const round2 = [
    chunk({ content: " Filed as T-1." }),
    `data: ${JSON.stringify({ id: "c1", choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { total_tokens: 12 } })}\n\n`,
    "data: [DONE]\n\n",
  ];
  const h = host([() => sse(round1, { chunk: 5 }), () => sse(round2, { chunk: 5 })]);
  const pieces: string[] = [];
  const Ticket = shape({ summary: text("one line") }, { name: "Ticket" });
  const filed: string[] = [];
  const fileTicket = tool("file a ticket", Ticket, (args) => { filed.push(args.summary); return { id: "T-1" }; }, { name: "file_a_ticket" });
  const { result, trace } = await run(async () => {
    const desk = assistant("you are a desk", { tools: [fileTicket], name: "desk" });
    let out = "";
    for await (const piece of desk.stream("my printer is down")) { pieces.push(piece); out += piece; }
    return { out, reply: desk.last?.text, stopped: desk.last?.stopped, turns: desk.last?.turns };
  }, { input: "x", fetchImpl: h.fetch, env: env(), config: CFG });

  assertEquals(result, { out: "Hello, desk. Filed as T-1.", reply: " Filed as T-1.", stopped: null, turns: 2 });
  assertEquals(pieces, ["Hello", ", desk.", " Filed as T-1."]);
  assertEquals(filed, ["printer is down"]);
  assertEquals(h.calls[0].stream, true);
  assertEquals(h.calls[0].stream_options, { include_usage: true });
  assertEquals(h.calls[0].tools[0].function.name, "file_a_ticket");
  assertEquals(h.calls[0].tools[0].function.parameters, { type: "object", properties: { summary: { type: "string" } }, required: ["summary"], additionalProperties: false });
  // the second round carries the assistant's tool_calls and the tool's answer
  assertEquals(h.calls[1].messages.at(-2).tool_calls[0].function.arguments, '{"summary":"printer is down"}');
  assertEquals(h.calls[1].messages.at(-1), { role: "tool", tool_call_id: "call_a", name: "file_a_ticket", content: '{"id":"T-1"}' });
  const names = trace.events.map((e) => e.name);
  assert(names.includes("sand.turn"), names.join(","));
  assertEquals(trace.events.filter((e) => e.name === "sand.tool").map((e) => e.kind), ["span.start", "span.end"]);
  assertEquals(trace.budget.spent.turns, 2);
  assertEquals(trace.budget.spent.tokens, 41 + 12);
});

Deno.test("stream: OpenAI-style tool-call fragments accumulate by index", async () => {
  const frames = [
    chunk({ tool_calls: [{ index: 0, id: "call_b", type: "function", function: { name: "lookup", arguments: '{"or' } }] }),
    chunk({ tool_calls: [{ index: 0, function: { arguments: 'der":"A' } }] }),
    chunk({ tool_calls: [{ index: 0, function: { arguments: '-12"}' } }] }),
    `data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }] })}\n\n`,
    "data: [DONE]\n\n",
  ];
  const seen: unknown[] = [];
  const Order = shape({ order: text("the order id") }, { name: "Order" });
  const lookup = tool("look an order up", Order, (args) => { seen.push(args.order); return "shipped"; }, { name: "lookup" });
  const h = host([() => sse(frames, { chunk: 11 }), () => sse([chunk({ content: "It shipped." }), "data: [DONE]\n\n"])]);
  const { result } = await run(async () => {
    const desk = assistant("you are a desk", { tools: [lookup], name: "desk" });
    const reply = await desk.say("where is A-12?", { emit: true });
    return { text: reply.text, calls: reply.calls.length, value: reply.value, arg: seen[0] };
  }, { input: "x", fetchImpl: h.fetch, env: env(), config: CFG });
  assertEquals(result, { text: "It shipped.", calls: 1, value: "It shipped.", arg: "A-12" });
});

Deno.test("stream: an error frame is E-STREAM in the provider's words; no usage is W-UNMETERED", async () => {
  const h = host([() => sse([chunk({ content: "Hel" }), `data: ${JSON.stringify({ error: { type: "engine_error", message: "the seat ran out of room" } })}\n\n`], { chunk: 9 })]);
  const err = await assertRejects(() => run(async () => {
    const desk = chat("you are a desk", { name: "desk" });
    return desk.say("hello", { emit: true });
  }, { input: "x", fetchImpl: h.fetch, env: env(), config: CFG }), SandError) as SandError;
  assertEquals(err.code, "E-STREAM");
  assert(err.message.includes("the seat ran out of room"), err.message);

  const quiet = host([() => sse([chunk({ content: "hello there" }), "data: [DONE]\n\n"])]);
  const { result, trace } = await run(async () => {
    const desk = chat("you are a desk", { name: "desk" });
    const r = await desk.say("hello", { emit: true });
    return r.tokens;
  }, { input: "x", fetchImpl: quiet.fetch, env: env(), config: CFG });
  assert((result as number) > 0, "tokens are estimated when nothing metered them");
  assert(trace.events.some((e) => e.name === "sand.log" && String(e.output).includes("W-UNMETERED")), "W-UNMETERED is said once");
  assertEquals(trace.events.filter((e) => e.name === "sand.log" && String(e.output).includes("W-UNMETERED")).length, 1);
  // emit coalesces into sand.output events carrying the text
  const out = trace.events.filter((e) => e.name === "sand.output");
  assertEquals(out.length, 1);
  assertEquals(out[0].output, "hello there");
  assertEquals(out[0].attrs?.chars, "11");
});

Deno.test("stream: an abort cancels the reader and is E-TIMEOUT", async () => {
  let cancelled = false;
  const frames = [chunk({ content: "Hel" }), chunk({ content: "lo" })];
  const h = host([() => sse(frames, { chunk: 6, stall: true, onCancel: () => { cancelled = true; } })]);
  const ctrl = new AbortController();
  const err = await assertRejects(() => run(async () => {
    const desk = chat("you are a desk", { name: "desk" });
    return desk.say("hello", { signal: ctrl.signal, onDelta: (p) => { if (p === "lo") ctrl.abort(); } });
  }, { input: "x", fetchImpl: h.fetch, env: env(), config: CFG }), SandError) as SandError;
  assertEquals(err.code, "E-TIMEOUT");
  assert(cancelled, "the reader was cancelled");
});

Deno.test("stream: breaking the loop stops the model and keeps what arrived", async () => {
  let cancelled = false;
  const h = host([() => sse([chunk({ content: "one " }), chunk({ content: "two " }), chunk({ content: "three" })], { chunk: 8, stall: true, onCancel: () => { cancelled = true; } })]);
  const { result } = await run(async () => {
    const desk = chat("you are a desk", { name: "desk" });
    let got = "";
    for await (const piece of desk.stream("count")) { got += piece; if (got.includes("two")) break; }
    return { got, stopped: desk.last?.stopped, value: desk.last?.value === unsure ? "unsure" : desk.last?.value };
  }, { input: "x", fetchImpl: h.fetch, env: env(), config: CFG });
  assertEquals((result as any).stopped, "aborted");
  assertEquals((result as any).value, "unsure");
  assert(String((result as any).got).startsWith("one two"), String((result as any).got));
  assert(cancelled, "breaking the loop cancelled the reader");
});

Deno.test("assistant: tools run one after another, past the turn bound the reply is unsure", async () => {
  const order = { id: "call_x", type: "function", function: { name: "ask_again", arguments: "{}" } };
  const call = () => json({ choices: [{ index: 0, message: { role: "assistant", content: null, tool_calls: [{ ...order }] }, finish_reason: "tool_calls" }], usage: { total_tokens: 5 } });
  const ran: number[] = [];
  const Empty = shape({ why: text("why") }, { name: "Empty" });
  const again = tool("ask again", Empty, async () => { ran.push(ran.length); await new Promise((r) => setTimeout(r, 1)); return "again"; }, { name: "ask_again" });
  const h = host([call]);
  const { result, trace } = await run(async () => {
    const desk = assistant("you are a desk", { tools: [again], settle: { turns: 3 }, name: "desk" });
    const r = await desk.say("go");
    return { value: r.value === unsure ? "unsure" : r.value, stopped: r.stopped, turns: r.turns, calls: r.calls.length, reason: r.reason };
  }, { input: "x", fetchImpl: h.fetch, env: env(), config: CFG });
  assertEquals((result as any).value, "unsure");
  assertEquals((result as any).stopped, "turns");
  assertEquals((result as any).turns, 3);
  assertEquals((result as any).calls, 3);
  assertEquals(ran, [0, 1, 2]);
  assertEquals(trace.budget.spent.turns, 3);
  assert(trace.events.some((e) => e.name === "sand.run" && e.kind === "span.end" && e.attrs?.turns === "3"));
});

Deno.test("assistant: a tool that throws is fed back as an error and the round continues; a budget stop is not", async () => {
  const bad = { id: "call_y", type: "function", function: { name: "risky", arguments: '{"why":"x"}' } };
  const h = host([
    () => json({ choices: [{ message: { role: "assistant", content: null, tool_calls: [bad] }, finish_reason: "tool_calls" }], usage: { total_tokens: 5 } }),
    () => said("I could not do that."),
  ]);
  const Why = shape({ why: text("why") }, { name: "Why" });
  const risky = tool("do the risky thing", Why, () => { throw new Error("the printer is on fire"); }, { name: "risky" });
  const { result } = await run(async () => {
    const desk = assistant("you are a desk", { tools: [risky], name: "desk" });
    const r = await desk.say("go");
    return { text: r.text, error: r.calls[0].error };
  }, { input: "x", fetchImpl: h.fetch, env: env(), config: CFG });
  assertEquals((result as any).text, "I could not do that.");
  assertEquals((result as any).error, "the printer is on fire");
  assertEquals(JSON.parse(String(h.calls[1].messages.at(-1).content)), { error: "the printer is on fire" });

  const stop = tool("stop the run", Why, () => { throw new SandError("E-BUDGET", "passes 1/1 spent"); }, { name: "risky" });
  const h2 = host([() => json({ choices: [{ message: { role: "assistant", content: null, tool_calls: [bad] }, finish_reason: "tool_calls" }], usage: { total_tokens: 5 } })]);
  const err = await assertRejects(() => run(async () => {
    const desk = assistant("you are a desk", { tools: [stop], name: "desk" });
    return desk.say("go");
  }, { input: "x", fetchImpl: h2.fetch, env: env(), config: CFG }), SandError) as SandError;
  assertEquals(err.code, "E-BUDGET");
});

Deno.test("chat takes no tools, assistant needs one, a tool needs a shape", async () => {
  const h = host([() => said("hi")]);
  const Why = shape({ why: text("why") }, { name: "Why" });
  const t = tool("do it", Why, () => "done", { name: "doit" });
  const e1 = await assertRejects(() => run(async () => chat("desk", { tools: [t] } as any), { input: "x", fetchImpl: h.fetch, env: env(), config: CFG }), SandError) as SandError;
  assertEquals(e1.code, "E-STATE"); assertEquals(e1.text, "chat takes no tools; use assistant");
  const e2 = await assertRejects(() => run(async () => assistant("desk", { tools: [] }), { input: "x", fetchImpl: h.fetch, env: env(), config: CFG }), SandError) as SandError;
  assertEquals(e2.code, "E-STATE"); assert(e2.message.includes("needs at least one tool"));
  const e3 = await assertRejects(() => run(async () => tool("do it", kind("k", { a: "x" }) as any, () => 1), { input: "x", fetchImpl: h.fetch, env: env(), config: CFG }), SandError) as SandError;
  assertEquals(e3.code, "E-TOOL-SHAPE");
  const e4 = await assertRejects(() => run(async () => assistant("desk", { tools: [{ name: "nope" } as any] }), { input: "x", fetchImpl: h.fetch, env: env(), config: CFG }), SandError) as SandError;
  assertEquals(e4.code, "E-TOOL-SHAPE");
});

Deno.test("history: `keep` bounds the wire, not what the conversation holds; forget clears it", async () => {
  const h = host([() => said("ok")]);
  const { result } = await run(async () => {
    const desk = chat("you are a desk", { keep: 2, name: "desk" });
    await desk.say("one"); await desk.say("two"); await desk.say("three");
    const held = desk.history.length;
    const last = desk.last?.text;
    desk.forget();
    return { held, after: desk.history.length, last };
  }, { input: "x", fetchImpl: h.fetch, env: env(), config: CFG });
  assertEquals(result, { held: 6, after: 0, last: "ok" });
  // keep counts messages, not turns: the two most recent plus the system line
  assertEquals(h.calls[2].messages.map((m: any) => m.role), ["system", "assistant", "user"]);
  assertEquals(h.calls[2].messages.at(-1).content, "three");
});

Deno.test("sessions: a conversation round-trips through run(), and { session: false } opts out", async () => {
  const h = host([() => said("the first answer"), () => said("the second answer")]);
  const first = await run(async () => {
    const desk = chat("you are a desk", { name: "desk" });
    await desk.say("hello");
  }, { input: "x", fetchImpl: h.fetch, env: env(), config: CFG });
  assertEquals(first.session!.state!.chats.desk.map((m: any) => m.role), ["user", "assistant"]);
  assertEquals(first.session!.state!.v, 1);

  const second = await run(async () => {
    const desk = chat("you are a desk", { name: "desk" });
    const r = await desk.say("and again");
    return r.text;
  }, { input: "x", fetchImpl: h.fetch, env: env(), config: CFG, session: first.session! });
  assertEquals(second.result, "the second answer");
  assertEquals(h.calls[1].messages.map((m: any) => m.content), ["you are a desk", "hello", "the first answer", "and again"]);
  assertEquals(second.session!.state!.chats.desk.length, 4);

  const none = await run(async () => { const desk = chat("you are a desk", { name: "desk" }); await desk.say("hello"); }, { input: "x", fetchImpl: h.fetch, env: env(), config: CFG, session: false });
  assertEquals(none.session, null);
});

Deno.test("emit: pieces reach the output port and coalesce into sand.output", async () => {
  const seen: string[] = [];
  const { trace } = await run(async () => {
    emit("one ");
    emit({ text: "two" });
    await new Promise((r) => setTimeout(r, 70));
    emit(" three");
  }, { input: "x", env: env(), config: CFG, output: (piece: string) => seen.push(piece) });
  assertEquals(seen, ["one ", "two", " three"]);
  const out = trace.events.filter((e) => e.name === "sand.output");
  assertEquals(out.map((e) => e.output), ["one two", " three"]);
});
