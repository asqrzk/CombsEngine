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
import { billable } from "../runtime/wires/chat.js";

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

Deno.test("stream: reasoning arrives apart from the answer, is its own event ahead of the output, and stays out of the history", async () => {
  const frames = [
    chunk({ role: "assistant" }),
    chunk({ reasoning_content: "The user wants " }),
    chunk({ reasoning_content: "a greeting." }),
    chunk({ content: "Hello" }),
    chunk({ content: " there." }),
    `data: ${JSON.stringify({ id: "c1", choices: [], usage: { total_tokens: 30 } })}\n\n`,
    "data: [DONE]\n\n",
  ];
  const h = host([() => sse(frames, { chunk: 9 }), () => sse([chunk({ content: "Fine." }), "data: [DONE]\n\n"])]);
  const { result, trace } = await run(async () => {
    const desk = chat("you are a desk", { name: "desk" });
    const first = await desk.say("hi", { emit: true });
    await desk.say("how are you", { emit: true });
    return { text: first.text, thinking: first.thinking, timed: first.thinkingMs >= 0, history: desk.history.map((m) => [m.role, m.content]) };
  }, { input: "x", fetchImpl: h.fetch, env: env(), config: CFG });
  const out = result as Record<string, unknown>;
  assertEquals(out.text, "Hello there.");
  assertEquals(out.thinking, "The user wants a greeting.");
  assertEquals(out.timed, true);
  // what the model thought is not part of what it said: the history, and so the next request, holds answers only
  assertEquals(out.history, [["user", "hi"], ["assistant", "Hello there."], ["user", "how are you"], ["assistant", "Fine."]]);
  assertEquals(JSON.stringify(h.calls[1].messages).includes("greeting"), false);
  assertEquals(trace.events.filter((e) => e.name === "sand.thinking").map((e) => e.output).join(""), "The user wants a greeting.");
  assertEquals(trace.events.filter((e) => e.name === "sand.output").map((e) => e.output).join(""), "Hello there.Fine.");
  const order = trace.events.map((e) => e.name).filter((n) => n === "sand.thinking" || n === "sand.output");
  assertEquals(order[0], "sand.thinking");
  assertEquals(order.indexOf("sand.output") > order.lastIndexOf("sand.thinking"), true);
});

Deno.test("buffered: a reply's reasoning_content is read apart from its content", async () => {
  const h = host([() => json({ choices: [{ index: 0, message: { role: "assistant", content: "42", reasoning_content: "six times seven" }, finish_reason: "stop" }], usage: { total_tokens: 9 } })]);
  const { result } = await run(async () => {
    const desk = chat("you are a desk", { name: "desk" });
    const r = await desk.say("6 x 7?");
    return { text: r.text, thinking: r.thinking, value: r.value };
  }, { input: "x", fetchImpl: h.fetch, env: env(), config: CFG });
  assertEquals(result, { text: "42", thinking: "six times seven", value: "42" });
});

Deno.test("stream: a model that spends its tokens thinking has said nothing, and the reply says so with its thinking kept", async () => {
  const frames = [
    chunk({ reasoning_content: "Let me think about this at length. " }),
    chunk({ reasoning_content: "And a little more." }),
    `data: ${JSON.stringify({ id: "c1", choices: [{ index: 0, delta: {}, finish_reason: "length" }], usage: { total_tokens: 64 } })}\n\n`,
    "data: [DONE]\n\n",
  ];
  const h = host([() => sse(frames, { chunk: 13 })]);
  const { result } = await run(async () => {
    const desk = chat("you are a desk", { name: "desk" });
    const r = await desk.say("hard question", { emit: true, maxTokens: 64 });
    return { text: r.text, stopped: r.stopped, thinking: r.thinking, isUnsure: r.value === unsure };
  }, { input: "x", fetchImpl: h.fetch, env: env(), config: CFG });
  const out = result as Record<string, unknown>;
  assertEquals(out.text, "");
  assertEquals(out.stopped, "length");
  assertEquals(out.thinking, "Let me think about this at length. And a little more.");
  assertEquals(out.isUnsure, false); // the reply is empty and stopped at the ceiling; `stopped` is what says why
});

Deno.test("stream: a tool call ends the thinking, and a stop in the middle of it keeps what was thought", async () => {
  const frames = [
    chunk({ reasoning_content: "I should draw it." }),
    chunk({ tool_calls: [{ index: 0, id: "call_a", type: "function", function: { name: "draw", arguments: "{}" } }] }),
    `data: ${JSON.stringify({ id: "c1", choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }], usage: { total_tokens: 20 } })}\n\n`,
    "data: [DONE]\n\n",
  ];
  const Draw = shape({}, { name: "Draw" });
  const draw = tool("draw", Draw, () => "done", { name: "draw" });
  const h = host([() => sse(frames, { chunk: 11 }), () => sse([chunk({ content: "Drawn." }), "data: [DONE]\n\n"])]);
  const { result } = await run(async () => {
    const painter = assistant("you paint", { tools: [draw], name: "p" });
    const r = await painter.say("draw", { emit: true });
    return { text: r.text, thinking: r.thinking, calls: r.calls.length };
  }, { input: "x", fetchImpl: h.fetch, env: env(), config: CFG });
  assertEquals(result, { text: "Drawn.", thinking: "I should draw it.", calls: 1 });
});

Deno.test("assistant: past the budget the person is asked; a yes carries the reply on, nobody keeps what was said", async () => {
  const order = { id: "call_b", type: "function", function: { name: "look", arguments: "{}" } };
  const call = () => json({ choices: [{ index: 0, message: { role: "assistant", content: null, tool_calls: [{ ...order }] }, finish_reason: "tool_calls" }], usage: { total_tokens: 5 } });
  const Empty = shape({ why: text("why") }, { name: "Empty" });
  const look = tool("look", Empty, async () => "seen", { name: "look" });
  const tight = { ...CFG, limits: { turns: 1 } };
  // nobody there to answer: the turn stops at the limit, after a round, and says so
  const h1 = host([call, () => said("done")]);
  const r1 = await run(async () => {
    const desk = assistant("you are a desk", { tools: [look], settle: { turns: 4 }, name: "desk" });
    const r = await desk.say("go");
    return { value: r.value === unsure ? "unsure" : r.value, stopped: r.stopped, turns: r.turns, reason: r.reason };
  }, { input: "x", fetchImpl: h1.fetch, env: env(), config: tight });
  assertEquals((r1.result as any).stopped, "budget");
  assertEquals((r1.result as any).value, "unsure");
  assertEquals((r1.result as any).turns, 1);
  assert(String((r1.result as any).reason).includes("turns 2/1"), String((r1.result as any).reason));
  assert(r1.trace.events.some((e) => e.name === "sand.confirm" && e.kind === "span.start" && e.attrs?.kind === "budget"), "the budget asked, as a confirm of kind budget");
  assertEquals(h1.calls.length, 1);
  // the person says go on: the limits grow by their first size, and the reply finishes
  const asked: { kind?: string; args?: { block?: { turns?: number } } }[] = [];
  const h2 = host([call, () => said("done")]);
  const r2 = await run(async () => {
    const desk = assistant("you are a desk", { tools: [look], settle: { turns: 4 }, name: "desk" });
    const r = await desk.say("go");
    return { text: r.text, stopped: r.stopped, turns: r.turns };
  }, { input: "x", fetchImpl: h2.fetch, env: env(), config: tight, confirm: (_q: string, ctx: { kind?: string | null; args?: unknown }) => { asked.push(ctx as any); return true; } });
  assertEquals((r2.result as any), { text: "done", stopped: null, turns: 2 });
  assertEquals(asked.length, 1);
  assertEquals(asked[0].kind, "budget");
  assertEquals(asked[0].args?.block?.turns, 1);
  assertEquals(r2.trace.budget.limits.turns, 2);
  assertEquals(r2.trace.budget.extensions, 1);
});

Deno.test("usage: a cached prompt token counts a tenth against the budget, the rest whole", () => {
  assertEquals(billable({ total_tokens: 1000, prompt_tokens: 900, completion_tokens: 100, prompt_tokens_details: { cached_tokens: 800 } }), 280);
  assertEquals(billable({ total_tokens: 1000, prompt_tokens: 900, completion_tokens: 100 }), 1000);
  assertEquals(billable({ total_tokens: 100, prompt_tokens: 50, completion_tokens: 50, prompt_tokens_details: { cached_tokens: 500 } }), 55);
  assertEquals(billable(null), null);
  assertEquals(billable({}), null);
});

Deno.test("assistant: a tool call the host could not read is told to the model once, off the record; twice ends the reply, not the run", async () => {
  const bad = () => new Response(JSON.stringify({ error: { code: 500, message: "Failed to parse tool call arguments as JSON: [json.exception.parse_error.101] parse error at line 1, column 1655: syntax error while parsing value - invalid string: missing closing quote; last read: '\"cat >" } }), { status: 500, headers: { "content-type": "application/json" } });
  const Why = shape({ why: text("why") }, { name: "Why" });
  const look = tool("look", Why, async () => "seen", { name: "look" });
  // once: the model is told, on the wire only, and answers
  const h = host([bad, () => said("done"), () => said("again")]);
  const r = await run(async () => {
    const desk = assistant("you are a desk", { tools: [look], settle: { turns: 4 }, name: "desk" });
    const a = await desk.say("go");
    const b = await desk.say("more");
    return { text: a.text, stopped: a.stopped, turns: a.turns, later: b.text };
  }, { input: "x", fetchImpl: h.fetch, env: env(), config: CFG });
  assertEquals(r.result as any, { text: "done", stopped: null, turns: 1, later: "again" });
  assertEquals(h.calls.length, 3);
  const nudge = h.calls[1].messages.at(-1);
  assertEquals(nudge.role, "user");
  assert(String(nudge.content).includes("could not be read by the host") && String(nudge.content).includes("missing closing quote"), String(nudge.content));
  assert(!String(nudge.content).includes("Nothing was charged"), "the host's tail is not repeated to the model");
  assertEquals(h.calls[1].messages.length, h.calls[0].messages.length + 1);
  // the note never entered the conversation: the next turn's wire holds the question, the answer and the new question
  assert(!h.calls[2].messages.some((m: { content?: unknown }) => String(m.content ?? "").includes("could not be read")), "off the record");
  assertEquals(h.calls[2].messages.slice(-3).map((m: { role: string }) => m.role), ["user", "assistant", "user"]);
  assert(r.trace.events.some((e) => e.name === "sand.log" && String(e.output).includes("W-TOOL-CALL")));
  // twice: the reply ends with what it has and says why; the run goes on
  const h2 = host([bad, bad]);
  const r2 = await run(async () => {
    const desk = assistant("you are a desk", { tools: [look], settle: { turns: 4 }, name: "desk" });
    const a = await desk.say("go");
    return { value: a.value === unsure ? "unsure" : a.value, stopped: a.stopped, reason: a.reason };
  }, { input: "x", fetchImpl: h2.fetch, env: env(), config: CFG });
  assertEquals((r2.result as any).stopped, "tool_call");
  assertEquals((r2.result as any).value, "unsure");
  assert(String((r2.result as any).reason).includes("twice"), String((r2.result as any).reason));
  assertEquals(h2.calls.length, 2);
  // any other host error still fails the turn as before
  const h3 = host([() => new Response(JSON.stringify({ error: { message: "the model host is on fire" } }), { status: 500, headers: { "content-type": "application/json" } })]);
  const err = await assertRejects(() => run(async () => {
    const desk = assistant("you are a desk", { tools: [look], name: "desk" });
    return desk.say("go");
  }, { input: "x", fetchImpl: h3.fetch, env: env(), config: CFG }), SandError) as SandError;
  assertEquals(err.code, "E-HOST");
});
