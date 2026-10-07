/**
 * Asking a person, pinned: the confirm port, nobody there, a wait that elapses,
 * and the clock that stops while the person thinks — so a slow human never
 * spends the `ms` budget. Every answer is a verdict, so `and`/`or`/`not`
 * compose and an `unsure` block can catch "nobody was there".
 */
import { assert, assertEquals, assertRejects } from "jsr:@std/assert";
import { and, Budget, confirm, not, or, run, SandError, settle, setPorts, shape, text, tool, unsure } from "../mod.ts";

const env = (vars: Record<string, string> = {}) => (n: string) => vars[n];

Deno.test("confirm: the port's answer is a verdict, and yes/no/anything-else all read", async () => {
  const asked: string[] = [];
  const answers = ["yes", "no", "maybe", true, { answer: "y", by: "tty" }];
  let i = 0;
  const { result, trace } = await run(async () => {
    const out = [];
    for (let k = 0; k < answers.length; k++) out.push(await confirm(`may I do thing ${k}?`, { at: `${k + 1}:1` }));
    return out.map((v) => (v === unsure ? "unsure" : v));
  }, {
    input: "x", env: env(),
    confirm: (q: string) => { asked.push(q); return answers[i++] as never; },
  });
  assertEquals(result, [true, false, "unsure", true, true]);
  assertEquals(asked.length, 5);
  const spans = trace.events.filter((e) => e.name === "sand.confirm");
  assertEquals(spans.length, 10);
  assertEquals(spans[0].kind, "span.start");
  assertEquals(spans[0].input, "may I do thing 0?");
  assertEquals(spans[1].attrs?.answer, "true");
  assertEquals(spans[1].attrs?.by, "port");
  assertEquals(spans.at(-1)!.attrs?.by, "tty");
  assert(spans[1].attrs?.waited !== undefined);
});

Deno.test("confirm: nobody is here, so it is unsure and the run says W-NOBODY once", async () => {
  setPorts({ confirm: null });
  const { result, trace } = await run(async () => {
    const a = await confirm("may I?", { at: "3:1" });
    const b = await confirm("may I?", { at: "3:1" });
    return { a: a === unsure ? "unsure" : a, b: b === unsure ? "unsure" : b, kleene: [and(a, true), or(a, true), not(a)].map((v) => (v === unsure ? "unsure" : v)) };
  }, { input: "x", env: env() });
  assertEquals((result as any).a, "unsure");
  assertEquals((result as any).kleene, ["unsure", true, "unsure"]);
  const said = trace.events.filter((e) => e.name === "sand.log" && String(e.output).includes("W-NOBODY"));
  assertEquals(said.length, 1, "said once per question");
  assertEquals(trace.events.filter((e) => e.name === "sand.confirm" && e.kind === "span.end")[0].attrs?.by, "nobody");
});

Deno.test("confirm: a wait that elapses is unsure, and the clock stops while the person thinks", async () => {
  const { result, trace } = await run(async () => {
    settle({ ms: 1000 });
    const slow = await confirm("take your time", { wait: 40, at: "4:1" });
    return slow === unsure ? "unsure" : slow;
  }, {
    input: "x", env: env(),
    confirm: () => new Promise<never>(() => {}), // a person who never answers
  });
  assertEquals(result, "unsure");
  const end = trace.events.filter((e) => e.name === "sand.confirm" && e.kind === "span.end")[0];
  assertEquals(end.attrs?.answer, "unsure");
  assert(Number(end.attrs?.waited) >= 35, String(end.attrs?.waited));
  // the wait is reported, and it is not charged to the ms limit
  assert(trace.budget.waited >= 35, String(trace.budget.waited));
  assert(trace.budget.ms < 35, `the clock kept running: ${trace.budget.ms} ms`);
  assert(trace.events.some((e) => e.name === "sand.log" && String(e.output).includes("W-NOBODY")));
});

Deno.test("budget: pause and resume take the human wait out of elapsed(), and turns are a limit", async () => {
  const b = new Budget({ ms: 50, turns: 2 });
  b.pause();
  await new Promise((r) => setTimeout(r, 60));
  b.resume();
  assert(b.elapsed() < 30, `elapsed ${b.elapsed()}`);
  assert(b.waited >= 55, `waited ${b.waited}`);
  await b.check();                  // the ms limit is not crossed: the wait was not ours
  b.record({ turns: 1 }); b.record({ turns: 1 });
  assertEquals(b.snapshot().spent.turns, 2);
  assertEquals(b.snapshot().waited, b.waited);
  let stopped: SandError | null = null;
  try { await b.check({ turns: 1 }); } catch (e) { stopped = e as SandError; }
  assertEquals(stopped?.code, "E-BUDGET");
  assert(stopped!.message.includes("turns 3/2"), stopped!.message);
  // nested waits are counted once
  const n = new Budget({});
  n.pause(); n.pause(); n.resume();
  await new Promise((r) => setTimeout(r, 20));
  n.resume();
  assert(n.waited >= 18, String(n.waited));
});

Deno.test("confirm: a tool may ask a person, and the answer decides what the tool returns", async () => {
  const call = { id: "call_r", type: "function", function: { name: "refund", arguments: '{"order":"A-12"}' } };
  const step = [
    () => new Response(JSON.stringify({ choices: [{ message: { role: "assistant", content: null, tool_calls: [call] }, finish_reason: "tool_calls" }], usage: { total_tokens: 8 } }), { status: 200, headers: { "content-type": "application/json" } }),
    () => new Response(JSON.stringify({ choices: [{ message: { role: "assistant", content: "I refunded A-12." }, finish_reason: "stop" }], usage: { total_tokens: 9 } }), { status: 200, headers: { "content-type": "application/json" } }),
  ];
  let n = 0;
  const f = (async () => step[Math.min(n++, 1)]()) as unknown as typeof fetch;
  const Order = shape({ order: text("the order id") }, { name: "Order" });
  const asked: string[] = [];
  const { result, trace } = await run(async (input) => {
    const { assistant } = await import("../mod.ts");
    const refund = tool("refund an order", Order, async (args) => {
      const ok = await confirm(`refund ${args.order}?`, { at: "9:3" });
      return ok === true ? { refunded: args.order } : { refunded: null, why: ok === unsure ? "nobody said" : "the person said no" };
    }, { name: "refund" });
    const desk = assistant("you are a desk", { tools: [refund], name: "desk" });
    const r = await desk.say(input);
    return { text: r.text, result: r.calls[0].result };
  }, {
    input: "refund A-12", fetchImpl: f, env: env(),
    config: { models: { gpt: { kind: "chat", url: "https://chat.example/v1", model: "m" } }, default: { ask: "gpt" } },
    confirm: (q: string) => { asked.push(q); return true; },
  });
  assertEquals(asked, ["refund A-12?"]);
  assertEquals((result as any).result, { refunded: "A-12" });
  assertEquals((result as any).text, "I refunded A-12.");
  // the confirm span sits inside the tool span
  const names = trace.events.filter((e) => ["sand.tool", "sand.confirm"].includes(e.name)).map((e) => `${e.name}:${e.kind}`);
  assertEquals(names, ["sand.tool:span.start", "sand.confirm:span.start", "sand.confirm:span.end", "sand.tool:span.end"]);
});

Deno.test("confirm: an empty question is refused", async () => {
  const err = await assertRejects(() => run(async () => confirm("  "), { input: "x", env: env() }), SandError) as SandError;
  assertEquals(err.code, "E-STATE");
});

Deno.test("confirm with a kind: the port is told what is asked for, and the answer carries data back", async () => {
  const seen: unknown[] = [];
  const { result, trace } = await run(async () => {
    const got = await confirm("Read the open ports from your computer?", { kind: "link_reading", args: { tool: "listening_ports" }, at: "5:1" });
    const plain = await confirm("and a plain question?", { at: "6:1" });
    return { answer: got.answer === unsure ? "unsure" : got.answer, data: got.data, by: got.by, plain: plain === unsure ? "unsure" : plain };
  }, {
    input: "x", env: env(),
    confirm: (q: string, ctx: Record<string, unknown>) => {
      seen.push({ q, kind: ctx.kind, args: ctx.args });
      return (ctx.kind ? { answer: true, data: { block: "Listening ports, read from this computer:\n```json\n{}\n```" }, by: "page" } : "no") as never;
    },
  });
  assertEquals(result, { answer: true, data: { block: "Listening ports, read from this computer:\n```json\n{}\n```" }, by: "page", plain: false });
  assertEquals(seen, [
    { q: "Read the open ports from your computer?", kind: "link_reading", args: { tool: "listening_ports" } },
    { q: "and a plain question?", kind: undefined, args: undefined },
  ]);
  const asks = trace.events.filter((e) => e.name === "sand.confirm" && e.kind === "span.start");
  assertEquals(asks[0].attrs?.kind, "link_reading");
  assertEquals(asks[0].input, { question: "Read the open ports from your computer?", kind: "link_reading", args: { tool: "listening_ports" } });
  assertEquals(asks[1].input, "and a plain question?");
  assertEquals(asks[1].attrs?.kind, undefined);
  const ends = trace.events.filter((e) => e.name === "sand.confirm" && e.kind === "span.end");
  assertEquals(ends[0].attrs?.kind, "link_reading");
  assertEquals(ends[0].attrs?.answer, "true");
  assertEquals(ends[0].output, true, "the data is the program's, never the trace's");
  assert(Number(ends[0].attrs?.data_bytes) > 10, String(ends[0].attrs?.data_bytes));
});

Deno.test("confirm with a kind: nobody, or a port that gives no data, answers with no data; a bad kind or oversize args are refused", async () => {
  setPorts({ confirm: null });
  const { result } = await run(async () => {
    const got = await confirm("may I read?", { kind: "link_reading", args: { tool: "dns" }, at: "7:1" });
    return { answer: got.answer === unsure ? "unsure" : got.answer, data: got.data };
  }, { input: "x", env: env() });
  assertEquals(result, { answer: "unsure", data: undefined });
  const { result: r2 } = await run(async () => {
    const got = await confirm("may I read?", { kind: "link_reading", at: "8:1" });
    return { answer: got.answer, data: got.data };
  }, { input: "x", env: env(), confirm: () => "yes" as never });
  assertEquals(r2, { answer: true, data: undefined });
  await assertRejects(
    () => run(async () => await confirm("x?", { kind: "Not A Kind", at: "9:1" }), { input: "x", env: env(), confirm: () => true as never }),
    SandError,
    "kind",
  );
  await assertRejects(
    () => run(async () => await confirm("x?", { kind: "link_files", args: { content: "x".repeat(60_000) }, at: "10:1" }), { input: "x", env: env(), confirm: () => true as never }),
    SandError,
    "args",
  );
});

Deno.test("budget: at a limit the person is asked, and a yes gives every limit its first size again", async () => {
  const asked: { limit: string; used: number; max: number; block: Record<string, number | null>; extensions: number }[] = [];
  const events: Record<string, unknown>[] = [];
  let say: boolean | null = true;
  const b = new Budget({ tokens: 100, turns: 2, ms: 60_000 }, (_name: string, _kind: string, attrs: Record<string, unknown>) => events.push(attrs), async (info: typeof asked[number]) => { asked.push(info); return say; });
  b.record({ tokens: 90, turns: 1 });
  await b.check({ tokens: 20 });                       // 110 > 100: asked, yes: tokens 200, turns 4, ms 120 s
  assertEquals(asked.length, 1);
  assertEquals(asked[0].limit, "tokens");
  assertEquals(asked[0].block, { passes: null, tokens: 100, turns: 2, ms: 60_000, credits: null });
  assertEquals(b.limits.tokens, 200);
  assertEquals(b.limits.turns, 4);
  assertEquals(b.limits.ms, 120_000);
  assertEquals(b.snapshot().extensions, 1);
  assertEquals(events.filter((e) => e.action === "ask").length, 1);
  assertEquals(events.filter((e) => e.action === "extended").length, 1);
  // a no stops it as before, with the person named in the message
  say = false;
  b.record({ tokens: 120 });
  let stopped: SandError | null = null;
  try { await b.check({ tokens: 1 }); } catch (e) { stopped = e as SandError; }
  assertEquals(stopped?.code, "E-BUDGET");
  assert(stopped!.message.includes("tokens 211/200") && stopped!.message.includes("did not carry it on"), stopped!.message);
  assertEquals(asked.length, 2);
  // a no sticks: nothing is asked again
  say = true;
  stopped = null;
  try { await b.check({ tokens: 1 }); } catch (e) { stopped = e as SandError; }
  assertEquals(stopped?.code, "E-BUDGET");
  assertEquals(asked.length, 2);
  // an asker that throws is a no
  const c = new Budget({ tokens: 10 }, null, async () => { throw new Error("gone"); });
  c.record({ tokens: 10 });
  stopped = null;
  try { await c.check({ tokens: 1 }); } catch (e) { stopped = e as SandError; }
  assertEquals(stopped?.code, "E-BUDGET");
  // nobody (null) is the plain stop, not a no: the next call may ask again
  let calls = 0;
  const d = new Budget({ tokens: 10 }, null, async () => { calls++; return null; });
  d.record({ tokens: 10 });
  stopped = null;
  try { await d.check({ tokens: 1 }); } catch (e) { stopped = e as SandError; }
  assert(stopped!.message.endsWith("bound the loop"), stopped!.message);
  assertEquals(d.declined, false);
});

Deno.test("budget: two calls at the limit raise one question, and a limit set after the first ask gets a block of its own", async () => {
  let resolve: ((v: boolean) => void) | null = null;
  let asks = 0;
  const b = new Budget({ tokens: 10 }, null, () => new Promise<boolean>((r) => { asks++; resolve = r; }));
  b.record({ tokens: 10 });
  const first = b.check({ tokens: 1 });
  const second = b.check({ tokens: 2 });
  await new Promise((r) => setTimeout(r, 5));
  assertEquals(asks, 1, "one question for both");
  resolve!(true);
  await first; await second;
  assertEquals(b.limits.tokens, 20);
  assertEquals(b.extensions, 1);
  // turns had no limit at the first ask; settle() sets one later, and a yes at that limit buys as much again of it
  b.narrow({ turns: 2 });
  b.record({ turns: 2 });
  const third = b.check({ turns: 1 });
  await new Promise((r) => setTimeout(r, 5));
  assertEquals(asks, 2);
  resolve!(true);
  await third;
  assertEquals(b.limits.turns, 4);
  assertEquals(b.block!.turns, 2);
  // narrowing after an extension lowers the block with the limits
  b.narrow({ tokens: 5 });
  assertEquals(b.limits.tokens, 5);
  assertEquals(b.block!.tokens, 5);
});

Deno.test("confirm: the kind budget is the runtime's own", async () => {
  const { result, trace } = await run(async () => {
    try { await confirm("go on?", { kind: "budget" }); return "allowed"; } catch (e) { return (e as SandError).code; }
  }, { input: "x", env: () => undefined, config: { models: {}, default: {} } as never });
  assertEquals(result, "E-STATE");
  assert(!trace.events.some((e) => e.name === "sand.confirm"));
});

Deno.test("budget: credits are counted at each model's rate, and the questions follow the schedule, then double", async () => {
  const asked: { limit: string; used: number; max: number; block: Record<string, number | null> }[] = [];
  const b = new Budget({ credits: 50, creditSteps: [100, 500, 1000, 5000], rates: { quick: 768, careful: 256 }, rate: null }, null, async (info: typeof asked[number]) => { asked.push(info); return true; });
  b.record({ passes: 2 });                                   // two decisions: 2 credits
  b.record({ tokens: 7680, model: "quick", completions: 1 }); // 10 credits
  b.record({ tokens: 2560, model: "careful", completions: 1 }); // 10 credits
  b.record({ tokens: 1000, model: "unknown", completions: 1 }); // no rate: not counted
  assertEquals(Math.round(b.spent.credits), 22);
  await b.check({ tokens: 0 });
  assertEquals(asked.length, 0, "under the first step: no question");
  b.record({ tokens: 256 * 30, model: "careful", completions: 1 }); // 52 credits now
  await b.check({ tokens: 0 });
  assertEquals(asked.length, 1);
  assertEquals(asked[0].limit, "credits");
  assertEquals(asked[0].used, 52);
  assertEquals(asked[0].max, 50);
  assertEquals(asked[0].block.credits, 100, "the next step is named");
  assertEquals(b.limits.credits, 100);
  b.record({ tokens: 256 * 60, model: "careful", completions: 1 }); // 112
  await b.check({ tokens: 0 });
  assertEquals(b.limits.credits, 500);
  b.record({ tokens: 256 * 1000, model: "careful", completions: 1 }); // 1112: past two steps at once
  await b.check({ tokens: 0 });
  assertEquals(asked.length, 3, "one question for the jump");
  assertEquals(asked[2].block.credits, 5000);
  assertEquals(b.limits.credits, 5000, "the first step above what is spent");
  await b.check({ tokens: 0 });
  assertEquals(asked.length, 3, "under the new cap: nothing asked");
  b.record({ tokens: 256 * 4000, model: "careful", completions: 1 }); // 5112: past the schedule's last step
  await b.check({ tokens: 0 });
  assertEquals(asked.length, 4);
  assertEquals(b.limits.credits, 10_000, "after the schedule, double until above");
});

Deno.test("budget: askMore asks the same question for another cap, once at a time, and a no sticks", async () => {
  let say: boolean | null = true;
  const asked: { limit: string }[] = [];
  const b = new Budget({ tokens: 1000 }, null, async (info: { limit: string }) => { asked.push(info); return say; });
  assertEquals(await b.askMore({ limit: "rounds", used: 8, max: 8, block: { rounds: 8 } }), true);
  assertEquals(asked.map((a) => a.limit), ["rounds"]);
  assertEquals(b.extensions, 1);
  say = false;
  assertEquals(await b.askMore({ limit: "rounds", used: 16, max: 16, block: { rounds: 8 } }), false);
  assertEquals(b.declined, true);
  assertEquals(await b.askMore({ limit: "rounds", used: 16, max: 16, block: { rounds: 8 } }), false);
  assertEquals(asked.length, 2, "a no is not asked twice");
  const nobody = new Budget({ tokens: 1000 });
  assertEquals(await nobody.askMore({ limit: "rounds", used: 1, max: 1 }), false);
});
