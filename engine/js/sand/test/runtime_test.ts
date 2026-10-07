/**
 * The runtime, pinned: the adapters' status map, coalescing, memo, speculation,
 * the budget, the floor, chat calibration, extraction across both classes,
 * configuration layering, and three-valued logic. No network: every wire is a
 * fake fetch.
 */
import { assert, assertEquals, assertRejects, assertThrows } from "jsr:@std/assert";
import { kind, scale, shape, text, settle, decide, decideAll, extract, ask, parallel, run, unsure, and, or, not, resolveConfig, mergeConfig, zeroConfig, deriveNet, shapeSchema, SandError, registerKind } from "../mod.ts";

/** A fake seat: one pass per request, `pick` chooses the winner per question. */
function seat(opts: { pick?: (q: any) => string; p?: number; status?: number; body?: unknown; passesHeader?: string } = {}) {
  const calls: any[] = [];
  const f = (async (_url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body)); calls.push({ url: String(_url), body, headers: init.headers });
    if (opts.status && opts.status !== 200) return new Response(JSON.stringify(opts.body ?? { error: { type: "invalid_request", message: "a noul needs at least one contrary in `tether.not` to score against" } }), { status: opts.status, headers: { "content-type": "application/json" } });
    const answers = body.questions.map((q: any) => {
      const ids = q.type === "noul" ? ["true", "false"] : (q.options ?? q.levels).map((o: any) => o.id);
      const top = opts.pick?.(q) ?? ids[0];
      const p = opts.p ?? 0.8;
      const dist: Record<string, number> = {}; ids.forEach((id: string) => { dist[id] = id === top ? p : (1 - p) / (ids.length - 1); });
      const a: any = { name: q.name, type: q.type, top, p_top: p, distribution: dist, decision: "act", reason: null, confidence: p, margin: 0.3, implications: [] };
      if (q.abstain && p < q.abstain.p_min) { a.top = q.abstain.option; a.decision = "abstain"; a.reason = `p_top ${p} < p_min ${q.abstain.p_min}`; a.abstain = { option: q.abstain.option, applied: true, from: top, reason: a.reason }; }
      if (q.type === "score") a.expected_level = ids.reduce((s: number, id: string, i: number) => s + i * dist[id], 0);
      return a;
    });
    const passes = String(_url).includes("/zs/") ? body.questions.reduce((n: number, q: any) => n + (q.options?.length ?? q.levels?.length ?? 0), 0) : 1;
    return new Response(JSON.stringify({ answers, model: "fake", usage: { forward_passes: passes, per_question: body.questions.map((q: any) => ({ name: q.name, option_tokens_cut: 0 })) } }), { status: 200, headers: { "content-type": "application/json", "x-combs-passes": opts.passesHeader ?? String(passes), "x-combs-balance": "9", "x-combs-request": "r1" } });
  }) as unknown as typeof fetch;
  return { fetch: f, calls };
}
const env = (vars: Record<string, string>) => (n: string) => vars[n];
const COMBS = { COMBS_API_KEY: "cmb_test", COMBS_API_URL: "https://seat.example" };
const runWith = (main: (input: string) => Promise<unknown>, fetchImpl: typeof fetch, extra: Record<string, string> = {}, config?: any) => run(main, { input: "the text", fetchImpl, env: env({ ...COMBS, ...extra }), config });

Deno.test("status map: refusals carry the API's words and the code the status names", async () => {
  const cases: [number, unknown, string][] = [
    [401, { error: "unauthorized — send: Authorization: Bearer <token>" }, "E-AUTH"],
    [402, { error: "insufficient_balance", balance: 0 }, "E-PASSES"],
    [403, { error: "account_suspended" }, "E-SUSPENDED"],
    [400, { error: { type: "invalid_request", message: "a noul needs at least one contrary in `tether.not` to score against" } }, "E-WIRE"],
    [503, { error: "keys_unavailable" }, "E-HOST"],
  ];
  for (const [status, body, code] of cases) {
    const s = seat({ status, body });
    const err = await assertRejects(() => runWith(async (i) => decide(i, "it is nice", { not: ["it is not"] }), s.fetch), SandError) as SandError;
    assertEquals(err.code, code, String(status));
    assert(err.message.includes("Nothing was charged"), err.message);
    if (status === 400) assert(err.message.includes("a noul needs at least one contrary"));
  }
});

Deno.test("coalescing: 20 same-state decides in one tick are two requests (16 + 4); a repeat hits the memo", async () => {
  const s = seat();
  const { trace } = await runWith(async (i) => {
    const specs = Array.from({ length: 20 }, (_, k) => ({ means: `proposition ${k}`, not: ["no"] }));
    const ds = await Promise.all(specs.map((sp) => decide(i, sp)));
    const again = await decide(i, specs[3]);
    assertEquals(again.request.memo, true);
    return ds.length;
  }, s.fetch);
  assertEquals(s.calls.length, 2);
  assertEquals(s.calls[0].body.questions.length, 16); assertEquals(s.calls[1].body.questions.length, 4);
  assertEquals(trace.budget.spent.passes, 2);
});

Deno.test("speculation: served from the memo on a seat, skipped on deberta", async () => {
  const K = kind("q", { a: "x", b: "y" }, { name: "K" });
  const M = scale("m", ["low", "high"], { name: "M" });
  const s = seat();
  await runWith(async (i) => { await decide(i, K, { speculate: [M] }); const d = await decide(i, M); assertEquals(d.request.memo, true); }, s.fetch);
  assertEquals(s.calls.length, 1); assertEquals(s.calls[0].body.questions.map((q: any) => q.type), ["choice", "score"]);
  const z = seat();
  await runWith(async (i) => { await decide(i, K, { speculate: [M] }); const d = await decide(i, M); assertEquals(d.request.memo, false); }, z.fetch, {}, { models: { combs: { road: "deberta" } } });
  assertEquals(z.calls.length, 2); assertEquals(z.calls[0].body.questions.length, 1);
});

Deno.test("floor: a low p_top is unsure with the reason in the wire's voice; a kind with a fallback abstains on the wire and keeps the honest top", async () => {
  const s = seat({ p: 0.4 });
  await runWith(async (i) => {
    settle({ floor: 0.55 });
    const d = await decide(i, "it is nice", { not: ["it is not"] });
    assertEquals(d.value, unsure); assertEquals(d.decision, "abstain"); assertEquals(d.reason, "p_top 0.400 < floor 0.55");
    const K = kind("q", { a: "x", other: "none" }, { fallback: "other", name: "K" });
    const k = await decide(i, K);
    assertEquals(k.value, unsure); assertEquals(k.top, "a"); assert(k.reason!.includes("p_min"));
    assertEquals(and(d.value, true), unsure); assertEquals(or(d.value, true), true); assertEquals(not(d.value), unsure);
  }, s.fetch);
  assertEquals(s.calls[0].body.questions[0].abstain, undefined);
  assertEquals(s.calls[1].body.questions[0].abstain, { option: "other", p_min: 0.55 });
});

Deno.test("budget: settle({ passes: 3 }) over five parallel items stops at three; tolerate returns errors per item", async () => {
  const s = seat();
  const err = await assertRejects(() => runWith(async () => {
    settle({ passes: 3 });
    await parallel(["a", "b", "c", "d", "e"], async (x) => decide(x, "it is nice", { not: ["no"] }), { limit: 1 });
  }, s.fetch), SandError) as SandError & { trace?: any };
  assertEquals(err.code, "E-BUDGET");
  assertEquals(err.trace.budget.spent.passes, 3);
  const t = seat();
  const { result } = await runWith(async () => { settle({ passes: 2 }); return parallel(["a", "b", "c"], async (x) => decide(x, "it is nice", { not: ["no"] }), { limit: 1, tolerate: true }); }, t.fetch);
  const r = result as unknown[];
  assertEquals(r.length, 3); assert(r[2] instanceof Error);
});

Deno.test("chat model: an enum answer with logprobs has a p; without them p is null; unsure takes the branch", async () => {
  const calls: any[] = [];
  const chat = (withLogprobs: boolean, answer = "true") => (async (_u: string, init: RequestInit) => {
    calls.push(JSON.parse(String(init.body)));
    const content = JSON.stringify({ answer });
    const lp = withLogprobs ? { content: [{ token: '{"', logprob: -0.01 }, { token: "answer", logprob: -0.01 }, { token: '":"', logprob: -0.01 }, { token: answer, logprob: -0.22, top_logprobs: [{ token: answer, logprob: -0.22 }, { token: answer === "true" ? "false" : "true", logprob: -1.7 }, { token: "unsure", logprob: -4 }] }, { token: '"}', logprob: -0.01 }] } : null;
    return new Response(JSON.stringify({ choices: [{ message: { role: "assistant", content }, logprobs: lp }], usage: { total_tokens: 30 } }), { status: 200, headers: { "content-type": "application/json" } });
  }) as unknown as typeof fetch;
  const cfg = { models: { gpt: { kind: "chat", url: "https://chat.example/v1", model: "m" } }, default: { decide: "gpt", ask: "gpt" } };
  const a = await run(async (i) => decide(i, "it is nice", { not: ["no"] }), { input: "x", fetchImpl: chat(true), env: env({}), config: cfg });
  const d = a.result as any;
  assertEquals(d.value, true); assert(d.p !== null && d.p > 0.7 && d.p < 0.9, String(d.p));
  assertEquals(calls[0].response_format.json_schema.schema.properties.answer.enum, ["true", "false", "unsure"]);
  const b = await run(async (i) => decide(i, "it is nice", { not: ["no"] }), { input: "x", fetchImpl: chat(false), env: env({}), config: cfg });
  assertEquals((b.result as any).p, null);
  assert(b.trace.events.some((e: any) => e.name === "sand.log" && String(e.output).includes("W-UNCALIBRATED")));
  const c = await run(async (i) => decide(i, "it is nice", { not: ["no"] }), { input: "x", fetchImpl: chat(true, "unsure"), env: env({}), config: cfg });
  assertEquals((c.result as any).value, unsure);
});

Deno.test("extract: kinded slots by decision, text slots by chat, one request each; using rules", async () => {
  const s = seat();
  const chatCalls: any[] = [];
  const both = (async (url: string, init: RequestInit) => {
    if (String(url).endsWith("/chat/completions")) { chatCalls.push(JSON.parse(String(init.body))); return new Response(JSON.stringify({ choices: [{ message: { content: JSON.stringify({ summary: "one line", topic: "a" }) } }], usage: { total_tokens: 20 } }), { status: 200, headers: { "content-type": "application/json" } }); }
    return s.fetch(url, init);
  }) as unknown as typeof fetch;
  const T = kind("topic", { a: "x", b: "y" }, { name: "T" });
  const S = shape({ topic: T, summary: text("one line") }, { name: "S" });
  const cfg = { models: { gpt: { kind: "chat", url: "https://chat.example/v1", model: "m" } }, default: { ask: "gpt" } };
  const { result } = await run(async (i) => extract(i, S), { input: "text", fetchImpl: both, env: env(COMBS), config: cfg });
  assertEquals(result, { topic: "a", summary: "one line" });
  assertEquals(s.calls.length, 1); assertEquals(chatCalls.length, 1);
  assertEquals(chatCalls[0].response_format.json_schema.schema, { type: "object", properties: { summary: { type: "string" } }, required: ["summary"], additionalProperties: false });
  const viaAsk = await run(async (i) => extract(i, S, { using: "ask" }), { input: "text", fetchImpl: both, env: env(COMBS), config: cfg });
  assertEquals(viaAsk.result, { topic: "a", summary: "one line" });
  assertEquals(chatCalls[1].response_format.json_schema.schema.properties.topic.enum, ["a", "b", "unsure"]);
  const err = await assertRejects(() => run(async (i) => extract(i, S, { using: "decide" }), { input: "text", fetchImpl: both, env: env(COMBS), config: cfg }), SandError) as SandError;
  assertEquals(err.code, "E-MODEL");
  const none = await assertRejects(() => run(async (i) => extract(i, S), { input: "text", fetchImpl: both, env: env(COMBS), config: { default: { ask: "none" } } }), SandError) as SandError;
  assert(none.message.includes('slot "summary" needs a chat model'), none.message);
});

Deno.test("ask needs a chat model; classify passes the wire through", async () => {
  const s = seat();
  const err = await assertRejects(() => runWith(async () => ask("hello"), s.fetch, {}, { default: { ask: "none" } }), SandError) as SandError;
  assertEquals(err.code, "E-MODEL"); assert(err.message.includes("set OPENAI_API_KEY"));
});

Deno.test("config: zero from env, layers merge per model, $NAME resolves or names the variable, net derives and only narrows", () => {
  const z = resolveConfig({ env: env({ COMBS_API_KEY: "k" }) });
  assertEquals(Object.keys(z.models), ["combs", "combs-chat", "combs-memory"]);
  assertEquals(z.default, { decide: "combs", ask: "combs-chat", memory: "combs-memory" });
  assertEquals(z.models["combs-chat"].url, "https://api.combs.network/v1"); assertEquals(z.net, ["api.combs.network"]);
  const b = resolveConfig({ env: env({ COMBS_API_KEY: "k", OPENAI_API_KEY: "s" }) });
  assertEquals(b.default, { decide: "combs", ask: "openai", memory: "combs-memory" }); assertEquals(b.models.openai.model, "gpt-4o-mini");
  const n = resolveConfig({ env: env({}) });
  assertEquals(n.default.decide, null); assertEquals(n.default.memory, "none");
  const layered = resolveConfig({ env: env({ COMBS_API_KEY: "k", MY_KEY: "zz" }), layers: [{ models: { combs: { road: "label-anchors" }, mine: { kind: "chat", url: "http://127.0.0.1:8080/v1", key: "$MY_KEY" } }, default: { ask: "mine" } }] });
  assertEquals(layered.models.combs.road, "label-anchors"); assertEquals(layered.models.mine.key, "zz"); assertEquals(layered.net.sort(), ["127.0.0.1:8080", "api.combs.network"]);
  const missing = resolveConfig({ env: env({ COMBS_API_KEY: "k" }), layers: [{ models: { mine: { kind: "chat", url: "http://x/v1", key: "$MISSING" } } }] });
  assertEquals(missing.models.mine.unavailable, "MISSING");
  assertThrows(() => deriveNet({ a: { url: "https://a.example" } }, ["b.example"]), SandError, "names b.example");
  assertEquals(deriveNet({ a: { url: "https://a.example" }, b: { url: "https://b.example:8443/v1" } }, ["b.example:8443"]), ["b.example:8443"]);
  assertEquals(mergeConfig({ models: { a: { kind: "chat", url: "u" } } }, { models: { a: { model: "m" } } }).models, { a: { kind: "chat", url: "u", model: "m" } });
  assertEquals(zeroConfig(env({})), { models: {} });
});

Deno.test("schema: shapes stay inside the engine subset; a refused keyword is named", () => {
  const T = kind("t", { a: "x" }, { name: "T" });
  const S = shape({ topic: T, note: text("n") }, { name: "S" });
  assertEquals(shapeSchema(S.spec), { type: "object", properties: { topic: { type: "string", enum: ["a", "unsure"] }, note: { type: "string" } }, required: ["topic", "note"], additionalProperties: false });
});

Deno.test("kleene: the truth table over {true, false, unsure}", () => {
  const U = unsure;
  assertEquals([and(true, true), and(true, false), and(true, U), and(false, U), and(U, U)], [true, false, U, false, U]);
  assertEquals([or(false, false), or(false, true), or(false, U), or(true, U), or(U, U)], [false, true, U, true, U]);
  assertEquals([not(true), not(false), not(U)], [false, true, U]);
});

Deno.test("registerKind: a new wire is one function", async () => {
  registerKind("echo", (model: any) => ({ kind: "echo", name: model.name, road: null, path: "/echo", caps: { batch: 4, perRequest: true, noul: true, abstain: "client", schema: false, logprobs: false, text: false },
    async decide(req: any) { return { answers: req.questions.map((q: any) => ({ name: q.name, type: q.type, top: q.type === "noul" ? "false" : (q.options ?? q.levels)[0].id, p_top: 1, distribution: {}, decision: "act", reason: null })), passes: 0, tokens: 0, ms: 0 }; } }));
  const { result } = await run(async (i) => decide(i, "it is nice", { not: ["no"] }), { input: "x", env: env({}), config: { models: { e: { kind: "echo", url: "http://e" } }, default: { decide: "e" } } });
  assertEquals((result as any).value, false);
});

Deno.test("config: an env list names the variables a program's own tools read, kept once each across layers", () => {
  const merged = mergeConfig({ env: ["COMBS_LINK_TOKEN", "$COMBS_LINK_FILES_TOKEN"] }, { env: ["COMBS_LINK_TOKEN", "OTHER"] }) as { env?: string[] };
  assertEquals(merged.env, ["COMBS_LINK_TOKEN", "COMBS_LINK_FILES_TOKEN", "OTHER"]);
  assertEquals((mergeConfig({ models: {} }) as { env?: string[] }).env, undefined);
});
