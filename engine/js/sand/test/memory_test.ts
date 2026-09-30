/**
 * The memory kind, pinned: the local file store through in-memory ports (idf
 * recall, serialised writes, a host with no writeFile), and the platform store
 * through a fake fetch (the three paths, the bearer, the metering headers, and
 * a 402 that is E-PASSES). No network, no disk.
 */
import { assert, assertEquals, assertRejects } from "jsr:@std/assert";
import { memory, rank, run, SandError, setPorts, termsOf } from "../mod.ts";

const env = (vars: Record<string, string> = {}) => (n: string) => vars[n];
const LOCAL = { models: { store: { kind: "memory", url: "file:/tmp/sand-test" } }, default: { memory: "store" } };
const PLATFORM = { models: { store: { kind: "memory", url: "https://api.example", key: "cmb_test" } }, default: { memory: "store" } };

/** A disk that is a Map: the two ports a local store needs. */
function disk(seed: Record<string, string> = {}) {
  const files = new Map(Object.entries(seed));
  const writes: string[] = [];
  return {
    files, writes,
    readFile: async (p: string) => files.get(p) ?? null,
    writeFile: async (p: string, text: string) => { writes.push(p); files.set(p, text); },
  };
}

Deno.test("local memory: remember, recall by idf-weighted overlap, forget", async () => {
  const d = disk();
  setPorts({ readFile: d.readFile, writeFile: d.writeFile });
  const { result, trace } = await run(async () => {
    const notes = memory("notes");
    const ids = await notes.remember([
      { text: "the printer on the second floor jams when it is humid" },
      { text: "the coffee machine is out of beans on Mondays" },
      { text: "the printer needs a new drum every eighteen months", record: { source: "the manual" } },
    ]);
    const hits = await notes.recall("what is wrong with the printer", { limit: 2 });
    const gone = await notes.forget(ids[1]);
    const after = await notes.recall("coffee");
    return { ids: ids.length, top: hits.map((h) => h.text.slice(0, 11)), record: hits.find((h) => h.record)?.record, gone, after: after.length };
  }, { input: "x", env: env(), config: LOCAL });
  const r = result as any;
  assertEquals(r.ids, 3);
  assertEquals(r.top, ["the printer", "the printer"]);
  assertEquals(r.record, { source: "the manual" });
  assertEquals(r.gone, 1);
  assertEquals(r.after, 0);
  assertEquals([...d.files.keys()], ["/tmp/sand-test/notes.json"]);
  assertEquals(JSON.parse(d.files.get("/tmp/sand-test/notes.json")!).items.length, 2);
  // one memory event per call, and no request event: a local store is free
  assertEquals(trace.events.filter((e) => e.name === "sand.memory").map((e) => e.attrs?.op), ["remember", "recall", "forget", "recall"]);
  assertEquals(trace.events.filter((e) => e.name === "sand.request").length, 0);
  assertEquals(trace.budget.spent.passes, 0);
  setPorts({ writeFile: null });
});

Deno.test("local memory: writes in one tick are serialised, so none is lost", async () => {
  const d = disk();
  setPorts({ readFile: d.readFile, writeFile: d.writeFile });
  const { result } = await run(async () => {
    const notes = memory("notes");
    await Promise.all(["one", "two", "three", "four"].map((t) => notes.remember(t)));
    return (await notes.recall("one two three four", { limit: 10 })).length;
  }, { input: "x", env: env(), config: LOCAL });
  assertEquals(result, 4);
  assertEquals(JSON.parse(d.files.get("/tmp/sand-test/notes.json")!).items.map((i: any) => i.text), ["one", "two", "three", "four"]);
  setPorts({ writeFile: null });
});

Deno.test("local memory: a host with no writeFile is told to use the platform store", async () => {
  const d = disk();
  setPorts({ readFile: d.readFile, writeFile: null });
  const err = await assertRejects(() => run(async () => memory("notes").remember("anything"), { input: "x", env: env(), config: LOCAL }), SandError) as SandError;
  assertEquals(err.code, "E-MEMORY");
  assertEquals(err.text, "no local store in this host; use the platform memory");
});

Deno.test("platform memory: the three paths, the bearer, the metering headers", async () => {
  const calls: { url: string; body: any; auth: unknown }[] = [];
  const answer = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
    new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", "x-combs-passes": "1", "x-combs-balance": "7", "x-combs-request": "r9", ...headers } });
  const f = (async (url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body));
    calls.push({ url: String(url), body, auth: (init.headers as Record<string, string>).authorization });
    if (String(url).endsWith("/remember")) return answer({ ids: ["m1"] });
    if (String(url).endsWith("/recall")) return answer({ hits: [{ id: "m1", text: "the printer jams", record: null, score: 0.8, at: "2026-09-30T00:00:00Z" }] });
    return answer({ forgotten: 1 });
  }) as unknown as typeof fetch;
  const { result, trace } = await run(async () => {
    const notes = memory("notes");
    const id = await notes.remember("the printer jams");
    const hits = await notes.recall("printer", { limit: 3 });
    const gone = await notes.forget(id);
    return { id, hits, gone };
  }, { input: "x", fetchImpl: f, env: env(), config: PLATFORM });
  const r = result as any;
  assertEquals(r.id, "m1");
  assertEquals(r.hits, [{ id: "m1", text: "the printer jams", record: null, score: 0.8, at: "2026-09-30T00:00:00Z" }]);
  assertEquals(r.gone, 1);
  assertEquals(calls.map((c) => c.url), [
    "https://api.example/v1/memory/notes/remember",
    "https://api.example/v1/memory/notes/recall",
    "https://api.example/v1/memory/notes/forget",
  ]);
  assertEquals(calls[0].auth, "Bearer cmb_test");
  assertEquals(calls[0].body, { items: [{ text: "the printer jams", record: null }] });
  assertEquals(calls[1].body, { query: "printer", limit: 3 });
  assertEquals(calls[2].body, { ids: ["m1"] });
  // metered: one request event each, the balance read from the header
  const reqs = trace.events.filter((e) => e.name === "sand.request");
  assertEquals(reqs.map((e) => e.attrs?.verb), ["remember", "recall", "forget"]);
  assertEquals(reqs[0].attrs?.balance, "7");
  assertEquals(trace.budget.spent.passes, 3);
});

Deno.test("platform memory: refusals go through the taxonomy — 402 is E-PASSES", async () => {
  const cases: [number, unknown, string][] = [
    [402, { error: "insufficient_balance", balance: 0 }, "E-PASSES"],
    [401, { error: "unauthorized" }, "E-AUTH"],
    [403, { error: "account_suspended" }, "E-SUSPENDED"],
    [500, { error: "boom" }, "E-HOST"],
  ];
  for (const [status, body, code] of cases) {
    const f = (async () => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } })) as unknown as typeof fetch;
    const err = await assertRejects(() => run(async () => memory("notes").recall("printer"), { input: "x", fetchImpl: f, env: env(), config: PLATFORM }), SandError) as SandError;
    assertEquals(err.code, code, String(status));
  }
  // an answer of the wrong shape is E-MEMORY in the store's name
  const odd = (async () => new Response(JSON.stringify({ ok: true }), { status: 200, headers: { "content-type": "application/json" } })) as unknown as typeof fetch;
  const err = await assertRejects(() => run(async () => memory("notes").recall("printer"), { input: "x", fetchImpl: odd, env: env(), config: PLATFORM }), SandError) as SandError;
  assertEquals(err.code, "E-MEMORY");
  assert(err.message.includes("hits"), err.message);
});

Deno.test("memory: a decision model is not a store, and zero config names one", async () => {
  const err = await assertRejects(() => run(async () => memory("notes").recall("x"), { input: "x", env: env({ COMBS_API_KEY: "k" }), config: { default: { memory: "combs" } } }), SandError) as SandError;
  assertEquals(err.code, "E-MEMORY");
  assert(err.message.includes("is a decision model"), err.message);
  const none = await assertRejects(() => run(async () => memory("notes").recall("x"), { input: "x", env: env() }), SandError) as SandError;
  assertEquals(none.code, "E-MEMORY");
  assert(none.message.includes("set COMBS_API_KEY"), none.message);
});

Deno.test("rank: a term everywhere counts for little, a rare one for much", () => {
  const items = [
    { id: "a", text: "the printer is broken" },
    { id: "b", text: "the printer is fine" },
    { id: "c", text: "the kettle is broken and the kettle is old" },
  ];
  assertEquals(rank("kettle", items, 5).map((h) => h.id), ["c"]);
  // the item carrying both terms wins; the other two follow, shortest first
  assertEquals(rank("printer broken", items, 5).map((h) => h.id), ["a", "b", "c"]);
  assertEquals(rank("printer", items, 5).map((h) => h.id).sort(), ["a", "b"]);
  assertEquals(rank("nothing here", items, 5), []);
  assertEquals(termsOf("The Printer's drum, 18 months."), ["the", "printer's", "drum", "18", "months"]);
});
