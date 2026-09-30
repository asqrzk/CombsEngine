/**
 * The transform, pinned: the ten examples of the ladder compile byte-for-byte
 * to their goldens (UPDATE_GOLDEN=1 regenerates), each one's `needs` is the one
 * the compiler derives, the first program's four lines cost one pass, every
 * refusal names what the program carries, and output stays line-for-line with
 * input.
 */
import { assert, assertEquals, assertStringIncludes, assertThrows } from "jsr:@std/assert";
import { needsOf, PROLOGUE_LINES, transform } from "../mod.ts";

const here = new URL(".", import.meta.url).pathname;
const read = (p: string) => Deno.readTextFileSync(here + p);
const LADDER: { id: string; level: number; needs: string[]; file: string; input: string; expect: string; blurb: string }[] =
  JSON.parse(read("../examples/index.json"));
const EXAMPLES = LADDER.map((e) => e.id);

Deno.test("examples: golden .ts and .sand.json", () => {
  for (const name of EXAMPLES) {
    const src = read(`../examples/${name}.sand`);
    const { code, plan } = transform(src, { file: `${name}.sand` });
    const planText = JSON.stringify({ ...plan, hash: "<hash>" }, null, 2) + "\n";
    if (Deno.env.get("UPDATE_GOLDEN")) {
      Deno.writeTextFileSync(here + `golden/${name}.ts`, code);
      Deno.writeTextFileSync(here + `golden/${name}.sand.json`, planText);
      continue;
    }
    assertEquals(code, read(`golden/${name}.ts`), `${name}.ts drifted from its golden`);
    assertEquals(planText, read(`golden/${name}.sand.json`), `${name}.sand.json drifted from its golden`);
  }
});

/** The first program as it was first written: bare condition, bare extract, under the three declarations. */
const FIRST_PROGRAM = `const Urgency = kind("how urgent is this", { now: "it must happen today", soon: "this week is fine", whenever: "no rush is expressed" });
const Channel = kind("how do they want to be reached", { email: "they ask for a written reply", phone: "they ask for a call", any: "no preference is expressed" });
const ServiceRequest = shape({ urgency: Urgency, channel: Channel });

if(the {{input}} is a service request )
{
extract ({{input}})
}
`;

Deno.test("first-program-verbatim: the four lines under three declarations cost one pass and speculate the shape", () => {
  const { code, plan, warnings } = transform(FIRST_PROGRAM);
  assertStringIncludes(code, 'const d1 = await decide(input, "the message is a service request"');
  assertStringIncludes(code, "speculate: [ServiceRequest]");
  assertStringIncludes(code, "(await extract (input, ServiceRequest, { at: \"7:1\" }))");
  assertEquals(plan.cost.min.passes, 1); assertEquals(plan.cost.max.passes, 1);
  assert(warnings.some((w) => w.code === "W-GENERIC-CONTRARY" && w.at === "5:4"));
  assert(warnings.some((w) => w.code === "N-EXTRACT-SHAPE" && w.at === "7:1"));
  assert(warnings.some((w) => w.code === "W-UNSETTLED"));
});

Deno.test("service-request example: unless names the contrary and extract names the shape; still one pass", () => {
  const { code, plan, warnings } = transform(read("../examples/service-request.sand"));
  assertStringIncludes(code, 'not: ["they are describing something that already happened"]');
  assertStringIncludes(code, "speculate: [ServiceRequest]");
  assertStringIncludes(code, "(await extract(input, ServiceRequest, { at: \"7:1\" }))");
  assertEquals(plan.cost.min.passes, 1); assertEquals(plan.cost.max.passes, 1);
  assert(!warnings.some((w) => w.code === "W-GENERIC-CONTRARY" || w.code === "N-EXTRACT-SHAPE"));
  assert(warnings.some((w) => w.code === "W-UNSETTLED"));
});

Deno.test("line-preserving: output lines = input lines + prologue + footer", () => {
  for (const name of EXAMPLES) {
    const src = read(`../examples/${name}.sand`);
    const { code } = transform(src);
    assertEquals(code.split("\n").length, src.split("\n").length + PROLOGUE_LINES + 3, name);
  }
});

Deno.test("if-else-unsure: a flat three-way chain", () => {
  const { code } = transform(`const K = kind("q", { a: "x", other: "y" }, { fallback: "other" });\nsettle({ floor: 0.5 });\nif (the {{input}} is an a) { log("a") } else { log("b") } unsure { log("c") }`);
  assertStringIncludes(code, "if ((d1.is(\"a\")) === true) { log(\"a\") } else if ((d1.is(\"a\")) !== unsure) { log(\"b\") } else { log(\"c\") }");
});

Deno.test("kleene: && ! over two atoms is one decideAll", () => {
  const { code, plan } = transform(`const K = kind("q", { refund: "x", other: "y" });\nconst M = scale("m", ["low", "high"]);\nsettle({ floor: 0.5 });\nif (the {{input}} is a refund && !(the {{input}} is at least high on M)) { log(1) }`);
  assertStringIncludes(code, "const [d1, d2] = await decideAll(input, [K, M]");
  assertStringIncludes(code, "if ((and(d1.is(\"refund\"), not(d2.atLeast(\"high\")))) === true)");
  assertEquals(plan.batches.length, 1); assertEquals(plan.batches[0].questions, 2);
});

Deno.test("switch: a verb discriminant reads .value; a template binding interpolates", () => {
  const { code } = transform(`const K = kind("q", { a: "x", b: "y" });\nsettle({ floor: 0.5 });\nswitch (decide({{input}}, K)) { case K.a: log(\`got {{input}}\`); break; case unsure: log("?"); }\nlog("plain {{input}} string")`);
  assertStringIncludes(code, "switch ((await decide(input, K, { at: \"3:9\" })).value)");
  assertStringIncludes(code, "log(`got ${input}`)");
  assertStringIncludes(code, "log(`plain ${input} string`)");
});

Deno.test("while: re-asked each turn, warned when unbounded, quiet under settle({ passes })", () => {
  const src = `let draft = ask("Draft {{input}}")\nwhile (the {{draft}} sounds rude unless it is only firm) { draft = ask("kinder: {{draft}}") }`;
  const a = transform(src);
  assertStringIncludes(a.code, 'while (((await decide(draft, "the draft sounds rude", { not: ["it is only firm"], noun: "the draft"');
  assert(a.warnings.some((w) => w.code === "W-UNBOUNDED-LOOP"));
  const b = transform("settle({ passes: 5 });\n" + src);
  assert(!b.warnings.some((w) => w.code === "W-UNBOUNDED-LOOP"));
});

Deno.test("hoisting: two consecutive ifs share a request; a reassignment blocks it; a loop body never hoists", () => {
  const K = `const K = kind("q", { a: "x", b: "y" });\nsettle({ floor: 0.5 });\n`;
  const merged = transform(K + `if (the {{input}} is an a) { log(1) }\nif (the {{input}} is a b) { log(2) }`);
  assertEquals(merged.plan.batches.length, 1);
  const blocked = transform(K + `let x = {{input}};\nif (the {{x}} is an a) { log(1) }\nx = "other";\nif (the {{x}} is a b) { log(2) }`);
  assertEquals(blocked.plan.batches.length, 2);
  const looped = transform(K + `if (the {{input}} is an a) { log(1) }\nfor (const y of [1]) { if (the {{input}} is a b) { log(2) } }`);
  assertEquals(looped.plan.batches.length, 2);
  assert(looped.plan.batches[1].loops.includes("for"));
});

Deno.test("user async functions keep their await; a sync arrow leaves the verb unawaited with a warning", () => {
  const { code, warnings } = transform(`async function lookup(id: string) { const r = await fetch(id); return r.text() }\nconst order = await lookup({{input}})\nconst f = (x: string) => ask(x)\nlog(order)`);
  assertStringIncludes(code, "const order = await lookup(input)");
  assert(warnings.some((w) => w.code === "W-SYNC-VERB"));
  assert(!code.includes("(await ask(x"));
});

Deno.test("using: the callback is made async and its calls carry the model", () => {
  const { code } = transform(`const K = kind("q", { a: "x" });\nsettle({ floor: 0.5 });\nusing("fast", () => { if (the {{input}} is an a) { log(1) } })`);
  assertStringIncludes(code, 'using("fast", async () => {');
  assertStringIncludes(code, 'model: "fast"');
});

Deno.test("errors: every compile refusal names what the program carries", () => {
  const cases: [string, string][] = [
    [`extract({{input}})`, "E-SHAPE"],
    [`const A = shape({ a: text("x") }); const B = shape({ b: text("y") }); extract({{input}})`, "E-SHAPE"],
    [`const K = kind("q", { a: "x" }); if (the {{input}} is an a unless it is b) { log(1) }`, "E-UNLESS"],
    [`const K = kind("q", { a: "x" }); const J = kind("r", { a: "z" }); if (the {{input}} is an a) { log(1) }`, "E-OPTION"],
    [`if (1 > 0) { log(1) } unsure { log(2) }`, "E-UNSURE"],
    [`export const x = 1`, "E-RESERVED"],
    [`const decide = 1`, "E-RESERVED"],
    [`if (the {{a}} matches the {{b}}) { log(1) }`, "E-BINDING"],
    [`const x = the {{input}} is nice ? 1 : 2`, "E-PROSE-PLACE"],
    [`function f() { if (the {{input}} is nice) { log(1) } }`, "E-ASYNC"],
    [`if (the {{input}} is nice) log(1)`, "E-BLOCK"],
    [`const s = "unterminated`, "E-PARSE"],
  ];
  for (const [src, code] of cases) {
    const err = assertThrows(() => transform(src), Error, undefined, src) as Error & { code?: string };
    assertEquals(err.code ?? err.message.split(" ")[0], code, src);
    assert(/^E-[A-Z-]+ \d+:\d+ · /.test(err.message), `message shape: ${err.message}`);
  }
});

Deno.test("never emits temperature, render or label", () => {
  for (const name of EXAMPLES) {
    const { code } = transform(read(`../examples/${name}.sand`));
    assert(!/\btemperature\s*:/.test(code) && !/\brender\s*:/.test(code) && !/\blabel\s*:/.test(code), name);
  }
});

Deno.test("global import style: the browser page reads globalThis.SAND", () => {
  const { code } = transform(`log("hi {{input}}")`, { importStyle: "global" });
  assert(code.startsWith("const { kind, scale, shape"));
  assertStringIncludes(code, "= globalThis.SAND;");
});

Deno.test("the ladder: every example's `needs` is the one the compiler derives, and its level is its own", () => {
  const levels = new Set<number>();
  for (const e of LADDER) {
    const { plan } = transform(read(`../examples/${e.file}`), { file: e.file });
    assertEquals(needsOf(plan), e.needs, `${e.id}: index.json says ${e.needs.join(", ")}`);
    assert(!levels.has(e.level), `${e.id}: level ${e.level} is used twice`);
    levels.add(e.level);
    assert(e.blurb && e.input && e.expect, `${e.id}: the gallery needs a blurb, an input and an expectation`);
  }
});

Deno.test("cost: every flow of the ladder prints the line the docs promise", () => {
  const want: Record<string, string> = {
    "service-request": "min 1 pass · max 1 pass",
    "support-desk": "min 1 pass · max 2 passes, 4 turns · asks a person up to 1 time",
    "order-assistant": "min 1 pass · max 2 passes, 12 turns · asks a person up to 1 time",
    "inbox-sweep": "min 0 passes · max 0 passes + n × 1 pass (capped at 200 by settle)",
    "doc-qa": "min 1 pass · max 2 passes, 1 turn, 2 memory calls",
    "moderator": "min 1 pass, 2 memory calls · max 1 pass, 7 memory calls · asks a person up to 1 time",
  };
  for (const [id, text] of Object.entries(want)) {
    const { plan } = transform(read(`../examples/${id}.sand`), { file: `${id}.sand` });
    assertEquals(plan.cost.text, text, id);
  }
});

Deno.test("T4b: a declared receiver's methods are awaited and stamped; another object's are left alone", () => {
  const src = `const desk = chat("you answer mail");\nconst notes = memory("notes");\nconst other = { say: (s: string) => s, remember: (s: string) => s };\n` +
    `const r = desk.say("hello");\nfor await (const p of desk.stream("again")) emit(p);\ndesk.forget();\n` +
    `const id = notes.remember(r);\nconst past = notes.recall("mail", { limit: 3 });\nnotes.forget(id);\n` +
    `other.say("x"); other.remember("y");\nlog(past.length);`;
  const { code, plan } = transform(src);
  assertStringIncludes(code, 'const r = (await desk.say("hello", { at: "4:11" }));');
  assertStringIncludes(code, 'for await (const p of desk.stream("again"))');
  assert(!code.includes("(await desk.stream"), "a stream is never awaited");
  assertStringIncludes(code, "desk.forget();");
  assert(!code.includes("(await desk.forget"), "a chat's forget is never awaited");
  assertStringIncludes(code, 'const id = (await notes.remember(r, { at: "7:12" }));');
  assertStringIncludes(code, 'const past = (await notes.recall("mail", { at: "8:14", limit: 3 }));');
  assertStringIncludes(code, '(await notes.forget(id, { at: "9:1" }));');
  assertStringIncludes(code, 'other.say("x"); other.remember("y");');
  assert(!code.includes("(await other."), "an undeclared receiver is nobody's business");
  assertEquals(plan.chats.map((c) => c.name), ["desk"]);
  assertEquals(plan.memories.map((m) => m.name), ["notes"]);
  assertEquals(plan.calls.filter((c) => c.on === "notes").map((c) => c.verb), ["remember", "recall", "forget"]);
});

Deno.test("T4b: a receiver inside an array or a combinator keeps its own await", () => {
  const { code } = transform(`const desk = chat("x");\nconst both = await Promise.all([desk.say("a"), desk.say("b")]);\nconst one = await race(desk.say("c"), desk.say("d"));\nlog(both, one);`);
  assert(!code.includes("(await desk.say"), code);
  assertStringIncludes(code, 'Promise.all([desk.say("a", { at: "2:33" }), desk.say("b", { at: "2:48" })])');
});

Deno.test("T3: a whole-confirm condition is three-valued and may carry an unsure block", () => {
  const { code } = transform(`settle({ wait: 1000 });\nif (confirm("ship it?")) { log("yes") } else { log("no") } unsure { log("nobody") }`);
  assertStringIncludes(code, 'let c1;  if ((c1 = (await confirm("ship it?", { at: "2:5" }))) === true) { log("yes") } else if ((c1) !== unsure) { log("no") } else { log("nobody") }');
  const bare = transform(`settle({ wait: 1000 });\nif (confirm("ship it?")) log("yes");`);
  assertStringIncludes(bare.code, '=== true) log("yes");');
  const written = transform(`settle({ wait: 1000 });\nif (confirm("ship it?") === true) { log("yes") }`);
  assertStringIncludes(written.code, 'if ((await confirm("ship it?", { at: "2:5" })) === true) { log("yes") }');
  assertThrows(() => transform(`if (1 > 0) { log(1) } unsure { log(2) }`), Error, "E-UNSURE");
});

Deno.test("T8: only a decision is a switch discriminant with .value", () => {
  const k = `const K = kind("q", { a: "x", b: "y" });\nsettle({ floor: 0.5, wait: 10 });\n`;
  assertStringIncludes(transform(k + `switch (decide({{input}}, K)) { case K.a: log(1); break; }`).code, ").value)");
  assert(!transform(k + `switch (ask("which? {{input}}")) { case "a": log(1); break; }`).code.includes(").value)"));
  assert(!transform(k + `switch (confirm("which?")) { case true: log(1); break; }`).code.includes(").value)"));
});

Deno.test("tools: the shape must be declared above, the list must be a list, and an unlisted tool is noted", () => {
  const S = `const Args = shape({ id: text("the id") });\n`;
  const ok = transform(S + `const t = tool("do it", Args, (a) => a.id);\nconst bot = assistant("you do things", { tools: [t] });\nemit(bot.say("go"));`);
  assertEquals(ok.plan.tools.map((t) => t.name), ["t"]);
  assert(!ok.warnings.some((w) => w.code === "N-TOOL-UNUSED"));
  const inline = transform(`const t = tool("do it", shape({ id: text("the id") }), (a) => a.id);\nlog(t.run({ id: "1" }));`);
  assertEquals(inline.plan.tools.length, 1);
  assertThrows(() => transform(S + `const t = tool("do it", { id: "x" }, (a) => a.id);`), Error, "E-TOOL-SHAPE");
  assertThrows(() => transform(S + `const t = tool("do it", Args, (a) => a.id);\nconst bot = assistant("you", { tools: t });`), Error, "E-TOOL-SHAPE");
  assertThrows(() => transform(`const t = tool("do it", Args, (a) => a.id);\nconst Args = shape({ id: text("the id") });`), Error, "E-TOOL-SHAPE");
  const unused = transform(S + `const t = tool("do it", Args, (a) => a.id);\nlog("nothing runs it");`);
  assert(unused.warnings.some((w) => w.code === "N-TOOL-UNUSED" && w.at === "2:7"), JSON.stringify(unused.warnings));
});

Deno.test("tools: net literals become the fence the run is given", () => {
  const { plan } = transform(`const Args = shape({ q: text("the query") });\nconst look = tool("look it up", Args, async (a) => (await fetch("https://api.example.com/?q=" + a.q)).text(), { net: ["api.example.com", "cdn.example.com"] });\nconst bot = assistant("you look things up", { tools: [look] });\nemit(bot.say("go"));`);
  assertEquals(plan.hosts, ["api.example.com", "cdn.example.com"]);
  assertEquals(plan.tools[0].net, ["api.example.com", "cdn.example.com"]);
});

Deno.test("bounds: an unbounded chat loop and a confirm with no wait are both said once", () => {
  const loop = transform(`const desk = chat("x");\nlet i = 0;\nwhile (i++ < 3) { emit(desk.say("again")) }`);
  assert(loop.warnings.some((w) => w.code === "W-UNBOUNDED-CHAT"), JSON.stringify(loop.warnings));
  const bounded = transform(`settle({ turns: 6 });\nconst desk = chat("x");\nlet i = 0;\nwhile (i++ < 3) { emit(desk.say("again")) }`);
  assert(!bounded.warnings.some((w) => w.code === "W-UNBOUNDED-CHAT"));
  const asked = transform(`let i = 0;\nwhile (i++ < 3) { emit(ask("again")) }`);
  assert(asked.warnings.some((w) => w.code === "W-UNBOUNDED-CHAT"));
  const nowait = transform(`if (confirm("ship it?") === true) { log(1) }`);
  assert(nowait.warnings.some((w) => w.code === "W-NO-CONFIRM-BUDGET"), JSON.stringify(nowait.warnings));
  const waited = transform(`settle({ wait: 60_000 });\nif (confirm("ship it?") === true) { log(1) }`);
  assert(!waited.warnings.some((w) => w.code === "W-NO-CONFIRM-BUDGET"));
  const unsettled = transform(`const notes = memory("notes");\nemit(notes.recall("anything"));`);
  assert(unsettled.warnings.some((w) => w.code === "W-UNSETTLED"));
});

Deno.test("E-RESERVED covers the six new names", () => {
  for (const name of ["chat", "assistant", "tool", "memory", "confirm", "emit"]) {
    const err = assertThrows(() => transform(`const ${name} = 1`), Error, undefined, name) as Error & { code?: string };
    assertEquals(err.code, "E-RESERVED", name);
  }
});

Deno.test("the plan: 0.2 carries uses, chats, tools, memories, confirms, hosts and the reshaped cost", () => {
  const { plan } = transform(read("../examples/support-desk.sand"), { file: "support-desk.sand" });
  assertEquals(plan.sand, "0.2");
  assertEquals(Object.keys(plan.uses), ["decide", "ask", "chat", "tools", "memory", "confirm", "emit"]);
  assertEquals(plan.chats, [{ name: "desk", at: "24:9", verb: "chat", turns: null }]);
  assertEquals(plan.tools, [{ name: "createTicket", at: "17:7", net: [] }]);
  assertEquals(plan.memories, []);
  assertEquals(plan.confirms.length, 1);
  assertEquals(plan.hosts, []);
  assertEquals(Object.keys(plan.cost), ["min", "max", "waits", "loops", "unbounded", "text"]);
  assertEquals(plan.cost.min, { passes: 1, turns: 0, memory: 0 });
  assertEquals(plan.cost.max, { passes: 2, turns: 4, memory: 0 });
  assertEquals(plan.cost.waits, 1);
  assertEquals(plan.settle, { at: "14:1", passes: true, turns: true, wait: true });
});

Deno.test("the diagnostics catalogue: the README lists exactly the codes the package emits", async () => {
  const CODE = /\b([EWN]-[A-Z][A-Z-]{2,})\b/g;
  const codesIn = (text: string) => new Set([...text.matchAll(CODE)].map((m) => m[1]));
  const files: string[] = [];
  for (const dir of ["compiler", "runtime", "runtime/wires", "cli"]) {
    for await (const e of Deno.readDir(here + "../" + dir)) if (e.isFile && /\.(js|ts)$/.test(e.name)) files.push(`${dir}/${e.name}`);
  }
  const emitted = new Set<string>();
  for (const f of files) for (const c of codesIn(read("../" + f))) emitted.add(c);
  const listed = codesIn(read("../README.md"));
  const missing = [...emitted].filter((c) => !listed.has(c)).sort();
  const invented = [...listed].filter((c) => !emitted.has(c)).sort();
  assertEquals(missing, [], `the README does not list: ${missing.join(", ")}`);
  assertEquals(invented, [], `the README lists what nothing emits: ${invented.join(", ")}`);
  assert(emitted.size > 30, `only ${emitted.size} codes were found; the scan is broken`);
});
