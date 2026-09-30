/**
 * The transform, pinned: the five examples compile byte-for-byte to their
 * goldens (UPDATE_GOLDEN=1 regenerates), the first program's four lines cost one pass,
 * every refusal names what the program carries, and output stays line-for-line
 * with input.
 */
import { assert, assertEquals, assertStringIncludes, assertThrows } from "jsr:@std/assert";
import { transform, PROLOGUE_LINES } from "../mod.ts";

const here = new URL(".", import.meta.url).pathname;
const read = (p: string) => Deno.readTextFileSync(here + p);
const EXAMPLES = ["service-request", "support-triage", "extract-shape", "review-sweep", "intent-router"];

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
