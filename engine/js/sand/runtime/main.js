/**
 * @combs/sand — the Deno entry behind `if (import.meta.main) run(main)`.
 *
 * Two modes. Under `sand run` (SAND_CHILD=1) the child speaks the **child
 * protocol v1**, one JSON object per line:
 *
 *   in   `{sand:"1", config, limits, file?, memoryDir?, traceId?}`   the preamble, first and once
 *   out  `{sand:"1", ready:true}`                                     it read the preamble and the fence held
 *   in   `{input, session?, traceId?}`                                one run (a bare JSON string still works)
 *   in   `{confirm:{id, answer}}`                                     at any time: a person's answer
 *   out  `{result, session?}` · `{error:{code,message,at}, session?}`  one envelope per input, in order
 *   err  one trace event per line, including the `sand.confirm` ask that carries the `id`
 *
 * Stdin is read continuously, so inputs queue behind the run in flight while a
 * control line resolves a pending confirm at once; it stays open until the
 * parent has every envelope. Run directly (`deno run program.ts "text"`) it
 * takes the input from the arguments or stdin, prints the result as JSON and a
 * one-line summary.
 *
 * The only file in the runtime that reads Deno.* (besides the worker).
 */

import { run, configure, setPorts } from "./verbs.js";

const encoder = new TextEncoder();
const envGet = (n) => { try { return Deno.env.get(n); } catch { return undefined; } };
const readFile = async (p) => { try { return await Deno.readTextFile(p); } catch { return null; } };
const writeFile = async (p, text) => {
  const dir = p.replace(/\/[^/]*$/, "");
  if (dir && dir !== p) { try { await Deno.mkdir(dir, { recursive: true }); } catch { /* already there, or not ours to make */ } }
  await Deno.writeTextFile(p, text);
};

async function* lines(reader) {
  const decoder = new TextDecoder();
  let buf = "";
  for await (const chunk of reader) {
    buf += decoder.decode(chunk, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) { const line = buf.slice(0, i); buf = buf.slice(i + 1); if (line.trim()) yield line; }
  }
  if (buf.trim()) yield buf;
}

const writeLine = (stream, o) => stream.writeSync(encoder.encode(JSON.stringify(o) + "\n"));

function stderrBus() {
  const events = [];
  return { events, publish(partial) { const ev = { id: `e${events.length + 1}`, ts: Date.now(), ...partial }; events.push(ev); writeLine(Deno.stderr, ev); return ev; } };
}

const summary = (trace) => {
  const b = trace.budget;
  const decisions = trace.events.filter((e) => e.name === "sand.decision").map((e) => `${e.attrs?.at ?? ""} ${e.attrs?.top ?? ""}${e.attrs?.p_top ? " p=" + e.attrs.p_top : ""}`).join(" · ");
  return `${b.spent.passes} pass${b.spent.passes === 1 ? "" : "es"} · ${b.spent.tokens} tokens · ${b.ms} ms${decisions ? " · " + decisions : ""}`;
};

/** The child half of protocol v1. Answers one envelope per input and never exits while one is owed. */
async function childMain(main, opts) {
  let pre = null, traceId = null, confirmSeq = 0, failures = 0, closed = false, wake = null;
  const queue = [], pending = new Map();
  const wakeUp = () => { const w = wake; wake = null; w?.(); };

  setPorts({
    env: envGet, readFile, writeFile, log: () => {},
    /** A person is asked over the wire: the ask goes out with an id, the answer comes back as a control line. */
    confirm: (question, ctx = {}) => new Promise((resolve) => {
      const id = `c${++confirmSeq}`;
      pending.set(id, resolve);
      writeLine(Deno.stderr, { id, ts: Date.now(), source: "sand", kind: "ask", name: "sand.confirm", traceId, attrs: { id, at: ctx.at ?? "", wait: ctx.wait ?? "" }, input: question });
    }),
  });

  const reader = (async () => {
    for await (const line of lines(Deno.stdin.readable)) {
      let msg;
      try { msg = JSON.parse(line); } catch { msg = line; }
      if (pre === null) {
        pre = msg && typeof msg === "object" && !Array.isArray(msg) ? msg : {};
        if (pre.config) configure(pre.config);
        if (pre.limits) configure({ limits: pre.limits });
        if (pre.memoryDir) configure({ models: { "local-memory": { kind: "memory", url: `file:${pre.memoryDir}` } }, default: { memory: "local-memory" } });
        writeLine(Deno.stdout, { sand: "1", ready: true });
        continue;
      }
      if (msg && typeof msg === "object" && msg.confirm) {
        const key = msg.confirm.id ?? pending.keys().next().value; // no id: the question being asked
        if (key != null && pending.has(key)) { const resolve = pending.get(key); pending.delete(key); resolve(msg.confirm.answer); }
        continue;
      }
      queue.push(typeof msg === "string" ? { input: msg } : msg && typeof msg === "object" ? msg : { input: line });
      wakeUp();
    }
    closed = true;
    for (const resolve of pending.values()) resolve(undefined); // the pipe closed: nobody is there
    pending.clear();
    wakeUp();
  })();

  while (true) {
    if (!queue.length) {
      if (closed) break;
      await new Promise((r) => { wake = r; });
      continue;
    }
    const job = queue.shift();
    traceId = job.traceId ?? pre?.traceId ?? null;
    const wanted = job.session !== undefined;
    try {
      const out = await run(main, { input: String(job.input ?? ""), session: job.session, bus: stderrBus(), file: pre?.file ?? opts.file ?? null, traceId: traceId ?? undefined });
      const held = Object.keys(out.session?.state?.chats ?? {}).length;
      writeLine(Deno.stdout, { result: out.result === undefined ? null : out.result, ...(wanted || held ? { session: out.session } : {}) });
    } catch (e) {
      failures++;
      writeLine(Deno.stdout, { error: { code: e?.code ?? "E-RUNTIME", message: String(e?.message ?? e), at: e?.at ?? null }, ...(e?.session ? { session: e.session } : {}) });
    }
  }
  await reader;
  Deno.exit(failures ? 3 : 0);
}

export async function denoMain(main, opts = {}) {
  if (envGet("SAND_CHILD") === "1") return childMain(main, opts);
  setPorts({ env: envGet, readFile, writeFile, log: (l) => console.error(l) });
  const input = Deno.args.length ? Deno.args.join(" ") : await new Response(Deno.stdin.readable).text();
  try {
    const { result, trace } = await run(main, { input: input.replace(/\n$/, ""), file: opts.file ?? null });
    console.log(JSON.stringify(result === undefined ? null : result));
    console.error(summary(trace));
    return { result, trace };
  } catch (e) {
    console.error(String(e?.message ?? e));
    Deno.exit(exitCodeOf(e));
  }
}
export const exitCodeOf = (e) => ({ "E-BUDGET": 3, "E-AUTH": 4, "E-PASSES": 4, "E-SUSPENDED": 4, "E-TIMEOUT": 5 })[e?.code] ?? 1;
