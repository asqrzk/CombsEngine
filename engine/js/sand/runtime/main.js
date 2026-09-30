/**
 * @combs/sand — the Deno entry behind `if (import.meta.main) run(main)`.
 *
 * Two modes. Under `sand run` (SAND_CHILD=1) the child speaks JSON lines: one
 * preamble `{sand, config, limits}` on stdin, then one input per line; one
 * output JSON per input on stdout; every trace event as one JSON line on
 * stderr. Run directly (`deno run program.ts "text"`), it takes the input from
 * the arguments or stdin, prints the result as JSON and a one-line summary.
 *
 * The only file in the runtime that reads Deno.* (besides the worker).
 */

import { run, configure, setPorts } from "./verbs.js";

const encoder = new TextEncoder();
const envGet = (n) => { try { return Deno.env.get(n); } catch { return undefined; } };
const readFile = async (p) => { try { return await Deno.readTextFile(p); } catch { return null; } };

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

function stderrBus() {
  const events = [];
  return { events, publish(partial) { const ev = { id: `e${events.length + 1}`, ts: Date.now(), ...partial }; events.push(ev); Deno.stderr.writeSync(encoder.encode(JSON.stringify(ev) + "\n")); return ev; } };
}

const summary = (trace) => {
  const b = trace.budget;
  const decisions = trace.events.filter((e) => e.name === "sand.decision").map((e) => `${e.attrs?.at ?? ""} ${e.attrs?.top ?? ""}${e.attrs?.p_top ? " p=" + e.attrs.p_top : ""}`).join(" · ");
  return `${b.spent.passes} pass${b.spent.passes === 1 ? "" : "es"} · ${b.spent.tokens} tokens · ${b.ms} ms${decisions ? " · " + decisions : ""}`;
};

export async function denoMain(main, opts = {}) {
  const child = envGet("SAND_CHILD") === "1";
  setPorts({ env: envGet, readFile, log: child ? () => {} : (l) => console.error(l) });
  if (child) {
    let first = true, failures = 0;
    for await (const line of lines(Deno.stdin.readable)) {
      if (first) {
        first = false;
        const pre = JSON.parse(line);
        if (pre.config) configure(pre.config);
        if (pre.limits) configure({ limits: pre.limits });
        continue;
      }
      let input = line;
      try { const j = JSON.parse(line); input = typeof j === "string" ? j : j?.input ?? line; } catch { /* a bare line is the input */ }
      try {
        const { result } = await run(main, { input, bus: stderrBus(), file: pre_file(opts) });
        Deno.stdout.writeSync(encoder.encode(JSON.stringify(result === undefined ? null : result) + "\n"));
      } catch (e) {
        failures++;
        Deno.stdout.writeSync(encoder.encode(JSON.stringify({ error: { code: e?.code ?? "E-RUNTIME", message: String(e?.message ?? e), at: e?.at ?? null } }) + "\n"));
      }
    }
    Deno.exit(failures ? 3 : 0);
  }
  const input = Deno.args.length ? Deno.args.join(" ") : await new Response(Deno.stdin.readable).text();
  try {
    const { result, trace } = await run(main, { input: input.replace(/\n$/, ""), file: pre_file(opts) });
    console.log(JSON.stringify(result === undefined ? null : result));
    console.error(summary(trace));
    return { result, trace };
  } catch (e) {
    console.error(String(e?.message ?? e));
    Deno.exit(exitCodeOf(e));
  }
}
const pre_file = (opts) => opts.file ?? null;
export const exitCodeOf = (e) => ({ "E-BUDGET": 3, "E-AUTH": 4, "E-PASSES": 4, "E-SUSPENDED": 4, "E-TIMEOUT": 5 })[e?.code] ?? 1;
