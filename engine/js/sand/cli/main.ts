#!/usr/bin/env -S deno run -A
/**
 * sand — the command line: check · build · run · log.
 *
 *   sand check <file.sand> [--strict]
 *   sand build <file.sand> [-o dir] [--js]
 *   sand run   <file.sand> (--input "…" | --stdin | --jsonl in.jsonl) [--workers N] [--trace out.ndjson]
 *                          [--model name] [--road seat] [--timeout ms] [--allow <deno flag>] [--quiet]
 *   sand log   <trace.ndjson>
 *
 * Install: deno install -g -A -n sand jsr:@combs/sand/cli
 */

import { build, importTarget } from "./build.ts";
import { runCommand } from "./run.ts";
import { logCommand } from "./log.ts";
import { loadLayers, resolveConfig, PROLOGUE_LINES, ROADS } from "../mod.ts";

const USAGE = `sand — semantic abstraction of neurolinguistic determinism

  sand check <file.sand> [--strict]
  sand build <file.sand> [-o dir] [--js]
  sand run   <file.sand> (--input "…" | --stdin | --jsonl in.jsonl) [--workers N] [--trace out.ndjson]
                         [--model name] [--road seat] [--timeout ms] [--allow <deno flag>] [--quiet]
  sand log   <trace.ndjson>

Config: the nearest sand.json upward, $SAND_CONFIG, ~/.config/combs/sand.json, or none at all —
COMBS_API_KEY alone gives a decision model, OPENAI_API_KEY alone a chat model.`;

function parse(argv: string[]) {
  const flags: Record<string, string | boolean | string[]> = {}; const pos: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "-o") { flags.out = argv[++i]; continue; }
    if (a.startsWith("--")) {
      const [k, v] = a.slice(2).split("=", 2);
      if (v !== undefined) { flags[k] = v; continue; }
      const next = argv[i + 1];
      if (["strict", "js", "stdin", "quiet"].includes(k) || next === undefined || next.startsWith("--")) { flags[k] = true; continue; }
      if (k === "allow") { (flags.allow as string[] | undefined) ? (flags.allow as string[]).push(next) : (flags.allow = [next]); i++; continue; }
      flags[k] = next; i++; continue;
    }
    pos.push(a);
  }
  return { flags, pos };
}

const envGet = (n: string) => { try { return Deno.env.get(n); } catch { return undefined; } };
const readFile = async (p: string) => { try { return await Deno.readTextFile(p); } catch { return null; } };

async function check(file: string, strict: boolean): Promise<number> {
  let built;
  try { built = await build(file, { outDir: await Deno.makeTempDir({ prefix: "sand-check-" }) }); }
  catch (e) { console.error(String((e as Error).message)); return 1; }
  const { plan, warnings } = built.result;
  const name = file.replace(/^.*\//, "");
  console.log(`${name} · ${plan.declarations.kinds.length + plan.declarations.scales.length + plan.declarations.shapes.length} declarations · ${plan.sites.length} predicate${plan.sites.length === 1 ? "" : "s"} · ${plan.batches.length} batch${plan.batches.length === 1 ? "" : "es"}`);
  for (const s of plan.sites) console.log(`  ${s.at.padEnd(7)} ${s.text.padEnd(52).slice(0, 52)} ${s.how.padEnd(16)} ${s.batch ? `batch ${s.batch}` : ""}${s.loops.length ? ` (in ${s.loops.join("/")})` : ""}`);
  for (const c of plan.calls) console.log(`  ${c.at.padEnd(7)} ${(c.verb + "(" + (c.binding ? "{{" + c.binding + "}}" : "") + (c.spec ? ", " + c.spec : "") + ")").padEnd(52).slice(0, 52)} ${c.speculative?.length ? "speculates " + c.speculative.join(", ") : c.verb}${c.loops.length ? ` (in ${c.loops.join("/")})` : ""}`);
  // the road check needs the config
  let roadNote = "";
  try {
    const cwd = file.includes("/") ? file.replace(/\/[^/]*$/, "") : ".";
    const layers = await loadLayers({ env: envGet, readFile, cwd: cwd.startsWith("/") ? cwd : `${Deno.cwd()}/${cwd}`, home: envGet("HOME") ?? null });
    const cfg = resolveConfig({ env: envGet, layers });
    const d = cfg.default.decide ? cfg.models[cfg.default.decide] : null;
    const road = d?.road ?? "seat";
    roadNote = ` · decide: ${cfg.default.decide ?? "none"}${d ? ` (${d.kind}${d.kind === "decision" ? " · " + road : ""})` : ""} · ask: ${cfg.default.ask ?? "none"}`;
    if (d && (road === "modernbert" || road === "deberta") && plan.sites.some((s) => s.how === "noul")) warnings.push({ code: "W-ROAD", at: plan.sites.find((s) => s.how === "noul")!.at, message: `the default decide model runs on ${road}, which cannot answer a prose predicate; use a seat road for this program` });
  } catch (e) { roadNote = ` · config: ${(e as Error).message}`; }
  const zs = plan.sites.some((s) => s.how === "noul") ? "n/a (a noul is asked)" : `${plan.batches.reduce((n, b) => n + b.questions, 0)}+ per option`;
  console.log(`plan    ${plan.batches.length} batch${plan.batches.length === 1 ? "" : "es"} · cost  ${plan.cost.text} · zero-shot: ${zs}${roadNote}`);
  for (const w of warnings) console.log(`${w.code.padEnd(19)}${w.at} · ${w.message}`);
  // types
  const proc = new Deno.Command("deno", { args: ["check", "--config", `${built.dir}/deno.json`, built.ts], stdout: "piped", stderr: "piped" });
  const out = await proc.output();
  const text = new TextDecoder().decode(out.stderr) + new TextDecoder().decode(out.stdout);
  if (out.success) console.log(`types   ok (deno check ${built.name}.ts)`);
  else {
    const mapped = text.replace(new RegExp(built.ts.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ":(\\d+):(\\d+)", "g"), (_, l, c) => `${name}:${Math.max(1, Number(l) - PROLOGUE_LINES)}:${c}`);
    console.log(mapped.replace(/\x1b\[[0-9;]*m/g, "").trim());
    return 1;
  }
  const bad = warnings.filter((w) => w.code.startsWith("W-"));
  return strict && bad.length ? 1 : 0;
}

if (import.meta.main) {
  const { flags, pos } = parse(Deno.args);
  const [cmd, file] = pos;
  let code = 2;
  try {
    if (cmd === "check" && file) code = await check(file, !!flags.strict);
    else if (cmd === "build" && file) {
      const b = await build(file, { outDir: (flags.out as string) ?? `${file.replace(/\.sand$/, "")}.build`, js: !!flags.js });
      console.log([b.ts, b.plan, b.js].filter(Boolean).map((p) => `${p}`).join("\n"));
      console.log(`imports @combs/sand from ${importTarget()}`);
      code = 0;
    }
    else if (cmd === "run" && file) code = await runCommand(file, { input: flags.input as string | undefined, stdin: !!flags.stdin, jsonl: flags.jsonl as string | undefined, workers: flags.workers ? Number(flags.workers) : undefined, trace: flags.trace as string | undefined, model: flags.model as string | undefined, road: flags.road as string | undefined, timeout: flags.timeout ? Number(flags.timeout) : undefined, allow: flags.allow as string[] | undefined, quiet: !!flags.quiet });
    else if (cmd === "log" && file) code = await logCommand(file);
    else { console.log(USAGE); code = cmd ? 2 : 0; }
  } catch (e) {
    console.error(String((e as Error)?.message ?? e));
    code = 1;
  }
  Deno.exit(code);
}
