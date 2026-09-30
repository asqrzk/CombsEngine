/**
 * sand run — build, fence, spawn, speak JSON lines.
 *
 * The child is `deno run --no-prompt` with exactly the permissions the config
 * implies: `--allow-net=<the model hosts>`, `--allow-env=<the variables the
 * config reads>`, nothing else. Secrets never touch argv: the child reads them
 * from the environment it inherits. Stdout carries one output per input;
 * stderr carries the trace as JSON lines, which the parent tees to `--trace`
 * and prints as decision lines. `--workers N` runs N children over the inputs
 * and keeps the outputs in index order.
 */

import { build } from "./build.ts";
import { loadLayers, resolveConfig, mergeConfig, zeroConfig, traceLines, ROADS } from "../mod.ts";

const envGet = (n: string) => { try { return Deno.env.get(n); } catch { return undefined; } };
const readFile = async (p: string) => { try { return await Deno.readTextFile(p); } catch { return null; } };
const enc = new TextEncoder();

export interface RunFlags { input?: string; stdin?: boolean; jsonl?: string; workers?: number; trace?: string; model?: string; road?: string; timeout?: number; allow?: string[]; quiet?: boolean }

/** The environment variable names a config reads (`$NAME` values), so the fence can be exact. */
function envNamesOf(cfg: unknown, out = new Set<string>()): Set<string> {
  if (typeof cfg === "string") { const m = /^\$([A-Za-z_][A-Za-z0-9_]*)$/.exec(cfg); if (m) out.add(m[1]); }
  else if (Array.isArray(cfg)) cfg.forEach((v) => envNamesOf(v, out));
  else if (cfg && typeof cfg === "object") Object.values(cfg).forEach((v) => envNamesOf(v, out));
  return out;
}

export async function runCommand(file: string, flags: RunFlags): Promise<number> {
  const env = envGet;
  const cwd = file.includes("/") ? file.replace(/\/[^/]*$/, "") : ".";
  const layers = await loadLayers({ env, readFile, cwd: cwd.startsWith("/") ? cwd : `${Deno.cwd()}/${cwd}`, home: env("HOME") ?? null });
  const explicit: any = {};
  if (flags.model) explicit.default = { decide: flags.model };
  // the effective, still-unresolved config goes to the child, which resolves $ENV itself
  const unresolved = mergeConfig({ models: {}, default: {}, settle: {}, limits: {} }, zeroConfig(env), ...layers, explicit);
  unresolved.models ??= {};
  if (flags.road) {
    const name = unresolved.default?.decide ?? (unresolved.models.combs ? "combs" : Object.keys(unresolved.models)[0]);
    if (!name) { console.error("--road names a road but no decision model is configured; set COMBS_API_KEY or write sand.json"); return 2; }
    if (!ROADS[flags.road]) { console.error(`--road ${flags.road} names no road; SAND knows ${Object.keys(ROADS).join(", ")}`); return 2; }
    unresolved.models[name] = { ...unresolved.models[name], road: flags.road };
  }
  let resolved;
  try { resolved = resolveConfig({ env, layers, explicit: { ...explicit, models: flags.road ? unresolved.models : undefined } }); }
  catch (e) { console.error(String((e as Error).message)); return 2; }
  if (!resolved.default.decide) { console.error("E-NO-MODEL · no decision model: set COMBS_API_KEY or write sand.json"); return 2; }

  const built = await build(file, {});
  const hosts = resolved.net;
  const names = new Set([...envNamesOf(unresolved), "SAND_CHILD"]);
  for (const n of ["COMBS_API_KEY", "COMBS_API_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]) if (env(n)) names.add(n);
  const permissions = [`--allow-net=${hosts.join(",")}`, `--allow-env=${[...names].join(",")}`, ...(flags.allow ?? [])];

  // inputs
  const inputs: string[] = [];
  if (flags.input !== undefined) inputs.push(flags.input);
  else if (flags.jsonl) for (const line of (await Deno.readTextFile(flags.jsonl)).split("\n")) { if (!line.trim()) continue; try { const j = JSON.parse(line); inputs.push(typeof j === "string" ? j : j?.input ?? line); } catch { inputs.push(line); } }
  else inputs.push((await new Response(Deno.stdin.readable).text()).replace(/\n$/, ""));
  if (!inputs.length) { console.error("nothing to run: give --input, --stdin or --jsonl"); return 2; }

  const traceFile = flags.trace ? await Deno.open(flags.trace, { write: true, create: true, truncate: true }) : null;
  const workers = Math.max(1, Math.min(flags.workers ?? 1, inputs.length));
  const outputs: (string | null)[] = new Array(inputs.length).fill(null);
  const totals = { passes: 0, tokens: 0, requests: 0, events: 0, failures: 0, ms: Date.now() };
  const preamble = JSON.stringify({ sand: "0", config: { ...unresolved, models: Object.fromEntries(Object.entries(unresolved.models ?? {}).map(([k, v]: [string, any]) => [k, { ...v, implicit: undefined }])) }, limits: resolved.limits });

  const childFor = async (indices: number[]) => {
    const cmd = new Deno.Command("deno", {
      args: ["run", "--no-prompt", ...permissions, "--config", `${built.dir}/deno.json`, built.ts],
      env: { SAND_CHILD: "1" }, stdin: "piped", stdout: "piped", stderr: "piped",
    });
    const child = cmd.spawn();
    const writer = child.stdin.getWriter();
    await writer.write(enc.encode(preamble + "\n"));
    for (const i of indices) await writer.write(enc.encode(JSON.stringify(inputs[i]) + "\n"));
    await writer.close();
    const timer = setTimeout(() => { try { child.kill("SIGTERM"); } catch { /* gone */ } setTimeout(() => { try { child.kill("SIGKILL"); } catch { /* gone */ } }, 5000); }, flags.timeout ?? 120_000);
    // stdout: one JSON per input, in the order sent
    const readLines = async (stream: ReadableStream<Uint8Array>, onLine: (l: string) => void) => {
      const dec = new TextDecoder(); let buf = "";
      for await (const chunk of stream) { buf += dec.decode(chunk, { stream: true }); let k; while ((k = buf.indexOf("\n")) >= 0) { const l = buf.slice(0, k); buf = buf.slice(k + 1); if (l.trim()) onLine(l); } }
      if (buf.trim()) onLine(buf);
    };
    let cursor = 0;
    const outDone = readLines(child.stdout, (l) => { const i = indices[cursor++]; if (i !== undefined) outputs[i] = l; });
    const errDone = readLines(child.stderr, (l) => {
      let ev: any = null; try { ev = JSON.parse(l); } catch { /* a plain line */ }
      if (!ev || !ev.name) { if (!flags.quiet) console.error(l); return; }
      totals.events++;
      if (traceFile) traceFile.writeSync(enc.encode(l + "\n"));
      if (ev.name === "sand.request" && ev.attrs?.status === "ok") { totals.requests++; totals.passes += Number(ev.attrs.passes ?? 0); totals.tokens += Number(ev.attrs.tokens ?? 0); }
      if (!flags.quiet && (ev.name === "sand.decision" || ev.name === "sand.effect" || ev.name === "sand.budget" || (ev.name === "sand.log"))) for (const line of traceLines([ev])) console.error(line.replace(/^\S+\s+/, ""));
    });
    const status = await child.status;
    clearTimeout(timer);
    await Promise.all([outDone, errDone]);
    return status;
  };
  const groups: number[][] = Array.from({ length: workers }, () => []);
  inputs.forEach((_, i) => groups[i % workers].push(i));
  const statuses = await Promise.all(groups.filter((g) => g.length).map(childFor));
  traceFile?.close();
  let code = 0;
  for (const s of statuses) if (!s.success) code = Math.max(code, s.code || 1);
  for (const o of outputs) {
    if (o == null) { totals.failures++; continue; }
    try { const j = JSON.parse(o); if (j && typeof j === "object" && j.error) { totals.failures++; console.error(`${j.error.code} ${j.error.at ?? ""} · ${j.error.message}`); } } catch { /* raw */ }
    console.log(o ?? "null");
  }
  if (!flags.quiet) console.error(`run   ${inputs.length} input${inputs.length === 1 ? "" : "s"} · ${totals.requests} request${totals.requests === 1 ? "" : "s"} · ${totals.passes} pass${totals.passes === 1 ? "" : "es"} · ${totals.tokens} tokens · ${Date.now() - totals.ms} ms${flags.trace ? ` · trace ${flags.trace} (${totals.events} events)` : ""}${totals.failures ? ` · ${totals.failures} failed` : ""}`);
  return code || (totals.failures ? 3 : 0);
}
