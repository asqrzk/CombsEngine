/**
 * sand run — build, fence, spawn, speak the child protocol.
 *
 * The fence is built from the plan and the config and from nothing else:
 * `--allow-net=` the model hosts plus every host a tool's `net` declares,
 * `--allow-env=` the variables the config reads and the names its `env` list adds, `--allow-read/write=` the
 * memory directory when there is one. If a flag is refused, `deno` is missing,
 * or the child dies before it acknowledges the preamble, the run ends `E-FENCE`
 * and exit 2 — **it is never retried wider**. Secrets never touch argv: the
 * child reads them from the environment it inherits.
 *
 * Stdout carries one result per input; stderr carries the trace as JSON lines,
 * which the parent tees to `--trace`, prints as decision lines, and reads for
 * two more things: a piece of `emit` output is printed as it arrives, and a
 * `confirm` is answered — from `--yes/--no/--unsure`, or by asking the person
 * at the terminal, with the kill clock stopped for as long as they take.
 */

import { build } from "./build.ts";
import { loadLayers, resolveConfig, mergeConfig, zeroConfig, traceLines, ROADS } from "../mod.ts";

const envGet = (n: string) => { try { return Deno.env.get(n); } catch { return undefined; } };
const readFile = async (p: string) => { try { return await Deno.readTextFile(p); } catch { return null; } };
const enc = new TextEncoder();
const dec = new TextDecoder();

export interface RunFlags {
  input?: string; stdin?: boolean; jsonl?: string; workers?: number; trace?: string; model?: string; road?: string;
  timeout?: number; allow?: string[]; quiet?: boolean; session?: string; memory?: string;
  yes?: boolean; no?: boolean; unsure?: boolean;
}

/** The environment variable names a config reads (`$NAME` values), so the fence can be exact. */
function envNamesOf(cfg: unknown, out = new Set<string>()): Set<string> {
  if (typeof cfg === "string") { const m = /^\$([A-Za-z_][A-Za-z0-9_]*)$/.exec(cfg); if (m) out.add(m[1]); }
  else if (Array.isArray(cfg)) cfg.forEach((v) => envNamesOf(v, out));
  else if (cfg && typeof cfg === "object") Object.values(cfg).forEach((v) => envNamesOf(v, out));
  return out;
}

/** The directory a `file:` memory store keeps its JSON in. */
const dirOfFileUrl = (url: string) => url.replace(/^file:(\/\/)?/, "").replace(/\/+$/, "") || ".";

/** Reads one line from the terminal without touching stdout, which belongs to the results. */
function askAtTerminal(question: string): string | null {
  Deno.stderr.writeSync(enc.encode(`\n${question} [y/n/?] `));
  const buf = new Uint8Array(4096);
  const n = Deno.stdin.readSync(buf);
  if (n === null) return null;
  return dec.decode(buf.subarray(0, n)).trim();
}

export async function runCommand(file: string, flags: RunFlags): Promise<number> {
  const env = envGet;
  const cwd = file.includes("/") ? file.replace(/\/[^/]*$/, "") : ".";
  const here = cwd.startsWith("/") ? cwd : `${Deno.cwd()}/${cwd}`;
  const layers = await loadLayers({ env, readFile, cwd: here, home: env("HOME") ?? null });
  const explicit: Record<string, unknown> = {};
  if (flags.model) explicit.default = { decide: flags.model };
  // the effective, still-unresolved config goes to the child, which resolves $ENV itself
  const unresolved = mergeConfig({ models: {}, default: {}, settle: {}, limits: {} }, zeroConfig(env), ...layers, explicit) as
    { models: Record<string, Record<string, unknown>>; default?: { decide?: string } };
  unresolved.models ??= {};
  if (flags.road) {
    const name = unresolved.default?.decide ?? (unresolved.models.combs ? "combs" : Object.keys(unresolved.models)[0]);
    if (!name) { console.error("--road names a setting but no decision model is configured; set COMBS_API_KEY or write sand.json"); return 2; }
    if (!ROADS[flags.road]) { console.error(`--road ${flags.road} names none SAND knows; it knows ${Object.keys(ROADS).join(", ")}`); return 2; }
    unresolved.models[name] = { ...unresolved.models[name], road: flags.road };
  }
  let resolved;
  try { resolved = resolveConfig({ env, layers, explicit: { ...explicit, models: flags.road ? unresolved.models : undefined } }); }
  catch (e) { console.error(String((e as Error).message)); return 2; }
  if (!resolved.default.decide) { console.error("E-NO-MODEL · no decision model: set COMBS_API_KEY or write sand.json"); return 2; }

  const built = await build(file, {});
  const plan = built.result.plan;

  // ── the fence, built from the plan and the config, and from nothing else ──
  const hosts = [...new Set([...resolved.net, ...plan.hosts])].filter(Boolean);
  const names = new Set([...envNamesOf(unresolved), ...((unresolved as { env?: string[] }).env ?? []), "SAND_CHILD"]);
  for (const n of ["COMBS_API_KEY", "COMBS_API_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]) if (env(n)) names.add(n);
  const memoryDirs = [...new Set([
    ...(flags.memory ? [flags.memory] : []),
    ...Object.values(resolved.models).filter((m) => m.kind === "memory" && String(m.url ?? "").startsWith("file:")).map((m) => dirOfFileUrl(String(m.url))),
  ])];
  const permissions = [
    ...(hosts.length ? [`--allow-net=${hosts.join(",")}`] : []),
    `--allow-env=${[...names].join(",")}`,
    ...(memoryDirs.length ? [`--allow-read=${memoryDirs.join(",")}`, `--allow-write=${memoryDirs.join(",")}`] : []),
    ...(flags.allow ?? []),
  ];

  // ── inputs ───────────────────────────────────────────────────────────────
  const inputs: string[] = [];
  if (flags.input !== undefined) inputs.push(flags.input);
  else if (flags.jsonl) for (const line of (await Deno.readTextFile(flags.jsonl)).split("\n")) { if (!line.trim()) continue; try { const j = JSON.parse(line); inputs.push(typeof j === "string" ? j : j?.input ?? line); } catch { inputs.push(line); } }
  else inputs.push((await new Response(Deno.stdin.readable).text()).replace(/\n$/, ""));
  if (!inputs.length) { console.error("nothing to run: give --input, --stdin or --jsonl"); return 2; }
  const stdinIsInput = flags.input === undefined && !flags.jsonl;

  // ── the session, kept across the inputs of this run and on disk by id ─────
  const sessionFile = flags.session ? `${here}/.sand-sessions/${flags.session.replace(/[^\w.@+-]/g, "_")}.json` : null;
  let session: Record<string, unknown> | null = null;
  if (sessionFile) {
    const held = await readFile(sessionFile);
    try { session = held ? JSON.parse(held) : { id: flags.session }; } catch { session = { id: flags.session }; }
  }

  // ── how a person answers, if one is there at all ─────────────────────────
  const workers = Math.max(1, Math.min(flags.session ? 1 : flags.workers ?? 1, inputs.length));
  const flagAnswer = flags.yes ? true : flags.no ? false : flags.unsure ? null : undefined;
  const canAsk = flagAnswer === undefined && workers === 1 && !stdinIsInput && Deno.stdin.isTerminal();
  let saidNobody = false, wrotePieces = false;

  const traceFile = flags.trace ? await Deno.open(flags.trace, { write: true, create: true, truncate: true }) : null;
  const results: (string | null)[] = new Array(inputs.length).fill(null);
  const totals = { passes: 0, tokens: 0, turns: 0, requests: 0, events: 0, failures: 0, fenced: false, ms: Date.now() };
  const preamble = JSON.stringify({
    sand: "1",
    config: { ...unresolved, models: Object.fromEntries(Object.entries(unresolved.models ?? {}).map(([k, v]) => [k, { ...v, implicit: undefined }])) },
    limits: resolved.limits,
    file: file.replace(/^.*\//, ""),
    ...(flags.memory ? { memoryDir: flags.memory } : {}),
  });

  const readLines = async (stream: ReadableStream<Uint8Array>, onLine: (l: string) => void) => {
    let buf = "";
    for await (const chunk of stream) { buf += dec.decode(chunk, { stream: true }); let k; while ((k = buf.indexOf("\n")) >= 0) { const l = buf.slice(0, k); buf = buf.slice(k + 1); if (l.trim()) onLine(l); } }
    if (buf.trim()) onLine(buf);
  };

  const fence = (why: string) => {
    if (!totals.fenced) console.error(`E-FENCE · ${why}. The run did not happen; nothing was charged.`);
    totals.fenced = true;
    return null;
  };

  const childFor = async (indices: number[]): Promise<Deno.CommandStatus | null> => {
    let child: Deno.ChildProcess;
    try {
      child = new Deno.Command("deno", {
        args: ["run", "--no-prompt", ...permissions, "--config", `${built.dir}/deno.json`, built.ts],
        env: { SAND_CHILD: "1" }, stdin: "piped", stdout: "piped", stderr: "piped",
      }).spawn();
    } catch (e) { return fence(`deno could not be started (${(e as Error).message})`); }

    const writer = child.stdin.getWriter();
    let closedIn = false;
    const closeIn = async () => { if (closedIn) return; closedIn = true; try { await writer.close(); } catch { /* the child is gone */ } };
    const tell = async (o: unknown) => { try { await writer.write(enc.encode(JSON.stringify(o) + "\n")); } catch { /* the child is gone */ } };

    // the kill clock, stopped for as long as a person is being asked
    let timer: ReturnType<typeof setTimeout> | null = null;
    const arm = () => { timer = setTimeout(() => { try { child.kill("SIGTERM"); } catch { /* gone */ } setTimeout(() => { try { child.kill("SIGKILL"); } catch { /* gone */ } }, 5000); }, flags.timeout ?? 120_000); };
    const hold = () => { if (timer != null) { clearTimeout(timer); timer = null; } };
    arm();

    let resolveReady: (ok: boolean) => void = () => {};
    const ready = new Promise<boolean>((r) => { resolveReady = r; });
    let sawReady = false, cursor = 0, got = 0;

    const outDone = readLines(child.stdout, (l) => {
      let j: Record<string, unknown> | null = null;
      try { j = JSON.parse(l); } catch { /* not the protocol */ }
      if (!sawReady) {
        sawReady = true;
        resolveReady(!!(j && j.sand === "1" && j.ready));
        return;
      }
      const i = indices[cursor++];
      got++;
      if (j && typeof j === "object") {
        if (j.session) session = j.session as Record<string, unknown>;
        if (j.error) {
          totals.failures++;
          const err = j.error as { code?: string; message?: string };
          console.error(String(err.message ?? "the run failed"));
          if (i !== undefined) results[i] = JSON.stringify({ error: j.error });
        } else if (i !== undefined) results[i] = JSON.stringify(j.result ?? null);
      } else if (i !== undefined) results[i] = l;
      if (got >= indices.length) closeIn();
    });

    const errDone = readLines(child.stderr, (l) => {
      let ev: Record<string, any> | null = null;
      try { ev = JSON.parse(l); } catch { /* a plain line */ }
      if (!ev || !ev.name) { if (!flags.quiet) console.error(l); return; }
      totals.events++;
      if (traceFile) traceFile.writeSync(enc.encode(l + "\n"));
      if (ev.name === "sand.request" && ev.attrs?.status === "ok") { totals.requests++; totals.passes += Number(ev.attrs.passes ?? 0); totals.tokens += Number(ev.attrs.tokens ?? 0); }
      if (ev.name === "sand.turn") totals.turns++;
      if (ev.name === "sand.output" && !flags.quiet && typeof ev.output === "string") { wrotePieces = true; Deno.stderr.writeSync(enc.encode(ev.output)); return; }
      if (ev.name === "sand.confirm" && ev.kind === "ask") {
        const question = typeof ev.input === "string" ? ev.input : typeof ev.input?.question === "string" ? ev.input.question : "may I?";
        if (canAsk) { hold(); const said = askAtTerminal(question); arm(); tell({ confirm: { id: ev.attrs?.id, answer: { answer: said, by: "tty" } } }); return; }
        if (flagAnswer !== undefined) { tell({ confirm: { id: ev.attrs?.id, answer: { answer: flagAnswer, by: "tty" } } }); return; }
        if (!saidNobody) { saidNobody = true; console.error(`W-NOBODY · nobody is here to answer "${question.slice(0, 80)}"; it is unsure. Answer with --yes, --no or --unsure, or run one input from a terminal.`); }
        tell({ confirm: { id: ev.attrs?.id, answer: { answer: null, by: "nobody" } } });
        return;
      }
      if (!flags.quiet && (ev.name === "sand.decision" || ev.name === "sand.effect" || ev.name === "sand.budget" || ev.name === "sand.log")) for (const line of traceLines([ev as unknown as Parameters<typeof traceLines>[0][number]])) console.error(line.replace(/^\S+\s+/, ""));
    });

    // a child that died on a refused flag breaks this pipe; the ready race says so
    try { await writer.write(enc.encode(preamble + "\n")); } catch { /* it is already gone */ }
    const status = child.status.then((s) => { resolveReady(false); return s; });
    if (!await ready) {
      hold();
      await closeIn();
      await Promise.all([outDone, errDone]);
      await status;
      return fence("the child refused the fence it was given, or could not start under it");
    }
    for (const i of indices) await tell({ input: inputs[i], ...(session ? { session } : {}) });
    const done = await status;
    hold();
    await closeIn();
    await Promise.all([outDone, errDone]);
    return done;
  };

  const groups: number[][] = Array.from({ length: workers }, () => []);
  inputs.forEach((_, i) => groups[i % workers].push(i));
  const statuses = await Promise.all(groups.filter((g) => g.length).map(childFor));
  traceFile?.close();
  if (totals.fenced) return 2;

  if (wrotePieces) Deno.stderr.writeSync(enc.encode("\n"));
  let code = 0;
  for (const s of statuses) if (s && !s.success) code = Math.max(code, s.code || 1);
  for (const r of results) {
    if (r == null) { totals.failures++; continue; }
    console.log(r);
  }
  if (sessionFile && session) {
    try {
      await Deno.mkdir(sessionFile.replace(/\/[^/]*$/, ""), { recursive: true });
      await Deno.writeTextFile(sessionFile, JSON.stringify(session, null, 2) + "\n");
    } catch (e) { console.error(`the session could not be kept in ${sessionFile}: ${(e as Error).message}`); }
  }
  if (!flags.quiet) {
    console.error(`\nrun   ${inputs.length} input${inputs.length === 1 ? "" : "s"} · ${totals.requests} request${totals.requests === 1 ? "" : "s"} · ${totals.passes} pass${totals.passes === 1 ? "" : "es"}${totals.turns ? ` · ${totals.turns} turn${totals.turns === 1 ? "" : "s"}` : ""} · ${totals.tokens} tokens · ${Date.now() - totals.ms} ms${flags.session ? ` · session ${flags.session}` : ""}${flags.trace ? ` · trace ${flags.trace} (${totals.events} events)` : ""}${totals.failures ? ` · ${totals.failures} failed` : ""}`);
  }
  return code || (totals.failures ? 3 : 0);
}
