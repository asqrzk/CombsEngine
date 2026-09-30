/**
 * @combs/sand — the verbs.
 *
 * What a program calls: the declarations (`kind`, `scale`, `shape`, `text`,
 * `settle`), the decisions (`decide`, `decideAll`, `classify`, `extract`), the
 * chat verb (`ask`), the helpers (`parallel`, `race`, `timeout`, `retry`,
 * `using`, `named`, `log`) and the entry (`run`, `trace`). A run owns one
 * context: its config, budget, batcher, memo and trace; the verbs read it.
 */

import { SandError } from "./errors.js";
import { unsure, verdict } from "./kleene.js";
import { Budget } from "./budget.js";
import { Batcher } from "./batcher.js";
import { emitter } from "./trace.js";
import { resolveConfig, mergeConfig, kindRegistry, registerKind } from "./config.js";
import { questionOf, normalizeSpec, idsOf, shapeSchema } from "./wires/schema.js";
import { decisionAdapter } from "./wires/decision.js";
import { chatAdapter } from "./wires/chat.js";

registerKind("decision", decisionAdapter);
registerKind("chat", chatAdapter);

// ── module state ──────────────────────────────────────────────────────────
let explicitConfig = null;
let ports = { env: () => undefined, readFile: async () => null, fetchImpl: null, layers: [], log: null };
let current = null;
const pendingSettle = {};

/** Programmatic sand.json: what the playground or a host passes instead of a file. */
export function configure(cfg) { explicitConfig = mergeConfig(explicitConfig ?? {}, cfg ?? {}); }
/** How the runtime reaches its host: env, files, fetch, the log line sink, config layers. */
export function setPorts(p) { ports = { ...ports, ...p }; }
export { registerKind };

const ctxNow = (at) => { if (!current) throw new SandError("E-STATE", "no run is active; call run(main, { input })", { at }); return current; };
const pick = (o, ...ks) => Object.fromEntries(ks.filter((k) => o && o[k] !== undefined).map((k) => [k, o[k]]));

// ── declarations ──────────────────────────────────────────────────────────
export function kind(instructions, options, opts = {}) {
  if (typeof instructions !== "string" || !options || typeof options !== "object") throw new SandError("E-STATE", "kind(instructions, { id: means, … }) needs a string and an object of options");
  for (const id of Object.keys(options)) if (id === "spec" || id === "unsure") throw new SandError("E-RESERVED", `\`${id}\` cannot be an option id`);
  if (opts.fallback && !(opts.fallback in options)) throw new SandError("E-OPTION", `fallback ${JSON.stringify(opts.fallback)} names no option of ${opts.name ?? "this kind"}`);
  const spec = { type: "kind", name: opts.name ?? null, instructions, options: { ...options }, fallback: opts.fallback ?? null };
  const obj = { spec };
  for (const id of Object.keys(options)) obj[id] = id;
  return Object.freeze(obj);
}
export function scale(instructions, levels, opts = {}) {
  const list = Array.isArray(levels) ? levels.map((l) => (typeof l === "string" ? { id: l, means: l } : l)) : Object.entries(levels ?? {}).map(([id, means]) => ({ id, means }));
  if (list.length < 2 || list.length > 10) throw new SandError("E-OPTION", `${opts.name ?? "this scale"} declares ${list.length} level${list.length === 1 ? "" : "s"}; a scale takes 2 to 10`);
  for (const l of list) if (l.id === "spec" || l.id === "unsure") throw new SandError("E-RESERVED", `\`${l.id}\` cannot be a level id`);
  const spec = { type: "scale", name: opts.name ?? null, instructions, levels: list };
  const obj = { spec };
  for (const l of list) obj[l.id] = l.id;
  return Object.freeze(obj);
}
export function shape(slots, opts = {}) {
  if (!slots || typeof slots !== "object") throw new SandError("E-SHAPE", "shape({ slot: Kind | Scale | text(…), … }) needs an object of slots");
  for (const [n, s] of Object.entries(slots)) if (!s || !s.spec) throw new SandError("E-SHAPE", `slot ${JSON.stringify(n)} is not a kind, a scale, a shape or text(…)`);
  return Object.freeze({ spec: { type: "shape", name: opts.name ?? null, slots: { ...slots } } });
}
export function text(instructions) { return Object.freeze({ spec: { type: "text", instructions: String(instructions ?? "") } }); }
export function named(txt, noun) { return { text: String(txt), noun: String(noun), toString() { return this.text; } }; }
export function settle(opts = {}) {
  const thresholds = pick(opts, "floor", "gap", "confidence");
  for (const [k, v] of Object.entries(thresholds)) if (typeof v !== "number" || v < 0 || v > 1) throw new SandError("E-OPTION", `settle({ ${k}: ${v} }) is not a number in [0, 1]`);
  const target = current ? current.settle : pendingSettle;
  Object.assign(target, thresholds);
  if (current) current.budget.narrow(pick(opts, "passes", "tokens", "ms"));
  else Object.assign(pendingSettle, pick(opts, "passes", "tokens", "ms"));
}

// ── the decision ──────────────────────────────────────────────────────────
export class Decision {
  constructor(spec, meta, settle = {}) {
    const a = meta.answer;
    this.spec = spec;
    this.type = spec.type;
    const honest = a.abstain?.applied && a.abstain.from ? a.abstain.from : a.top;
    this.top = spec.type === "noul" ? String(honest) === "true" : String(honest);
    this.p = a.p_top ?? null;
    this.distribution = a.distribution ?? null;
    this.confidence = a.confidence ?? null;
    this.margin = a.margin ?? null;
    if (a.expected_level != null) this.expected_level = a.expected_level;
    this.decision = a.decision === "abstain" ? "abstain" : "act";
    this.reason = a.reason ?? (a.abstain?.applied ? a.abstain.reason ?? `abstained to ${a.abstain.option}` : null);
    if (this.decision === "act" && settle.floor != null && this.p != null && this.p < settle.floor) { this.decision = "abstain"; this.reason = `p_top ${this.p.toFixed(3)} < floor ${settle.floor}`; }
    this.value = this.decision === "abstain" ? unsure : this.top;
    this.cut = !!meta.cut;
    this.request = { ...meta.request, memo: !!meta.memo };
    this.wire = a;
  }
  is(v) { if (this.decision === "abstain") return unsure; if (this.type === "noul") return this.top === (v === "true" ? true : v === "false" ? false : !!v); return this.top === v; }
  #idx(l) { if (this.type !== "scale") throw new SandError("E-OPTION", `${this.spec.name ?? "this decision"} is not a scale; atLeast/atMost/above/below read scales`); const i = this.spec.levels.findIndex((x) => x.id === l); if (i < 0) throw new SandError("E-OPTION", `\`${l}\` names no level of \`${this.spec.name ?? "the scale"}\`, which has ${this.spec.levels.map((x) => x.id).join(", ")}`); return i; }
  #lvl() { return this.expected_level != null ? Math.round(this.expected_level) : this.spec.levels.findIndex((x) => x.id === this.top); }
  atLeast(l) { const i = this.#idx(l); return this.decision === "abstain" ? unsure : this.#lvl() >= i; }
  atMost(l) { const i = this.#idx(l); return this.decision === "abstain" ? unsure : this.#lvl() <= i; }
  above(l) { const i = this.#idx(l); return this.decision === "abstain" ? unsure : this.#lvl() > i; }
  below(l) { const i = this.#idx(l); return this.decision === "abstain" ? unsure : this.#lvl() < i; }
  toJSON() { return { value: this.value === unsure ? "unsure" : this.value, top: this.top, p: this.p, distribution: this.distribution, confidence: this.confidence, margin: this.margin, decision: this.decision, reason: this.reason, expected_level: this.expected_level, cut: this.cut, request: this.request }; }
  toString() { return this.value === unsure ? "unsure" : String(this.value); }
}

// ── the run context ───────────────────────────────────────────────────────
async function createContext(opts) {
  const env = opts.env ?? ports.env;
  const cfg = resolveConfig({ env, layers: opts.layers ?? ports.layers ?? [], explicit: mergeConfig(explicitConfig ?? {}, opts.config ?? {}) });
  const traceId = opts.traceId ?? "run-" + Math.random().toString(36).slice(2, 10);
  const em = emitter(opts.bus ?? null, traceId);
  const ctx = {
    config: cfg, traceId, bus: em.bus, emit: em.emit, signal: opts.signal ?? null,
    settle: { ...(cfg.settle ?? {}), ...pick(pendingSettle, "floor", "gap", "confidence") },
    adapters: new Map(), fetchImpl: opts.fetchImpl ?? ports.fetchImpl, usingModel: null, log: opts.log ?? ports.log,
    file: opts.file ?? null,
  };
  ctx.budget = new Budget(cfg.limits, (name, kind, attrs) => ctx.emit(name, kind, attrs));
  ctx.budget.narrow(pick(pendingSettle, "passes", "tokens", "ms"));
  ctx.batcher = new Batcher(ctx);
  ctx.adapter = (name) => {
    if (ctx.adapters.has(name)) return ctx.adapters.get(name);
    const m = cfg.models[name];
    if (!m) throw new SandError("E-MODEL", `${JSON.stringify(name)} names no model; sand.json knows ${Object.keys(cfg.models).join(", ") || "none"}`);
    const a = kindRegistry.get(m.kind)(m, { fetchImpl: ctx.fetchImpl ?? undefined, bus: ctx.bus });
    ctx.adapters.set(name, a);
    return a;
  };
  ctx.trace = () => ({ traceId, file: ctx.file, events: ctx.bus.events ?? [], budget: ctx.budget.snapshot(), models: cfg.default });
  return ctx;
}

/**
 * Runs a program's main over one input. With no options under Deno, hands over
 * to the `sand run` child protocol (runtime/main.js).
 */
export async function run(main, opts) {
  if ((opts === undefined || opts?.input === undefined) && typeof Deno !== "undefined") {
    const m = await import("./main.js");
    return m.denoMain(main, opts ?? {});
  }
  if (typeof main !== "function") throw new SandError("E-STATE", "run(main, { input }) needs the program's main function");
  const ctx = await createContext(opts ?? {});
  const prev = current; current = ctx;
  ctx.emit("sand.run", "span.start", { file: ctx.file ?? "", decide: ctx.config.default.decide ?? "", ask: ctx.config.default.ask ?? "" });
  let result, status = "ok", error = null;
  try { result = await main(opts.input); } catch (e) { status = "error"; error = e; } finally { current = prev; }
  const snap = ctx.budget.snapshot();
  ctx.emit("sand.run", "span.end", { passes: snap.spent.passes, tokens: snap.spent.tokens, requests: snap.spent.requests, ms: snap.ms, status }, { status, error: error ? String(error?.message ?? error) : undefined });
  const trace = ctx.trace();
  if (error) { try { error.trace = trace; } catch { /* frozen */ } throw error; }
  return { result, trace };
}
export function trace() { return current ? current.trace() : null; }

// ── helpers ───────────────────────────────────────────────────────────────
const stateOf = (state, at) => {
  const s = state && typeof state === "object" && "text" in state ? state.text : state;
  if (typeof s !== "string") throw new SandError("E-STATE", `the state is not text (got ${state === null ? "null" : Array.isArray(state) ? "array" : typeof state}); decide on a field: {{x.notes}}`, { at });
  if (!s.trim()) throw new SandError("E-STATE", "the state is empty; there is nothing to decide on", { at });
  return s;
};
function resolveModel(ctx, opts, klass) {
  const name = opts.model ?? ctx.usingModel ?? ctx.config.default[klass];
  if (!name || name === "none") {
    throw new SandError("E-MODEL", klass === "ask" ? "ask needs a chat model and none is configured; set OPENAI_API_KEY or name one in sand.json" : "no decision model: set COMBS_API_KEY or write sand.json", { at: opts.at });
  }
  const adapter = ctx.adapter(name);
  if (klass === "ask" && !adapter.ask) throw new SandError("E-MODEL", `ask needs a chat model; \`${name}\` is a ${adapter.kind} model`, { at: opts.at });
  return { name, adapter };
}
const specQuestions = (s, settle) => {
  const sp = s?.spec ?? s;
  if (sp?.type === "shape") return Object.values(sp.slots).map((x) => x.spec ?? x).filter((x) => x.type === "kind" || x.type === "scale").map((x) => questionOf(x, settle));
  return [questionOf(normalizeSpec(s), settle)];
};

// ── decisions ─────────────────────────────────────────────────────────────
export async function decide(state, spec, opts = {}) {
  const ctx = ctxNow(opts.at);
  const st = stateOf(state, opts.at);
  const sp = normalizeSpec(spec, opts);
  const { name, adapter } = resolveModel(ctx, opts, "decide");
  const settleNow = { ...ctx.settle, ...pick(opts, "floor", "gap", "confidence") };
  const question = questionOf(sp, settleNow);
  const speculate = (opts.speculate ?? []).flatMap((s) => specQuestions(s, settleNow));
  const meta = await ctx.batcher.enqueue({ modelName: name, adapter, state: st, question, settle: settleNow, at: opts.at ?? null, speculate });
  const d = new Decision(sp, meta, settleNow);
  ctx.emit("sand.decision", "event", {
    at: opts.at, how: sp.type === "noul" ? "noul" : `${sp.type === "kind" ? "choice" : "score"}:${sp.name ?? ""}`, reads: question.name,
    top: String(d.top), p_top: d.p != null ? d.p.toFixed(3) : undefined, decision: d.decision, reason: d.reason ?? undefined, memo: meta.memo ? "true" : undefined,
  }, { input: sp.type === "noul" ? sp.means : question.instructions, output: d.toJSON() });
  return d;
}
export function decideAll(state, specs, opts = {}) {
  if (!Array.isArray(specs)) throw new SandError("E-STATE", "decideAll(state, [specs], opts) needs an array of specs", { at: opts.at });
  return Promise.all(specs.map((s, i) => decide(state, s, { ...opts, speculate: i === 0 ? opts.speculate : undefined })));
}
export async function classify(req, opts = {}) {
  const ctx = ctxNow(opts.at);
  const { name, adapter } = resolveModel(ctx, opts, "decide");
  if (adapter.kind !== "decision") throw new SandError("E-MODEL", `classify needs a decision model; \`${name}\` is a ${adapter.kind} model`, { at: opts.at });
  ctx.budget.check({ passes: adapter.caps.perRequest ? 1 : (req.questions ?? []).reduce((n, q) => n + (q.options?.length ?? q.levels?.length ?? 2), 0) }, opts.at);
  const res = await ctx.budget.gate.run(() => adapter.decide({ state: req.state, questions: req.questions, thresholds: req.thresholds }, { at: opts.at, signal: ctx.signal }));
  ctx.budget.record({ passes: res.passes ?? 0 });
  ctx.emit("sand.request", "event", { status: "ok", model: name, road: adapter.road ?? "", questions: (req.questions ?? []).length, passes: res.passes ?? 0, balance: res.balance ?? undefined, ms: res.ms, verb: "classify" }, { spanId: res.requestId ?? undefined, input: req, output: res.raw });
  return res.raw;
}

// ── chat ──────────────────────────────────────────────────────────────────
const coerceRecord = (shapeSpec, json) => {
  const out = {};
  for (const [n, s] of Object.entries(shapeSpec.slots)) {
    const sp = s.spec ?? s;
    const v = json?.[n];
    if (sp.type === "kind" || sp.type === "scale") out[n] = v === "unsure" || v == null || !idsOf(sp).includes(v) ? unsure : v;
    else if (sp.type === "shape") out[n] = coerceRecord(sp, v);
    else out[n] = v == null ? "" : String(v);
  }
  return out;
};
export async function ask(prompt, opts = {}) {
  const ctx = ctxNow(opts.at);
  if (typeof prompt !== "string" || !prompt.trim()) throw new SandError("E-STATE", "ask needs a prompt", { at: opts.at });
  const { name, adapter } = resolveModel(ctx, opts, "ask");
  const messages = [...(opts.system ? [{ role: "system", content: opts.system }] : []), { role: "user", content: prompt }];
  const schema = opts.as ? shapeSchema(opts.as.spec) : null;
  ctx.budget.check({ tokens: 0 }, opts.at);
  const r = await ctx.budget.gate.run(() => adapter.ask({ messages, schema, maxTokens: opts.maxTokens, temperature: opts.temperature }, { at: opts.at, signal: ctx.signal }));
  ctx.budget.record({ tokens: r.tokens ?? 0, completions: 1 });
  ctx.emit("sand.request", "event", { status: "ok", model: name, road: "", questions: 0, tokens: r.tokens ?? 0, ms: r.ms, verb: "ask", at: opts.at }, { input: messages, output: r.text });
  if (opts.as) return coerceRecord(opts.as.spec, r.json);
  return r.text;
}

// ── extraction ────────────────────────────────────────────────────────────
export async function extract(state, shapeObj, opts = {}) {
  const ctx = ctxNow(opts.at);
  const sp = shapeObj?.spec;
  if (!sp || sp.type !== "shape") throw new SandError("E-SHAPE", "extract(state, Shape) needs a shape declared with shape({ … })", { at: opts.at });
  const st = stateOf(state, opts.at);
  const slots = Object.entries(sp.slots).map(([n, s]) => [n, s.spec ?? s]);
  const kinded = slots.filter(([, s]) => s.type === "kind" || s.type === "scale");
  const texts = slots.filter(([, s]) => s.type === "text");
  const using = opts.using ?? null;
  const record = {};
  if (using === "decide" && texts.length) throw new SandError("E-MODEL", `slot ${JSON.stringify(texts[0][0])} needs a chat model and this extract says using: "decide"; drop the slot or the option`, { at: opts.at });
  const viaChat = async (only) => {
    const { name, adapter } = resolveModel(ctx, { ...opts, model: opts.askModel ?? (using === "ask" ? opts.model : undefined) }, "ask");
    if (!adapter.extract) throw new SandError("E-MODEL", `\`${name}\` cannot extract; it is a ${adapter.kind} model`, { at: opts.at });
    ctx.budget.check({ tokens: 0 }, opts.at);
    const r = await ctx.budget.gate.run(() => adapter.extract(st, sp, only, { at: opts.at, signal: ctx.signal }));
    ctx.budget.record({ tokens: r.tokens ?? 0, completions: 1 });
    ctx.emit("sand.request", "event", { status: "ok", model: name, road: "", questions: 0, tokens: r.tokens ?? 0, ms: r.ms, verb: "extract", at: opts.at }, { input: { state: st, slots: only ?? Object.keys(sp.slots) }, output: r.record });
    return r.record;
  };
  if (using === "ask" || (!kinded.length && texts.length)) {
    Object.assign(record, coerceRecord(sp, await viaChat(null)));
  } else {
    if (kinded.length) {
      const ds = await Promise.all(kinded.map(([, s]) => decide(st, { spec: s }, { at: opts.at, model: using === "decide" ? opts.model : opts.model })));
      kinded.forEach(([n], i) => { record[n] = ds[i].value; });
    }
    if (texts.length) {
      let missing = null;
      try { resolveModel(ctx, { ...opts, model: opts.askModel }, "ask"); } catch (e) { missing = e; }
      if (missing) throw new SandError("E-MODEL", `slot ${JSON.stringify(texts[0][0])} needs a chat model and none is configured; set OPENAI_API_KEY or name one in sand.json`, { at: opts.at });
      const part = await viaChat(texts.map(([n]) => n));
      for (const [n] of texts) record[n] = part?.[n] == null ? "" : String(part[n]);
    }
  }
  ctx.emit("sand.effect", "event", { at: opts.at, effect: "extract", name: sp.name ?? "", value: JSON.stringify(record, (k, v) => (v === unsure ? "unsure" : v)) }, { output: record });
  return record;
}

// ── control ───────────────────────────────────────────────────────────────
export function log(...parts) {
  const line = parts.map((p) => (typeof p === "string" ? p : p === unsure ? "unsure" : (() => { try { return JSON.stringify(p, (k, v) => (v === unsure ? "unsure" : v)); } catch { return String(p); } })())).join(" ");
  const ctx = current;
  ctx?.emit("sand.log", "log", { level: "info" }, { output: line, severity: "info" });
  const sink = ctx?.log ?? ports.log ?? ((l) => (typeof Deno !== "undefined" ? console.error(l) : console.log(l)));
  sink(line);
}
export async function parallel(items, fn, opts = {}) {
  const ctx = current;
  const limit = Math.max(1, opts.limit ?? 8), tolerate = !!opts.tolerate;
  const list = Array.from(items ?? []);
  const results = new Array(list.length);
  let next = 0, failed = null;
  ctx?.emit("sand.effect", "event", { at: opts.at, effect: "parallel", name: String(list.length), value: `limit ${limit}` });
  const worker = async () => {
    while (true) {
      if (failed && !tolerate) return;
      const i = next++;
      if (i >= list.length) return;
      try { results[i] = await fn(list[i], i); }
      catch (e) { if (tolerate) results[i] = e instanceof Error ? e : new Error(String(e)); else failed = failed ?? e; }
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, list.length) }, worker));
  if (failed) throw failed;
  return results;
}
export function race(...ps) { return Promise.race(ps.flat()); }
export function timeout(ms, p) {
  let timer;
  const t = new Promise((_, reject) => { timer = setTimeout(() => reject(new SandError("E-TIMEOUT", `did not settle within ${ms} ms`)), ms); });
  return Promise.race([p, t]).finally(() => clearTimeout(timer));
}
export async function retry(fn, opts = {}) {
  const times = opts.times ?? 3;
  const on = opts.on ?? ((e) => e?.code === "E-HOST" || e?.code === "E-TIMEOUT" || (e?.code === "E-WIRE" && (e.status ?? 0) >= 500));
  let last;
  for (let i = 0; i < times; i++) {
    try { return await fn(); }
    catch (e) { last = e; if (e?.code === "E-BUDGET" || !on(e) || i === times - 1) throw e; await new Promise((r) => setTimeout(r, 250 * 2 ** i)); }
  }
  throw last;
}
export async function using(model, fn) {
  const ctx = ctxNow();
  if (!ctx.config.models[model]) throw new SandError("E-MODEL", `using(${JSON.stringify(model)}) names no model; sand.json knows ${Object.keys(ctx.config.models).join(", ") || "none"}`);
  const prev = ctx.usingModel; ctx.usingModel = model;
  try { return await fn(); } finally { ctx.usingModel = prev; }
}
export { unsure, verdict };
