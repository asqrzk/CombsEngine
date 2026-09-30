/**
 * @combs/sand — the batcher.
 *
 * Every decision enqueues into a bucket keyed by (model, state, thresholds).
 * A microtask flush sends each bucket as one request, split at the model's
 * batch size, so `Promise.all`, `parallel` and the transform's hoisted
 * `decideAll` all cost one pass per distinct state on a seat. Answers are
 * memoised per (model, state, question) for the run; a memo hit costs nothing.
 * On a model priced per request, a head site's speculative questions ride
 * along and land in the memo for the calls that come later.
 */

import { SandError } from "./errors.js";
import { questionKey, idsOf } from "./wires/schema.js";

const thresholdsOf = (settle = {}) => (settle.confidence != null ? { confidence: settle.confidence } : {});
const chunks = (xs, n) => { const out = []; for (let i = 0; i < xs.length; i += n) out.push(xs.slice(i, i + n)); return out; };

export class Batcher {
  constructor(ctx) {
    this.ctx = ctx;
    this.buckets = new Map();
    this.memo = new Map();
    this.requests = 0;
    this.warned = new Set();
  }

  memoKey(modelName, state, question) { return `${modelName} ${state} ${questionKey(question)}`; }

  /**
   * @param {{modelName: string, adapter: object, state: string, question: object, settle: object, at?: string, speculate?: object[]}} p
   * @returns {Promise<{answer: object, request: object, cut: boolean, memo: boolean}>}
   */
  enqueue({ modelName, adapter, state, question, settle = {}, at = null, speculate = [] }) {
    const mk = this.memoKey(modelName, state, question);
    if (this.memo.has(mk)) return Promise.resolve({ ...this.memo.get(mk), memo: true });
    const bucketKey = `${modelName} ${JSON.stringify(thresholdsOf(settle))} ${state}`;
    let b = this.buckets.get(bucketKey);
    if (!b) { b = { modelName, adapter, state, settle, items: [], speculate: [], scheduled: false }; this.buckets.set(bucketKey, b); }
    for (const sq of speculate) b.speculate.push(sq);
    return new Promise((resolve, reject) => {
      b.items.push({ question, at, resolve, reject });
      if (!b.scheduled) { b.scheduled = true; queueMicrotask(() => this.flush(bucketKey)); }
    });
  }

  async flush(bucketKey) {
    const b = this.buckets.get(bucketKey);
    if (!b) return;
    this.buckets.delete(bucketKey);
    // dedupe by question key
    const byKey = new Map();
    for (const it of b.items) { const k = questionKey(it.question); if (!byKey.has(k)) byKey.set(k, { question: { ...it.question }, items: [] }); byKey.get(k).items.push(it); }
    const asked = [...byKey.values()].map((e) => e.question);
    // speculation: free on a per-request model, skipped elsewhere
    const spec = [];
    if (b.adapter.caps.perRequest) {
      for (const sq of b.speculate) {
        const k = questionKey(sq);
        if (byKey.has(k) || spec.some((x) => questionKey(x) === k) || this.memo.has(this.memoKey(b.modelName, b.state, sq))) continue;
        if (asked.length + spec.length >= b.adapter.caps.batch) break;
        spec.push({ ...sq });
      }
    }
    const all = [...asked, ...spec];
    const seen = new Set();
    for (const q of all) { let n = q.name, i = 2; while (seen.has(n)) n = `${q.name}_${i++}`; seen.add(n); q.name = n; }
    for (const chunk of chunks(all, Math.max(1, b.adapter.caps.batch))) {
      const items = chunk.flatMap((q) => byKey.get(questionKey(q))?.items ?? []);
      const at = items[0]?.at ?? b.items[0]?.at ?? null;
      const estimate = b.adapter.kind === "decision"
        ? { passes: b.adapter.caps.perRequest ? 1 : chunk.reduce((n, q) => n + (q.options?.length ?? q.levels?.length ?? 2), 0) }
        : { tokens: 0 };
      try { this.ctx.budget.check(estimate, at); } catch (e) { for (const it of items) it.reject(e); continue; }
      let res;
      const t0 = Date.now();
      try {
        res = await this.ctx.budget.gate.run(() => b.adapter.decide({ state: b.state, questions: chunk, settle: b.settle, thresholds: thresholdsOf(b.settle) }, { at, signal: this.ctx.signal, traceId: this.ctx.traceId }));
      } catch (e) {
        this.ctx.emit("sand.request", "event", { status: "error", model: b.modelName, road: b.adapter.road ?? "", questions: chunk.length, ms: Date.now() - t0, error: e?.code ?? e?.name }, { input: { state: b.state, questions: chunk }, status: "error", error: String(e?.message ?? e) });
        for (const it of items) it.reject(e);
        continue;
      }
      this.ctx.budget.record({ passes: res.passes ?? 0, tokens: res.tokens ?? 0, completions: res.completions ?? 0 });
      const id = ++this.requests;
      this.ctx.emit("sand.request", "event", { status: "ok", model: b.modelName, road: b.adapter.road ?? "", questions: chunk.length, passes: res.passes ?? 0, tokens: res.tokens ?? 0, balance: res.balance ?? undefined, ms: res.ms }, { spanId: res.requestId ?? undefined, input: { state: b.state, questions: chunk, thresholds: thresholdsOf(b.settle) }, output: res.raw ?? { answers: res.answers } });
      if (res.passesCharged != null && res.passesCharged !== res.passes) this.ctx.emit("sand.log", "log", { level: "warn", message: `the gate charged ${res.passesCharged} passes and the answer says ${res.passes}` }, { severity: "warn", output: `the gate charged ${res.passesCharged} passes and the answer says ${res.passes}` });
      // shape check: one answer per question, in order, top among the ids sent
      if (!Array.isArray(res.answers) || res.answers.length !== chunk.length) {
        const err = new SandError("E-ANSWER-SHAPE", `the answer carries ${res.answers?.length ?? 0} answers for ${chunk.length} questions`, { at });
        for (const it of items) it.reject(err); continue;
      }
      const byName = new Map(res.answers.map((a) => [a.name, a]));
      const cutOf = (name) => { const pq = (res.usage?.per_question ?? []).find((p) => p.name === name); return !!(pq && ((pq.option_tokens_cut ?? 0) > 0 || (pq.state_tokens_cut ?? 0) > 0)); };
      for (const q of chunk) {
        const a = byName.get(q.name) ?? res.answers[chunk.indexOf(q)];
        const entry = byKey.get(questionKey(q));
        const ids = q.type === "noul" ? ["true", "false"] : (q.options ?? q.levels).map((o) => o.id);
        if (!a || (a.name && a.name !== q.name)) {
          const err = new SandError("E-ANSWER-SHAPE", `answer ${chunk.indexOf(q) + 1} is named \`${a?.name ?? "?"}\`; the request asked \`${q.name}\``, { at });
          if (entry) for (const it of entry.items) it.reject(err); continue;
        }
        if (a.top != null && !ids.includes(String(a.top))) {
          const err = new SandError("E-ANSWER-SHAPE", `\`top\` is \`${a.top}\`, which was not among the ids sent (${ids.join(", ")})`, { at });
          if (entry) for (const it of entry.items) it.reject(err); continue;
        }
        if (a.uncalibrated && !this.warned.has(b.modelName)) { this.warned.add(b.modelName); this.ctx.emit("sand.log", "log", { level: "warn", message: `W-UNCALIBRATED: ${b.modelName} returned no logprobs; p is null and the floor is not applied` }, { severity: "warn", output: `W-UNCALIBRATED: ${b.modelName} returned no logprobs; p is null and the floor is not applied` }); }
        const meta = { answer: a, request: { id, model: b.modelName, road: b.adapter.road ?? null, path: b.adapter.path, passes: res.passes ?? 0, tokens: res.tokens ?? 0, ms: res.ms, memo: false }, cut: cutOf(q.name), memo: false };
        this.memo.set(this.memoKey(b.modelName, b.state, q), meta);
        if (entry) for (const it of entry.items) it.resolve(meta);
      }
    }
  }
}
