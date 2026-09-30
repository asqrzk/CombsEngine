/**
 * @combs/sand — the memory kind: what a program remembers between runs.
 *
 * Two adapters behind one verb. The **platform** store is the account's own,
 * named: `POST {url}/v1/memory/{name}/remember|recall|forget` with the bearer
 * and the same `x-combs-*` metering headers the classify wire reads; refusals
 * are named after the status exactly as there. The **local** store is a JSON
 * file per name under a `file:<dir>` url, recalled by idf-weighted lexical
 * overlap, written through the host's `writeFile` port and serialised per store
 * so two `remember`s in one tick cannot lose each other. A host with no
 * `writeFile` (a browser page) is told to use the platform store.
 *
 * Zero dependencies; all I/O goes through the injected ports.
 */

import { SandError } from "../errors.js";
import { hostOf } from "../config.js";
import { Semaphore } from "../budget.js";
import { hash, slug } from "./schema.js";
import { refusal } from "./decision.js";

/** The registry's factory: the url decides which store answers. */
export function memoryAdapter(model, ports = {}) {
  const url = String(model.url ?? "");
  return url.startsWith("file:") ? localMemory(model, ports) : platformMemory(model, ports);
}

// ── the platform store ────────────────────────────────────────────────────
function platformMemory(model, ports) {
  const fetchImpl = ports.fetchImpl ?? globalThis.fetch?.bind(globalThis);
  if (!fetchImpl) throw new SandError("E-HOST", "no fetch implementation is available");
  const base = String(model.url).replace(/\/+$/, "");

  async function call(name, op, body, ctx = {}) {
    if (model.unavailable) throw new SandError("E-MEMORY", `memory ${model.name} reads $${model.unavailable}, which is not set in the environment`, { at: ctx.at });
    const key = typeof model.key === "function" ? await model.key() : model.key;
    const url = `${base}/v1/memory/${encodeURIComponent(name)}/${op}`;
    const headers = { "content-type": "application/json", ...(model.headers ?? {}) };
    if (key) headers.authorization = `Bearer ${key}`;
    const t0 = Date.now();
    let res;
    try { res = await fetchImpl(url, { method: "POST", headers, body: JSON.stringify(body), signal: ctx.signal }); }
    catch (e) {
      if (e?.name === "AbortError") throw new SandError("E-TIMEOUT", `${model.name} was stopped before it answered`, { at: ctx.at, cause: e });
      throw new SandError("E-HOST", `could not reach ${hostOf(url) ?? url}: ${e?.message ?? e}. Nothing was charged.`, { at: ctx.at, cause: e });
    }
    const ms = Date.now() - t0;
    const text = await res.text();
    let json = null; try { json = JSON.parse(text); } catch { /* not JSON */ }
    const h = (n) => res.headers.get(n);
    const meta = {
      ms, status: res.status,
      passes: h("x-combs-passes") != null ? Number(h("x-combs-passes")) : 0,
      balance: h("x-combs-balance") != null ? Number(h("x-combs-balance")) : null,
      requestId: h("x-combs-request") ?? null,
    };
    if (!res.ok) throw refusal(model.name, res.status, json, text, ctx);
    if (!json || typeof json !== "object") throw new SandError("E-MEMORY", `${model.name} answered ${op} with something that is not an object: ${text.slice(0, 120)}`, { at: ctx.at, status: res.status });
    return { json, ...meta };
  }

  return {
    kind: "memory",
    name: model.name,
    road: null,
    path: "/v1/memory",
    metered: true,
    caps: { batch: 1, perRequest: true, noul: false, abstain: "client", schema: false, logprobs: false, text: true, tools: false, stream: false },
    async decide(_req, ctx = {}) { throw new SandError("E-MODEL", `\`${model.name}\` is a memory store, not a decision model`, { at: ctx.at }); },
    async remember(name, items, ctx = {}) {
      const r = await call(name, "remember", { items }, ctx);
      const ids = Array.isArray(r.json.ids) ? r.json.ids.map(String) : [];
      if (ids.length !== items.length) throw new SandError("E-MEMORY", `${model.name} kept ${ids.length} of ${items.length} item${items.length === 1 ? "" : "s"}`, { at: ctx.at });
      return { ids, ...r };
    },
    async recall(name, query, limit, ctx = {}) {
      const r = await call(name, "recall", { query, limit }, ctx);
      const hits = Array.isArray(r.json.hits) ? r.json.hits : null;
      if (!hits) throw new SandError("E-MEMORY", `${model.name} answered recall without a \`hits\` array`, { at: ctx.at });
      return { hits: hits.map((h) => ({ id: String(h.id ?? ""), text: String(h.text ?? ""), record: h.record ?? null, score: Number(h.score ?? 0), at: h.at ?? null })), ...r };
    },
    async forget(name, ids, ctx = {}) {
      const r = await call(name, "forget", { ids }, ctx);
      return { forgotten: Number(r.json.forgotten ?? ids.length), ...r };
    },
  };
}

// ── the local store ───────────────────────────────────────────────────────
const WORD = /[a-z0-9][a-z0-9'’]*/g;
/** The terms of a text: lowercase words of two characters or more. */
export const termsOf = (text) => (String(text ?? "").toLowerCase().match(WORD) ?? []).filter((w) => w.length > 1);

/**
 * idf-weighted lexical recall (BM25 with the usual k=1.2, b=0.75): a term that
 * is everywhere counts for little, a rare one for much, and a long item does
 * not win by length alone.
 */
export function rank(query, items, limit) {
  const q = [...new Set(termsOf(query))];
  if (!q.length || !items.length) return [];
  const N = items.length;
  const docs = items.map((it) => termsOf(`${it.text} ${it.record ? JSON.stringify(it.record) : ""}`));
  const avg = docs.reduce((n, d) => n + d.length, 0) / N || 1;
  const df = new Map();
  for (const d of docs) for (const t of new Set(d)) df.set(t, (df.get(t) ?? 0) + 1);
  const k = 1.2, b = 0.75;
  const scored = items.map((it, i) => {
    const d = docs[i];
    const tf = new Map();
    for (const t of d) tf.set(t, (tf.get(t) ?? 0) + 1);
    let score = 0;
    for (const t of q) {
      const f = tf.get(t) ?? 0;
      if (!f) continue;
      const idf = Math.log(1 + (N - (df.get(t) ?? 0) + 0.5) / ((df.get(t) ?? 0) + 0.5));
      score += idf * ((f * (k + 1)) / (f + k * (1 - b + (b * d.length) / avg)));
    }
    return { ...it, score };
  });
  return scored.filter((h) => h.score > 0).sort((a, b2) => b2.score - a.score || String(a.id).localeCompare(String(b2.id))).slice(0, Math.max(1, limit ?? 5));
}

function localMemory(model, ports) {
  const dir = String(model.url).replace(/^file:(\/\/)?/, "").replace(/\/+$/, "") || ".";
  const readFile = ports.readFile ?? null;
  const writeFile = ports.writeFile ?? null;
  const gates = new Map();
  const gateFor = (name) => { let g = gates.get(name); if (!g) { g = new Semaphore(1); gates.set(name, g); } return g; };
  const pathOf = (name) => `${dir}/${slug(name)}.json`;
  const needWrite = (at) => { if (!writeFile) throw new SandError("E-MEMORY", "no local store in this host; use the platform memory", { at }); };

  const load = async (name, at) => {
    if (!readFile) throw new SandError("E-MEMORY", "no local store in this host; use the platform memory", { at });
    const text = await readFile(pathOf(name));
    if (text == null) return { v: 1, items: [] };
    try {
      const json = JSON.parse(text);
      return { v: 1, items: Array.isArray(json?.items) ? json.items : [] };
    } catch (e) { throw new SandError("E-MEMORY", `${pathOf(name)} is not valid JSON: ${e.message}`, { at }); }
  };
  const save = async (name, store, at) => {
    needWrite(at);
    try { await writeFile(pathOf(name), JSON.stringify(store, null, 2) + "\n"); }
    catch (e) { throw new SandError("E-MEMORY", `could not write ${pathOf(name)}: ${e?.message ?? e}`, { at }); }
  };

  return {
    kind: "memory",
    name: model.name,
    road: null,
    path: pathOf("<name>"),
    metered: false,
    caps: { batch: 1, perRequest: true, noul: false, abstain: "client", schema: false, logprobs: false, text: true, tools: false, stream: false },
    async decide(_req, ctx = {}) { throw new SandError("E-MODEL", `\`${model.name}\` is a memory store, not a decision model`, { at: ctx.at }); },
    async remember(name, items, ctx = {}) {
      needWrite(ctx.at);
      const t0 = Date.now();
      return gateFor(name).run(async () => {
        const store = await load(name, ctx.at);
        const ids = [];
        for (const item of items) {
          const id = `m${(store.items.length + 1).toString(36)}_${hash(String(item.text) + "|" + store.items.length)}`;
          store.items.push({ id, text: String(item.text), record: item.record ?? null, at: new Date().toISOString() });
          ids.push(id);
        }
        await save(name, store, ctx.at);
        return { ids, ms: Date.now() - t0, passes: 0, balance: null, requestId: null, status: 200 };
      });
    },
    async recall(name, query, limit, ctx = {}) {
      const t0 = Date.now();
      const store = await load(name, ctx.at);
      return { hits: rank(query, store.items, limit), ms: Date.now() - t0, passes: 0, balance: null, requestId: null, status: 200 };
    },
    async forget(name, ids, ctx = {}) {
      needWrite(ctx.at);
      const t0 = Date.now();
      return gateFor(name).run(async () => {
        const store = await load(name, ctx.at);
        const want = new Set(ids.map(String));
        const before = store.items.length;
        store.items = want.has("*") ? [] : store.items.filter((it) => !want.has(String(it.id)));
        await save(name, store, ctx.at);
        return { forgotten: before - store.items.length, ms: Date.now() - t0, passes: 0, balance: null, requestId: null, status: 200 };
      });
    },
  };
}
