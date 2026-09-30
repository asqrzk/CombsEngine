/**
 * @combs/sand — the decision adapter: the Combs classify wire.
 *
 * `POST {url}{road}` with `{state, questions[≤16], thresholds?}` and a bearer.
 * Never sends `temperature`, `render` or a `label`. Reads the gate's metering
 * headers (`x-combs-passes`, `x-combs-balance`, `x-combs-request`) so the trace
 * can show what a request cost and what is left. Refusals are named after the
 * status and carry the API's own words; a refused request was never charged.
 *
 * Zero dependencies; all I/O goes through an injectable `fetchImpl`.
 */

import { SandError } from "../errors.js";
import { hostOf } from "../config.js";

export const ROADS = {
  seat: "/v1/classify",
  "id-means": "/roads/id-means/v1/classify",
  "label-means": "/roads/label-means/v1/classify",
  "label-anchors": "/roads/label-anchors/v1/classify",
  modernbert: "/zs/modernbert/v1/classify",
  deberta: "/zs/deberta/v1/classify",
};

/** @param {object} model the resolved model config @param {{fetchImpl?: typeof fetch}} ports */
export function decisionAdapter(model, ports = {}) {
  const fetchImpl = ports.fetchImpl ?? globalThis.fetch?.bind(globalThis);
  if (!fetchImpl) throw new SandError("E-HOST", "no fetch implementation is available");
  const road = model.road ?? "seat";
  if (!ROADS[road]) throw new SandError("E-MODEL", `road ${JSON.stringify(road)} names no road; SAND knows ${Object.keys(ROADS).join(", ")}`);
  const zeroShot = road === "modernbert" || road === "deberta";
  const caps = { batch: 16, perRequest: !zeroShot, noul: !zeroShot, abstain: zeroShot ? "client" : "wire", schema: false, logprobs: false, text: false };
  const base = String(model.url).replace(/\/+$/, "");

  return {
    kind: "decision",
    name: model.name,
    road,
    path: ROADS[road],
    caps,
    /** @param {{state: string, questions: object[], thresholds?: object}} req @param {{at?: string, signal?: AbortSignal}} ctx */
    async decide(req, ctx = {}) {
      if (model.unavailable) throw new SandError("E-MODEL", `model ${model.name} reads $${model.unavailable}, which is not set in the environment`, { at: ctx.at });
      if (zeroShot) for (const q of req.questions) if (q.type === "noul") throw new SandError("E-MODEL", `road ${road} cannot answer a prose predicate; it takes choice and score questions only`, { at: ctx.at });
      const key = typeof model.key === "function" ? await model.key() : model.key;
      const url = base + ROADS[road];
      const body = { state: req.state, questions: req.questions };
      if (req.thresholds && Object.keys(req.thresholds).length) body.thresholds = req.thresholds;
      const headers = { "content-type": "application/json", ...(model.headers ?? {}) };
      if (key) headers.authorization = `Bearer ${key}`;
      const t0 = Date.now();
      let res;
      try {
        res = await fetchImpl(url, { method: "POST", headers, body: JSON.stringify(body), signal: ctx.signal });
      } catch (e) {
        if (e?.name === "AbortError") throw new SandError("E-TIMEOUT", `${model.name} was stopped before it answered`, { at: ctx.at, cause: e });
        throw new SandError("E-HOST", `could not reach ${hostOf(url) ?? url}: ${e?.message ?? e}. Nothing was charged.`, { at: ctx.at, cause: e });
      }
      const ms = Date.now() - t0;
      const text = await res.text();
      let json = null; try { json = JSON.parse(text); } catch { /* not JSON */ }
      const h = (n) => res.headers.get(n);
      const meta = {
        ms, status: res.status,
        passesCharged: h("x-combs-passes") != null ? Number(h("x-combs-passes")) : null,
        balance: h("x-combs-balance") != null ? Number(h("x-combs-balance")) : null,
        requestId: h("x-combs-request") ?? null,
      };
      if (!res.ok) throw refusal(model.name, res.status, json, text, ctx);
      if (!json || !Array.isArray(json.answers)) throw new SandError("E-WIRE", `${model.name} answered without an \`answers\` array`, { at: ctx.at, status: res.status, api: json ?? text.slice(0, 240) });
      return { answers: json.answers, usage: json.usage ?? {}, passes: Number.isInteger(json.usage?.forward_passes) ? json.usage.forward_passes : 1, tokens: 0, completions: 0, raw: json, ...meta };
    },
  };
}

/** Maps a non-2xx answer to a named refusal carrying the API's own words. */
export function refusal(modelName, status, json, text, ctx = {}) {
  const e = json?.error;
  const msg = (typeof e === "string" ? e : e?.message) ?? (typeof json?.message === "string" ? json.message : null) ?? (text && text.length < 240 ? text : `HTTP ${status}`);
  const where = { at: ctx.at, status, api: json };
  const tail = " Nothing was charged.";
  if (status === 401) return new SandError("E-AUTH", `${msg}${tail}`, where);
  if (status === 402) return new SandError("E-PASSES", `${msg}${json?.balance != null ? ` (balance ${json.balance})` : ""}${tail}`, where);
  if (status === 403) return new SandError("E-SUSPENDED", `${msg}${tail}`, where);
  if (status >= 400 && status < 500) return new SandError("E-WIRE", `${modelName} answered ${status}: ${msg}${tail}`, where);
  return new SandError("E-HOST", `${e ? msg : `the model host did not answer (HTTP ${status})`}${tail}`, where);
}
