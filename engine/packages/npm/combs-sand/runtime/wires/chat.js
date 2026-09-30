/**
 * @combs/sand — the chat adapter: OpenAI Chat Completions.
 *
 * One completion per question, with `response_format: json_schema` so the
 * model can only answer with an option id (or "unsure"), `temperature: 0`, and
 * `logprobs` when the provider returns them — that is where a probability
 * comes from. The adapter answers in the decision wire's vocabulary
 * (`top`, `p_top`, `distribution`, `decision`, `reason`) so the evaluator has
 * one path for both model classes. Works against OpenAI, OpenRouter, Ollama,
 * vLLM and the Combs engine's own generative seat.
 */

import { SandError } from "../errors.js";
import { hostOf } from "../config.js";
import { promptOf, extractionPrompt, shapeSchema } from "./schema.js";
import { refusal } from "./decision.js";

export function chatAdapter(model, ports = {}) {
  const fetchImpl = ports.fetchImpl ?? globalThis.fetch?.bind(globalThis);
  if (!fetchImpl) throw new SandError("E-HOST", "no fetch implementation is available");
  const base = String(model.url).replace(/\/+$/, "");
  let logprobsWork = model.logprobs === true ? true : model.logprobs === false ? false : null; // null: "auto", learned on the first answer
  const caps = { batch: 1, perRequest: false, noul: true, abstain: "client", schema: true, logprobs: model.logprobs ?? "auto", text: true };

  async function completion({ messages, schema, schemaName = "decision", maxTokens = 64, temperature = 0, logprobs = false, signal, at }) {
    if (model.unavailable) throw new SandError("E-MODEL", `model ${model.name} reads $${model.unavailable}, which is not set in the environment`, { at });
    const key = typeof model.key === "function" ? await model.key() : model.key;
    const url = base + "/chat/completions";
    const body = {
      model: model.model ?? "default", messages, temperature, max_tokens: maxTokens, seed: 7,
      ...(schema ? { response_format: { type: "json_schema", json_schema: { name: schemaName, strict: true, schema } } } : {}),
      ...(logprobs ? { logprobs: true, top_logprobs: 5 } : {}),
      ...(model.extra ?? {}),
    };
    const headers = { "content-type": "application/json", ...(model.headers ?? {}) };
    if (key) headers.authorization = `Bearer ${key}`;
    const t0 = Date.now();
    let res;
    try { res = await fetchImpl(url, { method: "POST", headers, body: JSON.stringify(body), signal }); }
    catch (e) {
      if (e?.name === "AbortError") throw new SandError("E-TIMEOUT", `${model.name} was stopped before it answered`, { at, cause: e });
      throw new SandError("E-HOST", `could not reach ${hostOf(url) ?? url}: ${e?.message ?? e}`, { at, cause: e });
    }
    const ms = Date.now() - t0;
    const text = await res.text();
    let json = null; try { json = JSON.parse(text); } catch { /* not JSON */ }
    if (!res.ok) throw refusal(model.name, res.status, json, text, { at });
    const choice = json?.choices?.[0];
    if (!choice) throw new SandError("E-WIRE", `${model.name} answered without a choice`, { at, api: json });
    const content = typeof choice.message?.content === "string" ? choice.message.content : Array.isArray(choice.message?.content) ? choice.message.content.map((p) => p.text ?? "").join("") : "";
    return { content, logprobs: choice.logprobs?.content ?? null, tokens: json.usage?.total_tokens ?? 0, raw: json, ms, status: res.status };
  }

  return {
    kind: "chat",
    name: model.name,
    road: null,
    path: "/chat/completions",
    caps,
    async decide(req, ctx = {}) {
      const answers = []; let tokens = 0, ms = 0;
      for (const q of req.questions) {
        const ids = q.type === "noul" ? ["true", "false"] : (q.options ?? q.levels).map((o) => o.id);
        const schema = { type: "object", properties: { answer: { type: "string", enum: [...ids, "unsure"] } }, required: ["answer"], additionalProperties: false };
        const r = await completion({ messages: [{ role: "system", content: promptOf(q) }, { role: "user", content: req.state }], schema, maxTokens: 24, logprobs: logprobsWork !== false, signal: ctx.signal, at: ctx.at });
        if (logprobsWork === null) logprobsWork = Array.isArray(r.logprobs) && r.logprobs.length > 0;
        tokens += r.tokens; ms += r.ms;
        answers.push(answerFrom(q, ids, r, req.settle ?? {}));
      }
      return { answers, usage: { total_tokens: tokens }, passes: 0, tokens, completions: req.questions.length, raw: null, ms, status: 200, passesCharged: null, balance: null, requestId: null };
    },
    async ask(req, ctx = {}) {
      const r = await completion({ messages: req.messages, schema: req.schema ?? null, schemaName: "answer", maxTokens: req.maxTokens ?? 512, temperature: req.temperature ?? 0, signal: ctx.signal, at: ctx.at });
      let json;
      if (req.schema) {
        try { json = JSON.parse(r.content); } catch { throw new SandError("E-WIRE", `${model.name} answered with text that is not the JSON the schema asked for: ${r.content.slice(0, 120)}`, { at: ctx.at }); }
      }
      return { text: r.content, json, tokens: r.tokens, completions: 1, raw: r.raw, ms: r.ms };
    },
    /** Extraction in one completion: the shape as a schema. */
    async extract(state, shapeSpec, only, ctx = {}) {
      const schema = shapeSchema(shapeSpec, { only });
      const r = await completion({ messages: [{ role: "system", content: extractionPrompt(shapeSpec, only) }, { role: "user", content: state }], schema, schemaName: "extraction", maxTokens: 512, signal: ctx.signal, at: ctx.at });
      let json;
      try { json = JSON.parse(r.content); } catch { throw new SandError("E-WIRE", `${model.name} answered with text that is not the JSON the shape asked for: ${r.content.slice(0, 120)}`, { at: ctx.at }); }
      return { record: json, tokens: r.tokens, completions: 1, raw: r.raw, ms: r.ms };
    },
  };
}

/** Builds a decision-wire answer from a completion. */
export function answerFrom(q, ids, r, settle) {
  let parsed = null; try { parsed = JSON.parse(r.content); } catch { /* fall through */ }
  const ans = parsed?.answer;
  const valid = typeof ans === "string" && (ids.includes(ans) || ans === "unsure");
  let distribution = null, p_top = null;
  if (Array.isArray(r.logprobs) && r.logprobs.length && valid && ans !== "unsure") {
    const pos = r.content.indexOf(ans, Math.max(0, r.content.indexOf("answer")));
    if (pos >= 0) {
      let acc = 0;
      for (const tok of r.logprobs) {
        const start = acc; acc += (tok.token ?? "").length;
        if (start <= pos && pos < acc) {
          const cands = tok.top_logprobs?.length ? tok.top_logprobs : [{ token: tok.token, logprob: tok.logprob }];
          const mass = {}; for (const id of [...ids, "unsure"]) mass[id] = 0;
          for (const c of cands) {
            const piece = String(c.token ?? "").replace(/^[\s"']+/, "");
            if (!piece) continue;
            const hits = [...ids, "unsure"].filter((id) => id.startsWith(piece) || piece.startsWith(id));
            if (!hits.length) continue;
            const p = Math.exp(c.logprob) / hits.length;
            for (const h of hits) mass[h] += p;
          }
          const total = Object.values(mass).reduce((a, b) => a + b, 0);
          if (total > 0) { distribution = {}; for (const id of ids) distribution[id] = mass[id] / total; p_top = distribution[ans] ?? null; }
          break;
        }
      }
    }
  }
  const sorted = distribution ? Object.entries(distribution).sort((a, b) => b[1] - a[1]) : null;
  let top = valid && ans !== "unsure" ? ans : sorted?.[0]?.[0] ?? ids[0];
  let decision = "act", reason = null;
  if (!valid) { decision = "abstain"; reason = `the model's JSON did not name an option (W-PARSE): ${r.content.slice(0, 80)}`; }
  else if (ans === "unsure") { decision = "abstain"; reason = "the model answered unsure"; }
  else if (settle.floor != null && p_top != null && p_top < settle.floor) { decision = "abstain"; reason = `p_top ${p_top.toFixed(3)} < floor ${settle.floor} (client floor)`; }
  else if (settle.floor != null && p_top == null) reason = "no logprobs: the floor was not applied";
  const margin = sorted && sorted.length > 1 ? sorted[0][1] - sorted[1][1] : null;
  const out = { name: q.name, type: q.type, top, p_top, distribution, decision, reason, confidence: p_top, margin, calibrated: false, uncalibrated: p_top == null };
  if (q.type === "score") {
    const levels = q.levels.map((l) => l.id);
    out.expected_level = distribution ? levels.reduce((s, id, i) => s + i * (distribution[id] ?? 0), 0) : levels.indexOf(top);
  }
  return out;
}
