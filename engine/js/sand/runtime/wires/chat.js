/**
 * @combs/sand — the chat adapter: OpenAI Chat Completions.
 *
 * `complete()` is the one call underneath: messages in, `{text, toolCalls,
 * finish, tokens, usage, estimated, ms}` out, buffered or streamed. Streaming
 * reads `data:` frames off `res.body`, ignores `:` keepalive comments, stops at
 * `[DONE]`, accumulates `delta.tool_calls` **by index** — so the Combs engine's
 * one-whole-call-per-chunk and OpenAI's fragments both arrive whole — takes
 * `usage` from whichever frame carries it, and throws `E-STREAM` with the
 * provider's own words on an in-stream `error` frame. When nothing reports
 * usage, tokens are estimated at chars/4 and the run says so once (`W-UNMETERED`).
 *
 * A model that thinks first sends its reasoning in `delta.reasoning_content`
 * (`reasoning` on some hosts), apart from the answer in `delta.content`. The
 * two are never mixed: `text` is the answer, `thinking` is the reasoning and
 * `thinkingMs` how long it took (from the first reasoning piece to the first
 * answer piece or tool call), and `onThinking` hears each piece as it comes.
 *
 * `decide`, `ask` and `extract` are thin wrappers over it: one completion per
 * question with `response_format: json_schema` so the model can only answer
 * with an option id (or "unsure"), `temperature: 0`, and `logprobs` when the
 * provider returns them — that is where a probability comes from. The adapter
 * answers in the decision wire's vocabulary (`top`, `p_top`, `distribution`,
 * `decision`, `reason`) so the evaluator has one path for both model classes.
 * Works against OpenAI, OpenRouter, Ollama, vLLM and the Combs engine's own
 * generative seat. Zero dependencies; every byte of I/O goes through the
 * injectable `fetchImpl`.
 */

import { SandError } from "../errors.js";
import { hostOf } from "../config.js";
import { promptOf, extractionPrompt, shapeSchema } from "./schema.js";
import { refusal } from "./decision.js";

/** Tokens when the provider metered nothing: the usual four characters each. */
export const estimateTokens = (messages, text) => {
  const chars = (messages ?? []).reduce((n, m) => n + String(typeof m?.content === "string" ? m.content : JSON.stringify(m?.content ?? "")).length, 0) + String(text ?? "").length;
  return Math.max(1, Math.ceil(chars / 4));
};

export function chatAdapter(model, ports = {}) {
  const fetchImpl = ports.fetchImpl ?? globalThis.fetch?.bind(globalThis);
  if (!fetchImpl) throw new SandError("E-HOST", "no fetch implementation is available");
  const base = String(model.url).replace(/\/+$/, "");
  let logprobsWork = model.logprobs === true ? true : model.logprobs === false ? false : null; // null: "auto", learned on the first answer
  const caps = { batch: 1, perRequest: false, noul: true, abstain: "client", schema: true, logprobs: model.logprobs ?? "auto", text: true, tools: true, stream: true };

  /**
   * One completion, buffered or streamed.
   * @param {{messages: object[], tools?: object[]|null, schema?: object|null, schemaName?: string, stream?: boolean, maxTokens?: number, temperature?: number, logprobs?: boolean}} req
   * @param {{at?: string|null, signal?: AbortSignal|null, onDelta?: (piece: string) => void, onThinking?: (piece: string) => void}} ctx
   */
  async function complete(req, ctx = {}) {
    const { messages, tools = null, schema = null, schemaName = "decision", stream = false, maxTokens = 512, temperature = 0, logprobs = false } = req;
    const { signal = null, at = null, onDelta = null, onThinking = null } = ctx;
    if (model.unavailable) throw new SandError("E-MODEL", `model ${model.name} reads $${model.unavailable}, which is not set in the environment`, { at });
    const key = typeof model.key === "function" ? await model.key() : model.key;
    const url = base + "/chat/completions";
    const body = {
      model: model.model ?? "default", messages, temperature, max_tokens: maxTokens, seed: 7,
      ...(schema ? { response_format: { type: "json_schema", json_schema: { name: schemaName, strict: true, schema } } } : {}),
      ...(tools?.length ? { tools } : {}),
      ...(logprobs ? { logprobs: true, top_logprobs: 5 } : {}),
      ...(stream ? { stream: true, stream_options: { include_usage: true } } : {}),
      ...(model.extra ?? {}),
    };
    const headers = { "content-type": "application/json", ...(model.headers ?? {}) };
    if (key) headers.authorization = `Bearer ${key}`;
    const t0 = Date.now();
    let res;
    try { res = await fetchImpl(url, { method: "POST", headers, body: JSON.stringify(body), signal }); }
    catch (e) {
      if (e?.name === "AbortError" || signal?.aborted) throw new SandError("E-TIMEOUT", `${model.name} was stopped before it answered`, { at, cause: e });
      throw new SandError("E-HOST", `could not reach ${hostOf(url) ?? url}: ${e?.message ?? e}`, { at, cause: e });
    }
    if (!stream) {
      const ms0 = Date.now() - t0;
      const raw = await res.text();
      let json = null; try { json = JSON.parse(raw); } catch { /* not JSON */ }
      if (!res.ok) throw refusal(model.name, res.status, json, raw, { at });
      const choice = json?.choices?.[0];
      if (!choice) throw new SandError("E-WIRE", `${model.name} answered without a choice`, { at, api: json });
      const message = choice.message ?? {};
      const text = typeof message.content === "string" ? message.content : Array.isArray(message.content) ? message.content.map((p) => p.text ?? "").join("") : "";
      const toolCalls = (message.tool_calls ?? []).map((c, i) => toolCallOf(c, i));
      const reported = json.usage?.total_tokens ?? null;
      return {
        text, toolCalls, finish: choice.finish_reason ?? null,
        tokens: reported ?? estimateTokens(messages, text), usage: json.usage ?? null, estimated: reported == null,
        logprobs: choice.logprobs?.content ?? null, raw: json, ms: ms0, status: res.status,
        thinking: reasoningOf(message), thinkingMs: 0,
      };
    }
    if (!res.ok) {
      const raw = await res.text().catch(() => "");
      let json = null; try { json = JSON.parse(raw); } catch { /* not JSON */ }
      throw refusal(model.name, res.status, json, raw, { at });
    }
    const r = await readStream(res, { model, signal, at, onDelta, onThinking });
    const reported = r.usage?.total_tokens ?? null;
    return {
      text: r.text, toolCalls: r.toolCalls, finish: r.finish,
      tokens: reported ?? estimateTokens(messages, r.text), usage: r.usage, estimated: reported == null,
      logprobs: r.logprobs, raw: r.last, ms: Date.now() - t0, status: res.status,
      thinking: r.thinking, thinkingMs: r.thinkingMs,
    };
  }

  return {
    kind: "chat",
    name: model.name,
    road: null,
    path: "/chat/completions",
    caps,
    complete,
    async decide(req, ctx = {}) {
      const answers = []; let tokens = 0, ms = 0, estimated = false;
      for (const q of req.questions) {
        const ids = q.type === "noul" ? ["true", "false"] : (q.options ?? q.levels).map((o) => o.id);
        const schema = { type: "object", properties: { answer: { type: "string", enum: [...ids, "unsure"] } }, required: ["answer"], additionalProperties: false };
        const r = await complete({ messages: [{ role: "system", content: promptOf(q) }, { role: "user", content: req.state }], schema, maxTokens: 24, logprobs: logprobsWork !== false }, { signal: ctx.signal, at: ctx.at });
        if (logprobsWork === null) logprobsWork = Array.isArray(r.logprobs) && r.logprobs.length > 0;
        tokens += r.tokens; ms += r.ms; estimated = estimated || r.estimated;
        answers.push(answerFrom(q, ids, { content: r.text, logprobs: r.logprobs }, req.settle ?? {}));
      }
      return { answers, usage: { total_tokens: tokens }, passes: 0, tokens, completions: req.questions.length, estimated, raw: null, ms, status: 200, passesCharged: null, balance: null, requestId: null };
    },
    async ask(req, ctx = {}) {
      const r = await complete({ messages: req.messages, schema: req.schema ?? null, schemaName: "answer", maxTokens: req.maxTokens ?? 512, temperature: req.temperature ?? 0, stream: !!req.stream }, { signal: ctx.signal, at: ctx.at, onDelta: ctx.onDelta });
      let json;
      if (req.schema) {
        try { json = JSON.parse(r.text); } catch { throw new SandError("E-WIRE", `${model.name} answered with text that is not the JSON the schema asked for: ${r.text.slice(0, 120)}`, { at: ctx.at }); }
      }
      return { text: r.text, json, tokens: r.tokens, estimated: r.estimated, completions: 1, raw: r.raw, ms: r.ms };
    },
    /** Extraction in one completion: the shape as a schema. */
    async extract(state, shapeSpec, only, ctx = {}) {
      const schema = shapeSchema(shapeSpec, { only });
      const r = await complete({ messages: [{ role: "system", content: extractionPrompt(shapeSpec, only) }, { role: "user", content: state }], schema, schemaName: "extraction", maxTokens: 512 }, { signal: ctx.signal, at: ctx.at });
      let json;
      try { json = JSON.parse(r.text); } catch { throw new SandError("E-WIRE", `${model.name} answered with text that is not the JSON the shape asked for: ${r.text.slice(0, 120)}`, { at: ctx.at }); }
      return { record: json, tokens: r.tokens, estimated: r.estimated, completions: 1, raw: r.raw, ms: r.ms };
    },
  };
}

/** The reasoning a buffered message carries, apart from its content. */
function reasoningOf(message) {
  const r = message?.reasoning_content ?? message?.reasoning;
  return typeof r === "string" ? r : "";
}

/** One tool call off the wire: the name once, `arguments` a JSON string. */
function toolCallOf(c, i) {
  const args = c?.function?.arguments ?? "";
  let parsed = null;
  try { parsed = args ? JSON.parse(args) : {}; } catch { parsed = null; }
  return { id: c?.id ?? `call_${i}`, index: c?.index ?? i, name: c?.function?.name ?? "", arguments: String(args), args: parsed };
}

/**
 * The SSE reader: `\n`-split buffer, `:` comments ignored, `[DONE]` ends it,
 * tool calls accumulated by index, usage from whichever frame carries it.
 */
async function readStream(res, { model, signal, at, onDelta, onThinking }) {
  if (!res.body || typeof res.body.getReader !== "function") throw new SandError("E-STREAM", `${model.name} answered a stream with no body to read`, { at });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const calls = new Map();
  let text = "", buf = "", finish = null, usage = null, last = null, logprobs = null, done = false;
  let thinking = "", thinkFrom = null, thinkTo = null;
  const endThought = () => { if (thinkFrom != null && thinkTo == null) thinkTo = Date.now(); };
  const stop = () => { try { reader.cancel(); } catch { /* already gone */ } };
  const onAbort = () => stop();
  signal?.addEventListener?.("abort", onAbort, { once: true });
  const aborted = () => {
    if (!signal?.aborted) return false;
    stop();
    throw new SandError("E-TIMEOUT", `${model.name} was stopped before it answered`, { at });
  };
  const frame = (line) => {
    if (!line || line.startsWith(":")) return false; // a keepalive comment
    if (!line.startsWith("data:")) return false; // event:/id: fields carry nothing we read
    const payload = line.slice(5).trim();
    if (payload === "[DONE]") return true;
    if (!payload) return false;
    let json; try { json = JSON.parse(payload); } catch { return false; }
    last = json;
    if (json.error) {
      stop();
      const e = json.error;
      throw new SandError("E-STREAM", `${model.name} stopped mid-answer: ${(typeof e === "string" ? e : e?.message) ?? "the stream carried an error"}`, { at, api: json });
    }
    if (json.usage) usage = json.usage;
    const choice = json.choices?.[0];
    if (!choice) return false;
    const delta = choice.delta ?? {};
    const thought = typeof delta.reasoning_content === "string" ? delta.reasoning_content : typeof delta.reasoning === "string" ? delta.reasoning : "";
    if (thought) { thinkFrom ??= Date.now(); thinking += thought; try { onThinking?.(thought); } catch { /* a listener's fault is never the model's */ } }
    if (typeof delta.content === "string" && delta.content) { endThought(); text += delta.content; try { onDelta?.(delta.content); } catch { /* a listener's fault is never the model's */ } }
    for (const c of delta.tool_calls ?? []) {
      endThought();
      const i = c.index ?? calls.size;
      const cur = calls.get(i) ?? { id: null, index: i, name: "", arguments: "" };
      if (c.id) cur.id = c.id;
      if (c.function?.name) cur.name = c.function.name;
      if (typeof c.function?.arguments === "string") cur.arguments += c.function.arguments;
      calls.set(i, cur);
    }
    if (choice.finish_reason) finish = choice.finish_reason;
    if (choice.logprobs?.content) (logprobs ??= []).push(...choice.logprobs.content);
    return false;
  };
  try {
    while (!done) {
      aborted();
      let chunk;
      try { chunk = await reader.read(); }
      catch (e) {
        if (signal?.aborted || e?.name === "AbortError") throw new SandError("E-TIMEOUT", `${model.name} was stopped before it answered`, { at, cause: e });
        throw new SandError("E-STREAM", `${model.name}'s stream broke: ${e?.message ?? e}`, { at, cause: e });
      }
      aborted();
      if (chunk.done) break;
      buf += typeof chunk.value === "string" ? chunk.value : decoder.decode(chunk.value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).replace(/\r$/, "");
        buf = buf.slice(i + 1);
        if (frame(line.trim())) { done = true; break; }
      }
    }
    if (!done && buf.trim()) frame(buf.trim());
  } finally {
    signal?.removeEventListener?.("abort", onAbort);
    if (!done) stop();
  }
  const toolCalls = [...calls.values()].sort((a, b) => a.index - b.index).map((c, i) => toolCallOf({ id: c.id, index: c.index, function: { name: c.name, arguments: c.arguments } }, i));
  const thinkingMs = thinkFrom == null ? 0 : (thinkTo ?? Date.now()) - thinkFrom;
  return { text, toolCalls, finish: finish ?? (toolCalls.length ? "tool_calls" : null), usage, logprobs, last, thinking, thinkingMs };
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
