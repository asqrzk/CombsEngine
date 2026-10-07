/**
 * @combs/sand — what leaves the program: output, and asking a person.
 *
 * `emit(piece)` is a piece of output for whoever listens — the `output` port in
 * process, a `sand.output` event otherwise (coalesced over 50 ms, so a token
 * stream is about twenty events a second rather than one per token). It is
 * sync and it never fails: a listener that throws is the listener's fault.
 *
 * `confirm(question, { wait })` asks a person and answers a verdict, so
 * `and`/`or`/`not` compose and an `unsure` block can catch "nobody was there".
 * The budget clock is paused for as long as the person takes, so a slow human
 * never spends the `ms` limit.
 */

import { SandError } from "./errors.js";
import { unsure, verdict, isVerdict } from "./kleene.js";
import { ctxNow, getCurrent, ports, warnOnce } from "./context.js";

const COALESCE_MS = 50;

/** The text of a piece: a string, a Reply, a Decision, anything with a `text`. */
export function pieceText(piece) {
  if (piece == null) return "";
  if (typeof piece === "string") return piece;
  if (piece === unsure) return "unsure";
  if (typeof piece === "object" && typeof piece.text === "string") return piece.text;
  if (typeof piece?.toString === "function" && piece.toString !== Object.prototype.toString) return String(piece);
  try { return JSON.stringify(piece, (_k, v) => (v === unsure ? "unsure" : v)) ?? ""; } catch { return String(piece); }
}

/** Sends what has piled up as one `sand.output` event. Called on the timer and at the end of a run. */
export function flushOutput(ctx) {
  if (!ctx?.output) return;
  const { buffer, timer, at } = ctx.output;
  if (timer != null) { clearTimeout(timer); ctx.output.timer = null; }
  if (!buffer.length) return;
  const text = buffer.join("");
  ctx.output.buffer = [];
  ctx.emit("sand.output", "event", { at: at ?? undefined, chars: text.length }, { output: text });
}

/** Sends what the model's reasoning has piled up as one `sand.thinking` event. */
export function flushThinking(ctx) {
  if (!ctx?.thinking) return;
  const { buffer, timer, at } = ctx.thinking;
  if (timer != null) { clearTimeout(timer); ctx.thinking.timer = null; }
  if (!buffer.length) return;
  const text = buffer.join("");
  ctx.thinking.buffer = [];
  ctx.emit("sand.thinking", "event", { at: at ?? undefined, chars: text.length }, { output: text });
}

/**
 * A piece of the model's reasoning, coalesced like output and never mixed with it: the pieces of an answer are sent
 * first, so the record reads in the order things happened. Sync; never fails.
 */
export function emitThinking(piece, opts = {}) {
  const text = typeof piece === "string" ? piece : "";
  const ctx = getCurrent();
  if (!ctx || !text) return;
  flushOutput(ctx);
  ctx.thinking ??= { buffer: [], timer: null, at: null };
  ctx.thinking.at = opts.at ?? ctx.thinking.at;
  ctx.thinking.buffer.push(text);
  if (ctx.thinking.timer == null) ctx.thinking.timer = setTimeout(() => { ctx.thinking.timer = null; flushThinking(ctx); }, COALESCE_MS);
}

/** A piece of output. Sync; never fails. */
export function emit(piece, opts = {}) {
  const text = pieceText(piece);
  const ctx = getCurrent();
  const sink = ctx?.outputPort ?? ports.output ?? null;
  if (sink) { try { sink(text, { at: opts.at ?? null }); } catch { /* a listener's fault is never the program's */ } }
  if (!ctx) return;
  ctx.output ??= { buffer: [], timer: null, at: null };
  if (!text) return;
  flushThinking(ctx);
  ctx.output.at = opts.at ?? ctx.output.at;
  ctx.output.buffer.push(text);
  if (ctx.output.timer == null) ctx.output.timer = setTimeout(() => { ctx.output.timer = null; flushOutput(ctx); }, COALESCE_MS);
}

/** Turns whatever a person's answer arrived as into a verdict. */
export function verdictOf(answer) {
  if (answer === undefined || answer === null) return unsure;
  if (isVerdict(answer)) return answer;
  if (typeof answer === "string") {
    const s = answer.trim().toLowerCase();
    if (["y", "yes", "true", "ok", "okay", "go", "1"].includes(s)) return true;
    if (["n", "no", "false", "stop", "0"].includes(s)) return false;
    return unsure;
  }
  if (typeof answer === "object" && "answer" in answer) return verdictOf(answer.answer);
  return verdict(answer);
}

/** A kind names what a `confirm` asks for beyond a verdict: a short name of letters, digits and underscores. */
const KIND = /^[a-z][a-z0-9_]{0,63}$/;
/** marks the runtime's own asks (the budget's question), which a program cannot raise under that kind */
const OWN = Symbol("sand.confirm.own");
/** The most an ask's arguments may weigh, as JSON: they travel to the person as one trace line. */
const ARGS_MAX = 48 * 1024;

/**
 * Asks a person. `true`, `false`, or `unsure` when nobody was there or the
 * wait elapsed. The budget clock pauses for the whole wait. With a `kind`,
 * the ask is for something, not only a verdict: the answer is then
 * `{answer, data?, by}`, `data` being what the person gave with it (a
 * reading from their computer, a command's result), never kept in the trace.
 * @param {string} question @param {{wait?: number, at?: string, kind?: string, args?: unknown}} opts
 */
export async function confirm(question, opts = {}) {
  const ctx = ctxNow(opts.at);
  const q = String(question ?? "").trim();
  if (!q) throw new SandError("E-STATE", "confirm needs a question to ask", { at: opts.at });
  const kind = opts.kind === undefined || opts.kind === null ? null : String(opts.kind);
  if (kind !== null && !KIND.test(kind)) throw new SandError("E-STATE", "confirm's kind is a short name of letters, digits and underscores, like link_reading", { at: opts.at });
  if (kind === "budget" && opts[OWN] !== true) throw new SandError("E-STATE", "the kind budget is the runtime's own question; a program names another kind", { at: opts.at });
  const args = kind === null ? undefined : opts.args === undefined ? null : opts.args;
  if (kind !== null && JSON.stringify(args).length > ARGS_MAX) throw new SandError("E-STATE", `confirm's args are at most ${ARGS_MAX} bytes of JSON`, { at: opts.at });
  const port = ctx.confirmPort ?? ports.confirm ?? null;
  const wait = opts.wait ?? ctx.budget.limits.wait ?? null;
  ctx.emit("sand.confirm", "span.start", { at: opts.at ?? undefined, ...(kind === null ? {} : { kind }) }, { input: kind === null ? q : { question: q, kind, args } });
  const t0 = Date.now();
  let answer = unsure, by = "nobody", data;
  ctx.budget.pause();
  try {
    if (!port) {
      warnOnce(ctx, `nobody:${opts.at ?? q}`, `W-NOBODY: nobody is here to answer "${q.slice(0, 80)}"; it is unsure. Answer it with sand run --yes/--no/--unsure, or give the run a confirm port.`);
    } else {
      let timer = null;
      const asked = Promise.resolve().then(() => port(q, { wait, at: opts.at ?? null, signal: ctx.signal ?? null, ...(kind === null ? {} : { kind, args }) }));
      const raced = wait == null ? asked : Promise.race([asked, new Promise((r) => { timer = setTimeout(() => r(TIMED_OUT), wait); })]);
      let got;
      try { got = await raced; } finally { if (timer != null) clearTimeout(timer); }
      if (got === TIMED_OUT) {
        warnOnce(ctx, `nobody:wait`, `W-NOBODY: nobody answered "${q.slice(0, 80)}" within ${wait} ms; it is unsure.`);
      } else {
        answer = verdictOf(got);
        by = (got && typeof got === "object" && typeof got.by === "string" ? got.by : null) ?? "port";
        if (kind !== null && got && typeof got === "object" && "data" in got) data = got.data;
      }
    }
  } finally {
    ctx.budget.resume();
  }
  const waited = Date.now() - t0;
  const dataBytes = data === undefined ? undefined : JSON.stringify(data).length;
  ctx.emit("sand.confirm", "span.end", {
    at: opts.at ?? undefined, answer: answer === unsure ? "unsure" : String(answer), waited, by,
    ...(kind === null ? {} : { kind }), ...(dataBytes === undefined ? {} : { data_bytes: dataBytes }),
  }, { input: q, output: answer === unsure ? "unsure" : answer });
  return kind === null ? answer : { answer, data, by };
}
const TIMED_OUT = Symbol("sand.confirm.timeout");

/**
 * The budget's question, a `confirm` of kind `budget`: the person sees what
 * was used and what one more block holds, and answers whether to go on.
 * @param {{limit: string, used: number, max: number, spent: object, block: object, extensions: number}} info
 */
export async function askToContinue(info, at = null) {
  const ctx = ctxNow(at);
  if (!(ctx.confirmPort ?? ports.confirm ?? null)) return null;   // nobody is here: not a no, just nobody
  const n = (v) => (typeof v === "number" ? v.toLocaleString("en-US") : String(v ?? "?"));
  const what = info.limit === "ms" ? `${Math.round(info.used / 1000)} of ${Math.round(info.max / 1000)} seconds` : `${n(info.used)} of ${n(info.max)} ${info.limit}`;
  const more = [
    info.block.tokens != null ? `${n(info.block.tokens)} tokens` : null,
    info.block.turns != null ? `${n(info.block.turns)} turns` : null,
    info.block.ms != null ? `${Math.round(info.block.ms / 1000)} seconds` : null,
  ].filter(Boolean).join(", ");
  const q = `This run has used its budget (${what}). Go on with as much again${more ? ` (${more})` : ""}?`;
  const got = await confirm(q, { kind: "budget", args: info, at: at ?? undefined, [OWN]: true });
  return got && typeof got === "object" && got.answer === true;
}
