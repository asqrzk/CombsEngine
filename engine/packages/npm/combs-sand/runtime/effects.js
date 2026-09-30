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

/** A piece of output. Sync; never fails. */
export function emit(piece, opts = {}) {
  const text = pieceText(piece);
  const ctx = getCurrent();
  const sink = ctx?.outputPort ?? ports.output ?? null;
  if (sink) { try { sink(text, { at: opts.at ?? null }); } catch { /* a listener's fault is never the program's */ } }
  if (!ctx) return;
  ctx.output ??= { buffer: [], timer: null, at: null };
  if (!text) return;
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

/**
 * Asks a person. `true`, `false`, or `unsure` when nobody was there or the
 * wait elapsed. The budget clock pauses for the whole wait.
 * @param {string} question @param {{wait?: number, at?: string}} opts
 */
export async function confirm(question, opts = {}) {
  const ctx = ctxNow(opts.at);
  const q = String(question ?? "").trim();
  if (!q) throw new SandError("E-STATE", "confirm needs a question to ask", { at: opts.at });
  const port = ctx.confirmPort ?? ports.confirm ?? null;
  const wait = opts.wait ?? ctx.budget.limits.wait ?? null;
  ctx.emit("sand.confirm", "span.start", { at: opts.at ?? undefined }, { input: q });
  const t0 = Date.now();
  let answer = unsure, by = "nobody";
  ctx.budget.pause();
  try {
    if (!port) {
      warnOnce(ctx, `nobody:${opts.at ?? q}`, `W-NOBODY: nobody is here to answer "${q.slice(0, 80)}"; it is unsure. Answer it with sand run --yes/--no/--unsure, or give the run a confirm port.`);
    } else {
      let timer = null;
      const asked = Promise.resolve().then(() => port(q, { wait, at: opts.at ?? null, signal: ctx.signal ?? null }));
      const raced = wait == null ? asked : Promise.race([asked, new Promise((r) => { timer = setTimeout(() => r(TIMED_OUT), wait); })]);
      let got;
      try { got = await raced; } finally { if (timer != null) clearTimeout(timer); }
      if (got === TIMED_OUT) {
        warnOnce(ctx, `nobody:wait`, `W-NOBODY: nobody answered "${q.slice(0, 80)}" within ${wait} ms; it is unsure.`);
      } else {
        answer = verdictOf(got);
        by = (got && typeof got === "object" && typeof got.by === "string" ? got.by : null) ?? "port";
      }
    }
  } finally {
    ctx.budget.resume();
  }
  const waited = Date.now() - t0;
  ctx.emit("sand.confirm", "span.end", { at: opts.at ?? undefined, answer: answer === unsure ? "unsure" : String(answer), waited, by }, { input: q, output: answer === unsure ? "unsure" : answer });
  return answer;
}
const TIMED_OUT = Symbol("sand.confirm.timeout");
