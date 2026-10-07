/**
 * @combs/sand — the budget and the concurrency gate.
 *
 * Every model call passes `check()` before it leaves and `record()` when it
 * returns. Limits come from `sand.json` (`limits`) narrowed by the program's own
 * `settle({ passes, tokens, ms, turns, wait })`; a program can lower a limit,
 * never raise it. An overrun found after the fact refuses the *next* call;
 * nothing is clawed back. A chat turn counts against `turns`. While a person is
 * being asked the clock is paused (`pause()`/`resume()`), so a slow human never
 * spends the `ms` limit; what they took is reported as `waited`.
 *
 * When a limit is reached and someone can be asked (`asker`, set by the
 * runtime to a `confirm` of kind `budget`), the person is offered the same
 * budget again, every limit by its first size; on yes the run goes on, on no
 * or no answer it stops as before. Tokens are counted as billed: a token the
 * host read from its cache counts a tenth (see wires/chat.js).
 */

import { SandError } from "./errors.js";

/** Counting semaphore (after @combs/runtime src/primitives.ts:72). */
export class Semaphore {
  #waiting = [];
  constructor(permits) { this.permits = permits; }
  async acquire() {
    if (this.permits > 0) { this.permits--; return; }
    await new Promise((resolve) => this.#waiting.push(resolve));
  }
  release() {
    const next = this.#waiting.shift();
    if (next) next(); else this.permits++;
  }
  async run(fn) { await this.acquire(); try { return await fn(); } finally { this.release(); } }
}

export class Budget {
  #pausedAt = null;
  #depth = 0;
  /** @param {{passes?: number|null, tokens?: number|null, ms?: number|null, turns?: number|null, wait?: number|null, concurrency?: number}} limits */
  constructor(limits = {}, emit = null) {
    this.limits = { passes: limits.passes ?? null, tokens: limits.tokens ?? null, ms: limits.ms ?? null, turns: limits.turns ?? null, wait: limits.wait ?? null, concurrency: limits.concurrency ?? 4 };
    this.spent = { passes: 0, tokens: 0, requests: 0, completions: 0, turns: 0 };
    this.started = Date.now();
    this.waited = 0;
    this.gate = new Semaphore(this.limits.concurrency);
    this.emit = emit;
    this.stopped = null;
    /** asks the person whether to go on: `(info, at) => Promise<boolean>`; nobody by default */
    this.asker = null;
    /** the limits as they first stood, the block each extension adds */
    this.block = null;
    this.extensions = 0;
  }
  /** Narrows the limits from a program's settle(); a program can only lower them. */
  narrow({ passes, tokens, ms, turns, wait } = {}) {
    const lower = (a, b) => (b == null ? a : a == null ? b : Math.min(a, b));
    this.limits.passes = lower(this.limits.passes, passes);
    this.limits.tokens = lower(this.limits.tokens, tokens);
    this.limits.ms = lower(this.limits.ms, ms);
    this.limits.turns = lower(this.limits.turns, turns);
    this.limits.wait = lower(this.limits.wait, wait);
  }
  /** Stops the clock while a person is being asked; nested waits are counted once. */
  pause() { if (this.#depth++ === 0) this.#pausedAt = Date.now(); }
  resume() {
    if (this.#depth === 0) return;
    if (--this.#depth === 0 && this.#pausedAt != null) { this.waited += Date.now() - this.#pausedAt; this.#pausedAt = null; }
  }
  /** Wall clock since the run began, minus every ms a person was being waited on. */
  elapsed() {
    const paused = this.#pausedAt != null ? Date.now() - this.#pausedAt : 0;
    return Date.now() - this.started - this.waited - paused;
  }
  /** The first limit the estimate would cross, or null. */
  #over(estimate) {
    if (this.limits.ms != null && this.elapsed() > this.limits.ms) return { limit: "ms", used: this.elapsed(), max: this.limits.ms };
    for (const name of ["passes", "tokens", "turns"]) {
      const cap = this.limits[name];
      if (cap != null && this.spent[name] + (estimate[name] ?? 0) > cap) return { limit: name, used: this.spent[name] + (estimate[name] ?? 0), max: cap };
    }
    return null;
  }
  /**
   * Refuses before a call when the estimate would cross a limit. With an
   * `asker`, the person is asked first whether to go on with as much again.
   */
  async check(estimate = {}, at = null) {
    let hit = this.#over(estimate);
    while (hit && this.asker) {
      const block = this.block ??= { passes: this.limits.passes, tokens: this.limits.tokens, ms: this.limits.ms, turns: this.limits.turns };
      const info = {
        limit: hit.limit, used: hit.used, max: hit.max,
        spent: { ...this.spent, ms: this.elapsed() },
        block: { passes: block.passes, tokens: block.tokens, turns: block.turns, ms: block.ms },
        extensions: this.extensions,
      };
      this.emit?.("sand.budget", "event", { limit: hit.limit, used: String(hit.used), max: String(hit.max), action: "ask", at: at ?? "" });
      let yes = false;
      try { yes = (await this.asker(info, at)) === true; } catch { yes = false; }
      if (!yes) break;
      for (const name of ["passes", "tokens", "ms", "turns"]) {
        if (this.limits[name] != null && block[name] != null) this.limits[name] += block[name];
      }
      this.extensions += 1;
      this.emit?.("sand.budget", "event", { limit: hit.limit, used: String(hit.used), max: String(this.limits[hit.limit]), action: "extended", at: at ?? "" });
      hit = this.#over(estimate);
    }
    if (!hit) return;
    this.stopped = { limit: hit.limit, used: hit.used, max: hit.max };
    this.emit?.("sand.budget", "event", { limit: hit.limit, used: String(hit.used), max: String(hit.max), action: "stop", at: at ?? "" });
    const how = hit.limit === "ms" ? `ms ${hit.used}/${hit.max} elapsed; raise settle({ ms }) or do less` : `${hit.limit} ${hit.used}/${hit.max} spent; raise settle({ ${hit.limit} }) or bound the loop`;
    throw new SandError("E-BUDGET", how + (this.asker ? " (the person did not carry it on)" : ""), { at: at ?? undefined });
  }
  record({ passes = 0, tokens = 0, completions = 0, turns = 0 } = {}) {
    this.spent.passes += passes; this.spent.tokens += tokens; this.spent.requests += 1; this.spent.completions += completions; this.spent.turns += turns;
    const warnAt = (name, v, cap) => { if (cap != null && v > cap * 0.8 && v <= cap) this.emit?.("sand.budget", "event", { limit: name, used: String(v), max: String(cap), action: "warn" }); };
    warnAt("passes", this.spent.passes, this.limits.passes); warnAt("tokens", this.spent.tokens, this.limits.tokens); warnAt("turns", this.spent.turns, this.limits.turns);
  }
  snapshot() { return { limits: { ...this.limits }, spent: { ...this.spent }, ms: this.elapsed(), waited: this.waited, stopped: this.stopped, extensions: this.extensions }; }
}
