/**
 * @combs/sand — the budget and the concurrency gate.
 *
 * Every model call passes `check()` before it leaves and `record()` when it
 * returns. Limits come from `sand.json` (`limits`) narrowed by the program's own
 * `settle({ passes, tokens, ms })`; a program can lower a limit, never raise
 * it. An overrun found after the fact refuses the *next* call; nothing is
 * clawed back.
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
  /** @param {{passes?: number|null, tokens?: number|null, ms?: number|null, concurrency?: number}} limits */
  constructor(limits = {}, emit = null) {
    this.limits = { passes: limits.passes ?? null, tokens: limits.tokens ?? null, ms: limits.ms ?? null, concurrency: limits.concurrency ?? 4 };
    this.spent = { passes: 0, tokens: 0, requests: 0, completions: 0 };
    this.started = Date.now();
    this.gate = new Semaphore(this.limits.concurrency);
    this.emit = emit;
    this.stopped = null;
  }
  /** Narrows the limits from a program's settle(); a program can only lower them. */
  narrow({ passes, tokens, ms } = {}) {
    const lower = (a, b) => (b == null ? a : a == null ? b : Math.min(a, b));
    this.limits.passes = lower(this.limits.passes, passes);
    this.limits.tokens = lower(this.limits.tokens, tokens);
    this.limits.ms = lower(this.limits.ms, ms);
  }
  elapsed() { return Date.now() - this.started; }
  /** Refuses before a call when the estimate would cross a limit. */
  check(estimate = {}, at = null) {
    const over = (name, spentV, est, cap) => {
      if (cap == null) return;
      if (spentV + (est ?? 0) > cap) {
        this.stopped = { limit: name, used: spentV, max: cap };
        this.emit?.("sand.budget", "event", { limit: name, used: String(spentV), max: String(cap), action: "stop", at: at ?? "" });
        throw new SandError("E-BUDGET", `${name} ${spentV + (est ?? 0)}/${cap} spent; raise settle({ ${name} }) or bound the loop`, { at: at ?? undefined });
      }
    };
    if (this.limits.ms != null && this.elapsed() > this.limits.ms) {
      this.stopped = { limit: "ms", used: this.elapsed(), max: this.limits.ms };
      this.emit?.("sand.budget", "event", { limit: "ms", used: String(this.elapsed()), max: String(this.limits.ms), action: "stop", at: at ?? "" });
      throw new SandError("E-BUDGET", `ms ${this.elapsed()}/${this.limits.ms} elapsed; raise settle({ ms }) or do less`, { at: at ?? undefined });
    }
    over("passes", this.spent.passes, estimate.passes, this.limits.passes);
    over("tokens", this.spent.tokens, estimate.tokens, this.limits.tokens);
  }
  record({ passes = 0, tokens = 0, completions = 0 } = {}) {
    this.spent.passes += passes; this.spent.tokens += tokens; this.spent.requests += 1; this.spent.completions += completions;
    const warnAt = (name, v, cap) => { if (cap != null && v > cap * 0.8 && v <= cap) this.emit?.("sand.budget", "event", { limit: name, used: String(v), max: String(cap), action: "warn" }); };
    warnAt("passes", this.spent.passes, this.limits.passes); warnAt("tokens", this.spent.tokens, this.limits.tokens);
  }
  snapshot() { return { limits: { ...this.limits }, spent: { ...this.spent }, ms: this.elapsed(), stopped: this.stopped }; }
}
