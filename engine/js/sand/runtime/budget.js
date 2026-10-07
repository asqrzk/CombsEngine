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
 * Tokens are counted as billed: a token the host read from its cache counts a
 * tenth (see wires/chat.js). Credits are counted when the platform says what a
 * token costs (`limits.rates` per model, `limits.rate` for the rest): a pass is
 * a credit, a chat call its tokens at the rate, whole and at least one.
 *
 * When a limit is reached and someone can be asked (`asker`, given with the
 * budget: a `confirm` of kind `budget`), the person is asked whether to go on.
 * A yes to a credits question moves the credits cap to the next step of
 * `creditSteps` above what is spent (then double), and nothing else; a yes to
 * any other limit gives every limit its first size again. A no is remembered for
 * that limit alone; nobody to ask is the plain stop. `askMore()` lets another
 * cap, such as a conversation's rounds, raise the same question.
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
  /**
   * @param {{passes?: number|null, tokens?: number|null, ms?: number|null, turns?: number|null, wait?: number|null, concurrency?: number}} limits
   * @param {((name: string, kind: string, attrs: object) => void) | null} emit
   * @param {((info: object, at: string|null) => Promise<boolean|null>) | null} asker  asks the person to go on: true, false, or null for nobody
   */
  constructor(limits = {}, emit = null, asker = null) {
    this.limits = { passes: limits.passes ?? null, tokens: limits.tokens ?? null, ms: limits.ms ?? null, turns: limits.turns ?? null, wait: limits.wait ?? null, concurrency: limits.concurrency ?? 4 };
    // credits, when the platform says what a token costs: the first question at `credits`, the next ones at each
    // of `creditSteps`, then at double the last; `rates` maps a model to its tokens per credit, `rate` is the rest's
    this.limits.credits = limits.credits ?? null;
    this.limits.creditSteps = Array.isArray(limits.creditSteps) ? [...limits.creditSteps].filter((n) => typeof n === "number" && n > 0).sort((a, b) => a - b) : [];
    this.limits.rates = limits.rates && typeof limits.rates === "object" ? { ...limits.rates } : {};
    this.limits.rate = typeof limits.rate === "number" && limits.rate > 0 ? limits.rate : null;
    this.spent = { passes: 0, tokens: 0, requests: 0, completions: 0, turns: 0, credits: 0 };
    this.started = Date.now();
    this.waited = 0;
    this.gate = new Semaphore(this.limits.concurrency);
    this.emit = emit;
    this.stopped = null;
    /** asks the person whether to go on: `(info, at) => Promise<boolean|null>`; nobody by default */
    this.asker = typeof asker === "function" ? asker : null;
    /** the limits as they first stood, the block each extension adds */
    this.block = null;
    this.extensions = 0;
    /** the person said no once: that limit is not asked again this run (another limit still may be) */
    this.declined = false;
    this.#noTo = new Set();
    this.#warned = new Set();
    /** the question in flight, so two calls at a limit raise one question and both read its answer */
    this.#asking = null;
  }
  #asking;
  #noTo;
  #warned;
  /** Narrows the limits from a program's settle(); a program can only lower them. */
  narrow({ passes, tokens, ms, turns, wait } = {}) {
    const lower = (a, b) => (b == null ? a : a == null ? b : Math.min(a, b));
    this.limits.passes = lower(this.limits.passes, passes);
    this.limits.tokens = lower(this.limits.tokens, tokens);
    this.limits.ms = lower(this.limits.ms, ms);
    this.limits.turns = lower(this.limits.turns, turns);
    this.limits.wait = lower(this.limits.wait, wait);
    if (this.block) for (const k of ["passes", "tokens", "ms", "turns"]) this.block[k] = lower(this.block[k], { passes, tokens, ms, turns }[k]);
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
    // credits are counted after the fact: the question comes before the next call once a step is passed
    if (this.limits.credits != null && this.spent.credits > this.limits.credits) return { limit: "credits", used: Math.round(this.spent.credits), max: this.limits.credits };
    for (const name of ["passes", "tokens", "turns"]) {
      const cap = this.limits[name];
      if (cap != null && this.spent[name] + (estimate[name] ?? 0) > cap) return { limit: name, used: this.spent[name] + (estimate[name] ?? 0), max: cap };
    }
    return null;
  }
  /**
   * What the credits cap becomes after a yes: the first step of the schedule above what is spent (one big call may
   * pass several steps at once, and the person is asked once for it), else double until it is above.
   */
  #nextCredits() {
    const need = Math.max(this.limits.credits ?? 0, this.spent.credits);
    const step = this.limits.creditSteps.find((n) => n > need);
    if (step !== undefined) return step;
    let next = Math.max(1, this.limits.credits ?? 1) * 2;
    while (next <= need) next *= 2;
    return next;
  }
  /**
   * Asks the person whether to go on past a cap that is not the budget's own (a conversation's rounds): the same
   * question, the same single flight, the same memory of a no. Answers whether to go on; the caller extends its cap.
   */
  async askMore(info, at = null) {
    if (!this.asker || this.#noTo.has(info.limit)) return false;
    while (this.#asking) await this.#asking;
    if (this.#noTo.has(info.limit)) return false;
    this.emit?.("sand.budget", "event", { limit: info.limit, used: String(info.used), max: String(info.max), action: "ask", at: at ?? "" });
    this.#asking = (async () => { try { return await this.asker({ ...info, spent: { ...this.spent, ms: this.elapsed() }, extensions: this.extensions }, at); } catch { return false; } })();
    let got;
    try { got = await this.#asking; } finally { this.#asking = null; }
    if (got === null) return false;
    if (got !== true) { this.declined = true; this.#noTo.add(info.limit); return false; }
    this.extensions += 1;
    this.emit?.("sand.budget", "event", { limit: info.limit, used: String(info.used), max: String(info.max + (info.block?.[info.limit] ?? 0)), action: "extended", at: at ?? "" });
    return true;
  }
  /**
   * Refuses before a call when the estimate would cross a limit. With an
   * `asker`, the person is asked first whether to go on with as much again.
   */
  async check(estimate = {}, at = null) {
    let hit = this.#over(estimate);
    let nobody = false;
    while (hit && this.asker && !this.#noTo.has(hit.limit)) {
      // one question at a time: a second call that meets the limit waits for the first's answer, then looks again
      if (this.#asking) { await this.#asking; hit = this.#over(estimate); continue; }
      const block = this.block ??= { passes: this.limits.passes, tokens: this.limits.tokens, ms: this.limits.ms, turns: this.limits.turns };
      // a limit that was unset at the first ask (set by settle() since) gets a block of its own size, so a yes always buys something
      if (hit.limit !== "credits" && block[hit.limit] == null && this.limits[hit.limit] != null) block[hit.limit] = this.limits[hit.limit];
      // a credits question names the next step and nothing else: a yes to it moves no other cap, and no clock
      const next = hit.limit === "credits" ? this.#nextCredits() : null;
      const info = {
        limit: hit.limit, used: hit.used, max: hit.max,
        spent: { ...this.spent, ms: this.elapsed() },
        block: next !== null ? { credits: next } : { passes: block.passes, tokens: block.tokens, turns: block.turns, ms: block.ms },
        extensions: this.extensions,
      };
      this.emit?.("sand.budget", "event", { limit: hit.limit, used: String(hit.used), max: String(hit.max), action: "ask", at: at ?? "" });
      this.#asking = (async () => { try { return await this.asker(info, at); } catch { return false; } })();
      let got;
      try { got = await this.#asking; } finally { this.#asking = null; }
      if (got === null) { nobody = true; break; }   // no one could be asked: the plain stop, as without an asker
      if (got !== true) { this.declined = true; this.#noTo.add(hit.limit); break; }
      this.#warned.delete(hit.limit);
      if (next !== null) {
        // the schedule: the step the person was told, not another block; the other caps stand where they are
        this.limits.credits = next;
      } else {
        for (const name of ["passes", "tokens", "ms", "turns"]) {
          if (this.limits[name] != null && block[name] != null) this.limits[name] += block[name];
        }
      }
      this.extensions += 1;
      this.emit?.("sand.budget", "event", { limit: hit.limit, used: String(hit.used), max: String(this.limits[hit.limit]), action: "extended", at: at ?? "" });
      hit = this.#over(estimate);
    }
    if (!hit) return;
    this.stopped = { limit: hit.limit, used: hit.used, max: hit.max };
    this.emit?.("sand.budget", "event", { limit: hit.limit, used: String(hit.used), max: String(hit.max), action: "stop", at: at ?? "" });
    const how = hit.limit === "ms"
      ? `ms ${hit.used}/${hit.max} elapsed; raise settle({ ms }) or do less`
      : hit.limit === "credits"
      ? `credits ${hit.used}/${hit.max} spent`
      : `${hit.limit} ${hit.used}/${hit.max} spent; raise settle({ ${hit.limit} }) or bound the loop`;
    throw new SandError("E-BUDGET", how + (this.asker && !nobody ? " (the person did not carry it on)" : ""), { at: at ?? undefined });
  }
  /** The tokens one credit buys from a model, when the platform said; null when credits are not counted. */
  rateFor(model) {
    const r = model != null ? this.limits.rates[model] : undefined;
    return typeof r === "number" && r > 0 ? r : this.limits.rate;
  }
  record({ passes = 0, tokens = 0, completions = 0, turns = 0, model = null, credits = 0 } = {}) {
    this.spent.passes += passes; this.spent.tokens += tokens; this.spent.requests += 1; this.spent.completions += completions; this.spent.turns += turns;
    // a decision or a memory call is a credit a pass; a chat call is its tokens at the model's rate, whole and at
    // least one, as the platform bills it
    const rate = this.rateFor(model);
    this.spent.credits += passes + credits + (rate && tokens > 0 ? Math.max(1, Math.ceil(tokens / rate)) : 0);
    const warnAt = (name, v, cap) => {
      if (cap == null || !(v > cap * 0.8 && v <= cap) || this.#warned.has(name)) return;
      this.#warned.add(name);
      this.emit?.("sand.budget", "event", { limit: name, used: String(v), max: String(cap), action: "warn" });
    };
    warnAt("passes", this.spent.passes, this.limits.passes); warnAt("tokens", this.spent.tokens, this.limits.tokens); warnAt("turns", this.spent.turns, this.limits.turns);
    warnAt("credits", Math.round(this.spent.credits), this.limits.credits);
  }
  snapshot() { return { limits: { ...this.limits }, spent: { ...this.spent }, ms: this.elapsed(), waited: this.waited, stopped: this.stopped, extensions: this.extensions, declined: this.declined }; }
}
