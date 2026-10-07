/**
 * @combs/sand — chat and assistant: a conversation the program holds.
 *
 * `chat(system)` is a conversation with a chat model: one call per `say`, the
 * history kept, the `keep` most recent messages on the wire. `assistant(system,
 * { tools })` is the same with a tool loop inside every `say`: the model asks
 * for a tool, the program's own function runs — sequentially, so a tool may ask
 * a person — the result goes back as a `tool` message, and the round repeats up
 * to `settle.turns`. Past that bound the answer is `unsure`, `stopped: "turns"`;
 * a budget the person did not carry on ends the turn with what was said so far, `stopped: "budget"`.
 *
 * A model that thinks first hands its reasoning over apart from its answer: the
 * `Reply` carries it as `thinking` (and how long it took as `thinkingMs`), a
 * streamed `say` sends it as `sand.thinking` events ahead of the answer's
 * `sand.output`, and it is never put in the history, so a later turn does not
 * read the model's earlier thoughts back as if they were things said.
 *
 * A `Reply` is an object as a `Decision` is: `text` and `value`, plus what it
 * took. `toString()` and `toJSON()` are the text, so `"…{{reply}}…"`,
 * `emit(reply)`, `log(reply)` and a prose predicate on `{{reply}}` all read it.
 */

import { SandError } from "./errors.js";
import { unsure } from "./kleene.js";
import { ctxNow, resolveModel, warnOnce } from "./context.js";
import { emit as emitPiece, emitThinking } from "./effects.js";
import { toolsOf } from "./tools.js";

const DEFAULT_KEEP = 40;
/** A host's refusal of the model's own tool call: its arguments were not JSON it could read. */
const BAD_CALL = /tool[ _-]?call[\s\S]{0,80}(parse|json)|(parse|json)[\s\S]{0,80}tool[ _-]?call/i;

const DEFAULT_TURNS = 8;

export class Reply {
  constructor({ text = "", stopped = null, turns = 0, tokens = 0, calls = [], reason = null, finish = null, model = null, thinking = "", thinkingMs = 0 } = {}) {
    this.text = text;
    this.stopped = stopped;
    this.value = stopped === "turns" || stopped === "empty" || stopped === "aborted" || ((stopped === "budget" || stopped === "tool_call") && !text) ? unsure : text;
    this.turns = turns;
    this.tokens = tokens;
    this.calls = calls;
    this.reason = reason;
    this.finish = finish;
    this.model = model;
    this.thinking = thinking;
    this.thinkingMs = thinkingMs;
  }
  toString() { return this.text; }
  toJSON() { return this.text; }
}

/** The messages that go on the wire: the system line and the `keep` most recent, kept whole. */
export function onWire(system, messages, keep) {
  const n = Math.max(1, keep ?? DEFAULT_KEEP);
  let from = Math.max(0, messages.length - n);
  // never cut a `tool` answer away from the assistant turn that asked for it
  while (from > 0 && messages[from]?.role === "tool") from--;
  return [...(system ? [{ role: "system", content: system }] : []), ...messages.slice(from)];
}

export class Conversation {
  #messages = [];
  #last = null;
  #loaded = false;
  constructor(system, opts = {}, tools = null) {
    this.system = typeof system === "string" ? system : String(system?.text ?? system ?? "");
    this.name = opts.name ?? null;
    this.at = opts.at ?? null;
    this.model = opts.model ?? null;
    this.keep = opts.keep ?? DEFAULT_KEEP;
    this.tools = tools;
    this.settle = { turns: opts.settle?.turns ?? DEFAULT_TURNS, ...(opts.settle ?? {}) };
    this.maxTokens = opts.maxTokens ?? null;
    this.temperature = opts.temperature ?? null;
  }

  /** Every message this conversation holds, oldest first (the system line is not one). */
  get history() { return this.#messages.map((m) => ({ ...m })); }
  /** The last `Reply`, or null before the first `say`. */
  get last() { return this.#last; }
  /** Drops the history; the system line and the tools stay. */
  forget() { this.#messages = []; this.#last = null; this.#loaded = true; return this; }

  /** Loads this conversation's half of a session, once, before the first turn. */
  #load(ctx) {
    if (this.#loaded) return;
    this.#loaded = true;
    const key = this.#key(ctx);
    ctx.chats?.set(key, this);
    const held = ctx.session?.state?.chats?.[key];
    if (Array.isArray(held) && held.length) this.#messages = held.map((m) => ({ ...m }));
  }
  #key(ctx) {
    if (this.name) return this.name;
    this.name = `chat${(ctx.chats?.size ?? 0) + 1}`;
    return this.name;
  }

  /** One turn's worth of message-passing, up to `settle.turns` rounds of tools. */
  async say(text, opts = {}) {
    const ctx = ctxNow(opts.at ?? this.at);
    const said = typeof text === "string" ? text : String(text?.text ?? text ?? "");
    if (!said.trim()) throw new SandError("E-STATE", "say needs something to say", { at: opts.at ?? this.at });
    this.#load(ctx);
    const at = opts.at ?? this.at ?? undefined;
    const { name, adapter } = resolveModel(ctx, { model: opts.model ?? this.model, at }, "chat");
    if (typeof adapter.complete !== "function") throw new SandError("E-MODEL", `\`${name}\` cannot hold a conversation; it is a ${adapter.kind} model`, { at });
    const streaming = !!(opts.emit || opts.onDelta || opts.stream) && adapter.caps?.stream !== false;
    const onDelta = streaming ? (piece) => { if (opts.emit) emitPiece(piece, { at }); opts.onDelta?.(piece); } : null;
    const maxTurns = Math.max(1, opts.settle?.turns ?? this.settle.turns ?? DEFAULT_TURNS);
    const pending = [{ role: "user", content: said }];
    const calls = [];
    let tokens = 0, turns = 0, answer = "", finish = null, stopped = null, reason = null, partial = "";
    const keepPartial = (piece) => { partial += piece; };
    let thinking = "", thinkingMs = 0, thoughtNow = "";
    const onThought = streaming ? (piece) => { thoughtNow += piece; if (opts.emit) emitThinking(piece, { at }); opts.onThinking?.(piece); } : null;
    // a tool call the host could not read (its arguments were not JSON) is told to the model once, on the wire of the
    // next round only: the note is never part of the conversation
    let nudge = null, badCalls = 0;
    // what the last finished round said (already in the history when it asked for tools): a stop keeps it, once
    let lastText = "", kept = false;

    for (let round = 0; round < (this.tools ? maxTurns : 1); round++) {
      try {
        await ctx.budget.check({ turns: 1, tokens: 0 }, at);
      } catch (e) {
        // past the budget after a round or more: what was paid for is kept, and the reply says it stopped
        if (e?.code === "E-BUDGET" && turns > 0) { answer = lastText; kept = true; stopped = "budget"; reason = e.message; break; }
        throw e;
      }
      const wire = onWire(this.system, [...this.#messages, ...pending, ...(nudge ? [nudge] : [])], this.keep);
      const t0 = Date.now();
      let r;
      thoughtNow = "";
      partial = "";
      try {
        r = await ctx.budget.gate.run(() => adapter.complete({
          messages: wire,
          tools: this.tools ? [...this.tools.values()].map((t) => t.toSchema()) : null,
          stream: streaming,
          maxTokens: opts.maxTokens ?? this.maxTokens ?? 1024,
          temperature: opts.temperature ?? this.temperature ?? 0.2,
        }, { at, signal: opts.signal ?? ctx.signal, onDelta: onDelta ? (p) => { keepPartial(p); onDelta(p); } : null, onThinking: onThought }));
      } catch (e) {
        ctx.emit("sand.request", "event", { status: "error", model: name, road: "", questions: 0, ms: Date.now() - t0, verb: "say", at }, { input: wire, status: "error", error: String(e?.message ?? e) });
        if (opts.abortIsAnswer && (e?.code === "E-TIMEOUT" || e?.name === "AbortError")) {
          answer = partial; stopped = "aborted"; reason = "the reply was stopped before it finished"; turns++;
          thinking += (thinking && thoughtNow ? "\n\n" : "") + thoughtNow;
          break;
        }
        if (this.tools && BAD_CALL.test(String(e?.message ?? ""))) {
          const said = String(e?.message ?? "").replace(/\s*Nothing was charged\.\s*$/, "").slice(0, 300);
          if (badCalls === 0 && round + 1 < maxTurns) {
            badCalls++;
            nudge = { role: "user", content: `Your last tool call could not be read by the host: ${said}. Call the tool again with arguments that are valid JSON: one string per argument, quotes escaped, line breaks written as \\n and never raw, nothing outside the JSON; or answer without the tool.` };
            warnOnce(ctx, `badcall:${at ?? ""}`, `W-TOOL-CALL: ${name}'s tool call could not be read by the host (${said.slice(0, 120)}); it was asked once more`);
            continue;
          }
          // told once already and wrong again: the reply ends with what it has, and says why, instead of failing the run
          answer = lastText; kept = true; stopped = "tool_call"; reason = `the model wrote a tool call the host could not read, twice: ${said}`;
          break;
        }
        throw e;
      }
      nudge = null;
      turns++;
      thinking += (thinking && r.thinking ? "\n\n" : "") + (r.thinking ?? "");
      thinkingMs += r.thinkingMs ?? 0;
      tokens += r.tokens ?? 0;
      ctx.budget.record({ turns: 1, tokens: r.tokens ?? 0, completions: 1 });
      if (r.estimated) warnOnce(ctx, `unmetered:${name}`, `W-UNMETERED: ${name} reported no usage; tokens are estimated at one per four characters`);
      ctx.emit("sand.request", "event", { status: "ok", model: name, road: "", questions: 0, tokens: r.tokens ?? 0, ms: r.ms, verb: "say", at }, { input: wire, output: r.text });
      ctx.emit("sand.turn", "event", { at, chat: this.name ?? "", turn: turns, model: name, stop: r.finish ?? "", tokens: r.tokens ?? 0, ms: r.ms, calls: r.toolCalls?.length ?? 0 }, { output: r.text });
      finish = r.finish ?? null;
      lastText = r.text ?? "";
      if (this.tools && r.toolCalls?.length) {
        pending.push({ role: "assistant", content: r.text || null, tool_calls: r.toolCalls.map((c) => ({ id: c.id, type: "function", function: { name: c.name, arguments: c.arguments } })) });
        for (const call of r.toolCalls) {
          const done = await this.#runTool(ctx, call, at);
          calls.push(done.record);
          pending.push({ role: "tool", tool_call_id: call.id, name: call.name, content: done.content });
        }
        continue;
      }
      answer = r.text ?? "";
      break;
    }

    if (!stopped && this.tools && !answer && calls.length && turns >= maxTurns) {
      stopped = "turns";
      reason = `the assistant asked for a tool ${turns} times without answering; raise settle({ turns })`;
    }
    if (!stopped && finish === "length") { stopped = "length"; reason = "the model reached its token ceiling; raise maxTokens"; }
    if (!stopped && !answer.trim()) { stopped = "empty"; reason = "the model answered with nothing"; }
    if (answer && !kept) pending.push({ role: "assistant", content: answer });
    this.#messages.push(...pending);
    const reply = new Reply({ text: answer, stopped, turns, tokens, calls, reason, finish, model: name, thinking, thinkingMs });
    this.#last = reply;
    return reply;
  }

  /** The reply as it arrives. Breaking the loop stops the model. */
  async *stream(text, opts = {}) {
    const ctrl = new AbortController();
    const queue = [];
    let wake = null, done = false, failed = null;
    const wakeUp = () => { const w = wake; wake = null; w?.(); };
    const running = this.say(text, { ...opts, stream: true, signal: ctrl.signal, abortIsAnswer: true, onDelta: (p) => { queue.push(p); wakeUp(); } })
      .then(() => {}, (e) => { failed = e; })
      .finally(() => { done = true; wakeUp(); });
    try {
      while (true) {
        while (queue.length) yield queue.shift();
        if (done) break;
        await new Promise((r) => { wake = r; });
      }
      if (failed) throw failed;
    } finally {
      if (!done) { ctrl.abort(); try { await running; } catch { /* the abort is ours */ } }
    }
  }

  async #runTool(ctx, call, at) {
    const t0 = Date.now();
    const tool = this.tools.get(call.name);
    ctx.emit("sand.tool", "span.start", { at, chat: this.name ?? "", tool: call.name, id: call.id }, { input: call.args ?? call.arguments });
    const fail = (message, status) => {
      const ms = Date.now() - t0;
      ctx.emit("sand.tool", "span.end", { at, chat: this.name ?? "", tool: call.name, id: call.id, status, ms }, { status: "error", error: message, output: { error: message } });
      return { record: { id: call.id, name: call.name, args: call.args, result: null, error: message, ms }, content: JSON.stringify({ error: message }) };
    };
    if (!tool) return fail(`there is no tool named ${JSON.stringify(call.name)}; this conversation has ${[...this.tools.keys()].join(", ")}`, "unknown");
    if (call.args == null) return fail(`the arguments were not JSON: ${String(call.arguments).slice(0, 120)}`, "bad-arguments");
    let result;
    try { result = await tool.run(call.args); }
    catch (e) {
      // a budget stop, a timeout and an abort are the run's, not the tool's
      if (e?.code === "E-BUDGET" || e?.code === "E-TIMEOUT" || e?.name === "AbortError") {
        ctx.emit("sand.tool", "span.end", { at, chat: this.name ?? "", tool: call.name, id: call.id, status: "stopped", ms: Date.now() - t0 }, { status: "error", error: String(e?.message ?? e) });
        throw e;
      }
      return fail(String(e?.message ?? e), "error");
    }
    const ms = Date.now() - t0;
    const content = typeof result === "string" ? result : (() => { try { return JSON.stringify(result === undefined ? null : result, (_k, v) => (v === unsure ? "unsure" : v)); } catch { return String(result); } })();
    ctx.emit("sand.tool", "span.end", { at, chat: this.name ?? "", tool: call.name, id: call.id, status: "ok", ms }, { input: call.args, output: result === undefined ? null : result });
    return { record: { id: call.id, name: call.name, args: call.args, result, error: null, ms }, content };
  }
}

/** A conversation with a chat model. Sync: the first call to `say` reaches the model. */
export function chat(system, opts = {}) {
  if (opts.tools !== undefined) throw new SandError("E-STATE", "chat takes no tools; use assistant", { at: opts.at });
  return new Conversation(system, opts, null);
}

/** A conversation whose every `say` may run the tools it was given. */
export function assistant(system, opts = {}) {
  return new Conversation(system, opts, toolsOf(opts.tools, opts.at));
}
