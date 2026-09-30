/**
 * @combs/sand — the run context every verb reads.
 *
 * One object per run (config, budget, batcher, memo, trace, session) and one
 * bag of ports for how the runtime reaches its host (env, files, fetch, the log
 * sink, the person who confirms, where output goes). This module holds nothing
 * but that state and the two lookups over it, so `verbs.js`, `conversation.js`,
 * `tools.js` and the wires can share it without importing each other.
 */

import { SandError } from "./errors.js";

/** How the runtime reaches its host. `setPorts` mutates this object in place. */
export const ports = { env: () => undefined, readFile: async () => null, writeFile: null, fetchImpl: null, layers: [], log: null, confirm: null, output: null };

/** How the runtime reaches its host: env, files, fetch, the log line sink, the person, the output sink. */
export function setPorts(p) { Object.assign(ports, p ?? {}); }

let current = null;
export const getCurrent = () => current;
/** Makes `ctx` the active run and answers the one it replaced. */
export const setCurrent = (ctx) => { const prev = current; current = ctx; return prev; };

export const ctxNow = (at) => { if (!current) throw new SandError("E-STATE", "no run is active; call run(main, { input })", { at }); return current; };
export const pick = (o, ...ks) => Object.fromEntries(ks.filter((k) => o && o[k] !== undefined).map((k) => [k, o[k]]));

/** The model a verb reaches for, and a plain refusal when its class is wrong. */
export function resolveModel(ctx, opts, klass) {
  // a conversation reaches for the same model `ask` does
  const key = klass === "chat" ? "ask" : klass;
  const name = opts.model ?? ctx.usingModel ?? ctx.config.default[key];
  if (!name || name === "none") {
    const how = {
      ask: "ask needs a chat model and none is configured; set OPENAI_API_KEY or name one in sand.json",
      chat: "chat needs a chat model and none is configured; set COMBS_API_KEY or OPENAI_API_KEY, or name one in sand.json",
      memory: "memory needs a memory store and none is configured; set COMBS_API_KEY, or name one in sand.json",
    }[klass] ?? "no decision model: set COMBS_API_KEY or write sand.json";
    throw new SandError(klass === "memory" ? "E-MEMORY" : "E-MODEL", how, { at: opts.at });
  }
  const adapter = ctx.adapter(name);
  const needs = { ask: "chat", chat: "chat", memory: "memory", decide: "decision" }[klass];
  if (needs === "chat" && !adapter.ask && !adapter.complete) throw new SandError("E-MODEL", `${klass} needs a chat model; \`${name}\` is a ${adapter.kind} model`, { at: opts.at });
  if (needs === "memory" && adapter.kind !== "memory") throw new SandError("E-MEMORY", `memory needs a memory model; \`${name}\` is a ${adapter.kind} model`, { at: opts.at });
  return { name, adapter };
}

/** A `sand.log` line said once per run per key (the W-… lines the README lists). */
export function warnOnce(ctx, key, line) {
  if (!ctx) return;
  ctx.warned ??= new Set();
  if (ctx.warned.has(key)) return;
  ctx.warned.add(key);
  ctx.emit("sand.log", "log", { level: "warn", message: line }, { severity: "warn", output: line });
  (ctx.log ?? ports.log)?.(line);
}
