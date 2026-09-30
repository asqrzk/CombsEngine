/**
 * @combs/sand — configuration.
 *
 * Layered like @combs/core's config.ts: defaults → $SAND_CONFIG → the nearest
 * `sand.json` upward from the program → ~/.config/combs/sand.json → what the
 * program passes to configure() → what a call names. Zero config works: a
 * COMBS_API_KEY in the environment yields a decision model called `combs`, an
 * OPENAI_API_KEY a chat model called `openai`.
 *
 * This module never touches Deno.* or process.*: it takes `env` and `readFile`
 * ports, so the same code runs in the browser (with env: () => undefined).
 */

import { SandError } from "./errors.js";

export const DEFAULT_LIMITS = { passes: 64, tokens: 20000, ms: 60000, concurrency: 4 };
export const DEFAULTS = { models: {}, default: { decide: null, ask: null }, settle: {}, limits: { ...DEFAULT_LIMITS }, net: [] };

/** The registry of model kinds: name → (modelConfig, ports) => adapter. Filled by wires/*.js and registerKind(). */
export const kindRegistry = new Map();
export function registerKind(name, factory) {
  if (typeof factory !== "function") throw new SandError("E-MODEL", `registerKind(${JSON.stringify(name)}) needs a factory (model, ports) => adapter`);
  kindRegistry.set(name, factory);
}

/** Resolves `$NAME` references against env; `$$` escapes a literal dollar. */
export function resolveEnv(value, env, where) {
  if (typeof value !== "string") return value;
  if (value.startsWith("$$")) return value.slice(1);
  if (!value.startsWith("$")) return value;
  const name = value.slice(1);
  const v = env(name);
  if (v === undefined || v === "") throw new SandError("E-MODEL", `${where} reads $${name}, which is not set in the environment`);
  return v;
}

/** Defined values win; `models` merge per name; `net` is recomputed later. */
export function mergeConfig(...layers) {
  const out = { models: {}, default: {}, settle: {}, limits: {} };
  for (const layer of layers) {
    if (!layer) continue;
    for (const [name, m] of Object.entries(layer.models ?? {})) out.models[name] = { ...(out.models[name] ?? {}), ...m };
    for (const k of ["default", "settle", "limits"]) for (const [kk, v] of Object.entries(layer[k] ?? {})) if (v !== undefined) out[k][kk] = v;
    if (layer.net) out.net = layer.net;
  }
  return out;
}

/** The hostname[:port] of a model url. */
export function hostOf(url) {
  try { const u = new URL(url); return u.port ? `${u.hostname}:${u.port}` : u.hostname; } catch { return null; }
}

/** `net` is derived from the models; a hand-written list may only narrow it. */
export function deriveNet(models, written) {
  const derived = [...new Set(Object.values(models).map((m) => hostOf(m.url)).filter(Boolean))];
  if (!written) return derived;
  for (const h of written) if (!derived.includes(h)) throw new SandError("E-MODEL", `\`net\` names ${h}, which no model uses`);
  return written;
}

/** Models implied by the environment alone. Missing keys mean "absent", never an error. */
export function zeroConfig(env) {
  const models = {};
  if (env("COMBS_API_KEY")) models.combs = { kind: "decision", url: env("COMBS_API_URL") || "https://api.combs.network", key: "$COMBS_API_KEY", road: "seat", implicit: true };
  if (env("OPENAI_API_KEY")) models.openai = { kind: "chat", url: env("OPENAI_BASE_URL") || "https://api.openai.com/v1", key: "$OPENAI_API_KEY", model: env("OPENAI_MODEL") || "gpt-4o-mini", implicit: true };
  return { models };
}

/**
 * Resolves the effective config from the layers, env refs and defaults.
 * @param {{ env: (n: string) => string|undefined, layers?: object[], explicit?: object, local?: object|null }} p
 */
export function resolveConfig({ env, layers = [], explicit = null, local = null }) {
  const merged = mergeConfig(DEFAULTS, zeroConfig(env), local ? { models: { local } } : null, ...layers, explicit);
  const models = {};
  for (const [name, m] of Object.entries(merged.models)) {
    if (!m.kind) throw new SandError("E-MODEL", `model ${JSON.stringify(name)} has no kind; set kind: "decision" or "chat"`);
    if (!kindRegistry.has(m.kind)) throw new SandError("E-MODEL", `model ${JSON.stringify(name)} has kind ${JSON.stringify(m.kind)}, which nothing registered; SAND knows ${[...kindRegistry.keys()].join(", ")}`);
    if (!m.url) throw new SandError("E-MODEL", `model ${JSON.stringify(name)} has no url`);
    const resolved = { ...m, name };
    for (const f of ["url", "key", "model"]) {
      if (typeof m[f] === "string" && m[f].startsWith("$") && !m[f].startsWith("$$")) {
        const v = env(m[f].slice(1));
        if (v === undefined || v === "") { resolved.unavailable = m[f].slice(1); resolved[f] = null; }
        else resolved[f] = v;
      }
    }
    if (m.headers) resolved.headers = Object.fromEntries(Object.entries(m.headers).map(([k, v]) => [k, resolveEnv(v, env, `model ${name} header ${k}`)]));
    models[name] = resolved;
  }
  const byKind = (k) => Object.values(models).filter((m) => m.kind === k && !m.unavailable).map((m) => m.name);
  const decide = merged.default.decide ?? (models.combs && !models.combs.unavailable ? "combs" : byKind("decision")[0] ?? null);
  const ask = merged.default.ask ?? (models.openai && !models.openai.unavailable ? "openai" : byKind("chat")[0] ?? "none");
  return {
    models,
    default: { decide, ask },
    settle: merged.settle ?? {},
    limits: { ...DEFAULT_LIMITS, ...(merged.limits ?? {}) },
    net: deriveNet(models, merged.net?.length ? merged.net : null),
  };
}

/** The config file layers, in precedence order (lowest first). */
export async function loadLayers({ env, readFile, cwd = ".", home = null }) {
  const candidates = [];
  if (home) candidates.push(`${home}/.config/combs/sand.json`);
  // nearest sand.json upward from cwd
  const parts = cwd.split("/").filter(Boolean);
  const upward = [];
  for (let i = parts.length; i >= 0; i--) upward.push((cwd.startsWith("/") ? "/" : "") + parts.slice(0, i).join("/") + (i ? "/" : "") + "sand.json");
  for (const p of upward) { const text = await readFile(p); if (text != null) { candidates.push(p); break; } }
  if (env("SAND_CONFIG")) candidates.push(env("SAND_CONFIG"));
  const layers = [];
  for (const path of candidates) {
    const text = await readFile(path);
    if (text == null) continue;
    try { layers.push({ path, ...JSON.parse(text) }); }
    catch (e) { throw new SandError("E-MODEL", `${path} is not valid JSON: ${e.message}`); }
  }
  return layers;
}
