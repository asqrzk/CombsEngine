/**
 * @combs/sand — questions and schemas.
 *
 * Turns a program's declarations into what a wire understands: a classify
 * question (`choice` / `noul` / `score`) for the decision wire, and a JSON
 * Schema in the intersection of what the Combs engine's constrained decoder
 * accepts (type, enum, const, properties, items, required,
 * additionalProperties) and what OpenAI's strict mode requires (every property
 * required, additionalProperties: false) for a chat model.
 */

import { SandError } from "../errors.js";

export const slug = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 32) || "x";
export const hash = (s) => { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h.toString(16).padStart(8, "0"); };

const ENGINE_UNSUPPORTED = ["$ref", "$defs", "definitions", "anyOf", "oneOf", "allOf", "not", "if", "then", "else", "pattern", "format", "patternProperties", "propertyNames", "dependentRequired", "dependentSchemas", "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minItems", "maxItems", "uniqueItems", "contains", "prefixItems", "minProperties", "maxProperties"];

/** Normalises what `decide`/`decideAll` accept into a spec object. */
export function normalizeSpec(spec, opts = {}) {
  if (typeof spec === "string") return { type: "noul", means: spec, not: opts.not ?? [], noun: opts.noun ?? null };
  if (spec && spec.spec) return spec.spec;
  if (spec && typeof spec === "object" && typeof spec.means === "string") return { type: "noul", means: spec.means, not: spec.not ?? [], noun: spec.noun ?? null };
  throw new SandError("E-STATE", "decide needs a kind, a scale, or a proposition (a string or { means, not })", { at: opts.at });
}

/** The classify-wire question for a spec. */
export function questionOf(spec, settle = {}) {
  if (spec.type === "kind") {
    const options = Object.entries(spec.options).map(([id, o]) => ({ id, tether: tetherOf(o) }));
    const qn = { name: `k_${slug(spec.name ?? hash(JSON.stringify(spec.options)))}`, type: "choice", instructions: spec.instructions, options };
    if (spec.fallback && (settle.floor != null || settle.gap != null)) {
      qn.abstain = { option: spec.fallback };
      if (settle.floor != null) qn.abstain.p_min = settle.floor;
      if (settle.gap != null) qn.abstain.gap_max = settle.gap;
    }
    return qn;
  }
  if (spec.type === "scale") {
    return { name: `s_${slug(spec.name ?? hash(JSON.stringify(spec.levels)))}`, type: "score", instructions: spec.instructions, levels: spec.levels.map((l) => ({ id: l.id, tether: { means: l.means ?? l.id } })) };
  }
  if (spec.type === "noul") {
    if (!spec.not || !spec.not.length) throw new SandError("E-WIRE", "a noul needs at least one contrary in `not` to score against");
    return { name: `n_${hash(spec.means + "|" + spec.not.join("|"))}`, type: "noul", instructions: `is this true of ${spec.noun ?? "the text"}`, tether: { means: spec.means, not: spec.not } };
  }
  throw new SandError("E-STATE", `cannot ask a ${spec.type} as a question`);
}
const tetherOf = (o) => (typeof o === "string" ? { means: o } : { means: o.means, ...(o.not?.length ? { not: o.not } : {}), ...(o.requires?.length ? { requires: o.requires } : {}) });

/** Canonical JSON: object keys sorted at every depth, so equal questions serialise equally. */
export function canonical(v) {
  if (Array.isArray(v)) return "[" + v.map(canonical).join(",") + "]";
  if (v && typeof v === "object") return "{" + Object.keys(v).sort().filter((k) => v[k] !== undefined).map((k) => JSON.stringify(k) + ":" + canonical(v[k])).join(",") + "}";
  return JSON.stringify(v);
}

/** A stable key for memo and deduplication: everything but the wire name. */
export function questionKey(q) {
  const { name, ...rest } = q;
  return canonical(rest);
}

/** The option ids a spec can answer with. */
export function idsOf(spec) {
  if (spec.type === "kind") return Object.keys(spec.options);
  if (spec.type === "scale") return spec.levels.map((l) => l.id);
  if (spec.type === "noul") return ["true", "false"];
  return [];
}

/** JSON Schema for one decision on a chat model: an enum with "unsure" added. */
export function enumSchema(spec) {
  return { type: "object", properties: { answer: { type: "string", enum: [...idsOf(spec), "unsure"] } }, required: ["answer"], additionalProperties: false };
}

/** JSON Schema for a shape; refuses keywords the engine's decoder would refuse. */
export function shapeSchema(shapeSpec, { only = null } = {}) {
  const properties = {};
  for (const [name, slot] of Object.entries(shapeSpec.slots)) {
    if (only && !only.includes(name)) continue;
    const s = slot?.spec ?? slot;
    if (s.type === "text") properties[name] = { type: "string" };
    else if (s.type === "kind" || s.type === "scale") properties[name] = { type: "string", enum: [...idsOf(s), "unsure"] };
    else if (s.type === "shape") properties[name] = shapeSchema(s);
    else throw new SandError("E-MODEL", `slot ${JSON.stringify(name)} has no known type`);
  }
  const schema = { type: "object", properties, required: Object.keys(properties), additionalProperties: false };
  checkSchema(schema);
  return schema;
}

export function checkSchema(schema, path = "schema") {
  if (!schema || typeof schema !== "object") return;
  for (const k of Object.keys(schema)) if (ENGINE_UNSUPPORTED.includes(k)) throw new SandError("E-MODEL", `${path} carries \`${k}\`, which the Combs decoder refuses by name; SAND schemas use type, enum, const, properties, items, required and additionalProperties only`);
  if (schema.properties) { if (Object.keys(schema.properties).length > 64) throw new SandError("E-MODEL", `${path} has more than 64 properties`); for (const [k, v] of Object.entries(schema.properties)) checkSchema(v, `${path}.${k}`); }
  if (schema.items) checkSchema(schema.items, `${path}[]`);
}

/** The system message that renders a question for a chat model. */
export function promptOf(question) {
  const lines = [];
  if (question.type === "noul") {
    lines.push(`Decide whether the following is true of the text the user sends. Answer with JSON {"answer": "true" | "false" | "unsure"}.`);
    lines.push(`Proposition: ${question.tether.means}`);
    if (question.tether.not?.length) lines.push(`It is NOT true when instead: ${question.tether.not.join("; ")}.`);
    lines.push(`Answer "unsure" only when the text gives no basis to decide.`);
  } else {
    const items = question.type === "score" ? question.levels : question.options;
    lines.push(`${question.instructions ? question.instructions[0].toUpperCase() + question.instructions.slice(1) : "Classify the text the user sends"}. Answer with JSON {"answer": <one id>} choosing exactly one of these ids${question.type === "score" ? ", which are ordered from lowest to highest" : ""}:`);
    for (const it of items) lines.push(`- ${it.id}: ${it.tether.means}${it.tether.not?.length ? ` (not when: ${it.tether.not.join("; ")})` : ""}`);
    lines.push(`Answer "unsure" only when none fits and you cannot tell.`);
  }
  return lines.join("\n");
}

/** The system message for a shape extraction on a chat model. */
export function extractionPrompt(shapeSpec, only = null) {
  const lines = ["Read the text the user sends and fill every field of the JSON object exactly as the schema requires."];
  for (const [name, slot] of Object.entries(shapeSpec.slots)) {
    if (only && !only.includes(name)) continue;
    const s = slot?.spec ?? slot;
    if (s.type === "text") lines.push(`- ${name}: ${s.instructions}`);
    else if (s.type === "kind") lines.push(`- ${name}: ${s.instructions}; one of ${Object.entries(s.options).map(([id, o]) => `${id} (${typeof o === "string" ? o : o.means})`).join(", ")}, or "unsure"`);
    else if (s.type === "scale") lines.push(`- ${name}: ${s.instructions}; one of ${s.levels.map((l) => `${l.id} (${l.means ?? l.id})`).join(", ")} in rising order, or "unsure"`);
  }
  return lines.join("\n");
}
