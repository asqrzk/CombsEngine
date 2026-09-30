/**
 * @combs/sand — tools: what an assistant can do, written by the program.
 *
 * `tool(description, Shape, fn, { name?, net? })` declares one. The shape's
 * JSON Schema (the same strict subset `extract` sends) goes to the model; the
 * model can only ever emit a `tool_calls` entry naming it. The body is the
 * program's own function and runs where the program runs — inside the fenced
 * child or the page, never on the platform. `net` declares the hosts the body
 * reaches, so a fence can be built from the plan and nothing wider.
 */

import { SandError } from "./errors.js";
import { unsure } from "./kleene.js";
import { shapeSchema, slug, idsOf } from "./wires/schema.js";

/** Coerces a model's arguments into the shape's own vocabulary, as an extraction does. */
export function coerceArgs(shapeSpec, json) {
  const out = {};
  for (const [n, s] of Object.entries(shapeSpec.slots)) {
    const sp = s.spec ?? s;
    const v = json?.[n];
    if (sp.type === "kind" || sp.type === "scale") out[n] = v === "unsure" || v == null || !idsOf(sp).includes(v) ? unsure : v;
    else if (sp.type === "shape") out[n] = coerceArgs(sp, v);
    else out[n] = v == null ? "" : String(v);
  }
  return out;
}

export class Tool {
  constructor(description, shapeObj, fn, opts = {}) {
    if (typeof description !== "string" || !description.trim()) throw new SandError("E-TOOL-SHAPE", "tool(description, Shape, fn) needs a description the model reads", { at: opts.at });
    const sp = shapeObj?.spec;
    if (!sp || sp.type !== "shape") throw new SandError("E-TOOL-SHAPE", `tool(${JSON.stringify(description.slice(0, 40))}, Shape, fn) needs a shape declared with shape({ … }); it was given ${sp ? sp.type : shapeObj === undefined ? "nothing" : typeof shapeObj}`, { at: opts.at });
    if (typeof fn !== "function") throw new SandError("E-TOOL-SHAPE", "tool(description, Shape, fn) needs a function to run", { at: opts.at });
    this.description = description;
    this.shape = shapeObj;
    this.spec = sp;
    this.name = opts.name ?? sp.name ?? slug(description);
    this.net = opts.net ?? [];
    this.at = opts.at ?? null;
    this.fn = fn;
    this.schema = shapeSchema(sp);
  }
  /** What goes on the wire in `tools`. */
  toSchema() { return { type: "function", function: { name: this.name, description: this.description, parameters: this.schema } }; }
  /** Runs the body over the model's arguments, coerced like an extraction. */
  run(args) { return this.fn(coerceArgs(this.spec, args ?? {})); }
  toJSON() { return { name: this.name, description: this.description, net: this.net, at: this.at }; }
  toString() { return this.name; }
}

/** Declares a tool. Sync: the model sees the schema, the program keeps the body. */
export function tool(description, Shape, fn, opts = {}) { return new Tool(description, Shape, fn, opts); }

/** Every tool an assistant was given, refused plainly when it was given something else. */
export function toolsOf(list, at) {
  if (!Array.isArray(list) || !list.length) throw new SandError("E-STATE", "assistant(system, { tools: [ … ] }) needs at least one tool; a conversation with no tools is chat(…)", { at });
  const byName = new Map();
  for (const t of list) {
    if (!(t instanceof Tool)) throw new SandError("E-TOOL-SHAPE", `assistant's \`tools\` takes what tool(description, Shape, fn) makes; it was given ${t === null ? "null" : typeof t}`, { at });
    if (byName.has(t.name)) throw new SandError("E-TOOL-SHAPE", `two tools are named \`${t.name}\`; name one with tool(…, { name })`, { at });
    byName.set(t.name, t);
  }
  return byName;
}
