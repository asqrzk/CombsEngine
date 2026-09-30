/**
 * @combs/sand — semantic abstraction of neurolinguistic determinism.
 *
 * TypeScript plus three things: `{{name}}` bindings, prose predicates in
 * conditions, and the third truth value `unsure`. This is the whole package:
 * the compiler (`transform`, `plan`) and the runtime (the verbs, the two model
 * adapters, the batcher, the budget, the trace). Zero dependencies; runs in
 * browsers, Deno and Cloudflare Workers.
 */
export * from "./compiler/mod.js";
export * from "./runtime/mod.js";
