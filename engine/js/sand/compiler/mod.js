/** @combs/sand — the compiler barrel. */
export { transform, ASYNC_VERBS, SYNC_VERBS, GENERIC_CONTRARIES, PROLOGUE_LINES, NEEDS_ORDER, needsOf } from "./transform.js";
export { tokenize, pairBrackets, isTrivia } from "./tokenize.js";
import { transform } from "./transform.js";
/** The plan alone: what `sand check` and the playground's Compiled pane render. */
export function plan(src, opts = {}) { return transform(src, opts).plan; }
