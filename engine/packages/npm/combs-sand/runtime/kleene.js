/**
 * @combs/sand — three-valued logic.
 *
 * `unsure` is the third truth value: what a decision is worth when the model
 * abstained. `and`/`or`/`not` are strong Kleene: `false` beats `unsure` in an
 * `and`, `true` beats it in an `or`, and `not(unsure)` stays `unsure`.
 */

/** The third truth value. `Symbol.for` so every copy of the runtime agrees. */
export const unsure = Symbol.for("sand.unsure");

/** @param {unknown} v @returns {v is boolean} */
export const isVerdict = (v) => v === true || v === false || v === unsure;

/** Coerces a value to a verdict: booleans and `unsure` pass through; anything else by truthiness. */
export function verdict(v) {
  if (v === unsure) return unsure;
  if (v && typeof v === "object" && "value" in v) return verdict(v.value);
  return !!v;
}

export function and(a, b) {
  const x = verdict(a), y = verdict(b);
  if (x === false || y === false) return false;
  if (x === unsure || y === unsure) return unsure;
  return true;
}

export function or(a, b) {
  const x = verdict(a), y = verdict(b);
  if (x === true || y === true) return true;
  if (x === unsure || y === unsure) return unsure;
  return false;
}

export function not(a) {
  const x = verdict(a);
  return x === unsure ? unsure : !x;
}
