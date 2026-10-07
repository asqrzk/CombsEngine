/**
 * @combs/sand — errors that name what the program carries, then the fix.
 *
 * Every message reads `CODE line:col · text`. Compile-time codes come from the
 * transform; runtime codes from the verbs and adapters. `line`/`col` are the
 * program's own positions (the `.sand` file), never the emitted `.ts`.
 *
 * The runtime's codes, each raised in exactly one place and catalogued in the
 * README: `E-MODEL` (no model, the wrong class, or an unset variable),
 * `E-STATE`, `E-SHAPE`, `E-TOOL-SHAPE` (tool(…) was given something that is not
 * a shape, or an assistant something that is not a tool), `E-WIRE`, `E-AUTH`,
 * `E-PASSES`, `E-SUSPENDED`, `E-HOST`, `E-STREAM` (a stream carried an error
 * frame or broke mid-answer), `E-MEMORY` (a store refused, answered the wrong
 * shape, or is not reachable from this host), `E-FENCE` (the fence a run asked
 * for could not be built, so the run did not happen), `E-ANSWER-SHAPE`,
 * `E-BUDGET`, `E-TIMEOUT`, `E-RUNTIME`, `E-NO-MODEL`. Warnings are `sand.log`
 * lines, never codes: `W-UNCALIBRATED`, `W-UNMETERED`, `W-NOBODY`, `W-TOOL-CALL`.
 */

export class SandError extends Error {
  /**
   * @param {string} code    e.g. "E-BINDING"
   * @param {string} text    the message body
   * @param {{ line?: number, col?: number, at?: string, status?: number, api?: unknown, cause?: unknown }} [where]
   */
  constructor(code, text, where = {}) {
    const at = where.at ?? (where.line != null ? `${where.line}:${where.col ?? 1}` : null);
    super(at ? `${code} ${at} · ${text}` : `${code} · ${text}`);
    this.name = "SandError";
    this.code = code;
    this.text = text;
    this.at = at;
    this.line = where.line ?? (at ? Number(at.split(":")[0]) || null : null);
    this.col = where.col ?? (at ? Number(at.split(":")[1]) || null : null);
    this.status = where.status ?? null;
    this.api = where.api ?? null;
    if (where.cause !== undefined) this.cause = where.cause;
  }
}

/** A compile-time refusal at a token position. */
export function refuse(code, text, tok) {
  return new SandError(code, text, { line: tok?.line, col: tok?.col });
}
