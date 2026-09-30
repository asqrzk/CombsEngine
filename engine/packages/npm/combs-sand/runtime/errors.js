/**
 * @combs/sand — errors that name what the program carries, then the fix.
 *
 * Every message reads `CODE line:col · text`. Compile-time codes come from the
 * transform; runtime codes from the verbs and adapters. `line`/`col` are the
 * program's own positions (the `.sand` file), never the emitted `.ts`.
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
