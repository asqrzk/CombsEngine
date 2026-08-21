/**
 * sanitize — the ingestion filter every graph write passes through.
 *
 * Stored text reaches model prompts (recall lines return as tool
 * results), so the store never persists invisible or directional
 * poison. Two rules, in order:
 *
 * 1. Escape sequences go as WHOLE sequences (CSI/OSC/string + short
 *    escapes) — stripping only the ESC byte would leave the printable
 *    payload behind as garbage text.
 * 2. Code points strip by Unicode category: everything in C*
 *    (control, format, surrogate, private use, unassigned) except tab
 *    and newline, plus the invisible marks and fillers the categories
 *    miss (variation selectors, fillers, U+FFFD).
 *
 * Homoglyph normalization is NOT done here — confusable-but-visible
 * text is a rendering concern, and rewriting it would corrupt
 * legitimate non-Latin content.
 */

const ANSI = /\x1B(?:\[[0-?]*[ -\/]*[@-~]|\][^\x07\x1B]*(?:\x07|\x1B\\)?|[PX^_][^\x1B]*(?:\x1B\\)?|.)/g;

const POISON = new RegExp(
  "(?![\\t\\n])" +
    "[\\p{Cc}\\p{Cf}\\p{Co}\\p{Cs}\\p{Cn}" + // category C: control, format, private use, surrogate, unassigned
    "\\u034F\\u115F\\u1160\\u17B4\\u17B5\\u180B-\\u180F\\u3164\\uFFA0" + // invisible marks and fillers
    "\\uFE00-\\uFE0F\\u{E0100}-\\u{E01EF}" + // variation selectors
    "\\uFFFD" + // replacement character — decode damage, never content
    "]",
  "gu",
);

/** Strips poison sequences/code points, normalizes CRLF; visible text is unchanged. */
export function cleanText(s: string): string {
  return s.replace(/\r\n?/g, "\n").replace(ANSI, "").replace(POISON, "");
}

/**
 * True when decoded text smells like binary: a NUL, or a dense run of
 * replacement characters. A lone U+FFFD from a read cut mid-codepoint
 * stays below both thresholds.
 */
export function looksBinary(s: string): boolean {
  if (s.includes("\u0000")) return true;
  let bad = 0;
  for (const ch of s) if (ch === "\uFFFD") bad++;
  return bad >= 4 && bad / Math.max(1, s.length) > 0.02;
}
