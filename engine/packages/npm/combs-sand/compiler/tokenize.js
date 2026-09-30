/**
 * @combs/sand — the token scanner.
 *
 * A JavaScript/TypeScript lexer that is exactly as clever as the transform
 * needs: it keeps strings, template literals (with their `${}` nesting),
 * comments and regex literals whole, and adds one token of its own —
 * `{{name.path}}`, a binding. Every token carries `line`/`col` (1-based) and
 * `start`/`end` offsets into the source, so the transform can splice text
 * back into the original without ever building a syntax tree.
 *
 * Token kinds: ws · comment · string · template · regex · number · ident ·
 * punct · binding · eof.
 */

const PUNCT = [
  ">>>=", "...", "===", "!==", "**=", "<<=", ">>=", ">>>", "&&=", "||=", "??=",
  "=>", "==", "!=", "<=", ">=", "&&", "||", "??", "?.", "++", "--", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "**", "<<", ">>",
  "{", "}", "(", ")", "[", "]", ";", ",", "<", ">", "+", "-", "*", "/", "%", "&", "|", "^", "!", "~", "?", ":", "=", ".", "@", "#",
];

/** Keywords after which a `/` begins a regex literal rather than a division. */
const REGEX_AFTER = new Set([
  "return", "typeof", "instanceof", "in", "of", "new", "delete", "void", "throw",
  "case", "do", "else", "yield", "await",
]);

const IDENT_START = /[A-Za-z_$À-￿]/;
const IDENT_PART = /[\w$À-￿]/;
const BINDING = /^\{\{\s*([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\}\}/;

/**
 * @typedef {object} Token
 * @property {string} kind
 * @property {string} text
 * @property {number} line   1-based
 * @property {number} col    1-based
 * @property {number} start  offset into the source
 * @property {number} end    offset one past the last character
 * @property {string} [path] the identifier path of a binding token
 */

/**
 * @param {string} src
 * @returns {Token[]}
 */
export function tokenize(src) {
  /** @type {Token[]} */
  const out = [];
  let i = 0, line = 1, col = 1;
  const n = src.length;

  const push = (kind, start, end, extra) => {
    const text = src.slice(start, end);
    const tok = { kind, text, line, col, start, end, ...extra };
    out.push(tok);
    // advance line/col over the consumed text
    for (let k = start; k < end; k++) {
      if (src.charCodeAt(k) === 10) { line++; col = 1; } else col++;
    }
    i = end;
    return tok;
  };

  const lastSignificant = () => {
    for (let k = out.length - 1; k >= 0; k--) {
      const t = out[k];
      if (t.kind !== "ws" && t.kind !== "comment") return t;
    }
    return null;
  };

  const regexAllowed = () => {
    const p = lastSignificant();
    if (!p) return true;
    if (p.kind === "punct") return !(p.text === ")" || p.text === "]" || p.text === "}");
    if (p.kind === "ident") return REGEX_AFTER.has(p.text);
    return false;
  };

  // scans a template literal starting at `from` (the backtick); returns the end offset
  const scanTemplate = (from) => {
    let k = from + 1;
    while (k < n) {
      const c = src[k];
      if (c === "\\") { k += 2; continue; }
      if (c === "`") return k + 1;
      if (c === "$" && src[k + 1] === "{") { k = scanBraced(k + 2); continue; }
      k++;
    }
    throw rangeError("an unterminated template literal", from);
  };

  // scans a `${ ... }` body starting after the `{`; returns the offset after the matching `}`
  const scanBraced = (from) => {
    let depth = 1, k = from;
    while (k < n) {
      const c = src[k];
      if (c === "\\") { k += 2; continue; }
      if (c === '"' || c === "'") { k = scanString(k); continue; }
      if (c === "`") { k = scanTemplate(k); continue; }
      if (c === "/" && src[k + 1] === "/") { while (k < n && src[k] !== "\n") k++; continue; }
      if (c === "/" && src[k + 1] === "*") { const e = src.indexOf("*/", k + 2); k = e < 0 ? n : e + 2; continue; }
      if (c === "{") depth++;
      else if (c === "}") { depth--; if (depth === 0) return k + 1; }
      k++;
    }
    throw rangeError("an unterminated `${` in a template literal", from);
  };

  const scanString = (from) => {
    const q = src[from];
    let k = from + 1;
    while (k < n) {
      const c = src[k];
      if (c === "\\") { k += 2; continue; }
      if (c === q) return k + 1;
      if (c === "\n") throw rangeError("an unterminated string", from);
      k++;
    }
    throw rangeError("an unterminated string", from);
  };

  const rangeError = (what, at) => {
    let l = 1, c = 1;
    for (let k = 0; k < at; k++) { if (src.charCodeAt(k) === 10) { l++; c = 1; } else c++; }
    const e = new Error(`E-PARSE ${l}:${c} · this program carries ${what}`);
    e.code = "E-PARSE"; e.line = l; e.col = c;
    return e;
  };

  while (i < n) {
    const c = src[i];

    // whitespace
    if (c === " " || c === "\t" || c === "\n" || c === "\r") {
      let k = i; while (k < n && (src[k] === " " || src[k] === "\t" || src[k] === "\n" || src[k] === "\r")) k++;
      push("ws", i, k); continue;
    }
    // comments
    if (c === "/" && src[i + 1] === "/") { let k = i; while (k < n && src[k] !== "\n") k++; push("comment", i, k); continue; }
    if (c === "/" && src[i + 1] === "*") { const e = src.indexOf("*/", i + 2); push("comment", i, e < 0 ? n : e + 2); continue; }
    // binding
    if (c === "{" && src[i + 1] === "{") {
      const m = BINDING.exec(src.slice(i));
      if (m) { push("binding", i, i + m[0].length, { path: m[1] }); continue; }
    }
    // strings and templates
    if (c === '"' || c === "'") { push("string", i, scanString(i)); continue; }
    if (c === "`") { push("template", i, scanTemplate(i)); continue; }
    // regex
    if (c === "/" && regexAllowed()) {
      let k = i + 1, inClass = false;
      while (k < n) {
        const d = src[k];
        if (d === "\\") { k += 2; continue; }
        if (d === "\n") break;
        if (inClass) { if (d === "]") inClass = false; }
        else if (d === "[") inClass = true;
        else if (d === "/") { k++; while (k < n && IDENT_PART.test(src[k])) k++; break; }
        k++;
      }
      push("regex", i, k); continue;
    }
    // numbers
    if (/[0-9]/.test(c) || (c === "." && /[0-9]/.test(src[i + 1] || ""))) {
      let k = i + 1; while (k < n && /[\w.]/.test(src[k])) k++;
      push("number", i, k); continue;
    }
    // identifiers and keywords
    if (IDENT_START.test(c)) {
      let k = i + 1; while (k < n && IDENT_PART.test(src[k])) k++;
      push("ident", i, k); continue;
    }
    // punctuation (longest match first)
    let matched = false;
    for (const p of PUNCT) {
      if (src.startsWith(p, i)) { push("punct", i, i + p.length); matched = true; break; }
    }
    if (matched) continue;
    // anything else: one character of prose (an em dash, a curly quote, …)
    push("punct", i, i + 1);
  }
  out.push({ kind: "eof", text: "", line, col, start: n, end: n });
  return out;
}

/** True for tokens that carry no meaning to the parser. */
export const isTrivia = (t) => t.kind === "ws" || t.kind === "comment";

/**
 * Pairs every bracket with its match. Returns a Map from the opening token's
 * index to the closing token's index, and the reverse. Unbalanced input throws
 * E-PARSE at the offending token.
 */
export function pairBrackets(toks) {
  const open = new Map(), close = new Map();
  const stack = [];
  const pairs = { "(": ")", "[": "]", "{": "}" };
  for (let k = 0; k < toks.length; k++) {
    const t = toks[k];
    if (t.kind !== "punct") continue;
    if (pairs[t.text]) stack.push(k);
    else if (t.text === ")" || t.text === "]" || t.text === "}") {
      const o = stack.pop();
      if (o === undefined || pairs[toks[o].text] !== t.text) {
        const e = new Error(`E-PARSE ${t.line}:${t.col} · this program carries an unmatched \`${t.text}\``);
        e.code = "E-PARSE"; e.line = t.line; e.col = t.col; throw e;
      }
      open.set(o, k); close.set(k, o);
    }
  }
  if (stack.length) {
    const t = toks[stack[stack.length - 1]];
    const e = new Error(`E-PARSE ${t.line}:${t.col} · this program carries an unclosed \`${t.text}\``);
    e.code = "E-PARSE"; e.line = t.line; e.col = t.col; throw e;
  }
  return { open, close };
}
