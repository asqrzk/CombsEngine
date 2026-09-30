/**
 * @combs/sand — the transform: `.sand` → `.ts`.
 *
 * A `.sand` file is TypeScript plus three things: `{{name}}` bindings, prose
 * predicates in conditions, and the third truth value `unsure`. This module
 * rewrites exactly those three into calls on the runtime and leaves every other
 * character where it was. It works on tokens, never on a syntax tree, and every
 * edit is a splice into the original text, so the output is line-for-line with
 * the input after a fixed two-line prologue.
 *
 * Rules T0–T9: T0 wrapper · T1 bindings · T2 prose conditions ·
 * T3 unsure · T4 verb auto-await · T5 shape inference · T6 stamping · T7
 * hoisting · T7b speculation · T8 switch discriminants · T9 near-miss warning.
 */

import { tokenize, isTrivia, pairBrackets } from "./tokenize.js";
import { SandError } from "../runtime/errors.js";

export const ASYNC_VERBS = ["decide", "decideAll", "classify", "ask", "extract", "parallel", "race", "timeout", "retry", "using"];
export const SYNC_VERBS = ["log", "kind", "scale", "shape", "text", "settle", "configure", "named"];
const RESERVED = new Set([...ASYNC_VERBS, ...SYNC_VERBS, "and", "or", "not", "unsure", "main", "run", "trace"]);
/** index of the options argument per verb (the argument stamping targets) */
const OPTS_INDEX = { ask: 1, extract: 2, decide: 2, decideAll: 2, classify: 1, parallel: 2, retry: 1 };
const DECL_OPTS_INDEX = { kind: 2, scale: 2, shape: 1 };
const NOT_FUNCTION_HEADS = new Set(["if", "while", "for", "switch", "catch", "with"]);
const IMPORT_NAMES = "kind, scale, shape, text, settle, configure, decide, decideAll, classify, ask, extract, unsure, and, or, not, parallel, race, timeout, retry, using, named, log, run, trace";
export const GENERIC_CONTRARIES = ["the opposite is the case", "it is only mentioned, quoted or asked about, not the case"];
export const PROLOGUE_LINES = 2;

/**
 * @param {string} src
 * @param {{ importStyle?: "jsr"|"relative"|"global", importPath?: string, file?: string }} [opts]
 * @returns {{ code: string, plan: object, warnings: Array<{code:string, at:string, message:string}> }}
 */
export function transform(src, opts = {}) {
  const importStyle = opts.importStyle ?? "jsr";
  const file = opts.file ?? "program.sand";
  const toks = tokenize(src);
  const { open, close } = pairBrackets(toks);
  const warnings = [];
  const warn = (code, tok, message) => warnings.push({ code, at: tok ? `${tok.line}:${tok.col}` : "1:1", message });
  const at = (tok) => `${tok.line}:${tok.col}`;
  const q = (s) => JSON.stringify(s);
  const unquote = (text) => JSON.parse(text[0] === "'" ? '"' + text.slice(1, -1).replace(/\\'/g, "'").replace(/"/g, '\\"') + '"' : text);

  // ── token navigation ────────────────────────────────────────────────────
  const sig = (k, dir = 1) => { k += dir; while (k >= 0 && k < toks.length && isTrivia(toks[k])) k += dir; return k >= 0 && k < toks.length ? k : -1; };
  const isP = (k, text) => k >= 0 && toks[k].kind === "punct" && toks[k].text === text;
  const isI = (k, text) => k >= 0 && toks[k].kind === "ident" && (text === undefined || toks[k].text === text);
  const slice = (a, b) => src.slice(toks[a].start, toks[b].end);

  // ── the splice list ─────────────────────────────────────────────────────
  /** @type {{start:number, end:number, text:string, order:number}[]} */
  const edits = [];
  let editOrder = 0;
  const replace = (start, end, text) => edits.push({ start, end, text, order: editOrder++ });
  const insert = (pos, text) => edits.push({ start: pos, end: pos, text, order: editOrder++ });
  const consumed = new Set(); // binding token indices already rewritten as part of a larger edit

  // ── T0 · refuse `export` in a program ───────────────────────────────────
  for (let k = 0; k < toks.length; k++) {
    if (isI(k, "export") && depthOf(k) === 0) {
      throw new SandError("E-RESERVED", "a .sand file is a program, not a module; put shared kinds in a .ts file and import them", toks[k]);
    }
  }
  function depthOf(k) { let d = 0; for (let j = 0; j < k; j++) { if (isP(j, "{")) d++; else if (isP(j, "}")) d--; } return d; }

  // ── declarations: kinds, scales, shapes, nouns, settle ──────────────────
  const kinds = new Map(), scales = new Map(), shapes = new Map(), nouns = new Map([["input", "the message"]]);
  let settleTok = null, settlePasses = false;
  const topKeys = (openIdx) => { // keys of an object literal at token index openIdx (the `{`)
    const out = []; const end = open.get(openIdx); let depth = 0;
    for (let k = openIdx + 1; k < end; k++) {
      const t = toks[k];
      if (t.kind === "punct" && "([{".includes(t.text)) { depth++; continue; }
      if (t.kind === "punct" && ")]}".includes(t.text)) { depth--; continue; }
      if (depth !== 0) continue;
      if ((t.kind === "ident" || t.kind === "string" || t.kind === "number") && isP(sig(k), ":")) {
        const key = t.kind === "string" ? unquote(t.text) : t.text;
        out.push({ key, tok: t, valueStart: sig(sig(k)) });
      }
    }
    return out;
  };
  const arrayStrings = (openIdx) => { const out = []; const end = open.get(openIdx); for (let k = openIdx + 1; k < end; k++) if (toks[k].kind === "string") out.push(unquote(toks[k].text)); return out; };
  const callArgs = (openIdx) => { // argument token ranges [start, end] inclusive, split at top-level commas
    const end = open.get(openIdx); const args = []; let depth = 0, s = sig(openIdx);
    if (s === end) return args;
    for (let k = openIdx + 1; k < end; k++) {
      const t = toks[k];
      if (t.kind === "punct" && "([{".includes(t.text)) depth++;
      else if (t.kind === "punct" && ")]}".includes(t.text)) depth--;
      else if (depth === 0 && t.kind === "punct" && t.text === ",") { args.push([s, sig(k, -1)]); s = sig(k); }
    }
    if (s < end) args.push([s, sig(end, -1)]);
    return args;
  };

  for (let k = 0; k < toks.length; k++) {
    if (!isI(k)) continue;
    const t = toks[k];
    // const NAME = kind|scale|shape( … )
    if ((t.text === "const" || t.text === "let") && isI(sig(k)) && isP(sig(sig(k)), "=")) {
      const nameIdx = sig(k), eqIdx = sig(nameIdx), vIdx = sig(eqIdx);
      const name = toks[nameIdx].text;
      if (RESERVED.has(name)) throw new SandError("E-RESERVED", `\`${name}\` is a SAND name and cannot be declared here; choose another`, toks[nameIdx]);
      if (isI(vIdx) && isP(sig(vIdx), "(") && DECL_OPTS_INDEX[toks[vIdx].text] !== undefined) {
        const verb = toks[vIdx].text, args = callArgs(sig(vIdx));
        const decl = { name, verb, at: at(toks[nameIdx]), nameTok: toks[nameIdx], callOpen: sig(vIdx), args };
        if (verb === "kind" && args[1] && isP(args[1][0], "{")) {
          decl.options = topKeys(args[1][0]).map((o) => o.key);
          decl.fallback = null;
          if (args[2] && isP(args[2][0], "{")) for (const o of topKeys(args[2][0])) if (o.key === "fallback" && toks[o.valueStart].kind === "string") decl.fallback = unquote(toks[o.valueStart].text);
          kinds.set(name, decl);
        } else if (verb === "scale" && args[1]) {
          decl.levels = isP(args[1][0], "[") ? arrayStrings(args[1][0]) : isP(args[1][0], "{") ? topKeys(args[1][0]).map((o) => o.key) : [];
          scales.set(name, decl);
        } else if (verb === "shape" && args[0] && isP(args[0][0], "{")) {
          decl.slots = {};
          for (const o of topKeys(args[0][0])) {
            const v = toks[o.valueStart];
            decl.slots[o.key] = v.kind === "ident" && v.text === "text" ? "text" : v.kind === "ident" ? v.text : "?";
          }
          shapes.set(name, decl);
        }
      }
      // const NAME = {{x}} as "noun"
      if (vIdx >= 0 && toks[vIdx].kind === "binding" && isI(sig(vIdx), "as") && toks[sig(sig(vIdx))]?.kind === "string") {
        const strIdx = sig(sig(vIdx));
        nouns.set(name, unquote(toks[strIdx].text));
        replace(toks[vIdx].end, toks[strIdx].end, ""); // strip ` as "noun"`
      }
    }
    if (t.text === "settle" && isP(sig(k), "(") && !isP(sig(k, -1), ".")) {
      settleTok = settleTok ?? t;
      const inner = slice(sig(k), open.get(sig(k)));
      if (/\bpasses\s*:/.test(inner)) settlePasses = true;
    }
  }
  const nounOf = (path) => {
    const head = path.split(".")[0], last = path.split(".").pop();
    if (path.includes(".")) return "the " + last.replace(/([a-z])([A-Z])/g, "$1 $2").replace(/_/g, " ").toLowerCase();
    if (nouns.has(head)) return nouns.get(head);
    return "the " + head.replace(/([a-z])([A-Z])/g, "$1 $2").replace(/_/g, " ").toLowerCase();
  };

  // ── function contexts (async or not, using-model, loops) ────────────────
  const main = { async: true, isMain: true, usingModel: null, start: 0, end: toks.length };
  const fns = [];
  for (let k = 0; k < toks.length; k++) {
    if (!isP(k, "{")) continue;
    const p = sig(k, -1);
    let fn = null;
    if (isP(p, "=>")) {
      const pp = sig(p, -1);
      let asyncTok = -1, paramsStart = pp;
      if (isP(pp, ")")) { paramsStart = close.get(pp); asyncTok = sig(paramsStart, -1); }
      else if (isI(pp)) { asyncTok = sig(pp, -1); }
      const isAsync = isI(asyncTok, "async");
      fn = { async: isAsync, start: k, end: open.get(k), asyncInsertAt: toks[paramsStart].start, isArrow: true };
    } else if (isP(p, ")")) {
      const o = close.get(p), before = sig(o, -1);
      if (isI(before) && !NOT_FUNCTION_HEADS.has(toks[before].text)) {
        const b2 = sig(before, -1);
        if (toks[before].text === "function") {
          fn = { async: isI(b2, "async"), start: k, end: open.get(k), asyncInsertAt: toks[before].start };
        } else if (isI(b2, "function")) {
          const b3 = sig(b2, -1);
          fn = { async: isI(b3, "async"), start: k, end: open.get(k), asyncInsertAt: toks[b2].start };
        } else if (!isP(b2, ".") && !isI(b2, "return") && !isI(b2, "new")) {
          // a method shorthand — only when the head sits directly in an object/class body
          const encl = enclosingBrace(before);
          if (encl >= 0 && !isBlockBrace(encl)) fn = { async: isI(b2, "async"), start: k, end: open.get(k), asyncInsertAt: toks[before].start };
        }
      } else if (isI(before, "function")) {
        fn = { async: isI(sig(before, -1), "async"), start: k, end: open.get(k), asyncInsertAt: toks[before].start };
      }
    }
    if (fn) { fn.usingModel = null; fns.push(fn); }
  }
  // expression-bodied arrows: `=> expr` without a brace
  for (let k = 0; k < toks.length; k++) {
    if (!isP(k, "=>")) continue;
    const nx = sig(k);
    if (isP(nx, "{")) continue;
    const pp = sig(k, -1);
    let asyncTok = -1, paramsStart = pp;
    if (isP(pp, ")")) { paramsStart = close.get(pp); asyncTok = sig(paramsStart, -1); } else if (isI(pp)) asyncTok = sig(pp, -1);
    let depth = 0, e = nx;
    for (; e < toks.length; e++) {
      const t = toks[e];
      if (t.kind === "punct" && "([{".includes(t.text)) depth++;
      else if (t.kind === "punct" && ")]}".includes(t.text)) { if (depth === 0) break; depth--; }
      else if (depth === 0 && t.kind === "punct" && (t.text === "," || t.text === ";")) break;
      else if (depth === 0 && t.kind === "ws" && t.text.includes("\n")) break;
    }
    fns.push({ async: isI(asyncTok, "async"), start: nx, end: e, asyncInsertAt: toks[paramsStart].start, isArrow: true, exprBody: true, usingModel: null });
  }
  fns.sort((a, b) => a.start - b.start || b.end - a.end);
  function enclosingBrace(k) { let depth = 0; for (let j = k - 1; j >= 0; j--) { if (isP(j, "}")) depth++; else if (isP(j, "{")) { if (depth === 0) return j; depth--; } } return -1; }
  function isBlockBrace(k) { // a `{` that opens a statement block (not an object literal / class / type body)
    const p = sig(k, -1);
    if (p < 0) return true;
    if (isP(p, ")")) return true;
    if (isP(p, "=>")) return true;
    if (isI(p) && ["else", "try", "finally", "do", "unsure"].includes(toks[p].text)) return true;
    if (isP(p, "{") || isP(p, "}") || isP(p, ";")) return true;
    return false;
  }
  const argFn = (args, ai) => (args[ai] ? fns.find((f) => f.start >= args[ai][0] && f.start <= args[ai][1] + 1) : null);
  // using("m", fn): the callback's context carries the model; parallel/retry/using callbacks are made async
  for (let k = 0; k < toks.length; k++) {
    if (!isI(k) || !["parallel", "retry", "using"].includes(toks[k].text) || !isP(sig(k), "(") || isP(sig(k, -1), ".")) continue;
    const args = callArgs(sig(k));
    const ai = toks[k].text === "retry" ? 0 : 1;
    const fn = argFn(args, ai);
    if (!fn) continue;
    if (toks[k].text === "using" && toks[args[0][0]]?.kind === "string") fn.usingModel = unquote(toks[args[0][0]].text);
    if (toks[k].text === "parallel") fn.parallelBody = true;
    if (!fn.async) { fn.async = true; fn.madeAsync = true; insert(fn.asyncInsertAt, "async "); }
  }
  const ctxAt = new Array(toks.length).fill(main);
  for (const f of fns) for (let k = f.start; k <= f.end && k < toks.length; k++) ctxAt[k] = f; // inner functions overwrite
  // loops: token ranges of for/while/do bodies and parallel callbacks
  const loops = [];
  for (let k = 0; k < toks.length; k++) {
    if (isI(k, "for") || isI(k, "while")) { const o = sig(k); if (isP(o, "(")) { const c = open.get(o); const b = sig(c); loops.push({ start: k, end: isP(b, "{") ? open.get(b) : c, headTok: toks[k], kind: toks[k].text }); } }
    if (isI(k, "do") && isP(sig(k), "{")) loops.push({ start: k, end: open.get(sig(k)), headTok: toks[k], kind: "do" });
  }
  for (const f of fns) if (f.parallelBody) loops.push({ start: f.start, end: f.end, headTok: toks[f.start], kind: "parallel" });
  const loopsAround = (k) => loops.filter((l) => k > l.start && k < l.end);

  // ── verb calls ──────────────────────────────────────────────────────────
  const calls = [];
  for (let k = 0; k < toks.length; k++) {
    if (!isI(k)) continue;
    const name = toks[k].text;
    if (!RESERVED.has(name)) continue;
    const prev = sig(k, -1), next = sig(k);
    if (isP(prev, ".") || isP(prev, "?.")) continue; // a method of something else
    if (isI(prev, "function") || isI(prev, "class")) throw new SandError("E-RESERVED", `\`${name}\` is a SAND name and cannot be declared here; choose another`, toks[k]);
    if (isI(prev, "const") || isI(prev, "let") || isI(prev, "var")) throw new SandError("E-RESERVED", `\`${name}\` is a SAND name and cannot be declared here; choose another`, toks[k]);
    if (!isP(next, "(")) continue;
    if (!ASYNC_VERBS.includes(name) && !SYNC_VERBS.includes(name)) continue;
    calls.push({ verb: name, idx: k, open: next, close: open.get(next), args: callArgs(next), tok: toks[k] });
  }

  // ── prose conditions ────────────────────────────────────────────────────
  let dCounter = 0;
  const newVar = () => `d${++dCounter}`;
  const sites = [];
  const chains = [];
  const whiles = [];

  // parse a condition (tokens a..b inclusive) into a tree of atoms
  function parseCond(a, b) {
    const tokensIn = []; for (let k = a; k <= b; k++) if (!isTrivia(toks[k])) tokensIn.push(k);
    let pos = 0;
    const peek = () => tokensIn[pos], take = () => tokensIn[pos++];
    const isPk = (t) => peek() !== undefined && toks[peek()].kind === "punct" && toks[peek()].text === t;
    function pOr() { let l = pAnd(); while (isPk("||")) { take(); l = { op: "or", l, r: pAnd() }; } return l; }
    function pAnd() { let l = pUnary(); while (isPk("&&")) { take(); l = { op: "and", l, r: pUnary() }; } return l; }
    function pUnary() {
      if (isPk("!")) { take(); return { op: "not", x: pUnary() }; }
      if (isPk("(")) {
        const o = take(); const c = open.get(o);
        const inner = parseCond(o + 1, c - 1); while (peek() !== undefined && peek() <= c) take();
        return inner;
      }
      const s = peek(); let e = s; let depth = 0;
      while (peek() !== undefined) {
        const k = peek(); const t = toks[k];
        if (t.kind === "punct" && "([{".includes(t.text)) depth++;
        if (t.kind === "punct" && ")]}".includes(t.text)) depth--;
        if (depth === 0 && t.kind === "punct" && (t.text === "&&" || t.text === "||")) break;
        e = k; take();
      }
      return { op: "atom", a: s, b: e };
    }
    return pOr();
  }
  const atomBindings = (a, b) => { const out = []; for (let k = a; k <= b; k++) if (toks[k].kind === "binding") out.push(k); return out; };
  const wrapNot = (s, neg) => (neg ? `not(${s})` : s);
  const norm = (s) => s.toLowerCase().replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();

  // classify a prose atom
  function classifyAtom(a, b) {
    const binds = atomBindings(a, b);
    const tok = toks[a];
    if (binds.length > 1) {
      const p0 = toks[binds[0]].path, p1 = toks[binds[1]].path;
      throw new SandError("E-BINDING", `this predicate carries two bindings, {{${p0}}} and {{${p1}}}, and a decision reads one state; join them first: const pair = \`${p0.split(".").pop()}: \${${p0}} · ${p1.split(".").pop()}: \${${p1}}\` as "the pair"`, tok);
    }
    const path = toks[binds[0]].path;
    let raw = slice(a, b).replace(/\s+/g, " ").trim();
    let contraries = null;
    const um = /\sunless\s+([\s\S]+)$/.exec(raw);
    if (um) { contraries = um[1].split(";").map((s) => s.trim()).filter(Boolean); raw = raw.slice(0, um.index).trim(); }
    const noun = nounOf(path);
    const pathRe = path.replace(/[.$]/g, "\\$&");
    const means = raw.replace(new RegExp(`(?:\\b(?:the|a|an|this)\\s+)?\\{\\{\\s*${pathRe}\\s*\\}\\}`, "g"), noun);
    const m = new RegExp(`^(?:(?:the|a|an|this)\\s+)?\\{\\{\\s*${pathRe}\\s*\\}\\}\\s+is\\s+(not\\s+)?(?:(?:a|an)\\s+)?(.+?)(?:\\s+on\\s+([A-Z][\\w$]*))?$`).exec(raw);
    if (m) {
      const negated = !!m[1]; const phrase = m[2].trim(); const onName = m[3] ?? null;
      const cmp = /^(at least|at most|above|below)\s+(.+)$/.exec(phrase);
      if (onName && scales.has(onName)) {
        const sc = scales.get(onName);
        const lvl = cmp ? cmp[2] : phrase;
        const level = sc.levels.find((l) => norm(l) === norm(lvl));
        if (!level) throw new SandError("E-OPTION", `\`${phrase}\` names no level of \`${onName}\`, which has ${sc.levels.join(", ")}`, tok);
        if (contraries) throw new SandError("E-UNLESS", "this program carries `unless` on a scale test, which only a prose predicate reads", tok);
        const method = cmp ? { "at least": "atLeast", "at most": "atMost", above: "above", below: "below" }[cmp[1]] : "is";
        return { how: `score:${onName}`, spec: onName, specKey: `scale:${onName}`, verdict: (dv) => wrapNot(`${dv}.${method}(${q(level)})`, negated), binding: path, tok, negated, text: raw };
      }
      if (onName && kinds.has(onName)) {
        const kd = kinds.get(onName);
        const id = kd.options.find((o) => norm(o) === norm(phrase));
        if (!id) throw new SandError("E-OPTION", `\`${phrase}\` names no option of \`${onName}\`, which has ${kd.options.join(", ")}`, tok);
        if (contraries) throw new SandError("E-UNLESS", `this program carries \`unless\` on a kind test, which only a prose predicate reads; put the contraries on the option: ${id}: { means, not: [...] }`, tok);
        return { how: `choice:${onName}`, spec: onName, specKey: `kind:${onName}`, verdict: (dv) => wrapNot(`${dv}.is(${q(id)})`, negated), binding: path, tok, negated, text: raw };
      }
      if (onName) throw new SandError("E-OPTION", `\`on ${onName}\` names no kind or scale declared above it`, tok);
      if (!cmp) {
        const hits = [];
        for (const [kname, kd] of kinds) { const id = kd.options.find((o) => norm(o) === norm(phrase)); if (id) hits.push({ kname, id }); }
        if (hits.length > 1) throw new SandError("E-OPTION", `"${phrase}" is an option of ${hits.map((h) => h.kname).join(" and of ")}; say which: is a ${phrase} on ${hits[0].kname}`, tok);
        if (hits.length === 1) {
          if (contraries) throw new SandError("E-UNLESS", `this program carries \`unless\` on a kind test, which only a prose predicate reads; put the contraries on the option: ${hits[0].id}: { means, not: [...] }`, tok);
          return { how: `choice:${hits[0].kname}`, spec: hits[0].kname, specKey: `kind:${hits[0].kname}`, verdict: (dv) => wrapNot(`${dv}.is(${q(hits[0].id)})`, negated), binding: path, tok, negated, text: raw };
        }
        for (const [kname, kd] of kinds) for (const o of kd.options) if (norm(o).replace(/s$/, "") === norm(phrase).replace(/s$/, "")) warn("W-NEAR-OPTION", tok, `\`${phrase}\` reads like the option \`${o}\` of \`${kname}\` but is asked as a bare predicate; write \`is a ${o}\` for the kind test`);
      }
    }
    const not = contraries ?? GENERIC_CONTRARIES;
    if (!contraries) warn("W-GENERIC-CONTRARY", tok, "no contrary given; the decision will be weak. Add `unless <what this can be mistaken for>`");
    return { how: "noul", spec: `{ means: ${q(means)}, not: ${q(not)}, noun: ${q(noun)} }`, specKey: "noul:" + JSON.stringify([means, not]), means, not, noun, verdict: (dv) => `${dv}.value`, binding: path, tok, text: raw };
  }

  // resolve a parsed tree into a verdict expression, collecting entries into `group`
  function resolveTree(tree, group) {
    if (tree.op === "atom") {
      const binds = atomBindings(tree.a, tree.b);
      if (!binds.length) return { verdict: `(${slice(tree.a, tree.b).trim()})`, atoms: [] };
      const c = classifyAtom(tree.a, tree.b);
      let entry = group.entries.find((e) => e.specKey === c.specKey && e.binding === c.binding);
      if (!entry) { entry = { specKey: c.specKey, spec: c.spec, binding: c.binding, dv: newVar(), atoms: [] }; group.entries.push(entry); }
      entry.atoms.push(c);
      for (const k of binds) consumed.add(k);
      return { verdict: c.verdict(entry.dv), atoms: [c] };
    }
    if (tree.op === "not") { const x = resolveTree(tree.x, group); return { verdict: `not(${x.verdict})`, atoms: x.atoms }; }
    const l = resolveTree(tree.l, group), r = resolveTree(tree.r, group);
    return { verdict: `${tree.op}(${l.verdict}, ${r.verdict})`, atoms: [...l.atoms, ...r.atoms] };
  }

  // find every if-chain and while with a prose condition
  const seenIf = new Set();
  for (let k = 0; k < toks.length; k++) {
    if (isI(k, "if") && !seenIf.has(k)) {
      if (isI(sig(k, -1), "else")) continue;
      const o = sig(k); if (!isP(o, "(")) continue;
      const parts = []; let cur = k; let any = false;
      while (true) {
        const co = sig(cur); const cc = open.get(co); const bo = sig(cc);
        const hasB = atomBindings(co + 1, cc - 1).length > 0;
        if (!isP(bo, "{")) { if (hasB) throw new SandError("E-BLOCK", "this prose `if` has no `{`; SAND needs braces around its branches", toks[cur]); break; }
        parts.push({ kind: parts.length ? "elseif" : "if", ifTok: cur, condOpen: co, condClose: cc, bodyOpen: bo, bodyClose: open.get(bo), prose: hasB });
        any = any || hasB;
        seenIf.add(cur);
        const after = sig(open.get(bo));
        if (isI(after, "else")) {
          const nx = sig(after);
          if (isI(nx, "if")) { cur = nx; continue; }
          if (isP(nx, "{")) { parts.push({ kind: "else", elseTok: after, bodyOpen: nx, bodyClose: open.get(nx) }); break; }
          if (any) throw new SandError("E-BLOCK", "this prose `if` has an `else` without `{`; SAND needs braces around its branches", toks[after]);
          break;
        }
        break;
      }
      if (!parts.length) continue;
      const last = parts[parts.length - 1];
      const afterChain = sig(last.bodyClose);
      if (isI(afterChain, "unsure") && isP(sig(afterChain), "{")) {
        if (!any) throw new SandError("E-UNSURE", "this `unsure` follows a plain condition, which cannot be unsure; test the value: if (x === unsure)", toks[afterChain]);
        parts.push({ kind: "unsure", unsureTok: afterChain, bodyOpen: sig(afterChain), bodyClose: open.get(sig(afterChain)) });
      }
      if (any) chains.push({ headTok: toks[k], headIdx: k, parts, ctx: ctxAt[k], loops: loopsAround(k), encl: enclosingBrace(k) });
    }
    if (isI(k, "while")) {
      const o = sig(k); if (!isP(o, "(")) continue;
      const c = open.get(o);
      if (atomBindings(o + 1, c - 1).length) whiles.push({ tok: toks[k], idx: k, condOpen: o, condClose: c, ctx: ctxAt[k], loops: loopsAround(k) });
    }
    if (isI(k, "unsure") && isP(sig(k), "{") && !isP(sig(k, -1), "}")) throw new SandError("E-UNSURE", "this `unsure` block follows no `if` chain", toks[k]);
  }

  const chainEnd = (ch) => ch.parts[ch.parts.length - 1].bodyClose;
  for (const ch of chains) {
    if (!ch.ctx.async) throw new SandError("E-ASYNC", "this predicate sits in a function that is not async; write async (x) => … or use parallel(items, async (x) => …)", ch.headTok);
    const group = { entries: [], binding: null, inlineVars: [] };
    for (const part of ch.parts) {
      if (!part.prose) continue;
      const sub = { entries: [] };
      const r = resolveTree(parseCond(part.condOpen + 1, part.condClose - 1), sub);
      part.verdict = r.verdict;
      if (!group.binding) group.binding = sub.entries[0].binding;
      part.inline = [];
      for (const e of sub.entries) {
        if (e.binding === group.binding) {
          const dup = group.entries.find((g) => g.specKey === e.specKey);
          if (dup) { part.verdict = part.verdict.split(e.dv).join(dup.dv); for (const a of e.atoms) dup.atoms.push(a); }
          else group.entries.push(e);
        } else { part.inline.push(e); group.inlineVars.push(e); }
      }
    }
    for (const e of [...group.entries, ...group.inlineVars]) for (const a of e.atoms) sites.push({ id: e.dv, at: at(a.tok), text: a.text, how: a.how, binding: e.binding, model: ch.ctx.usingModel ?? "default:decide", loops: ch.loops.map((l) => l.kind), conditional: ch.encl >= 0 });
    ch.group = group;
  }
  for (const w of whiles) {
    if (!w.ctx.async) throw new SandError("E-ASYNC", "this predicate sits in a function that is not async; write async (x) => … or use parallel(items, async (x) => …)", w.tok);
    const sub = { entries: [] };
    const r = resolveTree(parseCond(w.condOpen + 1, w.condClose - 1), sub);
    w.verdict = r.verdict; w.entries = sub.entries;
    for (const e of sub.entries) for (const a of e.atoms) sites.push({ id: e.dv, at: at(a.tok), text: a.text, how: a.how, binding: e.binding, model: w.ctx.usingModel ?? "default:decide", loops: [...w.loops.map((l) => l.kind), "while"], conditional: false });
    if (!settlePasses) warn("W-UNBOUNDED-LOOP", w.tok, "this `while` asks a model on every turn and nothing bounds it; add settle({ passes: N })");
  }

  // ── T7 hoisting: consecutive chains in one block on one binding ─────────
  const assignsBinding = (fromEnd, toStart, binding) => {
    const head = binding.split(".")[0].replace(/\$/g, "\\$");
    const text = src.slice(fromEnd, toStart);
    return new RegExp(`(?:^|[^\\w$.])${head}\\s*(?:=[^=>]|\\+=|-=|\\*=|/=|\\+\\+|--)|\\b(?:let|const|var)\\s+${head}\\b`).test(text);
  };
  const explicitSites = calls.filter((c) => ["decide", "decideAll", "extract"].includes(c.verb) && c.args[0] && c.args[0][0] === c.args[0][1] && toks[c.args[0][0]].kind === "binding")
    .map((c) => ({ call: c, binding: toks[c.args[0][0]].path, spec: c.args[1] ? slice(c.args[1][0], c.args[1][1]) : null, using: c.args[2] ? (/using\s*:\s*["'](ask|decide)["']/.exec(slice(c.args[2][0], c.args[2][1])) || [])[1] ?? null : null, idx: c.idx, encl: enclosingBrace(c.idx), ctx: ctxAt[c.idx], loops: loopsAround(c.idx) }));
  for (let i = 0; i < chains.length; i++) {
    const A = chains[i]; if (A.mergedInto) continue;
    for (let j = i + 1; j < chains.length; j++) {
      const B = chains[j];
      if (B.mergedInto || B.group.binding !== A.group.binding || B.headIdx < chainEnd(A)) continue;
      if (enclosingBrace(A.headIdx) !== enclosingBrace(B.headIdx) || B.ctx !== A.ctx) continue;
      if (loopsAround(B.headIdx).length !== loopsAround(A.headIdx).length) continue;
      if (assignsBinding(toks[chainEnd(A)].end, toks[B.headIdx].start, A.group.binding)) break;
      for (const e of B.group.entries) {
        const dup = A.group.entries.find((g) => g.specKey === e.specKey);
        if (dup) { for (const p of B.parts) if (p.verdict) p.verdict = p.verdict.split(e.dv).join(dup.dv); for (const a of e.atoms) dup.atoms.push(a); }
        else A.group.entries.push(e);
      }
      A.group.inlineVars.push(...B.group.inlineVars);
      B.mergedInto = A;
    }
  }

  // ── T7b speculation: nested same-binding sites join the head request ────
  function soleShape(tok) {
    if (shapes.size === 1) return [...shapes.keys()][0];
    if (shapes.size === 0) throw new SandError("E-SHAPE", "extract needs a shape and this file declares none; declare one: const S = shape({ slot: Kind, … }) and write extract({{input}}, S)", tok);
    throw new SandError("E-SHAPE", `extract needs a shape and this file declares ${shapes.size} (${[...shapes.keys()].join(", ")}); name it: extract({{input}}, ${[...shapes.keys()][0]})`, tok);
  }
  function speculationFor(headIdx, binding, ctx, headKeys, headCallIdx = -1) {
    const encl = enclosingBrace(headIdx);
    const blockEnd = encl >= 0 ? open.get(encl) : toks.length;
    const out = []; const keys = new Set(headKeys); const covered = [];
    const push = (key, spec) => { if (keys.has(key)) return true; if (out.length >= 15) return false; keys.add(key); out.push(spec); return true; };
    for (const ch of chains) {
      if (ch.headIdx <= headIdx || ch.headIdx >= blockEnd || ch.mergedInto) continue;
      if (ch.group.binding !== binding || ch.ctx !== ctx) continue;
      if (loopsAround(ch.headIdx).length !== loopsAround(headIdx).length) continue;
      if (enclosingBrace(ch.headIdx) === encl) continue; // a sibling: hoisting's job, or reached anyway
      if (assignsBinding(toks[headIdx].end, toks[ch.headIdx].start, binding)) break;
      let all = true;
      for (const e of ch.group.entries) all = push(e.specKey, e.spec) && all;
      if (all) covered.push(ch);
    }
    for (const s of explicitSites) {
      if (s.idx <= headIdx || s.idx === headCallIdx || s.idx >= blockEnd || s.binding !== binding || s.ctx !== ctx) continue;
      if (loopsAround(s.idx).length !== loopsAround(headIdx).length) continue;
      if (assignsBinding(toks[headIdx].end, toks[s.idx].start, binding)) break;
      if (s.call.verb === "extract") { const sh = s.spec ?? soleShape(s.call.tok); push(`shape:${sh}`, sh); }
      else if (s.call.verb === "decide" && s.spec && (kinds.has(s.spec) || scales.has(s.spec))) push(`${kinds.has(s.spec) ? "kind" : "scale"}:${s.spec}`, s.spec);
    }
    for (const ch of covered) ch.speculated = true;
    return out;
  }
  // explicit decide/extract statements are heads too: the questions nested below them ride along
  const explicitSpeculation = new Map(); // call idx → spec list
  for (const s of explicitSites) {
    const prev = sig(s.idx, -1);
    const statementLevel = prev < 0 || isP(prev, ";") || isP(prev, "{") || isP(prev, "}") || isP(prev, "=") || isI(prev, "return") || (isP(prev, "(") && isI(sig(prev, -1), "switch"));
    if (!statementLevel || s.loops.length) continue;
    const own = s.call.verb === "extract" ? [`shape:${s.spec ?? (shapes.size === 1 ? [...shapes.keys()][0] : "")}`] : s.spec ? [`${kinds.has(s.spec) ? "kind" : "scale"}:${s.spec}`] : [];
    const specs = speculationFor(s.idx, s.binding, s.ctx, own, s.idx);
    if (specs.length) explicitSpeculation.set(s.call.idx, specs);
  }

  // ── emit chains ─────────────────────────────────────────────────────────
  const batches = [];
  const singleCall = (e, binding, atTok, modelOpt, extra = "") => e.specKey.startsWith("noul:")
    ? `decide(${binding}, ${q(e.atoms[0].means)}, { not: ${q(e.atoms[0].not)}, noun: ${q(e.atoms[0].noun)}, at: ${q(at(atTok))}${modelOpt}${extra} })`
    : `decide(${binding}, ${e.spec}, { at: ${q(at(atTok))}${modelOpt}${extra} })`;
  for (const ch of chains) {
    const g = ch.group;
    const modelOpt = ch.ctx.usingModel ? `, model: ${q(ch.ctx.usingModel)}` : "";
    if (!ch.mergedInto) {
      const spec = speculationFor(ch.headIdx, g.binding, ch.ctx, g.entries.map((e) => e.specKey));
      const specOpt = spec.length ? `, speculate: [${spec.join(", ")}]` : "";
      let prefix = "";
      if (g.inlineVars.length) prefix += `let ${[...new Set(g.inlineVars.map((e) => e.dv))].join(", ")};  `;
      if (g.entries.length === 1) {
        prefix += `const ${g.entries[0].dv} = await ${singleCall(g.entries[0], g.binding, ch.headTok, modelOpt, specOpt)};  `;
      } else if (g.entries.length > 1) {
        if (g.entries.length > 16) warn("W-BATCH-SPLIT", ch.headTok, `${g.entries.length} questions at one site; the runtime splits them into requests of 16`);
        prefix += `const [${g.entries.map((e) => e.dv).join(", ")}] = await decideAll(${g.binding}, [${g.entries.map((e) => e.spec).join(", ")}], { at: ${q(at(ch.headTok))}${modelOpt}${specOpt} });  `;
      }
      insert(toks[ch.headIdx].start, prefix);
      ch.batch = { id: batches.length + 1, at: at(ch.headTok), binding: g.binding, sites: g.entries.map((e) => e.dv), speculative: spec, speculated: false, questions: g.entries.length, loops: ch.loops.map((l) => l.kind), conditional: ch.encl >= 0, model: ch.ctx.usingModel ?? "default:decide" };
      batches.push(ch.batch);
    }
    const proseParts = ch.parts.filter((p) => p.prose);
    const unsureTest = proseParts.map((p) => `(${p.verdict}) === unsure`).join(" || ");
    const notUnsureTest = proseParts.map((p) => `(${p.verdict}) !== unsure`).join(" && ");
    for (const part of ch.parts) {
      if (part.prose) {
        let v = part.verdict;
        for (const e of part.inline) v = v.replace(new RegExp(`\\b${e.dv}\\b`), `(${e.dv} = await ${singleCall(e, e.binding, e.atoms[0].tok, modelOpt)})`);
        replace(toks[part.condOpen].start, toks[part.condClose].end, `((${v}) === true)`);
      }
      if (part.kind === "else" && ch.parts.some((p) => p.kind === "unsure")) replace(toks[part.elseTok].start, toks[part.bodyOpen].end, `else if (${notUnsureTest.replace(/\b(d\d+)\b\.value/g, (m, d) => (g.inlineVars.some((e) => e.dv === d) ? `${d}?.value` : m))}) {`);
      if (part.kind === "unsure") {
        const hasElse = ch.parts.some((p) => p.kind === "else");
        replace(toks[part.unsureTok].start, toks[part.bodyOpen].end, hasElse ? "else {" : `else if (${unsureTest.replace(/\b(d\d+)\b\.value/g, (m, d) => (g.inlineVars.some((e) => e.dv === d) ? `${d}?.value` : m))}) {`);
      }
    }
  }
  for (const w of whiles) {
    const modelOpt = w.ctx.usingModel ? `, model: ${q(w.ctx.usingModel)}` : "";
    let v;
    if (w.entries.length === 1) {
      const e = w.entries[0];
      v = w.verdict.replace(new RegExp(`\\b${e.dv}\\b`, "g"), `(await ${singleCall(e, e.binding, w.tok, modelOpt)})`);
    } else {
      const byBinding = new Map();
      for (const e of w.entries) { if (!byBinding.has(e.binding)) byBinding.set(e.binding, []); byBinding.get(e.binding).push(e); }
      const params = [], inner = [];
      for (const [b, es] of byBinding) { params.push(`[${es.map((e) => e.dv).join(", ")}]`); inner.push(`await decideAll(${b}, [${es.map((e) => e.spec).join(", ")}], { at: ${q(at(w.tok))}${modelOpt} })`); }
      v = `((${params.join(", ")}) => ${w.verdict})(${inner.join(", ")})`;
    }
    replace(toks[w.condOpen].start, toks[w.condClose].end, `((${v}) === true)`);
    batches.push({ id: batches.length + 1, at: at(w.tok), binding: w.entries[0]?.binding, sites: w.entries.map((e) => e.dv), speculative: [], questions: w.entries.length, loops: [...w.loops.map((l) => l.kind), "while"], conditional: false, model: w.ctx.usingModel ?? "default:decide" });
  }

  for (const ch of chains) if (ch.speculated && ch.batch) ch.batch.speculated = true;

  // ── T1 bindings outside conditions; strings and templates ───────────────
  const BINDING_RE = /\{\{\s*([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\}\}/g;
  for (let k = 0; k < toks.length; k++) {
    const t = toks[k];
    if (t.kind === "binding" && !consumed.has(k)) {
      const p = sig(k, -1), n = sig(k);
      const wordy = (j) => j >= 0 && toks[j].kind === "ident" && !["as", "in", "of", "instanceof", "typeof", "return", "await", "case", "const", "let", "var", "new", "void", "delete", "yield", "throw", "else", "do"].includes(toks[j].text);
      if (wordy(p) || wordy(n)) throw new SandError("E-PROSE-PLACE", "prose predicates are read only in `if`, `else if` and `while` conditions; put this one there", t);
      replace(t.start, t.end, t.path);
    }
    if (t.kind === "string" && t.text.includes("{{")) {
      const inner = t.text.slice(1, -1).replace(/`/g, "\\`").replace(/\$\{/g, "\\${");
      replace(t.start, t.end, "`" + inner.replace(BINDING_RE, "${$1}") + "`");
    }
    if (t.kind === "template" && t.text.includes("{{")) {
      let out = "", i = 0, depth = 0, seg = "";
      const flush = () => { out += seg.replace(BINDING_RE, "${$1}"); seg = ""; };
      while (i < t.text.length) {
        const c = t.text[i];
        if (depth === 0) {
          if (c === "\\") { seg += c + (t.text[i + 1] ?? ""); i += 2; continue; }
          if (c === "$" && t.text[i + 1] === "{") { flush(); out += "${"; depth = 1; i += 2; continue; }
          seg += c; i++;
        } else {
          if (c === "{") depth++; else if (c === "}") depth--;
          out += c; i++;
        }
      }
      flush();
      replace(t.start, t.end, out);
    }
  }

  // ── T4/T5/T6/T8: verb calls ─────────────────────────────────────────────
  const declaredNames = new Map();
  for (const d of [...kinds.values(), ...scales.values(), ...shapes.values()]) declaredNames.set(d.callOpen, d.name);
  function enclosingBracket(k) { let depth = 0; for (let j = k - 1; j >= 0; j--) { const t = toks[j]; if (t.kind !== "punct") continue; if (")]}".includes(t.text)) depth++; else if ("([{".includes(t.text)) { if (depth === 0) return j; depth--; } } return -1; }
  for (const c of calls) {
    const ctx = ctxAt[c.idx];
    if (c.verb === "extract" && c.args.length === 1) {
      const sh = soleShape(c.tok);
      insert(toks[c.args[0][1]].end, `, ${sh}`);
      c.args.push(null);
      warnings.push({ code: "N-EXTRACT-SHAPE", at: at(c.tok), message: `\`extract\` names no shape; the one in scope, \`${sh}\`, was used` });
    }
    const oi = OPTS_INDEX[c.verb] ?? DECL_OPTS_INDEX[c.verb];
    if (oi !== undefined) {
      const stamps = [];
      if (OPTS_INDEX[c.verb] !== undefined) {
        stamps.push(`at: ${q(at(c.tok))}`);
        if (ctx.usingModel && ["decide", "decideAll", "classify", "ask", "extract"].includes(c.verb)) stamps.push(`model: ${q(ctx.usingModel)}`);
        if (explicitSpeculation.has(c.idx)) stamps.push(`speculate: [${explicitSpeculation.get(c.idx).join(", ")}]`);
      }
      else if (declaredNames.has(c.open)) stamps.push(`name: ${q(declaredNames.get(c.open))}`);
      if (stamps.length) {
        const nargs = c.args.length;
        if (nargs === oi + 1 && c.args[oi] && isP(c.args[oi][0], "{")) {
          const o = c.args[oi][0];
          insert(toks[o].end, ` ${stamps.join(", ")}${sig(o) === open.get(o) ? " " : ","}`);
        } else if (nargs === oi) insert(toks[c.close].start, `, { ${stamps.join(", ")} }`);
      }
    }
    if (ASYNC_VERBS.includes(c.verb)) {
      if (isI(sig(c.idx, -1), "await")) continue;
      const encl = enclosingBracket(c.idx);
      const callee = encl >= 0 && isP(encl, "(") ? sig(encl, -1) : -1;
      const inArray = encl >= 0 && isP(encl, "[");
      const inCombinator = callee >= 0 && (isI(callee, "race") || isI(callee, "timeout") || (isI(callee) && isP(sig(callee, -1), ".") && isI(sig(sig(callee, -1), -1), "Promise")));
      if (inArray || inCombinator) continue;
      if (!ctx.async) { warn("W-SYNC-VERB", c.tok, `\`${c.verb}\` is called inside a function that is not async, so its promise is left unawaited`); continue; }
      insert(c.tok.start, "(await ");
      insert(toks[c.close].end, ")");
      const before = sig(c.idx, -1);
      if (isP(before, "(") && isI(sig(before, -1), "switch") && open.get(before) === sig(c.close)) insert(toks[c.close].end, ".value");
    }
  }

  // ── whole-file warnings ─────────────────────────────────────────────────
  if (!settleTok && (chains.length || whiles.length || explicitSites.length)) warn("W-UNSETTLED", toks[0], "no settle(); nothing can abstain and every unsure branch is dead");
  for (const d of [...kinds.values(), ...scales.values()]) {
    const used = sites.some((s) => s.how.endsWith(`:${d.name}`)) || explicitSites.some((s) => s.spec === d.name) || [...shapes.values()].some((sh) => Object.values(sh.slots).includes(d.name));
    if (!used) warnings.push({ code: "N-UNUSED", at: d.at, message: `\`${d.name}\` is declared and nothing reads it` });
  }

  // ── apply the splices ───────────────────────────────────────────────────
  edits.sort((a, b) => b.start - a.start || b.end - a.end || b.order - a.order);
  let body = src;
  for (const e of edits) body = body.slice(0, e.start) + e.text + body.slice(e.end);
  const importLines = [];
  body = body.replace(/^[ \t]*import\s[^\n]*?;?[ \t]*$/gm, (m) => { importLines.push(m.trim()); return ""; });
  const importLine = importStyle === "global"
    ? `const { ${IMPORT_NAMES} } = globalThis.SAND;${importLines.length ? " " + importLines.join(" ") : ""}`
    : `import { ${IMPORT_NAMES} } from ${q(opts.importPath ?? (importStyle === "relative" ? "./sand/mod.ts" : "@combs/sand"))};${importLines.length ? " " + importLines.join(" ") : ""}`;
  const code = `${importLine}\nexport default async function main(input: string) {\n${body}\n}\nif (import.meta.main) run(main);\n`;

  // ── the plan ────────────────────────────────────────────────────────────
  const explicitPlan = explicitSites.map((s) => ({ at: at(s.call.tok), verb: s.call.verb, binding: s.binding, spec: s.spec ?? (s.call.verb === "extract" && shapes.size === 1 ? [...shapes.keys()][0] : null), using: s.using, loops: s.loops.map((l) => l.kind), conditional: s.encl >= 0, speculative: explicitSpeculation.get(s.call.idx) ?? [] }));
  const askCalls = calls.filter((c) => c.verb === "ask").map((c) => ({ at: at(c.tok), verb: "ask", loops: loopsAround(c.idx).map((l) => l.kind), conditional: enclosingBrace(c.idx) >= 0 }));
  const cost = computeCost(batches, explicitPlan, askCalls, shapes, settlePasses);
  const plan = {
    sand: "0.1", file, hash: fnv1a(src), lines: src.split("\n").length,
    declarations: {
      kinds: [...kinds.values()].map((d) => ({ name: d.name, at: d.at, options: d.options, fallback: d.fallback })),
      scales: [...scales.values()].map((d) => ({ name: d.name, at: d.at, levels: d.levels })),
      shapes: [...shapes.values()].map((d) => ({ name: d.name, at: d.at, slots: d.slots })),
    },
    settle: settleTok ? { at: at(settleTok), passes: settlePasses } : null,
    sites: sites.map((s) => ({ id: s.id, at: s.at, text: s.text, how: s.how, binding: s.binding, model: s.model, loops: s.loops, batch: batches.find((b) => b.sites.includes(s.id))?.id ?? null })),
    batches,
    calls: [...explicitPlan, ...askCalls],
    loops: loops.map((l) => ({ at: at(l.headTok), kind: l.kind })),
    cost,
    warnings,
  };
  return { code, plan, warnings };
}

function computeCost(batches, explicit, asks, shapes, settlePasses) {
  let minP = 0, maxP = 0, minC = 0, maxC = 0; const loopTerms = []; let unbounded = false;
  for (const b of batches) {
    if (b.speculated) continue;
    if (b.loops.length) { if (b.loops.includes("while") && !settlePasses) unbounded = true; loopTerms.push("n × 1 pass"); continue; }
    maxP += 1; if (!b.conditional) minP += 1;
  }
  for (const e of explicit) {
    if (e.loops.length) { loopTerms.push("n × 1 pass"); continue; }
    const speculated = batches.some((b) => !b.loops.length && b.speculative.includes(e.spec ?? "")) || [...explicit].some((h) => h !== e && h.speculative?.includes(e.spec ?? ""));
    if (e.verb === "extract") {
      const sh = e.spec ? shapes.get(e.spec) : [...shapes.values()][0];
      const kinded = sh ? Object.values(sh.slots).some((t) => t !== "text") : true;
      const texts = sh ? Object.values(sh.slots).some((t) => t === "text") : false;
      if (e.using === "ask") { maxC += 1; if (!e.conditional) minC += 1; continue; }
      if (kinded && !speculated) { maxP += 1; if (!e.conditional) minP += 1; }
      if (texts) { maxC += 1; if (!e.conditional) minC += 1; }
    } else if (!speculated) { maxP += 1; if (!e.conditional) minP += 1; }
  }
  for (const a of asks) { if (a.loops.length) { loopTerms.push("n × 1 completion"); continue; } maxC += 1; if (!a.conditional) minC += 1; }
  const fmt = (p, c) => [p ? `${p} pass${p === 1 ? "" : "es"}` : null, c ? `${c} completion${c === 1 ? "" : "s"}` : null].filter(Boolean).join(", ") || "0 passes";
  const text = `min ${fmt(minP, minC)} · max ${fmt(maxP, maxC)}${loopTerms.length ? " + " + loopTerms.join(" + ") : ""}${unbounded ? " (unbounded: a while loop asks a model with no settle({ passes }))" : ""}`;
  return { min: { passes: minP, completions: minC }, max: { passes: maxP, completions: maxC }, loops: loopTerms, unbounded, text };
}

function fnv1a(s) { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return "fnv1a:" + h.toString(16).padStart(8, "0"); }
