//! Constrained decoding: OpenAI `response_format` enforcement.
//!
//! Three layers, all built dynamically — nothing is baked per model:
//!
//! 1. [`ConstraintSpec`] — plain-data request config parsed from the wire
//!    (`json_object` / `json_schema`), carried on `GenerationConfig`.
//! 2. [`TokenByteTable`] — per-model map from token id to the exact bytes
//!    that token appends to the output stream, derived from the tokenizer at
//!    first use and validated against the tokenizer's own decoder.
//! 3. [`ConstraintState`] — a byte-level JSON pushdown automaton with the
//!    schema overlay compiled at request time. The engine masks the logits
//!    row with it before *every* sample and advances it on the sampled
//!    token.
//!
//! The mask runs engine-side rather than as a `LogitsProcessor` on purpose:
//! the greedy sampler skips the temperature/top-k chain, and a constraint is
//! correctness, not preference — it must bind under every sampler.

use std::collections::HashSet;

use serde_json::Value;
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// Spec
// ---------------------------------------------------------------------------

/// A structured-output constraint, parsed from OpenAI `response_format`.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintSpec {
    /// Output must be one syntactically valid JSON object.
    JsonObject,
    /// Output must conform to a JSON Schema subset (see
    /// [`CompiledSchema::compile`] for the supported keywords).
    JsonSchema(Value),
}

impl ConstraintSpec {
    /// Parses an OpenAI `response_format` value. `{"type":"text"}` (the
    /// explicit default) yields `None`; unknown types and malformed
    /// `json_schema` payloads are request errors.
    pub fn from_response_format(v: &Value) -> Result<Option<ConstraintSpec>, String> {
        let ty = v
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| "response_format.type must be a string".to_string())?;
        match ty {
            "text" => Ok(None),
            "json_object" => Ok(Some(ConstraintSpec::JsonObject)),
            "json_schema" => {
                // Wire shape: {"type":"json_schema","json_schema":{"name":…,
                // "schema":{…}}}; some clients flatten "schema" to the top.
                let schema = v
                    .get("json_schema")
                    .and_then(|js| js.get("schema"))
                    .or_else(|| v.get("schema"))
                    .ok_or_else(|| {
                        "response_format.json_schema.schema is required".to_string()
                    })?;
                Ok(Some(ConstraintSpec::JsonSchema(schema.clone())))
            }
            other => Err(format!("unsupported response_format.type: {other:?}")),
        }
    }

    /// Compiles the spec; call at request time so malformed schemas are
    /// clean request errors, never generation-time panics.
    pub fn compile(&self) -> Result<CompiledSchema, String> {
        CompiledSchema::compile(self)
    }
}

// ---------------------------------------------------------------------------
// Schema compilation
// ---------------------------------------------------------------------------

type NodeId = u16;

/// Arena slot 0: unconstrained value.
const ANY: NodeId = 0;
/// Arena slot 1: any JSON object (json_object mode / nested Any objects).
const ANY_OBJ: NodeId = 1;
/// Arena slot 2: any JSON array.
const ANY_ARR: NodeId = 2;

#[derive(Debug)]
enum SchemaNode {
    Any,
    Object {
        /// Property name → value node, in schema order (≤ 64 — bitmask).
        props: Vec<(String, NodeId)>,
        /// Bitmask over `props` of required names.
        required: u64,
        /// Value node for undeclared keys; `None` = additionalProperties
        /// false (closed object).
        additional: Option<NodeId>,
    },
    Array {
        items: NodeId,
    },
    Str,
    Num {
        integer: bool,
    },
    Bool,
    Null,
    /// String enum: value must be exactly one of the literals (≤ 64).
    EnumStr {
        literals: Vec<String>,
    },
}

/// A compiled schema: node arena + root. Compilation is pure data → data,
/// so the server can validate a request before it ever reaches the worker.
#[derive(Debug)]
pub struct CompiledSchema {
    nodes: Vec<SchemaNode>,
    root: NodeId,
}

/// Keywords that change validation semantics and are not enforced in v1;
/// silently ignoring them would claim conformance we don't check, so they
/// are request errors instead.
const UNSUPPORTED_KEYWORDS: &[&str] = &[
    "$ref", "$defs", "definitions", "anyOf", "oneOf", "allOf", "not", "if", "then", "else",
    "pattern", "format", "patternProperties", "propertyNames", "dependentRequired",
    "dependentSchemas", "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum",
    "exclusiveMaximum", "multipleOf", "minItems", "maxItems", "uniqueItems", "contains",
    "prefixItems", "minProperties", "maxProperties",
];

impl CompiledSchema {
    /// Compiles a constraint spec into the node arena.
    pub fn compile(spec: &ConstraintSpec) -> Result<CompiledSchema, String> {
        let mut nodes = vec![
            SchemaNode::Any,
            SchemaNode::Object {
                props: Vec::new(),
                required: 0,
                additional: Some(ANY),
            },
            SchemaNode::Array { items: ANY },
        ];
        let root = match spec {
            ConstraintSpec::JsonObject => ANY_OBJ,
            ConstraintSpec::JsonSchema(v) => compile_node(&mut nodes, v, 0)?,
        };
        Ok(CompiledSchema { nodes, root })
    }

    fn node(&self, id: NodeId) -> &SchemaNode {
        &self.nodes[id as usize]
    }
}

fn compile_node(nodes: &mut Vec<SchemaNode>, v: &Value, depth: usize) -> Result<NodeId, String> {
    if depth > 64 {
        return Err("schema nesting deeper than 64 levels".to_string());
    }
    let obj = match v {
        Value::Bool(true) => return Ok(ANY),
        Value::Bool(false) => return Err("schema `false` matches nothing".to_string()),
        Value::Object(map) => map,
        _ => return Err("schema must be an object or boolean".to_string()),
    };
    for key in obj.keys() {
        if UNSUPPORTED_KEYWORDS.contains(&key.as_str()) {
            return Err(format!("unsupported schema keyword: {key:?}"));
        }
    }

    if let Some(lits) = obj.get("enum") {
        return push_enum(nodes, lits.as_array().ok_or("enum must be an array")?);
    }
    if let Some(c) = obj.get("const") {
        return push_enum(nodes, std::slice::from_ref(c));
    }

    let ty = match obj.get("type") {
        Some(Value::String(s)) => Some(s.as_str()),
        Some(Value::Array(_)) => return Err("type unions are not supported".to_string()),
        Some(_) => return Err("type must be a string".to_string()),
        // No type: infer from structural keywords, else unconstrained.
        None if obj.contains_key("properties") => Some("object"),
        None if obj.contains_key("items") => Some("array"),
        None => None,
    };

    let node = match ty {
        None => return Ok(ANY),
        Some("object") => {
            let mut props: Vec<(String, NodeId)> = Vec::new();
            if let Some(p) = obj.get("properties") {
                let p = p.as_object().ok_or("properties must be an object")?;
                if p.len() > 64 {
                    return Err("more than 64 properties in one object".to_string());
                }
                for (name, sub) in p {
                    let id = compile_node(nodes, sub, depth + 1)?;
                    props.push((name.clone(), id));
                }
            }
            let mut required = 0u64;
            if let Some(r) = obj.get("required") {
                for name in r.as_array().ok_or("required must be an array")? {
                    let name = name.as_str().ok_or("required entries must be strings")?;
                    let i = props
                        .iter()
                        .position(|(n, _)| n == name)
                        .ok_or_else(|| format!("required property {name:?} is not declared"))?;
                    required |= 1 << i;
                }
            }
            let additional = match obj.get("additionalProperties") {
                None | Some(Value::Bool(true)) => Some(ANY),
                Some(Value::Bool(false)) => None,
                Some(sub) => Some(compile_node(nodes, sub, depth + 1)?),
            };
            SchemaNode::Object {
                props,
                required,
                additional,
            }
        }
        Some("array") => {
            let items = match obj.get("items") {
                Some(sub) => compile_node(nodes, sub, depth + 1)?,
                None => ANY,
            };
            SchemaNode::Array { items }
        }
        Some("string") => SchemaNode::Str,
        Some("number") => SchemaNode::Num { integer: false },
        Some("integer") => SchemaNode::Num { integer: true },
        Some("boolean") => SchemaNode::Bool,
        Some("null") => SchemaNode::Null,
        Some(other) => return Err(format!("unsupported type: {other:?}")),
    };
    push_node(nodes, node)
}

fn push_enum(nodes: &mut Vec<SchemaNode>, lits: &[Value]) -> Result<NodeId, String> {
    if lits.is_empty() || lits.len() > 64 {
        return Err("enum must have 1..=64 entries".to_string());
    }
    let literals = lits
        .iter()
        .map(|l| {
            l.as_str()
                .map(str::to_string)
                .ok_or_else(|| "only string enum/const values are supported".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    push_node(nodes, SchemaNode::EnumStr { literals })
}

fn push_node(nodes: &mut Vec<SchemaNode>, node: SchemaNode) -> Result<NodeId, String> {
    let id = nodes.len();
    if id > u16::MAX as usize {
        return Err("schema too large".to_string());
    }
    nodes.push(node);
    Ok(id as NodeId)
}

// ---------------------------------------------------------------------------
// Token → bytes table
// ---------------------------------------------------------------------------

/// Per-model map from token id to the bytes it appends to the output
/// stream. `None` marks never-emit ids (special/control tokens, undecodable
/// ids); those are only ever allowed as EOS in an accept state.
pub struct TokenByteTable {
    bytes: Vec<Option<Box<[u8]>>>,
    /// Precomputed per-token string-content classification (the mask's
    /// fast path): most decode steps sit inside free string content where
    /// nearly the whole vocabulary is legal, and simulating ~150k tokens
    /// per step costs milliseconds. A "plain" token (no quote, backslash,
    /// or control byte; structurally valid UTF-8) is legal there exactly
    /// when its leading continuation-byte count matches the machine's
    /// pending count — no simulation needed.
    meta: Vec<TokMeta>,
}

/// String-content classification for one token.
#[derive(Clone, Copy, Default)]
struct TokMeta {
    /// All bytes are plain string content and the UTF-8 structure is
    /// stream-valid (incomplete sequences only at the very end).
    plain: bool,
    /// Leading continuation bytes (0x80..=0xBF) before the first full
    /// character.
    prefix_cont: u8,
    /// The token is continuation bytes only (a mid-character splice).
    all_cont: bool,
}

impl TokMeta {
    /// Legal as free string content when the machine has `pending`
    /// continuation bytes outstanding?
    fn legal_in_free_string(self, pending: u8) -> bool {
        if !self.plain {
            return false;
        }
        if self.all_cont {
            // A pure splice may leave the character still incomplete.
            self.prefix_cont >= 1 && self.prefix_cont <= pending
        } else {
            // Content after the splice requires the character to complete
            // exactly at the boundary.
            self.prefix_cont == pending
        }
    }
}

/// Classifies token bytes for the free-string fast path.
fn classify(bytes: &[u8]) -> TokMeta {
    let mut i = 0usize;
    let mut prefix_cont = 0u8;
    while i < bytes.len() && (0x80..=0xBF).contains(&bytes[i]) {
        if prefix_cont == 3 {
            // A 4th leading continuation can never be legal (no state
            // expects more than 3).
            return TokMeta::default();
        }
        prefix_cont += 1;
        i += 1;
    }
    if i == bytes.len() {
        return TokMeta {
            plain: !bytes.is_empty(),
            prefix_cont,
            all_cont: true,
        };
    }
    while i < bytes.len() {
        let need = match bytes[i] {
            0x00..=0x1F | b'"' | b'\\' => return TokMeta::default(),
            0x20..=0x7F => 0,
            0xC2..=0xDF => 1,
            0xE0..=0xEF => 2,
            0xF0..=0xF4 => 3,
            // C0/C1/F5..FF and mid-token bare continuations are never
            // legal string bytes.
            _ => return TokMeta::default(),
        };
        i += 1;
        for _ in 0..need {
            if i == bytes.len() {
                // Incomplete tail is fine: the next token continues it.
                break;
            }
            if !(0x80..=0xBF).contains(&bytes[i]) {
                return TokMeta::default();
            }
            i += 1;
        }
    }
    TokMeta {
        plain: true,
        prefix_cont,
        all_cont: false,
    }
}

impl TokenByteTable {
    /// Derives the table from the tokenizer.
    ///
    /// Fast path: unmap the vocab strings directly (GPT-2 byte-level
    /// reverse table for BPE vocabs; `▁`→space and `<0xNN>`→byte for SPM
    /// vocabs). The result is validated against `tokenizer.decode` on a
    /// probe sample — any disagreement drops the whole table to the slow
    /// decode-per-id path, so a family the fast mapping mispredicts is
    /// detected, never assumed.
    pub fn build(tokenizer: &Tokenizer) -> Self {
        let n = tokenizer.get_vocab_size(true);
        let special: HashSet<u32> = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter(|(_, tok)| tok.special)
            .map(|(id, _)| id)
            .collect();
        let strings: Vec<Option<String>> =
            (0..n as u32).map(|id| tokenizer.id_to_token(id)).collect();

        // Family detection: literal ▁ / <0xNN> only occur in SPM vocabs
        // (byte-level vocabs would show them as mapped multi-char runs).
        let spm = strings
            .iter()
            .flatten()
            .any(|s| s.contains('\u{2581}') || spm_byte_token(s).is_some());

        let unmap = gpt2_unmap();
        let mut bytes: Vec<Option<Box<[u8]>>> = strings
            .iter()
            .enumerate()
            .map(|(id, s)| {
                if special.contains(&(id as u32)) {
                    return None;
                }
                let s = s.as_ref()?;
                let out = if spm {
                    spm_bytes(s)
                } else {
                    byte_level_bytes(s, &unmap)
                };
                Some(out.into_boxed_slice())
            })
            .collect();

        // Anchor for reference pieces: a token whose decode is stable in
        // first position ("a" exists in every real vocab; fall back to any
        // plain id). decode([anchor, id]) − decode([anchor]) is the ground
        // truth for "what does this token append mid-stream".
        let anchor = tokenizer
            .get_vocab(true)
            .get("a")
            .copied()
            .or_else(|| {
                (0..n as u32).find(|id| {
                    !special.contains(id)
                        && bytes
                            .get(*id as usize)
                            .and_then(|b| b.as_ref())
                            .is_some_and(|b| !b.is_empty())
                })
            });
        let Some(anchor) = anchor else {
            return Self::from_bytes_vec(bytes);
        };
        let anchor_text = tokenizer.decode(&[anchor], false).unwrap_or_default();
        let reference = |id: u32| -> Option<String> {
            let full = tokenizer.decode(&[anchor, id], false).ok()?;
            full.strip_prefix(anchor_text.as_str()).map(str::to_string)
        };

        // Probe a deterministic sample of losslessly-decodable ids.
        let step = (n / 64).max(1);
        let mut probes = 0usize;
        let mut mismatches = 0usize;
        for id in (0..n as u32).step_by(step) {
            let Some(Some(fast)) = bytes.get(id as usize) else {
                continue;
            };
            let Ok(fast_text) = std::str::from_utf8(fast) else {
                continue;
            };
            if fast_text.is_empty() {
                continue;
            }
            probes += 1;
            if reference(id).as_deref() != Some(fast_text) {
                mismatches += 1;
            }
        }

        if mismatches > 0 || probes == 0 {
            eprintln!(
                "[constraint] token byte table fell back to per-id decode \
                 ({mismatches}/{probes} probe mismatches)"
            );
            for id in 0..n as u32 {
                if special.contains(&id) {
                    bytes[id as usize] = None;
                    continue;
                }
                bytes[id as usize] = reference(id)
                    .filter(|p| !p.contains('\u{FFFD}'))
                    .map(|p| p.into_bytes().into_boxed_slice());
            }
        }

        Self::from_bytes_vec(bytes)
    }

    fn from_bytes_vec(bytes: Vec<Option<Box<[u8]>>>) -> Self {
        let meta = bytes
            .iter()
            .map(|b| b.as_deref().map(classify).unwrap_or_default())
            .collect();
        TokenByteTable { bytes, meta }
    }

    /// Bytes for `id`; `None` = never emit.
    pub fn bytes(&self, id: u32) -> Option<&[u8]> {
        self.bytes.get(id as usize).and_then(|b| b.as_deref())
    }

    /// Number of ids in the table.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// True when the table is empty.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    #[cfg(test)]
    fn from_raw(bytes: Vec<Option<Vec<u8>>>) -> Self {
        Self::from_bytes_vec(
            bytes
                .into_iter()
                .map(|b| b.map(Vec::into_boxed_slice))
                .collect(),
        )
    }
}

/// `<0xNN>` SPM byte-fallback token → its byte.
fn spm_byte_token(s: &str) -> Option<u8> {
    let hex = s.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() == 2 {
        u8::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

fn spm_bytes(s: &str) -> Vec<u8> {
    if let Some(b) = spm_byte_token(s) {
        return vec![b];
    }
    let mut out = Vec::with_capacity(s.len());
    for c in s.chars() {
        if c == '\u{2581}' {
            out.push(b' ');
        } else {
            let mut buf = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
        }
    }
    out
}

/// Reverse of the GPT-2 byte→unicode mapping: 256 chars back to bytes.
fn gpt2_unmap() -> std::collections::HashMap<char, u8> {
    let mut printable: Vec<u8> = (b'!'..=b'~').collect();
    printable.extend(0xA1u8..=0xAC);
    printable.extend(0xAEu8..=0xFF);
    let mut map = std::collections::HashMap::with_capacity(256);
    for &b in &printable {
        map.insert(char::from_u32(b as u32).unwrap(), b);
    }
    let mut n = 0u32;
    for b in 0..=255u8 {
        if !printable.contains(&b) {
            map.insert(char::from_u32(256 + n).unwrap(), b);
            n += 1;
        }
    }
    map
}

fn byte_level_bytes(s: &str, unmap: &std::collections::HashMap<char, u8>) -> Vec<u8> {
    // Added tokens keep plain content even in byte-level vocabs; a char
    // outside the 256-entry map domain means "not byte-mapped" — fall back
    // to the raw UTF-8 for the whole string.
    let mut out = Vec::with_capacity(s.len());
    for c in s.chars() {
        match unmap.get(&c) {
            Some(&b) => out.push(b),
            None => return s.as_bytes().to_vec(),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Byte-level JSON automaton
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Frame {
    Object { node: NodeId, seen: u64 },
    Array { node: NodeId },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum StrKind {
    /// Unconstrained string content.
    Free,
    /// Object key: candidate bitmask over the top frame's unfilled
    /// properties + wildcard (additionalProperties) flag; `len` counts
    /// decoded bytes matched so far.
    Key { cand: u64, wild: bool, len: u16 },
    /// String enum literal: candidate bitmask over the node's literals.
    Enum { node: NodeId, cand: u64, len: u16 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Esc {
    None,
    /// Just consumed `\`.
    Bslash,
    /// Inside `\uXXXX` hex digits.
    Uni { left: u8, acc: u16 },
    /// High surrogate consumed; the low half's `\` is required next.
    PairBslash { high: u16 },
    /// …then its `u`.
    PairU { high: u16 },
    /// …then the low surrogate's hex digits.
    Uni2 { left: u8, acc: u16, high: u16 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum NumStage {
    Sign,
    Zero,
    Int,
    Dot,
    Frac,
    Exp,
    ExpSign,
    ExpDig,
}

impl NumStage {
    /// True when the digits consumed so far form a complete JSON number.
    fn complete(self) -> bool {
        matches!(
            self,
            NumStage::Zero | NumStage::Int | NumStage::Frac | NumStage::ExpDig
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Lit {
    True,
    False,
    Null,
}

impl Lit {
    fn bytes(self) -> &'static [u8] {
        match self {
            Lit::True => b"true",
            Lit::False => b"false",
            Lit::Null => b"null",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Mode {
    /// Expecting the start of a value of the given node (whitespace ok).
    Value { node: NodeId },
    /// Just after `[`: first element or `]`.
    ElemFirst,
    /// In an object, expecting `"` of a key (or `}` when not after a comma).
    KeyStart { require: bool },
    /// Expecting `:` after a key; `value` is the matched property's node.
    Colon { value: NodeId },
    /// Inside string content.
    Str { kind: StrKind, esc: Esc, utf8: u8 },
    /// Inside a number.
    Num { integer: bool, stage: NumStage },
    /// Inside `true`/`false`/`null`.
    Lit { lit: Lit, pos: u8 },
    /// In a container, after a complete value: `,` or the closer.
    AfterValue,
    /// Top-level value complete: trailing whitespace only (EOS allowed).
    Done,
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

fn hex_val(b: u8) -> Option<u16> {
    match b {
        b'0'..=b'9' => Some((b - b'0') as u16),
        b'a'..=b'f' => Some((b - b'a' + 10) as u16),
        b'A'..=b'F' => Some((b - b'A' + 10) as u16),
        _ => None,
    }
}

/// Maximum consecutive gap-whitespace bytes. Whitespace between JSON
/// tokens is legal but must be bounded: a model steered away from prose
/// otherwise satisfies the mask with whitespace forever (the observed
/// greedy failure mode — 200 tokens of spaces). 12 bytes covers
/// newline + deep pretty-print indentation; string-content whitespace is
/// unaffected.
const WS_MAX: u8 = 12;

#[derive(Clone, Debug)]
struct Machine {
    mode: Mode,
    stack: Vec<Frame>,
    /// Consecutive gap-whitespace bytes consumed (reset by any other byte).
    ws_run: u8,
}

impl Machine {
    fn new(root: NodeId) -> Self {
        Machine {
            mode: Mode::Value { node: root },
            stack: Vec::with_capacity(16),
            ws_run: 0,
        }
    }

    /// Copies `src` into `self`, reusing the stack allocation (the mask
    /// loop simulates ~vocab-size tokens per step; a fresh Vec per clone
    /// would dominate the cost).
    fn copy_from(&mut self, src: &Machine) {
        self.mode = src.mode;
        self.ws_run = src.ws_run;
        self.stack.clone_from(&src.stack);
    }

    /// Consumes one byte of gap whitespace, bounded by [`WS_MAX`].
    fn take_ws(&mut self) -> bool {
        if self.ws_run >= WS_MAX {
            return false;
        }
        self.ws_run += 1;
        true
    }

    /// True when the stream so far is one complete top-level value.
    fn accepting(&self) -> bool {
        match self.mode {
            Mode::Done => true,
            // A top-level number has no delimiter to end it: complete
            // digits at depth 0 are accepting as-is.
            Mode::Num { stage, .. } => self.stack.is_empty() && stage.complete(),
            _ => false,
        }
    }

    /// The value just finished: back to the container, or done at depth 0.
    fn value_done(&mut self) {
        self.mode = if self.stack.is_empty() {
            Mode::Done
        } else {
            Mode::AfterValue
        };
    }

    /// Advances one byte; `false` = the byte is not legal here.
    fn step(&mut self, s: &CompiledSchema, b: u8) -> bool {
        let ws_before = self.ws_run;
        let ok = self.step_inner(s, b);
        if ok && self.ws_run == ws_before {
            // Any non-gap byte (including string-content whitespace)
            // resets the gap counter; only `take_ws` increments it.
            self.ws_run = 0;
        }
        ok
    }

    fn step_inner(&mut self, s: &CompiledSchema, byte: u8) -> bool {
        let b = byte;
        // The loop re-feeds `b` after mode transitions that consume no
        // input (number termination, array first-element dispatch).
        loop {
            match self.mode {
                Mode::Value { node } => return self.value_start(s, node, b),
                Mode::ElemFirst => {
                    if is_ws(b) {
                        return self.take_ws();
                    }
                    let Some(Frame::Array { node }) = self.stack.last().copied() else {
                        return false;
                    };
                    if b == b']' {
                        self.stack.pop();
                        self.value_done();
                        return true;
                    }
                    let SchemaNode::Array { items } = s.node(node) else {
                        return false;
                    };
                    self.mode = Mode::Value { node: *items };
                    continue;
                }
                Mode::KeyStart { require } => {
                    if is_ws(b) {
                        return self.take_ws();
                    }
                    let Some(Frame::Object { node, seen }) = self.stack.last().copied() else {
                        return false;
                    };
                    let SchemaNode::Object {
                        props,
                        required,
                        additional,
                    } = s.node(node)
                    else {
                        return false;
                    };
                    return match b {
                        b'"' => {
                            let cand = unseen_mask(props.len(), seen);
                            let wild = additional.is_some();
                            if cand == 0 && !wild {
                                return false;
                            }
                            self.mode = Mode::Str {
                                kind: StrKind::Key { cand, wild, len: 0 },
                                esc: Esc::None,
                                utf8: 0,
                            };
                            true
                        }
                        b'}' if !require => {
                            if required & !seen != 0 {
                                return false;
                            }
                            self.stack.pop();
                            self.value_done();
                            true
                        }
                        _ => false,
                    };
                }
                Mode::Colon { value } => {
                    if is_ws(b) {
                        return self.take_ws();
                    }
                    if b == b':' {
                        self.mode = Mode::Value { node: value };
                        return true;
                    }
                    return false;
                }
                Mode::Str { kind, esc, utf8 } => return self.str_step(s, kind, esc, utf8, b),
                Mode::Num { integer, stage } => {
                    match num_step(integer, stage, b) {
                        NumOutcome::Next(next) => {
                            self.mode = Mode::Num {
                                integer,
                                stage: next,
                            };
                            return true;
                        }
                        NumOutcome::Reject => return false,
                        NumOutcome::Terminated => {
                            // The byte belongs to the surrounding context.
                            self.value_done();
                            continue;
                        }
                    }
                }
                Mode::Lit { lit, pos } => {
                    let bytes = lit.bytes();
                    if bytes.get(pos as usize) != Some(&b) {
                        return false;
                    }
                    if pos as usize + 1 == bytes.len() {
                        self.value_done();
                    } else {
                        self.mode = Mode::Lit { lit, pos: pos + 1 };
                    }
                    return true;
                }
                Mode::AfterValue => {
                    if is_ws(b) {
                        return self.take_ws();
                    }
                    let Some(frame) = self.stack.last().copied() else {
                        return false;
                    };
                    match (frame, b) {
                        (Frame::Object { node, seen }, b',') => {
                            let SchemaNode::Object {
                                props, additional, ..
                            } = s.node(node)
                            else {
                                return false;
                            };
                            // A comma commits to another key: legal only if
                            // one can still be written.
                            if additional.is_none() && unseen_mask(props.len(), seen) == 0 {
                                return false;
                            }
                            self.mode = Mode::KeyStart { require: true };
                            return true;
                        }
                        (Frame::Object { node, seen }, b'}') => {
                            let SchemaNode::Object { required, .. } = s.node(node) else {
                                return false;
                            };
                            if required & !seen != 0 {
                                return false;
                            }
                            self.stack.pop();
                            self.value_done();
                            return true;
                        }
                        (Frame::Array { node }, b',') => {
                            let SchemaNode::Array { items } = s.node(node) else {
                                return false;
                            };
                            self.mode = Mode::Value { node: *items };
                            return true;
                        }
                        (Frame::Array { .. }, b']') => {
                            self.stack.pop();
                            self.value_done();
                            return true;
                        }
                        _ => return false,
                    }
                }
                Mode::Done => return is_ws(b) && self.take_ws(),
            }
        }
    }

    /// Dispatches the first byte of a value for `node`.
    fn value_start(&mut self, s: &CompiledSchema, node: NodeId, b: u8) -> bool {
        if is_ws(b) {
            return self.take_ws();
        }
        let (allow_obj, allow_arr, allow_str, allow_num, allow_bool, allow_null, integer) =
            match s.node(node) {
                SchemaNode::Any => (true, true, true, true, true, true, false),
                SchemaNode::Object { .. } => (true, false, false, false, false, false, false),
                SchemaNode::Array { .. } => (false, true, false, false, false, false, false),
                SchemaNode::Str | SchemaNode::EnumStr { .. } => {
                    (false, false, true, false, false, false, false)
                }
                SchemaNode::Num { integer } => (false, false, false, true, false, false, *integer),
                SchemaNode::Bool => (false, false, false, false, true, false, false),
                SchemaNode::Null => (false, false, false, false, false, true, false),
            };
        match b {
            b'{' if allow_obj => {
                let obj_node = if matches!(s.node(node), SchemaNode::Any) {
                    ANY_OBJ
                } else {
                    node
                };
                self.stack.push(Frame::Object {
                    node: obj_node,
                    seen: 0,
                });
                self.mode = Mode::KeyStart { require: false };
                true
            }
            b'[' if allow_arr => {
                let arr_node = if matches!(s.node(node), SchemaNode::Any) {
                    ANY_ARR
                } else {
                    node
                };
                self.stack.push(Frame::Array { node: arr_node });
                self.mode = Mode::ElemFirst;
                true
            }
            b'"' if allow_str => {
                let kind = match s.node(node) {
                    SchemaNode::EnumStr { literals } => StrKind::Enum {
                        node,
                        cand: unseen_mask(literals.len(), 0),
                        len: 0,
                    },
                    _ => StrKind::Free,
                };
                self.mode = Mode::Str {
                    kind,
                    esc: Esc::None,
                    utf8: 0,
                };
                true
            }
            b'-' if allow_num => {
                self.mode = Mode::Num {
                    integer,
                    stage: NumStage::Sign,
                };
                true
            }
            b'0' if allow_num => {
                self.mode = Mode::Num {
                    integer,
                    stage: NumStage::Zero,
                };
                true
            }
            b'1'..=b'9' if allow_num => {
                self.mode = Mode::Num {
                    integer,
                    stage: NumStage::Int,
                };
                true
            }
            b't' if allow_bool => {
                self.mode = Mode::Lit {
                    lit: Lit::True,
                    pos: 1,
                };
                true
            }
            b'f' if allow_bool => {
                self.mode = Mode::Lit {
                    lit: Lit::False,
                    pos: 1,
                };
                true
            }
            b'n' if allow_null => {
                self.mode = Mode::Lit {
                    lit: Lit::Null,
                    pos: 1,
                };
                true
            }
            _ => false,
        }
    }

    /// One byte of string content (including escapes and the closing quote).
    fn str_step(&mut self, s: &CompiledSchema, kind: StrKind, esc: Esc, utf8: u8, b: u8) -> bool {
        let mut kind = kind;
        match esc {
            Esc::None if utf8 > 0 => {
                if !(0x80..=0xBF).contains(&b) {
                    return false;
                }
                if !feed_match(&mut kind, s, self.stack.last(), b) {
                    return false;
                }
                self.mode = Mode::Str {
                    kind,
                    esc: Esc::None,
                    utf8: utf8 - 1,
                };
                true
            }
            Esc::None => match b {
                b'"' => self.close_string(s, kind),
                b'\\' => {
                    self.mode = Mode::Str {
                        kind,
                        esc: Esc::Bslash,
                        utf8: 0,
                    };
                    true
                }
                0x00..=0x1F => false,
                0x80..=0xFF => {
                    let cont = match b {
                        0xC2..=0xDF => 1,
                        0xE0..=0xEF => 2,
                        0xF0..=0xF4 => 3,
                        // Bare continuation bytes / illegal leads would make
                        // the output invalid UTF-8.
                        _ => return false,
                    };
                    if !feed_match(&mut kind, s, self.stack.last(), b) {
                        return false;
                    }
                    self.mode = Mode::Str {
                        kind,
                        esc: Esc::None,
                        utf8: cont,
                    };
                    true
                }
                _ => {
                    if !feed_match(&mut kind, s, self.stack.last(), b) {
                        return false;
                    }
                    self.mode = Mode::Str {
                        kind,
                        esc: Esc::None,
                        utf8: 0,
                    };
                    true
                }
            },
            Esc::Bslash => {
                let decoded = match b {
                    b'"' => b'"',
                    b'\\' => b'\\',
                    b'/' => b'/',
                    b'b' => 0x08,
                    b'f' => 0x0C,
                    b'n' => b'\n',
                    b'r' => b'\r',
                    b't' => b'\t',
                    b'u' => {
                        self.mode = Mode::Str {
                            kind,
                            esc: Esc::Uni { left: 4, acc: 0 },
                            utf8: 0,
                        };
                        return true;
                    }
                    _ => return false,
                };
                if !feed_match(&mut kind, s, self.stack.last(), decoded) {
                    return false;
                }
                self.mode = Mode::Str {
                    kind,
                    esc: Esc::None,
                    utf8: 0,
                };
                true
            }
            Esc::Uni { left, acc } => {
                let Some(v) = hex_val(b) else { return false };
                let acc = acc << 4 | v;
                if left > 1 {
                    self.mode = Mode::Str {
                        kind,
                        esc: Esc::Uni {
                            left: left - 1,
                            acc,
                        },
                        utf8: 0,
                    };
                    return true;
                }
                match acc {
                    // High surrogate: strict JSON requires the low half
                    // (serde_json rejects lone surrogates).
                    0xD800..=0xDBFF => {
                        self.mode = Mode::Str {
                            kind,
                            esc: Esc::PairBslash { high: acc },
                            utf8: 0,
                        };
                        true
                    }
                    0xDC00..=0xDFFF => false,
                    _ => self.finish_unicode(s, kind, acc as u32),
                }
            }
            Esc::PairBslash { high } => {
                if b != b'\\' {
                    return false;
                }
                self.mode = Mode::Str {
                    kind,
                    esc: Esc::PairU { high },
                    utf8: 0,
                };
                true
            }
            Esc::PairU { high } => {
                if b != b'u' {
                    return false;
                }
                self.mode = Mode::Str {
                    kind,
                    esc: Esc::Uni2 {
                        left: 4,
                        acc: 0,
                        high,
                    },
                    utf8: 0,
                };
                true
            }
            Esc::Uni2 { left, acc, high } => {
                let Some(v) = hex_val(b) else { return false };
                let acc = acc << 4 | v;
                if left > 1 {
                    self.mode = Mode::Str {
                        kind,
                        esc: Esc::Uni2 {
                            left: left - 1,
                            acc,
                            high,
                        },
                        utf8: 0,
                    };
                    return true;
                }
                if !(0xDC00..=0xDFFF).contains(&acc) {
                    return false;
                }
                let cp =
                    0x10000 + (((high as u32 - 0xD800) << 10) | (acc as u32 - 0xDC00));
                self.finish_unicode(s, kind, cp)
            }
        }
    }

    /// Feeds a decoded `\uXXXX` code point into the matcher and returns to
    /// plain content state.
    fn finish_unicode(&mut self, s: &CompiledSchema, mut kind: StrKind, cp: u32) -> bool {
        let Some(c) = char::from_u32(cp) else {
            return false;
        };
        let mut buf = [0u8; 4];
        for &b in c.encode_utf8(&mut buf).as_bytes() {
            if !feed_match(&mut kind, s, self.stack.last(), b) {
                return false;
            }
        }
        self.mode = Mode::Str {
            kind,
            esc: Esc::None,
            utf8: 0,
        };
        true
    }

    /// Closing quote: resolve key/enum matches.
    fn close_string(&mut self, s: &CompiledSchema, kind: StrKind) -> bool {
        match kind {
            StrKind::Free => {
                self.value_done();
                true
            }
            StrKind::Enum { node, cand, len } => {
                let SchemaNode::EnumStr { literals } = s.node(node) else {
                    return false;
                };
                let exact = iter_bits(cand)
                    .any(|i| literals[i].as_bytes().len() == len as usize);
                if !exact {
                    return false;
                }
                self.value_done();
                true
            }
            StrKind::Key { cand, wild, len } => {
                let Some(Frame::Object { node, seen }) = self.stack.last().copied() else {
                    return false;
                };
                let SchemaNode::Object {
                    props, additional, ..
                } = s.node(node)
                else {
                    return false;
                };
                let exact =
                    iter_bits(cand).find(|&i| props[i].0.as_bytes().len() == len as usize);
                match exact {
                    Some(i) => {
                        if let Some(Frame::Object { seen: sn, .. }) = self.stack.last_mut() {
                            *sn = seen | 1 << i;
                        }
                        self.mode = Mode::Colon { value: props[i].1 };
                        true
                    }
                    None if wild => {
                        self.mode = Mode::Colon {
                            value: additional.unwrap_or(ANY),
                        };
                        true
                    }
                    None => false,
                }
            }
        }
    }
}

/// Bitmask of `count` candidates minus the already-seen set.
fn unseen_mask(count: usize, seen: u64) -> u64 {
    let all = if count >= 64 {
        u64::MAX
    } else {
        (1u64 << count) - 1
    };
    all & !seen
}

fn iter_bits(mask: u64) -> impl Iterator<Item = usize> {
    (0..64).filter(move |i| mask & (1 << i) != 0)
}

/// Filters key/enum candidates by the next decoded content byte.
fn feed_match(kind: &mut StrKind, s: &CompiledSchema, top: Option<&Frame>, b: u8) -> bool {
    match kind {
        StrKind::Free => true,
        StrKind::Key { cand, wild, len } => {
            let names: &[(String, NodeId)] = match top {
                Some(Frame::Object { node, .. }) => match s.node(*node) {
                    SchemaNode::Object { props, .. } => props,
                    _ => return false,
                },
                _ => return false,
            };
            let mut next = 0u64;
            for i in iter_bits(*cand) {
                let name = names[i].0.as_bytes();
                if name.get(*len as usize) == Some(&b) {
                    next |= 1 << i;
                }
            }
            *cand = next;
            *len = len.saturating_add(1);
            next != 0 || *wild
        }
        StrKind::Enum { node, cand, len } => {
            let SchemaNode::EnumStr { literals } = s.node(*node) else {
                return false;
            };
            let mut next = 0u64;
            for i in iter_bits(*cand) {
                if literals[i].as_bytes().get(*len as usize) == Some(&b) {
                    next |= 1 << i;
                }
            }
            *cand = next;
            *len = len.saturating_add(1);
            next != 0
        }
    }
}

enum NumOutcome {
    Next(NumStage),
    Reject,
    /// The byte is not part of the number and the digits so far are
    /// complete: hand the byte back to the surrounding context.
    Terminated,
}

fn num_step(integer: bool, stage: NumStage, b: u8) -> NumOutcome {
    use NumOutcome::*;
    use NumStage::*;
    match (stage, b) {
        (Sign, b'0') => Next(Zero),
        (Sign, b'1'..=b'9') => Next(Int),
        (Sign, _) => Reject,
        (Zero, b'.') | (Int, b'.') if !integer => Next(Dot),
        (Zero, b'e' | b'E') | (Int, b'e' | b'E') if !integer => Next(Exp),
        (Int, b'0'..=b'9') => Next(Int),
        (Zero, b'0'..=b'9') => Reject,
        (Dot, b'0'..=b'9') => Next(Frac),
        (Dot, _) => Reject,
        (Frac, b'0'..=b'9') => Next(Frac),
        (Frac, b'e' | b'E') => Next(Exp),
        (Exp, b'+' | b'-') => Next(ExpSign),
        (Exp, b'0'..=b'9') | (ExpSign, b'0'..=b'9') | (ExpDig, b'0'..=b'9') => Next(ExpDig),
        (Exp, _) | (ExpSign, _) => Reject,
        (st, _) if st.complete() => Terminated,
        _ => Reject,
    }
}

// ---------------------------------------------------------------------------
// ConstraintState — what the engine drives
// ---------------------------------------------------------------------------

/// Live constraint for one generation: masks the logits row before each
/// sample and advances on the sampled token.
pub struct ConstraintState<'t> {
    schema: CompiledSchema,
    table: &'t TokenByteTable,
    eos: Vec<u32>,
    machine: Machine,
    scratch: Machine,
    /// Mask results keyed by full machine state. States recur constantly
    /// (every content byte of the same string, structurally identical
    /// spots), so each unique state pays the vocabulary walk once.
    cache: std::collections::HashMap<MaskKey, CachedMask>,
}

#[derive(PartialEq, Eq, Hash)]
struct MaskKey {
    mode: Mode,
    ws_run: u8,
    stack: Box<[Frame]>,
}

struct CachedMask {
    /// Allowed-token bitset, vocab bits packed in u64 words.
    bits: Box<[u64]>,
    allowed: usize,
}

/// Cache reset threshold — bounds memory at ~vocab/8 bytes × entries.
const MASK_CACHE_MAX: usize = 512;

impl<'t> ConstraintState<'t> {
    /// Builds the live state from a compiled schema, the model's token
    /// table, and the request's eos ids (allowed only in accept states).
    pub fn new(schema: CompiledSchema, table: &'t TokenByteTable, eos: Vec<u32>) -> Self {
        let machine = Machine::new(schema.root);
        let scratch = machine.clone();
        ConstraintState {
            schema,
            table,
            eos,
            machine,
            scratch,
            cache: std::collections::HashMap::new(),
        }
    }

    /// Masks tokens that cannot legally continue the output (−∞), allowing
    /// eos ids only in accept states. Returns how many ids stay allowed —
    /// 0 means the vocabulary cannot spell any legal continuation and the
    /// generation must fail rather than emit garbage.
    pub fn mask(&mut self, logits: &mut [f32]) -> usize {
        let key = MaskKey {
            mode: self.machine.mode,
            ws_run: self.machine.ws_run,
            stack: self.machine.stack.as_slice().into(),
        };
        if let Some(hit) = self.cache.get(&key) {
            let mut allowed = 0usize;
            for (id, l) in logits.iter_mut().enumerate() {
                if hit.bits[id >> 6] & (1u64 << (id & 63)) != 0 {
                    allowed += 1;
                } else {
                    *l = f32::NEG_INFINITY;
                }
            }
            debug_assert_eq!(allowed, hit.allowed);
            return hit.allowed;
        }

        let ConstraintState {
            schema,
            table,
            eos,
            machine,
            scratch,
            ..
        } = self;
        let accept = machine.accepting();

        // 256-entry first-byte gate: one simulation per byte value, then
        // most tokens are rejected without touching the stack.
        let mut first = [false; 256];
        for (b, ok) in first.iter_mut().enumerate() {
            scratch.copy_from(machine);
            *ok = scratch.step(schema, b as u8);
        }

        // String-content fast path: inside plain string content nearly
        // every token is legal, and the per-token classification answers
        // without simulating (the dominant state by step count). Wildcard
        // keys (json_object mode) behave exactly like free content for
        // plain tokens — no quote/escape means no close and no candidate
        // resolution.
        let content_pending = match machine.mode {
            Mode::Str {
                kind: StrKind::Free | StrKind::Key { wild: true, .. },
                esc: Esc::None,
                utf8,
            } => Some(utf8),
            _ => None,
        };

        let mut sim = |bytes: &[u8]| -> bool {
            first[bytes[0] as usize]
                && (bytes.len() == 1 || {
                    scratch.copy_from(machine);
                    bytes.iter().all(|&b| scratch.step(schema, b))
                })
        };

        let mut bits = vec![0u64; logits.len().div_ceil(64)].into_boxed_slice();
        let mut allowed = 0usize;
        for (id, l) in logits.iter_mut().enumerate() {
            let ok = if eos.contains(&(id as u32)) {
                accept
            } else {
                match table.bytes(id as u32) {
                    None | Some([]) => false,
                    Some(bytes) => match content_pending {
                        Some(pending) if table.meta[id].plain => {
                            table.meta[id].legal_in_free_string(pending)
                        }
                        _ => sim(bytes),
                    },
                }
            };
            if ok {
                allowed += 1;
                bits[id >> 6] |= 1u64 << (id & 63);
            } else {
                *l = f32::NEG_INFINITY;
            }
        }
        if self.cache.len() >= MASK_CACHE_MAX {
            self.cache.clear();
        }
        self.cache.insert(key, CachedMask { bits, allowed });
        allowed
    }

    /// Advances past a sampled token (no-op for eos ids).
    pub fn advance(&mut self, id: u32) {
        if self.eos.contains(&id) {
            return;
        }
        let ConstraintState {
            schema,
            table,
            machine,
            ..
        } = self;
        if let Some(bytes) = table.bytes(id) {
            for &b in bytes {
                let ok = machine.step(schema, b);
                debug_assert!(ok, "sampled token {id} was not legal for the constraint");
                if !ok {
                    break;
                }
            }
        }
    }

    /// True when the output so far is a complete value (eos is legal).
    pub fn accepting(&self) -> bool {
        self.machine.accepting()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn accepts_spec(spec: &ConstraintSpec, text: &str) -> bool {
        let s = CompiledSchema::compile(spec).expect("schema compiles");
        let mut m = Machine::new(s.root);
        for &b in text.as_bytes() {
            if !m.step(&s, b) {
                return false;
            }
        }
        m.accepting()
    }

    fn accepts_obj(text: &str) -> bool {
        accepts_spec(&ConstraintSpec::JsonObject, text)
    }

    fn accepts_schema(schema: Value, text: &str) -> bool {
        accepts_spec(&ConstraintSpec::JsonSchema(schema), text)
    }

    #[test]
    fn object_mode_accepts_valid_objects() {
        assert!(accepts_obj("{}"));
        assert!(accepts_obj(r#"{"a":1}"#));
        assert!(accepts_obj(
            "{ \"a\" : [1, 2.5e-3, true, null, \"x\"] ,\n\t\"b\": {\"c\": \"\"} }"
        ));
        assert!(accepts_obj("{\"a\":1}  \n"));
        assert!(accepts_obj(r#"{"k":"és 世"}"#));
    }

    #[test]
    fn object_mode_rejects_invalid_streams() {
        assert!(!accepts_obj("[1]")); // must be an object
        assert!(!accepts_obj(r#""hi""#));
        assert!(!accepts_obj(r#"{"a":1,}"#)); // trailing comma
        assert!(!accepts_obj(r#"{"a":01}"#)); // leading zero
        assert!(!accepts_obj("{'a':1}"));
        assert!(!accepts_obj(r#"{"a":1}x"#));
        assert!(!accepts_obj(r#"{"a":+1}"#));
        assert!(!accepts_obj(r#"{"a":.5}"#));
        assert!(!accepts_obj(r#"{"a":1.}"#));
        assert!(!accepts_obj(r#"{"a":tru}"#));
        assert!(!accepts_obj("{\"a\":\"\u{0009}")); // raw control byte in string
        assert!(!accepts_obj(r#"{"a""#)); // incomplete: not accepting
        assert!(!accepts_obj(r#"{"a":"b"#)); // unterminated string
    }

    #[test]
    fn string_escapes() {
        assert!(accepts_obj(r#"{"a":"q\"\\\/\b\f\n\r\tA"}"#));
        assert!(accepts_obj(r#"{"a":"😀"}"#)); // surrogate pair
        assert!(!accepts_obj(r#"{"a":"\q"}"#));
        assert!(!accepts_obj(r#"{"a":"\ud800x"}"#)); // lone high surrogate
        assert!(!accepts_obj(r#"{"a":"\udc00"}"#)); // lone low surrogate
        assert!(!accepts_obj(r#"{"a":"\u00zz"}"#));
    }

    #[test]
    fn utf8_continuation_tracking() {
        // é = C3 A9: fine. A lone lead byte before the closing quote is not.
        assert!(accepts_obj("{\"a\":\"\u{e9}\"}"));
        let lead_then_quote = [b'{', b'"', b'a', b'"', b':', b'"', 0xC3, b'"'];
        let s = CompiledSchema::compile(&ConstraintSpec::JsonObject).unwrap();
        let mut m = Machine::new(s.root);
        let mut ok = true;
        for &b in &lead_then_quote {
            if !m.step(&s, b) {
                ok = false;
                break;
            }
        }
        assert!(!ok, "quote inside a UTF-8 sequence must be rejected");
        // Bare continuation byte outside a sequence.
        let mut m = Machine::new(s.root);
        for &b in &[b'{', b'"', b'a', b'"', b':', b'"'] {
            assert!(m.step(&s, b));
        }
        assert!(!m.step(&s, 0x80));
    }

    fn person_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"],
            "additionalProperties": false
        })
    }

    #[test]
    fn schema_objects() {
        assert!(accepts_schema(person_schema(), r#"{"name":"x","age":3}"#));
        assert!(accepts_schema(person_schema(), r#"{"age":-2,"name":""}"#));
        assert!(!accepts_schema(person_schema(), r#"{"name":"x"}"#)); // missing required
        assert!(!accepts_schema(person_schema(), r#"{"name":"x","age":3.5}"#));
        assert!(!accepts_schema(person_schema(), r#"{"name":"x","age":3,"z":1}"#));
        assert!(!accepts_schema(person_schema(), r#"{"nam":"x","age":3}"#));
        assert!(!accepts_schema(person_schema(), r#"{"name":"x","name":"y","age":3}"#));
    }

    #[test]
    fn schema_enums() {
        let s = json!({"type":"object","properties":{"c":{"enum":["red","green"]}},
                       "required":["c"],"additionalProperties":false});
        assert!(accepts_schema(s.clone(), r#"{"c":"red"}"#));
        assert!(accepts_schema(s.clone(), r#"{"c":"green"}"#));
        assert!(!accepts_schema(s.clone(), r#"{"c":"blue"}"#));
        assert!(!accepts_schema(s.clone(), r#"{"c":"re"}"#));
        assert!(!accepts_schema(s, r#"{"c":"redd"}"#));
    }

    #[test]
    fn schema_arrays_and_scalars() {
        let ints = json!({"type":"array","items":{"type":"integer"}});
        assert!(accepts_schema(ints.clone(), "[]"));
        assert!(accepts_schema(ints.clone(), "[1, -2, 0]"));
        assert!(!accepts_schema(ints.clone(), r#"[1,"a"]"#));
        assert!(!accepts_schema(ints, "[1,]"));

        assert!(accepts_schema(json!({"type":"string"}), r#""hi""#));
        assert!(accepts_schema(json!({"type":"number"}), "3.5"));
        assert!(accepts_schema(json!({"type":"number"}), "-0"));
        assert!(!accepts_schema(json!({"type":"number"}), "3.")); // incomplete
        assert!(accepts_schema(json!({"type":"integer"}), "12"));
        assert!(!accepts_schema(json!({"type":"integer"}), "1.0"));
        assert!(!accepts_schema(json!({"type":"integer"}), "1e3"));
        assert!(accepts_schema(json!({"type":"boolean"}), "false"));
        assert!(accepts_schema(json!({"type":"null"}), "null"));
    }

    #[test]
    fn compile_rejects_unsupported_keywords() {
        for schema in [
            json!({"anyOf": [{"type":"string"}]}),
            json!({"$ref": "#/x"}),
            json!({"type":"string", "pattern": "a+"}),
            json!({"type":"string", "minLength": 1}),
            json!({"type": ["string", "null"]}),
            json!({"enum": [1, 2]}),
            json!({"type":"object", "properties": {"a": {"type":"string"}},
                   "required": ["b"]}),
        ] {
            assert!(
                ConstraintSpec::JsonSchema(schema.clone()).compile().is_err(),
                "expected rejection: {schema}"
            );
        }
    }

    #[test]
    fn response_format_parsing() {
        assert_eq!(
            ConstraintSpec::from_response_format(&json!({"type":"text"})).unwrap(),
            None
        );
        assert_eq!(
            ConstraintSpec::from_response_format(&json!({"type":"json_object"})).unwrap(),
            Some(ConstraintSpec::JsonObject)
        );
        let wire = json!({"type":"json_schema",
                          "json_schema":{"name":"p","schema":{"type":"object"}}});
        assert_eq!(
            ConstraintSpec::from_response_format(&wire).unwrap(),
            Some(ConstraintSpec::JsonSchema(json!({"type":"object"})))
        );
        assert!(ConstraintSpec::from_response_format(&json!({"type":"yaml"})).is_err());
        assert!(ConstraintSpec::from_response_format(&json!({})).is_err());
    }

    /// Tiny synthetic vocabulary for mask/advance tests:
    /// 0 `{`, 1 `}`, 2 `"a"`, 3 `:`, 4 `1`, 5 eos (never-emit), 6 `,`,
    /// 7 `"z` (unterminated), 8 ` ` (space).
    fn toy_table() -> TokenByteTable {
        TokenByteTable::from_raw(vec![
            Some(b"{".to_vec()),
            Some(b"}".to_vec()),
            Some(b"\"a\"".to_vec()),
            Some(b":".to_vec()),
            Some(b"1".to_vec()),
            None,
            Some(b",".to_vec()),
            Some(b"\"z".to_vec()),
            Some(b" ".to_vec()),
        ])
    }

    fn masked_ids(state: &mut ConstraintState<'_>) -> Vec<u32> {
        let mut logits = vec![0.0f32; state.table.len()];
        state.mask(&mut logits);
        logits
            .iter()
            .enumerate()
            .filter(|(_, l)| l.is_finite())
            .map(|(i, _)| i as u32)
            .collect()
    }

    #[test]
    fn mask_walks_json_object_mode() {
        let table = toy_table();
        let schema = ConstraintSpec::JsonObject.compile().unwrap();
        let mut state = ConstraintState::new(schema, &table, vec![5]);

        // Start: `{` opens the object (leading whitespace is legal JSON).
        assert_eq!(masked_ids(&mut state), vec![0, 8]);
        state.advance(0);
        // After `{`: close, a key, key-prefix `"z`, or whitespace — never
        // eos, never `:`.
        assert_eq!(masked_ids(&mut state), vec![1, 2, 7, 8]);
        state.advance(2); // "a"
        assert_eq!(masked_ids(&mut state), vec![3, 8]);
        state.advance(3); // :
        state.advance(4); // 1
        // Number is terminable by `}`, `,`, or whitespace; digits continue.
        assert_eq!(masked_ids(&mut state), vec![1, 4, 6, 8]);
        state.advance(1); // }
        assert!(state.accepting());
        // Accept state: eos and trailing whitespace only.
        assert_eq!(masked_ids(&mut state), vec![5, 8]);
    }

    #[test]
    fn mask_dead_end_returns_zero() {
        // Schema requires a key "q" that no token in the vocabulary spells.
        let table = toy_table();
        let schema = ConstraintSpec::JsonSchema(json!({
            "type":"object",
            "properties":{"q":{"type":"integer"}},
            "required":["q"],
            "additionalProperties": false
        }))
        .compile()
        .unwrap();
        let mut state = ConstraintState::new(schema, &table, vec![5]);
        state.advance(0); // {
        let mut logits = vec![0.0f32; state.table.len()];
        // `}` is illegal (required key missing), `"a"` doesn't match "q",
        // `"z` matches nothing: only whitespace remains…
        assert_eq!(masked_ids(&mut state), vec![8]);
        // …and a vocabulary without whitespace would be a hard dead end.
        let bare = TokenByteTable::from_raw(vec![
            Some(b"{".to_vec()),
            Some(b"}".to_vec()),
            Some(b"\"a\"".to_vec()),
        ]);
        let schema = ConstraintSpec::JsonSchema(json!({
            "type":"object",
            "properties":{"q":{"type":"integer"}},
            "required":["q"],
            "additionalProperties": false
        }))
        .compile()
        .unwrap();
        let mut state = ConstraintState::new(schema, &bare, vec![]);
        state.advance(0);
        logits.truncate(3);
        assert_eq!(state.mask(&mut logits), 0);
    }

    #[test]
    fn gap_whitespace_is_bounded() {
        // 12 consecutive gap-whitespace bytes are fine, the 13th is not —
        // in the lead-in, between tokens, and after the close.
        assert!(accepts_obj(&format!("{}{{\"a\":1}}", " ".repeat(12))));
        assert!(!accepts_obj(&format!("{}{{\"a\":1}}", " ".repeat(13))));
        assert!(accepts_obj(&format!("{{\"a\":{}1}}", "\n ".repeat(6))));
        assert!(!accepts_obj(&format!("{{\"a\":{}1}}", " ".repeat(13))));
        assert!(accepts_obj(&format!("{{\"a\":1}}{}", " ".repeat(12))));
        assert!(!accepts_obj(&format!("{{\"a\":1}}{}", " ".repeat(13))));
        // Consuming a real token resets the run…
        assert!(accepts_obj(&format!(
            "{}{{{}\"a\"{}:{}1{}}}",
            " ".repeat(10),
            " ".repeat(10),
            " ".repeat(10),
            " ".repeat(10),
            " ".repeat(10)
        )));
        // …and string CONTENT whitespace is unbounded (and also resets).
        assert!(accepts_obj(&format!("{{\"a\":\"{}\"}}", " ".repeat(40))));
    }

    #[test]
    fn free_string_fast_path_utf8_splices() {
        // 0 `{`, 1 `"k"`, 2 `:`, 3 `"`, 4 `x` (plain ascii), 5 lead C3,
        // 6 continuation A9, 7 `é` complete, 8 `x"` (closes the string).
        let table = TokenByteTable::from_raw(vec![
            Some(b"{".to_vec()),
            Some(b"\"k\"".to_vec()),
            Some(b":".to_vec()),
            Some(b"\"".to_vec()),
            Some(b"x".to_vec()),
            Some(vec![0xC3]),
            Some(vec![0xA9]),
            Some(vec![0xC3, 0xA9]),
            Some(b"x\"".to_vec()),
        ]);
        let schema = ConstraintSpec::JsonObject.compile().unwrap();
        let mut state = ConstraintState::new(schema, &table, vec![]);
        for id in [0u32, 1, 2, 3] {
            state.advance(id); // {"k":"
        }
        // Plain content state: `{` and `:` are legal CONTENT, a fresh
        // lead, a full é, and both closing tokens work; a bare
        // continuation does not, and `"k"` closes the string then dies on
        // the stray `k`.
        assert_eq!(masked_ids(&mut state), vec![0, 2, 3, 4, 5, 7, 8]);
        state.advance(5); // lone C3: character now incomplete
        // Only the continuation byte may follow — not even `"`.
        assert_eq!(masked_ids(&mut state), vec![6]);
        state.advance(6);
        assert_eq!(masked_ids(&mut state), vec![0, 2, 3, 4, 5, 7, 8]);
    }

    #[test]
    fn spm_and_byte_level_unmapping() {
        assert_eq!(spm_bytes("\u{2581}foo"), b" foo");
        assert_eq!(spm_bytes("<0x0A>"), vec![0x0A]);
        assert_eq!(spm_bytes("caf\u{e9}"), "café".as_bytes());
        let unmap = gpt2_unmap();
        assert_eq!(byte_level_bytes("\u{120}foo", &unmap), b" foo");
        assert_eq!(byte_level_bytes("\u{10A}", &unmap), b"\n");
        // 世 in byte-level form: E4 B8 96 mapped char-by-char.
        let mapped: String = "世"
            .as_bytes()
            .iter()
            .map(|&b| {
                unmap
                    .iter()
                    .find(|(_, &v)| v == b)
                    .map(|(&c, _)| c)
                    .unwrap()
            })
            .collect();
        assert_eq!(byte_level_bytes(&mapped, &unmap), "世".as_bytes());
    }
}
