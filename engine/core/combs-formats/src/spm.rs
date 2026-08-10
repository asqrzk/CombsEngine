//! SentencePiece (`.model` protobuf) → HuggingFace `tokenizer.json`
//! (Unigram) converter block.
//!
//! `.task` bundles and `.litertlm` `SP_Tokenizer` sections carry a raw
//! SentencePiece model; the engine's tokenizer path is the `tokenizers`
//! crate, so this block converts once and caches the JSON next to the
//! source (`<file>.spm.tokenizer.json`). The protobuf reader is
//! hand-rolled (~60 lines) — no prost needed for a schema this small.
//!
//! Special tokens (CONTROL / USER_DEFINED / UNKNOWN pieces) are written
//! as `added_tokens` with `special: true` — without them the tokenizer
//! BPE/unigram-shreds chat-template markers (the GGUF E2 lesson).

use std::path::{Path, PathBuf};

use crate::protomin::Proto;
use crate::{FormatError, Result};

// ── minimal protobuf wire reader ─────────────────────────────────────
// (shared implementation lives in protomin.rs; `Proto` is re-exported
// here under the old private name so the parsing code below is unchanged)
type ProtoReader<'a> = Proto<'a>;

// ── SentencePiece model pieces ───────────────────────────────────────
const TYPE_NORMAL: u64 = 1;
const TYPE_UNKNOWN: u64 = 2;
const TYPE_CONTROL: u64 = 3;
const TYPE_USER_DEFINED: u64 = 4;
const TYPE_BYTE: u64 = 6;

struct Piece {
    text: String,
    score: f32,
    kind: u64,
}

struct SpModel {
    pieces: Vec<Piece>,
    unk_id: Option<u32>,
    normalizer_name: String,
    precompiled_charsmap: Option<Vec<u8>>,
    add_dummy_prefix: bool,
}

fn parse_model(buf: &[u8]) -> Result<SpModel> {
    let mut model = SpModel {
        pieces: Vec::new(),
        unk_id: None,
        normalizer_name: String::new(),
        precompiled_charsmap: None,
        add_dummy_prefix: true,
    };
    let mut p = ProtoReader::new(buf);
    while let Some((field, wire)) = p.tag()? {
        match (field, wire) {
            (1, 2) => {
                // SentencePiece { 1: piece string, 2: score float, 3: type }
                let sub = p.bytes()?;
                let mut sp = ProtoReader::new(sub);
                let mut piece = Piece { text: String::new(), score: 0.0, kind: TYPE_NORMAL };
                while let Some((f, w)) = sp.tag()? {
                    match (f, w) {
                        (1, 2) => piece.text = sp.string()?,
                        (2, 5) => piece.score = sp.f32()?,
                        (3, 0) => piece.kind = sp.varint()?,
                        (_, w) => sp.skip(w)?,
                    }
                }
                model.pieces.push(piece);
            }
            (2, 2) => {
                // TrainerSpec { 10: unk_id, … } — ids default to the
                // UNKNOWN-typed piece when absent.
                let sub = p.bytes()?;
                let mut sp = ProtoReader::new(sub);
                while let Some((f, w)) = sp.tag()? {
                    match (f, w) {
                        (10, 0) => model.unk_id = Some(sp.varint()? as u32),
                        (_, w) => sp.skip(w)?,
                    }
                }
            }
            (3, 2) => {
                // NormalizerSpec { 1: name, 2: precompiled_charsmap, 4: add_dummy_prefix }
                let sub = p.bytes()?;
                let mut sp = ProtoReader::new(sub);
                while let Some((f, w)) = sp.tag()? {
                    match (f, w) {
                        (1, 2) => model.normalizer_name = sp.string()?,
                        (2, 2) => model.precompiled_charsmap = Some(sp.bytes()?.to_vec()),
                        (4, 0) => model.add_dummy_prefix = sp.varint()? != 0,
                        (_, w) => sp.skip(w)?,
                    }
                }
            }
            (_, w) => p.skip(w)?,
        }
    }
    if model.pieces.is_empty() {
        return Err(FormatError::MissingField("spm: no pieces (not a SentencePiece model?)".into()));
    }
    // unk id: explicit trainer field → UNKNOWN-typed piece → 0.
    if model.unk_id.is_none() {
        model.unk_id = Some(
            model
                .pieces
                .iter()
                .position(|p| p.kind == TYPE_UNKNOWN)
                .map(|i| i as u32)
                .unwrap_or(0),
        );
    }
    Ok(model)
}

fn is_special(kind: u64) -> bool {
    kind == TYPE_UNKNOWN || kind == TYPE_CONTROL || kind == TYPE_USER_DEFINED
}

/// Converts a SentencePiece model to a Unigram tokenizer.json Value.
fn to_tokenizer_json(model: &SpModel) -> Result<serde_json::Value> {
    let mut vocab = Vec::with_capacity(model.pieces.len());
    let mut added_tokens = Vec::new();
    let mut has_byte_pieces = false;
    for (id, piece) in model.pieces.iter().enumerate() {
        vocab.push(serde_json::json!([piece.text, piece.score]));
        if is_special(piece.kind) {
            added_tokens.push(serde_json::json!({
                "id": id,
                "content": piece.text,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            }));
        }
        if piece.kind == TYPE_BYTE {
            has_byte_pieces = true;
        }
    }

    let normalizer = match &model.precompiled_charsmap {
        Some(map) if !map.is_empty() => {
            serde_json::json!({ "type": "Precompiled", "precompiled_charsmap": base64_encode(map) })
        }
        _ if model.normalizer_name.is_empty() || model.normalizer_name == "identity" => {
            serde_json::Value::Null
        }
        _ => serde_json::Value::Null, // named-but-mapless normalizers: approximate as none
    };
    let metaspace = |split: bool| {
        serde_json::json!({
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": if model.add_dummy_prefix { "always" } else { "never" },
            "split": split,
        })
    };

    Ok(serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": normalizer,
        "pre_tokenizer": metaspace(true),
        "post_processor": null,
        "decoder": metaspace(false),
        "model": {
            "type": "Unigram",
            "unk_id": model.unk_id,
            "byte_fallback": has_byte_pieces,
            "vocab": vocab,
        }
    }))
}

/// base64 without an extra dependency (format block stays self-contained).
fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

/// Ensures a tokenizer.json exists for a SentencePiece `.model` file,
/// converting + caching it next to the source on first use. Returns the
/// path (idempotent: an existing cache is reused).
pub fn ensure_tokenizer_json_from_spm(spm_path: &Path) -> Result<PathBuf> {
    let cached = spm_path.with_extension("spm.tokenizer.json");
    if cached.exists() {
        return Ok(cached);
    }
    let buf = std::fs::read(spm_path)?;
    let model = parse_model(&buf)?;
    let json = to_tokenizer_json(&model)?;
    let serialized = serde_json::to_string(&json)
        .map_err(|e| FormatError::Safetensors(format!("spm tokenizer json: {e}")))?;
    std::fs::write(&cached, serialized)?;
    Ok(cached)
}

/// Parses a SentencePiece model and returns its special-token map
/// (id → string), for `TokenizerSpec.added_tokens`.
pub fn spm_added_tokens(spm_path: &Path) -> Result<std::collections::HashMap<u32, String>> {
    let buf = std::fs::read(spm_path)?;
    let model = parse_model(&buf)?;
    Ok(model
        .pieces
        .iter()
        .enumerate()
        .filter(|(_, p)| is_special(p.kind))
        .map(|(i, p)| (i as u32, p.text.clone()))
        .collect())
}
