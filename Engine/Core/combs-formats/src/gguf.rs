//! GGUF adapter (llama.cpp ecosystem).
//!
//! Reads GGUF v2/v3 files: header, metadata KV pairs, tensor infos and the
//! aligned tensor data section (mmap-backed). Implements [`ModelSource`] by
//! mapping ggml names to HF names (`blk.0.attn_q.weight` →
//! `model.layers.0.self_attn.q_proj.weight`) and ggml dimensions to HF
//! layout (`[in, out]` → `[out, in]`, which for row-major data is just a
//! shape reversal — no data movement).
//!
//! Supported tensor types: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
//! Q4_K, Q5_K, Q6_K.
//! Quantized tensors are dequantized on load (CPU scalar path) — wiring
//! `QuantizedLinear` to keep them packed in VRAM is a follow-up. K-quant
//! superblock layouts follow ggml exactly (256-value blocks, 6-bit packed
//! scales/mins for 4/5_K, per-16 i8 scales for 6_K).
//!
//! Tokenizer: a sibling `tokenizer.json` is used when present; otherwise a
//! BPE `tokenizer.json` is synthesized from the GGUF tokenizer metadata
//! (tokens/scores/types/merges) and cached next to the model.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::metadata::ModelMetadata;
use crate::source::{ModelSource, SamplerConfig, TensorDtype, TensorReader};
use crate::tokenizer::TokenizerSpec;
use crate::{FormatError, Result};

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"

#[derive(Debug, Clone)]
enum MetaValue {
    U32(u32),
    I32(i32),
    U64(u64),
    F32(f32),
    Bool(bool),
    String(String),
    Strings(Vec<String>),
    F32s(Vec<f32>),
    I32s(Vec<i32>),
}

#[derive(Debug, Clone)]
struct TensorInfo {
    name: String,
    dims: Vec<usize>, // ggml order (fastest dim first)
    ggml_type: u32,
    offset: usize, // relative to data section
}

/// A parsed GGUF file.
pub struct GgufSource {
    path: PathBuf,
    mmap: Mmap,
    metadata: ModelMetadata,
    kv: HashMap<String, MetaValue>,
    tensors: HashMap<String, TensorInfo>,
    data_start: usize,
    tokenizer_json: PathBuf,
    added_tokens: HashMap<u32, String>,
    eos_ids: Vec<u32>,
    bos_id: Option<u32>,
}

// ---------------------------------------------------------------------------
// parsing helpers (little-endian cursor)

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(FormatError::Safetensors("gguf: unexpected end of file".into()));
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| FormatError::Safetensors(format!("gguf: bad utf8 string: {e}")))
    }
}

const META_U8: u32 = 0;
const META_I8: u32 = 1;
const META_U16: u32 = 2;
const META_I16: u32 = 3;
const META_U32: u32 = 4;
const META_I32: u32 = 5;
const META_F32: u32 = 6;
const META_BOOL: u32 = 7;
const META_STRING: u32 = 8;
const META_ARRAY: u32 = 9;
const META_U64: u32 = 10;
const META_I64: u32 = 11;
const META_F64: u32 = 12;

fn read_meta_value(c: &mut Cursor, ty: u32) -> Result<MetaValue> {
    Ok(match ty {
        META_U8 => MetaValue::U32(c.u8()? as u32),
        META_I8 => MetaValue::I32(c.u8()? as i8 as i32),
        META_U16 => MetaValue::U32(c.u16()? as u32),
        META_I16 => MetaValue::I32(c.u16()? as i16 as i32),
        META_U32 => MetaValue::U32(c.u32()?),
        META_I32 => MetaValue::I32(c.i32()?),
        META_U64 => MetaValue::U64(c.u64()?),
        META_I64 => MetaValue::U64(c.i64()? as u64),
        META_F32 => MetaValue::F32(c.f32()?),
        META_F64 => MetaValue::F32(f64::from_le_bytes(c.take(8)?.try_into().unwrap()) as f32),
        META_BOOL => MetaValue::Bool(c.u8()? != 0),
        META_STRING => MetaValue::String(c.string()?),
        META_ARRAY => {
            let elem_ty = c.u32()?;
            let len = c.u64()? as usize;
            match elem_ty {
                META_STRING => {
                    let mut out = Vec::with_capacity(len);
                    for _ in 0..len {
                        out.push(c.string()?);
                    }
                    MetaValue::Strings(out)
                }
                META_F32 => {
                    let mut out = Vec::with_capacity(len);
                    for _ in 0..len {
                        out.push(c.f32()?);
                    }
                    MetaValue::F32s(out)
                }
                META_I32 | META_I16 | META_I8 => {
                    let mut out = Vec::with_capacity(len);
                    for _ in 0..len {
                        out.push(read_meta_value(c, elem_ty).map(|v| match v {
                            MetaValue::I32(i) => i,
                            _ => 0,
                        })?);
                    }
                    MetaValue::I32s(out)
                }
                META_U32 | META_U16 | META_U8 => {
                    let mut out = Vec::with_capacity(len);
                    for _ in 0..len {
                        out.push(read_meta_value(c, elem_ty).map(|v| match v {
                            MetaValue::U32(u) => u as i32,
                            _ => 0,
                        })?);
                    }
                    MetaValue::I32s(out)
                }
                other => {
                    return Err(FormatError::Safetensors(format!(
                        "gguf: unsupported metadata array element type {other}"
                    )));
                }
            }
        }
        other => {
            return Err(FormatError::Safetensors(format!(
                "gguf: unsupported metadata type {other}"
            )));
        }
    })
}

// ggml tensor types we support.
const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
const GGML_Q4_0: u32 = 2;
const GGML_Q4_1: u32 = 3;
const GGML_Q5_0: u32 = 6;
const GGML_Q5_1: u32 = 7;
const GGML_Q8_0: u32 = 8;
const GGML_Q4_K: u32 = 12;
const GGML_Q5_K: u32 = 13;
const GGML_Q6_K: u32 = 14;
const GGML_BF16: u32 = 30;

impl GgufSource {
    /// Opens and parses a `.gguf` file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut c = Cursor::new(&mmap);

        if c.u32()? != GGUF_MAGIC {
            return Err(FormatError::Safetensors("gguf: bad magic".into()));
        }
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            return Err(FormatError::Safetensors(format!(
                "gguf: unsupported version {version}"
            )));
        }
        let tensor_count = c.u64()? as usize;
        let kv_count = c.u64()? as usize;

        let mut kv = HashMap::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = c.string()?;
            let ty = c.u32()?;
            let value = read_meta_value(&mut c, ty)?;
            kv.insert(key, value);
        }

        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = c.string()?;
            let n_dims = c.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(c.u64()? as usize);
            }
            let ggml_type = c.u32()?;
            let offset = c.u64()? as usize;
            tensors.insert(name.clone(), TensorInfo { name, dims, ggml_type, offset });
        }

        // Tensor data starts after the info section, aligned to 32 bytes.
        let alignment = match kv.get("general.alignment") {
            Some(MetaValue::U32(a)) => *a as usize,
            _ => 32,
        };
        let data_start = c.pos.div_ceil(alignment) * alignment;

        let metadata = build_model_metadata(&kv)?;
        let (eos_ids, bos_id, added_tokens) = tokenizer_ids(&kv);
        let tokenizer_json = ensure_tokenizer_json(&path, &kv)?;

        let mut source = GgufSource {
            path,
            mmap,
            metadata,
            kv,
            tensors,
            data_start,
            tokenizer_json,
            added_tokens,
            eos_ids,
            bos_id,
        };
        // GGUF llama files usually include output.weight; if absent, lm_head
        // is tied to the embedding matrix.
        source.metadata.tie_word_embeddings = !source.tensors.contains_key("output.weight");
        Ok(source)
    }

    fn kv_u64(&self, key: &str) -> Option<u64> {
        match self.kv.get(key) {
            Some(MetaValue::U32(v)) => Some(*v as u64),
            Some(MetaValue::U64(v)) => Some(*v),
            Some(MetaValue::I32(v)) => Some(*v as u64),
            _ => None,
        }
    }
}

fn build_model_metadata(kv: &HashMap<String, MetaValue>) -> Result<ModelMetadata> {
    let get_u64 = |key: &str| -> Option<u64> {
        match kv.get(key) {
            Some(MetaValue::U32(v)) => Some(*v as u64),
            Some(MetaValue::U64(v)) => Some(*v),
            Some(MetaValue::I32(v)) => Some(*v as u64),
            _ => None,
        }
    };
    let get_f32 = |key: &str| -> Option<f32> {
        match kv.get(key) {
            Some(MetaValue::F32(v)) => Some(*v),
            Some(MetaValue::U32(v)) => Some(*v as f32),
            _ => None,
        }
    };
    let get_str = |key: &str| -> Option<String> {
        match kv.get(key) {
            Some(MetaValue::String(s)) => Some(s.clone()),
            _ => None,
        }
    };

    let arch = get_str("general.architecture")
        .ok_or_else(|| FormatError::MissingField("general.architecture".into()))?;
    let prefix = arch.clone();
    let field = |name: &str| get_u64(&format!("{prefix}.{name}"));

    let hidden = field("embedding_length")
        .ok_or_else(|| FormatError::MissingField("embedding_length".into()))? as usize;
    let heads = field("attention.head_count")
        .ok_or_else(|| FormatError::MissingField("attention.head_count".into()))?
        as usize;
    let kv_heads = field("attention.head_count_kv").unwrap_or(heads as u64) as usize;
    let layers = field("block_count")
        .ok_or_else(|| FormatError::MissingField("block_count".into()))? as usize;
    let ctx = field("context_length").unwrap_or(2048) as usize;
    let ffn = field("feed_forward_length").unwrap_or((hidden * 4) as u64) as usize;
    let vocab = match kv.get(&format!("{prefix}.vocab_size")) {
        Some(MetaValue::U32(v)) => *v as usize,
        Some(MetaValue::U64(v)) => *v as usize,
        _ => match kv.get("tokenizer.ggml.tokens") {
            Some(MetaValue::Strings(t)) => t.len(),
            _ => 0,
        },
    };
    let (eos_ids, bos_id, _) = tokenizer_ids(kv);

    Ok(ModelMetadata {
        architecture: arch,
        hidden_size: hidden,
        intermediate_size: ffn,
        num_hidden_layers: layers,
        num_attention_heads: heads,
        num_key_value_heads: kv_heads,
        vocab_size: vocab,
        max_position_embeddings: ctx,
        rms_norm_eps: get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5) as f64,
        rope_theta: get_f32(&format!("{prefix}.rope.freq_base")).unwrap_or(10000.0) as f64,
        // GGUF files usually include output.weight; if absent, the head is tied.
        tie_word_embeddings: false, // refined in load() via tensor presence
        head_dim: hidden / heads,
        attention_bias: false,
        bos_token_id: bos_id,
        eos_token_ids: eos_ids,
        vision: None,
        // GGUF: llama-family defaults (all-global). Gemma GGUF keys
        // (attention.sliding_window etc.) are a follow-up — the
        // safetensors path is the Gemma reference for U1.
        attention_pattern: crate::metadata::AttentionPattern::default(),
    })
}

fn tokenizer_ids(kv: &HashMap<String, MetaValue>) -> (Vec<u32>, Option<u32>, HashMap<u32, String>) {
    let mut eos = Vec::new();
    let mut bos = None;
    let mut added = HashMap::new();
    if let Some(MetaValue::U32(v)) = kv.get("tokenizer.ggml.eos_token_id") {
        eos.push(*v);
    }
    if let Some(MetaValue::U32(v)) = kv.get("tokenizer.ggml.bos_token_id") {
        bos = Some(*v);
    }
    // Mark special tokens from the token_type array (3 = control).
    if let (Some(MetaValue::Strings(tokens)), Some(MetaValue::I32s(types))) =
        (kv.get("tokenizer.ggml.tokens"), kv.get("tokenizer.ggml.token_type"))
    {
        for (i, (tok, ty)) in tokens.iter().zip(types.iter()).enumerate() {
            if *ty == 3 {
                added.insert(i as u32, tok.clone());
            }
        }
    }
    (eos, bos, added)
}

/// Counts GGUF special tokens (token_type 3 = control, 4 = user-defined).
fn special_token_count(kv: &HashMap<String, MetaValue>) -> usize {
    match kv.get("tokenizer.ggml.token_type") {
        Some(MetaValue::I32s(types)) => types.iter().filter(|t| **t == 3 || **t == 4).count(),
        _ => 0,
    }
}

/// Builds a minimal HF BPE tokenizer.json from GGUF tokenizer metadata when
/// no sibling tokenizer.json exists (cached alongside the model file).
///
/// Staleness: v1 syntheses wrote `"added_tokens": []`, which BPE-shredded
/// ChatML control tokens (`<|im_start|>`/`<|im_end|>`) — the "garbled GGUF"
/// bug — and the poisoned cache was sticky. The cached file is regenerated
/// whenever its added_tokens count disagrees with the GGUF metadata. (No
/// in-file version marker: the tokenizers crate rejects unknown top-level
/// keys.)
fn ensure_tokenizer_json(path: &Path, kv: &HashMap<String, MetaValue>) -> Result<PathBuf> {
    let sibling = path.with_file_name("tokenizer.json");
    if sibling.exists() {
        return Ok(sibling);
    }
    let cached = path.with_extension("tokenizer.json");
    if cached.exists() {
        let have = std::fs::read_to_string(&cached)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v.get("added_tokens")?.as_array().map(Vec::len));
        if have == Some(special_token_count(kv)) {
            return Ok(cached);
        }
        // Stale synthesis — regenerate below.
    }

    let tokens = match kv.get("tokenizer.ggml.tokens") {
        Some(MetaValue::Strings(t)) => t,
        _ => {
            return Err(FormatError::MissingField(
                "tokenizer.ggml.tokens (and no sibling tokenizer.json)".into(),
            ));
        }
    };
    let scores: Vec<f32> = match kv.get("tokenizer.ggml.scores") {
        Some(MetaValue::F32s(s)) => s.clone(),
        _ => vec![0.0; tokens.len()],
    };
    let merges: Vec<String> = match kv.get("tokenizer.ggml.merges") {
        Some(MetaValue::Strings(m)) => m.clone(),
        _ => vec![],
    };

    let mut vocab = serde_json::Map::new();
    for (i, tok) in tokens.iter().enumerate() {
        vocab.insert(tok.clone(), serde_json::Value::from(i));
    }
    let mut ordered: Vec<usize> = (0..tokens.len()).collect();
    ordered.sort_by(|&a, &b| {
        scores[a].partial_cmp(&scores[b]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = serde_json::Map::new();
    for (rank, id) in ordered.iter().enumerate() {
        ranks.insert(id.to_string(), serde_json::Value::from(rank));
    }

    // Special tokens (GGUF token_type 3 = control, 4 = user-defined) must
    // be registered as added_tokens so the tokenizer keeps them atomic
    // instead of BPE-shredding them (e.g. <|im_start|>/<|im_end|>).
    let mut added_tokens = Vec::new();
    if let Some(MetaValue::I32s(types)) = kv.get("tokenizer.ggml.token_type") {
        for (i, (tok, ty)) in tokens.iter().zip(types.iter()).enumerate() {
            if *ty == 3 || *ty == 4 {
                added_tokens.push(serde_json::json!({
                    "id": i,
                    "content": tok,
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true,
                }));
            }
        }
    }

    let json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
            ]
        },
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": vocab,
            "merges": merges,
        }
    });
    let serialized = serde_json::to_string(&json).map_err(|e| FormatError::Safetensors(format!("tokenizer json: {e}")))?;
    std::fs::write(&cached, serialized)?;
    Ok(cached)
}

/// Maps a ggml tensor name to its HF equivalent (`None` = skip tensor).
fn map_tensor_name(ggml: &str) -> Option<String> {
    if ggml == "token_embd.weight" {
        return Some("model.embed_tokens.weight".into());
    }
    if ggml == "output.weight" {
        return Some("lm_head.weight".into());
    }
    if ggml == "output_norm.weight" {
        return Some("model.norm.weight".into());
    }
    let rest = ggml.strip_prefix("blk.")?;
    let (layer, rest) = rest.split_once('.')?;
    let hf = match rest {
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q.bias" => "self_attn.q_proj.bias",
        "attn_k.bias" => "self_attn.k_proj.bias",
        "attn_v.bias" => "self_attn.v_proj.bias",
        "attn_output.bias" => "self_attn.o_proj.bias",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    };
    Some(format!("model.layers.{layer}.{hf}"))
}

/// Number of elements in a ggml tensor.
fn num_elements(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// Byte size of a ggml tensor's data.
fn tensor_byte_size(info: &TensorInfo) -> Result<usize> {
    let n = num_elements(&info.dims);
    let block = |bs: usize| -> Result<usize> {
        if n % 32 != 0 {
            return Err(FormatError::Safetensors(format!(
                "gguf tensor {}: {n} elements not divisible by block size 32",
                info.name
            )));
        }
        Ok(n / 32 * bs)
    };
    let superblock = |bs: usize| -> Result<usize> {
        if n % 256 != 0 {
            return Err(FormatError::Safetensors(format!(
                "gguf tensor {}: {n} elements not divisible by K-quant superblock 256",
                info.name
            )));
        }
        Ok(n / 256 * bs)
    };
    Ok(match info.ggml_type {
        GGML_F32 => n * 4,
        GGML_F16 | GGML_BF16 => n * 2,
        GGML_Q4_0 => block(18)?, // 2-byte scale + 16 bytes per 32 values
        GGML_Q4_1 => block(20)?, // f16 scale + f16 min + 16 bytes
        GGML_Q5_0 => block(22)?, // f16 scale + u32 high bits + 16 bytes
        GGML_Q5_1 => block(24)?, // f16 scale + f16 min + u32 high bits + 16 bytes
        GGML_Q8_0 => block(34)?, // 2-byte scale + 32 int8 per 32 values
        // K-quants: 256-value superblocks.
        GGML_Q4_K => superblock(144)?, // 2×f16 + 12B packed scales/mins + 128B quants
        GGML_Q5_K => superblock(176)?, // + 32B high bits
        GGML_Q6_K => superblock(210)?, // 128B ql + 64B qh + 16×i8 scales + f16
        other => {
            return Err(FormatError::UnsupportedDtype {
                tensor: info.name.clone(),
                dtype: format!("ggml_type {other}"),
            });
        }
    })
}

/// Dequantizes a Q4_0 tensor to f32. Blocks of 32 values: f16 scale + 16
/// bytes; value i (i<16) is the low nibble of byte i, value i+16 the high.
///
/// Public via [`crate::quants`]: this scalar path is the golden reference
/// the fused CubeCL kernels validate against.
pub fn dequantize_q4_0(data: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut fixed = Vec::with_capacity(n);
    for block in data.chunks_exact(18) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let mut vals = [0f32; 32];
        for j in 0..16 {
            let byte = block[2 + j];
            vals[j] = ((byte & 0x0F) as i32 - 8) as f32 * d;
            vals[j + 16] = ((byte >> 4) as i32 - 8) as f32 * d;
        }
        fixed.extend_from_slice(&vals);
    }
    if fixed.len() != n {
        return Err(FormatError::Safetensors(format!(
            "q4_0 dequant size mismatch: {} != {n}",
            fixed.len()
        )));
    }
    Ok(fixed)
}

/// Dequantizes a Q8_0 tensor to f32.
///
/// Public via [`crate::quants`] as the golden reference for the GPU kernel.
pub fn dequantize_q8_0(data: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for block in data.chunks_exact(34) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        for &b in &block[2..34] {
            out.push((b as i8) as f32 * d);
        }
    }
    if out.len() != n {
        return Err(FormatError::Safetensors(format!(
            "q8_0 dequant size mismatch: {} != {n}",
            out.len()
        )));
    }
    Ok(out)
}

/// Dequantizes a Q4_1 tensor to f32. Like Q4_0 but asymmetric:
/// value = q·d + m (f16 scale + f16 min per 32-value block).
fn dequantize_q4_1(data: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for block in data.chunks_exact(20) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let mut vals = [0f32; 32];
        for j in 0..16 {
            let byte = block[4 + j];
            vals[j] = (byte & 0x0F) as f32 * d + m;
            vals[j + 16] = (byte >> 4) as f32 * d + m;
        }
        out.extend_from_slice(&vals);
    }
    if out.len() != n {
        return Err(FormatError::Safetensors(format!(
            "q4_1 dequant size mismatch: {} != {n}",
            out.len()
        )));
    }
    Ok(out)
}

/// Dequantizes a Q5_0 tensor to f32. 32-value blocks (22 bytes): f16
/// scale, u32 high bits (bit j → value j, bit j+16 → value j+16), 16
/// packed nibble bytes; values biased by -16.
///
/// Public via [`crate::quants`] as the golden reference for the GPU kernel.
pub fn dequantize_q5_0(data: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for block in data.chunks_exact(22) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let mut vals = [0f32; 32];
        for j in 0..16 {
            let byte = block[6 + j];
            let lo = ((byte & 0x0F) as u32 | (((qh >> j) & 1) << 4)) as i32 - 16;
            let hi = ((byte >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4)) as i32 - 16;
            vals[j] = lo as f32 * d;
            vals[j + 16] = hi as f32 * d;
        }
        out.extend_from_slice(&vals);
    }
    if out.len() != n {
        return Err(FormatError::Safetensors(format!(
            "q5_0 dequant size mismatch: {} != {n}",
            out.len()
        )));
    }
    Ok(out)
}

/// Dequantizes a Q5_1 tensor to f32. Like Q5_0 but asymmetric (scale +
/// min, no bias): value = q·d + m. 24-byte blocks.
fn dequantize_q5_1(data: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for block in data.chunks_exact(24) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let mut vals = [0f32; 32];
        for j in 0..16 {
            let byte = block[8 + j];
            let lo = (byte & 0x0F) as u32 | (((qh >> j) & 1) << 4);
            let hi = (byte >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4);
            vals[j] = lo as f32 * d + m;
            vals[j + 16] = hi as f32 * d + m;
        }
        out.extend_from_slice(&vals);
    }
    if out.len() != n {
        return Err(FormatError::Safetensors(format!(
            "q5_1 dequant size mismatch: {} != {n}",
            out.len()
        )));
    }
    Ok(out)
}

/// K-quant 6-bit scale/min unpacking (ggml `get_scale_min_k4`): the 12
/// scale bytes pack eight 6-bit scales and eight 6-bit mins, with the top
/// 2 bits of bytes 0..8 carrying the high bits of sub-blocks 4..8.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantizes a Q4_K tensor to f32. 256-value superblocks (144 bytes):
/// f16 d, f16 dmin, 12B packed scales/mins, 128B 4-bit quants.
///
/// Public via [`crate::quants`] as the golden reference for the GPU kernel.
pub fn dequantize_q4_k(data: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for sb in data.chunks_exact(144) {
        let d = half::f16::from_le_bytes([sb[0], sb[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([sb[2], sb[3]]).to_f32();
        let scales = &sb[4..16];
        let qs = &sb[16..144];
        for j in 0..4 {
            // Each 64-value group uses 32 qs bytes; low nibbles → first
            // sub-block, high nibbles → second.
            let (sc1, m1) = scale_min_k4(2 * j, scales);
            let (sc2, m2) = scale_min_k4(2 * j + 1, scales);
            let (d1, fmin1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, fmin2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                let byte = qs[32 * j + l];
                out.push(d1 * (byte & 0x0F) as f32 - fmin1);
            }
            for l in 0..32 {
                let byte = qs[32 * j + l];
                out.push(d2 * (byte >> 4) as f32 - fmin2);
            }
        }
    }
    if out.len() != n {
        return Err(FormatError::Safetensors(format!(
            "q4_k dequant size mismatch: {} != {n}",
            out.len()
        )));
    }
    Ok(out)
}

/// Dequantizes a Q5_K tensor to f32. Like Q4_K plus 32 high-bit bytes:
/// group j reads bit 2j (low sub-block) / 2j+1 (high sub-block) of qh.
fn dequantize_q5_k(data: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for sb in data.chunks_exact(176) {
        let d = half::f16::from_le_bytes([sb[0], sb[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([sb[2], sb[3]]).to_f32();
        let scales = &sb[4..16];
        let qh = &sb[16..48];
        let qs = &sb[48..176];
        for j in 0..4 {
            let (sc1, m1) = scale_min_k4(2 * j, scales);
            let (sc2, m2) = scale_min_k4(2 * j + 1, scales);
            let (d1, fmin1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, fmin2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                let lo = (qs[32 * j + l] & 0x0F) as u32;
                let hi = ((qh[l] >> (2 * j)) & 1) as u32;
                out.push(d1 * ((lo | (hi << 4)) as f32) - fmin1);
            }
            for l in 0..32 {
                let lo = (qs[32 * j + l] >> 4) as u32;
                let hi = ((qh[l] >> (2 * j + 1)) & 1) as u32;
                out.push(d2 * ((lo | (hi << 4)) as f32) - fmin2);
            }
        }
    }
    if out.len() != n {
        return Err(FormatError::Safetensors(format!(
            "q5_k dequant size mismatch: {} != {n}",
            out.len()
        )));
    }
    Ok(out)
}

/// Dequantizes a Q6_K tensor to f32. 256-value superblocks (210 bytes):
/// 128B low nibbles, 64B high 2-bit fields, 16 i8 scales (one per 16
/// values), f16 super-scale. Values are biased by -32.
///
/// Public via [`crate::quants`] as the golden reference for the GPU kernel.
pub fn dequantize_q6_k(data: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for sb in data.chunks_exact(210) {
        let d = half::f16::from_le_bytes([sb[208], sb[209]]).to_f32();
        // Process the superblock as two 128-value halves.
        for half_idx in 0..2 {
            let ql = &sb[64 * half_idx..64 * half_idx + 64];
            let qh = &sb[128 + 32 * half_idx..128 + 32 * half_idx + 32];
            let scales = &sb[192 + 8 * half_idx..192 + 8 * half_idx + 8];
            let mut vals = [0f32; 128];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 0x03) << 4)) as i8 as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | ((qh[l] & 0x0C) >> 2 << 4)) as i8 as i32 - 32;
                let q3 = ((ql[l] >> 4) | ((qh[l] & 0x30) >> 4 << 4)) as i8 as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | ((qh[l] & 0xC0) >> 6 << 4)) as i8 as i32 - 32;
                vals[l] = d * (scales[is] as i8) as f32 * q1 as f32;
                vals[l + 32] = d * (scales[is + 2] as i8) as f32 * q2 as f32;
                vals[l + 64] = d * (scales[is + 4] as i8) as f32 * q3 as f32;
                vals[l + 96] = d * (scales[is + 6] as i8) as f32 * q4 as f32;
            }
            out.extend_from_slice(&vals);
        }
    }
    if out.len() != n {
        return Err(FormatError::Safetensors(format!(
            "q6_k dequant size mismatch: {} != {n}",
            out.len()
        )));
    }
    Ok(out)
}

impl ModelSource for GgufSource {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn tensor_names(&self) -> Vec<String> {
        self.tensors
            .keys()
            .filter_map(|k| map_tensor_name(k))
            .collect()
    }

    fn open_tensor(&self, name: &str) -> Result<TensorReader<'_>> {
        // Find the ggml tensor mapping to this HF name.
        let (ggml_name, info) = self
            .tensors
            .iter()
            .find(|(k, _)| map_tensor_name(k).as_deref() == Some(name))
            .ok_or_else(|| FormatError::TensorNotFound(name.to_string()))?;
        let _ = ggml_name;

        let size = tensor_byte_size(info)?;
        let start = self.data_start + info.offset;
        let data = self
            .mmap
            .get(start..start + size)
            .ok_or_else(|| FormatError::Safetensors(format!("gguf tensor {} out of bounds", info.name)))?;

        // HF layout = ggml dims reversed (row-major data needs no movement).
        let shape: Vec<usize> = info.dims.iter().rev().copied().collect();
        let n = num_elements(&info.dims);

        match info.ggml_type {
            GGML_F32 | GGML_F16 | GGML_BF16 => {
                let dtype = match info.ggml_type {
                    GGML_F32 => TensorDtype::F32,
                    GGML_F16 => TensorDtype::F16,
                    _ => TensorDtype::BF16,
                };
                Ok(TensorReader::new(name.to_string(), shape, dtype, data))
            }
            GGML_Q4_0 => {
                let values = dequantize_q4_0(data, n)?;
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                Ok(TensorReader::owned(name.to_string(), shape, bytes))
            }
            GGML_Q4_1 => {
                let values = dequantize_q4_1(data, n)?;
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                Ok(TensorReader::owned(name.to_string(), shape, bytes))
            }
            GGML_Q5_0 => {
                let values = dequantize_q5_0(data, n)?;
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                Ok(TensorReader::owned(name.to_string(), shape, bytes))
            }
            GGML_Q5_1 => {
                let values = dequantize_q5_1(data, n)?;
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                Ok(TensorReader::owned(name.to_string(), shape, bytes))
            }
            GGML_Q8_0 => {
                let values = dequantize_q8_0(data, n)?;
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                Ok(TensorReader::owned(name.to_string(), shape, bytes))
            }
            GGML_Q4_K => {
                let values = dequantize_q4_k(data, n)?;
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                Ok(TensorReader::owned(name.to_string(), shape, bytes))
            }
            GGML_Q5_K => {
                let values = dequantize_q5_k(data, n)?;
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                Ok(TensorReader::owned(name.to_string(), shape, bytes))
            }
            GGML_Q6_K => {
                let values = dequantize_q6_k(data, n)?;
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                Ok(TensorReader::owned(name.to_string(), shape, bytes))
            }
            other => Err(FormatError::UnsupportedDtype {
                tensor: info.name.clone(),
                dtype: format!("ggml_type {other}"),
            }),
        }
    }

    fn tokenizer(&self) -> Result<TokenizerSpec> {
        Ok(TokenizerSpec {
            tokenizer_json: self.tokenizer_json.clone(),
            added_tokens: self.added_tokens.clone(),
            chat_template: None,
        })
    }

    fn open_tensor_quant(&self, name: &str) -> Result<Option<crate::QuantTensor<'_>>> {
        let Some((_, info)) = self
            .tensors
            .iter()
            .find(|(k, _)| map_tensor_name(k).as_deref() == Some(name))
        else {
            return Ok(None);
        };
        let format = match info.ggml_type {
            GGML_Q4_0 => crate::QuantFormat::Q4_0,
            GGML_Q5_0 => crate::QuantFormat::Q5_0,
            GGML_Q8_0 => crate::QuantFormat::Q8_0,
            GGML_Q4_K => crate::QuantFormat::Q4K,
            GGML_Q6_K => crate::QuantFormat::Q6K,
            _ => return Ok(None),
        };
        let size = tensor_byte_size(info)?;
        let start = self.data_start + info.offset;
        let data = self.mmap.get(start..start + size).ok_or_else(|| {
            FormatError::Safetensors(format!("gguf tensor {} out of bounds", info.name))
        })?;
        Ok(Some(crate::QuantTensor {
            format,
            shape: info.dims.iter().rev().copied().collect(),
            data,
        }))
    }

    fn sampler_defaults(&self) -> Option<SamplerConfig> {
        None
    }
}

impl GgufSource {
    /// End-of-sequence ids from tokenizer metadata.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    /// Model file path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod quant_access_tests {
    use super::*;
    use crate::ModelSource;

    /// Scratch diagnostic against a locally cached model; not part of CI.
    #[test]
    #[ignore]
    fn real_gguf_quant_access() {
        let path = dirs_home().join(".cache/combs/models/llama-3.2-1b-instruct-gguf/model.gguf");
        let src = GgufSource::load(&path).unwrap();
        for name in [
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ] {
            let qt = src.open_tensor_quant(name).unwrap();
            println!("{name}: {:?}", qt.map(|q| (q.format, q.shape, q.data.len())));
        }
    }

    fn dirs_home() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("HOME").unwrap())
    }
}
