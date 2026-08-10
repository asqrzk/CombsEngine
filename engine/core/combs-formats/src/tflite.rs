//! TFLite flatbuffer model source (`.tflite` / raw `.task` — `TFL3`).
//!
//! Recon-verified layout (see `documentations/formats-recon.md`):
//! - `Model { subgraphs, buffers, metadata }` at the flatbuffer root
//! - weight tensors live in `Buffer { offset, size }` absolute file ranges
//!   (no inline data vectors in ODML exports)
//! - the model graph in these exports is a weight carrier: tensor NAMES
//!   follow the ODML scheme (`transformer.layer_{i}.attn.q.w`), quantized
//!   as int8 (+ `w_quantized_scale` f32 per output row, `w.sum_i` i32
//!   compensation sums — unused by our f32 compute)
//! - `metadata[]` carries `odml.infra.proto.LlmParameters` (config proto)
//!   and `spm_vocab_model` (SentencePiece tokenizer)
//!
//! This block maps ODML names → canonical HF names so the architecture
//! blocks (gemma.rs) load unchanged: **format ⊥ architecture**.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::flatbuf::FlatBuffer;
use crate::metadata::AttentionPattern;
use crate::protomin::Proto;
use crate::source::{ModelSource, TensorDtype, TensorReader};
use crate::tokenizer::TokenizerSpec;
use crate::{FormatError, ModelMetadata, Result};

fn bad(what: impl Into<String>) -> FormatError {
    FormatError::Safetensors(format!("tflite: {}", what.into()))
}

/// Tensor types we read from the schema enum (subset).
const TFLITE_F32: u8 = 0;
const TFLITE_INT8: u8 = 9;

/// One tensor entry in the subgraph (pre-mapping).
struct RawTensor {
    name: String,
    shape: Vec<usize>,
    dtype: u8,
    buffer: usize,
}

/// A buffer's absolute file range.
#[derive(Clone, Copy)]
struct BufferSpan {
    offset: usize,
    size: usize,
}

/// Quantization companions for an int8 weight: per-output-row f32 scales.
struct QuantInfo {
    scale_buffer: usize,
    rows: usize,
}

/// ODML → HF name mapping for the Gemma family. Returns None for tensors
/// we deliberately skip (activation scales, cache quant scales, per-layer
/// embeddings in MatFormer-family exports — see gemma4 notes).
fn map_name(odml: &str) -> Option<String> {
    let rest = odml.strip_prefix("transformer.")?;
    if rest == "embedder.input_embedding.w" {
        return Some("model.embed_tokens.weight".into());
    }
    if rest == "final_norm.scale" {
        return Some("model.norm.weight".into());
    }
    let layer = rest.strip_prefix("layer_")?;
    let (idx, sub) = layer.split_once('.')?;
    let p = format!("model.layers.{idx}");
    let mapped = match sub {
        "attn.q.w" => "self_attn.q_proj.weight",
        "attn.k.w" => "self_attn.k_proj.weight",
        "attn.v.w" => "self_attn.v_proj.weight",
        "attn.attn_vec_einsum.w" => "self_attn.o_proj.weight",
        "attn.q_norm.scale" => "self_attn.q_norm.weight",
        "attn.k_norm.scale" => "self_attn.k_norm.weight",
        "pre_attention_norm.scale" => "input_layernorm.weight",
        "post_attention_norm.scale" => "post_attention_layernorm.weight",
        "pre_ffw_norm.scale" => "pre_feedforward_layernorm.weight",
        "post_ffw_norm.scale" => "post_feedforward_layernorm.weight",
        "mlp.ff_gate.w" => "mlp.gate_proj.weight",
        "mlp.ff1.w" | "mlp.ffn.w" => "mlp.up_proj.weight",
        "mlp.linear.w" => "mlp.down_proj.weight",
        _ => return None, // activation scales, caches, PLE (gemma4)…
    };
    Some(format!("{p}.{mapped}"))
}

/// Parsed `odml.infra.proto.LlmParameters` (fields decoded empirically —
/// see formats-recon.md; the proto is not published as a .proto).
#[derive(Default, Debug)]
struct LlmParameters {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
    sliding_window_pattern: usize,
    rope_theta: f64,
    rope_local_theta: f64,
    query_pre_attn_scalar: Option<f64>,
    eos_token_ids: Vec<u32>,
    bos_token_id: Option<u32>,
}

/// The model-params sub-message of LlmParameters (field 1). Field numbers
/// were recovered by decoding a gemma-4 export and matching known config
/// values; calibrated per model family at load time by the E2E tests.
fn parse_llm_parameters(buf: &[u8]) -> Result<LlmParameters> {
    let mut out = LlmParameters::default();
    let mut p = Proto::new(buf);
    while let Some((field, wire)) = p.tag()? {
        match (field, wire) {
            (1, 2) => {
                let sub = p.bytes()?;
                let mut sp = Proto::new(sub);
                while let Some((f, w)) = sp.tag()? {
                    match (f, w) {
                        (3, 0) => out.hidden_size = sp.varint()? as usize,
                        (4, 0) => out.intermediate_size = sp.varint()? as usize,
                        (5, 0) => out.head_dim = sp.varint()? as usize,
                        (6, 0) => out.num_attention_heads = sp.varint()? as usize,
                        (7, 0) => out.num_hidden_layers = sp.varint()? as usize,
                        (9, 0) => out.num_key_value_heads = sp.varint()? as usize,
                        (_, w) => sp.skip(w)?,
                    }
                }
            }
            (2, 0) => out.vocab_size = p.varint()? as usize,
            (4, 0) => out.bos_token_id = Some(p.varint()? as u32),
            (_, w) => p.skip(w)?,
        }
    }
    Ok(out)
}

/// A TFLite model file as a [`ModelSource`] (Gemma-family exports).
pub struct TfliteSource {
    _mmap: Mmap,
    spans: Vec<BufferSpan>,
    /// HF name → (raw tensor index, quant info).
    tensors: HashMap<String, (usize, Option<QuantInfo>)>,
    raws: Vec<RawTensor>,
    metadata: ModelMetadata,
    tokenizer_json: PathBuf,
    added_tokens: HashMap<u32, String>,
}

impl TfliteSource {
    /// Opens a raw `.tflite`/`.task` file (TFL3 flatbuffer).
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_at(path.as_ref(), 0)
    }

    /// Opens a TFL3 flatbuffer at byte offset `base` of `path` (base != 0
    /// when the flatbuffer is a section inside a `.litertlm` container).
    /// Buffer spans are section-relative in that case and are rebased to
    /// absolute file offsets, so tensor reads stay zero-copy.
    pub(crate) fn load_at(path: &Path, base: usize) -> Result<Self> {
        Self::load_at_with_spm(path, base, None)
    }

    /// [`TfliteSource::load_at`] with an optional tokenizer blob override
    /// (a `.litertlm` container can carry the tokenizer as its own
    /// section when the TFLite section's metadata doesn't).
    pub(crate) fn load_at_with_spm(
        path: &Path,
        base: usize,
        spm_override: Option<&[u8]>,
    ) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let fb = FlatBuffer::new(&mmap[base..], Some(b"TFL3"))?;
        let root = fb.root();

        // Buffers: { data(0) inline | offset(1), size(2) } — rebased to
        // absolute file offsets.
        let buffer_tables = fb.table_vector(root, 4)?;
        let mut spans = Vec::with_capacity(buffer_tables.len());
        for &b in &buffer_tables {
            let offset = fb.scalar_u64(b, 1).unwrap_or(0) as usize;
            let size = fb.scalar_u64(b, 2).unwrap_or(0) as usize;
            spans.push(BufferSpan { offset: base + offset, size });
        }

        // Subgraph 0 tensors.
        let subgraphs = fb.table_vector(root, 2)?;
        let sg = subgraphs
            .first()
            .ok_or_else(|| bad("no subgraphs"))?;
        let tensor_tables = fb.table_vector(*sg, 0)?;
        let mut raws = Vec::with_capacity(tensor_tables.len());
        for &t in &tensor_tables {
            let name = fb.string(t, 3)?.unwrap_or_default();
            let (sp, sn) = fb.vector(t, 0)?;
            let shape = match sp {
                Some(p) => fb
                    .i32_slice(p, sn)?
                    .into_iter()
                    .map(|d| d as usize)
                    .collect(),
                None => Vec::new(),
            };
            let dtype = fb.scalar_u8(t, 1).unwrap_or(TFLITE_F32);
            let buffer = fb.scalar_u32(t, 2).unwrap_or(0) as usize;
            raws.push(RawTensor { name, shape, dtype, buffer });
        }

        // Metadata entries: name → buffer index.
        let mut meta_buffers: HashMap<String, usize> = HashMap::new();
        for &m in &fb.table_vector(root, 6)? {
            if let Some(name) = fb.string(m, 0)? {
                let idx = fb.scalar_u32(m, 1).unwrap_or(0) as usize;
                meta_buffers.insert(name, idx);
            }
        }

        // Config proto → ModelMetadata.
        let params = {
            let idx = meta_buffers
                .get("odml.infra.proto.LlmParameters")
                .ok_or_else(|| bad("no LlmParameters metadata"))?;
            let bytes = read_span(&mmap, spans[*idx])?;
            parse_llm_parameters(bytes)?
        };
        let metadata = metadata_from_params(&params)?;

        // Tokenizer: spm blob → cached tokenizer.json (the U0 block).
        // Source: container section override, else the TFLite metadata.
        let spm_bytes: Vec<u8> = match spm_override {
            Some(b) => b.to_vec(),
            None => {
                let spm_idx = meta_buffers
                    .get("spm_vocab_model")
                    .ok_or_else(|| bad("no spm_vocab_model metadata"))?;
                read_span(&mmap, spans[*spm_idx])?.to_vec()
            }
        };
        let spm_path = path.with_extension("extracted.spm.model");
        if !spm_path.exists() {
            std::fs::write(&spm_path, &spm_bytes)?;
        }
        let tokenizer_json = crate::spm::ensure_tokenizer_json_from_spm(&spm_path)?;
        let added_tokens = crate::spm::spm_added_tokens(&spm_path)?;

        // Name mapping + quantization companions.
        let mut tensors: HashMap<String, (usize, Option<QuantInfo>)> = HashMap::new();
        for (i, raw) in raws.iter().enumerate() {
            let Some(hf) = map_name(&raw.name) else { continue };
            let quant = if raw.dtype == TFLITE_INT8 {
                let scale_name = format!("{}_quantized_scale", raw.name);
                let scale_idx = raws
                    .iter()
                    .position(|r| r.name == scale_name)
                    .ok_or_else(|| bad(format!("missing scales for {}", raw.name)))?;
                Some(QuantInfo {
                    scale_buffer: raws[scale_idx].buffer,
                    rows: raws[scale_idx].shape.first().copied().unwrap_or(0),
                })
            } else {
                None
            };
            tensors.insert(hf, (i, quant));
        }

        Ok(TfliteSource {
            _mmap: mmap,
            spans,
            tensors,
            raws,
            metadata,
            tokenizer_json,
            added_tokens,
        })
    }
}

fn read_span(mmap: &Mmap, span: BufferSpan) -> Result<&[u8]> {
    mmap.get(span.offset..span.offset + span.size)
        .ok_or_else(|| bad(format!("buffer out of bounds @{}+{}", span.offset, span.size)))
}

fn metadata_from_params(p: &LlmParameters) -> Result<ModelMetadata> {
    if p.hidden_size == 0 || p.num_hidden_layers == 0 || p.vocab_size == 0 {
        return Err(FormatError::MissingField(format!(
            "LlmParameters incomplete: {p:?}"
        )));
    }
    Ok(ModelMetadata {
        architecture: "gemma3_text".to_string(),
        hidden_size: p.hidden_size,
        intermediate_size: p.intermediate_size,
        num_hidden_layers: p.num_hidden_layers,
        num_attention_heads: p.num_attention_heads,
        num_key_value_heads: p.num_key_value_heads,
        vocab_size: p.vocab_size,
        max_position_embeddings: 32768,
        rms_norm_eps: 1e-6,
        rope_theta: if p.rope_theta > 0.0 { p.rope_theta } else { 1_000_000.0 },
        tie_word_embeddings: true, // ODML exports carry a single embedding table
        head_dim: p.head_dim,
        attention_bias: false,
        bos_token_id: p.bos_token_id,
        eos_token_ids: p.eos_token_ids.clone(),
        vision: None,
        attention_pattern: AttentionPattern {
            sliding_window: p.sliding_window.or(Some(512)),
            pattern: if p.sliding_window_pattern > 0 { p.sliding_window_pattern } else { 6 },
            rope_local_theta: if p.rope_local_theta > 0.0 { p.rope_local_theta } else { 10_000.0 },
            query_pre_attn_scalar: p.query_pre_attn_scalar,
        },
    })
}

impl ModelSource for TfliteSource {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn open_tensor(&self, name: &str) -> Result<TensorReader<'_>> {
        let (idx, quant) = self
            .tensors
            .get(name)
            .ok_or_else(|| FormatError::TensorNotFound(name.to_string()))?;
        let raw = &self.raws[*idx];
        let data = read_span(&self._mmap, self.spans[raw.buffer])?;
        let n: usize = raw.shape.iter().product();
        // ODML exports flatten weights to 1-D; the logical [in, out] shape
        // is rebuilt from the output dim (scale-companion length or the
        // architecture config).
        let out_dim = match quant {
            Some(q) => q.rows,
            None => logical_out_dim(&self.metadata, name, n)?,
        };
        let logical = vec![n / out_dim, out_dim];

        match raw.dtype {
            TFLITE_F32 => {
                let values: Vec<f32> = data
                    .chunks_exact(4)
                    .take(n)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let (values, shape) = reorient(&self.metadata, name, values, &logical)?;
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                Ok(TensorReader::owned(name.to_string(), shape, bytes))
            }
            TFLITE_INT8 => {
                let q = quant
                    .as_ref()
                    .ok_or_else(|| bad(format!("int8 tensor {} without scales", raw.name)))?;
                let scales_raw = read_span(&self._mmap, self.spans[q.scale_buffer])?;
                let scales: Vec<f32> = scales_raw
                    .chunks_exact(4)
                    .take(q.rows)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let values = dequant_i8(data, &scales, &logical)?;
                let (values, shape) = reorient(&self.metadata, name, values, &logical)?;
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                Ok(TensorReader::owned(name.to_string(), shape, bytes))
            }
            other => Err(FormatError::UnsupportedDtype {
                tensor: raw.name.clone(),
                dtype: format!("tflite type {other} (int4 packed: follow-up)"),
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

    fn sampler_defaults(&self) -> Option<crate::SamplerConfig> {
        None
    }
}

/// Output dimension for a mapped HF weight, from the architecture config
/// (used when no scale companion exists, e.g. f32 exports).
fn logical_out_dim(m: &ModelMetadata, hf_name: &str, n: usize) -> Result<usize> {
    if hf_name.ends_with("_layernorm.weight")
        || hf_name.ends_with("_norm.weight")
        || hf_name == "model.norm.weight"
    {
        return Ok(n); // 1-D: shape is [hidden] already
    }
    if hf_name == "model.embed_tokens.weight" || hf_name == "lm_head.weight" {
        return Ok(m.hidden_size); // [vocab, hidden] — no transpose below
    }
    Ok(if hf_name.ends_with("q_proj.weight") {
        m.num_attention_heads * m.head_dim
    } else if hf_name.ends_with("k_proj.weight") || hf_name.ends_with("v_proj.weight") {
        m.num_key_value_heads * m.head_dim
    } else if hf_name.ends_with("o_proj.weight") {
        m.hidden_size
    } else if hf_name.ends_with("gate_proj.weight") || hf_name.ends_with("up_proj.weight") {
        m.intermediate_size
    } else if hf_name.ends_with("down_proj.weight") {
        m.hidden_size
    } else {
        return Err(bad(format!("no logical shape rule for {hf_name}")));
    })
}

/// Reorients a logical [in, out] weight to HF [out, in] (transpose).
/// Embeddings and 1-D tensors pass through.
fn reorient(
    metadata: &ModelMetadata,
    hf_name: &str,
    values: Vec<f32>,
    logical: &[usize],
) -> Result<(Vec<f32>, Vec<usize>)> {
    if hf_name == "model.embed_tokens.weight" || hf_name == "lm_head.weight" {
        return Ok((values, logical.to_vec()));
    }
    if logical[0] == 1 || logical[1] == 1 {
        // 1-D tensors (norms): keep them 1-D, not [1, hidden].
        let shape = if logical[0] == 1 { vec![logical[1]] } else { logical.to_vec() };
        return Ok((values, shape));
    }
    if logical[0] == logical[1] && logical[0] == metadata.hidden_size && hf_name.ends_with("norm.weight") {
        return Ok((values, logical.to_vec()));
    }
    let (rows, cols) = (logical[0], logical[1]);
    let mut out = vec![0f32; values.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = values[r * cols + c];
        }
    }
    Ok((out, vec![cols, rows]))
}

/// int8 per-output-row dequant: `w[i,j] = q[i,j] * scale[j]` on the ODML
/// [in, out] layout (scales are per output channel = last dim).
fn dequant_i8(data: &[u8], scales: &[f32], logical: &[usize]) -> Result<Vec<f32>> {
    let (rows, cols) = (logical[0], logical[1]);
    if scales.len() != cols {
        return Err(bad(format!(
            "scale count {} != out dim {cols}",
            scales.len()
        )));
    }
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let q = data[r * cols + c] as i8;
            out.push(q as f32 * scales[c]);
        }
    }
    Ok(out)
}
