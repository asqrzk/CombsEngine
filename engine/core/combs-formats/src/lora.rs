//! LoRA merge-at-load, shared by the diffusion and text pipelines.
//!
//! Reads a LoRA safetensors file in the three wild formats — diffusers
//! (`unet.<path>.lora.down.weight`, incl. the `lora_A`/`lora_B`
//! spelling), kohya (`lora_unet_<flat>.lora_down.weight` + per-module
//! `.alpha`), and PEFT text adapters
//! (`base_model.model.<path>.lora_A.weight`, alpha in the sibling
//! `adapter_config.json`) — and fuses
//! `W' = W + scale * (alpha/rank) * (up @ down)` on the host in f32
//! before the weights reach the device. Base tensors the file does not
//! touch stay mmap-served; merged tensors force the dense path even on
//! packed-quant sources. Unmatched or unsupported-shape modules are
//! skipped loudly, never silently.

use std::collections::HashMap;
use std::path::Path;

use crate::metadata::ModelMetadata;
use crate::source::{SamplerConfig, TensorDtype, TensorReader};
use crate::tokenizer::TokenizerSpec;
use crate::{FormatError, ModelSource, Result};

/// A LoRA file to fuse into the pipeline at load time.
#[derive(Debug, Clone)]
pub struct LoraSpec {
    pub path: std::path::PathBuf,
    /// User strength multiplier on top of the file's own alpha/rank.
    pub scale: f32,
}

#[derive(Default)]
pub struct LoraPair {
    /// (row-major values, shape) as stored, trailing 1x1 conv dims kept.
    down: Option<(Vec<f32>, Vec<usize>)>,
    up: Option<(Vec<f32>, Vec<usize>)>,
    alpha: Option<f32>,
}

/// Parsed LoRA file: target base-tensor name -> low-rank pair, per component.
pub struct LoraFile {
    pub unet: HashMap<String, LoraPair>,
    pub text: HashMap<String, LoraPair>,
    /// File keys that matched no known LoRA naming scheme.
    pub unrecognized: Vec<String>,
    /// File-level alpha from a sibling `adapter_config.json` (PEFT keeps
    /// one `lora_alpha` for the whole adapter instead of per-module keys).
    pub file_alpha: Option<f32>,
}

fn dtype_of(d: safetensors::Dtype) -> Result<TensorDtype> {
    match d {
        safetensors::Dtype::F32 => Ok(TensorDtype::F32),
        safetensors::Dtype::F16 => Ok(TensorDtype::F16),
        safetensors::Dtype::BF16 => Ok(TensorDtype::BF16),
        safetensors::Dtype::F64 => Ok(TensorDtype::F64),
        other => Err(FormatError::Safetensors(format!(
            "lora tensor dtype {other:?} not supported"
        ))),
    }
}

/// kohya flattens module paths with underscores; the diffusers names the
/// base checkpoint uses keep underscores only inside these fixed words.
fn kohya_to_dotted(flat: &str) -> String {
    let mut s = flat.replace('_', ".");
    for (broken, fixed) in [
        ("down.blocks", "down_blocks"),
        ("mid.block", "mid_block"),
        ("up.blocks", "up_blocks"),
        ("transformer.blocks", "transformer_blocks"),
        ("proj.in", "proj_in"),
        ("proj.out", "proj_out"),
        ("to.q", "to_q"),
        ("to.k", "to_k"),
        ("to.v", "to_v"),
        ("to.out", "to_out"),
        ("conv.shortcut", "conv_shortcut"),
        ("time.emb.proj", "time_emb_proj"),
        ("conv.in", "conv_in"),
        ("conv.out", "conv_out"),
        ("conv.norm.out", "conv_norm_out"),
        ("time.embedding", "time_embedding"),
        ("text.model", "text_model"),
        ("self.attn", "self_attn"),
        ("q.proj", "q_proj"),
        ("k.proj", "k_proj"),
        ("v.proj", "v_proj"),
        ("out.proj", "out_proj"),
        ("final.layer.norm", "final_layer_norm"),
    ] {
        s = s.replace(broken, fixed);
    }
    s
}

enum Part {
    Down,
    Up,
    Alpha,
}

/// Classify one file key into (component, target base tensor, part).
fn classify(key: &str) -> Option<(bool, String, Part)> {
    // kohya: lora_unet_<flat>.(lora_down.weight | lora_up.weight | alpha)
    for (prefix, is_unet) in [("lora_unet_", true), ("lora_te_", false), ("lora_te1_", false)] {
        if let Some(rest) = key.strip_prefix(prefix) {
            let (flat, part) = if let Some(f) = rest.strip_suffix(".lora_down.weight") {
                (f, Part::Down)
            } else if let Some(f) = rest.strip_suffix(".lora_up.weight") {
                (f, Part::Up)
            } else if let Some(f) = rest.strip_suffix(".alpha") {
                (f, Part::Alpha)
            } else {
                return None;
            };
            return Some((is_unet, format!("{}.weight", kohya_to_dotted(flat)), part));
        }
    }
    // PEFT text adapters: base_model.model.<path>.(lora_A|lora_B).weight
    if let Some(rest) = key.strip_prefix("base_model.model.") {
        let (path, part) = if let Some(p) = rest.strip_suffix(".lora_A.weight") {
            (p, Part::Down)
        } else if let Some(p) = rest.strip_suffix(".lora_B.weight") {
            (p, Part::Up)
        } else {
            return None;
        };
        return Some((false, format!("{path}.weight"), part));
    }
    // diffusers: <comp>.<path>.(lora.down|lora.up|lora_A|lora_B).weight
    for (prefix, is_unet) in [("unet.", true), ("text_encoder.", false)] {
        if let Some(rest) = key.strip_prefix(prefix) {
            let (path, part) = if let Some(p) = rest.strip_suffix(".lora.down.weight") {
                (p, Part::Down)
            } else if let Some(p) = rest.strip_suffix(".lora.up.weight") {
                (p, Part::Up)
            } else if let Some(p) = rest.strip_suffix(".lora_A.weight") {
                (p, Part::Down)
            } else if let Some(p) = rest.strip_suffix(".lora_B.weight") {
                (p, Part::Up)
            } else {
                return None;
            };
            // PEFT nests the pair under `.lora_A/.lora_B` directly on the
            // module; older diffusers uses `.lora.down/.lora.up`. Either
            // way the target is the module's `.weight`.
            return Some((is_unet, format!("{path}.weight"), part));
        }
    }
    None
}

impl LoraFile {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(FormatError::Io)?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| FormatError::Safetensors(e.to_string()))?;
        let mut unet: HashMap<String, LoraPair> = HashMap::new();
        let mut text: HashMap<String, LoraPair> = HashMap::new();
        let mut unrecognized = Vec::new();
        for (key, view) in st.tensors() {
            let Some((is_unet, target, part)) = classify(&key) else {
                unrecognized.push(key);
                continue;
            };
            let shape: Vec<usize> = view.shape().to_vec();
            let values = TensorReader::new(
                key.clone(),
                if shape.is_empty() { vec![1] } else { shape.clone() },
                dtype_of(view.dtype())?,
                view.data(),
            )
            .load_data()?
            .to_vec::<f32>()
            .map_err(|e| FormatError::Safetensors(format!("{key}: {e:?}")))?;
            let pair = if is_unet { &mut unet } else { &mut text }
                .entry(target)
                .or_default();
            match part {
                Part::Down => pair.down = Some((values, shape)),
                Part::Up => pair.up = Some((values, shape)),
                Part::Alpha => pair.alpha = values.first().copied(),
            }
        }
        let file_alpha = path
            .parent()
            .map(|d| d.join("adapter_config.json"))
            .filter(|p| p.is_file())
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|v| v.get("lora_alpha").and_then(|a| a.as_f64()))
            .map(|a| a as f32);
        Ok(Self { unet, text, unrecognized, file_alpha })
    }
}

/// A `ModelSource` that serves base tensors with LoRA deltas fused in.
pub struct LoraMergedSource<S: ModelSource> {
    base: S,
    merged: HashMap<String, (Vec<f32>, Vec<usize>)>,
    pub applied: usize,
    pub skipped: Vec<String>,
}

/// Fuse `pairs` into `base`. Targets missing from the base checkpoint or
/// with shapes the merge cannot express (e.g. 3x3 conv LoRA) are recorded
/// in `skipped` — the caller reports them.
pub fn merge_lora<S: ModelSource>(
    base: S,
    pairs: &HashMap<String, LoraPair>,
    scale: f32,
    default_alpha: Option<f32>,
) -> Result<LoraMergedSource<S>> {
    let names: std::collections::HashSet<String> = base.tensor_names().into_iter().collect();
    let mut merged = HashMap::new();
    let mut skipped = Vec::new();
    for (target, pair) in pairs {
        let (Some((down, down_shape)), Some((up, up_shape))) = (&pair.down, &pair.up) else {
            skipped.push(format!("{target} (incomplete pair)"));
            continue;
        };
        if !names.contains(target) {
            skipped.push(format!("{target} (no such base tensor)"));
            continue;
        }
        // Accept [r,in] / [out,r] linears and their [.,.,1,1] conv-1x1
        // forms; anything with a real spatial kernel is out of scope.
        let flat_ok = |s: &[usize]| s.len() == 2 || (s.len() == 4 && s[2] == 1 && s[3] == 1);
        if !flat_ok(down_shape) || !flat_ok(up_shape) {
            skipped.push(format!("{target} (unsupported shapes {down_shape:?}/{up_shape:?})"));
            continue;
        }
        let (rank, in_dim) = (down_shape[0], down_shape[1]);
        let (out_dim, up_rank) = (up_shape[0], up_shape[1]);
        if rank != up_rank {
            skipped.push(format!("{target} (rank mismatch {rank} vs {up_rank})"));
            continue;
        }
        let reader = base.open_tensor(target)?;
        let base_shape = reader.shape().to_vec();
        let base_elems: usize = base_shape.iter().product();
        if base_elems != out_dim * in_dim {
            skipped.push(format!(
                "{target} (base shape {base_shape:?} vs lora {out_dim}x{in_dim})"
            ));
            continue;
        }
        let mut w = reader
            .load_data()?
            .to_vec::<f32>()
            .map_err(|e| FormatError::Safetensors(format!("{target}: {e:?}")))?;
        let alpha = pair.alpha.or(default_alpha).unwrap_or(rank as f32);
        let factor = scale * alpha / rank as f32;
        // w[o*in + i] += factor * sum_r up[o*rank + r] * down[r*in + i]
        for o in 0..out_dim {
            let w_row = &mut w[o * in_dim..(o + 1) * in_dim];
            for r in 0..rank {
                let coeff = factor * up[o * rank + r];
                if coeff == 0.0 {
                    continue;
                }
                let d_row = &down[r * in_dim..(r + 1) * in_dim];
                for (wi, di) in w_row.iter_mut().zip(d_row) {
                    *wi += coeff * di;
                }
            }
        }
        merged.insert(target.clone(), (w, base_shape));
    }
    let mut dense_fallback_bytes: u64 = 0;
    for (name, (w, _)) in &merged {
        if matches!(base.open_tensor_quant(name), Ok(Some(_))) {
            dense_fallback_bytes += (w.len() * 4) as u64;
        }
    }
    if dense_fallback_bytes > 0 {
        eprintln!(
            "lora: {} MB of packed-quant base tensors will run dense f32 (merged weights cannot stay packed)",
            dense_fallback_bytes / (1024 * 1024)
        );
    }
    Ok(LoraMergedSource { base, applied: merged.len(), merged, skipped })
}

impl<S: ModelSource> ModelSource for LoraMergedSource<S> {
    fn metadata(&self) -> &ModelMetadata {
        self.base.metadata()
    }

    fn tensor_names(&self) -> Vec<String> {
        self.base.tensor_names()
    }

    fn open_tensor(&self, name: &str) -> Result<TensorReader<'_>> {
        if let Some((values, shape)) = self.merged.get(name) {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            return Ok(TensorReader::owned_with_dtype(
                name.to_string(),
                shape.clone(),
                TensorDtype::F32,
                bytes,
            ));
        }
        self.base.open_tensor(name)
    }

    fn tokenizer(&self) -> Result<TokenizerSpec> {
        self.base.tokenizer()
    }

    fn sampler_defaults(&self) -> Option<SamplerConfig> {
        self.base.sampler_defaults()
    }

    fn open_tensor_quant(&self, name: &str) -> Result<Option<crate::QuantTensor<'_>>> {
        // A merged tensor no longer equals the packed bytes on disk — force
        // the dense path so the delta is never silently dropped.
        if self.merged.contains_key(name) {
            return Ok(None);
        }
        self.base.open_tensor_quant(name)
    }
}

#[cfg(test)]
mod tests {
    use super::{classify, kohya_to_dotted, Part};

    /// Real keys captured from the two verification files' headers.
    #[test]
    fn classifies_diffusers_and_kohya_keys() {
        let (u, t, p) = classify(
            "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora.down.weight",
        )
        .unwrap();
        assert!(u && matches!(p, Part::Down));
        assert_eq!(
            t,
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.weight"
        );

        let (u, t, p) =
            classify("lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight").unwrap();
        assert!(u && matches!(p, Part::Up));
        assert_eq!(t, "down_blocks.0.attentions.0.proj_in.weight");

        let (u, t, p) = classify(
            "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_ff_net_0_proj.alpha",
        )
        .unwrap();
        assert!(u && matches!(p, Part::Alpha));
        assert_eq!(
            t,
            "up_blocks.1.attentions.2.transformer_blocks.0.ff.net.0.proj.weight"
        );

        let (u, t, _) = classify(
            "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_down.weight",
        )
        .unwrap();
        assert!(u);
        assert_eq!(
            t,
            "mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.weight"
        );

        let (u, t, _) =
            classify("lora_te_text_model_encoder_layers_3_self_attn_q_proj.lora_down.weight")
                .unwrap();
        assert!(!u);
        assert_eq!(t, "text_model.encoder.layers.3.self_attn.q_proj.weight");

        let (u, t, p) = classify(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        )
        .unwrap();
        assert!(!u && matches!(p, Part::Down));
        assert_eq!(t, "model.layers.0.self_attn.q_proj.weight");

        assert!(classify("some_random_tensor").is_none());
    }

    #[test]
    fn kohya_flat_names_round_trip_the_fixed_vocabulary() {
        assert_eq!(
            kohya_to_dotted("down_blocks_0_attentions_1_transformer_blocks_0_ff_net_2"),
            "down_blocks.0.attentions.1.transformer_blocks.0.ff.net.2"
        );
        assert_eq!(kohya_to_dotted("mid_block_attentions_0_proj_out"), "mid_block.attentions.0.proj_out");
    }
}
