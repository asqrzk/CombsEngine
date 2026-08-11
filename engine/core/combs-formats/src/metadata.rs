//! Model metadata parsed from HuggingFace `config.json` (+ `generation_config.json`).

use crate::{FormatError, Result};

/// Architecture + hyperparameter description of a model, format-agnostic.
///
/// Field names follow the HuggingFace Llama config convention; other
/// architecture families remap their config onto this struct in their adapter.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Architecture identifier, e.g. `"llama"`, `"smollm2"` (from
    /// `config.json::model_type`). The model registry keys on this string.
    pub architecture: String,
    /// Hidden size (model dimension).
    pub hidden_size: usize,
    /// MLP intermediate size.
    pub intermediate_size: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Number of attention query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (GQA). Equal to `num_attention_heads` for MHA.
    pub num_key_value_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum positional embeddings the model was built for.
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f64,
    /// RoPE base frequency (theta).
    pub rope_theta: f64,
    /// Whether lm_head is tied to the embedding matrix.
    pub tie_word_embeddings: bool,
    /// Per-head dimension, derived: `hidden_size / num_attention_heads`.
    pub head_dim: usize,
    /// Whether attention projections carry biases.
    pub attention_bias: bool,
    /// Beginning-of-sequence token id, if defined.
    pub bos_token_id: Option<u32>,
    /// End-of-sequence token ids (merged from config + generation_config).
    pub eos_token_ids: Vec<u32>,
    /// Vision-tower hyperparameters for multimodal models (Idefics3/SmolVLM
    /// today); `None` for text-only models.
    pub vision: Option<VisionConfig>,
    /// Layer-type attention pattern (Gemma sliding-window interleave);
    /// defaults to all-global (Llama-family behavior).
    pub attention_pattern: AttentionPattern,
    /// MLP activation (`hidden_act` / `hidden_activation`).
    pub activation: Activation,
    /// RoPE frequency scaling (`rope_scaling`); `None` variant when absent.
    pub rope_scaling: RopeScaling,
}

/// MLP activation function, parsed from `hidden_act`/`hidden_activation`.
/// The tanh-approximation family (`gelu_pytorch_tanh`, `gelu_new`,
/// `gelu_fast`) all map to [`Activation::GeluTanh`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Activation {
    #[default]
    Silu,
    GeluTanh,
    Gelu,
}

impl Activation {
    fn parse(name: Option<&str>) -> Self {
        match name {
            Some("gelu_pytorch_tanh" | "gelu_new" | "gelu_fast") => Activation::GeluTanh,
            Some("gelu") => Activation::Gelu,
            // silu/swish and anything unknown: the llama-family default.
            _ => Activation::Silu,
        }
    }
}

/// RoPE frequency scaling, parsed from HF `rope_scaling` (accepts both the
/// modern `rope_type` and the legacy `type` key). Table math lives in
/// `combs-models::rope`; this is parse-only.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum RopeScaling {
    #[default]
    None,
    Linear {
        factor: f64,
    },
    /// Llama-3.1+ piecewise frequency scaling.
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position_embeddings: usize,
    },
    /// YaRN NTK-by-parts interpolation (+ attention temperature).
    Yarn {
        factor: f64,
        original_max_position_embeddings: usize,
        beta_fast: f64,
        beta_slow: f64,
        /// Explicit attention scaling; `None` = the YaRN default
        /// `0.1·ln(factor) + 1`.
        attention_factor: Option<f64>,
    },
    /// Phi-3 LongRoPE: per-dimension frequency divisors, one set for
    /// prompts within the pretraining context and one beyond it.
    LongRope {
        short_factor: Vec<f64>,
        long_factor: Vec<f64>,
        original_max_position_embeddings: usize,
        /// Context extension ratio `max_position / original_max` (phi
        /// derives the attention temperature from it, not from a config
        /// `factor` key).
        factor: f64,
        /// Explicit attention scaling; `None` = the LongRoPE default
        /// `sqrt(1 + ln(factor)/ln(original_max))` (1.0 when factor ≤ 1).
        attention_factor: Option<f64>,
    },
}

impl RopeScaling {
    fn parse(config: &serde_json::Value) -> Result<Self> {
        let Some(rs) = config.get("rope_scaling").filter(|v| !v.is_null()) else {
            return Ok(RopeScaling::None);
        };
        let kind = rs
            .get("rope_type")
            .or_else(|| rs.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        let f = |key: &str, default: f64| rs.get(key).and_then(|v| v.as_f64()).unwrap_or(default);
        let factor = f("factor", 1.0);
        match kind {
            "default" => Ok(RopeScaling::None),
            "linear" => Ok(RopeScaling::Linear { factor }),
            "llama3" => Ok(RopeScaling::Llama3 {
                factor,
                low_freq_factor: f("low_freq_factor", 1.0),
                high_freq_factor: f("high_freq_factor", 4.0),
                original_max_position_embeddings: rs
                    .get("original_max_position_embeddings")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(8192) as usize,
            }),
            "yarn" => Ok(RopeScaling::Yarn {
                factor,
                original_max_position_embeddings: rs
                    .get("original_max_position_embeddings")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(32768) as usize,
                beta_fast: f("beta_fast", 32.0),
                beta_slow: f("beta_slow", 1.0),
                attention_factor: rs.get("attention_factor").and_then(|v| v.as_f64()),
            }),
            "longrope" => {
                let factors = |key: &str| -> Result<Vec<f64>> {
                    rs.get(key)
                        .and_then(|v| v.as_array())
                        .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                        .ok_or_else(|| {
                            FormatError::MissingField(format!("rope_scaling.{key}"))
                        })
                };
                // Phi-3 keeps the pretraining context length top-level (not
                // inside rope_scaling), with max_position_embeddings already
                // raised to the extended value; their ratio drives the
                // attention temperature.
                let original = rs
                    .get("original_max_position_embeddings")
                    .or_else(|| config.get("original_max_position_embeddings"))
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| {
                        FormatError::MissingField(
                            "original_max_position_embeddings (longrope)".to_string(),
                        )
                    })? as usize;
                let max_pos = config
                    .get("max_position_embeddings")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(original as u64) as usize;
                Ok(RopeScaling::LongRope {
                    short_factor: factors("short_factor")?,
                    long_factor: factors("long_factor")?,
                    original_max_position_embeddings: original,
                    factor: max_pos as f64 / original as f64,
                    attention_factor: rs.get("attention_factor").and_then(|v| v.as_f64()),
                })
            }
            other => Err(FormatError::MissingField(format!(
                "unsupported rope_scaling type {other:?} (supported: linear, llama3, yarn, longrope)"
            ))),
        }
    }
}

/// Vision-encoder hyperparameters parsed from `config.json::vision_config`
/// (plus top-level `scale_factor` / `image_token_id`).
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Square input image size (pixels).
    pub image_size: usize,
    /// Patch size (pixels) of the patch embedding.
    pub patch_size: usize,
    /// Vision hidden size.
    pub hidden_size: usize,
    /// Vision MLP intermediate size.
    pub intermediate_size: usize,
    /// Vision transformer layers.
    pub num_hidden_layers: usize,
    /// Vision attention heads (MHA — kv heads == q heads).
    pub num_attention_heads: usize,
    /// LayerNorm epsilon (SigLIP: 1e-6).
    pub layer_norm_eps: f64,
    /// Pixel-shuffle scale factor of the connector (scale² patches are
    /// merged into one visual token).
    pub scale_factor: usize,
    /// Token id whose span in the prompt is replaced by visual embeddings.
    pub image_token_id: u32,
}

/// Attention RoPE/scale settings that vary by layer type (Gemma2/3):
/// `pattern`-th layers are "global" (full attention, `rope_theta`); the
/// rest are "local" (sliding-window attention, `rope_local_theta`).
#[derive(Debug, Clone)]
pub struct AttentionPattern {
    /// Sliding-window span for local layers; `None` = all layers global.
    pub sliding_window: Option<usize>,
    /// Every Nth layer is global (HF `sliding_window_pattern`, default 6).
    pub pattern: usize,
    /// RoPE base frequency for local layers (`rope_local_base_freq`).
    pub rope_local_theta: f64,
    /// Attention logit scale divisor (`query_pre_attn_scalar`); when
    /// `None`, the scale is `1/sqrt(head_dim)`.
    pub query_pre_attn_scalar: Option<f64>,
    /// Qwen2-style partition, stored raw: the first N layers are global and
    /// layers >= N slide — the inverse of `pattern`'s every-Nth-global.
    /// `ArchSpec::resolve` turns it into the per-layer layout. All shipped
    /// qwen2.5 checkpoints disable sliding anyway (`use_sliding_window:
    /// false` nulls `sliding_window` at parse).
    pub max_window_layers: Option<usize>,
}

impl Default for AttentionPattern {
    fn default() -> Self {
        AttentionPattern {
            sliding_window: None,
            pattern: 6,
            rope_local_theta: 10000.0,
            query_pre_attn_scalar: None,
            max_window_layers: None,
        }
    }
}

impl AttentionPattern {
    /// Whether layer `i` uses global attention (vs sliding-window local).
    pub fn is_global_layer(&self, i: usize) -> bool {
        self.sliding_window.is_none() || (i + 1) % self.pattern == 0
    }
}

impl VisionConfig {
    /// Visual tokens per image:
    /// `(image_size / patch_size)² / scale_factor²`.
    pub fn image_seq_len(&self) -> usize {
        let per_side = self.image_size / self.patch_size;
        (per_side * per_side) / (self.scale_factor * self.scale_factor)
    }

    /// Vision per-head dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Parses the vision section of a multimodal `config.json` (returns
    /// `None` when no `vision_config` object is present).
    fn from_hf_config(config: &serde_json::Value) -> Result<Option<Self>> {
        let Some(v) = config.get("vision_config").filter(|v| v.is_object()) else {
            return Ok(None);
        };
        let get = |key: &str| v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize);
        let image_token_id = config
            .get("image_token_id")
            .and_then(|x| x.as_u64())
            .ok_or_else(|| FormatError::MissingField("image_token_id".to_string()))?
            as u32;
        Ok(Some(VisionConfig {
            image_size: get("image_size").unwrap_or(512),
            patch_size: get("patch_size")
                .ok_or_else(|| FormatError::MissingField("vision_config.patch_size".to_string()))?,
            hidden_size: get("hidden_size")
                .ok_or_else(|| FormatError::MissingField("vision_config.hidden_size".to_string()))?,
            intermediate_size: get("intermediate_size")
                .ok_or_else(|| FormatError::MissingField("vision_config.intermediate_size".to_string()))?,
            num_hidden_layers: get("num_hidden_layers")
                .ok_or_else(|| FormatError::MissingField("vision_config.num_hidden_layers".to_string()))?,
            num_attention_heads: get("num_attention_heads")
                .ok_or_else(|| FormatError::MissingField("vision_config.num_attention_heads".to_string()))?,
            layer_norm_eps: v
                .get("layer_norm_eps")
                .and_then(|x| x.as_f64())
                .unwrap_or(1e-12),
            scale_factor: config
                .get("scale_factor")
                .and_then(|x| x.as_u64())
                .unwrap_or(2) as usize,
            image_token_id,
        }))
    }
}

fn get_u64(v: &serde_json::Value, key: &str) -> Result<u64> {
    v.get(key)
        .and_then(|x| x.as_u64())
        .ok_or_else(|| FormatError::MissingField(key.to_string()))
}

fn get_f64(v: &serde_json::Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
}

/// Extracts token ids from a config value that may be a single id or an array.
fn token_ids(v: Option<&serde_json::Value>) -> Vec<u32> {
    match v {
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|x| x.as_u64().map(|n| n as u32))
            .collect(),
        Some(x) => x.as_u64().map(|n| vec![n as u32]).unwrap_or_default(),
        None => Vec::new(),
    }
}

impl ModelMetadata {
    /// Minimal placeholder metadata for diffusion components that do not
    /// carry a language-model `config.json` (UNet, VAE, etc.).
    pub fn diffusion_placeholder(architecture: &str) -> Self {
        Self {
            architecture: architecture.to_string(),
            hidden_size: 0,
            intermediate_size: 0,
            num_hidden_layers: 0,
            num_attention_heads: 0,
            num_key_value_heads: 0,
            vocab_size: 0,
            max_position_embeddings: 0,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            tie_word_embeddings: false,
            head_dim: 0,
            attention_bias: false,
            bos_token_id: None,
            eos_token_ids: Vec::new(),
            vision: None,
            attention_pattern: AttentionPattern::default(),
            activation: Activation::default(),
            rope_scaling: RopeScaling::default(),
        }
    }

    /// Parses metadata from a HuggingFace `config.json` value, optionally
    /// merged with a `generation_config.json` value (which can override/add
    /// bos/eos ids).
    pub fn from_hf_config(
        config: &serde_json::Value,
        generation_config: Option<&serde_json::Value>,
    ) -> Result<Self> {
        // Multimodal configs (Idefics3/SmolVLM) nest the text hyperparameters
        // under `text_config`; the architecture id stays at the root.
        let text = config.get("text_config").unwrap_or(config);
        let hidden_size = get_u64(text, "hidden_size")? as usize;
        let num_attention_heads = get_u64(text, "num_attention_heads")? as usize;
        let num_key_value_heads = text
            .get("num_key_value_heads")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(num_attention_heads);

        let mut eos_token_ids = token_ids(text.get("eos_token_id"));
        if let Some(gc) = generation_config {
            for id in token_ids(gc.get("eos_token_id")) {
                if !eos_token_ids.contains(&id) {
                    eos_token_ids.push(id);
                }
            }
        }

        let bos_token_id = generation_config
            .and_then(|gc| gc.get("bos_token_id"))
            .and_then(|x| x.as_u64())
            .map(|x| x as u32)
            .or_else(|| {
                text.get("bos_token_id")
                    .and_then(|x| x.as_u64())
                    .map(|x| x as u32)
            });

        let architecture = config
            .get("model_type")
            .and_then(|x| x.as_str())
            .ok_or_else(|| FormatError::MissingField("model_type".to_string()))?
            .to_string();
        // HF config classes carry per-family defaults the JSON omits:
        // gemma3 ties word embeddings unless stated otherwise.
        let tie_default = matches!(architecture.as_str(), "gemma3" | "gemma3_text");

        if hidden_size % num_attention_heads != 0 {
            return Err(FormatError::MissingField(format!(
                "hidden_size ({hidden_size}) not divisible by num_attention_heads ({num_attention_heads})"
            )));
        }

        Ok(ModelMetadata {
            architecture,
            hidden_size,
            intermediate_size: get_u64(text, "intermediate_size")? as usize,
            num_hidden_layers: get_u64(text, "num_hidden_layers")? as usize,
            num_attention_heads,
            num_key_value_heads,
            vocab_size: get_u64(text, "vocab_size")? as usize,
            max_position_embeddings: text
                .get("max_position_embeddings")
                .and_then(|x| x.as_u64())
                .unwrap_or(2048) as usize,
            rms_norm_eps: get_f64(text, "rms_norm_eps", 1e-5),
            rope_theta: get_f64(text, "rope_theta", 10000.0),
            tie_word_embeddings: config
                .get("tie_word_embeddings")
                .or_else(|| text.get("tie_word_embeddings"))
                .and_then(|x| x.as_bool())
                .unwrap_or(tie_default),
            head_dim: text
                .get("head_dim")
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(hidden_size / num_attention_heads),
            attention_bias: text
                .get("attention_bias")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            bos_token_id,
            eos_token_ids,
            vision: VisionConfig::from_hf_config(config)?,
            attention_pattern: AttentionPattern {
                // Qwen2-family configs carry `sliding_window` even when
                // sliding is off (`use_sliding_window: false`); an explicit
                // false must null the window or the layer-pattern math would
                // invent gemma-style local layers.
                sliding_window: text
                    .get("sliding_window")
                    .and_then(|x| x.as_u64())
                    .map(|x| x as usize)
                    .filter(|_| {
                        text.get("use_sliding_window").and_then(|x| x.as_bool())
                            != Some(false)
                    }),
                pattern: get_u64(text, "sliding_window_pattern").unwrap_or(6) as usize,
                rope_local_theta: get_f64(text, "rope_local_base_freq", 10000.0),
                query_pre_attn_scalar: text
                    .get("query_pre_attn_scalar")
                    .and_then(|x| x.as_f64()),
                max_window_layers: text
                    .get("max_window_layers")
                    .and_then(|x| x.as_u64())
                    .map(|x| x as usize),
            },
            activation: Activation::parse(
                text.get("hidden_act")
                    .or_else(|| text.get("hidden_activation"))
                    .and_then(|x| x.as_str()),
            ),
            rope_scaling: RopeScaling::parse(text)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_smollm2_style_config() {
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 576,
            "intermediate_size": 1536,
            "num_hidden_layers": 30,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "vocab_size": 49152,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-5,
            "rope_theta": 100000,
            "tie_word_embeddings": true,
            "eos_token_id": 0,
            "bos_token_id": 0
        });
        let meta = ModelMetadata::from_hf_config(&config, None).unwrap();
        assert_eq!(meta.architecture, "llama");
        assert_eq!(meta.head_dim, 64);
        assert_eq!(meta.num_key_value_heads, 3);
        assert_eq!(meta.rope_theta, 100000.0);
        assert!(meta.tie_word_embeddings);
        assert_eq!(meta.eos_token_ids, vec![0]);
    }

    #[test]
    fn merges_generation_config_eos_array() {
        let config = serde_json::json!({
            "model_type": "llama", "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 1, "num_attention_heads": 2, "vocab_size": 10,
            "eos_token_id": 1
        });
        let gen = serde_json::json!({ "eos_token_id": [1, 2] });
        let meta = ModelMetadata::from_hf_config(&config, Some(&gen)).unwrap();
        assert_eq!(meta.eos_token_ids, vec![1, 2]);
        // GQA default: kv heads == q heads.
        assert_eq!(meta.num_key_value_heads, 2);
    }

    #[test]
    fn qwen2_use_sliding_window_false_nulls_the_window() {
        let base = serde_json::json!({
            "model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 2, "num_attention_heads": 2, "vocab_size": 10,
            "sliding_window": 131072, "use_sliding_window": false,
            "max_window_layers": 28
        });
        let meta = ModelMetadata::from_hf_config(&base, None).unwrap();
        assert_eq!(meta.attention_pattern.sliding_window, None);
        assert_eq!(meta.attention_pattern.max_window_layers, Some(28));

        // Explicitly enabled (rare long-context qwen) keeps the window, so
        // the registry guard can reject it loudly instead of running wrong.
        let mut on = base.clone();
        on["use_sliding_window"] = serde_json::json!(true);
        let meta = ModelMetadata::from_hf_config(&on, None).unwrap();
        assert_eq!(meta.attention_pattern.sliding_window, Some(131072));

        // Absent key (gemma/mistral style) keeps the window too.
        let mut absent = base.clone();
        absent.as_object_mut().unwrap().remove("use_sliding_window");
        let meta = ModelMetadata::from_hf_config(&absent, None).unwrap();
        assert_eq!(meta.attention_pattern.sliding_window, Some(131072));
    }

    #[test]
    fn gemma3_defaults_to_tied_embeddings() {
        // Gemma3 config.json omits tie_word_embeddings entirely; the HF
        // config class default (true) must apply — the checkpoints ship no
        // lm_head.weight.
        let config = serde_json::json!({
            "model_type": "gemma3_text", "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 1, "num_attention_heads": 2, "vocab_size": 10
        });
        let meta = ModelMetadata::from_hf_config(&config, None).unwrap();
        assert!(meta.tie_word_embeddings);

        // An explicit false still wins (hypothetical untied variant)…
        let mut untied = config.clone();
        untied["tie_word_embeddings"] = serde_json::json!(false);
        let meta = ModelMetadata::from_hf_config(&untied, None).unwrap();
        assert!(!meta.tie_word_embeddings);

        // …and non-gemma architectures keep the false default.
        let mut llama = config.clone();
        llama["model_type"] = serde_json::json!("llama");
        let meta = ModelMetadata::from_hf_config(&llama, None).unwrap();
        assert!(!meta.tie_word_embeddings);
    }

    #[test]
    fn parses_phi3_longrope_with_toplevel_original_max() {
        // Phi-3-mini-128k shape: `original_max_position_embeddings` lives at
        // the TOP level (not inside rope_scaling), max_position already
        // extended; factor derives from the ratio.
        let config = serde_json::json!({
            "model_type": "phi3", "hidden_size": 3072, "intermediate_size": 8192,
            "num_hidden_layers": 32, "num_attention_heads": 32, "vocab_size": 32064,
            "max_position_embeddings": 131072,
            "original_max_position_embeddings": 4096,
            "rope_scaling": {
                "type": "longrope",
                "short_factor": [1.0, 1.05, 1.1],
                "long_factor": [2.0, 2.5, 3.0]
            }
        });
        let meta = ModelMetadata::from_hf_config(&config, None).unwrap();
        match &meta.rope_scaling {
            RopeScaling::LongRope {
                short_factor,
                long_factor,
                original_max_position_embeddings,
                factor,
                attention_factor,
            } => {
                assert_eq!(short_factor, &[1.0, 1.05, 1.1]);
                assert_eq!(long_factor, &[2.0, 2.5, 3.0]);
                assert_eq!(*original_max_position_embeddings, 4096);
                assert_eq!(*factor, 32.0);
                assert_eq!(*attention_factor, None);
            }
            other => panic!("expected LongRope, got {other:?}"),
        }
    }

    #[test]
    fn parses_nested_idefics3_config() {
        let config = serde_json::json!({
            "model_type": "idefics3",
            "image_token_id": 49190,
            "scale_factor": 4,
            "tie_word_embeddings": false,
            "text_config": {
                "hidden_size": 576,
                "intermediate_size": 1536,
                "num_hidden_layers": 30,
                "num_attention_heads": 9,
                "num_key_value_heads": 3,
                "vocab_size": 49280,
                "max_position_embeddings": 8192,
                "rms_norm_eps": 1e-5,
                "rope_theta": 100000,
                "eos_token_id": 2
            },
            "vision_config": {
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "image_size": 512,
                "patch_size": 16,
                "layer_norm_eps": 1e-6
            }
        });
        let meta = ModelMetadata::from_hf_config(&config, None).unwrap();
        assert_eq!(meta.architecture, "idefics3");
        assert_eq!(meta.hidden_size, 576);
        assert_eq!(meta.eos_token_ids, vec![2]);
        let v = meta.vision.expect("vision config parsed");
        assert_eq!(v.hidden_size, 768);
        assert_eq!(v.image_token_id, 49190);
        assert_eq!(v.scale_factor, 4);
        assert_eq!(v.head_dim(), 64);
        // (512/16)² / 4² = 1024/16 = 64 visual tokens per image.
        assert_eq!(v.image_seq_len(), 64);
    }
}
