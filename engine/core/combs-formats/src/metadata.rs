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
    /// Gated DeltaNet (linear attention) geometry for hybrid families —
    /// Qwen3.5 runs three DeltaNet layers to every full-attention layer;
    /// `None` for every pure-attention model.
    pub linear_attention: Option<LinearAttention>,
    /// Head dimensions RoPE turns when only part of each head rotates
    /// (Qwen3.5: 64 of 256); `None` = the whole head.
    pub rotary_dim: Option<usize>,
    /// A BIDIRECTIONAL encoder's own geometry (ModernBERT today); `None`
    /// for every generative model. Present means the seat decodes
    /// nothing: it reads a whole sequence at once and answers from it.
    pub encoder: Option<EncoderConfig>,
}

/// One hybrid family's Gated DeltaNet geometry (see
/// [`ModelMetadata::linear_attention`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearAttention {
    /// Key (and query) heads; each repeats across the value heads.
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    /// The causal conv1d kernel over the concatenated q|k|v stream.
    pub conv_kernel: usize,
    /// Every Nth layer (1-based) is full attention; the rest are DeltaNet.
    pub full_attention_interval: usize,
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

/// A bidirectional encoder's geometry, parsed from `config.json` when
/// `model_type == "modernbert"`.
///
/// It is read INSTEAD of the generative fields, never beside them, and
/// that is the whole reason it exists: a ModernBERT config carries a
/// `rms_norm_eps` of nothing and no `rope_theta` at all, so the two
/// knobs a llama-shaped reader would reach for are silently plausible
/// and both wrong. This encoder normalises with LayerNorm, not RMSNorm,
/// and turns TWO rope bases — one for the layers that attend to the
/// whole sequence, another for the layers that slide.
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    /// Every Nth layer attends globally; the rest slide. Only consulted
    /// when `layer_types` is empty.
    pub global_attn_every_n_layers: usize,
    /// Half of HF's `local_attention`: a sliding layer sees keys within
    /// ±this many positions. HF states the whole width; the kernel wants
    /// the radius, and halving it at the reader means nobody halves it
    /// twice.
    pub local_window: usize,
    /// RoPE base for the layers that attend to everything.
    pub global_rope_theta: f64,
    /// RoPE base for the layers that slide.
    pub local_rope_theta: f64,
    /// LayerNorm epsilon.
    pub norm_eps: f64,
    /// Whether the LayerNorms carry biases (ModernBERT-large: they do not).
    pub norm_bias: bool,
    /// Whether the attention projections carry biases.
    pub attention_bias: bool,
    /// Whether the MLP carries biases.
    pub mlp_bias: bool,
    /// `[CLS]`, `[SEP]` and `[PAD]`, by id. A marker sequence splices
    /// these by NUMBER and never by string — `[MASK]` is `lstrip: true`
    /// in the tokenizer, so writing the text would not round-trip.
    pub cls_id: u32,
    pub sep_id: u32,
    pub pad_id: u32,
    /// The explicit per-layer list when the config carries one (`true` =
    /// global); empty when only `global_attn_every_n_layers` is given.
    /// The list is the authority where both are present.
    pub layer_types: Vec<bool>,
}

impl EncoderConfig {
    /// Whether layer `i` attends to the whole sequence.
    pub fn is_global_layer(&self, i: usize) -> bool {
        if let Some(&global) = self.layer_types.get(i) {
            return global;
        }
        self.global_attn_every_n_layers <= 1 || i % self.global_attn_every_n_layers == 0
    }

    /// Parses the encoder section of a `config.json`; `None` for every
    /// architecture that is not one.
    fn from_hf_config(config: &serde_json::Value) -> Result<Option<Self>> {
        if config.get("model_type").and_then(|v| v.as_str()) != Some("modernbert") {
            return Ok(None);
        }
        let u = |key: &str| config.get(key).and_then(|x| x.as_u64());
        let flag = |key: &str, default: bool| config.get(key).and_then(|x| x.as_bool()).unwrap_or(default);
        // Two spellings in the wild. transformers <= 4.x writes the bases
        // flat; 5.x nests them under `rope_parameters` by layer type, and
        // the laya checkpoint's config is the nested one — a reader that
        // knew only the flat names would take BOTH defaults and turn every
        // layer at 10000.
        let theta = |flat: &str, nested: &str, default: f64| -> f64 {
            if let Some(v) = config.get(flat).and_then(|x| x.as_f64()) {
                return v;
            }
            config
                .get("rope_parameters")
                .and_then(|r| r.get(nested))
                .and_then(|r| r.get("rope_theta"))
                .and_then(|x| x.as_f64())
                .unwrap_or(default)
        };
        let layer_types: Vec<bool> = config
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|t| t.as_str() == Some("full_attention")).collect())
            .unwrap_or_default();
        let id = |key: &str| -> Result<u32> {
            u(key)
                .map(|x| x as u32)
                .ok_or_else(|| FormatError::MissingField(key.to_string()))
        };
        Ok(Some(EncoderConfig {
            global_attn_every_n_layers: u("global_attn_every_n_layers").unwrap_or(3) as usize,
            local_window: (u("local_attention").unwrap_or(128) / 2) as usize,
            global_rope_theta: theta("global_rope_theta", "full_attention", 160000.0),
            local_rope_theta: theta("local_rope_theta", "sliding_attention", 10000.0),
            norm_eps: config
                .get("norm_eps")
                .or_else(|| config.get("layer_norm_eps"))
                .and_then(|x| x.as_f64())
                .unwrap_or(1e-5),
            norm_bias: flag("norm_bias", false),
            attention_bias: flag("attention_bias", false),
            mlp_bias: flag("mlp_bias", false),
            cls_id: id("cls_token_id")?,
            sep_id: id("sep_token_id")?,
            pad_id: id("pad_token_id")?,
            layer_types,
        }))
    }
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
    /// Layers that hold a paged KV arena: the global-attention layers, and
    /// in a hybrid family only its attention ones — a DeltaNet layer keeps
    /// a recurrent state instead. Never zero, so byte estimates stay
    /// defined for a malformed file.
    pub fn kv_layers(&self) -> usize {
        let every = self
            .linear_attention
            .map(|la| la.full_attention_interval.max(1));
        (0..self.num_hidden_layers)
            .filter(|&i| self.attention_pattern.is_global_layer(i))
            .filter(|&i| every.is_none_or(|n| (i + 1) % n == 0))
            .count()
            .max(1)
    }

    /// Bytes one sequence's recurrent state holds across every DeltaNet
    /// layer — the state matrix and the conv ring, f32 on every build.
    /// Zero for every attention-only model.
    pub fn recurrent_state_bytes(&self) -> u64 {
        let Some(la) = self.linear_attention else {
            return 0;
        };
        let every = la.full_attention_interval.max(1);
        let delta_layers = (0..self.num_hidden_layers)
            .filter(|&i| (i + 1) % every != 0)
            .count() as u64;
        let state = (la.value_heads * la.key_head_dim * la.value_head_dim) as u64;
        let channels = (2 * la.key_heads * la.key_head_dim + la.value_heads * la.value_head_dim) as u64;
        let ring = la.conv_kernel.saturating_sub(1) as u64 * channels;
        delta_layers * (state + ring) * 4
    }

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
            linear_attention: None,
            rotary_dim: None,
            encoder: None,
        }
    }

    /// Parses metadata from a HuggingFace `config.json` value, optionally
    /// merged with a `generation_config.json` value (which can override/add
    /// bos/eos ids).
    pub fn from_hf_config(
        config: &serde_json::Value,
        generation_config: Option<&serde_json::Value>,
    ) -> Result<Self> {
        // Whisper names its dimensions differently (d_model, encoder_layers,
        // …). Remap them onto the shared keys and re-enter; the guard on
        // `hidden_size` keeps the second pass out of this branch. Whisper-only
        // geometry (mel bins, audio positions) is derived from tensor shapes
        // at model load, so nothing else needs a slot here.
        if config.get("model_type").and_then(|x| x.as_str()) == Some("whisper")
            && config.get("hidden_size").is_none()
        {
            let mut remapped = config.clone();
            let obj = remapped
                .as_object_mut()
                .ok_or_else(|| FormatError::MissingField("config object".to_string()))?;
            for (from, to) in [
                ("d_model", "hidden_size"),
                ("encoder_attention_heads", "num_attention_heads"),
                ("encoder_ffn_dim", "intermediate_size"),
                ("encoder_layers", "num_hidden_layers"),
                ("max_target_positions", "max_position_embeddings"),
            ] {
                if let Some(v) = config.get(from).cloned() {
                    obj.insert(to.to_string(), v);
                }
            }
            obj.insert("tie_word_embeddings".to_string(), serde_json::json!(true));
            return Self::from_hf_config(&remapped, generation_config);
        }
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
            linear_attention: None,
            rotary_dim: None,
            // Read from the WHOLE config, not the nested text section: a
            // ModernBERT config has no text section, and the special ids
            // this needs sit at the top level.
            encoder: EncoderConfig::from_hf_config(config)?,
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

#[cfg(test)]
mod hybrid_tests {
    use super::*;

    fn qwen35_4b() -> ModelMetadata {
        let mut m = ModelMetadata::diffusion_placeholder("qwen35");
        m.num_hidden_layers = 32;
        m.num_key_value_heads = 4;
        m.head_dim = 256;
        m.linear_attention = Some(LinearAttention {
            key_heads: 16,
            value_heads: 32,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel: 4,
            full_attention_interval: 4,
        });
        m
    }

    #[test]
    fn a_hybrid_model_holds_kv_only_on_its_attention_layers() {
        let m = qwen35_4b();
        assert_eq!(m.kv_layers(), 8, "one attention layer in four holds KV");
        // 24 DeltaNet layers × (32 × 128 × 128 state + 3 × 8,192 conv ring) × 4 B
        assert_eq!(m.recurrent_state_bytes(), 52_690_944);
    }

    #[test]
    fn attention_only_models_keep_their_count() {
        let mut m = ModelMetadata::diffusion_placeholder("llama");
        m.num_hidden_layers = 16;
        assert_eq!(m.kv_layers(), 16);
        assert_eq!(m.recurrent_state_bytes(), 0);
        m.num_hidden_layers = 12;
        m.attention_pattern = AttentionPattern {
            sliding_window: Some(512),
            pattern: 6,
            ..AttentionPattern::default()
        };
        assert_eq!(m.kv_layers(), 2, "gemma: every sixth layer is global");
    }

    /// The real ModernBERT-large config the laya checkpoint ships, not a
    /// hand-written stand-in — because the thing worth catching here is a
    /// key that moved, and a stand-in would move with the reader.
    #[test]
    fn a_modernbert_config_is_read_as_an_encoder_and_not_as_a_decoder() {
        let raw = include_str!("../tests/fixtures/modernbert_large_config.json");
        let cfg: serde_json::Value = serde_json::from_str(raw).expect("the fixture parses");
        let m = ModelMetadata::from_hf_config(&cfg, None).expect("a modernbert config");
        let e = m.encoder.as_ref().expect("model_type modernbert means an encoder");

        assert_eq!(e.global_attn_every_n_layers, 3);
        // HF states the whole window; the kernel wants the radius. Halved
        // once, here, so nobody halves it twice.
        assert_eq!(e.local_window, 64, "local_attention 128 is +/-64");
        // This config is transformers 5.x, which nests the bases under
        // `rope_parameters` by layer type. A reader that knew only the
        // flat `global_rope_theta`/`local_rope_theta` spelling would take
        // BOTH defaults and turn every layer at 10000.
        assert_eq!(e.global_rope_theta, 160000.0, "the global base is not the default");
        assert_eq!(e.local_rope_theta, 10000.0);
        assert_eq!(e.norm_eps, 1e-5);
        assert!(!e.norm_bias && !e.attention_bias && !e.mlp_bias, "ModernBERT-large carries no biases");
        assert_eq!((e.cls_id, e.sep_id, e.pad_id), (50281, 50282, 50283));

        // The explicit list is the authority, and it agrees with the
        // every-third rule on this checkpoint.
        assert_eq!(e.layer_types.len(), 28);
        for i in 0..28 {
            assert_eq!(e.is_global_layer(i), i % 3 == 0, "layer {i}");
        }

        // The knobs a llama-shaped reader would reach for are absent from
        // this config, so they are whatever the defaults say — plausible,
        // and both wrong. The encoder must be read through `encoder`.
        assert!(cfg.get("rope_theta").is_none(), "there is no single rope base to read");
        assert!(cfg.get("rms_norm_eps").is_none(), "this encoder normalises with LayerNorm");
    }

    /// Both spellings of the rope bases, because the flat one is what
    /// every ModernBERT config written before transformers 5 carries.
    #[test]
    fn the_flat_rope_spelling_is_read_too_and_wins_over_the_nested_one() {
        // The shared fields are still read first — an encoder config
        // carries hidden_size and its layer count like any other.
        let base = serde_json::json!({
            "model_type": "modernbert",
            "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
            "num_attention_heads": 4, "vocab_size": 32, "max_position_embeddings": 128,
            "cls_token_id": 1, "sep_token_id": 2, "pad_token_id": 3,
            "global_rope_theta": 160000.0, "local_rope_theta": 10000.0,
            "local_attention": 128,
        });
        let m = ModelMetadata::from_hf_config(&base, None).expect("flat spelling");
        let e = m.encoder.expect("an encoder");
        assert_eq!((e.global_rope_theta, e.local_rope_theta), (160000.0, 10000.0));
        assert!(e.layer_types.is_empty(), "no list given");
        assert!(e.is_global_layer(0) && !e.is_global_layer(1) && e.is_global_layer(3));
    }

    /// A generative config must not grow an encoder, or every llama would
    /// suddenly claim to decode nothing.
    #[test]
    fn a_generative_config_carries_no_encoder() {
        let cfg = serde_json::json!({
            "model_type": "llama", "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
            "vocab_size": 32, "max_position_embeddings": 128,
        });
        let m = ModelMetadata::from_hf_config(&cfg, None).expect("a llama config");
        assert!(m.encoder.is_none());
    }

    /// An encoder config missing a special id is refused BY NAME: a
    /// marker sequence splices these by number, and a guessed [SEP] would
    /// put the options in the wrong half of the sequence.
    #[test]
    fn an_encoder_config_without_its_special_ids_is_refused_by_name() {
        let cfg = serde_json::json!({
            "model_type": "modernbert",
            "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
            "num_attention_heads": 4, "vocab_size": 32, "max_position_embeddings": 128,
            "cls_token_id": 1, "sep_token_id": 2,
        });
        let err = ModelMetadata::from_hf_config(&cfg, None).unwrap_err();
        assert!(format!("{err}").contains("pad_token_id"), "{err}");
    }
}
