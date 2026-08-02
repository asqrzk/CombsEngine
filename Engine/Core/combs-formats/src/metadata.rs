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
                .unwrap_or(false),
            head_dim: hidden_size / num_attention_heads,
            attention_bias: text
                .get("attention_bias")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            bos_token_id,
            eos_token_ids,
            vision: VisionConfig::from_hf_config(config)?,
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
