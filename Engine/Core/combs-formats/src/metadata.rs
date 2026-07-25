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
        let hidden_size = get_u64(config, "hidden_size")? as usize;
        let num_attention_heads = get_u64(config, "num_attention_heads")? as usize;
        let num_key_value_heads = config
            .get("num_key_value_heads")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(num_attention_heads);

        let mut eos_token_ids = token_ids(config.get("eos_token_id"));
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
                config
                    .get("bos_token_id")
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
            intermediate_size: get_u64(config, "intermediate_size")? as usize,
            num_hidden_layers: get_u64(config, "num_hidden_layers")? as usize,
            num_attention_heads,
            num_key_value_heads,
            vocab_size: get_u64(config, "vocab_size")? as usize,
            max_position_embeddings: config
                .get("max_position_embeddings")
                .and_then(|x| x.as_u64())
                .unwrap_or(2048) as usize,
            rms_norm_eps: get_f64(config, "rms_norm_eps", 1e-5),
            rope_theta: get_f64(config, "rope_theta", 10000.0),
            tie_word_embeddings: config
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            head_dim: hidden_size / num_attention_heads,
            attention_bias: config
                .get("attention_bias")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            bos_token_id,
            eos_token_ids,
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
}
