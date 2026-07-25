//! HuggingFace safetensors adapter: `config.json` + `model.safetensors`
//! (single-file or sharded with `model.safetensors.index.json`), mmap-backed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::SafeTensors;

use crate::metadata::ModelMetadata;
use crate::source::{ModelSource, SamplerConfig, TensorDtype, TensorReader};
use crate::tokenizer::TokenizerSpec;
use crate::{FormatError, Result};

/// One memory-mapped safetensors shard.
struct Shard {
    mmap: Mmap,
}

/// Lightweight per-tensor index entry (populated once at load).
struct TensorEntry {
    shard: usize,
    dtype: TensorDtype,
    shape: Vec<usize>,
}

/// [`ModelSource`] over a HuggingFace-format directory:
///
/// ```text
/// <dir>/config.json                   (required)
/// <dir>/generation_config.json        (optional)
/// <dir>/tokenizer.json                (required by the runtime)
/// <dir>/tokenizer_config.json         (optional; chat special tokens)
/// <dir>/model.safetensors             (single-file), or
/// <dir>/model.safetensors.index.json  (sharded)
/// ```
///
/// Files are memory-mapped; [`ModelSource::open_tensor`] returns zero-copy
/// views into the mapping.
pub struct SafetensorsSource {
    metadata: ModelMetadata,
    tokenizer: TokenizerSpec,
    sampler: Option<SamplerConfig>,
    shards: Vec<Shard>,
    index: HashMap<String, TensorEntry>,
}

fn read_json(path: &Path, required: bool) -> Result<Option<serde_json::Value>> {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text)
            .map(Some)
            .map_err(|source| FormatError::Json {
                context: path.display().to_string(),
                source,
            }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && !required => Ok(None),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Err(FormatError::MissingFile(path.display().to_string()))
        }
        Err(e) => Err(FormatError::Io(e)),
    }
}

fn convert_dtype(
    name: &str,
    dtype: safetensors::Dtype,
) -> Result<TensorDtype> {
    match dtype {
        safetensors::Dtype::F32 => Ok(TensorDtype::F32),
        safetensors::Dtype::F16 => Ok(TensorDtype::F16),
        safetensors::Dtype::BF16 => Ok(TensorDtype::BF16),
        safetensors::Dtype::U8 => Ok(TensorDtype::U8),
        other => Err(FormatError::UnsupportedDtype {
            tensor: name.to_string(),
            dtype: format!("{other:?}"),
        }),
    }
}

impl SafetensorsSource {
    /// Opens a model directory (see type docs for the expected layout).
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();

        // --- config + metadata -------------------------------------------------
        let config = read_json(&dir.join("config.json"), true)?.expect("required");
        let generation_config = read_json(&dir.join("generation_config.json"), false)?;
        let metadata =
            ModelMetadata::from_hf_config(&config, generation_config.as_ref())?;

        // --- tokenizer ----------------------------------------------------------
        let tokenizer_json = dir.join("tokenizer.json");
        if !tokenizer_json.exists() {
            return Err(FormatError::MissingFile(
                tokenizer_json.display().to_string(),
            ));
        }
        let mut added_tokens = HashMap::new();
        let mut chat_template = None;
        if let Some(tc) = read_json(&dir.join("tokenizer_config.json"), false)? {
            if let Some(map) = tc.get("added_tokens_decoder").and_then(|v| v.as_object()) {
                for (id, entry) in map {
                    if let (Ok(id), Some(content)) = (
                        id.parse::<u32>(),
                        entry.get("content").and_then(|c| c.as_str()),
                    ) {
                        added_tokens.insert(id, content.to_string());
                    }
                }
            }
            chat_template = tc
                .get("chat_template")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
        }
        let tokenizer = TokenizerSpec {
            tokenizer_json,
            added_tokens,
            chat_template,
        };

        // --- sampler defaults ----------------------------------------------------
        let sampler = generation_config.as_ref().map(|gc| SamplerConfig {
            temperature: gc
                .get("temperature")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            top_p: gc.get("top_p").and_then(|v| v.as_f64()).map(|v| v as f32),
            top_k: gc
                .get("top_k")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            repetition_penalty: gc
                .get("repetition_penalty")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            max_new_tokens: gc
                .get("max_new_tokens")
                .or_else(|| gc.get("max_length"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
        });

        // --- weight shards ---------------------------------------------------------
        let index_json = dir.join("model.safetensors.index.json");
        let single = dir.join("model.safetensors");
        let shard_files: Vec<PathBuf> = if index_json.exists() {
            let idx = read_json(&index_json, true)?.expect("required");
            let weight_map = idx
                .get("weight_map")
                .and_then(|v| v.as_object())
                .ok_or_else(|| FormatError::MissingField("weight_map".to_string()))?;
            let mut files: Vec<PathBuf> = weight_map
                .values()
                .filter_map(|v| v.as_str())
                .map(|f| dir.join(f))
                .collect();
            files.sort();
            files.dedup();
            files
        } else if single.exists() {
            vec![single]
        } else {
            return Err(FormatError::MissingFile(format!(
                "{single:?} (or model.safetensors.index.json)"
            )));
        };

        let mut shards = Vec::with_capacity(shard_files.len());
        let mut index = HashMap::new();
        for (shard_idx, file) in shard_files.iter().enumerate() {
            let f = std::fs::File::open(file)?;
            // SAFETY: the file is opened read-only and never mutated by us;
            // external mutation of an mmap'd model file is out of scope.
            let mmap = unsafe { Mmap::map(&f)? };
            let st = SafeTensors::deserialize(&mmap).map_err(|e| {
                FormatError::Safetensors(format!("{}: {e}", file.display()))
            })?;
            for (name, view) in st.tensors() {
                index.insert(
                    name.clone(),
                    TensorEntry {
                        shard: shard_idx,
                        dtype: convert_dtype(&name, view.dtype())?,
                        shape: view.shape().to_vec(),
                    },
                );
            }
            shards.push(Shard { mmap });
        }

        Ok(SafetensorsSource {
            metadata,
            tokenizer,
            sampler,
            shards,
            index,
        })
    }
}

impl ModelSource for SafetensorsSource {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.index.keys().cloned().collect();
        names.sort();
        names
    }

    fn open_tensor(&self, name: &str) -> Result<TensorReader<'_>> {
        let entry = self
            .index
            .get(name)
            .ok_or_else(|| FormatError::TensorNotFound(name.to_string()))?;
        let shard = &self.shards[entry.shard];
        let st = SafeTensors::deserialize(&shard.mmap)
            .map_err(|e| FormatError::Safetensors(e.to_string()))?;
        let view = st
            .tensor(name)
            .map_err(|e| FormatError::Safetensors(e.to_string()))?;
        Ok(TensorReader::new(
            name.to_string(),
            entry.shape.clone(),
            entry.dtype,
            view.data(),
        ))
    }

    fn tokenizer(&self) -> Result<TokenizerSpec> {
        Ok(self.tokenizer.clone())
    }

    fn sampler_defaults(&self) -> Option<SamplerConfig> {
        self.sampler.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a tiny synthetic HF model dir and checks the adapter surface.
    #[test]
    fn lists_and_reads_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        std::fs::write(
            root.join("config.json"),
            serde_json::to_string(&serde_json::json!({
                "model_type": "llama",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 32,
                "max_position_embeddings": 128,
                "rope_theta": 10000,
                "rms_norm_eps": 1e-5,
                "tie_word_embeddings": true,
                "eos_token_id": 0,
                "bos_token_id": 0
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(root.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(
            root.join("tokenizer_config.json"),
            serde_json::to_string(&serde_json::json!({
                "added_tokens_decoder": {
                    "0": {"content": "<|endoftext|>", "special": true},
                    "2": {"content": "<|im_end|>", "special": true}
                }
            }))
            .unwrap(),
        )
        .unwrap();

        // Two tensors: one F32, one BF16.
        let w1_bytes: Vec<u8> = (0..16)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let w2_bytes: Vec<u8> = (0..8)
            .flat_map(|i| half::bf16::from_f32(i as f32 * 0.5).to_le_bytes())
            .collect();
        let v1 = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![4, 4],
            &w1_bytes,
        )
        .unwrap();
        let v2 = safetensors::tensor::TensorView::new(
            safetensors::Dtype::BF16,
            vec![8],
            &w2_bytes,
        )
        .unwrap();
        safetensors::serialize_to_file(
            vec![("a.weight", v1), ("b.weight", v2)],
            None,
            root.join("model.safetensors").as_path(),
        )
        .unwrap();

        let src = SafetensorsSource::load(root).unwrap();
        assert_eq!(src.metadata().architecture, "llama");
        assert_eq!(src.metadata().head_dim, 4);
        assert_eq!(src.tensor_names(), vec!["a.weight", "b.weight"]);

        let a = src.open_tensor("a.weight").unwrap();
        assert_eq!(a.shape(), &[4, 4]);
        assert_eq!(a.dtype(), TensorDtype::F32);
        let data = a.load_data().unwrap();
        let vals: Vec<f32> = data.to_vec().unwrap();
        assert_eq!(vals[15], 15.0);

        let b = src.open_tensor("b.weight").unwrap();
        assert_eq!(b.dtype(), TensorDtype::BF16);
        let vals: Vec<f32> = b.load_data().unwrap().to_vec().unwrap();
        assert!((vals[3] - 1.5).abs() < 1e-3);

        let tok = src.tokenizer().unwrap();
        assert_eq!(tok.special_token_id("<|im_end|>"), Some(2));

        assert!(src.open_tensor("missing").is_err());
    }
}
