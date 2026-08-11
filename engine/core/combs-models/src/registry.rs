//! Architecture registry: maps `metadata.architecture` to a model loader.
//! New architectures are additive — one module + one `register` call.

use std::collections::HashMap;

use burn::tensor::{Device, backend::Backend};
use combs_formats::ModelSource;

use crate::traits::GenerativeModel;
use crate::{ModelError, Result};

/// A constructor for a boxed model of some architecture.
pub type Loader<B> =
    fn(&dyn ModelSource, &Device<B>) -> Result<Box<dyn GenerativeModel<B>>>;

/// Maps architecture identifiers (`config.json::model_type`, plus known
/// aliases) to loaders. Mirrors MLC's `model.py::MODELS` table.
pub struct ModelRegistry<B: Backend> {
    loaders: HashMap<String, Loader<B>>,
}

impl<B: Backend> Default for ModelRegistry<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> ModelRegistry<B> {
    /// Creates a registry with the built-in architectures registered.
    pub fn new() -> Self {
        let mut r = ModelRegistry {
            loaders: HashMap::new(),
        };
        // SmolLM2 reports model_type "llama" (older releases: "smollm2");
        // both are Llama-structured.
        r.register("llama", |source, device| {
            Ok(Box::new(crate::llama::LlamaModel::<B>::load(source, device)?))
        });
        r.register("smollm2", |source, device| {
            Ok(Box::new(crate::llama::LlamaModel::<B>::load(source, device)?))
        });
        // Qwen2/2.5 is llama-structured plus q/k/v projection biases (loaded
        // by presence) — but its optional upper-layer sliding partition
        // (`use_sliding_window: true`) is not expressible yet; all shipped
        // qwen2.5 checkpoints have it off, which the metadata parse maps to
        // `sliding_window: None`.
        r.register("qwen2", |source, device| {
            reject_active_sliding(source, "qwen2")?;
            Ok(Box::new(crate::llama::LlamaModel::<B>::load(source, device)?))
        });
        // Qwen3 is qwen2 plus per-head q/k RMSNorm (ArchSpec `qk_norm`) and
        // an explicit `head_dim` decoupled from hidden/heads; all released
        // configs ship `use_sliding_window: false`.
        r.register("qwen3", |source, device| {
            reject_active_sliding(source, "qwen3")?;
            Ok(Box::new(crate::llama::LlamaModel::<B>::load(source, device)?))
        });
        // Mistral v0.3+/Nemo report `sliding_window: null` and are plain
        // llama; v0.1's all-layer sliding window is not supported yet.
        r.register("mistral", |source, device| {
            reject_active_sliding(source, "mistral")?;
            Ok(Box::new(crate::llama::LlamaModel::<B>::load(source, device)?))
        });
        // Phi-3/3.5/4-mini: llama-structured with fused qkv/gate_up
        // projections (split at load) and an all-layer sliding window that
        // the per-layer attention layout expresses directly — every shipped
        // mini config activates it, so no sliding guard here.
        r.register("phi3", |source, device| {
            Ok(Box::new(crate::llama::LlamaModel::<B>::load(source, device)?))
        });
        // SmolVLM reports model_type "idefics3" (SigLIP + pixel-shuffle + SmolLM2).
        r.register("idefics3", |source, device| {
            Ok(Box::new(crate::smolvlm::SmolVlmModel::<B>::load(
                source, device,
            )?))
        });
        // Gemma 3 text ("gemma3_text") and the text trunk of multimodal
        // Gemma 3 ("gemma3") ride the universal decoder: ArchSpec supplies
        // the (1+w) norm flavor, qk/sandwich norms, sqrt(hidden) embed
        // scale, query_pre_attn_scalar, dual-RoPE local theta, and the
        // every-Nth-global sliding layout.
        r.register("gemma3_text", |source, device| {
            Ok(Box::new(crate::llama::LlamaModel::<B>::load(source, device)?))
        });
        r.register("gemma3", |source, device| {
            Ok(Box::new(crate::llama::LlamaModel::<B>::load(source, device)?))
        });
        r
    }

    /// Registers (or replaces) the loader for an architecture id.
    pub fn register(&mut self, architecture: &str, loader: Loader<B>) {
        self.loaders.insert(architecture.to_string(), loader);
    }

    /// Whether an architecture id has a loader.
    pub fn supports(&self, architecture: &str) -> bool {
        self.loaders.contains_key(architecture)
    }

    /// Registered architecture ids.
    pub fn architectures(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.loaders.keys().map(String::as_str).collect();
        v.sort();
        v
    }

    /// Loads the model described by `source`'s metadata.
    pub fn load(
        &self,
        source: &dyn ModelSource,
        device: &Device<B>,
    ) -> Result<Box<dyn GenerativeModel<B>>> {
        let arch = &source.metadata().architecture;
        let loader = self
            .loaders
            .get(arch)
            .ok_or_else(|| ModelError::UnsupportedArchitecture(arch.clone()))?;
        loader(source, device)
    }
}

/// Rejects checkpoints whose sliding window is actually active — the llama
/// block runs all layers global and unmasked, which would be silently wrong.
/// Landing in roadmap wave 2 (per-layer `AttentionLayout`).
fn reject_active_sliding(source: &dyn ModelSource, arch: &str) -> Result<()> {
    if let Some(w) = source.metadata().attention_pattern.sliding_window {
        return Err(ModelError::UnsupportedArchitecture(format!(
            "{arch} with an active sliding window (w={w}) — supported once the \
             per-layer attention layout lands (roadmap wave 2)"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_llama_aliases() {
        let r = ModelRegistry::<burn::backend::NdArray<f32>>::new();
        assert!(r.supports("llama"));
        assert!(r.supports("smollm2"));
        assert!(r.supports("qwen2"));
        assert!(r.supports("mistral"));
        assert!(r.supports("qwen3"));
    }
}
