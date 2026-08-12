//! The model-agnostic generation contract.

use std::ops::Range;

use burn::tensor::{Int, Tensor, backend::Backend, Device};
use combs_formats::{ModelMetadata, ModelSource};

use crate::Result;
use crate::kv::{CacheConfig, KVCache};

/// Fixed contract every generative architecture implements — the direct
/// analog of MLC's `embed / prefill / decode / create_kv_cache` function set.
/// The runtime only ever talks to models through this trait.
pub trait GenerativeModel<B: Backend>: Send {
    /// Metadata this model was built from.
    fn metadata(&self) -> &ModelMetadata;

    /// Loads all weights from a [`ModelSource`] onto `device`.
    fn load(source: &dyn ModelSource, device: &Device<B>) -> Result<Self>
    where
        Self: Sized;

    /// Creates a fresh KV cache for a new generation session, sized and
    /// implemented according to `config` (paged arena vs contiguous
    /// baseline).
    fn create_kv_cache(&self, config: &CacheConfig) -> Box<dyn KVCache<B>>;

    /// Embeds token ids: `[batch, seq] -> [batch, seq, hidden]`.
    fn embed(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3>;

    /// Embeds token ids, splicing vision-tower features into the image-token
    /// spans. `images` are preprocessed pixel batches `[1, channels, H, W]`,
    /// one per image-token span, in order. Text-only models keep the default
    /// impl, which rejects non-empty media and otherwise defers to `embed`.
    fn embed_multimodal(
        &self,
        tokens: Tensor<B, 2, Int>,
        images: &[Tensor<B, 4>],
    ) -> Result<Tensor<B, 3>> {
        if !images.is_empty() {
            return Err(crate::ModelError::UnsupportedMedia(format!(
                "{} image(s) passed to a text-only model",
                images.len()
            )));
        }
        Ok(self.embed(tokens))
    }

    /// Runs (a chunk of) the prompt through the model, filling the KV cache
    /// for positions `pos`. `pos.end - pos.start` must equal the input
    /// sequence length, and `pos.start` must equal the cache's current
    /// length (dense contiguous chunks). Returns the logits of the **last**
    /// position, shape `[batch, vocab]`.
    fn prefill(
        &mut self,
        input: Tensor<B, 3>,
        cache: &mut dyn KVCache<B>,
        pos: Range<u32>,
    ) -> Tensor<B, 2>;

    /// Runs one decode step (single new position at the end of the cache).
    /// Returns the logits of that position, shape `[batch, vocab]`.
    fn decode(&mut self, input: Tensor<B, 3>, cache: &mut dyn KVCache<B>) -> Tensor<B, 2>;

    /// Runs (a chunk of) the prompt and returns the final-norm hidden
    /// states for those positions, shape `[1, seq, hidden]` — the
    /// embeddings path. Same cache/position contract as
    /// [`GenerativeModel::prefill`]. Models that cannot expose hidden
    /// states keep the default error.
    fn prefill_hidden(
        &mut self,
        _input: Tensor<B, 3>,
        _cache: &mut dyn KVCache<B>,
        _pos: Range<u32>,
    ) -> Result<Tensor<B, 3>> {
        Err(crate::ModelError::Unsupported(
            "this model does not expose hidden states for embeddings".to_string(),
        ))
    }

    /// Whether [`GenerativeModel::prefill_hidden`] is implemented — the
    /// capability flag `/v1/model/info` advertises as `embeddings`.
    fn supports_hidden_states(&self) -> bool {
        false
    }

    /// Runs (a chunk of) the prompt and returns logits for **every**
    /// position, shape `[1, seq, vocab]` — the perplexity / speculative-
    /// decode path. Same cache/position contract as
    /// [`GenerativeModel::prefill`]. Memory scales with `seq × vocab`, so
    /// callers chunk accordingly. Default: unsupported.
    fn prefill_all_logits(
        &mut self,
        _input: Tensor<B, 3>,
        _cache: &mut dyn KVCache<B>,
        _pos: Range<u32>,
    ) -> Result<Tensor<B, 3>> {
        Err(crate::ModelError::Unsupported(
            "this model does not expose per-position logits".to_string(),
        ))
    }
}

/// Speech-to-text models (Whisper-style encoder–decoder). A separate
/// contract from [`GenerativeModel`]: the encoder runs once per audio
/// window, then the decoder is stepped over token prefixes against the
/// fixed encoder states.
pub trait SpeechToTextModel<B: Backend>: Send {
    /// Architecture + hyperparameter metadata.
    fn metadata(&self) -> &ModelMetadata;

    /// Mel bins the encoder expects (derived from its conv stem weights).
    fn n_mels(&self) -> usize;

    /// Encodes one `[1, n_mels, frames]` log-mel window into encoder
    /// states `[1, frames/2, hidden]`.
    fn encode_audio(&self, mel: Tensor<B, 3>) -> crate::Result<Tensor<B, 3>>;

    /// Runs the decoder over the whole token prefix and returns the final
    /// position's logits `[vocab]`.
    fn decode_step(&self, tokens: &[u32], encoded: &Tensor<B, 3>)
    -> crate::Result<Tensor<B, 1>>;
}
