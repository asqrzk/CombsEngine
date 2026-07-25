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
}
