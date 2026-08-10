//! # combs-runtime
//!
//! Generation engine: single-flight request queue (one worker thread owns
//! the model; `generate` queues requests and streams pieces back) over the
//! [`combs_models::GenerativeModel`] contract, chunked prefill, paged KV
//! cache (via [`CacheConfig`]), pluggable [`Sampler`]s with composable
//! [`LogitsProcessor`]s, incremental detokenization and stop detection.
//! Batching and the action scheduler are Phase 3.

mod detok;
mod engine;
mod logits;
mod sampler;
mod stop;

pub use combs_models::{CacheConfig, CacheKind};
pub use engine::{Engine, GenerationConfig, GenerationStats, check_context_len};

pub use logits::{
    FrequencyPenalty, LogitsProcessor, LogitsProcessorChain, PresencePenalty,
    RepetitionPenalty, TemperatureScaler, TopK, TopP,
};
pub use sampler::{GreedySampler, MultinomialSampler, Sampler, SamplingParams, sampler_from_params};
pub use stop::{StopDetector, StopStringMatcher};

/// Errors produced by the runtime.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// Model construction failed.
    #[error(transparent)]
    Model(#[from] combs_models::ModelError),

    /// Format adapter failed.
    #[error(transparent)]
    Format(#[from] combs_formats::FormatError),

    /// Tokenizer failure.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// Tensor data could not be read back from the device.
    #[error("tensor readback error: {0}")]
    Readback(String),

    /// Prompt + requested generation exceeds the context budget (KV cache
    /// capacity, capped by the model's `max_position_embeddings`).
    #[error(
        "prompt ({prompt_len} tokens) + max_tokens ({max_tokens}) exceeds the context \
         limit ({max_position_embeddings})"
    )]
    ContextTooLong {
        /// Prompt length in tokens.
        prompt_len: usize,
        /// Requested new tokens.
        max_tokens: usize,
        /// Context limit (cache capacity).
        max_position_embeddings: usize,
    },

    /// Generation was aborted via the request's cancel flag (or the caller
    /// dropped the streaming channel).
    #[error("generation cancelled")]
    Cancelled,

    /// The engine worker thread is gone (engine dropped or spawn failed).
    #[error("engine worker unavailable: {0}")]
    WorkerGone(String),
}

/// Convenient result alias for this crate.
pub type Result<T> = std::result::Result<T, EngineError>;
