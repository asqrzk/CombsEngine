//! # combs-runtime
//!
//! Generation engine: single-flight request queue (one worker thread owns
//! the model; `generate` queues requests and streams pieces back) over the
//! [`combs_models::GenerativeModel`] contract, chunked prefill, paged KV
//! cache (via [`CacheConfig`]), pluggable [`Sampler`]s with composable
//! [`LogitsProcessor`]s, incremental detokenization and stop detection.
//! Batching and the action scheduler are Phase 3.

mod constraint;
mod detok;
mod engine;
mod logits;
mod sampler;
mod stop;
mod template;
mod toolcall;
mod transcribe;

pub use combs_models::{CacheConfig, CacheKind};
pub use constraint::{CompiledSchema, ConstraintSpec, ConstraintState, TokenByteTable};
pub use engine::{
    EmbedOptions, EmbedOutput, Engine, EngineStatsSnapshot, GenerationConfig, GenerationStats,
    LastGeneration, PerplexityOutput, Pooling, SessionInfo, check_context_len,
};

pub use logits::{
    FrequencyPenalty, LogitsProcessor, LogitsProcessorChain, PresencePenalty,
    RepetitionPenalty, TemperatureScaler, TopK, TopP,
};
pub use sampler::{
    GreedySampler, MultinomialSampler, SampleOutcome, Sampler, SamplingParams, TokenLogprobs,
    sampler_from_params,
};
pub use stop::{StopDetector, StopStringMatcher};
pub use template::ChatTemplate;
pub use toolcall::{
    ChatMessage, ToolCall, ToolCallParser, ToolCallStyle, ToolEvent, ToolFunction,
};
pub use transcribe::SpeechEngine;

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

    /// Media payload (audio) could not be decoded.
    #[error("media error: {0}")]
    Media(String),

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

    /// Structured-output constraint failure: an invalid `response_format`
    /// spec/schema, or a generation dead end where no vocabulary token can
    /// legally continue the constrained output.
    #[error("constraint error: {0}")]
    Constraint(String),

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
