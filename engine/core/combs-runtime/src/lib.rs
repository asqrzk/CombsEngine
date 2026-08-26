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
mod local;
mod logits;
mod request;
mod sampler;
// Prompt-lookup drafting feeds the speculative decode path, which is
// native-only (it verifies with a blocking readback).
#[cfg(not(target_family = "wasm"))]
mod spec;
mod step;
mod stop;
mod time;
mod template;
mod toolcall;
// Speech transcription reads checkpoints off disk and syncs the device on
// every window; neither exists in a browser, so it stays native-only until
// it has an async readback of its own.
#[cfg(not(target_family = "wasm"))]
mod transcribe;

pub use combs_models::{CacheConfig, CacheKind};
pub use constraint::{CompiledSchema, ConstraintSpec, ConstraintState, TokenByteTable};
pub use engine::{
    EmbedOptions, EmbedOutput, EngineStatsSnapshot, GenerationConfig, GenerationStats,
    LastGeneration, PerplexityOutput, Pooling, SessionInfo, check_context_len, default_arena_len,
};
#[cfg(not(target_family = "wasm"))]
pub use engine::Engine;

pub use local::{LocalEngine, StepEvent};
pub use request::{
    ChatHost, ChatMessageJson, ChatRequestJson, EngineConfigJson, EngineMetadataJson,
    ResolvedChat, StatsJson, StreamEvent, finish_reason, resolve_chat_request,
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
#[cfg(not(target_family = "wasm"))]
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

    /// One resident tensor is larger than the device can bind — refused
    /// before any allocation, with the arithmetic shown.
    #[error(
        "tensor {tensor} needs {bytes} bytes resident but the device caps \
         storage bindings at {limit} bytes"
    )]
    TensorExceedsDevice {
        /// Offending tensor name.
        tensor: String,
        /// Estimated resident bytes.
        bytes: u64,
        /// Device `max_storage_buffer_binding_size`.
        limit: u64,
    },

    /// The model does not fit the configured VRAM budget — refused before
    /// any allocation, itemized so the number can be argued with.
    #[error(
        "model exceeds COMBS_VRAM_BUDGET_MB={budget_mb}: weights \
         ~{weights_mb} MB + kv arena ~{kv_mb} MB + activation slack \
         ~{slack_mb} MB = ~{total_mb} MB"
    )]
    OverBudget {
        /// Estimated resident weight bytes, in MB.
        weights_mb: u64,
        /// Full KV arena bytes, in MB.
        kv_mb: u64,
        /// Activation / autotune slack, in MB.
        slack_mb: u64,
        /// Sum of the above, in MB.
        total_mb: u64,
        /// The configured budget, in MB.
        budget_mb: u64,
    },

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
