//! # combs-runtime
//!
//! Generation engine: single-flight request queue (one worker thread owns
//! the model; `generate` queues requests and streams pieces back) over the
//! [`combs_models::GenerativeModel`] contract, chunked prefill, paged KV
//! cache (via [`CacheConfig`]), pluggable [`Sampler`]s with composable
//! [`LogitsProcessor`]s, incremental detokenization and stop detection.
//! Batching and the action scheduler are Phase 3.

mod constraint;
pub mod dialect;
mod detok;
mod engine;
mod kvevents;
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
// The System 1 seat reads checkpoints off disk and times itself with a
// std clock, which COMPILES in a browser and panics the moment it is
// called. It also decodes nothing a browser asks for: `combs-wasm` uses
// none of its exports. Native-only, like `spec` and `transcribe` beside
// it — the gate it was missing.
#[cfg(not(target_family = "wasm"))]
mod system_one;
mod time;
mod template;
mod toolcall;
pub mod budget;
pub mod stream_mount;
// Speech transcription reads checkpoints off disk and syncs the device on
// every window; neither exists in a browser, so it stays native-only until
// it has an async readback of its own.
#[cfg(not(target_family = "wasm"))]
mod transcribe;

pub use combs_models::{CacheConfig, CacheKind};
pub use constraint::{CompiledSchema, ConstraintSpec, ConstraintState, TokenByteTable};
pub use engine::{
    EmbedOptions, EmbedOutput, EmbedStats, EngineStatsSnapshot, GenerationConfig, GenerationStats,
    LastGeneration, PerplexityOutput, Pooling, SessionInfo, check_context_len, default_arena_len,
};
#[cfg(not(target_family = "wasm"))]
pub use engine::Engine;
pub use kvevents::{KvEvent, KvEventsPage, SeqEvent, kv_page_hash, page_hashes};

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
#[cfg(not(target_family = "wasm"))]
pub use system_one::{
    default_warm_shapes, warm_batch, BatchShape, Decided, DecisionAnswer, DecisionEngine,
    DecisionQuestion, WarmPass, WarmReport,
};
pub use template::ChatTemplate;
pub use toolcall::{
    ChatMessage, ToolCall, ToolCallParser, ToolCallStyle, ToolEvent, ToolFunction,
};
pub use dialect::{DialectParser, Off, Thinking, ToolDialect};
#[cfg(not(target_family = "wasm"))]
pub use transcribe::{SpeechEngine, TranscribeGates, Transcript};

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

    /// The card beside the weights, or a door naming a tool dialect,
    /// could not be honoured. A refusal rather than a quiet fallback:
    /// someone wrote it meaning it to be used, and a setting that did not
    /// take is worse than one that was refused out loud.
    #[error("model card: {0}")]
    Card(String),

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

#[cfg(test)]
mod flavour_tests {
    /// This crate reads its own `f16` flag and the tensors read the
    /// backend; the two must be one fact. Before this pin, `--features
    /// f16` on this crate alone switched the flag and left the backend
    /// f32 — an arena sized for two-byte elements over four-byte storage
    /// — and the tiers spelled `combs-core/f16` around it, which is the
    /// same mismatch the other way round. Either half-spelling fails
    /// here by name, in both flavours.
    #[test]
    fn the_crates_flavour_flag_and_the_backend_agree() {
        let flag = cfg!(feature = "f16");
        let backend = combs_core::BACKEND.dtype;
        assert_eq!(
            flag,
            backend == "f16",
            "this crate's `f16` feature is {flag} but combs-core serves {backend}: \
             spell the feature on the crate under test and let it forward — never \
             on a dependency alone, never on this crate without the forwarding"
        );
    }
}
