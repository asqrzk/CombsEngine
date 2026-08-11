//! Request/response types for the JSON FFI boundary.
//!
//! Everything crossing the C ABI is JSON (MLC `json_ffi` pattern): one
//! integration surface for Deno, Kotlin, Swift and the WASM shell.

use serde::{Deserialize, Serialize};

/// Engine creation config (`combs_engine_create`).
///
/// Only `model_dir` is required; every other field falls back to the model
/// metadata / engine defaults. This is the object the Deno `DevicePlanner`
/// emits.
#[derive(Debug, Clone, Deserialize)]
pub struct EngineConfigJson {
    /// Path to the model directory (HF layout).
    pub model_dir: String,
    /// KV cache capacity in tokens (default: `max_position_embeddings`).
    pub max_seq_len: Option<usize>,
    /// Tokens per KV page (default: 16).
    pub page_size: Option<usize>,
    /// `"paged"` (default) or `"contiguous"`.
    pub kv_cache: Option<String>,
    /// Default prefill chunk size (0 = single shot; default: engine's).
    pub prefill_chunk_size: Option<usize>,
}

/// One chat message in a [`ChatRequestJson`]. Mirrors the OpenAI message
/// shape: assistant messages may carry `tool_calls`, tool results arrive
/// as `role: "tool"` with `tool_call_id`/`name`.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessageJson {
    /// `system` | `user` | `assistant` | `tool` | `ipython`.
    pub role: String,
    /// Text content (may be empty on tool-call-only assistant turns).
    #[serde(default)]
    pub content: String,
    /// Assistant tool invocations (OpenAI wire shape; string `arguments`
    /// are normalized to objects before templating).
    #[serde(default)]
    pub tool_calls: Vec<serde_json::Value>,
    /// Correlation id on `role: "tool"` results.
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// Tool name on `role: "tool"` results.
    #[serde(default)]
    pub name: Option<String>,
}

/// A generation request (`combs_chat_completion`).
///
/// All sampling fields are optional; absent fields fall back to the
/// engine's defaults (model `generation_config.json` merged over engine
/// defaults). Explicit values always win.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatRequestJson {
    /// Raw prompt (used verbatim; no chat template applied).
    pub prompt: Option<String>,
    /// Chat messages; when present a ChatML template is applied and
    /// `prompt` is ignored.
    pub messages: Option<Vec<ChatMessageJson>>,
    /// Maximum new tokens.
    pub max_tokens: Option<usize>,
    /// Sampling temperature (0 = greedy).
    pub temperature: Option<f32>,
    /// Top-k cutoff.
    pub top_k: Option<usize>,
    /// Nucleus threshold.
    pub top_p: Option<f32>,
    /// HF-style repetition penalty.
    pub repetition_penalty: Option<f32>,
    /// OpenAI-style frequency penalty.
    pub frequency_penalty: Option<f32>,
    /// OpenAI-style presence penalty.
    pub presence_penalty: Option<f32>,
    /// RNG seed for reproducible sampling.
    pub seed: Option<u64>,
    /// Stop strings (boundary-safe).
    pub stop: Option<Vec<String>>,
    /// Extra stop token ids.
    pub stop_token_ids: Option<Vec<u32>>,
    /// Per-request prefill chunk size override.
    pub prefill_chunk_size: Option<usize>,
}

/// Streaming event kinds emitted to the `CombsStreamCallback`.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// One generated token's text piece.
    Delta {
        /// Newly decoded text.
        text: String,
        /// The token id.
        token_id: u32,
    },
    /// Terminal event with telemetry.
    Done {
        /// `stop` | `length` | `cancelled`.
        finish_reason: String,
        /// Generation stats.
        stats: StatsJson,
    },
    /// Terminal error event.
    Error {
        /// Human-readable message.
        message: String,
    },
}

/// Telemetry subset embedded in [`StreamEvent::Done`].
#[derive(Debug, Serialize)]
pub struct StatsJson {
    /// Prompt tokens processed.
    pub prompt_tokens: usize,
    /// Tokens generated.
    pub generated_tokens: usize,
    /// Time to first token, milliseconds.
    pub ttft_ms: f64,
    /// Decode throughput.
    pub decode_tokens_per_second: f64,
    /// Prefill throughput.
    pub prefill_tokens_per_second: f64,
    /// KV pages in use at end of generation (paged cache).
    pub cache_pages_used: usize,
}

/// Model metadata payload (`combs_engine_metadata_json`).
#[derive(Debug, Serialize)]
pub struct EngineMetadataJson {
    /// Architecture string (e.g. "llama").
    pub architecture: String,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Model context window.
    pub max_position_embeddings: usize,
    /// Configured KV cache capacity.
    pub max_seq_len: usize,
    /// KV page size.
    pub page_size: usize,
    /// End-of-sequence token ids.
    pub eos_token_ids: Vec<u32>,
    /// `<|im_end|>` id when the tokenizer defines one.
    pub im_end_id: Option<u32>,
}
