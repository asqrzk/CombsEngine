//! The JSON chat request, and what it means.
//!
//! Turning a request into a prompt and a [`GenerationConfig`] is not
//! transport work — it is the model's own semantics: which template wraps
//! the messages, which stop token ends a chat turn, which sampling default
//! a missing field falls back to, whether a `response_format` compiles.
//! That reasoning used to live inside the C FFI, which meant a second
//! caller (the browser bindings) could only have it by copying it, and a
//! copy is a place for the two to disagree about what a request means.
//!
//! So it lives here, expressed against [`ChatHost`] — the small surface a
//! request needs from whatever engine will run it. Every engine that
//! implements it gets identical request semantics, and a new one gets them
//! by implementing five methods rather than by copying ninety lines.

use serde::{Deserialize, Serialize};

use crate::constraint::ConstraintSpec;
use crate::engine::{GenerationConfig, GenerationStats};
use crate::sampler::SamplingParams;
use crate::toolcall::{ChatMessage, ToolCallStyle};

/// What a chat request needs to know about the engine that will run it.
pub trait ChatHost {
    /// Renders messages (and any tool definitions) into a prompt using the
    /// model's own chat template.
    fn wrap_chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&serde_json::Value>,
    ) -> String;

    /// Tokenizes a prompt verbatim (no special tokens added).
    fn encode(&self, text: &str) -> crate::Result<Vec<u32>>;

    /// The engine's defaults, already merged with the checkpoint's
    /// `generation_config.json`.
    fn default_config(&self) -> GenerationConfig;

    /// The `<|im_end|>` id when the tokenizer defines one.
    fn im_end_id(&self) -> Option<u32>;

    /// The tool-call phrasing this model's template teaches.
    fn tool_call_style(&self) -> ToolCallStyle;
}

/// Engine creation config.
///
/// Only the model is required; every other field falls back to the model
/// metadata / engine defaults. This is the object the Deno `DevicePlanner`
/// emits.
#[derive(Debug, Clone, Deserialize)]
pub struct EngineConfigJson {
    /// Path to the model file or directory. Absent in the browser, where
    /// the bytes are handed over directly instead.
    #[serde(default)]
    pub model_dir: Option<String>,
    /// KV cache capacity in tokens (default: `max_position_embeddings`).
    pub max_seq_len: Option<usize>,
    /// Tokens per KV page (default: 16).
    pub page_size: Option<usize>,
    /// `"paged"` (default) or `"contiguous"`.
    pub kv_cache: Option<String>,
    /// Default prefill chunk size (0 = single shot; default: engine's).
    pub prefill_chunk_size: Option<usize>,
}

/// One chat message. Mirrors the OpenAI message shape: assistant messages
/// may carry `tool_calls`, tool results arrive as `role: "tool"` with
/// `tool_call_id`/`name`.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessageJson {
    /// `system` | `user` | `assistant` | `tool` | `ipython`.
    pub role: String,
    /// Text content (may be empty on tool-call-only assistant turns).
    #[serde(default)]
    pub content: String,
    /// Assistant tool invocations (OpenAI wire shape).
    #[serde(default)]
    pub tool_calls: Vec<serde_json::Value>,
    /// Correlation id on `role: "tool"` results.
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// Tool name on `role: "tool"` results.
    #[serde(default)]
    pub name: Option<String>,
}

/// A generation request.
///
/// All sampling fields are optional; absent fields fall back to the
/// engine's defaults (model `generation_config.json` merged over engine
/// defaults). Explicit values always win.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatRequestJson {
    /// Raw prompt (used verbatim; no chat template applied).
    pub prompt: Option<String>,
    /// Chat messages; when present the chat template is applied and
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
    /// OpenAI-shaped tool definitions, rendered through the model's chat
    /// template. Generated tool calls arrive on the `Done` event.
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    /// OpenAI `response_format`: `json_object` / `json_schema` constrain
    /// the output through the engine's JSON automaton.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    /// Min-p probability floor relative to the top token (multinomial).
    #[serde(default)]
    pub min_p: Option<f32>,
    /// Per-token logit offsets; JSON keys are token-id strings.
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<u32, f32>>,
    /// Capture per-token log-probabilities (top-n; may be 0).
    #[serde(default)]
    pub logprobs: Option<usize>,
}

/// A request resolved against a specific engine and ready to run.
pub struct ResolvedChat {
    /// The tokenized prompt.
    pub prompt_tokens: Vec<u32>,
    /// Generation parameters, with request fields laid over the defaults.
    pub config: GenerationConfig,
    /// How to parse tool calls out of the stream (`None` when the request
    /// supplied no tools — an unasked-for parser only mangles prose).
    pub parser_style: ToolCallStyle,
}

/// Resolves a request against the engine that will run it: renders the
/// prompt, merges sampling, applies chat-mode stop tokens, and compiles
/// any structured-output schema so a bad one is a request error rather
/// than a generation failure.
pub fn resolve_chat_request(
    host: &dyn ChatHost,
    request: &ChatRequestJson,
    default_prefill_chunk_size: Option<usize>,
) -> Result<ResolvedChat, String> {
    let prompt = match (&request.messages, &request.prompt) {
        (Some(messages), _) => {
            let msgs: Vec<ChatMessage> = messages
                .iter()
                .map(|m| ChatMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                    tool_calls: m
                        .tool_calls
                        .iter()
                        .filter_map(|c| serde_json::from_value(c.clone()).ok())
                        .collect(),
                    tool_call_id: m.tool_call_id.clone(),
                    name: m.name.clone(),
                })
                .collect();
            host.wrap_chat_with_tools(&msgs, request.tools.as_ref())
        }
        (None, Some(p)) => p.clone(),
        (None, None) => return Err("request needs `prompt` or `messages`".into()),
    };
    // Parse tool calls out of the stream only when tools were supplied.
    let parser_style = if request.tools.is_some() {
        host.tool_call_style()
    } else {
        ToolCallStyle::None
    };
    let prompt_tokens = host
        .encode(&prompt)
        .map_err(|e| format!("tokenization failed: {e}"))?;

    // Defaults from the engine; explicit request fields win.
    let mut config = host.default_config();
    let sampling = SamplingParams {
        temperature: request.temperature.unwrap_or(config.sampling.temperature),
        top_k: request.top_k.or(config.sampling.top_k),
        top_p: request.top_p.or(config.sampling.top_p),
        repetition_penalty: request
            .repetition_penalty
            .or(config.sampling.repetition_penalty),
        frequency_penalty: request
            .frequency_penalty
            .or(config.sampling.frequency_penalty),
        presence_penalty: request.presence_penalty.or(config.sampling.presence_penalty),
        seed: request.seed.or(config.sampling.seed),
        min_p: request.min_p.or(config.sampling.min_p),
        logit_bias: request
            .logit_bias
            .clone()
            .or_else(|| config.sampling.logit_bias.clone()),
        logprobs: request.logprobs.or(config.sampling.logprobs),
        penalty_last_n: config.sampling.penalty_last_n,
        penalty_exempt: config.sampling.penalty_exempt.clone(),
    };
    config.sampling = sampling;
    if let Some(mt) = request.max_tokens {
        config.max_tokens = mt;
    }
    if let Some(stop) = request.stop.clone() {
        config.stop_strings = stop;
    }
    if let Some(ids) = request.stop_token_ids.clone() {
        config.stop_token_ids = ids;
    } else if request.messages.is_some() {
        // Chat mode: stop at <|im_end|> when the tokenizer defines it.
        if let Some(im_end) = host.im_end_id() {
            config.stop_token_ids.push(im_end);
        }
    }
    if let Some(chunk) = request.prefill_chunk_size.or(default_prefill_chunk_size) {
        config.prefill_chunk_size = chunk;
    }
    if let Some(rf) = request.response_format.as_ref() {
        config.constraint = ConstraintSpec::from_response_format(rf)
            .and_then(|spec| {
                if let Some(s) = &spec {
                    // Compile now so schema errors surface as request
                    // errors, not generation failures.
                    s.compile()?;
                }
                Ok(spec)
            })
            .map_err(|e| format!("response_format: {e}"))?;
    }

    Ok(ResolvedChat {
        prompt_tokens,
        config,
        parser_style,
    })
}

/// Streaming event kinds. One JSON object per event, identical across
/// every transport — the C callback, the worker `postMessage`, and the
/// HTTP stream all carry these exact shapes, so clients are written once.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// One generated token's text piece.
    Delta {
        /// Newly decoded text.
        text: String,
        /// The token id.
        token_id: u32,
        /// ln P of this token when the request asked for logprobs; absent
        /// otherwise (and on buffered tool-parse flushes).
        #[serde(skip_serializing_if = "Option::is_none")]
        logprob: Option<f32>,
    },
    /// Terminal event with telemetry.
    Done {
        /// `stop` | `length` | `cancelled` | `tool_calls`.
        finish_reason: String,
        /// Complete parsed tool calls (OpenAI wire shape, `arguments` as a
        /// JSON string); empty when the model made none.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<serde_json::Value>,
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
#[derive(Debug, Default, Serialize)]
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

impl From<&GenerationStats> for StatsJson {
    fn from(s: &GenerationStats) -> Self {
        StatsJson {
            prompt_tokens: s.prompt_tokens,
            generated_tokens: s.generated_tokens,
            ttft_ms: s.ttft.as_secs_f64() * 1000.0,
            decode_tokens_per_second: s.decode_tokens_per_second(),
            prefill_tokens_per_second: s.prefill_tokens_per_second(),
            cache_pages_used: s.cache_pages_used,
        }
    }
}

/// Model metadata payload.
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

/// The finish reason a completed run should report.
pub fn finish_reason(stats: &GenerationStats, max_tokens: usize, tool_calls: usize) -> &'static str {
    if stats.generated_tokens >= max_tokens {
        "length"
    } else if tool_calls > 0 {
        "tool_calls"
    } else {
        "stop"
    }
}
