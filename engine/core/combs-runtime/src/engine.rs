//! The generation engine.
//!
//! Single-flight request queue (LiteRT-LM ExecutionQueue pattern): the model
//! and KV cache live on one worker thread; `generate` (callable on a shared
//! `&Engine`, e.g. via `Arc<Engine>`) pushes a request onto an internal
//! `mpsc` queue and streams pieces back over a channel. Requests are
//! executed strictly serially — this is the seam Phase 3's threaded engine
//! and scheduler will hang off.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use burn::tensor::{Tensor, TensorData};
use combs_core::{BufferPool, CombsBackend, CombsDevice};
use combs_formats::{ModelMetadata, ModelSource, SamplerConfig};
use combs_media::PixelBatch;
use combs_models::{CacheConfig, CacheKind, GenerativeModel, KVCache, ModelRegistry, pixels_to_tensor};
use tokenizers::Tokenizer;

use crate::constraint::{ConstraintSpec, ConstraintState, TokenByteTable};
use crate::detok::IncrementalDetokenizer;
use crate::spec;
use crate::sampler::{
    Sampler, SamplingParams, TokenLogprobs, sampler_from_params_exempting,
};
use crate::stop::StopDetector;
use crate::{EngineError, Result};

/// Parameters for one generation call.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Sampling parameters (temperature, top-k/top-p, penalties, seed).
    pub sampling: SamplingParams,
    /// Extra stop token ids (e.g. `<|im_end|>` in chat mode), checked in
    /// addition to the model's eos ids.
    pub stop_token_ids: Vec<u32>,
    /// Stop strings matched against the detokenized output stream.
    pub stop_strings: Vec<String>,
    /// Prompt tokens processed per prefill call. 0 (or larger than the
    /// prompt) means single-shot prefill.
    ///
    /// Note: burn 0.21's wgpu path miscomputes matmuls with M >= 512 and
    /// K >= 512 (see workspace README "Known issues"); the model code
    /// sidesteps that region via `safe_matmul`, so any chunk size is safe.
    pub prefill_chunk_size: usize,
    /// Reuse the previous request's KV cache across calls: the worker keeps
    /// the last session alive and rolls it back to the longest common token
    /// prefix, so only the prompt suffix is prefilled (paged cache only).
    pub session_reuse: bool,
    /// Named session for KV prefix reuse (e.g. one per debate agent):
    /// requests with the same id share a rolling session, independent of
    /// other ids. `None` uses the anonymous default session.
    pub session_id: Option<String>,
    /// Structured-output constraint (OpenAI `response_format`): when set,
    /// the logits row is masked before every sample so only tokens that
    /// legally continue the JSON output survive. `None` leaves the decode
    /// path byte-identical to an unconstrained run.
    pub constraint: Option<ConstraintSpec>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        GenerationConfig {
            max_tokens: 64,
            sampling: SamplingParams::default(),
            stop_token_ids: Vec::new(),
            stop_strings: Vec::new(),
            prefill_chunk_size: 512,
            session_reuse: true,
            session_id: None,
            constraint: None,
        }
    }
}

/// Pooling strategy for the embeddings path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Hidden state of the last position (decoder-model convention,
    /// e.g. qwen3-embedding).
    Last,
    /// Mean over all positions.
    Mean,
}

/// Options for one [`Engine::embed_texts`] call — plain per-request data.
#[derive(Debug, Clone, Default)]
pub struct EmbedOptions {
    /// Pooling override; `None` uses the pooling detected from the
    /// checkpoint (`1_Pooling/config.json`, else last-token).
    pub pooling: Option<Pooling>,
    /// Keep only the first N dimensions, then re-normalize (matryoshka
    /// truncation). `None` returns the full hidden size.
    pub dimensions: Option<usize>,
}

/// Result of one embeddings call.
#[derive(Debug, Clone)]
pub struct EmbedOutput {
    /// One L2-normalized vector per input text, in order.
    pub vectors: Vec<Vec<f32>>,
    /// Total input tokens across all texts.
    pub prompt_tokens: usize,
}

/// Result of one perplexity evaluation.
#[derive(Debug, Clone)]
pub struct PerplexityOutput {
    /// Sum of per-token negative log-likelihoods (nats).
    pub nll_sum: f64,
    /// Number of scored tokens (`input length − 1`).
    pub tokens: usize,
}

impl PerplexityOutput {
    /// `exp(mean NLL)` — the standard perplexity.
    pub fn perplexity(&self) -> f64 {
        if self.tokens == 0 {
            return f64::NAN;
        }
        (self.nll_sum / self.tokens as f64).exp()
    }
}

/// Telemetry for one generation call.
#[derive(Debug, Clone)]
pub struct GenerationStats {
    /// Number of prompt tokens processed in prefill.
    pub prompt_tokens: usize,
    /// Number of tokens generated.
    pub generated_tokens: usize,
    /// Time from generation start to the first sampled token (≈ prefill).
    pub ttft: Duration,
    /// Wall time of the decode loop (after the first token).
    pub decode_time: Duration,
    /// Total wall time.
    pub total_time: Duration,
    /// KV cache pages allocated to the request at the end of generation
    /// (0 for the contiguous baseline cache).
    pub cache_pages_used: usize,
    /// Prompt tokens served from the previous session's KV cache instead of
    /// being re-prefilled (rolling-session prefix reuse; 0 on a cold start).
    pub cached_tokens: usize,
}

impl GenerationStats {
    /// Prefill throughput in tokens/second (0 if prefill took no time).
    ///
    /// Prefill time is approximated by TTFT, which includes the first
    /// sample + readback; at Phase 1 scales the difference is negligible.
    pub fn prefill_tokens_per_second(&self) -> f64 {
        if self.prompt_tokens == 0 || self.ttft.as_secs_f64() == 0.0 {
            return 0.0;
        }
        self.prompt_tokens as f64 / self.ttft.as_secs_f64()
    }

    /// Decode throughput in tokens/second (0 if nothing decoded).
    pub fn tokens_per_second(&self) -> f64 {
        if self.generated_tokens <= 1 || self.decode_time.as_secs_f64() == 0.0 {
            return 0.0;
        }
        (self.generated_tokens - 1) as f64 / self.decode_time.as_secs_f64()
    }

    /// Alias for [`GenerationStats::tokens_per_second`], for symmetry with
    /// [`GenerationStats::prefill_tokens_per_second`].
    pub fn decode_tokens_per_second(&self) -> f64 {
        self.tokens_per_second()
    }
}

/// Validates that prompt + requested generation fits the context budget
/// (the KV cache capacity).
pub fn check_context_len(
    prompt_len: usize,
    max_tokens: usize,
    max_position_embeddings: usize,
) -> Result<()> {
    if prompt_len + max_tokens > max_position_embeddings {
        return Err(EngineError::ContextTooLong {
            prompt_len,
            max_tokens,
            max_position_embeddings,
        });
    }
    Ok(())
}

/// One queued generation request.
struct GenerateRequest {
    prompt_tokens: Vec<u32>,
    /// Preprocessed images for multimodal prompts (empty for text-only).
    images: Vec<PixelBatch>,
    config: GenerationConfig,
    cancel: Arc<AtomicBool>,
    /// Streaming channel: (token id, new text piece, logprob capture) per
    /// generated token; the capture is `None` unless the request asked.
    pieces: mpsc::Sender<(u32, String, Option<TokenLogprobs>)>,
    /// Final outcome, sent once after `pieces` closes.
    reply: mpsc::Sender<Result<GenerationStats>>,
}

/// Instruction for the engine worker thread.
enum Command {
    Generate(Box<GenerateRequest>),
    Embed {
        texts: Vec<String>,
        opts: EmbedOptions,
        reply: mpsc::Sender<Result<EmbedOutput>>,
    },
    Perplexity {
        tokens: Vec<u32>,
        chunk: usize,
        reply: mpsc::Sender<Result<PerplexityOutput>>,
    },
    /// Drops one named session (or all when `id` is None), freeing its
    /// arena. Single-flight like everything else: waits behind an
    /// in-flight generation.
    ClearSessions {
        id: Option<String>,
        reply: mpsc::Sender<usize>,
    },
    Shutdown,
}

/// The worker's rolling session: the previous request's full token history
/// (prompt + decoded tokens — exactly what the KV cache contains) and its
/// cache. The next request rolls the cache back to the longest common
/// prefix with its own prompt and prefills only the suffix.
struct SessionState {
    history: Vec<u32>,
    cache: Box<dyn KVCache<CombsBackend>>,
    last_used: u64,
}

/// Maximum named sessions kept alive at once (LRU-evicted beyond this).
/// Each session owns its KV arena, so this also bounds cache VRAM.
const MAX_SESSIONS: usize = 4;

/// Default cap for the KV arena when the model advertises a larger
/// positional maximum. 32k tokens covers long coding sessions while keeping
/// the arena allocation bounded on 16–18 GB machines; `--context-size`
/// (CLI) / `load_with_cache_config` (API) override.
///
/// Under `--features f16` each cached token costs half the memory, so the
/// default cap doubles to 64k — the "spend the freed VRAM on longer context"
/// win. An explicit `--context-size` still overrides either default.
#[cfg(not(feature = "f16"))]
const DEFAULT_KV_ARENA_CAP: usize = 32768;
#[cfg(feature = "f16")]
const DEFAULT_KV_ARENA_CAP: usize = 65536;

/// The arena length [`Engine::load`] would pick for a model with the given
/// `max_position_embeddings` — for callers composing a partial
/// [`CacheConfig`] override (e.g. `combs serve --page-size` without
/// `--context-size`).
pub fn default_arena_len(max_position_embeddings: usize) -> usize {
    max_position_embeddings.min(DEFAULT_KV_ARENA_CAP)
}

/// The worker's session table: named rolling sessions with LRU eviction.
/// The anonymous session (empty key) serves requests without a session id.
struct SessionSet {
    map: std::collections::HashMap<String, SessionState>,
    tick: u64,
    evictions: u64,
}

impl SessionSet {
    fn new() -> Self {
        SessionSet {
            map: std::collections::HashMap::new(),
            tick: 0,
            evictions: 0,
        }
    }

    /// Removes and returns the session for `key` (caller re-inserts it
    /// after generation, or drops it to free its pages).
    fn take(&mut self, key: &str) -> Option<SessionState> {
        self.map.remove(key)
    }

    /// Drops every session, returning how many were removed.
    fn clear_all(&mut self) -> usize {
        let n = self.map.len();
        self.map.clear();
        n
    }

    /// Inserts a session, evicting the least-recently-used one past
    /// [`MAX_SESSIONS`].
    fn put(&mut self, key: String, mut session: SessionState) {
        self.tick += 1;
        session.last_used = self.tick;
        self.map.insert(key, session);
        if self.map.len() > MAX_SESSIONS {
            if let Some(oldest) = self
                .map
                .iter()
                .min_by_key(|(_, s)| s.last_used)
                .map(|(k, _)| k.clone())
            {
                self.map.remove(&oldest);
                self.evictions += 1;
            }
        }
    }
}

/// Point-in-time engine statistics, written by the worker thread after
/// every generation and read (lock held for microseconds) by observers —
/// never through the single-flight command queue, which would block a
/// `/v1/stats` poll behind an in-flight generation.
#[derive(Debug, Clone, Default)]
pub struct EngineStatsSnapshot {
    /// Completed generation requests (including failed ones).
    pub requests_total: u64,
    /// Requests that ended in an engine error.
    pub errors_total: u64,
    /// Requests the client aborted mid-stream. A normal outcome — kept
    /// apart from `errors_total` so a stopped turn does not read as a
    /// fault in the stats.
    pub cancelled_total: u64,
    /// Sum of prompt tokens across requests.
    pub prompt_tokens_total: u64,
    /// Sum of generated tokens across requests.
    pub generated_tokens_total: u64,
    /// Sum of prefix-reuse (KV cache hit) tokens across requests.
    pub cached_tokens_total: u64,
    /// Exponentially-weighted decode throughput (tok/s), α = 0.3.
    pub decode_tok_s_ewma: f64,
    /// The most recent completed generation.
    pub last: Option<LastGeneration>,
    /// Live rolling sessions and their KV footprint.
    pub sessions: Vec<SessionInfo>,
    /// Sessions LRU-evicted since load.
    pub session_evictions: u64,
    /// Session-table capacity ([`MAX_SESSIONS`]).
    pub max_sessions: usize,
    /// GPU allocator state, sampled on the worker after each generation.
    pub gpu: Option<combs_core::GpuMemory>,
    /// Bytes one KV page costs across all layers (both K and V), so
    /// consumers can turn page counts into memory: `pages × kv_page_bytes`.
    pub kv_page_bytes: u64,
}

/// Timing/count summary of one completed generation.
#[derive(Debug, Clone)]
pub struct LastGeneration {
    /// Prompt length in tokens.
    pub prompt_tokens: usize,
    /// Generated length in tokens.
    pub generated_tokens: usize,
    /// Prefix-reuse hit length in tokens.
    pub cached_tokens: usize,
    /// Time to first token, milliseconds.
    pub ttft_ms: f64,
    /// Prefill throughput, tokens/second.
    pub prefill_tok_s: f64,
    /// Decode throughput, tokens/second.
    pub decode_tok_s: f64,
    /// Wall time of the whole request, milliseconds.
    pub total_ms: f64,
    /// KV pages held by the request's cache at completion.
    pub cache_pages_used: usize,
}

/// One live rolling session's KV state.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    /// Session id (`"(anonymous)"` for the empty key).
    pub id: String,
    /// Tokens in the session history (== KV cache contents).
    pub history_len: usize,
    /// Page-table state of the session's cache, when paged.
    pub pages: Option<combs_models::PageStats>,
}

/// Length of the longest common prefix of two token slices.
fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Single-flight generation engine over the default wgpu backend.
///
/// The model lives on an internal worker thread; all public methods are
/// `&self`, so one `Engine` (typically behind an `Arc`) can be shared by
/// many owners. `generate` calls are queued and executed strictly one at a
/// time, in submission order.
pub struct Engine {
    device: CombsDevice,
    tokenizer: Tokenizer,
    metadata: ModelMetadata,
    spec: combs_formats::TokenizerSpec,
    /// The checkpoint's own Jinja chat template, when it ships one;
    /// `wrap_chat` falls back to the token-sniffed builtin wraps otherwise.
    template: Option<crate::template::ChatTemplate>,
    template_warned: std::sync::Once,
    default_config: GenerationConfig,
    cache_config: CacheConfig,
    tx: mpsc::Sender<Command>,
    worker: Mutex<Option<JoinHandle<()>>>,
    stats: Arc<Mutex<EngineStatsSnapshot>>,
    /// Whether the model implements `prefill_hidden` (captured before the
    /// model moves to the worker) — the `embeddings` capability.
    supports_embeddings: bool,
    /// Pooling convention detected from the checkpoint artifacts.
    default_pooling: Pooling,
    #[allow(dead_code)] // facade used by later phases for arena management
    pool: BufferPool,
}

impl Engine {
    /// Loads a model from any [`ModelSource`] via the default registry.
    ///
    /// The KV cache kind is selected by `COMBS_KV=paged|contiguous`
    /// (default: paged) with capacity `max_position_embeddings`, capped at
    /// [`DEFAULT_KV_ARENA_CAP`] so models advertising huge positional maxes
    /// (e.g. 128k) don't allocate enormous arenas by default. Use
    /// [`Engine::load_with_cache_config`] (CLI: `--context-size`) to raise.
    pub fn load(source: &dyn ModelSource, device: CombsDevice) -> Result<Self> {
        let kind = match std::env::var("COMBS_KV").as_deref() {
            Ok("contiguous") => CacheKind::Contiguous,
            Ok("paged") | Err(_) => CacheKind::Paged,
            Ok(other) => {
                tracing::warn!("unknown COMBS_KV={other:?}; using paged KV cache");
                CacheKind::Paged
            }
        };
        let positional = source.metadata().max_position_embeddings;
        let arena = positional.min(DEFAULT_KV_ARENA_CAP);
        if arena < positional {
            tracing::info!(
                "KV arena capped at {arena} tokens (model max {positional}); \
                 raise with --context-size"
            );
        }
        let mut config = CacheConfig::paged(arena);
        config.kind = kind;
        Self::load_with_cache_config(source, device, config)
    }

    /// Loads a model with an explicit KV cache configuration.
    ///
    /// `COMBS_KV_QUANT=1` turns on int8 KV storage for global-attention
    /// layers regardless of how the config was built (single choke point
    /// for both the CLI paths).
    pub fn load_with_cache_config(
        source: &dyn ModelSource,
        device: CombsDevice,
        mut cache_config: CacheConfig,
    ) -> Result<Self> {
        if matches!(std::env::var("COMBS_KV_QUANT").as_deref(), Ok("1")) {
            cache_config.quantize_kv = true;
        }
        let registry = ModelRegistry::<CombsBackend>::new();
        let model = registry.load(source, &device)?;
        let supports_embeddings = model.supports_hidden_states();

        let spec = source.tokenizer()?;
        let default_pooling = detect_pooling(&spec.tokenizer_json);
        let tokenizer = Tokenizer::from_file(&spec.tokenizer_json)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))?;

        // Observability snapshot: seeded with the static KV geometry so
        // consumers can turn page counts into bytes; the worker refreshes
        // the dynamic parts after every generation.
        let meta = source.metadata();
        let elem_bytes: u64 = if cfg!(feature = "f16") { 2 } else { 4 };
        // Only global-attention layers hold paged arenas; sliding layers
        // (gemma's 5:1 pattern) keep a small rolling tensor instead.
        let pattern = &meta.attention_pattern;
        let global_layers = (0..meta.num_hidden_layers)
            .filter(|&i| pattern.is_global_layer(i))
            .count()
            .max(1);
        // int8 KV packs to 1 byte/value + one f32 scale per 32 values.
        let bytes_per_value_x8 = if cache_config.quantize_kv {
            9 // (1 + 4/32) * 8
        } else {
            elem_bytes * 8
        };
        let stats = Arc::new(Mutex::new(EngineStatsSnapshot {
            max_sessions: MAX_SESSIONS,
            kv_page_bytes: cache_config.page_size as u64
                * meta.num_key_value_heads as u64
                * meta.head_dim as u64
                * bytes_per_value_x8
                / 8
                * 2 // K and V
                * global_layers as u64,
            // Priming the GPU sample here is ALSO the load-time sync:
            // gpu_memory blocks on the compute stream, so every enqueued
            // weight upload materializes before load() returns — an
            // allocation panic lands during load, not behind /health.
            gpu: combs_core::gpu_memory(&device),
            ..EngineStatsSnapshot::default()
        }));

        let (tx, rx) = mpsc::channel();
        let worker = {
            let tokenizer = tokenizer.clone();
            let device = device.clone();
            let max_position_embeddings = source.metadata().max_position_embeddings;
            let stats = stats.clone();
            std::thread::Builder::new()
                .name("combs-engine".to_string())
                .spawn(move || {
                    worker_loop(
                        model,
                        tokenizer,
                        device,
                        cache_config,
                        max_position_embeddings,
                        rx,
                        stats,
                    );
                })
                .map_err(|e| EngineError::WorkerGone(format!("spawning worker: {e}")))?
        };

        // Checkpoint chat template (tokenizer_config.json / GGUF
        // `tokenizer.chat_template`): resolve the bos/eos token STRINGS the
        // template context exposes from the special-token table.
        let token_string = |id: Option<u32>| {
            id.and_then(|id| spec.added_tokens.get(&id).cloned())
                .unwrap_or_default()
        };
        let template = spec.chat_template.clone().map(|src| {
            crate::template::ChatTemplate::new(
                src,
                token_string(meta.bos_token_id),
                token_string(meta.eos_token_ids.first().copied()),
            )
        });

        Ok(Engine {
            device,
            tokenizer,
            metadata: source.metadata().clone(),
            spec,
            template,
            template_warned: std::sync::Once::new(),
            default_config: default_config_from(source.sampler_defaults().as_ref()),
            cache_config,
            tx,
            worker: Mutex::new(Some(worker)),
            stats,
            supports_embeddings,
            default_pooling,
            pool: BufferPool::new(),
        })
    }

    /// Clones the current statistics snapshot (worker-maintained; safe to
    /// call at any time — never blocks on generation).
    pub fn stats_snapshot(&self) -> EngineStatsSnapshot {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Model metadata.
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// The device the engine runs on.
    pub fn device(&self) -> &CombsDevice {
        &self.device
    }

    /// The KV cache configuration new sessions are created with.
    pub fn cache_config(&self) -> CacheConfig {
        self.cache_config
    }

    /// Per-layer attention window layout resolved for this model — the
    /// same shape the KV cache is built with (`None` = global arena,
    /// `Some(w)` = rolling window).
    pub fn attention_windows(&self) -> Vec<Option<usize>> {
        combs_models::ArchSpec::resolve(&self.metadata).windows()
    }

    /// Drops one named KV session (`Some(id)`) or every session (`None`),
    /// freeing the arena memory. Returns how many sessions were removed.
    /// Single-flight: waits behind an in-flight generation.
    pub fn clear_sessions(&self, id: Option<&str>) -> Result<usize> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(Command::ClearSessions {
                id: id.map(str::to_string),
                reply: reply_tx,
            })
            .map_err(|_| EngineError::WorkerGone("worker thread terminated".to_string()))?;
        reply_rx
            .recv()
            .map_err(|_| EngineError::WorkerGone("worker dropped the reply".to_string()))
    }

    /// Whether the model exposes hidden states for `/v1/embeddings`.
    pub fn supports_embeddings(&self) -> bool {
        self.supports_embeddings
    }

    /// Pooling convention detected from the checkpoint
    /// (`1_Pooling/config.json`, else last-token).
    pub fn default_pooling(&self) -> Pooling {
        self.default_pooling
    }

    /// Embeds `texts` into L2-normalized vectors on the worker thread.
    /// Request-level pooling/dimensions come from `opts`; an unset pooling
    /// uses the checkpoint's detected default.
    pub fn embed_texts(&self, texts: &[String], opts: &EmbedOptions) -> Result<EmbedOutput> {
        if !self.supports_embeddings {
            return Err(EngineError::Model(combs_models::ModelError::Unsupported(
                "this model does not expose hidden states for embeddings".to_string(),
            )));
        }
        let mut opts = opts.clone();
        opts.pooling = Some(opts.pooling.unwrap_or(self.default_pooling));
        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(Command::Embed {
                texts: texts.to_vec(),
                opts,
                reply: reply_tx,
            })
            .map_err(|_| EngineError::WorkerGone("worker thread terminated".to_string()))?;
        reply_rx
            .recv()
            .map_err(|_| EngineError::WorkerGone("worker dropped the reply".to_string()))?
    }

    /// Scores `tokens` (chunked, `chunk` positions per pass; 0 = default
    /// 256) and returns the summed NLL — quantization QA. Position `p`'s
    /// logits score token `p + 1`.
    pub fn perplexity(&self, tokens: &[u32], chunk: usize) -> Result<PerplexityOutput> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(Command::Perplexity {
                tokens: tokens.to_vec(),
                chunk,
                reply: reply_tx,
            })
            .map_err(|_| EngineError::WorkerGone("worker thread terminated".to_string()))?;
        reply_rx
            .recv()
            .map_err(|_| EngineError::WorkerGone("worker dropped the reply".to_string()))?
    }

    /// Decodes one token id to its display text (specials kept) — the
    /// string OpenAI logprob alternatives carry on the wire.
    pub fn token_piece(&self, id: u32) -> String {
        self.tokenizer.decode(&[id], false).unwrap_or_default()
    }

    /// The `<|im_end|>` token id, if the tokenizer defines one (chat models).
    pub fn im_end_id(&self) -> Option<u32> {
        self.spec.special_token_id("<|im_end|>")
    }

    /// Gemma-style end-of-turn id (`<end_of_turn>`), when defined.
    pub fn end_turn_id(&self) -> Option<u32> {
        self.spec.special_token_id("<end_of_turn>")
    }

    /// Wraps (role, content) message pairs into the model's chat format,
    /// ending with the assistant turn open. Prefers the checkpoint's own
    /// Jinja chat template; falls back to the token-sniffed builtin wraps
    /// (ChatML / Gemma) when no template ships or rendering fails — a bad
    /// template logs once and degrades, it never breaks chat.
    pub fn wrap_chat(&self, messages: &[crate::ChatMessage]) -> String {
        self.wrap_chat_with_tools(messages, None)
    }

    /// [`Self::wrap_chat`] with tool definitions rendered through the
    /// model's own chat template (OpenAI-shaped schemas, passed verbatim
    /// like transformers does). The builtin fallback wraps ignore tools —
    /// they exist for models whose templates predate tool support anyway.
    pub fn wrap_chat_with_tools(
        &self,
        messages: &[crate::ChatMessage],
        tools: Option<&serde_json::Value>,
    ) -> String {
        if let Some(template) = &self.template {
            match template.render(messages, tools) {
                Ok(prompt) => return prompt,
                Err(e) => self.template_warned.call_once(|| {
                    tracing::warn!(
                        "chat template failed to render ({e}); falling back \
                         to the builtin {:?} wrap for this session",
                        self.spec.chat_template_kind()
                    );
                }),
            }
        }
        let pairs: Vec<(String, String)> = messages
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();
        self.spec.wrap_messages(&pairs)
    }

    /// Whether this model's chat template can express tool definitions —
    /// i.e. it references the `tools` context variable (qwen2.5/3,
    /// llama-3.x do; phi-3, gemma-3, smollm2 don't). Requests carrying
    /// tools should be rejected when this is false rather than silently
    /// rendered without them.
    pub fn supports_tools(&self) -> bool {
        self.spec
            .chat_template
            .as_deref()
            .is_some_and(|t| t.contains("tools"))
    }

    /// Whether this model was trained to emit `<think>` reasoning blocks —
    /// the template renders the tag or the vocab carries it as a control
    /// token (qwen3 both; qwen2.5 neither). Drives the client's reasoning
    /// affordance: a think-filter for a model that never thinks is noise.
    pub fn supports_thinking(&self) -> bool {
        self.spec
            .chat_template
            .as_deref()
            .is_some_and(|t| t.contains("<think>"))
            || self.spec.special_token_id("<think>").is_some()
    }

    /// The tool-call phrasing this model's template teaches (drives the
    /// streaming parser).
    pub fn tool_call_style(&self) -> crate::ToolCallStyle {
        crate::ToolCallStyle::detect(self.spec.chat_template.as_deref())
    }

    /// Default generation config: [`GenerationConfig::default`] merged with
    /// the model's `generation_config.json` sampler defaults. Callers clone
    /// this and override fields explicitly — explicit values always win.
    pub fn default_config(&self) -> GenerationConfig {
        self.default_config.clone()
    }

    /// Encodes text to token ids (no special tokens are added — prompts are
    /// used verbatim; wrap with a chat template yourself if needed).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Queues a generation request and streams the result: chunked prefill,
    /// then decode up to `max_tokens` or a stop condition (stop token id or
    /// stop string). `on_token` is called on the calling thread with each
    /// generated token id and its newly decoded text piece.
    ///
    /// Blocks until the request completes; concurrent callers queue behind
    /// the in-flight request (single-flight).
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        config: &GenerationConfig,
        on_token: impl FnMut(u32, &str, Option<&TokenLogprobs>),
    ) -> Result<GenerationStats> {
        self.generate_cancellable(
            prompt_tokens,
            config,
            Arc::new(AtomicBool::new(false)),
            on_token,
        )
    }

    /// [`Engine::generate`] with preprocessed images: the prompt must carry
    /// the model's image-token spans (one span per image, in order); the
    /// model splices vision-tower features in at `embed` time. Image turns
    /// never join the KV session cache (a fresh cache is used and dropped).
    pub fn generate_with_media(
        &self,
        prompt_tokens: &[u32],
        images: Vec<PixelBatch>,
        config: &GenerationConfig,
        on_token: impl FnMut(u32, &str, Option<&TokenLogprobs>),
    ) -> Result<GenerationStats> {
        self.generate_media_cancellable(
            prompt_tokens,
            images,
            config,
            Arc::new(AtomicBool::new(false)),
            on_token,
        )
    }

    /// [`Engine::generate`] with an abort flag: setting `cancel` to `true`
    /// stops the generation between tokens (checked once per decode step)
    /// and returns [`EngineError::Cancelled`].
    pub fn generate_cancellable(
        &self,
        prompt_tokens: &[u32],
        config: &GenerationConfig,
        cancel: Arc<AtomicBool>,
        on_token: impl FnMut(u32, &str, Option<&TokenLogprobs>),
    ) -> Result<GenerationStats> {
        self.generate_media_cancellable(prompt_tokens, Vec::new(), config, cancel, on_token)
    }

    /// [`Engine::generate_with_media`] with an abort flag.
    pub fn generate_media_cancellable(
        &self,
        prompt_tokens: &[u32],
        images: Vec<PixelBatch>,
        config: &GenerationConfig,
        cancel: Arc<AtomicBool>,
        mut on_token: impl FnMut(u32, &str, Option<&TokenLogprobs>),
    ) -> Result<GenerationStats> {
        // HF `add_special_tokens` semantics: prompts start with BOS when the
        // model declares one. Gemma collapses without it; templates that
        // already open with `<bos>` are detected and left alone. Qwen2
        // declares a BOS id but sets `add_bos_token: false` — honoring it
        // avoids prepending `<|endoftext|>` to every prompt.
        let mut prompt_tokens = prompt_tokens.to_vec();
        if self.spec.add_bos != Some(false) {
            if let Some(bos) = self.metadata.bos_token_id {
                if prompt_tokens.first() != Some(&bos) {
                    prompt_tokens.insert(0, bos);
                }
            }
        }
        let (pieces_tx, pieces_rx) = mpsc::channel();
        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(Command::Generate(Box::new(GenerateRequest {
                prompt_tokens,
                images,
                config: config.clone(),
                cancel,
                pieces: pieces_tx,
                reply: reply_tx,
            })))
            .map_err(|_| EngineError::WorkerGone("worker thread terminated".to_string()))?;

        // Stream pieces until the worker closes the channel, then collect
        // the final result (already sent by then).
        while let Ok((id, piece, logprobs)) = pieces_rx.recv() {
            on_token(id, &piece, logprobs.as_ref());
        }
        reply_rx
            .recv()
            .map_err(|_| EngineError::WorkerGone("worker dropped the reply".to_string()))?
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Shutdown);
        if let Some(handle) = self
            .worker
            .lock()
            .ok()
            .and_then(|mut guard| guard.take())
        {
            let _ = handle.join();
        }
    }
}

/// Worker-thread loop: executes queued requests serially until shutdown.
fn worker_loop(
    mut model: Box<dyn GenerativeModel<CombsBackend>>,
    tokenizer: Tokenizer,
    device: CombsDevice,
    cache_config: CacheConfig,
    max_position_embeddings: usize,
    rx: mpsc::Receiver<Command>,
    stats: Arc<Mutex<EngineStatsSnapshot>>,
) {
    // Rolling KV sessions — survive across requests so multi-turn callers
    // (and named per-agent sessions) only prefill the new prompt suffix.
    let mut sessions = SessionSet::new();
    // Token→bytes table for constrained decoding: derived from the
    // tokenizer on the first constrained request, then reused for the
    // worker's lifetime (vocab-sized, so built once, never per request).
    let mut token_table: Option<TokenByteTable> = None;
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Command::Shutdown => break,
            Command::Generate(req) => {
                let result = run_generation(
                    model.as_mut(),
                    &tokenizer,
                    &device,
                    &cache_config,
                    max_position_embeddings,
                    &req,
                    &mut sessions,
                    &mut token_table,
                );
                update_stats(&stats, &result, &sessions, &device);
                // `req.pieces` closes when `req` drops at the end of this
                // iteration; the reply is queued first, so the caller always
                // sees all pieces followed by the result.
                let _ = req.reply.send(result);
            }
            Command::Embed { texts, opts, reply } => {
                let result = run_embed(
                    model.as_mut(),
                    &tokenizer,
                    &device,
                    &cache_config,
                    max_position_embeddings,
                    &texts,
                    &opts,
                );
                let _ = reply.send(result);
            }
            Command::Perplexity {
                tokens,
                chunk,
                reply,
            } => {
                let result = run_perplexity(
                    model.as_mut(),
                    &device,
                    &cache_config,
                    max_position_embeddings,
                    &tokens,
                    chunk,
                );
                let _ = reply.send(result);
            }
            Command::ClearSessions { id, reply } => {
                let removed = match id {
                    Some(key) => sessions.take(&key).map(|_| 1).unwrap_or(0),
                    None => sessions.clear_all(),
                };
                if let Ok(mut snap) = stats.lock() {
                    snap.sessions = session_infos(&sessions);
                }
                let _ = reply.send(removed);
            }
        }
    }
}

/// Prompt tokens per `prefill_hidden` call on the embeddings path — the
/// same attention-memory bound chunked generation prefill uses.
const EMBED_CHUNK: usize = 512;

/// Executes one embeddings request on the worker thread.
///
/// Each text runs on a fresh contiguous cache (dropped afterwards — no
/// session interaction) through the model's hidden-state path, chunked
/// like generation prefill. Pooling accumulates across chunks: last-token
/// keeps the final chunk's last position, mean keeps a running sum (the
/// final L2 normalization makes the 1/n scale irrelevant). Tokenization
/// is the tokenizer's own (post-processor included): embedding
/// checkpoints bake their bos/eos convention into `tokenizer.json`, so
/// no engine-side BOS logic applies.
fn run_embed(
    model: &mut dyn GenerativeModel<CombsBackend>,
    tokenizer: &Tokenizer,
    device: &CombsDevice,
    cache_config: &CacheConfig,
    max_position_embeddings: usize,
    texts: &[String],
    opts: &EmbedOptions,
) -> Result<EmbedOutput> {
    let pooling = opts.pooling.unwrap_or(Pooling::Last);
    let mut vectors = Vec::with_capacity(texts.len());
    let mut prompt_tokens = 0usize;

    let embed_cache_config = CacheConfig {
        kind: CacheKind::Contiguous,
        ..*cache_config
    };

    for text in texts {
        let ids: Vec<u32> = tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec();
        if ids.is_empty() {
            return Err(EngineError::Tokenizer("empty embedding input".to_string()));
        }
        check_context_len(
            ids.len(),
            0,
            cache_config.max_seq_len.min(max_position_embeddings),
        )?;
        prompt_tokens += ids.len();

        let data: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
        let tokens = Tensor::from_data(TensorData::new(data, [1, ids.len()]), device);
        let embedded = model.embed(tokens);

        let mut cache = model.create_kv_cache(&embed_cache_config);
        let mut last_hidden: Option<Tensor<CombsBackend, 3>> = None;
        let mut sum_hidden: Option<Tensor<CombsBackend, 3>> = None;
        let mut offset = 0usize;
        while offset < ids.len() {
            let len = EMBED_CHUNK.min(ids.len() - offset);
            let input = embedded.clone().narrow(1, offset, len);
            let start = offset as u32;
            let hidden = model.prefill_hidden(input, cache.as_mut(), start..start + len as u32)?;
            match pooling {
                Pooling::Last => last_hidden = Some(hidden.narrow(1, len - 1, 1)),
                Pooling::Mean => {
                    let chunk_sum = hidden.sum_dim(1);
                    sum_hidden = Some(match sum_hidden.take() {
                        Some(acc) => acc + chunk_sum,
                        None => chunk_sum,
                    });
                }
            }
            offset += len;
        }

        let pooled = match pooling {
            Pooling::Last => last_hidden.expect("nonempty text ran at least one chunk"),
            Pooling::Mean => sum_hidden.expect("nonempty text ran at least one chunk"),
        };
        let mut v = pooled
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|e| EngineError::Readback(format!("hidden state must be f32: {e:?}")))?;
        if let Some(n) = opts.dimensions {
            if n == 0 || n > v.len() {
                return Err(EngineError::Model(combs_models::ModelError::Unsupported(
                    format!("dimensions must be 1..={}, got {n}", v.len()),
                )));
            }
            v.truncate(n);
        }
        l2_normalize(&mut v);
        vectors.push(v);
    }

    Ok(EmbedOutput {
        vectors,
        prompt_tokens,
    })
}

/// Evaluates perplexity of a token sequence on the worker thread.
///
/// Chunked prefill through the model's all-positions logits head; the
/// per-token NLL is computed on device (log-sum-exp minus the target
/// logit) so only one scalar per position is read back. Position `p`'s
/// logits score token `p + 1`; the final position of the last chunk has
/// no target and is dropped.
fn run_perplexity(
    model: &mut dyn GenerativeModel<CombsBackend>,
    device: &CombsDevice,
    cache_config: &CacheConfig,
    max_position_embeddings: usize,
    tokens: &[u32],
    chunk: usize,
) -> Result<PerplexityOutput> {
    if tokens.len() < 2 {
        return Err(EngineError::Tokenizer(
            "perplexity needs at least 2 tokens".to_string(),
        ));
    }
    check_context_len(
        tokens.len(),
        0,
        cache_config.max_seq_len.min(max_position_embeddings),
    )?;
    let chunk = if chunk == 0 { 256 } else { chunk };

    let data: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let all = Tensor::from_data(TensorData::new(data, [1, tokens.len()]), device);
    let embedded = model.embed(all);

    let mut cache = model.create_kv_cache(&CacheConfig {
        kind: CacheKind::Contiguous,
        ..*cache_config
    });

    let mut nll_sum = 0.0f64;
    let mut scored = 0usize;
    let mut offset = 0usize;
    while offset < tokens.len() {
        let len = chunk.min(tokens.len() - offset);
        let input = embedded.clone().narrow(1, offset, len);
        let start = offset as u32;
        let logits = model.prefill_all_logits(input, cache.as_mut(), start..start + len as u32)?;
        let [_, l, vocab] = logits.dims();

        // Targets for positions offset..offset+len are tokens shifted by
        // one; the sequence-final position has none.
        let n_score = if offset + len == tokens.len() { len - 1 } else { len };
        if n_score == 0 {
            break;
        }
        let logits = logits.narrow(1, 0, n_score);
        let targets: Vec<i32> = tokens[offset + 1..offset + 1 + n_score]
            .iter()
            .map(|&t| t as i32)
            .collect();
        let idx = Tensor::<CombsBackend, 3, burn::tensor::Int>::from_data(
            TensorData::new(targets, [1, n_score, 1]),
            device,
        );

        // lse - target_logit, all on device; readback is [n_score] floats.
        let max = logits.clone().max_dim(2);
        let lse = (logits.clone() - max.clone().expand([1, n_score, vocab]))
            .exp()
            .sum_dim(2)
            .log()
            + max;
        let target = logits.gather(2, idx);
        let nll = (lse - target).reshape([n_score]);
        let host = nll
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|e| EngineError::Readback(format!("nll must be f32: {e:?}")))?;
        nll_sum += host.iter().map(|&v| v as f64).sum::<f64>();
        scored += n_score;
        offset += len;
    }

    Ok(PerplexityOutput {
        nll_sum,
        tokens: scored,
    })
}

/// In-place L2 normalization (no-op on a zero vector).
fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
}

/// Detects the checkpoint's pooling convention: sentence-transformers
/// layouts ship `1_Pooling/config.json` next to the weights; absent (or
/// unrecognized), decoder embedding models pool the last token.
fn detect_pooling(tokenizer_json: &std::path::Path) -> Pooling {
    let path = match tokenizer_json.parent() {
        Some(dir) => dir.join("1_Pooling").join("config.json"),
        None => return Pooling::Last,
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Pooling::Last;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return Pooling::Last;
    };
    if v.get("pooling_mode_mean_tokens").and_then(serde_json::Value::as_bool) == Some(true) {
        return Pooling::Mean;
    }
    Pooling::Last
}

/// Refreshes the shared snapshot after a generation. Runs on the worker so
/// the GPU allocator sample (`submit_blocking`) never contends with an
/// in-flight generation; observers only ever pay a mutex clone.
fn session_infos(sessions: &SessionSet) -> Vec<SessionInfo> {
    sessions
        .map
        .iter()
        .map(|(k, s)| SessionInfo {
            id: if k.is_empty() {
                "(anonymous)".to_string()
            } else {
                k.clone()
            },
            history_len: s.history.len(),
            pages: s.cache.page_stats(),
        })
        .collect()
}

fn update_stats(
    stats: &Mutex<EngineStatsSnapshot>,
    result: &Result<GenerationStats>,
    sessions: &SessionSet,
    device: &CombsDevice,
) {
    let gpu = combs_core::gpu_memory(device);
    let session_infos: Vec<SessionInfo> = session_infos(sessions);

    let Ok(mut snap) = stats.lock() else { return };
    snap.requests_total += 1;
    match result {
        Ok(st) => {
            snap.prompt_tokens_total += st.prompt_tokens as u64;
            snap.generated_tokens_total += st.generated_tokens as u64;
            snap.cached_tokens_total += st.cached_tokens as u64;
            let decode = st.decode_tokens_per_second();
            if decode > 0.0 {
                snap.decode_tok_s_ewma = if snap.decode_tok_s_ewma == 0.0 {
                    decode
                } else {
                    0.7 * snap.decode_tok_s_ewma + 0.3 * decode
                };
            }
            snap.last = Some(LastGeneration {
                prompt_tokens: st.prompt_tokens,
                generated_tokens: st.generated_tokens,
                cached_tokens: st.cached_tokens,
                ttft_ms: st.ttft.as_secs_f64() * 1000.0,
                prefill_tok_s: st.prefill_tokens_per_second(),
                decode_tok_s: decode,
                total_ms: st.total_time.as_secs_f64() * 1000.0,
                cache_pages_used: st.cache_pages_used,
            });
        }
        // A client hanging up is a normal outcome, not an engine fault.
        // Counting cancels as errors made every stopped turn look like a
        // failure in /v1/stats.
        Err(EngineError::Cancelled) => snap.cancelled_total += 1,
        Err(_) => snap.errors_total += 1,
    }
    snap.sessions = session_infos;
    snap.session_evictions = sessions.evictions;
    snap.gpu = gpu;
}

/// Executes one generation request on the worker thread.
///
/// Rolling-session prefix reuse: when the request allows it (paged cache,
/// `session_reuse`), the previous request's KV cache is rolled back to the
/// longest common token prefix with the new prompt and only the suffix is
/// prefilled. The session is saved on every exit path after prefill, so a
/// cancelled request still leaves a consistent cache behind.
#[allow(clippy::too_many_arguments)]
fn run_generation(
    model: &mut dyn GenerativeModel<CombsBackend>,
    tokenizer: &Tokenizer,
    device: &CombsDevice,
    cache_config: &CacheConfig,
    max_position_embeddings: usize,
    req: &GenerateRequest,
    sessions: &mut SessionSet,
    token_table: &mut Option<TokenByteTable>,
) -> Result<GenerationStats> {
    let prompt_tokens = &req.prompt_tokens;
    let config = &req.config;
    if prompt_tokens.is_empty() {
        return Err(EngineError::Tokenizer("empty prompt".to_string()));
    }
    // Context budget: enforced against the cache capacity, which is itself
    // capped by the model's positional limit.
    check_context_len(
        prompt_tokens.len(),
        config.max_tokens,
        cache_config.max_seq_len.min(max_position_embeddings),
    )?;

    // Image turns never join the KV session cache: token prefixes would
    // collide with different image contents, so media requests run on a
    // fresh cache and are not saved back.
    let has_media = !req.images.is_empty();
    let reuse = config.session_reuse && cache_config.kind == CacheKind::Paged && !has_media;
    let key = config.session_id.clone().unwrap_or_default();

    // Longest common prefix with this session's previous request. Capped at
    // prompt.len() - 1 so at least one token is always prefilled (the last
    // position's logits are needed to start decoding, even for an
    // identical repeated prompt).
    let mut lcp = 0usize;
    let mut cache = if reuse {
        match sessions.take(&key) {
            Some(mut s) => {
                let shared = common_prefix(&s.history, prompt_tokens)
                    .min(prompt_tokens.len().saturating_sub(1));
                if shared > 0 {
                    let need = s.history.len() - shared;
                    let popped = s.cache.popn(need);
                    if popped == need {
                        // Invariant: lcp == cache seq_len == history len.
                        s.history.truncate(shared);
                        lcp = shared;
                        s.cache
                    } else {
                        // The cache cannot roll back to the shared prefix
                        // (sliding-window layers past eviction): a partial
                        // rollback would leave divergent tokens cached, so
                        // rebuild fresh. Pure extensions (need == 0) never
                        // hit this arm.
                        model.create_kv_cache(cache_config)
                    }
                } else {
                    model.create_kv_cache(cache_config)
                }
            }
            None => model.create_kv_cache(cache_config),
        }
    } else {
        model.create_kv_cache(cache_config)
    };

    let eos_ids = req_eos_ids(model, config);
    // Penalties must never touch the stop tokens: `<|im_end|>` appears
    // once per prior turn in a chat prompt, so an unexempted penalty
    // suppresses the model's ability to end its turn — and gets worse
    // the longer the conversation runs.
    let mut sampler: Box<dyn Sampler> =
        sampler_from_params_exempting(&config.sampling, &eos_ids);
    let mut stop = StopDetector::new(eos_ids.clone(), config.stop_strings.clone());
    // Structured output: compile the schema at request time and bind the
    // automaton to this model's token table (built lazily, cached on the
    // worker). The mask runs engine-side, not in the sampler's processor
    // chain — the greedy sampler skips that chain, and a constraint must
    // hold under every sampler.
    let mut constraint = match &config.constraint {
        Some(spec) => {
            let schema = spec.compile().map_err(EngineError::Constraint)?;
            let table = token_table.get_or_insert_with(|| TokenByteTable::build(tokenizer));
            Some(ConstraintState::new(schema, table, eos_ids))
        }
        None => None,
    };
    let mut detok = IncrementalDetokenizer::new();
    let mut history: Vec<u32> = prompt_tokens.to_vec();

    let t_start = Instant::now();

    // --- chunked prefill of the suffix (prompt[lcp..]) ---------------------
    let chunk = if config.prefill_chunk_size == 0 {
        usize::MAX
    } else {
        config.prefill_chunk_size
    };
    let suffix = &prompt_tokens[lcp..];
    let data: Vec<i32> = suffix.iter().map(|&t| t as i32).collect();
    let tokens = Tensor::from_data(TensorData::new(data, [1, suffix.len()]), device);
    let embedded = if has_media {
        let pixels: Vec<Tensor<CombsBackend, 4>> = req
            .images
            .iter()
            .map(|pb| pixels_to_tensor(pb.data.clone(), pb.shape(), device))
            .collect();
        model.embed_multimodal(tokens, &pixels)?
    } else {
        model.embed(tokens)
    };

    let mut offset = 0;
    let mut logits = None;
    while offset < suffix.len() {
        let len = chunk.min(suffix.len() - offset);
        let input = embedded.clone().narrow(1, offset, len);
        let start = (lcp + offset) as u32;
        logits = Some(model.prefill(input, cache.as_mut(), start..start + len as u32));
        offset += len;
    }
    let logits = logits.expect("nonempty suffix runs at least one chunk");

    let mut row = readback_logits(logits)?;
    if let Some(c) = constraint.as_mut() {
        if c.mask(&mut row) == 0 {
            return Err(EngineError::Constraint(DEAD_END.to_string()));
        }
    }
    let outcome = sampler.sample(&mut row, &history);
    let mut next = outcome.token;
    let mut next_logprobs = outcome.logprobs;
    if let Some(c) = constraint.as_mut() {
        c.advance(next);
    }
    let ttft = t_start.elapsed();
    let t_decode = Instant::now();
    let mut trace = DecodeTrace::from_env();

    // Speculative decoding (prompt-lookup drafts + multi-token verify):
    // greedy-only and only where raw argmax IS the sampler (no penalties,
    // bias, or logprob capture), never under constraints, and only on
    // non-sliding models with a paged cache — there `popn` rollback is
    // always total. Every other request takes the normal path untouched.
    let spec_active = spec_enabled()
        && constraint.is_none()
        && config.sampling.temperature <= 0.0
        && config.sampling.repetition_penalty.map_or(true, |v| v == 1.0)
        && config.sampling.frequency_penalty.map_or(true, |v| v == 0.0)
        && config.sampling.presence_penalty.map_or(true, |v| v == 0.0)
        && config.sampling.logit_bias.is_none()
        && config.sampling.logprobs.is_none()
        && model.supports_decode_all_logits()
        && model.metadata().attention_pattern.sliding_window.is_none()
        && cache.pages_used().is_some();
    // Pre-verified tokens waiting to be emitted, and how many cache rows
    // sit beyond `decoded` (drained back to zero as pending empties).
    let mut pending: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
    let mut cache_ahead = 0usize;
    let mut spec_cooldown = 0usize;

    // Tokens actually fed through `decode` (i.e. present in the KV cache).
    // The final sampled token and stop tokens never enter the cache.
    let mut decoded: Vec<u32> = Vec::new();
    let mut generated = 0usize;
    let mut loop_error: Option<EngineError> = None;

    for _ in 0..config.max_tokens {
        if req.cancel.load(Ordering::Relaxed) {
            loop_error = Some(EngineError::Cancelled);
            break;
        }
        if stop.is_stop_token(next) {
            break;
        }
        history.push(next);
        let piece = match detok.push(tokenizer, next) {
            Ok(p) => p,
            Err(e) => {
                loop_error = Some(e);
                break;
            }
        };
        // Emit the piece, truncating at a stop string if one completes.
        match stop.push_text(&piece) {
            Some(cut) => {
                if cut > 0
                    && req
                        .pieces
                        .send((next, piece[..cut].to_string(), next_logprobs.take()))
                        .is_err()
                {
                    loop_error = Some(EngineError::Cancelled); // caller hung up
                }
                generated += 1;
                break;
            }
            None => {
                if req.pieces.send((next, piece, next_logprobs.take())).is_err() {
                    loop_error = Some(EngineError::Cancelled); // caller hung up
                    break;
                }
            }
        }
        generated += 1;

        if generated >= config.max_tokens {
            break;
        }
        if let Some(t) = pending.pop_front() {
            // Pre-verified by the last speculative round: `next` is already
            // in the cache and `t` was proven to follow it.
            decoded.push(next);
            cache_ahead -= 1;
            next = t;
            next_logprobs = None;
        } else if let Some(draft) = (spec_active && spec_cooldown == 0)
            .then(|| spec::propose(&history, SPEC_DRAFT, SPEC_MIN_NGRAM, SPEC_MAX_NGRAM))
            .flatten()
        {
            // Feed [next, draft…] in one pass; row i's argmax is the
            // model's true next token after position i, so the longest
            // matching prefix of the draft is exactly what greedy decoding
            // would have produced one at a time.
            let k = draft.len();
            let mut ids: Vec<i32> = Vec::with_capacity(k + 1);
            ids.push(next as i32);
            ids.extend(draft.iter().map(|&t| t as i32));
            let input =
                model.embed(Tensor::from_data(TensorData::new(ids, [1, k + 1]), device));
            let all = match model.decode_all_logits(input, cache.as_mut()) {
                Ok(t) => t,
                Err(e) => {
                    loop_error = Some(e.into());
                    break;
                }
            };
            let rows: Vec<f32> = match all.into_data().convert::<f32>().to_vec::<f32>() {
                Ok(v) => v,
                Err(e) => {
                    loop_error = Some(EngineError::Readback(format!("spec logits: {e:?}")));
                    break;
                }
            };
            let vocab = rows.len() / (k + 1);
            let argmax = |r: usize| -> u32 {
                rows[r * vocab..(r + 1) * vocab]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            };
            let mut accepted = 0usize;
            while accepted < k && argmax(accepted) == draft[accepted] {
                accepted += 1;
            }
            let correction = argmax(accepted.min(k));
            // Roll the rejected tail out of the cache. The static guards
            // make a partial rollback impossible; treat one as fatal
            // rather than continuing on divergent state.
            let unwanted = k - accepted;
            if unwanted > 0 {
                let popped = cache.popn(unwanted);
                if popped != unwanted {
                    loop_error = Some(EngineError::Readback(format!(
                        "speculative rollback popped {popped} of {unwanted} rows"
                    )));
                    break;
                }
            }
            decoded.push(next);
            cache_ahead += accepted;
            spec_cooldown = if accepted == 0 { SPEC_BACKOFF } else { 0 };
            pending.extend(draft[..accepted].iter().copied());
            pending.push_back(correction);
            next = pending.pop_front().expect("correction token is always queued");
            next_logprobs = None;
        } else {
            spec_cooldown = spec_cooldown.saturating_sub(1);
            trace.step_start();
            let data = vec![next as i32];
            let input = model.embed(Tensor::from_data(TensorData::new(data, [1, 1]), device));
            let logits = model.decode(input, cache.as_mut());
            trace.submitted();
            decoded.push(next); // `next` is now in the KV cache
            let mut row = match readback_logits(logits) {
                Ok(r) => r,
                Err(e) => {
                    loop_error = Some(e);
                    break;
                }
            };
            trace.read_back();
            if let Some(c) = constraint.as_mut() {
                if c.mask(&mut row) == 0 {
                    loop_error = Some(EngineError::Constraint(DEAD_END.to_string()));
                    break;
                }
            }
            let outcome = sampler.sample(&mut row, &history);
            next = outcome.token;
            next_logprobs = outcome.logprobs;
            if let Some(c) = constraint.as_mut() {
                c.advance(next);
            }
            trace.sampled();
        }
    }

    // Release any character the detokenizer held back (an incomplete
    // UTF-8 sequence at the very end of generation).
    if let Ok(tail) = detok.flush(tokenizer) {
        if !tail.is_empty() {
            let _ = req.pieces.send((0, tail, None));
        }
    }

    // Pre-verified rows that never got emitted must leave the cache before
    // the session snapshot (history must mirror KV contents exactly).
    if cache_ahead > 0 {
        let popped = cache.popn(cache_ahead);
        debug_assert_eq!(popped, cache_ahead, "speculative exit rollback");
    }

    // Save the rolling session: history must mirror KV contents exactly
    // (prompt + decoded tokens — nothing else).
    let cache_pages_used = cache.pages_used().unwrap_or(0);
    if reuse {
        let mut hist = prompt_tokens.clone();
        hist.extend_from_slice(&decoded);
        sessions.put(
            key,
            SessionState {
                history: hist,
                cache,
                last_used: 0,
            },
        );
    }

    trace.report();
    let decode_time = t_decode.elapsed();
    let stats = GenerationStats {
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: generated,
        ttft,
        decode_time,
        total_time: t_start.elapsed(),
        cache_pages_used,
        cached_tokens: lcp,
    };
    match loop_error {
        Some(e) => Err(e),
        None => Ok(stats),
    }
}

/// Draft length proposed per speculative round.
const SPEC_DRAFT: usize = 6;
/// Shortest history suffix n-gram accepted as a draft trigger — bigrams
/// recur incidentally in all text and produce failed rounds that cost more
/// than they save.
const SPEC_MIN_NGRAM: usize = 3;
/// Longest history suffix n-gram searched for a recurrence.
const SPEC_MAX_NGRAM: usize = 8;
/// Tokens decoded normally after a fully-rejected round before drafting
/// again (a miss usually means this region of text is not repetitive).
const SPEC_BACKOFF: usize = 16;

/// Speculative decoding ships opt-in behind `COMBS_SPEC=1` until its
/// measured speedup clears the bar for default-on; checked once.
fn spec_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("COMBS_SPEC").as_deref(), Ok("1")))
}

/// Error text for a constraint dead end (no token can legally continue).
const DEAD_END: &str =
    "no vocabulary token can continue the constrained JSON output (schema dead end)";

/// Collects the eos ids that apply to a request.
fn req_eos_ids(
    model: &dyn GenerativeModel<CombsBackend>,
    config: &GenerationConfig,
) -> Vec<u32> {
    model
        .metadata()
        .eos_token_ids
        .iter()
        .chain(config.stop_token_ids.iter())
        .copied()
        .collect()
}

/// Reads a `[1, vocab]` logits row back to the host (one copy per step).
fn readback_logits(logits: Tensor<CombsBackend, 2>) -> Result<Vec<f32>> {
    logits
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|e| EngineError::Readback(format!("logits must be f32: {e:?}")))
}

/// Per-step decode timing behind `COMBS_TRACE_DECODE=1`: accumulates the
/// wall split of kernel submission (embed + decode enqueue), logits
/// readback (the host sync that absorbs GPU execution time), and host-side
/// mask + sample, then prints one stderr summary line per generation.
/// Inert when the env var is unset.
struct DecodeTrace {
    enabled: bool,
    t: Instant,
    submit: Duration,
    readback: Duration,
    sample: Duration,
    steps: u32,
}

impl DecodeTrace {
    fn from_env() -> Self {
        DecodeTrace {
            enabled: matches!(std::env::var("COMBS_TRACE_DECODE").as_deref(), Ok("1")),
            t: Instant::now(),
            submit: Duration::ZERO,
            readback: Duration::ZERO,
            sample: Duration::ZERO,
            steps: 0,
        }
    }

    fn step_start(&mut self) {
        if self.enabled {
            self.t = Instant::now();
        }
    }

    fn submitted(&mut self) {
        if self.enabled {
            let now = Instant::now();
            self.submit += now - self.t;
            self.t = now;
        }
    }

    fn read_back(&mut self) {
        if self.enabled {
            let now = Instant::now();
            self.readback += now - self.t;
            self.t = now;
        }
    }

    fn sampled(&mut self) {
        if self.enabled {
            self.sample += self.t.elapsed();
            self.steps += 1;
        }
    }

    fn report(&self) {
        if !self.enabled || self.steps == 0 {
            return;
        }
        let total = self.submit + self.readback + self.sample;
        let per = |d: Duration| d.as_secs_f64() * 1e3 / self.steps as f64;
        let pct = |d: Duration| 100.0 * d.as_secs_f64() / total.as_secs_f64().max(f64::EPSILON);
        eprintln!(
            "[trace] decode: {} steps, {:.2} ms/step — submit {:.2} ms ({:.0}%), readback {:.2} ms ({:.0}%), sample {:.3} ms ({:.0}%)",
            self.steps,
            per(total),
            per(self.submit),
            pct(self.submit),
            per(self.readback),
            pct(self.readback),
            per(self.sample),
            pct(self.sample),
        );
    }
}

/// Merges `generation_config.json` sampler defaults over the built-in
/// defaults. `None` fields keep the built-in default (greedy, no filters).
fn default_config_from(sampler: Option<&SamplerConfig>) -> GenerationConfig {
    let mut config = GenerationConfig::default();
    if let Some(sd) = sampler {
        if let Some(t) = sd.temperature {
            config.sampling.temperature = t;
        }
        config.sampling.top_k = sd.top_k;
        config.sampling.top_p = sd.top_p;
        config.sampling.repetition_penalty = sd.repetition_penalty;
        if let Some(m) = sd.max_new_tokens {
            config.max_tokens = m;
        }
    }
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_guard_rejects_overflow() {
        let err = check_context_len(8000, 200, 8192).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("8000"), "{msg}");
        assert!(msg.contains("8192"), "{msg}");
    }

    #[test]
    fn context_guard_accepts_exact_fit() {
        assert!(check_context_len(8000, 192, 8192).is_ok());
    }

    #[test]
    fn stats_throughputs() {
        let stats = GenerationStats {
            prompt_tokens: 100,
            generated_tokens: 11,
            ttft: Duration::from_millis(200),
            decode_time: Duration::from_secs(2),
            total_time: Duration::from_millis(2200),
            cache_pages_used: 7,
            cached_tokens: 42,
        };
        assert_eq!(stats.prefill_tokens_per_second(), 500.0);
        assert_eq!(stats.tokens_per_second(), 5.0);
        assert_eq!(stats.decode_tokens_per_second(), 5.0);
    }

    #[test]
    fn sampler_defaults_merge_into_config() {
        let sd = SamplerConfig {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            repetition_penalty: Some(1.1),
            max_new_tokens: Some(128),
        };
        let config = default_config_from(Some(&sd));
        assert_eq!(config.sampling.temperature, 0.7);
        assert_eq!(config.sampling.top_p, Some(0.9));
        assert_eq!(config.sampling.top_k, Some(40));
        assert_eq!(config.sampling.repetition_penalty, Some(1.1));
        assert_eq!(config.max_tokens, 128);
    }

    #[test]
    fn empty_sampler_defaults_keep_greedy_default() {
        let config = default_config_from(Some(&SamplerConfig::default()));
        assert_eq!(config.sampling.temperature, 0.0);
        assert_eq!(config.max_tokens, 64);
        assert_eq!(config.prefill_chunk_size, 512);
    }

    #[test]
    fn engine_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Engine>();
    }

    #[test]
    fn common_prefix_cases() {
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 4]), 2);
        assert_eq!(common_prefix(&[1, 2], &[1, 2, 3, 4]), 2);
        assert_eq!(common_prefix(&[1, 2, 3, 4], &[1, 2]), 2);
        assert_eq!(common_prefix(&[1], &[2]), 0);
        assert_eq!(common_prefix(&[], &[1]), 0);
        assert_eq!(common_prefix(&[1], &[]), 0);
    }
}
