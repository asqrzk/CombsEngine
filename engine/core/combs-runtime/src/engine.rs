//! The generation engine.
//!
//! Single-flight request queue (LiteRT-LM ExecutionQueue pattern): the model
//! and KV cache live on one worker thread; `generate` (callable on a shared
//! `&Engine`, e.g. via `Arc<Engine>`) pushes a request onto an internal
//! `mpsc` queue and streams pieces back over a channel. Requests are
//! executed strictly serially — this is the seam Phase 3's threaded engine
//! and scheduler will hang off.

use std::time::Duration;

use combs_formats::SamplerConfig;

use crate::constraint::ConstraintSpec;
use crate::sampler::SamplingParams;
use crate::{EngineError, Result};

// The threaded driver's own imports. Kept apart from the shared ones above
// so that cfg-ing the driver out does not leave a dozen unused names
// behind — the boundary is visible here rather than repeated per line.
#[cfg(not(target_family = "wasm"))]
use native_driver_imports::*;
#[cfg(not(target_family = "wasm"))]
mod native_driver_imports {
    pub(super) use std::sync::atomic::AtomicBool;
    pub(super) use std::sync::{Arc, Mutex, mpsc};
    pub(super) use std::thread::JoinHandle;

    pub(super) use burn::tensor::{Tensor, TensorData, backend::Backend as _};
    pub(super) use combs_core::{BufferPool, CombsBackend, CombsDevice};
    pub(super) use combs_formats::{ModelMetadata, ModelSource};
    pub(super) use combs_media::PixelBatch;
    pub(super) use combs_models::{CacheConfig, CacheKind, GenerativeModel, ModelRegistry};
    pub(super) use tokenizers::Tokenizer;

    pub(super) use crate::constraint::TokenByteTable;
    pub(super) use crate::sampler::TokenLogprobs;
    pub(super) use crate::step::{self, MAX_SESSIONS, SessionSet, Submitted, begin_generation};
}

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

// --- the threaded driver ------------------------------------------------
//
// Everything from here to `run_generation` is one engine shape: a worker
// thread that owns the model and blocks on each readback, fed over an mpsc
// queue. A browser has no thread to give it and no blocking readback to
// give it either, so rather than ship an API that compiles there and fails
// there, the whole driver is native. The browser's engine is
// `crate::LocalEngine`, and both are drivers over the same `crate::step`.
#[cfg(not(target_family = "wasm"))]
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

#[cfg(not(target_family = "wasm"))]
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
    /// Load-time estimate of resident weight bytes + the full KV arena —
    /// what the S2c budget refusal reasons over, exported so observers
    /// (and the footprint proof) can hold the allocator to it.
    pub estimated_model_bytes: u64,
    /// The weights alone — the part that must be live right after load
    /// (the arena grows lazily toward the rest).
    pub estimated_weight_bytes: u64,
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

#[cfg(not(target_family = "wasm"))]
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

#[cfg(not(target_family = "wasm"))]
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
        // S2c: refuse what cannot fit BEFORE any allocation. The paged
        // arena math below is the same the observability snapshot uses.
        let meta_pre = source.metadata();
        let elem_bytes_pre: u64 = if cfg!(feature = "f16") { 2 } else { 4 };
        let pattern_pre = &meta_pre.attention_pattern;
        let globals = (0..meta_pre.num_hidden_layers)
            .filter(|&i| pattern_pre.is_global_layer(i))
            .count()
            .max(1);
        let kv_bytes_per_value_x8 = if cache_config.quantize_kv {
            9
        } else {
            elem_bytes_pre * 8
        };
        let kv_bytes = cache_config.max_seq_len as u64
            * meta_pre.num_key_value_heads as u64
            * meta_pre.head_dim as u64
            * kv_bytes_per_value_x8
            / 8
            * 2
            * globals as u64;
        let (weight_bytes, largest) = estimate_weight_bytes(source, elem_bytes_pre);
        if let Some(caps_limit) = binding_limit(&device) {
            if largest.1 > caps_limit {
                return Err(EngineError::TensorExceedsDevice {
                    tensor: largest.0,
                    bytes: largest.1,
                    limit: caps_limit,
                });
            }
        }
        if let Ok(budget_mb) = std::env::var("COMBS_VRAM_BUDGET_MB") {
            if let Ok(budget_mb) = budget_mb.parse::<u64>() {
                let slack = ((weight_bytes + kv_bytes) / 10).max(256 << 20);
                let total = weight_bytes + kv_bytes + slack;
                if total > budget_mb << 20 {
                    return Err(EngineError::OverBudget {
                        weights_mb: weight_bytes >> 20,
                        kv_mb: kv_bytes >> 20,
                        slack_mb: slack >> 20,
                        total_mb: total >> 20,
                        budget_mb,
                    });
                }
            }
        }

        let registry = ModelRegistry::<CombsBackend>::new();
        let pool = BufferPool::new();
        let mount = combs_core::provenance::turn(
            "engine",
            "mount",
            &[
                ("arch", meta_pre.architecture.clone()),
                ("layers", meta_pre.num_hidden_layers.to_string()),
                ("kv_heads", meta_pre.num_key_value_heads.to_string()),
                ("head_dim", meta_pre.head_dim.to_string()),
                ("max_seq_len", cache_config.max_seq_len.to_string()),
                ("kv_quant", cache_config.quantize_kv.to_string()),
                ("weights_mb", (weight_bytes >> 20).to_string()),
                ("kv_mb", (kv_bytes >> 20).to_string()),
                ("largest_tensor", format!("{} ({} MB)", largest.0, largest.1 >> 20)),
            ],
        );
        let model = pool.pin_persistent(&device, || registry.load(source, &device))?;
        // The load path's transients (dequant staging, repack scratch) are
        // garbage from here on; without this the pool holds them forever.
        pool.cleanup(&device);
        let supports_embeddings = model.supports_hidden_states();
        mount.ok(&[("embeddings", supports_embeddings.to_string())]);

        let spec = source.tokenizer()?;
        let default_pooling = detect_pooling(spec.json_dir());
        let tokenizer = Tokenizer::from_bytes(spec.json_bytes()?)
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
            estimated_model_bytes: weight_bytes + kv_bytes,
            estimated_weight_bytes: weight_bytes,
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
        let messages = crate::template::sanitize_history(messages);
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

#[cfg(not(target_family = "wasm"))]
/// The threaded engine answers chat requests the same way every engine
/// does — see [`crate::request`], where that meaning lives.
impl crate::request::ChatHost for Engine {
    fn wrap_chat_with_tools(
        &self,
        messages: &[crate::ChatMessage],
        tools: Option<&serde_json::Value>,
    ) -> String {
        Engine::wrap_chat_with_tools(self, messages, tools)
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Engine::encode(self, text)
    }

    fn default_config(&self) -> GenerationConfig {
        Engine::default_config(self)
    }

    fn im_end_id(&self) -> Option<u32> {
        Engine::im_end_id(self)
    }

    fn tool_call_style(&self) -> crate::ToolCallStyle {
        Engine::tool_call_style(self)
    }
}

#[cfg(not(target_family = "wasm"))]
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

#[cfg(not(target_family = "wasm"))]
/// Estimates the bytes each weight tensor holds resident after load,
/// summed, plus the single largest — the honest input to the S2c
/// refusal. Quant tensors with a device kernel stay packed (their file
/// bytes); an embedding in a format without a gather kernel dequantizes
/// dense; everything else loads at the backend element size. Cheap by
/// construction: packed bytes come from the mmap'd source, and only
/// tensors that would dequantize at load anyway are opened.
#[cfg(not(target_family = "wasm"))]
fn estimate_weight_bytes(source: &dyn ModelSource, elem_bytes: u64) -> (u64, (String, u64)) {
    let mut total = 0u64;
    let mut largest = (String::new(), 0u64);
    for name in source.tensor_names() {
        let dense_embed = name.contains("embed_tokens") || name.contains("token_embd");
        let bytes = match source.open_tensor_quant(&name) {
            Ok(Some(qt)) => {
                let packs = !dense_embed
                    || matches!(qt.format, combs_formats::QuantFormat::Q8_0);
                if packs {
                    qt.data.len() as u64
                } else {
                    qt.shape.iter().product::<usize>() as u64 * elem_bytes
                }
            }
            _ => match source.open_tensor(&name) {
                Ok(reader) => {
                    reader.shape().iter().product::<usize>() as u64 * elem_bytes
                }
                Err(_) => continue,
            },
        };
        total += bytes;
        if bytes > largest.1 {
            largest = (name, bytes);
        }
    }
    (total, largest)
}

/// The device's storage-binding ceiling, when the platform can answer.
#[cfg(not(target_family = "wasm"))]
fn binding_limit(device: &CombsDevice) -> Option<u64> {
    Some(combs_core::device_caps(device).max_storage_buffer_binding_size)
}

/// Worker-thread loop: executes queued requests serially until shutdown.
#[cfg(not(target_family = "wasm"))]
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
    let mut token_table: Option<Arc<TokenByteTable>> = None;
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Command::Shutdown => break,
            Command::Generate(req) => {
                let evictions_before = sessions.evictions;
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
                if sessions.evictions > evictions_before {
                    combs_core::provenance::event(
                        "engine",
                        "kv.evict",
                        &[
                            ("evicted", (sessions.evictions - evictions_before).to_string()),
                            ("evictions_total", sessions.evictions.to_string()),
                        ],
                    );
                    // An LRU-evicted session just dropped its KV pages.
                    // Drain the fusion stream first — tensor drops ride
                    // its queue, and a cleanup submitted before the
                    // deregistrations land trims almost nothing.
                    let _ = CombsBackend::sync(&device);
                    BufferPool::new().cleanup(&device);
                }
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
                if removed > 0 {
                    let _ = CombsBackend::sync(&device);
                    BufferPool::new().cleanup(&device);
                }
                if let Ok(mut snap) = stats.lock() {
                    snap.sessions = session_infos(&sessions);
                    // Resample on this thread: cubecl's memory accounting
                    // is per-stream, and the arenas live on the worker's —
                    // a caller-side gpu_memory cannot see what just freed.
                    snap.gpu = combs_core::gpu_memory(&device);
                }
                let _ = reply.send(removed);
            }
        }
    }
}

#[cfg(not(target_family = "wasm"))]
/// Prompt tokens per `prefill_hidden` call on the embeddings path — the
/// same attention-memory bound chunked generation prefill uses.
const EMBED_CHUNK: usize = 512;

#[cfg(not(target_family = "wasm"))]
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

#[cfg(not(target_family = "wasm"))]
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
        let [_, _l, vocab] = logits.dims();

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

#[cfg(not(target_family = "wasm"))]
/// In-place L2 normalization (no-op on a zero vector).
fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
}

#[cfg(not(target_family = "wasm"))]
/// Detects the checkpoint's pooling convention: sentence-transformers
/// layouts ship `1_Pooling/config.json` next to the weights; absent (or
/// unrecognized), decoder embedding models pool the last token.
///
/// `None` means the tokenizer has no directory to look beside — an
/// in-memory checkpoint carries no sibling files, so there is nothing to
/// detect and the decoder default stands.
fn detect_pooling(dir: Option<&std::path::Path>) -> Pooling {
    let Some(dir) = dir else { return Pooling::Last };
    let path = dir.join("1_Pooling").join("config.json");
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

#[cfg(not(target_family = "wasm"))]
/// Refreshes the shared snapshot after a generation. Runs on the worker so
/// the GPU allocator sample (`submit_blocking`) never contends with an
/// in-flight generation; observers only ever pay a mutex clone.
fn session_infos(sessions: &SessionSet) -> Vec<SessionInfo> {
    sessions
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

#[cfg(not(target_family = "wasm"))]
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

#[cfg(not(target_family = "wasm"))]
/// Executes one generation request on the worker thread.
///
/// The decode loop itself lives in [`crate::step`]; what belongs here is
/// only the part that is specific to *this* driver — a dedicated thread
/// that can afford to block on each logits readback, and an mpsc channel
/// that can refuse a piece when the caller has hung up.
fn run_generation(
    model: &mut dyn GenerativeModel<CombsBackend>,
    tokenizer: &Tokenizer,
    device: &CombsDevice,
    cache_config: &CacheConfig,
    max_position_embeddings: usize,
    req: &GenerateRequest,
    sessions: &mut SessionSet,
    token_table: &mut Option<Arc<TokenByteTable>>,
) -> Result<GenerationStats> {
    let (mut active, logits) = begin_generation(
        model,
        tokenizer,
        device,
        cache_config,
        max_position_embeddings,
        &req.prompt_tokens,
        &req.images,
        &req.config,
        req.cancel.clone(),
        sessions,
        token_table,
    )?;

    // A send failure is the caller hanging up mid-stream, which the loop
    // treats as a cancel.
    let mut emit = |id: u32, piece: String, lp: Option<TokenLogprobs>| {
        req.pieces.send((id, piece, lp)).is_ok()
    };

    // Errors before the first token are request-level rejections: nothing
    // was produced and no session is written.
    let row = step::readback_logits(logits)?;
    active.sample_row(row)?;

    loop {
        active.advance(tokenizer, &mut emit);
        if active.is_done() {
            break;
        }
        match active.submit(model, device) {
            Submitted::Logits(t) => match step::readback_logits(t) {
                Ok(row) => {
                    if active.sample_row(row).is_err() {
                        // Unreachable: past the first token, sample_row
                        // records its failure instead of returning it.
                        break;
                    }
                }
                Err(e) => {
                    active.fail(e);
                    break;
                }
            },
            Submitted::Ready => continue,
            Submitted::Done => break,
        }
    }

    active.finish(tokenizer, sessions, &mut emit)
}


/// Merges `generation_config.json` sampler defaults over the built-in
/// defaults. `None` fields keep the built-in default (greedy, no filters).
pub(crate) fn default_config_from(sampler: Option<&SamplerConfig>) -> GenerationConfig {
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
    #[cfg(not(target_family = "wasm"))]
    fn engine_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Engine>();
    }
}
