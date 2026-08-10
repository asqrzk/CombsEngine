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

use crate::detok::IncrementalDetokenizer;
use crate::sampler::{Sampler, SamplingParams, sampler_from_params};
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
        }
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
    /// Streaming channel: (token id, new text piece) per generated token.
    pieces: mpsc::Sender<(u32, String)>,
    /// Final outcome, sent once after `pieces` closes.
    reply: mpsc::Sender<Result<GenerationStats>>,
}

/// Instruction for the engine worker thread.
enum Command {
    Generate(Box<GenerateRequest>),
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
    default_config: GenerationConfig,
    cache_config: CacheConfig,
    tx: mpsc::Sender<Command>,
    worker: Mutex<Option<JoinHandle<()>>>,
    stats: Arc<Mutex<EngineStatsSnapshot>>,
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
    pub fn load_with_cache_config(
        source: &dyn ModelSource,
        device: CombsDevice,
        cache_config: CacheConfig,
    ) -> Result<Self> {
        let registry = ModelRegistry::<CombsBackend>::new();
        let model = registry.load(source, &device)?;

        let spec = source.tokenizer()?;
        let tokenizer = Tokenizer::from_file(&spec.tokenizer_json)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))?;

        // Observability snapshot: seeded with the static KV geometry so
        // consumers can turn page counts into bytes; the worker refreshes
        // the dynamic parts after every generation.
        let meta = source.metadata();
        let elem_bytes: u64 = if cfg!(feature = "f16") { 2 } else { 4 };
        let stats = Arc::new(Mutex::new(EngineStatsSnapshot {
            max_sessions: MAX_SESSIONS,
            kv_page_bytes: cache_config.page_size as u64
                * meta.num_key_value_heads as u64
                * meta.head_dim as u64
                * elem_bytes
                * 2 // K and V
                * meta.num_hidden_layers as u64,
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

        Ok(Engine {
            device,
            tokenizer,
            metadata: source.metadata().clone(),
            spec,
            default_config: default_config_from(source.sampler_defaults().as_ref()),
            cache_config,
            tx,
            worker: Mutex::new(Some(worker)),
            stats,
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

    /// The `<|im_end|>` token id, if the tokenizer defines one (chat models).
    pub fn im_end_id(&self) -> Option<u32> {
        self.spec.special_token_id("<|im_end|>")
    }

    /// Gemma-style end-of-turn id (`<end_of_turn>`), when defined.
    pub fn end_turn_id(&self) -> Option<u32> {
        self.spec.special_token_id("<end_of_turn>")
    }

    /// Wraps (role, content) message pairs into the model's chat template
    /// (ChatML or Gemma turns, detected from the tokenizer's special
    /// tokens), ending with the assistant turn open.
    pub fn wrap_chat(&self, messages: &[(String, String)]) -> String {
        self.spec.wrap_messages(messages)
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
        on_token: impl FnMut(u32, &str),
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
        on_token: impl FnMut(u32, &str),
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
        on_token: impl FnMut(u32, &str),
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
        mut on_token: impl FnMut(u32, &str),
    ) -> Result<GenerationStats> {
        // HF `add_special_tokens` semantics: prompts start with BOS when the
        // model declares one. Gemma collapses without it; templates that
        // already open with `<bos>` are detected and left alone.
        let mut prompt_tokens = prompt_tokens.to_vec();
        if let Some(bos) = self.metadata.bos_token_id {
            if prompt_tokens.first() != Some(&bos) {
                prompt_tokens.insert(0, bos);
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
        while let Ok((id, piece)) = pieces_rx.recv() {
            on_token(id, &piece);
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
                );
                update_stats(&stats, &result, &sessions, &device);
                // `req.pieces` closes when `req` drops at the end of this
                // iteration; the reply is queued first, so the caller always
                // sees all pieces followed by the result.
                let _ = req.reply.send(result);
            }
        }
    }
}

/// Refreshes the shared snapshot after a generation. Runs on the worker so
/// the GPU allocator sample (`submit_blocking`) never contends with an
/// in-flight generation; observers only ever pay a mutex clone.
fn update_stats(
    stats: &Mutex<EngineStatsSnapshot>,
    result: &Result<GenerationStats>,
    sessions: &SessionSet,
    device: &CombsDevice,
) {
    let gpu = combs_core::gpu_memory(device);
    let session_infos: Vec<SessionInfo> = sessions
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
        .collect();

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
fn run_generation(
    model: &mut dyn GenerativeModel<CombsBackend>,
    tokenizer: &Tokenizer,
    device: &CombsDevice,
    cache_config: &CacheConfig,
    max_position_embeddings: usize,
    req: &GenerateRequest,
    sessions: &mut SessionSet,
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
                    let popped = s.cache.popn(s.history.len() - shared);
                    s.history.truncate(s.history.len() - popped);
                    // Invariant: lcp == cache seq_len == history len (paged
                    // cache always pops the full requested amount).
                    lcp = s.history.len();
                    s.cache
                } else {
                    model.create_kv_cache(cache_config)
                }
            }
            None => model.create_kv_cache(cache_config),
        }
    } else {
        model.create_kv_cache(cache_config)
    };

    let mut sampler: Box<dyn Sampler> = sampler_from_params(&config.sampling);
    let mut stop = StopDetector::new(
        req_eos_ids(model, config),
        config.stop_strings.clone(),
    );
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
    let mut next = sampler.sample(&mut row, &history);
    let ttft = t_start.elapsed();
    let t_decode = Instant::now();

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
                if cut > 0 && req.pieces.send((next, piece[..cut].to_string())).is_err() {
                    loop_error = Some(EngineError::Cancelled); // caller hung up
                }
                generated += 1;
                break;
            }
            None => {
                if req.pieces.send((next, piece)).is_err() {
                    loop_error = Some(EngineError::Cancelled); // caller hung up
                    break;
                }
            }
        }
        generated += 1;

        if generated >= config.max_tokens {
            break;
        }
        let data = vec![next as i32];
        let input = model.embed(Tensor::from_data(TensorData::new(data, [1, 1]), device));
        let logits = model.decode(input, cache.as_mut());
        decoded.push(next); // `next` is now in the KV cache
        let mut row = match readback_logits(logits) {
            Ok(r) => r,
            Err(e) => {
                loop_error = Some(e);
                break;
            }
        };
        next = sampler.sample(&mut row, &history);
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
