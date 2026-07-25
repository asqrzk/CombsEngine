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
use combs_models::{CacheConfig, CacheKind, GenerativeModel, ModelRegistry};
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
}

impl Default for GenerationConfig {
    fn default() -> Self {
        GenerationConfig {
            max_tokens: 64,
            sampling: SamplingParams::default(),
            stop_token_ids: Vec::new(),
            stop_strings: Vec::new(),
            prefill_chunk_size: 512,
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
    im_end_id: Option<u32>,
    default_config: GenerationConfig,
    cache_config: CacheConfig,
    tx: mpsc::Sender<Command>,
    worker: Mutex<Option<JoinHandle<()>>>,
    #[allow(dead_code)] // facade used by later phases for arena management
    pool: BufferPool,
}

impl Engine {
    /// Loads a model from any [`ModelSource`] via the default registry.
    ///
    /// The KV cache kind is selected by `COMBS_KV=paged|contiguous`
    /// (default: paged) with capacity `max_position_embeddings`.
    pub fn load(source: &dyn ModelSource, device: CombsDevice) -> Result<Self> {
        let kind = match std::env::var("COMBS_KV").as_deref() {
            Ok("contiguous") => CacheKind::Contiguous,
            Ok("paged") | Err(_) => CacheKind::Paged,
            Ok(other) => {
                tracing::warn!("unknown COMBS_KV={other:?}; using paged KV cache");
                CacheKind::Paged
            }
        };
        let mut config = CacheConfig::paged(source.metadata().max_position_embeddings);
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
        let im_end_id = spec.special_token_id("<|im_end|>");

        let (tx, rx) = mpsc::channel();
        let worker = {
            let tokenizer = tokenizer.clone();
            let device = device.clone();
            let max_position_embeddings = source.metadata().max_position_embeddings;
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
                    );
                })
                .map_err(|e| EngineError::WorkerGone(format!("spawning worker: {e}")))?
        };

        Ok(Engine {
            device,
            tokenizer,
            metadata: source.metadata().clone(),
            im_end_id,
            default_config: default_config_from(source.sampler_defaults().as_ref()),
            cache_config,
            tx,
            worker: Mutex::new(Some(worker)),
            pool: BufferPool::new(),
        })
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
        self.im_end_id
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

    /// [`Engine::generate`] with an abort flag: setting `cancel` to `true`
    /// stops the generation between tokens (checked once per decode step)
    /// and returns [`EngineError::Cancelled`].
    pub fn generate_cancellable(
        &self,
        prompt_tokens: &[u32],
        config: &GenerationConfig,
        cancel: Arc<AtomicBool>,
        mut on_token: impl FnMut(u32, &str),
    ) -> Result<GenerationStats> {
        let (pieces_tx, pieces_rx) = mpsc::channel();
        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(Command::Generate(Box::new(GenerateRequest {
                prompt_tokens: prompt_tokens.to_vec(),
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
) {
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
                );
                // `req.pieces` closes when `req` drops at the end of this
                // iteration; the reply is queued first, so the caller always
                // sees all pieces followed by the result.
                let _ = req.reply.send(result);
            }
        }
    }
}

/// Executes one generation request on the worker thread.
fn run_generation(
    model: &mut dyn GenerativeModel<CombsBackend>,
    tokenizer: &Tokenizer,
    device: &CombsDevice,
    cache_config: &CacheConfig,
    max_position_embeddings: usize,
    req: &GenerateRequest,
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

    let mut sampler: Box<dyn Sampler> = sampler_from_params(&config.sampling);
    let mut stop = StopDetector::new(
        req_eos_ids(model, config),
        config.stop_strings.clone(),
    );
    let mut detok = IncrementalDetokenizer::new();
    let mut cache = model.create_kv_cache(cache_config);
    let mut history: Vec<u32> = prompt_tokens.to_vec();

    let t_start = Instant::now();

    // --- chunked prefill ------------------------------------------------------
    let chunk = if config.prefill_chunk_size == 0 {
        usize::MAX
    } else {
        config.prefill_chunk_size
    };
    let data: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
    let tokens = Tensor::from_data(TensorData::new(data, [1, prompt_tokens.len()]), device);
    let embedded = model.embed(tokens);

    let mut offset = 0;
    let mut logits = None;
    while offset < prompt_tokens.len() {
        let len = chunk.min(prompt_tokens.len() - offset);
        let input = embedded.clone().narrow(1, offset, len);
        logits = Some(model.prefill(
            input,
            cache.as_mut(),
            offset as u32..(offset + len) as u32,
        ));
        offset += len;
    }
    let logits = logits.expect("nonempty prompt runs at least one chunk");

    let mut row = readback_logits(logits)?;
    let mut next = sampler.sample(&mut row, &history);
    let ttft = t_start.elapsed();
    let t_decode = Instant::now();

    let mut generated = 0usize;
    for _ in 0..config.max_tokens {
        if req.cancel.load(Ordering::Relaxed) {
            return Err(EngineError::Cancelled);
        }
        if stop.is_stop_token(next) {
            break;
        }
        history.push(next);
        let piece = detok.push(tokenizer, next)?;
        // Emit the piece, truncating at a stop string if one completes.
        match stop.push_text(&piece) {
            Some(cut) => {
                if cut > 0 && req.pieces.send((next, piece[..cut].to_string())).is_err() {
                    return Err(EngineError::Cancelled); // caller hung up
                }
                generated += 1;
                break;
            }
            None => {
                if req.pieces.send((next, piece)).is_err() {
                    return Err(EngineError::Cancelled); // caller hung up
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
        let mut row = readback_logits(logits)?;
        next = sampler.sample(&mut row, &history);
    }

    let decode_time = t_decode.elapsed();
    Ok(GenerationStats {
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: generated,
        ttft,
        decode_time,
        total_time: t_start.elapsed(),
        cache_pages_used: cache.pages_used().unwrap_or(0),
    })
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
}
