//! The decode loop, as a state machine you drive one step at a time.
//!
//! The engine's token loop used to be written inline, which tied it to the
//! one thing that could drive it: a dedicated OS thread blocking on a
//! logits readback. A browser has neither — no spare thread to block, and
//! no blocking readback at all — so the loop is expressed here as state
//! plus three moves, and *who* performs the readback (and whether it
//! blocks or awaits) becomes the caller's business:
//!
//! ```text
//!   begin_generation(...)  -> (ActiveGeneration, first logits)
//!   loop {
//!       sample_row(row)        mask + sample the row into `next`
//!       advance(emit)          stop tests, detokenize, emit `next`
//!       submit(model)          enqueue the work whose row comes next
//!   }
//!   finish(...)            -> stats
//! ```
//!
//! The native [`crate::Engine`] drives it with a blocking readback on its
//! worker thread; [`crate::LocalEngine`] drives it with an awaited one on
//! the caller's task. Same code decides what a token means in both.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use burn::tensor::{Tensor, TensorData};
use combs_core::{CombsBackend, CombsDevice};
use combs_media::PixelBatch;
use combs_models::{CacheConfig, CacheKind, GenerativeModel, KVCache, pixels_to_tensor};
use tokenizers::Tokenizer;

use crate::constraint::{ConstraintState, TokenByteTable};
use crate::detok::IncrementalDetokenizer;
use crate::engine::{GenerationConfig, GenerationStats, check_context_len};
use crate::sampler::{Sampler, TokenLogprobs, sampler_from_params_exempting};
use crate::stop::StopDetector;
use crate::time::Instant;
use crate::{EngineError, Result};

/// Error text for a constraint dead end (no token can legally continue).
pub(crate) const DEAD_END: &str =
    "no vocabulary token can continue the constrained JSON output (schema dead end)";

/// Maximum named sessions kept alive at once (LRU-evicted beyond this).
/// Each session owns its KV arena, so this also bounds cache VRAM.
pub(crate) const MAX_SESSIONS: usize = 4;

/// Where a generated token goes. Returns false when the sink is gone (the
/// caller hung up mid-stream), which ends the generation as a cancel.
pub(crate) type Emit<'a> = &'a mut dyn FnMut(u32, String, Option<TokenLogprobs>) -> bool;

/// Length of the longest common prefix of two token slices.
fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// A rolling session: the previous request's full token history (prompt +
/// decoded tokens — exactly what the KV cache contains) and its cache. The
/// next request rolls the cache back to the longest common prefix with its
/// own prompt and prefills only the suffix.
pub(crate) struct SessionState {
    pub(crate) history: Vec<u32>,
    pub(crate) cache: Box<dyn KVCache<CombsBackend>>,
    pub(crate) last_used: u64,
}

/// The session table: named rolling sessions with LRU eviction. The
/// anonymous session (empty key) serves requests without a session id.
pub(crate) struct SessionSet {
    map: std::collections::HashMap<String, SessionState>,
    tick: u64,
    pub(crate) evictions: u64,
}

impl SessionSet {
    pub(crate) fn new() -> Self {
        SessionSet {
            map: std::collections::HashMap::new(),
            tick: 0,
            evictions: 0,
        }
    }

    /// Removes and returns the session for `key` (caller re-inserts it
    /// after generation, or drops it to free its pages).
    pub(crate) fn take(&mut self, key: &str) -> Option<SessionState> {
        self.map.remove(key)
    }

    /// Drops every session, returning how many were removed.
    pub(crate) fn clear_all(&mut self) -> usize {
        let n = self.map.len();
        self.map.clear();
        n
    }

    /// Live sessions, for observability. Both engines publish a snapshot:
    /// a KV cache nobody can see is a KV cache nobody can debug, and the
    /// browser needs that more than the desktop does, not less.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&String, &SessionState)> {
        self.map.iter()
    }

    /// Inserts a session, evicting the least-recently-used one past
    /// [`MAX_SESSIONS`].
    pub(crate) fn put(&mut self, key: String, mut session: SessionState) {
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

/// What [`ActiveGeneration::submit`] left for the driver to do.
pub(crate) enum Submitted {
    /// A decode step was enqueued: read this tensor back and feed the row
    /// to [`ActiveGeneration::sample_row`].
    Logits(Tensor<CombsBackend, 2>),
    /// The next token was settled without a readback (a speculative round
    /// pre-verified it): go straight to [`ActiveGeneration::advance`].
    /// Only the native speculative path produces this.
    #[cfg(not(target_family = "wasm"))]
    Ready,
    /// Nothing more to do; call [`ActiveGeneration::finish`].
    Done,
}

/// One generation in flight: everything the token loop carries between
/// steps, and nothing about how it is driven.
/// Reasoning-marker discipline for the decode loop: while the CURRENT
/// generation holds an open `<think>` with no matching `</think>`, the
/// stop tokens are masked out of sampling — a model cannot legally end
/// its turn mid-thought. Depth counts markers over accepted tokens
/// only; `max_tokens` still bounds the run (finish_reason = length),
/// so a model that refuses to close cannot spin forever.
pub(crate) struct ThinkGuard {
    open: u32,
    close: u32,
    stop_ids: Vec<u32>,
    depth: u32,
}

/// The vocab's explicit reasoning markers, when both exist.
/// `COMBS_THINK_GUARD=0` disables the guard for A/B runs.
fn think_marker_ids(tokenizer: &Tokenizer) -> Option<(u32, u32)> {
    if matches!(std::env::var("COMBS_THINK_GUARD").as_deref(), Ok("0")) {
        return None;
    }
    Some((
        tokenizer.token_to_id("<think>")?,
        tokenizer.token_to_id("</think>")?,
    ))
}

pub(crate) struct ActiveGeneration {
    cache: Box<dyn KVCache<CombsBackend>>,
    sampler: Box<dyn Sampler>,
    stop: StopDetector,
    constraint: Option<ConstraintState>,
    think_guard: Option<ThinkGuard>,
    detok: IncrementalDetokenizer,
    /// Prompt + emitted tokens — what the sampler's penalties see.
    history: Vec<u32>,
    /// The token about to be emitted (already sampled, not yet accepted).
    next: u32,
    next_logprobs: Option<TokenLogprobs>,
    /// Tokens actually fed through `decode` (i.e. present in the KV cache).
    /// The final sampled token and stop tokens never enter the cache.
    decoded: Vec<u32>,
    generated: usize,
    /// Loop iterations consumed, bounding generation exactly as the
    /// original `for _ in 0..max_tokens` did.
    iterations: usize,
    max_tokens: usize,
    cancel: Arc<AtomicBool>,
    loop_error: Option<EngineError>,
    done: bool,
    /// Set once a row has been sampled: before that, a failure is a
    /// request-level rejection rather than a generation that stopped.
    started: bool,
    /// Set when the cache holds rows this machine can no longer account
    /// for. The session invariant is that `history` mirrors the cache
    /// exactly; once that is in doubt the session is dropped rather than
    /// written, because a wrong cache is served silently forever while a
    /// missing one costs one re-prefill.
    poisoned: bool,

    prompt_tokens: Vec<u32>,
    lcp: usize,
    reuse: bool,
    session_key: String,

    t_start: Instant,
    t_decode: Instant,
    ttft: Duration,
    trace: DecodeTrace,

    /// Prompt-lookup speculation state. Native only: it needs a blocking
    /// readback of the verification rows, and it is a throughput trick for
    /// machines that have throughput to spare.
    #[cfg(not(target_family = "wasm"))]
    spec: SpecState,
}

/// Prompt-lookup speculative decoding state.
#[cfg(not(target_family = "wasm"))]
struct SpecState {
    active: bool,
    /// Pre-verified tokens waiting to be emitted.
    pending: std::collections::VecDeque<u32>,
    /// Cache rows sitting beyond `decoded` (drained as `pending` empties).
    cache_ahead: usize,
    cooldown: usize,
}

/// Begins a generation: validates the request, takes over the session's KV
/// cache where it can, prefills the prompt suffix, and stops at the point
/// where the first logits row must be read back — which is exactly the
/// point where native and browser drivers diverge.
#[allow(clippy::too_many_arguments)]
pub(crate) fn begin_generation(
    model: &mut dyn GenerativeModel<CombsBackend>,
    tokenizer: &Tokenizer,
    device: &CombsDevice,
    cache_config: &CacheConfig,
    max_position_embeddings: usize,
    prompt_tokens: &[u32],
    images: &[PixelBatch],
    config: &GenerationConfig,
    cancel: Arc<AtomicBool>,
    sessions: &mut SessionSet,
    token_table: &mut Option<Arc<TokenByteTable>>,
) -> Result<(ActiveGeneration, Tensor<CombsBackend, 2>)> {
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

    // Structured output compiles first, before anything is taken from the
    // session table. An unsatisfiable schema is a request error that
    // produced nothing, and it must not cost a conversation the KV cache
    // it had accumulated — taking the session and then failing would
    // silently drop it on the floor.
    let eos_for_constraint = req_eos_ids(model, config);
    let constraint = match &config.constraint {
        Some(spec) => {
            let schema = spec.compile().map_err(EngineError::Constraint)?;
            let table = token_table
                .get_or_insert_with(|| Arc::new(TokenByteTable::build(tokenizer)));
            Some(ConstraintState::new(schema, table.clone(), eos_for_constraint))
        }
        None => None,
    };

    // Image turns never join the KV session cache: token prefixes would
    // collide with different image contents, so media requests run on a
    // fresh cache and are not saved back.
    let has_media = !images.is_empty();
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
    // the longer the conversation runs. The reasoning markers are
    // structural in exactly the same way: penalizing `</think>` for
    // having appeared in prior turns suppresses the model's ability to
    // stop thinking.
    let think_ids = think_marker_ids(tokenizer);
    let mut penalty_exempt = eos_ids.clone();
    if let Some((open, close)) = think_ids {
        penalty_exempt.push(open);
        penalty_exempt.push(close);
    }
    let sampler: Box<dyn Sampler> =
        sampler_from_params_exempting(&config.sampling, &penalty_exempt);
    let think_guard = think_ids.map(|(open, close)| ThinkGuard {
        open,
        close,
        stop_ids: eos_ids.clone(),
        depth: 0,
    });
    let stop = StopDetector::new(eos_ids, config.stop_strings.clone());

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
        let pixels: Vec<Tensor<CombsBackend, 4>> = images
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

    #[cfg(not(target_family = "wasm"))]
    let spec = SpecState {
        // Speculative decoding (prompt-lookup drafts + multi-token verify):
        // greedy-only and only where raw argmax IS the sampler (no
        // penalties, bias, or logprob capture), never under constraints,
        // and only on non-sliding models with a paged cache — there `popn`
        // rollback is always total. Every other request takes the normal
        // path untouched.
        active: spec_enabled()
            && constraint.is_none()
            && config.sampling.temperature <= 0.0
            && config.sampling.repetition_penalty.map_or(true, |v| v == 1.0)
            && config.sampling.frequency_penalty.map_or(true, |v| v == 0.0)
            && config.sampling.presence_penalty.map_or(true, |v| v == 0.0)
            && config.sampling.logit_bias.is_none()
            && config.sampling.logprobs.is_none()
            && model.supports_decode_all_logits()
            && model.metadata().attention_pattern.sliding_window.is_none()
            && cache.pages_used().is_some(),
        pending: std::collections::VecDeque::new(),
        cache_ahead: 0,
        cooldown: 0,
    };

    let active = ActiveGeneration {
        cache,
        sampler,
        stop,
        constraint,
        think_guard,
        detok: IncrementalDetokenizer::new(),
        history: prompt_tokens.to_vec(),
        next: 0,
        next_logprobs: None,
        decoded: Vec::new(),
        generated: 0,
        iterations: 0,
        max_tokens: config.max_tokens,
        cancel,
        loop_error: None,
        done: false,
        started: false,
        poisoned: false,
        prompt_tokens: prompt_tokens.to_vec(),
        lcp,
        reuse,
        session_key: key,
        t_start,
        t_decode: t_start,
        ttft: Duration::ZERO,
        trace: DecodeTrace::from_env(),
        #[cfg(not(target_family = "wasm"))]
        spec,
    };
    Ok((active, logits))
}

impl ActiveGeneration {
    /// Masks and samples one logits row into `next`.
    ///
    /// A constraint dead end here is returned as an error only before the
    /// first token: at that point nothing has been produced and the
    /// request is simply unsatisfiable, so it is rejected outright and no
    /// session is written. Once generation has started, the same condition
    /// ends the run and is reported through [`Self::finish`], keeping the
    /// tokens already emitted and the cache already built.
    pub(crate) fn sample_row(&mut self, mut row: Vec<f32>) -> Result<()> {
        // The prefill's own sample is not a decode step: counting it would
        // add a step with no submit time and skew the per-step split.
        let traced = self.started;
        if traced {
            self.trace.read_back();
        }
        if let Some(c) = self.constraint.as_mut() {
            if c.mask(&mut row) == 0 {
                let err = EngineError::Constraint(DEAD_END.to_string());
                if !self.started {
                    return Err(err);
                }
                self.fail(err);
                return Ok(());
            }
        }
        if let Some(g) = &self.think_guard {
            if g.depth > 0 {
                for &id in &g.stop_ids {
                    if let Some(v) = row.get_mut(id as usize) {
                        *v = f32::NEG_INFINITY;
                    }
                }
            }
        }
        let outcome = self.sampler.sample(&mut row, &self.history);
        self.next = outcome.token;
        self.next_logprobs = outcome.logprobs;
        if let Some(c) = self.constraint.as_mut() {
            c.advance(self.next);
        }
        if !self.started {
            self.started = true;
            self.ttft = self.t_start.elapsed();
            self.t_decode = Instant::now();
        }
        if traced {
            self.trace.sampled();
        }
        Ok(())
    }

    /// One turn of the loop head: the budget and stop tests, incremental
    /// detokenization, and emission of the current `next`.
    pub(crate) fn advance(&mut self, tokenizer: &Tokenizer, emit: Emit<'_>) {
        if self.done {
            return;
        }
        if self.iterations >= self.max_tokens {
            self.done = true;
            return;
        }
        self.iterations += 1;

        if self.cancel.load(Ordering::Relaxed) {
            self.fail(EngineError::Cancelled);
            return;
        }
        if self.stop.is_stop_token(self.next) {
            self.done = true;
            return;
        }
        self.history.push(self.next);
        if let Some(g) = self.think_guard.as_mut() {
            if self.next == g.open {
                g.depth += 1;
            } else if self.next == g.close {
                g.depth = g.depth.saturating_sub(1);
            }
        }
        let piece = match self.detok.push(tokenizer, self.next) {
            Ok(p) => p,
            Err(e) => {
                self.fail(e);
                return;
            }
        };
        // Emit the piece, truncating at a stop string if one completes.
        match self.stop.push_text(&piece) {
            Some(cut) => {
                if cut > 0 && !emit(self.next, piece[..cut].to_string(), self.next_logprobs.take())
                {
                    self.loop_error = Some(EngineError::Cancelled); // caller hung up
                }
                self.generated += 1;
                self.done = true;
                return;
            }
            None => {
                if !emit(self.next, piece, self.next_logprobs.take()) {
                    self.fail(EngineError::Cancelled); // caller hung up
                    return;
                }
            }
        }
        self.generated += 1;
        if self.generated >= self.max_tokens {
            self.done = true;
        }
    }

    /// Enqueues the work that produces the next token.
    pub(crate) fn submit(
        &mut self,
        model: &mut dyn GenerativeModel<CombsBackend>,
        device: &CombsDevice,
    ) -> Submitted {
        if self.done {
            return Submitted::Done;
        }
        #[cfg(not(target_family = "wasm"))]
        if let Some(ready) = self.submit_speculative(model, device) {
            return ready;
        }
        // A step taken the ordinary way is a step closer to drafting again.
        #[cfg(not(target_family = "wasm"))]
        {
            self.spec.cooldown = self.spec.cooldown.saturating_sub(1);
        }
        self.trace.step_start();
        let data = vec![self.next as i32];
        let input = model.embed(Tensor::from_data(TensorData::new(data, [1, 1]), device));
        let logits = model.decode(input, self.cache.as_mut());
        self.trace.submitted();
        self.decoded.push(self.next); // `next` is now in the KV cache
        Submitted::Logits(logits)
    }

    /// The speculative arms of the loop tail: drain a pre-verified token,
    /// or run one draft-and-verify round. `None` means "take the ordinary
    /// single-token decode".
    #[cfg(not(target_family = "wasm"))]
    fn submit_speculative(
        &mut self,
        model: &mut dyn GenerativeModel<CombsBackend>,
        device: &CombsDevice,
    ) -> Option<Submitted> {
        if let Some(t) = self.spec.pending.pop_front() {
            // Pre-verified by the last speculative round: `next` is already
            // in the cache and `t` was proven to follow it.
            self.decoded.push(self.next);
            self.spec.cache_ahead -= 1;
            self.next = t;
            self.next_logprobs = None;
            return Some(Submitted::Ready);
        }
        let draft = (self.spec.active && self.spec.cooldown == 0)
            .then(|| crate::spec::propose(&self.history, SPEC_DRAFT, SPEC_MIN_NGRAM, SPEC_MAX_NGRAM))
            .flatten()?;

        // Feed [next, draft…] in one pass; row i's argmax is the model's
        // true next token after position i, so the longest matching prefix
        // of the draft is exactly what greedy decoding would have produced
        // one at a time.
        let k = draft.len();
        let mut ids: Vec<i32> = Vec::with_capacity(k + 1);
        ids.push(self.next as i32);
        ids.extend(draft.iter().map(|&t| t as i32));
        let input = model.embed(Tensor::from_data(TensorData::new(ids, [1, k + 1]), device));
        // Past this call the cache holds k+1 rows that nothing has
        // accounted for yet: `decoded` gains `next` and `cache_ahead`
        // gains `accepted` only after the verification below succeeds.
        // Every failure in between therefore leaves the cache longer than
        // `history` describes, and the session must be dropped rather than
        // saved — see `poisoned`.
        let all = match model.decode_all_logits(input, self.cache.as_mut()) {
            Ok(t) => t,
            Err(e) => {
                self.fail_with_unaccounted_cache(e.into());
                return Some(Submitted::Done);
            }
        };
        let rows: Vec<f32> = match all.into_data().convert::<f32>().to_vec::<f32>() {
            Ok(v) => v,
            Err(e) => {
                self.fail_with_unaccounted_cache(EngineError::Readback(format!(
                    "spec logits: {e:?}"
                )));
                return Some(Submitted::Done);
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
        // Roll the rejected tail out of the cache. The static guards make a
        // partial rollback impossible; treat one as fatal rather than
        // continuing on divergent state.
        let unwanted = k - accepted;
        if unwanted > 0 {
            let popped = self.cache.popn(unwanted);
            if popped != unwanted {
                self.fail_with_unaccounted_cache(EngineError::Readback(format!(
                    "speculative rollback popped {popped} of {unwanted} rows"
                )));
                return Some(Submitted::Done);
            }
        }
        self.decoded.push(self.next);
        self.spec.cache_ahead += accepted;
        self.spec.cooldown = if accepted == 0 { SPEC_BACKOFF } else { 0 };
        self.spec.pending.extend(draft[..accepted].iter().copied());
        self.spec.pending.push_back(correction);
        self.next = self
            .spec
            .pending
            .pop_front()
            .expect("correction token is always queued");
        self.next_logprobs = None;
        Some(Submitted::Ready)
    }

    /// Whether the loop should stop.
    pub(crate) fn is_done(&self) -> bool {
        self.done
    }

    /// Records a terminal error, stops the loop, and abandons the session:
    /// the cache holds rows beyond what `decoded` describes and no
    /// accounting can recover which.
    #[cfg(not(target_family = "wasm"))]
    fn fail_with_unaccounted_cache(&mut self, err: EngineError) {
        self.poisoned = true;
        self.fail(err);
    }

    /// Records a terminal error and stops the loop.
    pub(crate) fn fail(&mut self, err: EngineError) {
        if self.loop_error.is_none() {
            self.loop_error = Some(err);
        }
        self.done = true;
    }

    /// Flushes the detokenizer tail, returns the cache to its session, and
    /// reports the run. Called on every exit path — including a cancelled
    /// one, so a stopped request still leaves a consistent cache behind.
    pub(crate) fn finish(
        mut self,
        tokenizer: &Tokenizer,
        sessions: &mut SessionSet,
        emit: Emit<'_>,
    ) -> Result<GenerationStats> {
        // Release any character the detokenizer held back (an incomplete
        // UTF-8 sequence at the very end of generation).
        if let Ok(tail) = self.detok.flush(tokenizer) {
            if !tail.is_empty() {
                let _ = emit(0, tail, None);
            }
        }

        // Pre-verified rows that never got emitted must leave the cache
        // before the session snapshot (history must mirror KV contents
        // exactly).
        #[cfg(not(target_family = "wasm"))]
        if self.spec.cache_ahead > 0 {
            let popped = self.cache.popn(self.spec.cache_ahead);
            debug_assert_eq!(popped, self.spec.cache_ahead, "speculative exit rollback");
        }

        let cache_pages_used = self.cache.pages_used().unwrap_or(0);
        if self.reuse && !self.poisoned {
            let mut hist = self.prompt_tokens.clone();
            hist.extend_from_slice(&self.decoded);
            sessions.put(
                self.session_key,
                SessionState {
                    history: hist,
                    cache: self.cache,
                    last_used: 0,
                },
            );
        }

        self.trace.report();
        let stats = GenerationStats {
            prompt_tokens: self.prompt_tokens.len(),
            generated_tokens: self.generated,
            ttft: self.ttft,
            decode_time: self.t_decode.elapsed(),
            total_time: self.t_start.elapsed(),
            cache_pages_used,
            cached_tokens: self.lcp,
        };
        match self.loop_error {
            Some(e) => Err(e),
            None => Ok(stats),
        }
    }
}

/// Collects the eos ids that apply to a request.
fn req_eos_ids(model: &dyn GenerativeModel<CombsBackend>, config: &GenerationConfig) -> Vec<u32> {
    model
        .metadata()
        .eos_token_ids
        .iter()
        .chain(config.stop_token_ids.iter())
        .copied()
        .collect()
}

/// Reads a `[1, vocab]` logits row back to the host (one copy per step).
///
/// Blocking, so native-only: this is the exact call a browser cannot make,
/// and the reason [`readback_logits_async`] exists.
#[cfg(not(target_family = "wasm"))]
pub(crate) fn readback_logits(logits: Tensor<CombsBackend, 2>) -> Result<Vec<f32>> {
    logits
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|e| EngineError::Readback(format!("logits must be f32: {e:?}")))
}

/// [`readback_logits`] for drivers that must not block the thread they run
/// on. The browser has no other option; natively it is the same copy with
/// the wait expressed as an await rather than a park.
pub(crate) async fn readback_logits_async(logits: Tensor<CombsBackend, 2>) -> Result<Vec<f32>> {
    logits
        .into_data_async()
        .await
        .map_err(|e| EngineError::Readback(format!("logits readback failed: {e:?}")))?
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|e| EngineError::Readback(format!("logits must be f32: {e:?}")))
}

/// Draft length proposed per speculative round.
#[cfg(not(target_family = "wasm"))]
const SPEC_DRAFT: usize = 6;
/// Shortest history suffix n-gram accepted as a draft trigger — bigrams
/// recur incidentally in all text and produce failed rounds that cost more
/// than they save.
#[cfg(not(target_family = "wasm"))]
const SPEC_MIN_NGRAM: usize = 3;
/// Longest history suffix n-gram searched for a recurrence.
#[cfg(not(target_family = "wasm"))]
const SPEC_MAX_NGRAM: usize = 8;
/// Tokens decoded normally after a fully-rejected round before drafting
/// again (a miss usually means this region of text is not repetitive).
#[cfg(not(target_family = "wasm"))]
const SPEC_BACKOFF: usize = 16;

/// Speculative decoding ships opt-in behind `COMBS_SPEC=1` until its
/// measured speedup clears the bar for default-on; checked once.
#[cfg(not(target_family = "wasm"))]
fn spec_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("COMBS_SPEC").as_deref(), Ok("1")))
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
            enabled: trace_enabled(),
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

#[cfg(not(target_family = "wasm"))]
fn trace_enabled() -> bool {
    matches!(std::env::var("COMBS_TRACE_DECODE").as_deref(), Ok("1"))
}

/// No environment to read in a browser; the trace stays off and costs a
/// branch that never taken.
#[cfg(target_family = "wasm")]
fn trace_enabled() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::common_prefix;

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
