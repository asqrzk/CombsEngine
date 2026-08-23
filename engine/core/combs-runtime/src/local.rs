//! The single-flight engine that runs on the caller's own task.
//!
//! [`crate::Engine`] owns a worker thread and blocks it on every logits
//! readback. That is the right shape on a desktop and an impossible one in
//! a browser tab, which has a single thread it must give back between
//! tokens or the page stops responding — and where a blocking readback is
//! not merely rude but unavailable.
//!
//! `LocalEngine` is the same generation, driven differently: no thread, no
//! queue, no channel. The caller asks for one token at a time and awaits
//! it, so cancellation, backpressure and rendering all interleave for free
//! between steps. The decode logic underneath is [`crate::step`], shared
//! verbatim with the threaded engine — the two differ in who waits and
//! how, never in what a token means.
//!
//! Single-flight is literal here: one generation at a time, and starting a
//! second before the first finishes is an error rather than a queue. A
//! page that wants concurrency wants a second worker.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use burn::tensor::Tensor;
use combs_core::{CombsBackend, CombsDevice};
use combs_formats::{ModelMetadata, ModelSource, TokenizerSpec};
use combs_models::{CacheConfig, GenerativeModel, ModelRegistry};
use tokenizers::Tokenizer;

use crate::constraint::TokenByteTable;
use crate::engine::{GenerationConfig, GenerationStats, default_config_from};
use crate::request::ChatHost;
use crate::sampler::TokenLogprobs;
use crate::step::{self, ActiveGeneration, SessionSet, Submitted, begin_generation};
use crate::toolcall::{ChatMessage, ToolCallStyle};
use crate::{EngineError, Result};

/// One turn of [`LocalEngine::step`].
pub enum StepEvent {
    /// A generated token and the text it added.
    Token {
        /// Token id.
        id: u32,
        /// Newly decoded text (may be empty while a character completes).
        text: String,
        /// Per-token log-probabilities when the request asked for them.
        logprobs: Option<TokenLogprobs>,
    },
    /// The generation ended; no further steps.
    Done {
        /// Any character the detokenizer was still holding — usually
        /// empty, non-empty when generation stopped mid-codepoint. Emit it
        /// before the terminal or the last character is lost.
        tail: String,
        /// Telemetry for the completed run.
        stats: Box<GenerationStats>,
    },
}

/// GPU work awaiting a readback, or a token already settled without one.
enum Pending {
    Logits(Tensor<CombsBackend, 2>),
    /// A token the speculative path already settled; nothing to read back.
    #[cfg(not(target_family = "wasm"))]
    Ready,
}

/// A single-flight engine driven one token at a time by its caller.
pub struct LocalEngine {
    device: CombsDevice,
    model: Box<dyn GenerativeModel<CombsBackend>>,
    tokenizer: Tokenizer,
    metadata: ModelMetadata,
    spec: TokenizerSpec,
    template: Option<crate::template::ChatTemplate>,
    default_config: GenerationConfig,
    cache_config: CacheConfig,
    sessions: SessionSet,
    token_table: Option<Arc<TokenByteTable>>,
    active: Option<ActiveGeneration>,
    pending: Option<Pending>,
    /// A terminal held back one step, because the step that produced it
    /// also produced a token and a step reports one thing.
    terminal: Option<Result<StepEvent>>,
    cancel: Arc<AtomicBool>,
}

impl LocalEngine {
    /// Loads a model with an explicit KV cache configuration.
    ///
    /// Deliberately narrower than [`crate::Engine::load_with_cache_config`]:
    /// no environment overrides (a browser has no environment), no worker
    /// thread, and no allocator probe — `gpu_memory` is a blocking submit,
    /// which is exactly what this engine exists to avoid.
    pub fn load_with_cache_config(
        source: &dyn ModelSource,
        device: CombsDevice,
        cache_config: CacheConfig,
    ) -> Result<Self> {
        let registry = ModelRegistry::<CombsBackend>::new();
        let model = registry.load(source, &device)?;

        let spec = source.tokenizer()?;
        let tokenizer = Tokenizer::from_bytes(spec.json_bytes()?)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))?;

        let meta = source.metadata();
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

        Ok(LocalEngine {
            device,
            model,
            tokenizer,
            metadata: meta.clone(),
            spec,
            template,
            default_config: default_config_from(source.sampler_defaults().as_ref()),
            cache_config,
            sessions: SessionSet::new(),
            token_table: None,
            active: None,
            pending: None,
            terminal: None,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Loads a model with a KV cache sized to the model's own context,
    /// capped by [`crate::default_arena_len`].
    pub fn load(source: &dyn ModelSource, device: CombsDevice) -> Result<Self> {
        let arena = crate::default_arena_len(source.metadata().max_position_embeddings);
        Self::load_with_cache_config(source, device, CacheConfig::paged(arena))
    }

    /// Model metadata.
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// The KV cache configuration new sessions are created with.
    pub fn cache_config(&self) -> CacheConfig {
        self.cache_config
    }

    /// Encodes text to token ids (no special tokens added).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// The `<|im_end|>` token id, if the tokenizer defines one.
    pub fn im_end_id(&self) -> Option<u32> {
        self.spec.special_token_id("<|im_end|>")
    }

    /// Whether this model's chat template can express tool definitions.
    pub fn supports_tools(&self) -> bool {
        self.spec
            .chat_template
            .as_deref()
            .is_some_and(|t| t.contains("tools"))
    }

    /// Whether this model was trained to emit `<think>` reasoning blocks.
    pub fn supports_thinking(&self) -> bool {
        self.spec
            .chat_template
            .as_deref()
            .is_some_and(|t| t.contains("<think>"))
            || self.spec.special_token_id("<think>").is_some()
    }

    /// Wraps messages into the model's chat format, ending with the
    /// assistant turn open.
    pub fn wrap_chat(&self, messages: &[ChatMessage]) -> String {
        ChatHost::wrap_chat_with_tools(self, messages, None)
    }

    /// Starts a generation. Errors if one is already running: this engine
    /// is single-flight by construction, and silently queueing would hide
    /// that from a caller who could have opened a second worker instead.
    pub fn begin(&mut self, prompt_tokens: &[u32], config: &GenerationConfig) -> Result<()> {
        if self.active.is_some() {
            return Err(EngineError::WorkerGone(
                "a generation is already running on this engine".to_string(),
            ));
        }
        // HF `add_special_tokens` semantics, matching the threaded engine:
        // prompts start with BOS when the model declares one, unless the
        // tokenizer says otherwise or the prompt already opens with it.
        let mut prompt = prompt_tokens.to_vec();
        if self.spec.add_bos != Some(false) {
            if let Some(bos) = self.metadata.bos_token_id {
                if prompt.first() != Some(&bos) {
                    prompt.insert(0, bos);
                }
            }
        }
        self.cancel = Arc::new(AtomicBool::new(false));
        let (active, logits) = begin_generation(
            self.model.as_mut(),
            &self.tokenizer,
            &self.device,
            &self.cache_config,
            self.metadata.max_position_embeddings,
            &prompt,
            &[],
            config,
            self.cancel.clone(),
            &mut self.sessions,
            &mut self.token_table,
        )?;
        self.active = Some(active);
        self.pending = Some(Pending::Logits(logits));
        Ok(())
    }

    /// Advances the generation by one token.
    ///
    /// The await is the point: it is where a browser's event loop gets its
    /// thread back, so a cancel, a render, or a user's Stop button can land
    /// between any two tokens without the engine knowing about any of them.
    ///
    /// Ends with a [`StepEvent::Done`], or an error. A cancelled run
    /// reports `EngineError::Cancelled`; the only thing lost with it is a
    /// partial character the detokenizer was holding, which by definition
    /// was never displayable.
    pub async fn step(&mut self) -> Result<StepEvent> {
        // A terminal the previous step had to hold back.
        if let Some(terminal) = self.terminal.take() {
            return terminal;
        }
        if self.active.is_none() {
            return Err(EngineError::WorkerGone(
                "no generation is running; call begin() first".to_string(),
            ));
        }

        // 1. Turn the outstanding GPU work into a sampled `next`. The
        //    readback is awaited outside any borrow of the generation.
        match self.pending.take() {
            Some(Pending::Logits(t)) => {
                let row = step::readback_logits_async(t).await;
                let active = self.active.as_mut().expect("checked above");
                match row {
                    Ok(row) => {
                        if let Err(e) = active.sample_row(row) {
                            // Reachable only before the first token, where
                            // a failure rejects the request outright and no
                            // session is written.
                            self.active = None;
                            return Err(e);
                        }
                    }
                    Err(e) => active.fail(e),
                }
            }
            #[cfg(not(target_family = "wasm"))]
            Some(Pending::Ready) => {}
            None => {}
        }

        // 2. Emit at most one token, enqueue the next unit of work, and
        //    report whichever of the two the caller is owed first.
        loop {
            let active = self.active.as_mut().expect("checked above");
            let mut token: Option<StepEvent> = None;
            active.advance(&self.tokenizer, &mut |id, text, logprobs| {
                token = Some(StepEvent::Token { id, text, logprobs });
                true
            });

            if !active.is_done() {
                match active.submit(self.model.as_mut(), &self.device) {
                    Submitted::Logits(t) => self.pending = Some(Pending::Logits(t)),
                    #[cfg(not(target_family = "wasm"))]
                    Submitted::Ready => self.pending = Some(Pending::Ready),
                    // The submit itself failed; `active` is done now.
                    Submitted::Done => {}
                }
            }

            if self.active.as_ref().expect("checked above").is_done() {
                let finished = self.active.take().expect("checked above");
                let mut tail = String::new();
                let outcome =
                    finished.finish(&self.tokenizer, &mut self.sessions, &mut |_id, text, _lp| {
                        tail.push_str(&text);
                        true
                    });
                let terminal = outcome.map(|stats| StepEvent::Done {
                    tail,
                    stats: Box::new(stats),
                });
                return match token {
                    Some(tok) => {
                        self.terminal = Some(terminal);
                        Ok(tok)
                    }
                    None => terminal,
                };
            }

            match token {
                Some(tok) => return Ok(tok),
                // Every path that emits nothing also ends the run, so this
                // is unreachable in practice; looping is still the correct
                // response if that ever stops being true.
                None => continue,
            }
        }
    }

    /// The live KV sessions this engine holds.
    ///
    /// The threaded engine publishes a stats snapshot its host can poll;
    /// this one has no thread to publish from, so its host asks instead.
    /// Same information, pulled rather than pushed.
    pub fn sessions(&self) -> Vec<crate::SessionInfo> {
        self.sessions
            .iter()
            .map(|(k, s)| crate::SessionInfo {
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

    /// Bytes one KV page costs across all layers (both K and V), so a
    /// caller can turn page counts into memory the way the native stats
    /// endpoint does.
    pub fn kv_page_bytes(&self) -> u64 {
        let meta = &self.metadata;
        let elem_bytes: u64 = if cfg!(feature = "f16") { 2 } else { 4 };
        let global_layers = (0..meta.num_hidden_layers)
            .filter(|&i| meta.attention_pattern.is_global_layer(i))
            .count()
            .max(1);
        let bytes_per_value_x8 = if self.cache_config.quantize_kv {
            9 // (1 + 4/32) * 8
        } else {
            elem_bytes * 8
        };
        self.cache_config.page_size as u64
            * meta.num_key_value_heads as u64
            * meta.head_dim as u64
            * bytes_per_value_x8
            / 8
            * 2 // K and V
            * global_layers as u64
    }

    /// Discards an in-flight generation, freeing its cache and leaving the
    /// engine ready for the next `begin`. Returns whether there was one.
    ///
    /// This is for a caller whose request ended without reaching a terminal
    /// step — a browser tab navigating away mid-stream, a dropped future.
    /// Without it, the abandoned generation would still hold the
    /// single-flight slot and every later request would be refused for a
    /// turn that nobody is waiting for any more.
    pub fn abandon(&mut self) -> bool {
        self.pending = None;
        self.terminal = None;
        self.active.take().is_some()
    }

    /// The current generation's cancel flag.
    ///
    /// Handed out so a caller can hold the ability to stop a run without
    /// holding the engine — which matters when the engine is borrowed for
    /// the duration of the request and a Stop must still land.
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    /// Requests cancellation of the running generation. The next step
    /// stops between tokens and reports `Cancelled`.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Drops one named KV session (`Some(id)`) or every session (`None`).
    pub fn clear_sessions(&mut self, id: Option<&str>) -> usize {
        match id {
            Some(key) => self.sessions.take(key).map(|_| 1).unwrap_or(0),
            None => self.sessions.clear_all(),
        }
    }
}

impl ChatHost for LocalEngine {
    fn wrap_chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&serde_json::Value>,
    ) -> String {
        if let Some(template) = &self.template {
            if let Ok(prompt) = template.render(messages, tools) {
                return prompt;
            }
            // A template that will not render degrades to the token-sniffed
            // wrap rather than breaking the turn.
        }
        let pairs: Vec<(String, String)> = messages
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();
        self.spec.wrap_messages(&pairs)
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        LocalEngine::encode(self, text)
    }

    fn default_config(&self) -> GenerationConfig {
        self.default_config.clone()
    }

    fn im_end_id(&self) -> Option<u32> {
        LocalEngine::im_end_id(self)
    }

    fn tool_call_style(&self) -> ToolCallStyle {
        ToolCallStyle::detect(self.spec.chat_template.as_deref())
    }
}
