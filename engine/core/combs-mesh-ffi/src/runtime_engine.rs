//! `RuntimeEngine` — the `engine`-feature [`CombsEngineCore`] adapter over
//! `combs_runtime::Engine`.
//!
//! Model loading is explicit (the `engine_load` op carries the model
//! directory); `infer` then runs a single non-streaming generation with the
//! engine's default config (greedy: `SamplingParams::default()` is
//! temperature 0). The combs-ffi single-flight queue, KV sessions and
//! sampler are reused unchanged — this module only *consumes* them.
//!
//! The wgpu device is process-global (cubecl: one device per adapter), so
//! it lives behind a `OnceLock` exactly like in combs-ffi.

use std::sync::{Mutex, OnceLock};

use combs_core::{CombsDevice, init_device};
use combs_formats::open_model_source;
use combs_mesh::ffi_trait::EngineError;
use combs_runtime::{CacheConfig, Engine};

/// The mesh inference engine: a lazily loaded `combs_runtime::Engine`.
pub struct RuntimeEngine {
    engine: Mutex<Option<Engine>>,
}

impl RuntimeEngine {
    /// The process-wide instance.
    pub fn global() -> &'static RuntimeEngine {
        static ENGINE: OnceLock<RuntimeEngine> = OnceLock::new();
        ENGINE.get_or_init(|| RuntimeEngine {
            engine: Mutex::new(None),
        })
    }

    /// The process-wide wgpu device, shared by every engine in the process
    /// (never init a second one — cubecl allows one device per adapter).
    fn shared_device() -> &'static CombsDevice {
        static DEVICE: OnceLock<CombsDevice> = OnceLock::new();
        DEVICE.get_or_init(init_device)
    }

    /// Loads a model directory (safetensors or GGUF), replacing any
    /// previously loaded engine.
    pub fn load_model(&self, model_dir: &str, max_seq_len: Option<usize>) -> Result<(), EngineError> {
        let source = open_model_source(model_dir)
            .map_err(|e| EngineError::Unsupported(format!("loading model source: {e}")))?;
        let engine = match max_seq_len {
            Some(cap) => Engine::load_with_cache_config(
                &*source,
                Self::shared_device().clone(),
                CacheConfig::paged(cap),
            ),
            None => Engine::load(&*source, Self::shared_device().clone()),
        }
        .map_err(|e| EngineError::Unsupported(format!("engine load failed: {e}")))?;
        let mut guard = self
            .engine
            .lock()
            .map_err(|_| EngineError::Unsupported("engine lock poisoned".into()))?;
        *guard = Some(engine);
        Ok(())
    }

    /// Runs one greedy, non-streaming generation on `prompt`.
    pub fn infer(&self, prompt: &str) -> Result<String, EngineError> {
        let guard = self
            .engine
            .lock()
            .map_err(|_| EngineError::Unsupported("engine lock poisoned".into()))?;
        let engine = guard.as_ref().ok_or(EngineError::NotInitialized)?;
        let tokens = engine
            .encode(prompt)
            .map_err(|e| EngineError::Unsupported(format!("tokenization failed: {e}")))?;
        let config = engine.default_config();
        let mut text = String::new();
        engine
            .generate(&tokens, &config, |_, piece| text.push_str(piece))
            .map_err(|e| EngineError::Unsupported(format!("generation failed: {e}")))?;
        Ok(text)
    }
}
