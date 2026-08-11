//! # combs-ffi — L1 boundary
//!
//! Stable C ABI + JSON FFI for the Combs Engine (MLC `json_ffi` pattern):
//! opaque engine handles, JSON request/response payloads, one streaming
//! callback. This is the *only* native integration surface — Deno (FFI),
//! Android (JNI), iOS (Swift) and the WASM shell all bind to these symbols.
//!
//! ## API contract
//!
//! - `combs_device_caps_json()` → device capabilities JSON (planner input).
//! - `combs_engine_create(config_json)` → opaque handle or NULL (see
//!   `combs_last_error()`).
//! - `combs_chat_completion(engine, request_json, request_id, cb, user_data)`
//!   blocks until the request finishes, streaming
//!   `{"type":"delta"|"done"|"error", ...}` events to `cb`. Call it from a
//!   worker thread (Deno: `nonblocking: true` + `UnsafeCallback.threadSafe`).
//! - `combs_cancel(request_id)` aborts a running request between tokens.
//! - Strings returned by `*_json` functions are owned by the library and
//!   must be released with `combs_string_free`.
//!
//! All functions are panic-fenced; errors land in thread-local storage and
//! are read back via `combs_last_error()`.

mod types;

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use combs_core::{CombsDevice, device_caps, init_device};
use combs_formats::{ModelSource, open_model_source};
use combs_models::{CacheConfig, CacheKind};
use combs_runtime::{Engine, SamplingParams};

use types::*;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: impl std::fmt::Display) {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(CString::new(msg.to_string().replace('\0', " ")).unwrap())
    });
}

/// Returns the last error message on this thread, or NULL. The pointer is
/// borrowed — do not free it; invalidated by the next FFI call on the same
/// thread.
#[no_mangle]
pub extern "C" fn combs_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| match &*slot.borrow() {
        Some(s) => s.as_ptr(),
        None => ptr::null(),
    })
}

/// Frees a string previously returned by a `combs_*_json` function.
///
/// # Safety
/// `s` must have been returned by this library and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn combs_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

fn into_raw_json<T: serde::Serialize>(value: &T) -> *mut c_char {
    match serde_json::to_string(value) {
        Ok(s) => CString::new(s).map(CString::into_raw).unwrap_or(ptr::null_mut()),
        Err(_) => ptr::null_mut(),
    }
}

unsafe fn read_str<'a>(s: *const c_char, what: &str) -> Result<&'a str, String> {
    if s.is_null() {
        return Err(format!("{what} is NULL"));
    }
    unsafe { CStr::from_ptr(s) }
        .to_str()
        .map_err(|e| format!("{what} is not valid UTF-8: {e}"))
}

/// Opaque engine handle.
pub struct CombsEngine {
    engine: Engine,
    /// Default prefill chunk size from the creation config (applied to each
    /// request unless the request overrides it).
    prefill_chunk_size: Option<usize>,
}

/// Cancellation flags for in-flight requests, keyed by request id.
fn cancel_registry() -> &'static Mutex<HashMap<String, Arc<AtomicBool>>> {
    static REG: OnceLock<Mutex<HashMap<String, Arc<AtomicBool>>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The process-wide wgpu device. cubecl allows only one device per adapter,
/// so all engines created through this library share it (handles are cheap
/// clones); this also makes engine create/destroy cycles safe.
fn shared_device() -> &'static CombsDevice {
    static DEVICE: OnceLock<CombsDevice> = OnceLock::new();
    DEVICE.get_or_init(init_device)
}

/// Returns the device capabilities JSON (name, backend, buffer limits,
/// compute limits, feature dump). Free with `combs_string_free`.
///
/// Capabilities are static per process, so the query runs once and the JSON
/// string is cached (each call returns a fresh copy for the caller to free).
#[no_mangle]
pub extern "C" fn combs_device_caps_json() -> *mut c_char {
    static CAPS: OnceLock<Result<CString, String>> = OnceLock::new();
    let caps = CAPS.get_or_init(|| {
        catch_unwind(|| {
            serde_json::to_string(&device_caps(shared_device()))
                .map_err(|e| e.to_string())
                .and_then(|s| CString::new(s).map_err(|e| e.to_string()))
        })
        .unwrap_or_else(|_| Err("panic while querying device capabilities".to_string()))
    });
    match caps {
        Ok(s) => CString::into_raw(s.clone()),
        Err(e) => {
            set_last_error(e.clone());
            ptr::null_mut()
        }
    }
}

/// Creates an engine from a JSON config (`{"model_dir": "...", ...}`).
/// Returns NULL on error (see `combs_last_error`).
///
/// # Safety
/// `config_json` must be a valid NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn combs_engine_create(config_json: *const c_char) -> *mut CombsEngine {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<Box<CombsEngine>, String> {
        let json = unsafe { read_str(config_json, "config_json") }?;
        let config: EngineConfigJson =
            serde_json::from_str(json).map_err(|e| format!("invalid engine config JSON: {e}"))?;

        let source = open_model_source(&config.model_dir)
            .map_err(|e| format!("loading model source: {e}"))?;

        let mut cache_config = CacheConfig::paged(
            config
                .max_seq_len
                .unwrap_or(source.metadata().max_position_embeddings),
        );
        if let Some(ps) = config.page_size {
            cache_config.page_size = ps;
        }
        cache_config.kind = match config.kv_cache.as_deref() {
            None | Some("paged") => CacheKind::Paged,
            Some("contiguous") => CacheKind::Contiguous,
            Some(other) => return Err(format!("unknown kv_cache kind: {other}")),
        };

        let engine = Engine::load_with_cache_config(&source, shared_device().clone(), cache_config)
            .map_err(|e| format!("engine load failed: {e}"))?;
        Ok(Box::new(CombsEngine {
            engine,
            prefill_chunk_size: config.prefill_chunk_size,
        }))
    }));

    match result {
        Ok(Ok(engine)) => Box::into_raw(engine),
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic during engine creation");
            ptr::null_mut()
        }
    }
}

/// Destroys an engine handle.
///
/// # Safety
/// `engine` must have been created by `combs_engine_create` and not yet
/// destroyed.
#[no_mangle]
pub unsafe extern "C" fn combs_engine_destroy(engine: *mut CombsEngine) {
    if !engine.is_null() {
        drop(unsafe { Box::from_raw(engine) });
    }
}

/// Returns the engine's model metadata JSON. Free with `combs_string_free`.
///
/// # Safety
/// `engine` must be a valid handle.
#[no_mangle]
pub unsafe extern "C" fn combs_engine_metadata_json(engine: *const CombsEngine) -> *mut c_char {
    if engine.is_null() {
        set_last_error("engine is NULL");
        return ptr::null_mut();
    }
    let engine = unsafe { &*engine };
    let md = engine.engine.metadata();
    let cc = engine.engine.cache_config();
    into_raw_json(&EngineMetadataJson {
        architecture: md.architecture.clone(),
        vocab_size: md.vocab_size,
        max_position_embeddings: md.max_position_embeddings,
        max_seq_len: cc.max_seq_len,
        page_size: cc.page_size,
        eos_token_ids: md.eos_token_ids.clone(),
        im_end_id: engine.engine.im_end_id(),
    })
}

/// Streaming callback: receives a NUL-terminated JSON event
/// (`{"type":"delta"|"done"|"error", ...}`) and the opaque user pointer.
/// Called from the thread running `combs_chat_completion` — embedders must
/// make this thread-safe (Deno: `Deno.UnsafeCallback.threadSafe`).
pub type CombsStreamCallback = extern "C" fn(event_json: *const c_char, user_data: *mut c_void);

fn emit(cb: Option<CombsStreamCallback>, user_data: *mut c_void, event: &StreamEvent) {
    let Some(cb) = cb else { return };
    if let Ok(json) = serde_json::to_string(event) {
        if let Ok(cstr) = CString::new(json) {
            cb(cstr.as_ptr(), user_data);
        }
    }
}

// Chat wrapping goes through `Engine::wrap_chat` — the same checkpoint
// template (or token-sniffed fallback) path as `combs serve` and
// `combs run --chat`, so all three surfaces produce identical prompts.

/// Runs a chat completion. Blocks until done; streams events to `cb`.
/// Returns 0 on success, -1 on error (also emitted as an `error` event).
///
/// # Safety
/// `engine`, `request_json`, `request_id` must be valid; `cb` (if set) must
/// be safe to call from this thread.
#[no_mangle]
pub unsafe extern "C" fn combs_chat_completion(
    engine: *const CombsEngine,
    request_json: *const c_char,
    request_id: *const c_char,
    cb: Option<CombsStreamCallback>,
    user_data: *mut c_void,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        if engine.is_null() {
            return Err("engine is NULL".into());
        }
        let engine = unsafe { &*engine };
        let request: ChatRequestJson =
            serde_json::from_str(unsafe { read_str(request_json, "request_json") }?)
                .map_err(|e| format!("invalid request JSON: {e}"))?;
        let req_id = unsafe { read_str(request_id, "request_id") }?.to_string();

        let prompt = match (&request.messages, &request.prompt) {
            (Some(messages), _) => {
                let msgs: Vec<combs_runtime::ChatMessage> = messages
                    .iter()
                    .map(|m| combs_runtime::ChatMessage {
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
                engine.engine.wrap_chat(&msgs)
            }
            (None, Some(p)) => p.clone(),
            (None, None) => return Err("request needs `prompt` or `messages`".into()),
        };
        let prompt_tokens = engine
            .engine
            .encode(&prompt)
            .map_err(|e| format!("tokenization failed: {e}"))?;

        // Defaults from the engine; explicit request fields win.
        let mut config = engine.engine.default_config();
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
        };
        config.sampling = sampling;
        if let Some(mt) = request.max_tokens {
            config.max_tokens = mt;
        }
        if let Some(stop) = request.stop {
            config.stop_strings = stop;
        }
        if let Some(ids) = request.stop_token_ids {
            config.stop_token_ids = ids;
        } else if request.messages.is_some() {
            // Chat mode: stop at <|im_end|> when the tokenizer defines it.
            if let Some(im_end) = engine.engine.im_end_id() {
                config.stop_token_ids.push(im_end);
            }
        }
        if let Some(chunk) = request.prefill_chunk_size.or(engine.prefill_chunk_size) {
            config.prefill_chunk_size = chunk;
        }

        let cancel = Arc::new(AtomicBool::new(false));
        cancel_registry()
            .lock()
            .map_err(|e| e.to_string())?
            .insert(req_id.clone(), cancel.clone());
        let _guard = scopeguard(req_id.clone());

        let mut first = true;
        let outcome = engine.engine.generate_cancellable(
            &prompt_tokens,
            &config,
            cancel,
            |token_id, text| {
                emit(
                    cb,
                    user_data,
                    &StreamEvent::Delta {
                        text: text.to_string(),
                        token_id,
                    },
                );
                first = false;
            },
        );
        let _ = first;

        match outcome {
            Ok(stats) => {
                let finish_reason = if stats.generated_tokens >= config.max_tokens {
                    "length"
                } else {
                    "stop"
                };
                emit(
                    cb,
                    user_data,
                    &StreamEvent::Done {
                        finish_reason: finish_reason.into(),
                        stats: StatsJson {
                            prompt_tokens: stats.prompt_tokens,
                            generated_tokens: stats.generated_tokens,
                            ttft_ms: stats.ttft.as_secs_f64() * 1000.0,
                            decode_tokens_per_second: stats.decode_tokens_per_second(),
                            prefill_tokens_per_second: stats.prefill_tokens_per_second(),
                            cache_pages_used: stats.cache_pages_used,
                        },
                    },
                );
                Ok(())
            }
            Err(e) => {
                let msg = e.to_string();
                let cancelled = msg.contains("cancelled") || msg.contains("canceled");
                if cancelled {
                    emit(
                        cb,
                        user_data,
                        &StreamEvent::Done {
                            finish_reason: "cancelled".into(),
                            stats: StatsJson {
                                prompt_tokens: 0,
                                generated_tokens: 0,
                                ttft_ms: 0.0,
                                decode_tokens_per_second: 0.0,
                                prefill_tokens_per_second: 0.0,
                                cache_pages_used: 0,
                            },
                        },
                    );
                    Ok(())
                } else {
                    emit(cb, user_data, &StreamEvent::Error {
                        message: msg.clone(),
                    });
                    Err(msg)
                }
            }
        }
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            -1
        }
        Err(_) => {
            set_last_error("panic during chat completion");
            -1
        }
    }
}

/// Removes a request's cancel flag when it finishes.
fn scopeguard(req_id: String) -> impl Drop {
    struct Guard(String);
    impl Drop for Guard {
        fn drop(&mut self) {
            if let Ok(mut reg) = cancel_registry().lock() {
                reg.remove(&self.0);
            }
        }
    }
    Guard(req_id)
}

/// Requests cancellation of an in-flight completion. Returns 0 if the
/// request was found, 1 if no such request is running.
///
/// # Safety
/// `request_id` must be a valid NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn combs_cancel(request_id: *const c_char) -> i32 {
    let Ok(id) = (unsafe { read_str(request_id, "request_id") }) else {
        set_last_error("request_id is NULL or invalid UTF-8");
        return -1;
    };
    let reg = cancel_registry().lock();
    match reg {
        Ok(reg) => match reg.get(id) {
            Some(flag) => {
                flag.store(true, Ordering::Relaxed);
                0
            }
            None => 1,
        },
        Err(e) => {
            set_last_error(e.to_string());
            -1
        }
    }
}
