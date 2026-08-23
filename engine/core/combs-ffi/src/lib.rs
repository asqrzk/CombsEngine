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

        let model_dir = config
            .model_dir
            .as_deref()
            .ok_or("engine config needs `model_dir`")?;
        let source = open_model_source(model_dir)
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

/// Embeds texts into L2-normalized vectors. Blocking; returns the response
/// JSON (`{"vectors": [[...]], "prompt_tokens": n}`) or NULL on error (see
/// `combs_last_error`). Free with `combs_string_free`.
///
/// Request: `{"input": "..." | ["..."], "dimensions"?: n,
/// "pooling"?: "last" | "mean"}` — absent pooling uses the checkpoint's
/// detected default (`1_Pooling/config.json`, else last-token).
///
/// # Safety
/// `engine` must be a valid handle and `request_json` a valid
/// NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn combs_embed_json(
    engine: *const CombsEngine,
    request_json: *const c_char,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<EmbedResponseJson, String> {
        if engine.is_null() {
            return Err("engine is NULL".to_string());
        }
        let engine = unsafe { &*engine };
        let json = unsafe { read_str(request_json, "request_json") }?;
        let request: EmbedRequestJson = serde_json::from_str(json)
            .map_err(|e| format!("invalid embed request JSON: {e}"))?;

        let texts: Vec<String> = match &request.input {
            serde_json::Value::String(s) => vec![s.clone()],
            serde_json::Value::Array(a) => a
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| "input array entries must be strings".to_string())
                })
                .collect::<Result<_, _>>()?,
            _ => return Err("input must be a string or an array of strings".to_string()),
        };
        if texts.is_empty() || texts.len() > 64 {
            return Err("input must contain 1..=64 texts".to_string());
        }
        let pooling = match request.pooling.as_deref() {
            None => None,
            Some("last") => Some(combs_runtime::Pooling::Last),
            Some("mean") => Some(combs_runtime::Pooling::Mean),
            Some(other) => return Err(format!("unknown pooling: {other:?}")),
        };
        let opts = combs_runtime::EmbedOptions {
            pooling,
            dimensions: request.dimensions,
        };
        let out = engine
            .engine
            .embed_texts(&texts, &opts)
            .map_err(|e| e.to_string())?;
        Ok(EmbedResponseJson {
            vectors: out.vectors,
            prompt_tokens: out.prompt_tokens,
        })
    }));

    match result {
        Ok(Ok(resp)) => into_raw_json(&resp),
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("panic during embeddings");
            ptr::null_mut()
        }
    }
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

        let resolved = combs_runtime::resolve_chat_request(
            &engine.engine,
            &request,
            engine.prefill_chunk_size,
        )?;
        let combs_runtime::ResolvedChat {
            prompt_tokens,
            config,
            parser_style,
        } = resolved;

        let cancel = Arc::new(AtomicBool::new(false));
        cancel_registry()
            .lock()
            .map_err(|e| e.to_string())?
            .insert(req_id.clone(), cancel.clone());
        let _guard = scopeguard(req_id.clone());

        let mut parser = combs_runtime::ToolCallParser::new(parser_style);
        let mut tool_calls: Vec<serde_json::Value> = Vec::new();
        let mut handle_event = |ev: combs_runtime::ToolEvent,
                                token_id: u32,
                                logprob: Option<f32>,
                                calls: &mut Vec<serde_json::Value>| match ev {
            combs_runtime::ToolEvent::Content(text) => {
                emit(
                    cb,
                    user_data,
                    &StreamEvent::Delta {
                        text,
                        token_id,
                        logprob,
                    },
                );
            }
            combs_runtime::ToolEvent::Call(c) => {
                calls.push(serde_json::json!({
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": serde_json::to_string(&c.function.arguments)
                            .unwrap_or_else(|_| "{}".to_string()),
                    },
                }));
            }
        };
        let outcome = engine.engine.generate_cancellable(
            &prompt_tokens,
            &config,
            cancel,
            |token_id, text, lp| {
                let logprob = lp.map(|l| l.logprob);
                for ev in parser.push(text) {
                    handle_event(ev, token_id, logprob, &mut tool_calls);
                }
            },
        );
        for ev in parser.finish() {
            handle_event(ev, 0, None, &mut tool_calls);
        }

        match outcome {
            Ok(stats) => {
                let reason =
                    combs_runtime::finish_reason(&stats, config.max_tokens, tool_calls.len());
                emit(
                    cb,
                    user_data,
                    &StreamEvent::Done {
                        finish_reason: reason.into(),
                        tool_calls,
                        stats: StatsJson::from(&stats),
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
                            tool_calls: Vec::new(),
                            stats: StatsJson::default(),
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
