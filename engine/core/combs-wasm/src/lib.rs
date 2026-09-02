//! # combs-wasm — the engine, in a browser tab
//!
//! The same Rust engine that runs the CLI, compiled to
//! `wasm32-unknown-unknown` on burn's WebGPU backend and exposed to
//! JavaScript. Nothing here is a browser-specific *engine*: it is a
//! browser-specific *driver* over [`combs_runtime::LocalEngine`], which
//! decodes with the same code as every other surface.
//!
//! ## What the browser changes
//!
//! Three things, and only three:
//!
//! 1. **Device setup is async.** cubecl cannot create a wgpu setup
//!    synchronously on the web, so [`combs_device_caps`] must complete
//!    before any tensor work. Rather than document that as a rule callers
//!    can get wrong, every entry point that needs a device waits on the
//!    same one-shot initialization itself.
//! 2. **The model arrives as bytes.** There is no path to open, so the
//!    caller hands over the whole GGUF image and the engine parses it in
//!    place.
//! 3. **Nothing may block.** One token per await, so the page keeps its
//!    thread between them — which is also what makes Stop work without the
//!    engine knowing what a Stop button is.
//!
//! ## Shape
//!
//! Events are JSON strings in the same shapes the C FFI emits, so a client
//! written against one transport reads the other unchanged. State lives in
//! thread-locals: a wasm module is single-threaded, and pretending
//! otherwise with locks would only add ways to fail.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use combs_runtime::{
    ChatRequestJson, EngineConfigJson, EngineMetadataJson, LocalEngine, ResolvedChat, StatsJson,
    StepEvent, StreamEvent, ToolCallParser, ToolEvent,
};
use wasm_bindgen::prelude::*;

mod mount;

/// Installs the panic hook so a Rust panic reaches the browser console as a
/// stack trace instead of an opaque `unreachable executed`.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    web_sys_log(&format!(
        "[combs-wasm] {} {}",
        env!("CARGO_PKG_VERSION"),
        SERVING_DTYPE
    ));
}

/// One line to the browser console, without pulling in web-sys.
fn web_sys_log(text: &str) {
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = console)]
        fn log(s: &str);
    }
    log(text);
}

thread_local! {
    /// Loaded engines by id. An engine is *removed* while a request runs,
    /// which is how single-flight is enforced: a second request for the
    /// same engine finds it absent and is refused rather than corrupting
    /// the first one's KV cache.
    static ENGINES: RefCell<HashMap<u32, LocalEngine>> = RefCell::new(HashMap::new());
    static NEXT_ID: RefCell<u32> = const { RefCell::new(1) };
    /// Cancel flags for in-flight requests, keyed by request id. Held apart
    /// from the engine so a Stop can land while the engine is checked out.
    static CANCELS: RefCell<HashMap<String, Arc<AtomicBool>>> = RefCell::new(HashMap::new());
    /// In-flight chunk-appended mounts by handle (ids share NEXT_ID with
    /// engines, so a mount handle can never be mistaken for an engine id
    /// that exists). `finish` REMOVES the state before doing anything
    /// else — a duplicate finish or a late append finds "no such mount"
    /// instead of a half-consumed buffer.
    static MOUNTS: RefCell<HashMap<u32, Mount>> = RefCell::new(HashMap::new());
    /// The one device this module ever creates, and its capabilities.
    /// cubecl panics on a second setup, so this is initialized once and
    /// every later caller reuses it.
    static DEVICE: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Initializes the wgpu device once and returns its capabilities as JSON.
///
/// Safe to call repeatedly: the setup runs on the first call and later
/// calls return the cached answer.
async fn ensure_device() -> Result<String, String> {
    if let Some(caps) = DEVICE.with(|d| d.borrow().clone()) {
        return Ok(caps);
    }
    let device = combs_core::init_device();
    let caps = combs_core::device_caps_async(&device).await;
    let json = serde_json::to_string(&caps).map_err(|e| e.to_string())?;
    DEVICE.with(|d| *d.borrow_mut() = Some(json.clone()));
    Ok(json)
}

fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Adapter name, backend, buffer limits and compute limits, as JSON.
///
/// This is the planner's input: it is what decides whether a given model
/// fits this machine before a byte of it is downloaded.
#[wasm_bindgen]
pub async fn combs_device_caps() -> Result<String, JsValue> {
    ensure_device().await.map_err(js_err)
}

/// Validates the hand-written kernel contract on THIS browser's WebGPU
/// stack: the scalar uniform layout and workgroup-memory barriers, judged
/// by output values rather than by the absence of errors — a shader that
/// fails Tint validation dispatches as a silent no-op here, so silence
/// proves nothing. Returns "ok" or throws with the first violation.
///
/// Run and record this green before enabling any COMBS_WGSL door in a
/// browser build; naga passing natively says nothing about Tint.
#[wasm_bindgen]
pub async fn combs_wgsl_probe() -> Result<String, JsValue> {
    ensure_device().await.map_err(js_err)?;
    combs_models::wgsl_probe_report()
        .await
        .map(|()| "ok".to_string())
        .map_err(js_err)
}

/// Value-checks the batched quant matmul — the path every paste takes
/// in a tab — against a host reference on THIS browser's compiler.
/// Always compiled: §70's class (a valid launch that writes nothing)
/// is exactly what only a value probe can catch, and the batched
/// path's types already ship in the bundle.
#[wasm_bindgen]
pub async fn combs_batched_probe() -> Result<String, JsValue> {
    ensure_device().await.map_err(js_err)?;
    combs_models::batched_probe_report()
        .await
        .map(|()| "ok".to_string())
        .map_err(js_err)
}

/// Value-checks the burn-composed forward paths (manual masked
/// attention, flash attention at head_dim 256) against a host reference
/// on THIS browser's compiler — the paths only GeluTanh-family models
/// exercise, which the WGSL probe cannot cover.
/// Compiled only with `--features forward-probe`: the burn-path canaries
/// monomorphize ~6.8 MB of module the shipped bundle doesn't need — the
/// always-on `combs_wgsl_probe` covers every Combs Kernel (tanh canary
/// included); this one exists for investigating burn-composed paths on
/// a misbehaving browser, which is an investigation build, not a ship.
#[cfg(feature = "forward-probe")]
#[wasm_bindgen]
pub async fn combs_forward_probe() -> Result<String, JsValue> {
    ensure_device().await.map_err(js_err)?;
    combs_models::forward_probe_report()
        .await
        .map(|()| "ok".to_string())
        .map_err(js_err)
}

/// Creates an engine from a config JSON and the model's bytes.
///
/// `model_bytes` must be the complete GGUF image: its tensor offsets are
/// absolute within the file, so a partial buffer is not a smaller model but
/// a wrong one. Returns an engine id for the calls below.
/// An explicit arena size must be one the model can actually use: the
/// f5 trap gate found both 0 and 50,000,000 loading without complaint —
/// a zero arena holds no prompt, and positions past the model's
/// positional maximum are unaddressable, so both failures deferred to
/// mid-chat instead of surfacing at load.
fn validate_arena(explicit: Option<usize>, positional_max: usize) -> Result<(), String> {
    if let Some(n) = explicit {
        if n == 0 {
            return Err("max_seq_len 0: a zero-token arena holds no prompt".to_string());
        }
        if n > positional_max {
            return Err(format!(
                "max_seq_len {n} exceeds the model's positional maximum {positional_max}"
            ));
        }
    }
    Ok(())
}

#[wasm_bindgen]
pub async fn combs_engine_create(
    config_json: String,
    model_bytes: Vec<u8>,
) -> Result<u32, JsValue> {
    ensure_device().await.map_err(js_err)?;
    let config: EngineConfigJson = serde_json::from_str(&config_json)
        .map_err(|e| js_err(format!("invalid engine config JSON: {e}")))?;
    create_engine_from_bytes(config, model_bytes)
}

/// The shared load tail: parse the in-memory image, build the engine,
/// register it. Both the one-shot create above and the chunked mount's
/// finish land here.
fn create_engine_from_bytes(
    config: EngineConfigJson,
    model_bytes: Vec<u8>,
) -> Result<u32, JsValue> {
    let source = combs_formats::open_model_source_bytes(model_bytes)
        .map_err(|e| js_err(format!("loading model: {e}")))?;
    create_engine_from_source(config, source)
}

/// The same engine tail as [`create_engine_from_source`], from weights a
/// stream already staged. Split rather than shared because the two
/// differ in exactly one thing — where the model comes from — and the
/// cache configuration below must not drift between them.
fn create_engine_from_staged(
    config: EngineConfigJson,
    staged: &mut combs_models::staged::StagedWeights<combs_core::CombsBackend>,
    header: combs_formats::GgufSource,
) -> Result<u32, JsValue> {
    use combs_formats::ModelSource as _;
    let mut cache_config = combs_runtime::CacheConfig::paged(
        {
            validate_arena(config.max_seq_len, header.metadata().max_position_embeddings)
                .map_err(js_err)?;
            config.max_seq_len.unwrap_or_else(|| {
                combs_runtime::default_arena_len(header.metadata().max_position_embeddings)
            })
        },
    );
    if let Some(ps) = config.page_size {
        cache_config.page_size = ps;
    }
    cache_config.kind = match config.kv_cache.as_deref() {
        None | Some("paged") => combs_runtime::CacheKind::Paged,
        Some("contiguous") => combs_runtime::CacheKind::Contiguous,
        Some(other) => return Err(js_err(format!("unknown kv_cache kind: {other}"))),
    };
    let engine = LocalEngine::load_staged(
        staged,
        &header,
        combs_core::init_device(),
        cache_config,
    )
    .map_err(|e| js_err(format!("engine load failed: {e}")))?;
    let id = NEXT_ID.with(|n| {
        let mut n = n.borrow_mut();
        let id = *n;
        *n += 1;
        id
    });
    ENGINES.with(|m| m.borrow_mut().insert(id, engine));
    Ok(id)
}

fn create_engine_from_source(
    config: EngineConfigJson,
    source: Box<dyn combs_formats::ModelSource>,
) -> Result<u32, JsValue> {

    let mut cache_config = combs_runtime::CacheConfig::paged(
        {
            validate_arena(config.max_seq_len, source.metadata().max_position_embeddings)
                .map_err(js_err)?;
            config.max_seq_len.unwrap_or_else(|| {
                combs_runtime::default_arena_len(source.metadata().max_position_embeddings)
            })
        },
    );
    if let Some(ps) = config.page_size {
        cache_config.page_size = ps;
    }
    cache_config.kind = match config.kv_cache.as_deref() {
        None | Some("paged") => combs_runtime::CacheKind::Paged,
        Some("contiguous") => combs_runtime::CacheKind::Contiguous,
        Some(other) => return Err(js_err(format!("unknown kv_cache kind: {other}"))),
    };

    let engine = LocalEngine::load_with_cache_config(&source, combs_core::init_device(), cache_config)
        .map_err(|e| js_err(format!("engine load failed: {e}")))?;

    let id = NEXT_ID.with(|n| {
        let mut n = n.borrow_mut();
        let id = *n;
        *n += 1;
        id
    });
    ENGINES.with(|m| m.borrow_mut().insert(id, engine));
    Ok(id)
}

/// Creates an engine from an ONNX graph and the siblings a directory
/// would have held: `{ config, tokenizer, tokenizer_config?,
/// generation_config?, chat_template? }`, each a JSON string.
///
/// Separate from [`combs_engine_create`] because ONNX has no magic
/// bytes to recognize it by — a graph is protobuf, and guessing at
/// protobuf is how you load the wrong thing confidently. The caller
/// says what it is handing over. The reader itself was already in the
/// module; it was only ever unreachable from a browser, which has no
/// directory to read siblings from.
#[wasm_bindgen]
pub async fn combs_engine_create_onnx(
    config_json: String,
    graph: Vec<u8>,
    siblings_json: String,
) -> Result<u32, JsValue> {
    ensure_device().await.map_err(js_err)?;
    let config: EngineConfigJson = serde_json::from_str(&config_json)
        .map_err(|e| js_err(format!("invalid engine config JSON: {e}")))?;
    let s: serde_json::Value = serde_json::from_str(&siblings_json)
        .map_err(|e| js_err(format!("invalid onnx siblings JSON: {e}")))?;
    let text = |key: &str| s.get(key).and_then(|v| v.as_str()).map(String::from);
    let required = |key: &str| {
        text(key).ok_or_else(|| js_err(format!("onnx siblings: {key} is required")))
    };
    let source = combs_formats::OnnxSource::from_parts(
        graph,
        combs_formats::OnnxSiblings {
            config: required("config")?,
            tokenizer_json: required("tokenizer")?.into_bytes(),
            tokenizer_config: text("tokenizer_config"),
            generation_config: text("generation_config"),
            chat_template: text("chat_template"),
            sidecars: std::collections::HashMap::new(),
        },
    )
    .map_err(|e| js_err(format!("loading onnx model: {e}")))?;
    create_engine_from_source(config, Box::new(source))
}

/// A mount in flight, in whichever of the two ways.
///
/// Buffering holds the image and hands it over whole at the end, which
/// is the right shape up to about three gigabytes. Streaming never
/// holds it: tensors are uploaded as their bytes pass and the bytes are
/// dropped, which is the only shape that fits a 7B checkpoint under a
/// tab's four gigabytes of address space.
enum Mount {
    Buffering(mount::MountState),
    Streaming {
        config_json: String,
        mount: combs_runtime::stream_mount::StreamMount,
    },
}

/// Opens a chunk-appended mount: validates the engine config and the
/// declared byte length eagerly, reserves what the mode needs while the
/// heap is still small, and initializes the device so it overlaps the
/// download. Mode is `"buffer"` or `"stream"`. Returns a mount handle
/// for [`combs_model_append`] / [`combs_model_finish`].
#[wasm_bindgen]
pub async fn combs_model_open(
    config_json: String,
    expected_len: f64,
    mode: String,
) -> Result<u32, JsValue> {
    ensure_device().await.map_err(js_err)?;
    // Config problems must surface before gigabytes move, not after.
    let _: EngineConfigJson = serde_json::from_str(&config_json)
        .map_err(|e| js_err(format!("invalid engine config JSON: {e}")))?;
    if mode == "stream" {
        if !expected_len.is_finite() || expected_len < 1.0 || expected_len.fract() != 0.0 {
            return Err(js_err(format!(
                "expected_len must be a positive whole byte count, got {expected_len}"
            )));
        }
        let state = Mount::Streaming {
            config_json,
            mount: combs_runtime::stream_mount::StreamMount::new(
                expected_len as u64,
                combs_core::init_device(),
            ),
        };
        let id = NEXT_ID.with(|n| {
            let mut n = n.borrow_mut();
            let id = *n;
            *n += 1;
            id
        });
        MOUNTS.with(|m| m.borrow_mut().insert(id, state));
        return Ok(id);
    }
    let state = Mount::Buffering(mount::open(config_json, expected_len, &mode).map_err(js_err)?);
    let id = NEXT_ID.with(|n| {
        let mut n = n.borrow_mut();
        let id = *n;
        *n += 1;
        id
    });
    MOUNTS.with(|m| m.borrow_mut().insert(id, state));
    Ok(id)
}

/// Appends one chunk to an open mount. Sync on purpose — bytes are
/// already in the wasm heap when this is called, and the async side
/// (the worker's reader loop) is the driver.
#[wasm_bindgen]
pub fn combs_model_append(handle: u32, chunk: &[u8]) -> Result<(), JsValue> {
    MOUNTS.with(|m| {
        let mut m = m.borrow_mut();
        let state = m
            .get_mut(&handle)
            .ok_or_else(|| js_err(format!("no such mount: {handle}")))?;
        match state {
            Mount::Buffering(state) => mount::append(state, chunk).map_err(js_err),
            // The streaming append does the uploading too, so this is
            // where a mount spends most of its time — and why the
            // worker calls it from its reader loop rather than the page.
            Mount::Streaming { mount, .. } => {
                mount.append(chunk).map(|_| ()).map_err(|e| js_err(e.to_string()))
            }
        }
    })
}

/// Completes a mount: verifies every declared byte arrived, then runs
/// the same load tail as [`combs_engine_create`] and returns the
/// engine id. The mount state is consumed FIRST, so a duplicate
/// finish or a late append is "no such mount", never double work.
#[wasm_bindgen]
pub async fn combs_model_finish(handle: u32) -> Result<u32, JsValue> {
    let state = MOUNTS
        .with(|m| m.borrow_mut().remove(&handle))
        .ok_or_else(|| js_err(format!("no such mount: {handle}")))?;
    match state {
        Mount::Buffering(state) => {
            mount::finish_check(&state).map_err(js_err)?;
            let config: EngineConfigJson = serde_json::from_str(&state.config_json)
                .map_err(|e| js_err(format!("invalid engine config JSON: {e}")))?;
            let source = combs_formats::open_model_source_segments(
                state.into_segments(),
                mount::SEGMENT_LEN,
            )
            .map_err(|e| js_err(format!("loading model: {e}")))?;
            create_engine_from_source(config, source)
        }
        Mount::Streaming { config_json, mount } => {
            let config: EngineConfigJson = serde_json::from_str(&config_json)
                .map_err(|e| js_err(format!("invalid engine config JSON: {e}")))?;
            let (mut staged, header) = mount.finish().map_err(|e| js_err(e.to_string()))?;
            create_engine_from_staged(config, &mut staged, header)
        }
    }
}

/// Drops a mount and frees its buffer. Idempotent: aborting a handle
/// that is gone (or never existed) is a no-op, so error-path cleanup
/// can never fail.
#[wasm_bindgen]
pub fn combs_model_abort(handle: u32) {
    MOUNTS.with(|m| m.borrow_mut().remove(&handle));
}

/// The engine's model metadata as JSON (same shape as the C FFI's).
#[wasm_bindgen]
pub fn combs_engine_metadata(engine_id: u32) -> Result<String, JsValue> {
    ENGINES.with(|m| {
        let map = m.borrow();
        let engine = map
            .get(&engine_id)
            .ok_or_else(|| js_err("no such engine, or a request is running on it"))?;
        let md = engine.metadata();
        let cc = engine.cache_config();
        serde_json::to_string(&EngineMetadataJson {
            architecture: md.architecture.clone(),
            vocab_size: md.vocab_size,
            max_position_embeddings: md.max_position_embeddings,
            max_seq_len: cc.max_seq_len,
            page_size: cc.page_size,
            eos_token_ids: md.eos_token_ids.clone(),
            im_end_id: engine.im_end_id(),
        })
        .map_err(js_err)
    })
}

/// The engine's live KV state as JSON: cache geometry and every rolling
/// session with its history length and page usage.
///
/// The native engine publishes a snapshot its host polls over HTTP; a
/// browser engine has no port and no thread to publish from, so its host
/// asks it directly. Without this, a page can watch its own model answer
/// and still have no idea what its cache is doing — which is precisely
/// the state a KV bug is invisible in.
#[wasm_bindgen]
pub fn combs_engine_stats(engine_id: u32) -> Result<String, JsValue> {
    ENGINES.with(|m| {
        let map = m.borrow();
        let engine = map
            .get(&engine_id)
            .ok_or_else(|| js_err("no such engine, or a request is running on it"))?;
        let cc = engine.cache_config();
        let sessions: Vec<serde_json::Value> = engine
            .sessions()
            .iter()
            .map(|s| {
                serde_json::json!({
                    "id": s.id,
                    "history_tokens": s.history_len,
                    "pages_used": s.pages.map(|p| p.pages_used),
                })
            })
            .collect();
        serde_json::to_string(&serde_json::json!({
            "kind": match cc.kind {
                combs_runtime::CacheKind::Paged => "paged",
                combs_runtime::CacheKind::Contiguous => "contiguous",
            },
            "quantized": cc.quantize_kv,
            "max_seq_len": cc.max_seq_len,
            "page_size": cc.page_size,
            "page_bytes": engine.kv_page_bytes(),
            "sessions": sessions,
        }))
        .map_err(js_err)
    })
}

/// Frees an engine and everything it holds (weights, KV arenas).
#[wasm_bindgen]
pub fn combs_engine_destroy(engine_id: u32) {
    ENGINES.with(|m| m.borrow_mut().remove(&engine_id));
}

/// Asks a running request to stop. Returns true if the request was found.
///
/// The flag is read between tokens, so a stopped turn keeps whatever text
/// already arrived rather than discarding it.
#[wasm_bindgen]
pub fn combs_cancel(request_id: &str) -> bool {
    CANCELS.with(|c| match c.borrow().get(request_id) {
        Some(flag) => {
            flag.store(true, Ordering::Relaxed);
            true
        }
        None => false,
    })
}

/// Returns the engine to the table and forgets the request's cancel flag,
/// on every exit path including an error.
struct RequestGuard {
    engine_id: u32,
    request_id: String,
    engine: Option<LocalEngine>,
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        if let Some(mut engine) = self.engine.take() {
            // The request may have ended without reaching a terminal step —
            // a page navigating away drops the future mid-await. Clearing
            // the generation here is what keeps the engine usable
            // afterwards instead of permanently busy with a turn nobody is
            // waiting for.
            engine.abandon();
            ENGINES.with(|m| m.borrow_mut().insert(self.engine_id, engine));
        }
        CANCELS.with(|c| {
            c.borrow_mut().remove(&self.request_id);
        });
    }
}

fn emit(on_event: &js_sys::Function, event: &StreamEvent) {
    if let Ok(json) = serde_json::to_string(event) {
        // A throwing callback is the page's problem, not the engine's: the
        // generation keeps its own state consistent either way.
        let _ = on_event.call1(&JsValue::NULL, &JsValue::from_str(&json));
    }
}

/// Runs a chat completion, calling `on_event` once per event with a JSON
/// string: `delta` per token, then exactly one terminal `done` or `error`.
///
/// Resolves when the request finishes. Between every two tokens the page
/// gets its thread back, so rendering, input and [`combs_cancel`] all
/// interleave with decoding.
#[wasm_bindgen]
pub async fn combs_chat_completion(
    engine_id: u32,
    request_id: String,
    request_json: String,
    on_event: js_sys::Function,
) -> Result<(), JsValue> {
    // Check the engine out for the duration. A concurrent request for the
    // same engine finds it gone and is told so, rather than interleaving
    // two generations through one KV cache.
    let engine = ENGINES.with(|m| m.borrow_mut().remove(&engine_id));
    let Some(engine) = engine else {
        let msg = "no such engine, or a request is already running on it".to_string();
        emit(&on_event, &StreamEvent::Error {
            message: msg.clone(),
        });
        return Err(js_err(msg));
    };
    let mut guard = RequestGuard {
        engine_id,
        request_id: request_id.clone(),
        engine: Some(engine),
    };
    let engine = guard.engine.as_mut().expect("just placed");

    let result = run_chat(engine, &request_id, &request_json, &on_event).await;
    match result {
        Ok(()) => Ok(()),
        Err(message) => {
            emit(&on_event, &StreamEvent::Error {
                message: message.clone(),
            });
            Err(js_err(message))
        }
    }
}

async fn run_chat(
    engine: &mut LocalEngine,
    request_id: &str,
    request_json: &str,
    on_event: &js_sys::Function,
) -> Result<(), String> {
    let request: ChatRequestJson =
        serde_json::from_str(request_json).map_err(|e| format!("invalid request JSON: {e}"))?;
    let ResolvedChat {
        prompt_tokens,
        config,
        parser_style,
    } = combs_runtime::resolve_chat_request(engine, &request, None)?;

    engine
        .begin(&prompt_tokens, &config)
        .map_err(|e| e.to_string())?;
    CANCELS.with(|c| {
        c.borrow_mut()
            .insert(request_id.to_string(), engine.cancel_handle())
    });

    let mut parser = ToolCallParser::new(parser_style);
    let mut tool_calls: Vec<serde_json::Value> = Vec::new();
    let push_text = |text: &str,
                         token_id: u32,
                         logprob: Option<f32>,
                         parser: &mut ToolCallParser,
                         calls: &mut Vec<serde_json::Value>| {
        for ev in parser.push(text) {
            match ev {
                ToolEvent::Content(text) => emit(on_event, &StreamEvent::Delta {
                    text,
                    token_id,
                    logprob,
                }),
                ToolEvent::Call(c) => calls.push(serde_json::json!({
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": serde_json::to_string(&c.function.arguments)
                            .unwrap_or_else(|_| "{}".to_string()),
                    },
                })),
            }
        }
    };

    loop {
        match engine.step().await {
            Ok(StepEvent::Token { id, text, logprobs }) => {
                let logprob = logprobs.map(|l| l.logprob);
                push_text(&text, id, logprob, &mut parser, &mut tool_calls);
            }
            Ok(StepEvent::Done { tail, stats }) => {
                if !tail.is_empty() {
                    push_text(&tail, 0, None, &mut parser, &mut tool_calls);
                }
                for ev in parser.finish() {
                    match ev {
                        ToolEvent::Content(text) => emit(on_event, &StreamEvent::Delta {
                            text,
                            token_id: 0,
                            logprob: None,
                        }),
                        ToolEvent::Call(c) => tool_calls.push(serde_json::json!({
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.function.name,
                                "arguments": serde_json::to_string(&c.function.arguments)
                                    .unwrap_or_else(|_| "{}".to_string()),
                            },
                        })),
                    }
                }
                let reason =
                    combs_runtime::finish_reason(&stats, config.max_tokens, tool_calls.len());
                emit(on_event, &StreamEvent::Done {
                    finish_reason: reason.into(),
                    tool_calls,
                    stats: StatsJson::from(&*stats),
                });
                return Ok(());
            }
            Err(combs_runtime::EngineError::Cancelled) => {
                // A stopped turn is a normal outcome, and the text that
                // already arrived stays on screen.
                for ev in parser.finish() {
                    if let ToolEvent::Content(text) = ev {
                        emit(on_event, &StreamEvent::Delta {
                            text,
                            token_id: 0,
                            logprob: None,
                        });
                    }
                }
                emit(on_event, &StreamEvent::Done {
                    finish_reason: "cancelled".into(),
                    tool_calls: Vec::new(),
                    stats: StatsJson::default(),
                });
                return Ok(());
            }
            Err(e) => return Err(e.to_string()),
        }
    }
}

/// What this module is, so a page checks rather than assumes. The
/// serving dtype is a compile-time choice that changes the backend
/// type, and a bundle is deployed by copying files — the two ways to
/// end up running something other than what you think you are.
#[wasm_bindgen]
pub fn combs_wasm_version() -> String {
    format!("{} {}", env!("CARGO_PKG_VERSION"), SERVING_DTYPE)
}

/// The float width this module serves in, fixed at compile time.
pub const SERVING_DTYPE: &str = if cfg!(feature = "f16") { "f16" } else { "f32" };
