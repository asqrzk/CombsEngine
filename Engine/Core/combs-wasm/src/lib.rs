//! # combs-wasm — browser bindings (EXPERIMENTAL, Phase 5 skeleton)
//!
//! This crate is the seam for browser inference: the same engine compiled
//! to `wasm32-unknown-unknown` with burn's WebGpu backend, exposed to JS via
//! wasm-bindgen, driven by the `@combs/core` WorkerEngine transport
//! (web-llm pattern: engine in a Web Worker, pull-based streaming).
//!
//! ## Status
//!
//! The desktop/mobile engine stack compiles for native targets today. The
//! remaining browser-specific work (tracked here so the seam is explicit):
//!
//! 1. burn 0.21 wgpu on wasm: `init_setup_async` + the `WebGpu` compiler
//!    alias instead of the `AutoGraphicsApi` sync init used on desktop.
//! 2. Weight delivery: OPFS read chunks → `ModelSource` over a JS-provided
//!    byte provider (the `WasmModelSource` below is the intended shape).
//! 3. `getrandom` needs the `js` feature; `console_error_panic_hook` for
//!    debugging; no threads in v1 (single-flight only).
//!
//! ## JS side (already compatible)
//!
//! `Engine/Js/core` defines the `EngineClient` contract; a `WorkerEngine`
//! implementation postMessages to a worker hosting this wasm module —
//! same `{type:"delegate"}` envelope as `@combs/runtime` agents, so
//! orchestration code doesn't change.

use wasm_bindgen::prelude::*;

/// Version probe so the JS side can verify the module loaded.
#[wasm_bindgen]
pub fn combs_wasm_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Planned JS-facing engine API (mirrors the FFI JSON envelope):
///
/// ```text
/// combs_engine_create(config_json) -> engine id
/// combs_chat_completion(id, request_json, on_event: js_sys::Function)
/// combs_cancel(request_id)
/// combs_device_caps() -> JsValue
/// ```
///
/// Implementation lands with the WebGpu backend enablement (see module docs).
#[wasm_bindgen]
pub fn combs_wasm_ping(input: &str) -> String {
    format!("combs-wasm ok: {input}")
}
