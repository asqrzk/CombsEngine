//! Request/response types for the JSON FFI boundary.
//!
//! Everything crossing the C ABI is JSON (MLC `json_ffi` pattern): one
//! integration surface for Deno, Kotlin, Swift and the WASM shell.
//!
//! The request and event *shapes* — and the rules for turning a request
//! into a prompt and a generation config — live in `combs-runtime`, so
//! this boundary and the browser bindings cannot drift apart about what a
//! request means. What remains here is what is genuinely C-specific.

pub use combs_runtime::{
    ChatMessageJson, ChatRequestJson, EngineConfigJson, EngineMetadataJson, StatsJson,
    StreamEvent,
};

use serde::{Deserialize, Serialize};

/// An embeddings request (`combs_embed_json`).
#[derive(Debug, Clone, Deserialize)]
pub struct EmbedRequestJson {
    /// One text or an array of texts (≤ 64).
    pub input: serde_json::Value,
    /// Matryoshka truncation: keep the first N dims, re-normalize.
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// `"last"` | `"mean"`; absent uses the checkpoint's detected default.
    #[serde(default)]
    pub pooling: Option<String>,
}

/// Embeddings response payload (`combs_embed_json`).
#[derive(Debug, Serialize)]
pub struct EmbedResponseJson {
    /// One L2-normalized vector per input text, in order.
    pub vectors: Vec<Vec<f32>>,
    /// Total input tokens across all texts.
    pub prompt_tokens: usize,
}

