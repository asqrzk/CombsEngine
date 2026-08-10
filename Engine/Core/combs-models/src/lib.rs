//! # combs-models
//!
//! Model architecture registry. The runtime drives any architecture through
//! the fixed [`GenerativeModel`] contract (MLC's `embed/prefill/decode/
//! create_kv_cache` function set); architectures register themselves in the
//! [`ModelRegistry`]. Phase 1 ships the Llama family (incl. SmolLM2).

mod kv;
mod gemma;
mod llama;
mod matmul;
mod norm;
mod precision;
mod quant_linear;
mod registry;
mod rope;
mod smolvlm;
mod traits;

pub use gemma::GemmaModel;
pub use kv::{CacheConfig, CacheKind, ContiguousKVCache, KVCache, PagedKVCache};
pub use llama::LlamaModel;
pub use norm::rms_norm;
pub use quant_linear::QuantizedLinear;
pub use registry::ModelRegistry;
pub use rope::RotaryEmbedding;
pub use smolvlm::{SmolVlmModel, image_prompt_expansion, pixels_to_tensor};
pub use traits::GenerativeModel;

/// Errors produced while constructing or running models.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// A format-adapter error.
    #[error(transparent)]
    Format(#[from] combs_formats::FormatError),

    /// No registered architecture matches the source metadata.
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    /// Media input (image/audio) was passed to a model that cannot take it.
    #[error("unsupported media input: {0}")]
    UnsupportedMedia(String),

    /// A required weight tensor is missing from the source.
    #[error("missing weight tensor: {0}")]
    MissingTensor(String),

    /// A weight tensor has an unexpected shape.
    #[error("bad shape for {tensor}: expected {expected:?}, got {got:?}")]
    BadShape {
        /// Tensor name.
        tensor: String,
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        got: Vec<usize>,
    },
}

/// Convenient result alias for this crate.
pub type Result<T> = std::result::Result<T, ModelError>;
