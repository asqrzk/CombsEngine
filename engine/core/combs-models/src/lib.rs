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
mod qkernel;
mod qlinear;
mod qmatmul;
mod quant_linear;
mod registry;
mod rope;
mod smolvlm;
mod traits;

pub use gemma::GemmaModel;
pub use kv::{CacheConfig, CacheKind, ContiguousKVCache, KVCache, PageStats, PagedKVCache};
pub use llama::LlamaModel;
pub use norm::rms_norm;
pub use qlinear::{Linear, QuantLinearOp, try_quant_linear};
pub use qmatmul::{
    Q4KWeight, Q6KWeight, Q40Weight, Q50Weight, Q80Weight, dequantize_q4_0_gpu,
    dequantize_q4_k_gpu, dequantize_q5_0_gpu, dequantize_q6_k_gpu, dequantize_q8_0_gpu,
    repack_q4_0, repack_q4_k, repack_q5_0, repack_q6_k, repack_q8_0,
};
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
