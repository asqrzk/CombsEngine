//! # combs-formats
//!
//! File-format adapter layer. The runtime and model crates never touch file
//! formats directly; they go through the [`ModelSource`] trait (the
//! LiteRT-LM `ModelResources` equivalent). Phase 1 ships the
//! [`safetensors`] adapter (HuggingFace `config.json` + `model.safetensors`,
//! mmap-backed, zero-copy views). GGUF / ONNX / litertlm adapters plug in
//! here later by implementing the same trait.

mod metadata;
mod safetensors;
mod source;
mod tokenizer;

pub use metadata::ModelMetadata;
pub use safetensors::SafetensorsSource;
pub use source::{ModelSource, SamplerConfig, TensorDtype, TensorReader};
pub use tokenizer::TokenizerSpec;

/// Errors produced by format adapters.
#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    /// An I/O error while reading model files.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// A JSON parse error (config.json, generation_config.json, …).
    #[error("json error in {context}: {source}")]
    Json {
        /// Which file/section failed to parse.
        context: String,
        /// The underlying serde error.
        source: serde_json::Error,
    },

    /// The safetensors container is malformed.
    #[error("safetensors error: {0}")]
    Safetensors(String),

    /// A requested tensor does not exist in the source.
    #[error("tensor not found: {0}")]
    TensorNotFound(String),

    /// A tensor has an unsupported dtype for this build.
    #[error("unsupported dtype for tensor {tensor}: {dtype}")]
    UnsupportedDtype {
        /// Tensor name.
        tensor: String,
        /// Dtype string from the container.
        dtype: String,
    },

    /// The model directory is missing a required file.
    #[error("missing file: {0}")]
    MissingFile(String),

    /// The config is missing a required field.
    #[error("missing config field: {0}")]
    MissingField(String),
}

/// Convenient result alias for this crate.
pub type Result<T> = std::result::Result<T, FormatError>;
