//! # combs-formats
//!
//! File-format adapter layer. The runtime and model crates never touch file
//! formats directly; they go through the [`ModelSource`] trait (the
//! LiteRT-LM `ModelResources` equivalent). Phase 1 ships the
//! [`safetensors`] adapter (HuggingFace `config.json` + `model.safetensors`,
//! mmap-backed, zero-copy views). GGUF / ONNX / litertlm adapters plug in
//! here later by implementing the same trait.

mod flatbuf;
mod gguf;
mod litertlm;
mod metadata;
mod protomin;
mod safetensors;
mod source;
mod spm;
mod tflite;
mod tokenizer;

pub use gguf::GgufSource;

/// Reference CPU dequantizers for GGUF quant formats. These scalar
/// implementations are the harmony reference that the fused GPU kernels in
/// `combs-models` validate against (QUANTIZATION_PLAN.md, principle #6:
/// every kernel is tested against a portable reference).
pub mod quants {
    pub use crate::gguf::{
        dequantize_q4_0, dequantize_q4_k, dequantize_q5_0, dequantize_q5_k, dequantize_q6_k,
        dequantize_q8_0,
    };
}
pub use metadata::{Activation, AttentionPattern, ModelMetadata, RopeScaling, VisionConfig};
pub use safetensors::SafetensorsSource;
pub use litertlm::{SectionInfo, read_sections as litertlm_read_sections};
pub use source::{ModelSource, QuantFormat, QuantTensor, SamplerConfig, TensorDtype, TensorReader};
pub use spm::{ensure_tokenizer_json_from_spm, spm_added_tokens};
pub use tflite::TfliteSource;
pub use tokenizer::TokenizerSpec;

use std::path::Path;

/// Opens any supported model path: a `.gguf` file, or a directory in the
/// HuggingFace safetensors layout. This is the single entry point the CLI,
/// FFI and server use — format detection lives here.
pub fn open_model_source(path: impl AsRef<Path>) -> Result<Box<dyn ModelSource>> {
    let path = path.as_ref();
    if path.is_file() && path.extension().is_some_and(|e| e == "gguf") {
        return Ok(Box::new(GgufSource::load(path)?));
    }
    if path.is_file() && path.extension().is_some_and(|e| e == "task" || e == "tflite") {
        return Ok(Box::new(TfliteSource::load(path)?));
    }
    if path.is_file() && path.extension().is_some_and(|e| e == "litertlm") {
        return litertlm::open_litertlm(path);
    }
    if path.is_dir() {
        return Ok(Box::new(SafetensorsSource::load(path)?));
    }
    Err(FormatError::MissingFile(path.display().to_string()))
}

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
