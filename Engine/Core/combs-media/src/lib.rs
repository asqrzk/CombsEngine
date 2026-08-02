//! # combs-media
//!
//! Media preprocessing — the traits-first basic block for non-text
//! modalities. Vision (SigLIP-style image preprocessing) ships first;
//! audio (mel spectrogram) lands here in the next phase. Preprocessors are
//! plain host-side code (no GPU dependency) producing normalized tensors
//! the runtime hands to the model's `embed_multimodal`.

mod images;

pub use images::{ImagePreprocessor, PixelBatch, SiglipPreprocessor};

/// Errors produced while decoding or preprocessing media.
#[derive(Debug, thiserror::Error)]
pub enum MediaError {
    /// The payload could not be decoded as an image.
    #[error("image decode failed: {0}")]
    Decode(String),
    /// The payload had an unsupported or empty shape.
    #[error("bad media shape: {0}")]
    BadShape(String),
}

/// Convenient result alias for this crate.
pub type Result<T> = std::result::Result<T, MediaError>;
