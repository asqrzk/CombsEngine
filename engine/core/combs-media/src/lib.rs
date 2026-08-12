//! # combs-media
//!
//! Media preprocessing — the traits-first basic block for non-text
//! modalities. Vision (SigLIP-style image preprocessing) and audio (WAV →
//! 16 kHz → Whisper log-mel) are both plain host-side code (no GPU
//! dependency) producing normalized tensors the runtime hands to the
//! model's embedding entry points.

mod audio;
mod images;

pub use audio::{
    decode_wav, pad_or_trim, resample_linear, LogMel, CHUNK_FRAMES, CHUNK_SAMPLES, HOP_LENGTH,
    N_FFT, N_MELS, SAMPLE_RATE,
};
pub use images::{ImagePreprocessor, PixelBatch, SiglipPreprocessor};

/// Errors produced while decoding or preprocessing media.
#[derive(Debug, thiserror::Error)]
pub enum MediaError {
    /// The payload could not be decoded as an image.
    #[error("image decode failed: {0}")]
    Decode(String),
    /// The payload could not be decoded as audio.
    #[error("audio decode failed: {0}")]
    AudioDecode(String),
    /// The payload had an unsupported or empty shape.
    #[error("bad media shape: {0}")]
    BadShape(String),
}

/// Convenient result alias for this crate.
pub type Result<T> = std::result::Result<T, MediaError>;
