//! Error type shared by every combs-mesh subsystem.

/// The single error type for the crate. Every fallible public API returns
/// this; readers of external data (binary/unicode/registry) must never
/// panic, so all malformed-input paths funnel here.
#[derive(Debug, thiserror::Error)]
pub enum MeshError {
    /// Binary container is malformed (bad magic, out-of-bounds directory,
    /// truncated payload, ...).
    #[error("format error: {0}")]
    Format(String),
    /// A block failed validation (e.g. atlas size does not match pixels).
    #[error("invalid block: {0}")]
    InvalidBlock(String),
    /// A block payload's CRC32 did not match the directory entry.
    #[error("crc mismatch")]
    CrcMismatch,
    /// The binary container version is not supported by this build.
    #[error("unsupported format version: {0}")]
    UnsupportedVersion(u16),
    /// Encryption/decryption failed (wrong key, tampered ciphertext, or a
    /// key was required but not supplied).
    #[error("crypto error: {0}")]
    Crypto(String),
    /// An operation required the process-wide keyring before `init`.
    #[error("keyring not initialized")]
    NotInitialized,
    /// The Unicode (PUA plane 15/16 + tag char) encoding was malformed.
    #[error("unicode error: {0}")]
    Unicode(String),
    /// Registry (content-addressed store) failure.
    #[error("registry error: {0}")]
    Registry(String),
    /// Filesystem I/O failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// JSON (de)serialization failure.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// Convenience alias used across the crate.
pub type Result<T> = std::result::Result<T, MeshError>;
