//! `enc` — encryption-at-rest directive.
//!
//! The block itself is never encrypted (it carries the policy); the binary
//! writer encrypts the payloads of every block type listed in `apply_to`
//! when a keyring is supplied.

use serde::{Deserialize, Serialize};

use super::BlockTag;

/// Encryption directive block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncryptionBlock {
    /// The AEAD algorithm to use.
    pub algorithm: EncryptionAlgorithm,
    /// Block types to encrypt at rest. Empty = encrypt nothing.
    #[serde(default)]
    pub apply_to: Vec<BlockTag>,
}

/// Supported AEAD algorithms (RustCrypto implementations, matching the
/// WebCrypto algorithms used by `@combs/zerotrust`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EncryptionAlgorithm {
    /// AES-256-GCM.
    Aes256Gcm,
    /// ChaCha20-Poly1305.
    ChaCha20Poly1305,
}
