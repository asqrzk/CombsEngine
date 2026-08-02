//! Key management + AEAD encryption (pure Rust — RustCrypto, no C deps, so
//! wasm32 and mobile cross-builds stay clean; same algorithms as
//! `@combs/zerotrust`'s WebCrypto stack).
//!
//! Layout of an encrypted payload: `nonce (12 bytes, random) || ciphertext
//! || tag` — the nonce travels with the message, keys never leave the
//! [`KeyRing`] (master key held in [`Zeroizing`]).
//!
//! A process-wide keyring lives behind `OnceLock<RwLock<…>>` with
//! [`init`]/[`shutdown`]/[`global`] so the FFI crate (`combsmesh_init` /
//! `combsmesh_shutdown`) is a thin shim over this module.

use std::sync::{OnceLock, RwLock};

use aes_gcm::aead::{Aead, AeadCore, KeyInit, OsRng};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::Zeroizing;

use crate::blocks::EncryptionAlgorithm;
use crate::error::{MeshError, Result};

/// Default HKDF info string for emoji-at-rest encryption subkeys.
pub const DEFAULT_HKDF_INFO: &str = "combsmesh-emoji-encryption";

/// Nonce size in bytes (96-bit, the standard AEAD nonce).
pub const NONCE_LEN: usize = 12;

/// Holds the master key and derives purpose-specific subkeys via
/// HKDF-SHA256. Key material is zeroized on drop.
#[derive(Clone)]
pub struct KeyRing {
    master: Zeroizing<Vec<u8>>,
}

impl KeyRing {
    /// Creates a keyring from `master`, or generates 32 random bytes when
    /// `None` is given.
    #[must_use]
    pub fn new(master: Option<&[u8]>) -> Self {
        let master = match master {
            Some(bytes) => bytes.to_vec(),
            None => Aes256Gcm::generate_key(&mut OsRng).to_vec(),
        };
        KeyRing {
            master: Zeroizing::new(master),
        }
    }

    /// Derives a 32-byte subkey via HKDF-SHA256 with the given info string.
    pub fn subkey(&self, info: &str) -> Result<Zeroizing<[u8; 32]>> {
        let hk = Hkdf::<Sha256>::new(None, &self.master);
        let mut okm = [0u8; 32];
        hk.expand(info.as_bytes(), &mut okm)
            .map_err(|_| MeshError::Crypto("HKDF expand failed".into()))?;
        Ok(Zeroizing::new(okm))
    }

    /// Encrypts with the default emoji-encryption subkey. Output is
    /// `nonce(12) || ciphertext`.
    pub fn encrypt(&self, data: &[u8], algorithm: EncryptionAlgorithm) -> Result<Vec<u8>> {
        let subkey = self.subkey(DEFAULT_HKDF_INFO)?;
        match algorithm {
            EncryptionAlgorithm::Aes256Gcm => {
                let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&subkey[..]));
                let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
                let ct = cipher
                    .encrypt(&nonce, data)
                    .map_err(|_| MeshError::Crypto("AES-256-GCM encrypt failed".into()))?;
                Ok([&nonce[..], &ct[..]].concat())
            }
            EncryptionAlgorithm::ChaCha20Poly1305 => {
                use chacha20poly1305::{ChaCha20Poly1305, Key as CKey};
                let cipher = ChaCha20Poly1305::new(CKey::from_slice(&subkey[..]));
                let nonce = ChaCha20Poly1305::generate_nonce(&mut OsRng);
                let ct = cipher
                    .encrypt(&nonce, data)
                    .map_err(|_| MeshError::Crypto("ChaCha20-Poly1305 encrypt failed".into()))?;
                Ok([&nonce[..], &ct[..]].concat())
            }
        }
    }

    /// Decrypts a `nonce(12) || ciphertext` payload produced by
    /// [`KeyRing::encrypt`]. Wrong keys and tampering both surface as
    /// [`MeshError::Crypto`].
    pub fn decrypt(&self, data: &[u8], algorithm: EncryptionAlgorithm) -> Result<Vec<u8>> {
        if data.len() < NONCE_LEN {
            return Err(MeshError::Crypto(
                "ciphertext shorter than the nonce".into(),
            ));
        }
        let (nonce, ct) = data.split_at(NONCE_LEN);
        let subkey = self.subkey(DEFAULT_HKDF_INFO)?;
        match algorithm {
            EncryptionAlgorithm::Aes256Gcm => {
                let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&subkey[..]));
                cipher
                    .decrypt(Nonce::from_slice(nonce), ct)
                    .map_err(|_| MeshError::Crypto("AES-256-GCM decrypt failed".into()))
            }
            EncryptionAlgorithm::ChaCha20Poly1305 => {
                use chacha20poly1305::{ChaCha20Poly1305, Key as CKey, Nonce as CNonce};
                let cipher = ChaCha20Poly1305::new(CKey::from_slice(&subkey[..]));
                cipher
                    .decrypt(CNonce::from_slice(nonce), ct)
                    .map_err(|_| MeshError::Crypto("ChaCha20-Poly1305 decrypt failed".into()))
            }
        }
    }
}

impl std::fmt::Debug for KeyRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("KeyRing(<redacted>)")
    }
}

fn slot() -> &'static RwLock<Option<KeyRing>> {
    static KEYRING: OnceLock<RwLock<Option<KeyRing>>> = OnceLock::new();
    KEYRING.get_or_init(|| RwLock::new(None))
}

/// Initializes the process-wide keyring (`combsmesh_init` semantics).
/// `None` generates a random master key. Replaces any existing keyring.
pub fn init(master: Option<&[u8]>) -> Result<()> {
    let mut guard = slot()
        .write()
        .map_err(|_| MeshError::Crypto("keyring lock poisoned".into()))?;
    *guard = Some(KeyRing::new(master));
    Ok(())
}

/// Drops the process-wide keyring, zeroizing the master key
/// (`combsmesh_shutdown` semantics).
pub fn shutdown() {
    if let Ok(mut guard) = slot().write() {
        *guard = None;
    }
}

/// Returns a clone of the process-wide keyring, or
/// [`MeshError::NotInitialized`] when [`init`] was never called.
pub fn global() -> Result<KeyRing> {
    let guard = slot()
        .read()
        .map_err(|_| MeshError::Crypto("keyring lock poisoned".into()))?;
    guard.clone().ok_or(MeshError::NotInitialized)
}
