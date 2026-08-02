//! The [`CombsEngineCore`] trait — the contract the C ABI crate
//! (`combs-mesh-ffi`) binds to — plus [`DefaultEngine`], the standalone
//! implementation that ships with this crate.
//!
//! Owned by `combs-mesh` (not the ffi crate) so tests and alternative
//! transports can implement/consume it without linking the cdylib.
//! `DefaultEngine` covers everything except `infer`, which requires the
//! optional `engine` feature (combs-runtime); that adapter lands in
//! combs-mesh-ffi.

use std::sync::Mutex;

use crate::blocks::EncryptionAlgorithm;
use crate::crypto::KeyRing;
use crate::engine::Emoji;
use crate::error::MeshError;
use crate::render::{CpuRenderer, Renderer};

/// Errors from a [`CombsEngineCore`] implementation.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// The engine was used before `init` (or after `shutdown`).
    #[error("engine not initialized")]
    NotInitialized,
    /// Encryption/decryption failed.
    #[error("crypto error: {0}")]
    Crypto(String),
    /// Rendering failed.
    #[error("render error: {0}")]
    Render(String),
    /// The operation is not supported by this engine (e.g. `infer` without
    /// the `engine` feature).
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// An underlying mesh error.
    #[error(transparent)]
    Mesh(#[from] MeshError),
}

/// The engine contract surfaced over the C ABI.
pub trait CombsEngineCore: Send + Sync {
    /// Initializes the engine with a master key (HKDF input).
    fn init(&self, key: &[u8]) -> Result<(), EngineError>;

    /// Runs inference on `prompt`. Requires the `engine` feature.
    fn infer(&self, prompt: &str) -> Result<String, EngineError>;

    /// Encrypts a memory blob (nonce-prefixed AEAD).
    fn encrypt_memory(&self, data: &[u8]) -> Result<Vec<u8>, EngineError>;

    /// Decrypts a blob produced by [`CombsEngineCore::encrypt_memory`].
    fn decrypt_memory(&self, data: &[u8]) -> Result<Vec<u8>, EngineError>;

    /// Renders frame `frame_index` of the emoji's first sprite atlas to
    /// RGBA8 bytes.
    fn render_sprite(&self, emoji: &Emoji, frame_index: u32) -> Result<Vec<u8>, EngineError>;

    /// Shuts the engine down, zeroizing key material.
    fn shutdown(&self) -> Result<(), EngineError>;
}

/// Standalone engine: crypto via [`KeyRing`], sprites via [`CpuRenderer`].
/// `infer` returns [`EngineError::Unsupported`] — real inference needs the
/// `engine` feature (combs-runtime), wired up in combs-mesh-ffi.
pub struct DefaultEngine {
    keyring: Mutex<Option<KeyRing>>,
    renderer: CpuRenderer,
}

impl DefaultEngine {
    /// Creates an uninitialized engine (call `init` before crypto ops).
    #[must_use]
    pub fn new() -> Self {
        DefaultEngine {
            keyring: Mutex::new(None),
            renderer: CpuRenderer::new(),
        }
    }
}

impl Default for DefaultEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl CombsEngineCore for DefaultEngine {
    fn init(&self, key: &[u8]) -> Result<(), EngineError> {
        let mut guard = self
            .keyring
            .lock()
            .map_err(|_| EngineError::Crypto("keyring lock poisoned".into()))?;
        *guard = Some(KeyRing::new(Some(key)));
        Ok(())
    }

    fn infer(&self, _prompt: &str) -> Result<String, EngineError> {
        Err(EngineError::Unsupported(
            "inference requires the `engine` feature".into(),
        ))
    }

    fn encrypt_memory(&self, data: &[u8]) -> Result<Vec<u8>, EngineError> {
        let guard = self
            .keyring
            .lock()
            .map_err(|_| EngineError::Crypto("keyring lock poisoned".into()))?;
        let keyring = guard.as_ref().ok_or(EngineError::NotInitialized)?;
        Ok(keyring.encrypt(data, EncryptionAlgorithm::Aes256Gcm)?)
    }

    fn decrypt_memory(&self, data: &[u8]) -> Result<Vec<u8>, EngineError> {
        let guard = self
            .keyring
            .lock()
            .map_err(|_| EngineError::Crypto("keyring lock poisoned".into()))?;
        let keyring = guard.as_ref().ok_or(EngineError::NotInitialized)?;
        Ok(keyring.decrypt(data, EncryptionAlgorithm::Aes256Gcm)?)
    }

    fn render_sprite(&self, emoji: &Emoji, frame_index: u32) -> Result<Vec<u8>, EngineError> {
        let image = emoji.get_image().ok_or(EngineError::Render(
            "emoji has no image block".into(),
        ))?;
        Ok(self.renderer.render_frame(&image.atlas, frame_index)?)
    }

    fn shutdown(&self) -> Result<(), EngineError> {
        let mut guard = self
            .keyring
            .lock()
            .map_err(|_| EngineError::Crypto("keyring lock poisoned".into()))?;
        *guard = None;
        Ok(())
    }
}
