//! Browser bindings (feature `wasm`): thin `#[wasm_bindgen]` wrappers over
//! the existing APIs, for a downstream cdylib (the way combs-wasm wraps
//! the engine). Input JSON shapes match `combsmesh_op_json` ops.
//!
//! Key handling: `mesh_encrypt`/`mesh_decrypt` use the process-wide
//! keyring; call `mesh_init` first (no key = random master, generated via
//! WebCrypto through getrandom's `js` shim). There is intentionally no
//! `mesh_shutdown` export — the keyring outlives any single call and the
//! process key can be replaced by calling `mesh_init` again.

use wasm_bindgen::JsError;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::crypto::{self, KeyRing};
use crate::blocks::EncryptionAlgorithm;
use crate::{Emoji, EmojiBuilder, EmojiExporter};

fn mesh_err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

/// Crate version string.
#[wasm_bindgen]
pub fn mesh_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Initializes the process keyring. `key == null/undefined` (or an empty
/// array) generates a random master key via WebCrypto.
#[wasm_bindgen]
pub fn mesh_init(key: Option<Vec<u8>>) -> Result<(), JsError> {
    let key = key.filter(|k| !k.is_empty());
    crypto::init(key.as_deref()).map_err(mesh_err)
}

/// `{"name", "description?", "blocks?"}` (same shape as the
/// `combsmesh_op_json` "build" op) → `.cmse` binary.
#[wasm_bindgen]
pub fn emoji_build(json: &str) -> Result<Vec<u8>, JsError> {
    #[derive(serde::Deserialize)]
    struct BuildRequest {
        name: String,
        description: Option<String>,
        blocks: Option<Vec<crate::Block>>,
    }
    let request: BuildRequest = serde_json::from_str(json).map_err(mesh_err)?;
    let mut builder = EmojiBuilder::new(&request.name);
    if let Some(description) = request.description {
        builder = builder.description(&description);
    }
    if let Some(blocks) = request.blocks {
        for block in blocks {
            builder = builder.add_block(block);
        }
    }
    EmojiExporter::to_binary(&builder.build()).map_err(mesh_err)
}

/// Emoji JSON (`{"name", "blocks"}`) → unicode envelope string.
#[wasm_bindgen]
pub fn emoji_to_unicode(json: &str) -> Result<String, JsError> {
    #[derive(serde::Deserialize)]
    struct EmojiJson {
        name: String,
        #[serde(default)]
        blocks: Vec<crate::Block>,
    }
    let json: EmojiJson = serde_json::from_str(json).map_err(mesh_err)?;
    let emoji = Emoji {
        name: json.name,
        blocks: json.blocks,
    };
    EmojiExporter::to_unicode(&emoji).map_err(mesh_err)
}

/// `.cmse` binary → emoji JSON string. Encrypted containers are accepted
/// when the process keyring holds the right key.
#[wasm_bindgen]
pub fn emoji_from_binary(bytes: &[u8]) -> Result<String, JsError> {
    let emoji = match crypto::global() {
        Ok(keyring) => EmojiExporter::from_binary_decrypted(bytes, &keyring),
        Err(_) => EmojiExporter::from_binary(bytes),
    }
    .map_err(mesh_err)?;
    serde_json::to_string(&emoji_json(&emoji)).map_err(mesh_err)
}

/// Unicode envelope string → emoji JSON string.
#[wasm_bindgen]
pub fn emoji_from_unicode(s: &str) -> Result<String, JsError> {
    let emoji = EmojiExporter::from_unicode(s).map_err(mesh_err)?;
    serde_json::to_string(&emoji_json(&emoji)).map_err(mesh_err)
}

/// Encrypts with the process keyring (AES-256-GCM, nonce-prefixed).
/// Requires `mesh_init` first.
#[wasm_bindgen]
pub fn mesh_encrypt(bytes: &[u8]) -> Result<Vec<u8>, JsError> {
    crypto::global()
        .and_then(|k: KeyRing| k.encrypt(bytes, EncryptionAlgorithm::Aes256Gcm))
        .map_err(mesh_err)
}

/// Decrypts a blob produced by `mesh_encrypt`.
#[wasm_bindgen]
pub fn mesh_decrypt(bytes: &[u8]) -> Result<Vec<u8>, JsError> {
    crypto::global()
        .and_then(|k: KeyRing| k.decrypt(bytes, EncryptionAlgorithm::Aes256Gcm))
        .map_err(mesh_err)
}

/// `{"name", "blocks"}` wire shape (Emoji itself is not serde).
fn emoji_json(emoji: &Emoji) -> serde_json::Value {
    serde_json::json!({"name": emoji.name, "blocks": emoji.blocks})
}
