//! Request/response types for the `combsmesh_op_json` boundary.
//!
//! Mirrors combs-ffi: everything crossing the C ABI is JSON, one stable
//! symbol (`combsmesh_op_json`) for every present and future op.

use serde::{Deserialize, Serialize};

use combs_mesh::{Block, Emoji, RegistryEntry};

/// An [`Emoji`] as JSON (the struct itself is not serde; this is the wire
/// shape — `{name, blocks}`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmojiJson {
    /// Emoji name.
    pub name: String,
    /// Typed blocks (externally tagged by the `type` field).
    #[serde(default)]
    pub blocks: Vec<Block>,
}

impl From<Emoji> for EmojiJson {
    fn from(emoji: Emoji) -> Self {
        EmojiJson {
            name: emoji.name,
            blocks: emoji.blocks,
        }
    }
}

impl From<EmojiJson> for Emoji {
    fn from(json: EmojiJson) -> Self {
        Emoji {
            name: json.name,
            blocks: json.blocks,
        }
    }
}

/// An op request: `{"op": "...", ...}` — fields are op-specific and kept
/// flat for ergonomics (unknown fields ignored, missing fields default).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OpRequest {
    /// The operation name.
    pub op: String,
    /// `build` / `registry_register`: emoji name.
    pub name: Option<String>,
    /// `build`: description for the text block.
    pub description: Option<String>,
    /// `build`: initial blocks.
    pub blocks: Option<Vec<Block>>,
    /// `to_unicode`: the emoji.
    pub emoji: Option<EmojiJson>,
    /// `from_binary` / `registry_register` / `render`: base64 `.cmse`.
    pub binary_b64: Option<String>,
    /// `from_unicode`: the unicode envelope string.
    pub unicode: Option<String>,
    /// `registry_resolve`: name or sha256 hex.
    pub name_or_hash: Option<String>,
    /// `render`: frame index.
    pub frame: Option<u32>,
    /// `engine_load` (`engine` feature): model directory.
    pub model_dir: Option<String>,
    /// `engine_load`: KV cache capacity in tokens.
    pub max_seq_len: Option<usize>,
}

// ---- op responses (typed, serialized directly — see dispatch docs) ----

/// `build` response.
#[derive(Debug, Serialize)]
pub struct BuildResponse {
    /// The built emoji.
    pub emoji: EmojiJson,
    /// Base64 `.cmse` binary.
    pub binary_b64: String,
    /// Unicode envelope string.
    pub unicode: String,
}

/// `from_binary` / `from_unicode` response.
#[derive(Debug, Serialize)]
pub struct EmojiResponse {
    /// The emoji.
    pub emoji: EmojiJson,
}

/// `to_unicode` response.
#[derive(Debug, Serialize)]
pub struct UnicodeResponse {
    /// Unicode envelope string.
    pub unicode: String,
}

/// `registry_register` response.
#[derive(Debug, Serialize)]
pub struct HashResponse {
    /// SHA-256 hex of the binary.
    pub hash: String,
}

/// `registry_resolve` response.
#[derive(Debug, Serialize)]
pub struct ResolveResponse {
    /// The emoji.
    pub emoji: EmojiJson,
    /// Base64 `.cmse` binary.
    pub binary_b64: String,
}

/// `registry_list` response.
#[derive(Debug, Serialize)]
pub struct ListResponse {
    /// All entries.
    pub entries: Vec<RegistryEntryJson>,
}

/// `render` response.
#[derive(Debug, Serialize)]
pub struct RenderResponse {
    /// Base64 RGBA8 frame (`width * height * 4` bytes).
    pub rgba_b64: String,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
}

/// `engine_load` response.
#[cfg(feature = "engine")]
#[derive(Debug, Serialize)]
pub struct LoadedResponse {
    /// Always true (errors are reported via the error channel).
    pub loaded: bool,
}

/// One registry entry in a `registry_list` response.
#[derive(Debug, Clone, Serialize)]
pub struct RegistryEntryJson {
    /// Registered name.
    pub name: String,
    /// SHA-256 hex of the binary.
    pub hash: String,
    /// Path of the `.cmse` file.
    pub path: String,
    /// Size of the binary in bytes.
    pub bytes: usize,
}

impl From<RegistryEntry> for RegistryEntryJson {
    fn from(entry: RegistryEntry) -> Self {
        RegistryEntryJson {
            name: entry.name,
            hash: entry.hash,
            path: entry.path.display().to_string(),
            bytes: entry.bytes,
        }
    }
}
