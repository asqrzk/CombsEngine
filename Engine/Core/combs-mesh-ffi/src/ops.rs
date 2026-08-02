//! `combsmesh_op_json` op handlers. Everything returns JSON strings;
//! errors are plain `String`s (the C layer drops them into the TLS slot).

use base64::{Engine as _, engine::general_purpose::STANDARD as B64};

use combs_mesh::{CpuRenderer, Emoji, EmojiBuilder, EmojiExporter, Registry, Renderer, crypto};

use crate::types::{
    BuildResponse, EmojiJson, EmojiResponse, HashResponse, ListResponse, OpRequest,
    RegistryEntryJson, RenderResponse, ResolveResponse, UnicodeResponse,
};
#[cfg(feature = "engine")]
use crate::types::LoadedResponse;

/// Dispatches one op request to its handler.
///
/// Handlers serialize TYPED response structs directly (never through
/// `serde_json::Value`, whose BTreeMap would alphabetize keys): the emoji
/// JSON the ops return must keep struct field order so clients can strip
/// the `type` tag and re-serialize payloads byte-identically to the
/// envelope encoding (the TS unicode codec's parity contract).
pub fn dispatch(json: &str) -> Result<String, String> {
    let request: OpRequest =
        serde_json::from_str(json).map_err(|e| format!("invalid request JSON: {e}"))?;
    match request.op.as_str() {
        "build" => build(&request),
        "from_binary" => from_binary(&request),
        "to_unicode" => to_unicode(&request),
        "from_unicode" => from_unicode(&request),
        "registry_register" => registry_register(&request),
        "registry_resolve" => registry_resolve(&request),
        "registry_list" => registry_list(),
        "render" => render(&request),
        "engine_load" => engine_load(&request),
        other => Err(format!("unknown op: {other}")),
    }
}

fn to_json<T: serde::Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_string(value).map_err(|e| e.to_string())
}

/// Decodes a base64 `.cmse` into an emoji, using the process keyring when
/// one is initialized (plaintext containers are accepted either way).
pub fn decode_emoji(bytes: &[u8]) -> Result<Emoji, String> {
    match crypto::global() {
        Ok(keyring) => EmojiExporter::from_binary_decrypted(bytes, &keyring),
        Err(_) => EmojiExporter::from_binary(bytes),
    }
    .map_err(|e| e.to_string())
}

fn require<'a, T>(value: Option<&'a T>, what: &str) -> Result<&'a T, String> {
    value.ok_or_else(|| format!("missing `{what}`"))
}

fn b64_to_emoji(b64: Option<&String>) -> Result<Emoji, String> {
    let b64 = require(b64, "binary_b64")?;
    let bytes = B64
        .decode(b64)
        .map_err(|e| format!("invalid binary_b64: {e}"))?;
    decode_emoji(&bytes)
}

fn build(request: &OpRequest) -> Result<String, String> {
    let name = require(request.name.as_ref(), "name")?;
    let mut builder = EmojiBuilder::new(name);
    if let Some(description) = &request.description {
        builder = builder.description(description);
    }
    if let Some(blocks) = &request.blocks {
        for block in blocks {
            builder = builder.add_block(block.clone());
        }
    }
    let emoji = builder.build();
    let binary = EmojiExporter::to_binary(&emoji).map_err(|e| e.to_string())?;
    let unicode = EmojiExporter::to_unicode(&emoji).map_err(|e| e.to_string())?;
    to_json(&BuildResponse {
        emoji: EmojiJson::from(emoji),
        binary_b64: B64.encode(binary),
        unicode,
    })
}

fn from_binary(request: &OpRequest) -> Result<String, String> {
    let emoji = b64_to_emoji(request.binary_b64.as_ref())?;
    to_json(&EmojiResponse {
        emoji: EmojiJson::from(emoji),
    })
}

fn to_unicode(request: &OpRequest) -> Result<String, String> {
    let emoji: Emoji = require(request.emoji.as_ref(), "emoji")?.clone().into();
    let unicode = EmojiExporter::to_unicode(&emoji).map_err(|e| e.to_string())?;
    to_json(&UnicodeResponse { unicode })
}

fn from_unicode(request: &OpRequest) -> Result<String, String> {
    let unicode = require(request.unicode.as_ref(), "unicode")?;
    let emoji = EmojiExporter::from_unicode(unicode).map_err(|e| e.to_string())?;
    to_json(&EmojiResponse {
        emoji: EmojiJson::from(emoji),
    })
}

fn registry_register(request: &OpRequest) -> Result<String, String> {
    let mut emoji = b64_to_emoji(request.binary_b64.as_ref())?;
    if let Some(name) = &request.name {
        emoji.name = name.clone();
    }
    let hash = Registry::open()
        .and_then(|r| r.register(&emoji))
        .map_err(|e| e.to_string())?;
    to_json(&HashResponse { hash })
}

fn registry_resolve(request: &OpRequest) -> Result<String, String> {
    let name_or_hash = require(request.name_or_hash.as_ref(), "name_or_hash")?;
    let emoji = Registry::open()
        .and_then(|r| r.resolve(name_or_hash))
        .map_err(|e| e.to_string())?;
    let binary = EmojiExporter::to_binary(&emoji).map_err(|e| e.to_string())?;
    to_json(&ResolveResponse {
        emoji: EmojiJson::from(emoji),
        binary_b64: B64.encode(binary),
    })
}

fn registry_list() -> Result<String, String> {
    let entries = Registry::open()
        .and_then(|r| r.list())
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(RegistryEntryJson::from)
        .collect::<Vec<_>>();
    to_json(&ListResponse { entries })
}

fn render(request: &OpRequest) -> Result<String, String> {
    let emoji = b64_to_emoji(request.binary_b64.as_ref())?;
    let frame = request.frame.unwrap_or(0);
    let image = emoji
        .get_image()
        .ok_or_else(|| "emoji has no image block".to_string())?;
    let rgba = CpuRenderer::new()
        .render_frame(&image.atlas, frame)
        .map_err(|e| e.to_string())?;
    to_json(&RenderResponse {
        rgba_b64: B64.encode(rgba),
        width: image.atlas.frame_width,
        height: image.atlas.frame_height,
    })
}

#[cfg(feature = "engine")]
fn engine_load(request: &OpRequest) -> Result<String, String> {
    let model_dir = require(request.model_dir.as_ref(), "model_dir")?;
    crate::runtime_engine::RuntimeEngine::global()
        .load_model(model_dir, request.max_seq_len)
        .map_err(|e| e.to_string())?;
    to_json(&LoadedResponse { loaded: true })
}

#[cfg(not(feature = "engine"))]
fn engine_load(request: &OpRequest) -> Result<String, String> {
    // Fields are read only by the engine-feature handler.
    let _ = (&request.model_dir, &request.max_seq_len);
    Err("op `engine_load` requires the `engine` feature".into())
}
