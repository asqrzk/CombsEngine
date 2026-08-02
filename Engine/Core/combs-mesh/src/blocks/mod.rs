//! The 10 typed block payloads that make up an [`crate::Emoji`].
//!
//! Each block type is a plain serde struct (one module per type). On disk /
//! on the wire a block is identified by its [`BlockTag`] — a 3-byte ASCII
//! tag in the binary directory, and a tag char + plane-15 sub-range in the
//! Unicode encoding — so payloads stay self-describing without embedding
//! type names in the JSON.

mod api;
mod character;
mod emotion;
mod encryption;
mod function;
mod image;
mod lifecycle;
mod orchestration;
mod text;
mod todo;

pub use api::{ApiBlock, ApiEndpoint};
pub use character::CharacterBlock;
pub use emotion::{EmotionBlock, EmotionState};
pub use encryption::{EncryptionAlgorithm, EncryptionBlock};
pub use function::{FunctionBlock, FunctionDef, FunctionKind};
pub use image::{ImageBlock, SpriteAtlas};
pub use lifecycle::{LifecycleBlock, LifecycleState, LifecycleTransition};
pub use orchestration::{DirectiveKind, OrchestrationBlock, OrchestrationDirective};
pub use text::TextBlock;
pub use todo::{TodoBlock, TodoItem, TodoStatus};

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Identifies a block type. The index (0..10) doubles as the plane-15
/// sub-range selector in the Unicode encoding, so the order is part of the
/// wire format and must never change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BlockTag {
    /// Free text / spec sheet.
    Txt,
    /// Sprite atlas (RGBA8 pixels).
    Img,
    /// Task list with statuses + dependencies.
    Tdo,
    /// Callable function definitions.
    Fnc,
    /// HTTP endpoint descriptions.
    Api,
    /// Agent lifecycle state machine.
    Lfc,
    /// Character traits + backstory.
    Chr,
    /// Emotion states with intensities.
    Emo,
    /// Encryption-at-rest directive.
    Enc,
    /// Orchestration directives for runtimes.
    Orc,
}

impl BlockTag {
    /// All tags, in wire order.
    pub const ALL: [BlockTag; 10] = [
        BlockTag::Txt,
        BlockTag::Img,
        BlockTag::Tdo,
        BlockTag::Fnc,
        BlockTag::Api,
        BlockTag::Lfc,
        BlockTag::Chr,
        BlockTag::Emo,
        BlockTag::Enc,
        BlockTag::Orc,
    ];

    /// The 3-byte ASCII tag used in the binary directory.
    #[must_use]
    pub fn tag_bytes(self) -> [u8; 3] {
        match self {
            BlockTag::Txt => *b"txt",
            BlockTag::Img => *b"img",
            BlockTag::Tdo => *b"tdo",
            BlockTag::Fnc => *b"fnc",
            BlockTag::Api => *b"api",
            BlockTag::Lfc => *b"lfc",
            BlockTag::Chr => *b"chr",
            BlockTag::Emo => *b"emo",
            BlockTag::Enc => *b"enc",
            BlockTag::Orc => *b"orc",
        }
    }

    /// Wire index (0..10); also the plane-15 sub-range index.
    #[must_use]
    pub fn index(self) -> u8 {
        match self {
            BlockTag::Txt => 0,
            BlockTag::Img => 1,
            BlockTag::Tdo => 2,
            BlockTag::Fnc => 3,
            BlockTag::Api => 4,
            BlockTag::Lfc => 5,
            BlockTag::Chr => 6,
            BlockTag::Emo => 7,
            BlockTag::Enc => 8,
            BlockTag::Orc => 9,
        }
    }

    /// Inverse of [`BlockTag::index`].
    #[must_use]
    pub fn from_index(index: u8) -> Option<Self> {
        BlockTag::ALL.iter().copied().find(|t| t.index() == index)
    }

    /// Inverse of [`BlockTag::tag_bytes`].
    #[must_use]
    pub fn from_tag_bytes(tag: &[u8; 3]) -> Option<Self> {
        BlockTag::ALL.iter().copied().find(|t| &t.tag_bytes() == tag)
    }
}

/// A typed block payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Block {
    /// See [`TextBlock`].
    Txt(TextBlock),
    /// See [`ImageBlock`].
    Img(ImageBlock),
    /// See [`TodoBlock`].
    Tdo(TodoBlock),
    /// See [`FunctionBlock`].
    Fnc(FunctionBlock),
    /// See [`ApiBlock`].
    Api(ApiBlock),
    /// See [`LifecycleBlock`].
    Lfc(LifecycleBlock),
    /// See [`CharacterBlock`].
    Chr(CharacterBlock),
    /// See [`EmotionBlock`].
    Emo(EmotionBlock),
    /// See [`EncryptionBlock`].
    Enc(EncryptionBlock),
    /// See [`OrchestrationBlock`].
    Orc(OrchestrationBlock),
}

impl Block {
    /// The tag identifying this block's type.
    #[must_use]
    pub fn tag(&self) -> BlockTag {
        match self {
            Block::Txt(_) => BlockTag::Txt,
            Block::Img(_) => BlockTag::Img,
            Block::Tdo(_) => BlockTag::Tdo,
            Block::Fnc(_) => BlockTag::Fnc,
            Block::Api(_) => BlockTag::Api,
            Block::Lfc(_) => BlockTag::Lfc,
            Block::Chr(_) => BlockTag::Chr,
            Block::Emo(_) => BlockTag::Emo,
            Block::Enc(_) => BlockTag::Enc,
            Block::Orc(_) => BlockTag::Orc,
        }
    }

    /// Serializes the *inner* block struct to JSON bytes (the type is
    /// carried by the container, not embedded in the payload).
    pub fn payload(&self) -> Result<Vec<u8>> {
        let json = match self {
            Block::Txt(b) => serde_json::to_vec(b),
            Block::Img(b) => serde_json::to_vec(b),
            Block::Tdo(b) => serde_json::to_vec(b),
            Block::Fnc(b) => serde_json::to_vec(b),
            Block::Api(b) => serde_json::to_vec(b),
            Block::Lfc(b) => serde_json::to_vec(b),
            Block::Chr(b) => serde_json::to_vec(b),
            Block::Emo(b) => serde_json::to_vec(b),
            Block::Enc(b) => serde_json::to_vec(b),
            Block::Orc(b) => serde_json::to_vec(b),
        }?;
        Ok(json)
    }

    /// Inverse of [`Block::payload`]: parses a JSON payload for `tag`.
    pub fn from_payload(tag: BlockTag, bytes: &[u8]) -> Result<Block> {
        let block = match tag {
            BlockTag::Txt => Block::Txt(serde_json::from_slice(bytes)?),
            BlockTag::Img => Block::Img(serde_json::from_slice(bytes)?),
            BlockTag::Tdo => Block::Tdo(serde_json::from_slice(bytes)?),
            BlockTag::Fnc => Block::Fnc(serde_json::from_slice(bytes)?),
            BlockTag::Api => Block::Api(serde_json::from_slice(bytes)?),
            BlockTag::Lfc => Block::Lfc(serde_json::from_slice(bytes)?),
            BlockTag::Chr => Block::Chr(serde_json::from_slice(bytes)?),
            BlockTag::Emo => Block::Emo(serde_json::from_slice(bytes)?),
            BlockTag::Enc => Block::Enc(serde_json::from_slice(bytes)?),
            BlockTag::Orc => Block::Orc(serde_json::from_slice(bytes)?),
        };
        Ok(block)
    }

    /// If this is an `Enc` block, returns the encryption directive.
    #[must_use]
    pub fn as_encryption(&self) -> Option<&EncryptionBlock> {
        match self {
            Block::Enc(b) => Some(b),
            _ => None,
        }
    }
}

/// Validates a block's internal consistency (sizes, ranges).
pub(crate) fn validate_block(block: &Block) -> Result<()> {
    if let Block::Img(img) = block {
        img.atlas.validate()?;
    }
    Ok(())
}
