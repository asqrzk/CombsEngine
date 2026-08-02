//! `.cmse` writer: serializes an [`Emoji`] to the v1 container, applying
//! per-block encryption when the emoji carries an `Enc` block and a
//! keyring is supplied.

use crate::binary::{
    BLOCK_FLAG_ENCRYPTED, DIR_ENTRY_LEN, FLAG_ENCRYPTED, HEADER_LEN, MAGIC, VERSION,
};
use crate::blocks::{Block, BlockTag, EncryptionBlock, validate_block};
use crate::crypto::KeyRing;
use crate::engine::Emoji;
use crate::error::Result;

/// Serializes `emoji` to a `.cmse` container.
///
/// With `keyring: Some(..)` and an `Enc` block present, every block type
/// listed in `apply_to` is encrypted at rest (payload becomes
/// `nonce(12) || ciphertext`, directory flag bit0 set, header flag bit0
/// set). The `Enc` block itself is always stored in plaintext. Without an
/// `Enc` block the keyring is ignored and the output is deterministic —
/// the registry relies on that for content addressing.
pub(crate) fn write_emoji(emoji: &Emoji, keyring: Option<&KeyRing>) -> Result<Vec<u8>> {
    let enc: Option<&EncryptionBlock> = emoji.blocks.iter().find_map(Block::as_encryption);

    let mut payloads: Vec<(BlockTag, u8, Vec<u8>)> = Vec::with_capacity(emoji.blocks.len());
    let mut any_encrypted = false;
    for block in &emoji.blocks {
        validate_block(block)?;
        let tag = block.tag();
        let mut payload = block.payload()?;
        let mut flags = 0u8;
        if let (Some(enc), Some(keyring)) = (enc, keyring) {
            if tag != BlockTag::Enc && enc.apply_to.contains(&tag) {
                payload = keyring.encrypt(&payload, enc.algorithm)?;
                flags |= BLOCK_FLAG_ENCRYPTED;
                any_encrypted = true;
            }
        }
        payloads.push((tag, flags, payload));
    }

    let block_count = payloads.len() as u32;
    let mut out = Vec::with_capacity(HEADER_LEN + payloads.len() * DIR_ENTRY_LEN);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    let flags = if any_encrypted { FLAG_ENCRYPTED } else { 0 };
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&block_count.to_le_bytes());

    // Directory first (offsets computed against the payload region start),
    // then the payloads themselves.
    let payload_start = HEADER_LEN + payloads.len() * DIR_ENTRY_LEN;
    let mut offset = payload_start as u32;
    for (tag, flags, payload) in &payloads {
        out.extend_from_slice(&tag.tag_bytes());
        out.push(*flags);
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&crc32fast::hash(payload).to_le_bytes());
        offset += payload.len() as u32;
    }
    for (_, _, payload) in &payloads {
        out.extend_from_slice(payload);
    }
    Ok(out)
}
