//! [`EmojiExporter`] — the single entry point for moving an [`Emoji`]
//! between the in-memory model and its two serializations: `.cmse` binary
//! (storage/FFI) and the Unicode envelope string (text channels).

use crate::binary;
use crate::codepoint;
use crate::crypto::KeyRing;
use crate::engine::Emoji;
use crate::error::Result;

/// Stateless serializer/deserializer for [`Emoji`].
pub struct EmojiExporter;

impl EmojiExporter {
    /// Serializes to the `.cmse` binary container (plaintext).
    pub fn to_binary(emoji: &Emoji) -> Result<Vec<u8>> {
        binary::write_emoji(emoji, None)
    }

    /// Serializes to `.cmse`, encrypting every block type named by the
    /// emoji's `Enc` block with `keyring`.
    pub fn to_binary_encrypted(emoji: &Emoji, keyring: &KeyRing) -> Result<Vec<u8>> {
        binary::write_emoji(emoji, Some(keyring))
    }

    /// Parses a `.cmse` container. Fails if any block is encrypted.
    pub fn from_binary(bytes: &[u8]) -> Result<Emoji> {
        Ok(Emoji::from_blocks(binary::read_blocks(bytes, None)?))
    }

    /// Parses a `.cmse` container, decrypting encrypted blocks with
    /// `keyring` (plaintext containers are accepted too).
    pub fn from_binary_decrypted(bytes: &[u8], keyring: &KeyRing) -> Result<Emoji> {
        Ok(Emoji::from_blocks(binary::read_blocks(
            bytes,
            Some(keyring),
        )?))
    }

    /// Encodes to the Unicode envelope string (plane 15/16 PUA + tag
    /// chars). Always plaintext — encrypt bytes first if a confidential
    /// text-channel transport is needed.
    pub fn to_unicode(emoji: &Emoji) -> Result<String> {
        codepoint::encode_blocks(&emoji.blocks)
    }

    /// Decodes every block envelope found in `s` (other text is skipped).
    pub fn from_unicode(s: &str) -> Result<Emoji> {
        Ok(Emoji::from_blocks(codepoint::decode_blocks(s)?))
    }
}
