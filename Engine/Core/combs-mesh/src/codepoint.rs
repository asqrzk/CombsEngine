//! Unicode transport encoding — embeds an emoji's blocks in a plain
//! `String` that survives text-only channels (chat, MCP tool results).
//!
//! ## Scheme (self-describing envelope, one per block)
//!
//! ```text
//! [TAG char]  [2 × plane-15 length chars]  [ceil(len/2) × plane-16 data chars]
//! ```
//!
//! - **Tag char**: `U+E0061 + block-type index` (TAG glyphs U+E0061..U+E006A
//!   for the 10 block types). Marks the start of a block and names its type.
//! - **Length**: payload length (bytes) as two 12-bit chunks, big-endian,
//!   each stored in plane 15 (U+F0000..U+FFFFF, the Supplementary Private
//!   Use Area-A): `U+F0000 + block-type index × 4096 + chunk`. Plane 15 is
//!   split into 16 sub-ranges of 4096; the block type occupies sub-range
//!   `index`, so length chars are type-checked against the tag char.
//! - **Data**: the JSON payload bytes, two bytes (big-endian u16) per
//!   codepoint in plane 16 (U+100000..U+10FFFF, Supplementary PUA-B):
//!   `U+100000 + u16`. Odd payloads are zero-padded; the length field
//!   trims the pad.
//!
//! Max payload is `2^24 - 1` bytes (24-bit length). All codepoints used
//! are valid Unicode scalar values (no surrogates), so the output is a
//! well-formed UTF-8/UTF-16 string on any platform. Decoders skip any
//! non-marker text, so envelopes can ride inside ordinary prose.

use crate::blocks::{Block, BlockTag};
use crate::error::{MeshError, Result};

/// Base of the TAG block-type marker chars (U+E0061 + index).
pub const TAG_CHAR_BASE: u32 = 0xE0061;
/// Base of plane 15 (Supplementary PUA-A).
pub const PLANE15_BASE: u32 = 0xF0000;
/// Base of plane 16 (Supplementary PUA-B).
pub const PLANE16_BASE: u32 = 0x100000;
/// Plane-15 sub-range size (16 sub-ranges of 4096 cover the plane).
pub const SUBRANGE_SIZE: u32 = 4096;
/// Maximum payload size (24-bit length field).
pub const MAX_PAYLOAD: usize = 0xFF_FFFF;

/// The tag char marking the start of a block of type `tag`.
#[must_use]
pub fn tag_char(tag: BlockTag) -> char {
    // U+E0061 + index (0..10) is always a valid scalar value.
    char::from_u32(TAG_CHAR_BASE + tag.index() as u32).unwrap_or('\u{E0061}')
}

/// Inverse of [`tag_char`]; `None` for any other char.
#[must_use]
pub fn tag_from_char(c: char) -> Option<BlockTag> {
    let v = c as u32;
    let idx = v.checked_sub(TAG_CHAR_BASE)?;
    BlockTag::from_index(u8::try_from(idx).ok()?)
}

fn plane15_char(tag: BlockTag, chunk: u16) -> Result<char> {
    if chunk >= SUBRANGE_SIZE as u16 {
        return Err(MeshError::Unicode(format!("chunk {chunk} out of range")));
    }
    let v = PLANE15_BASE + tag.index() as u32 * SUBRANGE_SIZE + chunk as u32;
    char::from_u32(v).ok_or_else(|| MeshError::Unicode(format!("invalid char U+{v:X}")))
}

fn plane15_value(c: char, tag: BlockTag) -> Result<u16> {
    let v = c as u32;
    let base = PLANE15_BASE + tag.index() as u32 * SUBRANGE_SIZE;
    if !(base..base + SUBRANGE_SIZE).contains(&v) {
        return Err(MeshError::Unicode(format!(
            "expected plane-15 length char for {:?}, got U+{:X}",
            tag, v
        )));
    }
    Ok((v - base) as u16)
}

fn plane16_char(value: u16) -> Result<char> {
    let v = PLANE16_BASE + value as u32;
    char::from_u32(v).ok_or_else(|| MeshError::Unicode(format!("invalid char U+{v:X}")))
}

fn plane16_value(c: char) -> Result<u16> {
    let v = c as u32;
    if !(PLANE16_BASE..=PLANE16_BASE + 0xFFFF).contains(&v) {
        return Err(MeshError::Unicode(format!(
            "expected plane-16 data char, got U+{v:X}"
        )));
    }
    Ok((v - PLANE16_BASE) as u16)
}

/// Encodes blocks to the Unicode envelope string (concatenated per block).
pub fn encode_blocks(blocks: &[Block]) -> Result<String> {
    let mut out = String::new();
    for block in blocks {
        let tag = block.tag();
        let payload = block.payload()?;
        if payload.len() > MAX_PAYLOAD {
            return Err(MeshError::Unicode(format!(
                "payload of {} bytes exceeds the 24-bit limit",
                payload.len()
            )));
        }
        out.push(tag_char(tag));
        let len = payload.len() as u32;
        out.push(plane15_char(tag, ((len >> 12) & 0xFFF) as u16)?);
        out.push(plane15_char(tag, (len & 0xFFF) as u16)?);
        let mut chunks = payload.chunks_exact(2);
        for pair in &mut chunks {
            out.push(plane16_char(u16::from_be_bytes([pair[0], pair[1]]))?);
        }
        let rem = chunks.remainder();
        if let [last] = rem {
            out.push(plane16_char(u16::from_be_bytes([*last, 0]))?);
        }
    }
    Ok(out)
}

/// Decodes all block envelopes found in `s`. Non-marker chars are skipped,
/// so the emoji may be embedded in arbitrary text. Any *started* envelope
/// that is malformed (bad length chars, truncated data, bad JSON) is an
/// error — never a panic.
pub fn decode_blocks(s: &str) -> Result<Vec<Block>> {
    let mut blocks = Vec::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        let Some(tag) = tag_from_char(c) else { continue };
        let hi = next_char(&mut chars).and_then(|c| plane15_value(c, tag))?;
        let lo = next_char(&mut chars).and_then(|c| plane15_value(c, tag))?;
        let len = ((hi as usize) << 12) | lo as usize;
        let count = len.div_ceil(2);
        let mut payload = Vec::with_capacity(count * 2);
        for _ in 0..count {
            let value = next_char(&mut chars).and_then(plane16_value)?;
            payload.extend_from_slice(&value.to_be_bytes());
        }
        payload.truncate(len);
        blocks.push(Block::from_payload(tag, &payload)?);
    }
    Ok(blocks)
}

fn next_char(chars: &mut std::str::Chars<'_>) -> Result<char> {
    chars
        .next()
        .ok_or_else(|| MeshError::Unicode("truncated envelope".into()))
}
