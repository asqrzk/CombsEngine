//! The `.cmse` binary container, format v1.
//!
//! Layout (all integers little-endian):
//!
//! ```text
//! offset  size  field
//! 0       4     magic b"CMSE"
//! 4       2     u16 version (= 1)
//! 6       2     u16 flags (bit0 = at least one block is encrypted)
//! 8       4     u32 block_count
//! 12      16×N  directory entries:
//!               [u8;3] tag | u8 flags (bit0 = this block encrypted) |
//!               u32 offset | u32 len | u32 crc32 (IEEE, of stored payload)
//! 12+16N  ..    payload bytes (serde_json of the block struct; encrypted
//!               blocks store nonce(12) || ciphertext, CRC'd as stored)
//! ```
//!
//! Offsets are absolute from byte 0. Payloads are versioned via the
//! container version — no per-block versioning in v1.

mod reader;
mod writer;

pub(crate) use reader::read_blocks;
pub(crate) use writer::write_emoji;

/// Container magic.
pub const MAGIC: &[u8; 4] = b"CMSE";
/// Container version this build reads/writes.
pub const VERSION: u16 = 1;
/// Header flag: at least one block payload is encrypted.
pub const FLAG_ENCRYPTED: u16 = 0x1;
/// Directory-entry flag: this block's payload is encrypted.
pub const BLOCK_FLAG_ENCRYPTED: u8 = 0x1;
/// Header size in bytes.
pub const HEADER_LEN: usize = 12;
/// Directory entry size in bytes (3 tag + 1 flags + 4 offset + 4 len + 4 crc).
pub const DIR_ENTRY_LEN: usize = 16;
