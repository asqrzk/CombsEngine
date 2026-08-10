//! Strict, panic-free `.cmse` reader over a byte slice (no mmap — keeps the
//! crate wasm-friendly). Every bounds/parse failure is a [`MeshError`].

use crate::binary::{
    BLOCK_FLAG_ENCRYPTED, DIR_ENTRY_LEN, HEADER_LEN, MAGIC, VERSION,
};
use crate::blocks::{Block, BlockTag, EncryptionAlgorithm, EncryptionBlock};
use crate::crypto::KeyRing;
use crate::error::{MeshError, Result};

fn u16_le(bytes: &[u8], at: usize) -> Result<u16> {
    let s = bytes
        .get(at..at + 2)
        .ok_or_else(|| MeshError::Format("truncated header".into()))?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn u32_le(bytes: &[u8], at: usize) -> Result<u32> {
    let s = bytes
        .get(at..at + 4)
        .ok_or_else(|| MeshError::Format("truncated directory".into()))?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

struct Entry {
    tag: BlockTag,
    encrypted: bool,
    payload: Vec<u8>,
}

/// Parses a `.cmse` container into blocks. When any block is encrypted,
/// `keyring` must be `Some` and an `Enc` block (never itself encrypted)
/// must be present to name the algorithm.
pub(crate) fn read_blocks(bytes: &[u8], keyring: Option<&KeyRing>) -> Result<Vec<Block>> {
    if bytes.len() < HEADER_LEN {
        return Err(MeshError::Format("buffer smaller than header".into()));
    }
    if &bytes[0..4] != MAGIC {
        return Err(MeshError::Format("bad magic (expected CMSE)".into()));
    }
    let version = u16_le(bytes, 4)?;
    if version != VERSION {
        return Err(MeshError::UnsupportedVersion(version));
    }
    let block_count = u32_le(bytes, 8)? as usize;
    let dir_len = block_count
        .checked_mul(DIR_ENTRY_LEN)
        .and_then(|d| HEADER_LEN.checked_add(d))
        .ok_or_else(|| MeshError::Format("block count overflow".into()))?;
    if bytes.len() < dir_len {
        return Err(MeshError::Format("truncated directory".into()));
    }

    let mut entries = Vec::with_capacity(block_count);
    for i in 0..block_count {
        let base = HEADER_LEN + i * DIR_ENTRY_LEN;
        let tag: [u8; 3] = [bytes[base], bytes[base + 1], bytes[base + 2]];
        let tag = BlockTag::from_tag_bytes(&tag)
            .ok_or_else(|| MeshError::Format(format!("unknown block tag {tag:?}")))?;
        let flags = bytes[base + 3];
        let offset = u32_le(bytes, base + 4)? as usize;
        let len = u32_le(bytes, base + 8)? as usize;
        let crc = u32_le(bytes, base + 12)?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| MeshError::Format("payload range overflow".into()))?;
        if offset < dir_len || end > bytes.len() {
            return Err(MeshError::Format(format!(
                "payload of block {i} out of bounds (offset {offset}, len {len})"
            )));
        }
        let payload = bytes[offset..end].to_vec();
        if crc32fast::hash(&payload) != crc {
            return Err(MeshError::CrcMismatch);
        }
        entries.push(Entry {
            tag,
            encrypted: flags & BLOCK_FLAG_ENCRYPTED != 0,
            payload,
        });
    }
    // First pass: locate the (never-encrypted) Enc block for the algorithm.
    let mut algorithm: Option<EncryptionAlgorithm> = None;
    let mut blocks: Vec<(BlockTag, Vec<u8>)> = Vec::with_capacity(entries.len());
    for entry in &entries {
        if entry.tag == BlockTag::Enc {
            if entry.encrypted {
                return Err(MeshError::Format(
                    "the Enc block itself must not be encrypted".into(),
                ));
            }
            let enc: EncryptionBlock = serde_json::from_slice(&entry.payload)?;
            algorithm = Some(enc.algorithm);
        }
    }

    for entry in entries {
        let payload = if entry.encrypted {
            let algorithm = algorithm.ok_or_else(|| {
                MeshError::Crypto("encrypted blocks but no Enc block".into())
            })?;
            let keyring = keyring.ok_or_else(|| {
                MeshError::Crypto("encrypted blocks but no keyring supplied".into())
            })?;
            keyring.decrypt(&entry.payload, algorithm)?
        } else {
            entry.payload
        };
        blocks.push((entry.tag, payload));
    }

    blocks
        .into_iter()
        .map(|(tag, payload)| Block::from_payload(tag, &payload))
        .collect()
}
