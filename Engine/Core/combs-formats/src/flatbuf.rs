//! Minimal flatbuffer navigation (schema-less): vtable lookup, uoffset
//! indirection, vectors, strings, scalars. Hand-rolled like the GGUF and
//! SentencePiece readers — the `flatbuffers` crate would only give us
//! these same primitives, and generated schema code would tie us to one
//! TFLite schema version.
//!
//! All reads are bounds-checked against the underlying byte slice and
//! return `FormatError` on malformed input (never panic on untrusted
//! bytes beyond a clear error).

use crate::{FormatError, Result};

/// A parsed flatbuffer document (borrows the backing bytes — mmap-friendly).
pub struct FlatBuffer<'a> {
    pub(crate) data: &'a [u8],
    root: usize,
}

fn bad(what: impl Into<String>) -> FormatError {
    FormatError::Safetensors(format!("flatbuf: {}", what.into()))
}

impl<'a> FlatBuffer<'a> {
    /// Opens a flatbuffer document. `file_identifier` (e.g. `b"TFL3"`) is
    /// checked when present at offset 4.
    pub fn new(data: &'a [u8], file_identifier: Option<&[u8; 4]>) -> Result<Self> {
        if data.len() < 8 {
            return Err(bad("file too small"));
        }
        if let Some(id) = file_identifier {
            if &data[4..8] != id {
                return Err(bad(format!(
                    "bad file identifier {:?} (expected {:?})",
                    &data[4..8],
                    id
                )));
            }
        }
        let root = u32_at(data, 0)? as usize;
        Ok(FlatBuffer { data, root })
    }

    /// Root table position.
    pub fn root(&self) -> usize {
        self.root
    }

    /// Absolute position of field `field_idx` in the table at `pos`
    /// (None when the field is absent — flatbuffers omit default values).
    /// Note: vtable offsets may be NEGATIVE — deduplicated vtables can
    /// live after the table that references them.
    pub fn table_pos(&self, pos: usize, field_idx: usize) -> Option<usize> {
        let vt_off = i32_at(self.data, pos).ok()?;
        let vt = (pos as i64 - vt_off as i64).try_into().ok()?;
        let vtsize = u16_at(self.data, vt).ok()? as usize;
        let idx = 4 + field_idx * 2;
        if idx + 2 > vtsize {
            return None;
        }
        let fo = u16_at(self.data, vt + idx).ok()? as usize;
        if fo == 0 {
            return None;
        }
        Some(pos + fo)
    }

    /// Follows a uoffset field to its target (string/vector/sub-table).
    pub fn uoffset(&self, pos: usize, field_idx: usize) -> Option<usize> {
        let p = self.table_pos(pos, field_idx)?;
        let rel = u32_at(self.data, p).ok()? as usize;
        Some(p + rel)
    }

    /// Reads a string field.
    pub fn string(&self, pos: usize, field_idx: usize) -> Result<Option<String>> {
        let Some(p) = self.uoffset(pos, field_idx) else {
            return Ok(None);
        };
        let len = u32_at(self.data, p)? as usize;
        let start = p + 4;
        let bytes = self
            .data
            .get(start..start + len)
            .ok_or_else(|| bad("string out of bounds"))?;
        Ok(Some(
            String::from_utf8(bytes.to_vec()).map_err(|e| bad(format!("utf8: {e}")))?,
        ))
    }

    /// A vector field as (absolute position of element 0, element count).
    pub fn vector(&self, pos: usize, field_idx: usize) -> Result<(Option<usize>, usize)> {
        let Some(p) = self.uoffset(pos, field_idx) else {
            return Ok((None, 0));
        };
        let len = u32_at(self.data, p)? as usize;
        Ok((Some(p + 4), len))
    }

    /// Absolute positions of the sub-tables in a vector-of-tables field.
    pub fn table_vector(&self, pos: usize, field_idx: usize) -> Result<Vec<usize>> {
        let (Some(p), len) = self.vector(pos, field_idx)? else {
            return Ok(Vec::new());
        };
        if len > 1_000_000 {
            return Err(bad(format!("absurd table vector length {len}")));
        }
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let rel = u32_at(self.data, p + i * 4)? as usize;
            out.push(p + i * 4 + rel);
        }
        Ok(out)
    }

    /// A scalar field (None when absent = schema default).
    pub fn scalar_u32(&self, pos: usize, field_idx: usize) -> Option<u32> {
        u32_at(self.data, self.table_pos(pos, field_idx)?).ok()
    }

    pub fn scalar_u64(&self, pos: usize, field_idx: usize) -> Option<u64> {
        u64_at(self.data, self.table_pos(pos, field_idx)?).ok()
    }

    pub fn scalar_u8(&self, pos: usize, field_idx: usize) -> Option<u8> {
        self.data.get(self.table_pos(pos, field_idx)?).copied()
    }

    pub fn scalar_i32(&self, pos: usize, field_idx: usize) -> Option<i32> {
        i32_at(self.data, self.table_pos(pos, field_idx)?).ok()
    }

    /// Reads `count` little-endian i32 from an absolute position.
    pub fn i32_slice(&self, abs_pos: usize, count: usize) -> Result<Vec<i32>> {
        let bytes = self
            .data
            .get(abs_pos..abs_pos + count * 4)
            .ok_or_else(|| bad("i32 vector out of bounds"))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Raw bytes of a buffer's data (inline vector), when present.
    pub fn byte_vector(&self, pos: usize, field_idx: usize) -> Result<Option<&'a [u8]>> {
        let (Some(p), len) = self.vector(pos, field_idx)? else {
            return Ok(None);
        };
        Ok(Some(
            self.data
                .get(p..p + len)
                .ok_or_else(|| bad("byte vector out of bounds"))?,
        ))
    }
}

fn u32_at(d: &[u8], pos: usize) -> Result<u32> {
    let b = d.get(pos..pos + 4).ok_or_else(|| bad("u32 out of bounds"))?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn i32_at(d: &[u8], pos: usize) -> Result<i32> {
    Ok(u32_at(d, pos)? as i32)
}

fn u16_at(d: &[u8], pos: usize) -> Result<u16> {
    let b = d.get(pos..pos + 2).ok_or_else(|| bad("u16 out of bounds"))?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

fn u64_at(d: &[u8], pos: usize) -> Result<u64> {
    let b = d.get(pos..pos + 8).ok_or_else(|| bad("u64 out of bounds"))?;
    Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
}
