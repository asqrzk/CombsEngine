//! Minimal protobuf wire-format reader (varint / fixed32 / length-
//! delimited). Hand-rolled — a schema this small doesn't need prost.
//! Shared by the SentencePiece converter (spm.rs) and the TFLite
//! LlmParameters parser (tflite.rs).

use crate::{FormatError, Result};

fn bad(what: impl Into<String>) -> FormatError {
    FormatError::Safetensors(format!("proto: {}", what.into()))
}

/// A cursor over a protobuf message's bytes.
pub struct Proto<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Proto<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Proto { buf, pos: 0 }
    }

    pub fn varint(&mut self) -> Result<u64> {
        let mut out: u64 = 0;
        let mut shift = 0;
        loop {
            let b = *self.buf.get(self.pos).ok_or_else(|| bad("truncated varint"))?;
            self.pos += 1;
            out |= ((b & 0x7F) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(out);
            }
            shift += 7;
            if shift >= 64 {
                return Err(bad("varint overflow"));
            }
        }
    }

    /// Next (field_number, wire_type), or None at EOF.
    pub fn tag(&mut self) -> Result<Option<(u32, u32)>> {
        if self.pos >= self.buf.len() {
            return Ok(None);
        }
        let v = self.varint()?;
        Ok(Some(((v >> 3) as u32, (v & 0x07) as u32)))
    }

    pub fn bytes(&mut self) -> Result<&'a [u8]> {
        let len = self.varint()? as usize;
        let end = self.pos + len;
        let out = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| bad("truncated bytes field"))?;
        self.pos = end;
        Ok(out)
    }

    pub fn string(&mut self) -> Result<String> {
        String::from_utf8(self.bytes()?.to_vec()).map_err(|e| bad(format!("bad utf8: {e}")))
    }

    pub fn f32(&mut self) -> Result<f32> {
        let b = self
            .buf
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| bad("truncated f32"))?;
        self.pos += 4;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn f64(&mut self) -> Result<f64> {
        let b = self
            .buf
            .get(self.pos..self.pos + 8)
            .ok_or_else(|| bad("truncated f64"))?;
        self.pos += 8;
        Ok(f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    pub fn skip(&mut self, wire: u32) -> Result<()> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => self.pos += 8,
            2 => {
                self.bytes()?;
            }
            5 => self.pos += 4,
            other => return Err(bad(format!("unsupported wire type {other}"))),
        }
        Ok(())
    }
}
