//! `.litertlm` container reader (LiteRT-LM archive format).
//!
//! Layout (recon + official `litertlm_header_schema.fbs` /
//! `litertlm_read.cc`):
//!
//! ```text
//! 0x00  "LITERTLM"                magic
//! 0x08  u32 major / minor / patch version
//! 0x14  4 bytes padding
//! 0x18  u64 header_end_offset
//! 0x20  LiteRTLMMetaData flatbuffer (system metadata + section directory)
//! …     section payloads, each aligned to BLOCK_SIZE = 16 KiB
//! ```
//!
//! Sections (`AnySectionDataType`): TFLiteModel (3) → handed to the
//! TFLite block at its absolute offset; SP_Tokenizer (4) → the
//! SentencePiece block; HF_Tokenizer_Zlib (6) → zlib-inflated HF
//! tokenizer.json. Everything else (LlmMetadataProto, executor metadata,
//! generic blobs) is ignored for now — config comes from the TFLite
//! section's own LlmParameters.

use std::path::Path;

use crate::flatbuf::FlatBuffer;
use crate::source::ModelSource;
use crate::tflite::TfliteSource;
use crate::{FormatError, Result};

fn bad(what: impl Into<String>) -> FormatError {
    FormatError::Safetensors(format!("litertlm: {}", what.into()))
}

const SECTION_TFLITE_MODEL: u8 = 3;
const SECTION_SP_TOKENIZER: u8 = 4;
const SECTION_HF_TOKENIZER_ZLIB: u8 = 6;

/// A section directory entry.
#[derive(Debug)]
pub struct SectionInfo {
    pub begin: usize,
    pub end: usize,
    pub data_type: u8,
}

/// Reads the section directory of a `.litertlm` file header.
pub fn read_sections(header: &[u8]) -> Result<Vec<SectionInfo>> {
    if header.len() < 0x20 || &header[0..8] != b"LITERTLM" {
        return Err(bad("bad magic"));
    }
    let major = u32_le(header, 0x08)?;
    if major != 1 {
        return Err(bad(format!("unsupported major version {major}")));
    }
    let header_end = u64_le(header, 0x18)? as usize;
    if header_end > header.len() || header_end < 0x20 {
        return Err(bad("header end out of range"));
    }
    let fb = FlatBuffer::new(&header[0x20..header_end], None)?;
    let root = fb.root();
    // LiteRTLMMetaData.section_metadata (field 1) → SectionMetadata.objects (field 0)
    let sm = fb
        .uoffset(root, 1)
        .ok_or_else(|| bad("no section_metadata"))?;
    let objects = fb.table_vector(sm, 0)?;
    let mut out = Vec::with_capacity(objects.len());
    for obj in objects {
        let begin = fb.scalar_u64(obj, 1).ok_or_else(|| bad("section missing begin"))? as usize;
        let end = fb.scalar_u64(obj, 2).ok_or_else(|| bad("section missing end"))? as usize;
        let data_type = fb.scalar_u8(obj, 3).unwrap_or(0);
        out.push(SectionInfo { begin, end, data_type });
    }
    Ok(out)
}

/// Opens a `.litertlm` file as a [`ModelSource`]: finds the TFLiteModel
/// section and delegates to the TFLite block at its absolute offset;
/// an SP_Tokenizer section, when present, overrides the TFLite
/// section's own tokenizer metadata. (HF_Tokenizer_Zlib override lands
/// with the first archive that carries one.)
pub fn open_litertlm(path: &Path) -> Result<Box<dyn ModelSource>> {
    let head = {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut buf = vec![0u8; 1024 * 1024]; // header_end is always < 1MB
        let n = f.read(&mut buf)?;
        buf.truncate(n);
        buf
    };
    let sections = read_sections(&head)?;
    let tflite = sections
        .iter()
        .find(|s| s.data_type == SECTION_TFLITE_MODEL)
        .ok_or_else(|| bad("no TFLiteModel section"))?;
    // SP section bytes live at section offsets — read them eagerly
    // (tokenizer blobs are a few MB at most).
    let spm: Option<Vec<u8>> = sections
        .iter()
        .find(|s| s.data_type == SECTION_SP_TOKENIZER)
        .map(|s| {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = std::fs::File::open(path)?;
            f.seek(SeekFrom::Start(s.begin as u64))?;
            let mut buf = vec![0u8; s.end - s.begin];
            f.read_exact(&mut buf)?;
            Ok::<Vec<u8>, FormatError>(buf)
        })
        .transpose()?;
    Ok(Box::new(TfliteSource::load_at_with_spm(
        path,
        tflite.begin,
        spm.as_deref(),
    )?))
}

fn u32_le(d: &[u8], pos: usize) -> Result<u32> {
    let b = d.get(pos..pos + 4).ok_or_else(|| bad("u32 out of bounds"))?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn u64_le(d: &[u8], pos: usize) -> Result<u64> {
    let b = d.get(pos..pos + 8).ok_or_else(|| bad("u64 out of bounds"))?;
    Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
}
