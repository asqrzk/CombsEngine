//! GGUF adapter (llama.cpp ecosystem).
//!
//! Reads GGUF v2/v3 files: header, metadata KV pairs, tensor infos and the
//! aligned tensor data section (mmap-backed). Implements [`ModelSource`] by
//! mapping ggml names to HF names (`blk.0.attn_q.weight` →
//! `model.layers.0.self_attn.q_proj.weight`) and ggml dimensions to HF
//! layout (`[in, out]` → `[out, in]`, which for row-major data is just a
//! shape reversal — no data movement).
//!
//! Supported tensor types: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
//! Q4_K, Q5_K, Q6_K.
//! Quantized tensors are dequantized on load (CPU scalar path) — wiring
//! `QuantizedLinear` to keep them packed in VRAM is a follow-up. K-quant
//! superblock layouts follow ggml exactly (256-value blocks, 6-bit packed
//! scales/mins for 4/5_K, per-16 i8 scales for 6_K).
//!
//! Tokenizer: a sibling `tokenizer.json` is used when present; otherwise a
//! BPE `tokenizer.json` is synthesized from the GGUF tokenizer metadata
//! (tokens/scores/types/merges) and cached next to the model.
//!
//! Two ways in. [`GgufSource::load`] maps a file — the native path, and
//! the reason GGUF exists: weights are read where they lie.
//! [`GgufSource::from_bytes`] takes the file image already in memory, for
//! callers that have no filesystem to map (a browser tab holding a
//! downloaded model). Parsing, name mapping and every reader are shared;
//! only the byte custody and the tokenizer's origin differ.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(not(target_family = "wasm"))]
use memmap2::Mmap;

use crate::metadata::ModelMetadata;
use crate::source::{ModelSource, SamplerConfig, TensorDtype, TensorReader};
use crate::tokenizer::{TokenizerSource, TokenizerSpec};
use crate::{FormatError, Result};

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"

#[derive(Debug, Clone)]
pub(crate) enum MetaValue {
    U32(u32),
    I32(i32),
    U64(u64),
    F32(f32),
    Bool(bool),
    String(String),
    Strings(Vec<String>),
    F32s(Vec<f32>),
    I32s(Vec<i32>),
}

#[derive(Debug, Clone)]
pub(crate) struct TensorInfo {
    name: String,
    dims: Vec<usize>, // ggml order (fastest dim first)
    ggml_type: u32,
    /// Relative to the data section. `u64` because a 7B checkpoint's
    /// later tensors sit past 4 GB, and `usize` is 32 bits on wasm32 —
    /// the one target where a model that large is the whole point.
    offset: u64,
}

/// The GGUF byte image and who holds it.
///
/// Mapped from disk natively; owned outright when the bytes arrived
/// without a file behind them; SEGMENTED when they arrived by chunk
/// append on a 32-bit target — Rust caps any single allocation at
/// `isize::MAX`, which on wasm32 is ~2.14 GB, so a bigger image can
/// never be one contiguous `Vec<u8>` no matter how much linear memory
/// exists. All access goes through [`GgufData::slice`]; a payload
/// inside one segment borrows, one straddling a boundary joins into
/// an owned copy (bounded by the largest tensor, transient).
pub(crate) enum GgufData {
    #[cfg(not(target_family = "wasm"))]
    Mmap(Mmap),
    Owned(Vec<u8>),
    Segmented {
        /// Every segment is exactly `seg_len` bytes except the last.
        segments: Vec<Vec<u8>>,
        seg_len: usize,
        total: usize,
    },
    /// A moving view of an image that is still arriving, or has already
    /// been consumed and dropped. Only `[base, base + buf.len())` is
    /// readable; everything else is gone or not here yet, and asking
    /// for it fails rather than returning the wrong bytes. This is what
    /// lets a model be mounted without ever holding it whole.
    Window {
        buf: Vec<u8>,
        base: u64,
        total: u64,
    },
}

impl GgufData {
    fn len(&self) -> u64 {
        match self {
            #[cfg(not(target_family = "wasm"))]
            GgufData::Mmap(m) => m.len() as u64,
            GgufData::Owned(v) => v.len() as u64,
            GgufData::Segmented { total, .. } => *total as u64,
            GgufData::Window { total, .. } => *total,
        }
    }

    /// The contiguous head of the image — the whole slice for the
    /// contiguous holders, segment 0 for the segmented one. The header
    /// parser reads from here; segments are gigabyte-scale while GGUF
    /// headers are megabytes, and a header that outruns the first
    /// segment fails the parse loudly rather than silently.
    fn prefix(&self) -> &[u8] {
        match self {
            #[cfg(not(target_family = "wasm"))]
            GgufData::Mmap(m) => m,
            GgufData::Owned(v) => v,
            GgufData::Segmented { segments, .. } => {
                segments.first().map(|s| s.as_slice()).unwrap_or(&[])
            }
            // The header lives at offset zero; once the window has moved
            // past it there is no prefix to read and saying so is the
            // honest answer.
            GgufData::Window { buf, base, .. } => {
                if *base == 0 { buf.as_slice() } else { &[] }
            }
        }
    }

    /// `start..start + len` of the image; `None` past the end or, for a
    /// window, outside what it holds.
    fn slice(&self, start: u64, len: u64) -> Option<std::borrow::Cow<'_, [u8]>> {
        use std::borrow::Cow;
        let end = start.checked_add(len)?;
        if end > self.len() {
            return None;
        }
        // Everything below indexes a buffer that is in memory, so it fits
        // `usize` by construction — a held buffer cannot be larger than
        // the address space that holds it.
        let (start_us, end_us, len_us) = (
            usize::try_from(start).ok(),
            usize::try_from(end).ok(),
            usize::try_from(len).ok()?,
        );
        match self {
            #[cfg(not(target_family = "wasm"))]
            GgufData::Mmap(m) => m.get(start_us?..end_us?).map(Cow::Borrowed),
            GgufData::Owned(v) => v.get(start_us?..end_us?).map(Cow::Borrowed),
            GgufData::Segmented { segments, seg_len, .. } => {
                let seg = *seg_len as u64;
                let first = (start / seg) as usize;
                let last = ((end - 1) / seg) as usize;
                if first == last {
                    let off = (start - first as u64 * seg) as usize;
                    return segments.get(first)?.get(off..off + len_us).map(Cow::Borrowed);
                }
                // Boundary-straddling payload: join. Rare (one tensor
                // per gigabyte boundary) and bounded by tensor size.
                let mut out = Vec::with_capacity(len_us);
                let mut pos = start;
                while pos < end {
                    let idx = (pos / seg) as usize;
                    let off = (pos - idx as u64 * seg) as usize;
                    let take = ((seg - off as u64).min(end - pos)) as usize;
                    out.extend_from_slice(segments.get(idx)?.get(off..off + take)?);
                    pos += take as u64;
                }
                Some(Cow::Owned(out))
            }
            GgufData::Window { buf, base, .. } => {
                let off = usize::try_from(start.checked_sub(*base)?).ok()?;
                buf.get(off..off.checked_add(len_us)?).map(Cow::Borrowed)
            }
        }
    }
}

/// Everything before the tensor payloads: what a mount must hold in
/// full before it can place a single byte of weight. Parsed apart from
/// the rest of the file so a mount that is still arriving can ask for
/// it and be told, honestly, that it is not all here yet.
pub(crate) struct GgufHeader {
    pub(crate) version: u32,
    pub(crate) kv: HashMap<String, MetaValue>,
    pub(crate) tensors: HashMap<String, TensorInfo>,
    /// Offset of the first payload byte — the alignment boundary after
    /// the info section.
    pub(crate) data_start: u64,
}

impl GgufHeader {
    /// Parse from a prefix of the image. `Ok(None)` means "not all here
    /// yet, feed more"; `Err` means this will never parse however many
    /// bytes arrive. Pass `total` when the image's eventual length is
    /// known — it is what separates a header still in flight from a
    /// corrupt length field claiming the file is enormous.
    ///
    /// `Ok(Some)` only once `data_start` bytes are in hand, the alignment
    /// padding included: a mount files the next byte off the wire at
    /// `data_start`, so a prefix that stops inside the padding is not a
    /// header it can steer by.
    pub(crate) fn try_parse(bytes: &[u8], total: Option<u64>) -> Result<Option<Self>> {
        let mut c = Cursor::with_limit(bytes, total);
        macro_rules! more {
            ($e:expr) => {
                match $e {
                    Ok(v) => v,
                    Err(err) => {
                        return if c.exhausted { Ok(None) } else { Err(err) };
                    }
                }
            };
        }

        if more!(c.u32()) != GGUF_MAGIC {
            return Err(FormatError::Safetensors("gguf: bad magic".into()));
        }
        let version = more!(c.u32());
        if !(2..=3).contains(&version) {
            return Err(FormatError::Safetensors(format!(
                "gguf: unsupported version {version}"
            )));
        }
        let tensor_count = more!(c.u64()) as usize;
        let kv_count = more!(c.u64()) as usize;

        // Counts come off the wire before the entries they describe, so
        // they size allocations before anything has validated them. A
        // corrupt pair would otherwise reserve gigabytes on a header
        // that is only a few bytes in.
        let entry_floor = 12usize; // smallest possible kv or tensor record
        if let Some(limit) = total {
            let claimed = (tensor_count
                .saturating_add(kv_count)
                .saturating_mul(entry_floor)) as u64;
            if claimed > limit {
                return Err(FormatError::Safetensors(format!(
                    "gguf: header claims {kv_count} metadata entries and \
                     {tensor_count} tensors, which cannot fit in {limit} bytes"
                )));
            }
        }

        let mut kv = HashMap::with_capacity(kv_count.min(4096));
        for _ in 0..kv_count {
            let key = more!(c.string());
            let ty = more!(c.u32());
            let value = more!(read_meta_value(&mut c, ty));
            kv.insert(key, value);
        }

        let mut tensors = HashMap::with_capacity(tensor_count.min(4096));
        for _ in 0..tensor_count {
            let name = more!(c.string());
            let n_dims = more!(c.u32()) as usize;
            let mut dims = Vec::with_capacity(n_dims.min(8));
            for _ in 0..n_dims {
                dims.push(more!(c.u64()) as usize);
            }
            let ggml_type = more!(c.u32());
            let offset = more!(c.u64());
            tensors.insert(name.clone(), TensorInfo { name, dims, ggml_type, offset });
        }

        // Tensor data starts after the info section, aligned to 32 bytes.
        let alignment = match kv.get("general.alignment") {
            Some(MetaValue::U32(a)) => *a as usize,
            _ => 32,
        };
        let alignment = alignment.max(1);
        let data_start = (c.pos.div_ceil(alignment) * alignment) as u64;

        // Split GGUFs (llama.cpp `gguf-split` shards, `…-00001-of-0000N`)
        // carry only a slice of the tensors; loading one would fail later
        // with a baffling missing-tensor error and a wrong tied-head guess.
        // Refused HERE so a streamed mount finds out before it has pulled
        // a gigabyte rather than after.
        if let Some(MetaValue::U32(count)) = kv.get("split.count") {
            if *count > 1 {
                let no = match kv.get("split.no") {
                    Some(MetaValue::U32(n)) => *n + 1,
                    _ => 1,
                };
                return Err(FormatError::Safetensors(format!(
                    "split GGUF: this file is shard {no} of {count} — \
                     multi-file GGUF loading is not supported yet; pull a \
                     single-file quant or merge the shards with llama.cpp's \
                     `llama-gguf-split --merge`"
                )));
            }
        }

        // `data_start` is the alignment boundary after the info section,
        // not its end: the bytes between are padding this parser never
        // reads. A mount files the first byte after the header at
        // `data_start`, so a prefix that ends inside the padding is a
        // header it cannot steer by — every tensor would land the width
        // of the padding early. Measured on nine cached files the gap is
        // 12 to 28 bytes and never zero.
        let have = bytes.len() as u64;
        if have < data_start {
            return match total {
                Some(t) if t < data_start => Err(FormatError::Safetensors(format!(
                    "gguf: tensor data starts at byte {data_start} but the file is {t} bytes"
                ))),
                _ => Ok(None),
            };
        }

        Ok(Some(GgufHeader { version, kv, tensors, data_start }))
    }
}

/// Where every tensor lives, and where the payloads begin — the map a
/// streaming mount steers by. Read from the first bytes off the wire,
/// before any payload has been asked for.
#[derive(Debug, Clone)]
pub struct GgufHeaderInfo {
    /// Offset of the first payload byte.
    pub data_start: u64,
    /// Model architecture, so a mount can refuse an unsupported one
    /// here rather than a gigabyte later.
    pub architecture: String,
    /// `(ggml name, absolute start, byte length)`, **sorted by start**.
    /// One `Response.body` delivers bytes in file order, so this is the
    /// order a mount will meet them in.
    pub tensors: Vec<(String, u64, u64)>,
}

/// Read a GGUF header from a prefix of the file. `Ok(None)` means the
/// header is not all here yet — feed more and ask again; `Err` means it
/// will never parse. `total` is the file's eventual length when known,
/// which is what separates the two. `Some` arrives only once `data_start`
/// bytes are in hand, the alignment padding included.
pub fn read_gguf_header(bytes: &[u8], total: Option<u64>) -> Result<Option<GgufHeaderInfo>> {
    let Some(header) = GgufHeader::try_parse(bytes, total)? else {
        return Ok(None);
    };
    let architecture = match header.kv.get("general.architecture") {
        Some(MetaValue::String(s)) => s.clone(),
        _ => String::new(),
    };
    let mut tensors: Vec<(String, u64, u64)> = header
        .tensors
        .values()
        .map(|info| {
            tensor_byte_size(info)
                .map(|size| (info.name.clone(), header.data_start + info.offset, size as u64))
        })
        .collect::<Result<Vec<_>>>()?;
    tensors.sort_by_key(|(_, start, _)| *start);
    Ok(Some(GgufHeaderInfo {
        data_start: header.data_start,
        architecture,
        tensors,
    }))
}

/// Narrow a payload to a byte range, borrowing when it already borrows.
fn narrow_cow<'a>(
    data: std::borrow::Cow<'a, [u8]>,
    start: usize,
    len: usize,
) -> std::borrow::Cow<'a, [u8]> {
    use std::borrow::Cow;
    match data {
        Cow::Borrowed(b) => Cow::Borrowed(&b[start..start + len]),
        Cow::Owned(v) => Cow::Owned(v[start..start + len].to_vec()),
    }
}

/// A parsed GGUF file.
pub struct GgufSource {
    /// The file this was read from, when it came from one.
    path: Option<PathBuf>,
    data: GgufData,
    metadata: ModelMetadata,
    kv: HashMap<String, MetaValue>,
    tensors: HashMap<String, TensorInfo>,
    data_start: u64,
    tokenizer: TokenizerSource,
    added_tokens: HashMap<u32, String>,
    eos_ids: Vec<u32>,
    bos_id: Option<u32>,
}

// ---------------------------------------------------------------------------
// parsing helpers (little-endian cursor)

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
    /// Set when a read ran past the end of `buf` while still inside what
    /// the image is eventually going to be. A streaming mount has to
    /// tell "not yet" from "never": the first means feed more bytes, the
    /// second means the file is wrong, and an error string cannot be
    /// asked which it was.
    exhausted: bool,
    /// The image's full length when it is known. A read past this is
    /// malformed however few bytes are in hand — without it, a corrupt
    /// string length asking for a terabyte would masquerade as a
    /// truncated header forever.
    limit: Option<u64>,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0, exhausted: false, limit: None }
    }
    fn with_limit(buf: &'a [u8], limit: Option<u64>) -> Self {
        Cursor { buf, pos: 0, exhausted: false, limit }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            let within_image = match self.limit {
                Some(limit) => self.pos.saturating_add(n) as u64 <= limit,
                None => true,
            };
            self.exhausted = within_image;
            return Err(FormatError::Safetensors("gguf: unexpected end of file".into()));
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| FormatError::Safetensors(format!("gguf: bad utf8 string: {e}")))
    }
}

const META_U8: u32 = 0;
const META_I8: u32 = 1;
const META_U16: u32 = 2;
const META_I16: u32 = 3;
const META_U32: u32 = 4;
const META_I32: u32 = 5;
const META_F32: u32 = 6;
const META_BOOL: u32 = 7;
const META_STRING: u32 = 8;
const META_ARRAY: u32 = 9;
const META_U64: u32 = 10;
const META_I64: u32 = 11;
const META_F64: u32 = 12;

fn read_meta_value(c: &mut Cursor, ty: u32) -> Result<MetaValue> {
    Ok(match ty {
        META_U8 => MetaValue::U32(c.u8()? as u32),
        META_I8 => MetaValue::I32(c.u8()? as i8 as i32),
        META_U16 => MetaValue::U32(c.u16()? as u32),
        META_I16 => MetaValue::I32(c.u16()? as i16 as i32),
        META_U32 => MetaValue::U32(c.u32()?),
        META_I32 => MetaValue::I32(c.i32()?),
        META_U64 => MetaValue::U64(c.u64()?),
        META_I64 => MetaValue::U64(c.i64()? as u64),
        META_F32 => MetaValue::F32(c.f32()?),
        META_F64 => MetaValue::F32(f64::from_le_bytes(c.take(8)?.try_into().unwrap()) as f32),
        META_BOOL => MetaValue::Bool(c.u8()? != 0),
        META_STRING => MetaValue::String(c.string()?),
        META_ARRAY => {
            let elem_ty = c.u32()?;
            let len = c.u64()? as usize;
            match elem_ty {
                META_STRING => {
                    let mut out = Vec::with_capacity(len);
                    for _ in 0..len {
                        out.push(c.string()?);
                    }
                    MetaValue::Strings(out)
                }
                META_F32 => {
                    let mut out = Vec::with_capacity(len);
                    for _ in 0..len {
                        out.push(c.f32()?);
                    }
                    MetaValue::F32s(out)
                }
                META_I32 | META_I16 | META_I8 => {
                    let mut out = Vec::with_capacity(len);
                    for _ in 0..len {
                        out.push(read_meta_value(c, elem_ty).map(|v| match v {
                            MetaValue::I32(i) => i,
                            _ => 0,
                        })?);
                    }
                    MetaValue::I32s(out)
                }
                META_U32 | META_U16 | META_U8 => {
                    let mut out = Vec::with_capacity(len);
                    for _ in 0..len {
                        out.push(read_meta_value(c, elem_ty).map(|v| match v {
                            MetaValue::U32(u) => u as i32,
                            _ => 0,
                        })?);
                    }
                    MetaValue::I32s(out)
                }
                other => {
                    return Err(FormatError::Safetensors(format!(
                        "gguf: unsupported metadata array element type {other}"
                    )));
                }
            }
        }
        other => {
            return Err(FormatError::Safetensors(format!(
                "gguf: unsupported metadata type {other}"
            )));
        }
    })
}

// ggml tensor types we support.
const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
const GGML_Q4_0: u32 = 2;
const GGML_Q4_1: u32 = 3;
const GGML_Q5_0: u32 = 6;
const GGML_Q5_1: u32 = 7;
const GGML_Q8_0: u32 = 8;
const GGML_Q4_K: u32 = 12;
const GGML_Q5_K: u32 = 13;
const GGML_Q6_K: u32 = 14;
const GGML_BF16: u32 = 30;

impl GgufSource {
    /// Opens and parses a `.gguf` file, mapping it into memory.
    #[cfg(not(target_family = "wasm"))]
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse(GgufData::Mmap(mmap), Some(path))
    }

    /// Parses a `.gguf` file image already held in memory.
    ///
    /// The whole file must be present — GGUF's tensor offsets are absolute
    /// within it, so a partial buffer is not a smaller model, it is a
    /// wrong one. With no file to sit beside, the tokenizer is synthesized
    /// into memory rather than cached to disk.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::parse(GgufData::Owned(bytes), None)
    }

    /// [`GgufSource::from_bytes`] for an image held as fixed-size
    /// segments — how chunk-appended mounts arrive on wasm32, where a
    /// single allocation caps at `isize::MAX` (~2.14 GB) and a bigger
    /// image can never be one `Vec<u8>`. Every segment must be exactly
    /// `seg_len` bytes except the last.
    pub fn from_segments(segments: Vec<Vec<u8>>, seg_len: usize) -> Result<Self> {
        if seg_len == 0 {
            return Err(FormatError::Safetensors("gguf: zero segment length".into()));
        }
        let total: usize = segments.iter().map(Vec::len).sum();
        for (i, seg) in segments.iter().enumerate() {
            let want = if i + 1 == segments.len() { seg.len() } else { seg_len };
            if seg.len() != want || (i + 1 < segments.len() && seg.len() != seg_len) {
                return Err(FormatError::Safetensors(format!(
                    "gguf: segment {i} is {} bytes, expected {seg_len}",
                    seg.len()
                )));
            }
        }
        Self::parse(GgufData::Segmented { segments, seg_len, total }, None).map_err(|e| {
            FormatError::Safetensors(format!(
                "{e} (segmented image: the header must fit the first segment —                  {seg_len} bytes here)"
            ))
        })
    }

    fn parse(data: GgufData, path: Option<PathBuf>) -> Result<Self> {
        // The header reads from the contiguous head of the image; a
        // segmented image's first segment is gigabyte-scale while GGUF
        // headers are megabytes, and a header that outruns it fails
        // loudly rather than silently.
        let total = data.len();
        let header = GgufHeader::try_parse(data.prefix(), Some(total))?.ok_or_else(|| {
            FormatError::Safetensors(
                "gguf: header runs past the bytes in hand — a truncated file, \
                 or a header larger than the first segment"
                    .into(),
            )
        })?;
        Self::from_header(header, data, path)
    }

    /// A source over a MOVING WINDOW of an image: `window` holds bytes
    /// `[base, base + window.len())` of a file `total` bytes long, and
    /// nothing else is readable. `header_bytes` must cover the header,
    /// which a mount has in hand from the first bytes off the wire.
    ///
    /// Every tensor whose payload lies inside the window reads exactly
    /// as it would from the whole file; every other tensor fails by
    /// name. That is the whole contract, and it is what lets a model be
    /// mounted without the file ever existing anywhere entire.
    pub fn from_window(
        header_bytes: &[u8],
        window: Vec<u8>,
        base: u64,
        total: u64,
    ) -> Result<Self> {
        let header = GgufHeader::try_parse(header_bytes, Some(total))?.ok_or_else(|| {
            FormatError::Safetensors(
                "gguf: window given a header that is not all there".into(),
            )
        })?;
        Self::from_header(header, GgufData::Window { buf: window, base, total }, None)
    }

    /// Take the window's buffer back.
    ///
    /// A mount hands its window in to read from and wants it back to
    /// keep filling — copying it in and out would cost the whole model
    /// in memcpy and, worse, double the live bytes at the moment the
    /// largest tensor is resident.
    pub fn into_window_buf(self) -> Option<Vec<u8>> {
        match self.data {
            GgufData::Window { buf, .. } => Some(buf),
            _ => None,
        }
    }

    /// Assemble a source from a header that has ALREADY been parsed and
    /// an image that may be less than whole. The streaming mount reads
    /// the header from the first bytes off the wire and then never has
    /// the file entire, so the two steps have to be separable; the
    /// whole-file path just does them back to back.
    pub(crate) fn from_header(
        header: GgufHeader,
        data: GgufData,
        path: Option<PathBuf>,
    ) -> Result<Self> {
        let GgufHeader { kv, tensors, data_start, .. } = header;
        let metadata = build_model_metadata(&kv)?;
        let (eos_ids, bos_id, added_tokens) = tokenizer_ids(&kv);
        // A file on disk gets the sibling-or-cached tokenizer.json (written
        // once, reused by every later run); bytes in memory get the same
        // JSON synthesized in memory. Same text either way.
        let tokenizer = match &path {
            #[cfg(not(target_family = "wasm"))]
            Some(p) => TokenizerSource::Path(ensure_tokenizer_json(p, &kv)?),
            _ => TokenizerSource::Bytes(synthesize_tokenizer_json(&kv)?.into_bytes()),
        };

        let mut source = GgufSource {
            path,
            data,
            metadata,
            kv,
            tensors,
            data_start,
            tokenizer,
            added_tokens,
            eos_ids,
            bos_id,
        };
        // GGUF llama files usually include output.weight; if absent, lm_head
        // is tied to the embedding matrix.
        source.metadata.tie_word_embeddings = !source.tensors.contains_key("output.weight");

        // A tensor the map doesn't know is a loud one-line warning, never a
        // silent drop — unmapped weights mean wrong output, not no output.
        let arch = source.metadata.architecture.clone();
        let mut unmapped: Vec<&str> = source
            .tensors
            .keys()
            .filter(|k| {
                map_tensor_name(k, &arch).is_none()
                    && !KNOWN_UNMAPPED.contains(&k.as_str())
                    && !is_fused_source(k, &arch)
            })
            .map(String::as_str)
            .collect();
        if !unmapped.is_empty() {
            unmapped.sort();
            eprintln!(
                "[gguf] {} unmapped tensors (arch {arch}); first: {}",
                unmapped.len(),
                unmapped[0]
            );
        }
        Ok(source)
    }

    fn kv_u64(&self, key: &str) -> Option<u64> {
        match self.kv.get(key) {
            Some(MetaValue::U32(v)) => Some(*v as u64),
            Some(MetaValue::U64(v)) => Some(*v),
            Some(MetaValue::I32(v)) => Some(*v as u64),
            _ => None,
        }
    }
}

fn build_model_metadata(kv: &HashMap<String, MetaValue>) -> Result<ModelMetadata> {
    let get_u64 = |key: &str| -> Option<u64> {
        match kv.get(key) {
            Some(MetaValue::U32(v)) => Some(*v as u64),
            Some(MetaValue::U64(v)) => Some(*v),
            Some(MetaValue::I32(v)) => Some(*v as u64),
            _ => None,
        }
    };
    let get_f32 = |key: &str| -> Option<f32> {
        match kv.get(key) {
            Some(MetaValue::F32(v)) => Some(*v),
            Some(MetaValue::U32(v)) => Some(*v as f32),
            _ => None,
        }
    };
    let get_str = |key: &str| -> Option<String> {
        match kv.get(key) {
            Some(MetaValue::String(s)) => Some(s.clone()),
            _ => None,
        }
    };

    let arch = get_str("general.architecture")
        .ok_or_else(|| FormatError::MissingField("general.architecture".into()))?;
    let prefix = arch.clone();
    let field = |name: &str| get_u64(&format!("{prefix}.{name}"));

    let hidden = field("embedding_length")
        .ok_or_else(|| FormatError::MissingField("embedding_length".into()))? as usize;
    let heads = field("attention.head_count")
        .ok_or_else(|| FormatError::MissingField("attention.head_count".into()))?
        as usize;
    let kv_heads = field("attention.head_count_kv").unwrap_or(heads as u64) as usize;
    let layers = field("block_count")
        .ok_or_else(|| FormatError::MissingField("block_count".into()))? as usize;
    let ctx = field("context_length").unwrap_or(2048) as usize;
    let ffn = field("feed_forward_length").unwrap_or((hidden * 4) as u64) as usize;
    let vocab = match kv.get(&format!("{prefix}.vocab_size")) {
        Some(MetaValue::U32(v)) => *v as usize,
        Some(MetaValue::U64(v)) => *v as usize,
        _ => match kv.get("tokenizer.ggml.tokens") {
            Some(MetaValue::Strings(t)) => t.len(),
            _ => 0,
        },
    };
    let (eos_ids, bos_id, _) = tokenizer_ids(kv);

    Ok(ModelMetadata {
        architecture: arch,
        hidden_size: hidden,
        intermediate_size: ffn,
        num_hidden_layers: layers,
        num_attention_heads: heads,
        num_key_value_heads: kv_heads,
        vocab_size: vocab,
        max_position_embeddings: ctx,
        rms_norm_eps: get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5) as f64,
        rope_theta: get_f32(&format!("{prefix}.rope.freq_base")).unwrap_or(10000.0) as f64,
        // GGUF files usually include output.weight; if absent, the head is tied.
        tie_word_embeddings: false, // refined in load() via tensor presence
        // Explicit head size when present — gemma3 (256) and qwen3 (128)
        // decouple it from hidden/heads.
        head_dim: field("attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(hidden / heads),
        // Bias loading is presence-driven (the loader probes bias tensors),
        // so this flag is informational only for GGUF.
        attention_bias: false,
        bos_token_id: bos_id,
        eos_token_ids: eos_ids,
        vision: None,
        // Sliding window when the file declares one (phi3 writes
        // `attention.sliding_window`); the remaining gemma layout keys
        // (pattern, local rope base) land with the arch-aware map.
        attention_pattern: crate::metadata::AttentionPattern {
            sliding_window: field("attention.sliding_window").map(|v| v as usize),
            ..Default::default()
        },
        // GGUF stores no activation key; the per-arch resolver supplies it.
        activation: crate::metadata::Activation::default(),
        rope_scaling: gguf_rope_scaling(kv, &prefix)?,
    })
}

/// GGUF `rope.scaling.*` keys (llama.cpp writes `linear`/`yarn`; llama3
/// scaling is baked as a `rope_freqs.weight` tensor instead — that tensor
/// is a separate follow-up and absent from our cached files).
fn gguf_rope_scaling(
    kv: &HashMap<String, MetaValue>,
    prefix: &str,
) -> Result<crate::metadata::RopeScaling> {
    use crate::metadata::RopeScaling;
    let get_f32 = |key: String| match kv.get(&key) {
        Some(MetaValue::F32(v)) => Some(*v as f64),
        _ => None,
    };
    let kind = match kv.get(&format!("{prefix}.rope.scaling.type")) {
        Some(MetaValue::String(s)) => s.clone(),
        _ => return Ok(RopeScaling::None),
    };
    let factor = get_f32(format!("{prefix}.rope.scaling.factor")).unwrap_or(1.0);
    let orig = match kv.get(&format!("{prefix}.rope.scaling.original_context_length")) {
        Some(MetaValue::U32(v)) => *v as usize,
        Some(MetaValue::U64(v)) => *v as usize,
        _ => 32768,
    };
    match kind.as_str() {
        "none" => Ok(RopeScaling::None),
        "linear" => Ok(RopeScaling::Linear { factor }),
        "yarn" => Ok(RopeScaling::Yarn {
            factor,
            original_max_position_embeddings: orig,
            beta_fast: get_f32(format!("{prefix}.rope.scaling.yarn_beta_fast")).unwrap_or(32.0),
            beta_slow: get_f32(format!("{prefix}.rope.scaling.yarn_beta_slow")).unwrap_or(1.0),
            attention_factor: None,
        }),
        other => Err(FormatError::MissingField(format!(
            "unsupported GGUF rope scaling type {other:?}"
        ))),
    }
}

/// End-of-generation control tokens that chat finetunes emit to terminate a
/// turn while the file's declared eos stays the base `<|endoftext|>`-style id
/// (phi-3's `<|end|>`, llama-3's `<|eot_id|>`, gemma's `<end_of_turn>`).
/// Mirrors llama.cpp's `special_eog_ids` name scan.
const EOG_TOKENS: &[&str] = &[
    "<|end|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<|im_end|>",
    "<end_of_turn>",
    "<|end_of_text|>",
    "<|endoftext|>",
    "<EOT>",
];

fn tokenizer_ids(kv: &HashMap<String, MetaValue>) -> (Vec<u32>, Option<u32>, HashMap<u32, String>) {
    let mut eos = Vec::new();
    let mut bos = None;
    let mut added = HashMap::new();
    if let Some(MetaValue::U32(v)) = kv.get("tokenizer.ggml.eos_token_id") {
        eos.push(*v);
    }
    if let Some(MetaValue::U32(v)) = kv.get("tokenizer.ggml.bos_token_id") {
        bos = Some(*v);
    }
    // Mark special tokens from the token_type array (3 = control).
    if let (Some(MetaValue::Strings(tokens)), Some(MetaValue::I32s(types))) =
        (kv.get("tokenizer.ggml.tokens"), kv.get("tokenizer.ggml.token_type"))
    {
        for (i, (tok, ty)) in tokens.iter().zip(types.iter()).enumerate() {
            if *ty == 3 {
                added.insert(i as u32, tok.clone());
            }
            if *ty == 3 && EOG_TOKENS.contains(&tok.as_str()) && !eos.contains(&(i as u32)) {
                eos.push(i as u32);
            }
        }
    }
    (eos, bos, added)
}

/// Counts GGUF special tokens (token_type 3 = control, 4 = user-defined).
fn special_token_count(kv: &HashMap<String, MetaValue>) -> usize {
    match kv.get("tokenizer.ggml.token_type") {
        Some(MetaValue::I32s(types)) => types.iter().filter(|t| **t == 3 || **t == 4).count(),
        _ => 0,
    }
}

/// BPE split regex for the synthesized tokenizer, selected by the GGUF
/// `tokenizer.ggml.pre` family. Qwen2 splits digit runs into single digits
/// (`\p{N}`) where the GPT-2/llama-bpe families group up to three
/// (`\p{N}{1,3}`) — digit tokenization drift is user-visible on coder models.
fn pretokenizer_regex(kv: &HashMap<String, MetaValue>) -> &'static str {
    const DEFAULT: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    const QWEN2: &str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    match kv.get("tokenizer.ggml.pre") {
        Some(MetaValue::String(pre)) if pre == "qwen2" => QWEN2,
        _ => DEFAULT,
    }
}

/// Builds a minimal HF BPE tokenizer.json from GGUF tokenizer metadata when
/// no sibling tokenizer.json exists (cached alongside the model file).
///
/// Staleness: v1 syntheses wrote `"added_tokens": []`, which BPE-shredded
/// ChatML control tokens (`<|im_start|>`/`<|im_end|>`) — the "garbled GGUF"
/// bug — and the poisoned cache was sticky. The cached file is regenerated
/// whenever its added_tokens count disagrees with the GGUF metadata. (No
/// in-file version marker: the tokenizers crate rejects unknown top-level
/// keys.)
#[cfg(not(target_family = "wasm"))]
fn ensure_tokenizer_json(path: &Path, kv: &HashMap<String, MetaValue>) -> Result<PathBuf> {
    let sibling = path.with_file_name("tokenizer.json");
    if sibling.exists() {
        return Ok(sibling);
    }
    let cached = path.with_extension("tokenizer.json");
    if cached.exists() {
        let parsed = std::fs::read_to_string(&cached)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
        let have_added = parsed
            .as_ref()
            .and_then(|v| v.get("added_tokens")?.as_array().map(Vec::len));
        // Older syntheses hardcoded the GPT-2 split regex for every family;
        // regenerate when the cached regex disagrees with the `pre` family.
        let have_regex = parsed.as_ref().and_then(|v| {
            v.get("pre_tokenizer")?
                .get("pretokenizers")?
                .get(0)?
                .get("pattern")?
                .get("Regex")?
                .as_str()
                .map(str::to_string)
        });
        if have_added == Some(special_token_count(kv))
            && have_regex.as_deref() == Some(pretokenizer_regex(kv))
        {
            return Ok(cached);
        }
        // Stale synthesis — regenerate below.
    }

    let serialized = synthesize_tokenizer_json(kv)?;
    std::fs::write(&cached, serialized)?;
    Ok(cached)
}

/// Builds the HF BPE `tokenizer.json` text from GGUF tokenizer metadata.
///
/// Pure: no filesystem, no caching, no decisions about where the result
/// should live — those belong to the caller, and only the caller knows
/// whether there is a disk to put it on.
fn synthesize_tokenizer_json(kv: &HashMap<String, MetaValue>) -> Result<String> {
    let tokens = match kv.get("tokenizer.ggml.tokens") {
        Some(MetaValue::Strings(t)) => t,
        _ => {
            return Err(FormatError::MissingField(
                "tokenizer.ggml.tokens (and no sibling tokenizer.json)".into(),
            ));
        }
    };
    let scores: Vec<f32> = match kv.get("tokenizer.ggml.scores") {
        Some(MetaValue::F32s(s)) => s.clone(),
        _ => vec![0.0; tokens.len()],
    };
    let merges: Vec<String> = match kv.get("tokenizer.ggml.merges") {
        Some(MetaValue::Strings(m)) => m.clone(),
        _ => vec![],
    };

    // Two tokenizer families hide behind GGUF metadata:
    // `tokenizer.ggml.model = "gpt2"` is byte-level BPE (qwen, smollm,
    // llama3), synthesized below as before; `= "llama"` is SentencePiece
    // Unigram (gemma, llama1/2, mistral) — token strings carry the U+2581
    // word marker and <0xNN> byte-fallback entries, scores are unigram
    // log-probs and there are no merges. Synthesizing BPE for an SPM
    // vocab produces a tokenizer that mangles every prompt and prints
    // literal U+2581 in output — exactly what the browser (which has no
    // sibling tokenizer.json to fall back on) shipped for gemma3.
    let spm = matches!(
        kv.get("tokenizer.ggml.model"),
        Some(MetaValue::String(m)) if m == "llama"
    );
    if spm {
        return synthesize_spm_tokenizer_json(kv, tokens, &scores);
    }

    let mut vocab = serde_json::Map::new();
    for (i, tok) in tokens.iter().enumerate() {
        vocab.insert(tok.clone(), serde_json::Value::from(i));
    }
    let mut ordered: Vec<usize> = (0..tokens.len()).collect();
    ordered.sort_by(|&a, &b| {
        scores[a].partial_cmp(&scores[b]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = serde_json::Map::new();
    for (rank, id) in ordered.iter().enumerate() {
        ranks.insert(id.to_string(), serde_json::Value::from(rank));
    }

    // Special tokens (GGUF token_type 3 = control, 4 = user-defined) must
    // be registered as added_tokens so the tokenizer keeps them atomic
    // instead of BPE-shredding them (e.g. <|im_start|>/<|im_end|>).
    let mut added_tokens = Vec::new();
    if let Some(MetaValue::I32s(types)) = kv.get("tokenizer.ggml.token_type") {
        for (i, (tok, ty)) in tokens.iter().zip(types.iter()).enumerate() {
            if *ty == 3 || *ty == 4 {
                added_tokens.push(serde_json::json!({
                    "id": i,
                    "content": tok,
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true,
                }));
            }
        }
    }

    let json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": pretokenizer_regex(kv)}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
            ]
        },
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": vocab,
            "merges": merges,
        }
    });
    serde_json::to_string(&json)
        .map_err(|e| FormatError::Safetensors(format!("tokenizer json: {e}")))
}

/// Builds the HF `tokenizer.json` for the SentencePiece ("llama") GGUF
/// family — SPM-flavored BPE, the shape transformers' own converter
/// produces and the shape every sibling tokenizer.json of this family
/// ships: vocab with U+2581 word markers and <0xNN> byte fallback,
/// merges DERIVED from the piece scores (GGUF stores score = -merge_rank
/// and no merges), a space -> U+2581 normalizer with the family's
/// optional dummy prefix, and a Split-space pre-tokenizer.
fn synthesize_spm_tokenizer_json(
    kv: &HashMap<String, MetaValue>,
    tokens: &[String],
    scores: &[f32],
) -> Result<String> {
    let unk_token = match kv.get("tokenizer.ggml.unknown_token_id") {
        Some(MetaValue::U32(v)) => tokens
            .get(*v as usize)
            .cloned()
            .unwrap_or_else(|| "<unk>".to_string()),
        _ => "<unk>".to_string(),
    };
    let prepend = match kv.get("tokenizer.ggml.add_space_prefix") {
        Some(MetaValue::Bool(b)) => *b,
        _ => true,
    };

    let mut vocab = serde_json::Map::new();
    let mut ids: HashMap<&str, usize> = HashMap::with_capacity(tokens.len());
    for (i, tok) in tokens.iter().enumerate() {
        vocab.insert(tok.clone(), serde_json::Value::from(i));
        ids.insert(tok.as_str(), i);
    }

    // transformers' generate_merges, verbatim in shape: every split of
    // every piece whose halves are both in the vocab is a candidate,
    // locally ordered by the halves' ids, globally ordered by the parent
    // piece's score descending (score = -rank, so this restores the
    // original merge order). A pair (l, r) determines its parent l+r
    // uniquely, so no dedupe is needed.
    let mut merges: Vec<(usize, usize, f32)> = Vec::new();
    for (piece, &score) in tokens.iter().zip(scores) {
        let mut local: Vec<(usize, usize)> = Vec::new();
        for (split, _) in piece.char_indices().skip(1) {
            let (l, r) = piece.split_at(split);
            if let (Some(&li), Some(&ri)) = (ids.get(l), ids.get(r)) {
                local.push((li, ri));
            }
        }
        local.sort_unstable();
        merges.extend(local.into_iter().map(|(l, r)| (l, r, score)));
    }
    merges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    let merges: Vec<serde_json::Value> = merges
        .into_iter()
        .map(|(l, r, _)| serde_json::json!([tokens[l], tokens[r]]))
        .collect();

    let mut added_tokens = Vec::new();
    if let Some(MetaValue::I32s(types)) = kv.get("tokenizer.ggml.token_type") {
        for (i, (tok, ty)) in tokens.iter().zip(types.iter()).enumerate() {
            if *ty == 3 || *ty == 4 {
                added_tokens.push(serde_json::json!({
                    "id": i,
                    "content": tok,
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true,
                }));
            }
        }
    }

    let mut normalizers = Vec::new();
    if prepend {
        normalizers.push(serde_json::json!({"type": "Prepend", "prepend": "\u{2581}"}));
    }
    normalizers.push(serde_json::json!({
        "type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"
    }));
    let mut decoders = vec![
        serde_json::json!({"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "}),
        serde_json::json!({"type": "ByteFallback"}),
        serde_json::json!({"type": "Fuse"}),
    ];
    if prepend {
        decoders.push(serde_json::json!({"type": "Strip", "content": " ", "start": 1, "stop": 0}));
    }

    let json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": {"type": "Sequence", "normalizers": normalizers},
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"String": " "},
            "behavior": "MergedWithPrevious",
            "invert": false
        },
        "post_processor": null,
        "decoder": {"type": "Sequence", "decoders": decoders},
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": unk_token,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": true,
            "byte_fallback": true,
            "ignore_merges": false,
            "vocab": vocab,
            "merges": merges,
        }
    });
    serde_json::to_string(&json)
        .map_err(|e| FormatError::Safetensors(format!("tokenizer json: {e}")))
}

/// Maps a ggml tensor name to its HF equivalent (`None` = skip tensor).
/// Arch-keyed where a ggml name means different HF norms per family:
/// llama's `ffn_norm` is the pre-MLP `post_attention_layernorm`, but
/// gemma3's `ffn_norm` is `pre_feedforward_layernorm` (its
/// `post_attention_norm`/`post_ffw_norm` are the sandwich norms).
fn map_tensor_name(ggml: &str, arch: &str) -> Option<String> {
    if ggml == "token_embd.weight" {
        return Some("model.embed_tokens.weight".into());
    }
    if ggml == "output.weight" {
        return Some("lm_head.weight".into());
    }
    if ggml == "output_norm.weight" {
        return Some("model.norm.weight".into());
    }
    let rest = ggml.strip_prefix("blk.")?;
    let (layer, rest) = rest.split_once('.')?;
    let gemma = matches!(arch, "gemma3" | "gemma3_text");
    let hf = match rest {
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" if gemma => "pre_feedforward_layernorm.weight",
        "post_attention_norm.weight" if gemma => "post_attention_layernorm.weight",
        "post_ffw_norm.weight" if gemma => "post_feedforward_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        // Per-head QK norms (qwen3, gemma3); the loader probes these only
        // when the resolved spec asks for them.
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q.bias" => "self_attn.q_proj.bias",
        "attn_k.bias" => "self_attn.k_proj.bias",
        "attn_v.bias" => "self_attn.v_proj.bias",
        "attn_output.bias" => "self_attn.o_proj.bias",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    };
    Some(format!("model.layers.{layer}.{hf}"))
}

/// Tensors some converters emit that the engine deliberately does not map
/// (suppressed from the unmapped-tensor warning).
const KNOWN_UNMAPPED: &[&str] = &[
    // llama-3.1+ long-rope frequency table; scaling comes from metadata
    // keys when present.
    "rope_freqs.weight",
];

/// Fused ggml tensors served through `fused_slice` instead of the name map
/// (phi3's `attn_qkv`; its fused `ffn_up` also feeds gate/up slices but
/// maps directly as `up_proj`). Consumed, so not "unmapped".
fn is_fused_source(ggml: &str, arch: &str) -> bool {
    arch == "phi3"
        && ggml
            .strip_prefix("blk.")
            .and_then(|r| r.split_once('.'))
            .is_some_and(|(_, rest)| rest == "attn_qkv.weight" || rest == "attn_qkv.bias")
}

/// Number of elements in a ggml tensor.
fn num_elements(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// Byte size of a ggml tensor's data.
fn tensor_byte_size(info: &TensorInfo) -> Result<usize> {
    let n = num_elements(&info.dims);
    let block = |bs: usize| -> Result<usize> {
        if n % 32 != 0 {
            return Err(FormatError::Safetensors(format!(
                "gguf tensor {}: {n} elements not divisible by block size 32",
                info.name
            )));
        }
        Ok(n / 32 * bs)
    };
    let superblock = |bs: usize| -> Result<usize> {
        if n % 256 != 0 {
            return Err(FormatError::Safetensors(format!(
                "gguf tensor {}: {n} elements not divisible by K-quant superblock 256",
                info.name
            )));
        }
        Ok(n / 256 * bs)
    };
    Ok(match info.ggml_type {
        GGML_F32 => n * 4,
        GGML_F16 | GGML_BF16 => n * 2,
        GGML_Q4_0 => block(18)?, // 2-byte scale + 16 bytes per 32 values
        GGML_Q4_1 => block(20)?, // f16 scale + f16 min + 16 bytes
        GGML_Q5_0 => block(22)?, // f16 scale + u32 high bits + 16 bytes
        GGML_Q5_1 => block(24)?, // f16 scale + f16 min + u32 high bits + 16 bytes
        GGML_Q8_0 => block(34)?, // 2-byte scale + 32 int8 per 32 values
        // K-quants: 256-value superblocks.
        GGML_Q4_K => superblock(144)?, // 2×f16 + 12B packed scales/mins + 128B quants
        GGML_Q5_K => superblock(176)?, // + 32B high bits
        GGML_Q6_K => superblock(210)?, // 128B ql + 64B qh + 16×i8 scales + f16
        other => {
            return Err(FormatError::UnsupportedDtype {
                tensor: info.name.clone(),
                dtype: format!("ggml_type {other}"),
            });
        }
    })
}

/// Dequantizes a Q4_0 tensor to f32. Blocks of 32 values: f16 scale + 16
/// bytes; value i (i<16) is the low nibble of byte i, value i+16 the high.
///
/// Public via [`crate::quants`]: this scalar path is the harmony reference
/// the fused CubeCL kernels validate against.
pub fn dequantize_q4_0(data: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut fixed = Vec::with_capacity(n);
    for block in data.chunks_exact(18) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let mut vals = [0f32; 32];
        for j in 0..16 {
            let byte = block[2 + j];
            vals[j] = ((byte & 0x0F) as i32 - 8) as f32 * d;
            vals[j + 16] = ((byte >> 4) as i32 - 8) as f32 * d;
        }
        fixed.extend_from_slice(&vals);
    }
    if fixed.len() != n {
        return Err(FormatError::Safetensors(format!(
            "q4_0 dequant size mismatch: {} != {n}",
            fixed.len()
        )));
    }
    Ok(fixed)
}

/// Dequantizes a Q8_0 tensor to f32.
///
/// Public via [`crate::quants`] as the harmony reference for the GPU kernel.
/// Quantizes f32 values into Q8_0 blocks (per-32 f16 scale + 32 i8),
/// ggml's own rounding: `d = amax/127`, `q = round(x/d)`. The inverse
/// of [`dequantize_q8_0`]; used to pack float checkpoints onto the
/// quant kernels at load time. `values.len()` must be a multiple of 32.
pub fn quantize_q8_0(values: &[f32]) -> Result<Vec<u8>> {
    if values.len() % 32 != 0 {
        return Err(FormatError::Safetensors(format!(
            "q8_0 quantize: {} values is not a multiple of 32",
            values.len()
        )));
    }
    let mut out = Vec::with_capacity(values.len() / 32 * 34);
    for block in values.chunks_exact(32) {
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for &v in block {
            out.push((v * inv).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
    Ok(out)
}

pub fn dequantize_q8_0(data: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for block in data.chunks_exact(34) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        for &b in &block[2..34] {
            out.push((b as i8) as f32 * d);
        }
    }
    if out.len() != n {
        return Err(FormatError::Safetensors(format!(
            "q8_0 dequant size mismatch: {} != {n}",
            out.len()
        )));
    }
    Ok(out)
}

/// Dequantizes a Q4_1 tensor to f32. Like Q4_0 but asymmetric:
/// value = q·d + m (f16 scale + f16 min per 32-value block).
fn dequantize_q4_1(data: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for block in data.chunks_exact(20) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let mut vals = [0f32; 32];
        for j in 0..16 {
            let byte = block[4 + j];
            vals[j] = (byte & 0x0F) as f32 * d + m;
            vals[j + 16] = (byte >> 4) as f32 * d + m;
        }
        out.extend_from_slice(&vals);
    }
    if out.len() != n {
        return Err(FormatError::Safetensors(format!(
            "q4_1 dequant size mismatch: {} != {n}",
            out.len()
        )));
    }
    Ok(out)
}

/// Dequantizes a Q5_0 tensor to f32. 32-value blocks (22 bytes): f16
/// scale, u32 high bits (bit j → value j, bit j+16 → value j+16), 16
/// packed nibble bytes; values biased by -16.
///
/// Public via [`crate::quants`] as the harmony reference for the GPU kernel.
pub fn dequantize_q5_0(data: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for block in data.chunks_exact(22) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let mut vals = [0f32; 32];
        for j in 0..16 {
            let byte = block[6 + j];
            let lo = ((byte & 0x0F) as u32 | (((qh >> j) & 1) << 4)) as i32 - 16;
            let hi = ((byte >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4)) as i32 - 16;
            vals[j] = lo as f32 * d;
            vals[j + 16] = hi as f32 * d;
        }
        out.extend_from_slice(&vals);
    }
    if out.len() != n {
        return Err(FormatError::Safetensors(format!(
            "q5_0 dequant size mismatch: {} != {n}",
            out.len()
        )));
    }
    Ok(out)
}

/// Dequantizes a Q5_1 tensor to f32. Like Q5_0 but asymmetric (scale +
/// min, no bias): value = q·d + m. 24-byte blocks.
fn dequantize_q5_1(data: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for block in data.chunks_exact(24) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let m = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let mut vals = [0f32; 32];
        for j in 0..16 {
            let byte = block[8 + j];
            let lo = (byte & 0x0F) as u32 | (((qh >> j) & 1) << 4);
            let hi = (byte >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4);
            vals[j] = lo as f32 * d + m;
            vals[j + 16] = hi as f32 * d + m;
        }
        out.extend_from_slice(&vals);
    }
    if out.len() != n {
        return Err(FormatError::Safetensors(format!(
            "q5_1 dequant size mismatch: {} != {n}",
            out.len()
        )));
    }
    Ok(out)
}

/// K-quant 6-bit scale/min unpacking (ggml `get_scale_min_k4`): the 12
/// scale bytes pack eight 6-bit scales and eight 6-bit mins, with the top
/// 2 bits of bytes 0..8 carrying the high bits of sub-blocks 4..8.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantizes a Q4_K tensor to f32. 256-value superblocks (144 bytes):
/// f16 d, f16 dmin, 12B packed scales/mins, 128B 4-bit quants.
///
/// Public via [`crate::quants`] as the harmony reference for the GPU kernel.
pub fn dequantize_q4_k(data: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for sb in data.chunks_exact(144) {
        let d = half::f16::from_le_bytes([sb[0], sb[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([sb[2], sb[3]]).to_f32();
        let scales = &sb[4..16];
        let qs = &sb[16..144];
        for j in 0..4 {
            // Each 64-value group uses 32 qs bytes; low nibbles → first
            // sub-block, high nibbles → second.
            let (sc1, m1) = scale_min_k4(2 * j, scales);
            let (sc2, m2) = scale_min_k4(2 * j + 1, scales);
            let (d1, fmin1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, fmin2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                let byte = qs[32 * j + l];
                out.push(d1 * (byte & 0x0F) as f32 - fmin1);
            }
            for l in 0..32 {
                let byte = qs[32 * j + l];
                out.push(d2 * (byte >> 4) as f32 - fmin2);
            }
        }
    }
    if out.len() != n {
        return Err(FormatError::Safetensors(format!(
            "q4_k dequant size mismatch: {} != {n}",
            out.len()
        )));
    }
    Ok(out)
}

/// Dequantizes a Q5_K tensor to f32. Like Q4_K plus 32 high-bit bytes:
/// group j reads bit 2j (low sub-block) / 2j+1 (high sub-block) of qh.
pub fn dequantize_q5_k(data: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for sb in data.chunks_exact(176) {
        let d = half::f16::from_le_bytes([sb[0], sb[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([sb[2], sb[3]]).to_f32();
        let scales = &sb[4..16];
        let qh = &sb[16..48];
        let qs = &sb[48..176];
        for j in 0..4 {
            let (sc1, m1) = scale_min_k4(2 * j, scales);
            let (sc2, m2) = scale_min_k4(2 * j + 1, scales);
            let (d1, fmin1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, fmin2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                let lo = (qs[32 * j + l] & 0x0F) as u32;
                let hi = ((qh[l] >> (2 * j)) & 1) as u32;
                out.push(d1 * ((lo | (hi << 4)) as f32) - fmin1);
            }
            for l in 0..32 {
                let lo = (qs[32 * j + l] >> 4) as u32;
                let hi = ((qh[l] >> (2 * j + 1)) & 1) as u32;
                out.push(d2 * ((lo | (hi << 4)) as f32) - fmin2);
            }
        }
    }
    if out.len() != n {
        return Err(FormatError::Safetensors(format!(
            "q5_k dequant size mismatch: {} != {n}",
            out.len()
        )));
    }
    Ok(out)
}

/// Dequantizes a Q6_K tensor to f32. 256-value superblocks (210 bytes):
/// 128B low nibbles, 64B high 2-bit fields, 16 i8 scales (one per 16
/// values), f16 super-scale. Values are biased by -32.
///
/// Public via [`crate::quants`] as the harmony reference for the GPU kernel.
pub fn dequantize_q6_k(data: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    for sb in data.chunks_exact(210) {
        let d = half::f16::from_le_bytes([sb[208], sb[209]]).to_f32();
        // Process the superblock as two 128-value halves.
        for half_idx in 0..2 {
            let ql = &sb[64 * half_idx..64 * half_idx + 64];
            let qh = &sb[128 + 32 * half_idx..128 + 32 * half_idx + 32];
            let scales = &sb[192 + 8 * half_idx..192 + 8 * half_idx + 8];
            let mut vals = [0f32; 128];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 0x03) << 4)) as i8 as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | ((qh[l] & 0x0C) >> 2 << 4)) as i8 as i32 - 32;
                let q3 = ((ql[l] >> 4) | ((qh[l] & 0x30) >> 4 << 4)) as i8 as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | ((qh[l] & 0xC0) >> 6 << 4)) as i8 as i32 - 32;
                vals[l] = d * (scales[is] as i8) as f32 * q1 as f32;
                vals[l + 32] = d * (scales[is + 2] as i8) as f32 * q2 as f32;
                vals[l + 64] = d * (scales[is + 4] as i8) as f32 * q3 as f32;
                vals[l + 96] = d * (scales[is + 6] as i8) as f32 * q4 as f32;
            }
            out.extend_from_slice(&vals);
        }
    }
    if out.len() != n {
        return Err(FormatError::Safetensors(format!(
            "q6_k dequant size mismatch: {} != {n}",
            out.len()
        )));
    }
    Ok(out)
}

/// llama.cpp's `convert_hf_to_gguf` permutes each attention head's
/// `attn_q`/`attn_k` rows for llama-family architectures — HF rotate-half
/// halves `[0..d/2 | d/2..d]` become ggml interleaved pairs
/// `[0, d/2, 1, d/2+1, …]`. This engine applies HF rotate-half RoPE, so the
/// rows must be de-interleaved back on load (transformers'
/// `_reverse_permute_weights`). Returns, for each HF row, the source ggml
/// row: `hf[j] = ggml[2j]` for `j < d/2`, else `ggml[2(j − d/2) + 1]`,
/// per head.
fn rope_depermute_src_rows(rows: usize, n_head: usize) -> Vec<usize> {
    let d = rows / n_head;
    let half = d / 2;
    let mut map = Vec::with_capacity(rows);
    for h in 0..n_head {
        let base = h * d;
        for j in 0..d {
            let src = if j < half { 2 * j } else { 2 * (j - half) + 1 };
            map.push(base + src);
        }
    }
    map
}

/// Reorders `values` (f32, `rows` equal-length rows) by the de-permute map.
fn depermute_rows_f32(values: Vec<f32>, rows: usize, n_head: usize) -> Vec<f32> {
    let row_len = values.len() / rows;
    let map = rope_depermute_src_rows(rows, n_head);
    let mut out = Vec::with_capacity(values.len());
    for src in map {
        out.extend_from_slice(&values[src * row_len..(src + 1) * row_len]);
    }
    out
}

impl GgufSource {
    /// Fused-projection resolution: phi3 GGUFs store `attn_qkv` = `[q|k|v]`
    /// rows and `ffn_up` = `[gate|up]` rows (HF phi3 order). Given a split
    /// HF name, returns the fused ggml name plus the `(start, len)` row
    /// range to slice. Row slicing is exact for every supported dtype:
    /// packed formats store whole blocks per row (the column count is a
    /// multiple of the 256-value superblock), so a row range is a
    /// contiguous byte range.
    fn fused_slice(&self, name: &str) -> Option<(String, usize, usize)> {
        if self.metadata.architecture != "phi3" {
            return None;
        }
        let m = &self.metadata;
        let rest = name.strip_prefix("model.layers.")?;
        let (layer, rest) = rest.split_once('.')?;
        let q_rows = m.num_attention_heads * m.head_dim;
        let kv_rows = m.num_key_value_heads * m.head_dim;
        let ffn = m.intermediate_size;
        let (fused, start, len) = match rest {
            "self_attn.q_proj.weight" => ("attn_qkv.weight", 0, q_rows),
            "self_attn.k_proj.weight" => ("attn_qkv.weight", q_rows, kv_rows),
            "self_attn.v_proj.weight" => ("attn_qkv.weight", q_rows + kv_rows, kv_rows),
            "mlp.gate_proj.weight" => ("ffn_up.weight", 0, ffn),
            "mlp.up_proj.weight" => ("ffn_up.weight", ffn, ffn),
            _ => return None,
        };
        Some((format!("blk.{layer}.{fused}"), start, len))
    }

    /// The payload bytes of one tensor. The single place a tensor's
    /// byte range is turned into bytes, so the residency rule lives in
    /// one spot: a windowed image answers only for what it still holds,
    /// and a tensor asked for out of order fails here with its name on
    /// it rather than somewhere downstream with a shape mismatch.
    fn tensor_bytes(&self, info: &TensorInfo) -> Result<std::borrow::Cow<'_, [u8]>> {
        let size = tensor_byte_size(info)? as u64;
        let start = self.data_start + info.offset;
        self.data.slice(start, size).ok_or_else(|| {
            FormatError::Safetensors(format!(
                "gguf tensor {}: bytes {}..{} are not available{}",
                info.name,
                start,
                start + size,
                match &self.data {
                    GgufData::Window { base, buf, .. } => format!(
                        " — the mount window holds {}..{}",
                        base,
                        base + buf.len() as u64
                    ),
                    _ => " (out of bounds)".to_string(),
                }
            ))
        })
    }

    /// Which HF tensors this ggml tensor supplies — [`resolve_tensor`]
    /// read backwards. A load driven by the file rather than by the
    /// model walks tensors in the order the bytes arrive, so it needs
    /// to go from the name on the wire to the names the model asked
    /// for; a fused projection answers with all of its parts, each with
    /// its row range. An empty result means nothing in the model wants
    /// these bytes, which is a fact to act on and not an error.
    pub fn hf_names_for_ggml(&self, ggml: &str) -> Vec<(String, Option<(usize, usize)>)> {
        // Fused projections are asked about FIRST, and the order is
        // load-bearing rather than tidy. On phi3 the fused gate+up
        // tensor is named `ffn_up.weight`, which the forward map — which
        // knows nothing of architectures here — happily reads as the
        // whole `mlp.up_proj.weight`. `resolve_tensor` gets the right
        // answer only because it tries the splitter first, so this must
        // too, or the two describe different files.
        let fused = self.fused_parts(ggml);
        if !fused.is_empty() {
            return fused;
        }
        let arch = self.metadata.architecture.as_str();
        if let Some(direct) = map_tensor_name(ggml, arch) {
            return vec![(direct, None)];
        }
        Vec::new()
    }

    /// The HF names living inside `ggml` when it is a fused projection.
    /// Candidates come from `fused_slice` itself — asking each name the
    /// splitter knows whether it lands here — so the two cannot drift.
    fn fused_parts(&self, ggml: &str) -> Vec<(String, Option<(usize, usize)>)> {
        let Some(rest) = ggml.strip_prefix("blk.") else {
            return Vec::new();
        };
        let Some((layer, _)) = rest.split_once('.') else {
            return Vec::new();
        };
        const FUSED_PARTS: [&str; 5] = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
        ];
        let mut out: Vec<(String, Option<(usize, usize)>)> = FUSED_PARTS
            .iter()
            .filter_map(|part| {
                let hf = format!("model.layers.{layer}.{part}");
                let (fused, start, len) = self.fused_slice(&hf)?;
                (fused == ggml).then_some((hf, Some((start, len))))
            })
            .collect();
        out.sort_by_key(|(_, range)| range.map(|(start, _)| start).unwrap_or(0));
        out
    }

    /// Looks up the ggml tensor serving HF `name`, with the row range to
    /// slice when it lives inside a fused projection (`None` range = whole
    /// tensor).
    fn resolve_tensor(&self, name: &str) -> Option<(&String, &TensorInfo, Option<(usize, usize)>)> {
        if let Some((fused, start, len)) = self.fused_slice(name) {
            if let Some((k, info)) = self.tensors.get_key_value(&fused) {
                return Some((k, info, Some((start, len))));
            }
        }
        let arch = self.metadata.architecture.as_str();
        self.tensors
            .iter()
            .find(|(k, _)| map_tensor_name(k, arch).as_deref() == Some(name))
            .map(|(k, info)| (k, info, None))
    }

    /// Head count to de-permute `ggml_name` with, when this file's arch
    /// stores Q/K in llama.cpp's interleaved-pairs layout. `None` = serve
    /// the tensor verbatim.
    fn depermute_heads(&self, ggml_name: &str) -> Option<usize> {
        if !matches!(self.metadata.architecture.as_str(), "llama" | "mistral") {
            return None;
        }
        let rest = ggml_name.strip_prefix("blk.")?;
        let (_, rest) = rest.split_once('.')?;
        match rest {
            "attn_q.weight" | "attn_q.bias" => Some(self.metadata.num_attention_heads),
            "attn_k.weight" | "attn_k.bias" => Some(self.metadata.num_key_value_heads),
            _ => None,
        }
    }
}

impl ModelSource for GgufSource {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn tensor_names(&self) -> Vec<String> {
        let arch = self.metadata.architecture.as_str();
        self.tensors
            .keys()
            .filter_map(|k| map_tensor_name(k, arch))
            .collect()
    }

    fn open_tensor(&self, name: &str) -> Result<TensorReader<'_>> {
        // Find the ggml tensor mapping to this HF name (possibly a row
        // range of a fused projection).
        let (ggml_name, info, slice) = self
            .resolve_tensor(name)
            .ok_or_else(|| FormatError::TensorNotFound(name.to_string()))?;

        let size = tensor_byte_size(info)?;
        let data = self.tensor_bytes(info)?;

        // HF layout = ggml dims reversed (row-major data needs no movement).
        let mut shape: Vec<usize> = info.dims.iter().rev().copied().collect();
        let data = match slice {
            None => data,
            Some((row_start, row_len)) => {
                let rows_total = shape.first().copied().unwrap_or(1).max(1);
                if size % rows_total != 0 {
                    return Err(FormatError::Safetensors(format!(
                        "gguf tensor {}: rows not byte-addressable for fused split",
                        info.name
                    )));
                }
                let row_bytes = size / rows_total;
                shape[0] = row_len;
                narrow_cow(data, row_start * row_bytes, row_len * row_bytes)
            }
        };
        let data_ref: &[u8] = &data;
        let n: usize = shape.iter().product();
        let rows = shape.first().copied().unwrap_or(1).max(1);
        // Fused slices and RoPE de-permutation never co-occur (phi3 vs
        // llama/mistral arch gates).
        let permute = self.depermute_heads(ggml_name);
        // llama.cpp's gemma converters bake the `(1+w)` into stored norm
        // weights (their graph applies plain x̂·w); the engine keeps HF
        // semantics (x̂·(1+w) via the gemma norm flavor), so the +1 is
        // removed here — the same normalize-at-the-adapter rule as the
        // RoPE de-permutation. Verified: gemma-3-1b GGUF norms are exactly
        // safetensors + 1.0.
        let gemma_norm_offset = matches!(
            self.metadata.architecture.as_str(),
            "gemma3" | "gemma3_text"
        ) && ggml_name.ends_with("norm.weight");

        // Passthrough dtypes: serve the mmap slice, unless the rows must be
        // de-interleaved (RoPE de-permutation) — then copy row-reordered.
        if gemma_norm_offset {
            if info.ggml_type != GGML_F32 {
                return Err(FormatError::UnsupportedDtype {
                    tensor: info.name.clone(),
                    dtype: format!("gemma norm must be F32, got ggml_type {}", info.ggml_type),
                });
            }
            let values: Vec<f32> = data_ref
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()) - 1.0)
                .collect();
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            return Ok(TensorReader::owned(name.to_string(), shape, bytes));
        }
        if let GGML_F32 | GGML_F16 | GGML_BF16 = info.ggml_type {
            let dtype = match info.ggml_type {
                GGML_F32 => TensorDtype::F32,
                GGML_F16 => TensorDtype::F16,
                _ => TensorDtype::BF16,
            };
            return Ok(match permute {
                None => match data {
                    std::borrow::Cow::Borrowed(b) => {
                        TensorReader::new(name.to_string(), shape, dtype, b)
                    }
                    std::borrow::Cow::Owned(v) => {
                        TensorReader::owned_with_dtype(name.to_string(), shape, dtype, v)
                    }
                },
                Some(n_head) => {
                    let row_bytes = size / rows;
                    let map = rope_depermute_src_rows(rows, n_head);
                    let mut out = Vec::with_capacity(size);
                    for src in map {
                        out.extend_from_slice(&data_ref[src * row_bytes..(src + 1) * row_bytes]);
                    }
                    TensorReader::owned_with_dtype(name.to_string(), shape, dtype, out)
                }
            });
        }

        let values = match info.ggml_type {
            GGML_Q4_0 => dequantize_q4_0(data_ref, n)?,
            GGML_Q4_1 => dequantize_q4_1(data_ref, n)?,
            GGML_Q5_0 => dequantize_q5_0(data_ref, n)?,
            GGML_Q5_1 => dequantize_q5_1(data_ref, n)?,
            GGML_Q8_0 => dequantize_q8_0(data_ref, n)?,
            GGML_Q4_K => dequantize_q4_k(data_ref, n)?,
            GGML_Q5_K => dequantize_q5_k(data_ref, n)?,
            GGML_Q6_K => dequantize_q6_k(data_ref, n)?,
            other => {
                return Err(FormatError::UnsupportedDtype {
                    tensor: info.name.clone(),
                    dtype: format!("ggml_type {other}"),
                });
            }
        };
        let values = match permute {
            Some(n_head) => depermute_rows_f32(values, rows, n_head),
            None => values,
        };
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(TensorReader::owned(name.to_string(), shape, bytes))
    }

    fn tokenizer(&self) -> Result<TokenizerSpec> {
        let add_bos = match self.kv.get("tokenizer.ggml.add_bos_token") {
            Some(MetaValue::Bool(b)) => Some(*b),
            _ => None,
        };
        let chat_template = match self.kv.get("tokenizer.chat_template") {
            Some(MetaValue::String(t)) => Some(t.clone()),
            _ => None,
        };
        Ok(TokenizerSpec {
            tokenizer: self.tokenizer.clone(),
            added_tokens: self.added_tokens.clone(),
            chat_template,
            add_bos,
        })
    }

    fn open_tensor_quant(&self, name: &str) -> Result<Option<crate::QuantTensor<'_>>> {
        let Some((ggml_name, info, slice)) = self.resolve_tensor(name) else {
            return Ok(None);
        };
        let format = match info.ggml_type {
            GGML_Q4_0 => crate::QuantFormat::Q4_0,
            GGML_Q5_0 => crate::QuantFormat::Q5_0,
            GGML_Q8_0 => crate::QuantFormat::Q8_0,
            GGML_Q4_K => crate::QuantFormat::Q4K,
            GGML_Q5_K => crate::QuantFormat::Q5K,
            GGML_Q6_K => crate::QuantFormat::Q6K,
            _ => return Ok(None),
        };
        let size = tensor_byte_size(info)?;
        let data = self.tensor_bytes(info)?;
        let mut shape: Vec<usize> = info.dims.iter().rev().copied().collect();
        // Row range of a fused projection: a contiguous packed byte range
        // (whole blocks per row), so the payload narrows in place.
        let data = match slice {
            None => data,
            Some((row_start, row_len)) => {
                let rows_total = shape.first().copied().unwrap_or(1).max(1);
                if size % rows_total != 0 {
                    return Ok(None);
                }
                let row_bytes = size / rows_total;
                shape[0] = row_len;
                narrow_cow(data, row_start * row_bytes, row_len * row_bytes)
            }
        };

        // RoPE de-permutation for packed weights: every supported block
        // format stores whole blocks per row, so reordering the packed
        // stream row-chunk-wise is exact. If the packed rows aren't
        // cleanly addressable, fall back to the dense path (which
        // de-permutes after dequantization).
        let data = match self.depermute_heads(ggml_name) {
            None => data,
            Some(n_head) => {
                let rows = shape.first().copied().unwrap_or(1).max(1);
                if size % rows != 0 {
                    return Ok(None);
                }
                let row_bytes = size / rows;
                let map = rope_depermute_src_rows(rows, n_head);
                let mut out = Vec::with_capacity(size);
                for src in map {
                    out.extend_from_slice(&data[src * row_bytes..(src + 1) * row_bytes]);
                }
                std::borrow::Cow::Owned(out)
            }
        };
        Ok(Some(crate::QuantTensor { format, shape, data }))
    }

    fn sampler_defaults(&self) -> Option<SamplerConfig> {
        None
    }
}

impl GgufSource {
    /// End-of-sequence ids from tokenizer metadata.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    /// Model file path, when this source was read from a file.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
}

#[cfg(test)]
mod depermute_tests {
    use super::rope_depermute_src_rows;

    /// llama.cpp's forward permute: per head, rows [2, d/2] -> swap -> flat,
    /// i.e. ggml[2i] = hf[i], ggml[2i+1] = hf[d/2 + i].
    fn forward_permute(rows: &[Vec<u32>], n_head: usize) -> Vec<Vec<u32>> {
        let d = rows.len() / n_head;
        let mut out = Vec::with_capacity(rows.len());
        for h in 0..n_head {
            let head = &rows[h * d..(h + 1) * d];
            for i in 0..d / 2 {
                out.push(head[i].clone());
                out.push(head[d / 2 + i].clone());
            }
        }
        out
    }

    #[test]
    fn depermute_inverts_llama_cpp_permute() {
        // 2 heads x head_dim 8 = 16 rows, each row tagged by its HF index.
        let hf: Vec<Vec<u32>> = (0..16).map(|i| vec![i, 100 + i]).collect();
        let ggml = forward_permute(&hf, 2);
        assert_ne!(hf, ggml, "permute must actually move rows");
        let map = rope_depermute_src_rows(16, 2);
        let recovered: Vec<Vec<u32>> = map.iter().map(|&src| ggml[src].clone()).collect();
        assert_eq!(recovered, hf, "de-permute must invert llama.cpp's layout");
    }
}

#[cfg(test)]
mod wide_offset_tests {
    use super::*;

    /// A header describing one tensor at `offset`, `elements` f32 long.
    fn header_with_tensor_at(offset: u64, elements: u64) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        out.extend_from_slice(&3u32.to_le_bytes()); // version
        out.extend_from_slice(&1u64.to_le_bytes()); // one tensor
        out.extend_from_slice(&0u64.to_le_bytes()); // no metadata
        let name = b"output.weight";
        out.extend_from_slice(&(name.len() as u64).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&1u32.to_le_bytes()); // one dimension
        out.extend_from_slice(&elements.to_le_bytes());
        out.extend_from_slice(&GGML_F32.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        // Padded to the 32-byte boundary: a header ends at `data_start`,
        // not at the last info byte.
        out.resize(out.len().div_ceil(32) * 32, 0);
        out
    }

    /// A tensor's end must be computed in a width that can hold it.
    ///
    /// The numbers are the ones that failed: a 447,068,160-byte tensor at
    /// offset 4,236,006,176 in a 4.68 GB checkpoint. Their sum is
    /// 4,683,074,336, which is past `u32::MAX` — and `usize` is 32 bits
    /// on wasm32, the one target where a model that large is the entire
    /// point. The wrap turned an end into something smaller than its own
    /// start, and the reader reported a range it could not possibly
    /// serve. Nothing under 4 GB was reachable before per-tensor
    /// streaming, which is why this waited to be found.
    ///
    /// On a 64-bit host this passes either way; its job is to state the
    /// requirement in the numbers, next to the reader it constrains.
    #[test]
    fn a_tensor_past_four_gigabytes_keeps_its_place() {
        const OFFSET: u64 = 4_236_006_176;
        const SIZE: u64 = 447_068_160;
        let bytes = header_with_tensor_at(OFFSET, SIZE / 4);
        let total = OFFSET + SIZE + 4096;
        let header = read_gguf_header(&bytes, Some(total))
            .expect("parses")
            .expect("the whole header is here");

        let (name, start, size) = &header.tensors[0];
        assert_eq!(name, "output.weight");
        assert_eq!(*size, SIZE);
        assert_eq!(*start, header.data_start + OFFSET);
        let end = start + size;
        assert!(
            end > u32::MAX as u64,
            "the case only bites past 4 GB; this one ends at {end}"
        );
        assert!(end > *start, "a tensor's end came out before its start");
    }
}

#[cfg(test)]
mod window_tests {
    use super::*;
    use crate::ModelSource;

    fn cached(dir: &str) -> Option<std::path::PathBuf> {
        let path = std::path::PathBuf::from(std::env::var("HOME").ok()?)
            .join(".cache/combs/models")
            .join(dir)
            .join("model.gguf");
        path.exists().then_some(path)
    }

    /// One per architecture that makes the loader do something special:
    /// llama de-permutes Q/K, qwen3 carries per-head norms and an untied
    /// head, gemma3 renames three norms, phi3 keeps Q/K/V and gate/up
    /// FUSED and has to be sliced back apart.
    const ARCHS: [&str; 4] = [
        "smollm2-360m-instruct-gguf",
        "qwen3-0.6b-gguf",
        "gemma-3-1b-it-gguf",
        "phi-3.1-mini-4k-instruct-gguf",
    ];

    /// The keystone. A source holding ONLY one tensor's bytes must
    /// answer for that tensor exactly what the whole file answers —
    /// same format, same shape, same bytes, through both the quantized
    /// and the dense reader. If this holds for every tensor of every
    /// architecture, then a mount never needs the file entire, which is
    /// the whole of the streaming design resting on one property.
    #[test]
    fn a_window_of_one_tensor_answers_like_the_whole_file() {
        let mut ran = 0;
        for dir in ARCHS {
            let Some(path) = cached(dir) else {
                eprintln!("skip {dir}: not in the local cache");
                continue;
            };
            let full = GgufSource::load(&path).unwrap();
            let total = full.data.len();
            let header_bytes = full
                .data
                .slice(0, full.data_start)
                .expect("the header is always resident")
                .into_owned();

            let ggml_names: Vec<String> = full.tensors.keys().cloned().collect();
            let mut checked = 0usize;
            for ggml in ggml_names {
                let info = &full.tensors[&ggml];
                let Ok(size) = tensor_byte_size(info).map(|n| n as u64) else { continue };
                let start = full.data_start + info.offset;
                let Some(window) = full.data.slice(start, size) else { continue };
                let win =
                    GgufSource::from_window(&header_bytes, window.into_owned(), start, total)
                        .unwrap();

                let views = full.hf_names_for_ggml(&ggml);
                for (hf, _range) in &views {
                    match (
                        full.open_tensor_quant(hf).unwrap(),
                        win.open_tensor_quant(hf).unwrap(),
                    ) {
                        (Some(a), Some(b)) => {
                            assert_eq!(a.format, b.format, "{dir} {ggml} -> {hf}: format");
                            assert_eq!(a.shape, b.shape, "{dir} {ggml} -> {hf}: shape");
                            assert_eq!(&*a.data, &*b.data, "{dir} {ggml} -> {hf}: packed bytes");
                        }
                        (None, None) => {}
                        (a, b) => panic!(
                            "{dir} {ggml} -> {hf}: quant path disagrees on whether it applies \
                             (whole {}, window {})",
                            a.is_some(),
                            b.is_some()
                        ),
                    }
                    let a = full.open_tensor(hf).unwrap().load_data().unwrap();
                    let b = win.open_tensor(hf).unwrap().load_data().unwrap();
                    assert_eq!(a.shape, b.shape, "{dir} {ggml} -> {hf}: dense shape");
                    assert_eq!(
                        a.to_vec::<f32>().unwrap(),
                        b.to_vec::<f32>().unwrap(),
                        "{dir} {ggml} -> {hf}: dense values"
                    );
                    checked += 1;
                }
            }
            assert!(checked > 0, "{dir}: the reverse map named no tensors at all");
            eprintln!("{dir}: {checked} tensor views identical through a one-tensor window");
            ran += 1;
        }
        assert!(ran > 0, "no architecture was available to check");
    }

    /// Drift guard. The reverse map and the forward lookup are two
    /// descriptions of one relation, written apart and edited apart, so
    /// they are pinned to each other: every name the reverse map emits
    /// must resolve back to the tensor it came from, with the same row
    /// range, and every name the model can ask for must be produced by
    /// exactly one tensor. A rename on one side and not the other stops
    /// here rather than in a model that loads and answers wrongly.
    #[test]
    fn the_reverse_map_and_the_forward_lookup_agree() {
        let mut ran = 0;
        for dir in ARCHS {
            let Some(path) = cached(dir) else { continue };
            let src = GgufSource::load(&path).unwrap();

            let mut producers: HashMap<String, Vec<String>> = HashMap::new();
            for ggml in src.tensors.keys() {
                for (hf, range) in src.hf_names_for_ggml(ggml) {
                    let (back, _, back_range) = src
                        .resolve_tensor(&hf)
                        .unwrap_or_else(|| panic!("{dir}: {ggml} -> {hf} resolves to nothing"));
                    assert_eq!(back, ggml, "{dir}: {hf} came from {ggml} but resolves to {back}");
                    assert_eq!(back_range, range, "{dir}: {hf} row range disagrees");
                    producers.entry(hf).or_default().push(ggml.clone());
                }
            }
            for hf in src.tensor_names() {
                let from = producers.get(&hf);
                assert!(
                    from.map(|v| v.len()) == Some(1),
                    "{dir}: {hf} is asked for by the model but produced by {:?}",
                    from
                );
            }
            if src.metadata.architecture == "phi3" {
                // The case the map exists for: fused projections have no
                // forward mapping at all, so if the reverse map ever
                // stopped emitting them nothing else would notice.
                let fused: Vec<_> = src
                    .tensors
                    .keys()
                    .filter(|k| k.ends_with("attn_qkv.weight"))
                    .collect();
                assert!(!fused.is_empty(), "phi3 file without a fused qkv");
                for f in fused {
                    let parts = src.hf_names_for_ggml(f);
                    assert_eq!(parts.len(), 3, "{f}: expected q, k and v, got {parts:?}");
                }
            }
            ran += 1;
        }
        assert!(ran > 0, "no architecture was available to check");
    }
}

#[cfg(test)]
mod quant_access_tests {
    use super::*;
    use crate::ModelSource;

    /// Scratch diagnostic against a locally cached model; not part of CI.
    #[test]
    #[ignore]
    fn real_gguf_quant_access() {
        let path = dirs_home().join(".cache/combs/models/llama-3.2-1b-instruct-gguf/model.gguf");
        let src = GgufSource::load(&path).unwrap();
        for name in [
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ] {
            let qt = src.open_tensor_quant(name).unwrap();
            println!("{name}: {:?}", qt.map(|q| (q.format, q.shape, q.data.len())));
        }
    }

    fn dirs_home() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("HOME").unwrap())
    }
}
