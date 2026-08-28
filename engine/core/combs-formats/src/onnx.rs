//! Minimal ONNX container reader — a weights table, not a graph
//! runtime.
//!
//! ONNX is protobuf; this module walks the wire format with the shared
//! [`crate::protomin::Proto`] cursor (the same reader spm.rs and
//! tflite.rs use — no protobuf dependency) and keeps exactly what a
//! weights-container needs:
//! - every graph initializer: name, dtype, dims, and WHERE its bytes
//!   live — a range inside the model file (`raw_data`), or an
//!   (offset, length) window into a sibling external-data file, which
//!   the spec 4 KB-aligns precisely so readers can mmap or
//!   range-request it;
//! - the `MatMulNBits` nodes' quantization attributes (bits,
//!   block_size, K, N) and input names — the ONE place quant metadata
//!   lives as NODE attributes rather than tensor metadata;
//! - graph input/output names (KV-cache streams show up as
//!   `input.X`/`output.X` pairs and are skipped by weight loaders).
//!
//! Everything else in the file is skipped field-by-field. Unknown
//! wire types and truncations are hard errors, never silent.

use std::collections::HashMap;

use crate::protomin::Proto;
use crate::{FormatError, Result};

/// ONNX `TensorProto.DataType` values this reader understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxDtype {
    F32,
    F16,
    BF16,
    F64,
    I64,
    I32,
    I8,
    U8,
    /// 4-bit packed pairs (low nibble first), ONNX 1.16+.
    I4,
    U4,
    Bool,
}

impl OnnxDtype {
    fn from_code(code: u64) -> Option<Self> {
        Some(match code {
            1 => Self::F32,
            2 => Self::U8,
            3 => Self::I8,
            6 => Self::I32,
            7 => Self::I64,
            9 => Self::Bool,
            10 => Self::F16,
            11 => Self::F64,
            16 => Self::BF16,
            21 => Self::U4,
            22 => Self::I4,
            _ => return None,
        })
    }

    /// Bytes for `n` elements (4-bit types pack two per byte, rounded
    /// up — the ONNX convention).
    pub fn byte_len(&self, elements: u64) -> u64 {
        match self {
            Self::F64 | Self::I64 => elements * 8,
            Self::F32 | Self::I32 => elements * 4,
            Self::F16 | Self::BF16 => elements * 2,
            Self::I8 | Self::U8 | Self::Bool => elements,
            Self::I4 | Self::U4 => elements.div_ceil(2),
        }
    }
}

/// Where an initializer's bytes live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OnnxData {
    /// `raw_data` inside the .onnx file itself: a byte range of the
    /// parsed buffer.
    Inline { offset: usize, len: usize },
    /// External data file (relative `location`, byte `offset`, `length`).
    External { location: String, offset: u64, length: u64 },
}

#[derive(Debug, Clone)]
pub struct OnnxTensorInfo {
    pub name: String,
    pub dtype: OnnxDtype,
    pub dims: Vec<u64>,
    pub data: OnnxData,
}

impl OnnxTensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

/// One `com.microsoft.MatMulNBits` node: block-quantized matmul whose
/// packed weight `B` is `inputs[1]`, scales `inputs[2]`, optional
/// zero_points `inputs[3]`, optional g_idx `inputs[4]`.
#[derive(Debug, Clone)]
pub struct MatMulNBitsNode {
    pub name: String,
    pub inputs: Vec<String>,
    pub k: u64,
    pub n: u64,
    pub bits: u64,
    pub block_size: u64,
}

/// The parsed weights table of one .onnx file.
#[derive(Debug, Default)]
pub struct OnnxModel {
    pub tensors: HashMap<String, OnnxTensorInfo>,
    pub matmul_nbits: Vec<MatMulNBitsNode>,
    pub graph_inputs: Vec<String>,
    pub graph_outputs: Vec<String>,
}

fn bad(what: impl Into<String>) -> FormatError {
    FormatError::Safetensors(format!("onnx parse: {}", what.into()))
}

/// Byte offset of `slice` inside `base` — every sub-message the cursor
/// hands out borrows the one file image, so raw_data ranges recover
/// their absolute position from pointer distance.
fn offset_in(base: &[u8], slice: &[u8]) -> usize {
    slice.as_ptr() as usize - base.as_ptr() as usize
}

impl OnnxModel {
    /// Parse a whole .onnx file image.
    pub fn parse(buf: &[u8]) -> Result<Self> {
        let mut model = OnnxModel::default();
        let mut cur = Proto::new(buf);
        let mut saw_graph = false;
        while let Some((field, wire)) = cur.tag()? {
            match field {
                // ModelProto.graph
                7 if wire == 2 => {
                    model.parse_graph(buf, cur.bytes()?)?;
                    saw_graph = true;
                }
                _ => cur.skip(wire)?,
            }
        }
        if !saw_graph {
            return Err(bad("no graph in model (truncated or not an ONNX file)"));
        }
        Ok(model)
    }

    fn parse_graph(&mut self, base: &[u8], msg: &[u8]) -> Result<()> {
        let mut cur = Proto::new(msg);
        while let Some((field, wire)) = cur.tag()? {
            match field {
                // GraphProto.node
                1 if wire == 2 => self.parse_node(cur.bytes()?)?,
                // GraphProto.initializer
                5 if wire == 2 => {
                    let info = parse_tensor(base, cur.bytes()?)?;
                    self.tensors.insert(info.name.clone(), info);
                }
                // GraphProto.input / output (ValueInfoProto.name = field 1)
                11 if wire == 2 => {
                    if let Some(name) = value_info_name(cur.bytes()?)? {
                        self.graph_inputs.push(name);
                    }
                }
                12 if wire == 2 => {
                    if let Some(name) = value_info_name(cur.bytes()?)? {
                        self.graph_outputs.push(name);
                    }
                }
                _ => cur.skip(wire)?,
            }
        }
        Ok(())
    }

    fn parse_node(&mut self, msg: &[u8]) -> Result<()> {
        let mut cur = Proto::new(msg);
        let mut inputs = Vec::new();
        let mut name = String::new();
        let mut op_type = String::new();
        let mut attrs: Vec<(String, u64)> = Vec::new();
        while let Some((field, wire)) = cur.tag()? {
            match field {
                1 if wire == 2 => inputs.push(cur.string()?),
                3 if wire == 2 => name = cur.string()?,
                4 if wire == 2 => op_type = cur.string()?,
                5 if wire == 2 => {
                    if let Some(kv) = parse_int_attribute(cur.bytes()?)? {
                        attrs.push(kv);
                    }
                }
                _ => cur.skip(wire)?,
            }
        }
        if op_type == "MatMulNBits" {
            let get = |key: &str| attrs.iter().find(|(k, _)| k == key).map(|&(_, v)| v);
            let (Some(k), Some(n), Some(bits)) = (get("K"), get("N"), get("bits")) else {
                return Err(bad(format!(
                    "MatMulNBits node {name:?} lacks K/N/bits attributes"
                )));
            };
            self.matmul_nbits.push(MatMulNBitsNode {
                name,
                inputs,
                k,
                n,
                bits,
                block_size: get("block_size").unwrap_or(32),
            });
        }
        Ok(())
    }
}

/// AttributeProto: name (1), int value `i` (3). Non-int attributes
/// return None — the quant attrs are all ints.
fn parse_int_attribute(msg: &[u8]) -> Result<Option<(String, u64)>> {
    let mut cur = Proto::new(msg);
    let mut name = String::new();
    let mut value: Option<u64> = None;
    while let Some((field, wire)) = cur.tag()? {
        match field {
            1 if wire == 2 => name = cur.string()?,
            3 if wire == 0 => value = Some(cur.varint()?),
            _ => cur.skip(wire)?,
        }
    }
    Ok(value.map(|v| (name, v)))
}

fn value_info_name(msg: &[u8]) -> Result<Option<String>> {
    let mut cur = Proto::new(msg);
    while let Some((field, wire)) = cur.tag()? {
        match field {
            1 if wire == 2 => return Ok(Some(cur.string()?)),
            _ => cur.skip(wire)?,
        }
    }
    Ok(None)
}

/// TensorProto: dims (1), data_type (2), name (8), raw_data (9),
/// external_data (13, key/value entries), data_location (14).
fn parse_tensor(base: &[u8], msg: &[u8]) -> Result<OnnxTensorInfo> {
    let mut cur = Proto::new(msg);
    let mut dims = Vec::new();
    let mut dtype_code = 0u64;
    let mut name = String::new();
    let mut raw: Option<(usize, usize)> = None;
    let mut external: Vec<(String, String)> = Vec::new();
    let mut location_external = false;
    let mut typed_payload = false;
    while let Some((field, wire)) = cur.tag()? {
        match (field, wire) {
            (1, 0) => dims.push(cur.varint()?),
            (1, 2) => {
                // Packed repeated int64: raw back-to-back varints, no tags.
                let slice = cur.bytes()?;
                let mut p = 0usize;
                while p < slice.len() {
                    let mut v: u64 = 0;
                    let mut shift = 0u32;
                    loop {
                        let b = *slice.get(p).ok_or_else(|| bad("truncated packed dim"))?;
                        p += 1;
                        v |= u64::from(b & 0x7f) << shift;
                        if b & 0x80 == 0 {
                            break;
                        }
                        shift += 7;
                        if shift >= 64 {
                            return Err(bad("packed dim varint overflow"));
                        }
                    }
                    dims.push(v);
                }
            }
            (2, 0) => dtype_code = cur.varint()?,
            (8, 2) => name = cur.string()?,
            (9, 2) => {
                let slice = cur.bytes()?;
                raw = Some((offset_in(base, slice), slice.len()));
            }
            // float_data / int32_data / int64_data / double_data —
            // the non-raw encodings small exports sometimes carry.
            (4, _) | (5, _) | (7, _) | (10, _) => {
                typed_payload = true;
                cur.skip(wire)?;
            }
            (13, 2) => {
                let mut sub = Proto::new(cur.bytes()?);
                let mut key = String::new();
                let mut val = String::new();
                while let Some((f, w)) = sub.tag()? {
                    match f {
                        1 if w == 2 => key = sub.string()?,
                        2 if w == 2 => val = sub.string()?,
                        _ => sub.skip(w)?,
                    }
                }
                external.push((key, val));
            }
            (14, 0) => location_external = cur.varint()? == 1,
            _ => cur.skip(wire)?,
        }
    }
    let dtype = OnnxDtype::from_code(dtype_code).ok_or_else(|| FormatError::UnsupportedDtype {
        tensor: name.clone(),
        dtype: format!("onnx data_type {dtype_code}"),
    })?;
    let elements: u64 = dims.iter().product();
    let data = if location_external {
        let find = |key: &str| external.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone());
        let location = find("location")
            .ok_or_else(|| bad(format!("external tensor {name:?} has no location")))?;
        let offset: u64 = find("offset").and_then(|v| v.parse().ok()).unwrap_or(0);
        let length: u64 = match find("length").and_then(|v| v.parse().ok()) {
            Some(l) => l,
            None => dtype.byte_len(elements),
        };
        OnnxData::External { location, offset, length }
    } else if let Some((offset, len)) = raw {
        let expected = dtype.byte_len(elements);
        if len as u64 != expected {
            return Err(bad(format!(
                "tensor {name:?} raw_data is {len} bytes, dims want {expected}"
            )));
        }
        OnnxData::Inline { offset, len }
    } else if typed_payload {
        return Err(bad(format!(
            "tensor {name:?} uses a typed payload (float_data/…) — re-export with raw data"
        )));
    } else if elements == 0 {
        OnnxData::Inline { offset: 0, len: 0 }
    } else {
        return Err(bad(format!("tensor {name:?} has no data")));
    };
    Ok(OnnxTensorInfo { name, dtype, dims, data })
}
