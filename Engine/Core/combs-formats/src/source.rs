//! The format-agnostic [`ModelSource`] trait and tensor reader.

use burn::tensor::{Device, Int, Tensor, TensorData, backend::Backend};

use crate::metadata::ModelMetadata;
use crate::tokenizer::TokenizerSpec;
use crate::{FormatError, Result};

/// The central adapter trait: a source of model weights + config, independent
/// of the on-disk format (LiteRT-LM `ModelResources` equivalent).
///
/// Implementations must be cheap to query for metadata and names, and lazy /
/// zero-copy (e.g. mmap-backed) when opening tensors.
pub trait ModelSource: Send + Sync {
    /// Architecture + hyperparameter metadata.
    fn metadata(&self) -> &ModelMetadata;

    /// Names of all tensors available in this source.
    fn tensor_names(&self) -> Vec<String>;

    /// Opens a tensor by name, returning a lazy reader over the raw bytes.
    fn open_tensor(&self, name: &str) -> Result<TensorReader<'_>>;

    /// Tokenizer specification (path to `tokenizer.json` + added tokens).
    fn tokenizer(&self) -> Result<TokenizerSpec>;

    /// Sampler defaults from `generation_config.json`, if present.
    fn sampler_defaults(&self) -> Option<SamplerConfig>;
}

/// Element dtypes supported by the loaders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    /// IEEE 64-bit float.
    F64,
    /// IEEE 32-bit float.
    F32,
    /// IEEE 16-bit half float.
    F16,
    /// bfloat16.
    BF16,
    /// Signed 64-bit integer.
    I64,
    /// Signed 32-bit integer.
    I32,
    /// Signed 16-bit integer.
    I16,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 64-bit integer.
    U64,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 8-bit integer (raw packed data, e.g. quantized weights).
    U8,
    /// Boolean (stored as one byte).
    Bool,
}

impl TensorDtype {
    /// Byte size of one element.
    pub fn size(&self) -> usize {
        match self {
            TensorDtype::F64 | TensorDtype::I64 | TensorDtype::U64 => 8,
            TensorDtype::F32 | TensorDtype::I32 | TensorDtype::U32 => 4,
            TensorDtype::F16 | TensorDtype::BF16 | TensorDtype::I16 | TensorDtype::U16 => 2,
            TensorDtype::I8 | TensorDtype::U8 | TensorDtype::Bool => 1,
        }
    }
}

impl std::fmt::Display for TensorDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorDtype::F64 => write!(f, "F64"),
            TensorDtype::F32 => write!(f, "F32"),
            TensorDtype::F16 => write!(f, "F16"),
            TensorDtype::BF16 => write!(f, "BF16"),
            TensorDtype::I64 => write!(f, "I64"),
            TensorDtype::I32 => write!(f, "I32"),
            TensorDtype::I16 => write!(f, "I16"),
            TensorDtype::I8 => write!(f, "I8"),
            TensorDtype::U64 => write!(f, "U64"),
            TensorDtype::U32 => write!(f, "U32"),
            TensorDtype::U16 => write!(f, "U16"),
            TensorDtype::U8 => write!(f, "U8"),
            TensorDtype::Bool => write!(f, "Bool"),
        }
    }
}

/// A lazy view over one tensor's raw bytes inside a [`ModelSource`].
///
/// The byte slice borrows from the source (e.g. an mmap region) — no copy is
/// made until [`TensorReader::load_data`] is called. Format adapters that
/// decode on open (e.g. GGUF quantization) use [`TensorReader::owned`].
pub struct TensorReader<'a> {
    name: String,
    shape: Vec<usize>,
    dtype: TensorDtype,
    data: std::borrow::Cow<'a, [u8]>,
}

impl<'a> TensorReader<'a> {
    /// Creates a reader from raw parts. `data` must be
    /// `shape.iter().product::<usize>() * dtype.size()` little-endian bytes.
    pub fn new(name: String, shape: Vec<usize>, dtype: TensorDtype, data: &'a [u8]) -> Self {
        TensorReader {
            name,
            shape,
            dtype,
            data: std::borrow::Cow::Borrowed(data),
        }
    }

    /// Creates a reader over owned (already-decoded) f32 bytes.
    pub fn owned(name: String, shape: Vec<usize>, data: Vec<u8>) -> Self {
        TensorReader {
            name,
            shape,
            dtype: TensorDtype::F32,
            data: std::borrow::Cow::Owned(data),
        }
    }

    /// Element shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// On-disk dtype.
    pub fn dtype(&self) -> TensorDtype {
        self.dtype
    }

    /// Number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Converts the raw bytes to f32 `TensorData`. Integer and boolean tensors
    /// are cast to f32 so weight loaders never abort on buffer dtypes such as
    /// I64 `position_ids`.
    pub fn load_data(&self) -> Result<TensorData> {
        let values: Vec<f32> = match self.dtype {
            TensorDtype::F64 => self
                .data
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
                .collect(),
            TensorDtype::F32 => self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            TensorDtype::F16 => self
                .data
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            TensorDtype::BF16 => self
                .data
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            TensorDtype::I64 => self
                .data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
                .collect(),
            TensorDtype::I32 => self
                .data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect(),
            TensorDtype::I16 => self
                .data
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
                .collect(),
            TensorDtype::I8 => self
                .data
                .iter()
                .map(|&b| b as i8 as f32)
                .collect(),
            TensorDtype::U64 => self
                .data
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
                .collect(),
            TensorDtype::U32 => self
                .data
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect(),
            TensorDtype::U16 => self
                .data
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32)
                .collect(),
            TensorDtype::U8 => self.data.iter().map(|&b| b as f32).collect(),
            TensorDtype::Bool => self.data.iter().map(|&b| (b != 0) as i32 as f32).collect(),
        };
        if values.len() != self.num_elements() {
            return Err(FormatError::Safetensors(format!(
                "tensor {}: expected {} elements, got {} (dtype {})",
                self.name,
                self.num_elements(),
                values.len(),
                self.dtype
            )));
        }
        Ok(TensorData::new(values, self.shape.clone()))
    }

    /// Loads the tensor onto a backend device as an f32 tensor of rank `D`.
    pub fn load_to_tensor<B: Backend, const D: usize>(
        &self,
        device: &Device<B>,
    ) -> Result<Tensor<B, D>> {
        let data = self.load_data()?;
        if self.shape.len() != D {
            return Err(FormatError::Safetensors(format!(
                "tensor {}: expected rank {D}, got {}",
                self.name,
                self.shape.len()
            )));
        }
        Ok(Tensor::from_data(data, device))
    }

    /// Loads raw unsigned-byte data onto a backend device as an i32 tensor
    /// of rank `D` (values 0..=255). Only valid for `U8` tensors; used to
    /// feed packed quantized weights to GPU-side dequantization.
    pub fn load_int_tensor<B: Backend, const D: usize>(
        &self,
        device: &Device<B>,
    ) -> Result<Tensor<B, D, Int>> {
        if self.dtype != TensorDtype::U8 {
            return Err(FormatError::Safetensors(format!(
                "tensor {}: load_int_tensor requires U8, got {}",
                self.name, self.dtype
            )));
        }
        if self.shape.len() != D {
            return Err(FormatError::Safetensors(format!(
                "tensor {}: expected rank {D}, got {}",
                self.name,
                self.shape.len()
            )));
        }
        let values: Vec<i32> = self.data.iter().map(|&b| b as i32).collect();
        Ok(Tensor::from_data(
            TensorData::new(values, self.shape.clone()),
            device,
        ))
    }

    /// The raw little-endian bytes, borrowing from the source (zero-copy).
    pub fn raw_bytes(&self) -> &[u8] {
        &self.data
    }
}

/// Blanket forwarding so `Box<dyn ModelSource>` (returned by
/// `open_model_source`) can be passed anywhere a `&dyn ModelSource` goes.
impl<T: ModelSource + ?Sized> ModelSource for Box<T> {
    fn metadata(&self) -> &crate::ModelMetadata {
        (**self).metadata()
    }
    fn tensor_names(&self) -> Vec<String> {
        (**self).tensor_names()
    }
    fn open_tensor(&self, name: &str) -> crate::Result<TensorReader<'_>> {
        (**self).open_tensor(name)
    }
    fn tokenizer(&self) -> crate::Result<TokenizerSpec> {
        (**self).tokenizer()
    }
    fn sampler_defaults(&self) -> Option<SamplerConfig> {
        (**self).sampler_defaults()
    }
}

/// Default sampler parameters, typically from `generation_config.json`.
#[derive(Debug, Clone, Default)]
pub struct SamplerConfig {    /// Sampling temperature (1.0 = neutral, 0.0 = greedy).
    pub temperature: Option<f32>,
    /// Top-p (nucleus) threshold.
    pub top_p: Option<f32>,
    /// Top-k cutoff.
    pub top_k: Option<usize>,
    /// Repetition penalty.
    pub repetition_penalty: Option<f32>,
    /// Suggested maximum new tokens.
    pub max_new_tokens: Option<usize>,
}
