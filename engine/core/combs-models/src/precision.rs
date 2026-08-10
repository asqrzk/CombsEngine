//! Mixed-precision helpers.
//!
//! When the backend float is f16 (`--features f16`), a few ops are numerically
//! unsafe in half precision — RMS/LayerNorm reductions (`mean(x²)`) and the
//! attention scores + softmax (`exp` overflows f16's ~65504 ceiling). We run
//! just those in f32 and cast back, keeping the memory-heavy weight matmuls and
//! KV cache in f16. These casts are **no-ops when the backend is already f32**
//! (`cast` to the current dtype is a no-op), so the default f32 build is
//! byte-for-byte unchanged.

use burn::tensor::{DType, FloatDType, Tensor, backend::Backend};

/// Cast a float tensor up to f32 for a stable reduction.
pub fn to_f32<B: Backend, const D: usize>(t: Tensor<B, D>) -> Tensor<B, D> {
    t.cast(FloatDType::F32)
}

/// Cast a float tensor to the given (backend-native) float dtype. Use with a
/// dtype captured from an input tensor via [`Tensor::dtype`] so the result
/// matches the surrounding tensors' precision.
pub fn to_float<B: Backend, const D: usize>(t: Tensor<B, D>, dtype: DType) -> Tensor<B, D> {
    let fd = match dtype {
        DType::F16 => FloatDType::F16,
        DType::F64 => FloatDType::F64,
        _ => FloatDType::F32,
    };
    t.cast(fd)
}
