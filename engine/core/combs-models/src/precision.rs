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

/// Attention scores, widened BEFORE the product rather than after.
///
/// This module's own rule, stated at the top of this file, is that the
/// attention scores run in f32 because `exp` overflows f16's ~65504
/// ceiling. Two seats honoured half of it: they computed `q·kᵀ` at the
/// build's dtype and cast the RESULT to f32, so the widening arrived
/// after the arithmetic it was supposed to protect. The estate's own
/// pattern (`kv.rs`) widens q, k and v first, and that is the pattern.
///
/// The dot is over `head_dim` terms, so a row of LayerNormed values is
/// nowhere near the ceiling in practice — which is why nothing has gone
/// visibly wrong. It is also where most of the f16 parity gap lives, and
/// "implausible" is not the same as "cannot", especially for a seat that
/// takes arbitrary text from a caller.
///
/// Returns f32 on every build. In an f32 build both casts are no-ops and
/// the arithmetic is byte-for-byte what it was.
pub fn attention_scores<B: Backend>(
    q: Tensor<B, 4>,
    k_transposed: Tensor<B, 4>,
    head_dim: usize,
) -> Tensor<B, 4> {
    let scores = crate::matmul::safe_matmul(to_f32(q), to_f32(k_transposed));
    scores / (head_dim as f64).sqrt()
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::qlinear::UnfusedF16;
    use burn::tensor::TensorData;
    use combs_core::CombsBackend;

    /// The scale is the dot's, not the product's: dividing by
    /// `sqrt(head_dim)` after widening is the same number as before, and
    /// this pins it on every build so the f16 arm below is only about
    /// the widening.
    #[test]
    fn the_scores_carry_the_dot_scaled_by_the_head_dim() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = combs_core::init_device();
        let head_dim = 4usize;
        // q and k both all-ones over 4 terms: the dot is 4, scaled 4/2 = 2.
        let q = Tensor::<CombsBackend, 4>::ones([1, 1, 2, head_dim], &device);
        let k_t = Tensor::<CombsBackend, 4>::ones([1, 1, head_dim, 3], &device);
        let s = attention_scores(q, k_t, head_dim);
        assert_eq!(s.dims(), [1, 1, 2, 3]);
        assert_eq!(s.dtype(), DType::F32, "scores are f32 on every build");
        let got = s
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32");
        for v in got {
            assert!((v - 2.0).abs() < 1e-4, "expected 4/sqrt(4) = 2, got {v}");
        }
    }

    /// The reason the widening has to come FIRST.
    ///
    /// A dot of 64 terms at 120 each is 921,600 — fourteen times f16's
    /// 65,504 ceiling. Computed at f16 it saturates to infinity and the
    /// `to_f32` that used to follow preserves the infinity faithfully.
    /// Widened first, the same arithmetic is ordinary.
    ///
    /// The operands are f16 by name, not by build: every GPU test here
    /// pins the shared test device to f32 first (`skip_no_gpu`), so the
    /// width is stated at creation and this red runs in both flavours'
    /// lib tests. A no-op cast cannot be observed on an f32 tensor, which
    /// is exactly why the defect survived every f32 gate until the f16
    /// tensor was named.
    #[test]
    fn scores_are_widened_before_the_product_so_f16_cannot_overflow() {
        if crate::skip_no_gpu() {
            return;
        }
        let device: burn::tensor::Device<UnfusedF16> = Default::default();
        let head_dim = 64usize;
        let f16_full = |shape: [usize; 4]| {
            let n: usize = shape.iter().product();
            Tensor::<UnfusedF16, 4>::from_data(
                TensorData::new(vec![120.0f32; n], shape),
                (&device, DType::F16),
            )
        };
        let q = f16_full([1, 1, 1, head_dim]);
        let k_t = f16_full([1, 1, head_dim, 1]);
        assert_eq!(q.dtype(), DType::F16, "the operand is f16 by name");
        let s = attention_scores(q, k_t, head_dim);
        let got = s
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32")[0];
        // 64 * 120 * 120 / 8 = 115,200.
        assert!(got.is_finite(), "the product saturated: {got}");
        assert!(
            (got - 115_200.0).abs() / 115_200.0 < 0.01,
            "expected ~115200, got {got}"
        );
    }
}
