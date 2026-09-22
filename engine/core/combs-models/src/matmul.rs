//! `safe_matmul`: works around a burn 0.21 / cubecl 0.10 matmul kernel bug
//! on wgpu/Metal.
//!
//! # Root cause (verified, see `combs-models/tests/attn_cliff.rs`)
//!
//! cubecl's matmul autotuner picks, for shapes with **M >= 512 AND K >= 512**
//! (any N, any rank, contiguous or transposed rhs), a tile configuration that
//! requests 40960 bytes of shared memory; Metal exposes at most 32768. The
//! dispatch fails wgpu validation and is silently dropped (wgpu reports
//! validation errors asynchronously; nothing in the release path checks
//! them), so the zero-initialized output buffer is returned unwritten —
//! deterministically "garbage" logits for any prefill chunk of >= 512
//! tokens. The cliff is exact: 511 is correct, 512 is not, on fused and
//! unfused cubecl backends alike (fusion is not involved).
//!
//! # Workaround
//!
//! Slab the matmul along the M dimension (rows of the output) so every
//! launched kernel has M < 512 and `cat` the slabs. M-slabbing is exact —
//! each output row is still a single K-reduction — so results are
//! numerically identical to a single matmul modulo the kernels' internal
//! tiling. Below the danger region this is a zero-cost passthrough.

use burn::tensor::{Tensor, backend::Backend};

/// First broken value of M (with K also >= this) on wgpu/Metal.
pub(crate) const BROKEN_MATMUL_BOUNDARY: usize = 512;

/// Rows per slab when splitting; comfortably below the boundary.
const SAFE_M_SLAB: usize = 256;

/// `lhs @ rhs`, transparently slabbed along M when the shape falls into the
/// broken wgpu/Metal region (M >= 512 and K >= 512).
pub(crate) fn safe_matmul<B: Backend, const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
) -> Tensor<B, D> {
    let dims = lhs.dims();
    let (m, k) = (dims[D - 2], dims[D - 1]);
    if m < BROKEN_MATMUL_BOUNDARY || k < BROKEN_MATMUL_BOUNDARY {
        return lhs.matmul(rhs);
    }
    let mut slabs = Vec::with_capacity(m.div_ceil(SAFE_M_SLAB));
    let mut offset = 0;
    while offset < m {
        let len = SAFE_M_SLAB.min(m - offset);
        slabs.push(
            lhs.clone()
                .narrow(D - 2, offset, len)
                .matmul(rhs.clone()),
        );
        offset += len;
    }
    Tensor::cat(slabs, D - 2)
}

/// One line describing the DENSE road — `linear` → [`safe_matmul`] →
/// burn's own matmul — so a seat whose weights are plain tensors records
/// the road it takes instead of the quantized one it never enters.
///
/// `batched_matmul_summary` (qlinear.rs) describes the quantized batched
/// road and its operand door. A dense seat that printed that line said
/// `f32 operands` under an f16 build — true of a road not taken. Here the
/// operands are simply the tensors' own dtype, which the caller knows.
pub fn dense_matmul_summary(dtype: &str, fusion: bool) -> String {
    format!(
        "dense matmul (burn autotune), {dtype} operands, {SAFE_M_SLAB}-row slabs past \
         M>={BROKEN_MATMUL_BOUNDARY} & K>={BROKEN_MATMUL_BOUNDARY}, fusion {}",
        if fusion { "on" } else { "off" }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    type B = burn::backend::NdArray<f32>;

    /// M-slabbing must be numerically identical to a single matmul (checked
    /// on the CPU backend, where both paths are correct).
    #[test]
    fn slabbed_matches_single_matmul() {
        let device = Default::default();
        let (m, k, n) = (600, 576, 1536);
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 5 % 11) as f32 - 5.0) / 8.0).collect();
        let lhs = Tensor::<B, 2>::from_data(TensorData::new(a, [m, k]), &device);
        let rhs = Tensor::<B, 2>::from_data(TensorData::new(b, [k, n]), &device);

        let single = lhs.clone().matmul(rhs.clone());
        let slabbed = safe_matmul(lhs, rhs);
        let single: Vec<f32> = single.into_data().to_vec().unwrap();
        let slabbed: Vec<f32> = slabbed.into_data().to_vec().unwrap();
        assert_eq!(single, slabbed, "M-slabbing must be exact");
    }

    /// Below the boundary the wrapper must be a passthrough (same values).
    #[test]
    fn passthrough_below_boundary() {
        let device = Default::default();
        let (m, k, n) = (511, 64, 32);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32).collect();
        let lhs = Tensor::<B, 2>::from_data(TensorData::new(a, [m, k]), &device);
        let rhs = Tensor::<B, 2>::from_data(TensorData::new(b, [k, n]), &device);
        let single: Vec<f32> = lhs.clone().matmul(rhs.clone()).into_data().to_vec().unwrap();
        let wrapped: Vec<f32> = safe_matmul(lhs, rhs).into_data().to_vec().unwrap();
        assert_eq!(single, wrapped);
    }

    /// The dense line must describe the dense road and nothing about the
    /// quantized batched one (whose line once stood in for it).
    #[test]
    fn the_dense_summary_names_the_road_and_not_the_quantized_one() {
        let line = dense_matmul_summary("f16", false);
        assert!(line.starts_with("dense matmul"), "{line}");
        assert!(line.contains("f16 operands"), "{line}");
        assert!(
            line.contains("256-row slabs past M>=512 & K>=512"),
            "{line}"
        );
        assert!(line.ends_with("fusion off"), "{line}");
        for word in ["batched", "tuned kernel", " rows,"] {
            assert!(
                !line.contains(word),
                "the dense line must not describe the quantized road: {line}"
            );
        }
        assert!(dense_matmul_summary("f32", true).ends_with("fusion on"));
    }
}
