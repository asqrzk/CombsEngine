//! Group-wise 4-bit quantization support (GGUF `q4_0`-style layout).
//!
//! Phase 2 ships the dequantize op as portable burn tensor ops (nibble
//! extraction via `remainder`/`div` — no bitwise ops or custom kernels
//! required), so it runs on any backend. A fused dequant-matmul CubeCL
//! kernel (dequantize tiles inside the matmul, avoiding a materialized f32
//! weight) is future work; until then [`dequantize_q4`] materializes the
//! f32 weight and callers use the normal matmul path.
//!
//! # Packed layout (GGUF `q4_0`)
//!
//! Weights are grouped into blocks of `group_size` (32) values. Each block
//! stores 16 bytes: byte `j`'s **low** nibble is value `j` of the block and
//! its **high** nibble is value `j + 16`. The dequantized value is
//! `(nibble - 8) * scale` (symmetric quantization with an implicit zero
//! point of 8). Scales are stored per block as f32 on device (f16 scales
//! are widened at load time by the format adapter).

use burn::tensor::{Int, Tensor, backend::Backend};

/// Default quantization group size (GGUF `q4_0`).
pub const DEFAULT_Q4_GROUP_SIZE: usize = 32;

/// Dequantizes packed 4-bit weights to f32.
///
/// - `packed`: `[rows, cols / 2]` int tensor, byte values 0..=255 (GGUF
///   `q4_0` nibble order: low nibble = first half of the block, high nibble
///   = second half).
/// - `scales`: `[rows, cols / group_size]` f32 per-block scales.
///
/// Returns the `[rows, cols]` f32 weight matrix.
///
/// Panics (via shape assertions) if `cols % group_size != 0`; group sizes
/// other than 32 keep the low/high split-half convention within each group
/// (i.e. byte `j` of a group holds values `j` and `j + group_size/2`).
pub fn dequantize_q4<B: Backend>(
    packed: Tensor<B, 2, Int>,
    scales: Tensor<B, 2>,
    group_size: usize,
) -> Tensor<B, 2> {
    let [rows, packed_cols] = packed.dims();
    let half = group_size / 2;
    assert_eq!(
        packed_cols % half,
        0,
        "packed width {packed_cols} not a multiple of half group {half}"
    );
    let groups_per_row = packed_cols / half;
    let cols = groups_per_row * group_size;
    let [s_rows, s_cols] = scales.dims();
    assert_eq!(
        (s_rows, s_cols),
        (rows, groups_per_row),
        "scales shape [{s_rows}, {s_cols}] does not match [rows, cols/group] = [{rows}, {groups_per_row}]"
    );

    // Nibble extraction (portable: remainder/div instead of bitwise ops).
    let nibbles = packed.reshape([rows, groups_per_row, half]);
    let lo = nibbles.clone().remainder_scalar(16); // values j (first half)
    let hi = nibbles.div_scalar(16); // values j + half (second half)

    // [rows, groups, group_size] nibbles, split-half order restored.
    let nibbles = Tensor::cat(vec![lo, hi], 2).float();

    // Symmetric dequant: w = (nibble - 8) * scale.
    let scales = scales
        .unsqueeze_dim::<3>(2)
        .expand([rows, groups_per_row, group_size]);
    let w = (nibbles - 8.0) * scales;
    w.reshape([rows, cols])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    type B = burn::backend::NdArray<f32>;

    /// CPU scalar reference for one GGUF q4_0 block.
    fn reference_block(bytes: &[u8; 16], scale: f32) -> [f32; 32] {
        let mut out = [0.0f32; 32];
        for j in 0..16 {
            out[j] = ((bytes[j] & 0x0F) as f32 - 8.0) * scale;
            out[j + 16] = ((bytes[j] >> 4) as f32 - 8.0) * scale;
        }
        out
    }

    #[test]
    fn dequantize_matches_scalar_reference() {
        let device = Default::default();
        // 2 rows x 2 groups (64 values per row, 32 packed bytes per row).
        let packed_bytes: Vec<i32> = (0..64).map(|i| ((i * 37 + 11) % 256) as i32).collect();
        let scales: Vec<f32> = vec![0.5, -1.25, 2.0, 0.75];
        let packed = Tensor::<B, 2, Int>::from_data(
            TensorData::new(packed_bytes.clone(), [2, 32]),
            &device,
        );
        let scales_t = Tensor::<B, 2>::from_data(TensorData::new(scales.clone(), [2, 2]), &device);

        let w = dequantize_q4(packed, scales_t, DEFAULT_Q4_GROUP_SIZE);
        let got: Vec<f32> = w.into_data().to_vec().unwrap();

        let mut expected = Vec::with_capacity(128);
        for row in 0..2 {
            for g in 0..2 {
                let mut block = [0u8; 16];
                for j in 0..16 {
                    block[j] = packed_bytes[row * 32 + g * 16 + j] as u8;
                }
                expected.extend_from_slice(&reference_block(&block, scales[row * 2 + g]));
            }
        }
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
        }
    }

    #[test]
    fn all_eights_pack_to_zero_weights() {
        let device = Default::default();
        // Nibble 8 everywhere -> (8 - 8) * scale == 0 regardless of scale.
        let packed = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![0x88i32; 16], [1, 16]),
            &device,
        );
        let scales = Tensor::<B, 2>::from_data(TensorData::new(vec![3.5f32], [1, 1]), &device);
        let w = dequantize_q4(packed, scales, DEFAULT_Q4_GROUP_SIZE);
        let got: Vec<f32> = w.into_data().to_vec().unwrap();
        assert_eq!(got, vec![0.0; 32]);
    }
}
