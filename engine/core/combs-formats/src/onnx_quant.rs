//! MatMulNBits → Q4_0 repack.
//!
//! `com.microsoft.MatMulNBits` at 4 bits with the default zero-point
//! is FORMULA-identical to ggml's Q4_0: `w = (q - 8) * scale` over
//! 32-element blocks along K. What differs is byte layout — ORT packs
//! sequential nibble pairs (element 2i low, 2i+1 high) with scales in
//! a separate f32 tensor; Q4_0 interleaves halves (byte j holds
//! elements j and j+16) behind an f16 scale. The repack is therefore
//! a pure nibble shuffle plus one f32→f16 scale cast — the shuffle is
//! lossless by construction (proved bit-exact against the reference
//! dequantizers below); the scale cast rounds at f16 precision
//! (relative ≤ 2^-11, uniform across the block — the same order as
//! Q4 quantization noise itself, measured and recorded rather than
//! wished away).
//!
//! Models with explicit zero_points or g_idx don't fit Q4_0's
//! zero-point-8 and take the dequant-to-float fallback instead —
//! [`dequantize_matmul_nbits`] is both that fallback and the test
//! oracle.

use crate::{FormatError, Result};

const QK: usize = 32;
const BLOCK_BYTES: usize = 18;

/// Reference dequantization of a 4-bit MatMulNBits weight: `packed`
/// is `[n][ceil(k/block)][block/2]` bytes (sequential nibble pairs,
/// low first), `scales` is `[n * ceil(k/block)]`, `zero_points`
/// (packed nibbles, `[n * ceil(ceil(k/block)/2)]`... ORT packs them
/// per column) — only the ABSENT case (zp = 8) is supported here;
/// models carrying explicit zero points take this path with their own
/// zp handling once one actually shows up in the wild.
pub fn dequantize_matmul_nbits(
    packed: &[u8],
    scales: &[f32],
    k: usize,
    n: usize,
    block_size: usize,
) -> Result<Vec<f32>> {
    let blocks_per_col = k.div_ceil(block_size);
    let blob = block_size / 2;
    if packed.len() != n * blocks_per_col * blob {
        return Err(FormatError::Safetensors(format!(
            "matmulnbits: packed len {} != n {n} x blocks {blocks_per_col} x blob {blob}",
            packed.len()
        )));
    }
    if scales.len() != n * blocks_per_col {
        return Err(FormatError::Safetensors(format!(
            "matmulnbits: {} scales for {} blocks",
            scales.len(),
            n * blocks_per_col
        )));
    }
    let mut out = vec![0f32; n * k];
    for row in 0..n {
        for b in 0..blocks_per_col {
            let scale = scales[row * blocks_per_col + b];
            let blob_at = (row * blocks_per_col + b) * blob;
            for j in 0..block_size {
                let col = b * block_size + j;
                if col >= k {
                    break;
                }
                let byte = packed[blob_at + j / 2];
                let q = if j % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                out[row * k + col] = (q as i32 - 8) as f32 * scale;
            }
        }
    }
    Ok(out)
}

/// Repack a 4-bit MatMulNBits weight (no zero_points, block 32) into
/// a Q4_0 stream: `n` rows of `k/32` 18-byte blocks, ready for the
/// existing Q4_0 device kernels. Anything that doesn't fit Q4_0's
/// shape errs — the caller falls back to dequantization.
pub fn repack_matmul_nbits_q4_0(
    packed: &[u8],
    scales: &[f32],
    k: usize,
    n: usize,
    block_size: usize,
) -> Result<Vec<u8>> {
    if block_size != QK {
        return Err(FormatError::Safetensors(format!(
            "matmulnbits repack: block_size {block_size} != {QK} (Q4_0 blocks are 32)"
        )));
    }
    if k % QK != 0 {
        return Err(FormatError::Safetensors(format!(
            "matmulnbits repack: k {k} not a multiple of {QK}"
        )));
    }
    let blocks_per_col = k / QK;
    let blob = QK / 2;
    if packed.len() != n * blocks_per_col * blob || scales.len() != n * blocks_per_col {
        return Err(FormatError::Safetensors(
            "matmulnbits repack: packed/scales sizes disagree with k x n".to_string(),
        ));
    }
    let mut out = vec![0u8; n * blocks_per_col * BLOCK_BYTES];
    for (block_idx, chunk) in out.chunks_exact_mut(BLOCK_BYTES).enumerate() {
        let d = half::f16::from_f32(scales[block_idx]);
        chunk[0..2].copy_from_slice(&d.to_le_bytes());
        let blob_at = block_idx * blob;
        // Sequential pairs → split halves: q[j] low, q[j+16] high.
        let nibble = |j: usize| {
            let byte = packed[blob_at + j / 2];
            if j % 2 == 0 { byte & 0x0f } else { byte >> 4 }
        };
        for j in 0..16 {
            chunk[2 + j] = nibble(j) | (nibble(j + 16) << 4);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quants::dequantize_q4_0;

    fn fixture(k: usize, n: usize) -> (Vec<u8>, Vec<f32>) {
        let blocks = k / QK;
        let packed: Vec<u8> = (0..n * blocks * (QK / 2))
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .collect();
        let scales: Vec<f32> = (0..n * blocks)
            .map(|i| 0.003 + 0.001 * ((i * 7) % 13) as f32)
            .collect();
        (packed, scales)
    }

    /// The layout shuffle is lossless: repacked-then-Q4_0-dequantized
    /// must equal the reference semantics computed with the SAME
    /// f16-rounded scales, bit for bit.
    #[test]
    fn repack_is_a_pure_shuffle() {
        let (k, n) = (96, 5);
        let (packed, scales) = fixture(k, n);
        let f16_scales: Vec<f32> =
            scales.iter().map(|&s| half::f16::from_f32(s).to_f32()).collect();

        let q4 = repack_matmul_nbits_q4_0(&packed, &scales, k, n, QK).unwrap();
        let via_q4_0 = dequantize_q4_0(&q4, n * k).unwrap();
        let oracle = dequantize_matmul_nbits(&packed, &f16_scales, k, n, QK).unwrap();
        assert_eq!(via_q4_0.len(), oracle.len());
        for (i, (a, b)) in via_q4_0.iter().zip(&oracle).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "element {i}: {a} vs {b}");
        }
    }

    /// With f16-representable scales the whole path is bit-exact
    /// against the full-precision reference.
    #[test]
    fn f16_clean_scales_are_bit_exact() {
        let (k, n) = (64, 3);
        let (packed, _) = fixture(k, n);
        let scales: Vec<f32> = (0..n * (k / QK)).map(|i| 0.25 * (i as f32 + 1.0)).collect();
        let q4 = repack_matmul_nbits_q4_0(&packed, &scales, k, n, QK).unwrap();
        let via_q4_0 = dequantize_q4_0(&q4, n * k).unwrap();
        let oracle = dequantize_matmul_nbits(&packed, &scales, k, n, QK).unwrap();
        for (a, b) in via_q4_0.iter().zip(&oracle) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// The scale cast's drift stays inside f16 rounding: relative
    /// ≤ 2^-11 per element, measured.
    #[test]
    fn scale_rounding_stays_bounded() {
        let (k, n) = (128, 4);
        let (packed, scales) = fixture(k, n);
        let q4 = repack_matmul_nbits_q4_0(&packed, &scales, k, n, QK).unwrap();
        let via_q4_0 = dequantize_q4_0(&q4, n * k).unwrap();
        let oracle = dequantize_matmul_nbits(&packed, &scales, k, n, QK).unwrap();
        let mut worst = 0f32;
        for (a, b) in via_q4_0.iter().zip(&oracle) {
            if *b != 0.0 {
                worst = worst.max((a - b).abs() / b.abs());
            }
        }
        println!("[matmulnbits] worst relative scale-cast drift {worst:e}");
        assert!(worst <= 1.0 / 2048.0, "beyond f16 rounding: {worst}");
    }

    #[test]
    fn wrong_shapes_err() {
        let (packed, scales) = fixture(64, 2);
        assert!(repack_matmul_nbits_q4_0(&packed, &scales, 64, 2, 16).is_err(), "block 16");
        assert!(repack_matmul_nbits_q4_0(&packed, &scales, 60, 2, QK).is_err(), "ragged k");
        assert!(repack_matmul_nbits_q4_0(&packed[1..], &scales, 64, 2, QK).is_err(), "short blob");
    }
}
