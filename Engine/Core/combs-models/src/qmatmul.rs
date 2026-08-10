//! Fused Q4_0 dequant-matmul CubeCL kernel (QUANTIZATION_PLAN.md Phase 1B).
//!
//! Weights stay packed at 4 bits in VRAM and are dequantized *inside* the
//! matmul kernel — never materialized as f32. This is the memory win that
//! lets a 7B Q4 model run in ~4 GB instead of ~28 GB of weight VRAM.
//!
//! Follows the two-layer design from the plan's "Kernel architecture":
//!
//! - **Layout** ([`repack_q4_0`], [`Q40Weight`]): GGUF's 18-byte blocks
//!   (f16 scale + 16 nibble bytes) are not word-aligned, so at load we
//!   repack once into a GPU-friendly structure-of-arrays — nibble bytes as
//!   `u32` words plus an `f32` scale per block (20 B / 32 weights = 5.0
//!   bits/weight; f32 scales keep the kernel bit-exact with the CPU
//!   reference, packing them back to f16 pairs is a later 0.5-bit saving).
//! - **Compute** ([`q4_0_dequant_kernel`], [`q4_0_matmul_kernel`]): unpack
//!   nibbles, apply scale, accumulate in f32. The dequant-only kernel
//!   exists to validate the layout bit-exactly against the golden CPU
//!   reference (`combs_formats::quants::dequantize_q4_0`); the fused
//!   matmul is the production path.
//!
//! The portable fallback (dequantize at load + burn matmul) remains the
//! default; this kernel is the opt-in fast path behind the linear seam.

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::{ModelError, Result};

/// Values per GGUF Q4_0 block.
pub const Q4_0_BLOCK: usize = 32;
/// Bytes per GGUF Q4_0 block: 2-byte f16 scale + 16 packed nibble bytes.
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Layout step: repack a raw GGUF Q4_0 block stream into the device layout
/// the kernels consume — nibble bytes as little-endian `u32` words
/// (4 words per block) and one `f32` scale per block. The f16→f32 scale
/// conversion is exact, so no precision is lost relative to the reference.
pub fn repack_q4_0(data: &[u8]) -> Result<(Vec<u32>, Vec<f32>)> {
    if data.is_empty() || data.len() % Q4_0_BLOCK_BYTES != 0 {
        return Err(ModelError::BadShape {
            tensor: "q4_0 block stream".into(),
            expected: vec![Q4_0_BLOCK_BYTES],
            got: vec![data.len()],
        });
    }
    let n_blocks = data.len() / Q4_0_BLOCK_BYTES;
    let mut qs = Vec::with_capacity(n_blocks * 4);
    let mut d = Vec::with_capacity(n_blocks);
    for block in data.chunks_exact(Q4_0_BLOCK_BYTES) {
        d.push(burn::tensor::f16::from_le_bytes([block[0], block[1]]).to_f32());
        for w in 0..4 {
            let o = 2 + 4 * w;
            qs.push(u32::from_le_bytes([
                block[o],
                block[o + 1],
                block[o + 2],
                block[o + 3],
            ]));
        }
    }
    Ok((qs, d))
}

/// Dequantize-only kernel: `out[i]` = value `i` of the block stream, using
/// the exact arithmetic of the CPU reference (`(nibble as i32 - 8) as f32
/// * d`), so results are bit-identical. One thread per output value.
#[cube(launch_unchecked)]
fn q4_0_dequant_kernel(qs: &Array<u32>, d: &Array<f32>, out: &mut Array<f32>, n: usize) {
    if ABSOLUTE_POS < n {
        let block = ABSOLUTE_POS / 32;
        let j = ABSOLUTE_POS % 32;
        let byte_idx = j % 16;
        let word = qs[block * 4 + byte_idx / 4];
        let byte = (word >> (u32::cast_from(byte_idx % 4) * 8)) & 0xFF;
        let mut nib = byte & 0xF;
        if j >= 16 {
            nib = byte >> 4;
        }
        out[ABSOLUTE_POS] = f32::cast_from(i32::cast_from(nib) - 8) * d[block];
    }
}

/// Fused dequant-matmul: `out[row, col] = Σ_k x[row, k] · dequant(w[col, k])`
/// for `x: [m, k]` f32 activations and `w: [n_out, k]` Q4_0 weights packed
/// row-major with blocks along `k`. One thread per output element; per-block
/// products accumulate unscaled and are multiplied by the block scale once
/// (fewer multiplies, and the f32 accumulator never sees f16 range limits).
#[cube(launch_unchecked)]
fn q4_0_matmul_kernel(
    x: &Array<f32>,
    qs: &Array<u32>,
    d: &Array<f32>,
    out: &mut Array<f32>,
    m: usize,
    k: usize,
    n_out: usize,
) {
    if ABSOLUTE_POS < m * n_out {
        let row = ABSOLUTE_POS / n_out;
        let col = ABSOLUTE_POS % n_out;
        let blocks_per_row = k / 32;
        let mut acc = 0.0f32;
        for kb in 0..blocks_per_row {
            let block = col * blocks_per_row + kb;
            let x_base = row * k + kb * 32;
            let mut block_acc = 0.0f32;
            for w in 0..4usize {
                let word = qs[block * 4 + w];
                for b in 0..4usize {
                    let byte = (word >> (u32::cast_from(b) * 8)) & 0xFF;
                    let jj = w * 4 + b;
                    let lo = f32::cast_from(i32::cast_from(byte & 0xF) - 8);
                    let hi = f32::cast_from(i32::cast_from(byte >> 4) - 8);
                    block_acc += lo * x[x_base + jj];
                    block_acc += hi * x[x_base + 16 + jj];
                }
            }
            acc += d[block] * block_acc;
        }
        out[row * n_out + col] = acc;
    }
}

/// Threads per cube for the 1-D launches below.
const CUBE_DIM: u32 = 256;

fn cube_count_1d(total: u32) -> CubeCount {
    CubeCount::Static(total.div_ceil(CUBE_DIM).max(1), 1, 1)
}

/// Runs the dequant-only kernel over a raw Q4_0 block stream. Exists for
/// validation (bit-exact vs the CPU reference) and debugging, not the hot
/// path.
pub fn dequantize_q4_0_gpu<R: Runtime>(client: &ComputeClient<R>, data: &[u8]) -> Result<Vec<f32>> {
    let (qs, d) = repack_q4_0(data)?;
    let n = d.len() * Q4_0_BLOCK;
    let qs_h = client.create_from_slice(u32::as_bytes(&qs));
    let d_h = client.create_from_slice(f32::as_bytes(&d));
    let out_h = client.empty(n * core::mem::size_of::<f32>());
    unsafe {
        q4_0_dequant_kernel::launch_unchecked::<R>(
            client,
            cube_count_1d(n as u32),
            CubeDim::new_1d(CUBE_DIM),
            ArrayArg::from_raw_parts(qs_h, qs.len()),
            ArrayArg::from_raw_parts(d_h, d.len()),
            ArrayArg::from_raw_parts(out_h.clone(), n),
            n,
        );
    }
    let bytes = client.read_one_unchecked(out_h);
    Ok(f32::from_bytes(&bytes).to_vec())
}

/// A weight matrix resident in VRAM in packed Q4_0 form. `[n_out, k]`
/// row-major, `k % 32 == 0`, blocks along `k` — exactly the GGUF tensor
/// layout, so `from_gguf_bytes` takes the mmap'd tensor bytes unchanged.
pub struct Q40Weight<R: Runtime> {
    qs: Handle,
    d: Handle,
    n_out: usize,
    k: usize,
    _runtime: PhantomData<R>,
}

impl<R: Runtime> Q40Weight<R> {
    /// Repacks a GGUF Q4_0 tensor onto the device. `data` is the raw block
    /// stream for an `[n_out, k]` weight (the bytes `GgufSource` maps).
    pub fn from_gguf_bytes(
        client: &ComputeClient<R>,
        data: &[u8],
        n_out: usize,
        k: usize,
    ) -> Result<Self> {
        if k == 0 || k % Q4_0_BLOCK != 0 || data.len() != n_out * k / Q4_0_BLOCK * Q4_0_BLOCK_BYTES
        {
            return Err(ModelError::BadShape {
                tensor: "q4_0 weight".into(),
                expected: vec![n_out, k / Q4_0_BLOCK.max(1) * Q4_0_BLOCK_BYTES],
                got: vec![data.len()],
            });
        }
        let (qs, d) = repack_q4_0(data)?;
        Ok(Q40Weight {
            qs: client.create_from_slice(u32::as_bytes(&qs)),
            d: client.create_from_slice(f32::as_bytes(&d)),
            n_out,
            k,
            _runtime: PhantomData,
        })
    }

    /// Output features.
    pub fn n_out(&self) -> usize {
        self.n_out
    }

    /// Input features.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Bytes this weight occupies in VRAM (packed nibbles + f32 scales) —
    /// 20 bytes per 32 weights, vs 128 for f32 (6.4×) or 64 for f16 (3.2×).
    pub fn vram_bytes(&self) -> usize {
        let n_blocks = self.n_out * self.k / Q4_0_BLOCK;
        n_blocks * (16 + core::mem::size_of::<f32>())
    }

    /// `y = x @ W^T` for host-side `x: [m, k]`, returning `[m, n_out]`.
    /// Host-slice convenience for tests/CLI probes; the device-tensor path
    /// (activation handle in, handle out) lands with the linear-seam wiring.
    pub fn matmul_host(&self, client: &ComputeClient<R>, x: &[f32], m: usize) -> Result<Vec<f32>> {
        if m == 0 || x.len() != m * self.k {
            return Err(ModelError::BadShape {
                tensor: "q4_0 matmul input".into(),
                expected: vec![m, self.k],
                got: vec![x.len()],
            });
        }
        let x_h = client.create_from_slice(f32::as_bytes(x));
        let out_len = m * self.n_out;
        let out_h = client.empty(out_len * core::mem::size_of::<f32>());
        let n_blocks = self.n_out * self.k / Q4_0_BLOCK;
        unsafe {
            q4_0_matmul_kernel::launch_unchecked::<R>(
                client,
                cube_count_1d(out_len as u32),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(x_h, x.len()),
                ArrayArg::from_raw_parts(self.qs.clone(), n_blocks * 4),
                ArrayArg::from_raw_parts(self.d.clone(), n_blocks),
                ArrayArg::from_raw_parts(out_h.clone(), out_len),
                m,
                self.k,
                self.n_out,
            );
        }
        let bytes = client.read_one_unchecked(out_h);
        Ok(f32::from_bytes(&bytes).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::WgpuRuntime;

    /// Deterministic pseudo-random Q4_0 block stream: valid finite f16
    /// scales, LCG nibble bytes covering the full 0..=255 range.
    fn synth_q4_0(n_blocks: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n_blocks * Q4_0_BLOCK_BYTES);
        let mut s = 0x12345678u32;
        for b in 0..n_blocks {
            let scale = burn::tensor::f16::from_f32(0.003 * ((b % 11) as f32 + 1.0));
            out.extend_from_slice(&scale.to_le_bytes());
            for _ in 0..16 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                out.push((s >> 24) as u8);
            }
        }
        out
    }

    /// Plain f32 reference matmul over the CPU-dequantized weight.
    fn ref_matmul(x: &[f32], w: &[f32], m: usize, k: usize, n_out: usize) -> Vec<f32> {
        let mut out = vec![0f32; m * n_out];
        for r in 0..m {
            for c in 0..n_out {
                let mut acc = 0f32;
                for i in 0..k {
                    acc += x[r * k + i] * w[c * k + i];
                }
                out[r * n_out + c] = acc;
            }
        }
        out
    }

    /// The GPU dequant must be **bit-exact** with the golden CPU reference
    /// (`combs_formats::quants::dequantize_q4_0`) — same unpack, same
    /// arithmetic, same f16→f32 scale conversion. This validates the whole
    /// Layout layer: any repack/indexing slip shows up as a hard mismatch.
    #[test]
    fn dequant_kernel_is_bit_exact_vs_cpu_reference() {
        let n_blocks = 33; // deliberately not a multiple of the cube dim
        let data = synth_q4_0(n_blocks);
        let n = n_blocks * Q4_0_BLOCK;
        let expect = combs_formats::quants::dequantize_q4_0(&data, n).unwrap();

        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let got = dequantize_q4_0_gpu::<WgpuRuntime>(&client, &data).unwrap();

        assert_eq!(got, expect, "GPU dequant must be bit-exact vs gguf.rs");
    }

    /// The fused kernel must match a reference matmul over the reference
    /// dequant within accumulation-order tolerance, for both the decode
    /// shape (m=1) and a prefill shape (m>1), across multiple cubes.
    #[test]
    fn fused_matmul_matches_reference() {
        let (n_out, k) = (67, 128); // 67 forces a partial second cube
        let n_blocks = n_out * k / Q4_0_BLOCK;
        let data = synth_q4_0(n_blocks);
        let w = combs_formats::quants::dequantize_q4_0(&data, n_out * k).unwrap();

        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let weight = Q40Weight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k).unwrap();
        assert_eq!(weight.vram_bytes(), n_blocks * 20);

        for m in [1usize, 3] {
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0)
                .collect();
            let expect = ref_matmul(&x, &w, m, k, n_out);
            let got = weight.matmul_host(&client, &x, m).unwrap();
            assert_eq!(got.len(), expect.len());
            for (i, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
                let tol = 1e-4 * e.abs().max(1.0);
                assert!(
                    (g - e).abs() <= tol,
                    "m={m} out[{i}]: got {g}, expect {e}"
                );
            }
        }
    }

    /// Malformed inputs must be rejected, not mis-indexed.
    #[test]
    fn shape_validation() {
        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        // Truncated block stream.
        assert!(repack_q4_0(&[0u8; 17]).is_err());
        // k not a multiple of the block size.
        assert!(Q40Weight::<WgpuRuntime>::from_gguf_bytes(&client, &synth_q4_0(2), 2, 31).is_err());
        // Byte count disagrees with [n_out, k].
        assert!(Q40Weight::<WgpuRuntime>::from_gguf_bytes(&client, &synth_q4_0(2), 2, 64).is_err());
        // Bad x length.
        let w = Q40Weight::<WgpuRuntime>::from_gguf_bytes(&client, &synth_q4_0(2), 2, 32).unwrap();
        assert!(w.matmul_host(&client, &[0f32; 31], 1).is_err());
    }
}
