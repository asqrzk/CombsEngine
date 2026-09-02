//! Fused GGUF dequant-matmul CubeCL kernels: Q4_0, Q5_0, Q8_0 and the
//! K-quants Q4_K, Q5_K, Q6_K — the formats real model files actually use.
//! The `#[cube]` half of the Combs Kernel (see `wgsl/` for the
//! hand-written half and the suite's contract).
//!
//! Weights stay packed at 4–6 bits in VRAM and are dequantized *inside*
//! the matmul kernel — never materialized as f32. This is the memory win
//! that lets a 7B Q4 model run in ~4 GB instead of ~28 GB of weight VRAM.
//!
//! Follows a two-layer design:
//!
//! - **Layout** (`repack_*` plus a per-format weight struct): GGUF
//!   block streams are not word-aligned (18–210-byte blocks), so at
//!   load we repack once into a GPU-friendly structure-of-arrays — packed
//!   quant bytes as `u32` words plus `f32` super-scales (f16→f32 host
//!   conversion is exact, keeping the kernels bit-comparable with the CPU
//!   reference; re-packing scales to f16 pairs is a later small saving).
//! - **Compute** (`*_dequant_kernel`, `*_matmul_kernel`): unpack, apply
//!   scales, accumulate in f32. Each dequant-only kernel exists to
//!   validate the layout bit-exactly against the harmony CPU reference
//!   (`combs_formats::quants`); the fused matmuls are the production path.
//!
//! The portable fallback (dequantize at load + burn matmul) remains the
//! default; these kernels are the opt-in fast path behind the linear seam.

use core::marker::PhantomData;
use std::sync::OnceLock;

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
fn q4_0_dequant_kernel<F: Float>(qs: &Array<u32>, d: &Array<f32>, out: &mut Array<F>, n: usize) {
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
        out[ABSOLUTE_POS] = F::cast_from(f32::cast_from(i32::cast_from(nib) - 8) * d[block]);
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

/// Max cubes per grid dimension (wgpu/Metal limit).
const MAX_CUBES_PER_DIM: u32 = 65535;

fn cube_count_1d(total: u32) -> CubeCount {
    CubeCount::Static(total.div_ceil(CUBE_DIM).max(1), 1, 1)
}

/// Like [`cube_count_1d`] but splits across the Y grid dimension when the
/// thread count exceeds one dimension's limit (large prefill × vocab
/// launches). Over-provisioned cubes are discarded by the in-kernel bound
/// guard, which indexes by the linear `ABSOLUTE_POS`.
fn cube_count_capped(total: u32) -> CubeCount {
    let cubes = total.div_ceil(CUBE_DIM).max(1);
    if cubes <= MAX_CUBES_PER_DIM {
        CubeCount::Static(cubes, 1, 1)
    } else {
        let y = cubes.div_ceil(MAX_CUBES_PER_DIM);
        CubeCount::Static(MAX_CUBES_PER_DIM, y, 1)
    }
}

/// Grid for the tiled matmul kernels: one cube per (row, CUBE_DIM-wide
/// column block) — X = column blocks, Y = rows. Both stay far under
/// [`MAX_CUBES_PER_DIM`] for real weights (n_out ≤ 262k → ≤ 1024 column
/// blocks; m is a prefill chunk).
fn cube_count_tiled(n_out: u32, m: u32) -> CubeCount {
    CubeCount::Static(n_out.div_ceil(CUBE_DIM).max(1), m.max(1), 1)
}

/// The tiled prefill kernels can be disabled with `COMBS_NO_TILED_MATMUL=1`
/// (runtime A/B comparisons and triage); checked once per process.
fn tiled_enabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    !*DISABLED.get_or_init(|| matches!(std::env::var("COMBS_NO_TILED_MATMUL").as_deref(), Ok("1")))
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
        q4_0_dequant_kernel::launch_unchecked::<f32, R>(
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

    /// Transient full-width f32 copy on device (`[n_out, k]` row-major)
    /// for the batched-matmul path — see the Q8_0 twin for the why.
    pub fn dequant_device<F: Float>(&self, client: &ComputeClient<R>) -> Handle {
        let n = self.n_out * self.k;
        let n_blocks = n / Q4_0_BLOCK;
        let out_h = client.empty(n * core::mem::size_of::<F>());
        unsafe {
            q4_0_dequant_kernel::launch_unchecked::<F, R>(
                client,
                cube_count_capped(n as u32),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(self.qs.clone(), n_blocks * 4),
                ArrayArg::from_raw_parts(self.d.clone(), n_blocks),
                ArrayArg::from_raw_parts(out_h.clone(), n),
                n,
            );
        }
        out_h
    }

    /// Device path: `y = x @ W^T` with `x` already resident as a contiguous
    /// f32 buffer of `[m, k]`. Launch only — returns the output handle
    /// (`[m, n_out]` f32) without any host round-trip.
    pub fn matmul_device(&self, client: &ComputeClient<R>, x: Handle, m: usize) -> Handle {
        let out_len = m * self.n_out;
        let out_h = client.empty(out_len * core::mem::size_of::<f32>());
        let n_blocks = self.n_out * self.k / Q4_0_BLOCK;
        unsafe {
            q4_0_matmul_kernel::launch_unchecked::<R>(
                client,
                cube_count_capped(out_len as u32),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(x, m * self.k),
                ArrayArg::from_raw_parts(self.qs.clone(), n_blocks * 4),
                ArrayArg::from_raw_parts(self.d.clone(), n_blocks),
                ArrayArg::from_raw_parts(out_h.clone(), out_len),
                m,
                self.k,
                self.n_out,
            );
        }
        out_h
    }

    /// `y = x @ W^T` for host-side `x: [m, k]`, returning `[m, n_out]`.
    /// Host-slice convenience for tests/CLI probes.
    pub fn matmul_host(&self, client: &ComputeClient<R>, x: &[f32], m: usize) -> Result<Vec<f32>> {
        if m == 0 || x.len() != m * self.k {
            return Err(ModelError::BadShape {
                tensor: "q4_0 matmul input".into(),
                expected: vec![m, self.k],
                got: vec![x.len()],
            });
        }
        let x_h = client.create_from_slice(f32::as_bytes(x));
        let out_h = self.matmul_device(client, x_h, m);
        let bytes = client.read_one_unchecked(out_h);
        Ok(f32::from_bytes(&bytes).to_vec())
    }
}

// ---------------------------------------------------------------------------
// Q5_0 / Q8_0 (32-value blocks) — the formats ggml falls back to for
// tensors whose row size is not a 256 multiple (e.g. SmolLM2's hidden 960),
// so a "Q4_K_M" file of such a model is mostly Q5_0 with Q8_0 embeddings.
// ---------------------------------------------------------------------------

/// Bytes per GGUF Q5_0 block: f16 scale + u32 high bits + 16 nibble bytes.
pub const Q5_0_BLOCK_BYTES: usize = 22;
/// Bytes per GGUF Q8_0 block: f16 scale + 32 i8 values.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Layout step for Q5_0: SoA of `(nibble words [4/blk], high-bit words
/// [1/blk], f32 scales)` — 24 B / 32 weights = 6.0 bits/weight.
pub fn repack_q5_0(data: &[u8]) -> Result<(Vec<u32>, Vec<u32>, Vec<f32>)> {
    if data.is_empty() || data.len() % Q5_0_BLOCK_BYTES != 0 {
        return Err(ModelError::BadShape {
            tensor: "q5_0 block stream".into(),
            expected: vec![Q5_0_BLOCK_BYTES],
            got: vec![data.len()],
        });
    }
    let n_blocks = data.len() / Q5_0_BLOCK_BYTES;
    let mut qs = Vec::with_capacity(n_blocks * 4);
    let mut qh = Vec::with_capacity(n_blocks);
    let mut d = Vec::with_capacity(n_blocks);
    for block in data.chunks_exact(Q5_0_BLOCK_BYTES) {
        d.push(burn::tensor::f16::from_le_bytes([block[0], block[1]]).to_f32());
        qh.push(u32::from_le_bytes([block[2], block[3], block[4], block[5]]));
        for w in 0..4 {
            let o = 6 + 4 * w;
            qs.push(u32::from_le_bytes([
                block[o],
                block[o + 1],
                block[o + 2],
                block[o + 3],
            ]));
        }
    }
    Ok((qs, qh, d))
}

/// Layout step for Q8_0: SoA of `(i8 words [8/blk], f32 scales)` —
/// 36 B / 32 weights = 9.0 bits/weight.
pub fn repack_q8_0(data: &[u8]) -> Result<(Vec<u32>, Vec<f32>)> {
    if data.is_empty() || data.len() % Q8_0_BLOCK_BYTES != 0 {
        return Err(ModelError::BadShape {
            tensor: "q8_0 block stream".into(),
            expected: vec![Q8_0_BLOCK_BYTES],
            got: vec![data.len()],
        });
    }
    let n_blocks = data.len() / Q8_0_BLOCK_BYTES;
    let mut qs = Vec::with_capacity(n_blocks * 8);
    let mut d = Vec::with_capacity(n_blocks);
    for block in data.chunks_exact(Q8_0_BLOCK_BYTES) {
        d.push(burn::tensor::f16::from_le_bytes([block[0], block[1]]).to_f32());
        for w in 0..8 {
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

/// Q5_0 dequant-only kernel, bit-exact mirror of the CPU reference:
/// `((nibble | high_bit«4) − 16) · d`, high bit `j` of the block's u32 for
/// value `j` (low nibbles), `j+16` for the highs.
#[cube(launch_unchecked)]
fn q5_0_dequant_kernel<F: Float>(
    qs: &Array<u32>,
    qh: &Array<u32>,
    d: &Array<f32>,
    out: &mut Array<F>,
    n: usize,
) {
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
        let hi_bit = (qh[block] >> u32::cast_from(j)) & 1;
        let q = i32::cast_from(nib | (hi_bit << 4)) - 16;
        out[ABSOLUTE_POS] = F::cast_from(f32::cast_from(q) * d[block]);
    }
}

/// Fused Q5_0 dequant-matmul (see `q4_0_matmul_kernel` for the scheme).
#[cube(launch_unchecked)]
fn q5_0_matmul_kernel(
    x: &Array<f32>,
    qs: &Array<u32>,
    qh: &Array<u32>,
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
            let bits = qh[block];
            let mut block_acc = 0.0f32;
            for w in 0..4usize {
                let word = qs[block * 4 + w];
                for b in 0..4usize {
                    let byte = (word >> (u32::cast_from(b) * 8)) & 0xFF;
                    let jj = w * 4 + b;
                    let lo_bit = (bits >> u32::cast_from(jj)) & 1;
                    let hi_bit = (bits >> u32::cast_from(jj + 16)) & 1;
                    let lo = f32::cast_from(i32::cast_from((byte & 0xF) | (lo_bit << 4)) - 16);
                    let hi = f32::cast_from(i32::cast_from((byte >> 4) | (hi_bit << 4)) - 16);
                    block_acc += lo * x[x_base + jj];
                    block_acc += hi * x[x_base + 16 + jj];
                }
            }
            acc += d[block] * block_acc;
        }
        out[row * n_out + col] = acc;
    }
}

/// Q8_0 dequant-only kernel: sign-extended i8 times the block scale.
#[cube(launch_unchecked)]
fn q8_0_dequant_kernel<F: Float>(qs: &Array<u32>, d: &Array<f32>, out: &mut Array<F>, n: usize) {
    if ABSOLUTE_POS < n {
        let block = ABSOLUTE_POS / 32;
        let j = ABSOLUTE_POS % 32;
        let word = qs[block * 8 + j / 4];
        let byte = (word >> (u32::cast_from(j % 4) * 8)) & 0xFF;
        let q = (i32::cast_from(byte) << 24) >> 24;
        out[ABSOLUTE_POS] = F::cast_from(f32::cast_from(q) * d[block]);
    }
}

/// Q8_0 dequant-gather: `out[t*k + c]` is column `c` of row `ids[t]`,
/// dequantized. The block arithmetic is byte-identical to
/// [`q8_0_dequant_kernel`] with a row indirection in front — which is
/// what keeps it bit-exact against the CPU reference. This is the
/// embedding lookup for a table kept packed in VRAM.
#[cube(launch_unchecked)]
fn q8_0_gather_kernel(
    ids: &Array<u32>,
    qs: &Array<u32>,
    d: &Array<f32>,
    out: &mut Array<f32>,
    total: usize,
    k: usize,
) {
    if ABSOLUTE_POS < total {
        let t = ABSOLUTE_POS / k;
        let c = ABSOLUTE_POS % k;
        let row = usize::cast_from(ids[t]);
        let block = row * (k / 32) + c / 32;
        let j = c % 32;
        let word = qs[block * 8 + j / 4];
        let byte = (word >> (u32::cast_from(j % 4) * 8)) & 0xFF;
        let q = (i32::cast_from(byte) << 24) >> 24;
        out[ABSOLUTE_POS] = f32::cast_from(q) * d[block];
    }
}

/// Fused Q8_0 dequant-matmul.
#[cube(launch_unchecked)]
fn q8_0_matmul_kernel(
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
            for w in 0..8usize {
                let word = qs[block * 8 + w];
                for b in 0..4usize {
                    let byte = (word >> (u32::cast_from(b) * 8)) & 0xFF;
                    let q = (i32::cast_from(byte) << 24) >> 24;
                    block_acc += f32::cast_from(q) * x[x_base + w * 4 + b];
                }
            }
            acc += d[block] * block_acc;
        }
        out[row * n_out + col] = acc;
    }
}

/// Tiled variant of [`q8_0_matmul_kernel`] for m > 1: one cube per
/// (row, CUBE_DIM-wide column block). Each 256-value k-tile of the row's
/// activations is staged in shared memory once by the whole cube instead of
/// being re-read from global memory by every column. Per-output arithmetic
/// (ascending k, per-block sum then one scale multiply) is identical to the
/// untiled kernel, so outputs are bit-identical; only the `x` load path
/// changes. Both barriers sit outside the column guard: every thread of the
/// cube reaches them even in a ragged final column block.
#[cube(launch_unchecked)]
fn q8_0_matmul_tiled_kernel(
    x: &Array<f32>,
    qs: &Array<u32>,
    d: &Array<f32>,
    out: &mut Array<f32>,
    k: usize,
    n_out: usize,
) {
    let mut staged = SharedMemory::<f32>::new(256usize);
    let unit = UNIT_POS as usize;
    let row = CUBE_POS_Y as usize;
    let col = (CUBE_POS_X * CUBE_DIM + UNIT_POS) as usize;
    let blocks_per_row = k / 32;
    let n_tiles = (k + 255) / 256;
    let mut acc = 0.0f32;
    for t in 0..n_tiles {
        let k0 = t * 256;
        if k0 + unit < k {
            staged[unit] = x[row * k + k0 + unit];
        }
        sync_cube();
        if col < n_out {
            let mut kb_end = (k0 + 256) / 32;
            if blocks_per_row < kb_end {
                kb_end = blocks_per_row;
            }
            for kb in (k0 / 32)..kb_end {
                let block = col * blocks_per_row + kb;
                let s_base = kb * 32 - k0;
                let mut block_acc = 0.0f32;
                for w in 0..8usize {
                    let word = qs[block * 8 + w];
                    for b in 0..4usize {
                        let byte = (word >> (u32::cast_from(b) * 8)) & 0xFF;
                        let q = (i32::cast_from(byte) << 24) >> 24;
                        block_acc += f32::cast_from(q) * staged[s_base + w * 4 + b];
                    }
                }
                acc += d[block] * block_acc;
            }
        }
        sync_cube();
    }
    if col < n_out {
        out[row * n_out + col] = acc;
    }
}

/// Runs the Q5_0 dequant-only kernel (validation/debugging path).
pub fn dequantize_q5_0_gpu<R: Runtime>(client: &ComputeClient<R>, data: &[u8]) -> Result<Vec<f32>> {
    let (qs, qh, d) = repack_q5_0(data)?;
    let n = d.len() * Q4_0_BLOCK;
    let qs_h = client.create_from_slice(u32::as_bytes(&qs));
    let qh_h = client.create_from_slice(u32::as_bytes(&qh));
    let d_h = client.create_from_slice(f32::as_bytes(&d));
    let out_h = client.empty(n * core::mem::size_of::<f32>());
    unsafe {
        q5_0_dequant_kernel::launch_unchecked::<f32, R>(
            client,
            cube_count_1d(n as u32),
            CubeDim::new_1d(CUBE_DIM),
            ArrayArg::from_raw_parts(qs_h, qs.len()),
            ArrayArg::from_raw_parts(qh_h, qh.len()),
            ArrayArg::from_raw_parts(d_h, d.len()),
            ArrayArg::from_raw_parts(out_h.clone(), n),
            n,
        );
    }
    let bytes = client.read_one_unchecked(out_h);
    Ok(f32::from_bytes(&bytes).to_vec())
}

/// Runs the Q8_0 dequant-only kernel (validation/debugging path).
pub fn dequantize_q8_0_gpu<R: Runtime>(client: &ComputeClient<R>, data: &[u8]) -> Result<Vec<f32>> {
    let (qs, d) = repack_q8_0(data)?;
    let n = d.len() * Q4_0_BLOCK;
    let qs_h = client.create_from_slice(u32::as_bytes(&qs));
    let d_h = client.create_from_slice(f32::as_bytes(&d));
    let out_h = client.empty(n * core::mem::size_of::<f32>());
    unsafe {
        q8_0_dequant_kernel::launch_unchecked::<f32, R>(
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

/// A weight matrix resident in VRAM in packed Q5_0 form (`[n_out, k]`,
/// `k % 32 == 0`, blocks along `k`).
pub struct Q50Weight<R: Runtime> {
    qs: Handle,
    qh: Handle,
    d: Handle,
    n_out: usize,
    k: usize,
    _runtime: PhantomData<R>,
}

impl<R: Runtime> Q50Weight<R> {
    /// Repacks a GGUF Q5_0 tensor onto the device.
    pub fn from_gguf_bytes(
        client: &ComputeClient<R>,
        data: &[u8],
        n_out: usize,
        k: usize,
    ) -> Result<Self> {
        if k == 0 || k % Q4_0_BLOCK != 0 || data.len() != n_out * k / Q4_0_BLOCK * Q5_0_BLOCK_BYTES
        {
            return Err(ModelError::BadShape {
                tensor: "q5_0 weight".into(),
                expected: vec![n_out, k],
                got: vec![data.len()],
            });
        }
        let (qs, qh, d) = repack_q5_0(data)?;
        Ok(Q50Weight {
            qs: client.create_from_slice(u32::as_bytes(&qs)),
            qh: client.create_from_slice(u32::as_bytes(&qh)),
            d: client.create_from_slice(f32::as_bytes(&d)),
            n_out,
            k,
            _runtime: PhantomData,
        })
    }

    /// Bytes in VRAM: 24 per 32 weights (6.0 bits/weight).
    pub fn vram_bytes(&self) -> usize {
        (self.n_out * self.k / Q4_0_BLOCK) * 24
    }

    /// Transient full-width f32 copy on device (`[n_out, k]` row-major)
    /// for the batched-matmul path — see the Q8_0 twin for the why.
    pub fn dequant_device<F: Float>(&self, client: &ComputeClient<R>) -> Handle {
        let n = self.n_out * self.k;
        let n_blocks = n / Q4_0_BLOCK;
        let out_h = client.empty(n * core::mem::size_of::<F>());
        unsafe {
            q5_0_dequant_kernel::launch_unchecked::<F, R>(
                client,
                cube_count_capped(n as u32),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(self.qs.clone(), n_blocks * 4),
                ArrayArg::from_raw_parts(self.qh.clone(), n_blocks),
                ArrayArg::from_raw_parts(self.d.clone(), n_blocks),
                ArrayArg::from_raw_parts(out_h.clone(), n),
                n,
            );
        }
        out_h
    }

    /// Device path: launch only, output handle returned.
    pub fn matmul_device(&self, client: &ComputeClient<R>, x: Handle, m: usize) -> Handle {
        let out_len = m * self.n_out;
        let out_h = client.empty(out_len * core::mem::size_of::<f32>());
        let n_blocks = self.n_out * self.k / Q4_0_BLOCK;
        unsafe {
            q5_0_matmul_kernel::launch_unchecked::<R>(
                client,
                cube_count_capped(out_len as u32),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(x, m * self.k),
                ArrayArg::from_raw_parts(self.qs.clone(), n_blocks * 4),
                ArrayArg::from_raw_parts(self.qh.clone(), n_blocks),
                ArrayArg::from_raw_parts(self.d.clone(), n_blocks),
                ArrayArg::from_raw_parts(out_h.clone(), out_len),
                m,
                self.k,
                self.n_out,
            );
        }
        out_h
    }

    /// Host-slice convenience for tests.
    pub fn matmul_host(&self, client: &ComputeClient<R>, x: &[f32], m: usize) -> Result<Vec<f32>> {
        if m == 0 || x.len() != m * self.k {
            return Err(ModelError::BadShape {
                tensor: "q5_0 matmul input".into(),
                expected: vec![m, self.k],
                got: vec![x.len()],
            });
        }
        let x_h = client.create_from_slice(f32::as_bytes(x));
        let out_h = self.matmul_device(client, x_h, m);
        let bytes = client.read_one_unchecked(out_h);
        Ok(f32::from_bytes(&bytes).to_vec())
    }
}

/// A weight matrix resident in VRAM in packed Q8_0 form (`[n_out, k]`,
/// `k % 32 == 0`, blocks along `k`).
pub struct Q80Weight<R: Runtime> {
    qs: Handle,
    d: Handle,
    n_out: usize,
    k: usize,
    _runtime: PhantomData<R>,
}

impl<R: Runtime> Q80Weight<R> {
    /// Repacks a GGUF Q8_0 tensor onto the device.
    pub fn from_gguf_bytes(
        client: &ComputeClient<R>,
        data: &[u8],
        n_out: usize,
        k: usize,
    ) -> Result<Self> {
        if k == 0 || k % Q4_0_BLOCK != 0 || data.len() != n_out * k / Q4_0_BLOCK * Q8_0_BLOCK_BYTES
        {
            return Err(ModelError::BadShape {
                tensor: "q8_0 weight".into(),
                expected: vec![n_out, k],
                got: vec![data.len()],
            });
        }
        let (qs, d) = repack_q8_0(data)?;
        Ok(Q80Weight {
            qs: client.create_from_slice(u32::as_bytes(&qs)),
            d: client.create_from_slice(f32::as_bytes(&d)),
            n_out,
            k,
            _runtime: PhantomData,
        })
    }

    /// Bytes in VRAM: 36 per 32 weights (9.0 bits/weight).
    pub fn vram_bytes(&self) -> usize {
        (self.n_out * self.k / Q4_0_BLOCK) * 36
    }

    /// Dequantize the whole weight into a transient f32 device buffer
    /// (`[n_out, k]` row-major). The batched-matmul path pays this once
    /// per call so the weight is READ once — the fused kernels re-read
    /// it per activation row, which is the §54 wall.
    pub fn dequant_device<F: Float>(&self, client: &ComputeClient<R>) -> Handle {
        let n = self.n_out * self.k;
        let n_blocks = n / Q4_0_BLOCK;
        let out_h = client.empty(n * core::mem::size_of::<F>());
        unsafe {
            q8_0_dequant_kernel::launch_unchecked::<F, R>(
                client,
                cube_count_capped(n as u32),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(self.qs.clone(), n_blocks * 8),
                ArrayArg::from_raw_parts(self.d.clone(), n_blocks),
                ArrayArg::from_raw_parts(out_h.clone(), n),
                n,
            );
        }
        out_h
    }

    /// Device path: launch only, output handle returned. Decode (`m == 1`)
    /// keeps the untiled kernel; prefill (`m > 1`) takes the shared-memory
    /// tiled kernel unless `COMBS_NO_TILED_MATMUL=1`.
    pub fn matmul_device(&self, client: &ComputeClient<R>, x: Handle, m: usize) -> Handle {
        self.matmul_device_with(client, x, m, m > 1 && tiled_enabled())
    }

    /// Launch with an explicit kernel choice (the parity tests compare
    /// tiled vs untiled on identical inputs).
    pub(crate) fn matmul_device_with(
        &self,
        client: &ComputeClient<R>,
        x: Handle,
        m: usize,
        tiled: bool,
    ) -> Handle {
        let out_len = m * self.n_out;
        let out_h = client.empty(out_len * core::mem::size_of::<f32>());
        let n_blocks = self.n_out * self.k / Q4_0_BLOCK;
        if tiled {
            unsafe {
                q8_0_matmul_tiled_kernel::launch_unchecked::<R>(
                    client,
                    cube_count_tiled(self.n_out as u32, m as u32),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(x, m * self.k),
                    ArrayArg::from_raw_parts(self.qs.clone(), n_blocks * 8),
                    ArrayArg::from_raw_parts(self.d.clone(), n_blocks),
                    ArrayArg::from_raw_parts(out_h.clone(), out_len),
                    self.k,
                    self.n_out,
                );
            }
        } else {
            unsafe {
                q8_0_matmul_kernel::launch_unchecked::<R>(
                    client,
                    cube_count_capped(out_len as u32),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(x, m * self.k),
                    ArrayArg::from_raw_parts(self.qs.clone(), n_blocks * 8),
                    ArrayArg::from_raw_parts(self.d.clone(), n_blocks),
                    ArrayArg::from_raw_parts(out_h.clone(), out_len),
                    m,
                    self.k,
                    self.n_out,
                );
            }
        }
        out_h
    }

    /// Split-K decode gemv through the Combs Kernel's WGSL path: 16
    /// lanes cooperate per output row instead of one thread walking it —
    /// 16x the occupancy on the projection shapes that dominate decode's
    /// weight reads. WgpuRuntime only (the WGSL seam's runtime); the
    /// caller routes m == 1 here behind the door and keeps the cube
    /// kernels for everything else.
    pub(crate) fn decode_gemv_wgsl(
        &self,
        client: &ComputeClient<cubecl::wgpu::WgpuRuntime>,
        x: Handle,
    ) -> Handle {
        let out_h = client.empty(self.n_out * core::mem::size_of::<f32>());
        crate::wgsl::launch_gemv(
            client,
            crate::wgsl::GemvKernel::Q8,
            (self.n_out as u32).div_ceil(16),
            vec![
                x.binding(),
                self.qs.clone().binding(),
                self.d.clone().binding(),
                out_h.clone().binding(),
            ],
            vec![self.n_out as u64, self.k as u64],
        );
        out_h
    }

    /// Dequantizes the rows named by `ids` (u32 device handle, `n_tokens`
    /// entries) into a dense `[n_tokens, k]` f32 buffer — the embedding
    /// lookup for a packed table. Row-for-row bit-exact with the dequant
    /// kernel, and therefore with the CPU reference.
    pub fn gather_rows_device(
        &self,
        client: &ComputeClient<R>,
        ids: Handle,
        n_tokens: usize,
    ) -> Handle {
        let total = n_tokens * self.k;
        let n_blocks = self.n_out * self.k / Q4_0_BLOCK;
        let out_h = client.empty(total * core::mem::size_of::<f32>());
        unsafe {
            q8_0_gather_kernel::launch_unchecked::<R>(
                client,
                cube_count_capped(total as u32),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(ids, n_tokens),
                ArrayArg::from_raw_parts(self.qs.clone(), n_blocks * 8),
                ArrayArg::from_raw_parts(self.d.clone(), n_blocks),
                ArrayArg::from_raw_parts(out_h.clone(), total),
                total,
                self.k,
            );
        }
        out_h
    }

    /// Host-slice convenience for tests.
    pub fn matmul_host(&self, client: &ComputeClient<R>, x: &[f32], m: usize) -> Result<Vec<f32>> {
        if m == 0 || x.len() != m * self.k {
            return Err(ModelError::BadShape {
                tensor: "q8_0 matmul input".into(),
                expected: vec![m, self.k],
                got: vec![x.len()],
            });
        }
        let x_h = client.create_from_slice(f32::as_bytes(x));
        let out_h = self.matmul_device(client, x_h, m);
        let bytes = client.read_one_unchecked(out_h);
        Ok(f32::from_bytes(&bytes).to_vec())
    }
}

// ---------------------------------------------------------------------------
// K-quants (256-value superblocks). Shared in-kernel byte helpers first.
// ---------------------------------------------------------------------------

/// Values per K-quant superblock.
pub const K_SUPERBLOCK: usize = 256;
/// Bytes per GGUF Q4_K superblock: f16 d + f16 dmin + 12B scales + 128B quants.
pub const Q4_K_BLOCK_BYTES: usize = 144;
/// Bytes per GGUF Q5_K superblock: Q4_K's layout + 32B high bits.
pub const Q5_K_BLOCK_BYTES: usize = 176;
/// Bytes per GGUF Q6_K superblock: 128B ql + 64B qh + 16 i8 scales + f16 d.
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// Reads byte `idx` from a byte stream stored as little-endian u32 words.
#[cube]
fn byte_at(words: &Array<u32>, idx: usize) -> u32 {
    (words[idx / 4] >> (u32::cast_from(idx % 4) * 8)) & 0xFF
}

/// Sign-extends byte `idx` of a word-packed stream as an i8.
#[cube]
fn i8_at(words: &Array<u32>, idx: usize) -> i32 {
    (i32::cast_from(byte_at(words, idx)) << 24) >> 24
}

/// Little-endian u32 assembled from four bytes of a word-packed stream,
/// starting at any (unaligned) byte offset. The staging repack reads
/// GGUF blocks of 18–210 bytes, none word-aligned, so every output word
/// crosses word boundaries somewhere.
#[cube]
fn word_at(words: &Array<u32>, byte: usize) -> u32 {
    byte_at(words, byte)
        | (byte_at(words, byte + 1) << 8)
        | (byte_at(words, byte + 2) << 16)
        | (byte_at(words, byte + 3) << 24)
}

/// Exact f16 → f32 widen from the raw bits, no float arithmetic that
/// could round: normals and subnormals become integer-valued floats
/// times a power of two (both exact in f32); inf/NaN map bitwise with
/// the mantissa shifted into place, which is the same widening
/// `half::f16::to_f32` performs on the host.
#[cube]
fn f16_bits_to_f32(h: u32) -> f32 {
    let sign = h & 0x8000;
    let exp = (h >> 10) & 0x1F;
    let mant = h & 0x3FF;
    // Normal: (1024 + mant) · 2^(exp − 25). Subnormal: mant · 2^−24 —
    // whose power constant is written `(exp + 103)` because exp is zero
    // on that arm, keeping every binding runtime-typed (a cube `mut`
    // initialized from a bare literal cannot take a runtime value later).
    let mut base = 1024 + mant;
    let mut pow_bits = (exp + 102) << 23;
    if exp == 0 {
        base = mant;
        pow_bits = (exp + 103) << 23;
    }
    let mut out = f32::cast_from(base) * f32::reinterpret(pow_bits);
    if sign != 0 {
        out = -out;
    }
    if exp == 31 {
        out = f32::reinterpret((sign << 16) | 0x7F80_0000 | (mant << 13));
    }
    out
}

/// Staging repack on the device: one thread per 18-byte Q4_0 block,
/// writing its four packed words and one widened scale — the same
/// layout `repack_q4_0` builds on the host, without the host copy.
#[cube(launch_unchecked)]
fn repack_q4_0_kernel(
    raw: &Array<u32>,
    qs: &mut Array<u32>,
    d: &mut Array<f32>,
    block0: usize,
    n_blocks: usize,
) {
    if ABSOLUTE_POS < n_blocks {
        let base = ABSOLUTE_POS * 18;
        let out = block0 + ABSOLUTE_POS;
        d[out] = f16_bits_to_f32(byte_at(raw, base) | (byte_at(raw, base + 1) << 8));
        for w in 0..4 {
            qs[out * 4 + w] = word_at(raw, base + 2 + 4 * w);
        }
    }
}

/// Q5_0: 22-byte block → 4 packed words, one high-bit word, one scale.
#[cube(launch_unchecked)]
fn repack_q5_0_kernel(
    raw: &Array<u32>,
    qs: &mut Array<u32>,
    qh: &mut Array<u32>,
    d: &mut Array<f32>,
    block0: usize,
    n_blocks: usize,
) {
    if ABSOLUTE_POS < n_blocks {
        let base = ABSOLUTE_POS * 22;
        let out = block0 + ABSOLUTE_POS;
        d[out] = f16_bits_to_f32(byte_at(raw, base) | (byte_at(raw, base + 1) << 8));
        qh[out] = word_at(raw, base + 2);
        for w in 0..4 {
            qs[out * 4 + w] = word_at(raw, base + 6 + 4 * w);
        }
    }
}

/// Q8_0: 34-byte block → 8 packed words, one scale.
#[cube(launch_unchecked)]
fn repack_q8_0_kernel(
    raw: &Array<u32>,
    qs: &mut Array<u32>,
    d: &mut Array<f32>,
    block0: usize,
    n_blocks: usize,
) {
    if ABSOLUTE_POS < n_blocks {
        let base = ABSOLUTE_POS * 34;
        let out = block0 + ABSOLUTE_POS;
        d[out] = f16_bits_to_f32(byte_at(raw, base) | (byte_at(raw, base + 1) << 8));
        for w in 0..8 {
            qs[out * 8 + w] = word_at(raw, base + 2 + 4 * w);
        }
    }
}

/// Q4_K: 144-byte superblock → 32 quant words, 2 widened scales, 3
/// packed scale words.
#[cube(launch_unchecked)]
fn repack_q4_k_kernel(
    raw: &Array<u32>,
    qs: &mut Array<u32>,
    dd: &mut Array<f32>,
    scales: &mut Array<u32>,
    block0: usize,
    n_sb: usize,
) {
    if ABSOLUTE_POS < n_sb {
        let base = ABSOLUTE_POS * 144;
        let out = block0 + ABSOLUTE_POS;
        dd[out * 2] = f16_bits_to_f32(byte_at(raw, base) | (byte_at(raw, base + 1) << 8));
        dd[out * 2 + 1] = f16_bits_to_f32(byte_at(raw, base + 2) | (byte_at(raw, base + 3) << 8));
        for w in 0..3 {
            scales[out * 3 + w] = word_at(raw, base + 4 + 4 * w);
        }
        for w in 0..32 {
            qs[out * 32 + w] = word_at(raw, base + 16 + 4 * w);
        }
    }
}

/// Q5_K: 176-byte superblock → 32 quant words, 8 high-bit words, 2
/// widened scales, 3 packed scale words.
#[cube(launch_unchecked)]
fn repack_q5_k_kernel(
    raw: &Array<u32>,
    qs: &mut Array<u32>,
    qh: &mut Array<u32>,
    dd: &mut Array<f32>,
    scales: &mut Array<u32>,
    block0: usize,
    n_sb: usize,
) {
    if ABSOLUTE_POS < n_sb {
        let base = ABSOLUTE_POS * 176;
        let out = block0 + ABSOLUTE_POS;
        dd[out * 2] = f16_bits_to_f32(byte_at(raw, base) | (byte_at(raw, base + 1) << 8));
        dd[out * 2 + 1] = f16_bits_to_f32(byte_at(raw, base + 2) | (byte_at(raw, base + 3) << 8));
        for w in 0..3 {
            scales[out * 3 + w] = word_at(raw, base + 4 + 4 * w);
        }
        for w in 0..8 {
            qh[out * 8 + w] = word_at(raw, base + 16 + 4 * w);
        }
        for w in 0..32 {
            qs[out * 32 + w] = word_at(raw, base + 48 + 4 * w);
        }
    }
}

/// Q6_K: 210-byte superblock → 32 low words, 16 high words, 4 scale
/// words, one widened scale.
#[cube(launch_unchecked)]
fn repack_q6_k_kernel(
    raw: &Array<u32>,
    ql: &mut Array<u32>,
    qh: &mut Array<u32>,
    sc: &mut Array<u32>,
    d: &mut Array<f32>,
    block0: usize,
    n_sb: usize,
) {
    if ABSOLUTE_POS < n_sb {
        let base = ABSOLUTE_POS * 210;
        let out = block0 + ABSOLUTE_POS;
        for w in 0..32 {
            ql[out * 32 + w] = word_at(raw, base + 4 * w);
        }
        for w in 0..16 {
            qh[out * 16 + w] = word_at(raw, base + 128 + 4 * w);
        }
        for w in 0..4 {
            sc[out * 4 + w] = word_at(raw, base + 192 + 4 * w);
        }
        d[out] = f16_bits_to_f32(byte_at(raw, base + 208) | (byte_at(raw, base + 209) << 8));
    }
}

/// Whether staging repacks on the device. In a tab the answer is
/// always yes — the host copy is exactly what the 4 GiB ceiling cannot
/// afford; natively the door opens with `COMBS_STAGE_REPACK=device`
/// until the identity sweep blesses a default flip.
pub(crate) fn stage_repack_device() -> bool {
    if cfg!(target_family = "wasm") {
        return true;
    }
    matches!(std::env::var("COMBS_STAGE_REPACK").as_deref(), Ok("device"))
}

/// Flush after each device repack: the raw upload's staging copy is
/// enqueued, not executed, and without a submit several tensors' copies
/// stack in the queue inside one drain — measured as the difference
/// between a mount that fits its gate and one that does not.
fn flush_repack<R: Runtime>(client: &ComputeClient<R>) {
    let _ = client.flush();
}

/// Slab size for the raw upload. One contiguous copy of a whole tensor
/// is exactly what a 4 GiB arena cannot afford — the first tab
/// measurement of a whole-tensor upload cost 413 MB of high-water over
/// the host path, an allocation-shape effect, not a volume one. Slabs
/// keep every host allocation bounded and let each staging copy be
/// released (flushed) before the next.
fn repack_slab_bytes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("COMBS_REPACK_SLAB")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n| *n > 0)
            .unwrap_or(64 << 20)
    })
}

/// Uploads a raw GGUF block stream in block-aligned slabs, each
/// word-padded so a kernel can read it as `Array<u32>`; yields
/// `(handle, first block, blocks in slab)` per slab.
fn upload_slabs<R: Runtime>(
    client: &ComputeClient<R>,
    data: &[u8],
    block_bytes: usize,
) -> Vec<(Handle, usize, usize)> {
    let blocks_per_slab = (repack_slab_bytes() / block_bytes).max(1);
    let slab_bytes = blocks_per_slab * block_bytes;
    data.chunks(slab_bytes)
        .enumerate()
        .map(|(i, chunk)| {
            let h = if chunk.len() % 4 == 0 {
                client.create_from_slice(chunk)
            } else {
                let mut padded = Vec::with_capacity(chunk.len() + 3);
                padded.extend_from_slice(chunk);
                padded.resize(chunk.len().div_ceil(4) * 4, 0);
                client.create_from_slice(&padded)
            };
            (h, i * blocks_per_slab, chunk.len() / block_bytes)
        })
        .collect()
}

macro_rules! device_shape_check {
    ($data:expr, $n_out:expr, $k:expr, $blk:expr, $blk_bytes:expr, $what:expr) => {
        if $k == 0 || $k % $blk != 0 || $data.len() != $n_out * $k / $blk * $blk_bytes {
            return Err(ModelError::BadShape {
                tensor: $what.into(),
                expected: vec![$n_out, $k],
                got: vec![$data.len()],
            });
        }
    };
}

impl<R: Runtime> Q40Weight<R> {
    /// [`Self::from_gguf_bytes`] with the repack on the device: the raw
    /// blocks go up once and the SoA layout is written by a kernel, so
    /// the host never holds a second copy of the tensor.
    pub fn from_gguf_bytes_device(
        client: &ComputeClient<R>,
        data: &[u8],
        n_out: usize,
        k: usize,
    ) -> Result<Self> {
        device_shape_check!(data, n_out, k, Q4_0_BLOCK, Q4_0_BLOCK_BYTES, "q4_0 weight");
        let n_blocks = data.len() / Q4_0_BLOCK_BYTES;
        let qs = client.empty(n_blocks * 4 * 4);
        let d = client.empty(n_blocks * 4);
        for (raw, block0, nblk) in upload_slabs(client, data, Q4_0_BLOCK_BYTES) {
            let raw_words = (nblk * Q4_0_BLOCK_BYTES).div_ceil(4);
            unsafe {
                repack_q4_0_kernel::launch_unchecked::<R>(
                    client,
                    cube_count_capped(nblk as u32),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(raw, raw_words),
                    ArrayArg::from_raw_parts(qs.clone(), n_blocks * 4),
                    ArrayArg::from_raw_parts(d.clone(), n_blocks),
                    block0,
                    nblk,
                );
            }
            // Release this slab's staging copy before the next uploads.
            flush_repack(client);
        }
        Ok(Q40Weight {
            qs,
            d,
            n_out,
            k,
            _runtime: PhantomData,
        })
    }
}

impl<R: Runtime> Q50Weight<R> {
    /// Device-side twin of [`Self::from_gguf_bytes`]; see `Q40Weight`.
    pub fn from_gguf_bytes_device(
        client: &ComputeClient<R>,
        data: &[u8],
        n_out: usize,
        k: usize,
    ) -> Result<Self> {
        device_shape_check!(data, n_out, k, Q4_0_BLOCK, Q5_0_BLOCK_BYTES, "q5_0 weight");
        let n_blocks = data.len() / Q5_0_BLOCK_BYTES;
        let qs = client.empty(n_blocks * 4 * 4);
        let qh = client.empty(n_blocks * 4);
        let d = client.empty(n_blocks * 4);
        for (raw, block0, nblk) in upload_slabs(client, data, Q5_0_BLOCK_BYTES) {
            let raw_words = (nblk * Q5_0_BLOCK_BYTES).div_ceil(4);
            unsafe {
                repack_q5_0_kernel::launch_unchecked::<R>(
                    client,
                    cube_count_capped(nblk as u32),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(raw, raw_words),
                    ArrayArg::from_raw_parts(qs.clone(), n_blocks * 4),
                    ArrayArg::from_raw_parts(qh.clone(), n_blocks),
                    ArrayArg::from_raw_parts(d.clone(), n_blocks),
                    block0,
                    nblk,
                );
            }
            // Release this slab's staging copy before the next uploads.
            flush_repack(client);
        }
        Ok(Q50Weight {
            qs,
            qh,
            d,
            n_out,
            k,
            _runtime: PhantomData,
        })
    }
}

impl<R: Runtime> Q80Weight<R> {
    /// Device-side twin of [`Self::from_gguf_bytes`]; see `Q40Weight`.
    pub fn from_gguf_bytes_device(
        client: &ComputeClient<R>,
        data: &[u8],
        n_out: usize,
        k: usize,
    ) -> Result<Self> {
        device_shape_check!(data, n_out, k, Q4_0_BLOCK, Q8_0_BLOCK_BYTES, "q8_0 weight");
        let n_blocks = data.len() / Q8_0_BLOCK_BYTES;
        let qs = client.empty(n_blocks * 8 * 4);
        let d = client.empty(n_blocks * 4);
        for (raw, block0, nblk) in upload_slabs(client, data, Q8_0_BLOCK_BYTES) {
            let raw_words = (nblk * Q8_0_BLOCK_BYTES).div_ceil(4);
            unsafe {
                repack_q8_0_kernel::launch_unchecked::<R>(
                    client,
                    cube_count_capped(nblk as u32),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(raw, raw_words),
                    ArrayArg::from_raw_parts(qs.clone(), n_blocks * 8),
                    ArrayArg::from_raw_parts(d.clone(), n_blocks),
                    block0,
                    nblk,
                );
            }
            // Release this slab's staging copy before the next uploads.
            flush_repack(client);
        }
        Ok(Q80Weight {
            qs,
            d,
            n_out,
            k,
            _runtime: PhantomData,
        })
    }
}

impl<R: Runtime> Q4KWeight<R> {
    /// Device-side twin of [`Self::from_gguf_bytes`]; see `Q40Weight`.
    pub fn from_gguf_bytes_device(
        client: &ComputeClient<R>,
        data: &[u8],
        n_out: usize,
        k: usize,
    ) -> Result<Self> {
        device_shape_check!(
            data,
            n_out,
            k,
            K_SUPERBLOCK,
            Q4_K_BLOCK_BYTES,
            "q4_k weight"
        );
        let n_sb = data.len() / Q4_K_BLOCK_BYTES;
        let qs = client.empty(n_sb * 32 * 4);
        let dd = client.empty(n_sb * 2 * 4);
        let scales = client.empty(n_sb * 3 * 4);
        for (raw, block0, nblk) in upload_slabs(client, data, Q4_K_BLOCK_BYTES) {
            let raw_words = (nblk * Q4_K_BLOCK_BYTES).div_ceil(4);
            unsafe {
                repack_q4_k_kernel::launch_unchecked::<R>(
                    client,
                    cube_count_capped(nblk as u32),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(raw, raw_words),
                    ArrayArg::from_raw_parts(qs.clone(), n_sb * 32),
                    ArrayArg::from_raw_parts(dd.clone(), n_sb * 2),
                    ArrayArg::from_raw_parts(scales.clone(), n_sb * 3),
                    block0,
                    nblk,
                );
            }
            // Release this slab's staging copy before the next uploads.
            flush_repack(client);
        }
        Ok(Q4KWeight {
            qs,
            dd,
            scales,
            n_out,
            k,
            _runtime: PhantomData,
        })
    }
}

impl<R: Runtime> Q5KWeight<R> {
    /// Device-side twin of [`Self::from_gguf_bytes`]; see `Q40Weight`.
    pub fn from_gguf_bytes_device(
        client: &ComputeClient<R>,
        data: &[u8],
        n_out: usize,
        k: usize,
    ) -> Result<Self> {
        device_shape_check!(
            data,
            n_out,
            k,
            K_SUPERBLOCK,
            Q5_K_BLOCK_BYTES,
            "q5_k weight"
        );
        let n_sb = data.len() / Q5_K_BLOCK_BYTES;
        let qs = client.empty(n_sb * 32 * 4);
        let qh = client.empty(n_sb * 8 * 4);
        let dd = client.empty(n_sb * 2 * 4);
        let scales = client.empty(n_sb * 3 * 4);
        for (raw, block0, nblk) in upload_slabs(client, data, Q5_K_BLOCK_BYTES) {
            let raw_words = (nblk * Q5_K_BLOCK_BYTES).div_ceil(4);
            unsafe {
                repack_q5_k_kernel::launch_unchecked::<R>(
                    client,
                    cube_count_capped(nblk as u32),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(raw, raw_words),
                    ArrayArg::from_raw_parts(qs.clone(), n_sb * 32),
                    ArrayArg::from_raw_parts(qh.clone(), n_sb * 8),
                    ArrayArg::from_raw_parts(dd.clone(), n_sb * 2),
                    ArrayArg::from_raw_parts(scales.clone(), n_sb * 3),
                    block0,
                    nblk,
                );
            }
            // Release this slab's staging copy before the next uploads.
            flush_repack(client);
        }
        Ok(Q5KWeight {
            qs,
            qh,
            dd,
            scales,
            n_out,
            k,
            _runtime: PhantomData,
        })
    }
}

impl<R: Runtime> Q6KWeight<R> {
    /// Device-side twin of [`Self::from_gguf_bytes`]; see `Q40Weight`.
    pub fn from_gguf_bytes_device(
        client: &ComputeClient<R>,
        data: &[u8],
        n_out: usize,
        k: usize,
    ) -> Result<Self> {
        device_shape_check!(
            data,
            n_out,
            k,
            K_SUPERBLOCK,
            Q6_K_BLOCK_BYTES,
            "q6_k weight"
        );
        let n_sb = data.len() / Q6_K_BLOCK_BYTES;
        let ql = client.empty(n_sb * 32 * 4);
        let qh = client.empty(n_sb * 16 * 4);
        let sc = client.empty(n_sb * 4 * 4);
        let d = client.empty(n_sb * 4);
        for (raw, block0, nblk) in upload_slabs(client, data, Q6_K_BLOCK_BYTES) {
            let raw_words = (nblk * Q6_K_BLOCK_BYTES).div_ceil(4);
            unsafe {
                repack_q6_k_kernel::launch_unchecked::<R>(
                    client,
                    cube_count_capped(nblk as u32),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(raw, raw_words),
                    ArrayArg::from_raw_parts(ql.clone(), n_sb * 32),
                    ArrayArg::from_raw_parts(qh.clone(), n_sb * 16),
                    ArrayArg::from_raw_parts(sc.clone(), n_sb * 4),
                    ArrayArg::from_raw_parts(d.clone(), n_sb),
                    block0,
                    nblk,
                );
            }
            // Release this slab's staging copy before the next uploads.
            flush_repack(client);
        }
        Ok(Q6KWeight {
            ql,
            qh,
            sc,
            d,
            n_out,
            k,
            _runtime: PhantomData,
        })
    }
}

/// ggml `get_scale_min_k4`, scale half: 6-bit scale of sub-block `j` from
/// the 12 packed bytes starting at `base` (top 2 bits of bytes 0..4 carry
/// the high bits of sub-blocks 4..8).
#[cube]
fn k4_scale(scales: &Array<u32>, base: usize, j: usize) -> u32 {
    let mut v = 0u32;
    if j < 4 {
        v = byte_at(scales, base + j) & 63;
    } else {
        v = (byte_at(scales, base + j + 4) & 0xF) | ((byte_at(scales, base + j - 4) >> 6) << 4);
    }
    v
}

/// ggml `get_scale_min_k4`, min half.
#[cube]
fn k4_min(scales: &Array<u32>, base: usize, j: usize) -> u32 {
    let mut v = 0u32;
    if j < 4 {
        v = byte_at(scales, base + j + 4) & 63;
    } else {
        v = (byte_at(scales, base + j + 4) >> 4) | ((byte_at(scales, base + j) >> 6) << 4);
    }
    v
}

/// Layout step for Q4_K: split each 144-byte superblock into SoA device
/// arrays — `(qs words, [d, dmin] f32 pairs, scale words)`. 148 B per 256
/// weights = 4.63 bits/weight (GGUF native is 4.5).
pub fn repack_q4_k(data: &[u8]) -> Result<(Vec<u32>, Vec<f32>, Vec<u32>)> {
    if data.is_empty() || data.len() % Q4_K_BLOCK_BYTES != 0 {
        return Err(ModelError::BadShape {
            tensor: "q4_k superblock stream".into(),
            expected: vec![Q4_K_BLOCK_BYTES],
            got: vec![data.len()],
        });
    }
    let n_sb = data.len() / Q4_K_BLOCK_BYTES;
    let mut qs = Vec::with_capacity(n_sb * 32);
    let mut dd = Vec::with_capacity(n_sb * 2);
    let mut scales = Vec::with_capacity(n_sb * 3);
    for sb in data.chunks_exact(Q4_K_BLOCK_BYTES) {
        dd.push(burn::tensor::f16::from_le_bytes([sb[0], sb[1]]).to_f32());
        dd.push(burn::tensor::f16::from_le_bytes([sb[2], sb[3]]).to_f32());
        for w in 0..3 {
            let o = 4 + 4 * w;
            scales.push(u32::from_le_bytes([sb[o], sb[o + 1], sb[o + 2], sb[o + 3]]));
        }
        for w in 0..32 {
            let o = 16 + 4 * w;
            qs.push(u32::from_le_bytes([sb[o], sb[o + 1], sb[o + 2], sb[o + 3]]));
        }
    }
    Ok((qs, dd, scales))
}

/// Q4_K dequant-only kernel, arithmetic mirrored from the CPU reference:
/// `out = (d·sc) · q - (dmin·m)` per 32-value sub-block.
#[cube(launch_unchecked)]
fn q4_k_dequant_kernel<F: Float>(
    qs: &Array<u32>,
    dd: &Array<f32>,
    scales: &Array<u32>,
    out: &mut Array<F>,
    n: usize,
) {
    if ABSOLUTE_POS < n {
        let sb = ABSOLUTE_POS / 256;
        let r = ABSOLUTE_POS % 256;
        let j = r / 64; // 64-value group: 32 low-nibble values then 32 high
        let t = (r % 64) / 32; // 0 = low nibble, 1 = high nibble
        let l = r % 32;
        let byte = byte_at(qs, sb * 128 + j * 32 + l);
        let mut q = byte & 0xF;
        if t == 1 {
            q = byte >> 4;
        }
        let sidx = 2 * j + t;
        let sc = k4_scale(scales, sb * 12, sidx);
        let mn = k4_min(scales, sb * 12, sidx);
        let d1 = dd[sb * 2] * f32::cast_from(sc);
        let fmin = dd[sb * 2 + 1] * f32::cast_from(mn);
        out[ABSOLUTE_POS] = F::cast_from(d1 * f32::cast_from(q) - fmin);
    }
}

/// Canary for the cubecl-compiled `tanh`: Metal misbehaves above 43.0
/// (NaN), and the compiler's safe-tanh workaround must be active on
/// every target that can end up lowering to Metal — including wasm via
/// Dawn, where a missing gate cost gemma3 its browser output (all-NaN
/// logits, endless token 0). The probe checks the VALUES.
#[cube(launch_unchecked)]
fn tanh_canary_kernel(x: &Array<f32>, out: &mut Array<f32>, n: usize) {
    if ABSOLUTE_POS < n {
        out[ABSOLUTE_POS] = f32::tanh(x[ABSOLUTE_POS]);
    }
}

/// Launches the tanh canary over `xs`; the caller reads the handle back.
pub(crate) fn tanh_canary_device<R: Runtime>(
    client: &ComputeClient<R>,
    xs: &[f32],
) -> cubecl::server::Handle {
    let n = xs.len();
    let x_h = client.create_from_slice(f32::as_bytes(xs));
    let out_h = client.empty(n * core::mem::size_of::<f32>());
    unsafe {
        tanh_canary_kernel::launch_unchecked::<R>(
            client,
            cube_count_1d(n as u32),
            CubeDim::new_1d(CUBE_DIM),
            ArrayArg::from_raw_parts(x_h, n),
            ArrayArg::from_raw_parts(out_h.clone(), n),
            n,
        );
    }
    out_h
}

/// Q4_K dequant-gather: `out[t·k + c]` is column `c` of row `ids[t]`,
/// dequantized — [`q4_k_dequant_kernel`]'s arithmetic with a row
/// indirection in front, which keeps it bit-exact vs the CPU reference.
#[cube(launch_unchecked)]
fn q4_k_gather_kernel(
    ids: &Array<u32>,
    qs: &Array<u32>,
    dd: &Array<f32>,
    scales: &Array<u32>,
    out: &mut Array<f32>,
    total: usize,
    k: usize,
) {
    if ABSOLUTE_POS < total {
        let t = ABSOLUTE_POS / k;
        let c = ABSOLUTE_POS % k;
        let row = usize::cast_from(ids[t]);
        let src = row * k + c;
        let sb = src / 256;
        let r = src % 256;
        let j = r / 64;
        let tt = (r % 64) / 32;
        let l = r % 32;
        let byte = byte_at(qs, sb * 128 + j * 32 + l);
        let mut q = byte & 0xF;
        if tt == 1 {
            q = byte >> 4;
        }
        let sidx = 2 * j + tt;
        let sc = k4_scale(scales, sb * 12, sidx);
        let mn = k4_min(scales, sb * 12, sidx);
        let d1 = dd[sb * 2] * f32::cast_from(sc);
        let fmin = dd[sb * 2 + 1] * f32::cast_from(mn);
        out[ABSOLUTE_POS] = d1 * f32::cast_from(q) - fmin;
    }
}

/// Fused Q4_K dequant-matmul. Uses the ggml sum-split: within a sub-block,
/// `Σ (d·sc·q − dmin·m)·x = d·sc·Σ q·x − dmin·m·Σ x`, so the packed bytes
/// are touched once and the scales applied once per 32 values.
#[cube(launch_unchecked)]
fn q4_k_matmul_kernel(
    x: &Array<f32>,
    qs: &Array<u32>,
    dd: &Array<f32>,
    scales: &Array<u32>,
    out: &mut Array<f32>,
    m: usize,
    k: usize,
    n_out: usize,
) {
    if ABSOLUTE_POS < m * n_out {
        let row = ABSOLUTE_POS / n_out;
        let col = ABSOLUTE_POS % n_out;
        let sb_per_row = k / 256;
        let mut acc = 0.0f32;
        for sbi in 0..sb_per_row {
            let sb = col * sb_per_row + sbi;
            let d = dd[sb * 2];
            let dmin = dd[sb * 2 + 1];
            let s_base = sb * 12;
            let x_base = row * k + sbi * 256;
            for j in 0..4usize {
                let mut sum_lo = 0.0f32;
                let mut sum_hi = 0.0f32;
                let mut xs_lo = 0.0f32;
                let mut xs_hi = 0.0f32;
                for w in 0..8usize {
                    let word = qs[sb * 32 + j * 8 + w];
                    for b in 0..4usize {
                        let byte = (word >> (u32::cast_from(b) * 8)) & 0xFF;
                        let l = 4 * w + b;
                        let x1 = x[x_base + 64 * j + l];
                        let x2 = x[x_base + 64 * j + 32 + l];
                        sum_lo += f32::cast_from(byte & 0xF) * x1;
                        sum_hi += f32::cast_from(byte >> 4) * x2;
                        xs_lo += x1;
                        xs_hi += x2;
                    }
                }
                let sc1 = f32::cast_from(k4_scale(scales, s_base, 2 * j));
                let mn1 = f32::cast_from(k4_min(scales, s_base, 2 * j));
                let sc2 = f32::cast_from(k4_scale(scales, s_base, 2 * j + 1));
                let mn2 = f32::cast_from(k4_min(scales, s_base, 2 * j + 1));
                acc += d * sc1 * sum_lo - dmin * mn1 * xs_lo;
                acc += d * sc2 * sum_hi - dmin * mn2 * xs_hi;
            }
        }
        out[row * n_out + col] = acc;
    }
}

/// Tiled variant of [`q4_k_matmul_kernel`] for m > 1: the 256-value
/// K-superblock is exactly one shared-memory tile (`k % 256 == 0` always
/// holds for K-quants, so there is no ragged tail). The cube stages the
/// superblock's activation slice once, barriers, and every column applies
/// the same sum-split in the same ascending-k order as the untiled kernel —
/// outputs are bit-identical; only the `x` load path changes.
#[cube(launch_unchecked)]
fn q4_k_matmul_tiled_kernel(
    x: &Array<f32>,
    qs: &Array<u32>,
    dd: &Array<f32>,
    scales: &Array<u32>,
    out: &mut Array<f32>,
    k: usize,
    n_out: usize,
) {
    let mut staged = SharedMemory::<f32>::new(256usize);
    let unit = UNIT_POS as usize;
    let row = CUBE_POS_Y as usize;
    let col = (CUBE_POS_X * CUBE_DIM + UNIT_POS) as usize;
    let sb_per_row = k / 256;
    let mut acc = 0.0f32;
    for sbi in 0..sb_per_row {
        staged[unit] = x[row * k + sbi * 256 + unit];
        sync_cube();
        if col < n_out {
            let sb = col * sb_per_row + sbi;
            let d = dd[sb * 2];
            let dmin = dd[sb * 2 + 1];
            let s_base = sb * 12;
            for j in 0..4usize {
                let mut sum_lo = 0.0f32;
                let mut sum_hi = 0.0f32;
                let mut xs_lo = 0.0f32;
                let mut xs_hi = 0.0f32;
                for w in 0..8usize {
                    let word = qs[sb * 32 + j * 8 + w];
                    for b in 0..4usize {
                        let byte = (word >> (u32::cast_from(b) * 8)) & 0xFF;
                        let l = 4 * w + b;
                        let x1 = staged[64 * j + l];
                        let x2 = staged[64 * j + 32 + l];
                        sum_lo += f32::cast_from(byte & 0xF) * x1;
                        sum_hi += f32::cast_from(byte >> 4) * x2;
                        xs_lo += x1;
                        xs_hi += x2;
                    }
                }
                let sc1 = f32::cast_from(k4_scale(scales, s_base, 2 * j));
                let mn1 = f32::cast_from(k4_min(scales, s_base, 2 * j));
                let sc2 = f32::cast_from(k4_scale(scales, s_base, 2 * j + 1));
                let mn2 = f32::cast_from(k4_min(scales, s_base, 2 * j + 1));
                acc += d * sc1 * sum_lo - dmin * mn1 * xs_lo;
                acc += d * sc2 * sum_hi - dmin * mn2 * xs_hi;
            }
        }
        sync_cube();
    }
    if col < n_out {
        out[row * n_out + col] = acc;
    }
}

/// Layout step for Q5_K: split each 176-byte superblock into SoA device
/// arrays — `(qs words, qh words, [d, dmin] f32 pairs, scale words)`.
/// 180 B per 256 weights = 5.63 bits/weight (GGUF native is 5.5).
pub fn repack_q5_k(data: &[u8]) -> Result<(Vec<u32>, Vec<u32>, Vec<f32>, Vec<u32>)> {
    if data.is_empty() || data.len() % Q5_K_BLOCK_BYTES != 0 {
        return Err(ModelError::BadShape {
            tensor: "q5_k superblock stream".into(),
            expected: vec![Q5_K_BLOCK_BYTES],
            got: vec![data.len()],
        });
    }
    let n_sb = data.len() / Q5_K_BLOCK_BYTES;
    let word = |sb: &[u8], o: usize| u32::from_le_bytes([sb[o], sb[o + 1], sb[o + 2], sb[o + 3]]);
    let mut qs = Vec::with_capacity(n_sb * 32);
    let mut qh = Vec::with_capacity(n_sb * 8);
    let mut dd = Vec::with_capacity(n_sb * 2);
    let mut scales = Vec::with_capacity(n_sb * 3);
    for sb in data.chunks_exact(Q5_K_BLOCK_BYTES) {
        dd.push(burn::tensor::f16::from_le_bytes([sb[0], sb[1]]).to_f32());
        dd.push(burn::tensor::f16::from_le_bytes([sb[2], sb[3]]).to_f32());
        for w in 0..3 {
            scales.push(word(sb, 4 + 4 * w));
        }
        for w in 0..8 {
            qh.push(word(sb, 16 + 4 * w));
        }
        for w in 0..32 {
            qs.push(word(sb, 48 + 4 * w));
        }
    }
    Ok((qs, qh, dd, scales))
}

/// Q5_K dequant-only kernel, arithmetic mirrored from the CPU reference:
/// Q4_K plus the high-bit plane — group `j` reads bit `2j + t` of `qh[l]`
/// and the value is `(d·sc) · (nib | hi«4) - (dmin·m)`.
#[cube(launch_unchecked)]
fn q5_k_dequant_kernel<F: Float>(
    qs: &Array<u32>,
    qh: &Array<u32>,
    dd: &Array<f32>,
    scales: &Array<u32>,
    out: &mut Array<F>,
    n: usize,
) {
    if ABSOLUTE_POS < n {
        let sb = ABSOLUTE_POS / 256;
        let r = ABSOLUTE_POS % 256;
        let j = r / 64; // 64-value group: 32 low-nibble values then 32 high
        let t = (r % 64) / 32; // 0 = low nibble, 1 = high nibble
        let l = r % 32;
        let byte = byte_at(qs, sb * 128 + j * 32 + l);
        let mut nib = byte & 0xF;
        if t == 1 {
            nib = byte >> 4;
        }
        let hi = (byte_at(qh, sb * 32 + l) >> u32::cast_from(2 * j + t)) & 1;
        let sidx = 2 * j + t;
        let sc = k4_scale(scales, sb * 12, sidx);
        let mn = k4_min(scales, sb * 12, sidx);
        let d1 = dd[sb * 2] * f32::cast_from(sc);
        let fmin = dd[sb * 2 + 1] * f32::cast_from(mn);
        out[ABSOLUTE_POS] = F::cast_from(d1 * f32::cast_from(nib | (hi << 4)) - fmin);
    }
}

/// Fused Q5_K dequant-matmul: the Q4_K sum-split with the 5th bit folded
/// into `q` before the multiply.
#[cube(launch_unchecked)]
fn q5_k_matmul_kernel(
    x: &Array<f32>,
    qs: &Array<u32>,
    qh: &Array<u32>,
    dd: &Array<f32>,
    scales: &Array<u32>,
    out: &mut Array<f32>,
    m: usize,
    k: usize,
    n_out: usize,
) {
    if ABSOLUTE_POS < m * n_out {
        let row = ABSOLUTE_POS / n_out;
        let col = ABSOLUTE_POS % n_out;
        let sb_per_row = k / 256;
        let mut acc = 0.0f32;
        for sbi in 0..sb_per_row {
            let sb = col * sb_per_row + sbi;
            let d = dd[sb * 2];
            let dmin = dd[sb * 2 + 1];
            let s_base = sb * 12;
            let x_base = row * k + sbi * 256;
            for j in 0..4usize {
                let mut sum_lo = 0.0f32;
                let mut sum_hi = 0.0f32;
                let mut xs_lo = 0.0f32;
                let mut xs_hi = 0.0f32;
                for w in 0..8usize {
                    let word = qs[sb * 32 + j * 8 + w];
                    for b in 0..4usize {
                        let byte = (word >> (u32::cast_from(b) * 8)) & 0xFF;
                        let l = 4 * w + b;
                        let hb = byte_at(qh, sb * 32 + l);
                        let hi_lo = (hb >> u32::cast_from(2 * j)) & 1;
                        let hi_hi = (hb >> u32::cast_from(2 * j + 1)) & 1;
                        let x1 = x[x_base + 64 * j + l];
                        let x2 = x[x_base + 64 * j + 32 + l];
                        sum_lo += f32::cast_from((byte & 0xF) | (hi_lo << 4)) * x1;
                        sum_hi += f32::cast_from((byte >> 4) | (hi_hi << 4)) * x2;
                        xs_lo += x1;
                        xs_hi += x2;
                    }
                }
                let sc1 = f32::cast_from(k4_scale(scales, s_base, 2 * j));
                let mn1 = f32::cast_from(k4_min(scales, s_base, 2 * j));
                let sc2 = f32::cast_from(k4_scale(scales, s_base, 2 * j + 1));
                let mn2 = f32::cast_from(k4_min(scales, s_base, 2 * j + 1));
                acc += d * sc1 * sum_lo - dmin * mn1 * xs_lo;
                acc += d * sc2 * sum_hi - dmin * mn2 * xs_hi;
            }
        }
        out[row * n_out + col] = acc;
    }
}

/// Layout step for Q6_K: split each 210-byte superblock into SoA device
/// arrays — `(ql words, qh words, i8 scale words, d f32)`. 212 B per 256
/// weights = 6.63 bits/weight (GGUF native is 6.56).
pub fn repack_q6_k(data: &[u8]) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<f32>)> {
    if data.is_empty() || data.len() % Q6_K_BLOCK_BYTES != 0 {
        return Err(ModelError::BadShape {
            tensor: "q6_k superblock stream".into(),
            expected: vec![Q6_K_BLOCK_BYTES],
            got: vec![data.len()],
        });
    }
    let n_sb = data.len() / Q6_K_BLOCK_BYTES;
    let word = |sb: &[u8], o: usize| u32::from_le_bytes([sb[o], sb[o + 1], sb[o + 2], sb[o + 3]]);
    let mut ql = Vec::with_capacity(n_sb * 32);
    let mut qh = Vec::with_capacity(n_sb * 16);
    let mut sc = Vec::with_capacity(n_sb * 4);
    let mut d = Vec::with_capacity(n_sb);
    for sb in data.chunks_exact(Q6_K_BLOCK_BYTES) {
        for w in 0..32 {
            ql.push(word(sb, 4 * w));
        }
        for w in 0..16 {
            qh.push(word(sb, 128 + 4 * w));
        }
        for w in 0..4 {
            sc.push(word(sb, 192 + 4 * w));
        }
        d.push(burn::tensor::f16::from_le_bytes([sb[208], sb[209]]).to_f32());
    }
    Ok((ql, qh, sc, d))
}

/// Q6_K dequant-only kernel, mirroring the CPU reference: each 128-value
/// half yields quadrants t 0..4 with `q = (ql nibble) | (qh 2-bit « 4)`,
/// biased −32, times `d · scales[i8]`.
#[cube(launch_unchecked)]
fn q6_k_dequant_kernel<F: Float>(
    ql: &Array<u32>,
    qh: &Array<u32>,
    sc: &Array<u32>,
    d: &Array<f32>,
    out: &mut Array<F>,
    n: usize,
) {
    if ABSOLUTE_POS < n {
        let sb = ABSOLUTE_POS / 256;
        let r = ABSOLUTE_POS % 256;
        let half = r / 128;
        let t = (r % 128) / 32; // quadrant within the half
        let l = r % 32;
        let ql_byte = byte_at(ql, sb * 128 + half * 64 + (t % 2) * 32 + l);
        let mut nib = ql_byte & 0xF;
        if t >= 2 {
            nib = ql_byte >> 4;
        }
        let hi = (byte_at(qh, sb * 64 + half * 32 + l) >> (u32::cast_from(t) * 2)) & 3;
        let q = i32::cast_from(nib | (hi << 4)) - 32;
        let scale = i8_at(sc, sb * 16 + half * 8 + l / 16 + 2 * t);
        out[ABSOLUTE_POS] = F::cast_from(d[sb] * f32::cast_from(scale) * f32::cast_from(q));
    }
}

/// Q6_K dequant-gather: [`q6_k_dequant_kernel`]'s arithmetic with a row
/// indirection in front — bit-exact vs the CPU reference, per row.
#[cube(launch_unchecked)]
fn q6_k_gather_kernel(
    ids: &Array<u32>,
    ql: &Array<u32>,
    qh: &Array<u32>,
    sc: &Array<u32>,
    d: &Array<f32>,
    out: &mut Array<f32>,
    total: usize,
    k: usize,
) {
    if ABSOLUTE_POS < total {
        let t = ABSOLUTE_POS / k;
        let c = ABSOLUTE_POS % k;
        let row = usize::cast_from(ids[t]);
        let src = row * k + c;
        let sb = src / 256;
        let r = src % 256;
        let half = r / 128;
        let tt = (r % 128) / 32;
        let l = r % 32;
        let ql_byte = byte_at(ql, sb * 128 + half * 64 + (tt % 2) * 32 + l);
        let mut nib = ql_byte & 0xF;
        if tt >= 2 {
            nib = ql_byte >> 4;
        }
        let hi = (byte_at(qh, sb * 64 + half * 32 + l) >> (u32::cast_from(tt) * 2)) & 3;
        let q = i32::cast_from(nib | (hi << 4)) - 32;
        let scale = i8_at(sc, sb * 16 + half * 8 + l / 16 + 2 * tt);
        out[ABSOLUTE_POS] = d[sb] * f32::cast_from(scale) * f32::cast_from(q);
    }
}

/// Fused Q6_K dequant-matmul: per 16-value scale group,
/// `acc += d · sc · Σ (q − 32) · x`.
#[cube(launch_unchecked)]
fn q6_k_matmul_kernel(
    x: &Array<f32>,
    ql: &Array<u32>,
    qh: &Array<u32>,
    sc: &Array<u32>,
    d: &Array<f32>,
    out: &mut Array<f32>,
    m: usize,
    k: usize,
    n_out: usize,
) {
    if ABSOLUTE_POS < m * n_out {
        let row = ABSOLUTE_POS / n_out;
        let col = ABSOLUTE_POS % n_out;
        let sb_per_row = k / 256;
        let mut acc = 0.0f32;
        for sbi in 0..sb_per_row {
            let sb = col * sb_per_row + sbi;
            let dsb = d[sb];
            let x_base = row * k + sbi * 256;
            for half in 0..2usize {
                for t in 0..4usize {
                    for g in 0..2usize {
                        let mut sum = 0.0f32;
                        for l0 in 0..16usize {
                            let l = g * 16 + l0;
                            let ql_byte = byte_at(ql, sb * 128 + half * 64 + (t % 2) * 32 + l);
                            let mut nib = ql_byte & 0xF;
                            if t >= 2 {
                                nib = ql_byte >> 4;
                            }
                            let hi = (byte_at(qh, sb * 64 + half * 32 + l)
                                >> (u32::cast_from(t) * 2))
                                & 3;
                            let q = i32::cast_from(nib | (hi << 4)) - 32;
                            sum += f32::cast_from(q) * x[x_base + half * 128 + t * 32 + l];
                        }
                        let scale = i8_at(sc, sb * 16 + half * 8 + g + 2 * t);
                        acc += dsb * f32::cast_from(scale) * sum;
                    }
                }
            }
        }
        out[row * n_out + col] = acc;
    }
}

/// Runs the Q4_K dequant-only kernel (validation/debugging path).
pub fn dequantize_q4_k_gpu<R: Runtime>(client: &ComputeClient<R>, data: &[u8]) -> Result<Vec<f32>> {
    let (qs, dd, scales) = repack_q4_k(data)?;
    let n = (dd.len() / 2) * K_SUPERBLOCK;
    let qs_h = client.create_from_slice(u32::as_bytes(&qs));
    let dd_h = client.create_from_slice(f32::as_bytes(&dd));
    let sc_h = client.create_from_slice(u32::as_bytes(&scales));
    let out_h = client.empty(n * core::mem::size_of::<f32>());
    unsafe {
        q4_k_dequant_kernel::launch_unchecked::<f32, R>(
            client,
            cube_count_1d(n as u32),
            CubeDim::new_1d(CUBE_DIM),
            ArrayArg::from_raw_parts(qs_h, qs.len()),
            ArrayArg::from_raw_parts(dd_h, dd.len()),
            ArrayArg::from_raw_parts(sc_h, scales.len()),
            ArrayArg::from_raw_parts(out_h.clone(), n),
            n,
        );
    }
    let bytes = client.read_one_unchecked(out_h);
    Ok(f32::from_bytes(&bytes).to_vec())
}

/// Runs the Q5_K dequant-only kernel (validation/debugging path).
pub fn dequantize_q5_k_gpu<R: Runtime>(client: &ComputeClient<R>, data: &[u8]) -> Result<Vec<f32>> {
    let (qs, qh, dd, scales) = repack_q5_k(data)?;
    let n = (dd.len() / 2) * K_SUPERBLOCK;
    let qs_h = client.create_from_slice(u32::as_bytes(&qs));
    let qh_h = client.create_from_slice(u32::as_bytes(&qh));
    let dd_h = client.create_from_slice(f32::as_bytes(&dd));
    let sc_h = client.create_from_slice(u32::as_bytes(&scales));
    let out_h = client.empty(n * core::mem::size_of::<f32>());
    unsafe {
        q5_k_dequant_kernel::launch_unchecked::<f32, R>(
            client,
            cube_count_1d(n as u32),
            CubeDim::new_1d(CUBE_DIM),
            ArrayArg::from_raw_parts(qs_h, qs.len()),
            ArrayArg::from_raw_parts(qh_h, qh.len()),
            ArrayArg::from_raw_parts(dd_h, dd.len()),
            ArrayArg::from_raw_parts(sc_h, scales.len()),
            ArrayArg::from_raw_parts(out_h.clone(), n),
            n,
        );
    }
    let bytes = client.read_one_unchecked(out_h);
    Ok(f32::from_bytes(&bytes).to_vec())
}

/// Runs the Q6_K dequant-only kernel (validation/debugging path).
pub fn dequantize_q6_k_gpu<R: Runtime>(client: &ComputeClient<R>, data: &[u8]) -> Result<Vec<f32>> {
    let (ql, qh, sc, d) = repack_q6_k(data)?;
    let n = d.len() * K_SUPERBLOCK;
    let ql_h = client.create_from_slice(u32::as_bytes(&ql));
    let qh_h = client.create_from_slice(u32::as_bytes(&qh));
    let sc_h = client.create_from_slice(u32::as_bytes(&sc));
    let d_h = client.create_from_slice(f32::as_bytes(&d));
    let out_h = client.empty(n * core::mem::size_of::<f32>());
    unsafe {
        q6_k_dequant_kernel::launch_unchecked::<f32, R>(
            client,
            cube_count_1d(n as u32),
            CubeDim::new_1d(CUBE_DIM),
            ArrayArg::from_raw_parts(ql_h, ql.len()),
            ArrayArg::from_raw_parts(qh_h, qh.len()),
            ArrayArg::from_raw_parts(sc_h, sc.len()),
            ArrayArg::from_raw_parts(d_h, d.len()),
            ArrayArg::from_raw_parts(out_h.clone(), n),
            n,
        );
    }
    let bytes = client.read_one_unchecked(out_h);
    Ok(f32::from_bytes(&bytes).to_vec())
}

/// A weight matrix resident in VRAM in packed Q4_K form (`[n_out, k]`,
/// `k % 256 == 0`, superblocks along `k`).
pub struct Q4KWeight<R: Runtime> {
    qs: Handle,
    dd: Handle,
    scales: Handle,
    n_out: usize,
    k: usize,
    _runtime: PhantomData<R>,
}

impl<R: Runtime> Q4KWeight<R> {
    /// Repacks a GGUF Q4_K tensor onto the device.
    pub fn from_gguf_bytes(
        client: &ComputeClient<R>,
        data: &[u8],
        n_out: usize,
        k: usize,
    ) -> Result<Self> {
        if k == 0
            || k % K_SUPERBLOCK != 0
            || data.len() != n_out * k / K_SUPERBLOCK * Q4_K_BLOCK_BYTES
        {
            return Err(ModelError::BadShape {
                tensor: "q4_k weight".into(),
                expected: vec![n_out, k],
                got: vec![data.len()],
            });
        }
        let (qs, dd, scales) = repack_q4_k(data)?;
        Ok(Q4KWeight {
            qs: client.create_from_slice(u32::as_bytes(&qs)),
            dd: client.create_from_slice(f32::as_bytes(&dd)),
            scales: client.create_from_slice(u32::as_bytes(&scales)),
            n_out,
            k,
            _runtime: PhantomData,
        })
    }

    /// Transient full-width f32 copy on device (`[n_out, k]` row-major)
    /// for the batched-matmul path — see `Q80Weight::dequant_device`.
    pub fn dequant_device<F: Float>(&self, client: &ComputeClient<R>) -> Handle {
        let n = self.n_out * self.k;
        let n_sb = n / K_SUPERBLOCK;
        let out_h = client.empty(n * core::mem::size_of::<F>());
        unsafe {
            q4_k_dequant_kernel::launch_unchecked::<F, R>(
                client,
                cube_count_capped(n as u32),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(self.qs.clone(), n_sb * 32),
                ArrayArg::from_raw_parts(self.dd.clone(), n_sb * 2),
                ArrayArg::from_raw_parts(self.scales.clone(), n_sb * 3),
                ArrayArg::from_raw_parts(out_h.clone(), n),
                n,
            );
        }
        out_h
    }

    /// Bytes in VRAM: 148 per 256 weights (4.63 bits/weight).
    pub fn vram_bytes(&self) -> usize {
        (self.n_out * self.k / K_SUPERBLOCK) * (128 + 12 + 8)
    }

    /// Split-K decode gemv through the Combs Kernel — see
    /// `Q80Weight::decode_gemv_wgsl`. `k % 256 == 0` (whole superblocks)
    /// is guaranteed by construction for Q4_K tables.
    pub(crate) fn decode_gemv_wgsl(
        &self,
        client: &ComputeClient<cubecl::wgpu::WgpuRuntime>,
        x: Handle,
    ) -> Handle {
        let out_h = client.empty(self.n_out * core::mem::size_of::<f32>());
        crate::wgsl::launch_gemv(
            client,
            crate::wgsl::GemvKernel::Q4K,
            (self.n_out as u32).div_ceil(16),
            vec![
                x.binding(),
                self.qs.clone().binding(),
                self.dd.clone().binding(),
                self.scales.clone().binding(),
                out_h.clone().binding(),
            ],
            vec![self.n_out as u64, self.k as u64],
        );
        out_h
    }

    /// Dequantizes the rows named by `ids` into `[n_tokens, k]` f32 —
    /// the embedding lookup for a packed Q4_K table, row-for-row
    /// bit-exact with the dequant kernel.
    pub fn gather_rows_device(
        &self,
        client: &ComputeClient<R>,
        ids: Handle,
        n_tokens: usize,
    ) -> Handle {
        let total = n_tokens * self.k;
        let n_sb = self.n_out * self.k / K_SUPERBLOCK;
        let out_h = client.empty(total * core::mem::size_of::<f32>());
        unsafe {
            q4_k_gather_kernel::launch_unchecked::<R>(
                client,
                cube_count_capped(total as u32),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(ids, n_tokens),
                ArrayArg::from_raw_parts(self.qs.clone(), n_sb * 32),
                ArrayArg::from_raw_parts(self.dd.clone(), n_sb * 2),
                ArrayArg::from_raw_parts(self.scales.clone(), n_sb * 3),
                ArrayArg::from_raw_parts(out_h.clone(), total),
                total,
                self.k,
            );
        }
        out_h
    }

    /// Device path: launch only, output handle returned. Decode (`m == 1`)
    /// keeps the untiled kernel; prefill (`m > 1`) takes the shared-memory
    /// tiled kernel unless `COMBS_NO_TILED_MATMUL=1`.
    pub fn matmul_device(&self, client: &ComputeClient<R>, x: Handle, m: usize) -> Handle {
        self.matmul_device_with(client, x, m, m > 1 && tiled_enabled())
    }

    /// Launch with an explicit kernel choice (the parity tests compare
    /// tiled vs untiled on identical inputs).
    pub(crate) fn matmul_device_with(
        &self,
        client: &ComputeClient<R>,
        x: Handle,
        m: usize,
        tiled: bool,
    ) -> Handle {
        let out_len = m * self.n_out;
        let out_h = client.empty(out_len * core::mem::size_of::<f32>());
        let n_sb = self.n_out * self.k / K_SUPERBLOCK;
        if tiled {
            unsafe {
                q4_k_matmul_tiled_kernel::launch_unchecked::<R>(
                    client,
                    cube_count_tiled(self.n_out as u32, m as u32),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(x, m * self.k),
                    ArrayArg::from_raw_parts(self.qs.clone(), n_sb * 32),
                    ArrayArg::from_raw_parts(self.dd.clone(), n_sb * 2),
                    ArrayArg::from_raw_parts(self.scales.clone(), n_sb * 3),
                    ArrayArg::from_raw_parts(out_h.clone(), out_len),
                    self.k,
                    self.n_out,
                );
            }
        } else {
            unsafe {
                q4_k_matmul_kernel::launch_unchecked::<R>(
                    client,
                    cube_count_capped(out_len as u32),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(x, m * self.k),
                    ArrayArg::from_raw_parts(self.qs.clone(), n_sb * 32),
                    ArrayArg::from_raw_parts(self.dd.clone(), n_sb * 2),
                    ArrayArg::from_raw_parts(self.scales.clone(), n_sb * 3),
                    ArrayArg::from_raw_parts(out_h.clone(), out_len),
                    m,
                    self.k,
                    self.n_out,
                );
            }
        }
        out_h
    }

    /// `y = x @ W^T` for host-side `x: [m, k]`, returning `[m, n_out]`.
    pub fn matmul_host(&self, client: &ComputeClient<R>, x: &[f32], m: usize) -> Result<Vec<f32>> {
        if m == 0 || x.len() != m * self.k {
            return Err(ModelError::BadShape {
                tensor: "q4_k matmul input".into(),
                expected: vec![m, self.k],
                got: vec![x.len()],
            });
        }
        let x_h = client.create_from_slice(f32::as_bytes(x));
        let out_h = self.matmul_device(client, x_h, m);
        let bytes = client.read_one_unchecked(out_h);
        Ok(f32::from_bytes(&bytes).to_vec())
    }
}

/// A weight matrix resident in VRAM in packed Q6_K form (`[n_out, k]`,
/// `k % 256 == 0`, superblocks along `k`).
/// A weight matrix resident in VRAM in packed Q5_K form (`[n_out, k]`,
/// `k % 256 == 0`, superblocks along `k`).
pub struct Q5KWeight<R: Runtime> {
    qs: Handle,
    qh: Handle,
    dd: Handle,
    scales: Handle,
    n_out: usize,
    k: usize,
    _runtime: PhantomData<R>,
}

impl<R: Runtime> Q5KWeight<R> {
    /// Repacks a GGUF Q5_K tensor onto the device.
    pub fn from_gguf_bytes(
        client: &ComputeClient<R>,
        data: &[u8],
        n_out: usize,
        k: usize,
    ) -> Result<Self> {
        if k == 0
            || k % K_SUPERBLOCK != 0
            || data.len() != n_out * k / K_SUPERBLOCK * Q5_K_BLOCK_BYTES
        {
            return Err(ModelError::BadShape {
                tensor: "q5_k weight".into(),
                expected: vec![n_out, k],
                got: vec![data.len()],
            });
        }
        let (qs, qh, dd, scales) = repack_q5_k(data)?;
        Ok(Q5KWeight {
            qs: client.create_from_slice(u32::as_bytes(&qs)),
            qh: client.create_from_slice(u32::as_bytes(&qh)),
            dd: client.create_from_slice(f32::as_bytes(&dd)),
            scales: client.create_from_slice(u32::as_bytes(&scales)),
            n_out,
            k,
            _runtime: PhantomData,
        })
    }

    /// Bytes in VRAM: 180 per 256 weights (5.63 bits/weight).
    pub fn vram_bytes(&self) -> usize {
        (self.n_out * self.k / K_SUPERBLOCK) * (128 + 32 + 12 + 8)
    }

    /// Transient full-width f32 copy on device (`[n_out, k]` row-major)
    /// for the batched-matmul path — see `Q80Weight::dequant_device`.
    pub fn dequant_device<F: Float>(&self, client: &ComputeClient<R>) -> Handle {
        let n = self.n_out * self.k;
        let n_sb = n / K_SUPERBLOCK;
        let out_h = client.empty(n * core::mem::size_of::<F>());
        unsafe {
            q5_k_dequant_kernel::launch_unchecked::<F, R>(
                client,
                cube_count_capped(n as u32),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(self.qs.clone(), n_sb * 32),
                ArrayArg::from_raw_parts(self.qh.clone(), n_sb * 8),
                ArrayArg::from_raw_parts(self.dd.clone(), n_sb * 2),
                ArrayArg::from_raw_parts(self.scales.clone(), n_sb * 3),
                ArrayArg::from_raw_parts(out_h.clone(), n),
                n,
            );
        }
        out_h
    }

    /// Device path: launch only, output handle returned (see
    /// [`Q40Weight::matmul_device`]).
    pub fn matmul_device(&self, client: &ComputeClient<R>, x: Handle, m: usize) -> Handle {
        let out_len = m * self.n_out;
        let out_h = client.empty(out_len * core::mem::size_of::<f32>());
        let n_sb = self.n_out * self.k / K_SUPERBLOCK;
        unsafe {
            q5_k_matmul_kernel::launch_unchecked::<R>(
                client,
                cube_count_capped(out_len as u32),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(x, m * self.k),
                ArrayArg::from_raw_parts(self.qs.clone(), n_sb * 32),
                ArrayArg::from_raw_parts(self.qh.clone(), n_sb * 8),
                ArrayArg::from_raw_parts(self.dd.clone(), n_sb * 2),
                ArrayArg::from_raw_parts(self.scales.clone(), n_sb * 3),
                ArrayArg::from_raw_parts(out_h.clone(), out_len),
                m,
                self.k,
                self.n_out,
            );
        }
        out_h
    }

    /// `y = x @ W^T` for host-side `x: [m, k]`, returning `[m, n_out]`.
    pub fn matmul_host(&self, client: &ComputeClient<R>, x: &[f32], m: usize) -> Result<Vec<f32>> {
        if m == 0 || x.len() != m * self.k {
            return Err(ModelError::BadShape {
                tensor: "q5_k matmul input".into(),
                expected: vec![m, self.k],
                got: vec![x.len()],
            });
        }
        let x_h = client.create_from_slice(f32::as_bytes(x));
        let out_h = self.matmul_device(client, x_h, m);
        let bytes = client.read_one_unchecked(out_h);
        Ok(f32::from_bytes(&bytes).to_vec())
    }
}

pub struct Q6KWeight<R: Runtime> {
    ql: Handle,
    qh: Handle,
    sc: Handle,
    d: Handle,
    n_out: usize,
    k: usize,
    _runtime: PhantomData<R>,
}

impl<R: Runtime> Q6KWeight<R> {
    /// Repacks a GGUF Q6_K tensor onto the device.
    pub fn from_gguf_bytes(
        client: &ComputeClient<R>,
        data: &[u8],
        n_out: usize,
        k: usize,
    ) -> Result<Self> {
        if k == 0
            || k % K_SUPERBLOCK != 0
            || data.len() != n_out * k / K_SUPERBLOCK * Q6_K_BLOCK_BYTES
        {
            return Err(ModelError::BadShape {
                tensor: "q6_k weight".into(),
                expected: vec![n_out, k],
                got: vec![data.len()],
            });
        }
        let (ql, qh, sc, d) = repack_q6_k(data)?;
        Ok(Q6KWeight {
            ql: client.create_from_slice(u32::as_bytes(&ql)),
            qh: client.create_from_slice(u32::as_bytes(&qh)),
            sc: client.create_from_slice(u32::as_bytes(&sc)),
            d: client.create_from_slice(f32::as_bytes(&d)),
            n_out,
            k,
            _runtime: PhantomData,
        })
    }

    /// Bytes in VRAM: 212 per 256 weights (6.63 bits/weight).
    pub fn vram_bytes(&self) -> usize {
        (self.n_out * self.k / K_SUPERBLOCK) * (128 + 64 + 16 + 4)
    }

    /// Transient full-width f32 copy on device (`[n_out, k]` row-major)
    /// for the batched-matmul path — see `Q80Weight::dequant_device`.
    pub fn dequant_device<F: Float>(&self, client: &ComputeClient<R>) -> Handle {
        let n = self.n_out * self.k;
        let n_sb = n / K_SUPERBLOCK;
        let out_h = client.empty(n * core::mem::size_of::<F>());
        unsafe {
            q6_k_dequant_kernel::launch_unchecked::<F, R>(
                client,
                cube_count_capped(n as u32),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(self.ql.clone(), n_sb * 32),
                ArrayArg::from_raw_parts(self.qh.clone(), n_sb * 16),
                ArrayArg::from_raw_parts(self.sc.clone(), n_sb * 4),
                ArrayArg::from_raw_parts(self.d.clone(), n_sb),
                ArrayArg::from_raw_parts(out_h.clone(), n),
                n,
            );
        }
        out_h
    }

    /// Split-K decode gemv through the Combs Kernel — see
    /// `Q80Weight::decode_gemv_wgsl`.
    pub(crate) fn decode_gemv_wgsl(
        &self,
        client: &ComputeClient<cubecl::wgpu::WgpuRuntime>,
        x: Handle,
    ) -> Handle {
        let out_h = client.empty(self.n_out * core::mem::size_of::<f32>());
        crate::wgsl::launch_gemv(
            client,
            crate::wgsl::GemvKernel::Q6K,
            (self.n_out as u32).div_ceil(16),
            vec![
                x.binding(),
                self.ql.clone().binding(),
                self.qh.clone().binding(),
                self.sc.clone().binding(),
                self.d.clone().binding(),
                out_h.clone().binding(),
            ],
            vec![self.n_out as u64, self.k as u64],
        );
        out_h
    }

    /// Dequantizes the rows named by `ids` into `[n_tokens, k]` f32 —
    /// the embedding lookup for a packed Q6_K table, row-for-row
    /// bit-exact with the dequant kernel.
    pub fn gather_rows_device(
        &self,
        client: &ComputeClient<R>,
        ids: Handle,
        n_tokens: usize,
    ) -> Handle {
        let total = n_tokens * self.k;
        let n_sb = self.n_out * self.k / K_SUPERBLOCK;
        let out_h = client.empty(total * core::mem::size_of::<f32>());
        unsafe {
            q6_k_gather_kernel::launch_unchecked::<R>(
                client,
                cube_count_capped(total as u32),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(ids, n_tokens),
                ArrayArg::from_raw_parts(self.ql.clone(), n_sb * 32),
                ArrayArg::from_raw_parts(self.qh.clone(), n_sb * 16),
                ArrayArg::from_raw_parts(self.sc.clone(), n_sb * 4),
                ArrayArg::from_raw_parts(self.d.clone(), n_sb),
                ArrayArg::from_raw_parts(out_h.clone(), total),
                total,
                self.k,
            );
        }
        out_h
    }

    /// Device path: launch only, output handle returned (see
    /// [`Q40Weight::matmul_device`]).
    pub fn matmul_device(&self, client: &ComputeClient<R>, x: Handle, m: usize) -> Handle {
        let out_len = m * self.n_out;
        let out_h = client.empty(out_len * core::mem::size_of::<f32>());
        let n_sb = self.n_out * self.k / K_SUPERBLOCK;
        unsafe {
            q6_k_matmul_kernel::launch_unchecked::<R>(
                client,
                cube_count_capped(out_len as u32),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(x, m * self.k),
                ArrayArg::from_raw_parts(self.ql.clone(), n_sb * 32),
                ArrayArg::from_raw_parts(self.qh.clone(), n_sb * 16),
                ArrayArg::from_raw_parts(self.sc.clone(), n_sb * 4),
                ArrayArg::from_raw_parts(self.d.clone(), n_sb),
                ArrayArg::from_raw_parts(out_h.clone(), out_len),
                m,
                self.k,
                self.n_out,
            );
        }
        out_h
    }

    /// `y = x @ W^T` for host-side `x: [m, k]`, returning `[m, n_out]`.
    pub fn matmul_host(&self, client: &ComputeClient<R>, x: &[f32], m: usize) -> Result<Vec<f32>> {
        if m == 0 || x.len() != m * self.k {
            return Err(ModelError::BadShape {
                tensor: "q6_k matmul input".into(),
                expected: vec![m, self.k],
                got: vec![x.len()],
            });
        }
        let x_h = client.create_from_slice(f32::as_bytes(x));
        let out_h = self.matmul_device(client, x_h, m);
        let bytes = client.read_one_unchecked(out_h);
        Ok(f32::from_bytes(&bytes).to_vec())
    }
}

/// A device-resident quantized weight of any supported format, fixed to the
/// engine's wgpu runtime. This is what the linear seam (`qlinear`) stores;
/// format dispatch happens once per call, not per element.
pub enum QuantWeight {
    /// GGUF Q4_0.
    Q40(Q40Weight<cubecl::wgpu::WgpuRuntime>),
    /// GGUF Q5_0.
    Q50(Q50Weight<cubecl::wgpu::WgpuRuntime>),
    /// GGUF Q8_0.
    Q80(Q80Weight<cubecl::wgpu::WgpuRuntime>),
    /// GGUF Q4_K.
    Q4K(Q4KWeight<cubecl::wgpu::WgpuRuntime>),
    /// GGUF Q5_K.
    Q5K(Q5KWeight<cubecl::wgpu::WgpuRuntime>),
    /// GGUF Q6_K.
    Q6K(Q6KWeight<cubecl::wgpu::WgpuRuntime>),
}

impl QuantWeight {
    /// Builds from a raw packed tensor as handed out by
    /// `combs_formats::ModelSource::open_tensor_quant`.
    pub fn from_quant_tensor(
        client: &ComputeClient<cubecl::wgpu::WgpuRuntime>,
        format: combs_formats::QuantFormat,
        data: &[u8],
        n_out: usize,
        k: usize,
    ) -> Result<Self> {
        use combs_formats::QuantFormat;
        // The device path exists for the streamed mount: the host repack
        // briefly holds a full second copy of the widest tensor, which is
        // most of what stood between the in-tab peak and its gate.
        if stage_repack_device() {
            return Ok(match format {
                QuantFormat::Q4_0 => {
                    QuantWeight::Q40(Q40Weight::from_gguf_bytes_device(client, data, n_out, k)?)
                }
                QuantFormat::Q5_0 => {
                    QuantWeight::Q50(Q50Weight::from_gguf_bytes_device(client, data, n_out, k)?)
                }
                QuantFormat::Q8_0 => {
                    QuantWeight::Q80(Q80Weight::from_gguf_bytes_device(client, data, n_out, k)?)
                }
                QuantFormat::Q4K => {
                    QuantWeight::Q4K(Q4KWeight::from_gguf_bytes_device(client, data, n_out, k)?)
                }
                QuantFormat::Q5K => {
                    QuantWeight::Q5K(Q5KWeight::from_gguf_bytes_device(client, data, n_out, k)?)
                }
                QuantFormat::Q6K => {
                    QuantWeight::Q6K(Q6KWeight::from_gguf_bytes_device(client, data, n_out, k)?)
                }
            });
        }
        Ok(match format {
            QuantFormat::Q4_0 => {
                QuantWeight::Q40(Q40Weight::from_gguf_bytes(client, data, n_out, k)?)
            }
            QuantFormat::Q5_0 => {
                QuantWeight::Q50(Q50Weight::from_gguf_bytes(client, data, n_out, k)?)
            }
            QuantFormat::Q8_0 => {
                QuantWeight::Q80(Q80Weight::from_gguf_bytes(client, data, n_out, k)?)
            }
            QuantFormat::Q4K => {
                QuantWeight::Q4K(Q4KWeight::from_gguf_bytes(client, data, n_out, k)?)
            }
            QuantFormat::Q5K => {
                QuantWeight::Q5K(Q5KWeight::from_gguf_bytes(client, data, n_out, k)?)
            }
            QuantFormat::Q6K => {
                QuantWeight::Q6K(Q6KWeight::from_gguf_bytes(client, data, n_out, k)?)
            }
        })
    }

    /// Output features.
    pub fn n_out(&self) -> usize {
        match self {
            QuantWeight::Q40(w) => w.n_out,
            QuantWeight::Q50(w) => w.n_out,
            QuantWeight::Q80(w) => w.n_out,
            QuantWeight::Q4K(w) => w.n_out,
            QuantWeight::Q5K(w) => w.n_out,
            QuantWeight::Q6K(w) => w.n_out,
        }
    }

    /// Input features.
    pub fn k(&self) -> usize {
        match self {
            QuantWeight::Q40(w) => w.k,
            QuantWeight::Q50(w) => w.k,
            QuantWeight::Q80(w) => w.k,
            QuantWeight::Q4K(w) => w.k,
            QuantWeight::Q5K(w) => w.k,
            QuantWeight::Q6K(w) => w.k,
        }
    }

    /// Bytes this weight occupies in VRAM.
    pub fn vram_bytes(&self) -> usize {
        match self {
            QuantWeight::Q40(w) => w.vram_bytes(),
            QuantWeight::Q50(w) => w.vram_bytes(),
            QuantWeight::Q80(w) => w.vram_bytes(),
            QuantWeight::Q4K(w) => w.vram_bytes(),
            QuantWeight::Q5K(w) => w.vram_bytes(),
            QuantWeight::Q6K(w) => w.vram_bytes(),
        }
    }

    /// Row gather (embedding lookup) off the packed table, when this
    /// format has a gather kernel. Q8_0 only in the first landing — other
    /// formats return `None` and the caller keeps its dense table, per
    /// the fallback discipline.
    pub fn gather_rows_device(
        &self,
        client: &ComputeClient<cubecl::wgpu::WgpuRuntime>,
        ids: Handle,
        n_tokens: usize,
    ) -> Option<Handle> {
        match self {
            QuantWeight::Q80(w) => Some(w.gather_rows_device(client, ids, n_tokens)),
            QuantWeight::Q4K(w) => Some(w.gather_rows_device(client, ids, n_tokens)),
            QuantWeight::Q6K(w) => Some(w.gather_rows_device(client, ids, n_tokens)),
            _ => None,
        }
    }

    /// Whether [`Self::gather_rows_device`] is implemented for this format.
    pub fn supports_gather(&self) -> bool {
        matches!(
            self,
            QuantWeight::Q80(_) | QuantWeight::Q4K(_) | QuantWeight::Q6K(_)
        )
    }

    /// Transient full-width dequant on device, every format — the
    /// batched-matmul path's supply. The output element is the caller's
    /// choice: `f32` reproduces the CPU reference bit for bit, `f16`
    /// halves both the transient and the traffic the multiply that
    /// follows has to move, which is where the time actually goes
    /// (`notes/proposed/narrow-operand-matmul.md`). The arithmetic is
    /// identical either way — only the final store narrows. (`Option`
    /// stays in the signature: a future format without a dequant kernel
    /// falls back safely.)
    pub fn dequant_device_as<F: Float>(
        &self,
        client: &ComputeClient<cubecl::wgpu::WgpuRuntime>,
    ) -> Option<Handle> {
        match self {
            QuantWeight::Q40(w) => Some(w.dequant_device::<F>(client)),
            QuantWeight::Q50(w) => Some(w.dequant_device::<F>(client)),
            QuantWeight::Q80(w) => Some(w.dequant_device::<F>(client)),
            QuantWeight::Q4K(w) => Some(w.dequant_device::<F>(client)),
            QuantWeight::Q5K(w) => Some(w.dequant_device::<F>(client)),
            QuantWeight::Q6K(w) => Some(w.dequant_device::<F>(client)),
        }
    }

    /// The f32 dequant, which is what every existing caller means.
    pub fn dequant_device(
        &self,
        client: &ComputeClient<cubecl::wgpu::WgpuRuntime>,
    ) -> Option<Handle> {
        self.dequant_device_as::<f32>(client)
    }

    pub fn matmul_device(
        &self,
        client: &ComputeClient<cubecl::wgpu::WgpuRuntime>,
        x: Handle,
        m: usize,
    ) -> Handle {
        match self {
            QuantWeight::Q40(w) => w.matmul_device(client, x, m),
            QuantWeight::Q50(w) => w.matmul_device(client, x, m),
            QuantWeight::Q80(w) => {
                if m == 1 && w.k % Q4_0_BLOCK == 0 && crate::wgsl::gemv_enabled() {
                    return w.decode_gemv_wgsl(client, x);
                }
                w.matmul_device(client, x, m)
            }
            QuantWeight::Q4K(w) => {
                if m == 1 && w.k % K_SUPERBLOCK == 0 && crate::wgsl::gemv_enabled() {
                    return w.decode_gemv_wgsl(client, x);
                }
                w.matmul_device(client, x, m)
            }
            QuantWeight::Q5K(w) => w.matmul_device(client, x, m),
            QuantWeight::Q6K(w) => {
                if m == 1 && w.k % K_SUPERBLOCK == 0 && crate::wgsl::gemv_enabled() {
                    return w.decode_gemv_wgsl(client, x);
                }
                w.matmul_device(client, x, m)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::f16 as F16;
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

    /// The GPU dequant must be **bit-exact** with the harmony CPU reference
    /// (`combs_formats::quants::dequantize_q4_0`) — same unpack, same
    /// arithmetic, same f16→f32 scale conversion. This validates the whole
    /// Layout layer: any repack/indexing slip shows up as a hard mismatch.
    #[test]
    fn dequant_kernel_is_bit_exact_vs_cpu_reference() {
        if crate::skip_no_gpu() {
            return;
        }
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
        if crate::skip_no_gpu() {
            return;
        }
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
                assert!((g - e).abs() <= tol, "m={m} out[{i}]: got {g}, expect {e}");
            }
        }
    }

    /// The staging repack, host vs device: a permutation plus an exact
    /// widen, so every output buffer must be BYTE-identical. LCG quant
    /// bytes exercise every bit pattern; the widen has its own cell
    /// below for the corners valid scales never produce.
    fn assert_bytes_eq(label: &str, host: &[u8], device: &[u8]) {
        assert_eq!(host.len(), device.len(), "{label}: length");
        if host != device {
            let i = host
                .iter()
                .zip(device.iter())
                .position(|(a, b)| a != b)
                .unwrap();
            panic!(
                "{label}: first byte diff at {i}: host {} device {}",
                host[i], device[i]
            );
        }
    }

    #[test]
    fn device_repack_matches_host_q4_0() {
        if crate::skip_no_gpu() {
            return;
        }
        let data = synth_q4_0(67);
        let (qs, d) = repack_q4_0(&data).unwrap();
        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let w = Q40Weight::<WgpuRuntime>::from_gguf_bytes_device(&client, &data, 67, 32).unwrap();
        assert_bytes_eq(
            "q4_0 qs",
            u32::as_bytes(&qs),
            &client.read_one_unchecked(w.qs.clone()),
        );
        assert_bytes_eq(
            "q4_0 d",
            f32::as_bytes(&d),
            &client.read_one_unchecked(w.d.clone()),
        );
    }

    #[test]
    fn device_repack_matches_host_q5_0() {
        if crate::skip_no_gpu() {
            return;
        }
        let data = synth_q5_0(53);
        let (qs, qh, d) = repack_q5_0(&data).unwrap();
        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let w = Q50Weight::<WgpuRuntime>::from_gguf_bytes_device(&client, &data, 53, 32).unwrap();
        assert_bytes_eq(
            "q5_0 qs",
            u32::as_bytes(&qs),
            &client.read_one_unchecked(w.qs.clone()),
        );
        assert_bytes_eq(
            "q5_0 qh",
            u32::as_bytes(&qh),
            &client.read_one_unchecked(w.qh.clone()),
        );
        assert_bytes_eq(
            "q5_0 d",
            f32::as_bytes(&d),
            &client.read_one_unchecked(w.d.clone()),
        );
    }

    #[test]
    fn device_repack_matches_host_q8_0() {
        if crate::skip_no_gpu() {
            return;
        }
        let data = synth_q8_0(41);
        let (qs, d) = repack_q8_0(&data).unwrap();
        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let w = Q80Weight::<WgpuRuntime>::from_gguf_bytes_device(&client, &data, 41, 32).unwrap();
        assert_bytes_eq(
            "q8_0 qs",
            u32::as_bytes(&qs),
            &client.read_one_unchecked(w.qs.clone()),
        );
        assert_bytes_eq(
            "q8_0 d",
            f32::as_bytes(&d),
            &client.read_one_unchecked(w.d.clone()),
        );
    }

    #[test]
    fn device_repack_matches_host_q4_k() {
        if crate::skip_no_gpu() {
            return;
        }
        let data = synth_q4_k(19);
        let (qs, dd, scales) = repack_q4_k(&data).unwrap();
        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let w = Q4KWeight::<WgpuRuntime>::from_gguf_bytes_device(&client, &data, 19, 256).unwrap();
        assert_bytes_eq(
            "q4_k qs",
            u32::as_bytes(&qs),
            &client.read_one_unchecked(w.qs.clone()),
        );
        assert_bytes_eq(
            "q4_k dd",
            f32::as_bytes(&dd),
            &client.read_one_unchecked(w.dd.clone()),
        );
        assert_bytes_eq(
            "q4_k scales",
            u32::as_bytes(&scales),
            &client.read_one_unchecked(w.scales.clone()),
        );
    }

    #[test]
    fn device_repack_matches_host_q5_k() {
        if crate::skip_no_gpu() {
            return;
        }
        let data = synth_q5_k(17);
        let (qs, qh, dd, scales) = repack_q5_k(&data).unwrap();
        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let w = Q5KWeight::<WgpuRuntime>::from_gguf_bytes_device(&client, &data, 17, 256).unwrap();
        assert_bytes_eq(
            "q5_k qs",
            u32::as_bytes(&qs),
            &client.read_one_unchecked(w.qs.clone()),
        );
        assert_bytes_eq(
            "q5_k qh",
            u32::as_bytes(&qh),
            &client.read_one_unchecked(w.qh.clone()),
        );
        assert_bytes_eq(
            "q5_k dd",
            f32::as_bytes(&dd),
            &client.read_one_unchecked(w.dd.clone()),
        );
        assert_bytes_eq(
            "q5_k scales",
            u32::as_bytes(&scales),
            &client.read_one_unchecked(w.scales.clone()),
        );
    }

    #[test]
    fn device_repack_matches_host_q6_k() {
        if crate::skip_no_gpu() {
            return;
        }
        let data = synth_q6_k(13);
        let (ql, qh, sc, d) = repack_q6_k(&data).unwrap();
        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let w = Q6KWeight::<WgpuRuntime>::from_gguf_bytes_device(&client, &data, 13, 256).unwrap();
        assert_bytes_eq(
            "q6_k ql",
            u32::as_bytes(&ql),
            &client.read_one_unchecked(w.ql.clone()),
        );
        assert_bytes_eq(
            "q6_k qh",
            u32::as_bytes(&qh),
            &client.read_one_unchecked(w.qh.clone()),
        );
        assert_bytes_eq(
            "q6_k sc",
            u32::as_bytes(&sc),
            &client.read_one_unchecked(w.sc.clone()),
        );
        assert_bytes_eq(
            "q6_k d",
            f32::as_bytes(&d),
            &client.read_one_unchecked(w.d.clone()),
        );
    }

    /// The widen alone, over raw LCG bit patterns — the corners a valid
    /// scale never produces (subnormals, infinities, NaNs) plus a pinned
    /// set. Q4_0 blocks are built whose SCALE bytes are the LCG stream,
    /// so the device path widens exactly those bits.
    #[test]
    fn device_f16_widen_matches_host_on_every_bit_pattern_class() {
        if crate::skip_no_gpu() {
            return;
        }
        let mut patterns: Vec<u16> = vec![
            0x0000, 0x8000, // ±0
            0x0001, 0x8001, // smallest subnormals
            0x03FF, 0x83FF, // largest subnormals
            0x0400, 0x8400, // smallest normals
            0x7BFF, 0xFBFF, // largest finite
            0x7C00, 0xFC00, // ±inf
            0x7C01, 0x7E00, 0xFE00, // NaNs
            0x3C00, 0xBC00, // ±1
        ];
        let mut s = 0xC0FFEEu32;
        for _ in 0..1024 {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            patterns.push((s >> 16) as u16);
        }
        let n_blocks = patterns.len();
        let mut data = Vec::with_capacity(n_blocks * Q4_0_BLOCK_BYTES);
        for bits in &patterns {
            data.extend_from_slice(&bits.to_le_bytes());
            data.extend_from_slice(&[0u8; 16]);
        }
        let (_, host_d) = repack_q4_0(&data).unwrap();
        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let w =
            Q40Weight::<WgpuRuntime>::from_gguf_bytes_device(&client, &data, n_blocks, 32).unwrap();
        let dev_bytes = client.read_one_unchecked(w.d.clone());
        let dev_d: &[f32] = f32::from_bytes(&dev_bytes);
        for (i, (h, g)) in host_d.iter().zip(dev_d.iter()).enumerate() {
            let same_bits = h.to_bits() == g.to_bits();
            let both_nan = h.is_nan() && g.is_nan();
            assert!(
                same_bits || both_nan,
                "widen bits {:#06x}: host {h} ({:#010x}) device {g} ({:#010x})",
                patterns[i],
                h.to_bits(),
                g.to_bits()
            );
        }
    }

    /// The slab seams, forced: with COMBS_REPACK_SLAB tiny (a OnceLock,
    /// so this cell owns its process), every tensor crosses many slab
    /// boundaries including a partial tail — and every byte must still
    /// land where the host repack puts it.
    ///
    /// COMBS_REPACK_SLAB=1024 cargo test --release -p combs-models \
    ///   --lib device_repack_matches_host_under_tiny_slabs -- --nocapture
    #[test]
    fn device_repack_matches_host_under_tiny_slabs() {
        if crate::skip_no_gpu() {
            return;
        }
        let ok = std::env::var("COMBS_REPACK_SLAB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .is_some_and(|n| n <= 4096);
        if !ok {
            eprintln!("skip: COMBS_REPACK_SLAB not set tiny (own-process cell)");
            return;
        }
        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        {
            let data = synth_q4_0(301); // 301 x 18 B: many slabs, partial tail
            let (qs, d) = repack_q4_0(&data).unwrap();
            let w =
                Q40Weight::<WgpuRuntime>::from_gguf_bytes_device(&client, &data, 301, 32).unwrap();
            assert_bytes_eq(
                "slab q4_0 qs",
                u32::as_bytes(&qs),
                &client.read_one_unchecked(w.qs.clone()),
            );
            assert_bytes_eq(
                "slab q4_0 d",
                f32::as_bytes(&d),
                &client.read_one_unchecked(w.d.clone()),
            );
        }
        {
            let data = synth_q6_k(23); // 23 x 210 B: misaligned slab tails
            let (ql, qh, sc, d) = repack_q6_k(&data).unwrap();
            let w =
                Q6KWeight::<WgpuRuntime>::from_gguf_bytes_device(&client, &data, 23, 256).unwrap();
            assert_bytes_eq(
                "slab q6_k ql",
                u32::as_bytes(&ql),
                &client.read_one_unchecked(w.ql.clone()),
            );
            assert_bytes_eq(
                "slab q6_k qh",
                u32::as_bytes(&qh),
                &client.read_one_unchecked(w.qh.clone()),
            );
            assert_bytes_eq(
                "slab q6_k sc",
                u32::as_bytes(&sc),
                &client.read_one_unchecked(w.sc.clone()),
            );
            assert_bytes_eq(
                "slab q6_k d",
                f32::as_bytes(&d),
                &client.read_one_unchecked(w.d.clone()),
            );
        }
        println!("tiny-slab parity: q4_0 + q6_k byte-identical across slab seams");
    }

    /// Deterministic pseudo-random byte stream for K-quant payloads.
    fn lcg_bytes(n: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8
            })
            .collect()
    }

    /// Q4_K superblock stream: valid small f16 d/dmin, LCG scales + quants.
    fn synth_q4_k(n_sb: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n_sb * Q4_K_BLOCK_BYTES);
        for b in 0..n_sb {
            let d = burn::tensor::f16::from_f32(0.002 * ((b % 9) as f32 + 1.0));
            let dmin = burn::tensor::f16::from_f32(0.001 * ((b % 5) as f32 + 1.0));
            out.extend_from_slice(&d.to_le_bytes());
            out.extend_from_slice(&dmin.to_le_bytes());
            out.extend_from_slice(&lcg_bytes(140, 0xC0FFEE ^ b as u32));
        }
        out
    }

    /// Q5_K superblock stream: LCG scales/qh/qs, valid small f16 d/dmin.
    fn synth_q5_k(n_sb: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n_sb * Q5_K_BLOCK_BYTES);
        for b in 0..n_sb {
            let d = burn::tensor::f16::from_f32(0.003 * ((b % 7) as f32 + 1.0));
            let dmin = burn::tensor::f16::from_f32(0.001 * ((b % 5) as f32 + 1.0));
            out.extend_from_slice(&d.to_le_bytes());
            out.extend_from_slice(&dmin.to_le_bytes());
            out.extend_from_slice(&lcg_bytes(172, 0x5EED ^ b as u32));
        }
        out
    }

    /// Q6_K superblock stream: LCG ql/qh/scales, valid small f16 d.
    fn synth_q6_k(n_sb: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n_sb * Q6_K_BLOCK_BYTES);
        for b in 0..n_sb {
            out.extend_from_slice(&lcg_bytes(208, 0xBEE5 ^ b as u32));
            let d = burn::tensor::f16::from_f32(0.002 * ((b % 9) as f32 + 1.0));
            out.extend_from_slice(&d.to_le_bytes());
        }
        out
    }

    fn assert_close(got: &[f32], expect: &[f32], rel: f32, what: &str) {
        assert_eq!(got.len(), expect.len(), "{what}: length");
        for (i, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
            // Finite first: a NaN passes every |diff| <= tol comparison
            // backwards (all comparisons on NaN are false, but so is the
            // assert's), and an Inf-vs-Inf diff is NaN — name the index
            // before the tolerance can lie about it.
            assert!(
                g.is_finite() && e.is_finite(),
                "{what}[{i}]: non-finite (got {g}, expect {e})"
            );
            let tol = rel * e.abs().max(1.0);
            assert!((g - e).abs() <= tol, "{what}[{i}]: got {g}, expect {e}");
        }
    }

    /// Q4_K GPU dequant vs the harmony CPU reference. The kernel mirrors the
    /// reference arithmetic exactly; tolerance only allows for backend FMA
    /// contraction of `d1·q − fmin` (a last-ulp effect, bounded far below
    /// the quantization step).
    #[test]
    fn q4_k_dequant_matches_cpu_reference() {
        if crate::skip_no_gpu() {
            return;
        }
        let n_sb = 9;
        let data = synth_q4_k(n_sb);
        let n = n_sb * K_SUPERBLOCK;
        let expect = combs_formats::quants::dequantize_q4_k(&data, n).unwrap();

        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let got = dequantize_q4_k_gpu::<WgpuRuntime>(&client, &data).unwrap();
        assert_close(&got, &expect, 1e-6, "q4_k dequant");
    }

    /// Q5_K GPU dequant vs the harmony CPU reference.
    #[test]
    fn q5_k_dequant_matches_cpu_reference() {
        if crate::skip_no_gpu() {
            return;
        }
        let n_sb = 9;
        let data = synth_q5_k(n_sb);
        let n = n_sb * K_SUPERBLOCK;
        let expect = combs_formats::quants::dequantize_q5_k(&data, n).unwrap();

        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let got = dequantize_q5_k_gpu::<WgpuRuntime>(&client, &data).unwrap();
        assert_close(&got, &expect, 1e-6, "q5_k dequant");
    }

    /// Q6_K GPU dequant vs the harmony CPU reference.
    #[test]
    fn q6_k_dequant_matches_cpu_reference() {
        if crate::skip_no_gpu() {
            return;
        }
        let n_sb = 9;
        let data = synth_q6_k(n_sb);
        let n = n_sb * K_SUPERBLOCK;
        let expect = combs_formats::quants::dequantize_q6_k(&data, n).unwrap();

        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let got = dequantize_q6_k_gpu::<WgpuRuntime>(&client, &data).unwrap();
        assert_close(&got, &expect, 1e-6, "q6_k dequant");
    }

    /// Fused Q5_K matmul vs a reference matmul over the reference dequant,
    /// decode (m=1) and prefill (m>1) shapes, multi-superblock rows.
    #[test]
    fn q5_k_fused_matmul_matches_reference() {
        if crate::skip_no_gpu() {
            return;
        }
        let (n_out, k) = (35, 512); // 2 superblocks per row, partial cube
        let n_sb = n_out * k / K_SUPERBLOCK;
        let data = synth_q5_k(n_sb);
        let w = combs_formats::quants::dequantize_q5_k(&data, n_out * k).unwrap();

        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let weight = Q5KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k).unwrap();
        assert_eq!(weight.vram_bytes(), n_sb * 180);

        for m in [1usize, 3] {
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0)
                .collect();
            let expect = ref_matmul(&x, &w, m, k, n_out);
            let got = weight.matmul_host(&client, &x, m).unwrap();
            assert_close(&got, &expect, 1e-3, &format!("q5_k matmul m={m}"));
        }
    }

    /// Fused Q4_K matmul vs a reference matmul over the reference dequant,
    /// decode (m=1) and prefill (m>1) shapes, multi-superblock rows.
    #[test]
    fn q4_k_fused_matmul_matches_reference() {
        if crate::skip_no_gpu() {
            return;
        }
        let (n_out, k) = (35, 512); // 2 superblocks per row, partial cube
        let n_sb = n_out * k / K_SUPERBLOCK;
        let data = synth_q4_k(n_sb);
        let w = combs_formats::quants::dequantize_q4_k(&data, n_out * k).unwrap();

        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let weight = Q4KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k).unwrap();
        assert_eq!(weight.vram_bytes(), n_sb * 148);

        for m in [1usize, 3] {
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0)
                .collect();
            let expect = ref_matmul(&x, &w, m, k, n_out);
            let got = weight.matmul_host(&client, &x, m).unwrap();
            assert_close(&got, &expect, 1e-3, &format!("q4_k matmul m={m}"));
        }
    }

    /// Fused Q6_K matmul vs a reference matmul over the reference dequant.
    #[test]
    fn q6_k_fused_matmul_matches_reference() {
        if crate::skip_no_gpu() {
            return;
        }
        let (n_out, k) = (35, 512);
        let n_sb = n_out * k / K_SUPERBLOCK;
        let data = synth_q6_k(n_sb);
        let w = combs_formats::quants::dequantize_q6_k(&data, n_out * k).unwrap();

        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let weight = Q6KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k).unwrap();
        assert_eq!(weight.vram_bytes(), n_sb * 212);

        for m in [1usize, 3] {
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0)
                .collect();
            let expect = ref_matmul(&x, &w, m, k, n_out);
            let got = weight.matmul_host(&client, &x, m).unwrap();
            assert_close(&got, &expect, 1e-3, &format!("q6_k matmul m={m}"));
        }
    }

    /// Q5_0 stream: valid f16 scales, LCG high bits + nibbles.
    fn synth_q5_0(n_blocks: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n_blocks * Q5_0_BLOCK_BYTES);
        for b in 0..n_blocks {
            let scale = burn::tensor::f16::from_f32(0.003 * ((b % 11) as f32 + 1.0));
            out.extend_from_slice(&scale.to_le_bytes());
            out.extend_from_slice(&lcg_bytes(20, 0x51D0 ^ b as u32));
        }
        out
    }

    /// Q8_0 stream: valid f16 scales, LCG i8 payload (full range).
    fn synth_q8_0(n_blocks: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n_blocks * Q8_0_BLOCK_BYTES);
        for b in 0..n_blocks {
            let scale = burn::tensor::f16::from_f32(0.003 * ((b % 11) as f32 + 1.0));
            out.extend_from_slice(&scale.to_le_bytes());
            out.extend_from_slice(&lcg_bytes(32, 0x80C0 ^ b as u32));
        }
        out
    }

    /// The embedding gather must be bit-exact against the CPU reference
    /// row slices: it is the dequant kernel with a row indirection, and
    /// any drift would make a packed table a subtly different model.
    /// Ragged k-multiple vocab, out-of-order and repeated ids, and a
    /// single-token gather all covered.
    #[test]
    fn q8_0_gather_is_bit_exact_vs_cpu_reference_rows() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        let client = WgpuRuntime::client(&device);

        // vocab=67 rows (ragged vs every launch granularity), k=64.
        let (vocab, k) = (67usize, 64usize);
        let data = synth_q8_0(vocab * k / 32);
        let w = Q80Weight::<WgpuRuntime>::from_gguf_bytes(&client, &data, vocab, k)
            .expect("packed table");
        let dense = combs_formats::quants::dequantize_q8_0(&data, vocab * k).unwrap();

        for ids in [vec![3u32], vec![3, 0, 66, 3, 41]] {
            let ids_h = client.create_from_slice(u32::as_bytes(&ids));
            let out_h = w.gather_rows_device(&client, ids_h, ids.len());
            let bytes = client.read_one_unchecked(out_h);
            let got = f32::from_bytes(&bytes);
            for (t, &row) in ids.iter().enumerate() {
                let want = &dense[row as usize * k..(row as usize + 1) * k];
                assert_eq!(
                    &got[t * k..(t + 1) * k],
                    want,
                    "row {row} at slot {t} must be bit-exact vs gguf.rs"
                );
            }
        }
    }

    /// The split-K WGSL decode gemvs vs the cube kernels and the CPU
    /// reference, per format: different accumulation shape, same
    /// arithmetic — ragged n_out (crosses the 16-row workgroup),
    /// gemma-ish k for Q8, superblock-multiple k for the K-quants.
    #[test]
    fn wgsl_decode_gemv_matches_cube_and_reference() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let check = |got: &[f32],
                     cube: &[f32],
                     x: &[f32],
                     dense: &[f32],
                     n_out: usize,
                     k: usize,
                     what: &str| {
            for r in 0..n_out {
                let expect: f32 = (0..k).map(|c| x[c] * dense[r * k + c]).sum();
                let tol = 1e-3 * expect.abs().max(1.0);
                assert!(
                    (got[r] - expect).abs() <= tol,
                    "{what} n_out={n_out} k={k} row {r}: wgsl {} vs reference {expect}",
                    got[r]
                );
                assert!(
                    (got[r] - cube[r]).abs() <= tol,
                    "{what} n_out={n_out} k={k} row {r}: wgsl {} vs cube {}",
                    got[r],
                    cube[r]
                );
            }
        };
        let mk_x = |k: usize| -> Vec<f32> {
            (0..k)
                .map(|i| ((i as f32 * 0.311).sin() * 1.4) - 0.1)
                .collect()
        };

        for (n_out, k) in [(37usize, 64usize), (1024, 1024), (211, 1152)] {
            let data = synth_q8_0(n_out * k / 32);
            let w = Q80Weight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k)
                .expect("packed table");
            let dense = combs_formats::quants::dequantize_q8_0(&data, n_out * k).unwrap();
            let x = mk_x(k);
            let x_h = client.create_from_slice(f32::as_bytes(&x));
            let got_h = w.decode_gemv_wgsl(&client, x_h.clone());
            let cube_h = w.matmul_device_with(&client, x_h, 1, false);
            let got = f32::from_bytes(&client.read_one_unchecked(got_h)).to_vec();
            let cube = f32::from_bytes(&client.read_one_unchecked(cube_h)).to_vec();
            check(&got, &cube, &x, &dense, n_out, k, "q8_0");
        }
        for (n_out, k) in [(37usize, 256usize), (211, 1024)] {
            let data = synth_q4_k(n_out * k / 256);
            let w = Q4KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k)
                .expect("packed table");
            let dense = combs_formats::quants::dequantize_q4_k(&data, n_out * k).unwrap();
            let x = mk_x(k);
            let x_h = client.create_from_slice(f32::as_bytes(&x));
            let got_h = w.decode_gemv_wgsl(&client, x_h.clone());
            let cube_h = w.matmul_device_with(&client, x_h, 1, false);
            let got = f32::from_bytes(&client.read_one_unchecked(got_h)).to_vec();
            let cube = f32::from_bytes(&client.read_one_unchecked(cube_h)).to_vec();
            check(&got, &cube, &x, &dense, n_out, k, "q4_k");
        }
        for (n_out, k) in [(37usize, 256usize), (211, 1024)] {
            let data = synth_q6_k(n_out * k / 256);
            let w = Q6KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k)
                .expect("packed table");
            let dense = combs_formats::quants::dequantize_q6_k(&data, n_out * k).unwrap();
            let x = mk_x(k);
            let x_h = client.create_from_slice(f32::as_bytes(&x));
            let got_h = w.decode_gemv_wgsl(&client, x_h.clone());
            let cube_h = w.matmul_device(&client, x_h, 1);
            let got = f32::from_bytes(&client.read_one_unchecked(got_h)).to_vec();
            let cube = f32::from_bytes(&client.read_one_unchecked(cube_h)).to_vec();
            check(&got, &cube, &x, &dense, n_out, k, "q6_k");
        }
    }

    /// The K-quant gathers under the same bar as Q8_0's: bit-exact
    /// against the CPU reference row slices, with out-of-order and
    /// repeated ids. k = 256 = one superblock per row keeps the fixture
    /// small while touching every sub-block/quadrant lane.
    #[test]
    fn q4_k_and_q6_k_gather_are_bit_exact_vs_cpu_reference_rows() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let (vocab, k) = (13usize, 256usize);
        let ids_sets: [&[u32]; 2] = [&[5], &[5, 0, 12, 5, 7]];

        let data = synth_q4_k(vocab);
        let w = Q4KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, vocab, k)
            .expect("packed q4_k table");
        let dense = combs_formats::quants::dequantize_q4_k(&data, vocab * k).unwrap();
        for ids in ids_sets {
            let ids_h = client.create_from_slice(u32::as_bytes(ids));
            let out_h = w.gather_rows_device(&client, ids_h, ids.len());
            let bytes = client.read_one_unchecked(out_h);
            let got = f32::from_bytes(&bytes);
            for (t, &row) in ids.iter().enumerate() {
                assert_eq!(
                    &got[t * k..(t + 1) * k],
                    &dense[row as usize * k..(row as usize + 1) * k],
                    "q4_k row {row} at slot {t} must be bit-exact"
                );
            }
        }

        let data = synth_q6_k(vocab);
        let w = Q6KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, vocab, k)
            .expect("packed q6_k table");
        let dense = combs_formats::quants::dequantize_q6_k(&data, vocab * k).unwrap();
        for ids in ids_sets {
            let ids_h = client.create_from_slice(u32::as_bytes(ids));
            let out_h = w.gather_rows_device(&client, ids_h, ids.len());
            let bytes = client.read_one_unchecked(out_h);
            let got = f32::from_bytes(&bytes);
            for (t, &row) in ids.iter().enumerate() {
                assert_eq!(
                    &got[t * k..(t + 1) * k],
                    &dense[row as usize * k..(row as usize + 1) * k],
                    "q6_k row {row} at slot {t} must be bit-exact"
                );
            }
        }
    }

    /// Q5_0/Q8_0 GPU dequant must be **bit-exact** vs the CPU references —
    /// both are a single f32 multiply per value, same as Q4_0.
    /// `dequant_device` (the batched-matmul supply, launched from the
    /// RESIDENT weight handles with the capped grid) must match the CPU
    /// reference bit-exactly for every format — same kernels as the
    /// `dequantize_*_gpu` validation path, different plumbing.
    #[test]
    fn resident_dequant_matches_reference_all_formats() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let read = |h: Handle, n: usize| -> Vec<f32> {
            let bytes = client.read_one_unchecked(h);
            f32::from_bytes(&bytes)[..n].to_vec()
        };
        // The narrow store must not be a second arithmetic: the kernel
        // computes the same f32 value and rounds it once on the way out,
        // so every element has to equal the reference put through the
        // same rounding. Anything else means the dequant maths moved.
        let read_f16 = |h: Handle, n: usize| -> Vec<f32> {
            let bytes = client.read_one_unchecked(h);
            F16::from_bytes(&bytes)[..n]
                .iter()
                .map(|v| v.to_f32())
                .collect()
        };
        let assert_narrowed = |got: &[f32], expect: &[f32], what: &str| {
            assert_eq!(got.len(), expect.len(), "{what}: length");
            for (i, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
                let rounded = F16::from_f32(*e).to_f32();
                assert_eq!(*g, rounded, "{what}[{i}]: f16 store drifted");
            }
        };

        let (n_out, k) = (6, 256);
        let n = n_out * k;
        let sb = n / K_SUPERBLOCK;
        let blk = n / Q4_0_BLOCK;

        let data = synth_q4_0(blk);
        let w = Q40Weight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k).unwrap();
        let expect = combs_formats::quants::dequantize_q4_0(&data, n).unwrap();
        assert_eq!(read(w.dequant_device::<f32>(&client), n), expect, "q4_0");
        assert_narrowed(
            &read_f16(w.dequant_device::<F16>(&client), n),
            &expect,
            "q4_0",
        );

        let data = synth_q5_0(blk);
        let w = Q50Weight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k).unwrap();
        let expect = combs_formats::quants::dequantize_q5_0(&data, n).unwrap();
        assert_eq!(read(w.dequant_device::<f32>(&client), n), expect, "q5_0");
        assert_narrowed(
            &read_f16(w.dequant_device::<F16>(&client), n),
            &expect,
            "q5_0",
        );

        let data = synth_q8_0(blk);
        let w = Q80Weight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k).unwrap();
        let expect = combs_formats::quants::dequantize_q8_0(&data, n).unwrap();
        assert_eq!(read(w.dequant_device::<f32>(&client), n), expect, "q8_0");
        assert_narrowed(
            &read_f16(w.dequant_device::<F16>(&client), n),
            &expect,
            "q8_0",
        );

        let data = synth_q4_k(sb);
        let w = Q4KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k).unwrap();
        let expect = combs_formats::quants::dequantize_q4_k(&data, n).unwrap();
        assert_eq!(read(w.dequant_device::<f32>(&client), n), expect, "q4_k");
        assert_narrowed(
            &read_f16(w.dequant_device::<F16>(&client), n),
            &expect,
            "q4_k",
        );

        let data = synth_q5_k(sb);
        let w = Q5KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k).unwrap();
        let expect = combs_formats::quants::dequantize_q5_k(&data, n).unwrap();
        assert_eq!(read(w.dequant_device::<f32>(&client), n), expect, "q5_k");
        assert_narrowed(
            &read_f16(w.dequant_device::<F16>(&client), n),
            &expect,
            "q5_k",
        );

        let data = synth_q6_k(sb);
        let w = Q6KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k).unwrap();
        let expect = combs_formats::quants::dequantize_q6_k(&data, n).unwrap();
        assert_eq!(read(w.dequant_device::<f32>(&client), n), expect, "q6_k");
        assert_narrowed(
            &read_f16(w.dequant_device::<F16>(&client), n),
            &expect,
            "q6_k",
        );
    }

    #[test]
    fn q5_0_and_q8_0_dequant_are_bit_exact() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        let client = WgpuRuntime::client(&device);

        let data = synth_q5_0(33);
        let n = 33 * Q4_0_BLOCK;
        let expect = combs_formats::quants::dequantize_q5_0(&data, n).unwrap();
        let got = dequantize_q5_0_gpu::<WgpuRuntime>(&client, &data).unwrap();
        assert_eq!(got, expect, "q5_0 GPU dequant must be bit-exact");

        let data = synth_q8_0(33);
        let expect = combs_formats::quants::dequantize_q8_0(&data, n).unwrap();
        let got = dequantize_q8_0_gpu::<WgpuRuntime>(&client, &data).unwrap();
        assert_eq!(got, expect, "q8_0 GPU dequant must be bit-exact");
    }

    /// Fused Q5_0/Q8_0 matmuls vs reference matmuls over the reference
    /// dequants, decode and prefill shapes.
    #[test]
    fn q5_0_and_q8_0_fused_matmul_match_reference() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let (n_out, k) = (67, 128);
        let n_blocks = n_out * k / Q4_0_BLOCK;

        let data5 = synth_q5_0(n_blocks);
        let w5 = combs_formats::quants::dequantize_q5_0(&data5, n_out * k).unwrap();
        let q5 = Q50Weight::<WgpuRuntime>::from_gguf_bytes(&client, &data5, n_out, k).unwrap();
        assert_eq!(q5.vram_bytes(), n_blocks * 24);

        let data8 = synth_q8_0(n_blocks);
        let w8 = combs_formats::quants::dequantize_q8_0(&data8, n_out * k).unwrap();
        let q8 = Q80Weight::<WgpuRuntime>::from_gguf_bytes(&client, &data8, n_out, k).unwrap();
        assert_eq!(q8.vram_bytes(), n_blocks * 36);

        for m in [1usize, 3] {
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0)
                .collect();
            let got5 = q5.matmul_host(&client, &x, m).unwrap();
            assert_close(
                &got5,
                &ref_matmul(&x, &w5, m, k, n_out),
                1e-3,
                &format!("q5_0 m={m}"),
            );
            let got8 = q8.matmul_host(&client, &x, m).unwrap();
            assert_close(
                &got8,
                &ref_matmul(&x, &w8, m, k, n_out),
                1e-3,
                &format!("q8_0 m={m}"),
            );
        }
    }

    /// Q5_0/Q8_0 malformed input rejection.
    #[test]
    fn q5_q8_shape_validation() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        assert!(repack_q5_0(&[0u8; 21]).is_err());
        assert!(repack_q8_0(&[0u8; 33]).is_err());
        assert!(Q50Weight::<WgpuRuntime>::from_gguf_bytes(&client, &synth_q5_0(2), 2, 31).is_err());
        assert!(Q80Weight::<WgpuRuntime>::from_gguf_bytes(&client, &synth_q8_0(2), 2, 64).is_err());
    }

    /// K-quant shape validation mirrors the Q4_0 rules.
    #[test]
    fn k_quant_shape_validation() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        assert!(repack_q4_k(&[0u8; 143]).is_err());
        assert!(repack_q6_k(&[0u8; 209]).is_err());
        // k must be a superblock multiple.
        assert!(
            Q4KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &synth_q4_k(1), 1, 128).is_err()
        );
        assert!(
            Q6KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &synth_q6_k(1), 1, 128).is_err()
        );
    }

    /// Compares the tiled and untiled kernels on identical device inputs
    /// and demands **bit-identical** outputs (`to_bits`, not a tolerance):
    /// the tiled kernels change only the activation load path, never the
    /// accumulation order.
    fn assert_tiled_bit_identical(untiled: &[f32], tiled: &[f32], label: &str) {
        assert_eq!(untiled.len(), tiled.len(), "{label}: length mismatch");
        for (i, (u, t)) in untiled.iter().zip(tiled.iter()).enumerate() {
            assert_eq!(
                u.to_bits(),
                t.to_bits(),
                "{label} out[{i}]: untiled {u} vs tiled {t} — accumulation order drifted"
            );
        }
    }

    /// Q8_0 tiled-vs-untiled bit identity across ragged shapes: n_out not a
    /// multiple of the cube dim (ragged final column block), k not a
    /// multiple of the tile (ragged final k-tile), m from tiny to a full
    /// prefill chunk.
    #[test]
    fn tiled_q8_0_matmul_is_bit_identical() {
        if crate::skip_no_gpu() {
            return;
        }
        let (n_out, k) = (300, 320);
        let n_blocks = n_out * k / Q4_0_BLOCK;
        let data = synth_q8_0(n_blocks);
        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let w = Q80Weight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k).unwrap();
        for m in [2usize, 3, 17, 256, 1024] {
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i * 11 % 29) as f32 - 14.0) / 16.0)
                .collect();
            let x_h = client.create_from_slice(f32::as_bytes(&x));
            let un_h = w.matmul_device_with(&client, x_h.clone(), m, false);
            let ti_h = w.matmul_device_with(&client, x_h, m, true);
            let un = f32::from_bytes(&client.read_one_unchecked(un_h)).to_vec();
            let ti = f32::from_bytes(&client.read_one_unchecked(ti_h)).to_vec();
            assert_tiled_bit_identical(&un, &ti, &format!("q8_0 m={m}"));
        }
    }

    /// Q4_K tiled-vs-untiled bit identity (superblock-aligned k by
    /// construction; ragged final column block still exercised).
    #[test]
    fn tiled_q4_k_matmul_is_bit_identical() {
        if crate::skip_no_gpu() {
            return;
        }
        let (n_out, k) = (300, 512);
        let n_sb = n_out * k / K_SUPERBLOCK;
        let data = synth_q4_k(n_sb);
        let device = Default::default();
        let client = WgpuRuntime::client(&device);
        let w = Q4KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k).unwrap();
        for m in [2usize, 3, 17, 256, 1024] {
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i * 13 % 31) as f32 - 15.0) / 16.0)
                .collect();
            let x_h = client.create_from_slice(f32::as_bytes(&x));
            let un_h = w.matmul_device_with(&client, x_h.clone(), m, false);
            let ti_h = w.matmul_device_with(&client, x_h, m, true);
            let un = f32::from_bytes(&client.read_one_unchecked(un_h)).to_vec();
            let ti = f32::from_bytes(&client.read_one_unchecked(ti_h)).to_vec();
            assert_tiled_bit_identical(&un, &ti, &format!("q4_k m={m}"));
        }
    }

    /// Malformed inputs must be rejected, not mis-indexed.
    #[test]
    fn shape_validation() {
        if crate::skip_no_gpu() {
            return;
        }
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
