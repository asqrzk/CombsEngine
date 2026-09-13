{{f16_enable}}// Split-K decode gemv over a packed Q5_K table — decode_gemv_q4k's
// occupancy shape (16 rows x 16 lanes per workgroup, lane-strided
// 32-value chunks, smem tree combine) with Q5_K's fifth bit folded in,
// mirrored from q5_k_dequant_kernel: per sub-block, value =
// (d·sc)·(nib | hi«4) − (dmin·m), applied via the same sum-split
// Σ(dsc·q − dm)·x = dsc·Σq·x − dm·Σx. Chunk index cc within a superblock
// IS the scale index AND the high-bit index: value l of chunk cc reads
// bit cc of high byte l, and high byte l of every chunk lives in word
// sb·8 + l/4 — the same word stride as the nibbles. Scalars: n_out, k.

struct Params {
  n_out: vec2<u32>,
  k: vec2<u32>,
  fuse: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> x: array<{{act4}}>;
@group(0) @binding(1) var<storage, read_write> qs: array<u32>;
@group(0) @binding(2) var<storage, read_write> qh: array<u32>;
@group(0) @binding(3) var<storage, read_write> dd: array<f32>;
@group(0) @binding(4) var<storage, read_write> scales: array<u32>;
@group(0) @binding(5) var<storage, read_write> out: array<{{act}}>;
// The residual stream to add in the epilogue when params.fuse.x == 1u;
// bound to `out` itself (never read) when nothing is fused, so the
// pipeline layout never changes. Read_write like every buffer here.
@group(0) @binding(6) var<storage, read_write> residual: array<{{act}}>;
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

const LANES: u32 = {{lanes}}u;
const ROWS: u32 = {{rows}}u;

fn byte_sc(idx: u32) -> u32 {
  return (scales[idx / 4u] >> ((idx % 4u) * 8u)) & 0xffu;
}

// ggml get_scale_min_k4, both halves (j = sidx in 0..8).
fn k4_scale(base: u32, j: u32) -> u32 {
  if (j < 4u) {
    return byte_sc(base + j) & 63u;
  }
  return (byte_sc(base + j + 4u) & 0xfu) | ((byte_sc(base + j - 4u) >> 6u) << 4u);
}

fn k4_min(base: u32, j: u32) -> u32 {
  if (j < 4u) {
    return byte_sc(base + j + 4u) & 63u;
  }
  return (byte_sc(base + j + 4u) >> 4u) | ((byte_sc(base + j) >> 6u) << 4u);
}

// One 5-bit value: the nibble at `shift` of `word`, the high bit at
// `hbit` of `high` (both already offset to the value's byte).
fn q5(word: u32, high: u32, shift: u32, hbit: u32) -> f32 {
  return f32(((word >> shift) & 0xfu) | (((high >> hbit) & 1u) << 4u));
}

@compute @workgroup_size({{rows}} * {{lanes}})
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let local = lid.x;
  let lane = local % LANES;
  let row = wg.x * ROWS + local / LANES;
  let k = params.k.x;
  let chunks = k / 32u;

  var acc = 0.0;
  if (row < params.n_out.x) {
    for (var c = lane; c < chunks; c += LANES) {
      let abs_c = row * chunks + c;
      let sb = abs_c / 8u;
      let cc = abs_c % 8u;
      let j = cc / 2u;
      let t = cc % 2u;
      let dsc = dd[sb * 2u] * f32(k4_scale(sb * 12u, cc));
      let dmn = dd[sb * 2u + 1u] * f32(k4_min(sb * 12u, cc));
      // 32 nibble-bytes of group j start at byte sb·128 + j·32; the 32
      // high bytes of the superblock at byte sb·32 of qh.
      let wb = sb * 32u + j * 8u;
      let hb = sb * 8u;
      let xb = c * 8u;
      let shift = t * 4u;
      var s1 = 0.0;
      var s2 = 0.0;
      for (var w = 0u; w < 8u; w += 1u) {
        let word = qs[wb + w];
        let high = qh[hb + w];
        let xv = vec4<f32>(x[xb + w]);
        s1 += q5(word, high, shift, cc) * xv.x;
        s1 += q5(word, high, 8u + shift, 8u + cc) * xv.y;
        s1 += q5(word, high, 16u + shift, 16u + cc) * xv.z;
        s1 += q5(word, high, 24u + shift, 24u + cc) * xv.w;
        s2 += xv.x + xv.y + xv.z + xv.w;
      }
      acc += dsc * s1 - dmn * s2;
    }
  }
  partial[local] = acc;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      partial[local] += partial[local + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u && row < params.n_out.x) {
    var y = partial[local];
    if (params.fuse.x == 1u) {
      y += f32(residual[row]);
    }
    out[row] = {{act}}(y);
  }
}
