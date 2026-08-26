// Split-K decode gemv over a packed Q4_K table — decode_gemv_q8's
// occupancy shape (16 rows x 16 lanes per workgroup, lane-strided
// 32-value chunks, smem tree combine) with Q4_K's arithmetic mirrored
// from q4_k_dequant_kernel: per sub-block, value = (d·sc)·q − (dmin·m),
// applied via the ggml sum-split Σ(dsc·q − dm)·x = dsc·Σq·x − dm·Σx so
// scales touch each chunk once. Chunk index cc within a superblock IS
// the scale index (sidx = 2j + t = cc). Scalars: n_out, k.

struct Params {
  n_out: vec2<u32>,
  k: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> qs: array<u32>;
@group(0) @binding(2) var<storage, read_write> dd: array<f32>;
@group(0) @binding(3) var<storage, read_write> scales: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

const LANES: u32 = 16u;
const ROWS: u32 = 16u;

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

@compute @workgroup_size(256)
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
      // 32 nibble-bytes of group j start at byte sb·128 + j·32.
      let wb = sb * 32u + j * 8u;
      let xb = c * 8u;
      let shift = t * 4u;
      var s1 = 0.0;
      var s2 = 0.0;
      for (var w = 0u; w < 8u; w += 1u) {
        let word = qs[wb + w];
        let xv = x[xb + w];
        s1 += f32((word >> (shift)) & 0xfu) * xv.x;
        s1 += f32((word >> (8u + shift)) & 0xfu) * xv.y;
        s1 += f32((word >> (16u + shift)) & 0xfu) * xv.z;
        s1 += f32((word >> (24u + shift)) & 0xfu) * xv.w;
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
    out[row] = partial[local];
  }
}
