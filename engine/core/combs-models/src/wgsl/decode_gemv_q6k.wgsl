// Split-K decode gemv over a packed Q6_K table — the q8 kernel's
// occupancy shape with Q6_K's arithmetic mirrored from
// q6_k_dequant_kernel: q = ((ql nibble) | (qh 2-bit << 4)) − 32, times
// d · scales[i8], one i8 scale per 16 values (two per 32-value chunk,
// so the chunk sum splits into its two 16-halves). Scalars: n_out, k.

struct Params {
  n_out: vec2<u32>,
  k: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> ql: array<u32>;
@group(0) @binding(2) var<storage, read_write> qh: array<u32>;
@group(0) @binding(3) var<storage, read_write> sc: array<u32>;
@group(0) @binding(4) var<storage, read_write> d: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

const LANES: u32 = 16u;
const ROWS: u32 = 16u;

fn i8_sc(idx: u32) -> f32 {
  return f32(bitcast<i32>((sc[idx / 4u] << ((3u - idx % 4u) * 8u))) >> 24u);
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
      let half = cc / 4u;
      let t = cc % 4u;
      // ql bytes: sb·128 + half·64 + (t%2)·32 + l; qh bytes:
      // sb·64 + half·32 + l — both 32-byte runs, 8 words each.
      let qlb = sb * 32u + half * 16u + (t % 2u) * 8u;
      let qhb = sb * 16u + half * 8u;
      let nib_shift = (t / 2u) * 4u; // t >= 2 takes the high nibble
      let h_shift = t * 2u;
      let xb = c * 8u;
      // One i8 scale per 16 values: words 0..3 are l 0..15, words 4..7
      // are l 16..31.
      let s_lo = i8_sc(sb * 16u + half * 8u + 2u * t);
      let s_hi = i8_sc(sb * 16u + half * 8u + 1u + 2u * t);
      var sum_lo = 0.0;
      var sum_hi = 0.0;
      for (var w = 0u; w < 8u; w += 1u) {
        let lw = ql[qlb + w];
        let hw = qh[qhb + w];
        let xv = x[xb + w];
        let q0 = f32(((lw >> nib_shift) & 0xfu) | (((hw >> h_shift) & 3u) << 4u)) - 32.0;
        let q1 = f32(((lw >> (8u + nib_shift)) & 0xfu) | (((hw >> (8u + h_shift)) & 3u) << 4u)) - 32.0;
        let q2 = f32(((lw >> (16u + nib_shift)) & 0xfu) | (((hw >> (16u + h_shift)) & 3u) << 4u)) - 32.0;
        let q3 = f32(((lw >> (24u + nib_shift)) & 0xfu) | (((hw >> (24u + h_shift)) & 3u) << 4u)) - 32.0;
        let contrib = q0 * xv.x + q1 * xv.y + q2 * xv.z + q3 * xv.w;
        if (w < 4u) {
          sum_lo += contrib;
        } else {
          sum_hi += contrib;
        }
      }
      acc += d[sb] * (s_lo * sum_lo + s_hi * sum_hi);
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
