// Fused decode attention over the int8-quantized paged arena (K3c):
// the packed words and group scales are read in place and dequantized
// in-register — one f32 multiply per value, bit-for-bit the inverse of
// kv_quantize's packing (lanes 0..2 offset-binary as q+128 in [1,255],
// lane 3 two's-complement in the top byte, one scale per 32 values).
//
// PAGED ONLY: quant arenas exist only for global layers. `mode` is
// declared for scalar-layout symmetry with decode_attn.wgsl and must be
// 0 — dispatching this kernel at a contiguous store would silently
// apply paged addressing to it.
//
// The packed arenas bind as array<u32>: the backends fix IntElem = i32,
// so one Int element is one 32-bit word of 4 lanes. Binding budget note:
// 7 storage buffers + 1 uniform sits exactly at WebGPU's default
// maxStorageBuffersPerShaderStage — no headroom here.
//
// Everything else — online softmax tiling, tree reduces, the
// doubled-duty barrier, masking discipline, 3 KB workgroup storage — is
// decode_attn.wgsl transplanted; see that file and mod.rs for the
// contract this honors.

struct Params {
  n_q: vec2<u32>,
  n_kv: vec2<u32>,
  d: vec2<u32>,
  page_size: vec2<u32>,
  total: vec2<u32>,
  window: vec2<u32>,
  mode: vec2<u32>,
  scale: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> k_scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_packed: array<u32>;
@group(0) @binding(4) var<storage, read_write> v_scales: array<f32>;
@group(0) @binding(5) var<storage, read_write> table: array<u32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> q_s: array<f32, 256>;
var<workgroup> p_s: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

const NEG_MAX: f32 = -3.0e38;

// Arena ROW index for key j and kv head g (not an element index).
fn kv_row(j: u32, g: u32) -> u32 {
  let phys = table[j / params.page_size.x];
  return (phys * params.n_kv.x + g) * params.page_size.x + (j % params.page_size.x);
}

// One packed word -> four dequantized values. Core WGSL only: shifts,
// masks, bitcast — the arithmetic >> on i32 sign-extends lane 3.
fn unpack4(w: u32, scale: f32) -> vec4<f32> {
  return vec4<f32>(
    f32(i32(w & 0xffu) - 128),
    f32(i32((w >> 8u) & 0xffu) - 128),
    f32(i32((w >> 16u) & 0xffu) - 128),
    f32(bitcast<i32>(w) >> 24u),
  ) * scale;
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let h = wg.x;
  let lane = lid.x;
  let d = params.d.x;
  let dw = d / 4u;
  let dg = d / 32u;
  let total = params.total.x;
  let window = params.window.x;
  let g = h / (params.n_q.x / params.n_kv.x);
  let scale = bitcast<f32>(params.scale.x);

  if (lane < d) {
    q_s[lane] = q[h * d + lane];
  }
  // This lane's fixed geometry as an output column of V.
  let wp = lane / 4u;
  let shift = 8u * (lane % 4u);
  let hi = (lane % 4u) == 3u;
  let sg = lane / 32u;
  workgroupBarrier();

  var m = NEG_MAX;
  var s = 0.0;
  var o = 0.0;

  let tiles = (total + 255u) / 256u;
  for (var t = 0u; t < tiles; t += 1u) {
    let j = t * 256u + lane;
    let visible = j < total && (window == 0u || j + window >= total);
    var score = NEG_MAX;
    if (visible) {
      let r = kv_row(j, g);
      let wb = r * dw;
      let sb = r * dg;
      var dot = 0.0;
      for (var p = 0u; p < dw; p += 1u) {
        let kd = unpack4(k_packed[wb + p], k_scales[sb + p / 8u]);
        let c = p * 4u;
        dot += q_s[c] * kd.x;
        dot += q_s[c + 1u] * kd.y;
        dot += q_s[c + 2u] * kd.z;
        dot += q_s[c + 3u] * kd.w;
      }
      score = dot * scale;
    }

    red[lane] = score;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
      if (lane < stride) {
        red[lane] = max(red[lane], red[lane + stride]);
      }
      workgroupBarrier();
    }
    let m_new = max(m, red[0]);
    let p = select(0.0, exp(score - m_new), visible);
    p_s[lane] = p;
    // One barrier serves twice: every lane has read red[0] before the sum
    // reduce overwrites it, and p_s is fully published.
    workgroupBarrier();

    red[lane] = p;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
      if (lane < stride) {
        red[lane] += red[lane + stride];
      }
      workgroupBarrier();
    }

    let corr = exp(m - m_new);
    s = s * corr + red[0];
    m = m_new;

    if (lane < d) {
      o = o * corr;
      let limit = min(256u, total - t * 256u);
      for (var jj = 0u; jj < limit; jj += 1u) {
        let pv = p_s[jj];
        // Zero probability doubles as the bounds guard: kv_row is only
        // evaluated for keys that were visible when scored.
        if (pv != 0.0) {
          let r = kv_row(t * 256u + jj, g);
          let w = v_packed[r * dw + wp];
          let b = (w >> shift) & 0xffu;
          let qv = select(i32(b) - 128, bitcast<i32>(w) >> 24u, hi);
          o += pv * (f32(qv) * v_scales[r * dg + sg]);
        }
      }
    }
    workgroupBarrier();
  }

  if (lane < d) {
    out[h * d + lane] = o / s;
  }
}
