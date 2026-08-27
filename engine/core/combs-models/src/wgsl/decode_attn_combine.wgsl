// K8, pass 2: merge the per-segment flash partials into the final
// attention output. One workgroup per query head; every lane serially
// folds the (few) segments — a global max first, then the standard
// rescaled sum: out[c] = Σ exp(m_i − M)·o_i[c] / Σ exp(m_i − M)·s_i.
// No shared memory, no barriers; every segment holds at least one key
// by construction so the denominator is positive.
//
// Scalars: n_q, d, segs.

struct Params {
  n_q: vec2<u32>,
  d: vec2<u32>,
  segs: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> m_part: array<f32>;
@group(0) @binding(1) var<storage, read_write> s_part: array<f32>;
@group(0) @binding(2) var<storage, read_write> o_part: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let h = wg.x;
  let lane = lid.x;
  let d = params.d.x;
  let segs = params.segs.x;
  if (lane >= d) {
    return;
  }

  var m = m_part[h * segs];
  for (var i = 1u; i < segs; i += 1u) {
    m = max(m, m_part[h * segs + i]);
  }
  var s = 0.0;
  var o = 0.0;
  for (var i = 0u; i < segs; i += 1u) {
    let slot = h * segs + i;
    let w = exp(m_part[slot] - m);
    s += w * s_part[slot];
    o += w * o_part[slot * d + lane];
  }
  out[h * d + lane] = o / s;
}
