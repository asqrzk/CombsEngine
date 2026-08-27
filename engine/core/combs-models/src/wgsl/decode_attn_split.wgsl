// K8, pass 1: decode attention over ONE key segment, emitting flash
// partials. Deep contexts starved the single-pass kernel — one
// workgroup per query head walking every tile serially, 16 workgroups
// on a machine that seats hundreds — so the key range splits into
// segments and the grid becomes (n_q, segs): each workgroup runs the
// proven online-softmax over its own segment and writes its running
// max, sum and unnormalized output column; the combine pass merges
// them. Paged, global-visibility only (the dispatcher routes windowed
// and short totals to the single-pass kernel).
//
// Scalars: n_q, n_kv, d, page_size, total, seg_len, segs, scale.
// Partial layout: m/s at [h·segs + seg], o at [(h·segs + seg)·d + c].
// Contract per mod.rs: all storage read_write, vec4 q/k as in
// decode_attn.wgsl, workgroup 256, barriers uniform (segment bounds
// are uniform within a workgroup).

struct Params {
  n_q: vec2<u32>,
  n_kv: vec2<u32>,
  d: vec2<u32>,
  page_size: vec2<u32>,
  total: vec2<u32>,
  seg_len: vec2<u32>,
  segs: vec2<u32>,
  scale: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> k: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> table: array<u32>;
@group(0) @binding(4) var<storage, read_write> m_part: array<f32>;
@group(0) @binding(5) var<storage, read_write> s_part: array<f32>;
@group(0) @binding(6) var<storage, read_write> o_part: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> q_v: array<vec4<f32>, 64>;
var<workgroup> p_s: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

const NEG_MAX: f32 = -3.0e38;

fn kv_base(j: u32, g: u32) -> u32 {
  let phys = table[j / params.page_size.x];
  return ((phys * params.n_kv.x + g) * params.page_size.x
          + (j % params.page_size.x)) * params.d.x;
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let h = wg.x;
  let seg = wg.y;
  let lane = lid.x;
  let d = params.d.x;
  let total = params.total.x;
  let seg_start = seg * params.seg_len.x;
  let seg_end = min(seg_start + params.seg_len.x, total);
  let g = h / (params.n_q.x / params.n_kv.x);
  let scale = bitcast<f32>(params.scale.x);

  if (lane < d / 4u) {
    q_v[lane] = q[(h * d) / 4u + lane];
  }
  workgroupBarrier();

  var m = NEG_MAX;
  var s = 0.0;
  var o = 0.0;

  let span = seg_end - seg_start;
  let tiles = (span + 255u) / 256u;
  for (var t = 0u; t < tiles; t += 1u) {
    let j = seg_start + t * 256u + lane;
    let visible = j < seg_end;
    var score = NEG_MAX;
    if (visible) {
      let base = kv_base(j, g) / 4u;
      var acc = 0.0;
      for (var c = 0u; c < d / 4u; c += 1u) {
        acc += dot(q_v[c], k[base + c]);
      }
      score = acc * scale;
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
      let limit = min(256u, span - t * 256u);
      for (var jj = 0u; jj < limit; jj += 1u) {
        let pv = p_s[jj];
        if (pv != 0.0) {
          o += pv * v[kv_base(seg_start + t * 256u + jj, g) + lane];
        }
      }
    }
    workgroupBarrier();
  }

  let slot = h * params.segs.x + seg;
  if (lane == 0u) {
    m_part[slot] = m;
    s_part[slot] = s;
  }
  if (lane < d) {
    o_part[slot * d + lane] = o;
  }
}
