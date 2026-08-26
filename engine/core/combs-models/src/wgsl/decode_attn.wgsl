// Fused decode attention (flash-decode, M = 1): one workgroup per query
// head, K/V read straight from the paged arena — no gather, no repeat_kv,
// no materialized scores row.
//
// mode 0 (paged): k/v are the arena [num_pages, n_kv, page_size, d] and
// key j lives at ((table[j/page_size] * n_kv + g) * page_size +
// j % page_size) * d for kv head g = h / (n_q / n_kv).
// mode 1 (contiguous): k/v are [1, n_kv, total, d] and key j lives at
// (g * total + j) * d.
//
// Visibility for the single query (absolute position total-1):
// j < total, and when window != 0 additionally j + window >= total.
// Masked lanes contribute an exact 0 probability via select — never a
// skipped barrier, and never exp(0) leaking in from a fully-masked tile.
//
// Online softmax per 256-key tile: each lane scores one key, a tree
// reduce finds the tile max, probabilities go to shared memory, and each
// lane c < d accumulates its output column over the tile with the
// standard running-max correction. Workgroup storage: 3 KB (q + p + red),
// a recorded deviation from the 1 KB house line, far under wasm's 16 KB.
//
// Contract notes (see mod.rs): all storage buffers read_write; scalars as
// u64 slots (low word .x, f32 as bits); no infinity literals — the
// sentinel is -3.0e38; every barrier sits in uniform control flow (tile
// count, reduce strides and the dot/accumulate bounds are all uniform).

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

// q and k bind as vec4 words: the K-dot walks whole rows per lane, and
// a 16-byte load per four values is the difference between one load
// instruction per element and one per four. Row bases stay element
// counts; divide by 4 at the access. Requires d % 4 == 0 (the
// dispatcher guards; every real head dim qualifies). v stays scalar —
// its access is one column per lane, already coalesced across lanes.
@group(0) @binding(0) var<storage, read_write> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> k: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> table: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> q_v: array<vec4<f32>, 64>;
var<workgroup> p_s: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

const NEG_MAX: f32 = -3.0e38;

fn kv_base(j: u32, g: u32) -> u32 {
  if (params.mode.x == 0u) {
    let phys = table[j / params.page_size.x];
    return ((phys * params.n_kv.x + g) * params.page_size.x
            + (j % params.page_size.x)) * params.d.x;
  }
  return (g * params.total.x + j) * params.d.x;
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let h = wg.x;
  let lane = lid.x;
  let d = params.d.x;
  let total = params.total.x;
  let window = params.window.x;
  let g = h / (params.n_q.x / params.n_kv.x);
  let scale = bitcast<f32>(params.scale.x);

  if (lane < d / 4u) {
    q_v[lane] = q[(h * d) / 4u + lane];
  }
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
      let limit = min(256u, total - t * 256u);
      for (var jj = 0u; jj < limit; jj += 1u) {
        let pv = p_s[jj];
        // Zero probability doubles as the bounds guard: kv_base is only
        // evaluated for keys that were visible when scored.
        if (pv != 0.0) {
          o += pv * v[kv_base(t * 256u + jj, g) + lane];
        }
      }
    }
    workgroupBarrier();
  }

  if (lane < d) {
    out[h * d + lane] = o / s;
  }
}
