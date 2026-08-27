// K9: paged causal prefill attention — seq > 1 queries attend straight
// from the paged arena, no gathered window, no autotune.
//
// Tile-resident flash: a workgroup owns one head and QT = 16 query
// rows whose q vectors stage into shared memory ONCE; key tiles of
// KT = 8 stage per iteration; scores, the per-row online-softmax state
// and the PV accumulation all work from shared memory. d <= 128 (the
// dispatcher gates; larger head dims keep the materialized path) keeps
// the whole working set — q 8 KB + k 4 KB + scores/state under 1 KB —
// inside the 16 KB wasm budget.
//
// Lane maps: score phase (m, n) = (lane / KT, lane % KT) with lanes
// >= 128 idle; PV phase (m, c-lane) = (lane / 16, lane % 16), each
// c-lane owning d/16 interleaved output columns. Global visibility
// only; the absolute position of query row r is pos + q0 + r.
//
// Scalars: n_q, n_kv, d, page_size, total, pos, seq, scale.
// Contract per mod.rs: storage read_write, q/k as vec4, no subgroups,
// barriers uniform (masks select values, never skip barriers).

struct Params {
  n_q: vec2<u32>,
  n_kv: vec2<u32>,
  d: vec2<u32>,
  page_size: vec2<u32>,
  total: vec2<u32>,
  pos: vec2<u32>,
  seq: vec2<u32>,
  scale: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> k: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> table: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const QT: u32 = 16u;
const KT: u32 = 8u;
const NEG_MAX: f32 = -3.0e38;

// q tile: 16 rows x 128/4 = 512 vec4 max (8 KB).
var<workgroup> q_tile: array<vec4<f32>, 512>;
// k tile: 8 keys x 32 vec4 = 256 vec4 max (4 KB).
var<workgroup> k_tile: array<vec4<f32>, 256>;
// score tile [QT][KT], then probabilities in place.
var<workgroup> score: array<f32, 128>;
var<workgroup> row_m: array<f32, 16>;
var<workgroup> row_l: array<f32, 16>;
var<workgroup> row_corr: array<f32, 16>;

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
  let q0 = wg.y * QT;
  let lane = lid.x;
  let d = params.d.x;
  let dw = d / 4u;
  let seq = params.seq.x;
  let total = params.total.x;
  let g = h / (params.n_q.x / params.n_kv.x);
  let scale = bitcast<f32>(params.scale.x);

  // Stage the q tile once: QT rows x dw words.
  for (var w = lane; w < QT * dw; w += 256u) {
    let r = w / dw;
    let c = w % dw;
    if (q0 + r < seq) {
      q_tile[r * dw + c] = q[(h * seq + q0 + r) * dw + c];
    }
  }
  if (lane < QT) {
    row_m[lane] = NEG_MAX;
    row_l[lane] = 0.0;
  }
  workgroupBarrier();

  let m = lane / KT;
  let n = lane % KT;
  let pm = lane / 16u;
  let pc = lane % 16u;
  let q_abs = params.pos.x + q0 + m;

  var o: array<f32, 8>;
  for (var i = 0u; i < 8u; i += 1u) {
    o[i] = 0.0;
  }

  let last_abs = min(params.pos.x + q0 + QT - 1u, params.pos.x + seq - 1u);
  let kmax = min(last_abs + 1u, total);
  let tiles = (kmax + KT - 1u) / KT;

  for (var t = 0u; t < tiles; t += 1u) {
    let jbase = t * KT;
    // Stage the key tile.
    for (var w = lane; w < KT * dw; w += 256u) {
      let kr = w / dw;
      let kc = w % dw;
      let j = jbase + kr;
      if (j < kmax) {
        k_tile[kr * dw + kc] = k[kv_base(j, g) / 4u + kc];
      }
    }
    workgroupBarrier();

    // Scores from shared memory only; lanes >= QT*KT idle here.
    if (lane < QT * KT) {
      let j = jbase + n;
      let visible = (q0 + m) < seq && j < kmax && j <= q_abs;
      var sc = NEG_MAX;
      if (visible) {
        var acc = 0.0;
        for (var c = 0u; c < dw; c += 1u) {
          acc += dot(q_tile[m * dw + c], k_tile[n * dw + c]);
        }
        sc = acc * scale;
      }
      score[m * KT + n] = sc;
    }
    workgroupBarrier();

    // Per-row online-softmax update; one lane per row scans its KT slots.
    if (lane < QT) {
      var tile_max = NEG_MAX;
      for (var i = 0u; i < KT; i += 1u) {
        tile_max = max(tile_max, score[lane * KT + i]);
      }
      let m_old = row_m[lane];
      let m_new = max(m_old, tile_max);
      var l_add = 0.0;
      for (var i = 0u; i < KT; i += 1u) {
        let s = score[lane * KT + i];
        var p = 0.0;
        if (s > NEG_MAX) {
          p = exp(s - m_new);
        }
        score[lane * KT + i] = p;
        l_add += p;
      }
      let corr = exp(m_old - m_new);
      row_m[lane] = m_new;
      row_l[lane] = row_l[lane] * corr + l_add;
      row_corr[lane] = corr;
    }
    workgroupBarrier();

    // PV: lane (pm, pc) folds the tile into its interleaved columns.
    let corr = row_corr[pm];
    for (var s = 0u; s < d / 16u; s += 1u) {
      let c = pc + s * 16u;
      var acc = o[s] * corr;
      for (var i = 0u; i < KT; i += 1u) {
        let p = score[pm * KT + i];
        if (p != 0.0) {
          acc += p * v[kv_base(jbase + i, g) + c];
        }
      }
      o[s] = acc;
    }
    workgroupBarrier();
  }

  if (q0 + pm < seq) {
    let l = row_l[pm];
    for (var s = 0u; s < d / 16u; s += 1u) {
      let c = pc + s * 16u;
      out[(h * seq + q0 + pm) * d + c] = o[s] / l;
    }
  }
}
