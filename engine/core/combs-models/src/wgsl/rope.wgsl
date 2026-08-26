// Fused RoPE: q and k rotated in one elementwise dispatch, half-split
// rotate_half convention (out = x·cos + rotate_half(x)·sin), tables
// indexed at absolute position pos + r. No shared memory, no barriers —
// the early return on the ragged tail is legal here and nowhere else in
// this module.
//
// Element space is q's elements followed by k's: e < n_q·seq·d addresses
// q, the rest address k. Scalar slots: n_q, n_kv, seq, d, pos.

struct Params {
  n_q: vec2<u32>,
  n_kv: vec2<u32>,
  seq: vec2<u32>,
  d: vec2<u32>,
  pos: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> cos_t: array<f32>;
@group(0) @binding(3) var<storage, read_write> sin_t: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_q: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_k: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = params.d.x;
  let seq = params.seq.x;
  let q_total = params.n_q.x * seq * d;
  let total = q_total + params.n_kv.x * seq * d;
  let e = gid.x;
  if (e >= total) {
    return;
  }

  let in_q = e < q_total;
  let i = select(e - q_total, e, in_q);
  let r = (i / d) % seq;
  let c = i % d;
  let half = d / 2u;
  let partner = select(i - half, i + half, c < half);

  var x: f32;
  var mate: f32;
  if (in_q) {
    x = q[i];
    mate = q[partner];
  } else {
    x = k[i];
    mate = k[partner];
  }
  let rot = select(mate, -mate, c < half);

  let t = (params.pos.x + r) * d + c;
  let y = x * cos_t[t] + rot * sin_t[t];
  if (in_q) {
    out_q[i] = y;
  } else {
    out_k[i] = y;
  }
}
