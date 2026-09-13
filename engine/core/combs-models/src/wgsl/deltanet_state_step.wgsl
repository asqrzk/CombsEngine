// The Gated DeltaNet state step for ONE decode token. One invocation per
// VALUE head h; key head h % kh (converted files tile V heads, so key heads
// repeat by tiling). Per head, all in f32 — mirroring qwen35.rs `delta()` at
// seq = 1, whose host road is the canary's reference:
//   q, k   L2-normalized key-head slices of the conv output; q / sqrt(kd)
//   beta   sigmoid(b[h]);  decay = exp(A[h] * softplus(a[h] + dt[h]))
//   S'     = decay * S + k (delta)^T,  delta = (v - k^T (decay * S)) * beta
//   y      = rmsnorm(q^T S') * normw * silu(z[h])
// Reads the old state, writes the new one beside it (never a live buffer).
//
//   conv      [2*kh*kd + vh*vd]  the conv step's SiLU output
//   z         [vh*vd]            the output gate
//   b, a      [vh]               beta logits, decay alpha
//   decay_a   [vh]               -exp(A_log), as converted
//   dt        [vh]               dt bias
//   normw     [vd]               gated RMSNorm weight
//   state_in  [vh*kd*vd]         row h, then key i, then value j
//   state_out [vh*kd*vd]
//   out       [vh*vd]
// Scalars: kh, vh, kd, vd, rms eps bits, l2 eps bits.

struct Params {
  kh: vec2<u32>,
  vh: vec2<u32>,
  kd: vec2<u32>,
  vd: vec2<u32>,
  rms_eps: vec2<u32>,
  l2_eps: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> conv: array<f32>;
@group(0) @binding(1) var<storage, read_write> z: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> a: array<f32>;
@group(0) @binding(4) var<storage, read_write> decay_a: array<f32>;
@group(0) @binding(5) var<storage, read_write> dt: array<f32>;
@group(0) @binding(6) var<storage, read_write> normw: array<f32>;
@group(0) @binding(7) var<storage, read_write> state_in: array<f32>;
@group(0) @binding(8) var<storage, read_write> state_out: array<f32>;
@group(0) @binding(9) var<storage, read_write> out: array<f32>;
@group(0) @binding(10) var<uniform> params: Params;

// Per-invocation scratch (function scope); kd and vd are 128 on every
// shipped geometry, and the door refuses anything past 256.
fn softplus(x: f32) -> f32 {
  return max(x, 0.0) + log(1.0 + exp(-abs(x)));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  let kh = params.kh.x;
  let vh = params.vh.x;
  let kd = params.kd.x;
  let vd = params.vd.x;
  if (h >= vh) {
    return;
  }
  var qn: array<f32, 256>;
  var kn: array<f32, 256>;
  var recalled: array<f32, 256>;
  var o: array<f32, 256>;
  let rms_eps = bitcast<f32>(params.rms_eps.x);
  let l2_eps = bitcast<f32>(params.l2_eps.x);
  let g = h % kh;
  let key_dim = kh * kd;
  let qb = g * kd;
  let kb = key_dim + g * kd;
  let vb = 2u * key_dim + h * vd;

  var qs = 0.0;
  var ks = 0.0;
  for (var i = 0u; i < kd; i += 1u) {
    qs += conv[qb + i] * conv[qb + i];
    ks += conv[kb + i] * conv[kb + i];
  }
  let qinv = inverseSqrt(qs + l2_eps) * inverseSqrt(f32(kd));
  let kinv = inverseSqrt(ks + l2_eps);
  for (var i = 0u; i < kd; i += 1u) {
    qn[i] = conv[qb + i] * qinv;
    kn[i] = conv[kb + i] * kinv;
  }

  let beta = 1.0 / (1.0 + exp(-b[h]));
  let decay = exp(decay_a[h] * softplus(a[h] + dt[h]));
  let sb = h * kd * vd;

  // recall k^T (decay * S)
  for (var j = 0u; j < vd; j += 1u) {
    recalled[j] = 0.0;
  }
  for (var i = 0u; i < kd; i += 1u) {
    for (var j = 0u; j < vd; j += 1u) {
      recalled[j] += kn[i] * state_in[sb + i * vd + j] * decay;
    }
  }
  // S' = decay * S + k delta^T, and read out q^T S'
  for (var j = 0u; j < vd; j += 1u) {
    o[j] = 0.0;
  }
  for (var i = 0u; i < kd; i += 1u) {
    for (var j = 0u; j < vd; j += 1u) {
      let delta = (conv[vb + j] - recalled[j]) * beta;
      let s = state_in[sb + i * vd + j] * decay + kn[i] * delta;
      state_out[sb + i * vd + j] = s;
      o[j] += qn[i] * s;
    }
  }
  // gated RMSNorm over the value dim
  var ms = 0.0;
  for (var j = 0u; j < vd; j += 1u) {
    ms += o[j] * o[j];
  }
  let inv = inverseSqrt(ms / f32(vd) + rms_eps);
  let zb = h * vd;
  for (var j = 0u; j < vd; j += 1u) {
    let gate = z[zb + j];
    out[zb + j] = o[j] * inv * normw[j] * (gate / (1.0 + exp(-gate)));
  }
}
