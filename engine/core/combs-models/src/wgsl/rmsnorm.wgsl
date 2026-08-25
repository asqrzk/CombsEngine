// Fused RMSNorm: one workgroup per row, y = x * inv_rms(x) * (w + flavor).
//
// flavor crosses as f32 bits: 0.0 is the plain norm, 1.0 the gemma
// zero-centered weight. eps is added to mean(x²) inside the sqrt, matching
// the reference in norm.rs. Scalar slots: rows, n, eps(bits), flavor(bits).
//
// Contract notes (see mod.rs): every storage buffer is read_write; the
// strided loops contain no barriers, and the reduction barriers sit in
// uniform control flow — lane guards mask writes, they never skip a
// barrier.

struct Params {
  rows: vec2<u32>,
  n: vec2<u32>,
  eps: vec2<u32>,
  flavor: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wg.x;
  let lane = lid.x;
  let n = params.n.x;
  let base = row * n;

  var sum = 0.0;
  for (var i = lane; i < n; i += 256u) {
    let v = x[base + i];
    sum += v * v;
  }
  scratch[lane] = sum;
  workgroupBarrier();

  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      scratch[lane] += scratch[lane + stride];
    }
    workgroupBarrier();
  }

  let inv_rms = 1.0 / sqrt(scratch[0] / f32(n) + bitcast<f32>(params.eps.x));
  let flavor = bitcast<f32>(params.flavor.x);
  for (var i = lane; i < n; i += 256u) {
    out[base + i] = x[base + i] * inv_rms * (w[i] + flavor);
  }
}
