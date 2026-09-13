// The Gated DeltaNet conv step for ONE decode token. One invocation per
// channel of the q|k|v stream: the causal depthwise conv reads the ring's
// kernel-1 previous columns and this token, SiLU applies, and the next ring
// (oldest column dropped, this token appended) is written beside it. The
// house writes fresh buffers, never a live tensor's; all f32, since the ring
// is recurrent state and state never narrows.
//
//   qkv      [channels]              this token's in_proj_qkv output
//   w        [channels * kernel]     the conv kernel, row c = channel c
//   ring_in  [channels * (kernel-1)] the previous columns, oldest first
//   ring_out [channels * (kernel-1)] the next ring
//   out      [channels]              SiLU(conv), what the state step reads
// Scalars: channels, kernel.

struct Params {
  channels: vec2<u32>,
  kernel: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> ring_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> ring_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= params.channels.x) {
    return;
  }
  let k = params.kernel.x;
  let hist = k - 1u;
  let rb = c * hist;
  let wb = c * k;
  // torch's conv1d is a correlation: window column j takes w[j], the
  // window being [ring oldest .. ring newest, this token].
  var acc = w[wb + hist] * qkv[c];
  for (var j = 0u; j < hist; j += 1u) {
    acc += w[wb + j] * ring_in[rb + j];
  }
  // SiLU: x * sigmoid(x)
  out[c] = acc / (1.0 + exp(-acc));
  for (var j = 0u; j + 1u < hist; j += 1u) {
    ring_out[rb + j] = ring_in[rb + j + 1u];
  }
  if (hist > 0u) {
    ring_out[rb + hist - 1u] = qkv[c];
  }
}
