// Split-K decode gemv over a packed Q8_0 table: y = W·x for one token.
//
// The #[cube] untiled gemv launches one thread per output row — 1–3k
// threads for the per-layer projections, an order of magnitude under
// what the GPU seats, while each thread serially walks the whole row.
// Here a workgroup covers ROWS=16 rows with LANES=16 lanes each: lane l
// of a row sums the row's blocks {l, l+16, l+32, ...} (block-strided —
// aligned for every k % 32 == 0), partials tree-reduce through shared
// memory, lane 0 writes the row. 16× the occupancy, same arithmetic
// per value, one f32 multiply per block scale exactly like the cube
// kernel it shadows.
//
// x binds as vec4 (k % 4 == 0 always holds when k % 32 == 0); qs binds
// as the same u32 words the arena stores (4 signed bytes each,
// sign-extended by shift pairs exactly like q8_0_matmul_kernel).
// Scalars: n_out, k. Contract per mod.rs: all storage read_write,
// workgroup 256, barriers uniform (row/lane guards mask values only).

struct Params {
  n_out: vec2<u32>,
  k: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> qs: array<u32>;
@group(0) @binding(2) var<storage, read_write> d: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partial: array<f32, 256>;

const LANES: u32 = 16u;
const ROWS: u32 = 16u;

fn sext8(w: u32, byte: u32) -> f32 {
  return f32((bitcast<i32>(w << ((3u - byte) * 8u))) >> 24u);
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let local = lid.x;
  let lane = local % LANES;
  let row_in_wg = local / LANES;
  let row = wg.x * ROWS + row_in_wg;
  let k = params.k.x;
  let blocks_per_row = k / 32u;

  var acc = 0.0;
  if (row < params.n_out.x) {
    for (var b = lane; b < blocks_per_row; b += LANES) {
      let block = row * blocks_per_row + b;
      let scale = d[block];
      let qbase = block * 8u;
      // x spans ONE row: index by the in-row block, not the absolute one.
      let xbase = b * 8u; // 32 values = 8 vec4 words
      var s = 0.0;
      for (var w = 0u; w < 8u; w += 1u) {
        let word = qs[qbase + w];
        let xv = x[xbase + w];
        s += sext8(word, 0u) * xv.x;
        s += sext8(word, 1u) * xv.y;
        s += sext8(word, 2u) * xv.z;
        s += sext8(word, 3u) * xv.w;
      }
      acc += s * scale;
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
