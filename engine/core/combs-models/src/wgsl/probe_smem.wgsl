// Shared-memory + barrier probe — the cooperative-correctness proof every
// staged kernel stands on, in the same spirit as the #[cube] mirror probe
// the quant family runs before trusting its tiled kernels.
//
// Each workgroup stages its 256-element window REVERSED into workgroup
// memory, barriers, and reads back its own lane: out[i] must equal the
// value thread (255 - lane) staged, i.e. the element mirrored within the
// window, plus one. A broken barrier or a miscompiled workgroup array
// yields wrong VALUES here, not a compile error — which is exactly why
// this runs before any kernel that stages data.
//
// The barrier sits OUTSIDE the bounds guard: every lane of the workgroup
// reaches it even in a ragged final window (WGSL uniformity requirement).
// Out-of-range lanes stage 0.0 and never write out.

struct Params {
    n: vec2<u32>,
};

var<workgroup> scratch: array<f32, 256>;

// Both storage buffers are read_write even though input is only read:
// the pool sub-slices allocations, so two bindings routinely live in ONE
// wgpu buffer, and wgpu rejects a dispatch that uses one buffer as both
// read-only and read-write. Uniform read_write declarations make the
// usage compatible regardless of which page each binding landed on.
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n = params.n.x;
    let base = wid.x * 256u;
    let idx = base + lid.x;

    var v = 0.0;
    if (idx < n) {
        v = input[idx];
    }
    scratch[255u - lid.x] = v;

    workgroupBarrier();

    if (idx < n) {
        out[idx] = scratch[lid.x] + 1.0;
    }
}
