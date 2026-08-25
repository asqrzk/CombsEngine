// Scalar-echo probe — pins the host->kernel scalar contract by test.
//
// The runtime packs launch scalars as raw u64 slots into a uniform buffer
// appended as the LAST binding. Each slot arrives here as a vec2<u32>,
// low word in .x. Nothing else in the kernel suite is allowed to assume
// that layout until this kernel has proven it on the backend in use —
// naga natively, Tint in a browser — because a layout disagreement would
// not fail loudly: kernels would just read garbage parameters.
//
// out[0..2]: first slot's low and high words
// out[2]:    second slot's low word
// out[3]:    third slot's low word
// out[4]:    bitcast round-trip: f32 bits in slot 3's low word, doubled

struct Params {
    a: vec2<u32>,
    b: vec2<u32>,
    c: vec2<u32>,
    d: vec2<u32>,
};

@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x == 0u) {
        out[0] = params.a.x;
        out[1] = params.a.y;
        out[2] = params.b.x;
        out[3] = params.c.x;
        let f = bitcast<f32>(params.d.x);
        out[4] = bitcast<u32>(f * 2.0);
    }
}
