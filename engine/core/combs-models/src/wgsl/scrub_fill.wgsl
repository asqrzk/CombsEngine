// Scrub fill — writes zero into every u32 word of one storage buffer, whatever the tensor above it
// holds (f32, f16 or packed i32), so a session's cache bytes are gone before the buffer returns to
// the pool. A constant write, never arithmetic: x * 0.0 keeps the sign bit (-0.0) and keeps NaN.
//
// slot a: words to clear (the binding's bytes / 4)
// slot b: groups in one row of the dispatch grid (the grid turns 2-D past 65,535 groups)

struct Params {
    words: vec2<u32>,
    row: vec2<u32>,
};

@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * params.row.x * 256u;
    if (i < params.words.x) {
        buf[i] = 0u;
    }
}
