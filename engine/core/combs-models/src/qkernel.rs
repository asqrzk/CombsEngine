//! Custom CubeCL kernels (native, owned by us). This module starts with a
//! minimal proof that the CubeCL toolchain compiles and launches on our
//! wgpu runtime; the fused dequant-matmul kernels build on the same launch
//! machinery.

#![allow(dead_code)]

use cubecl::prelude::*;

/// Minimal proof kernel: `output[i] = input[i] + 1`. Exists only to validate
/// the `#[cube]` → launch → readback path end to end.
#[cube(launch_unchecked)]
fn add_one_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1.0;
    }
}

/// Cooperative shared-memory proof: each cube stages its 256-wide slice of
/// `input` in shared memory, barriers, then every thread reads the slot
/// staged by the *mirrored* thread of the same cube —
/// `output[c*256 + j] = input[c*256 + (255-j)]`.
///
/// A broken shared-memory or barrier implementation produces wrong values,
/// not a compile error, so this must pass on a backend before any fused
/// kernel is allowed to rely on cooperative staging there.
#[cube(launch_unchecked)]
fn smem_mirror_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    let mut staged = SharedMemory::<f32>::new(256usize);
    let unit = UNIT_POS as usize;
    let cdim = CUBE_DIM as usize;
    if ABSOLUTE_POS < input.len() {
        staged[unit] = input[ABSOLUTE_POS];
    }
    sync_cube();
    // XOR with cdim-1 mirrors the lane within a power-of-two cube.
    let mirror = unit ^ (cdim - 1);
    let src = CUBE_POS * cdim + mirror;
    if ABSOLUTE_POS < output.len() && src < input.len() {
        output[ABSOLUTE_POS] = staged[mirror];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::WgpuRuntime;

    #[test]
    fn add_one_runs_on_gpu() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        let client = WgpuRuntime::client(&device);

        let input = [1.0f32, 2.0, 3.0, 4.0];
        let n = input.len();
        let in_h = client.create_from_slice(f32::as_bytes(&input));
        let out_h = client.empty(n * core::mem::size_of::<f32>());

        unsafe {
            add_one_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(n as u32),
                ArrayArg::from_raw_parts(in_h, n),
                ArrayArg::from_raw_parts(out_h.clone(), n),
            );
        }

        let bytes = client.read_one_unchecked(out_h);
        let got = f32::from_bytes(&bytes);
        assert_eq!(got, &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn shared_memory_mirror_runs_on_gpu() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        let client = WgpuRuntime::client(&device);

        let n = 512usize; // two full 256-thread cubes
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let in_h = client.create_from_slice(f32::as_bytes(&input));
        let out_h = client.empty(n * core::mem::size_of::<f32>());

        unsafe {
            smem_mirror_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(2, 1, 1),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts(in_h, n),
                ArrayArg::from_raw_parts(out_h.clone(), n),
            );
        }

        let bytes = client.read_one_unchecked(out_h);
        let got = f32::from_bytes(&bytes);
        for cube in 0..2usize {
            for j in 0..256usize {
                let want = input[cube * 256 + (255 - j)];
                assert_eq!(
                    got[cube * 256 + j],
                    want,
                    "cube {cube} lane {j}: shared-memory staging or barrier is broken"
                );
            }
        }
    }
}
