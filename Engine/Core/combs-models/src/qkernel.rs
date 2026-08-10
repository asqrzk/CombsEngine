//! Custom CubeCL kernels (native, owned by us — see QUANTIZATION_PLAN.md
//! "Kernel architecture"). This module starts with a minimal proof that the
//! CubeCL toolchain compiles and launches on our wgpu runtime; the fused Q4
//! dequant-matmul kernel builds on the same launch machinery.

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

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::WgpuRuntime;

    #[test]
    fn add_one_runs_on_gpu() {
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
}
