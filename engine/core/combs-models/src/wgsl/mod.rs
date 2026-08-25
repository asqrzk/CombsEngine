//! Hand-written WGSL kernels, launched on the runtime's own stream.
//!
//! The #[cube] DSL serves the quant-gemv family well; these kernels exist
//! for the places where owning the exact WGSL matters — reductions and
//! attention whose selection must be fixed by shape, not autotuned, and
//! whose text must be auditable against the two validators it will meet
//! (naga natively, Tint in a browser).
//!
//! ## The contract every kernel here honors
//!
//! - Entry point named `main`; buffers bound at `@group(0) @binding(i)`
//!   in the exact order pushed into [`KernelArguments`].
//! - Launch scalars travel as raw u64 slots in a uniform buffer appended
//!   as the LAST binding. In WGSL that is a struct of `vec2<u32>` fields,
//!   low word in `.x`; f32 scalars cross as bits (`to_bits` host-side,
//!   `bitcast<f32>` in the shader). This layout is not assumed — the
//!   scalar-echo probe pins it by test on every backend before any other
//!   kernel's door opens.
//! - The runtime injects no bounds checks into template kernels: every
//!   access self-guards. No `enable` directives, ever — a browser rejects
//!   the shader and then dispatches the invalid pipeline as a silent
//!   no-op, which is how a model decodes zeros with no error. No
//!   subgroup operations for the same reason. No infinity literals (WGSL
//!   forbids them); large finite sentinels instead.
//! - One-dimensional workgroups of [`WORKGROUP`] threads; barriers only
//!   in uniform control flow — guards mask values, they never skip a
//!   barrier.
//! - **Every storage buffer is declared `read_write`**, including pure
//!   inputs. The pool sub-slices allocations, so two bindings routinely
//!   share one `wgpu::Buffer` — and wgpu rejects a dispatch using one
//!   buffer as both read-only and read-write. The shared-memory probe
//!   found this on day one; uniform declarations retire the whole class.
//! - f32 compute throughout; dtype casts happen outside the launch.
//!
//! Kernels are accelerators, never load-bearing: every integration point
//! offers the kernel through a `try_*` door that returns `None` into the
//! existing burn path, and every door reads its environment once per
//! process.

// The probes are exercised from tests until the first production kernel
// (Stage 2) takes `launch` into the forward path.
#![cfg_attr(not(test), allow(dead_code))]

use burn_cubecl::template::{KernelSource, SourceKernel, SourceTemplate};
use cubecl::prelude::{CubeCount, CubeDim, KernelId, Runtime};
use cubecl::server::{Binding, KernelArguments, MetadataBindingInfo};

use burn::backend::wgpu::WgpuRuntime;

/// Threads per workgroup for every kernel in this module. Must match the
/// literal `@workgroup_size(256)` in each `.wgsl` file — the two cannot
/// drift, because [`launch`] passes this value as the task's `CubeDim`
/// and the runtime validates launches against it.
pub(crate) const WORKGROUP: u32 = 256;

/// Master door for the WGSL kernel suite. `COMBS_WGSL=0` restores the
/// burn path everywhere; per-kernel doors layer on top. Read once — a
/// door that flips mid-process would let autotuned and fixed paths
/// interleave within one generation, which no test covers.
pub(crate) fn wgsl_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("COMBS_WGSL").as_deref(), Ok("0")))
}

/// Declares a [`KernelSource`] over an included `.wgsl` file.
macro_rules! wgsl_kernel {
    ($name:ident, $file:expr) => {
        pub(crate) struct $name;

        impl KernelSource for $name {
            fn source(&self) -> SourceTemplate {
                SourceTemplate::new(include_str!($file))
            }

            fn id(&self) -> KernelId {
                KernelId::new::<Self>().cube_dim(CubeDim::new_1d(WORKGROUP))
            }
        }
    };
}

wgsl_kernel!(ProbeEcho, "probe_echo.wgsl");
wgsl_kernel!(ProbeSmem, "probe_smem.wgsl");

/// Launches one WGSL kernel on the runtime's stream.
///
/// The task joins the same encoder and the same submit as the surrounding
/// burn operations — ordering with neighbouring ops needs no flush and no
/// fence. Checked execution mode is used deliberately: template kernels
/// receive no injected guards either way, and Checked avoids the
/// unchecked entry point's extra obligations.
pub(crate) fn launch(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    kernel: impl KernelSource,
    count: CubeCount,
    buffers: Vec<Binding>,
    scalars: Vec<u64>,
) {
    let task: <<WgpuRuntime as Runtime>::Server as cubecl::server::ComputeServer>::Kernel =
        Box::new(SourceKernel::new(kernel, CubeDim::new_1d(WORKGROUP)));
    client.launch(
        task,
        count,
        KernelArguments::new()
            .with_buffers(buffers)
            .with_info(MetadataBindingInfo::custom(scalars)),
    );
}

/// Runs both probes against the current device and reports the first
/// contract violation, if any.
///
/// Async because a browser can only read buffers back asynchronously;
/// the sync tests below remain the native gate. The wasm surface exposes
/// this so the SAME checks that naga passed can be demanded of Tint in
/// the actual browser before any WGSL door opens there — a shader that
/// fails validation in a browser dispatches as a silent no-op, so the
/// probe checks values, not absence of errors.
pub async fn probe_report() -> Result<(), String> {
    use cubecl::prelude::*;

    let device = Default::default();
    let client = WgpuRuntime::client(&device);

    // --- scalar layout ----------------------------------------------------
    let out = client.empty(5 * core::mem::size_of::<u32>());
    launch(
        &client,
        ProbeEcho,
        CubeCount::Static(1, 1, 1),
        vec![out.clone().binding()],
        vec![1, 0xAABB_CCDD_0011_2233, 7, f32::to_bits(1.5) as u64],
    );
    let bytes = client
        .read_async(vec![out])
        .await
        .map_err(|e| format!("echo readback failed: {e:?}"))?;
    let words = u32::from_bytes(&bytes[0]).to_vec();
    let expect = [1u32, 0, 0x0011_2233, 7, f32::to_bits(3.0)];
    if words[..5] != expect {
        return Err(format!(
            "scalar layout mismatch: got {words:?}, expected {expect:?} — \
             the uniform u64-slot contract does not hold on this backend"
        ));
    }

    // --- shared memory + barriers -----------------------------------------
    let n = 300usize;
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let in_h = client.create_from_slice(f32::as_bytes(&input));
    let out_h = client.empty(n * core::mem::size_of::<f32>());
    launch(
        &client,
        ProbeSmem,
        CubeCount::Static(2, 1, 1),
        vec![in_h.binding(), out_h.clone().binding()],
        vec![n as u64],
    );
    let bytes = client
        .read_async(vec![out_h])
        .await
        .map_err(|e| format!("smem readback failed: {e:?}"))?;
    let got = f32::from_bytes(&bytes[0]);
    for i in 0..n {
        let base = (i / 256) * 256;
        let mirrored = base + (255 - i % 256);
        let staged = if mirrored < n { mirrored as f32 } else { 0.0 };
        if got[i] != staged + 1.0 {
            return Err(format!(
                "smem mirror wrong at {i}: got {}, expected {} — workgroup \
                 memory or barriers are broken on this backend",
                got[i],
                staged + 1.0
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::prelude::*;

    /// The host->kernel scalar layout, proven rather than assumed: u64
    /// slots as vec2<u32>, low word first, f32 via bit round-trip.
    #[test]
    fn scalar_layout_is_what_every_kernel_assumes() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        let client = WgpuRuntime::client(&device);

        let out = client.empty(5 * core::mem::size_of::<u32>());
        launch(
            &client,
            ProbeEcho,
            CubeCount::Static(1, 1, 1),
            vec![out.clone().binding()],
            vec![
                1,
                0xAABB_CCDD_0011_2233,
                7,
                f32::to_bits(1.5) as u64,
            ],
        );
        let bytes = client.read_one_unchecked(out);
        let words = u32::from_bytes(&bytes);
        assert_eq!(words[0], 1, "slot 0 low word");
        assert_eq!(words[1], 0, "slot 0 high word");
        assert_eq!(words[2], 0x0011_2233, "slot 1 low word");
        assert_eq!(words[3], 7, "slot 2 low word");
        assert_eq!(
            f32::from_bits(words[4]),
            3.0,
            "f32 must survive the bits round-trip"
        );
    }

    /// Workgroup memory and barriers produce correct cross-thread values,
    /// including in a ragged final window. Wrong values here mean staged
    /// kernels cannot be trusted at all on this backend.
    #[test]
    fn shared_memory_mirror_survives_a_ragged_tail() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        let client = WgpuRuntime::client(&device);

        // 300 elements: one full workgroup and one ragged 44-lane tail.
        let n = 300usize;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let in_h = client.create_from_slice(f32::as_bytes(&input));
        let out_h = client.empty(n * core::mem::size_of::<f32>());

        launch(
            &client,
            ProbeSmem,
            CubeCount::Static(2, 1, 1),
            vec![in_h.binding(), out_h.clone().binding()],
            vec![n as u64],
        );

        let bytes = client.read_one_unchecked(out_h);
        let got = f32::from_bytes(&bytes);
        for i in 0..n {
            let base = (i / 256) * 256;
            let lane = i % 256;
            let mirrored = base + (255 - lane);
            let staged = if mirrored < n { mirrored as f32 } else { 0.0 };
            assert_eq!(
                got[i],
                staged + 1.0,
                "lane {i}: expected the mirror of its window"
            );
        }
    }
}
