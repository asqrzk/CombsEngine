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
wgsl_kernel!(RmsNorm, "rmsnorm.wgsl");
wgsl_kernel!(DecodeAttn, "decode_attn.wgsl");

mod decode_attn;
mod rmsnorm;

pub(crate) use decode_attn::try_decode_attention;
pub(crate) use rmsnorm::try_rms_norm;

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

    // --- rmsnorm canary (K1) ----------------------------------------------
    // Two ragged rows through the production kernel, gemma flavor to
    // exercise every scalar slot, checked against a host reduction.
    let (rows, n) = (2usize, 300usize);
    let xs: Vec<f32> = (0..rows * n).map(|i| (i as f32 * 0.37).sin() * 2.0).collect();
    let ws: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32 * 0.11).cos()).collect();
    let (eps, flavor) = (1e-6f32, 1.0f32);
    let x_h = client.create_from_slice(f32::as_bytes(&xs));
    let w_h = client.create_from_slice(f32::as_bytes(&ws));
    let y_h = client.empty(rows * n * core::mem::size_of::<f32>());
    launch(
        &client,
        RmsNorm,
        CubeCount::Static(rows as u32, 1, 1),
        vec![x_h.binding(), w_h.binding(), y_h.clone().binding()],
        vec![
            rows as u64,
            n as u64,
            f32::to_bits(eps) as u64,
            f32::to_bits(flavor) as u64,
        ],
    );
    let bytes = client
        .read_async(vec![y_h])
        .await
        .map_err(|e| format!("rmsnorm readback failed: {e:?}"))?;
    let got = f32::from_bytes(&bytes[0]);
    for r in 0..rows {
        let row = &xs[r * n..(r + 1) * n];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for c in 0..n {
            let expect = row[c] * inv_rms * (ws[c] + flavor);
            let g = got[r * n + c];
            if (g - expect).abs() > 1e-4 * expect.abs().max(1.0) {
                return Err(format!(
                    "rmsnorm canary wrong at [{r},{c}]: got {g}, expected \
                     {expect} — the K1 kernel must stay doored off here"
                ));
            }
        }
    }

    // --- decode-attention canary (K3a) ------------------------------------
    // A small paged fixture with a shuffled table, GQA 2:1, ragged total,
    // checked against a host softmax — the same reference the harmony
    // matrix uses natively.
    let (n_q, n_kv, d, page_size, total) = (4usize, 2usize, 64usize, 16usize, 21usize);
    let pages = total.div_ceil(page_size);
    let num_pages = pages + 2;
    let table: Vec<i32> = (0..pages).map(|p| ((p * 3 + 1) % num_pages) as i32).collect();
    let val = |i: usize, salt: f32| ((i as f32 * 0.437 + salt).sin() * 1.5) as f32;
    let qv: Vec<f32> = (0..n_q * d).map(|i| val(i, 42.5)).collect();
    let mut ak = vec![0.0f32; num_pages * n_kv * page_size * d];
    let mut av = vec![0.0f32; num_pages * n_kv * page_size * d];
    let key = |g: usize, j: usize, c: usize| val(c + (g * total + j) * d, 0.71);
    let value = |g: usize, j: usize, c: usize| val(c + (g * total + j) * d, 90.0);
    for j in 0..total {
        let phys = table[j / page_size] as usize;
        for g in 0..n_kv {
            let base = ((phys * n_kv + g) * page_size + j % page_size) * d;
            for c in 0..d {
                ak[base + c] = key(g, j, c);
                av[base + c] = value(g, j, c);
            }
        }
    }
    let scale = 0.125f32;
    let q_h = client.create_from_slice(f32::as_bytes(&qv));
    let k_h = client.create_from_slice(f32::as_bytes(&ak));
    let v_h = client.create_from_slice(f32::as_bytes(&av));
    let t_h = client.create_from_slice(i32::as_bytes(&table));
    let o_h = client.empty(n_q * d * core::mem::size_of::<f32>());
    launch(
        &client,
        DecodeAttn,
        CubeCount::Static(n_q as u32, 1, 1),
        vec![
            q_h.binding(),
            k_h.binding(),
            v_h.binding(),
            t_h.binding(),
            o_h.clone().binding(),
        ],
        vec![
            n_q as u64,
            n_kv as u64,
            d as u64,
            page_size as u64,
            total as u64,
            0,
            0,
            f32::to_bits(scale) as u64,
        ],
    );
    let bytes = client
        .read_async(vec![o_h])
        .await
        .map_err(|e| format!("decode-attn readback failed: {e:?}"))?;
    let got = f32::from_bytes(&bytes[0]);
    for h in 0..n_q {
        let g = h / (n_q / n_kv);
        let qi = &qv[h * d..(h + 1) * d];
        let scores: Vec<f32> = (0..total)
            .map(|j| {
                (0..d).map(|c| qi[c] * key(g, j, c)).sum::<f32>() * scale
            })
            .collect();
        let m = scores.iter().cloned().fold(f32::MIN, f32::max);
        let ps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let sum: f32 = ps.iter().sum();
        for c in 0..d {
            let expect: f32 =
                (0..total).map(|j| ps[j] / sum * value(g, j, c)).sum();
            let gv = got[h * d + c];
            if (gv - expect).abs() > 1e-3 * expect.abs().max(1.0) {
                return Err(format!(
                    "decode-attn canary wrong at [{h},{c}]: got {gv}, \
                     expected {expect} — the K3 kernel must stay doored \
                     off here"
                ));
            }
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
