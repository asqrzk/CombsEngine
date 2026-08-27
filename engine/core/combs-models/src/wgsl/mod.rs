//! The Combs Kernel: the engine's owned GPU path.
//!
//! This module is the hand-written-WGSL half of the suite — launched on
//! the runtime's own stream through the template seam below; the
//! `#[cube]` quant family in `qmatmul.rs` is the other half. Together
//! they carry the whole decode hot path: packed embed gather, rms_norm,
//! rope, decode attention (paged / sliding / int8), and every quant
//! projection. Hand WGSL exists for the places where owning the exact
//! text matters — reductions and attention whose selection must be
//! fixed by shape, not autotuned, and whose source must be auditable
//! against the two validators it will meet (naga natively, Tint in a
//! browser).
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
wgsl_kernel!(DecodeAttnQ8, "decode_attn_q8.wgsl");
wgsl_kernel!(DecodeAttnSplit, "decode_attn_split.wgsl");
wgsl_kernel!(DecodeAttnCombine, "decode_attn_combine.wgsl");
wgsl_kernel!(RopeQk, "rope.wgsl");
wgsl_kernel!(DecodeGemvQ8, "decode_gemv_q8.wgsl");
wgsl_kernel!(DecodeGemvQ4K, "decode_gemv_q4k.wgsl");
wgsl_kernel!(DecodeGemvQ6K, "decode_gemv_q6k.wgsl");

/// Door for the split-K decode gemv (`COMBS_WGSL_GEMV=0` restores the
/// cube untiled kernel); read once, like every door here.
pub(crate) fn gemv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        wgsl_enabled() && !matches!(std::env::var("COMBS_WGSL_GEMV").as_deref(), Ok("0"))
    })
}

mod decode_attn;
mod decode_attn_q8;
mod rmsnorm;
mod rope;

pub(crate) use decode_attn::{try_decode_attention, try_sliding_decode_attention};
pub(crate) use decode_attn_q8::{QuantArena, try_decode_attention_q8};
pub(crate) use rmsnorm::try_rms_norm;
pub(crate) use rope::try_rope_qk;

/// Which split-K decode gemv kernel to launch — the qmatmul family
/// names its format, this seam owns the kernel objects.
pub(crate) enum GemvKernel {
    /// Q8_0 packed words + per-block scales.
    Q8,
    /// Q4_K superblocks (qs, [d, dmin] pairs, packed 6-bit scales).
    Q4K,
    /// Q6_K superblocks (ql, qh, i8 scales, d).
    Q6K,
}

/// [`launch`] for the decode gemv, callable from the qmatmul family
/// (which is hardwired to WgpuRuntime, same as this seam).
pub(crate) fn launch_gemv(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    kernel: GemvKernel,
    workgroups: u32,
    buffers: Vec<Binding>,
    scalars: Vec<u64>,
) {
    let count = CubeCount::Static(workgroups, 1, 1);
    match kernel {
        GemvKernel::Q8 => launch(client, DecodeGemvQ8, count, buffers, scalars),
        GemvKernel::Q4K => launch(client, DecodeGemvQ4K, count, buffers, scalars),
        GemvKernel::Q6K => launch(client, DecodeGemvQ6K, count, buffers, scalars),
    }
}

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

    // --- decode-attention canaries (K3a paged, K3b contiguous) ------------
    // Paged fixtures with a shuffled table, GQA 2:1, ragged totals,
    // checked against a host softmax — the same reference the harmony
    // matrix uses natively. d = 256 is gemma3's head dim and exercises
    // the full-workgroup column path; the contiguous case adds mode 1
    // with a window.
    // The totals straddle the 256-key tile boundary on purpose: the
    // multi-tile online-softmax correction is exactly the code a
    // single-tile canary would leave unproven, and gemma3's second turn
    // (context > 256) is where an unproven tile shows up as garbage.
    for (mode, d, window, n_kv, total) in [
        (0usize, 64usize, 0usize, 2usize, 21usize),
        (0, 256, 0, 2, 21),
        (1, 256, 5, 2, 21),
        (0, 256, 0, 2, 500),
        (1, 256, 512, 1, 500),
        (0, 128, 0, 1, 300),
    ] {
        decode_attn_canary(&client, mode, d, window, n_kv, total).await?;
    }

    // --- split-KV decode-attention canary (K8) ----------------------------
    // The two-pass deep path (segment partials + combine), value-checked
    // at a multi-segment total with a ragged last segment and d = 256.
    split_attn_canary(&client, 256, 1300).await?;
    split_attn_canary(&client, 128, 700).await?;

    // --- quantized decode-attention canaries (K3c) ------------------------
    // Grid-exact packed lanes (the i32::MAX corner word included) through
    // the q8 kernel's bitcast/shift unpack — constructs no other canary
    // exercises, value-checked against a host softmax at gemma3's d too.
    // Two checks per head dim: total=1 makes the softmax weight exactly
    // 1.0, so the output must be the dequantized V row to the last bit —
    // an unpack bug (sign extension, lane order, scale indexing) cannot
    // hide, and accumulation order cannot interfere. The long case then
    // bounds cross-compiler accumulation noise against an f64 reference
    // (Tint and naga contract fma differently; the quant grid's dynamic
    // range makes 1e-3-relative too tight for a 300-key online softmax).
    for d in [64usize, 256] {
        decode_attn_q8_canary(&client, d, 1).await?;
        decode_attn_q8_canary(&client, d, 300).await?;
    }

    // --- tiled quant-matmul canaries --------------------------------------
    // Prefill (m > 1) takes the shared-memory TILED kernels while decode
    // takes the untiled gemv — a browser model can therefore decode
    // perfectly while its prefill lies. Natively tiled and untiled are
    // bit-identical; hold Tint to the same bar, per format, and check
    // untiled against the CPU dequant reference so a both-wrong tie
    // cannot slip through.
    for format in ["q8_0", "q4_k", "q6_k"] {
        tiled_matmul_canary(&client, format).await?;
    }

    // --- split-K decode gemv canaries -------------------------------------
    // The same fixtures at m = 1 through the WGSL gemv vs the cube
    // untiled kernel: the decode path's dominant kernels, value-checked
    // per format on this compiler before the door means anything here.
    for format in ["q8_0", "q4_k", "q6_k"] {
        gemv_canary(&client, format).await?;
    }

    // --- compiled-tanh canary ---------------------------------------------
    // Goes through cubecl's WGSL compiler (not this module's hand-written
    // kernels) on purpose: it proves the safe-tanh workaround is active
    // on THIS target. Metal returns NaN for tanh past 43.0, and a build
    // that misses the clamp turns every GeluTanh model into token-0 soup.
    let xs = [-88.0f32, -50.0, -10.0, 0.5, 10.0, 43.5, 50.0, 88.0];
    let out_h = crate::qmatmul::tanh_canary_device(&client, &xs);
    let bytes = client
        .read_async(vec![out_h])
        .await
        .map_err(|e| format!("tanh readback failed: {e:?}"))?;
    let got = f32::from_bytes(&bytes[0]);
    for (i, (&x, &g)) in xs.iter().zip(got.iter()).enumerate() {
        let expect = x.tanh();
        if !g.is_finite() || (g - expect).abs() > 1e-6 {
            return Err(format!(
                "tanh canary wrong at [{i}] (x = {x}): got {g}, expected                  {expect} — the safe-tanh workaround is not active on this                  target and GeluTanh models will decode garbage"
            ));
        }
    }

    // --- rope canary (K2) -------------------------------------------------
    // Two heads of q, one of k, a mid-table position — both outputs
    // checked against the host rotate_half formula.
    let (n_q, n_kv, seq, d, pos, max_position) = (2usize, 1usize, 3usize, 64usize, 40usize, 64usize);
    let half = d / 2;
    let table = |t: usize, salt: f64| ((t as f64) * 0.031 + salt).sin() as f32;
    let cos_v: Vec<f32> = (0..max_position * d).map(|t| table(t, 0.0)).collect();
    let sin_v: Vec<f32> = (0..max_position * d).map(|t| table(t, 2.0)).collect();
    let xq: Vec<f32> = (0..n_q * seq * d).map(|i| table(i, 5.0)).collect();
    let xk: Vec<f32> = (0..n_kv * seq * d).map(|i| table(i, 7.0)).collect();
    let q_h = client.create_from_slice(f32::as_bytes(&xq));
    let k_h = client.create_from_slice(f32::as_bytes(&xk));
    let c_h = client.create_from_slice(f32::as_bytes(&cos_v));
    let s_h = client.create_from_slice(f32::as_bytes(&sin_v));
    let oq_h = client.empty(xq.len() * core::mem::size_of::<f32>());
    let ok_h = client.empty(xk.len() * core::mem::size_of::<f32>());
    let elems = (n_q + n_kv) * seq * d;
    launch(
        &client,
        RopeQk,
        CubeCount::Static(elems.div_ceil(WORKGROUP as usize) as u32, 1, 1),
        vec![
            q_h.binding(),
            k_h.binding(),
            c_h.binding(),
            s_h.binding(),
            oq_h.clone().binding(),
            ok_h.clone().binding(),
        ],
        vec![n_q as u64, n_kv as u64, seq as u64, d as u64, pos as u64],
    );
    let bytes = client
        .read_async(vec![oq_h, ok_h])
        .await
        .map_err(|e| format!("rope readback failed: {e:?}"))?;
    for (which, x, heads, got) in [
        ("q", &xq, n_q, f32::from_bytes(&bytes[0])),
        ("k", &xk, n_kv, f32::from_bytes(&bytes[1])),
    ] {
        for h in 0..heads {
            for r in 0..seq {
                for c in 0..d {
                    let i = (h * seq + r) * d + c;
                    let mate = if c < half { -x[i + half] } else { x[i - half] };
                    let t = (pos + r) * d + c;
                    let expect = x[i] * cos_v[t] + mate * sin_v[t];
                    if (got[i] - expect).abs() > 1e-5 * expect.abs().max(1.0) {
                        return Err(format!(
                            "rope canary wrong at {which}[{i}]: got {}, \
                             expected {expect} — the K2 kernel must stay \
                             doored off here",
                            got[i]
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// One split-KV decode-attention canary: the deep-context two-pass
/// pipeline (segment partials, then combine) against a host softmax,
/// with the same fixtures the single-pass canary uses.
async fn split_attn_canary(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    d: usize,
    total: usize,
) -> Result<(), String> {
    use cubecl::prelude::*;

    let (n_q, n_kv, page_size, seg_len) = (4usize, 2usize, 16usize, 512usize);
    let segs = total.div_ceil(seg_len);
    let pages = total.div_ceil(page_size);
    let num_pages = pages + 2;
    let table: Vec<i32> = (0..pages).map(|p| ((p + 2) % num_pages) as i32).collect();
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
    let scale = 1.0 / (d as f32).sqrt();
    let q_h = client.create_from_slice(f32::as_bytes(&qv));
    let k_h = client.create_from_slice(f32::as_bytes(&ak));
    let v_h = client.create_from_slice(f32::as_bytes(&av));
    let t_h = client.create_from_slice(i32::as_bytes(&table));
    let m_h = client.empty(n_q * segs * core::mem::size_of::<f32>());
    let s_h = client.empty(n_q * segs * core::mem::size_of::<f32>());
    let o_h = client.empty(n_q * segs * d * core::mem::size_of::<f32>());
    let out_h = client.empty(n_q * d * core::mem::size_of::<f32>());
    launch(
        client,
        DecodeAttnSplit,
        CubeCount::Static(n_q as u32, segs as u32, 1),
        vec![
            q_h.binding(),
            k_h.binding(),
            v_h.binding(),
            t_h.binding(),
            m_h.clone().binding(),
            s_h.clone().binding(),
            o_h.clone().binding(),
        ],
        vec![
            n_q as u64,
            n_kv as u64,
            d as u64,
            page_size as u64,
            total as u64,
            seg_len as u64,
            segs as u64,
            f32::to_bits(scale) as u64,
        ],
    );
    launch(
        client,
        DecodeAttnCombine,
        CubeCount::Static(n_q as u32, 1, 1),
        vec![
            m_h.binding(),
            s_h.binding(),
            o_h.binding(),
            out_h.clone().binding(),
        ],
        vec![n_q as u64, d as u64, segs as u64],
    );
    let bytes = client
        .read_async(vec![out_h])
        .await
        .map_err(|e| format!("split-attn readback failed: {e:?}"))?;
    let got = f32::from_bytes(&bytes[0]);
    for h in 0..n_q {
        let g = h / (n_q / n_kv);
        let qi = &qv[h * d..(h + 1) * d];
        let sc: Vec<f64> = (0..total)
            .map(|j| {
                (0..d).map(|c| qi[c] as f64 * key(g, j, c) as f64).sum::<f64>()
                    * scale as f64
            })
            .collect();
        let m = sc.iter().cloned().fold(f64::MIN, f64::max);
        let ps: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
        let sum: f64 = ps.iter().sum();
        for c in 0..d {
            let expect: f64 = (0..total)
                .map(|j| ps[j] / sum * value(g, j, c) as f64)
                .sum();
            let gv = got[h * d + c] as f64;
            if !gv.is_finite() || (gv - expect).abs() > 5e-3 * expect.abs().max(1.0) {
                return Err(format!(
                    "split-attn canary (d {d}, total {total}) wrong at                      [{h},{c}]: got {gv}, expected {expect} — the K8 path                      must stay doored off here"
                ));
            }
        }
    }
    Ok(())
}

/// One split-K gemv canary: the tiled canary's synthetic weights at
/// m = 1 through the WGSL kernel, checked against the CPU dequant
/// reference (the cube kernel is checked by the tiled canary already).
async fn gemv_canary(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    format: &str,
) -> Result<(), String> {
    use cubecl::prelude::*;

    let (n_out, k) = (211usize, 1024usize);
    let synth = |block_bytes: usize, scale_offsets: &[usize]| -> Vec<u8> {
        let blocks = match format {
            "q8_0" => n_out * k / 32,
            _ => n_out * k / 256,
        };
        let mut data: Vec<u8> = (0..blocks * block_bytes)
            .map(|i| ((i * 37 + 11) % 251) as u8)
            .collect();
        for b in 0..blocks {
            for (which, &off) in scale_offsets.iter().enumerate() {
                let bits: u16 = if which == 0 { 0x3C00 } else { 0x3400 };
                data[b * block_bytes + off..b * block_bytes + off + 2]
                    .copy_from_slice(&bits.to_le_bytes());
            }
        }
        data
    };
    let x: Vec<f32> = (0..k)
        .map(|i| ((i as f32 * 0.311).sin() * 1.4) - 0.1)
        .collect();
    let x_h = client.create_from_slice(f32::as_bytes(&x));
    let (dense, out_h) = match format {
        "q8_0" => {
            let data = synth(34, &[0]);
            let w = crate::qmatmul::Q80Weight::<WgpuRuntime>::from_gguf_bytes(
                client, &data, n_out, k,
            )
            .map_err(|e| format!("q8_0 pack failed: {e}"))?;
            let dense = combs_formats::quants::dequantize_q8_0(&data, n_out * k)
                .map_err(|e| format!("q8_0 reference failed: {e}"))?;
            (dense, w.decode_gemv_wgsl(client, x_h))
        }
        "q4_k" => {
            let data = synth(144, &[0, 2]);
            let w = crate::qmatmul::Q4KWeight::<WgpuRuntime>::from_gguf_bytes(
                client, &data, n_out, k,
            )
            .map_err(|e| format!("q4_k pack failed: {e}"))?;
            let dense = combs_formats::quants::dequantize_q4_k(&data, n_out * k)
                .map_err(|e| format!("q4_k reference failed: {e}"))?;
            (dense, w.decode_gemv_wgsl(client, x_h))
        }
        _ => {
            let data = synth(210, &[208]);
            let w = crate::qmatmul::Q6KWeight::<WgpuRuntime>::from_gguf_bytes(
                client, &data, n_out, k,
            )
            .map_err(|e| format!("q6_k pack failed: {e}"))?;
            let dense = combs_formats::quants::dequantize_q6_k(&data, n_out * k)
                .map_err(|e| format!("q6_k reference failed: {e}"))?;
            (dense, w.decode_gemv_wgsl(client, x_h))
        }
    };
    let bytes = client
        .read_async(vec![out_h])
        .await
        .map_err(|e| format!("{format} gemv canary readback failed: {e:?}"))?;
    let got = f32::from_bytes(&bytes[0]);
    for r in 0..n_out {
        let expect: f32 = (0..k).map(|c| x[c] * dense[r * k + c]).sum();
        let g = got[r];
        if !g.is_finite() || (g - expect).abs() > 1e-3 * expect.abs().max(1.0) {
            return Err(format!(
                "{format} gemv canary wrong at row {r}: got {g}, expected                  {expect} — the split-K gemv must stay doored off here"
            ));
        }
    }
    Ok(())
}

/// One quantized decode-attention canary: chosen int8 lanes packed
/// exactly like `kv_quantize` (lanes 0..2 offset-binary, lane 3 signed
/// in the top byte — including the `i32::MAX` corner), power-of-two
/// scales so the host dequant is exact, a shuffled page table, and a
/// host softmax over the dequantized rows as the reference.
async fn decode_attn_q8_canary(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    d: usize,
    total: usize,
) -> Result<(), String> {
    use cubecl::prelude::*;

    let (n_q, n_kv, page_size) = (4usize, 2usize, 16usize);
    let (dw, dg) = (d / 4, d / 32);
    let pages = total.div_ceil(page_size);
    let num_pages = pages + 2;
    let table: Vec<i32> = (0..pages).map(|p| ((p + 1) % num_pages) as i32).collect();
    let lanes = [-127i32, -1, 0, 1, 63, 127, -64, 5];
    let scales = [0.5f32, 0.25, 1.0, 0.125];
    let pack = |q0: i32, q1: i32, q2: i32, q3: i32| -> i32 {
        (q0 + 128) | ((q1 + 128) << 8) | ((q2 + 128) << 16) | (q3 << 24)
    };
    let mut build = |salt: usize| {
        let mut packed = vec![0i32; num_pages * n_kv * page_size * dw];
        let mut scal = vec![0.0f32; num_pages * n_kv * page_size * dg];
        let mut dense = vec![vec![0.0f32; d]; n_kv * total];
        for j in 0..total {
            let phys = table[j / page_size] as usize;
            for g in 0..n_kv {
                let row = (phys * n_kv + g) * page_size + j % page_size;
                for grp in 0..dg {
                    let sc = scales[(j + g + grp + salt) % scales.len()];
                    scal[row * dg + grp] = sc;
                    for w in 0..8 {
                        let word = if j == 0 && g == 0 && grp == 0 && w == 0 {
                            // The exact-i32::MAX corner: all-255 low
                            // bytes, +127 top lane.
                            pack(127, 127, 127, 127)
                        } else {
                            let pick = |l: usize| {
                                lanes[(j * 31 + g * 7 + grp * 5 + w * 3 + l + salt)
                                    % lanes.len()]
                            };
                            pack(pick(0), pick(1), pick(2), pick(3))
                        };
                        packed[row * dw + grp * 8 + w] = word;
                        let base = grp * 32 + w * 4;
                        let out = &mut dense[g * total + j];
                        out[base] = ((word & 0xff) - 128) as f32 * sc;
                        out[base + 1] = (((word >> 8) & 0xff) - 128) as f32 * sc;
                        out[base + 2] = (((word >> 16) & 0xff) - 128) as f32 * sc;
                        out[base + 3] = (word >> 24) as f32 * sc;
                    }
                }
            }
        }
        (packed, scal, dense)
    };
    let (kp, ks, kd) = build(1);
    let (vp, vs, vd) = build(9);
    let qv: Vec<f32> = (0..n_q * d)
        .map(|i| ((i as f32 * 0.437 + 4.2).sin() * 1.5))
        .collect();
    let scale = 1.0 / (d as f32).sqrt();

    let q_h = client.create_from_slice(f32::as_bytes(&qv));
    let kp_h = client.create_from_slice(i32::as_bytes(&kp));
    let ks_h = client.create_from_slice(f32::as_bytes(&ks));
    let vp_h = client.create_from_slice(i32::as_bytes(&vp));
    let vs_h = client.create_from_slice(f32::as_bytes(&vs));
    let t_h = client.create_from_slice(i32::as_bytes(&table));
    let o_h = client.empty(n_q * d * core::mem::size_of::<f32>());
    launch(
        client,
        DecodeAttnQ8,
        CubeCount::Static(n_q as u32, 1, 1),
        vec![
            q_h.binding(),
            kp_h.binding(),
            ks_h.binding(),
            vp_h.binding(),
            vs_h.binding(),
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
        .map_err(|e| format!("q8 decode-attn readback failed: {e:?}"))?;
    let got = f32::from_bytes(&bytes[0]);
    for h in 0..n_q {
        let g = h / (n_q / n_kv);
        if total == 1 {
            // Single visible key: weight is exactly 1.0 and the output is
            // the dequantized V row, bit for bit.
            for c in 0..d {
                let expect = vd[g][c];
                let gv = got[h * d + c];
                if gv.to_bits() != expect.to_bits() {
                    return Err(format!(
                        "q8 decode-attn canary (d {d}) unpack wrong at [{h},{c}]: \
                         got {gv}, expected exactly {expect} — the K3c kernel \
                         must stay doored off here"
                    ));
                }
            }
            continue;
        }
        let qi = &qv[h * d..(h + 1) * d];
        let sc: Vec<f64> = (0..total)
            .map(|j| {
                qi.iter()
                    .zip(&kd[g * total + j])
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>()
                    * scale as f64
            })
            .collect();
        let m = sc.iter().cloned().fold(f64::MIN, f64::max);
        let ps: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
        let sum: f64 = ps.iter().sum();
        for c in 0..d {
            let expect: f64 = (0..total)
                .map(|j| ps[j] / sum * vd[g * total + j][c] as f64)
                .sum();
            let gv = got[h * d + c] as f64;
            if !gv.is_finite() || (gv - expect).abs() > 5e-3 * expect.abs().max(1.0) {
                return Err(format!(
                    "q8 decode-attn canary (d {d}) wrong at [{h},{c}]: got \
                     {gv}, expected {expect} — the K3c kernel must stay \
                     doored off here"
                ));
            }
        }
    }
    Ok(())
}

/// One tiled-vs-untiled quant matmul canary: synthetic packed weights,
/// a prefill-shaped input (m = 32), both kernel variants read back and
/// the untiled one checked against the CPU dequant reference.
async fn tiled_matmul_canary(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    format: &str,
) -> Result<(), String> {
    use cubecl::prelude::*;

    let (n_out, k, m) = (48usize, 256usize, 32usize);
    // Deterministic packed bytes with sane f16 scales patched in at each
    // block's scale offsets (0x3C00 = 1.0, 0x3400 = 0.25).
    let synth = |block_bytes: usize, scale_offsets: &[usize]| -> Vec<u8> {
        let blocks = match format {
            "q8_0" => n_out * k / 32,
            _ => n_out * k / 256,
        };
        let mut data: Vec<u8> = (0..blocks * block_bytes)
            .map(|i| ((i * 37 + 11) % 251) as u8)
            .collect();
        for b in 0..blocks {
            for (which, &off) in scale_offsets.iter().enumerate() {
                let bits: u16 = if which == 0 { 0x3C00 } else { 0x3400 };
                data[b * block_bytes + off..b * block_bytes + off + 2]
                    .copy_from_slice(&bits.to_le_bytes());
            }
        }
        data
    };
    let (x_host, dense, tiled_h, untiled_h) = match format {
        "q8_0" => {
            let data = synth(34, &[0]);
            let w = crate::qmatmul::Q80Weight::<WgpuRuntime>::from_gguf_bytes(
                client, &data, n_out, k,
            )
            .map_err(|e| format!("q8_0 pack failed: {e}"))?;
            let dense = combs_formats::quants::dequantize_q8_0(&data, n_out * k)
                .map_err(|e| format!("q8_0 reference failed: {e}"))?;
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i as f32 * 0.377).sin() * 1.3) - 0.2)
                .collect();
            let x_h = client.create_from_slice(f32::as_bytes(&x));
            let t = w.matmul_device_with(client, x_h.clone(), m, true);
            let u = w.matmul_device_with(client, x_h, m, false);
            (x, dense, t, u)
        }
        "q4_k" => {
            let data = synth(144, &[0, 2]);
            let w = crate::qmatmul::Q4KWeight::<WgpuRuntime>::from_gguf_bytes(
                client, &data, n_out, k,
            )
            .map_err(|e| format!("q4_k pack failed: {e}"))?;
            let dense = combs_formats::quants::dequantize_q4_k(&data, n_out * k)
                .map_err(|e| format!("q4_k reference failed: {e}"))?;
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i as f32 * 0.377).sin() * 1.3) - 0.2)
                .collect();
            let x_h = client.create_from_slice(f32::as_bytes(&x));
            let t = w.matmul_device_with(client, x_h.clone(), m, true);
            let u = w.matmul_device_with(client, x_h, m, false);
            (x, dense, t, u)
        }
        _ => {
            // Q6_K has no tiled variant — prefill and decode share the
            // one kernel, so the reference check alone covers it.
            let data = synth(210, &[208]);
            let w = crate::qmatmul::Q6KWeight::<WgpuRuntime>::from_gguf_bytes(
                client, &data, n_out, k,
            )
            .map_err(|e| format!("q6_k pack failed: {e}"))?;
            let dense = combs_formats::quants::dequantize_q6_k(&data, n_out * k)
                .map_err(|e| format!("q6_k reference failed: {e}"))?;
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i as f32 * 0.377).sin() * 1.3) - 0.2)
                .collect();
            let x_h = client.create_from_slice(f32::as_bytes(&x));
            let t = w.matmul_device(client, x_h.clone(), m);
            let u = w.matmul_device(client, x_h, m);
            (x, dense, t, u)
        }
    };
    let bytes = client
        .read_async(vec![tiled_h, untiled_h])
        .await
        .map_err(|e| format!("{format} tiled canary readback failed: {e:?}"))?;
    let tiled = f32::from_bytes(&bytes[0]);
    let untiled = f32::from_bytes(&bytes[1]);
    for i in 0..m * n_out {
        if tiled[i].to_bits() != untiled[i].to_bits() {
            return Err(format!(
                "{format} tiled canary: tiled[{i}] = {} differs from                  untiled {} — the tiled prefill kernel computes different                  values on this target",
                tiled[i], untiled[i]
            ));
        }
    }
    for r in 0..m {
        for c in 0..n_out {
            let expect: f32 = (0..k)
                .map(|j| x_host[r * k + j] * dense[c * k + j])
                .sum();
            let g = untiled[r * n_out + c];
            if !g.is_finite() || (g - expect).abs() > 1e-3 * expect.abs().max(1.0) {
                return Err(format!(
                    "{format} tiled canary: untiled[{r},{c}] = {g}, host                      reference {expect} — the quant matmul itself is wrong                      on this target"
                ));
            }
        }
    }
    Ok(())
}

/// One decode-attention canary case: build the fixture, launch, check
/// every output element against a host softmax.
async fn decode_attn_canary(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    mode: usize,
    d: usize,
    window: usize,
    n_kv: usize,
    total: usize,
) -> Result<(), String> {
    use cubecl::prelude::*;

    let (n_q, page_size) = (4usize, 16usize);
    let pages = total.div_ceil(page_size);
    let num_pages = pages + 2;
    // A rotation is collision-free for every page count — the stride-3
    // shuffle this used to be collides whenever gcd(3, num_pages) > 1.
    let table: Vec<i32> = (0..pages).map(|p| ((p + 2) % num_pages) as i32).collect();
    let val = |i: usize, salt: f32| ((i as f32 * 0.437 + salt).sin() * 1.5) as f32;
    let qv: Vec<f32> = (0..n_q * d).map(|i| val(i, 42.5)).collect();
    let mut ak = vec![0.0f32; num_pages * n_kv * page_size * d];
    let mut av = vec![0.0f32; num_pages * n_kv * page_size * d];
    let key = |g: usize, j: usize, c: usize| val(c + (g * total + j) * d, 0.71);
    let value = |g: usize, j: usize, c: usize| val(c + (g * total + j) * d, 90.0);
    if mode == 0 {
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
    } else {
        ak = (0..n_kv * total * d)
            .map(|i| key(i / (total * d), (i / d) % total, i % d))
            .collect();
        av = (0..n_kv * total * d)
            .map(|i| value(i / (total * d), (i / d) % total, i % d))
            .collect();
    }
    let scale = 0.125f32;
    let q_h = client.create_from_slice(f32::as_bytes(&qv));
    let k_h = client.create_from_slice(f32::as_bytes(&ak));
    let v_h = client.create_from_slice(f32::as_bytes(&av));
    let t_h = client.create_from_slice(i32::as_bytes(&table));
    let o_h = client.empty(n_q * d * core::mem::size_of::<f32>());
    launch(
        client,
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
            window as u64,
            mode as u64,
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
        let visible: Vec<usize> = (0..total)
            .filter(|&j| window == 0 || j + window >= total)
            .collect();
        let scores: Vec<f32> = visible
            .iter()
            .map(|&j| {
                (0..d).map(|c| qi[c] * key(g, j, c)).sum::<f32>() * scale
            })
            .collect();
        let m = scores.iter().cloned().fold(f32::MIN, f32::max);
        let ps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let sum: f32 = ps.iter().sum();
        for c in 0..d {
            let expect: f32 = visible
                .iter()
                .enumerate()
                .map(|(idx, &j)| ps[idx] / sum * value(g, j, c))
                .sum();
            let gv = got[h * d + c];
            if (gv - expect).abs() > 1e-3 * expect.abs().max(1.0) {
                return Err(format!(
                    "decode-attn canary (mode {mode}, d {d}, window \
                     {window}) wrong at [{h},{c}]: got {gv}, expected \
                     {expect} — the K3 kernel must stay doored off here"
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

    /// The exact probe the browser runs, on the native compiler — when
    /// the Chrome probe fails a canary that passes here, the divergence
    /// is Tint's, not the fixture's.
    #[test]
    fn probe_report_passes_natively() {
        if crate::skip_no_gpu() {
            return;
        }
        cubecl::future::block_on(probe_report()).expect("all canaries green on naga");
    }

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
