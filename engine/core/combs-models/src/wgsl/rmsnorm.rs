//! K1: the fused RMSNorm dispatch — first WGSL kernel in the forward path.
//!
//! One workgroup per row over the flattened leading dimensions, so the
//! same kernel serves the hidden-state norms (rows = seq) and the
//! per-head qk-norms (rows = heads × seq). The gemma `(1 + w)` variant is
//! the same kernel with `flavor = 1.0` — the add happens in-register
//! instead of materializing a shifted weight tensor per call.
//!
//! Fallback discipline as everywhere: any refusal — doors, a foreign
//! backend, a row count past the dispatch limit — returns `None` and
//! the caller runs the burn reference unchanged. Both wgpu element
//! types are served: f16 activations cast around the f32 launch.

use std::any::{Any, TypeId};

use burn::backend::wgpu::{CubeBackend, CubeTensor, FloatElement, WgpuRuntime};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Shape, Tensor, TensorPrimitive};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl_fusion::CubeFusionHandle;
use burn_fusion::stream::{Operation, OperationStreams};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr, TensorStatus};
use cubecl::prelude::CubeCount;

use super::{RmsNorm, Storage, launch, wgsl_enabled};
use super::{as_f32, from_f32};
use crate::qlinear::{FusedF32, InnerF32, UnfusedF16, UnfusedF32};

/// `CubeCount::Static` caps each grid dimension at 65535; rows beyond
/// that (no real shape today) fall back to the burn path.
const MAX_ROWS: usize = 65535;

/// Per-kernel door on top of the master `COMBS_WGSL`; read once.
pub(super) fn rmsnorm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        wgsl_enabled() && !matches!(std::env::var("COMBS_WGSL_RMSNORM").as_deref(), Ok("0"))
    })
}

/// Runtime type-equality bridge, tensor edition: `From` and `To` are the
/// same type exactly when the backends match, and the downcast proves it.
pub(super) fn cast_any<From: 'static, To: 'static>(v: From) -> Option<To> {
    let any: Box<dyn Any> = Box::new(v);
    any.downcast::<To>().ok().map(|b| *b)
}

/// Tries the WGSL RMSNorm for `x` rows over the last dimension. `flavor`
/// is added to the weight in-kernel (0.0 plain, 1.0 gemma). `None` →
/// caller falls back to the burn reference.
pub(crate) fn try_rms_norm<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    weight: Tensor<B, 1>,
    eps: f64,
    flavor: f32,
) -> Option<Tensor<B, D>> {
    if !rmsnorm_enabled() {
        return None;
    }
    let dims = x.dims();
    let n = dims[D - 1];
    let rows: usize = dims[..D - 1].iter().product();
    if rows == 0 || n == 0 || rows > MAX_ROWS {
        return None;
    }
    if TypeId::of::<B>() == TypeId::of::<UnfusedF32>() {
        let x: Tensor<UnfusedF32, D> = cast_any(x)?;
        let w: Tensor<UnfusedF32, 1> = cast_any(weight)?;
        return cast_any(rms_norm_unfused(x, w, eps, flavor));
    }
    if TypeId::of::<B>() == TypeId::of::<FusedF32>() {
        let x: Tensor<FusedF32, D> = cast_any(x)?;
        let w: Tensor<FusedF32, 1> = cast_any(weight)?;
        return cast_any(rms_norm_fused(x, w, eps, flavor));
    }
    // The f16 backend the app ships takes the same launch with a cast on
    // either side. It used to be refused here, and the refusal cost it
    // the whole burn chain — about thirty dispatches a layer where four
    // will do (the ledger §90, §92).
    if TypeId::of::<B>() == TypeId::of::<UnfusedF16>() {
        let x: Tensor<UnfusedF16, D> = cast_any(x)?;
        let w: Tensor<UnfusedF16, 1> = cast_any(weight)?;
        return cast_any(rms_norm_unfused(x, w, eps, flavor));
    }
    None
}

/// Scalar slots in the order the kernel's `Params` declares them.
fn scalar_slots(rows: usize, n: usize, eps: f64, flavor: f32) -> Vec<u64> {
    vec![
        rows as u64,
        n as u64,
        f32::to_bits(eps as f32) as u64,
        f32::to_bits(flavor) as u64,
    ]
}

/// The launch itself, shared by both backends once primitives are in hand.
fn launch_rms_norm(
    x: CubeTensor<WgpuRuntime>,
    w: CubeTensor<WgpuRuntime>,
    shape: Shape,
    rows: usize,
    n: usize,
    eps: f64,
    flavor: f32,
    storage: Storage,
) -> CubeTensor<WgpuRuntime> {
    let x = into_contiguous(x);
    let w = into_contiguous(w);
    let client = x.client.clone();
    // The output buffer is sized for what the kernel will WRITE. An f16
    // rendering that wrote into an f32-sized buffer would be correct and
    // waste half of it; one that wrote f32 into an f16-sized buffer
    // would run off the end.
    let elem = match storage {
        Storage::F32 => core::mem::size_of::<f32>(),
        Storage::F16 => core::mem::size_of::<f32>() / 2,
    };
    let out = client.empty(rows * n * elem);
    launch(
        &client,
        RmsNorm(storage),
        CubeCount::Static(rows as u32, 1, 1),
        vec![
            x.handle.binding(),
            w.handle.binding(),
            out.clone().binding(),
        ],
        scalar_slots(rows, n, eps, flavor),
    );
    let dtype = match storage {
        Storage::F32 => DType::F32,
        Storage::F16 => DType::F16,
    };
    CubeTensor::new_contiguous(client, x.device.clone(), shape, out, dtype)
}

/// One body for every unfused wgpu element type, and two roads through
/// it.
///
/// The f32 road casts f16 activations up, launches, and casts the result
/// back — three small kernels around one, which is still far cheaper
/// than the multi-op chain it replaced (§92).
///
/// The f16 road, when the adapter admits the type and the door is open,
/// skips all three: the kernel reads and writes f16 IN PLACE and does
/// its arithmetic in f32 regardless, which is the law (§66). Half the
/// bytes cross the bus, and on a box where bytes cost more than launches
/// that is the whole point.
fn rms_norm_unfused<F: FloatElement, const D: usize>(
    x: Tensor<CubeBackend<WgpuRuntime, F, i32, u32>, D>,
    w: Tensor<CubeBackend<WgpuRuntime, F, i32, u32>, 1>,
    eps: f64,
    flavor: f32,
) -> Tensor<CubeBackend<WgpuRuntime, F, i32, u32>, D> {
    let dims = x.dims();
    let n = dims[D - 1];
    let rows: usize = dims[..D - 1].iter().product();
    let shape = Shape::from(dims.to_vec());
    let in_dtype = x.dtype();

    let xp = x.into_primitive().tensor();
    let wp = w.into_primitive().tensor();
    if in_dtype == DType::F16 && super::f16_storage_enabled(&xp.client) {
        let out = launch_rms_norm(xp, wp, shape, rows, n, eps, flavor, Storage::F16);
        return Tensor::from_primitive(TensorPrimitive::Float(out));
    }

    let xp = as_f32(Tensor::<CubeBackend<WgpuRuntime, F, i32, u32>, D>::from_primitive(
        TensorPrimitive::Float(xp),
    ))
    .into_primitive()
    .tensor();
    let wp = as_f32(Tensor::<CubeBackend<WgpuRuntime, F, i32, u32>, 1>::from_primitive(
        TensorPrimitive::Float(wp),
    ))
    .into_primitive()
    .tensor();
    let out = launch_rms_norm(xp, wp, shape, rows, n, eps, flavor, Storage::F32);
    from_f32(Tensor::from_primitive(TensorPrimitive::Float(out)), in_dtype)
}

/// The fusion-stream operation: executed when the stream drains, with
/// both inputs resolved to real device tensors.
struct RmsNormOp {
    desc: CustomOpIr,
    shape: Vec<usize>,
    eps: f64,
    flavor: f32,
}

impl core::fmt::Debug for RmsNormOp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "RmsNormOp {{ shape: {:?} }}", self.shape)
    }
}

impl Operation<FusionCubeRuntime<WgpuRuntime>> for RmsNormOp {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle<WgpuRuntime>>) {
        let ([x, w], [output]) = self.desc.as_fixed::<2, 1>();
        let xp: CubeTensor<WgpuRuntime> = handles.get_float_tensor::<InnerF32>(x);
        let wp: CubeTensor<WgpuRuntime> = handles.get_float_tensor::<InnerF32>(w);
        let n = *self.shape.last().expect("rank >= 1");
        let rows: usize = self.shape[..self.shape.len() - 1].iter().product();
        let out = launch_rms_norm(
            xp,
            wp,
            Shape::from(self.shape.clone()),
            rows,
            n,
            self.eps,
            self.flavor,
            // The fused road is the f32 backend's; f16 never reaches it,
            // since burn-fusion cannot carry f16 at all (§91).
            Storage::F32,
        );
        handles.register_float_tensor::<InnerF32>(&output.id, out);
    }
}

fn rms_norm_fused<const D: usize>(
    x: Tensor<FusedF32, D>,
    w: Tensor<FusedF32, 1>,
    eps: f64,
    flavor: f32,
) -> Tensor<FusedF32, D> {
    let dims = x.dims();
    let xp = x.into_primitive().tensor();
    let wp = w.into_primitive().tensor();
    let client = xp.client.clone();

    let mut streams = OperationStreams::default();
    streams.tensor(&xp);
    streams.tensor(&wp);
    let x_ir = xp.into_ir();
    let w_ir = wp.into_ir();
    let out_ir = TensorIr {
        id: client.create_empty_handle(),
        shape: Shape::from(dims.to_vec()),
        status: TensorStatus::NotInit,
        dtype: DType::F32,
    };
    let desc = CustomOpIr::new("combs_rms_norm", &[x_ir, w_ir], &[out_ir]);
    let op = RmsNormOp {
        desc: desc.clone(),
        shape: dims.to_vec(),
        eps,
        flavor,
    };
    let mut outputs = client.register(streams, OperationIr::Custom(desc), op);
    let out = outputs.pop().expect("custom op declares one output");
    Tensor::from_primitive(TensorPrimitive::Float(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use crate::qlinear::UnfusedF16;
    use burn_cubecl::template::KernelSource;

    type Ref = burn::backend::NdArray<f32>;

    /// Deterministic non-trivial data: no RNG in tests, and values large
    /// and mixed enough that a broken reduction cannot hide.
    fn signal(len: usize, salt: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.719 + salt).sin() * 3.0) + ((i % 7) as f32 - 3.0))
            .collect()
    }

    fn reference(xs: &[f32], ws: &[f32], rows: usize, n: usize, eps: f64, flavor: f32) -> Vec<f32> {
        let device = Default::default();
        let x: Tensor<Ref, 2> =
            Tensor::from_data(TensorData::new(xs.to_vec(), [rows, n]), &device);
        let mut w = ws.to_vec();
        for v in &mut w {
            *v += flavor;
        }
        let w: Tensor<Ref, 1> = Tensor::from_data(TensorData::new(w, [n]), &device);
        crate::norm::rms_norm(x, w, eps)
            .into_data()
            .to_vec()
            .unwrap()
    }

    fn assert_close(got: &[f32], expect: &[f32], rel: f32, what: &str) {
        assert_eq!(got.len(), expect.len(), "{what}: length");
        for (i, (g, e)) in got.iter().zip(expect).enumerate() {
            // Finite first: a NaN passes every |diff| <= tol comparison
            // backwards (all comparisons on NaN are false, but so is the
            // assert's), and an Inf-vs-Inf diff is NaN — name the index
            // before the tolerance can lie about it.
            assert!(
                g.is_finite() && e.is_finite(),
                "{what}[{i}]: non-finite (got {g}, expect {e})"
            );
            let tol = rel * e.abs().max(1.0);
            assert!((g - e).abs() <= tol, "{what}[{i}]: got {g}, expect {e}");
        }
    }

    /// Harmony vs the burn reference across every decode-relevant width,
    /// ragged included, on both wgpu backends and both flavors. Tiny
    /// buffers — safe in a shared test process.
    /// H3c's contract on the helper itself: a NaN names its index
    /// instead of sliding under the tolerance. CPU-only.
    #[test]
    #[should_panic(expected = "non-finite")]
    fn assert_close_names_the_first_non_finite_index() {
        assert_close(&[1.0, f32::NAN, 3.0], &[1.0, 2.0, 3.0], 1e-4, "probe");
    }

    /// eps = 1.0 conditions the mean-square (the rsqrt argument sits
    /// near 1 instead of near float dust), so backends should agree far
    /// tighter than the real-eps tolerance — this is the arm expected
    /// to hold unchanged on Vulkan when the pod runs (H3e).
    #[test]
    fn wgsl_rms_norm_conditioned_at_unit_eps_agrees_tightly() {
        if crate::skip_no_gpu() {
            return;
        }
        for &n in &[64usize, 896, 1024, 3072] {
            let rows = 3usize;
            let xs = signal(rows * n, 0.3);
            let ws = signal(n, 7.7);
            let expect = reference(&xs, &ws, rows, n, 1.0, 0.0);
            let device = Default::default();
            let x: Tensor<UnfusedF32, 2> =
                Tensor::from_data(TensorData::new(xs.clone(), [rows, n]), &device);
            let w: Tensor<UnfusedF32, 1> =
                Tensor::from_data(TensorData::new(ws.clone(), [n]), &device);
            let got = rms_norm_unfused(x, w, 1.0, 0.0)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_close(&got, &expect, 1e-6, &format!("conditioned n={n}"));
        }
    }

    #[test]
    fn wgsl_rms_norm_matches_the_reference_across_widths_and_flavors() {
        if crate::skip_no_gpu() {
            return;
        }
        let widths = [64usize, 96, 128, 256, 576, 896, 1000, 1024, 2048, 3072];
        for &n in &widths {
            let rows = 3usize;
            let xs = signal(rows * n, 0.3);
            let ws = signal(n, 7.7);
            for flavor in [0.0f32, 1.0] {
                let expect = reference(&xs, &ws, rows, n, 1e-6, flavor);

                let device = Default::default();
                let x: Tensor<UnfusedF32, 2> =
                    Tensor::from_data(TensorData::new(xs.clone(), [rows, n]), &device);
                let w: Tensor<UnfusedF32, 1> =
                    Tensor::from_data(TensorData::new(ws.clone(), [n]), &device);
                let got = rms_norm_unfused(x, w, 1e-6, flavor)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                assert_close(&got, &expect, 1e-4, &format!("unfused n={n} flavor={flavor}"));

                let x: Tensor<FusedF32, 2> =
                    Tensor::from_data(TensorData::new(xs.clone(), [rows, n]), &device);
                let w: Tensor<FusedF32, 1> =
                    Tensor::from_data(TensorData::new(ws.clone(), [n]), &device);
                let got = rms_norm_fused(x, w, 1e-6, flavor)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                assert_close(&got, &expect, 1e-4, &format!("fused n={n} flavor={flavor}"));
            }
        }
    }

    /// The f16-STORAGE kernel on this adapter: does it compute, and does
    /// it agree with the host?
    ///
    /// The fixture is ROUNDED TO f16 ON THE HOST FIRST and the
    /// expectation computed from the rounded values — otherwise operand
    /// rounding would be scored as kernel error and the tolerance would
    /// have to be widened to hide it, which is how a real mistake gets
    /// through. Arithmetic is f32 in both renderings, so what remains is
    /// one store rounding at 2^-11, and a real mistake lands near 1e0.
    ///
    /// Skips itself, by name, where the adapter does not admit f16.
    #[test]
    fn wgsl_rms_norm_f16_storage_matches_the_host_on_rounded_input() {
        if crate::skip_no_gpu() {
            return;
        }
        // The door is opt-in; this test is about the kernel, so it asks
        // for the rendering directly rather than through the door.
        let device = Default::default();
        let (rows, n) = (2usize, 256usize);
        let xs = signal(rows * n, 0.3);
        let ws = signal(n, 7.7);
        // Round the fixture the way the f16 bindings will.
        let xs: Vec<f32> = xs.iter().map(|v| half_round(*v)).collect();
        let ws: Vec<f32> = ws.iter().map(|v| half_round(*v)).collect();
        let expect = reference(&xs, &ws, rows, n, 1e-6, 0.0);

        // `from_data` keeps the dtype of the DATA, not the backend's
        // element type: handing it f32 values builds an F32 tensor even
        // on the f16 backend. The first version of this test did exactly
        // that and fed f32 bytes to an f16 kernel, which is how it came
        // to read 5.65625 where the host said 0.0259. Cast, and then
        // ASSERT the cast, so this can never quietly measure the wrong
        // type again.
        // The seam's own cast-down helper, rather than hoping `from_data`
        // infers the element type from the backend — it does not.
        let x: Tensor<UnfusedF16, 2> = from_f32(
            Tensor::from_data(TensorData::new(xs, [rows, n]), &device),
            DType::F16,
        );
        let w: Tensor<UnfusedF16, 1> = from_f32(
            Tensor::from_data(TensorData::new(ws, [n]), &device),
            DType::F16,
        );
        assert_eq!(x.dtype(), DType::F16, "the fixture must be f16 before an f16 kernel sees it");
        let xp = x.into_primitive().tensor();
        if !crate::wgsl::f16_storage_enabled_for_test(&xp.client) {
            eprintln!("skipped: this adapter does not admit f16 in a shader");
            return;
        }
        let wp = w.into_primitive().tensor();
        let out = launch_rms_norm(
            xp,
            wp,
            Shape::from(vec![rows, n]),
            rows,
            n,
            1e-6,
            0.0,
            Storage::F16,
        );
        let got = Tensor::<UnfusedF16, 2>::from_primitive(TensorPrimitive::Float(out))
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        // f32 arithmetic plus ONE f16 store rounding at 2^-11.
        assert_close(&got, &expect, 1e-3, "f16 storage n=256");
    }

    /// Round a value the way an f16 binding will, so a fixture measures
    /// the kernel rather than the format.
    fn half_round(v: f32) -> f32 {
        f32::from(burn::tensor::f16::from_f32(v))
    }

    /// The f32 rendering must come out EXACTLY as the audited shader
    /// did, or templating a kernel silently re-reviews it. The `enable`
    /// line must be absent — a browser rejects it and then dispatches
    /// the invalid pipeline as a silent no-op — and no placeholder may
    /// survive into the text handed to the compiler.
    #[test]
    fn the_f32_rendering_carries_no_enable_and_no_placeholder() {
        let text = RmsNorm(Storage::F32).source().complete();
        assert!(!text.contains("enable"), "an f32 shader must not enable anything");
        assert!(!text.contains("f16"), "no f16 may survive into the f32 rendering");
        assert!(!text.contains("{{"), "an unfilled placeholder reached the compiler");
        assert!(text.contains("array<f32>"), "the bindings must be f32");
        // The reduction and the scale are f32 in BOTH renderings; that is
        // the law (§66), not a property of this one.
        assert!(text.contains("var<workgroup> scratch: array<f32, 256>"));
    }

    /// And the f16 rendering must enable the type, bind it, and STILL
    /// reduce in f32 — accumulating in f16 diverges 3.4e-3 and overflows
    /// at a sum of 14,467.
    #[test]
    fn the_f16_rendering_enables_the_type_and_still_reduces_in_f32() {
        let text = RmsNorm(Storage::F16).source().complete();
        assert!(text.starts_with("enable f16;"), "the directive must be line one");
        assert!(text.contains("array<f16>"), "the bindings must be f16");
        assert!(!text.contains("{{"), "an unfilled placeholder reached the compiler");
        assert!(
            text.contains("var<workgroup> scratch: array<f32, 256>"),
            "the reduction stays f32 in every rendering"
        );
        assert!(
            text.contains("var sum = 0.0;"),
            "the accumulator stays f32 in every rendering"
        );
    }

    /// The two renderings must never share a pipeline-cache entry, or
    /// whichever compiled first would serve both — and the second model
    /// would read f16 bytes as f32.
    #[test]
    fn the_two_renderings_have_different_kernel_ids() {
        assert_ne!(
            format!("{:?}", RmsNorm(Storage::F32).id()),
            format!("{:?}", RmsNorm(Storage::F16).id())
        );
    }

    /// The f16 door: the backend the app ships must take this kernel,
    /// not the multi-op Burn fallback behind it. The `expect` is the
    /// proof the door opened; the values prove the casts around the
    /// launch are the right way round, since a mixed-dtype slip returns
    /// power-of-two-wrong numbers silently (see the KV cache's note).
    #[test]
    fn wgsl_rms_norm_matches_the_reference_on_the_f16_backend() {
        if crate::skip_no_gpu() {
            return;
        }
        let (rows, n) = (3usize, 896usize);
        let xs = signal(rows * n, 0.3);
        let ws = signal(n, 7.7);
        let expect = reference(&xs, &ws, rows, n, 1e-6, 0.0);

        let device = burn::backend::wgpu::WgpuDevice::default();
        let x: Tensor<UnfusedF16, 2> =
            Tensor::from_data(TensorData::new(xs, [rows, n]), (&device, burn::tensor::DType::F16));
        let w: Tensor<UnfusedF16, 1> = Tensor::from_data(TensorData::new(ws, [n]), (&device, burn::tensor::DType::F16));
        let got = try_rms_norm(x, w, 1e-6, 0.0)
            .expect("the f16 backend takes the kernel path")
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        // f16 storage on both sides of the launch: one rounding at 2^-11
        // on the way in and one on the way out, over f32 arithmetic.
        assert_close(&got, &expect, 2e-3, "f16 n=896");
    }

    /// The second guard, and it CANNOT live in the parallel suite: the
    /// dispatch counter is process-wide and monotone, so a sibling test
    /// launching a kernel between the two reads lands in this delta.
    /// Written in the suite first, it passed one gate by luck and failed
    /// the next — which is exactly the lie the battery law exists to
    /// forbid. It runs deliberately, alone:
    ///
    /// ```text
    /// cargo test --release -p combs-models --lib -- --ignored \
    ///   --test-threads=1 costs_one_kernel
    /// ```
    ///
    /// What it guards: a future change where the door still returns
    /// Some but the work quietly goes back to burn ops. Measured at
    /// eleven — three casts and the launch, plus the contiguous copy
    /// each cast carries — against roughly thirty for the chain this
    /// replaced.
    #[test]
    #[ignore = "gpu: reads the process-wide dispatch counter, so it needs its own process"]
    fn wgsl_rms_norm_costs_one_kernel_on_the_f16_backend() {
        if crate::skip_no_gpu() {
            return;
        }
        let (rows, n) = (3usize, 896usize);
        let xs = signal(rows * n, 0.3);
        let ws = signal(n, 7.7);
        let device = burn::backend::wgpu::WgpuDevice::default();
        let mk = || {
            let x: Tensor<UnfusedF16, 2> =
                Tensor::from_data(TensorData::new(xs.clone(), [rows, n]), (&device, burn::tensor::DType::F16));
            let w: Tensor<UnfusedF16, 1> =
                Tensor::from_data(TensorData::new(ws.clone(), [n]), (&device, burn::tensor::DType::F16));
            (x, w)
        };
        // Warm the pipeline: a cold launch compiles, and compilation is
        // not the token's cost.
        let (x, w) = mk();
        let _ = try_rms_norm(x, w, 1e-6, 0.0)
            .expect("the f16 backend takes the kernel path")
            .into_data();

        let (x, w) = mk();
        let (d0, _) = combs_core::gpu_counters();
        let _ = try_rms_norm(x, w, 1e-6, 0.0)
            .expect("the f16 backend takes the kernel path")
            .into_data();
        let (d1, _) = combs_core::gpu_counters();
        assert!(
            d1 - d0 <= 12,
            "f16 rms_norm cost {} dispatches — the kernel was not taken",
            d1 - d0
        );
    }

    /// The qk-norm shape: rank 4, rows = heads × seq flattened by the
    /// dispatcher, weight over the last dim only.
    #[test]
    fn wgsl_rms_norm_handles_the_per_head_rank() {
        if crate::skip_no_gpu() {
            return;
        }
        let (heads, seq, d) = (5usize, 7, 96);
        let xs = signal(heads * seq * d, 1.9);
        let ws = signal(d, 4.2);
        let expect = reference(&xs, &ws, heads * seq, d, 1e-6, 0.0);

        let device = Default::default();
        let x: Tensor<UnfusedF32, 4> =
            Tensor::from_data(TensorData::new(xs, [1, heads, seq, d]), &device);
        let w: Tensor<UnfusedF32, 1> = Tensor::from_data(TensorData::new(ws, [d]), &device);
        let got = try_rms_norm(x, w, 1e-6, 0.0)
            .expect("wgpu f32 backend takes the kernel path")
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_close(&got, &expect, 1e-4, "rank-4 qk-norm shape");
    }
}
