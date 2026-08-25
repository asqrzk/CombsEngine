//! K1: the fused RMSNorm dispatch — first WGSL kernel in the forward path.
//!
//! One workgroup per row over the flattened leading dimensions, so the
//! same kernel serves the hidden-state norms (rows = seq) and the
//! per-head qk-norms (rows = heads × seq). The gemma `(1 + w)` variant is
//! the same kernel with `flavor = 1.0` — the add happens in-register
//! instead of materializing a shifted weight tensor per call.
//!
//! Fallback discipline as everywhere: any refusal — doors, foreign
//! backend, f16 arenas, a row count past the dispatch limit — returns
//! `None` and the caller runs the burn reference unchanged.

use std::any::{Any, TypeId};

use burn::backend::wgpu::{CubeTensor, WgpuRuntime};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Shape, Tensor, TensorPrimitive};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl_fusion::CubeFusionHandle;
use burn_fusion::stream::{Operation, OperationStreams};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr, TensorStatus};
use cubecl::prelude::CubeCount;

use super::{RmsNorm, launch, wgsl_enabled};
use crate::qlinear::{FusedF32, InnerF32, UnfusedF32};

/// `CubeCount::Static` caps each grid dimension at 65535; rows beyond
/// that (no real shape today) fall back to the burn path.
const MAX_ROWS: usize = 65535;

/// Per-kernel door on top of the master `COMBS_WGSL`; read once.
fn rmsnorm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        wgsl_enabled() && !matches!(std::env::var("COMBS_WGSL_RMSNORM").as_deref(), Ok("0"))
    })
}

/// Runtime type-equality bridge, tensor edition: `From` and `To` are the
/// same type exactly when the backends match, and the downcast proves it.
fn cast_any<From: 'static, To: 'static>(v: From) -> Option<To> {
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
    // f16 builds keep the burn path: its f32 reduction + dtype casts are
    // the numerics the f16 transcripts were gated on.
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
) -> CubeTensor<WgpuRuntime> {
    let x = into_contiguous(x);
    let w = into_contiguous(w);
    let client = x.client.clone();
    let out = client.empty(rows * n * core::mem::size_of::<f32>());
    launch(
        &client,
        RmsNorm,
        CubeCount::Static(rows as u32, 1, 1),
        vec![
            x.handle.binding(),
            w.handle.binding(),
            out.clone().binding(),
        ],
        scalar_slots(rows, n, eps, flavor),
    );
    CubeTensor::new_contiguous(client, x.device.clone(), shape, out, DType::F32)
}

fn rms_norm_unfused<const D: usize>(
    x: Tensor<UnfusedF32, D>,
    w: Tensor<UnfusedF32, 1>,
    eps: f64,
    flavor: f32,
) -> Tensor<UnfusedF32, D> {
    let dims = x.dims();
    let n = dims[D - 1];
    let rows: usize = dims[..D - 1].iter().product();
    let xp = x.into_primitive().tensor();
    let wp = w.into_primitive().tensor();
    let out = launch_rms_norm(xp, wp, Shape::from(dims.to_vec()), rows, n, eps, flavor);
    Tensor::from_primitive(TensorPrimitive::Float(out))
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
            let tol = rel * e.abs().max(1.0);
            assert!((g - e).abs() <= tol, "{what}[{i}]: got {g}, expect {e}");
        }
    }

    /// Harmony vs the burn reference across every decode-relevant width,
    /// ragged included, on both wgpu backends and both flavors. Tiny
    /// buffers — safe in a shared test process.
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
