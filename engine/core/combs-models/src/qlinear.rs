//! The quantized linear seam: per-tensor
//! kernel dispatch behind a stable `Linear` type, so model code calls
//! `layer.q.forward(x, bias)` and never knows whether the weight is a dense
//! burn tensor or packed GGUF blocks fed to our fused CubeCL kernels.
//!
//! Dispatch happens **once, at load time**: [`try_quant_linear`] returns a
//! backend-specific op only when (a) the source stores the tensor packed in
//! a supported GGUF format (Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K), and (b) the
//! backend runs on the wgpu runtime the kernels target. Every other
//! combination falls back to the portable dense
//! path — HF-kernels principle #2 (kernels are accelerators, never
//! load-bearing for correctness).
//!
//! Three backends get the fast path:
//! - `Fusion<CubeBackend<WgpuRuntime, f32, …>>` — the default build. The
//!   matmul enters the fusion stream as a custom operation (burn's
//!   sanctioned escape hatch), so ops before/after it still fuse and the
//!   packed weight is read directly by our kernel at execution time.
//! - `CubeBackend<WgpuRuntime, f32, …>` — unfused f32; direct launch.
//! - `CubeBackend<WgpuRuntime, f16, …>` — the `--features f16` build; the
//!   activation is cast f16→f32 around the kernel (weights dominate memory,
//!   activations are negligible).
//!
//! Backend selection uses `Any` downcasts keyed on the backend type — safe,
//! no `unsafe`, and models stay generic over `B: Backend`.

use std::any::{Any, TypeId};
use std::sync::Arc;

use burn::backend::wgpu::{CubeBackend, CubeTensor, WgpuDevice, WgpuRuntime};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Device, FloatDType, Shape, Tensor, TensorPrimitive};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::kernel::matmul::{matmul, MatmulStrategy};
use burn_cubecl::ops::permute;
use burn_cubecl_fusion::CubeFusionHandle;
use burn_fusion::Fusion;
use burn_fusion::stream::{Operation, OperationStreams};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr, TensorStatus};
use combs_formats::ModelSource;

use crate::llama::linear as dense_linear;
use crate::qmatmul::QuantWeight;
use crate::{ModelError, Result};

/// The default engine backend (fused f32 wgpu).
pub(crate) type FusedF32 = Fusion<CubeBackend<WgpuRuntime, f32, i32, u32>>;
/// Unfused f32 wgpu (used when the fusion feature is off).
pub(crate) type UnfusedF32 = CubeBackend<WgpuRuntime, f32, i32, u32>;
/// The `--features f16` backend.
pub(crate) type UnfusedF16 = CubeBackend<WgpuRuntime, burn::tensor::f16, i32, u32>;
/// The inner (non-fusion) backend the custom op executes on.
pub(crate) type InnerF32 = CubeBackend<WgpuRuntime, f32, i32, u32>;

/// A backend-specific quantized-linear forward. Boxed into [`Linear::Quant`]
/// at load time by [`try_quant_linear`].
pub trait QuantLinearOp<B: Backend>: Send + Sync {
    /// `y = x @ W^T` for `x: [batch, seq, k]` → `[batch, seq, n_out]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3>;
    /// `[n_out, k]`, matching a dense weight's `dims()`.
    fn dims(&self) -> [usize; 2];
    /// Bytes the packed weight occupies in VRAM.
    fn vram_bytes(&self) -> usize;
}

/// A linear layer weight: dense tensor (portable path) or packed quant
/// blocks bound to a fused device kernel.
pub enum Linear<B: Backend> {
    /// Portable path: `[n_out, k]` dense tensor, burn matmul.
    Dense(Tensor<B, 2>),
    /// Fast path: packed weight + backend-specific kernel dispatch.
    Quant(Box<dyn QuantLinearOp<B>>),
}

impl<B: Backend> Linear<B> {
    /// `[n_out, k]`.
    pub fn dims(&self) -> [usize; 2] {
        match self {
            Linear::Dense(w) => w.dims(),
            Linear::Quant(op) => op.dims(),
        }
    }

    /// `y = x @ W^T (+ b)` for `x: [batch, seq, k]`.
    pub fn forward(&self, x: Tensor<B, 3>, bias: Option<&Tensor<B, 1>>) -> Tensor<B, 3> {
        match self {
            Linear::Dense(w) => dense_linear(x, w, bias),
            Linear::Quant(op) => {
                let out = op.forward(x);
                match bias {
                    Some(b) => {
                        let [batch, seq, dim] = out.dims();
                        out + b.clone().reshape([1, 1, dim]).expand([batch, seq, dim])
                    }
                    None => out,
                }
            }
        }
    }
}

/// The one concrete op implementation: a [`QuantWeight`] on the wgpu
/// runtime. Implements [`QuantLinearOp`] for each supported backend.
struct CubeQuantLinear {
    w: Arc<QuantWeight>,
}

impl CubeQuantLinear {
    fn dims(&self) -> [usize; 2] {
        [self.w.n_out(), self.w.k()]
    }

    /// Shared unfused path: contiguous f32 `CubeTensor` in, f32 out.
    fn forward_cube(&self, x: CubeTensor<WgpuRuntime>, batch: usize, seq: usize) -> CubeTensor<WgpuRuntime> {
        let x = into_contiguous(x);
        if let Some(out) = try_batched_matmul(&self.w, &x, batch, seq) {
            return out;
        }
        let out_h = self.w.matmul_device(&x.client, x.handle.clone(), batch * seq);
        CubeTensor::new_contiguous(
            x.client.clone(),
            x.device.clone(),
            Shape::from([batch, seq, self.w.n_out()]),
            out_h,
            DType::F32,
        )
    }
}

/// The kernels read/write f32; tensors of other float dtypes are cast
/// around the launch and the output follows the input dtype. (burn 0.21
/// resolves a tensor's dtype from per-device default settings, so even an
/// f32 backend can hand us f16 tensors.)
fn to_f32<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    match x.dtype() {
        DType::F32 => x,
        _ => x.cast(FloatDType::F32),
    }
}

fn to_dtype<B: Backend>(out: Tensor<B, 3>, dtype: DType) -> Tensor<B, 3> {
    match dtype {
        DType::F16 => out.cast(FloatDType::F16),
        DType::BF16 => out.cast(FloatDType::BF16),
        _ => out,
    }
}

/// Row count at which the batched path (transient device dequant +
/// burn's tuned matmul) replaces the fused per-row kernels. The fused
/// kernels re-read the packed weight once PER ACTIVATION ROW — decode
/// never notices, but prompt-shaped calls drown in redundant weight
/// traffic (§ the klein profile: ~95% of a step at ~2.6% of peak).
/// `COMBS_QMATMUL_BATCHED=0` closes the door; any other value overrides
/// the threshold.
fn batched_threshold() -> usize {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| match std::env::var("COMBS_QMATMUL_BATCHED").as_deref() {
        Ok("0") => usize::MAX,
        Ok(v) => v.parse().unwrap_or(8),
        _ => 8,
    })
}

/// Which matmul the batched path uses. `MatmulStrategy::Cube` is a
/// DIAGNOSTIC ONLY: it is known to produce flat-grey images in the live
/// diffusion pipeline while agreeing with the tuned kernel to 1e-3 when
/// the same matmul runs in isolation. It exists so that discrepancy can
/// be investigated with real logging instead of inference.
fn matmul_strategy() -> MatmulStrategy {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let untuned = *S.get_or_init(|| {
        let untuned =
            matches!(std::env::var("COMBS_QMATMUL_STRATEGY").as_deref(), Ok("cube"));
        if untuned {
            eprintln!(
                "[qmatmul] WARNING: COMBS_QMATMUL_STRATEGY=cube — diagnostic only, \
                 known to produce wrong images in the diffusion pipeline"
            );
        }
        untuned
    });
    if untuned { MatmulStrategy::Cube } else { MatmulStrategy::default() }
}

/// One line describing what the batched path will actually do, so a run
/// records its configuration instead of leaving it to be assumed.
pub fn batched_matmul_summary() -> String {
    let threshold = batched_threshold();
    let strategy = if matches!(matmul_strategy(), MatmulStrategy::Cube) {
        "cube (DIAGNOSTIC)"
    } else {
        "tuned"
    };
    if threshold == usize::MAX {
        "batched matmul OFF (COMBS_QMATMUL_BATCHED=0)".to_string()
    } else {
        format!("batched matmul from {threshold} rows, {strategy} kernel")
    }
}

/// First few batched calls report the shapes and dtypes they actually
/// saw under `COMBS_QMATMUL_DEBUG=1` — the ground truth an A/B needs,
/// since burn resolves tensor dtypes from per-device defaults and an
/// f32 backend can still be handed f16 tensors.
fn debug_first_calls(m: usize, k: usize, n: usize, x_dtype: DType, out_dtype: DType) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("COMBS_QMATMUL_DEBUG").as_deref() == Ok("1")) {
        return;
    }
    static SEEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n_seen = SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n_seen < 8 {
        eprintln!(
            "[qmatmul] call {n_seen}: m {m} k {k} n {n} | x {x_dtype:?} out {out_dtype:?} | {}",
            batched_matmul_summary()
        );
    }
}

/// The batched path: weight read ONCE (dequantized into a transient f32
/// buffer that dies with the call), FLOPs handed to the tuned matmul.
/// `None` below the threshold or for formats without a dequant kernel —
/// the caller falls through to the fused kernels.
fn try_batched_matmul(
    w: &QuantWeight,
    x: &CubeTensor<WgpuRuntime>,
    batch: usize,
    seq: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    let m = batch * seq;
    if m < batched_threshold() {
        return None;
    }
    let (n_out, k) = (w.n_out(), w.k());
    let w_h = w.dequant_device(&x.client)?;
    let x2 = CubeTensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        Shape::from([m, k]),
        x.handle.clone(),
        DType::F32,
    );
    let w2 = CubeTensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        Shape::from([n_out, k]),
        w_h,
        DType::F32,
    );
    debug_first_calls(m, k, n_out, x.dtype, DType::F32);
    // The strategy is load-bearing for CORRECTNESS here, not only for
    // speed: the untuned kernel turns generated images flat grey even
    // though the two agree to 1e-3 when the same matmul runs in
    // isolation at these shapes (see the strategy test). Diagnostic
    // door only — the default stays on what demonstrably works.
    let out = matmul(x2, permute(w2, &[1, 0]), None, matmul_strategy(), DType::F32).ok()?;
    // Submit now so the dequant transient's slot recycles before the
    // NEXT block allocates its own — unflushed, the transients stack
    // across a step's 25 block linears and the peak walks into
    // machine-killing territory on unified memory (the 2026-08-28 OOM).
    let _ = x.client.flush();
    Some(CubeTensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        Shape::from([batch, seq, n_out]),
        out.handle,
        DType::F32,
    ))
}

impl QuantLinearOp<UnfusedF32> for CubeQuantLinear {
    fn forward(&self, x: Tensor<UnfusedF32, 3>) -> Tensor<UnfusedF32, 3> {
        let in_dtype = x.dtype();
        let [batch, seq, _] = x.dims();
        let prim = to_f32(x).into_primitive().tensor();
        let out = self.forward_cube(prim, batch, seq);
        to_dtype(
            Tensor::from_primitive(TensorPrimitive::Float(out)),
            in_dtype,
        )
    }

    fn dims(&self) -> [usize; 2] {
        CubeQuantLinear::dims(self)
    }

    fn vram_bytes(&self) -> usize {
        self.w.vram_bytes()
    }
}

impl QuantLinearOp<UnfusedF16> for CubeQuantLinear {
    fn forward(&self, x: Tensor<UnfusedF16, 3>) -> Tensor<UnfusedF16, 3> {
        // Casting the (small) activation up costs nothing next to the
        // weight win, and the f32-accumulated matmul is *better*
        // numerically than an f16 one.
        let in_dtype = x.dtype();
        let [batch, seq, _] = x.dims();
        let prim = to_f32(x).into_primitive().tensor();
        let out = self.forward_cube(prim, batch, seq);
        to_dtype(
            Tensor::<UnfusedF16, 3>::from_primitive(TensorPrimitive::Float(out)),
            in_dtype,
        )
    }

    fn dims(&self) -> [usize; 2] {
        CubeQuantLinear::dims(self)
    }

    fn vram_bytes(&self) -> usize {
        self.w.vram_bytes()
    }
}

/// The fusion-stream operation for the fused backend: executed when the
/// stream drains, with inputs resolved to real device tensors.
struct QuantMatmulOp {
    desc: CustomOpIr,
    w: Arc<QuantWeight>,
    batch: usize,
    seq: usize,
}

impl core::fmt::Debug for QuantMatmulOp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "QuantMatmulOp {{ w: [{}, {}], m: {} }}",
            self.w.n_out(),
            self.w.k(),
            self.batch * self.seq
        )
    }
}

impl Operation<FusionCubeRuntime<WgpuRuntime>> for QuantMatmulOp {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle<WgpuRuntime>>) {
        let ([input], [output]) = self.desc.as_fixed::<1, 1>();
        let x: CubeTensor<WgpuRuntime> = handles.get_float_tensor::<InnerF32>(input);
        let x = into_contiguous(x);
        if let Some(out) = try_batched_matmul(&self.w, &x, self.batch, self.seq) {
            handles.register_float_tensor::<InnerF32>(&output.id, out);
            return;
        }
        let out_h = self.w.matmul_device(&x.client, x.handle.clone(), self.batch * self.seq);
        let out = CubeTensor::new_contiguous(
            x.client.clone(),
            x.device.clone(),
            Shape::from([self.batch, self.seq, self.w.n_out()]),
            out_h,
            DType::F32,
        );
        handles.register_float_tensor::<InnerF32>(&output.id, out);
    }
}

impl QuantLinearOp<FusedF32> for CubeQuantLinear {
    fn forward(&self, x: Tensor<FusedF32, 3>) -> Tensor<FusedF32, 3> {
        let in_dtype = x.dtype();
        let [batch, seq, _] = x.dims();
        let prim = to_f32(x).into_primitive().tensor();
        let client = prim.client.clone();

        let mut streams = OperationStreams::default();
        streams.tensor(&prim);
        let input_ir = prim.into_ir();
        let out_ir = TensorIr {
            id: client.create_empty_handle(),
            shape: Shape::from([batch, seq, self.w.n_out()]),
            status: TensorStatus::NotInit,
            dtype: DType::F32,
        };
        let desc = CustomOpIr::new("combs_quant_matmul", &[input_ir], &[out_ir]);
        let op = QuantMatmulOp {
            desc: desc.clone(),
            w: self.w.clone(),
            batch,
            seq,
        };
        let mut outputs = client.register(streams, OperationIr::Custom(desc), op);
        let out = outputs.pop().expect("custom op declares one output");
        to_dtype(
            Tensor::from_primitive(TensorPrimitive::Float(out)),
            in_dtype,
        )
    }

    fn dims(&self) -> [usize; 2] {
        CubeQuantLinear::dims(self)
    }

    fn vram_bytes(&self) -> usize {
        self.w.vram_bytes()
    }
}

/// Boxes `op` as a `QuantLinearOp<B>` iff `B` is `T` — a safe runtime
/// type-equality bridge (models stay generic; no specialization needed).
fn cast_op<B: Backend, T: Backend>(op: Box<dyn QuantLinearOp<T>>) -> Option<Box<dyn QuantLinearOp<B>>> {
    let any: Box<dyn Any> = Box::new(op);
    any.downcast::<Box<dyn QuantLinearOp<B>>>().ok().map(|b| *b)
}

/// Tries to build the quantized fast path for `name`: packed bytes from the
/// source + a kernel dispatch matching `B`. `None` → caller uses the dense
/// fallback. Errors only on malformed packed data.
fn debug_quant(name: &str, outcome: &str) {
    if std::env::var_os("COMBS_DEBUG_QUANT").is_some() {
        eprintln!("quant-linear {name}: {outcome}");
    }
}

/// Wraps an already-packed weight as a [`QuantLinearOp`] for `B` — the
/// bridge that lets a tied lm_head share the embedding's packed table
/// (one `Arc`, one copy in VRAM) instead of loading a second one.
pub(crate) fn quant_linear_from_weight<B: Backend>(
    w: Arc<QuantWeight>,
) -> Option<Box<dyn QuantLinearOp<B>>> {
    let lin = CubeQuantLinear { w };
    if TypeId::of::<B>() == TypeId::of::<FusedF32>() {
        return cast_op::<B, FusedF32>(Box::new(lin));
    }
    if TypeId::of::<B>() == TypeId::of::<UnfusedF32>() {
        return cast_op::<B, UnfusedF32>(Box::new(lin));
    }
    if TypeId::of::<B>() == TypeId::of::<UnfusedF16>() {
        return cast_op::<B, UnfusedF16>(Box::new(lin));
    }
    None
}

pub fn try_quant_linear<B: Backend>(
    source: &dyn ModelSource,
    name: &str,
    device: &Device<B>,
) -> Result<Option<Box<dyn QuantLinearOp<B>>>> {
    // Escape hatch: force the portable dense path (weights dequantized to
    // float at load). Costs the VRAM win; useful to isolate kernel issues.
    if std::env::var_os("COMBS_NO_QUANT_KERNELS").is_some_and(|v| v != "0") {
        return Ok(None);
    }
    let supported = [
        TypeId::of::<FusedF32>(),
        TypeId::of::<UnfusedF32>(),
        TypeId::of::<UnfusedF16>(),
    ];
    if !supported.contains(&TypeId::of::<B>()) {
        debug_quant(name, "backend not wgpu f32/f16 — dense fallback");
        return Ok(None);
    }
    let device_any: &dyn Any = device;
    let Some(wgpu_device) = device_any.downcast_ref::<WgpuDevice>() else {
        debug_quant(name, "device not WgpuDevice — dense fallback");
        return Ok(None);
    };
    let Some(qt) = source.open_tensor_quant(name).map_err(ModelError::Format)? else {
        debug_quant(name, "no packed quant tensor — dense fallback");
        return Ok(None);
    };
    let &[n_out, k] = qt.shape.as_slice() else {
        debug_quant(name, "not rank-2 — dense fallback");
        return Ok(None);
    };

    let client = <WgpuRuntime as cubecl::prelude::Runtime>::client(wgpu_device);
    // A tensor the kernels can't take (e.g. k not a block multiple — ggml
    // itself falls back to 32-block formats for such shapes) is not an
    // error: the dense path handles it. Kernels are accelerators, never
    // load-bearing.
    let Ok(w) = QuantWeight::from_quant_tensor(&client, qt.format, &qt.data, n_out, k) else {
        debug_quant(name, "kernel-incompatible shape — dense fallback");
        return Ok(None);
    };
    debug_quant(name, "packed on device");
    let lin = CubeQuantLinear { w: Arc::new(w) };

    if TypeId::of::<B>() == TypeId::of::<FusedF32>() {
        return Ok(cast_op::<B, FusedF32>(Box::new(lin)));
    }
    if TypeId::of::<B>() == TypeId::of::<UnfusedF32>() {
        return Ok(cast_op::<B, UnfusedF32>(Box::new(lin)));
    }
    if TypeId::of::<B>() == TypeId::of::<UnfusedF16>() {
        return Ok(cast_op::<B, UnfusedF16>(Box::new(lin)));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use combs_formats::QuantFormat;
    use cubecl::prelude::Runtime;

    /// Synthetic Q4_0 stream (mirrors qmatmul's test generator).
    fn synth_q4_0(n_blocks: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n_blocks * 18);
        let mut s = 0x12345678u32;
        for b in 0..n_blocks {
            let scale = burn::tensor::f16::from_f32(0.003 * ((b % 11) as f32 + 1.0));
            out.extend_from_slice(&scale.to_le_bytes());
            for _ in 0..16 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                out.push((s >> 24) as u8);
            }
        }
        out
    }

    /// burn 0.21 locks per-device default dtypes to whichever backend
    /// touches the device first, and tests share one wgpu device — without
    /// pinning, the f16 test can lock the device to F16 and the f32 test's
    /// `from_data` would round its reference tensors through f16. Pin F32
    /// defaults before any tensor exists; each test then casts explicitly
    /// to the dtype it intends.
    fn pin_device_dtypes() {
        use std::sync::Once;
        static PIN: Once = Once::new();
        PIN.call_once(|| {
            let device = WgpuDevice::default();
            let _ = burn::tensor::set_default_dtypes::<UnfusedF32>(
                &device,
                FloatDType::F32,
                burn::tensor::IntDType::I32,
            );
        });
    }

    fn quant_and_dense<B: Backend>(
        device: &Device<B>,
        n_out: usize,
        k: usize,
        dtype: FloatDType,
    ) -> (Linear<B>, Linear<B>)
    where
        CubeQuantLinear: QuantLinearOp<B>,
    {
        let data = synth_q4_0(n_out * k / 32);
        let client = <WgpuRuntime as Runtime>::client(&Default::default());
        let w = Arc::new(
            QuantWeight::from_quant_tensor(&client, QuantFormat::Q4_0, &data, n_out, k).unwrap(),
        );
        let quant = Linear::Quant(Box::new(CubeQuantLinear { w }) as Box<dyn QuantLinearOp<B>>);
        let wf = combs_formats::quants::dequantize_q4_0(&data, n_out * k).unwrap();
        let dense = Linear::Dense(
            Tensor::<B, 2>::from_data(TensorData::new(wf, [n_out, k]), device).cast(dtype),
        );
        (quant, dense)
    }

    fn assert_close(got: &[f32], expect: &[f32], rel: f32) {
        assert_eq!(got.len(), expect.len());
        for (i, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
            let tol = rel * e.abs().max(1.0);
            assert!((g - e).abs() <= tol, "[{i}]: got {g}, expect {e}");
        }
    }

    /// The default engine backend: the quantized linear runs as a custom op
    /// inside the fusion stream and must match the dense path, with a bias.
    #[test]
    fn fused_backend_matches_dense() {
        if crate::skip_no_gpu() {
            return;
        }
        pin_device_dtypes();
        let device: Device<FusedF32> = Default::default();
        let (n_out, k) = (48, 64);
        let (quant, dense) = quant_and_dense::<FusedF32>(&device, n_out, k, FloatDType::F32);
        assert_eq!(quant.dims(), [n_out, k]);

        let x: Vec<f32> = (0..3 * k).map(|i| ((i % 32) as f32) / 16.0 - 1.0).collect();
        let x = Tensor::<FusedF32, 3>::from_data(TensorData::new(x, [1, 3, k]), &device)
            .cast(FloatDType::F32);
        let b: Vec<f32> = (0..n_out).map(|i| (i as f32) / 100.0).collect();
        let bias = Tensor::<FusedF32, 1>::from_data(TensorData::new(b, [n_out]), &device)
            .cast(FloatDType::F32);

        let got: Vec<f32> = quant
            .forward(x.clone(), Some(&bias))
            .into_data()
            .to_vec()
            .unwrap();
        let expect: Vec<f32> = dense
            .forward(x, Some(&bias))
            .into_data()
            .to_vec()
            .unwrap();
        assert_close(&got, &expect, 1e-4);
    }

    /// The batched path hands its FLOPs to burn's matmul, so which
    /// strategy that resolves to is load-bearing. In ISOLATION the
    /// tuned and untuned kernels agree to 1e-3 at the diffusion
    /// blocks' real shape and token count — which is the surprise
    /// worth pinning: in the live pipeline the untuned one yields flat
    /// grey images. Whatever the difference is, it is not this matmul
    /// alone, and this test is the boundary of what has been ruled
    /// out.
    #[test]
    fn matmul_strategies_must_agree_before_one_is_chosen() {
        if crate::skip_no_gpu() {
            return;
        }
        pin_device_dtypes();
        let device: Device<UnfusedF32> = Default::default();
        // Shapes the live pipeline actually runs, logged from a real
        // generation: the text encoder's projections first (its output
        // is the conditioning, and a dead conditioning pathway is
        // exactly what grey images look like), then a single-stream
        // block's fused projection.
        for (m, k, n) in [
            (512usize, 2560usize, 4096usize),
            (512, 2560, 1024),
            (512, 4096, 2560),
            (768, 3072, 27648),
        ] {
        let x: Vec<f32> = (0..m * k).map(|i| ((i % 37) as f32) / 18.0 - 1.0).collect();
        let w: Vec<f32> = (0..n * k).map(|i| ((i % 53) as f32) / 26.0 - 1.0).collect();
        let xt = Tensor::<UnfusedF32, 2>::from_data(TensorData::new(x, [m, k]), &device);
        let wt = Tensor::<UnfusedF32, 2>::from_data(TensorData::new(w, [n, k]), &device);
        // The reference: burn's own tensor-level matmul.
        let expect: Vec<f32> = xt
            .clone()
            .matmul(wt.clone().transpose())
            .into_data()
            .to_vec()
            .unwrap();

        let prim = |t: Tensor<UnfusedF32, 2>| t.into_primitive().tensor();
        for (label, strategy) in [
            ("default", MatmulStrategy::default()),
            ("cube", MatmulStrategy::Cube),
        ] {
            let out = matmul(
                prim(xt.clone()),
                permute(prim(wt.clone()), &[1, 0]),
                None,
                strategy,
                DType::F32,
            )
            .expect("matmul launches");
            let got: Vec<f32> =
                Tensor::<UnfusedF32, 2>::from_primitive(TensorPrimitive::Float(out))
                    .into_data()
                    .to_vec()
                    .unwrap();
            assert_close(&got, &expect, 1e-3);
            let _ = label;
            }
        }
    }

    /// Above the batched threshold the SAME public path routes through
    /// the transient-dequant + tuned matmul; it must match the dense
    /// reference too (looser tolerance — the tuned matmul's accumulation
    /// order legitimately differs from the fused kernels'). Runs on both
    /// wgpu backends so both dispatch seams cover the new branch.
    #[test]
    fn batched_prompt_shape_matches_dense() {
        if crate::skip_no_gpu() {
            return;
        }
        pin_device_dtypes();
        let (n_out, k, seq) = (48, 64, 64);

        fn run<B: Backend>(n_out: usize, k: usize, seq: usize)
        where
            CubeQuantLinear: QuantLinearOp<B>,
        {
            let device: Device<B> = Default::default();
            let (quant, dense) = quant_and_dense::<B>(&device, n_out, k, FloatDType::F32);
            let x: Vec<f32> =
                (0..seq * k).map(|i| ((i % 29) as f32) / 14.0 - 1.0).collect();
            let x = Tensor::<B, 3>::from_data(TensorData::new(x, [1, seq, k]), &device)
                .cast(FloatDType::F32);
            let got: Vec<f32> =
                quant.forward(x.clone(), None).into_data().to_vec().unwrap();
            let expect: Vec<f32> =
                dense.forward(x, None).into_data().to_vec().unwrap();
            assert_close(&got, &expect, 1e-3);
        }

        run::<FusedF32>(n_out, k, seq);
        run::<UnfusedF32>(n_out, k, seq);
    }

    /// The f16 build: activation is cast around the f32 kernel; tolerance
    /// covers the final f16 rounding of the output.
    #[test]
    fn f16_backend_matches_dense() {
        if crate::skip_no_gpu() {
            return;
        }
        pin_device_dtypes();
        let device: Device<UnfusedF16> = Default::default();
        let (n_out, k) = (48, 64);
        let (quant, dense) = quant_and_dense::<UnfusedF16>(&device, n_out, k, FloatDType::F16);

        let x: Vec<f32> = (0..3 * k).map(|i| ((i % 32) as f32) / 16.0 - 1.0).collect();
        let x = Tensor::<UnfusedF16, 3>::from_data(TensorData::new(x, [1, 3, k]), &device)
            .cast(FloatDType::F16);

        let got: Vec<f32> = quant
            .forward(x.clone(), None)
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        let expect: Vec<f32> = dense
            .forward(x, None)
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        assert_close(&got, &expect, 1e-2);
    }
}
