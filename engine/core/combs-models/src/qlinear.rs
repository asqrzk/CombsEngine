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
use burn_cubecl::kernel::{cast, into_contiguous};
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

/// Which matmul the batched path uses.
///
/// `MatmulStrategy::Cube` is DIAGNOSTIC ONLY, and now for a known
/// reason: it launches with cubek's DEFAULT strategy rather than one
/// chosen for the shape, and at the widths a transformer block runs
/// that launch reports success and writes nothing. The caller gets the
/// output buffer exactly as it was allocated — every element zero —
/// and because the launch returned `Ok`, nothing falls back. A
/// conditioning of zeros, or a velocity field of zeros, is a grey
/// image. Measured: 1,572,864 of 1,572,864 outputs zero at
/// m 768, k 3072, n 2048, on both backends; correct at m 64, k 512,
/// n 256. Autotune is immune because it measures its candidates, and a
/// routine that writes nothing loses.
/// Which `k` widths the diagnostic untuned kernel is allowed to take,
/// when the door is open. Empty = all of them.
///
/// The grey-image fault appears only in the diffusion pipeline, and
/// that pipeline runs two models: a text encoder with hidden 2560 and a
/// transformer with hidden 3072. Their `k` widths are disjoint, so `k`
/// is a usable proxy for WHICH model a call belongs to — the crudest
/// possible bisection, and the one that needs no plumbing through five
/// layers to ask a question worth two runs.
fn cube_k_filter() -> &'static [usize] {
    static F: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    F.get_or_init(|| {
        std::env::var("COMBS_QMATMUL_CUBE_K")
            .ok()
            .map(|v| v.split(',').filter_map(|n| n.trim().parse().ok()).collect())
            .unwrap_or_default()
    })
}

/// The strategy for a call of this width. Identical to
/// [`matmul_strategy`] unless the k-filter narrows the diagnostic door
/// to part of the pipeline.
fn matmul_strategy_for(k: usize) -> MatmulStrategy {
    let chosen = matmul_strategy();
    if matches!(chosen, MatmulStrategy::Cube) {
        let filter = cube_k_filter();
        if !filter.is_empty() && !filter.contains(&k) {
            return MatmulStrategy::default();
        }
    }
    chosen
}

fn matmul_strategy() -> MatmulStrategy {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let untuned = *S.get_or_init(|| {
        let untuned =
            matches!(std::env::var("COMBS_QMATMUL_STRATEGY").as_deref(), Ok("cube"));
        if untuned {
            eprintln!(
                "[qmatmul] WARNING: COMBS_QMATMUL_STRATEGY=cube — diagnostic only. \
                 At transformer-block widths this launch writes nothing and leaves \
                 the output buffer at zero, which reaches you as a grey image."
            );
        }
        untuned
    });
    if untuned { MatmulStrategy::Cube } else { MatmulStrategy::default() }
}

/// The element width the batched path hands its operands to the matmul
/// in. The multiply is 88–97% of a batched call and this machine has no
/// matrix units reachable from WGSL, so the multiply's cost tracks the
/// bytes it moves: narrowing both operands to f16, with the accumulator
/// and the output left f32, measured a 2.3x image step and a 1.8x
/// 641-token prefill. Activations are never STORED narrow — the tensor
/// arriving is f32 and the tensor leaving is f32 — which is what
/// separates this from the f16 pipeline the flow-matching model's range
/// closed off.
///
/// **Off by default, and the reason is a gate that failed rather than a
/// doubt.** The narrow operands answer the wide ones to ~1.3e-3 of the
/// output RMS, but a four-step sampler amplifies that: the same prompt
/// and seed produce the same fox, matching in mean and standard
/// deviation to 0.02% and visually indistinguishable, with 0.07% of
/// channels differing by more than 10 and a worst channel of 27. The
/// gate written down beforehand asked for 2. It is a real change to
/// what images come out, so it is offered, not imposed:
/// `COMBS_QMATMUL_OPERAND=f16` takes it.
fn operand_dtype() -> DType {
    static D: std::sync::OnceLock<DType> = std::sync::OnceLock::new();
    *D.get_or_init(
        || match std::env::var("COMBS_QMATMUL_OPERAND").as_deref() {
            Ok("f16") => DType::F16,
            _ => DType::F32,
        },
    )
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
    let operand = match operand_dtype() {
        DType::F16 => "f16 operands/f32 accumulator",
        _ => "f32 operands",
    };
    let scope = match cube_k_filter() {
        [] => String::new(),
        widths => format!(", cube limited to k in {widths:?}"),
    };
    if threshold == usize::MAX {
        "batched matmul OFF (COMBS_QMATMUL_BATCHED=0)".to_string()
    } else {
        format!("batched matmul from {threshold} rows, {strategy} kernel, {operand}{scope}")
    }
}

/// First few batched calls report the shapes and dtypes they actually
/// saw under `COMBS_QMATMUL_DEBUG=1` — the ground truth an A/B needs,
/// since burn resolves tensor dtypes from per-device defaults and an
/// f32 backend can still be handed f16 tensors.
fn debug_first_calls(m: usize, k: usize, n: usize, x_dtype: DType, operand: DType) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("COMBS_QMATMUL_DEBUG").as_deref() == Ok("1")) {
        return;
    }
    static SEEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n_seen = SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n_seen < 8 {
        eprintln!(
            "[qmatmul] call {n_seen}: m {m} k {k} n {n} | x {x_dtype:?} operands {operand:?} out F32 | {}",
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
    try_batched_matmul_with(w, x, batch, seq, operand_dtype())
}

/// The batched path with its operand width named rather than read from
/// the door, so a single process can run both widths and compare them.
fn try_batched_matmul_with(
    w: &QuantWeight,
    x: &CubeTensor<WgpuRuntime>,
    batch: usize,
    seq: usize,
    operand: DType,
) -> Option<CubeTensor<WgpuRuntime>> {
    let m = batch * seq;
    if m < batched_threshold() {
        return None;
    }
    let (n_out, k) = (w.n_out(), w.k());
    // The weight is written at the operand width by the dequant kernel
    // itself. A separate narrowing pass would cost about what the whole
    // dequant costs — a wasted full-weight pass is expensive, which the
    // cost-split instrument prices directly.
    let w_h = match operand {
        DType::F16 => w.dequant_device_as::<burn::tensor::f16>(&x.client)?,
        _ => w.dequant_device(&x.client)?,
    };
    let x2 = CubeTensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        Shape::from([m, k]),
        x.handle.clone(),
        DType::F32,
    );
    // Both operands must share a dtype for the matmul to bind them, so
    // narrowing the weight means narrowing the activation too — and the
    // activation is the one with a range problem. Measured on a
    // flow-matching image run: the deep single-stream blocks reach
    // 60551, against f16's largest finite value of 65504. An 8% margin,
    // on a quantity that moves with prompt, seed, canvas and step, and
    // whose overflow is Inf rather than a small error. So the operands
    // are normalized first: divide by a power of two that brings the
    // largest magnitude to 1, and multiply the f32 result back by the
    // same. A power of two only moves exponents, so the scaling is
    // exact both ways and the sole remaining error is still the f16
    // store. The scale is computed and applied ON DEVICE — reading it
    // back would sync the queue once per linear, which costs far more
    // than it saves.
    let (x2, scale) = match operand {
        DType::F32 => (x2, None),
        dt => {
            let xt = Tensor::<InnerF32, 2>::from_primitive(TensorPrimitive::Float(x2));
            let ln2 = core::f32::consts::LN_2;
            let amax = xt.clone().abs().max().clamp_min(f32::MIN_POSITIVE);
            let exponent = (amax.log() / ln2).ceil();
            let scale = (exponent * ln2).exp().reshape([1, 1]);
            let normalized = xt.div(scale.clone().expand([m, k])).cast(FloatDType::F16);
            (
                cast(normalized.into_primitive().tensor(), dt),
                Some(scale),
            )
        }
    };
    let w2 = CubeTensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        Shape::from([n_out, k]),
        w_h,
        operand,
    );
    debug_first_calls(m, k, n_out, x.dtype, operand);
    // The strategy is load-bearing for CORRECTNESS here, not only for
    // speed: the untuned kernel turns generated images flat grey even
    // though the two agree to 1e-3 when the same matmul runs in
    // isolation at these shapes (see the strategy test). Diagnostic
    // door only — the default stays on what demonstrably works.
    let out = matmul(x2, permute(w2, &[1, 0]), None, matmul_strategy_for(k), DType::F32).ok()?;
    // Undo the normalization on the f32 result, where the product has
    // room. Exact: the scale is a power of two.
    let out = match scale {
        None => out,
        Some(scale) => {
            let ot = Tensor::<InnerF32, 2>::from_primitive(TensorPrimitive::Float(out));
            ot.mul(scale.expand([m, n_out]))
                .into_primitive()
                .tensor()
        }
    };
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
/// Every weight's dispatch decision is counted, whether or not anyone
/// is watching: the per-tensor detail stays behind `COMBS_DEBUG_QUANT`,
/// but the totals answer "did this model actually get the fast path?"
/// — a question that has been answered by assumption before.
static QUANT_DECISIONS: std::sync::Mutex<Vec<(String, usize)>> =
    std::sync::Mutex::new(Vec::new());

fn debug_quant(name: &str, outcome: &str) {
    if std::env::var_os("COMBS_DEBUG_QUANT").is_some() {
        eprintln!("quant-linear {name}: {outcome}");
    }
    if let Ok(mut counts) = QUANT_DECISIONS.lock() {
        match counts.iter_mut().find(|(o, _)| o == outcome) {
            Some((_, n)) => *n += 1,
            None => counts.push((outcome.to_string(), 1)),
        }
    }
}

/// The dispatch census so far, most common first — e.g.
/// `"packed on device=196, no packed quant tensor=4"`. Empty when no
/// quantized linear has been considered.
pub fn quant_census() -> String {
    let Ok(counts) = QUANT_DECISIONS.lock() else {
        return String::new();
    };
    let mut rows: Vec<&(String, usize)> = counts.iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    rows.iter().map(|(o, n)| format!("{o}={n}")).collect::<Vec<_>>().join(", ")
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

    /// The live path does not upload its weight — it WRITES one with
    /// the dequant kernel into a fresh device allocation and hands that
    /// straight to matmul. This builds the right-hand side exactly that
    /// way, which is the last difference between the isolated test
    /// (where the strategies agree) and the pipeline (where the untuned
    /// one produces grey).
    #[test]
    fn strategies_agree_on_a_freshly_dequantized_weight() {
        if crate::skip_no_gpu() {
            return;
        }
        pin_device_dtypes();
        let device: Device<UnfusedF32> = Default::default();
        let client = <WgpuRuntime as Runtime>::client(&Default::default());
        let (m, k, n) = (768usize, 2560usize, 1024usize);

        let data = synth_q4_0(n * k / 32);
        let w = QuantWeight::from_quant_tensor(
            &client,
            combs_formats::QuantFormat::Q4_0,
            &data,
            n,
            k,
        )
        .unwrap();
        let reference_w = combs_formats::quants::dequantize_q4_0(&data, n * k).unwrap();

        let x: Vec<f32> = (0..m * k).map(|i| ((i % 37) as f32) / 18.0 - 1.0).collect();
        let xt = Tensor::<UnfusedF32, 2>::from_data(TensorData::new(x, [m, k]), &device);
        let wt = Tensor::<UnfusedF32, 2>::from_data(
            TensorData::new(reference_w, [n, k]),
            &device,
        );
        let expect: Vec<f32> = xt
            .clone()
            .matmul(wt.transpose())
            .into_data()
            .to_vec()
            .unwrap();

        for (label, strategy) in [
            ("default", MatmulStrategy::default()),
            ("cube", MatmulStrategy::Cube),
        ] {
            // A fresh dequant per arm: the transient is single-use.
            let w_h = w.dequant_device(&client).expect("dequant kernel");
            let w2 = CubeTensor::new_contiguous(
                client.clone(),
                Default::default(),
                Shape::from([n, k]),
                w_h,
                DType::F32,
            );
            let x2 = xt.clone().into_primitive().tensor();
            let out = matmul(x2, permute(w2, &[1, 0]), None, strategy, DType::F32)
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

    fn lcg_bytes(n: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8
            })
            .collect()
    }

    fn synth_q8_0(n_blocks: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n_blocks * crate::qmatmul::Q8_0_BLOCK_BYTES);
        for b in 0..n_blocks {
            let scale = burn::tensor::f16::from_f32(0.003 * ((b % 11) as f32 + 1.0));
            out.extend_from_slice(&scale.to_le_bytes());
            out.extend_from_slice(&lcg_bytes(32, 0x80C0 ^ b as u32));
        }
        out
    }

    fn synth_q4_k(n_sb: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n_sb * crate::qmatmul::Q4_K_BLOCK_BYTES);
        for b in 0..n_sb {
            let d = burn::tensor::f16::from_f32(0.002 * ((b % 9) as f32 + 1.0));
            let dmin = burn::tensor::f16::from_f32(0.001 * ((b % 5) as f32 + 1.0));
            out.extend_from_slice(&d.to_le_bytes());
            out.extend_from_slice(&dmin.to_le_bytes());
            out.extend_from_slice(&lcg_bytes(140, 0xC0FFEE ^ b as u32));
        }
        out
    }

    /// The narrow-operand path must answer what the wide one answers.
    /// Shapes are the live pipeline's, with the widest output dimension
    /// cut down: the arithmetic that could drift is per-element and per
    /// accumulation, so `n` only multiplies the count of independent
    /// checks — it does not add a way to be wrong. The row counts, which
    /// DO change which kernel runs, are the real ones (512 encoder, 768
    /// at a 256 canvas, 1536 at 512). Full-size shapes are covered by
    /// the image parity gate.
    #[test]
    fn narrow_operands_agree_with_wide_ones() {
        use combs_formats::QuantFormat;
        use cubecl::server::Handle;

        if crate::skip_no_gpu() {
            return;
        }
        pin_device_dtypes();
        let device: Device<UnfusedF32> = Default::default();
        let client = <WgpuRuntime as Runtime>::client(&Default::default());
        let mk = |h: &Handle, dims: [usize; 2], dt: DType| {
            CubeTensor::new_contiguous(
                client.clone(),
                WgpuDevice::default(),
                Shape::from(dims),
                h.clone(),
                dt,
            )
        };

        for (fmt, m, k, n) in [
            (QuantFormat::Q4K, 512usize, 2560usize, 1024usize),
            (QuantFormat::Q4K, 512, 4096, 1024),
            (QuantFormat::Q8_0, 768, 3072, 1024),
            (QuantFormat::Q8_0, 1536, 3072, 1024),
        ] {
            let data = match fmt {
                QuantFormat::Q8_0 => synth_q8_0(n * k / 32),
                _ => synth_q4_k(n * k / 256),
            };
            let w = QuantWeight::from_quant_tensor(&client, fmt, &data, n, k).unwrap();
            let xv: Vec<f32> = (0..m * k).map(|i| ((i % 37) as f32) / 18.0 - 1.0).collect();
            let x_h = Tensor::<UnfusedF32, 2>::from_data(TensorData::new(xv, [m, k]), &device)
                .into_primitive()
                .tensor()
                .handle;

            // The production path itself, both widths, in one process —
            // not a copy of it that could drift from what ships.
            let run = |operand: DType| -> Vec<f32> {
                let out =
                    try_batched_matmul_with(&w, &mk(&x_h, [m, k], DType::F32), 1, m, operand)
                        .expect("batched path engages at these row counts");
                Tensor::<UnfusedF32, 3>::from_primitive(TensorPrimitive::Float(out))
                    .into_data()
                    .to_vec()
                    .unwrap()
            };

            let wide = run(DType::F32);
            let narrow = run(DType::F16);
            assert_eq!(wide.len(), narrow.len());
            let mut worst_abs = 0.0f32;
            let mut worst_elem = 0.0f32;
            let mut sq = 0.0f64;
            let mut peak = 0.0f32;
            for (g, e) in narrow.iter().zip(wide.iter()) {
                worst_abs = worst_abs.max((g - e).abs());
                worst_elem = worst_elem.max((g - e).abs() / e.abs().max(1.0));
                sq += (*e as f64) * (*e as f64);
                peak = peak.max(e.abs());
            }
            let rms = (sq / wide.len() as f64).sqrt() as f32;
            // Two denominators, because they answer different questions.
            // Elementwise (the one first written down) divides by the
            // output's own magnitude, so at an output that cancelled to
            // near zero it reports the cancellation, not the operands'
            // precision — a matmul whose terms are large and whose sum
            // is small will fail it at any operand width below the
            // reference. Against the output RMS the number is the one
            // the pre-registered 1e-2 was reasoning about: how much of
            // the signal the narrow operands cost.
            println!(
                "narrow-vs-wide {fmt:?} m {m} k {k} n {n}: \
                 worst abs {worst_abs:.3e} · rms {rms:.3e} · peak {peak:.3e} · \
                 rms-relative {:.2e} · elementwise {worst_elem:.2e}",
                worst_abs / rms
            );
            assert!(
                worst_abs / rms <= 1e-2,
                "{fmt:?} m {m} k {k} n {n}: worst delta {worst_abs:.3e} is \
                 {:.3e} of the output RMS {rms:.3e}, over 1e-2",
                worst_abs / rms
            );
        }
    }

    /// Many quantized linears, interleaved with the ordinary tensor work
    /// a real forward does, on the fusion backend.
    ///
    /// This is the last thing that differs. The untuned kernel ruins a
    /// diffusion run and leaves text untouched, and everything else that
    /// separated the two has been excluded: the model (the same
    /// Qwen3-4B is byte-identical as text and corrupt as an encoder),
    /// the sequence length (630 tokens, identical), the taps forward
    /// path (matches the reference under either kernel), determinism
    /// (six million outputs agree between two identical calls), and the
    /// half of the pipeline (BOTH halves corrupt, differently). What is
    /// left is that a generation puts hundreds of these calls into a
    /// busy fusion stream, and no test did.
    ///
    /// Run under `COMBS_QMATMUL_STRATEGY=cube` to ask the question this
    /// exists for; under the default it guards the shipping path.
    #[test]
    fn many_interleaved_calls_agree_with_dense() {
        if crate::skip_no_gpu() {
            return;
        }
        pin_device_dtypes();
        // Both backends, small widths and the widths a transformer block
        // actually runs. Four cells, and the fault lives in exactly one.
        fn round_trip<B: Backend>(n_out: usize, k: usize, seq: usize, rounds: usize, label: &str)
        where
            CubeQuantLinear: QuantLinearOp<B>,
        {
        let device: Device<B> = Default::default();
        let (quant, dense) = quant_and_dense::<B>(&device, n_out, k, FloatDType::F32);

        let x0: Vec<f32> = (0..seq * k).map(|i| ((i % 29) as f32) / 14.0 - 1.0).collect();
        let base = Tensor::<B, 3>::from_data(TensorData::new(x0, [1, seq, k]), &device)
            .cast(FloatDType::F32);

        // Between the linears, the shapes of work a block actually does:
        // a scale, a residual add, a normalization, a transpose. Enough
        // to keep the stream busy and to make the custom op one
        // participant among many rather than the only thing in flight.
        let mut worst = 0.0f32;
        for round in 0..rounds {
            let drift = (round as f32) * 0.01;
            let x = base.clone() * (1.0 + drift) + drift;
            let x = x.clone() - x.clone().mean_dim(2);
            let got: Vec<f32> = quant.forward(x.clone(), None).into_data().to_vec().unwrap();
            let expect: Vec<f32> = dense.forward(x, None).into_data().to_vec().unwrap();
            assert_eq!(got.len(), expect.len());
            for (g, e) in got.iter().zip(expect.iter()) {
                worst = worst.max((g - e).abs() / e.abs().max(1.0));
            }
            if worst > 1e-3 {
                // How it is wrong matters more than that it is. All
                // zeros means the kernel was launched and wrote
                // nothing, leaving the output buffer as allocated —
                // which is a different fault from arithmetic drift and
                // points at a different place.
                let zeros = got.iter().filter(|v| **v == 0.0).count();
                panic!(
                    "{label} at m {seq} k {k} n {n_out}, round {round}: drifted from \
                     dense by {worst:.3e}; {zeros} of {} outputs are exactly zero",
                    got.len()
                );
            }
        }
        println!(
            "{label:8} m {seq:>4} k {k:>5} n {n_out:>5}: {rounds} rounds, worst relative {worst:.2e}"
        );
        }

        println!("{}", batched_matmul_summary());
        for (n_out, k, seq, rounds) in
            [(256usize, 512usize, 64usize, 48usize), (2048, 3072, 768, 8)]
        {
            round_trip::<UnfusedF32>(n_out, k, seq, rounds, "unfused");
            round_trip::<FusedF32>(n_out, k, seq, rounds, "fused");
        }
    }

    /// The same multiply, twice, in one process: bit for bit or not.
    ///
    /// A one-ulp run-to-run jitter has been on record since the image
    /// pipeline was first measured, seen at a 512 canvas and never at a
    /// 256 one — and the pipeline is now known to be bit-identical at
    /// 256 both within a process and across two. The two canvases put
    /// different row counts through the batched path, and different row
    /// counts get different tuned routines: 768 rows resolve to one
    /// kernel and 1536 to another, three times apart in throughput. If
    /// the jitter is a kernel that accumulates in a nondeterministic
    /// order — a split reduction, an atomic — then the shape that
    /// engages it is the one to name, and it costs a gigabyte to ask
    /// here rather than eleven to ask through a generation.
    #[test]
    fn the_same_multiply_twice_gives_the_same_bytes() {
        use combs_formats::QuantFormat;
        use cubecl::server::Handle;

        if crate::skip_no_gpu() {
            return;
        }
        pin_device_dtypes();
        let device: Device<UnfusedF32> = Default::default();
        let client = <WgpuRuntime as Runtime>::client(&Default::default());
        let mk = |h: &Handle, dims: [usize; 2]| {
            CubeTensor::new_contiguous(
                client.clone(),
                WgpuDevice::default(),
                Shape::from(dims),
                h.clone(),
                DType::F32,
            )
        };

        // The row counts the two canvases actually produce, either side
        // of the point where the tuned choice changes.
        let (k, n) = (3072usize, 4096usize);
        let data = synth_q8_0(n * k / 32);
        let w = QuantWeight::from_quant_tensor(&client, QuantFormat::Q8_0, &data, n, k).unwrap();
        drop(data);

        for m in [768usize, 1024, 1536] {
            let xv: Vec<f32> = (0..m * k)
                .map(|i| ((i % 37) as f32) / 18.0 - 1.0)
                .collect();
            let x_h = Tensor::<UnfusedF32, 2>::from_data(TensorData::new(xv, [m, k]), &device)
                .into_primitive()
                .tensor()
                .handle;
            let once = || -> Vec<f32> {
                let out = try_batched_matmul_with(&w, &mk(&x_h, [m, k]), 1, m, DType::F32)
                    .expect("batched path engages");
                Tensor::<UnfusedF32, 3>::from_primitive(TensorPrimitive::Float(out))
                    .into_data()
                    .to_vec()
                    .unwrap()
            };
            let a = once();
            let b = once();
            let differing = a
                .iter()
                .zip(b.iter())
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            println!(
                "m {m:>5}: {} of {} outputs differ between two identical calls",
                differing,
                a.len()
            );
            assert_eq!(
                differing, 0,
                "m {m}: the same multiply gave different bytes twice — the tuned \
                 routine for this row count accumulates in a nondeterministic order"
            );
        }
    }

    /// Range safety, which is the reason the narrow path normalizes at
    /// all. A flow-matching image run puts 60551 through these linears,
    /// against f16's largest finite value of 65504 — and an overflow
    /// there is Inf, which the pipeline turns into a dead image rather
    /// than a slightly wrong one. Magnitudes here straddle that ceiling
    /// and go three orders past it, plus one far below where an
    /// unnormalized cast would flush to zero instead.
    #[test]
    fn narrow_operands_survive_activations_past_the_f16_ceiling() {
        use combs_formats::QuantFormat;
        use cubecl::server::Handle;

        if crate::skip_no_gpu() {
            return;
        }
        pin_device_dtypes();
        let device: Device<UnfusedF32> = Default::default();
        let client = <WgpuRuntime as Runtime>::client(&Default::default());
        let mk = |h: &Handle, dims: [usize; 2]| {
            CubeTensor::new_contiguous(
                client.clone(),
                WgpuDevice::default(),
                Shape::from(dims),
                h.clone(),
                DType::F32,
            )
        };

        let (m, k, n) = (64usize, 256usize, 128usize);
        let data = synth_q8_0(n * k / 32);
        let w = QuantWeight::from_quant_tensor(&client, QuantFormat::Q8_0, &data, n, k).unwrap();

        for magnitude in [6.0e4f32, 6.5e4, 1.0e6, 1.0e9, 1.0e-6] {
            let xv: Vec<f32> = (0..m * k)
                .map(|i| (((i % 37) as f32) / 18.0 - 1.0) * magnitude)
                .collect();
            let x_h = Tensor::<UnfusedF32, 2>::from_data(TensorData::new(xv, [m, k]), &device)
                .into_primitive()
                .tensor()
                .handle;
            let run = |operand: DType| -> Vec<f32> {
                let out =
                    try_batched_matmul_with(&w, &mk(&x_h, [m, k]), 1, m, operand)
                        .expect("batched path engages");
                Tensor::<UnfusedF32, 3>::from_primitive(TensorPrimitive::Float(out))
                    .into_data()
                    .to_vec()
                    .unwrap()
            };
            let wide = run(DType::F32);
            let narrow = run(DType::F16);

            assert!(
                narrow.iter().all(|v| v.is_finite()),
                "magnitude {magnitude:e}: narrow operands produced a non-finite value"
            );
            let mut worst = 0.0f32;
            let mut sq = 0.0f64;
            for (g, e) in narrow.iter().zip(wide.iter()) {
                worst = worst.max((g - e).abs());
                sq += (*e as f64) * (*e as f64);
            }
            let rms = (sq / wide.len() as f64).sqrt() as f32;
            println!(
                "range {magnitude:e}: rms {rms:.3e} worst {worst:.3e} rms-relative {:.2e}",
                worst / rms
            );
            assert!(
                worst / rms <= 1e-2,
                "magnitude {magnitude:e}: worst delta is {:.3e} of the output RMS",
                worst / rms
            );
        }
    }

    /// Where a batched call's time actually goes, at the shapes the live
    /// pipeline runs. An instrument, not a gate — it asserts nothing and
    /// stays `#[ignore]`d; the numbers are the product:
    ///
    /// ```text
    /// cargo test -p combs-models --release cost_split -- --ignored --nocapture
    /// ```
    ///
    /// Every phase forces completion with a one-element readback before
    /// stopping its clock (wgpu ops are lazy; unforced spans measure
    /// enqueue time). Forcing serializes, so the phase sum runs slower
    /// than the live path — read the ratios, and take `full` as the
    /// like-for-like total.
    #[test]
    #[ignore = "measurement instrument; run explicitly with --ignored"]
    fn batched_call_cost_split() {
        use combs_formats::QuantFormat;
        use cubecl::server::Handle;
        use std::time::Instant;

        if crate::skip_no_gpu() {
            return;
        }
        pin_device_dtypes();
        let device: Device<UnfusedF32> = Default::default();
        let client = <WgpuRuntime as Runtime>::client(&Default::default());

        let mk = |h: &Handle, dims: [usize; 2]| {
            CubeTensor::new_contiguous(
                client.clone(),
                WgpuDevice::default(),
                Shape::from(dims),
                h.clone(),
                DType::F32,
            )
        };
        let mk_dt = |h: &Handle, dims: [usize; 2], dt: DType| {
            CubeTensor::new_contiguous(
                client.clone(),
                WgpuDevice::default(),
                Shape::from(dims),
                h.clone(),
                dt,
            )
        };
        // One element back is enough to prove the queue drained: submission
        // order guarantees everything enqueued earlier finished first.
        let force_dt = |h: &Handle, len: usize, dt: DType| {
            let ct = CubeTensor::new_contiguous(
                client.clone(),
                WgpuDevice::default(),
                Shape::from([len]),
                h.clone(),
                dt,
            );
            let t = Tensor::<UnfusedF32, 1>::from_primitive(TensorPrimitive::Float(ct));
            let _ = t.narrow(0, 0, 1).into_data();
        };
        let force = |h: &Handle, len: usize| force_dt(h, len, DType::F32);
        fn time(reps: usize, body: &mut dyn FnMut()) -> f64 {
            body(); // warm: autotune resolves per key, and the first launch compiles
            let t0 = Instant::now();
            for _ in 0..reps {
                body();
            }
            t0.elapsed().as_secs_f64() / reps as f64
        }

        // Shapes logged from a real generation. The encoder's projections
        // (Q4_K_M Qwen3-4B) run once per image; the DiT's (Q8_0) run once
        // per block per STEP, which is what the arc is about.
        let rows: &[(&str, QuantFormat, usize, usize, usize)] = &[
            ("enc.qkv", QuantFormat::Q4K, 512, 2560, 4096),
            ("enc.ff", QuantFormat::Q4K, 512, 4096, 2560),
            ("dit.single.fused 256", QuantFormat::Q8_0, 768, 3072, 27648),
            ("dit.single.fused 512", QuantFormat::Q8_0, 1536, 3072, 27648),
            ("dit.single.out 512", QuantFormat::Q8_0, 1536, 9216, 3072),
        ];
        const REPS: usize = 3;

        println!(
            "\n{:22} {:>6} {:>6} {:>6} | {:>9} {:>9} {:>9} {:>9} {:>8} {:>9} {:>9}",
            "shape", "m", "k", "n", "dequant", "mm(perm)", "mm(cont)", "relayout", "alloc",
            "full", "no-flush"
        );
        for &(label, fmt, m, k, n) in rows {
            let data = match fmt {
                QuantFormat::Q8_0 => synth_q8_0(n * k / 32),
                QuantFormat::Q4K => synth_q4_k(n * k / 256),
                _ => unreachable!("instrument covers the formats the pipeline ships"),
            };
            let w = QuantWeight::from_quant_tensor(&client, fmt, &data, n, k).unwrap();
            drop(data);

            let xv: Vec<f32> = (0..m * k).map(|i| ((i % 37) as f32) / 18.0 - 1.0).collect();
            let x_h = Tensor::<UnfusedF32, 2>::from_data(TensorData::new(xv, [m, k]), &device)
                .into_primitive()
                .tensor()
                .handle;

            let dequant = time(
                REPS,
                &mut || {
                    let h = w.dequant_device(&client).unwrap();
                    force(&h, n * k);
                },
            );

            // Staged once, so the matmul phases price the multiply alone.
            let w_h = w.dequant_device(&client).unwrap();
            force(&w_h, n * k);

            let mm_perm = time(
                REPS,
                &mut || {
                    let out = matmul(
                        mk(&x_h, [m, k]),
                        permute(mk(&w_h, [n, k]), &[1, 0]),
                        None,
                        MatmulStrategy::default(),
                        DType::F32,
                    )
                    .expect("matmul launches");
                    force(&out.handle, m * n);
                },
            );

            // The same multiply with the weight already laid out [k, n]:
            // what option (i) — a transposed dequant output — would buy.
            let wt_h = into_contiguous(permute(mk(&w_h, [n, k]), &[1, 0])).handle;
            force(&wt_h, n * k);
            let mm_cont = time(
                REPS,
                &mut || {
                    let out = matmul(
                        mk(&x_h, [m, k]),
                        mk(&wt_h, [k, n]),
                        None,
                        MatmulStrategy::default(),
                        DType::F32,
                    )
                    .expect("matmul launches");
                    force(&out.handle, m * n);
                },
            );

            // What a full-weight layout copy costs, if one were happening.
            let relayout = time(
                REPS,
                &mut || {
                    let c = into_contiguous(permute(mk(&w_h, [n, k]), &[1, 0]));
                    force(&c.handle, n * k);
                },
            );

            // Pool churn alone: the transient is re-requested every call.
            let alloc = time(
                REPS,
                &mut || {
                    let h = client.empty(n * k * 4);
                    std::hint::black_box(&h);
                },
            );

            let full = time(
                REPS,
                &mut || {
                    let wh = w.dequant_device(&client).unwrap();
                    let out = matmul(
                        mk(&x_h, [m, k]),
                        permute(mk(&wh, [n, k]), &[1, 0]),
                        None,
                        MatmulStrategy::default(),
                        DType::F32,
                    )
                    .expect("matmul launches");
                    let _ = client.flush();
                    force(&out.handle, m * n);
                },
            );

            let no_flush = time(
                REPS,
                &mut || {
                    let wh = w.dequant_device(&client).unwrap();
                    let out = matmul(
                        mk(&x_h, [m, k]),
                        permute(mk(&wh, [n, k]), &[1, 0]),
                        None,
                        MatmulStrategy::default(),
                        DType::F32,
                    )
                    .expect("matmul launches");
                    force(&out.handle, m * n);
                },
            );

            // Apple's GPUs have no f32 matrix acceleration — the f32
            // kernel runs on plain ALUs. Price narrower OPERANDS with an
            // f32 accumulator before committing to narrow dequant
            // kernels: this is what option (ii) would buy, measured
            // without writing a line of it.
            let narrow = |fdt: FloatDType| -> Option<f64> {
                let as_t = |h: &Handle, dims: [usize; 2]| {
                    Tensor::<UnfusedF32, 2>::from_primitive(TensorPrimitive::Float(mk(h, dims)))
                        .cast(fdt)
                        .into_primitive()
                        .tensor()
                };
                let wf = as_t(&w_h, [n, k]);
                let xf = as_t(&x_h, [m, k]);
                let dt = wf.dtype;
                let (wf_h, xf_h) = (wf.handle.clone(), xf.handle.clone());
                force_dt(&wf_h, n * k, dt);
                force_dt(&xf_h, m * k, dt);
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    time(
                        REPS,
                        &mut || {
                            let out = matmul(
                                mk_dt(&xf_h, [m, k], dt),
                                permute(mk_dt(&wf_h, [n, k], dt), &[1, 0]),
                                None,
                                MatmulStrategy::default(),
                                DType::F32,
                            )
                            .expect("matmul launches");
                            force(&out.handle, m * n);
                        },
                    )
                }))
                .ok()
            };
            // bf16 is deliberately absent: the WGSL compiler rejects it
            // ("bf16 is not a valid WgpuElement"), and the failed autotune
            // takes the compute channel down with it, poisoning every
            // later measurement in the process. Recorded, not retried.
            let mm_f16 = narrow(FloatDType::F16);

            let ms = |s: f64| s * 1e3;
            println!(
                "{label:22} {m:>6} {k:>6} {n:>6} | {:>8.1}ms {:>8.1}ms {:>8.1}ms {:>8.1}ms {:>6.2}ms {:>8.1}ms {:>8.1}ms",
                ms(dequant),
                ms(mm_perm),
                ms(mm_cont),
                ms(relayout),
                ms(alloc),
                ms(full),
                ms(no_flush)
            );
            // Bandwidth the dequant achieves writing the f32 transient —
            // the number that says whether the kernel or the memory is the
            // wall.
            let written = (n * k * 4) as f64 / 1e9;
            println!(
                "{:22} dequant writes {written:.2} GB at {:.1} GB/s · matmul {:.2} TFLOP/s · x {:.0} MB out {:.0} MB",
                "",
                written / dequant,
                2.0 * (m * k * n) as f64 / mm_perm / 1e12,
                (m * k * 4) as f64 / 1e6,
                (m * n * 4) as f64 / 1e6
            );
            let show = |t: Option<f64>| match t {
                Some(s) => format!(
                    "{:.1}ms ({:.2} TFLOP/s, {:.2}x)",
                    ms(s),
                    2.0 * (m * k * n) as f64 / s / 1e12,
                    mm_perm / s
                ),
                None => "unsupported".to_string(),
            };
            println!(
                "{:22} narrow operands, f32 accumulator: f16 {}",
                "",
                show(mm_f16)
            );
        }

        // Narrowing has a fixed cost per call — a reduction over the
        // activation, a scaling of it, and a scaling of the result — and
        // a win proportional to the weight traffic it halves. At a short
        // prompt the fixed cost wins and the narrow path is a loss. This
        // sweep, on a text model's linear shape, is where the crossover
        // gets measured instead of assumed.
        {
            let (k, n) = (2048usize, 2048usize);
            let data = synth_q8_0(n * k / 32);
            let w =
                QuantWeight::from_quant_tensor(&client, QuantFormat::Q8_0, &data, n, k).unwrap();
            drop(data);
            println!("\nnarrow-path crossover (k {k}, n {n}):");
            for m in [8usize, 16, 32, 64, 128, 256, 512, 1024] {
                let xv: Vec<f32> =
                    (0..m * k).map(|i| ((i % 37) as f32) / 18.0 - 1.0).collect();
                let x_h =
                    Tensor::<UnfusedF32, 2>::from_data(TensorData::new(xv, [m, k]), &device)
                        .into_primitive()
                        .tensor()
                        .handle;
                let arm = |operand: DType| {
                    time(
                        REPS,
                        &mut || {
                            let out = try_batched_matmul_with(
                                &w,
                                &mk(&x_h, [m, k]),
                                1,
                                m,
                                operand,
                            )
                            .expect("batched path engages");
                            force(&out.handle, m * n);
                        },
                    )
                };
                let wide = arm(DType::F32);
                let narrow = arm(DType::F16);
                println!(
                    "  m {m:>5} | f32 {:>7.2}ms | f16 {:>7.2}ms | {:.2}x{}",
                    wide * 1e3,
                    narrow * 1e3,
                    wide / narrow,
                    if narrow < wide { "" } else { "  <- narrow loses" }
                );
            }
        }

        // The 256-canvas anomaly: m=768 measured SLOWER than m=1536 at the
        // same weight. Sweep the row count at the DiT's fused shape to see
        // where the tuned routine changes its mind.
        let (k, n) = (3072usize, 27648usize);
        let data = synth_q8_0(n * k / 32);
        let w = QuantWeight::from_quant_tensor(&client, QuantFormat::Q8_0, &data, n, k).unwrap();
        drop(data);
        let w_h = w.dequant_device(&client).unwrap();
        force(&w_h, n * k);
        println!("\nrow sweep at the DiT fused shape (k {k}, n {n}):");
        for m in [512usize, 768, 1024, 1280, 1536, 2048] {
            let xv: Vec<f32> = (0..m * k).map(|i| ((i % 37) as f32) / 18.0 - 1.0).collect();
            let x_h = Tensor::<UnfusedF32, 2>::from_data(TensorData::new(xv, [m, k]), &device)
                .into_primitive()
                .tensor()
                .handle;
            let t = time(
                REPS,
                &mut || {
                    let out = matmul(
                        mk(&x_h, [m, k]),
                        permute(mk(&w_h, [n, k]), &[1, 0]),
                        None,
                        MatmulStrategy::default(),
                        DType::F32,
                    )
                    .expect("matmul launches");
                    force(&out.handle, m * n);
                },
            );
            println!(
                "  m {m:>5} | {:>8.1}ms | {:.2} TFLOP/s | {:>6.3}ms per row",
                t * 1e3,
                2.0 * (m * k * n) as f64 / t / 1e12,
                t * 1e3 / m as f64
            );
        }
    }
}
