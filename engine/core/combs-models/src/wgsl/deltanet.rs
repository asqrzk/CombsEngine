//! The Gated DeltaNet decode step as two WGSL dispatches.
//!
//! On the generic road a DeltaNet layer costs ~69 launches a token (the
//! conv, the norms, the gates, the rank-1 update, the readout, each a burn
//! op): Qwen3.5-4B's 24 such layers put it at 2,801 dispatches a token
//! against qwen3-4b's 1,151, and at 15 µs a launch the count is the wall.
//! Here the conv step and the state step are one dispatch each.
//!
//! Decode only (seq = 1): prefill keeps the host road. The model asks
//! [`deltanet_step_available`] BEFORE it projects or takes the state, so a
//! refusal costs nothing and loses nothing. Every buffer is f32 on every
//! build: the conv ring and the state are recurrent memory and never narrow,
//! so the f16 backend casts its projection outputs up around the launch.
//! The kernels read the old ring and state and write new buffers — the house
//! never writes into a live tensor's buffer.

use std::any::TypeId;

use burn::backend::wgpu::{CubeBackend, CubeTensor, FloatElement, WgpuRuntime};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Shape, Tensor, TensorPrimitive};
use burn_cubecl::kernel::into_contiguous;
use cubecl::prelude::CubeCount;

use super::rmsnorm::cast_any;
use super::{DeltanetConvStep, DeltanetStateStep, WORKGROUP, as_f32, launch, wgsl_enabled};
use crate::qlinear::{UnfusedF16, UnfusedF32};

/// The state shader's per-invocation scratch is 256 wide.
const MAX_HEAD_DIM: usize = 256;

/// Per-kernel door on top of the master `COMBS_WGSL`; read once.
pub(crate) fn deltanet_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        wgsl_enabled() && !matches!(std::env::var("COMBS_WGSL_DELTANET").as_deref(), Ok("0"))
    })
}

/// One layer's geometry, as the file declares it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StepGeometry {
    pub kh: usize,
    pub vh: usize,
    pub kd: usize,
    pub vd: usize,
    pub kernel: usize,
    pub rms_eps: f32,
    pub l2_eps: f32,
}

impl StepGeometry {
    /// The q|k|v stream's width.
    pub(crate) fn channels(&self) -> usize {
        2 * self.kh * self.kd + self.vh * self.vd
    }
}

/// One token's inputs, flattened in the host road's row order.
pub(crate) struct StepInputs<B: Backend> {
    /// `[channels]`
    pub qkv: Tensor<B, 1>,
    /// `[vh * vd]`
    pub z: Tensor<B, 1>,
    /// `[vh]` each: beta logits, decay alpha, -exp(A_log), dt bias.
    pub b: Tensor<B, 1>,
    pub a: Tensor<B, 1>,
    pub decay_a: Tensor<B, 1>,
    pub dt: Tensor<B, 1>,
    /// `[vd]`
    pub normw: Tensor<B, 1>,
    /// `[channels * kernel]`
    pub conv_w: Tensor<B, 1>,
    /// `[channels * (kernel - 1)]`
    pub ring: Tensor<B, 1>,
    /// `[vh * kd * vd]`
    pub state: Tensor<B, 1>,
}

/// `(out [vh * vd], ring', state')`, all f32.
pub(crate) type StepOutputs<B> = (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>);

/// Whether [`try_deltanet_step`] will run for this backend and geometry —
/// the same predicate it routes on, asked first so a refusal is free.
pub(crate) fn deltanet_step_available<B: Backend>(g: &StepGeometry) -> bool {
    deltanet_enabled()
        && g.kh > 0
        && g.vh % g.kh == 0
        && g.kd <= MAX_HEAD_DIM
        && g.vd <= MAX_HEAD_DIM
        && g.kernel > 1
        && g.channels().div_ceil(WORKGROUP as usize) <= 65_535
        // The fused f32 build would need a fusion custom op; until one is
        // written it keeps the host road, correct and slower.
        && (TypeId::of::<B>() == TypeId::of::<UnfusedF32>()
            || TypeId::of::<B>() == TypeId::of::<UnfusedF16>())
}

pub(crate) fn try_deltanet_step<B: Backend>(
    inputs: StepInputs<B>,
    g: StepGeometry,
) -> Option<StepOutputs<B>> {
    if !deltanet_step_available::<B>(&g) {
        return None;
    }
    if TypeId::of::<B>() == TypeId::of::<UnfusedF32>() {
        return cast_outputs(step_unfused(cast_inputs::<B, UnfusedF32>(inputs)?, g));
    }
    cast_outputs(step_unfused(cast_inputs::<B, UnfusedF16>(inputs)?, g))
}

fn cast_inputs<From: Backend, To: Backend>(i: StepInputs<From>) -> Option<StepInputs<To>> {
    Some(StepInputs {
        qkv: cast_any(i.qkv)?,
        z: cast_any(i.z)?,
        b: cast_any(i.b)?,
        a: cast_any(i.a)?,
        decay_a: cast_any(i.decay_a)?,
        dt: cast_any(i.dt)?,
        normw: cast_any(i.normw)?,
        conv_w: cast_any(i.conv_w)?,
        ring: cast_any(i.ring)?,
        state: cast_any(i.state)?,
    })
}

fn cast_outputs<From: Backend, To: Backend>(o: StepOutputs<From>) -> Option<StepOutputs<To>> {
    Some((cast_any(o.0)?, cast_any(o.1)?, cast_any(o.2)?))
}

fn step_unfused<F: FloatElement>(
    i: StepInputs<CubeBackend<WgpuRuntime, F, i32, u32>>,
    g: StepGeometry,
) -> StepOutputs<CubeBackend<WgpuRuntime, F, i32, u32>> {
    let prim = |t: Tensor<CubeBackend<WgpuRuntime, F, i32, u32>, 1>| {
        into_contiguous(as_f32(t).into_primitive().tensor())
    };
    let qkv = prim(i.qkv);
    let (z, b, a) = (prim(i.z), prim(i.b), prim(i.a));
    let (decay_a, dt, normw) = (prim(i.decay_a), prim(i.dt), prim(i.normw));
    let (conv_w, ring, state) = (prim(i.conv_w), prim(i.ring), prim(i.state));
    let client = qkv.client.clone();
    let device = qkv.device.clone();
    let f = core::mem::size_of::<f32>();
    let channels = g.channels();
    let hist = g.kernel - 1;
    let wg = WORKGROUP as usize;

    let conv_out = client.empty(channels * f);
    let ring_out = client.empty(channels * hist * f);
    launch(
        &client,
        DeltanetConvStep,
        CubeCount::Static(channels.div_ceil(wg) as u32, 1, 1),
        vec![
            qkv.handle.binding(),
            conv_w.handle.binding(),
            ring.handle.binding(),
            ring_out.clone().binding(),
            conv_out.clone().binding(),
        ],
        vec![channels as u64, g.kernel as u64],
    );

    let state_len = g.vh * g.kd * g.vd;
    let state_out = client.empty(state_len * f);
    let out = client.empty(g.vh * g.vd * f);
    launch(
        &client,
        DeltanetStateStep,
        CubeCount::Static(g.vh.div_ceil(wg) as u32, 1, 1),
        vec![
            conv_out.binding(),
            z.handle.binding(),
            b.handle.binding(),
            a.handle.binding(),
            decay_a.handle.binding(),
            dt.handle.binding(),
            normw.handle.binding(),
            state.handle.binding(),
            state_out.clone().binding(),
            out.clone().binding(),
        ],
        vec![
            g.kh as u64,
            g.vh as u64,
            g.kd as u64,
            g.vd as u64,
            f32::to_bits(g.rms_eps) as u64,
            f32::to_bits(g.l2_eps) as u64,
        ],
    );

    let wrap = |handle, n: usize| {
        Tensor::from_primitive(TensorPrimitive::Float(CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            Shape::from(vec![n]),
            handle,
            DType::F32,
        )))
    };
    (
        wrap(out, g.vh * g.vd),
        wrap(ring_out, channels * hist),
        wrap(state_out, state_len),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    /// The step in plain f32 loops, row for row the host road in qwen35.rs
    /// at seq = 1: the reference the kernels answer to.
    #[allow(clippy::too_many_arguments)]
    fn host_step(
        g: &StepGeometry,
        qkv: &[f32],
        z: &[f32],
        b: &[f32],
        a: &[f32],
        decay_a: &[f32],
        dt: &[f32],
        normw: &[f32],
        conv_w: &[f32],
        ring: &[f32],
        state: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (ch, hist) = (g.channels(), g.kernel - 1);
        let mut conv = vec![0f32; ch];
        let mut ring2 = vec![0f32; ch * hist];
        for c in 0..ch {
            let mut acc = conv_w[c * g.kernel + hist] * qkv[c];
            for j in 0..hist {
                acc += conv_w[c * g.kernel + j] * ring[c * hist + j];
            }
            conv[c] = acc / (1.0 + (-acc).exp());
            for j in 0..hist - 1 {
                ring2[c * hist + j] = ring[c * hist + j + 1];
            }
            ring2[c * hist + hist - 1] = qkv[c];
        }
        let key_dim = g.kh * g.kd;
        let mut out = vec![0f32; g.vh * g.vd];
        let mut state2 = vec![0f32; state.len()];
        for h in 0..g.vh {
            let gk = h % g.kh;
            let q = &conv[gk * g.kd..(gk + 1) * g.kd];
            let k = &conv[key_dim + gk * g.kd..key_dim + (gk + 1) * g.kd];
            let v = &conv[2 * key_dim + h * g.vd..2 * key_dim + (h + 1) * g.vd];
            let qs: f32 = q.iter().map(|x| x * x).sum();
            let ks: f32 = k.iter().map(|x| x * x).sum();
            let qn: Vec<f32> = q.iter().map(|x| x / (qs + g.l2_eps).sqrt() / (g.kd as f32).sqrt()).collect();
            let kn: Vec<f32> = k.iter().map(|x| x / (ks + g.l2_eps).sqrt()).collect();
            let beta = 1.0 / (1.0 + (-b[h]).exp());
            let x = a[h] + dt[h];
            let decay = (decay_a[h] * (x.max(0.0) + (-x.abs()).exp().ln_1p())).exp();
            let sb = h * g.kd * g.vd;
            let mut recalled = vec![0f32; g.vd];
            for i in 0..g.kd {
                for j in 0..g.vd {
                    recalled[j] += kn[i] * state[sb + i * g.vd + j] * decay;
                }
            }
            let mut o = vec![0f32; g.vd];
            for i in 0..g.kd {
                for j in 0..g.vd {
                    let s = state[sb + i * g.vd + j] * decay + kn[i] * (v[j] - recalled[j]) * beta;
                    state2[sb + i * g.vd + j] = s;
                    o[j] += qn[i] * s;
                }
            }
            let ms: f32 = o.iter().map(|x| x * x).sum::<f32>() / g.vd as f32;
            let inv = 1.0 / (ms + g.rms_eps).sqrt();
            for j in 0..g.vd {
                let gate = z[h * g.vd + j];
                out[h * g.vd + j] = o[j] * inv * normw[j] * (gate / (1.0 + (-gate).exp()));
            }
        }
        (out, ring2, state2)
    }

    fn wave(n: usize, seed: f32, scale: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 * 0.37 + seed).sin() * scale)).collect()
    }

    /// Relative tolerance: f32 arithmetic on both sides, the same operation
    /// order in every sum (≤ 16 terms), and exp/log/sqrt from two libm's —
    /// agreement near 1e-6; a wrong index, gate or head order lands near 1e0.
    const TOL: f32 = 1e-4;

    fn close(what: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                g.is_finite() && (g - w).abs() <= TOL * w.abs().max(1.0),
                "{what}[{i}]: got {g}, expected {w}"
            );
        }
    }

    fn max_rel(got: &[f32], want: &[f32]) -> f32 {
        got.iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs() / w.abs().max(1.0))
            .fold(0.0, f32::max)
    }

    fn check_on<B: Backend>(device: &B::Device, f16_backend: bool) {
        let g = StepGeometry { kh: 2, vh: 4, kd: 8, vd: 8, kernel: 4, rms_eps: 1e-6, l2_eps: 1e-6 };
        let ch = g.channels();
        // On an f16 backend the projections are rounded on the HOST first and
        // those exact values are both uploaded and fed to the reference, so
        // no second rounding (burn's, at upload) can differ from the one the
        // reference assumed. The recurrent memory is f32 on every backend.
        let round = |v: Vec<f32>| -> Vec<f32> {
            if f16_backend {
                v.iter().map(|x| f32::from(burn::tensor::f16::from_f32(*x))).collect()
            } else {
                v
            }
        };
        let qkv = round(wave(ch, 0.1, 1.3));
        let z = round(wave(g.vh * g.vd, 0.7, 1.1));
        let b = round(wave(g.vh, 1.9, 2.0));
        let a = round(wave(g.vh, 2.3, 1.5));
        let decay_a = round(wave(g.vh, 3.1, 1.0).iter().map(|x| -(x.abs() + 0.05)).collect());
        let dt = round(wave(g.vh, 4.2, 3.0));
        let normw = round(wave(g.vd, 5.0, 0.3).iter().map(|x| 1.0 + x).collect());
        let conv_w = round(wave(ch * g.kernel, 6.1, 0.6));
        let ring = wave(ch * (g.kernel - 1), 7.3, 0.9);
        let state = wave(g.vh * g.kd * g.vd, 8.4, 0.5);
        let (want_out, want_ring, want_state) =
            host_step(&g, &qkv, &z, &b, &a, &decay_a, &dt, &normw, &conv_w, &ring, &state);

        let proj = |v: &[f32]| Tensor::<B, 1>::from_data(TensorData::new(v.to_vec(), [v.len()]), device);
        let f32s = |v: &[f32]| {
            Tensor::<B, 1>::from_data(TensorData::new(v.to_vec(), [v.len()]), (device, DType::F32))
        };
        assert!(deltanet_step_available::<B>(&g), "the door is shut on this backend");
        let inputs = StepInputs {
            qkv: proj(&qkv),
            z: proj(&z),
            b: proj(&b),
            a: proj(&a),
            decay_a: proj(&decay_a),
            dt: proj(&dt),
            normw: proj(&normw),
            conv_w: proj(&conv_w),
            ring: f32s(&ring),
            state: f32s(&state),
        };
        let (out, ring2, state2) =
            try_deltanet_step(inputs, g).expect("the step runs where it is available");
        let read = |t: Tensor<B, 1>| t.into_data().convert::<f32>().to_vec::<f32>().expect("readback");
        let (out, ring2, state2) = (read(out), read(ring2), read(state2));
        let arm = if f16_backend { "f16 backend" } else { "f32 backend" };
        println!(
            "{arm}: max rel out {:.2e} · ring {:.2e} · state {:.2e}",
            max_rel(&out, &want_out),
            max_rel(&ring2, &want_ring),
            max_rel(&state2, &want_state)
        );
        close(&format!("out ({arm})"), &out, &want_out);
        close(&format!("ring ({arm})"), &ring2, &want_ring);
        close(&format!("state ({arm})"), &state2, &want_state);
    }

    #[test]
    fn deltanet_step_matches_the_host_road() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = burn::backend::wgpu::WgpuDevice::default();
        check_on::<UnfusedF32>(&device, false);
        check_on::<UnfusedF16>(&device, true);
    }
}
