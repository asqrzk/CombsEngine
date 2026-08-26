//! K3c: fused decode attention over the int8-quantized paged arena.
//!
//! With `COMBS_KV_QUANT=1` the global-layer arenas hold packed int8
//! words + group scales; until now every decode step gathered, ran the
//! elementwise `kv_dequantize`, and attended over a materialized f32
//! window. This dispatch reads the packed arenas in place and
//! dequantizes in-register — the quantized cache finally decodes like
//! the float one: one dispatch per layer.
//!
//! Paged mode only (quant arenas exist only for global layers), f32
//! backends only (the f16 build stores f16 scales — its fallback stays
//! the materialized path). Every refusal returns `None` into that path.

use std::any::TypeId;

use burn::backend::wgpu::{CubeTensor, WgpuRuntime};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Shape, Tensor, TensorPrimitive};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl_fusion::CubeFusionHandle;
use burn_fusion::stream::{Operation, OperationStreams};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr, TensorStatus};
use cubecl::prelude::CubeCount;

use super::decode_attn::{AttnGeom, attn_enabled};
use super::rmsnorm::cast_any;
use super::{DecodeAttnQ8, WORKGROUP, launch};
use crate::qlinear::{FusedF32, InnerF32, UnfusedF32};

/// The int8 arena for one of K or V: packed words + group scales.
pub(crate) struct QuantArena<B: Backend> {
    /// `[np, n_kv, page_size, d/4]`, one i32 word = 4 packed lanes.
    pub packed: Tensor<B, 4, Int>,
    /// `[np, n_kv, page_size, d/32]`, one f32 scale per 32 values.
    pub scales: Tensor<B, 4>,
}

/// Tries the fused decode-attention kernel against the QUANTIZED paged
/// arena. Geometry contract mirrors [`super::try_decode_attention`];
/// `d` must additionally be a multiple of 32 (the quant group), which
/// `kv_quantize`'s own gate already guarantees at the call site.
pub(crate) fn try_decode_attention_q8<B: Backend>(
    q: Tensor<B, 4>,
    k: QuantArena<B>,
    v: QuantArena<B>,
    table: Tensor<B, 1, Int>,
    total: usize,
    scale: f64,
) -> Option<Tensor<B, 4>> {
    if !attn_enabled() {
        return None;
    }
    let [_, n_q, seq, d] = q.dims();
    let [_, n_kv, page_size, dw] = k.packed.dims();
    if seq != 1
        || d > WORKGROUP as usize
        || d == 0
        || d % 32 != 0
        || dw != d / 4
        || n_kv == 0
        || n_q % n_kv != 0
        || total == 0
        || table.dims()[0] < total.div_ceil(page_size)
    {
        return None;
    }
    let geom = AttnGeom {
        n_q,
        n_kv,
        d,
        page_size,
        total,
        window: 0,
        mode: 0,
    };
    if TypeId::of::<B>() == TypeId::of::<UnfusedF32>() {
        let q: Tensor<UnfusedF32, 4> = cast_any(q)?;
        let kp: Tensor<UnfusedF32, 4, Int> = cast_any(k.packed)?;
        let ks: Tensor<UnfusedF32, 4> = cast_any(k.scales)?;
        let vp: Tensor<UnfusedF32, 4, Int> = cast_any(v.packed)?;
        let vs: Tensor<UnfusedF32, 4> = cast_any(v.scales)?;
        let t: Tensor<UnfusedF32, 1, Int> = cast_any(table)?;
        return cast_any(q8_unfused(q, kp, ks, vp, vs, t, geom, scale));
    }
    if TypeId::of::<B>() == TypeId::of::<FusedF32>() {
        let q: Tensor<FusedF32, 4> = cast_any(q)?;
        let kp: Tensor<FusedF32, 4, Int> = cast_any(k.packed)?;
        let ks: Tensor<FusedF32, 4> = cast_any(k.scales)?;
        let vp: Tensor<FusedF32, 4, Int> = cast_any(v.packed)?;
        let vs: Tensor<FusedF32, 4> = cast_any(v.scales)?;
        let t: Tensor<FusedF32, 1, Int> = cast_any(table)?;
        return cast_any(q8_fused(q, kp, ks, vp, vs, t, geom, scale));
    }
    None
}

/// The launch itself, shared by both backends once primitives are in hand.
#[allow(clippy::too_many_arguments)]
fn launch_q8(
    q: CubeTensor<WgpuRuntime>,
    kp: CubeTensor<WgpuRuntime>,
    ks: CubeTensor<WgpuRuntime>,
    vp: CubeTensor<WgpuRuntime>,
    vs: CubeTensor<WgpuRuntime>,
    table: CubeTensor<WgpuRuntime>,
    geom: AttnGeom,
    scale: f64,
) -> CubeTensor<WgpuRuntime> {
    let q = into_contiguous(q);
    let kp = into_contiguous(kp);
    let ks = into_contiguous(ks);
    let vp = into_contiguous(vp);
    let vs = into_contiguous(vs);
    let table = into_contiguous(table);
    let client = q.client.clone();
    let out = client.empty(geom.n_q * geom.d * core::mem::size_of::<f32>());
    launch(
        &client,
        DecodeAttnQ8,
        CubeCount::Static(geom.n_q as u32, 1, 1),
        vec![
            q.handle.binding(),
            kp.handle.binding(),
            ks.handle.binding(),
            vp.handle.binding(),
            vs.handle.binding(),
            table.handle.binding(),
            out.clone().binding(),
        ],
        geom.scalar_slots(scale),
    );
    CubeTensor::new_contiguous(
        client,
        q.device.clone(),
        Shape::from([1, geom.n_q, 1, geom.d]),
        out,
        DType::F32,
    )
}

#[allow(clippy::too_many_arguments)]
fn q8_unfused(
    q: Tensor<UnfusedF32, 4>,
    kp: Tensor<UnfusedF32, 4, Int>,
    ks: Tensor<UnfusedF32, 4>,
    vp: Tensor<UnfusedF32, 4, Int>,
    vs: Tensor<UnfusedF32, 4>,
    table: Tensor<UnfusedF32, 1, Int>,
    geom: AttnGeom,
    scale: f64,
) -> Tensor<UnfusedF32, 4> {
    let out = launch_q8(
        q.into_primitive().tensor(),
        kp.into_primitive(),
        ks.into_primitive().tensor(),
        vp.into_primitive(),
        vs.into_primitive().tensor(),
        table.into_primitive(),
        geom,
        scale,
    );
    Tensor::from_primitive(TensorPrimitive::Float(out))
}

/// The fusion-stream operation: six read-only inputs (four of them the
/// live arenas — the cache keeps its handles, registration must never
/// free or mutate them), one output.
struct DecodeAttnQ8Op {
    desc: CustomOpIr,
    geom: AttnGeom,
    scale: f64,
}

impl core::fmt::Debug for DecodeAttnQ8Op {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "DecodeAttnQ8Op {{ n_q: {}, n_kv: {}, d: {}, total: {} }}",
            self.geom.n_q, self.geom.n_kv, self.geom.d, self.geom.total
        )
    }
}

impl Operation<FusionCubeRuntime<WgpuRuntime>> for DecodeAttnQ8Op {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle<WgpuRuntime>>) {
        let ([q, kp, ks, vp, vs, t], [output]) = self.desc.as_fixed::<6, 1>();
        let qp: CubeTensor<WgpuRuntime> = handles.get_float_tensor::<InnerF32>(q);
        let kpp: CubeTensor<WgpuRuntime> = handles.get_int_tensor::<InnerF32>(kp);
        let ksp: CubeTensor<WgpuRuntime> = handles.get_float_tensor::<InnerF32>(ks);
        let vpp: CubeTensor<WgpuRuntime> = handles.get_int_tensor::<InnerF32>(vp);
        let vsp: CubeTensor<WgpuRuntime> = handles.get_float_tensor::<InnerF32>(vs);
        let tp: CubeTensor<WgpuRuntime> = handles.get_int_tensor::<InnerF32>(t);
        let out = launch_q8(qp, kpp, ksp, vpp, vsp, tp, self.geom, self.scale);
        handles.register_float_tensor::<InnerF32>(&output.id, out);
    }
}

#[allow(clippy::too_many_arguments)]
fn q8_fused(
    q: Tensor<FusedF32, 4>,
    kp: Tensor<FusedF32, 4, Int>,
    ks: Tensor<FusedF32, 4>,
    vp: Tensor<FusedF32, 4, Int>,
    vs: Tensor<FusedF32, 4>,
    table: Tensor<FusedF32, 1, Int>,
    geom: AttnGeom,
    scale: f64,
) -> Tensor<FusedF32, 4> {
    let qp = q.into_primitive().tensor();
    let kpp = kp.into_primitive();
    let ksp = ks.into_primitive().tensor();
    let vpp = vp.into_primitive();
    let vsp = vs.into_primitive().tensor();
    let tp = table.into_primitive();
    let client = qp.client.clone();

    let mut streams = OperationStreams::default();
    streams.tensor(&qp);
    streams.tensor(&kpp);
    streams.tensor(&ksp);
    streams.tensor(&vpp);
    streams.tensor(&vsp);
    streams.tensor(&tp);
    let q_ir = qp.into_ir();
    let kp_ir = kpp.into_ir();
    let ks_ir = ksp.into_ir();
    let vp_ir = vpp.into_ir();
    let vs_ir = vsp.into_ir();
    let t_ir = tp.into_ir();
    let out_ir = TensorIr {
        id: client.create_empty_handle(),
        shape: Shape::from([1, geom.n_q, 1, geom.d]),
        status: TensorStatus::NotInit,
        dtype: DType::F32,
    };
    let desc = CustomOpIr::new(
        "combs_decode_attn_q8",
        &[q_ir, kp_ir, ks_ir, vp_ir, vs_ir, t_ir],
        &[out_ir],
    );
    let op = DecodeAttnQ8Op {
        desc: desc.clone(),
        geom,
        scale,
    };
    let mut outputs = client.register(streams, OperationIr::Custom(desc), op);
    let out = outputs.pop().expect("custom op declares one output");
    Tensor::from_primitive(TensorPrimitive::Float(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    /// Packs four chosen int8 lanes exactly like `kv_quantize`: lanes
    /// 0..2 offset-binary, lane 3 signed in the top byte.
    fn pack(q0: i32, q1: i32, q2: i32, q3: i32) -> i32 {
        (q0 + 128) | ((q1 + 128) << 8) | ((q2 + 128) << 16) | (q3 << 24)
    }

    /// Grid fixture: chosen integer lanes and power-of-two scales, so the
    /// host dequant is exact and the kernel's unpack must match it
    /// bit-for-bit before any softmax arithmetic enters.
    #[allow(clippy::type_complexity)]
    fn fixture(
        n_kv: usize,
        total: usize,
        d: usize,
        page_size: usize,
        salt: i32,
    ) -> (Vec<i32>, Vec<f32>, Vec<Vec<f32>>, Vec<i32>, usize) {
        let lanes = [-127i32, -1, 0, 1, 63, 127, -64, 5];
        let scales = [0.5f32, 0.25, 1.0, 0.125];
        let pages = total.div_ceil(page_size);
        let num_pages = pages + 2;
        let table: Vec<i32> = (0..pages).map(|p| ((p + 1) % num_pages) as i32).collect();
        let dw = d / 4;
        let dg = d / 32;
        let mut packed = vec![0i32; num_pages * n_kv * page_size * dw];
        let mut scal = vec![0.0f32; num_pages * n_kv * page_size * dg];
        let mut dense = vec![vec![0.0f32; d]; n_kv * total];
        for j in 0..total {
            let phys = table[j / page_size] as usize;
            for g in 0..n_kv {
                let row = (phys * n_kv + g) * page_size + j % page_size;
                for grp in 0..dg {
                    let sc = scales[(j + g + grp + salt as usize) % scales.len()];
                    scal[row * dg + grp] = sc;
                    for w in 0..8 {
                        let pick = |l: usize| {
                            lanes[(j * 31 + g * 7 + grp * 5 + w * 3 + l + salt as usize)
                                % lanes.len()]
                        };
                        let (a, b, c, e) = (pick(0), pick(1), pick(2), pick(3));
                        packed[row * dw + grp * 8 + w] = pack(a, b, c, e);
                        let base = grp * 32 + w * 4;
                        let out = &mut dense[g * total + j];
                        out[base] = a as f32 * sc;
                        out[base + 1] = b as f32 * sc;
                        out[base + 2] = c as f32 * sc;
                        out[base + 3] = e as f32 * sc;
                    }
                }
            }
        }
        (packed, scal, dense, table, num_pages)
    }

    fn signal(len: usize, salt: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.437 + salt).sin() * 1.5) + ((i % 5) as f32 - 2.0) * 0.25)
            .collect()
    }

    /// Host softmax attention over the exactly-dequantized rows.
    fn reference(
        q: &[f32],
        keys: &[Vec<f32>],
        vals: &[Vec<f32>],
        n_q: usize,
        n_kv: usize,
        d: usize,
        scale: f32,
    ) -> Vec<f32> {
        let total = keys.len() / n_kv;
        let mut out = vec![0.0f32; n_q * d];
        for h in 0..n_q {
            let g = h / (n_q / n_kv);
            let qi = &q[h * d..(h + 1) * d];
            let scores: Vec<f32> = (0..total)
                .map(|j| {
                    qi.iter()
                        .zip(&keys[g * total + j])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let m = scores.iter().cloned().fold(f32::MIN, f32::max);
            let ps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let sum: f32 = ps.iter().sum();
            for (j, p) in ps.iter().enumerate() {
                for c in 0..d {
                    out[h * d + c] += p / sum * vals[g * total + j][c];
                }
            }
        }
        out
    }

    /// The harmony matrix: head dims incl. gemma3's 256, GQA ratios,
    /// totals straddling page and tile boundaries — kernel vs the host
    /// reference over exactly-representable packed data.
    #[test]
    fn q8_decode_attention_matches_the_reference() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        for &d in &[64usize, 128, 256] {
            for &n_rep in &[1usize, 2, 4] {
                for &total in &[1usize, 15, 17, 257, 300] {
                    let n_kv = 2usize;
                    let n_q = n_kv * n_rep;
                    let page_size = 16usize;
                    let scale = 1.0 / (d as f32).sqrt();
                    let (kp, ks, kd, table, np) = fixture(n_kv, total, d, page_size, 1);
                    let (vp, vs, vd, _, _) = fixture(n_kv, total, d, page_size, 9);
                    let qv = signal(n_q * d, 4.2);
                    let expect = reference(&qv, &kd, &vd, n_q, n_kv, d, scale);

                    let q: Tensor<UnfusedF32, 4> =
                        Tensor::from_data(TensorData::new(qv, [1, n_q, 1, d]), &device);
                    let mk = |p: Vec<i32>, s: Vec<f32>| QuantArena::<UnfusedF32> {
                        packed: Tensor::from_data(
                            TensorData::new(p, [np, n_kv, page_size, d / 4]),
                            &device,
                        ),
                        scales: Tensor::from_data(
                            TensorData::new(s, [np, n_kv, page_size, d / 32]),
                            &device,
                        ),
                    };
                    let len = table.len();
                    let t: Tensor<UnfusedF32, 1, Int> =
                        Tensor::from_data(TensorData::new(table, [len]), &device);
                    let got = try_decode_attention_q8(
                        q,
                        mk(kp, ks),
                        mk(vp, vs),
                        t,
                        total,
                        scale as f64,
                    )
                    .expect("geometry inside the kernel's envelope")
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                    for (i, (g, e)) in got.iter().zip(&expect).enumerate() {
                        let tol = 1e-3 * e.abs().max(1.0);
                        assert!(
                            (g - e).abs() <= tol,
                            "d={d} n_rep={n_rep} T={total} [{i}]: got {g}, expect {e}"
                        );
                    }
                }
            }
        }
    }

    /// Fused and unfused dispatch must agree bit-for-bit — six inputs,
    /// four of them long-lived arenas registered read-only.
    #[test]
    fn fused_dispatch_agrees_with_unfused_exactly() {
        if crate::skip_no_gpu() {
            return;
        }
        let (n_kv, n_q, d, total, page_size) = (2usize, 4usize, 64usize, 33usize, 16usize);
        let scale = 0.125f64;
        let (kp, ks, _, table, np) = fixture(n_kv, total, d, page_size, 3);
        let (vp, vs, _, _, _) = fixture(n_kv, total, d, page_size, 11);
        let qv = signal(n_q * d, 7.7);

        let run_unfused = {
            let device = Default::default();
            let q: Tensor<UnfusedF32, 4> =
                Tensor::from_data(TensorData::new(qv.clone(), [1, n_q, 1, d]), &device);
            let k = QuantArena::<UnfusedF32> {
                packed: Tensor::from_data(
                    TensorData::new(kp.clone(), [np, n_kv, page_size, d / 4]),
                    &device,
                ),
                scales: Tensor::from_data(
                    TensorData::new(ks.clone(), [np, n_kv, page_size, d / 32]),
                    &device,
                ),
            };
            let v = QuantArena::<UnfusedF32> {
                packed: Tensor::from_data(
                    TensorData::new(vp.clone(), [np, n_kv, page_size, d / 4]),
                    &device,
                ),
                scales: Tensor::from_data(
                    TensorData::new(vs.clone(), [np, n_kv, page_size, d / 32]),
                    &device,
                ),
            };
            let len = table.len();
            let t: Tensor<UnfusedF32, 1, Int> =
                Tensor::from_data(TensorData::new(table.clone(), [len]), &device);
            try_decode_attention_q8(q, k, v, t, total, scale)
                .expect("kernel path")
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        };
        let run_fused = {
            let device = Default::default();
            let q: Tensor<FusedF32, 4> =
                Tensor::from_data(TensorData::new(qv, [1, n_q, 1, d]), &device);
            let k = QuantArena::<FusedF32> {
                packed: Tensor::from_data(
                    TensorData::new(kp, [np, n_kv, page_size, d / 4]),
                    &device,
                ),
                scales: Tensor::from_data(
                    TensorData::new(ks, [np, n_kv, page_size, d / 32]),
                    &device,
                ),
            };
            let v = QuantArena::<FusedF32> {
                packed: Tensor::from_data(
                    TensorData::new(vp, [np, n_kv, page_size, d / 4]),
                    &device,
                ),
                scales: Tensor::from_data(
                    TensorData::new(vs, [np, n_kv, page_size, d / 32]),
                    &device,
                ),
            };
            let len = table.len();
            let t: Tensor<FusedF32, 1, Int> =
                Tensor::from_data(TensorData::new(table, [len]), &device);
            try_decode_attention_q8(q, k, v, t, total, scale)
                .expect("kernel path")
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        };
        assert_eq!(run_unfused, run_fused, "one kernel, two dispatch routes");
    }
}
