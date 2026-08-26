//! K3a: fused decode attention over the paged arena.
//!
//! For the seq == 1 step this replaces the whole gather-window +
//! repeat_kv + attend chain with one dispatch per layer that reads K/V
//! in place — the page table provides the indirection the gather used to
//! materialize. GQA never expands: each query head walks its kv head's
//! rows directly.
//!
//! The kernel also implements a contiguous mode (`mode = 1`) for the
//! sliding/rolling stores; the dispatcher here wires the paged mode, and
//! Stage 4 (K3b) opens the second one.
//!
//! Fallback discipline as everywhere: doors, foreign backends, f16,
//! d > 256, or any geometry surprise → `None`, and the caller runs the
//! materialized path unchanged.

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

use super::rmsnorm::cast_any;
use super::{DecodeAttn, WORKGROUP, launch, wgsl_enabled};
use crate::qlinear::{FusedF32, InnerF32, UnfusedF32};

/// Per-kernel door on top of the master `COMBS_WGSL`; read once.
pub(super) fn attn_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        wgsl_enabled() && !matches!(std::env::var("COMBS_WGSL_ATTN").as_deref(), Ok("0"))
    })
}

/// The geometry one dispatch needs, resolved host-side once. Shared
/// with the q8 sibling, whose Params layout is identical by design.
#[derive(Clone, Copy)]
pub(super) struct AttnGeom {
    pub(super) n_q: usize,
    pub(super) n_kv: usize,
    pub(super) d: usize,
    pub(super) page_size: usize,
    pub(super) total: usize,
    pub(super) window: usize,
    pub(super) mode: usize,
}

impl AttnGeom {
    pub(super) fn scalar_slots(self, scale: f64) -> Vec<u64> {
        vec![
            self.n_q as u64,
            self.n_kv as u64,
            self.d as u64,
            self.page_size as u64,
            self.total as u64,
            self.window as u64,
            self.mode as u64,
            f32::to_bits(scale as f32) as u64,
        ]
    }
}

/// Tries the fused decode-attention kernel against the paged arena.
///
/// `q` is `[1, n_q, 1, d]`; the arenas are `[num_pages, n_kv, page_size,
/// d]`; `table` maps the first `ceil(total / page_size)` logical pages to
/// physical ids. `window == 0` means global visibility.
pub(crate) fn try_decode_attention<B: Backend>(
    q: Tensor<B, 4>,
    k_arena: Tensor<B, 4>,
    v_arena: Tensor<B, 4>,
    table: Tensor<B, 1, Int>,
    total: usize,
    window: usize,
    scale: f64,
) -> Option<Tensor<B, 4>> {
    if !attn_enabled() {
        return None;
    }
    let [_, n_q, seq, d] = q.dims();
    let [_, n_kv, page_size, arena_d] = k_arena.dims();
    // Every lane owns one output column, so the head dim must fit one
    // workgroup; the other guards are geometry that should never happen.
    if seq != 1
        || d > WORKGROUP as usize
        || d == 0
        || d % 4 != 0
        || arena_d != d
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
        window,
        mode: 0,
    };
    if TypeId::of::<B>() == TypeId::of::<UnfusedF32>() {
        let q: Tensor<UnfusedF32, 4> = cast_any(q)?;
        let k: Tensor<UnfusedF32, 4> = cast_any(k_arena)?;
        let v: Tensor<UnfusedF32, 4> = cast_any(v_arena)?;
        let t: Tensor<UnfusedF32, 1, Int> = cast_any(table)?;
        return cast_any(attn_unfused(q, k, v, t, geom, scale));
    }
    if TypeId::of::<B>() == TypeId::of::<FusedF32>() {
        let q: Tensor<FusedF32, 4> = cast_any(q)?;
        let k: Tensor<FusedF32, 4> = cast_any(k_arena)?;
        let v: Tensor<FusedF32, 4> = cast_any(v_arena)?;
        let t: Tensor<FusedF32, 1, Int> = cast_any(table)?;
        return cast_any(attn_fused(q, k, v, t, geom, scale));
    }
    None
}

/// K3b: the same kernel in contiguous mode, for the sliding/rolling
/// stores — `k`/`v` are the concatenated `[1, n_kv, total, d]` window and
/// key j lives at `(g·total + j)·d`. The page table is a one-element
/// dummy (the binding is always present; mode 1 never reads it).
pub(crate) fn try_sliding_decode_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    window: usize,
    scale: f64,
) -> Option<Tensor<B, 4>> {
    if !attn_enabled() {
        return None;
    }
    let [_, n_q, seq, d] = q.dims();
    let [_, n_kv, total, arena_d] = k.dims();
    if seq != 1
        || d > WORKGROUP as usize
        || d == 0
        || d % 4 != 0
        || arena_d != d
        || n_kv == 0
        || n_q % n_kv != 0
        || total == 0
    {
        return None;
    }
    let geom = AttnGeom {
        n_q,
        n_kv,
        d,
        page_size: 1,
        total,
        window,
        mode: 1,
    };
    if TypeId::of::<B>() == TypeId::of::<UnfusedF32>() {
        let q: Tensor<UnfusedF32, 4> = cast_any(q)?;
        let k: Tensor<UnfusedF32, 4> = cast_any(k)?;
        let v: Tensor<UnfusedF32, 4> = cast_any(v)?;
        let t = Tensor::<UnfusedF32, 1, Int>::zeros([1], &q.device());
        return cast_any(attn_unfused(q, k, v, t, geom, scale));
    }
    if TypeId::of::<B>() == TypeId::of::<FusedF32>() {
        let q: Tensor<FusedF32, 4> = cast_any(q)?;
        let k: Tensor<FusedF32, 4> = cast_any(k)?;
        let v: Tensor<FusedF32, 4> = cast_any(v)?;
        let t = Tensor::<FusedF32, 1, Int>::zeros([1], &q.device());
        return cast_any(attn_fused(q, k, v, t, geom, scale));
    }
    None
}

/// The launch itself, shared by both backends once primitives are in hand.
fn launch_attn(
    q: CubeTensor<WgpuRuntime>,
    k: CubeTensor<WgpuRuntime>,
    v: CubeTensor<WgpuRuntime>,
    table: CubeTensor<WgpuRuntime>,
    geom: AttnGeom,
    scale: f64,
) -> CubeTensor<WgpuRuntime> {
    let q = into_contiguous(q);
    let k = into_contiguous(k);
    let v = into_contiguous(v);
    let table = into_contiguous(table);
    let client = q.client.clone();
    let out = client.empty(geom.n_q * geom.d * core::mem::size_of::<f32>());
    launch(
        &client,
        DecodeAttn,
        CubeCount::Static(geom.n_q as u32, 1, 1),
        vec![
            q.handle.binding(),
            k.handle.binding(),
            v.handle.binding(),
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

fn attn_unfused(
    q: Tensor<UnfusedF32, 4>,
    k: Tensor<UnfusedF32, 4>,
    v: Tensor<UnfusedF32, 4>,
    table: Tensor<UnfusedF32, 1, Int>,
    geom: AttnGeom,
    scale: f64,
) -> Tensor<UnfusedF32, 4> {
    let out = launch_attn(
        q.into_primitive().tensor(),
        k.into_primitive().tensor(),
        v.into_primitive().tensor(),
        table.into_primitive(),
        geom,
        scale,
    );
    Tensor::from_primitive(TensorPrimitive::Float(out))
}

/// The fusion-stream operation: q, both arenas and the page table resolve
/// to device tensors when the stream drains. The arenas arrive as
/// read-only inputs — the cache keeps its handles alive, so registering
/// them must never free or mutate them (the interleave test stands guard).
struct DecodeAttnOp {
    desc: CustomOpIr,
    geom: AttnGeom,
    scale: f64,
}

impl core::fmt::Debug for DecodeAttnOp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "DecodeAttnOp {{ n_q: {}, n_kv: {}, d: {}, total: {} }}",
            self.geom.n_q, self.geom.n_kv, self.geom.d, self.geom.total
        )
    }
}

impl Operation<FusionCubeRuntime<WgpuRuntime>> for DecodeAttnOp {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle<WgpuRuntime>>) {
        let ([q, k, v, t], [output]) = self.desc.as_fixed::<4, 1>();
        let qp: CubeTensor<WgpuRuntime> = handles.get_float_tensor::<InnerF32>(q);
        let kp: CubeTensor<WgpuRuntime> = handles.get_float_tensor::<InnerF32>(k);
        let vp: CubeTensor<WgpuRuntime> = handles.get_float_tensor::<InnerF32>(v);
        let tp: CubeTensor<WgpuRuntime> = handles.get_int_tensor::<InnerF32>(t);
        let out = launch_attn(qp, kp, vp, tp, self.geom, self.scale);
        handles.register_float_tensor::<InnerF32>(&output.id, out);
    }
}

fn attn_fused(
    q: Tensor<FusedF32, 4>,
    k: Tensor<FusedF32, 4>,
    v: Tensor<FusedF32, 4>,
    table: Tensor<FusedF32, 1, Int>,
    geom: AttnGeom,
    scale: f64,
) -> Tensor<FusedF32, 4> {
    let qp = q.into_primitive().tensor();
    let kp = k.into_primitive().tensor();
    let vp = v.into_primitive().tensor();
    let tp = table.into_primitive();
    let client = qp.client.clone();

    let mut streams = OperationStreams::default();
    streams.tensor(&qp);
    streams.tensor(&kp);
    streams.tensor(&vp);
    streams.tensor(&tp);
    let q_ir = qp.into_ir();
    let k_ir = kp.into_ir();
    let v_ir = vp.into_ir();
    let t_ir = tp.into_ir();
    let out_ir = TensorIr {
        id: client.create_empty_handle(),
        shape: Shape::from([1, geom.n_q, 1, geom.d]),
        status: TensorStatus::NotInit,
        dtype: DType::F32,
    };
    let desc = CustomOpIr::new("combs_decode_attn", &[q_ir, k_ir, v_ir, t_ir], &[out_ir]);
    let op = DecodeAttnOp {
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

    /// Deterministic mixed-sign data, distinct per salt.
    fn signal(len: usize, salt: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.437 + salt).sin() * 1.5) + ((i % 5) as f32 - 2.0) * 0.25)
            .collect()
    }

    /// Host reference: single-query attention over the logical K/V rows,
    /// plain f32 softmax — the same arithmetic kv.rs::attend runs, without
    /// its tensor plumbing.
    fn reference(
        q: &[f32],
        keys: &[Vec<f32>],
        vals: &[Vec<f32>],
        n_q: usize,
        n_kv: usize,
        d: usize,
        window: usize,
        scale: f32,
    ) -> Vec<f32> {
        let total = keys.len() / n_kv;
        let mut out = vec![0.0f32; n_q * d];
        for h in 0..n_q {
            let g = h / (n_q / n_kv);
            let qi = &q[h * d..(h + 1) * d];
            let visible: Vec<usize> = (0..total)
                .filter(|&j| window == 0 || j + window >= total)
                .collect();
            let scores: Vec<f32> = visible
                .iter()
                .map(|&j| {
                    let kj = &keys[g * total + j];
                    qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let m = scores.iter().cloned().fold(f32::MIN, f32::max);
            let ps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let sum: f32 = ps.iter().sum();
            for (idx, &j) in visible.iter().enumerate() {
                let vj = &vals[g * total + j];
                for c in 0..d {
                    out[h * d + c] += ps[idx] / sum * vj[c];
                }
            }
        }
        out
    }

    /// Builds a paged arena + shuffled table holding the logical rows, and
    /// returns (arena_k, arena_v, table) as flat host vectors.
    #[allow(clippy::type_complexity)]
    fn paged_layout(
        keys: &[Vec<f32>],
        vals: &[Vec<f32>],
        n_kv: usize,
        d: usize,
        total: usize,
        page_size: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<i32>, usize) {
        let pages = total.div_ceil(page_size);
        let num_pages = pages + 3;
        // A non-identity permutation: logical page p lives at physical
        // (p * 5 + 2) mod num_pages — collision-free when gcd(5, n) = 1,
        // which holding num_pages away from multiples of 5 guarantees.
        let num_pages = if num_pages % 5 == 0 { num_pages + 1 } else { num_pages };
        let table: Vec<i32> = (0..pages).map(|p| ((p * 5 + 2) % num_pages) as i32).collect();
        let mut ak = vec![0.0f32; num_pages * n_kv * page_size * d];
        let mut av = vec![0.0f32; num_pages * n_kv * page_size * d];
        for j in 0..total {
            let phys = table[j / page_size] as usize;
            let slot = j % page_size;
            for g in 0..n_kv {
                let base = ((phys * n_kv + g) * page_size + slot) * d;
                ak[base..base + d].copy_from_slice(&keys[g * total + j]);
                av[base..base + d].copy_from_slice(&vals[g * total + j]);
            }
        }
        (ak, av, table, num_pages)
    }

    fn assert_close(got: &[f32], expect: &[f32], rel: f32, what: &str) {
        assert_eq!(got.len(), expect.len(), "{what}: length");
        for (i, (g, e)) in got.iter().zip(expect).enumerate() {
            let tol = rel * e.abs().max(1.0);
            assert!((g - e).abs() <= tol, "{what}[{i}]: got {g}, expect {e}");
        }
    }

    /// The plan's harmony matrix: head dims, GQA ratios, totals straddling
    /// page and tile boundaries, windows, and a non-default scale — every
    /// case against the host reference, on the unfused backend.
    #[test]
    fn decode_attention_matches_the_reference_across_the_matrix() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        for &d in &[64usize, 96, 128, 256] {
            for &n_rep in &[1usize, 2, 3, 7] {
                for &total in &[1usize, 15, 16, 17, 33, 257] {
                    for &window in &[0usize, 5, 512] {
                        for &scale in &[1.0 / (d as f32).sqrt(), 1.0 / 16.0] {
                            let n_kv = 2usize;
                            let n_q = n_kv * n_rep;
                            let page_size = 16usize;
                            let keys: Vec<Vec<f32>> = (0..n_kv * total)
                                .map(|i| signal(d, i as f32 * 0.71))
                                .collect();
                            let vals: Vec<Vec<f32>> = (0..n_kv * total)
                                .map(|i| signal(d, i as f32 * 0.71 + 90.0))
                                .collect();
                            let q = signal(n_q * d, 42.5);
                            let expect =
                                reference(&q, &keys, &vals, n_q, n_kv, d, window, scale);

                            let (ak, av, table, num_pages) =
                                paged_layout(&keys, &vals, n_kv, d, total, page_size);
                            let qt: Tensor<UnfusedF32, 4> = Tensor::from_data(
                                TensorData::new(q, [1, n_q, 1, d]),
                                &device,
                            );
                            let kt: Tensor<UnfusedF32, 4> = Tensor::from_data(
                                TensorData::new(ak, [num_pages, n_kv, page_size, d]),
                                &device,
                            );
                            let vt: Tensor<UnfusedF32, 4> = Tensor::from_data(
                                TensorData::new(av, [num_pages, n_kv, page_size, d]),
                                &device,
                            );
                            let len = table.len();
                            let tt: Tensor<UnfusedF32, 1, Int> = Tensor::from_data(
                                TensorData::new(table, [len]),
                                &device,
                            );
                            let got = try_decode_attention(
                                qt,
                                kt,
                                vt,
                                tt,
                                total,
                                window,
                                scale as f64,
                            )
                            .expect("geometry inside the kernel's envelope")
                            .into_data()
                            .to_vec::<f32>()
                            .unwrap();
                            assert_close(
                                &got,
                                &expect,
                                1e-3,
                                &format!("d={d} n_rep={n_rep} T={total} w={window} s={scale}"),
                            );
                        }
                    }
                }
            }
        }
    }

    /// Contiguous mode (K3b) against the same reference: the rolling-store
    /// layout, windows on and off, totals straddling the tile boundary.
    #[test]
    fn contiguous_mode_matches_the_reference() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        for &d in &[64usize, 128] {
            for &total in &[1usize, 15, 257, 300] {
                for &window in &[0usize, 5, 256] {
                    let (n_kv, n_rep) = (2usize, 3usize);
                    let n_q = n_kv * n_rep;
                    let scale = 1.0 / (d as f32).sqrt();
                    let keys: Vec<Vec<f32>> = (0..n_kv * total)
                        .map(|i| signal(d, i as f32 * 0.31))
                        .collect();
                    let vals: Vec<Vec<f32>> = (0..n_kv * total)
                        .map(|i| signal(d, i as f32 * 0.31 + 70.0))
                        .collect();
                    let q = signal(n_q * d, 8.8);
                    let expect = reference(&q, &keys, &vals, n_q, n_kv, d, window, scale);

                    let flat_k: Vec<f32> = keys.iter().flatten().copied().collect();
                    let flat_v: Vec<f32> = vals.iter().flatten().copied().collect();
                    let qt: Tensor<UnfusedF32, 4> =
                        Tensor::from_data(TensorData::new(q, [1, n_q, 1, d]), &device);
                    let kt: Tensor<UnfusedF32, 4> = Tensor::from_data(
                        TensorData::new(flat_k, [1, n_kv, total, d]),
                        &device,
                    );
                    let vt: Tensor<UnfusedF32, 4> = Tensor::from_data(
                        TensorData::new(flat_v, [1, n_kv, total, d]),
                        &device,
                    );
                    let got = try_sliding_decode_attention(qt, kt, vt, window, scale as f64)
                        .expect("geometry inside the kernel's envelope")
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap();
                    assert_close(
                        &got,
                        &expect,
                        1e-3,
                        &format!("contiguous d={d} T={total} w={window}"),
                    );
                }
            }
        }
    }

    /// Fused and unfused answers must agree bit-for-bit: same kernel, same
    /// data — the fused path only adds the custom-op detour through the
    /// fusion stream, including the read-only arena registration the plan
    /// flags as this stage's named risk.
    #[test]
    fn fused_dispatch_agrees_with_unfused_exactly() {
        if crate::skip_no_gpu() {
            return;
        }
        let (n_kv, n_rep, d, total, page_size) = (2usize, 3usize, 128usize, 33usize, 16usize);
        let n_q = n_kv * n_rep;
        let keys: Vec<Vec<f32>> = (0..n_kv * total).map(|i| signal(d, i as f32)).collect();
        let vals: Vec<Vec<f32>> =
            (0..n_kv * total).map(|i| signal(d, i as f32 + 55.0)).collect();
        let q = signal(n_q * d, 3.14);
        let (ak, av, table, num_pages) =
            paged_layout(&keys, &vals, n_kv, d, total, page_size);

        let run_unfused = {
            let device = Default::default();
            let qt: Tensor<UnfusedF32, 4> =
                Tensor::from_data(TensorData::new(q.clone(), [1, n_q, 1, d]), &device);
            let kt: Tensor<UnfusedF32, 4> = Tensor::from_data(
                TensorData::new(ak.clone(), [num_pages, n_kv, page_size, d]),
                &device,
            );
            let vt: Tensor<UnfusedF32, 4> = Tensor::from_data(
                TensorData::new(av.clone(), [num_pages, n_kv, page_size, d]),
                &device,
            );
            let len = table.len();
            let tt: Tensor<UnfusedF32, 1, Int> =
                Tensor::from_data(TensorData::new(table.clone(), [len]), &device);
            try_decode_attention(qt, kt, vt, tt, total, 0, 0.125)
                .expect("kernel path")
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        };
        let run_fused = {
            let device = Default::default();
            let qt: Tensor<FusedF32, 4> =
                Tensor::from_data(TensorData::new(q, [1, n_q, 1, d]), &device);
            let kt: Tensor<FusedF32, 4> = Tensor::from_data(
                TensorData::new(ak, [num_pages, n_kv, page_size, d]),
                &device,
            );
            let vt: Tensor<FusedF32, 4> = Tensor::from_data(
                TensorData::new(av, [num_pages, n_kv, page_size, d]),
                &device,
            );
            let len = table.len();
            let tt: Tensor<FusedF32, 1, Int> =
                Tensor::from_data(TensorData::new(table, [len]), &device);
            try_decode_attention(qt, kt, vt, tt, total, 0, 0.125)
                .expect("kernel path")
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        };
        assert_eq!(run_unfused, run_fused, "one kernel, two dispatch routes");
    }
}
