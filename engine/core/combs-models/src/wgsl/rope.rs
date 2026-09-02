//! K2: fused RoPE — q and k rotated in one dispatch, closing S1.
//!
//! The burn path narrows both tables, materializes rotate_half via
//! neg + cat, and runs the mul-add chain twice (q, then k). Here the
//! kernel reads the full resident tables at `pos + r` and writes both
//! rotated tensors from one elementwise launch.
//!
//! Fallback discipline as everywhere: doors, foreign backends, f16, an
//! odd head dim, a position past the table, or a grid past the dispatch
//! limit → `None`, and the caller applies the tables the old way.

use std::any::TypeId;

use burn::backend::wgpu::{CubeTensor, WgpuRuntime};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Shape, Tensor, TensorPrimitive};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl_fusion::CubeFusionHandle;
use burn_fusion::stream::{Operation, OperationStreams};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr, TensorStatus};
use cubecl::prelude::CubeCount;

use super::rmsnorm::cast_any;
use super::{RopeQk, WORKGROUP, launch, wgsl_enabled};
use crate::qlinear::{FusedF32, InnerF32, UnfusedF32};

/// Per-kernel door on top of the master `COMBS_WGSL`; read once.
fn rope_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        wgsl_enabled() && !matches!(std::env::var("COMBS_WGSL_ROPE").as_deref(), Ok("0"))
    })
}

#[derive(Clone, Copy)]
struct RopeGeom {
    n_q: usize,
    n_kv: usize,
    seq: usize,
    d: usize,
    pos: usize,
}

impl RopeGeom {
    fn scalar_slots(self) -> Vec<u64> {
        vec![
            self.n_q as u64,
            self.n_kv as u64,
            self.seq as u64,
            self.d as u64,
            self.pos as u64,
        ]
    }

    fn workgroups(self) -> usize {
        ((self.n_q + self.n_kv) * self.seq * self.d).div_ceil(WORKGROUP as usize)
    }
}

/// Tries the fused RoPE for `q: [1, n_q, seq, d]` and `k: [1, n_kv, seq,
/// d]` against the resident `[max_position, d]` tables. `None` → caller
/// applies the tables through the burn path, twice.
pub(crate) fn try_rope_qk<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
    pos: usize,
) -> Option<(Tensor<B, 4>, Tensor<B, 4>)> {
    if !rope_enabled() {
        return None;
    }
    let [qb, n_q, seq, d] = q.dims();
    let [kb, n_kv, k_seq, k_d] = k.dims();
    let [max_position, t_d] = cos.dims();
    let geom = RopeGeom {
        n_q,
        n_kv,
        seq,
        d,
        pos,
    };
    if qb != 1
        || kb != 1
        || k_seq != seq
        || k_d != d
        || t_d != d
        || d == 0
        || d % 2 != 0
        || seq == 0
        || pos + seq > max_position
        || geom.workgroups() > 65535
    {
        return None;
    }
    if TypeId::of::<B>() == TypeId::of::<UnfusedF32>() {
        let q: Tensor<UnfusedF32, 4> = cast_any(q)?;
        let k: Tensor<UnfusedF32, 4> = cast_any(k)?;
        let c: Tensor<UnfusedF32, 2> = cast_any(cos)?;
        let s: Tensor<UnfusedF32, 2> = cast_any(sin)?;
        return cast_any(rope_unfused(q, k, c, s, geom));
    }
    if TypeId::of::<B>() == TypeId::of::<FusedF32>() {
        let q: Tensor<FusedF32, 4> = cast_any(q)?;
        let k: Tensor<FusedF32, 4> = cast_any(k)?;
        let c: Tensor<FusedF32, 2> = cast_any(cos)?;
        let s: Tensor<FusedF32, 2> = cast_any(sin)?;
        return cast_any(rope_fused(q, k, c, s, geom));
    }
    None
}

/// The launch itself, shared by both backends once primitives are in
/// hand. Returns (q_rotated, k_rotated).
fn launch_rope(
    q: CubeTensor<WgpuRuntime>,
    k: CubeTensor<WgpuRuntime>,
    cos: CubeTensor<WgpuRuntime>,
    sin: CubeTensor<WgpuRuntime>,
    geom: RopeGeom,
) -> (CubeTensor<WgpuRuntime>, CubeTensor<WgpuRuntime>) {
    let q = into_contiguous(q);
    let k = into_contiguous(k);
    let cos = into_contiguous(cos);
    let sin = into_contiguous(sin);
    let client = q.client.clone();
    let out_q = client.empty(geom.n_q * geom.seq * geom.d * core::mem::size_of::<f32>());
    let out_k = client.empty(geom.n_kv * geom.seq * geom.d * core::mem::size_of::<f32>());
    launch(
        &client,
        RopeQk,
        CubeCount::Static(geom.workgroups() as u32, 1, 1),
        vec![
            q.handle.binding(),
            k.handle.binding(),
            cos.handle.binding(),
            sin.handle.binding(),
            out_q.clone().binding(),
            out_k.clone().binding(),
        ],
        geom.scalar_slots(),
    );
    (
        CubeTensor::new_contiguous(
            client.clone(),
            q.device.clone(),
            Shape::from([1, geom.n_q, geom.seq, geom.d]),
            out_q,
            DType::F32,
        ),
        CubeTensor::new_contiguous(
            client,
            q.device.clone(),
            Shape::from([1, geom.n_kv, geom.seq, geom.d]),
            out_k,
            DType::F32,
        ),
    )
}

fn rope_unfused(
    q: Tensor<UnfusedF32, 4>,
    k: Tensor<UnfusedF32, 4>,
    cos: Tensor<UnfusedF32, 2>,
    sin: Tensor<UnfusedF32, 2>,
    geom: RopeGeom,
) -> (Tensor<UnfusedF32, 4>, Tensor<UnfusedF32, 4>) {
    let (oq, ok) = launch_rope(
        q.into_primitive().tensor(),
        k.into_primitive().tensor(),
        cos.into_primitive().tensor(),
        sin.into_primitive().tensor(),
        geom,
    );
    (
        Tensor::from_primitive(TensorPrimitive::Float(oq)),
        Tensor::from_primitive(TensorPrimitive::Float(ok)),
    )
}

/// The fusion-stream operation: two inputs rotated, two outputs, tables
/// as read-only inputs (the rotary embedding keeps them alive).
struct RopeQkOp {
    desc: CustomOpIr,
    geom: RopeGeom,
}

impl core::fmt::Debug for RopeQkOp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "RopeQkOp {{ n_q: {}, n_kv: {}, seq: {}, pos: {} }}",
            self.geom.n_q, self.geom.n_kv, self.geom.seq, self.geom.pos
        )
    }
}

impl Operation<FusionCubeRuntime<WgpuRuntime>> for RopeQkOp {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle<WgpuRuntime>>) {
        let ([q, k, c, s], [oq_ir, ok_ir]) = self.desc.as_fixed::<4, 2>();
        let qp: CubeTensor<WgpuRuntime> = handles.get_float_tensor::<InnerF32>(q);
        let kp: CubeTensor<WgpuRuntime> = handles.get_float_tensor::<InnerF32>(k);
        let cp: CubeTensor<WgpuRuntime> = handles.get_float_tensor::<InnerF32>(c);
        let sp: CubeTensor<WgpuRuntime> = handles.get_float_tensor::<InnerF32>(s);
        let (oq, ok) = launch_rope(qp, kp, cp, sp, self.geom);
        handles.register_float_tensor::<InnerF32>(&oq_ir.id, oq);
        handles.register_float_tensor::<InnerF32>(&ok_ir.id, ok);
    }
}

fn rope_fused(
    q: Tensor<FusedF32, 4>,
    k: Tensor<FusedF32, 4>,
    cos: Tensor<FusedF32, 2>,
    sin: Tensor<FusedF32, 2>,
    geom: RopeGeom,
) -> (Tensor<FusedF32, 4>, Tensor<FusedF32, 4>) {
    let qp = q.into_primitive().tensor();
    let kp = k.into_primitive().tensor();
    let cp = cos.into_primitive().tensor();
    let sp = sin.into_primitive().tensor();
    let client = qp.client.clone();

    let mut streams = OperationStreams::default();
    streams.tensor(&qp);
    streams.tensor(&kp);
    streams.tensor(&cp);
    streams.tensor(&sp);
    let q_ir = qp.into_ir();
    let k_ir = kp.into_ir();
    let c_ir = cp.into_ir();
    let s_ir = sp.into_ir();
    let oq_ir = TensorIr {
        id: client.create_empty_handle(),
        shape: Shape::from([1, geom.n_q, geom.seq, geom.d]),
        status: TensorStatus::NotInit,
        dtype: DType::F32,
    };
    let ok_ir = TensorIr {
        id: client.create_empty_handle(),
        shape: Shape::from([1, geom.n_kv, geom.seq, geom.d]),
        status: TensorStatus::NotInit,
        dtype: DType::F32,
    };
    let desc = CustomOpIr::new(
        "combs_rope_qk",
        &[q_ir, k_ir, c_ir, s_ir],
        &[oq_ir, ok_ir],
    );
    let op = RopeQkOp {
        desc: desc.clone(),
        geom,
    };
    let mut outputs = client.register(streams, OperationIr::Custom(desc), op);
    let ok = outputs.pop().expect("custom op declares two outputs");
    let oq = outputs.pop().expect("custom op declares two outputs");
    (
        Tensor::from_primitive(TensorPrimitive::Float(oq)),
        Tensor::from_primitive(TensorPrimitive::Float(ok)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rope::build_tables_scaled;
    use burn::tensor::TensorData;
    use combs_formats::RopeScaling;

    fn signal(len: usize, salt: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.613 + salt).sin() * 2.0) + ((i % 9) as f32 - 4.0) * 0.125)
            .collect()
    }

    /// Host reference: x·cos + rotate_half(x)·sin at pos + r, half-split.
    fn reference(
        x: &[f32],
        cos: &[f32],
        sin: &[f32],
        heads: usize,
        seq: usize,
        d: usize,
        pos: usize,
    ) -> Vec<f32> {
        let half = d / 2;
        let mut out = vec![0.0f32; x.len()];
        for h in 0..heads {
            for r in 0..seq {
                for c in 0..d {
                    let i = (h * seq + r) * d + c;
                    let mate = if c < half { -x[i + half] } else { x[i - half] };
                    let t = (pos + r) * d + c;
                    out[i] = x[i] * cos[t] + mate * sin[t];
                }
            }
        }
        out
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

    /// The plan's harmony matrix: head dims, plain + llama3 + YaRN scaled
    /// tables, decode and chunk shapes, large positions.
    #[test]
    fn fused_rope_matches_the_reference_across_tables_and_positions() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        let scalings = [
            RopeScaling::None,
            RopeScaling::Llama3 {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8192,
            },
            RopeScaling::Yarn {
                factor: 4.0,
                original_max_position_embeddings: 4096,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attention_factor: None,
            },
        ];
        for &d in &[64usize, 96, 128, 256] {
            for scaling in &scalings {
                let max_position = 8192;
                let (cos, sin) = build_tables_scaled(d, 10000.0, max_position, scaling);
                for &(seq, pos) in &[(1usize, 0usize), (1, 8000), (17, 4000)] {
                    let (n_q, n_kv) = (4usize, 2usize);
                    let qv = signal(n_q * seq * d, 1.1);
                    let kv = signal(n_kv * seq * d, 60.6);
                    let eq = reference(&qv, &cos, &sin, n_q, seq, d, pos);
                    let ek = reference(&kv, &cos, &sin, n_kv, seq, d, pos);

                    let qt: Tensor<UnfusedF32, 4> = Tensor::from_data(
                        TensorData::new(qv, [1, n_q, seq, d]),
                        &device,
                    );
                    let kt: Tensor<UnfusedF32, 4> = Tensor::from_data(
                        TensorData::new(kv, [1, n_kv, seq, d]),
                        &device,
                    );
                    let ct: Tensor<UnfusedF32, 2> = Tensor::from_data(
                        TensorData::new(cos.clone(), [max_position, d]),
                        &device,
                    );
                    let st: Tensor<UnfusedF32, 2> = Tensor::from_data(
                        TensorData::new(sin.clone(), [max_position, d]),
                        &device,
                    );
                    let (gq, gk) = try_rope_qk(qt, kt, ct, st, pos)
                        .expect("geometry inside the kernel's envelope");
                    assert_close(
                        &gq.into_data().to_vec::<f32>().unwrap(),
                        &eq,
                        1e-5,
                        &format!("q d={d} seq={seq} pos={pos}"),
                    );
                    assert_close(
                        &gk.into_data().to_vec::<f32>().unwrap(),
                        &ek,
                        1e-5,
                        &format!("k d={d} seq={seq} pos={pos}"),
                    );
                }
            }
        }
    }

    /// Fused and unfused dispatch must agree bit-for-bit, two-output
    /// custom op included.
    #[test]
    fn fused_dispatch_agrees_with_unfused_exactly() {
        if crate::skip_no_gpu() {
            return;
        }
        let (n_q, n_kv, seq, d, pos, max_position) = (4usize, 2usize, 3usize, 64usize, 100usize, 256usize);
        let (cos, sin) = build_tables_scaled(d, 10000.0, max_position, &RopeScaling::None);
        let qv = signal(n_q * seq * d, 5.5);
        let kv = signal(n_kv * seq * d, 66.0);

        let run_unfused = {
            let device = Default::default();
            let qt: Tensor<UnfusedF32, 4> =
                Tensor::from_data(TensorData::new(qv.clone(), [1, n_q, seq, d]), &device);
            let kt: Tensor<UnfusedF32, 4> =
                Tensor::from_data(TensorData::new(kv.clone(), [1, n_kv, seq, d]), &device);
            let ct: Tensor<UnfusedF32, 2> =
                Tensor::from_data(TensorData::new(cos.clone(), [max_position, d]), &device);
            let st: Tensor<UnfusedF32, 2> =
                Tensor::from_data(TensorData::new(sin.clone(), [max_position, d]), &device);
            let (a, b) = try_rope_qk(qt, kt, ct, st, pos).expect("kernel path");
            (
                a.into_data().to_vec::<f32>().unwrap(),
                b.into_data().to_vec::<f32>().unwrap(),
            )
        };
        let run_fused = {
            let device = Default::default();
            let qt: Tensor<FusedF32, 4> =
                Tensor::from_data(TensorData::new(qv, [1, n_q, seq, d]), &device);
            let kt: Tensor<FusedF32, 4> =
                Tensor::from_data(TensorData::new(kv, [1, n_kv, seq, d]), &device);
            let ct: Tensor<FusedF32, 2> =
                Tensor::from_data(TensorData::new(cos, [max_position, d]), &device);
            let st: Tensor<FusedF32, 2> =
                Tensor::from_data(TensorData::new(sin, [max_position, d]), &device);
            let (a, b) = try_rope_qk(qt, kt, ct, st, pos).expect("kernel path");
            (
                a.into_data().to_vec::<f32>().unwrap(),
                b.into_data().to_vec::<f32>().unwrap(),
            )
        };
        assert_eq!(run_unfused, run_fused, "one kernel, two dispatch routes");
    }
}
