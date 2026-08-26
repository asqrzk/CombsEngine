//! Browser probe for the burn-composed forward paths our WGSL canaries
//! cannot cover: the manual masked-attention composite (sliding-layer
//! prefill — scores matmul, arange/bool mask, softmax, P@V) and the
//! fused flash-attention module at head_dim 256 (global-layer prefill).
//!
//! Both run on the production fused backend against a host softmax
//! reference, mirroring `wgsl::probe_report`'s discipline: values, not
//! absence of errors. gemma3 is the only browser model that exercises
//! either path, which is how they stayed unproven until it shipped
//! garbage — this probe makes that class impossible to miss again.

use burn::tensor::{Tensor, TensorData};

use crate::kv::attend;

type FB = combs_core::CombsBackend;

fn signal(len: usize, salt: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32 * 0.517 + salt).sin() * 1.2) + ((i % 5) as f32 - 2.0) * 0.2)
        .collect()
}

/// Host reference: causal (+ window) softmax attention, `q` rows at
/// absolute positions `pos..pos+seq` over `total` keys, GQA by head
/// group. Plain f32 — the same arithmetic the device path claims.
#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q: usize,
    n_kv: usize,
    seq: usize,
    total: usize,
    d: usize,
    pos: usize,
    scale: f32,
    window: Option<usize>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_q * seq * d];
    for h in 0..n_q {
        let g = h / (n_q / n_kv);
        for r in 0..seq {
            let p = pos + r;
            let qi = &q[(h * seq + r) * d..(h * seq + r + 1) * d];
            let visible: Vec<usize> = (0..total)
                .filter(|&j| j <= p && window.is_none_or(|w| j + w > p))
                .collect();
            let scores: Vec<f32> = visible
                .iter()
                .map(|&j| {
                    let kj = &k[(g * total + j) * d..(g * total + j + 1) * d];
                    qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let m = scores.iter().cloned().fold(f32::MIN, f32::max);
            let ps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let sum: f32 = ps.iter().sum();
            for (idx, &j) in visible.iter().enumerate() {
                let vj = &v[(g * total + j) * d..(g * total + j + 1) * d];
                for c in 0..d {
                    out[(h * seq + r) * d + c] += ps[idx] / sum * vj[c];
                }
            }
        }
    }
    out
}

async fn run_case(
    name: &str,
    n_q: usize,
    n_kv: usize,
    seq: usize,
    total: usize,
    d: usize,
    scale: f64,
    window: Option<usize>,
) -> Result<(), String> {
    let pos = total - seq;
    let qv = signal(n_q * seq * d, 3.3);
    let kv = signal(n_kv * total * d, 41.0);
    let vv = signal(n_kv * total * d, 77.0);
    let expect = reference(
        &qv, &kv, &vv, n_q, n_kv, seq, total, d, pos, scale as f32, window,
    );

    let device = Default::default();
    let q: Tensor<FB, 4> =
        Tensor::from_data(TensorData::new(qv, [1, n_q, seq, d]), &device);
    let k: Tensor<FB, 4> =
        Tensor::from_data(TensorData::new(kv, [1, n_kv, total, d]), &device);
    let v: Tensor<FB, 4> =
        Tensor::from_data(TensorData::new(vv, [1, n_kv, total, d]), &device);
    let out = attend(q, k, v, pos, scale, window);
    let data = out
        .into_data_async()
        .await
        .map_err(|e| format!("{name}: readback failed: {e:?}"))?
        .to_vec::<f32>()
        .map_err(|e| format!("{name}: conversion failed: {e:?}"))?;
    let mut worst = (0usize, 0.0f32, 0.0f32, 0.0f32);
    for (i, (g, e)) in data.iter().zip(&expect).enumerate() {
        let err = (g - e).abs();
        if !g.is_finite() || err > worst.1 {
            worst = (i, err, *g, *e);
            if !g.is_finite() {
                break;
            }
        }
    }
    let tol = 2e-3;
    if !worst.2.is_finite() || worst.1 > tol {
        return Err(format!(
            "{name}: worst at [{}]: got {}, expected {} (|err| {}) — this \
             burn path computes wrong values on this target",
            worst.0, worst.2, worst.3, worst.1
        ));
    }
    Ok(())
}

/// Runs the burn-path canaries on the current device; `Err` names the
/// first path whose values diverge from the host reference.
pub async fn forward_probe_report() -> Result<(), String> {
    let d = 256usize;
    let flash_scale = 1.0 / (d as f64).sqrt();
    // Sliding-layer prefill: window masking forces the manual composite.
    run_case("manual-attend seq=8 window=5", 4, 1, 8, 40, d, 0.11, Some(5)).await?;
    run_case(
        "manual-attend seq=32 window=512",
        4,
        1,
        32,
        300,
        d,
        flash_scale,
        Some(512),
    )
    .await?;
    // Global-layer prefill: default scale + no window takes burn's fused
    // flash-attention module — the d=256 autotune key gemma3 alone uses.
    run_case("flash seq=8 d=256", 4, 1, 8, 40, d, flash_scale, None).await?;
    run_case("flash seq=32 d=256", 4, 1, 32, 300, d, flash_scale, None).await?;
    // The proven-in-browser geometry as a control: if THIS fails, the
    // harness (not the kernels) is suspect.
    run_case("flash seq=8 d=128 control", 4, 2, 8, 40, 128, 1.0 / (128f64).sqrt(), None)
        .await?;
    Ok(())
}
