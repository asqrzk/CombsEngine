//! Rotary positional embeddings (standard Llama RoPE, half-split /
//! `rotate_half` convention as used by HuggingFace `transformers`),
//! including the frequency-scaling variants from `modeling_rope_utils`
//! (linear, llama3 piecewise, YaRN NTK-by-parts).

use burn::tensor::{Tensor, TensorData, backend::Backend, Device};
use combs_formats::RopeScaling;

/// Precomputed RoPE cosine/sine tables.
///
/// Frequencies: `inv_freq[i] = theta^(-2i / head_dim)` for `i in 0..head_dim/2`.
/// Tables are `[max_position, head_dim]` with the half-dim frequencies
/// duplicated (`cat([cos, cos])`) to match the half-split convention
/// (`config.rope_interleaved = false`, the HF Llama default).
pub struct RotaryEmbedding<B: Backend> {
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
}

/// Computes the `[max_position, head_dim]` cos/sin tables on the host.
/// Exposed for unit tests.
pub fn build_tables(
    head_dim: usize,
    theta: f64,
    max_position: usize,
) -> (Vec<f32>, Vec<f32>) {
    build_tables_scaled(head_dim, theta, max_position, &RopeScaling::None)
}

/// Per-frequency scaled `inv_freq` plus the attention (mscale) multiplier
/// applied to both tables — pure host f64 math matching HF
/// `modeling_rope_utils` formulas. Exposed for the formula golden tests.
pub fn scaled_inv_freq(head_dim: usize, theta: f64, scaling: &RopeScaling) -> (Vec<f64>, f64) {
    let half = head_dim / 2;
    let base: Vec<f64> = (0..half)
        .map(|i| theta.powf(-2.0 * i as f64 / head_dim as f64))
        .collect();
    match scaling {
        RopeScaling::None => (base, 1.0),
        RopeScaling::Linear { factor } => (base.iter().map(|f| f / factor).collect(), 1.0),
        RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position_embeddings,
        } => {
            let orig = *original_max_position_embeddings as f64;
            let low_wavelen = orig / low_freq_factor;
            let high_wavelen = orig / high_freq_factor;
            let scaled = base
                .iter()
                .map(|&f| {
                    let wavelen = 2.0 * std::f64::consts::PI / f;
                    if wavelen < high_wavelen {
                        f
                    } else if wavelen > low_wavelen {
                        f / factor
                    } else {
                        let smooth = (orig / wavelen - low_freq_factor)
                            / (high_freq_factor - low_freq_factor);
                        (1.0 - smooth) * f / factor + smooth * f
                    }
                })
                .collect();
            (scaled, 1.0)
        }
        RopeScaling::Yarn {
            factor,
            original_max_position_embeddings,
            beta_fast,
            beta_slow,
            attention_factor,
        } => {
            let dim = head_dim as f64;
            let orig = *original_max_position_embeddings as f64;
            let corr_dim = |rotations: f64| {
                dim * (orig / (rotations * 2.0 * std::f64::consts::PI)).ln()
                    / (2.0 * theta.ln())
            };
            let low = corr_dim(*beta_fast).floor().max(0.0);
            let mut high = corr_dim(*beta_slow).ceil().min(dim - 1.0);
            if (high - low).abs() < f64::EPSILON {
                high += 0.001; // avoid a zero-width ramp
            }
            let scaled = (0..half)
                .map(|i| {
                    let pos_freq = theta.powf(2.0 * i as f64 / dim);
                    let extrapolation = 1.0 / pos_freq;
                    let interpolation = 1.0 / (factor * pos_freq);
                    let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
                    let extrapolation_factor = 1.0 - ramp;
                    interpolation * (1.0 - extrapolation_factor)
                        + extrapolation * extrapolation_factor
                })
                .collect();
            let mscale = attention_factor.unwrap_or(0.1 * factor.ln() + 1.0);
            (scaled, mscale)
        }
    }
}

/// [`build_tables`] with RoPE frequency scaling. The YaRN attention factor
/// multiplies both tables (temperature on the rotation, HF convention).
pub fn build_tables_scaled(
    head_dim: usize,
    theta: f64,
    max_position: usize,
    scaling: &RopeScaling,
) -> (Vec<f32>, Vec<f32>) {
    let (inv_freq, mscale) = scaled_inv_freq(head_dim, theta, scaling);
    let mut cos = Vec::with_capacity(max_position * head_dim);
    let mut sin = Vec::with_capacity(max_position * head_dim);
    for pos in 0..max_position {
        // Half-split layout: the half-dim frequencies appear twice so the
        // table can be applied elementwise against [x1, x2].
        for _ in 0..2 {
            for f in &inv_freq {
                let angle = pos as f64 * f;
                cos.push((angle.cos() * mscale) as f32);
                sin.push((angle.sin() * mscale) as f32);
            }
        }
    }
    (cos, sin)
}

impl<B: Backend> RotaryEmbedding<B> {
    /// Builds tables on `device` (no frequency scaling).
    pub fn new(head_dim: usize, theta: f64, max_position: usize, device: &Device<B>) -> Self {
        Self::new_scaled(head_dim, theta, max_position, &RopeScaling::None, device)
    }

    /// Builds tables on `device` with RoPE frequency scaling applied.
    pub fn new_scaled(
        head_dim: usize,
        theta: f64,
        max_position: usize,
        scaling: &RopeScaling,
        device: &Device<B>,
    ) -> Self {
        let (cos, sin) = build_tables_scaled(head_dim, theta, max_position, scaling);
        RotaryEmbedding {
            cos: Tensor::from_data(
                TensorData::new(cos, [max_position, head_dim]),
                device,
            ),
            sin: Tensor::from_data(
                TensorData::new(sin, [max_position, head_dim]),
                device,
            ),
        }
    }

    /// Applies RoPE to a `[batch, heads, seq, head_dim]` tensor whose first
    /// sequence position is at absolute position `pos`.
    pub fn apply(&self, x: Tensor<B, 4>, pos: usize) -> Tensor<B, 4> {
        let [batch, heads, seq, dim] = x.dims();
        let half = dim / 2;
        let cos = self
            .cos
            .clone()
            .narrow(0, pos, seq)
            .reshape([1, 1, seq, dim]);
        let sin = self
            .sin
            .clone()
            .narrow(0, pos, seq)
            .reshape([1, 1, seq, dim]);

        // rotate_half(x) = cat([-x2, x1]) along the head_dim axis.
        let x1 = x.clone().narrow(3, 0, half);
        let x2 = x.clone().narrow(3, half, half);
        let rotated = Tensor::cat(vec![x2.neg(), x1], 3);

        let out = x * cos + rotated * sin;
        debug_assert_eq!(out.dims(), [batch, heads, seq, dim]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_match_known_math() {
        let head_dim = 8;
        let theta = 10000.0f64;
        let max_pos = 4;
        let (cos, sin) = build_tables(head_dim, theta, max_pos);
        // inv_freq[i] = theta^(-2i/d): i=0 -> 1, i=1 -> 10000^-0.25 = 0.1,
        // i=2 -> 0.01, i=3 -> 0.001.
        let inv = [1.0f64, 0.1, 0.01, 0.001];
        for pos in 0..max_pos {
            for i in 0..4 {
                let angle = pos as f64 * inv[i];
                // half-split layout: index i and i + half share the frequency.
                for idx in [pos * head_dim + i, pos * head_dim + i + 4] {
                    assert!(
                        (cos[idx] as f64 - angle.cos()).abs() < 1e-5,
                        "cos mismatch at pos {pos} i {i}"
                    );
                    assert!(
                        (sin[idx] as f64 - angle.sin()).abs() < 1e-5,
                        "sin mismatch at pos {pos} i {i}"
                    );
                }
            }
        }
        // Position 0 must be the identity rotation.
        for i in 0..head_dim {
            assert!((cos[i] - 1.0).abs() < 1e-6);
            assert!(sin[i].abs() < 1e-6);
        }
    }

    /// Reference values computed independently (Python f64) from the HF
    /// `modeling_rope_utils` formulas.
    #[test]
    fn linear_scaling_divides_frequencies() {
        let (inv, mscale) = scaled_inv_freq(8, 10_000.0, &RopeScaling::Linear { factor: 2.0 });
        let expected = [0.5, 0.05, 0.005, 0.0005];
        for (i, e) in expected.iter().enumerate() {
            assert!((inv[i] - e).abs() < 1e-12, "linear inv[{i}]");
        }
        assert_eq!(mscale, 1.0);
    }

    #[test]
    fn llama3_scaling_matches_reference() {
        // llama-3.2-1b config: dim 64, theta 500000, factor 32, low 1,
        // high 4, original 8192.
        let scaling = RopeScaling::Llama3 {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        };
        let (inv, mscale) = scaled_inv_freq(64, 500_000.0, &scaling);
        let expected = [
            (0usize, 1.0),
            (8, 0.037606030931),
            (16, 0.000429556797),
            (24, 1.661967e-06),
            (31, 9.4183e-08),
        ];
        for (i, e) in expected {
            let rel = ((inv[i] - e) / e).abs();
            assert!(rel < 1e-6, "llama3 inv[{i}]: {} vs {e}", inv[i]);
        }
        assert_eq!(mscale, 1.0);
    }

    #[test]
    fn yarn_scaling_matches_reference() {
        // Qwen-style: dim 128, theta 1e6, factor 4, original 32768,
        // beta 32/1 -> correction range [23, 40], mscale 0.1·ln4 + 1.
        let scaling = RopeScaling::Yarn {
            factor: 4.0,
            original_max_position_embeddings: 32768,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: None,
        };
        let (inv, mscale) = scaled_inv_freq(128, 1_000_000.0, &scaling);
        let expected = [
            (0usize, 1.0),
            (16, 0.03162277660168379),
            (32, 0.0006029411764705882),
            (48, 7.905694150420949e-06),
            (63, 3.102344401879299e-07),
        ];
        for (i, e) in expected {
            let rel = ((inv[i] - e) / e).abs();
            assert!(rel < 1e-6, "yarn inv[{i}]: {} vs {e}", inv[i]);
        }
        assert!((mscale - 1.138629436112).abs() < 1e-9, "mscale {mscale}");
    }

    #[test]
    fn scaled_tables_apply_mscale() {
        // Position 0 cos = mscale (not 1) under YaRN.
        let scaling = RopeScaling::Yarn {
            factor: 4.0,
            original_max_position_embeddings: 32768,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: Some(1.25),
        };
        let (cos, sin) = build_tables_scaled(8, 10_000.0, 2, &scaling);
        assert!((cos[0] - 1.25).abs() < 1e-6);
        assert!(sin[0].abs() < 1e-6);
    }
}
