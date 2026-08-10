//! Rotary positional embeddings (standard Llama RoPE, half-split /
//! `rotate_half` convention as used by HuggingFace `transformers`).

use burn::tensor::{Tensor, TensorData, backend::Backend, Device};

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
    let half = head_dim / 2;
    let inv_freq: Vec<f64> = (0..half)
        .map(|i| theta.powf(-2.0 * i as f64 / head_dim as f64))
        .collect();
    let mut cos = Vec::with_capacity(max_position * head_dim);
    let mut sin = Vec::with_capacity(max_position * head_dim);
    for pos in 0..max_position {
        // Half-split layout: the half-dim frequencies appear twice so the
        // table can be applied elementwise against [x1, x2].
        for _ in 0..2 {
            for f in &inv_freq {
                let angle = pos as f64 * f;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
    }
    (cos, sin)
}

impl<B: Backend> RotaryEmbedding<B> {
    /// Builds tables on `device`.
    pub fn new(head_dim: usize, theta: f64, max_position: usize, device: &Device<B>) -> Self {
        let (cos, sin) = build_tables(head_dim, theta, max_position);
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
}
