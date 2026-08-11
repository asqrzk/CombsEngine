//! Time-step embedding for the UNet.
//!
//! Matches the standard sinusoidal timestep embedding used in Stable
//! Diffusion: `sin(t / 10000^(i/d))` for even channels, cosine for odd,
//! followed by a small SiLU-MLP.

use burn::nn::Linear;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::weights::load_linear;
use combs_formats::{ModelSource, Result};

/// Sinusoidal timestep embedding, shape `[batch, dim]`.
pub fn timestep_embedding<B: Backend>(
    timesteps: &Tensor<B, 1, burn::tensor::Int>,
    dim: usize,
    device: &burn::tensor::Device<B>,
) -> Tensor<B, 2> {
    assert!(dim % 2 == 0, "embedding dim must be even");
    let half = dim / 2;
    let t: Tensor<B, 1> = timesteps.clone().float();

    // Geometric frequency sweep, diffusers `Timesteps` convention with
    // SD 1.5's `freq_shift = 0`: inv_freq[i] = exp(-ln(10000) · i / half),
    // sweeping 1 → ~1e-4. (The earlier form divided by a growing period —
    // wrong scale AND direction; every channel sat near cos(0).)
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(10_000f32.ln()) * i as f32 / (half as f32).max(1.0)).exp())
        .collect();
    let freqs = Tensor::<B, 1>::from_floats(freqs.as_slice(), device);

    let args = t.unsqueeze::<2>().permute([1, 0]) * freqs.unsqueeze::<2>();
    let cos = args.clone().cos();
    let sin = args.sin();
    // SD 1.5 uses flip_sin_to_cos = true.
    Tensor::cat(vec![cos, sin], 1)
}

/// MLP that projects the sinusoidal embedding up and down.
pub struct TimeEmbedding<B: Backend> {
    pub(crate) linear_1: Linear<B>,
    pub(crate) linear_2: Linear<B>,
}

impl<B: Backend> TimeEmbedding<B> {
    pub fn new(in_dim: usize, out_dim: usize, device: &burn::tensor::Device<B>) -> Self {
        let linear_1 = burn::nn::LinearConfig::new(in_dim, out_dim).init(device);
        let linear_2 = burn::nn::LinearConfig::new(out_dim, out_dim).init(device);
        Self { linear_1, linear_2 }
    }

    pub fn load_from(
        source: &dyn ModelSource,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        device: &burn::tensor::Device<B>,
    ) -> Result<Self> {
        let linear_1 = load_linear(source, &format!("{prefix}.linear_1"), in_dim, out_dim, true, device)?;
        let linear_2 = load_linear(source, &format!("{prefix}.linear_2"), out_dim, out_dim, true, device)?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, t_emb: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.linear_1.forward(t_emb);
        self.linear_2.forward(burn::tensor::activation::silu(h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::{Int, TensorData};

    type B = NdArray<f32>;

    /// Golden: diffusers `Timesteps` (freq_shift 0, flip_sin_to_cos):
    /// dim 8 -> inv_freq [1, 1e-1, 1e-2, 1e-3]; t = 3 -> args
    /// [3, 0.3, 0.03, 0.003], layout [cos.., sin..].
    #[test]
    fn timestep_embedding_matches_reference() {
        let device = Default::default();
        let t = Tensor::<B, 1, Int>::from_data(TensorData::from([3i64].as_slice()), &device);
        let emb: Vec<f32> = timestep_embedding::<B>(&t, 8, &device)
            .into_data()
            .to_vec()
            .unwrap();
        let args = [3.0f64, 0.3, 0.03, 0.003];
        for (i, a) in args.iter().enumerate() {
            assert!((emb[i] as f64 - a.cos()).abs() < 1e-5, "cos[{i}]");
            assert!((emb[4 + i] as f64 - a.sin()).abs() < 1e-5, "sin[{i}]");
        }
        // The sweep must actually sweep: t=951 highest channel is
        // 951/10^3 -> args ~0.95..951, not a near-constant vector.
        let t = Tensor::<B, 1, Int>::from_data(TensorData::from([951i64].as_slice()), &device);
        let emb: Vec<f32> = timestep_embedding::<B>(&t, 8, &device)
            .into_data()
            .to_vec()
            .unwrap();
        assert!((emb[0] as f64 - (951.0f64).cos()).abs() < 1e-3, "cos[0] at t=951");
        assert!((emb[3] as f64 - (0.951f64).cos()).abs() < 1e-4, "cos[3] at t=951");
    }
}
