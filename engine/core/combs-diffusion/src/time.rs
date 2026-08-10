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

    let freqs: Vec<f32> = (0..half)
        .map(|i| (-(i as f32) / (half as f32 - 1.0).max(1.0)).exp() * 10_000.0)
        .collect();
    let freqs = Tensor::<B, 1>::from_floats(freqs.as_slice(), device);

    let args = t.unsqueeze::<2>().permute([1, 0]) / freqs.unsqueeze::<2>();
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
