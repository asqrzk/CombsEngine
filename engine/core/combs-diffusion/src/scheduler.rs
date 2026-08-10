//! DDPM scheduler for the Burn-native diffusion pipeline.
//!
//! Uses the scaled-linear beta schedule from Stable Diffusion 1.5.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Fixed scaled-linear beta schedule and the derived DDPM alphas.
pub struct DdpmScheduler {
    betas: Vec<f32>,
    alphas_cumprod: Vec<f32>,
    posterior_variance: Vec<f32>,
    pub init_noise_sigma: f32,
}

impl DdpmScheduler {
    /// `train_steps` is the number of diffusion steps the model was trained
    /// with (typically 1000 for SD 1.5).
    pub fn new(train_steps: usize) -> Self {
        let beta_start = 0.00085_f32;
        let beta_end = 0.012_f32;
        let betas: Vec<f32> = (0..train_steps)
            .map(|i| {
                let t = i as f32 / (train_steps.max(2) - 1) as f32;
                let beta = beta_start.sqrt() * (1.0 - t) + beta_end.sqrt() * t;
                beta * beta
            })
            .collect();
        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();
        let alphas_cumprod: Vec<f32> = alphas
            .iter()
            .scan(1.0, |acc, &a| {
                *acc *= a;
                Some(*acc)
            })
            .collect();
        let posterior_variance: Vec<f32> = betas
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                if i == 0 {
                    b
                } else {
                    b * (1.0 - alphas_cumprod[i - 1]) / (1.0 - alphas_cumprod[i])
                }
            })
            .collect();
        let init_noise_sigma = 1.0;
        Self {
            betas,
            alphas_cumprod,
            posterior_variance,
            init_noise_sigma,
        }
    }

    /// Prepare the scheduler for `num_steps` inference steps, spaced evenly
    /// across the training timestep range.
    pub fn inference_steps(&self, num_steps: usize) -> InferenceSchedule<'_> {
        let num_steps = num_steps.max(1);
        let max_t = self.betas.len() - 1;
        let step = (max_t as f32) / num_steps as f32;
        let timesteps: Vec<usize> = (0..num_steps)
            .map(|i| (max_t as f32 - step * (i as f32 + 0.5)) as usize)
            .collect();
        InferenceSchedule {
            timesteps,
            index: 0,
            scheduler: self,
        }
    }

    /// DDPM update: predict x_0 from noisy latent and noise prediction, then
    /// re-noise to the previous timestep.
    pub fn step<B: Backend>(
        &self,
        noisy_latent: &Tensor<B, 4>,
        noise_pred: &Tensor<B, 4>,
        t: usize,
        prev_t: usize,
    ) -> Tensor<B, 4> {
        let t = t.min(self.betas.len() - 1);
        let prev_t = prev_t.min(self.betas.len() - 1);

        // predict x_0
        let alpha_prod_t = self.alphas_cumprod[t];
        let beta_prod_t = 1.0 - alpha_prod_t;
        let pred_original = (noisy_latent.clone()
            - noise_pred.clone().mul_scalar(beta_prod_t.sqrt()))
            .div_scalar(alpha_prod_t.sqrt());

        // re-noise to prev_t
        let alpha_prod_prev = self.alphas_cumprod[prev_t];
        let beta_prod_prev = 1.0 - alpha_prod_prev;
        let pred_noisy = pred_original.mul_scalar(alpha_prod_prev.sqrt())
            + noise_pred.clone().mul_scalar(beta_prod_prev.sqrt());

        // DDPM adds posterior variance noise except at t=0
        if prev_t == 0 {
            pred_noisy
        } else {
            let variance = self.posterior_variance[prev_t];
            let std = (variance as f64).sqrt();
            let noise = Tensor::random_like(
                noise_pred,
                burn::tensor::Distribution::Normal(0.0, std),
            );
            pred_noisy + noise
        }
    }
}

/// Iterator-like view over the selected inference timesteps.
pub struct InferenceSchedule<'a> {
    timesteps: Vec<usize>,
    index: usize,
    scheduler: &'a DdpmScheduler,
}

impl<'a> InferenceSchedule<'a> {
    pub fn timestep(&self) -> usize {
        self.timesteps[self.index.min(self.timesteps.len() - 1)]
    }

    pub fn has_next(&self) -> bool {
        self.index < self.timesteps.len()
    }

    /// Advance one step and return the denoised latent. The last step lands
    /// at t=0.
    pub fn step<B: Backend>(
        &mut self,
        noisy_latent: &Tensor<B, 4>,
        noise_pred: &Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let t = self.timestep();
        let prev_t = if self.index + 1 < self.timesteps.len() {
            self.timesteps[self.index + 1]
        } else {
            0
        };
        self.index += 1;
        self.scheduler.step(noisy_latent, noise_pred, t, prev_t)
    }
}
