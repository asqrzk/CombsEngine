//! Denoising schedulers for the Burn-native diffusion pipeline.
//!
//! All schedulers share the SD 1.5 scaled-linear beta schedule and the
//! diffusers "leading" timestep spacing with `steps_offset = 1`, computed
//! in f64 and harmony-tested against reference values derived from the
//! diffusers formulas. Three algorithms:
//!
//! - [`Ddpm`] — ancestral sampling with the spaced posterior (needs many
//!   steps; kept as the reference sampler),
//! - [`Ddim`] — deterministic (eta = 0), `set_alpha_to_one = false`,
//! - [`DpmPP2M`] — DPM-Solver++ 2M multistep (midpoint, epsilon
//!   prediction, final sigma zero, first-order final step). Converged
//!   output at ~20 steps; the default.

use burn::tensor::{Tensor, backend::Backend};

use crate::noise::NoiseSource;

/// SD 1.5 training schedule length.
pub const TRAIN_STEPS: usize = 1000;
const STEPS_OFFSET: usize = 1;

/// Scaled-linear cumulative alphas (`beta_start 0.00085`, `beta_end 0.012`).
pub(crate) fn alphas_cumprod(train_steps: usize) -> Vec<f64> {
    let (bs, be) = (0.00085_f64, 0.012_f64);
    let denom = (train_steps.max(2) - 1) as f64;
    let mut acc = 1.0;
    (0..train_steps)
        .map(|i| {
            let sqrt_beta = bs.sqrt() + (i as f64 / denom) * (be.sqrt() - bs.sqrt());
            acc *= 1.0 - sqrt_beta * sqrt_beta;
            acc
        })
        .collect()
}

/// diffusers "leading" spacing: `arange(n) * (train // n)`, descending,
/// plus `steps_offset` (1 for SD 1.5).
pub(crate) fn leading_timesteps(train_steps: usize, n: usize) -> Vec<usize> {
    let n = n.max(1);
    let ratio = train_steps / n;
    (0..n).map(|i| i * ratio + STEPS_OFFSET).rev().collect()
}

/// One denoising algorithm over the shared schedule. `step` consumes the
/// model's epsilon prediction for `timesteps()[step_index]` and returns the
/// latent advanced one step. DDPM and DPM++ land exactly on x0 at the final
/// step; DDIM (`set_alpha_to_one = false`) keeps the SD-conventional
/// `sqrt(1 - ac[0]) ≈ 0.029` noise residual.
pub trait Scheduler<B: Backend>: Send {
    fn timesteps(&self) -> &[usize];

    /// Multiplier for the initial N(0,1) latent (1.0 for VP schedulers).
    fn init_noise_sigma(&self) -> f32 {
        1.0
    }

    /// Per-step model-input scaling (identity for VP schedulers; sigma
    /// scaling once a Karras/VE sampler lands).
    fn scale_model_input(&self, latent: Tensor<B, 4>, _step_index: usize) -> Tensor<B, 4> {
        latent
    }

    fn step(
        &mut self,
        noise_pred: Tensor<B, 4>,
        step_index: usize,
        latent: Tensor<B, 4>,
        noise: &mut NoiseSource,
    ) -> Tensor<B, 4>;
}

/// Which scheduler a request runs. `DpmPP2M` is the default: converged
/// output at 20 steps where DDPM under-denoises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchedulerKind {
    Ddpm,
    Ddim,
    #[default]
    DpmPP2M,
}

impl SchedulerKind {
    /// Accepts the common spellings ("ddpm", "ddim", "dpm++2m",
    /// "dpmpp2m", "dpm"); `None` for anything else.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().replace(['-', '_', ' '], "").as_str() {
            "ddpm" => Some(Self::Ddpm),
            "ddim" => Some(Self::Ddim),
            "dpm" | "dpmpp2m" | "dpm++2m" | "dpmsolver++2m" | "dpmsolvermultistep" => {
                Some(Self::DpmPP2M)
            }
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Ddpm => "ddpm",
            Self::Ddim => "ddim",
            Self::DpmPP2M => "dpm++2m",
        }
    }

    /// Builds the scheduler prepared for `num_inference_steps` (clamped to
    /// `1..=TRAIN_STEPS`: above the training length the integer step ratio
    /// collapses to 0, degenerating every timestep to 1 — NaNs in DPM++).
    pub fn build<B: Backend>(self, num_inference_steps: usize) -> Box<dyn Scheduler<B>> {
        let n = num_inference_steps.clamp(1, TRAIN_STEPS);
        match self {
            Self::Ddpm => Box::new(Ddpm::new(n)),
            Self::Ddim => Box::new(Ddim::new(n)),
            Self::DpmPP2M => Box::new(DpmPP2M::new(n)),
        }
    }
}

// ---------------------------------------------------------------------------
// DDPM (ancestral, spaced posterior, fixed_small variance)

pub struct Ddpm {
    timesteps: Vec<usize>,
    ac: Vec<f64>,
    ratio: usize,
}

impl Ddpm {
    pub fn new(n: usize) -> Self {
        Self {
            timesteps: leading_timesteps(TRAIN_STEPS, n),
            ac: alphas_cumprod(TRAIN_STEPS),
            ratio: TRAIN_STEPS / n.max(1),
        }
    }

    /// Posterior coefficients for `x0`, `x_t`, and the variance at `t`
    /// (spaced steps: `alpha_prod_prev = 1` past the start).
    fn coefficients(&self, t: usize) -> (f64, f64, f64) {
        let at = self.ac[t];
        let ap = t
            .checked_sub(self.ratio)
            .map(|p| self.ac[p])
            .unwrap_or(1.0);
        let current_alpha = at / ap;
        let current_beta = 1.0 - current_alpha;
        let c_x0 = ap.sqrt() * current_beta / (1.0 - at);
        let c_xt = current_alpha.sqrt() * (1.0 - ap) / (1.0 - at);
        let variance = (current_beta * (1.0 - ap) / (1.0 - at)).max(1e-20);
        (c_x0, c_xt, variance)
    }
}

impl<B: Backend> Scheduler<B> for Ddpm {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn step(
        &mut self,
        noise_pred: Tensor<B, 4>,
        step_index: usize,
        latent: Tensor<B, 4>,
        noise: &mut NoiseSource,
    ) -> Tensor<B, 4> {
        let t = self.timesteps[step_index];
        let at = self.ac[t];
        let (c_x0, c_xt, variance) = self.coefficients(t);

        let x0 = (latent.clone() - noise_pred.mul_scalar((1.0 - at).sqrt() as f32))
            .div_scalar(at.sqrt() as f32);
        let mean = x0.mul_scalar(c_x0 as f32) + latent.mul_scalar(c_xt as f32);

        let last = step_index + 1 == self.timesteps.len();
        if last || variance <= 1e-12 {
            mean
        } else {
            let dims = mean.dims();
            let eps = noise.normal_tensor::<B>(dims, &mean.device());
            mean + eps.mul_scalar(variance.sqrt() as f32)
        }
    }
}

// ---------------------------------------------------------------------------
// DDIM (eta = 0, deterministic; set_alpha_to_one = false)

pub struct Ddim {
    timesteps: Vec<usize>,
    ac: Vec<f64>,
    ratio: usize,
}

impl Ddim {
    pub fn new(n: usize) -> Self {
        Self {
            timesteps: leading_timesteps(TRAIN_STEPS, n),
            ac: alphas_cumprod(TRAIN_STEPS),
            ratio: TRAIN_STEPS / n.max(1),
        }
    }
}

impl<B: Backend> Scheduler<B> for Ddim {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn step(
        &mut self,
        noise_pred: Tensor<B, 4>,
        step_index: usize,
        latent: Tensor<B, 4>,
        _noise: &mut NoiseSource,
    ) -> Tensor<B, 4> {
        let t = self.timesteps[step_index];
        let at = self.ac[t];
        // `set_alpha_to_one = false` (SD 1.5): the final boundary uses
        // alphas_cumprod[0], not 1.
        let ap = t
            .checked_sub(self.ratio)
            .map(|p| self.ac[p])
            .unwrap_or(self.ac[0]);

        let x0 = (latent - noise_pred.clone().mul_scalar((1.0 - at).sqrt() as f32))
            .div_scalar(at.sqrt() as f32);
        x0.mul_scalar(ap.sqrt() as f32) + noise_pred.mul_scalar((1.0 - ap).sqrt() as f32)
    }
}

// ---------------------------------------------------------------------------
// DPM-Solver++ 2M (multistep midpoint, epsilon prediction, final sigma 0)

pub struct DpmPP2M<B: Backend> {
    timesteps: Vec<usize>,
    /// Per step 0..n plus the fully-denoised boundary at index n:
    /// VP alpha_t, sigma_t and lambda = ln(alpha/sigma) (+inf at the end).
    alpha_t: Vec<f64>,
    sigma_t: Vec<f64>,
    lambda: Vec<f64>,
    prev_x0: Option<Tensor<B, 4>>,
    prev_h: Option<f64>,
    /// Multistep history is keyed to sequential stepping; guard misuse.
    next_index: usize,
}

impl<B: Backend> DpmPP2M<B> {
    pub fn new(n: usize) -> Self {
        let timesteps = leading_timesteps(TRAIN_STEPS, n);
        let ac = alphas_cumprod(TRAIN_STEPS);
        // Karras-convention sigma per step, 0 at the boundary
        // (`final_sigmas_type = "zero"`), mapped back to VP alpha/sigma.
        let sigmas: Vec<f64> = timesteps
            .iter()
            .map(|&t| ((1.0 - ac[t]) / ac[t]).sqrt())
            .chain(std::iter::once(0.0))
            .collect();
        let alpha_t: Vec<f64> = sigmas.iter().map(|s| 1.0 / (1.0 + s * s).sqrt()).collect();
        let sigma_t: Vec<f64> = sigmas
            .iter()
            .zip(&alpha_t)
            .map(|(s, a)| s * a)
            .collect();
        let lambda: Vec<f64> = sigmas
            .iter()
            .map(|s| if *s > 0.0 { -s.ln() } else { f64::INFINITY })
            .collect();
        Self {
            timesteps,
            alpha_t,
            sigma_t,
            lambda,
            prev_x0: None,
            prev_h: None,
            next_index: 0,
        }
    }
}

impl<B: Backend> Scheduler<B> for DpmPP2M<B> {
    fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    fn step(
        &mut self,
        noise_pred: Tensor<B, 4>,
        i: usize,
        latent: Tensor<B, 4>,
        _noise: &mut NoiseSource,
    ) -> Tensor<B, 4> {
        assert_eq!(
            i, self.next_index,
            "DPM++ 2M is multistep: steps must be taken in order (expected \
             step {}, got {i})",
            self.next_index
        );
        self.next_index += 1;
        let n = self.timesteps.len();
        let x0 = (latent.clone() - noise_pred.mul_scalar(self.sigma_t[i] as f32))
            .div_scalar(self.alpha_t[i] as f32);

        let h = self.lambda[i + 1] - self.lambda[i];
        // First-order on the first step (no history) and the final step
        // (`lower_order_final`; h is infinite there and r0 would blow up).
        let d = match (&self.prev_x0, self.prev_h) {
            (Some(prev_x0), Some(prev_h)) if i + 1 < n => {
                let r0 = prev_h / h;
                x0.clone().mul_scalar((1.0 + 1.0 / (2.0 * r0)) as f32)
                    - prev_x0.clone().mul_scalar((1.0 / (2.0 * r0)) as f32)
            }
            _ => x0.clone(),
        };

        // exp(-h) - 1 with h = +inf collapses to -1 (the boundary step
        // returns the denoised estimate exactly).
        let ehm1 = if h.is_finite() { (-h).exp() - 1.0 } else { -1.0 };
        let sigma_ratio = if self.sigma_t[i] > 0.0 {
            self.sigma_t[i + 1] / self.sigma_t[i]
        } else {
            0.0
        };
        let next = latent.mul_scalar(sigma_ratio as f32)
            - d.mul_scalar((self.alpha_t[i + 1] * ehm1) as f32);

        self.prev_x0 = Some(x0);
        self.prev_h = Some(h);
        next
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray<f32>;

    /// Reference values computed independently in Python (f64) from the
    /// diffusers formulas: scaled-linear betas, leading spacing offset 1,
    /// spaced-DDPM posterior, DDIM with set_alpha_to_one=false, DPM++ 2M
    /// midpoint with final sigma zero.
    const AC_HEAD: [f64; 5] = [
        0.99915,
        0.998296027838,
        0.997438081988,
        0.996576161011,
        0.995710263563,
    ];
    const AC_TAIL: [f64; 5] = [
        0.004890135151,
        0.004831711885,
        0.004773901585,
        0.0047166989,
        0.004660098513,
    ];

    fn scalar(v: f32) -> Tensor<B, 4> {
        Tensor::from_data(TensorData::new(vec![v], [1, 1, 1, 1]), &Default::default())
    }

    fn value(t: Tensor<B, 4>) -> f64 {
        t.into_data().to_vec::<f32>().unwrap()[0] as f64
    }

    #[test]
    fn schedule_matches_diffusers_reference() {
        let ac = alphas_cumprod(TRAIN_STEPS);
        for (i, expected) in AC_HEAD.iter().enumerate() {
            assert!((ac[i] - expected).abs() < 1e-9, "ac[{i}]");
        }
        for (i, expected) in AC_TAIL.iter().enumerate() {
            let idx = TRAIN_STEPS - 5 + i;
            assert!((ac[idx] - expected).abs() < 1e-9, "ac[{idx}]");
        }
        assert_eq!(
            leading_timesteps(TRAIN_STEPS, 20),
            vec![
                951, 901, 851, 801, 751, 701, 651, 601, 551, 501, 451, 401, 351, 301, 251,
                201, 151, 101, 51, 1
            ]
        );
    }

    #[test]
    fn ddpm_coefficients_match_reference() {
        let s = Ddpm::new(20);
        let rows = [
            (0usize, [0.049838408675, 0.758582958609, 0.415239778939]),
            (10, [0.181689592553, 0.797159478239, 0.161244783444]),
        ];
        for (idx, expected) in rows {
            let (c0, cx, var) = s.coefficients(s.timesteps[idx]);
            assert!((c0 - expected[0]).abs() < 1e-9, "c_x0[{idx}]");
            assert!((cx - expected[1]).abs() < 1e-9, "c_xt[{idx}]");
            assert!((var - expected[2]).abs() < 1e-9, "var[{idx}]");
        }
        // Final step collapses to x0 exactly.
        let (c0, cx, var) = s.coefficients(1);
        assert!((c0 - 1.0).abs() < 1e-12 && cx.abs() < 1e-12 && var <= 1e-12);
    }

    /// Scalar chains: x = 1.0, constant eps_hat = 0.5, 20 steps — final
    /// values must match the Python reference to float32 tolerance.
    #[test]
    fn scalar_chains_match_reference() {
        let mut noise = NoiseSource::new(Some(0));

        let mut s: Box<dyn Scheduler<B>> = SchedulerKind::Ddim.build(20);
        let mut x = scalar(1.0);
        for i in 0..20 {
            x = s.step(scalar(0.5), i, x, &mut noise);
        }
        assert!(
            (value(x) - 5.571624674353).abs() < 1e-4,
            "ddim chain diverged"
        );

        let mut s: Box<dyn Scheduler<B>> = SchedulerKind::DpmPP2M.build(20);
        let mut x = scalar(1.0);
        for i in 0..20 {
            x = s.step(scalar(0.5), i, x, &mut noise);
        }
        assert!(
            (value(x) - 5.559410546395).abs() < 1e-4,
            "dpm++2m chain diverged"
        );
    }

    #[test]
    fn kind_parsing_accepts_common_spellings() {
        assert_eq!(SchedulerKind::parse("DPM++2M"), Some(SchedulerKind::DpmPP2M));
        assert_eq!(SchedulerKind::parse("dpm_pp_2m"), Some(SchedulerKind::DpmPP2M));
        assert_eq!(SchedulerKind::parse("ddim"), Some(SchedulerKind::Ddim));
        assert_eq!(SchedulerKind::parse("DDPM"), Some(SchedulerKind::Ddpm));
        assert_eq!(SchedulerKind::parse("euler"), None);
        assert_eq!(SchedulerKind::default(), SchedulerKind::DpmPP2M);
    }
}
