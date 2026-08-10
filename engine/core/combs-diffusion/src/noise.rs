//! Seeded Gaussian noise for the diffusion loop, computed on the HOST and
//! uploaded — never `Tensor::random`. Backend RNGs are global, unseedable
//! per-request, and not contractually reproducible across backends; a
//! host-side generator gives byte-identical images for a given seed on
//! wgpu, CPU, or any future backend. Same house RNG as the sampler
//! (xorshift64* seeded through splitmix64, no external crates), plus a
//! Box–Muller transform for normals.

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

pub struct NoiseSource {
    state: u64,
    seed: u64,
    spare: Option<f32>,
}

impl NoiseSource {
    /// `None` seeds from wall-clock entropy; the effective seed is always
    /// recorded so responses can echo it for reproduction.
    pub fn new(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9E37_79B9_7F4A_7C15)
        });
        // splitmix64 finalizer: decorrelates small/sequential seeds.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let state = (z ^ (z >> 31)).max(1); // xorshift state must be non-zero
        Self { state, seed, spare: None }
    }

    /// The seed this source runs on (caller-provided or entropy-drawn).
    pub fn effective_seed(&self) -> u64 {
        self.seed
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in (0, 1] — never 0, safe for `ln`.
    fn next_unit(&mut self) -> f64 {
        (((self.next_u64() >> 11) + 1) as f64) / ((1u64 << 53) as f64)
    }

    /// Standard normal via Box–Muller (pair-cached).
    pub fn normal(&mut self) -> f32 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some((r * theta.sin()) as f32);
        (r * theta.cos()) as f32
    }

    /// A `[b, c, h, w]` tensor of standard normals.
    pub fn normal_tensor<B: Backend>(
        &mut self,
        shape: [usize; 4],
        device: &burn::tensor::Device<B>,
    ) -> Tensor<B, 4> {
        let n: usize = shape.iter().product();
        let values: Vec<f32> = (0..n).map(|_| self.normal()).collect();
        Tensor::from_data(TensorData::new(values, shape), device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_streams_are_reproducible_and_distinct() {
        let a: Vec<f32> = {
            let mut n = NoiseSource::new(Some(42));
            (0..64).map(|_| n.normal()).collect()
        };
        let b: Vec<f32> = {
            let mut n = NoiseSource::new(Some(42));
            (0..64).map(|_| n.normal()).collect()
        };
        let c: Vec<f32> = {
            let mut n = NoiseSource::new(Some(43));
            (0..64).map(|_| n.normal()).collect()
        };
        assert_eq!(a, b, "same seed must replay the identical stream");
        assert_ne!(a, c, "different seeds must diverge");
    }

    #[test]
    fn normals_look_gaussian() {
        let mut n = NoiseSource::new(Some(7));
        let xs: Vec<f64> = (0..100_000).map(|_| n.normal() as f64).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / xs.len() as f64;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.03, "var {var}");
    }
}
