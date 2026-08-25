//! RMSNorm (Llama-style, no learnable bias).

use burn::tensor::{Tensor, backend::Backend};

use crate::precision::{to_f32, to_float};

/// `y = x / rms(x) * w` where `rms` is taken over the last dimension and
/// `eps` is added inside the square root.
///
/// The reduction runs in f32 for f16 stability (no-op in f32 builds).
/// On the wgpu f32 backends this dispatches the K1 WGSL kernel — one
/// workgroup per row instead of burn's mean/sqrt/mul chain; every
/// refusal (doors, other backends) falls through to that chain.
pub fn rms_norm<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    weight: Tensor<B, 1>,
    eps: f64,
) -> Tensor<B, D> {
    if let Some(y) = crate::wgsl::try_rms_norm(x.clone(), weight.clone(), eps, 0.0) {
        return y;
    }
    let out_dtype = x.dtype();
    let dims = x.dims();
    let hidden = dims[D - 1];

    let xf = to_f32(x);
    // mean(x^2) over the last dim, keeping rank for broadcasting.
    let mean_sq = xf.clone().powf_scalar(2.0).mean_dim(D - 1);
    let inv_rms = mean_sq.add_scalar(eps).sqrt().recip();

    let mut shape = [1usize; D];
    shape[D - 1] = hidden;
    let y = xf * inv_rms * to_f32(weight).reshape(shape);
    to_float(y, out_dtype)
}

/// Gemma-style RMSNorm: `y = x / rms(x) * (1 + w)` — the learnable weight
/// is zero-centered (HF `Gemma3RMSNorm`).
pub fn gemma_rms_norm<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    weight: Tensor<B, 1>,
    eps: f64,
) -> Tensor<B, D> {
    // The kernel takes the raw weight and adds the 1 in-register; the
    // fallback materializes the shifted weight as before.
    if let Some(y) = crate::wgsl::try_rms_norm(x.clone(), weight.clone(), eps, 1.0) {
        return y;
    }
    rms_norm(x, weight.add_scalar(1.0), eps)
}

/// LayerNorm (SigLIP-style, learnable weight + bias):
/// `y = (x - μ) / sqrt(σ² + eps) * w + b`, statistics over the last dim.
pub fn layer_norm<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    eps: f64,
) -> Tensor<B, D> {
    let dims = x.dims();
    let hidden = dims[D - 1];

    let out_dtype = x.dtype();
    let xf = to_f32(x);
    let mean = xf.clone().mean_dim(D - 1);
    let centered = xf - mean;
    let var = centered.clone().powf_scalar(2.0).mean_dim(D - 1);
    let inv_std = var.add_scalar(eps).sqrt().recip();

    let mut shape = [1usize; D];
    shape[D - 1] = hidden;
    let y = centered * inv_std * to_f32(weight).reshape(shape.clone())
        + to_f32(bias).reshape(shape);
    to_float(y, out_dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    type TestBackend = burn::backend::NdArray<f32>;

    #[test]
    fn normalizes_rows_to_unit_rms() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let x: Tensor<TestBackend, 2> = Tensor::from_data(
            TensorData::new(vec![3.0f32, 4.0, 1.0, -2.0, 0.5, 2.5], [2, 3]),
            &device,
        );
        let w: Tensor<TestBackend, 1> = Tensor::ones([3], &device);
        let y = rms_norm(x, w, 1e-6);
        // With unit weights, each row of y must have RMS == 1 (up to eps).
        let rms = y
            .powf_scalar(2.0)
            .mean_dim(1)
            .sqrt()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (i, r) in rms.iter().enumerate() {
            assert!((r - 1.0).abs() < 1e-4, "row {i} rms = {r}");
        }
    }

    #[test]
    fn applies_weight() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let x: Tensor<TestBackend, 2> =
            Tensor::from_data(TensorData::new(vec![1.0f32, 2.0], [1, 2]), &device);
        let w: Tensor<TestBackend, 1> =
            Tensor::from_data(TensorData::new(vec![2.0f32, 2.0], [2]), &device);
        let y = rms_norm(x.clone(), w, 1e-6);
        let z = rms_norm(x, Tensor::ones([2], &device), 1e-6);
        let yv: Vec<f32> = y.into_data().to_vec().unwrap();
        let zv: Vec<f32> = z.into_data().to_vec().unwrap();
        for i in 0..2 {
            assert!((yv[i] - 2.0 * zv[i]).abs() < 1e-4);
        }
    }
}
