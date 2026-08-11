//! MLP activation dispatch for the universal decoder, with the f16-safe
//! precision policy (cubed terms overflow f16; compute in f32).

use burn::tensor::{Tensor, backend::Backend};
use combs_formats::Activation;

use crate::precision::{to_f32, to_float};

/// tanh-approximated GELU (HF `gelu_pytorch_tanh`): the gemma-family MLP
/// activation.
pub(crate) fn gelu_tanh<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let out_dtype = x.dtype();
    let x = to_f32(x);
    const C: f64 = 0.797_884_560_802_865_4; // sqrt(2/pi)
    let inner = (x.clone() + x.clone().powf_scalar(3.0).mul_scalar(0.044715)).mul_scalar(C);
    let y = x * inner.tanh().add_scalar(1.0).mul_scalar(0.5);
    to_float(y, out_dtype)
}

/// Applies the resolved activation.
pub(crate) fn apply<B: Backend, const D: usize>(
    activation: Activation,
    x: Tensor<B, D>,
) -> Tensor<B, D> {
    match activation {
        Activation::Silu => burn::tensor::activation::silu(x),
        Activation::GeluTanh => gelu_tanh(x),
        Activation::Gelu => burn::tensor::activation::gelu(x),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    #[test]
    fn gelu_tanh_matches_known_values() {
        let device = Default::default();
        let x = Tensor::<NdArray<f32>, 1>::from_data(
            TensorData::from([-1.0f32, 0.0, 1.0, 2.0].as_slice()),
            &device,
        );
        let y: Vec<f32> = gelu_tanh(x).into_data().to_vec().unwrap();
        // Reference values from torch.nn.functional.gelu(..., approximate="tanh").
        let expected = [-0.158808, 0.0, 0.841192, 1.954597];
        for (a, b) in y.iter().zip(expected) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
