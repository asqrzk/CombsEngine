//! Tensor-loading helpers for diffusion models.
//!
//! Follows the house style from `combs-models`: open tensors by HuggingFace
//! name from a `ModelSource` and wrap them in `burn::module::Param` for use
//! in `burn::nn` modules.

use burn::module::Param;
use burn::nn::conv::Conv2d;
use burn::nn::{Embedding, GroupNorm, LayerNorm, Linear};
use burn::tensor::{backend::Backend, Tensor};
use combs_formats::{FormatError, ModelSource, Result};

/// Load a tensor of rank `D` by name, widening F16/BF16 to F32 if needed.
pub fn load_tensor<B: Backend, const D: usize>(
    source: &dyn ModelSource,
    name: &str,
    device: &burn::tensor::Device<B>,
) -> Result<Tensor<B, D>> {
    source.open_tensor(name)?.load_to_tensor(device)
}

/// Load a tensor and wrap it in a `Param`.
pub fn load_param<B: Backend, const D: usize>(
    source: &dyn ModelSource,
    name: &str,
    device: &burn::tensor::Device<B>,
) -> Result<Param<Tensor<B, D>>> {
    Ok(Param::from_tensor(load_tensor(source, name, device)?))
}

/// Load an optional tensor. A missing tensor becomes `None`; other errors
/// are propagated.
pub fn load_optional_tensor<B: Backend, const D: usize>(
    source: &dyn ModelSource,
    name: &str,
    device: &burn::tensor::Device<B>,
) -> Result<Option<Tensor<B, D>>> {
    match source.open_tensor(name) {
        Ok(reader) => Ok(Some(reader.load_to_tensor(device)?)),
        Err(FormatError::TensorNotFound(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Load an optional parameter.
pub fn load_optional_param<B: Backend, const D: usize>(
    source: &dyn ModelSource,
    name: &str,
    device: &burn::tensor::Device<B>,
) -> Result<Option<Param<Tensor<B, D>>>> {
    Ok(load_optional_tensor(source, name, device)?.map(Param::from_tensor))
}

/// Verify that a loaded tensor has the expected shape.
pub fn expect_shape(name: &str, actual: &[usize], expected: &[usize]) -> Result<()> {
    if actual != expected {
        return Err(FormatError::Safetensors(format!(
            "shape mismatch for {name}: expected {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

/// Load a `Conv2d` module from HF-style `weight` and optional `bias`.
pub(crate) fn load_conv2d<B: Backend>(
    source: &dyn ModelSource,
    prefix: &str,
    channels: [usize; 2],
    kernel_size: [usize; 2],
    device: &burn::tensor::Device<B>,
) -> Result<Conv2d<B>> {
    let weight = load_param::<B, 4>(source, &format!("{prefix}.weight"), device)?;
    let bias = load_optional_param::<B, 1>(source, &format!("{prefix}.bias"), device)?;
    let [out_ch, in_ch, kh, kw] = weight.dims();
    expect_shape(
        &format!("{prefix}.weight"),
        &[out_ch, in_ch, kh, kw],
        &[channels[1], channels[0], kernel_size[0], kernel_size[1]],
    )?;
    Ok(Conv2d {
        weight,
        bias,
        stride: [1, 1],
        kernel_size,
        dilation: [1, 1],
        groups: 1,
        padding: burn::nn::PaddingConfig2d::Same,
    })
}

/// Load a `Linear` module. PyTorch stores weights as `[out, in]`; Burn
/// expects `[in, out]`, so we transpose on load.
pub(crate) fn load_linear<B: Backend>(
    source: &dyn ModelSource,
    prefix: &str,
    d_input: usize,
    d_output: usize,
    bias: bool,
    device: &burn::tensor::Device<B>,
) -> Result<Linear<B>> {
    let weight_tensor = load_tensor::<B, 2>(source, &format!("{prefix}.weight"), device)?;
    expect_shape(
        &format!("{prefix}.weight"),
        &weight_tensor.dims(),
        &[d_output, d_input],
    )?;
    let weight = Param::from_tensor(weight_tensor.transpose());
    let bias = if bias {
        Some(load_param::<B, 1>(source, &format!("{prefix}.bias"), device)?)
    } else {
        None
    };
    Ok(Linear { weight, bias })
}

/// Load an affine `GroupNorm` module.
pub(crate) fn load_group_norm<B: Backend>(
    source: &dyn ModelSource,
    prefix: &str,
    num_groups: usize,
    num_channels: usize,
    eps: f64,
    device: &burn::tensor::Device<B>,
) -> Result<GroupNorm<B>> {
    let gamma = load_param::<B, 1>(source, &format!("{prefix}.weight"), device)?;
    let beta = load_param::<B, 1>(source, &format!("{prefix}.bias"), device)?;
    expect_shape(
        &format!("{prefix}.weight"),
        &gamma.dims(),
        &[num_channels],
    )?;
    expect_shape(&format!("{prefix}.bias"), &beta.dims(), &[num_channels])?;
    Ok(GroupNorm {
        gamma: Some(gamma),
        beta: Some(beta),
        num_groups,
        num_channels,
        epsilon: eps,
        affine: true,
    })
}

/// Load an `Embedding` module.
pub(crate) fn load_embedding<B: Backend>(
    source: &dyn ModelSource,
    prefix: &str,
    n_embedding: usize,
    d_model: usize,
    device: &burn::tensor::Device<B>,
) -> Result<Embedding<B>> {
    let weight = load_param::<B, 2>(source, &format!("{prefix}.weight"), device)?;
    expect_shape(
        &format!("{prefix}.weight"),
        &weight.dims(),
        &[n_embedding, d_model],
    )?;
    Ok(Embedding { weight })
}

/// Load an affine `LayerNorm` module.
pub(crate) fn load_layer_norm<B: Backend>(
    source: &dyn ModelSource,
    prefix: &str,
    d_model: usize,
    device: &burn::tensor::Device<B>,
) -> Result<LayerNorm<B>> {
    let gamma = load_param::<B, 1>(source, &format!("{prefix}.weight"), device)?;
    let beta = load_param::<B, 1>(source, &format!("{prefix}.bias"), device)?;
    expect_shape(
        &format!("{prefix}.weight"),
        &gamma.dims(),
        &[d_model],
    )?;
    expect_shape(&format!("{prefix}.bias"), &beta.dims(), &[d_model])?;
    // `epsilon` is private, so init a default LayerNorm and overwrite the
    // public gamma/beta fields.
    let mut ln = burn::nn::LayerNormConfig::new(d_model).init(device);
    ln.gamma = gamma;
    ln.beta = Some(beta);
    Ok(ln)
}
