//! UNet / VAE building blocks for Stable Diffusion 1.5.

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{GroupNorm, GroupNormConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::weights::{
    load_conv2d, load_group_norm, load_layer_norm, load_linear,
};
use combs_formats::{ModelSource, Result};

/// Number of groups for GroupNorm in SD 1.5 (channels are all multiples of 32).
const NORM_GROUPS: usize = 32;

/// A ResNet block with optional time-embedding injection.
pub struct ResnetBlock2D<B: Backend> {
    pub(crate) norm1: GroupNorm<B>,
    pub(crate) conv1: Conv2d<B>,
    pub(crate) time_emb_proj: Option<Linear<B>>,
    pub(crate) norm2: GroupNorm<B>,
    pub(crate) conv2: Conv2d<B>,
    pub(crate) conv_shortcut: Option<Conv2d<B>>,
    pub(crate) out_channels: usize,
}

impl<B: Backend> ResnetBlock2D<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        time_emb_dim: Option<usize>,
        eps: f64,
        device: &burn::tensor::Device<B>,
    ) -> Self {
        let groups1 = norm_groups(in_channels);
        let groups2 = norm_groups(out_channels);
        let norm1 = GroupNormConfig::new(groups1, in_channels)
            .with_epsilon(eps)
            .init(device);
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
        let time_emb_proj = time_emb_dim.map(|d| {
            LinearConfig::new(d, out_channels)
                .init(device)
        });
        let norm2 = GroupNormConfig::new(groups2, out_channels)
            .with_epsilon(eps)
            .init(device);
        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
        let conv_shortcut = if in_channels != out_channels {
            Some(Conv2dConfig::new([in_channels, out_channels], [1, 1]).init(device))
        } else {
            None
        };
        Self {
            norm1,
            conv1,
            time_emb_proj,
            norm2,
            conv2,
            conv_shortcut,
            out_channels,
        }
    }

    pub fn load_from(
        source: &dyn ModelSource,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        time_emb_dim: Option<usize>,
        eps: f64,
        device: &burn::tensor::Device<B>,
    ) -> Result<Self> {
        let groups1 = norm_groups(in_channels);
        let groups2 = norm_groups(out_channels);

        let norm1 = load_group_norm(source, &format!("{prefix}.norm1"), groups1, in_channels, eps, device)?;
        let conv1 = load_conv2d(source, &format!("{prefix}.conv1"), [in_channels, out_channels], [3, 3], [1, 1], device)?;
        let time_emb_proj = if let Some(d) = time_emb_dim {
            Some(load_linear(source, &format!("{prefix}.time_emb_proj"), d, out_channels, true, device)?)
        } else {
            None
        };
        let norm2 = load_group_norm(source, &format!("{prefix}.norm2"), groups2, out_channels, eps, device)?;
        let conv2 = load_conv2d(source, &format!("{prefix}.conv2"), [out_channels, out_channels], [3, 3], [1, 1], device)?;
        let conv_shortcut = if in_channels != out_channels {
            Some(load_conv2d(source, &format!("{prefix}.conv_shortcut"), [in_channels, out_channels], [1, 1], [1, 1], device)?)
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            time_emb_proj,
            norm2,
            conv2,
            conv_shortcut,
            out_channels,
        })
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        t_emb: Option<&Tensor<B, 2>>,
    ) -> Tensor<B, 4> {
        let [b, _, _h, _w] = x.dims();
        let mut hidden = silu(self.norm1.forward(x.clone()));
        hidden = self.conv1.forward(hidden);

        if let (Some(t), Some(proj)) = (t_emb, self.time_emb_proj.as_ref()) {
            let t_proj = proj.forward(silu(t.clone()));
            hidden = hidden.add(t_proj.reshape([b, self.out_channels, 1, 1]));
        }

        hidden = silu(self.norm2.forward(hidden));
        hidden = self.conv2.forward(hidden);

        let shortcut = if let Some(ref conv) = self.conv_shortcut {
            conv.forward(x)
        } else {
            x
        };

        hidden.add(shortcut)
    }
}

/// Scaled dot-product attention operating on a sequence `[B, N, C]`.
pub struct Attention<B: Backend> {
    pub(crate) to_q: Linear<B>,
    pub(crate) to_k: Linear<B>,
    pub(crate) to_v: Linear<B>,
    pub(crate) to_out: Linear<B>,
    pub(crate) heads: usize,
    pub(crate) dim_head: usize,
    pub(crate) scale: f32,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        query_dim: usize,
        context_dim: usize,
        heads: usize,
        dim_head: usize,
        qkv_bias: bool,
        device: &burn::tensor::Device<B>,
    ) -> Self {
        let inner_dim = dim_head * heads;
        let to_q = LinearConfig::new(query_dim, inner_dim)
            .with_bias(qkv_bias)
            .init(device);
        let to_k = LinearConfig::new(context_dim, inner_dim)
            .with_bias(qkv_bias)
            .init(device);
        let to_v = LinearConfig::new(context_dim, inner_dim)
            .with_bias(qkv_bias)
            .init(device);
        let to_out = LinearConfig::new(inner_dim, query_dim).init(device);
        Self {
            to_q,
            to_k,
            to_v,
            to_out,
            heads,
            dim_head,
            scale: 1.0 / (dim_head as f32).sqrt(),
        }
    }

    pub fn load_from(
        source: &dyn ModelSource,
        prefix: &str,
        query_dim: usize,
        context_dim: usize,
        heads: usize,
        dim_head: usize,
        qkv_bias: bool,
        device: &burn::tensor::Device<B>,
    ) -> Result<Self> {
        let inner_dim = dim_head * heads;
        let to_q = load_linear(source, &format!("{prefix}.to_q"), query_dim, inner_dim, qkv_bias, device)?;
        let to_k = load_linear(source, &format!("{prefix}.to_k"), context_dim, inner_dim, qkv_bias, device)?;
        let to_v = load_linear(source, &format!("{prefix}.to_v"), context_dim, inner_dim, qkv_bias, device)?;
        let to_out = load_linear(source, &format!("{prefix}.to_out.0"), inner_dim, query_dim, true, device)?;
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            heads,
            dim_head,
            scale: 1.0 / (dim_head as f32).sqrt(),
        })
    }

    /// `x` and optional `context` are `[B, N, C]`. When `context` is `None`,
    /// performs self-attention.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        context: Option<&Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let [b, n, _c] = x.dims();
        let context = context.unwrap_or(&x);
        let [_, context_n, _context_c] = context.dims();

        let q = self.to_q.forward(x.clone());
        let k = self.to_k.forward(context.clone());
        let v = self.to_v.forward(context.clone());

        let q = q.reshape([b, n, self.heads, self.dim_head]).permute([0, 2, 1, 3]);
        let k = k.reshape([b, context_n, self.heads, self.dim_head]).permute([0, 2, 1, 3]);
        let v = v.reshape([b, context_n, self.heads, self.dim_head]).permute([0, 2, 1, 3]);

        let scores = q
            .matmul(k.permute([0, 1, 3, 2]))
            .mul_scalar(self.scale);
        let attn = softmax(scores, 3);
        let out = attn.matmul(v);

        let out = out.permute([0, 2, 1, 3])
            .reshape([b, n, self.heads * self.dim_head]);
        self.to_out.forward(out)
    }
}

/// GEGLU activation: splits a linear projection in two, GELUs one half,
/// then multiplies elementwise.
pub struct Geglu<B: Backend> {
    pub(crate) proj: Linear<B>,
}

impl<B: Backend> Geglu<B> {
    pub fn new(dim: usize, hidden_dim: usize, device: &burn::tensor::Device<B>) -> Self {
        let proj = LinearConfig::new(dim, hidden_dim * 2).init(device);
        Self { proj }
    }

    pub fn load_from(
        source: &dyn ModelSource,
        prefix: &str,
        dim: usize,
        hidden_dim: usize,
        device: &burn::tensor::Device<B>,
    ) -> Result<Self> {
        let proj = load_linear(source, prefix, dim, hidden_dim * 2, true, device)?;
        Ok(Self { proj })
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, _] = x.dims();
        let projected = self.proj.forward(x);
        let [_, _, total] = projected.dims();
        let hidden_dim = total / 2;
        let chunks = projected.reshape([b, n, 2, hidden_dim]);
        let a = chunks.clone().narrow(2, 0, 1).reshape([b, n, hidden_dim]);
        let b_gate = chunks.narrow(2, 1, 1).reshape([b, n, hidden_dim]);
        let gated = gelu(b_gate).mul(a);
        gated
    }
}

/// Feed-forward block used inside UNet transformer blocks: GEGLU + Linear.
pub struct FeedForward<B: Backend> {
    pub(crate) geglu: Geglu<B>,
    pub(crate) proj: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(dim: usize, multiplier: usize, device: &burn::tensor::Device<B>) -> Self {
        let hidden_dim = dim * multiplier;
        let geglu = Geglu::new(dim, hidden_dim, device);
        let proj = LinearConfig::new(hidden_dim, dim).init(device);
        Self { geglu, proj }
    }

    pub fn load_from(
        source: &dyn ModelSource,
        prefix: &str,
        dim: usize,
        multiplier: usize,
        device: &burn::tensor::Device<B>,
    ) -> Result<Self> {
        let hidden_dim = dim * multiplier;
        let geglu = Geglu::load_from(source, &format!("{prefix}.net.0.proj"), dim, hidden_dim, device)?;
        let proj = load_linear(source, &format!("{prefix}.net.2"), hidden_dim, dim, true, device)?;
        Ok(Self { geglu, proj })
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.proj.forward(self.geglu.forward(x))
    }
}

/// A single transformer block matching diffusers `BasicTransformerBlock`:
/// self-attention -> cross-attention -> feed-forward, each pre-normalized.
pub struct BasicTransformerBlock<B: Backend> {
    pub(crate) norm1: LayerNorm<B>,
    pub(crate) attn1: Attention<B>,
    pub(crate) norm2: LayerNorm<B>,
    pub(crate) attn2: Attention<B>,
    pub(crate) norm3: LayerNorm<B>,
    pub(crate) ff: FeedForward<B>,
}

impl<B: Backend> BasicTransformerBlock<B> {
    pub fn new(
        dim: usize,
        context_dim: usize,
        heads: usize,
        dim_head: usize,
        qkv_bias: bool,
        device: &burn::tensor::Device<B>,
    ) -> Self {
        let norm1 = LayerNormConfig::new(dim).init(device);
        let attn1 = Attention::new(dim, dim, heads, dim_head, qkv_bias, device);
        let norm2 = LayerNormConfig::new(dim).init(device);
        let attn2 = Attention::new(dim, context_dim, heads, dim_head, qkv_bias, device);
        let norm3 = LayerNormConfig::new(dim).init(device);
        let ff = FeedForward::new(dim, 4, device);
        Self {
            norm1,
            attn1,
            norm2,
            attn2,
            norm3,
            ff,
        }
    }

    pub fn load_from(
        source: &dyn ModelSource,
        prefix: &str,
        dim: usize,
        context_dim: usize,
        heads: usize,
        dim_head: usize,
        qkv_bias: bool,
        device: &burn::tensor::Device<B>,
    ) -> Result<Self> {
        let norm1 = load_layer_norm(source, &format!("{prefix}.norm1"), dim, device)?;
        let attn1 = Attention::load_from(source, &format!("{prefix}.attn1"), dim, dim, heads, dim_head, qkv_bias, device)?;
        let norm2 = load_layer_norm(source, &format!("{prefix}.norm2"), dim, device)?;
        let attn2 = Attention::load_from(source, &format!("{prefix}.attn2"), dim, context_dim, heads, dim_head, qkv_bias, device)?;
        let norm3 = load_layer_norm(source, &format!("{prefix}.norm3"), dim, device)?;
        let ff = FeedForward::load_from(source, &format!("{prefix}.ff"), dim, 4, device)?;
        Ok(Self {
            norm1,
            attn1,
            norm2,
            attn2,
            norm3,
            ff,
        })
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        context: &Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let x = x.clone().add(self.attn1.forward(self.norm1.forward(x.clone()), None));
        let x = x.clone().add(self.attn2.forward(self.norm2.forward(x.clone()), Some(context)));
        let ff_out = self.ff.forward(self.norm3.forward(x.clone()));
        x.add(ff_out)
    }
}

/// Spatial transformer block for UNet: GroupNorm -> 1x1 conv -> sequence
/// transformer -> 1x1 conv, with a residual around the whole block.
pub struct SpatialTransformer<B: Backend> {
    pub(crate) norm: GroupNorm<B>,
    pub(crate) proj_in: Conv2d<B>,
    pub(crate) transformer: BasicTransformerBlock<B>,
    pub(crate) proj_out: Conv2d<B>,
}

impl<B: Backend> SpatialTransformer<B> {
    pub fn new(
        channels: usize,
        context_dim: usize,
        heads: usize,
        dim_head: usize,
        device: &burn::tensor::Device<B>,
    ) -> Self {
        let norm = GroupNormConfig::new(norm_groups(channels), channels).init(device);
        let proj_in = Conv2dConfig::new([channels, channels], [1, 1]).init(device);
        let transformer = BasicTransformerBlock::new(channels, context_dim, heads, dim_head, false, device);
        let proj_out = Conv2dConfig::new([channels, channels], [1, 1]).init(device);
        Self {
            norm,
            proj_in,
            transformer,
            proj_out,
        }
    }

    pub fn load_from(
        source: &dyn ModelSource,
        prefix: &str,
        channels: usize,
        context_dim: usize,
        heads: usize,
        dim_head: usize,
        device: &burn::tensor::Device<B>,
    ) -> Result<Self> {
        let norm = load_group_norm(source, &format!("{prefix}.norm"), norm_groups(channels), channels, 1e-6, device)?;
        let proj_in = load_conv2d(source, &format!("{prefix}.proj_in"), [channels, channels], [1, 1], [1, 1], device)?;
        let transformer = BasicTransformerBlock::load_from(
            source,
            &format!("{prefix}.transformer_blocks.0"),
            channels,
            context_dim,
            heads,
            dim_head,
            false,
            device,
        )?;
        let proj_out = load_conv2d(source, &format!("{prefix}.proj_out"), [channels, channels], [1, 1], [1, 1], device)?;
        Ok(Self {
            norm,
            proj_in,
            transformer,
            proj_out,
        })
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        context: &Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
        let residual = x.clone();
        let h_norm = self.norm.forward(x);
        let h_seq = self
            .proj_in
            .forward(h_norm)
            .reshape([b, c, h * w])
            .permute([0, 2, 1]);
        let h_seq = self.transformer.forward(h_seq, context);
        let h_img = h_seq
            .permute([0, 2, 1])
            .reshape([b, c, h, w]);
        self.proj_out.forward(h_img).add(residual)
    }
}

/// Strided 3x3 convolution for downsampling.
pub struct Downsample2D<B: Backend> {
    pub(crate) conv: Conv2d<B>,
}

impl<B: Backend> Downsample2D<B> {
    pub fn new(channels: usize, device: &burn::tensor::Device<B>) -> Self {
        let conv = Conv2dConfig::new([channels, channels], [3, 3])
            .with_stride([2, 2])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
        Self { conv }
    }

    pub fn load_from(
        source: &dyn ModelSource,
        prefix: &str,
        channels: usize,
        device: &burn::tensor::Device<B>,
    ) -> Result<Self> {
        let conv = load_conv2d(source, prefix, [channels, channels], [3, 3], [2, 2], device)?;
        Ok(Self { conv })
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(x)
    }
}

/// Nearest-neighbor upsample + 3x3 convolution.
pub struct Upsample2D<B: Backend> {
    pub(crate) conv: Conv2d<B>,
}

impl<B: Backend> Upsample2D<B> {
    pub fn new(channels: usize, device: &burn::tensor::Device<B>) -> Self {
        let conv = Conv2dConfig::new([channels, channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
        Self { conv }
    }

    pub fn load_from(
        source: &dyn ModelSource,
        prefix: &str,
        channels: usize,
        device: &burn::tensor::Device<B>,
    ) -> Result<Self> {
        let conv = load_conv2d(source, prefix, [channels, channels], [3, 3], [1, 1], device)?;
        Ok(Self { conv })
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = x.repeat_dim(2, 2).repeat_dim(3, 2);
        self.conv.forward(x)
    }
}

/// Self-attention block used in the VAE mid block.
pub struct AttentionBlock2D<B: Backend> {
    pub(crate) group_norm: GroupNorm<B>,
    pub(crate) attn: Attention<B>,
}

impl<B: Backend> AttentionBlock2D<B> {
    pub fn new(channels: usize, qkv_bias: bool, device: &burn::tensor::Device<B>) -> Self {
        let group_norm = GroupNormConfig::new(norm_groups(channels), channels)
            .with_epsilon(1e-6)
            .init(device);
        let attn = Attention::new(channels, channels, 1, channels, qkv_bias, device);
        Self {
            group_norm,
            attn,
        }
    }

    pub fn load_from(
        source: &dyn ModelSource,
        prefix: &str,
        channels: usize,
        qkv_bias: bool,
        device: &burn::tensor::Device<B>,
    ) -> Result<Self> {
        let group_norm = load_group_norm(source, &format!("{prefix}.group_norm"), norm_groups(channels), channels, 1e-6, device)?;
        let attn = Attention::load_from(source, &format!("{prefix}"), channels, channels, 1, channels, qkv_bias, device)?;
        Ok(Self { group_norm, attn })
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
        let residual = x.clone();
        let h_norm = self.group_norm.forward(x);
        let h_seq = h_norm.reshape([b, c, h * w]).permute([0, 2, 1]);
        let h_seq = self.attn.forward(h_seq, None);
        let h_img = h_seq.permute([0, 2, 1]).reshape([b, c, h, w]);
        h_img.add(residual)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn norm_groups(channels: usize) -> usize {
    if channels >= NORM_GROUPS {
        NORM_GROUPS
    } else {
        channels.max(1)
    }
}
