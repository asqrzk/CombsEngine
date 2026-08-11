//! VAE decoder for Stable Diffusion 1.5 (AutoencoderKL decode branch).
//!
//! Converts a 4-channel latent `[B, 4, H, W]` into a 3-channel RGB image
//! `[B, 3, 8H, 8W]`. The architecture mirrors diffusers' `Decoder`:
//!   block_out_channels = [128, 256, 512, 512]
//!   layers_per_block   = 2   (decoder uses layers_per_block + 1)
//!   latent_channels    = 4
//!   out_channels       = 3
//!   norm_num_groups    = 32
//!   resnet eps         = 1e-6

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{GroupNorm, GroupNormConfig};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::blocks::{AttentionBlock2D, ResnetBlock2D, Upsample2D};
use crate::weights::{load_conv2d, load_group_norm};
use combs_formats::{ModelSource, Result};

const LATENT_CHANNELS: usize = 4;
const OUT_CHANNELS: usize = 3;
const BLOCK_OUT_CHANNELS: [usize; 4] = [128, 256, 512, 512];
const LAYERS_PER_BLOCK: usize = 2;
const NORM_GROUPS: usize = 32;
const NORM_EPS: f64 = 1e-6;

/// One decoder up block: a stack of ResNet blocks followed by an optional
/// nearest-neighbour upsample + conv.
struct UpDecoderBlock2D<B: Backend> {
    resnets: Vec<ResnetBlock2D<B>>,
    upsampler: Option<Upsample2D<B>>,
}

impl<B: Backend> UpDecoderBlock2D<B> {
    fn new(
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        add_upsample: bool,
        device: &burn::tensor::Device<B>,
    ) -> Self {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let res_in = if i == 0 { in_channels } else { out_channels };
            resnets.push(ResnetBlock2D::new(
                res_in,
                out_channels,
                None,
                NORM_EPS,
                device,
            ));
        }
        let upsampler = if add_upsample {
            Some(Upsample2D::new(out_channels, device))
        } else {
            None
        };
        Self { resnets, upsampler }
    }

    fn load_from(
        source: &dyn ModelSource,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        num_layers: usize,
        add_upsample: bool,
        device: &burn::tensor::Device<B>,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let res_in = if i == 0 { in_channels } else { out_channels };
            resnets.push(ResnetBlock2D::load_from(
                source,
                &format!("{prefix}.resnets.{i}"),
                res_in,
                out_channels,
                None,
                NORM_EPS,
                device,
            )?);
        }
        let upsampler = if add_upsample {
            Some(Upsample2D::load_from(
                source,
                &format!("{prefix}.upsamplers.0"),
                out_channels,
                device,
            )?)
        } else {
            None
        };
        Ok(Self { resnets, upsampler })
    }

    fn forward(&self, mut h: Tensor<B, 4>) -> Tensor<B, 4> {
        for resnet in &self.resnets {
            h = resnet.forward(h, None);
        }
        if let Some(ref us) = self.upsampler {
            h = us.forward(h);
        }
        h
    }
}

/// Mid block for the decoder: ResNet -> self-attention -> ResNet.
struct MidBlock<B: Backend> {
    resnet_1: ResnetBlock2D<B>,
    attention: AttentionBlock2D<B>,
    resnet_2: ResnetBlock2D<B>,
}

impl<B: Backend> MidBlock<B> {
    fn new(channels: usize, device: &burn::tensor::Device<B>) -> Self {
        Self {
            resnet_1: ResnetBlock2D::new(channels, channels, None, NORM_EPS, device),
            attention: AttentionBlock2D::new(channels, true, device),
            resnet_2: ResnetBlock2D::new(channels, channels, None, NORM_EPS, device),
        }
    }

    fn load_from(
        source: &dyn ModelSource,
        prefix: &str,
        channels: usize,
        device: &burn::tensor::Device<B>,
    ) -> Result<Self> {
        Ok(Self {
            resnet_1: ResnetBlock2D::load_from(
                source,
                &format!("{prefix}.resnets.0"),
                channels,
                channels,
                None,
                NORM_EPS,
                device,
            )?,
            attention: AttentionBlock2D::load_from(
                source,
                &format!("{prefix}.attentions.0"),
                channels,
                true,
                device,
            )?,
            resnet_2: ResnetBlock2D::load_from(
                source,
                &format!("{prefix}.resnets.1"),
                channels,
                channels,
                None,
                NORM_EPS,
                device,
            )?,
        })
    }

    fn forward(&self, h: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.resnet_1.forward(h, None);
        let h = self.attention.forward(h);
        self.resnet_2.forward(h, None)
    }
}

/// Stable Diffusion 1.5 VAE decoder.
pub struct VAEDecoder<B: Backend> {
    /// AutoencoderKL's top-level 1×1 latent projection, applied BEFORE the
    /// decoder proper. `None` only for random-init scaffolds.
    post_quant_conv: Option<Conv2d<B>>,
    conv_in: Conv2d<B>,
    mid_block: MidBlock<B>,
    up_blocks: Vec<UpDecoderBlock2D<B>>,
    conv_norm_out: GroupNorm<B>,
    conv_out: Conv2d<B>,
}

impl<B: Backend> VAEDecoder<B> {
    /// Build a randomly initialized SD 1.5 VAE decoder.
    pub fn new(device: &burn::tensor::Device<B>) -> Self {
        let conv_in = Conv2dConfig::new([LATENT_CHANNELS, BLOCK_OUT_CHANNELS[3]], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);

        let mid_block = MidBlock::new(BLOCK_OUT_CHANNELS[3], device);

        let reversed: Vec<usize> = BLOCK_OUT_CHANNELS.iter().rev().copied().collect();
        let mut up_blocks = Vec::with_capacity(reversed.len());
        let num_layers = LAYERS_PER_BLOCK + 1;
        let n_blocks = reversed.len();
        for (i, &out_ch) in reversed.iter().enumerate() {
            let prev_ch = if i == 0 {
                reversed[0]
            } else {
                reversed[i - 1]
            };
            let add_upsample = i < n_blocks - 1;
            up_blocks.push(UpDecoderBlock2D::new(
                prev_ch,
                out_ch,
                num_layers,
                add_upsample,
                device,
            ));
        }

        let conv_norm_out = GroupNormConfig::new(NORM_GROUPS, BLOCK_OUT_CHANNELS[0])
            .with_epsilon(NORM_EPS)
            .init(device);
        let conv_out = Conv2dConfig::new([BLOCK_OUT_CHANNELS[0], OUT_CHANNELS], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);

        Self {
            post_quant_conv: None,
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
        }
    }

    /// Load SD 1.5 VAE decoder weights from a `ModelSource`.
    pub fn load_from(source: &dyn ModelSource, device: &burn::tensor::Device<B>) -> Result<Self> {
        // AutoencoderKL applies a 1x1 `post_quant_conv` to the latent before
        // the decoder; skipping it decodes an un-projected latent. Absent
        // only from decoder-only extracts — warn, don't fail.
        let post_quant_conv = if source
            .tensor_names()
            .contains(&"post_quant_conv.weight".to_string())
        {
            Some(load_conv2d(
                source,
                "post_quant_conv",
                [LATENT_CHANNELS, LATENT_CHANNELS],
                [1, 1],
                [1, 1],
                device,
            )?)
        } else {
            eprintln!(
                "[vae] warning: checkpoint has no post_quant_conv — decoding \
                 raw latents (image quality will suffer)"
            );
            None
        };

        let conv_in = load_conv2d(
            source,
            "decoder.conv_in",
            [LATENT_CHANNELS, BLOCK_OUT_CHANNELS[3]],
            [3, 3],
            [1, 1],
            device,
        )?;

        let mid_block = MidBlock::load_from(
            source,
            "decoder.mid_block",
            BLOCK_OUT_CHANNELS[3],
            device,
        )?;

        let reversed: Vec<usize> = BLOCK_OUT_CHANNELS.iter().rev().copied().collect();
        let mut up_blocks = Vec::with_capacity(reversed.len());
        let num_layers = LAYERS_PER_BLOCK + 1;
        let n_blocks = reversed.len();
        for (i, &out_ch) in reversed.iter().enumerate() {
            let prev_ch = if i == 0 {
                reversed[0]
            } else {
                reversed[i - 1]
            };
            let add_upsample = i < n_blocks - 1;
            up_blocks.push(UpDecoderBlock2D::load_from(
                source,
                &format!("decoder.up_blocks.{i}"),
                prev_ch,
                out_ch,
                num_layers,
                add_upsample,
                device,
            )?);
        }

        let conv_norm_out = load_group_norm(
            source,
            "decoder.conv_norm_out",
            NORM_GROUPS,
            BLOCK_OUT_CHANNELS[0],
            NORM_EPS,
            device,
        )?;
        let conv_out = load_conv2d(
            source,
            "decoder.conv_out",
            [BLOCK_OUT_CHANNELS[0], OUT_CHANNELS],
            [3, 3],
            [1, 1],
            device,
        )?;

        Ok(Self {
            post_quant_conv,
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
        })
    }

    /// Decode a latent tensor into an RGB image.
    ///
    /// - input:  `[batch, 4, height, width]`
    /// - output: `[batch, 3, height * 8, width * 8]`
    pub fn forward(&self, latent: Tensor<B, 4>) -> Tensor<B, 4> {
        let latent = match &self.post_quant_conv {
            Some(conv) => conv.forward(latent),
            None => latent,
        };
        let mut h = self.conv_in.forward(latent);
        h = self.mid_block.forward(h);
        for up in &self.up_blocks {
            h = up.forward(h);
        }
        h = silu(self.conv_norm_out.forward(h));
        self.conv_out.forward(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vae_decoder_runs() {
        type B = burn::backend::NdArray;
        let device = Default::default();
        let vae = VAEDecoder::<B>::new(&device);

        let latent = Tensor::zeros([1, 4, 8, 8], &device);
        let image = vae.forward(latent);
        assert_eq!(image.dims(), [1, 3, 64, 64]);
    }

    #[test]
    fn vae_decoder_512_output() {
        type B = burn::backend::NdArray;
        let device = Default::default();
        let vae = VAEDecoder::<B>::new(&device);

        let latent = Tensor::zeros([1, 4, 64, 64], &device);
        let image = vae.forward(latent);
        assert_eq!(image.dims(), [1, 3, 512, 512]);
    }
}
