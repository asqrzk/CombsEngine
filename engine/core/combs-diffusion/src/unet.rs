//! UNet2DConditionModel for Stable Diffusion 1.5, built on Burn.

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{GroupNorm, GroupNormConfig};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::blocks::{Downsample2D, ResnetBlock2D, SpatialTransformer, Upsample2D};
use crate::time::{timestep_embedding, TimeEmbedding};
use crate::weights::{load_conv2d, load_group_norm};
use combs_formats::{ModelSource, Result};

const TIME_EMBED_DIM: usize = 1280;
const CONTEXT_DIM: usize = 768;
const NORM_GROUPS: usize = 32;
const NORM_EPS: f64 = 1e-5;

/// One down block with optional cross-attention transformer(s).
enum DownBlock<B: Backend> {
    Basic(BasicDownBlock<B>),
    CrossAttn(CrossAttnDownBlock<B>),
}

struct BasicDownBlock<B: Backend> {
    resnets: Vec<ResnetBlock2D<B>>,
    downsampler: Option<Downsample2D<B>>,
}

struct CrossAttnDownBlock<B: Backend> {
    resnets: Vec<ResnetBlock2D<B>>,
    attentions: Vec<SpatialTransformer<B>>,
    downsampler: Option<Downsample2D<B>>,
}

/// One up block with optional cross-attention transformer(s).
enum UpBlock<B: Backend> {
    Basic(BasicUpBlock<B>),
    CrossAttn(CrossAttnUpBlock<B>),
}

struct BasicUpBlock<B: Backend> {
    resnets: Vec<ResnetBlock2D<B>>,
    upsampler: Option<Upsample2D<B>>,
}

struct CrossAttnUpBlock<B: Backend> {
    resnets: Vec<ResnetBlock2D<B>>,
    attentions: Vec<SpatialTransformer<B>>,
    upsampler: Option<Upsample2D<B>>,
}

struct MidBlock<B: Backend> {
    resnet_1: ResnetBlock2D<B>,
    attention: SpatialTransformer<B>,
    resnet_2: ResnetBlock2D<B>,
}

/// Full SD 1.5 UNet with cross-attention conditioning.
pub struct UNet2DConditionModel<B: Backend> {
    conv_in: Conv2d<B>,
    time_emb: TimeEmbedding<B>,
    down_blocks: Vec<DownBlock<B>>,
    mid_block: MidBlock<B>,
    up_blocks: Vec<UpBlock<B>>,
    conv_norm_out: GroupNorm<B>,
    conv_out: Conv2d<B>,
}

impl<B: Backend> UNet2DConditionModel<B> {
    /// Build a fresh (randomly initialized) SD 1.5 UNet.
    pub fn new(device: &burn::tensor::Device<B>) -> Self {
        let block_out_channels = [320, 640, 1280, 1280];
        let down_use_cross_attn = [true, true, true, false];
        let up_use_cross_attn = [false, true, true, true];
        let layers_per_block = 2;
        let n_blocks = block_out_channels.len();

        let conv_in = Conv2dConfig::new([4, block_out_channels[0]], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);
        let time_emb = TimeEmbedding::new(block_out_channels[0], TIME_EMBED_DIM, device);

        // Down blocks.
        let mut down_blocks = Vec::with_capacity(n_blocks);
        for (i, &out_ch) in block_out_channels.iter().enumerate() {
            let in_ch = if i == 0 {
                block_out_channels[0]
            } else {
                block_out_channels[i - 1]
            };
            let add_downsample = i < n_blocks - 1;
            if down_use_cross_attn[i] {
                down_blocks.push(DownBlock::CrossAttn(CrossAttnDownBlock::new(
                    in_ch,
                    out_ch,
                    TIME_EMBED_DIM,
                    add_downsample,
                    layers_per_block,
                    device,
                )));
            } else {
                down_blocks.push(DownBlock::Basic(BasicDownBlock::new(
                    in_ch,
                    out_ch,
                    TIME_EMBED_DIM,
                    add_downsample,
                    layers_per_block,
                    device,
                )));
            }
        }

        // Mid block.
        let mid_ch = block_out_channels[n_blocks - 1];
        let mid_block = MidBlock {
            resnet_1: ResnetBlock2D::new(
                mid_ch,
                mid_ch,
                Some(TIME_EMBED_DIM),
                NORM_EPS,
                device,
            ),
            attention: SpatialTransformer::new(
                mid_ch,
                CONTEXT_DIM,
                8,
                mid_ch / 8,
                device,
            ),
            resnet_2: ResnetBlock2D::new(
                mid_ch,
                mid_ch,
                Some(TIME_EMBED_DIM),
                NORM_EPS,
                device,
            ),
        };

        // Up blocks: reversed channel list.
        let reversed: Vec<usize> = block_out_channels.iter().rev().copied().collect();
        let mut up_blocks = Vec::with_capacity(n_blocks);
        for (i, &out_ch) in reversed.iter().enumerate() {
            let prev_out_ch = if i == 0 {
                mid_ch
            } else {
                reversed[i - 1]
            };
            let in_ch = if i == n_blocks - 1 {
                block_out_channels[0]
            } else {
                reversed[i + 1]
            };
            let add_upsample = i < n_blocks - 1;
            if up_use_cross_attn[i] {
                up_blocks.push(UpBlock::CrossAttn(CrossAttnUpBlock::new(
                    in_ch,
                    prev_out_ch,
                    out_ch,
                    TIME_EMBED_DIM,
                    add_upsample,
                    layers_per_block + 1,
                    device,
                )));
            } else {
                up_blocks.push(UpBlock::Basic(BasicUpBlock::new(
                    in_ch,
                    prev_out_ch,
                    out_ch,
                    TIME_EMBED_DIM,
                    add_upsample,
                    layers_per_block + 1,
                    device,
                )));
            }
        }

        let conv_norm_out = GroupNormConfig::new(NORM_GROUPS, block_out_channels[0])
            .with_epsilon(NORM_EPS)
            .init(device);
        let conv_out = Conv2dConfig::new([block_out_channels[0], 4], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);

        Self {
            conv_in,
            time_emb,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
        }
    }

    /// Load SD 1.5 UNet weights from a `ModelSource`.
    pub fn load_from(source: &dyn ModelSource, device: &burn::tensor::Device<B>) -> Result<Self> {
        let block_out_channels = [320, 640, 1280, 1280];
        let down_use_cross_attn = [true, true, true, false];
        let up_use_cross_attn = [false, true, true, true];
        let layers_per_block = 2;
        let n_blocks = block_out_channels.len();

        let conv_in = load_conv2d(source, "conv_in", [4, block_out_channels[0]], [3, 3], [1, 1], device)?;
        let time_emb = TimeEmbedding::load_from(
            source,
            "time_embedding",
            block_out_channels[0],
            TIME_EMBED_DIM,
            device,
        )?;

        // Down blocks.
        let mut down_blocks = Vec::with_capacity(n_blocks);
        for (i, &out_ch) in block_out_channels.iter().enumerate() {
            let in_ch = if i == 0 {
                block_out_channels[0]
            } else {
                block_out_channels[i - 1]
            };
            let add_downsample = i < n_blocks - 1;
            let prefix = format!("down_blocks.{i}");
            if down_use_cross_attn[i] {
                down_blocks.push(DownBlock::CrossAttn(CrossAttnDownBlock::load_from(
                    source,
                    &prefix,
                    in_ch,
                    out_ch,
                    TIME_EMBED_DIM,
                    add_downsample,
                    layers_per_block,
                    device,
                )?));
            } else {
                down_blocks.push(DownBlock::Basic(BasicDownBlock::load_from(
                    source,
                    &prefix,
                    in_ch,
                    out_ch,
                    TIME_EMBED_DIM,
                    add_downsample,
                    layers_per_block,
                    device,
                )?));
            }
        }

        // Mid block.
        let mid_ch = block_out_channels[n_blocks - 1];
        let mid_block = MidBlock {
            resnet_1: ResnetBlock2D::load_from(
                source,
                "mid_block.resnets.0",
                mid_ch,
                mid_ch,
                Some(TIME_EMBED_DIM),
                NORM_EPS,
                device,
            )?,
            attention: SpatialTransformer::load_from(
                source,
                "mid_block.attentions.0",
                mid_ch,
                CONTEXT_DIM,
                8,
                mid_ch / 8,
                device,
            )?,
            resnet_2: ResnetBlock2D::load_from(
                source,
                "mid_block.resnets.1",
                mid_ch,
                mid_ch,
                Some(TIME_EMBED_DIM),
                NORM_EPS,
                device,
            )?,
        };

        // Up blocks.
        let reversed: Vec<usize> = block_out_channels.iter().rev().copied().collect();
        let mut up_blocks = Vec::with_capacity(n_blocks);
        for (i, &out_ch) in reversed.iter().enumerate() {
            let prev_out_ch = if i == 0 {
                mid_ch
            } else {
                reversed[i - 1]
            };
            let in_ch = if i == n_blocks - 1 {
                block_out_channels[0]
            } else {
                reversed[i + 1]
            };
            let add_upsample = i < n_blocks - 1;
            let prefix = format!("up_blocks.{i}");
            if up_use_cross_attn[i] {
                up_blocks.push(UpBlock::CrossAttn(CrossAttnUpBlock::load_from(
                    source,
                    &prefix,
                    in_ch,
                    prev_out_ch,
                    out_ch,
                    TIME_EMBED_DIM,
                    add_upsample,
                    layers_per_block + 1,
                    device,
                )?));
            } else {
                up_blocks.push(UpBlock::Basic(BasicUpBlock::load_from(
                    source,
                    &prefix,
                    in_ch,
                    prev_out_ch,
                    out_ch,
                    TIME_EMBED_DIM,
                    add_upsample,
                    layers_per_block + 1,
                    device,
                )?));
            }
        }

        let conv_norm_out = load_group_norm(
            source,
            "conv_norm_out",
            NORM_GROUPS,
            block_out_channels[0],
            NORM_EPS,
            device,
        )?;
        let conv_out = load_conv2d(
            source,
            "conv_out",
            [block_out_channels[0], 4],
            [3, 3],
            [1, 1],
            device,
        )?;

        Ok(Self {
            conv_in,
            time_emb,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
        })
    }

    pub fn forward(
        &self,
        latent: Tensor<B, 4>,
        timestep: f32,
        context: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        self.forward_traced(latent, timestep, context).0
    }

    /// Forward with per-stage taps for the torch parity harness. The plain
    /// [`Self::forward`] delegates here, so the traced path can never drift
    /// from the real one. Taps are cheap clones of Arc-backed tensors.
    #[doc(hidden)]
    pub fn forward_traced(
        &self,
        latent: Tensor<B, 4>,
        timestep: f32,
        context: Tensor<B, 3>,
    ) -> (Tensor<B, 4>, Vec<(String, Tensor<B, 4>)>) {
        let mut taps: Vec<(String, Tensor<B, 4>)> = Vec::new();
        let [b, _c, _h, _w] = latent.dims();
        // One timestep entry per batch row: the resnet time projection is
        // reshaped to [batch, C, 1, 1], so CFG's batched [uncond; cond]
        // pass needs a batch-sized embedding.
        let t = Tensor::from_data(
            burn::tensor::TensorData::from(vec![timestep as i64; b].as_slice()),
            &latent.device(),
        );
        let t_emb = timestep_embedding(&t, 320, &latent.device());
        let t_emb = self.time_emb.forward(t_emb);

        let mut h = self.conv_in.forward(latent);
        taps.push(("conv_in".into(), h.clone()));
        let mut skip_connections: Vec<Tensor<B, 4>> = vec![h.clone()];

        // Down.
        for (i, down) in self.down_blocks.iter().enumerate() {
            let (h_out, mut states) = match down {
                DownBlock::Basic(db) => db.forward(h, &t_emb),
                DownBlock::CrossAttn(cb) => cb.forward(h, &t_emb, &context),
            };
            h = h_out;
            taps.push((format!("down{i}"), h.clone()));
            skip_connections.append(&mut states);
        }

        // Mid.
        h = self.mid_block.resnet_1.forward(h, Some(&t_emb));
        h = self.mid_block.attention.forward(h, &context);
        h = self.mid_block.resnet_2.forward(h, Some(&t_emb));
        taps.push(("mid".into(), h.clone()));

        // Up.
        for (i, up) in self.up_blocks.iter().enumerate() {
            let n_resnets = match up {
                UpBlock::Basic(ub) => ub.resnets.len(),
                UpBlock::CrossAttn(cub) => cub.resnets.len(),
            };
            let mut res_xs: Vec<Tensor<B, 4>> = Vec::with_capacity(n_resnets);
            for _ in 0..n_resnets {
                res_xs.push(skip_connections.pop().expect("missing skip"));
            }
            h = match up {
                UpBlock::Basic(ub) => ub.forward(h, &res_xs, &t_emb),
                UpBlock::CrossAttn(cub) => cub.forward(h, &res_xs, &t_emb, &context),
            };
            taps.push((format!("up{i}"), h.clone()));
        }

        let h = silu(self.conv_norm_out.forward(h));
        (self.conv_out.forward(h), taps)
    }
}

// ---------------------------------------------------------------------------
// Down / mid / up block helpers.
// ---------------------------------------------------------------------------

impl<B: Backend> BasicDownBlock<B> {
    fn new(
        in_channels: usize,
        out_channels: usize,
        time_emb_dim: usize,
        add_downsample: bool,
        num_layers: usize,
        device: &burn::tensor::Device<B>,
    ) -> Self {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let res_in = if i == 0 { in_channels } else { out_channels };
            resnets.push(ResnetBlock2D::new(
                res_in,
                out_channels,
                Some(time_emb_dim),
                NORM_EPS,
                device,
            ));
        }
        let downsampler = if add_downsample {
            Some(Downsample2D::new(out_channels, device))
        } else {
            None
        };
        Self {
            resnets,
            downsampler,
        }
    }

    fn load_from(
        source: &dyn ModelSource,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        time_emb_dim: usize,
        add_downsample: bool,
        num_layers: usize,
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
                Some(time_emb_dim),
                NORM_EPS,
                device,
            )?);
        }
        let downsampler = if add_downsample {
            Some(Downsample2D::load_from(
                source,
                &format!("{prefix}.downsamplers.0"),
                out_channels,
                device,
            )?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            downsampler,
        })
    }

    fn forward(&self, mut h: Tensor<B, 4>, t_emb: &Tensor<B, 2>) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        let mut states = Vec::with_capacity(self.resnets.len() + 1);
        for resnet in &self.resnets {
            h = resnet.forward(h, Some(t_emb));
            states.push(h.clone());
        }
        if let Some(ref ds) = self.downsampler {
            h = ds.forward(h);
            states.push(h.clone());
        }
        (h, states)
    }
}

impl<B: Backend> CrossAttnDownBlock<B> {
    fn new(
        in_channels: usize,
        out_channels: usize,
        time_emb_dim: usize,
        add_downsample: bool,
        num_layers: usize,
        device: &burn::tensor::Device<B>,
    ) -> Self {
        let mut resnets = Vec::with_capacity(num_layers);
        let mut attentions = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let res_in = if i == 0 { in_channels } else { out_channels };
            resnets.push(ResnetBlock2D::new(
                res_in,
                out_channels,
                Some(time_emb_dim),
                NORM_EPS,
                device,
            ));
            attentions.push(SpatialTransformer::new(
                out_channels,
                CONTEXT_DIM,
                8,
                out_channels / 8,
                device,
            ));
        }
        let downsampler = if add_downsample {
            Some(Downsample2D::new(out_channels, device))
        } else {
            None
        };
        Self {
            resnets,
            attentions,
            downsampler,
        }
    }

    fn load_from(
        source: &dyn ModelSource,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        time_emb_dim: usize,
        add_downsample: bool,
        num_layers: usize,
        device: &burn::tensor::Device<B>,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        let mut attentions = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let res_in = if i == 0 { in_channels } else { out_channels };
            resnets.push(ResnetBlock2D::load_from(
                source,
                &format!("{prefix}.resnets.{i}"),
                res_in,
                out_channels,
                Some(time_emb_dim),
                NORM_EPS,
                device,
            )?);
            attentions.push(SpatialTransformer::load_from(
                source,
                &format!("{prefix}.attentions.{i}"),
                out_channels,
                CONTEXT_DIM,
                8,
                out_channels / 8,
                device,
            )?);
        }
        let downsampler = if add_downsample {
            Some(Downsample2D::load_from(
                source,
                &format!("{prefix}.downsamplers.0"),
                out_channels,
                device,
            )?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            attentions,
            downsampler,
        })
    }

    fn forward(
        &self,
        mut h: Tensor<B, 4>,
        t_emb: &Tensor<B, 2>,
        context: &Tensor<B, 3>,
    ) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        let mut states = Vec::with_capacity(self.resnets.len() + 1);
        for (resnet, attn) in self.resnets.iter().zip(self.attentions.iter()) {
            h = resnet.forward(h, Some(t_emb));
            h = attn.forward(h, context);
            states.push(h.clone());
        }
        if let Some(ref ds) = self.downsampler {
            h = ds.forward(h);
            states.push(h.clone());
        }
        (h, states)
    }
}

impl<B: Backend> BasicUpBlock<B> {
    fn new(
        in_channels: usize,
        prev_output_channels: usize,
        out_channels: usize,
        time_emb_dim: usize,
        add_upsample: bool,
        num_layers: usize,
        device: &burn::tensor::Device<B>,
    ) -> Self {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let res_skip = if i == num_layers - 1 { in_channels } else { out_channels };
            let res_in = if i == 0 { prev_output_channels } else { out_channels };
            resnets.push(ResnetBlock2D::new(
                res_in + res_skip,
                out_channels,
                Some(time_emb_dim),
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
        prev_output_channels: usize,
        out_channels: usize,
        time_emb_dim: usize,
        add_upsample: bool,
        num_layers: usize,
        device: &burn::tensor::Device<B>,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let res_skip = if i == num_layers - 1 { in_channels } else { out_channels };
            let res_in = if i == 0 { prev_output_channels } else { out_channels };
            resnets.push(ResnetBlock2D::load_from(
                source,
                &format!("{prefix}.resnets.{i}"),
                res_in + res_skip,
                out_channels,
                Some(time_emb_dim),
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

    fn forward(
        &self,
        mut h: Tensor<B, 4>,
        res_xs: &[Tensor<B, 4>],
        t_emb: &Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        for (i, resnet) in self.resnets.iter().enumerate() {
            let skip = &res_xs[i];
            h = Tensor::cat(vec![h, skip.clone()], 1);
            h = resnet.forward(h, Some(t_emb));
        }
        if let Some(ref us) = self.upsampler {
            h = us.forward(h);
        }
        h
    }
}

impl<B: Backend> CrossAttnUpBlock<B> {
    fn new(
        in_channels: usize,
        prev_output_channels: usize,
        out_channels: usize,
        time_emb_dim: usize,
        add_upsample: bool,
        num_layers: usize,
        device: &burn::tensor::Device<B>,
    ) -> Self {
        let mut resnets = Vec::with_capacity(num_layers);
        let mut attentions = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let res_skip = if i == num_layers - 1 { in_channels } else { out_channels };
            let res_in = if i == 0 { prev_output_channels } else { out_channels };
            resnets.push(ResnetBlock2D::new(
                res_in + res_skip,
                out_channels,
                Some(time_emb_dim),
                NORM_EPS,
                device,
            ));
            attentions.push(SpatialTransformer::new(
                out_channels,
                CONTEXT_DIM,
                8,
                out_channels / 8,
                device,
            ));
        }
        let upsampler = if add_upsample {
            Some(Upsample2D::new(out_channels, device))
        } else {
            None
        };
        Self {
            resnets,
            attentions,
            upsampler,
        }
    }

    fn load_from(
        source: &dyn ModelSource,
        prefix: &str,
        in_channels: usize,
        prev_output_channels: usize,
        out_channels: usize,
        time_emb_dim: usize,
        add_upsample: bool,
        num_layers: usize,
        device: &burn::tensor::Device<B>,
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        let mut attentions = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let res_skip = if i == num_layers - 1 { in_channels } else { out_channels };
            let res_in = if i == 0 { prev_output_channels } else { out_channels };
            resnets.push(ResnetBlock2D::load_from(
                source,
                &format!("{prefix}.resnets.{i}"),
                res_in + res_skip,
                out_channels,
                Some(time_emb_dim),
                NORM_EPS,
                device,
            )?);
            attentions.push(SpatialTransformer::load_from(
                source,
                &format!("{prefix}.attentions.{i}"),
                out_channels,
                CONTEXT_DIM,
                8,
                out_channels / 8,
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
        Ok(Self {
            resnets,
            attentions,
            upsampler,
        })
    }

    fn forward(
        &self,
        mut h: Tensor<B, 4>,
        res_xs: &[Tensor<B, 4>],
        t_emb: &Tensor<B, 2>,
        context: &Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        for (i, resnet) in self.resnets.iter().enumerate() {
            let skip = &res_xs[i];
            h = Tensor::cat(vec![h, skip.clone()], 1);
            h = resnet.forward(h, Some(t_emb));
            h = self.attentions[i].forward(h, context);
        }
        if let Some(ref us) = self.upsampler {
            h = us.forward(h);
        }
        h
    }
}
