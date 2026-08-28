//! FLUX.2 flow-matching transformer (the klein family).
//!
//! MM-DiT with double-stream blocks (separate image/text weights,
//! joint attention) followed by single-stream blocks (fused parallel
//! QKV+MLP, ViT-22B style) over the text-first concatenated sequence.
//! Faithful to the diffusers `Flux2Transformer2DModel` semantics:
//! modulation is GLOBAL — one (img, txt) pair of shift/scale/gate sets
//! and one single-stream set computed from the time embedding and
//! shared by every block — all linears are bias-free, norms inside
//! blocks are affine-free LayerNorms, q/k get per-head affine RMSNorm,
//! and RoPE is 4-axis over (T, H, W, L) ids with interleaved-pair
//! rotation. Reference-image KV caching is out of scope here.

use burn::nn::Linear;
use burn::tensor::activation::{silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};
use combs_formats::{FormatError, ModelSource, QuantFormat, QuantTensor, Result, TokenizerSpec};
use combs_models::Linear as BlockLinear;

use crate::weights::{load_linear, load_param, load_tensor};

/// A source decorator that packs big rank-2 float tensors into Q8_0 on
/// demand, so [`combs_models::try_quant_linear`] can bind them to the
/// canaried quant kernels. This is how a bf16 DiT checkpoint runs on
/// the f32 backend at 4.3 GB instead of 15.5: packed weights, f32
/// activations — no f16 range cliffs, no Metal overcommit (which
/// hands back buffers that silently READ AS ZERO instead of failing).
pub struct QuantizingSource<'a> {
    inner: &'a dyn ModelSource,
}

impl<'a> QuantizingSource<'a> {
    pub fn new(inner: &'a dyn ModelSource) -> Self {
        Self { inner }
    }
}

impl ModelSource for QuantizingSource<'_> {
    fn metadata(&self) -> &combs_formats::ModelMetadata {
        self.inner.metadata()
    }
    fn tensor_names(&self) -> Vec<String> {
        self.inner.tensor_names()
    }
    fn open_tensor(&self, name: &str) -> Result<combs_formats::TensorReader<'_>> {
        self.inner.open_tensor(name)
    }
    fn tokenizer(&self) -> Result<TokenizerSpec> {
        self.inner.tokenizer()
    }
    fn sampler_defaults(&self) -> Option<combs_formats::SamplerConfig> {
        self.inner.sampler_defaults()
    }
    fn open_tensor_quant(&self, name: &str) -> Result<Option<QuantTensor<'_>>> {
        // Pass through anything already packed.
        if let Some(qt) = self.inner.open_tensor_quant(name)? {
            return Ok(Some(qt));
        }
        let reader = self.inner.open_tensor(name)?;
        let shape = reader.shape().to_vec();
        let [n_out, k] = shape[..] else { return Ok(None) };
        if k % 32 != 0 || n_out * k < (1 << 20) {
            return Ok(None);
        }
        let values: Vec<f32> = reader
            .load_data()?
            .to_vec()
            .map_err(|e| FormatError::Safetensors(format!("quantize {name}: {e:?}")))?;
        let packed = combs_formats::quants::quantize_q8_0(&values)?;
        Ok(Some(QuantTensor {
            format: QuantFormat::Q8_0,
            shape,
            data: std::borrow::Cow::Owned(packed),
        }))
    }
}

/// A block linear: the quant kernels when the source serves packed
/// blocks and the backend has them, the dense house matmul otherwise.
fn load_block_linear<B: Backend>(
    source: &dyn ModelSource,
    prefix: &str,
    k: usize,
    n_out: usize,
    device: &Device<B>,
) -> Result<BlockLinear<B>> {
    let name = format!("{prefix}.weight");
    match combs_models::try_quant_linear::<B>(source, &name, device) {
        Ok(Some(op)) => return Ok(BlockLinear::Quant(op)),
        Ok(None) => {}
        Err(e) => {
            return Err(FormatError::Safetensors(format!("quant bind {name}: {e}")));
        }
    }
    let w: Tensor<B, 2> = load_tensor(source, &name, device)?;
    crate::weights::expect_shape(&name, &w.dims(), &[n_out, k])?;
    Ok(BlockLinear::Dense(w))
}

/// Geometry of a FLUX.2 transformer checkpoint.
#[derive(Debug, Clone)]
pub struct Flux2Config {
    pub in_channels: usize,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub attention_head_dim: usize,
    pub num_attention_heads: usize,
    pub joint_attention_dim: usize,
    pub mlp_ratio: usize,
    pub axes_dims_rope: Vec<usize>,
    pub rope_theta: f64,
    pub eps: f64,
    /// Sinusoidal width of the timestep projection.
    pub timestep_channels: usize,
}

impl Flux2Config {
    /// FLUX.2 [klein] 4B geometry (Apache-2.0 line).
    pub fn klein_4b() -> Self {
        Self {
            in_channels: 128,
            num_layers: 5,
            num_single_layers: 20,
            attention_head_dim: 128,
            num_attention_heads: 24,
            joint_attention_dim: 7680,
            mlp_ratio: 3,
            axes_dims_rope: vec![32, 32, 32, 32],
            rope_theta: 2000.0,
            eps: 1e-6,
            timestep_channels: 256,
        }
    }

    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Read the diffusers `transformer/config.json` shape, falling
    /// back to klein-4B values for absent keys.
    pub fn from_json(config: &serde_json::Value) -> Self {
        let d = Self::klein_4b();
        let get = |key: &str, fallback: usize| {
            config.get(key).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(fallback)
        };
        Self {
            in_channels: get("in_channels", d.in_channels),
            num_layers: get("num_layers", d.num_layers),
            num_single_layers: get("num_single_layers", d.num_single_layers),
            attention_head_dim: get("attention_head_dim", d.attention_head_dim),
            num_attention_heads: get("num_attention_heads", d.num_attention_heads),
            joint_attention_dim: get("joint_attention_dim", d.joint_attention_dim),
            mlp_ratio: config
                .get("mlp_ratio")
                .and_then(|v| v.as_f64())
                .map(|v| v as usize)
                .unwrap_or(d.mlp_ratio),
            axes_dims_rope: config
                .get("axes_dims_rope")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect())
                .unwrap_or(d.axes_dims_rope),
            rope_theta: config
                .get("rope_theta")
                .and_then(|v| v.as_f64())
                .unwrap_or(d.rope_theta),
            eps: d.eps,
            timestep_channels: get("timestep_guidance_channels", d.timestep_channels),
        }
    }
}

/// LayerNorm without affine parameters (torch biased-variance form).
///
/// Rows are pre-scaled by their absolute maximum before any square or
/// sum: the text stream carries |x| ~ 1.6e4 outliers (Qwen3 residual
/// sinks), and x² or a 3072-wide sum overflows f16 to inf — the whole
/// conditioning pathway then silently normalizes to ZERO on the f16
/// backend. Normalization is scale-invariant, so dividing by the max
/// first is mathematically free and keeps every intermediate bounded
/// by the head/row width. (eps lands on the scaled variance; its only
/// job — guarding all-zero rows — is done by the max clamp instead.)
fn layer_norm<B: Backend>(x: Tensor<B, 3>, _eps: f64) -> Tensor<B, 3> {
    let m = x.clone().abs().max_dim(2).clamp_min(1e-3);
    let y = x / m;
    let mean = y.clone().mean_dim(2);
    let centered = y - mean;
    let var = centered.clone().powf_scalar(2.0).mean_dim(2);
    centered / (var + 1e-12).sqrt()
}

/// Per-head RMSNorm over the trailing head dim of `[b, s, heads, d]`,
/// with an affine weight of shape `[d]`. Same max-prescaling as
/// [`layer_norm`] — q/k projections of the text stream overflow f16
/// squares without it.
fn rms_norm_head<B: Backend>(x: Tensor<B, 4>, weight: &Tensor<B, 1>, _eps: f64) -> Tensor<B, 4> {
    let m = x.clone().abs().max_dim(3).clamp_min(1e-3);
    let y = x / m;
    let ms = y.clone().powf_scalar(2.0).mean_dim(3);
    let normed = y / (ms + 1e-12).sqrt();
    normed * weight.clone().reshape([1, 1, 1, weight.dims()[0]])
}

/// Rotary tables for a token sequence: half-width cos/sin `[s, d/2]`
/// computed host-side in f64 (the reference uses f64 frequencies).
/// Each id is a 4-axis coordinate; axis i contributes
/// `axes_dims[i] / 2` frequencies `1 / theta^(2j / axes_dims[i])`.
pub fn rope_tables<B: Backend>(
    ids: &[[f64; 4]],
    axes_dims: &[usize],
    theta: f64,
    device: &Device<B>,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let half: usize = axes_dims.iter().map(|d| d / 2).sum();
    let mut cos = Vec::with_capacity(ids.len() * half);
    let mut sin = Vec::with_capacity(ids.len() * half);
    for id in ids {
        for (axis, &dim) in axes_dims.iter().enumerate() {
            for j in 0..dim / 2 {
                let freq = 1.0 / theta.powf(2.0 * j as f64 / dim as f64);
                let angle = id[axis] * freq;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
    }
    let shape = [ids.len(), half];
    (
        Tensor::from_data(burn::tensor::TensorData::new(cos, shape), device),
        Tensor::from_data(burn::tensor::TensorData::new(sin, shape), device),
    )
}

/// Interleaved-pair rotation: even' = even·cos − odd·sin,
/// odd' = odd·cos + even·sin (the flux `use_real_unbind_dim=-1` form).
fn apply_rope<B: Backend>(
    x: Tensor<B, 4>,
    cos: &Tensor<B, 2>,
    sin: &Tensor<B, 2>,
) -> Tensor<B, 4> {
    let [b, s, h, d] = x.dims();
    let x = x.reshape([b, s, h, d / 2, 2]);
    let x0 = x.clone().narrow(4, 0, 1);
    let x1 = x.narrow(4, 1, 1);
    let c = cos.clone().reshape([1, s, 1, d / 2, 1]);
    let sn = sin.clone().reshape([1, s, 1, d / 2, 1]);
    let out0 = x0.clone() * c.clone() - x1.clone() * sn.clone();
    let out1 = x1 * c + x0 * sn;
    Tensor::cat(vec![out0, out1], 4).reshape([b, s, h, d])
}

/// Plain softmax attention over `[b, s, heads, d]`, returning
/// `[b, s, heads*d]`.
fn attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
) -> Tensor<B, 3> {
    let [b, s, h, d] = q.dims();
    let q = q.permute([0, 2, 1, 3]);
    let k = k.permute([0, 2, 1, 3]);
    let v = v.permute([0, 2, 1, 3]);
    let scores = q.matmul(k.permute([0, 1, 3, 2])).mul_scalar(1.0 / (d as f32).sqrt());
    let out = softmax(scores, 3).matmul(v);
    out.permute([0, 2, 1, 3]).reshape([b, s, h * d])
}

/// One (shift, scale, gate) set broadcast as `[b, 1, dim]`.
pub type ModSet<B> = (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>);

/// Split a modulation projection `[b, 3·sets·dim]` into per-set
/// (shift, scale, gate) triples.
pub fn split_mods<B: Backend>(mods: Tensor<B, 2>, sets: usize, dim: usize) -> Vec<ModSet<B>> {
    let [b, _] = mods.dims();
    let m = mods.reshape([b, 1, 3 * sets * dim]);
    (0..sets)
        .map(|i| {
            (
                m.clone().narrow(2, (3 * i) * dim, dim),
                m.clone().narrow(2, (3 * i + 1) * dim, dim),
                m.clone().narrow(2, (3 * i + 2) * dim, dim),
            )
        })
        .collect()
}

/// SwiGLU feed-forward with the gate fused into `linear_in`.
struct Flux2FeedForward<B: Backend> {
    linear_in: BlockLinear<B>,
    linear_out: BlockLinear<B>,
}

impl<B: Backend> Flux2FeedForward<B> {
    fn load(
        source: &dyn ModelSource,
        prefix: &str,
        dim: usize,
        inner: usize,
        device: &Device<B>,
    ) -> Result<Self> {
        Ok(Self {
            linear_in: load_block_linear(source, &format!("{prefix}.linear_in"), dim, inner * 2, device)?,
            linear_out: load_block_linear(source, &format!("{prefix}.linear_out"), inner, dim, device)?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear_in.forward(x, None);
        let half = x.dims()[2] / 2;
        let gate = silu(x.clone().narrow(2, 0, half));
        self.linear_out.forward(gate * x.narrow(2, half, half), None)
    }
}

/// Double-stream block: separate img/txt projections, joint attention
/// over the text-first concatenation, separate feed-forwards.
pub struct Flux2DoubleBlock<B: Backend> {
    to_q: BlockLinear<B>,
    to_k: BlockLinear<B>,
    to_v: BlockLinear<B>,
    to_out: BlockLinear<B>,
    add_q: BlockLinear<B>,
    add_k: BlockLinear<B>,
    add_v: BlockLinear<B>,
    to_add_out: BlockLinear<B>,
    norm_q: Tensor<B, 1>,
    norm_k: Tensor<B, 1>,
    norm_added_q: Tensor<B, 1>,
    norm_added_k: Tensor<B, 1>,
    ff: Flux2FeedForward<B>,
    ff_context: Flux2FeedForward<B>,
}

impl<B: Backend> Flux2DoubleBlock<B> {
    pub fn load(
        source: &dyn ModelSource,
        prefix: &str,
        cfg: &Flux2Config,
        device: &Device<B>,
    ) -> Result<Self> {
        let dim = cfg.inner_dim();
        let a = format!("{prefix}.attn");
        let lin = |name: &str| -> Result<BlockLinear<B>> {
            load_block_linear(source, &format!("{a}.{name}"), dim, dim, device)
        };
        let head = |name: &str| -> Result<Tensor<B, 1>> {
            Ok(load_param::<B, 1>(source, &format!("{a}.{name}.weight"), device)?.val())
        };
        Ok(Self {
            to_q: lin("to_q")?,
            to_k: lin("to_k")?,
            to_v: lin("to_v")?,
            to_out: lin("to_out.0")?,
            add_q: lin("add_q_proj")?,
            add_k: lin("add_k_proj")?,
            add_v: lin("add_v_proj")?,
            to_add_out: lin("to_add_out")?,
            norm_q: head("norm_q")?,
            norm_k: head("norm_k")?,
            norm_added_q: head("norm_added_q")?,
            norm_added_k: head("norm_added_k")?,
            ff: Flux2FeedForward::load(source, &format!("{prefix}.ff"), dim, dim * cfg.mlp_ratio, device)?,
            ff_context: Flux2FeedForward::load(source, &format!("{prefix}.ff_context"), dim, dim * cfg.mlp_ratio, device)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: Tensor<B, 3>,
        txt: Tensor<B, 3>,
        mods_img: &[ModSet<B>],
        mods_txt: &[ModSet<B>],
        rope: (&Tensor<B, 2>, &Tensor<B, 2>),
        cfg: &Flux2Config,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let (heads, hd, eps) = (cfg.num_attention_heads, cfg.attention_head_dim, cfg.eps);
        let [b, s_img, _] = img.dims();
        let s_txt = txt.dims()[1];
        let (shift_a, scale_a, gate_a) = &mods_img[0];
        let (shift_m, scale_m, gate_m) = &mods_img[1];
        let (c_shift_a, c_scale_a, c_gate_a) = &mods_txt[0];
        let (c_shift_m, c_scale_m, c_gate_m) = &mods_txt[1];

        let n_img = layer_norm(img.clone(), eps) * (scale_a.clone() + 1.0) + shift_a.clone();
        let n_txt = layer_norm(txt.clone(), eps) * (c_scale_a.clone() + 1.0) + c_shift_a.clone();

        let to_heads = |t: Tensor<B, 3>, s: usize| t.reshape([b, s, heads, hd]);
        let q = rms_norm_head(to_heads(self.to_q.forward(n_img.clone(), None), s_img), &self.norm_q, eps);
        let k = rms_norm_head(to_heads(self.to_k.forward(n_img.clone(), None), s_img), &self.norm_k, eps);
        let v = to_heads(self.to_v.forward(n_img, None), s_img);
        let cq = rms_norm_head(to_heads(self.add_q.forward(n_txt.clone(), None), s_txt), &self.norm_added_q, eps);
        let ck = rms_norm_head(to_heads(self.add_k.forward(n_txt.clone(), None), s_txt), &self.norm_added_k, eps);
        let cv = to_heads(self.add_v.forward(n_txt, None), s_txt);

        // Text first, matching the rope table order.
        let q = apply_rope(Tensor::cat(vec![cq, q], 1), rope.0, rope.1);
        let k = apply_rope(Tensor::cat(vec![ck, k], 1), rope.0, rope.1);
        let v = Tensor::cat(vec![cv, v], 1);
        let joint = attention(q, k, v);

        let txt_attn = self.to_add_out.forward(joint.clone().narrow(1, 0, s_txt), None);
        let img_attn = self.to_out.forward(joint.narrow(1, s_txt, s_img), None);

        let img = img + img_attn * gate_a.clone();
        let n = layer_norm(img.clone(), eps) * (scale_m.clone() + 1.0) + shift_m.clone();
        let img = img + self.ff.forward(n) * gate_m.clone();

        let txt = txt + txt_attn * c_gate_a.clone();
        let n = layer_norm(txt.clone(), eps) * (c_scale_m.clone() + 1.0) + c_shift_m.clone();
        let txt = txt + self.ff_context.forward(n) * c_gate_m.clone();

        (txt, img)
    }
}

/// Single-stream block: fused QKV+MLP projection, parallel attention
/// and SwiGLU over the joint sequence, one fused output projection.
pub struct Flux2SingleBlock<B: Backend> {
    to_qkv_mlp: BlockLinear<B>,
    to_out: BlockLinear<B>,
    norm_q: Tensor<B, 1>,
    norm_k: Tensor<B, 1>,
}

impl<B: Backend> Flux2SingleBlock<B> {
    pub fn load(
        source: &dyn ModelSource,
        prefix: &str,
        cfg: &Flux2Config,
        device: &Device<B>,
    ) -> Result<Self> {
        let dim = cfg.inner_dim();
        let mlp = dim * cfg.mlp_ratio;
        let a = format!("{prefix}.attn");
        Ok(Self {
            to_qkv_mlp: load_block_linear(source, &format!("{a}.to_qkv_mlp_proj"), dim, dim * 3 + mlp * 2, device)?,
            to_out: load_block_linear(source, &format!("{a}.to_out"), dim + mlp, dim, device)?,
            norm_q: load_param::<B, 1>(source, &format!("{a}.norm_q.weight"), device)?.val(),
            norm_k: load_param::<B, 1>(source, &format!("{a}.norm_k.weight"), device)?.val(),
        })
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mods: &ModSet<B>,
        rope: (&Tensor<B, 2>, &Tensor<B, 2>),
        cfg: &Flux2Config,
    ) -> Tensor<B, 3> {
        let (heads, hd, eps) = (cfg.num_attention_heads, cfg.attention_head_dim, cfg.eps);
        let dim = cfg.inner_dim();
        let mlp = dim * cfg.mlp_ratio;
        let [b, s, _] = x.dims();
        let (shift, scale, gate) = mods;

        let n = layer_norm(x.clone(), eps) * (scale.clone() + 1.0) + shift.clone();
        let fused = self.to_qkv_mlp.forward(n, None);

        let to_heads = |t: Tensor<B, 3>| t.reshape([b, s, heads, hd]);
        let q = rms_norm_head(to_heads(fused.clone().narrow(2, 0, dim)), &self.norm_q, eps);
        let k = rms_norm_head(to_heads(fused.clone().narrow(2, dim, dim)), &self.norm_k, eps);
        let v = to_heads(fused.clone().narrow(2, 2 * dim, dim));
        let q = apply_rope(q, rope.0, rope.1);
        let k = apply_rope(k, rope.0, rope.1);
        let attn_out = attention(q, k, v);

        let gate_half = silu(fused.clone().narrow(2, 3 * dim, mlp));
        let mlp_out = gate_half * fused.narrow(2, 3 * dim + mlp, mlp);

        let out = self.to_out.forward(Tensor::cat(vec![attn_out, mlp_out], 2), None);
        x + out * gate.clone()
    }
}

/// The full FLUX.2 transformer.
pub struct Flux2Transformer<B: Backend> {
    pub config: Flux2Config,
    time_linear_1: Linear<B>,
    time_linear_2: Linear<B>,
    mod_img: Linear<B>,
    mod_txt: Linear<B>,
    mod_single: Linear<B>,
    x_embedder: Linear<B>,
    context_embedder: Linear<B>,
    blocks: Vec<Flux2DoubleBlock<B>>,
    single_blocks: Vec<Flux2SingleBlock<B>>,
    norm_out_linear: Linear<B>,
    proj_out: Linear<B>,
}

impl<B: Backend> Flux2Transformer<B> {
    pub fn load(
        source: &dyn ModelSource,
        config: Flux2Config,
        device: &Device<B>,
    ) -> Result<Self> {
        let dim = config.inner_dim();
        let blocks = (0..config.num_layers)
            .map(|i| Flux2DoubleBlock::load(source, &format!("transformer_blocks.{i}"), &config, device))
            .collect::<Result<Vec<_>>>()?;
        let single_blocks = (0..config.num_single_layers)
            .map(|i| {
                Flux2SingleBlock::load(source, &format!("single_transformer_blocks.{i}"), &config, device)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            time_linear_1: load_linear(
                source,
                "time_guidance_embed.timestep_embedder.linear_1",
                config.timestep_channels,
                dim,
                false,
                device,
            )?,
            time_linear_2: load_linear(
                source,
                "time_guidance_embed.timestep_embedder.linear_2",
                dim,
                dim,
                false,
                device,
            )?,
            mod_img: load_linear(source, "double_stream_modulation_img.linear", dim, dim * 6, false, device)?,
            mod_txt: load_linear(source, "double_stream_modulation_txt.linear", dim, dim * 6, false, device)?,
            mod_single: load_linear(source, "single_stream_modulation.linear", dim, dim * 3, false, device)?,
            x_embedder: load_linear(source, "x_embedder", config.in_channels, dim, false, device)?,
            context_embedder: load_linear(
                source,
                "context_embedder",
                config.joint_attention_dim,
                dim,
                false,
                device,
            )?,
            blocks,
            single_blocks,
            norm_out_linear: load_linear(source, "norm_out.linear", dim, dim * 2, false, device)?,
            proj_out: load_linear(source, "proj_out", dim, config.in_channels, false, device)?,
            config,
        })
    }

    /// Sinusoidal timestep projection, diffusers `Timesteps` with
    /// `flip_sin_to_cos = true`, `downscale_freq_shift = 0`: [cos, sin].
    fn time_proj(&self, t: f32, device: &Device<B>) -> Tensor<B, 2> {
        let half = self.config.timestep_channels / 2;
        let mut row = vec![0.0f32; self.config.timestep_channels];
        for i in 0..half {
            let freq = (-(10_000f64.ln()) * i as f64 / half as f64).exp();
            let arg = t as f64 * freq;
            row[i] = arg.cos() as f32;
            row[half + i] = arg.sin() as f32;
        }
        Tensor::from_data(
            burn::tensor::TensorData::new(row, [1, self.config.timestep_channels]),
            device,
        )
    }

    /// One denoise evaluation. `timestep` is the pipeline-scale value
    /// in `[0, 1]` (scaled by 1000 internally, matching the reference);
    /// ids are the 4-axis (T, H, W, L) coordinates per token.
    pub fn forward(
        &self,
        img: Tensor<B, 3>,
        txt: Tensor<B, 3>,
        timestep: f32,
        img_ids: &[[f64; 4]],
        txt_ids: &[[f64; 4]],
    ) -> Tensor<B, 3> {
        let cfg = &self.config;
        let dim = cfg.inner_dim();
        let device = img.device();
        let s_txt = txt.dims()[1];
        let s_img = img.dims()[1];

        let temb = self
            .time_linear_2
            .forward(silu(self.time_linear_1.forward(self.time_proj(timestep * 1000.0, &device))));
        let debug = std::env::var("COMBS_KLEIN_DEBUG").is_ok_and(|v| v != "0");
        let peek = |label: &str, t: &Tensor<B, 3>| {
            if !debug {
                return;
            }
            let v: Vec<f32> = t.clone().into_data().convert::<f32>().to_vec().unwrap_or_default();
            let n = v.len().max(1) as f32;
            let mean = v.iter().sum::<f32>() / n;
            let amax = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let nan = v.iter().filter(|x| x.is_nan()).count();
            eprintln!("[flux2-debug] {label}: mean {mean:.5} amax {amax:.4} nan {nan}");
        };
        if debug {
            let v: Vec<f32> = temb.clone().into_data().convert::<f32>().to_vec().unwrap_or_default();
            let amax = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            eprintln!("[flux2-debug] temb: amax {amax:.4}");
        }

        let mods_img = split_mods(self.mod_img.forward(silu(temb.clone())), 2, dim);
        let mods_txt = split_mods(self.mod_txt.forward(silu(temb.clone())), 2, dim);
        let mods_single = split_mods(self.mod_single.forward(silu(temb.clone())), 1, dim);

        let mut img_h = self.x_embedder.forward(img);
        let mut txt_h = self.context_embedder.forward(txt);

        // Rope tables over the text-first joint sequence.
        let mut ids: Vec<[f64; 4]> = Vec::with_capacity(s_txt + s_img);
        ids.extend_from_slice(txt_ids);
        ids.extend_from_slice(img_ids);
        let (cos, sin) = rope_tables::<B>(&ids, &cfg.axes_dims_rope, cfg.rope_theta, &device);

        peek("img embed", &img_h);
        peek("txt embed", &txt_h);
        for (bi, block) in self.blocks.iter().enumerate() {
            let (t, i) = block.forward(img_h, txt_h, &mods_img, &mods_txt, (&cos, &sin), cfg);
            txt_h = t;
            img_h = i;
            peek(&format!("double_{bi} img"), &img_h);
        }

        let mut h = Tensor::cat(vec![txt_h, img_h], 1);
        for (bi, block) in self.single_blocks.iter().enumerate() {
            h = block.forward(h, &mods_single[0], (&cos, &sin), cfg);
            if bi % 5 == 0 || bi + 1 == self.single_blocks.len() {
                peek(&format!("single_{bi}"), &h);
            }
        }
        let h = h.narrow(1, s_txt, s_img);

        // AdaLayerNormContinuous: chunk order is (scale, shift) — the
        // REVERSE of the block modulation order.
        let cond = self.norm_out_linear.forward(silu(temb));
        let scale = cond.clone().narrow(1, 0, dim).reshape([1, 1, dim]);
        let shift = cond.narrow(1, dim, dim).reshape([1, 1, dim]);
        let h = layer_norm(h, cfg.eps) * (scale + 1.0) + shift;
        self.proj_out.forward(h)
    }
}

/// The latent batch-norm statistics the flux2 autoencoder carries:
/// generation runs in the normalized patchified-latent space, and the
/// decoder wants `latent * sqrt(running_var + eps) + running_mean`.
pub struct Flux2LatentStats<B: Backend> {
    mean: Tensor<B, 1>,
    std: Tensor<B, 1>,
}

impl<B: Backend> Flux2LatentStats<B> {
    /// Loads `bn.running_mean` / `bn.running_var` from the autoencoder
    /// checkpoint (batch_norm_eps 1e-4 per the shipped config).
    pub fn load(source: &dyn ModelSource, device: &Device<B>) -> Result<Self> {
        let mean: Tensor<B, 1> = source.open_tensor("bn.running_mean")?.load_to_tensor(device)?;
        let var: Tensor<B, 1> = source.open_tensor("bn.running_var")?.load_to_tensor(device)?;
        Ok(Self { mean, std: (var + 1e-4).sqrt() })
    }

    /// Denormalize a patchified latent `[b, c, h, w]` (c = the bn width,
    /// 4 x latent_channels).
    pub fn denormalize(&self, latent: Tensor<B, 4>) -> Tensor<B, 4> {
        let c = self.mean.dims()[0];
        latent * self.std.clone().reshape([1, c, 1, 1])
            + self.mean.clone().reshape([1, c, 1, 1])
    }
}

/// Packed tokens `[b, gh*gw, c]` (row-major grid order, the id scheme
/// `image_ids` emits) back to the patchified grid `[b, c, gh, gw]`.
pub fn unpack_latents<B: Backend>(
    tokens: Tensor<B, 3>,
    grid_h: usize,
    grid_w: usize,
) -> Tensor<B, 4> {
    let [b, s, c] = tokens.dims();
    assert_eq!(s, grid_h * grid_w, "token count must fill the grid");
    tokens.permute([0, 2, 1]).reshape([b, c, grid_h, grid_w])
}

/// Inverse of the pipeline's 2x2 patchify: `[b, 4c, h, w]` →
/// `[b, c, 2h, 2w]` (patchify was view(b,c,h,2,w,2) → permute
/// (0,1,3,5,2,4) → reshape, so the channel axis orders as
/// `[c][py][px]`).
pub fn unpatchify_latents<B: Backend>(latent: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, c4, h, w] = latent.dims();
    let c = c4 / 4;
    latent
        .reshape([b, c, 2, 2, h, w])
        .permute([0, 1, 4, 2, 5, 3])
        .reshape([b, c, h * 2, w * 2])
}

/// Flow-matching Euler schedule with resolution-dependent exponential
/// time shifting — the klein sampling loop. Deliberately NOT behind
/// the `Scheduler` trait: that contract is epsilon-prediction over
/// `[b, c, h, w]` VP latents with integer timesteps, while flow
/// matching is velocity-prediction over packed `[b, seq, c]` latents
/// on a continuous sigma axis.
pub struct FlowMatchEuler {
    sigmas: Vec<f32>,
}

impl FlowMatchEuler {
    /// The empirical mu fit shipped with the klein pipeline: piecewise
    /// linear in image sequence length, interpolated by step count
    /// below 4300 tokens.
    pub fn empirical_mu(image_seq_len: usize, num_steps: usize) -> f64 {
        let (a1, b1) = (8.738_095_24e-5, 1.898_333_33);
        let (a2, b2) = (1.6927e-4, 0.456_666_66);
        let seq = image_seq_len as f64;
        if image_seq_len > 4300 {
            return a2 * seq + b2;
        }
        let m_200 = a2 * seq + b2;
        let m_10 = a1 * seq + b1;
        let a = (m_200 - m_10) / 190.0;
        let b = m_200 - 200.0 * a;
        a * num_steps as f64 + b
    }

    /// linspace(1, 1/steps, steps) through the exponential shift
    /// `exp(mu) / (exp(mu) + (1/sigma - 1))`, terminating at 0.
    pub fn new(num_steps: usize, image_seq_len: usize) -> Self {
        let n = num_steps.max(1);
        let mu = Self::empirical_mu(image_seq_len, n);
        let emu = mu.exp();
        let mut sigmas: Vec<f32> = (0..n)
            .map(|i| {
                let s = 1.0 - (1.0 - 1.0 / n as f64) * i as f64 / (n as f64 - 1.0).max(1.0);
                (emu / (emu + (1.0 / s - 1.0))) as f32
            })
            .collect();
        sigmas.push(0.0);
        Self { sigmas }
    }

    pub fn num_steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// The DiT timestep for step `i` — the sigma itself (the pipeline
    /// scale where the transformer multiplies by 1000 internally).
    pub fn timestep(&self, i: usize) -> f32 {
        self.sigmas[i]
    }

    /// Euler step: `x + (sigma_next - sigma) * velocity`.
    pub fn step<B: Backend>(
        &self,
        latent: Tensor<B, 3>,
        velocity: Tensor<B, 3>,
        i: usize,
    ) -> Tensor<B, 3> {
        let dt = self.sigmas[i + 1] - self.sigmas[i];
        latent + velocity.mul_scalar(dt)
    }
}

/// The pipeline's position-id schemes.
///
/// Image tokens walk the patch grid row-major with (T=0, H=row,
/// W=col, L=0); text tokens sit at (0, 0, 0, L=index).
pub fn image_ids(height_tokens: usize, width_tokens: usize) -> Vec<[f64; 4]> {
    let mut ids = Vec::with_capacity(height_tokens * width_tokens);
    for h in 0..height_tokens {
        for w in 0..width_tokens {
            ids.push([0.0, h as f64, w as f64, 0.0]);
        }
    }
    ids
}

pub fn text_ids(seq: usize) -> Vec<[f64; 4]> {
    (0..seq).map(|l| [0.0, 0.0, 0.0, l as f64]).collect()
}

#[cfg(test)]
mod tests {
    use super::FlowMatchEuler;

    // Reference sigmas from the diffusers FlowMatchEulerDiscreteScheduler
    // (klein config: dynamic exponential shifting) at the klein
    // geometries; mu from the pipeline's empirical fit.
    fn assert_schedule(seq: usize, steps: usize, want_mu: f64, want: &[f32]) {
        let mu = FlowMatchEuler::empirical_mu(seq, steps);
        assert!((mu - want_mu).abs() < 1e-9, "mu {mu} vs {want_mu}");
        let s = FlowMatchEuler::new(steps, seq);
        assert_eq!(s.sigmas().len(), want.len());
        for (i, (&got, &w)) in s.sigmas().iter().zip(want).enumerate() {
            assert!((got - w).abs() < 1e-6, "sigma[{i}] {got} vs {w}");
        }
    }

    #[test]
    fn schedule_matches_reference_512px_4step() {
        assert_schedule(
            1024,
            4,
            2.0306897079499455,
            &[1.0, 0.9580854, 0.8839819, 0.7174966, 0.0],
        );
    }

    #[test]
    fn schedule_matches_reference_1024px_4step() {
        assert_schedule(
            4096,
            4,
            2.291179894115571,
            &[1.0, 0.96738404, 0.9081439, 0.76719993, 0.0],
        );
    }

    #[test]
    fn schedule_matches_reference_512px_50step_endpoints() {
        let s = FlowMatchEuler::new(50, 1024);
        assert!((FlowMatchEuler::empirical_mu(1024, 50) - 1.7019562073086316).abs() < 1e-9);
        assert_eq!(s.num_steps(), 50);
        let sig = s.sigmas();
        for (i, w) in [
            (0usize, 1.0f32),
            (1, 0.99629283),
            (25, 0.84579003),
            (49, 0.10066439),
            (50, 0.0),
        ] {
            assert!((sig[i] - w).abs() < 1e-6, "sigma[{i}] {} vs {w}", sig[i]);
        }
    }
}
