//! Diffusion model runtime built directly on Burn.
//!
//! This crate provides a `DiffusionModel` trait and a minimal Stable
//! Diffusion-style pipeline implemented without external diffusion
//! frameworks. The first increment is intentionally skeletal: the trait,
//! a DDPM scheduler, and a tiny UNet scaffold. The UNet, VAE, and text
//! encoder grow in later loops.

use burn::tensor::{backend::Backend, Tensor};
use combs_formats::Result;
use combs_formats::{AttentionPattern, ModelMetadata, ModelSource, TokenizerSpec};

pub mod blocks;
pub mod clip;
pub mod loader;
pub mod noise;
pub mod scheduler;
pub mod time;
pub mod unet;
pub mod vae;
pub mod weights;

pub use clip::{input_ids_to_tensor, ClipTextModel};
pub use loader::{load_diffusion_model, DiffusionArchitecture};
pub use combs_formats::{LoraFile, LoraSpec};
pub use noise::NoiseSource;
pub use scheduler::{Scheduler, SchedulerKind};
pub use unet::UNet2DConditionModel;
pub use vae::VAEDecoder;

/// Embeddings produced by the text encoder for a (prompt, negative prompt)
/// pair. Kept opaque so different pipelines (SD 1.5, SDXL, Flux) can store
/// whatever guidance tensors they need.
pub struct PromptEmbed<B: Backend> {
    /// Text conditioning tensor, shape `[batch, seq, hidden]`.
    pub positive: Tensor<B, 3>,
    /// Optional negative / unconditioned guidance tensor.
    pub negative: Option<Tensor<B, 3>>,
}

/// Model-agnostic diffusion contract. Analogous to `GenerativeModel` but for
/// latent denoising instead of autoregressive token generation.
pub trait DiffusionModel<B: Backend>: Send {
    fn metadata(&self) -> &ModelMetadata;

    fn load(source: &dyn ModelSource, device: &burn::tensor::Device<B>) -> Result<Self>
    where
        Self: Sized;

    /// Encode prompt strings into conditioning embeddings.
    fn encode_prompt(&self, prompt: &str, negative_prompt: Option<&str>) -> Result<PromptEmbed<B>>;

    /// Run the full denoising loop and return the image tensor
    /// `[batch, channels, height, width]` in RGB order plus the effective
    /// seed (caller-provided, or entropy-drawn and echoed for reproduction).
    /// `on_step` fires after each completed denoise step with
    /// `(completed, total)`, 1-based; `&mut dyn` keeps the trait object-safe.
    fn generate(
        &mut self,
        prompt: PromptEmbed<B>,
        width: u32,
        height: u32,
        num_inference_steps: usize,
        guidance_scale: f32,
        seed: Option<u64>,
        scheduler: SchedulerKind,
        on_step: Option<&mut dyn FnMut(usize, usize)>,
    ) -> Result<(Tensor<B, 4>, u64)>;
}

fn diffusion_metadata(architecture: &str) -> ModelMetadata {
    ModelMetadata {
        architecture: architecture.to_string(),
        hidden_size: 0,
        intermediate_size: 0,
        num_hidden_layers: 0,
        num_attention_heads: 0,
        num_key_value_heads: 0,
        vocab_size: 0,
        max_position_embeddings: 0,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings: false,
        head_dim: 0,
        attention_bias: false,
        bos_token_id: None,
        eos_token_ids: Vec::new(),
        vision: None,
        attention_pattern: AttentionPattern::default(),
        activation: combs_formats::Activation::default(),
        rope_scaling: combs_formats::RopeScaling::default(),
    }
}

/// A minimal Stable Diffusion 1.5-style pipeline built from scratch in Burn.
/// The VAE decoder is still a stub; the text encoder and UNet are real.
pub struct StableDiffusionPipeline<B: Backend> {
    metadata: ModelMetadata,
    device: burn::tensor::Device<B>,
    unet: UNet2DConditionModel<B>,
    vae_decoder: VAEDecoder<B>,
    text_encoder: Option<ClipTextModel<B>>,
    tokenizer_json: Option<std::path::PathBuf>,
}

impl<B: Backend> DiffusionModel<B> for StableDiffusionPipeline<B> {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn load(source: &dyn ModelSource, device: &burn::tensor::Device<B>) -> Result<Self> {
        // TODO: read config.json, load UNet/VAE/text-encoder weights from safetensors.
        let metadata = diffusion_metadata("stable-diffusion");
        let unet = UNet2DConditionModel::new(device);
        let mut vae_decoder = VAEDecoder::new(device);

        // Load the VAE decoder if the source provides the expected weights.
        if source.tensor_names().contains(&"decoder.conv_in.weight".to_string()) {
            vae_decoder = VAEDecoder::load_from(source, device)?;
        }

        // Load the CLIP text encoder if the source provides the expected weights.
        let text_encoder = if source.tensor_names().contains(&
            "text_model.embeddings.token_embedding.weight".to_string(),
        ) {
            Some(ClipTextModel::load_from(source, device)?)
        } else {
            None
        };

        let tokenizer_json = source
            .tokenizer()
            .ok()
            .map(|spec| spec.tokenizer_json.clone());

        Ok(Self {
            metadata,
            device: device.clone(),
            unet,
            vae_decoder,
            text_encoder,
            tokenizer_json,
        })
    }

    fn encode_prompt(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
    ) -> Result<PromptEmbed<B>> {
        let (Some(text_encoder), Some(tokenizer_json)) = (&self.text_encoder, &self.tokenizer_json) else {
            // Fallback for scaffold / weight-only sources: zero embeddings.
            let zeros = Tensor::zeros([1, 77, 768], &self.device);
            return Ok(PromptEmbed {
                positive: zeros.clone(),
                negative: Some(zeros),
            });
        };

        let positive_ids = tokenize_clip(&tokenizer_json, prompt)?;
        let positive_ids = crate::clip::input_ids_to_tensor::<B>(&positive_ids, &self.device);
        let positive = text_encoder.forward(positive_ids);

        let negative_text = negative_prompt.unwrap_or("");
        let negative_ids = tokenize_clip(&tokenizer_json, negative_text)?;
        let negative_ids = crate::clip::input_ids_to_tensor::<B>(&negative_ids, &self.device);
        let negative = text_encoder.forward(negative_ids);

        Ok(PromptEmbed {
            positive,
            negative: Some(negative),
        })
    }

    fn generate(
        &mut self,
        prompt: PromptEmbed<B>,
        width: u32,
        height: u32,
        num_inference_steps: usize,
        guidance_scale: f32,
        seed: Option<u64>,
        scheduler: SchedulerKind,
        mut on_step: Option<&mut dyn FnMut(usize, usize)>,
    ) -> Result<(Tensor<B, 4>, u64)> {
        let [b, _seq, _hidden] = prompt.positive.dims();
        let latent_h = (height / 8).max(1) as usize;
        let latent_w = (width / 8).max(1) as usize;
        let latent_channels = 4;

        // Classifier-free guidance: one batched UNet pass over
        // [uncond; cond], split, then `uncond + scale * (cond - uncond)`.
        // `scale <= 1` skips the uncond half entirely (single-batch pass).
        let do_cfg = guidance_scale > 1.0 && prompt.negative.is_some();
        let context = if do_cfg {
            Tensor::cat(
                vec![prompt.negative.clone().unwrap(), prompt.positive.clone()],
                0,
            )
        } else {
            prompt.positive.clone()
        };

        // 1. seeded latent noise (host-generated: reproducible everywhere)
        let mut noise = NoiseSource::new(seed);
        let mut sched = scheduler.build::<B>(num_inference_steps);
        let mut latent = noise
            .normal_tensor::<B>([b, latent_channels, latent_h, latent_w], &self.device)
            .mul_scalar(sched.init_noise_sigma());

        // 2. denoising loop
        let steps = sched.timesteps().to_vec();
        for (i, &t) in steps.iter().enumerate() {
            let input = sched.scale_model_input(latent.clone(), i);
            let input = if do_cfg {
                Tensor::cat(vec![input.clone(), input], 0)
            } else {
                input
            };
            let pred = self.unet.forward(input, t as f32, context.clone());
            let noise_pred = if do_cfg {
                let uncond = pred.clone().narrow(0, 0, b);
                let cond = pred.narrow(0, b, b);
                uncond.clone() + (cond - uncond).mul_scalar(guidance_scale)
            } else {
                pred
            };
            latent = sched.step(noise_pred, i, latent, &mut noise);
            if let Some(cb) = on_step.as_mut() {
                cb(i + 1, steps.len());
            }
        }

        // 3. VAE decode latent -> image
        //    Latents are in scaled space; undo the scaling before decoding.
        let latent = latent.div_scalar(0.18215f32);
        let image = self.vae_decoder.forward(latent);
        let image = image.clamp(-1.0, 1.0).mul_scalar(0.5).add_scalar(0.5);
        Ok((image, noise.effective_seed()))
    }
}

impl<B: Backend> StableDiffusionPipeline<B> {
    /// Build a pipeline from separate component sources (UNet, VAE, text encoder,
    /// tokenizer). This is the real checkpoint loading path; the `DiffusionModel`
    /// trait loader is kept for scaffold / random-init sources.
    pub fn load_from_dirs(
        unet_source: &dyn ModelSource,
        vae_source: &dyn ModelSource,
        text_source: &dyn ModelSource,
        tokenizer: TokenizerSpec,
        device: &burn::tensor::Device<B>,
    ) -> Result<Self> {
        let metadata = diffusion_metadata("stable-diffusion");
        let unet = UNet2DConditionModel::load_from(unet_source, device)?;
        let vae_decoder = VAEDecoder::load_from(vae_source, device)?;
        let text_encoder = ClipTextModel::load_from(text_source, device)?;

        Ok(Self {
            metadata,
            device: device.clone(),
            unet,
            vae_decoder,
            text_encoder: Some(text_encoder),
            tokenizer_json: Some(tokenizer.tokenizer_json),
        })
    }
}

/// Tokenize a prompt for CLIP (SD 1.5): add bos/eos, truncate to 77, and pad
/// with the end-of-text token.
fn tokenize_clip(tokenizer_json: &std::path::Path, text: &str) -> Result<Vec<u32>> {
    const MAX_LEN: usize = 77;
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_json).map_err(|e| {
        combs_formats::FormatError::Safetensors(format!(
            "failed to load tokenizer {}: {e}",
            tokenizer_json.display()
        ))
    })?;
    let encoding = tokenizer.encode(text, true).map_err(|e| {
        combs_formats::FormatError::Safetensors(format!("tokenize failed: {e}"))
    })?;
    let mut ids = encoding.get_ids().to_vec();
    let pad_id = tokenizer.token_to_id("<|endoftext|>").unwrap_or(49407);
    if ids.len() > MAX_LEN {
        // Keep the trailing EOS — CLIP's pooled/text semantics assume every
        // sequence ends with it; a bare truncate would chop it off.
        ids.truncate(MAX_LEN - 1);
        ids.push(pad_id);
    }
    while ids.len() < MAX_LEN {
        ids.push(pad_id);
    }
    Ok(ids)
}
