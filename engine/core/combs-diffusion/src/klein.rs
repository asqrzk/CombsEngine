//! The FLUX.2 [klein] pipeline: Qwen3 text conditioning through the
//! flux2 transformer and autoencoder.
//!
//! Three components, faithful to the reference pipeline:
//! - text encoder: a stock Qwen3 causal LM (an architecture the llama
//!   family already loads, GGUF quants included). The prompt is
//!   rendered through the checkpoint's own CHAT TEMPLATE with
//!   `add_generation_prompt = true, enable_thinking = false`, then
//!   the conditioning is the RAW residual stream tapped after layers
//!   9/18/27 and concatenated — width 3 x hidden.
//! - transformer: [`crate::flux2::Flux2Transformer`], driven by the
//!   [`crate::flux2::FlowMatchEuler`] schedule (klein is 4-step
//!   distilled; guidance is ignored — `guidance_embeds` is absent
//!   and the reference warns and drops CFG for distilled weights).
//! - autoencoder: packed tokens → grid → batch-norm denorm → 2x2
//!   unpatchify → the shared decoder at 32 latent channels.

use std::sync::Mutex;

use burn::tensor::backend::Backend;
use burn::tensor::{Device, Int, Tensor, TensorData};
use combs_formats::{FormatError, ModelMetadata, ModelSource, Result};
use combs_models::{CacheConfig, GenerativeModel, LlamaModel};
use minijinja::{context, Environment};

use crate::flux2::{
    image_ids, text_ids, unpack_latents, unpatchify_latents, FlowMatchEuler, Flux2Config,
    Flux2LatentStats, Flux2Transformer,
};
use crate::vae::VAEDecoder;
use crate::{DiffusionModel, GenerationHooks, NoiseSource, PromptEmbed, SchedulerKind};

/// The reference pipeline's default encoder taps and prompt budget.
const ENCODER_TAPS: [usize; 3] = [9, 18, 27];
const MAX_PROMPT_TOKENS: usize = 512;

/// Render the prompt through the checkpoint's chat template (thinking
/// disabled, generation prompt appended), matching the reference
/// conditioning exactly. A checkpoint without a template gets the
/// qwen conversation wrap.
pub(crate) fn render_prompt(chat_template: Option<&str>, prompt: &str) -> String {
    if let Some(template) = chat_template {
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        let rendered = env.render_str(
            template,
            context! {
                messages => vec![serde_json::json!({"role": "user", "content": prompt})],
                tools => serde_json::Value::Null,
                add_generation_prompt => true,
                enable_thinking => false,
            },
        );
        match rendered {
            Ok(text) => return text,
            Err(e) => eprintln!("[klein] chat template failed ({e}); using the plain wrap"),
        }
    }
    format!(
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

#[cfg(test)]
mod tests {
    use super::render_prompt;

    /// The real qwen3 chat template (from the cached qwen3-0.6b
    /// checkout — the same family template klein's encoder ships)
    /// must render a single user turn with thinking DISABLED: the
    /// assistant turn opens with an empty think block, exactly the
    /// conditioning text the reference pipeline builds.
    #[test]
    fn qwen3_template_renders_thinking_disabled() {
        let home = std::env::var("HOME").expect("HOME");
        let path = format!("{home}/.cache/combs/models/qwen3-0.6b/tokenizer_config.json");
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("skipping: no cached qwen3-0.6b checkout");
            return;
        };
        let config: serde_json::Value = serde_json::from_str(&text).unwrap();
        let template = config["chat_template"].as_str().expect("chat_template");

        let out = render_prompt(Some(template), "a red fox in the snow");
        assert!(
            out.contains("<|im_start|>user\na red fox in the snow<|im_end|>"),
            "user turn wrapped: {out:?}"
        );
        assert!(
            out.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "assistant turn must open with the EMPTY think block (thinking disabled): {out:?}"
        );
    }

    #[test]
    fn missing_template_falls_back_to_the_qwen_wrap() {
        let out = render_prompt(None, "hello");
        assert!(out.starts_with("<|im_start|>user\nhello<|im_end|>"));
        assert!(out.ends_with("</think>\n\n"));
    }
}

pub struct Flux2KleinPipeline<B: Backend> {
    metadata: ModelMetadata,
    device: Device<B>,
    transformer: Flux2Transformer<B>,
    vae: VAEDecoder<B>,
    latent_stats: Flux2LatentStats<B>,
    /// `prefill_taps` takes `&mut` (the cache contract), encode_prompt
    /// is `&self` — one prompt at a time behind a lock.
    encoder: Mutex<(LlamaModel<B>, Box<dyn combs_models::KVCache<B>>)>,
    tokenizer: tokenizers::Tokenizer,
    chat_template: Option<String>,
}

impl<B: Backend> Flux2KleinPipeline<B> {
    /// Assemble from the three-part recipe: the DiT weights, the Qwen3
    /// language model (any source the llama family loads — GGUF quants
    /// are the practical choice), and the flux2 autoencoder.
    pub fn load_recipe(
        dit: &dyn ModelSource,
        dit_config: Flux2Config,
        llm: &dyn ModelSource,
        vae: &dyn ModelSource,
        device: &Device<B>,
    ) -> Result<Self> {
        let transformer = Flux2Transformer::load(dit, dit_config, device)?;
        let vae_decoder = VAEDecoder::load_with_latent_channels(vae, 32, device)?;
        let latent_stats = Flux2LatentStats::load(vae, device)?;

        let encoder = LlamaModel::<B>::load(llm, device)
            .map_err(|e| FormatError::Safetensors(format!("klein text encoder: {e}")))?;
        let hidden = encoder.metadata().hidden_size;
        let expected = ENCODER_TAPS.len() * hidden;
        if expected != transformer.config.joint_attention_dim {
            return Err(FormatError::Safetensors(format!(
                "encoder/transformer mismatch: {} taps x hidden {hidden} = {expected}, \
                 but the transformer wants joint_attention_dim {}",
                ENCODER_TAPS.len(),
                transformer.config.joint_attention_dim
            )));
        }
        let cache = encoder.create_kv_cache(&CacheConfig::contiguous(MAX_PROMPT_TOKENS));

        let spec = llm.tokenizer()?;
        let tokenizer = tokenizers::Tokenizer::from_bytes(spec.json_bytes()?)
            .map_err(|e| FormatError::Safetensors(format!("klein tokenizer: {e}")))?;

        Ok(Self {
            metadata: crate::diffusion_metadata("flux2-klein"),
            device: device.clone(),
            transformer,
            vae: vae_decoder,
            latent_stats,
            encoder: Mutex::new((encoder, cache)),
            tokenizer,
            chat_template: spec.chat_template.clone(),
        })
    }

    fn wrap_prompt(&self, prompt: &str) -> String {
        render_prompt(self.chat_template.as_deref(), prompt)
    }

    /// Packed latent tokens → RGB in [0, 1].
    fn decode_tokens(&self, tokens: Tensor<B, 3>, grid_h: usize, grid_w: usize) -> Tensor<B, 4> {
        let grid = unpack_latents(tokens, grid_h, grid_w);
        let latent = unpatchify_latents(self.latent_stats.denormalize(grid));
        let image = self.vae.forward(latent);
        image.mul_scalar(0.5).add_scalar(0.5).clamp(0.0, 1.0)
    }
}

impl<B: Backend> DiffusionModel<B> for Flux2KleinPipeline<B> {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn load(_source: &dyn ModelSource, _device: &Device<B>) -> Result<Self> {
        Err(FormatError::Safetensors(
            "flux2-klein assembles from a three-part recipe (transformer + language \
             model + autoencoder) — use Flux2KleinPipeline::load_recipe or the \
             directory loader"
                .to_string(),
        ))
    }

    fn encode_prompt(&self, prompt: &str, _negative_prompt: Option<&str>) -> Result<PromptEmbed<B>> {
        let text = self.wrap_prompt(prompt);
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| FormatError::Safetensors(format!("klein encode: {e}")))?;
        let mut ids: Vec<i32> = encoding.get_ids().iter().map(|&t| t as i32).collect();
        ids.truncate(MAX_PROMPT_TOKENS);
        let seq = ids.len();

        let mut guard = self.encoder.lock().expect("encoder lock");
        let (encoder, cache) = &mut *guard;
        cache.reset();
        let tokens: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::new(ids, [1, seq]), &self.device);
        let embedded = encoder.embed(tokens);
        let positive = encoder
            .prefill_taps(embedded, cache.as_mut(), 0..seq as u32, &ENCODER_TAPS)
            .map_err(|e| FormatError::Safetensors(format!("klein taps: {e}")))?;
        cache.reset();

        // Distilled klein runs guidance-free; the negative arm exists
        // only for future -base weights and stays empty here.
        Ok(PromptEmbed { positive, negative: None })
    }

    fn generate(
        &mut self,
        prompt: PromptEmbed<B>,
        width: u32,
        height: u32,
        num_inference_steps: usize,
        guidance_scale: f32,
        seed: Option<u64>,
        _scheduler: SchedulerKind,
        mut hooks: GenerationHooks<'_, B>,
    ) -> Result<(Tensor<B, 4>, u64)> {
        if guidance_scale > 1.0 {
            eprintln!(
                "[klein] guidance {guidance_scale} ignored — distilled weights run guidance-free"
            );
        }
        // 8x VAE then 2x2 patchify: one token per 16x16 image pixels.
        let grid_h = (height as usize / 16).max(1);
        let grid_w = (width as usize / 16).max(1);
        let channels = self.transformer.config.in_channels;
        let seq_txt = prompt.positive.dims()[1];

        let mut noise = NoiseSource::new(seed);
        let schedule = FlowMatchEuler::new(num_inference_steps, grid_h * grid_w);
        let img_ids = image_ids(grid_h, grid_w);
        let txt_ids = text_ids(seq_txt);

        let flat = noise.normal_tensor::<B>([1, channels, grid_h, grid_w], &self.device);
        let mut latent = flat.reshape([1, channels, grid_h * grid_w]).permute([0, 2, 1]);

        let total = schedule.num_steps();
        for i in 0..total {
            let velocity = self.transformer.forward(
                latent.clone(),
                prompt.positive.clone(),
                schedule.timestep(i),
                &img_ids,
                &txt_ids,
            );
            latent = schedule.step(latent, velocity, i);
            let completed = i + 1;
            if let Some(cb) = hooks.on_step.as_mut() {
                cb(completed, total);
            }
            if hooks.preview_every > 0
                && completed % hooks.preview_every == 0
                && completed < total
            {
                if let Some(cb) = hooks.on_preview.as_mut() {
                    let img = self.decode_tokens(latent.clone(), grid_h, grid_w);
                    cb(completed, img);
                }
            }
        }

        let image = self.decode_tokens(latent, grid_h, grid_w);
        Ok((image, noise.effective_seed()))
    }
}
