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
    image_ids, prof_mark, prof_report, prof_reset, text_ids, unpack_latents, unpatchify_latents,
    FlowMatchEuler, Flux2Config, Flux2LatentStats, Flux2Transformer,
};
use crate::vae::VAEDecoder;
use crate::{
    DiffusionModel, GenerationHooks, NoiseSource, PromptEmbed, SchedulerKind, WorkingSet,
};

/// The reference pipeline's default encoder taps and prompt budget.
const ENCODER_TAPS: [usize; 3] = [9, 18, 27];
const MAX_PROMPT_TOKENS: usize = 512;

fn debug_enabled() -> bool {
    std::env::var("COMBS_KLEIN_DEBUG").is_ok_and(|v| v != "0")
}

fn stats<B: Backend, const D: usize>(label: &str, t: &Tensor<B, D>) {
    let v: Vec<f32> = t.clone().into_data().convert::<f32>().to_vec().unwrap_or_default();
    if v.is_empty() {
        eprintln!("[klein-debug] {label}: EMPTY");
        return;
    }
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let nan = v.iter().filter(|x| x.is_nan()).count();
    let max = v.iter().fold(f32::MIN, |m, &x| m.max(x));
    let min = v.iter().fold(f32::MAX, |m, &x| m.min(x));
    eprintln!(
        "[klein-debug] {label}: mean {mean:.4} std {:.4} min {min:.3} max {max:.3} nan {nan}",
        var.sqrt()
    );
}

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

/// `VB` is the autoencoder's backend — the decoder overflows f16
/// (GroupNorm squares + 512-wide conv stacks), so the f16 twin runs
/// it on the f32 backend over the same wgpu device; the latent hop
/// between backends is a ~0.5 MB host round-trip per decode.
pub struct Flux2KleinPipeline<B: Backend, VB: Backend = B> {
    metadata: ModelMetadata,
    device: Device<B>,
    vae_device: Device<VB>,
    transformer: Flux2Transformer<B>,
    vae: VAEDecoder<VB>,
    latent_stats: Flux2LatentStats<VB>,
    /// `prefill_taps` takes `&mut` (the cache contract), encode_prompt
    /// is `&self` — one prompt at a time behind a lock.
    encoder: Mutex<(LlamaModel<B>, Box<dyn combs_models::KVCache<B>>)>,
    tokenizer: tokenizers::Tokenizer,
    chat_template: Option<String>,
}

impl<B: Backend, VB: Backend> Flux2KleinPipeline<B, VB> {
    /// Assemble from the three-part recipe: the DiT weights, the Qwen3
    /// language model (any source the llama family loads — GGUF quants
    /// are the practical choice), and the flux2 autoencoder (on its
    /// own backend, see the type docs).
    pub fn load_recipe(
        dit: &dyn ModelSource,
        dit_config: Flux2Config,
        llm: &dyn ModelSource,
        vae: &dyn ModelSource,
        device: &Device<B>,
        vae_device: &Device<VB>,
    ) -> Result<Self> {
        let transformer = Flux2Transformer::load(dit, dit_config, device)?;
        let vae_decoder = VAEDecoder::load_with_latent_channels(vae, 32, vae_device)?;
        let latent_stats = Flux2LatentStats::load(vae, vae_device)?;

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
            vae_device: vae_device.clone(),
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

    /// Packed latent tokens → RGB in [0, 1] (computed on `VB`, handed
    /// back on `B`).
    fn decode_tokens(&self, tokens: Tensor<B, 3>, grid_h: usize, grid_w: usize) -> Tensor<B, 4> {
        let tokens: Tensor<VB, 3> = Tensor::from_data(
            tokens.into_data().convert::<f32>(),
            &self.vae_device,
        );
        let grid = unpack_latents(tokens, grid_h, grid_w);
        let denormed = self.latent_stats.denormalize(grid);
        let latent = unpatchify_latents(denormed);
        if debug_enabled() {
            stats("decode: denormed latent", &latent);
        }
        let image = self.vae.forward(latent);
        if debug_enabled() {
            stats("decode: vae output", &image);
        }
        let image = image.mul_scalar(0.5).add_scalar(0.5).clamp(0.0, 1.0);
        Tensor::from_data(image.into_data().convert::<f32>(), &self.device)
    }
}

impl<B: Backend, VB: Backend> DiffusionModel<B> for Flux2KleinPipeline<B, VB> {
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
            .encode(text.as_str(), false)
            .map_err(|e| FormatError::Safetensors(format!("klein encode: {e}")))?;
        let mut ids: Vec<i32> = encoding.get_ids().iter().map(|&t| t as i32).collect();
        ids.truncate(MAX_PROMPT_TOKENS);
        // The reference pads EVERY prompt to the full budget
        // (`padding="max_length"`) and feeds all positions — pad-derived
        // states included — to the transformer; the model calibrated to
        // that shape, and natural-length conditioning decodes to mush.
        // (Known deviation: the reference masks pad KEYS inside the
        // encoder; a causal LM with right-padding differs only in what
        // pad positions themselves attend to.)
        let pad = self
            .tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| self.tokenizer.token_to_id("<|im_end|>"))
            .unwrap_or(0) as i32;
        ids.resize(MAX_PROMPT_TOKENS, pad);
        let seq = ids.len();

        let mut guard = self.encoder.lock().expect("encoder lock");
        let (encoder, cache) = &mut *guard;
        cache.reset();
        prof_reset();
        let tokens: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::new(ids, [1, seq]), &self.device);
        let embedded = encoder.embed(tokens);
        let positive = encoder
            .prefill_taps(embedded, cache.as_mut(), 0..seq as u32, &ENCODER_TAPS)
            .map_err(|e| FormatError::Safetensors(format!("klein taps: {e}")))?;
        cache.reset();
        prof_mark("encode", &positive);
        prof_report("encode");
        if debug_enabled() {
            eprintln!("[klein-debug] wrapped prompt ({} tokens): {:?}...", seq, &text[..text.len().min(120)]);
            stats("prompt embeds", &positive);
        }

        // Distilled klein runs guidance-free; the negative arm exists
        // only for future -base weights and stays empty here.
        Ok(PromptEmbed { positive, negative: None })
    }

    fn fixed_sampler(&self) -> Option<&'static str> {
        Some("flow-match-euler")
    }

    /// Fitted to a controlled series on THIS recipe (Q8 transformer +
    /// Q4_K_M Qwen3-4B encoder + f32 autoencoder, 18 GB M3 Pro): four
    /// steps, previews off, one fresh process per point, because the
    /// device pool never shrinks and a second run in the same process
    /// measures the first one's high-water. 256x256 added 1828 MB over
    /// the resident weights; 512x512 added 3649 MB.
    ///
    /// The earlier pair this replaced differed in two variables at once
    /// (size AND whether previews ran) and came from a build before the
    /// encoder joined the batched matmul path — it under-estimated
    /// 512x512 by 782 MB, which is the direction that lets a run
    /// through that should have been refused. Re-measure after any
    /// change to how weights are staged: this curve describes the code
    /// as much as the model.
    ///
    /// Per-step previews cost ~207 MB on top (measured, same series) —
    /// inside the caller's headroom, so the curve stays preview-free
    /// rather than pretending to know the cadence.
    fn working_set(&self) -> Option<WorkingSet> {
        Some(WorkingSet {
            fixed_bytes: 1_280_311_296,
            bytes_per_pixel: 9_712,
            measured_max_pixels: 512 * 512,
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
        if debug_enabled() {
            stats("initial latent", &latent);
        }
        for i in 0..total {
            prof_reset();
            let velocity = self.transformer.forward(
                latent.clone(),
                prompt.positive.clone(),
                schedule.timestep(i),
                &img_ids,
                &txt_ids,
            );
            if debug_enabled() {
                eprintln!("[klein-debug] step {i}: sigma {}", schedule.timestep(i));
                stats("velocity", &velocity);
            }
            latent = schedule.step(latent, velocity, i);
            if debug_enabled() {
                stats("latent", &latent);
            }
            let completed = i + 1;
            prof_mark("sched", &latent);
            prof_report(&format!("step {completed}/{total}"));
            if let Some(cb) = hooks.on_step.as_mut() {
                // wgpu ops are lazy: without a readback the callback would
                // fire at queue-submission time and step counts / ETAs
                // would describe enqueued, not finished, work. One element
                // is enough to force completion.
                let _ = latent.clone().narrow(1, 0, 1).narrow(2, 0, 1).into_data();
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

        prof_reset();
        let image = self.decode_tokens(latent, grid_h, grid_w);
        prof_mark("decode", &image);
        prof_report("decode");
        Ok((image, noise.effective_seed()))
    }
}
