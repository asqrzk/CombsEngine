//! `combs generate-image` — image generation with a local Stable Diffusion checkpoint.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;

use combs_diffusion::{DiffusionArchitecture, DiffusionModel, SchedulerKind};

#[derive(Args, Clone)]
pub struct GenerateImageArgs {
    /// Path to the model directory (HF Diffusers layout with unet/, vae/,
    /// text_encoder/ and tokenizer.json).
    #[arg(long)]
    pub model: PathBuf,
    /// Text prompt.
    #[arg(long)]
    pub prompt: String,
    /// Negative / unconditioned prompt.
    #[arg(long)]
    pub negative_prompt: Option<String>,
    /// Output image width in pixels (must be a multiple of 8).
    #[arg(long, default_value_t = 512)]
    pub width: u32,
    /// Output image height in pixels (must be a multiple of 8).
    #[arg(long, default_value_t = 512)]
    pub height: u32,
    /// Number of denoising steps.
    #[arg(long, default_value_t = 20)]
    pub steps: usize,
    /// Classifier-free guidance scale.
    #[arg(long, default_value_t = 7.5)]
    pub guidance_scale: f32,
    /// RNG seed.
    #[arg(long)]
    pub seed: Option<u64>,
    /// Denoising scheduler: ddpm | ddim | dpm++2m.
    #[arg(long, default_value = "dpm++2m")]
    pub scheduler: String,
    /// Output PNG path.
    #[arg(long, default_value = "output.png")]
    pub output: PathBuf,
    /// LoRA safetensors file to fuse into the pipeline at load time
    /// (diffusers or kohya key format).
    #[arg(long)]
    pub lora: Option<PathBuf>,
    /// LoRA strength multiplier.
    #[arg(long, default_value_t = 1.0)]
    pub lora_scale: f32,
    /// Language-model weights for recipe-assembled pipelines
    /// (flux2-klein: a Qwen3 GGUF file or safetensors directory).
    /// With --vae, --model names the transformer directory instead of
    /// a full diffusers checkout.
    #[arg(long)]
    pub llm: Option<PathBuf>,
    /// Autoencoder directory for recipe-assembled pipelines.
    #[arg(long)]
    pub vae: Option<PathBuf>,
}

pub fn cmd_generate_image(args: GenerateImageArgs) -> Result<()> {
    let model_dir = super::resolve_model_arg(&args.model)?;

    eprintln!(
        "generating {}x{} image with prompt: {}",
        args.width, args.height, args.prompt
    );

    let device = combs_core::init_device();
    let lora = args
        .lora
        .as_ref()
        .map(|path| super::resolve_lora_arg(path))
        .transpose()?
        .map(|path| combs_diffusion::LoraSpec { path, scale: args.lora_scale });
    // The recipe pipeline runs on the build's serving dtype
    // (CombsBackend: f16 in the f16 twin) — the klein DiT is 7.75 GB
    // of bf16 that must NOT widen to f32 on an 18 GB machine: Metal
    // overcommit hands back zero-reading buffers instead of failing.
    // The classic SD path keeps its proven f32 backend.
    match (&args.llm, &args.vae) {
        (Some(llm), Some(vae)) => {
            anyhow::ensure!(lora.is_none(), "LoRA is not wired for recipe pipelines yet");
            let pipeline = combs_diffusion::loader::load_flux2_klein_recipe_split::<
                combs_core::CombsBackendF32,
                combs_core::CombsBackendF32,
            >(&model_dir, llm, vae, &device, &device)
            .context("loading flux2-klein recipe")?;
            run_generate(pipeline, &args)
        }
        (None, None) => {
            let architecture = DiffusionArchitecture::detect(&model_dir)
                .context("detecting diffusion architecture")?;
            let pipeline = combs_diffusion::loader::load_diffusion_model_with_lora::<
                combs_core::CombsBackendF32,
            >(architecture, &model_dir, &device, lora.as_ref())
            .context("loading diffusion pipeline")?;
            run_generate(pipeline, &args)
        }
        _ => anyhow::bail!("--llm and --vae come as a pair (the recipe needs both)"),
    }
}

fn run_generate<B: burn::tensor::backend::Backend>(
    mut pipeline: Box<dyn DiffusionModel<B>>,
    args: &GenerateImageArgs,
) -> Result<()> {

    let scheduler = match SchedulerKind::parse(&args.scheduler) {
        Some(kind) => kind,
        // Fixed-schedule pipelines ignore the choice entirely; the success
        // line prints "flow-match-euler", and that spelling must be
        // replayable as a flag rather than failing before any work.
        None if pipeline.fixed_sampler().is_some() => SchedulerKind::default(),
        None => anyhow::bail!(
            "unknown scheduler {:?} (ddpm | ddim | dpm++2m)",
            args.scheduler
        ),
    };


    let embed = pipeline
        .encode_prompt(&args.prompt, args.negative_prompt.as_deref())
        .context("encoding prompt")?;

    let mut report = |step: usize, total: usize| {
        eprint!("\r  step {step}/{total}");
        if step == total {
            eprintln!();
        }
    };
    let (image, effective_seed) = pipeline
        .generate(
            embed,
            args.width,
            args.height,
            args.steps,
            args.guidance_scale,
            args.seed,
            scheduler,
            combs_diffusion::GenerationHooks {
                on_step: Some(&mut report),
                ..Default::default()
            },
        )
        .context("generating image")?;

    save_tensor_as_png(&image, &args.output).context("saving output image")?;
    // Report the sampler that ran, not the flag: fixed-schedule pipelines
    // (klein's flow-match) ignore --scheduler.
    println!(
        "saved {} ({}, seed {effective_seed})",
        args.output.display(),
        pipeline.fixed_sampler().unwrap_or_else(|| scheduler.name())
    );
    Ok(())
}

/// Save a `[batch, 3, height, width]` float tensor in [0, 1] as a PNG.
fn save_tensor_as_png<B: burn::tensor::backend::Backend>(
    tensor: &burn::tensor::Tensor<B, 4>,
    path: &PathBuf,
) -> Result<()> {
    let img = tensor_to_rgb_image(tensor)?;
    img.save(path)?;
    Ok(())
}

/// Convert a `[1, 3, H, W]` float tensor in [0, 1] to an RGB image.
/// Shared with `serve-images`, which encodes PNG bytes in memory.
pub(crate) fn tensor_to_rgb_image<B: burn::tensor::backend::Backend>(
    tensor: &burn::tensor::Tensor<B, 4>,
) -> Result<image::RgbImage> {
    let [batch, channels, height, width] = tensor.dims();
    anyhow::ensure!(batch == 1, "expected batch size 1, got {batch}");
    anyhow::ensure!(channels == 3, "expected 3 channels, got {channels}");

    // NaN clamps to 0 and casts to black — a numerically-exploded latent
    // would silently return a 200 with an all-black PNG. Fail loudly.
    let data: Vec<f32> = tensor
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("tensor data conversion failed: {e}"))?;
    anyhow::ensure!(
        data.iter().all(|v| v.is_finite()),
        "generation produced non-finite pixels (NaN/inf latent — check \
         guidance_scale/steps)"
    );

    // Round to nearest, matching the reference postprocess
    // ((image * 255).round()); a bare `as u8` truncates, which darkens
    // every pixel by half a step on average and turns each integer
    // boundary into a cliff where one-ulp float jitter flips the byte.
    let quant = |v: f32| (v * 255.0).round().clamp(0.0, 255.0) as u8;
    let img = image::RgbImage::from_fn(width as u32, height as u32, |x, y| {
        let y = y as usize;
        let x = x as usize;
        let plane = height * width;
        let r = quant(data[0 * plane + y * width + x]);
        let g = quant(data[1 * plane + y * width + x]);
        let b = quant(data[2 * plane + y * width + x]);
        image::Rgb([r, g, b])
    });
    Ok(img)
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::tensor::Tensor;

    // 0.999999 * 255 = 254.99975: truncation said 254, the reference's
    // round says 255. 0.5 * 255 = 127.5 rounds away to 128. The low
    // end must not lift zero, and out-of-range must clamp.
    #[test]
    fn pixel_quantization_rounds_to_nearest() {
        let device = Default::default();
        let vals = [0.999_999f32, 0.5, 0.001, 0.0, 1.2, -0.3];
        let per_plane = vals.len();
        let mut data = Vec::with_capacity(per_plane * 3);
        for _ in 0..3 {
            data.extend_from_slice(&vals);
        }
        let t: Tensor<NdArray, 4> = Tensor::<NdArray, 1>::from_floats(
            data.as_slice(),
            &device,
        )
        .reshape([1, 3, 1, per_plane]);
        let img = super::tensor_to_rgb_image(&t).unwrap();
        let got: Vec<u8> = (0..per_plane as u32).map(|x| img.get_pixel(x, 0).0[0]).collect();
        assert_eq!(got, vec![255, 128, 0, 0, 255, 0]);
    }
}
