//! `combs generate-image` — image generation with a local Stable Diffusion checkpoint.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;

use combs_diffusion::{
    DiffusionArchitecture, DiffusionModel, SchedulerKind, load_diffusion_model,
};

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
    let mut pipeline = match (&args.llm, &args.vae) {
        (Some(llm), Some(vae)) => {
            anyhow::ensure!(lora.is_none(), "LoRA is not wired for recipe pipelines yet");
            combs_diffusion::loader::load_flux2_klein_recipe::<combs_core::CombsBackendF32>(
                &model_dir, llm, vae, &device,
            )
            .context("loading flux2-klein recipe")?
        }
        (None, None) => {
            let architecture = DiffusionArchitecture::detect(&model_dir)
                .context("detecting diffusion architecture")?;
            combs_diffusion::loader::load_diffusion_model_with_lora::<
                combs_core::CombsBackendF32,
            >(architecture, &model_dir, &device, lora.as_ref())
            .context("loading diffusion pipeline")?
        }
        _ => anyhow::bail!("--llm and --vae come as a pair (the recipe needs both)"),
    };

    let scheduler = SchedulerKind::parse(&args.scheduler)
        .with_context(|| format!("unknown scheduler {:?} (ddpm | ddim | dpm++2m)", args.scheduler))?;

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
    println!(
        "saved {} ({}, seed {effective_seed})",
        args.output.display(),
        scheduler.name()
    );
    Ok(())
}

/// Save a `[batch, 3, height, width]` float tensor in [0, 1] as a PNG.
fn save_tensor_as_png(tensor: &burn::tensor::Tensor<combs_core::CombsBackendF32, 4>, path: &PathBuf) -> Result<()> {
    let img = tensor_to_rgb_image(tensor)?;
    img.save(path)?;
    Ok(())
}

/// Convert a `[1, 3, H, W]` float tensor in [0, 1] to an RGB image.
/// Shared with `serve-images`, which encodes PNG bytes in memory.
pub(crate) fn tensor_to_rgb_image(
    tensor: &burn::tensor::Tensor<combs_core::CombsBackendF32, 4>,
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

    let img = image::RgbImage::from_fn(width as u32, height as u32, |x, y| {
        let y = y as usize;
        let x = x as usize;
        let plane = height * width;
        let r = (data[0 * plane + y * width + x] * 255.0).clamp(0.0, 255.0) as u8;
        let g = (data[1 * plane + y * width + x] * 255.0).clamp(0.0, 255.0) as u8;
        let b = (data[2 * plane + y * width + x] * 255.0).clamp(0.0, 255.0) as u8;
        image::Rgb([r, g, b])
    });
    Ok(img)
}
