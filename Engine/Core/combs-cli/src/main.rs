//! `combs` — Combs Engine CLI.
//!
//! Phase 1: `run` (streaming generation) and `devices` work;
//! `pull` / `convert` / `serve` are stubs for later phases.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};

use combs_formats::{ModelSource, open_model_source};
use combs_media::ImagePreprocessor;
use combs_runtime::{Engine, GenerationConfig};

mod chew;
mod generate_audio;
mod generate_image;
mod http;
mod pull;
mod serve;
mod serve_images;

#[derive(Parser)]
#[command(name = "combs", version, about = "Combs Engine — cross-platform edge AI inference")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Args, Clone)]
struct RunArgs {
    /// Path to the model directory (HF safetensors layout).
    #[arg(long)]
    model: PathBuf,
    /// Prompt text.
    #[arg(long)]
    prompt: String,
    /// Maximum number of tokens to generate (default: model's
    /// generation_config.json, else 64).
    #[arg(long)]
    max_tokens: Option<usize>,
    /// Sampling temperature (0 = greedy; default: model's generation_config).
    #[arg(long)]
    temperature: Option<f32>,
    /// Top-k cutoff (0 = disabled).
    #[arg(long)]
    top_k: Option<usize>,
    /// Nucleus (top-p) threshold.
    #[arg(long)]
    top_p: Option<f32>,
    /// HF-style repetition penalty (1.0 = disabled).
    #[arg(long)]
    repetition_penalty: Option<f32>,
    /// OpenAI-style frequency penalty.
    #[arg(long)]
    frequency_penalty: Option<f32>,
    /// OpenAI-style presence penalty.
    #[arg(long)]
    presence_penalty: Option<f32>,
    /// RNG seed for reproducible sampling.
    #[arg(long)]
    seed: Option<u64>,
    /// Stop string (repeatable); generation halts when one appears in the output.
    #[arg(long = "stop")]
    stop_strings: Vec<String>,
    /// Wrap the prompt in the model's ChatML template and stop at <|im_end|>.
    #[arg(long)]
    chat: bool,
    /// Attach an image (repeatable; requires a vision model like SmolVLM).
    #[arg(long = "image")]
    images: Vec<PathBuf>,
    /// Prompt tokens per prefill call (0 = single-shot; default 512).
    /// Values >= 512 hit a burn/wgpu prefill miscompilation on long prompts.
    #[arg(long)]
    prefill_chunk_size: Option<usize>,
}

#[derive(Subcommand)]
enum Command {
    /// Run a local model and stream generated text.
    Run(RunArgs),
    /// Print wgpu device information.
    Devices,
    /// Download a model into the local cache (~/.cache/combs/models).
    Pull {
        /// Preset id (smollm2-135m) or HuggingFace repo (org/name).
        source: String,
    },
    /// Convert/repackage a model (Phase 5).
    Convert {
        /// Input model path.
        input: PathBuf,
    },
    /// Generate an image with a local Stable Diffusion checkpoint.
    GenerateImage(generate_image::GenerateImageArgs),
    /// Generate speech (WAV) with a local Kokoro ONNX TTS checkpoint.
    GenerateAudio(generate_audio::GenerateAudioArgs),
    /// Start a persistent image-generation worker (loads the diffusion
    /// pipeline once, serves OpenAI-style /v1/images/generations).
    ServeImages {
        /// Path to the Diffusers checkpoint (unet/, vae/, text_encoder/).
        #[arg(long)]
        model: PathBuf,
        /// Port to listen on.
        #[arg(long, default_value_t = 8082)]
        port: u16,
    },
    /// Start an OpenAI-compatible HTTP server.
    Serve {
        /// Path to the model directory (HF safetensors layout) or .gguf file.
        #[arg(long)]
        model: PathBuf,
        /// Port to listen on.
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// KV arena size in tokens. Default: min(model's
        /// max_position_embeddings, 32768). Raise for long-context models
        /// when memory allows.
        #[arg(long)]
        context_size: Option<usize>,
        /// Prompt tokens per prefill call (0 = single-shot; default 512).
        #[arg(long)]
        prefill_chunk_size: Option<usize>,
    },
    /// Scaffold a UI app from the embedded template (interactive or via flags).
    Chew(ChewCommand),
}

#[derive(Args)]
struct ChewCommand {
    #[command(subcommand)]
    mode: ChewMode,
}

#[derive(Subcommand)]
enum ChewMode {
    /// Scaffold a chat UI app.
    ChatUi(chew::ChewArgs),
    /// Scaffold a multi-agent debate UI app.
    DebateUi(chew::ChewArgs),
    /// Scaffold a two-role roleplay UI (each role on its own engine process).
    RoleplayUi(chew::ChewArgs),
    /// Scaffold a multi-turn chat UI with the Control Tower observability view.
    MultiTurnUi(chew::ChewArgs),
    /// Scaffold a multi-agent orchestration UI (scenario + roles) watched from the Control Tower.
    OrchestrationObserveUi(chew::ChewArgs),
    /// Scaffold a chat UI demonstrating rolling-session KV cache reuse (live hit stats).
    KvCacheUi(chew::ChewArgs),
    /// Scaffold a debate UI whose agents each keep a named KV session (live hit stats).
    DebateKvUi(chew::ChewArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => cmd_run(args),
        Command::Devices => cmd_devices(),
        Command::Pull { source } => {
            let dir = pull::pull(&source)?;
            println!("cached at: {}", dir.display());
            Ok(())
        }
        Command::Convert { .. } => not_yet("convert", "Phase 5 (GGUF/burnpack adapters)"),
        Command::GenerateImage(args) => generate_image::cmd_generate_image(args),
        Command::GenerateAudio(args) => generate_audio::cmd_generate_audio(args),
        Command::ServeImages { model, port } => serve_images::cmd_serve_images(model, port),
        Command::Serve { model, port, context_size, prefill_chunk_size } => {
            cmd_serve(model, port, context_size, prefill_chunk_size)
        }
        Command::Chew(cmd) => match cmd.mode {
            ChewMode::ChatUi(args) => chew::chew("chat-ui", args),
            ChewMode::DebateUi(args) => chew::chew("debate-ui", args),
            ChewMode::RoleplayUi(args) => chew::chew("roleplay-ui", args),
            ChewMode::MultiTurnUi(args) => chew::chew("multi-turn-ui", args),
            ChewMode::OrchestrationObserveUi(args) => chew::chew("orchestration-observe-ui", args),
            ChewMode::KvCacheUi(args) => chew::chew("kv-cache-ui", args),
            ChewMode::DebateKvUi(args) => chew::chew("debate-kv-ui", args),
        },
    }
}

fn not_yet(cmd: &str, phase: &str) -> Result<()> {
    eprintln!("`combs {cmd}` is not yet implemented — planned for {phase}.");
    std::process::exit(2);
}

/// Resolves a --model argument: direct path, or a preset id in the local
/// cache (see `combs pull`).
fn resolve_model_arg(model: &PathBuf) -> Result<PathBuf> {
    if model.exists() {
        return Ok(model.clone());
    }
    let name = model.to_string_lossy();
    if let Some(dir) = pull::cached_dir(&name) {
        return Ok(dir);
    }
    if let Some(dir) = pull::cached_diffusion_dir(&name) {
        return Ok(dir);
    }
    if let Some(dir) = pull::cached_tts_dir(&name) {
        return Ok(dir);
    }
    anyhow::bail!("model '{name}' not found — download it first: combs pull {name}")
}

fn cmd_serve(
    model: PathBuf,
    port: u16,
    context_size: Option<usize>,
    prefill_chunk_size: Option<usize>,
) -> Result<()> {
    let model = resolve_model_arg(&model)?;
    let source = open_model_source(&model)?;
    let model_id = model
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "combs-model".to_string());
    eprintln!("loading weights...");
    let device = combs_core::init_device();
    let engine = if let Some(size) = context_size {
        let mut cc = combs_runtime::CacheConfig::paged(size);
        if matches!(std::env::var("COMBS_KV").as_deref(), Ok("contiguous")) {
            cc.kind = combs_runtime::CacheKind::Contiguous;
        }
        std::sync::Arc::new(Engine::load_with_cache_config(&source, device, cc)?)
    } else {
        std::sync::Arc::new(Engine::load(&source, device)?)
    };
    serve::serve(engine, model_id, &format!("0.0.0.0:{port}"), prefill_chunk_size)
}

fn cmd_devices() -> Result<()> {
    let device = combs_core::init_device();
    let info = combs_core::device_info(&device);
    println!("wgpu device:");
    println!("  name:        {}", info.name);
    println!("  backend:     {}", info.backend);
    println!("  device type: {}", info.device_type);
    println!("  driver:      {}", info.driver);
    Ok(())
}

fn cmd_run(args: RunArgs) -> Result<()> {
    let model = resolve_model_arg(&args.model)?;
    let source = open_model_source(&model)
        .with_context(|| format!("loading {}", model.display()))?;
    let meta = source.metadata();
    eprintln!(
        "model: {} ({} layers, hidden {}, vocab {}, {})",
        meta.architecture,
        meta.num_hidden_layers,
        meta.hidden_size,
        meta.vocab_size,
        if meta.tie_word_embeddings {
            "tied embeddings"
        } else {
            "separate lm_head"
        }
    );

    let device = combs_core::init_device();
    let mut pixel_batches = Vec::new();
    let prompt = if !args.images.is_empty() {
        // VLM path: Idefics3/SmolVLM template with one expanded image span
        // per attached image, then the question.
        let vision = meta
            .vision
            .clone()
            .context("--image needs a vision model (config.json has no vision_config)")?;
        let pp = combs_media::SiglipPreprocessor::new(vision.image_size);
        let mut prompt = String::from("<|im_start|>User:");
        for path in &args.images {
            let bytes = std::fs::read(path)
                .with_context(|| format!("reading {}", path.display()))?;
            let batch = pp
                .preprocess(&bytes)
                .map_err(|e| anyhow::anyhow!("preprocessing {}: {e}", path.display()))?;
            eprintln!("image: {} ({}x{})", path.display(), batch.width, batch.height);
            pixel_batches.push(batch);
            prompt.push_str(&combs_models::image_prompt_expansion(vision.image_seq_len()));
        }
        prompt.push_str(&args.prompt);
        prompt.push_str("\nAssistant:");
        prompt
    } else if args.chat {
        source
            .tokenizer()?
            .chatml_wrap(&args.prompt)
            .context("tokenizer has no <|im_start|>/<|im_end|> tokens; cannot use --chat")?
    } else {
        args.prompt
    };

    eprintln!("loading weights...");
    let engine = Engine::load(&source, device)?;
    // Real GPU-allocator numbers (weights resident after load) — this is
    // where packed-quant weights show their VRAM win; process RSS/footprint
    // is dominated by pool reservations and unified-memory accounting.
    if let Ok(mem) =
        <cubecl::wgpu::WgpuRuntime as cubecl::prelude::Runtime>::client(&Default::default())
            .memory_usage()
    {
        eprintln!("gpu memory after load: {mem}");
    }
    let cc = engine.cache_config();
    eprintln!(
        "kv cache: {:?} (max_seq_len {}, page size {})",
        cc.kind, cc.max_seq_len, cc.page_size
    );

    let tokens = engine.encode(&prompt)?;
    eprintln!("prompt: {} tokens", tokens.len());

    // Start from the engine defaults (model generation_config merged in);
    // explicit CLI flags always win.
    let mut config: GenerationConfig = engine.default_config();
    if let Some(m) = args.max_tokens {
        config.max_tokens = m;
    }
    let sp = &mut config.sampling;
    if let Some(t) = args.temperature {
        sp.temperature = t;
    }
    if args.top_k.is_some() {
        sp.top_k = args.top_k;
    }
    if args.top_p.is_some() {
        sp.top_p = args.top_p;
    }
    if args.repetition_penalty.is_some() {
        sp.repetition_penalty = args.repetition_penalty;
    }
    if args.frequency_penalty.is_some() {
        sp.frequency_penalty = args.frequency_penalty;
    }
    if args.presence_penalty.is_some() {
        sp.presence_penalty = args.presence_penalty;
    }
    if args.seed.is_some() {
        sp.seed = args.seed;
    }
    config.stop_strings = args.stop_strings;
    if let Some(c) = args.prefill_chunk_size {
        config.prefill_chunk_size = c;
    }

    if args.chat {
        if let Some(id) = engine.im_end_id() {
            config.stop_token_ids.push(id);
        }
    }

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut emit = |_id: u32, piece: &str| {
        let _ = out.write_all(piece.as_bytes());
        let _ = out.flush();
    };
    let stats = if pixel_batches.is_empty() {
        engine.generate(&tokens, &config, &mut emit)?
    } else {
        engine.generate_with_media(&tokens, pixel_batches, &config, &mut emit)?
    };

    println!();
    let cache_note = if stats.cache_pages_used > 0 {
        format!(" | kv cache {} pages", stats.cache_pages_used)
    } else {
        String::new()
    };
    eprintln!(
        "---\n{} prompt tokens ({:.0} tok/s prefill), {} generated | TTFT {:.0} ms | decode {:.1} tok/s | total {:.2} s{cache_note}",
        stats.prompt_tokens,
        stats.prefill_tokens_per_second(),
        stats.generated_tokens,
        stats.ttft.as_secs_f64() * 1e3,
        stats.decode_tokens_per_second(),
        stats.total_time.as_secs_f64(),
    );
    Ok(())
}
