//! `combs` — Combs Engine CLI.
//!
//! Phase 1: `run` (streaming generation) and `devices` work;
//! `pull` / `convert` / `serve` are stubs for later phases.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};

use combs_formats::{ModelSource, open_model_source};
use combs_runtime::{Engine, GenerationConfig};

mod chew;
mod serve;

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
    /// Download a model into the local store (Phase 3+).
    Pull {
        /// HuggingFace repo id or URL.
        source: String,
    },
    /// Convert/repackage a model (Phase 5).
    Convert {
        /// Input model path.
        input: PathBuf,
    },
    /// Start an OpenAI-compatible HTTP server.
    Serve {
        /// Path to the model directory (HF safetensors layout) or .gguf file.
        #[arg(long)]
        model: PathBuf,
        /// Port to listen on.
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Scaffold a chat UI app (interactive or via flags).
    ChewChatUi(chew::ChewArgs),
    /// Scaffold a multi-agent debate UI app (interactive or via flags).
    ChewDebateUi(chew::ChewArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => cmd_run(args),
        Command::Devices => cmd_devices(),
        Command::Pull { .. } => not_yet("pull", "Phase 5 (model store)"),
        Command::Convert { .. } => not_yet("convert", "Phase 5 (GGUF/burnpack adapters)"),
        Command::Serve { model, port } => cmd_serve(model, port),
        Command::ChewChatUi(args) => chew::chew("chat-ui", args),
        Command::ChewDebateUi(args) => chew::chew("debate-ui", args),
    }
}

fn not_yet(cmd: &str, phase: &str) -> Result<()> {
    eprintln!("`combs {cmd}` is not yet implemented — planned for {phase}.");
    std::process::exit(2);
}

fn cmd_serve(model: PathBuf, port: u16) -> Result<()> {
    let source = open_model_source(&model)?;
    let model_id = model
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "combs-model".to_string());
    eprintln!("loading weights...");
    let engine = std::sync::Arc::new(Engine::load(&source, combs_core::init_device())?);
    serve::serve(engine, model_id, &format!("0.0.0.0:{port}"))
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
    let source = open_model_source(&args.model)
        .with_context(|| format!("loading {}", args.model.display()))?;
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
    let prompt = if args.chat {
        source
            .tokenizer()?
            .chatml_wrap(&args.prompt)
            .context("tokenizer has no <|im_start|>/<|im_end|> tokens; cannot use --chat")?
    } else {
        args.prompt
    };

    eprintln!("loading weights...");
    let engine = Engine::load(&source, device)?;
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
    let stats = engine.generate(&tokens, &config, |_id, piece| {
        let _ = out.write_all(piece.as_bytes());
        let _ = out.flush();
    })?;

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
