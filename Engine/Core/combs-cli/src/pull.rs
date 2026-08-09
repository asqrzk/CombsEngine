//! `combs pull` — downloads a model from Hugging Face into the local
//! cache (`$COMBS_HOME/models/<id>`, default `~/.cache/combs/models/<id>`).
//!
//! The cache holds PLAINTEXT weights: the engine (L0, trusted compute
//! boundary) mmaps safetensors/GGUF directly from disk — encrypting public
//! weights would break mmap and add load-time cost for zero threat-model
//! gain. Zero-trust encryption-at-rest covers data crossing the proxy
//! boundary (chats, downloads, agent data), not the engine's model store.
//!
//! Accepts a preset id (`smollm2-135m`) or a full HF repo
//! (`HuggingFaceTB/SmolLM2-135M-Instruct`).

use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use console::style;

/// Files every preset needs (tokenizer.json covers vocab/merges).
const FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
];

/// Preset id → Hugging Face repo (must match @combs/core presets).
pub const PRESETS: &[(&str, &str)] = &[
    ("smollm2-135m", "HuggingFaceTB/SmolLM2-135M-Instruct"),
    ("smollm2-360m", "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("smolvlm-256m", "HuggingFaceTB/SmolVLM-256M-Instruct"),
    ("sd-1.5", "runwayml/stable-diffusion-v1-5"),
];

pub fn resolve_repo(source: &str) -> (String, String) {
    for (id, repo) in PRESETS {
        if source.eq_ignore_ascii_case(id) || source.eq_ignore_ascii_case(repo) {
            return ((*id).to_string(), (*repo).to_string());
        }
    }
    // Full repo id: cache under the model slug.
    let slug = source.rsplit('/').next().unwrap_or(source).to_lowercase();
    (slug, source.to_string())
}

pub fn cache_root() -> Result<PathBuf> {
    let home = std::env::var("COMBS_HOME")
        .map(PathBuf::from)
        .or_else(|_| {
            std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .map(|h| PathBuf::from(h).join(".cache/combs"))
        })
        .context("cannot locate a home directory (set COMBS_HOME)")?;
    Ok(home.join("models"))
}

/// True when `source` resolves to a preset/repo already in the cache.
pub fn cached_dir(source: &str) -> Option<PathBuf> {
    let (id, _) = resolve_repo(source);
    let dir = cache_root().ok()?.join(id);
    dir.join("model.safetensors").is_file().then_some(dir)
}

/// True when `source` resolves to a diffusion checkpoint in the cache.
pub fn cached_diffusion_dir(source: &str) -> Option<PathBuf> {
    let (id, _) = resolve_repo(source);
    let dir = cache_root().ok()?.join(id);
    if dir.join("unet").is_dir()
        && dir.join("vae").is_dir()
        && dir.join("text_encoder").is_dir()
    {
        Some(dir)
    } else {
        None
    }
}

/// Downloads `source` (preset id or HF repo) into the cache; returns the
/// cache directory. Existing files are skipped (resume-friendly).
pub fn pull(source: &str, diffusion: bool) -> Result<PathBuf> {
    if diffusion {
        return pull_diffusion(source);
    }

    let (id, repo) = resolve_repo(source);
    let dir = cache_root()?.join(&id);
    fs::create_dir_all(&dir)?;

    println!(
        "{}",
        style(format!("pulling {repo} → {}", dir.display())).bold().cyan()
    );
    for file in FILES {
        let target = dir.join(file);
        if target.is_file() {
            println!("  {} {file}", style("✓ cached").dim());
            continue;
        }
        let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
        download(&url, &target).with_context(|| format!("downloading {file}"))?;
    }
    println!("{}", style("model ready").bold().green());
    Ok(dir)
}

/// Stable Diffusion checkpoint layout used by `combs-diffusion`:
///   unet/diffusion_pytorch_model.safetensors
///   vae/diffusion_pytorch_model.safetensors
///   text_encoder/model.safetensors
///   tokenizer.json   (from openai/clip-vit-large-patch14)
fn pull_diffusion(source: &str) -> Result<PathBuf> {
    let repo = source.to_string();
    let slug = source.rsplit('/').next().unwrap_or(source).to_lowercase();
    let dir = cache_root()?.join(slug);

    println!(
        "{}",
        style(format!("pulling diffusion checkpoint {repo} → {}", dir.display())).bold().cyan()
    );

    // UNet: prefer fp16, fall back to full precision.
    download_subdir_file(&repo, &dir, "unet", "diffusion_pytorch_model", &[".fp16.safetensors", ".safetensors"])?;
    // VAE
    download_subdir_file(&repo, &dir, "vae", "diffusion_pytorch_model", &[".fp16.safetensors", ".safetensors"])?;
    // Text encoder (CLIP)
    download_subdir_file(&repo, &dir, "text_encoder", "model", &[".fp16.safetensors", ".safetensors"])?;

    // CLIP tokenizer — SD 1.5 uses openai/clip-vit-large-patch14's tokenizer.json.
    let tok_target = dir.join("tokenizer.json");
    if !tok_target.is_file() {
        let tok_url = "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer.json";
        download(tok_url, &tok_target).context("downloading CLIP tokenizer.json")?;
    } else {
        println!("  {} tokenizer.json", style("✓ cached").dim());
    }

    println!("{}", style("diffusion checkpoint ready").bold().green());
    Ok(dir)
}

/// Try a list of suffixes in order and save the first hit as `{name}.safetensors`.
fn download_subdir_file(repo: &str, dir: &PathBuf, subdir: &str, name: &str, suffixes: &[&str]) -> Result<()> {
    let target = dir.join(subdir).join(format!("{name}.safetensors"));
    if target.is_file() {
        println!("  {} {subdir}/{name}.safetensors", style("✓ cached").dim());
        return Ok(());
    }
    fs::create_dir_all(target.parent().unwrap())?;
    for suffix in suffixes {
        let url = format!("https://huggingface.co/{repo}/resolve/main/{subdir}/{name}{suffix}");
        match download(&url, &target) {
            Ok(_) => return Ok(()),
            Err(_) if suffix != suffixes.last().unwrap() => continue,
            Err(e) => return Err(e).with_context(|| format!("downloading {subdir}/{name}.safetensors")),
        }
    }
    Ok(())
}

fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGINGFACE_TOKEN"))
        .ok()
}

fn download(url: &str, target: &PathBuf) -> Result<()> {
    let part = target.with_extension("part");
    let mut req = ureq::get(url);
    if let Some(token) = hf_token() {
        req = req.header("Authorization", &format!("Bearer {token}"));
    }
    let resp = req.call().context("request failed")?;
    let total: u64 = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let mut reader = resp.into_body().into_reader();
    let mut out = fs::File::create(&part)?;
    let mut buf = vec![0u8; 512 * 1024];
    let mut done: u64 = 0;
    let mut last_mb = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        done += n as u64;
        let mb = done / (10 * 1024 * 1024);
        if mb != last_mb {
            last_mb = mb;
            eprint!(
                "\r  {} {:.0}/{:.0} MB",
                style(target.file_name().unwrap().to_string_lossy().to_string()).bold(),
                done as f64 / 1e6,
                total as f64 / 1e6
            );
        }
    }
    eprintln!();
    out.flush()?;
    if total > 0 && done != total {
        fs::remove_file(&part).ok();
        bail!("short download: {done}/{total} bytes");
    }
    fs::rename(&part, target)?;
    println!("  {} {} ({:.1} MB)", style("✓").green(), target.file_name().unwrap().to_string_lossy(), done as f64 / 1e6);
    Ok(())
}
