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

/// Downloads `source` (preset id or HF repo) into the cache; returns the
/// cache directory. Existing files are skipped (resume-friendly).
pub fn pull(source: &str) -> Result<PathBuf> {
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

fn download(url: &str, target: &PathBuf) -> Result<()> {
    let part = target.with_extension("part");
    let resp = ureq::get(url).call().context("request failed")?;
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
