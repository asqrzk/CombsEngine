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
//! (`HuggingFaceTB/SmolLM2-135M-Instruct`). Models are detected from the
//! remote file tree: text/vision (config.json), diffusion (model_index.json),
//! or GGUF (single `.gguf` file).

use std::collections::HashSet;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use console::style;

/// Preset id, HF repo, cache slug.
pub const PRESETS: &[(&str, &str, &str)] = &[
    ("smollm2-135m", "HuggingFaceTB/SmolLM2-135M-Instruct", "smollm2-135m"),
    ("smollm2-360m", "HuggingFaceTB/SmolLM2-360M-Instruct", "smollm2-360m"),
    ("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "smollm2-1.7b"),
    ("qwen3-0.6b", "Qwen/Qwen3-0.6B", "qwen3-0.6b"),
    ("phi-3.1-mini", "bartowski/Phi-3.1-mini-4k-instruct-GGUF", "phi-3.1-mini-4k-instruct-gguf"),
    ("smolvlm-256m", "HuggingFaceTB/SmolVLM-256M-Instruct", "smolvlm-256m"),
    ("sd-1.5", "runwayml/stable-diffusion-v1-5", "stable-diffusion-v1-5"),
    ("kokoro-82m", "onnx-community/Kokoro-82M-v1.0-ONNX", "kokoro-82m"),
];

/// Quant-GGUF repos that need the original checkpoint's `tokenizer.json`
/// staged as a sibling: SPM-vocab GGUFs (phi-3's `tokenizer.ggml.model =
/// "llama"`) can't ride the BPE tokenizer synthesis, and quant repos don't
/// ship the file. Keyed by cache slug → source repo for the tokenizer.
const GGUF_TOKENIZER_COMPANIONS: &[(&str, &str)] = &[
    ("phi-3.1-mini-4k-instruct-gguf", "microsoft/Phi-3-mini-4k-instruct"),
];

#[derive(Debug, Clone, serde::Deserialize)]
struct TreeEntry {
    path: String,
    #[serde(rename = "type")]
    r#type: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    TextVision,
    Diffusion,
    Gguf,
    /// Kokoro-style ONNX TTS: voices/*.bin style vectors + an .onnx model.
    Tts,
}

pub fn resolve_repo(source: &str) -> (String, String) {
    for (id, repo, slug) in PRESETS {
        if source.eq_ignore_ascii_case(id) || source.eq_ignore_ascii_case(repo) {
            return ((*slug).to_string(), (*repo).to_string());
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

/// Returns the cached path for a text/vision or GGUF model, if present.
pub fn cached_dir(source: &str) -> Option<PathBuf> {
    let (id, _) = resolve_repo(source);
    let dir = cache_root().ok()?.join(id);
    if dir.join("model.safetensors").is_file() {
        return Some(dir);
    }
    // GGUF is cached as a single file inside the slug dir.
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            if entry.path().extension().is_some_and(|e| e == "gguf") {
                return Some(entry.path());
            }
        }
    }
    None
}

/// Returns the cached path for a Kokoro-style ONNX TTS model, if present.
pub fn cached_tts_dir(source: &str) -> Option<PathBuf> {
    let (id, _) = resolve_repo(source);
    let dir = cache_root().ok()?.join(id);
    if dir.join("voices").is_dir() {
        Some(dir)
    } else {
        None
    }
}

/// Returns the cached path for a diffusion checkpoint, if present.
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
/// cache directory (or file, for GGUF). Existing files are skipped.
pub fn pull(source: &str) -> Result<PathBuf> {
    let (slug, repo) = resolve_repo(source);
    let dir = cache_root()?.join(&slug);
    fs::create_dir_all(&dir)?;

    let tree = fetch_tree(&repo)?;
    let kind = detect_kind(&tree)?;

    println!(
        "{}",
        style(format!("pulling {repo} → {}", dir.display())).bold().cyan()
    );

    match kind {
        Kind::Diffusion => pull_diffusion(&repo, &dir, &tree),
        Kind::TextVision => pull_text(&repo, &dir, &tree),
        Kind::Gguf => pull_gguf(&repo, &dir, &tree),
        Kind::Tts => pull_tts(&repo, &dir, &tree),
    }?;

    println!("{}", style("model ready").bold().green());
    Ok(dir)
}

fn fetch_tree(repo: &str) -> Result<Vec<TreeEntry>> {
    let url = format!(
        "https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    );
    let mut req = ureq::get(&url);
    if let Some(token) = hf_token() {
        req = req.header("Authorization", &format!("Bearer {token}"));
    }
    let resp = req
        .call()
        .with_context(|| format!("fetching file tree for {repo}"))?;
    let text = resp
        .into_body()
        .read_to_string()
        .context("reading tree response")?;
    let mut entries: Vec<TreeEntry> =
        serde_json::from_str(&text).context("parsing HF tree JSON")?;
    entries.retain(|e| e.r#type == "file");
    Ok(entries)
}

fn detect_kind(tree: &[TreeEntry]) -> Result<Kind> {
    let paths: HashSet<&str> = tree.iter().map(|e| e.path.as_str()).collect();
    if paths.contains("model_index.json") {
        return Ok(Kind::Diffusion);
    }
    // TTS before TextVision: Kokoro-style repos also carry a config.json,
    // but the voices/ style-vector dir is unique to them.
    if paths.iter().any(|p| p.starts_with("voices/") && p.ends_with(".bin"))
        && paths.iter().any(|p| p.ends_with(".onnx"))
    {
        return Ok(Kind::Tts);
    }
    if paths.contains("config.json") {
        return Ok(Kind::TextVision);
    }
    if paths.iter().any(|p| p.ends_with(".gguf")) {
        return Ok(Kind::Gguf);
    }
    bail!(
        "cannot detect model kind: no config.json, model_index.json, or .gguf found in the repo"
    )
}

fn pull_text(repo: &str, dir: &PathBuf, tree: &[TreeEntry]) -> Result<()> {
    let paths: HashSet<&str> = tree.iter().map(|e| e.path.as_str()).collect();

    // Required/optional config files.
    let required = ["config.json", "tokenizer.json"];
    for file in required {
        if paths.contains(file) {
            download_required(repo, dir, file)?;
        } else {
            bail!("text model missing required file: {file}");
        }
    }
    download_optional(repo, dir, "generation_config.json")?;
    download_optional(repo, dir, "tokenizer_config.json")?;

    // Weight files: single → sharded → GGUF.
    if paths.contains("model.safetensors") {
        download_required(repo, dir, "model.safetensors")?;
    } else if paths.contains("model.safetensors.index.json") {
        download_required(repo, dir, "model.safetensors.index.json")?;
        let index_text = fs::read_to_string(dir.join("model.safetensors.index.json"))?;
        let index: serde_json::Value =
            serde_json::from_str(&index_text).context("parsing safetensors index")?;
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .context("model.safetensors.index.json missing weight_map")?;
        let mut shard_files: Vec<&str> = weight_map
            .values()
            .filter_map(|v| v.as_str())
            .collect();
        shard_files.sort();
        shard_files.dedup();
        for file in shard_files {
            download_required(repo, dir, file)?;
        }
    } else if let Some(file) = pick_safetensors_file(tree) {
        download_required(repo, dir, &file)?;
    } else if let Some(file) = pick_gguf_file(tree) {
        // Save GGUF as model.gguf inside the slug dir so serve can find it.
        let target = dir.join("model.gguf");
        download_required_as(repo, &target, &format!("{repo}/resolve/main/{file}"))?;
    } else {
        bail!("no safetensors or GGUF weight files found");
    }

    Ok(())
}

/// Kokoro-style ONNX TTS repo: tokenizer + one .onnx + all voice styles.
fn pull_tts(repo: &str, dir: &PathBuf, tree: &[TreeEntry]) -> Result<()> {
    let paths: HashSet<&str> = tree.iter().map(|e| e.path.as_str()).collect();

    download_optional(repo, dir, "config.json")?;
    if paths.contains("tokenizer.json") {
        download_required(repo, dir, "tokenizer.json")?;
    } else {
        bail!("TTS repo missing tokenizer.json");
    }

    // Model: prefer the q8 dynamic quant (small, CPU-friendly), then fp32.
    let onnx_prefs = [
        "onnx/model_quantized.onnx",
        "onnx/model.onnx",
        "model_quantized.onnx",
        "model.onnx",
    ];
    let onnx = onnx_prefs
        .iter()
        .find(|p| paths.contains(**p))
        .map(|p| p.to_string())
        .or_else(|| {
            tree.iter()
                .map(|e| e.path.clone())
                .find(|p| p.ends_with(".onnx"))
        })
        .context("no .onnx model file found in TTS repo")?;
    download_required(repo, dir, &onnx)?;

    // Voice style vectors (each ~0.5 MB; the voice picker needs them all).
    let voices: Vec<String> = tree
        .iter()
        .filter(|e| e.path.starts_with("voices/") && e.path.ends_with(".bin"))
        .map(|e| e.path.clone())
        .collect();
    if voices.is_empty() {
        bail!("TTS repo has no voices/*.bin style vectors");
    }
    for voice in voices {
        download_required(repo, dir, &voice)?;
    }
    Ok(())
}

fn pull_gguf(repo: &str, dir: &PathBuf, tree: &[TreeEntry]) -> Result<()> {
    let Some(file) = pick_gguf_file(tree) else {
        if tree.iter().any(|e| is_split_shard(&e.path)) {
            bail!(
                "this repo only publishes split GGUFs (…-00001-of-0000N shards); \
                 multi-file GGUF loading is not supported yet — pick a repo with \
                 single-file quants (e.g. bartowski/lmstudio-community mirrors)"
            );
        }
        bail!("no .gguf file found");
    };
    let target = dir.join("model.gguf");
    download_required_as(repo, &target, &format!("{repo}/resolve/main/{file}"))?;
    // SPM-vocab quants need the original tokenizer.json next to the file.
    let slug = dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if let Some((_, tok_repo)) =
        GGUF_TOKENIZER_COMPANIONS.iter().find(|(s, _)| *s == slug)
    {
        download_required_as(
            tok_repo,
            &dir.join("tokenizer.json"),
            &format!("{tok_repo}/resolve/main/tokenizer.json"),
        )?;
    }
    Ok(())
}

fn pick_safetensors_file(tree: &[TreeEntry]) -> Option<String> {
    // Prefer the canonical single file, then any sharded-style file.
    for file in tree.iter().map(|e| &e.path) {
        if file == "model.safetensors" {
            return Some(file.clone());
        }
    }
    for file in tree.iter().map(|e| &e.path) {
        if file.ends_with(".safetensors") && !file.contains(".") {
            return Some(file.clone());
        }
    }
    tree.iter()
        .map(|e| &e.path)
        .find(|p| p.ends_with(".safetensors"))
        .cloned()
}

/// llama.cpp shard naming: `<name>-00001-of-00002.gguf`. Downloading one
/// shard yields a model missing most of its layers, so shards are excluded
/// from single-file selection.
fn is_split_shard(path: &str) -> bool {
    let Some(stem) = path.strip_suffix(".gguf") else {
        return false;
    };
    let mut parts = stem.rsplitn(3, '-');
    let (Some(count), Some(of), Some(rest)) = (parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    let is_index = |s: &str| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit());
    of == "of"
        && is_index(count)
        && rest.rsplit('-').next().is_some_and(is_index)
}

fn pick_gguf_file(tree: &[TreeEntry]) -> Option<String> {
    let mut files: Vec<String> = tree
        .iter()
        .filter(|e| e.path.ends_with(".gguf") && !is_split_shard(&e.path))
        .map(|e| e.path.clone())
        .collect();
    if files.is_empty() {
        return None;
    }
    // Prefer sensible quant ordering; Q4_K_M is a good default. Repos name
    // quants in either case (TheBloke upper, HuggingFaceTB lower).
    let prefs = ["Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M", "Q5_K_S", "Q8_0"];
    for pref in prefs {
        if let Some(f) = files
            .iter()
            .find(|f| f.to_ascii_uppercase().contains(pref))
        {
            return Some(f.clone());
        }
    }
    files.sort();
    files.into_iter().next()
}

fn pull_diffusion(repo: &str, dir: &PathBuf, tree: &[TreeEntry]) -> Result<()> {
    let paths: HashSet<&str> = tree.iter().map(|e| e.path.as_str()).collect();

    // Model metadata used for architecture detection.
    download_required(repo, dir, "model_index.json")?;

    // Component dirs we currently support.
    let components = ["unet", "vae", "text_encoder"];
    for comp in components {
        if !paths.iter().any(|p| p.starts_with(&format!("{comp}/"))) {
            bail!("diffusion checkpoint missing required component: {comp}/");
        }
        download_required(repo, dir, &format!("{comp}/config.json"))?;

        let (candidates, canonical) = match comp {
            "unet" | "vae" => (
                vec![
                    format!("{comp}/diffusion_pytorch_model.fp16.safetensors"),
                    format!("{comp}/diffusion_pytorch_model.safetensors"),
                ],
                format!("{comp}/diffusion_pytorch_model.safetensors"),
            ),
            "text_encoder" => (
                vec![
                    format!("{comp}/model.fp16.safetensors"),
                    format!("{comp}/model.safetensors"),
                    format!("{comp}/diffusion_pytorch_model.fp16.safetensors"),
                    format!("{comp}/diffusion_pytorch_model.safetensors"),
                ],
                format!("{comp}/model.safetensors"),
            ),
            _ => unreachable!(),
        };
        download_first_as(repo, dir, &candidates, &canonical)?;
    }

    // Tokenizer: prefer in-repo fast tokenizer, then root tokenizer.json,
    // then the CLIP repo fallback, then BPE vocab+merges.
    if paths.contains("tokenizer/tokenizer.json") {
        // Download all small tokenizer files we might need.
        for file in ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"] {
            download_optional(repo, dir, file)?;
        }
    } else if paths.contains("tokenizer.json") {
        download_required(repo, dir, "tokenizer.json")?;
    } else {
        let tok_target = dir.join("tokenizer.json");
        if !tok_target.is_file() {
            let tok_url = "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer.json";
            if !try_download(tok_url, &tok_target)? {
                // Fall back to old-style BPE files.
                if paths.contains("tokenizer/vocab.json") && paths.contains("tokenizer/merges.txt") {
                    download_required(repo, dir, "tokenizer/vocab.json")?;
                    download_required(repo, dir, "tokenizer/merges.txt")?;
                    download_optional(repo, dir, "tokenizer/tokenizer_config.json")?;
                } else {
                    bail!("diffusion checkpoint has no recognizable tokenizer");
                }
            }
        } else {
            println!("  {} tokenizer.json", style("✓ cached").dim());
        }
    }

    // Scheduler config is optional but useful.
    download_optional(repo, dir, "scheduler/scheduler_config.json")?;

    Ok(())
}

/// Download `source_path` from the repo, erroring if it is missing.
fn download_required(repo: &str, dir: &PathBuf, source_path: &str) -> Result<()> {
    let target = dir.join(source_path);
    if target.is_file() {
        println!("  {} {source_path}", style("✓ cached").dim());
        return Ok(());
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let url = format!("https://huggingface.co/{repo}/resolve/main/{source_path}");
    download(&url, &target).with_context(|| format!("downloading {source_path}"))
}

/// Download from an arbitrary URL and save as `target`.
fn download_required_as(_repo: &str, target: &PathBuf, url_relative: &str) -> Result<()> {
    if target.is_file() {
        println!("  {} {}", style("✓ cached").dim(), target.file_name().unwrap().to_string_lossy());
        return Ok(());
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let url = format!("https://huggingface.co/{url_relative}");
    download(&url, target).with_context(|| format!("downloading {}", target.display()))
}

/// Try each source path in order; save the first hit as `target_relative`.
fn download_first_as(
    repo: &str,
    dir: &PathBuf,
    source_paths: &[String],
    target_relative: &str,
) -> Result<()> {
    let target = dir.join(target_relative);
    if target.is_file() {
        println!("  {} {target_relative}", style("✓ cached").dim());
        return Ok(());
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    for (i, source) in source_paths.iter().enumerate() {
        let url = format!("https://huggingface.co/{repo}/resolve/main/{source}");
        match try_download(&url, &target) {
            Ok(true) => return Ok(()),
            Ok(false) if i + 1 < source_paths.len() => continue,
            Ok(false) => bail!("none of the candidate weight files found: {source_paths:?}"),
            Err(e) => return Err(e).with_context(|| format!("downloading {target_relative}")),
        }
    }
    Ok(())
}

/// Download an optional file; silently skip 404.
fn download_optional(repo: &str, dir: &PathBuf, source_path: &str) -> Result<()> {
    let target = dir.join(source_path);
    if target.is_file() {
        println!("  {} {source_path}", style("✓ cached").dim());
        return Ok(());
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let url = format!("https://huggingface.co/{repo}/resolve/main/{source_path}");
    if try_download(&url, &target)? {
        println!("  {} {source_path} ({:.1} MB)", style("✓").green(), file_mb(&target));
    }
    Ok(())
}

fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGINGFACE_TOKEN"))
        .ok()
}

/// Download `url` to `target`, returning true on success. 404 returns false;
/// other HTTP errors return Err.
fn try_download(url: &str, target: &PathBuf) -> Result<bool> {
    let mut req = ureq::get(url);
    if let Some(token) = hf_token() {
        req = req.header("Authorization", &format!("Bearer {token}"));
    }
    match req.call() {
        Ok(resp) => {
            let part = target.with_extension("part");
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
            Ok(true)
        }
        Err(ureq::Error::StatusCode(404)) => Ok(false),
        Err(e) => Err(e).context("request failed"),
    }
}

/// Download `url` to `target`, erroring on any failure.
fn download(url: &str, target: &PathBuf) -> Result<()> {
    if !try_download(url, target)? {
        bail!("HTTP 404: {url}");
    }
    Ok(())
}

fn file_mb(path: &PathBuf) -> f64 {
    fs::metadata(path).map(|m| m.len() as f64 / 1e6).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(paths: &[&str]) -> Vec<TreeEntry> {
        paths
            .iter()
            .map(|p| TreeEntry { path: p.to_string(), r#type: "file".to_string() })
            .collect()
    }

    #[test]
    fn split_shards_are_recognized() {
        assert!(is_split_shard("qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf"));
        assert!(is_split_shard("Model-Q4_K_M-00002-of-00003.gguf"));
        assert!(!is_split_shard("model.gguf"));
        assert!(!is_split_shard("Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"));
        assert!(!is_split_shard("best-of-model.gguf"));
        assert!(!is_split_shard("model-00001-of-00002.safetensors"));
    }

    #[test]
    fn gguf_pick_skips_shards() {
        // Single-file quant preferred even when shards list first.
        let t = tree(&[
            "m-q4_k_m-00001-of-00002.gguf",
            "m-q4_k_m-00002-of-00002.gguf",
            "m-Q8_0.gguf",
        ]);
        assert_eq!(pick_gguf_file(&t), Some("m-Q8_0.gguf".to_string()));

        // Only shards available -> no pick (pull_gguf reports why).
        let t = tree(&["m-q4_k_m-00001-of-00002.gguf", "m-q4_k_m-00002-of-00002.gguf"]);
        assert_eq!(pick_gguf_file(&t), None);
    }
}
