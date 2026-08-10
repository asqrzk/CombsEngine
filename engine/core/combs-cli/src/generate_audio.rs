//! `combs generate-audio` — text-to-speech with a local Kokoro ONNX
//! checkpoint (see `combs pull kokoro-82m`).
//!
//! Pipeline: text → IPA phonemes (espeak-ng subprocess) → char-level token
//! ids (tokenizer.json vocab) → ONNX inference (input_ids + per-voice style
//! vector + speed) → 24 kHz mono 16-bit PCM WAV.
//!
//! Voices are the "tones": each `voices/<name>.bin` is a 510×256 f32 style
//! table (row = phoneme count) — e.g. `af_heart` (warm US female),
//! `af_nicole` (whispering), `bm_george` (British male). Long inputs are
//! split into sentences and concatenated, which also respects the model's
//! 510-token context.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use anyhow::{bail, Context, Result};
use clap::Args;

const SAMPLE_RATE: u32 = 24_000;
const STYLE_DIM: usize = 256;
const STYLE_ROWS: usize = 510;
/// Max phoneme tokens per model call (context 512 minus the two `$` pads).
const MAX_PHONEME_TOKENS: usize = 510;

#[derive(Args, Clone)]
pub struct GenerateAudioArgs {
    /// Model directory (Kokoro ONNX layout: *.onnx + voices/*.bin +
    /// tokenizer.json), or a preset id like `kokoro-82m`.
    #[arg(long)]
    pub model: PathBuf,
    /// Text to speak (not needed with --list-voices).
    #[arg(long, default_value = "")]
    pub text: String,
    /// Voice / tone (a `voices/*.bin` name, e.g. af_heart, af_nicole
    /// for a whisper, bm_george). Default: af_heart.
    #[arg(long, default_value = "af_heart")]
    pub voice: String,
    /// Speaking speed multiplier (1.0 = normal).
    #[arg(long, default_value_t = 1.0)]
    pub speed: f32,
    /// espeak-ng language (default: derived from the voice prefix —
    /// `b?_*` → en-gb, otherwise en-us).
    #[arg(long)]
    pub lang: Option<String>,
    /// Output WAV path.
    #[arg(long, default_value = "output.wav")]
    pub output: PathBuf,
    /// List the available voices in the model directory and exit.
    #[arg(long)]
    pub list_voices: bool,
}

pub fn cmd_generate_audio(args: GenerateAudioArgs) -> Result<()> {
    let model_dir = super::resolve_model_arg(&args.model)?;

    if args.list_voices {
        for v in list_voices(&model_dir)? {
            println!("{v}");
        }
        return Ok(());
    }

    eprintln!(
        "generating speech (voice: {}, speed: {}): {}",
        args.voice,
        args.speed,
        truncate(&args.text, 80),
    );

    let mut engine = TtsEngine::load(&model_dir)?;
    let waveform =
        engine.synthesize(&args.text, &args.voice, args.speed, args.lang.as_deref())?;
    write_wav(&args.output, &waveform)?;
    let secs = waveform.len() as f32 / SAMPLE_RATE as f32;
    eprintln!("wrote {} ({secs:.1}s)", args.output.display());
    Ok(())
}

/// The load-once TTS engine: ONNX session, phoneme vocab, and per-voice
/// style tables (cached on first use). `combs serve-audio` keeps one of
/// these resident — the per-request cold start (session build + style
/// load) only ever happens once per process.
pub struct TtsEngine {
    session: ort::session::Session,
    vocab: HashMap<char, i64>,
    model_dir: PathBuf,
    styles: HashMap<String, Vec<f32>>,
}

impl TtsEngine {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let onnx = find_onnx(model_dir)?;
        let vocab = load_vocab(model_dir)?;
        // ort's error type carries non-Send state, so it can't ride `?`
        // into anyhow — flatten every ort error to a string via `ort_err`.
        let session = ort::session::Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(ort_err)?
            .commit_from_file(&onnx)
            .map_err(ort_err)
            .with_context(|| format!("loading ONNX model {}", onnx.display()))?;
        Ok(Self {
            session,
            vocab,
            model_dir: model_dir.to_path_buf(),
            styles: HashMap::new(),
        })
    }

    /// Available voice names (`voices/*.bin` stems).
    pub fn voices(&self) -> Result<Vec<String>> {
        list_voices(&self.model_dir)
    }

    /// Synthesizes 24 kHz mono f32 samples. `lang: None` derives the
    /// espeak language from the voice prefix (`b?_*` → en-gb).
    pub fn synthesize(
        &mut self,
        text: &str,
        voice: &str,
        speed: f32,
        lang: Option<&str>,
    ) -> Result<Vec<f32>> {
        if text.trim().is_empty() {
            bail!("text is empty");
        }
        if speed <= 0.0 {
            bail!("speed must be positive");
        }
        if !self.styles.contains_key(voice) {
            let style = load_style(&self.model_dir, voice)?;
            self.styles.insert(voice.to_string(), style);
        }
        let style = self.styles.get(voice).expect("just inserted").clone();
        let lang = lang.map(str::to_string).unwrap_or_else(|| {
            if voice.starts_with('b') { "en-gb".into() } else { "en-us".into() }
        });

        let mut waveform: Vec<f32> = Vec::new();
        for sentence in split_sentences(text) {
            let phonemes = phonemize(&sentence, &lang)?;
            if phonemes.is_empty() {
                continue;
            }
            let ids = tokenize(&phonemes, &self.vocab);
            if ids.len() <= 2 {
                continue;
            }
            let chunk = run_inference(&mut self.session, &ids, &style, speed)
                .with_context(|| format!("synthesizing: {}", truncate(&sentence, 60)))?;
            waveform.extend_from_slice(&chunk);
        }
        if waveform.is_empty() {
            bail!("no audio produced (text phonemized to nothing)");
        }
        Ok(waveform)
    }
}

// ── Model discovery ─────────────────────────────────────────────────

fn find_onnx(dir: &Path) -> Result<PathBuf> {
    let prefs = [
        "onnx/model_quantized.onnx",
        "onnx/model.onnx",
        "model_quantized.onnx",
        "model.onnx",
    ];
    for p in prefs {
        let path = dir.join(p);
        if path.is_file() {
            return Ok(path);
        }
    }
    // Fallback: first .onnx anywhere in dir or dir/onnx.
    for sub in [dir.to_path_buf(), dir.join("onnx")] {
        if let Ok(entries) = std::fs::read_dir(&sub) {
            for entry in entries.flatten() {
                if entry.path().extension().is_some_and(|e| e == "onnx") {
                    return Ok(entry.path());
                }
            }
        }
    }
    bail!("no .onnx model found in {}", dir.display())
}

pub fn list_voices(dir: &Path) -> Result<Vec<String>> {
    let mut voices = Vec::new();
    for entry in std::fs::read_dir(dir.join("voices"))
        .with_context(|| format!("no voices/ dir in {}", dir.display()))?
        .flatten()
    {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "bin") {
            if let Some(stem) = path.file_stem() {
                voices.push(stem.to_string_lossy().into_owned());
            }
        }
    }
    voices.sort();
    Ok(voices)
}

/// Loads the char-level phoneme vocab from tokenizer.json.
fn load_vocab(dir: &Path) -> Result<HashMap<char, i64>> {
    let raw = std::fs::read_to_string(dir.join("tokenizer.json"))
        .with_context(|| format!("reading {}/tokenizer.json", dir.display()))?;
    let json: serde_json::Value = serde_json::from_str(&raw)?;
    let vocab = json
        .pointer("/model/vocab")
        .and_then(|v| v.as_object())
        .context("tokenizer.json missing model.vocab")?;
    let mut map = HashMap::new();
    for (token, id) in vocab {
        let mut chars = token.chars();
        if let (Some(c), None) = (chars.next(), chars.next()) {
            if let Some(id) = id.as_i64() {
                map.insert(c, id);
            }
        }
    }
    if map.is_empty() {
        bail!("tokenizer.json vocab is empty or not char-level");
    }
    Ok(map)
}

/// Loads the requested voice's 510×256 style table.
fn load_style(dir: &Path, voice: &str) -> Result<Vec<f32>> {
    let path = dir.join("voices").join(format!("{voice}.bin"));
    let bytes = std::fs::read(&path).with_context(|| {
        let available = list_voices(dir)
            .map(|v| v.join(", "))
            .unwrap_or_else(|_| "(none)".into());
        format!("unknown voice '{voice}' — available: {available}")
    })?;
    let expected = STYLE_ROWS * STYLE_DIM * 4;
    if bytes.len() < expected {
        bail!(
            "style file {} too small: {} bytes (expected {expected})",
            path.display(),
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

// ── Text → phonemes → ids ───────────────────────────────────────────

/// Split into sentence-ish chunks so each stays under the model context
/// and gets its own prosody contour.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?' | '…' | ';' | '\n') && current.trim().len() > 1 {
            out.push(std::mem::take(&mut current).trim().to_string());
        }
    }
    let rest = current.trim();
    if !rest.is_empty() {
        out.push(rest.to_string());
    }
    out.retain(|s| !s.is_empty());
    out
}

/// IPA phonemes for `text` via the espeak-ng CLI, post-processed the way
/// Kokoro's (misaki espeak-fallback) pipeline expects.
fn phonemize(text: &str, lang: &str) -> Result<String> {
    let bin = find_espeak()?;
    let output = ProcessCommand::new(&bin)
        .args(["-q", "--ipa", "-v", lang, text])
        .output()
        .with_context(|| format!("running {bin}"))?;
    if !output.status.success() {
        bail!(
            "espeak-ng failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let ipa = String::from_utf8_lossy(&output.stdout);
    let joined = ipa
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    // Kokoro-style cleanup: drop espeak language-switch markers and ties,
    // then apply the standard substitutions (kokoro-js phonemize parity).
    let mut cleaned = String::with_capacity(joined.len());
    let mut in_lang_marker = false;
    for c in joined.chars() {
        match c {
            '(' => in_lang_marker = true,
            ')' => in_lang_marker = false,
            _ if in_lang_marker => {}
            '\u{0361}' | '\u{203F}' => {} // tie bars
            'ʲ' => cleaned.push('j'),
            'r' => cleaned.push('ɹ'),
            'x' => cleaned.push('k'),
            'ɬ' => cleaned.push('l'),
            _ => cleaned.push(c),
        }
    }
    Ok(cleaned.split_whitespace().collect::<Vec<_>>().join(" "))
}

fn find_espeak() -> Result<String> {
    if let Ok(bin) = std::env::var("COMBS_ESPEAK_NG") {
        return Ok(bin);
    }
    let candidates = [
        "espeak-ng",
        "/opt/homebrew/bin/espeak-ng",
        "/usr/local/bin/espeak-ng",
        "/usr/bin/espeak-ng",
    ];
    for bin in candidates {
        if ProcessCommand::new(bin)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
        {
            return Ok(bin.to_string());
        }
    }
    bail!(
        "espeak-ng not found — install it (macOS: `brew install espeak-ng`, \
         Debian: `apt install espeak-ng`) or set COMBS_ESPEAK_NG"
    )
}

/// Char-level encoding with `$` (id 0) boundary pads, skipping any char
/// not in the vocab (espeak occasionally emits marks Kokoro never saw).
fn tokenize(phonemes: &str, vocab: &HashMap<char, i64>) -> Vec<i64> {
    let mut ids = Vec::with_capacity(phonemes.len() + 2);
    ids.push(0);
    for c in phonemes.chars() {
        if let Some(&id) = vocab.get(&c) {
            ids.push(id);
        }
    }
    ids.truncate(MAX_PHONEME_TOKENS + 1);
    ids.push(0);
    ids
}

// ── Inference + WAV ─────────────────────────────────────────────────

fn run_inference(
    session: &mut ort::session::Session,
    ids: &[i64],
    style_table: &[f32],
    speed: f32,
) -> Result<Vec<f32>> {
    // Style row is indexed by the unpadded phoneme count (kokoro-js parity).
    let row = (ids.len().saturating_sub(2)).min(STYLE_ROWS - 1);
    let style: Vec<f32> = style_table[row * STYLE_DIM..(row + 1) * STYLE_DIM].to_vec();

    let ids_value =
        ort::value::Tensor::from_array(([1usize, ids.len()], ids.to_vec()))
            .map_err(ort_err)?;
    let style_value =
        ort::value::Tensor::from_array(([1usize, STYLE_DIM], style)).map_err(ort_err)?;
    let speed_value =
        ort::value::Tensor::from_array(([1usize], vec![speed])).map_err(ort_err)?;

    let outputs = session
        .run(ort::inputs![
            "input_ids" => ids_value,
            "style" => style_value,
            "speed" => speed_value,
        ])
        .map_err(ort_err)?;
    let (_, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
    Ok(data.to_vec())
}

/// Minimal 16-bit PCM mono WAV writer (24 kHz).
fn write_wav(path: &Path, samples: &[f32]) -> Result<()> {
    std::fs::write(path, encode_wav(samples))
        .with_context(|| format!("writing {}", path.display()))
}

/// 16-bit PCM mono WAV bytes (24 kHz) — the in-memory form the audio
/// worker serves directly.
pub fn encode_wav(samples: &[f32]) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        pcm.extend_from_slice(&v.to_le_bytes());
    }
    let data_len = pcm.len() as u32;
    let byte_rate = SAMPLE_RATE * 2;

    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(&pcm);
    wav
}

/// `ort::Error` carries non-Send state and can't convert through `?` into
/// anyhow; stringify it here.
fn ort_err(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("onnxruntime: {e}")
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}
