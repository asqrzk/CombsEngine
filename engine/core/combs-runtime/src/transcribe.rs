//! Greedy speech-to-text transcription over a [`SpeechToTextModel`].
//!
//! The forced decoder prefix (`<|startoftranscript|>`, language,
//! `<|transcribe|>`, `<|notimestamps|>`) is resolved from the tokenizer by
//! token *string* — ids differ between Whisper sizes and multilingual/en
//! exports, so hardcoding them would silently corrupt transcripts.
//! Suppress lists come from the checkpoint's `generation_config.json`.
//! Audio is processed in sequential 30 s windows (no overlap in v1).
//!
//! # The ears have gates
//!
//! Whisper is trained to always say something: at the first step the
//! end-of-text and the blank are suppressed and `<|nocaptions|>` is
//! never allowed, so a window of near-silence is FORCED to speak — and
//! what it says is whatever the decoder's prior prefers: Korean
//! byte-merges, a lone "I", a stray `<|27.18|>` whose timestamp id the
//! tokenizer does not mark special. One mask and three gates end that:
//!
//! - **the control mask** — after the forced prefix every control id
//!   (the languages, the tasks, `<|nocaptions|>`, `<|notimestamps|>`,
//!   every timestamp) is suppressed at every step and stripped again
//!   before decoding, belt and braces;
//! - **silence** — a window whose RMS is under the floor never reaches
//!   the decoder;
//! - **no-speech** — the probability of `<|nocaptions|>` read at the
//!   start-of-transcript position before any suppression (whisper's own
//!   definition) refuses a window above the gate;
//! - **confidence** — a window whose mean token log-probability falls
//!   under the floor says nothing: a decoder forced to speak over
//!   nothing speaks badly, and badly is where the Korean came from.
//!
//! A gated window contributes no words; the [`Transcript`] carries the
//! counts and the means so a caller can see what was refused and why.

use std::path::Path;

use burn::tensor::{Tensor, TensorData};
use combs_core::{CombsBackend, init_device};
use combs_formats::{ModelSource, SafetensorsSource};
use combs_media::{CHUNK_SAMPLES, LogMel, SAMPLE_RATE, decode_wav, pad_or_trim, resample_linear};
use combs_models::SpeechToTextModel;
use combs_models::load_speech_model;
use tokenizers::Tokenizer;

use crate::{EngineError, Result};

/// What the ears refuse to hear as words. Every field is a door; the
/// defaults are whisper's own where whisper has one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TranscribeGates {
    /// A window whose no-speech probability is above this says nothing
    /// (whisper's `no_speech_threshold`, 0.6).
    pub no_speech: f32,
    /// A window whose mean token log-probability is under this says
    /// nothing (whisper's `logprob_threshold`, -1.0).
    pub min_avg_logprob: f32,
    /// A window whose RMS amplitude is under this skips the decoder.
    /// 1e-4 is -80 dBFS: digital silence and little else.
    pub silence_rms: f32,
}

impl Default for TranscribeGates {
    fn default() -> Self {
        Self {
            no_speech: 0.6,
            min_avg_logprob: -1.0,
            silence_rms: 1e-4,
        }
    }
}

/// What a transcription heard, beyond its words.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Transcript {
    /// The words of every window that passed the gates, space-joined.
    pub text: String,
    /// Windows of audio the payload held (30 s each, the last shorter).
    pub windows: usize,
    /// Windows that contributed no words: skipped as silence, or
    /// decoded and refused by the no-speech or confidence gate.
    pub silent_windows: usize,
    /// The mean no-speech probability over the windows that reached the
    /// decoder (0 when none did).
    pub no_speech_prob: f32,
    /// The mean token log-probability over the windows that reached the
    /// decoder (0 when none did).
    pub avg_logprob: f32,
}

/// One window's decode, before the gates rule on it.
struct WindowHeard {
    text: String,
    no_speech_prob: f32,
    avg_logprob: f32,
}

/// A loaded speech model + tokenizer, reusable across requests.
pub struct SpeechEngine {
    model: Box<dyn SpeechToTextModel<CombsBackend>>,
    tokenizer: Tokenizer,
    mel: LogMel,
    device: burn::tensor::Device<CombsBackend>,
    forced_prefix: Vec<u32>,
    sot: u32,
    eot: u32,
    /// `<|nocaptions|>` (older exports) or `<|nospeech|>` (newer); a
    /// tokenizer without either reads a no-speech probability of zero.
    no_speech: Option<u32>,
    /// Every id at or above this is a control token — languages, tasks,
    /// no-speech, no-timestamps and the timestamps — never a word.
    control_floor: u32,
    suppress: Vec<u32>,
    begin_suppress: Vec<u32>,
    max_ctx: usize,
    gates: TranscribeGates,
}

impl SpeechEngine {
    /// Loads a Whisper-style safetensors checkpoint directory with the
    /// default gates.
    pub fn load(dir: &Path, language: &str) -> Result<Self> {
        let source = SafetensorsSource::load(dir)?;
        let device = init_device();
        let model = load_speech_model::<CombsBackend>(&source, &device)?;
        let spec = source.tokenizer()?;
        let tokenizer = Tokenizer::from_bytes(spec.json_bytes()?)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))?;

        let id = |s: &str| -> Result<u32> {
            tokenizer.token_to_id(s).ok_or_else(|| {
                EngineError::Tokenizer(format!("tokenizer has no {s} token"))
            })
        };
        let sot = id("<|startoftranscript|>")?;
        let forced_prefix = vec![
            sot,
            id(&format!("<|{language}|>"))?,
            id("<|transcribe|>")?,
            id("<|notimestamps|>")?,
        ];
        let eot = id("<|endoftext|>")?;
        let no_speech = tokenizer
            .token_to_id("<|nocaptions|>")
            .or_else(|| tokenizer.token_to_id("<|nospeech|>"));
        // Whisper lays its vocabulary out as words, then end-of-text,
        // then every control token from start-of-transcript up; when an
        // export breaks that order the timestamps alone are masked.
        let control_floor = if eot < sot {
            sot
        } else {
            tokenizer.token_to_id("<|0.00|>").unwrap_or(u32::MAX)
        };

        // Suppress lists ship in generation_config.json; absent lists mean
        // "suppress nothing" rather than an error.
        let (suppress, begin_suppress) = {
            let gc: Option<serde_json::Value> = std::fs::read_to_string(
                dir.join("generation_config.json"),
            )
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());
            let ids = |key: &str| -> Vec<u32> {
                gc.as_ref()
                    .and_then(|v| v.get(key))
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_u64().map(|x| x as u32))
                            .collect()
                    })
                    .unwrap_or_default()
            };
            (ids("suppress_tokens"), ids("begin_suppress_tokens"))
        };

        let max_ctx = model.metadata().max_position_embeddings;
        Ok(SpeechEngine {
            model,
            tokenizer,
            mel: LogMel::new(),
            device,
            forced_prefix,
            sot,
            eot,
            no_speech,
            control_floor,
            suppress,
            begin_suppress,
            max_ctx,
            gates: TranscribeGates::default(),
        })
    }

    /// The same engine with other gates.
    pub fn with_gates(mut self, gates: TranscribeGates) -> Self {
        self.gates = gates;
        self
    }

    /// The gates this engine rules by.
    pub fn gates(&self) -> TranscribeGates {
        self.gates
    }

    /// Transcribes a WAV payload (16-bit PCM mono/stereo, any rate) to
    /// its words alone.
    pub fn transcribe_wav(&self, wav: &[u8]) -> Result<String> {
        Ok(self.transcribe(wav)?.text)
    }

    /// Transcribes a WAV payload and says what the gates refused.
    pub fn transcribe(&self, wav: &[u8]) -> Result<Transcript> {
        let (samples, rate) = decode_wav(wav).map_err(|e| EngineError::Media(e.to_string()))?;
        // (decode_wav validates the RIFF layout; anything else is media junk)
        let samples = resample_linear(&samples, rate, SAMPLE_RATE as u32);
        if samples.is_empty() {
            return Ok(Transcript::default());
        }

        let mut out = Transcript::default();
        let mut no_speech_sum = 0.0f32;
        let mut logprob_sum = 0.0f32;
        let mut decoded = 0usize;
        for window in samples.chunks(CHUNK_SAMPLES) {
            out.windows += 1;
            if rms(window) < self.gates.silence_rms {
                out.silent_windows += 1;
                continue;
            }
            let heard = self.transcribe_window(window)?;
            no_speech_sum += heard.no_speech_prob;
            logprob_sum += heard.avg_logprob;
            decoded += 1;
            if gate_window(&self.gates, heard.no_speech_prob, heard.avg_logprob).is_some() {
                out.silent_windows += 1;
                continue;
            }
            let piece = heard.text.trim();
            if piece.is_empty() {
                out.silent_windows += 1;
                continue;
            }
            if !out.text.is_empty() {
                out.text.push(' ');
            }
            out.text.push_str(piece);
        }
        if decoded > 0 {
            out.no_speech_prob = no_speech_sum / decoded as f32;
            out.avg_logprob = logprob_sum / decoded as f32;
        }
        Ok(out)
    }

    /// The final position's logits for a token prefix, on the host.
    fn logits_row(&self, tokens: &[u32], encoded: &Tensor<CombsBackend, 3>) -> Result<Vec<f32>> {
        self.model
            .decode_step(tokens, encoded)?
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|e| EngineError::Readback(format!("asr logits: {e:?}")))
    }

    /// Greedy decode of one ≤30 s window.
    fn transcribe_window(&self, samples: &[f32]) -> Result<WindowHeard> {
        let padded = pad_or_trim(samples, CHUNK_SAMPLES);
        let (mel, frames) = self.mel.compute(&padded);
        let n_mels = self.model.n_mels();
        let mel: Tensor<CombsBackend, 3> = Tensor::from_data(
            TensorData::new(mel, [1, n_mels, frames]),
            &self.device,
        );
        let encoded = self.model.encode_audio(mel)?;

        // whisper's no-speech read: the probability of the no-speech
        // token right after start-of-transcript, on the raw logits
        let no_speech_prob = match self.no_speech {
            Some(ns) => softmax_at(&self.logits_row(&[self.sot], &encoded)?, ns as usize),
            None => 0.0,
        };

        let mut tokens = self.forced_prefix.clone();
        let mut generated: Vec<u32> = Vec::new();
        let mut logprob_sum = 0.0f32;
        let budget = self.max_ctx.saturating_sub(tokens.len() + 1);
        for step in 0..budget {
            let mut row = self.logits_row(&tokens, &encoded)?;
            for &id in &self.suppress {
                if (id as usize) < row.len() {
                    row[id as usize] = f32::NEG_INFINITY;
                }
            }
            if step == 0 {
                for &id in &self.begin_suppress {
                    if (id as usize) < row.len() {
                        row[id as usize] = f32::NEG_INFINITY;
                    }
                }
            }
            // the control mask: after the prefix, no language, task,
            // no-speech, no-timestamps or timestamp id is ever a word
            let floor = (self.control_floor as usize).min(row.len());
            for v in &mut row[floor..] {
                *v = f32::NEG_INFINITY;
            }
            let next = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(self.eot);
            logprob_sum += log_softmax_at(&row, next as usize);
            if next == self.eot {
                break;
            }
            tokens.push(next);
            generated.push(next);
        }
        // whisper's mean: the generated tokens and the end-of-text
        let avg_logprob = logprob_sum / (generated.len() as f32 + 1.0);
        let kept = strip_control(&generated, self.control_floor);
        let text = self
            .tokenizer
            .decode(&kept, true)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))?;
        Ok(WindowHeard {
            text,
            no_speech_prob,
            avg_logprob,
        })
    }
}

/// The braces under the belt: any control id that reached the output
/// is dropped before decoding, so a timestamp never prints as `<|27.18|>`.
pub fn strip_control(ids: &[u32], control_floor: u32) -> Vec<u32> {
    ids.iter().copied().filter(|&id| id < control_floor).collect()
}

/// Which gate, if any, refuses a decoded window.
pub fn gate_window(gates: &TranscribeGates, no_speech_prob: f32, avg_logprob: f32) -> Option<&'static str> {
    if no_speech_prob > gates.no_speech {
        return Some("no speech");
    }
    if avg_logprob < gates.min_avg_logprob {
        return Some("low confidence");
    }
    None
}

/// Root-mean-square amplitude of a window; 0 for an empty one.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// The log-sum-exp of a row that may carry `-inf` entries; `-inf` for a
/// row with none finite.
fn log_sum_exp(row: &[f32]) -> f32 {
    let m = row
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return f32::NEG_INFINITY;
    }
    let z: f32 = row
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| (x - m).exp())
        .sum();
    m + z.ln()
}

/// softmax(row)[idx]; 0 for an index out of range or masked out.
fn softmax_at(row: &[f32], idx: usize) -> f32 {
    match row.get(idx) {
        Some(x) if x.is_finite() => (x - log_sum_exp(row)).exp(),
        _ => 0.0,
    }
}

/// log_softmax(row)[idx]; `-inf` for an index out of range or masked out.
fn log_softmax_at(row: &[f32], idx: usize) -> f32 {
    match row.get(idx) {
        Some(x) if x.is_finite() => x - log_sum_exp(row),
        _ => f32::NEG_INFINITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_control_mask_strips_at_and_above_the_floor() {
        assert_eq!(strip_control(&[5, 50257, 50258, 50364, 7], 50258), vec![5, 50257, 7]);
        assert!(strip_control(&[50364, 50400], 50258).is_empty());
        assert_eq!(strip_control(&[1, 2], u32::MAX), vec![1, 2]);
    }

    #[test]
    fn softmax_reads_are_probabilities_and_masks_read_zero() {
        let row = [0.0, 0.0, 0.0];
        assert!((softmax_at(&row, 1) - 1.0 / 3.0).abs() < 1e-6);
        let masked = [1.0, f32::NEG_INFINITY, 1.0];
        assert_eq!(softmax_at(&masked, 1), 0.0);
        assert!((softmax_at(&masked, 0) - 0.5).abs() < 1e-6);
        assert_eq!(softmax_at(&masked, 9), 0.0);
        let sum: f32 = (0..3).map(|i| softmax_at(&[0.3, -1.2, 2.5], i)).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn log_softmax_agrees_with_softmax_and_masks_read_minus_inf() {
        let row = [0.3, -1.2, 2.5, f32::NEG_INFINITY];
        for i in 0..3 {
            assert!((log_softmax_at(&row, i).exp() - softmax_at(&row, i)).abs() < 1e-6);
        }
        assert_eq!(log_softmax_at(&row, 3), f32::NEG_INFINITY);
        assert_eq!(log_sum_exp(&[f32::NEG_INFINITY]), f32::NEG_INFINITY);
    }

    #[test]
    fn the_gates_refuse_by_name_and_let_speech_through() {
        let g = TranscribeGates::default();
        assert_eq!(gate_window(&g, 0.9, -0.4), Some("no speech"));
        assert_eq!(gate_window(&g, 0.1, -1.7), Some("low confidence"));
        assert_eq!(gate_window(&g, 0.1, -0.4), None);
        // the gates are strict: sitting on the line is not past it
        assert_eq!(gate_window(&g, 0.6, -1.0), None);
        let shut = TranscribeGates {
            no_speech: 0.0,
            ..TranscribeGates::default()
        };
        assert_eq!(gate_window(&shut, 0.01, -0.1), Some("no speech"));
    }

    #[test]
    fn rms_of_silence_is_zero_and_of_a_square_wave_one() {
        assert_eq!(rms(&[]), 0.0);
        assert_eq!(rms(&[0.0; 160]), 0.0);
        let square: Vec<f32> = (0..160).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        assert!((rms(&square) - 1.0).abs() < 1e-6);
        assert!(rms(&[0.00005; 16]) < TranscribeGates::default().silence_rms);
    }
}
