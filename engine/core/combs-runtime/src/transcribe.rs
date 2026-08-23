//! Greedy speech-to-text transcription over a [`SpeechToTextModel`].
//!
//! The forced decoder prefix (`<|startoftranscript|>`, language,
//! `<|transcribe|>`, `<|notimestamps|>`) is resolved from the tokenizer by
//! token *string* — ids differ between Whisper sizes and multilingual/en
//! exports, so hardcoding them would silently corrupt transcripts.
//! Suppress lists come from the checkpoint's `generation_config.json`.
//! Audio is processed in sequential 30 s windows (no overlap in v1).

use std::path::Path;

use burn::tensor::{Tensor, TensorData};
use combs_core::{CombsBackend, init_device};
use combs_formats::{ModelSource, SafetensorsSource};
use combs_media::{CHUNK_SAMPLES, LogMel, SAMPLE_RATE, decode_wav, pad_or_trim, resample_linear};
use combs_models::load_speech_model;
use combs_models::SpeechToTextModel;
use tokenizers::Tokenizer;

use crate::{EngineError, Result};

/// A loaded speech model + tokenizer, reusable across requests.
pub struct SpeechEngine {
    model: Box<dyn SpeechToTextModel<CombsBackend>>,
    tokenizer: Tokenizer,
    mel: LogMel,
    device: burn::tensor::Device<CombsBackend>,
    forced_prefix: Vec<u32>,
    eot: u32,
    suppress: Vec<u32>,
    begin_suppress: Vec<u32>,
    max_ctx: usize,
}

impl SpeechEngine {
    /// Loads a Whisper-style safetensors checkpoint directory.
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
        let forced_prefix = vec![
            id("<|startoftranscript|>")?,
            id(&format!("<|{language}|>"))?,
            id("<|transcribe|>")?,
            id("<|notimestamps|>")?,
        ];
        let eot = id("<|endoftext|>")?;

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
            eot,
            suppress,
            begin_suppress,
            max_ctx,
        })
    }

    /// Transcribes a WAV payload (16-bit PCM mono/stereo, any rate).
    pub fn transcribe_wav(&self, wav: &[u8]) -> Result<String> {
        let (samples, rate) = decode_wav(wav).map_err(|e| EngineError::Media(e.to_string()))?;
        // (decode_wav validates the RIFF layout; anything else is media junk)
        let samples = resample_linear(&samples, rate, SAMPLE_RATE as u32);
        if samples.is_empty() {
            return Ok(String::new());
        }

        let mut text = String::new();
        for window in samples.chunks(CHUNK_SAMPLES) {
            let piece = self.transcribe_window(window)?;
            if !text.is_empty() && !piece.is_empty() {
                text.push(' ');
            }
            text.push_str(piece.trim());
        }
        Ok(text)
    }

    /// Greedy decode of one ≤30 s window.
    fn transcribe_window(&self, samples: &[f32]) -> Result<String> {
        let padded = pad_or_trim(samples, CHUNK_SAMPLES);
        let (mel, frames) = self.mel.compute(&padded);
        let n_mels = self.model.n_mels();
        let mel: Tensor<CombsBackend, 3> = Tensor::from_data(
            TensorData::new(mel, [1, n_mels, frames]),
            &self.device,
        );
        let encoded = self.model.encode_audio(mel)?;

        let mut tokens = self.forced_prefix.clone();
        let mut generated: Vec<u32> = Vec::new();
        let budget = self.max_ctx.saturating_sub(tokens.len() + 1);
        for step in 0..budget {
            let logits = self.model.decode_step(&tokens, &encoded)?;
            let mut row: Vec<f32> = logits
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|e| EngineError::Readback(format!("asr logits: {e:?}")))?;
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
            let next = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(self.eot);
            if next == self.eot {
                break;
            }
            tokens.push(next);
            generated.push(next);
        }

        self.tokenizer
            .decode(&generated, true)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))
    }
}
