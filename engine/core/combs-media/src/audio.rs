//! Audio preprocessing for speech-to-text models: WAV decoding, linear
//! resampling to 16 kHz, and Whisper-style log-mel spectrograms.
//!
//! The mel pipeline mirrors the Whisper reference exactly: 400-sample
//! **periodic** Hann window, hop 160, centered STFT with reflect padding
//! and the final frame dropped, power spectrum, 80 **slaney-scale /
//! slaney-normalized** mel filters over 0–8 kHz, `log10` clamped at 1e-10,
//! a global dynamic-range floor of `max − 8`, then `(x + 4) / 4`. Any
//! deviation shifts every downstream encoder activation, so the constants
//! here are contract, not preference.

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

use crate::{MediaError, Result};

/// Sample rate every speech model input is resampled to.
pub const SAMPLE_RATE: usize = 16_000;
/// STFT window length (25 ms at 16 kHz).
pub const N_FFT: usize = 400;
/// STFT hop (10 ms at 16 kHz).
pub const HOP_LENGTH: usize = 160;
/// Mel bins.
pub const N_MELS: usize = 80;
/// Samples in one 30 s model window.
pub const CHUNK_SAMPLES: usize = 30 * SAMPLE_RATE;
/// Frames one 30 s window produces (CHUNK_SAMPLES / HOP_LENGTH).
pub const CHUNK_FRAMES: usize = CHUNK_SAMPLES / HOP_LENGTH;

/// Decodes a WAV payload to mono f32 samples in [-1, 1].
///
/// v1 scope: 16-bit PCM (format tag 1), mono or stereo (stereo is averaged
/// to mono), any sample rate (resample separately). Unknown RIFF chunks
/// are skipped, including the padding byte after odd-sized chunks.
pub fn decode_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let err = |m: &str| MediaError::AudioDecode(m.to_string());
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(err("not a RIFF/WAVE payload"));
    }
    let u16_at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);

    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (tag, channels, rate, bits)
    let mut data: Option<(usize, usize)> = None; // (offset, len)
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32_at(pos + 4) as usize;
        let body = pos + 8;
        if body + size > bytes.len() {
            return Err(err("chunk overruns file"));
        }
        match id {
            b"fmt " => {
                if size < 16 {
                    return Err(err("fmt chunk too short"));
                }
                fmt = Some((
                    u16_at(body),
                    u16_at(body + 2),
                    u32_at(body + 4),
                    u16_at(body + 14),
                ));
            }
            b"data" => {
                data = Some((body, size));
            }
            _ => {}
        }
        // Chunks are word-aligned; odd sizes carry one padding byte.
        pos = body + size + (size & 1);
    }

    let (tag, channels, rate, bits) = fmt.ok_or_else(|| err("missing fmt chunk"))?;
    let (off, len) = data.ok_or_else(|| err("missing data chunk"))?;
    if tag != 1 {
        return Err(err("only PCM (format tag 1) is supported"));
    }
    if bits != 16 {
        return Err(err("only 16-bit samples are supported"));
    }
    if channels == 0 || channels > 2 {
        return Err(err("only mono or stereo is supported"));
    }
    let ch = channels as usize;
    let frame_bytes = 2 * ch;
    let n_frames = len / frame_bytes;
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let base = off + f * frame_bytes;
        let mut acc = 0.0f32;
        for c in 0..ch {
            let s = i16::from_le_bytes([bytes[base + 2 * c], bytes[base + 2 * c + 1]]);
            acc += f32::from(s) / 32768.0;
        }
        out.push(acc / ch as f32);
    }
    Ok((out, rate))
}

/// Linear-interpolation resampler. Documented v1 simplification: no
/// low-pass filter, which is adequate for speech into a 16 kHz pipeline;
/// a windowed-sinc resampler can replace this without changing callers.
pub fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let n_out = ((samples.len() as u64 * to_rate as u64) / from_rate as u64) as usize;
    let step = from_rate as f64 / to_rate as f64;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let pos = i as f64 * step;
        let i0 = pos as usize;
        let frac = (pos - i0 as f64) as f32;
        let a = samples[i0.min(samples.len() - 1)];
        let b = samples[(i0 + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Zero-pads or truncates to exactly `len` samples (the 30 s model window).
pub fn pad_or_trim(samples: &[f32], len: usize) -> Vec<f32> {
    let mut out = samples.to_vec();
    out.resize(len, 0.0);
    out
}

/// Whisper log-mel extractor. Owns the FFT plan, window, and filterbank;
/// build once, reuse per utterance.
pub struct LogMel {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    /// Row-major `[N_MELS, N_FFT/2 + 1]` slaney filterbank.
    filters: Vec<f32>,
}

impl Default for LogMel {
    fn default() -> Self {
        Self::new()
    }
}

impl LogMel {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        // Periodic Hann: divide by N, not N-1 (torch.hann_window default).
        let window: Vec<f32> = (0..N_FFT)
            .map(|i| {
                let x = core::f32::consts::TAU * i as f32 / N_FFT as f32;
                0.5 * (1.0 - x.cos())
            })
            .collect();
        LogMel {
            fft,
            window,
            filters: mel_filterbank(),
        }
    }

    /// Computes the log-mel spectrogram of `samples` (16 kHz mono).
    /// Returns row-major `[N_MELS, n_frames]` with
    /// `n_frames = samples.len() / HOP_LENGTH` (centered STFT, last frame
    /// dropped — the Whisper convention).
    pub fn compute(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let n = samples.len();
        let half = N_FFT / 2;
        // Reflect padding (no edge repeat): [s[half]..s[1]] + s + [s[n-2]..s[n-half-1]]
        let mut padded = Vec::with_capacity(n + N_FFT);
        for i in (1..=half).rev() {
            padded.push(samples[i.min(n.saturating_sub(1))]);
        }
        padded.extend_from_slice(samples);
        for i in 2..=(half + 1) {
            padded.push(samples[n.saturating_sub(i)]);
        }

        let n_frames_full = if padded.len() >= N_FFT {
            1 + (padded.len() - N_FFT) / HOP_LENGTH
        } else {
            0
        };
        // Whisper drops the final STFT frame.
        let n_frames = n_frames_full.saturating_sub(1);
        let n_bins = half + 1;

        // Power spectrum per frame, then mel projection.
        let mut frame = vec![0.0f32; N_FFT];
        let mut spectrum = vec![Complex::new(0.0f32, 0.0f32); n_bins];
        let mut power = vec![0.0f32; n_bins * n_frames];
        let mut scratch = self.fft.make_scratch_vec();
        for f in 0..n_frames {
            let start = f * HOP_LENGTH;
            for i in 0..N_FFT {
                frame[i] = padded[start + i] * self.window[i];
            }
            self.fft
                .process_with_scratch(&mut frame, &mut spectrum, &mut scratch)
                .expect("fft length is fixed");
            for (k, c) in spectrum.iter().enumerate() {
                power[k * n_frames + f] = c.re * c.re + c.im * c.im;
            }
        }

        let mut mel = vec![0.0f32; N_MELS * n_frames];
        for m in 0..N_MELS {
            for k in 0..n_bins {
                let w = self.filters[m * n_bins + k];
                if w != 0.0 {
                    let row = &power[k * n_frames..(k + 1) * n_frames];
                    let out = &mut mel[m * n_frames..(m + 1) * n_frames];
                    for f in 0..n_frames {
                        out[f] += w * row[f];
                    }
                }
            }
        }

        // log10 clamp, global dynamic-range floor, (x + 4) / 4.
        let mut max_val = f32::MIN;
        for v in mel.iter_mut() {
            *v = v.max(1e-10).log10();
            if *v > max_val {
                max_val = *v;
            }
        }
        let floor = max_val - 8.0;
        for v in mel.iter_mut() {
            *v = (v.max(floor) + 4.0) / 4.0;
        }
        (mel, n_frames)
    }
}

/// Slaney mel scale (librosa `htk=False`): linear below 1 kHz, log above.
fn hz_to_mel(f: f32) -> f32 {
    if f < 1000.0 {
        f * 3.0 / 200.0
    } else {
        15.0 + 27.0 * (f / 1000.0).ln() / 6.4f32.ln()
    }
}

fn mel_to_hz(m: f32) -> f32 {
    if m < 15.0 {
        m * 200.0 / 3.0
    } else {
        1000.0 * (6.4f32.ln() * (m - 15.0) / 27.0).exp()
    }
}

/// Builds the `[N_MELS, N_FFT/2 + 1]` slaney-normalized triangular
/// filterbank over 0–8 kHz, matching librosa/transformers for Whisper.
fn mel_filterbank() -> Vec<f32> {
    let n_bins = N_FFT / 2 + 1;
    let fmax = SAMPLE_RATE as f32 / 2.0;
    let mel_max = hz_to_mel(fmax);
    // N_MELS + 2 corner points, uniform in mel space.
    let corners: Vec<f32> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(mel_max * i as f32 / (N_MELS + 1) as f32))
        .collect();
    let mut fb = vec![0.0f32; N_MELS * n_bins];
    for m in 0..N_MELS {
        let (lo, mid, hi) = (corners[m], corners[m + 1], corners[m + 2]);
        let norm = 2.0 / (hi - lo);
        for k in 0..n_bins {
            let f = k as f32 * SAMPLE_RATE as f32 / N_FFT as f32;
            let rising = (f - lo) / (mid - lo);
            let falling = (hi - f) / (hi - mid);
            let w = rising.min(falling).max(0.0);
            fb[m * n_bins + k] = w * norm;
        }
    }
    fb
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal WAV byte stream around raw i16 frames.
    fn wav_bytes(channels: u16, rate: u32, samples: &[i16], extra_chunk: bool) -> Vec<u8> {
        let data_len = samples.len() * 2;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&0u32.to_le_bytes()); // size (unchecked)
        out.extend_from_slice(b"WAVE");
        if extra_chunk {
            out.extend_from_slice(b"LIST");
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(b"abc");
            out.push(0); // word-alignment padding for the odd size
        }
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes()); // PCM
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * u32::from(channels) * 2).to_le_bytes());
        out.extend_from_slice(&(channels * 2).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data_len as u32).to_le_bytes());
        for s in samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    #[test]
    fn wav_mono_roundtrip() {
        let bytes = wav_bytes(1, 16_000, &[0, 16384, -16384, 32767], false);
        let (samples, rate) = decode_wav(&bytes).unwrap();
        assert_eq!(rate, 16_000);
        assert_eq!(samples.len(), 4);
        assert!((samples[0]).abs() < 1e-6);
        assert!((samples[1] - 0.5).abs() < 1e-4);
        assert!((samples[2] + 0.5).abs() < 1e-4);
        assert!(samples[3] > 0.999);
    }

    #[test]
    fn wav_stereo_averages_and_skips_chunks() {
        // Frames: (1000, 3000) -> 2000; (-2000, -4000) -> -3000.
        let bytes = wav_bytes(2, 44_100, &[1000, 3000, -2000, -4000], true);
        let (samples, rate) = decode_wav(&bytes).unwrap();
        assert_eq!(rate, 44_100);
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - 2000.0 / 32768.0).abs() < 1e-6);
        assert!((samples[1] + 3000.0 / 32768.0).abs() < 1e-6);
    }

    #[test]
    fn wav_rejects_non_pcm() {
        let mut bytes = wav_bytes(1, 16_000, &[0, 0], false);
        bytes[20] = 3; // format tag -> IEEE float
        assert!(decode_wav(&bytes).is_err());
    }

    #[test]
    fn resample_identity_and_halving() {
        let s: Vec<f32> = (0..100).map(|i| i as f32).collect();
        assert_eq!(resample_linear(&s, 16_000, 16_000), s);
        let half = resample_linear(&s, 32_000, 16_000);
        assert_eq!(half.len(), 50);
        // A linear ramp stays a linear ramp under linear interpolation.
        assert!((half[10] - 20.0).abs() < 1e-4);
    }

    #[test]
    fn pad_and_trim() {
        let s = vec![1.0f32; 10];
        let padded = pad_or_trim(&s, 16);
        assert_eq!(padded.len(), 16);
        assert_eq!(padded[9], 1.0);
        assert_eq!(padded[10], 0.0);
        assert_eq!(pad_or_trim(&s, 4).len(), 4);
    }

    #[test]
    fn hann_window_is_periodic() {
        let lm = LogMel::new();
        assert!(lm.window[0].abs() < 1e-7);
        assert!((lm.window[N_FFT / 2] - 1.0).abs() < 1e-6);
        for k in 1..N_FFT {
            assert!(
                (lm.window[k] - lm.window[N_FFT - k]).abs() < 1e-6,
                "periodic Hann symmetry broke at {k}"
            );
        }
    }

    #[test]
    fn filterbank_shape_and_coverage() {
        let fb = mel_filterbank();
        let n_bins = N_FFT / 2 + 1;
        assert_eq!(fb.len(), N_MELS * n_bins);
        for m in 0..N_MELS {
            let row = &fb[m * n_bins..(m + 1) * n_bins];
            let sum: f32 = row.iter().sum();
            assert!(sum > 0.0, "mel filter {m} is empty");
            assert!(row.iter().all(|w| *w >= 0.0));
        }
        // Filter peaks must be strictly ordered in frequency.
        let peak = |m: usize| {
            let row = &fb[m * n_bins..(m + 1) * n_bins];
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };
        assert!(peak(0) < peak(N_MELS / 2));
        assert!(peak(N_MELS / 2) < peak(N_MELS - 1));
    }

    #[test]
    fn log_mel_frame_counts() {
        let lm = LogMel::new();
        let clip: Vec<f32> = (0..8000)
            .map(|i| (core::f32::consts::TAU * 440.0 * i as f32 / 16_000.0).sin() * 0.1)
            .collect();
        let (mel, frames) = lm.compute(&clip);
        assert_eq!(frames, 50);
        assert_eq!(mel.len(), N_MELS * 50);
        assert!(mel.iter().all(|v| v.is_finite()));

        let (mel30, frames30) = lm.compute(&pad_or_trim(&clip, CHUNK_SAMPLES));
        assert_eq!(frames30, CHUNK_FRAMES);
        assert_eq!(mel30.len(), N_MELS * CHUNK_FRAMES);
    }

    #[test]
    fn log_mel_range_is_normalized() {
        let lm = LogMel::new();
        let clip: Vec<f32> = (0..16_000)
            .map(|i| (core::f32::consts::TAU * 1000.0 * i as f32 / 16_000.0).sin() * 0.5)
            .collect();
        let (mel, _) = lm.compute(&clip);
        let max = mel.iter().cloned().fold(f32::MIN, f32::max);
        let min = mel.iter().cloned().fold(f32::MAX, f32::min);
        // After (x+4)/4 with a max-8 floor, the span is exactly ≤ 2.
        assert!(max - min <= 2.0 + 1e-5);
        assert!(max < 3.0);
    }
}
