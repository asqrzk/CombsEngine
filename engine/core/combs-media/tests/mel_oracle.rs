//! Harmony test: the log-mel pipeline vs the transformers reference.
//!
//! `fixtures/mel_oracle.bin` holds a deterministic multi-tone clip and the
//! mel computed by `transformers.audio_utils` with Whisper's exact
//! parameters (periodic Hann 400, hop 160, centered reflect-padded power
//! spectrogram with the last frame dropped, 80 slaney/slaney mel filters,
//! log10 at 1e-10, max−8 floor, (x+4)/4). Layout: three little-endian u32
//! counts (samples, mels, frames), then the f32 samples, then the f32 mel
//! row-major `[mels, frames]`.

use combs_media::{LogMel, N_MELS};

#[test]
fn log_mel_matches_transformers_reference() {
    let bytes = include_bytes!("fixtures/mel_oracle.bin");
    let u32_at = |o: usize| {
        u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize
    };
    let (n_samples, n_mels, n_frames) = (u32_at(0), u32_at(4), u32_at(8));
    assert_eq!(n_mels, N_MELS);

    let f32_at = |o: usize| {
        f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
    };
    let base = 12;
    let samples: Vec<f32> = (0..n_samples).map(|i| f32_at(base + 4 * i)).collect();
    let mel_base = base + 4 * n_samples;
    let expect: Vec<f32> = (0..n_mels * n_frames)
        .map(|i| f32_at(mel_base + 4 * i))
        .collect();

    let (got, frames) = LogMel::new().compute(&samples);
    assert_eq!(frames, n_frames, "frame count disagrees with the reference");

    let mut worst = 0.0f32;
    for (i, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
        let d = (g - e).abs();
        if d > worst {
            worst = d;
        }
        assert!(
            d <= 1e-4,
            "mel[{}][{}]: got {g}, reference {e} (|diff| {d})",
            i / n_frames,
            i % n_frames,
        );
    }
    eprintln!("mel oracle worst |diff|: {worst:e}");
}
