//! The ears' gates on a real whisper checkpoint — silence says nothing,
//! noise says nothing, and speech still speaks without a control token
//! in it. Requires a checkpoint and a GPU, so it is ignored by default:
//!
//! ```text
//! COMBS_TEST_WHISPER=$HOME/.cache/combs/models/whisper-base \
//!   cargo test -p combs-runtime --test transcribe_gates -- --ignored --nocapture
//! ```
//!
//! Set COMBS_TEST_WHISPER_WAV to a 16-bit PCM WAV of spoken English and
//! COMBS_TEST_WHISPER_SAYS to a phrase it carries to run the speech leg;
//! without them that leg is skipped by name. Every number is printed so
//! the ledger can carry it.

use combs_runtime::{SpeechEngine, TranscribeGates};

fn engine() -> Option<SpeechEngine> {
    let dir = std::env::var("COMBS_TEST_WHISPER").ok()?;
    Some(SpeechEngine::load(std::path::Path::new(&dir), "en").expect("load whisper"))
}

/// A 16 kHz mono 16-bit WAV around the given samples.
fn wav(samples: &[f32]) -> Vec<u8> {
    let rate: u32 = 16_000;
    let mut data = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        data.extend_from_slice(&v.to_le_bytes());
    }
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    out
}

/// Deterministic white noise at the given peak (an LCG, no crate).
fn noise(seconds: f32, peak: f32) -> Vec<f32> {
    let n = (16_000.0 * seconds) as usize;
    let mut x: u32 = 0x9E37_79B9;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((x >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * peak
        })
        .collect()
}

#[test]
#[ignore = "requires COMBS_TEST_WHISPER (a whisper checkpoint dir) and a GPU"]
fn silence_and_noise_say_nothing_and_speech_still_speaks() {
    let Some(engine) = engine() else {
        eprintln!("skipping: set COMBS_TEST_WHISPER");
        return;
    };
    println!("gates {:?}", engine.gates());

    // 1. two seconds of digital silence: the RMS gate, no decoder
    let t = engine.transcribe(&wav(&vec![0.0f32; 32_000])).expect("silence");
    println!("[silence] {t:?}");
    assert_eq!(t.text, "", "digital silence was heard as words: {:?}", t.text);
    assert_eq!(t.windows, 1);
    assert_eq!(t.silent_windows, 1);

    // 2. three seconds of white noise at -30 dBFS: the decoder runs,
    //    the no-speech or the confidence gate refuses the window
    let t = engine.transcribe(&wav(&noise(3.0, 0.03))).expect("noise");
    println!("[noise -30dBFS] {t:?}");
    assert_eq!(t.text, "", "noise was heard as words: {:?}", t.text);
    assert_eq!(t.silent_windows, 1);

    // 3. the gates shut: the same noise is FORCED to speak, which is
    //    how the Korean and the fragments came in — and even then no
    //    control token reaches the text
    let open = TranscribeGates {
        no_speech: 1.0,
        min_avg_logprob: f32::NEG_INFINITY,
        silence_rms: 0.0,
    };
    let engine = engine.with_gates(open);
    let t = engine.transcribe(&wav(&noise(3.0, 0.03))).expect("noise, gates open");
    println!("[noise, gates open] {t:?}");
    assert!(!t.text.contains("<|"), "a control token reached the text: {:?}", t.text);
    let engine = engine.with_gates(TranscribeGates::default());

    // 4. speech, when the desk has some
    match (
        std::env::var("COMBS_TEST_WHISPER_WAV"),
        std::env::var("COMBS_TEST_WHISPER_SAYS"),
    ) {
        (Ok(path), Ok(says)) => {
            let bytes = std::fs::read(&path).expect("read the spoken wav");
            let t = engine.transcribe(&bytes).expect("speech");
            println!("[speech] {t:?}");
            let heard = t.text.to_lowercase();
            assert!(
                heard.contains(&says.to_lowercase()),
                "the speech leg lost its phrase: wanted {says:?}, heard {:?}",
                t.text
            );
            assert!(!t.text.contains("<|"));
            assert_eq!(t.silent_windows, 0, "a spoken window was refused: {t:?}");
        }
        _ => eprintln!("speech leg skipped: set COMBS_TEST_WHISPER_WAV and COMBS_TEST_WHISPER_SAYS"),
    }
}
