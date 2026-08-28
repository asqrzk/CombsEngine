//! Reference parity for the residual taps against the transformers
//! hidden-state stack: hidden_states[k] there = the stream after k
//! layers, and `prefill_taps` must mean the SAME k — published tap
//! indices (a diffusion pipeline's layer choices) carry over verbatim
//! or the conditioning is silently wrong.
//!
//! Constants dumped from Qwen3-0.6B fp32 on CPU (transformers
//! reference; the dump script feeds the identical token ids). Two-tier
//! assertion: per-position means catch global drift, a fixed 16-wide
//! slice catches localized corruption a mean would average away.
//!
//! Ignored by default; run with:
//! `cargo test -p combs-models --test taps_reference --release -- --ignored --nocapture`
//! (uses $HOME/.cache/combs/models/qwen3-0.6b, or COMBS_TEST_QWEN3).

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use combs_formats::{ModelSource, SafetensorsSource};
use combs_models::{CacheConfig, GenerativeModel, LlamaModel};

type B = NdArray<f32>;

// "The capital of France is Paris, and the capital of Italy is"
// tokenized by the model's own tokenizer, add_special_tokens=false.
const TOKENS: &[u32] = &[
    785, 6722, 315, 9625, 374, 12095, 11, 323, 279, 6722, 315, 15344, 374,
];

const TAPS: &[usize] = &[7, 14, 21];

// Mean over the hidden dim, positions 0..8, per tap.
#[rustfmt::skip]
const MEANS8: [[f32; 8]; 3] = [
    [7.1236334, 0.009708432, 0.039928474, 0.019017301, 0.008490283, 0.027207321, 0.015442053, 0.0031721375],
    [7.1209598, 0.07896635, 0.10223127, 0.082185254, 0.03385194, 0.119286723, 0.050049797, 0.034136347],
    [7.1519098, 0.26826739, 0.24044602, 0.16053550, 0.18177554, 0.22562754, 0.084702909, 0.23341468],
];

// tap[last_position, 0..16], per tap.
#[rustfmt::skip]
const LAST_SLICE16: [[f32; 16]; 3] = [
    [-1.4597695, 0.9072963, 0.43216819, -1.8243908, 0.34087744, 1.3217051, -0.20360091, -3.5159619, 1.6269513, 0.23206761, -1.0736624, 0.13022181, -0.72736025, -1.7287837, 0.43655980, -0.17480326],
    [-0.59424925, 4.0296307, -0.32141596, -3.2944944, 1.3269160, 1.8186721, -1.2060933, -6.0469046, 4.1932487, -0.62802768, -4.2390571, 1.3141286, -1.6195167, -4.1412077, 3.0319335, -0.34096664],
    [-1.5320184, 8.6262722, 30.167416, 0.80172449, 7.3842144, -2.5369949, -4.0861859, -35.639744, 2.3881860, -1.9694457, -12.348632, -2.7879565, -5.6728296, 9.5829477, 6.2419853, -6.0324078],
];

fn model_dir() -> String {
    std::env::var("COMBS_TEST_QWEN3").unwrap_or_else(|_| {
        format!(
            "{}/.cache/combs/models/qwen3-0.6b",
            std::env::var("HOME").expect("HOME")
        )
    })
}

#[test]
#[ignore = "requires the qwen3-0.6b model directory (COMBS_TEST_QWEN3)"]
fn taps_match_the_transformers_hidden_state_stack() {
    let source = SafetensorsSource::load(model_dir()).expect("load source");
    let device = Default::default();
    let mut model =
        LlamaModel::<B>::load(&source as &dyn ModelSource, &device).expect("load model");
    assert_eq!(model.metadata().num_hidden_layers, 28, "qwen3-0.6b expected");

    let data: Vec<i32> = TOKENS.iter().map(|&t| t as i32).collect();
    let embedded = model.embed(Tensor::from_data(
        TensorData::new(data, [1, TOKENS.len()]),
        &device,
    ));
    let mut cache = model.create_kv_cache(&CacheConfig::contiguous(4096));
    let out = model
        .prefill_taps(embedded, cache.as_mut(), 0..TOKENS.len() as u32, TAPS)
        .expect("taps");
    let hidden = model.metadata().hidden_size;
    let [_, seq, width] = out.dims();
    assert_eq!(seq, TOKENS.len());
    assert_eq!(width, TAPS.len() * hidden);
    let flat: Vec<f32> = out.into_data().to_vec().unwrap();
    let at = |pos: usize, tap: usize, c: usize| flat[pos * width + tap * hidden + c];

    // Both sides are f32 CPU with different reduction orders; the bar
    // is f32-noise scaled to the value (deep-layer activations reach
    // |35|). Measured max scaled error ~1e-4 territory.
    let close = |a: f32, b: f32| (a - b).abs() < 1e-3 + 2e-4 * b.abs();

    let mut worst = 0.0f32;
    for (t, _) in TAPS.iter().enumerate() {
        for pos in 0..8 {
            let mean = (0..hidden).map(|c| at(pos, t, c)).sum::<f32>() / hidden as f32;
            let want = MEANS8[t][pos];
            worst = worst.max((mean - want).abs() / (1.0 + want.abs()));
            assert!(
                close(mean, want),
                "tap {} pos {pos}: mean {mean} vs reference {want}",
                TAPS[t]
            );
        }
        for c in 0..16 {
            let got = at(TOKENS.len() - 1, t, c);
            let want = LAST_SLICE16[t][c];
            worst = worst.max((got - want).abs() / (1.0 + want.abs()));
            assert!(
                close(got, want),
                "tap {} slice[{c}]: {got} vs reference {want}",
                TAPS[t]
            );
        }
    }
    println!("[taps] transformers parity: worst scaled error {worst:e}");
}
