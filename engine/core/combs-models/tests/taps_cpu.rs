//! CPU harmony for the multi-depth residual taps: the conditioning
//! surface diffusion text encoders read. NdArray backend isolates the
//! tap plumbing from wgpu kernel behavior.
//!
//! Ignored by default; run with:
//! `cargo test -p combs-models --test taps_cpu --release -- --ignored --nocapture`
//! (uses ../../../models/SmolLM2-135M, or COMBS_TEST_MODEL).

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use combs_formats::{ModelSource, SafetensorsSource};
use combs_models::{CacheConfig, GenerativeModel, LlamaModel};

type B = NdArray<f32>;

fn model_dir() -> String {
    std::env::var("COMBS_TEST_MODEL").unwrap_or_else(|_| "../../../models/SmolLM2-135M".to_string())
}

fn load() -> (LlamaModel<B>, Vec<u32>) {
    let source = SafetensorsSource::load(model_dir()).expect("load source");
    let tokenizer =
        tokenizers::Tokenizer::from_bytes(source.tokenizer().unwrap().json_bytes().unwrap())
            .unwrap();
    let tokens = tokenizer
        .encode("The capital of France is Paris, and the capital of Italy is", false)
        .unwrap()
        .get_ids()
        .to_vec();
    let device = Default::default();
    let model =
        LlamaModel::<B>::load(&source as &dyn ModelSource, &device).expect("load model");
    (model, tokens)
}

fn embed(model: &LlamaModel<B>, tokens: &[u32]) -> Tensor<B, 3> {
    let data: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    model.embed(Tensor::from_data(
        TensorData::new(data, [1, tokens.len()]),
        &Default::default(),
    ))
}

fn taps(
    model: &mut LlamaModel<B>,
    tokens: &[u32],
    which: &[usize],
    chunk: usize,
) -> Tensor<B, 3> {
    let mut cache = model.create_kv_cache(&CacheConfig::contiguous(4096));
    let embedded = embed(model, tokens);
    let chunk = if chunk == 0 { usize::MAX } else { chunk };
    let mut offset = 0;
    let mut parts = Vec::new();
    while offset < tokens.len() {
        let len = chunk.min(tokens.len() - offset);
        let input = embedded.clone().narrow(1, offset, len);
        parts.push(
            model
                .prefill_taps(
                    input,
                    cache.as_mut(),
                    offset as u32..(offset + len) as u32,
                    which,
                )
                .expect("taps"),
        );
        offset += len;
    }
    Tensor::cat(parts, 1)
}

fn max_abs_diff(a: &Tensor<B, 3>, b: &Tensor<B, 3>) -> f32 {
    let av: Vec<f32> = a.clone().into_data().to_vec().unwrap();
    let bv: Vec<f32> = b.clone().into_data().to_vec().unwrap();
    assert_eq!(av.len(), bv.len(), "shape mismatch");
    av.iter().zip(&bv).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn tap_zero_is_the_embedding_stream() {
    let (mut model, tokens) = load();
    let t0 = taps(&mut model, &tokens, &[0], 0);
    let e = embed(&model, &tokens);
    assert_eq!(max_abs_diff(&t0, &e), 0.0, "tap 0 must be the raw embeddings, bitwise");
}

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn taps_concat_in_order_and_differ_by_depth() {
    let (mut model, tokens) = load();
    let layers = model.metadata().num_hidden_layers;
    let (a, b) = (layers / 3, 2 * layers / 3);
    let both = taps(&mut model, &tokens, &[a, b], 0);
    let ta = taps(&mut model, &tokens, &[a], 0);
    let tb = taps(&mut model, &tokens, &[b], 0);
    let cat = Tensor::cat(vec![ta.clone(), tb.clone()], 2);
    assert_eq!(
        max_abs_diff(&both, &cat),
        0.0,
        "multi-tap must equal the ordered concat of single taps, bitwise"
    );
    assert!(
        max_abs_diff(&ta, &tb) > 0.0,
        "different depths must produce different streams"
    );
    let [_, seq, width] = both.dims();
    assert_eq!(seq, tokens.len());
    assert_eq!(width, 2 * model.metadata().hidden_size);
}

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn chunked_taps_match_single_shot() {
    let (mut model, tokens) = load();
    let layers = model.metadata().num_hidden_layers;
    let which = [0, layers / 2, layers];
    let single = taps(&mut model, &tokens, &which, 0);
    let chunked = taps(&mut model, &tokens, &which, 5);
    let diff = max_abs_diff(&single, &chunked);
    println!("[taps] chunked-vs-single max abs diff {diff:e}");
    // CPU matmuls reduce in shape-dependent order (the prefill_cpu bar
    // for the amplified logits is 0.05); the raw residual stream must
    // stay well inside f32-noise. Measured 2.1e-4 on SmolLM2-135M.
    assert!(diff < 1e-3, "chunked taps drifted: {diff}");
}

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn final_tap_is_pre_norm() {
    let (mut model, tokens) = load();
    let layers = model.metadata().num_hidden_layers;
    let tap_n = taps(&mut model, &tokens, &[layers], 0);
    let mut cache = model.create_kv_cache(&CacheConfig::contiguous(4096));
    let embedded = embed(&model, &tokens);
    let n = tokens.len() as u32;
    let hidden = model
        .prefill_hidden(embedded, cache.as_mut(), 0..n)
        .expect("hidden");
    assert!(
        max_abs_diff(&tap_n, &hidden) > 0.0,
        "the deepest tap must be PRE-final-norm (prefill_hidden is post-norm)"
    );
}

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn bad_taps_err_cleanly() {
    let (mut model, tokens) = load();
    let layers = model.metadata().num_hidden_layers;
    let embedded = embed(&model, &tokens);
    let n = tokens.len() as u32;
    let mut cache = model.create_kv_cache(&CacheConfig::contiguous(4096));
    assert!(
        model
            .prefill_taps(embedded.clone(), cache.as_mut(), 0..n, &[layers + 1])
            .is_err(),
        "out-of-range tap must be refused"
    );
    let mut cache = model.create_kv_cache(&CacheConfig::contiguous(4096));
    assert!(
        model
            .prefill_taps(embedded, cache.as_mut(), 0..n, &[])
            .is_err(),
        "empty tap list must be refused"
    );
}
