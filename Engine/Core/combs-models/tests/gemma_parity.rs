//! Gemma parity diagnostic: prints the top-8 last-position logits for a
//! fixed prompt so they can be compared against an HF transformers
//! reference run (same model, same tokens, f32).
//!
//! Ignored by default; run with:
//! `COMBS_TEST_MODEL=~/.cache/combs/models/gemma-3-1b-it cargo test --release \
//!   -p combs-models --test gemma_parity -- --ignored --nocapture`

use burn::tensor::{Int, Tensor, TensorData};
use combs_core::{CombsBackend, init_device};
use combs_formats::SafetensorsSource;
use combs_models::{CacheConfig, GenerativeModel, ModelRegistry};

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn print_top_logits() {
    let dir = std::env::var("COMBS_TEST_MODEL").expect("COMBS_TEST_MODEL");
    let source = SafetensorsSource::load(&dir).expect("load source");
    let device = init_device();
    let registry = ModelRegistry::<CombsBackend>::new();
    let mut model = registry.load(&source, &device).expect("load model");

    // "<bos>The capital of France is" — <bos> is an added token, so this
    // matches HF's add_special_tokens=True ids [2, 818, 5279, 529, 7001, 563].
    let tokenizer = tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
    let enc = tokenizer
        .encode("<bos>The capital of France is", false)
        .unwrap();
    let ids: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
    println!("token ids: {ids:?}");

    let seq = ids.len();
    let tokens: Tensor<CombsBackend, 2, Int> =
        Tensor::from_data(TensorData::new(ids, [1, seq]), &device);
    let mut cache = model.create_kv_cache(&CacheConfig::paged(32768));
    let input = model.embed(tokens);
    let logits = model.prefill(input, cache.as_mut(), 0..seq as u32);
    let data: Vec<f32> = logits.into_data().to_vec().expect("logits to vec");

    let mut idx: Vec<usize> = (0..data.len()).collect();
    idx.sort_by(|&a, &b| data[b].partial_cmp(&data[a]).unwrap());
    println!("top-8 next-token logits:");
    for &i in idx.iter().take(8) {
        let piece = tokenizer.decode(&[i as u32], false).unwrap_or_default();
        println!("  {:9.4}  {:7}  {:?}", data[i], i, piece);
    }
}
