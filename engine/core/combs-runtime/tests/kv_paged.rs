//! End-to-end equivalence of the paged and contiguous KV caches against a
//! real model, plus chunked-prefill equivalence.
//!
//! Ignored by default; run with:
//! `COMBS_TEST_MODEL=/path/to/SmolLM2-135M cargo test -p combs-runtime --test kv_paged -- --ignored`

use combs_formats::{ModelSource, SafetensorsSource};
use combs_runtime::{CacheConfig, CacheKind, Engine, GenerationConfig};

fn model_dir() -> String {
    std::env::var("COMBS_TEST_MODEL").unwrap_or_else(|_| "../../../models/SmolLM2-135M".to_string())
}

/// Greedy-generates token ids with an explicit cache configuration.
fn greedy_tokens(kind: CacheKind, prompt: &str, max_tokens: usize, chunk: usize) -> Vec<u32> {
    let source = SafetensorsSource::load(model_dir()).expect("load source");
    let device = combs_core::init_device();
    let mut config = CacheConfig::paged(source.metadata().max_position_embeddings);
    config.kind = kind;
    let engine = Engine::load_with_cache_config(&source, device, config).expect("load engine");

    let tokens = engine.encode(prompt).unwrap();
    let mut ids = Vec::new();
    engine
        .generate(
            &tokens,
            &GenerationConfig {
                max_tokens,
                prefill_chunk_size: chunk,
                ..Default::default() // greedy
            },
            |id, _piece| ids.push(id),
        )
        .expect("generate");
    ids
}

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn paged_matches_contiguous_greedy() {
    let prompt = "The capital of France is";
    let contiguous = greedy_tokens(CacheKind::Contiguous, prompt, 32, 0);
    let paged = greedy_tokens(CacheKind::Paged, prompt, 32, 0);
    assert!(
        !contiguous.is_empty() && contiguous == paged,
        "greedy token mismatch:\ncontiguous: {contiguous:?}\npaged:      {paged:?}"
    );
}

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn chunked_prefill_matches_single_shot() {
    let prompt = "Explain what a hash map is in one paragraph.";
    let single = greedy_tokens(CacheKind::Paged, prompt, 32, 0);
    let chunked = greedy_tokens(CacheKind::Paged, prompt, 32, 4);
    assert!(
        !single.is_empty() && single == chunked,
        "chunked prefill mismatch:\nsingle:  {single:?}\nchunked: {chunked:?}"
    );
}
