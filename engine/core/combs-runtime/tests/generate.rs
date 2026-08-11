//! End-to-end generation against a real model directory.
//!
//! Ignored by default; run with:
//! `COMBS_TEST_MODEL=/path/to/SmolLM2-135M cargo test -p combs-runtime -- --ignored`

use combs_formats::SafetensorsSource;
use combs_runtime::{Engine, GenerationConfig};

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn generates_text_on_gpu() {
    let dir = std::env::var("COMBS_TEST_MODEL")
        .unwrap_or_else(|_| "../../../models/SmolLM2-135M".to_string());
    let source = SafetensorsSource::load(&dir).expect("load source");
    let device = combs_core::init_device();
    let engine = Engine::load(&source, device).expect("load engine");

    let tokens = engine.encode("The capital of France is").unwrap();
    let mut text = String::new();
    let stats = engine
        .generate(
            &tokens,
            &GenerationConfig {
                max_tokens: 16,
                ..Default::default() // greedy
            },
            |_id, piece, _lp| text.push_str(piece),
        )
        .expect("generate");

    println!("generated: {text:?} ({:.1} tok/s)", stats.tokens_per_second());
    assert!(stats.generated_tokens > 0);
    assert!(!text.trim().is_empty());
}

fn load_engine() -> Engine {
    let dir = std::env::var("COMBS_TEST_MODEL")
        .unwrap_or_else(|_| "../../../models/SmolLM2-135M".to_string());
    let source = SafetensorsSource::load(&dir).expect("load source");
    Engine::load(&source, combs_core::init_device()).expect("load engine")
}

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn cancel_flag_stops_generation_between_tokens() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let engine = load_engine();
    let tokens = engine.encode("Tell me a long story about the sea.").unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = cancel.clone();
    std::thread::spawn(move || {
        // Fire after prefill + a few decode steps (TTFT ~250 ms here).
        std::thread::sleep(std::time::Duration::from_millis(600));
        flag.store(true, Ordering::Relaxed);
    });

    let mut pieces = 0;
    let result = engine.generate_cancellable(
        &tokens,
        &GenerationConfig {
            max_tokens: 512,
            ..Default::default()
        },
        cancel,
        |_id, _piece, _lp| pieces += 1,
    );
    assert!(matches!(result, Err(combs_runtime::EngineError::Cancelled)));
    assert!(pieces > 0 && pieces < 512, "partial stream: {pieces} pieces");
}

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn concurrent_generates_queue_single_flight() {
    use std::sync::Arc;

    let engine = Arc::new(load_engine());
    let prompts = ["The capital of France is", "Water is made of"];
    let mut handles = Vec::new();
    for prompt in prompts {
        let engine = engine.clone();
        handles.push(std::thread::spawn(move || {
            let tokens = engine.encode(prompt).unwrap();
            let mut text = String::new();
            let stats = engine
                .generate(
                    &tokens,
                    &GenerationConfig {
                        max_tokens: 8,
                        ..Default::default()
                    },
                    |_id, piece, _lp| text.push_str(piece),
                )
                .expect("generate");
            assert_eq!(stats.generated_tokens, 8);
            text
        }));
    }
    for h in handles {
        assert!(!h.join().unwrap().is_empty());
    }
}
