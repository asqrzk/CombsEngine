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
fn cancel_before_first_token_returns_cancelled() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    // The flag is already set when the worker picks the request up, so the
    // first loop iteration sees it. Nothing here depends on how fast the
    // machine decodes or on what the model would have said.
    let engine = load_engine();
    let tokens = engine.encode("The capital of France is").unwrap();
    let mut pieces = 0;
    let result = engine.generate_cancellable(
        &tokens,
        &GenerationConfig {
            max_tokens: 64,
            ..Default::default()
        },
        Arc::new(AtomicBool::new(true)),
        |_id, _piece, _lp| pieces += 1,
    );
    assert!(
        matches!(result, Err(combs_runtime::EngineError::Cancelled)),
        "expected Cancelled, got {result:?}"
    );
    assert_eq!(pieces, 0, "a request cancelled before decoding emits nothing");
}

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn cancel_flag_stops_generation_between_tokens() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let engine = load_engine();
    // A continuation prompt, not an instruction: the test model is a base
    // model and answers an instruction by stopping immediately, which would
    // leave nothing to cancel and the test passing vacuously. This prompt
    // runs to the budget when left alone.
    let tokens = engine.encode("The capital of France is").unwrap();
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = cancel.clone();

    // Raised from the stream itself rather than from a timer: racing a
    // sleep against decode speed measures the machine, not the engine.
    let mut pieces = 0;
    let result = engine.generate_cancellable(
        &tokens,
        &GenerationConfig {
            max_tokens: 256,
            ..Default::default()
        },
        cancel,
        |_id, _piece, _lp| {
            pieces += 1;
            if pieces == 1 {
                flag.store(true, Ordering::Relaxed);
            }
        },
    );
    assert!(
        matches!(result, Err(combs_runtime::EngineError::Cancelled)),
        "expected Cancelled after {pieces} pieces, got {result:?}"
    );
    assert!(pieces > 0 && pieces < 256, "partial stream: {pieces} pieces");
}

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn an_engine_shuts_down_and_its_worker_is_joined() {
    let engine = load_engine();
    let tokens = engine.encode("The capital of France is").unwrap();
    let mut text = String::new();
    engine
        .generate(
            &tokens,
            &GenerationConfig {
                max_tokens: 4,
                ..Default::default()
            },
            |_id, piece, _lp| text.push_str(piece),
        )
        .expect("a turn before shutdown");
    assert!(!text.is_empty());
    engine.shutdown();
    // The worker is gone: the next request says so instead of hanging.
    let after = engine.generate(
        &tokens,
        &GenerationConfig {
            max_tokens: 4,
            ..Default::default()
        },
        |_id, _piece, _lp| {},
    );
    assert!(
        matches!(after, Err(combs_runtime::EngineError::WorkerGone(_))),
        "expected WorkerGone after shutdown, got {after:?}"
    );
    // Idempotent: a second shutdown finds nothing and returns.
    engine.shutdown();
}

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn a_dropped_engine_still_joins() {
    // No shutdown on purpose: Drop must send Shutdown, join, and come
    // back — a hang here is the failure this test exists to catch.
    let engine = load_engine();
    let tokens = engine.encode("Water is").unwrap();
    engine
        .generate(
            &tokens,
            &GenerationConfig {
                max_tokens: 2,
                ..Default::default()
            },
            |_id, _piece, _lp| {},
        )
        .expect("one turn");
    drop(engine);
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

