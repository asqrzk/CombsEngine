//! S2d: the memory program's proof.
//!
//! Loads a real model and holds the allocator to the S2c estimate: live
//! bytes after load must stay within 1.25× of it, generation may grow
//! the pool, and dropping the session (which fires the worker's
//! `BufferPool::cleanup`) must hand memory back rather than hoard it —
//! the failure mode that took the machine down twice (MEASUREMENTS §30).
//!
//! Requires a real model and a GPU, so ignored by default:
//!
//! ```text
//! COMBS_TEST_GGUF=$HOME/.cache/combs/models/qwen3-0.6b-gguf/model.gguf \
//!   cargo test --release -p combs-runtime --test footprint -- --ignored --nocapture
//! ```
//!
//! Run it under both pool policies (`COMBS_MEM_POOLS=exclusive` and
//! default) — the numbers it prints are what decide the native default.
//! One test per process, like every model-loading test.

use combs_runtime::{Engine, GenerationConfig, SamplingParams};

#[test]
#[ignore = "requires COMBS_TEST_GGUF and a GPU"]
fn footprint_stays_bounded_and_cleanup_returns_memory() {
    let Ok(path) = std::env::var("COMBS_TEST_GGUF") else {
        eprintln!("skipping: set COMBS_TEST_GGUF");
        return;
    };
    let source = combs_formats::open_model_source(&path).expect("open model");
    let device = combs_core::init_device();
    let engine = Engine::load(&source, device.clone()).expect("load engine");

    // All memory numbers come from the engine's snapshot: cubecl's
    // accounting is per-stream, so only samples taken on the allocating
    // thread (the loader here, the worker below) see the truth.
    let snap0 = engine.stats_snapshot();
    let estimate = snap0.estimated_model_bytes;
    let weights = snap0.estimated_weight_bytes;
    assert!(weights > 0, "the estimate must exist for the budget to mean anything");
    let at_load = snap0.gpu.expect("gpu sample at load");
    println!(
        "[footprint] estimate={:.1}MB (weights {:.1}MB) | load: in_use={:.1}MB reserved={:.1}MB",
        estimate as f64 / 1e6,
        weights as f64 / 1e6,
        at_load.bytes_in_use as f64 / 1e6,
        at_load.bytes_reserved as f64 / 1e6,
    );
    assert!(
        (at_load.bytes_in_use as f64) <= weights as f64 * 1.25,
        "live bytes after load ({}) blew past 1.25x the weight estimate ({weights})",
        at_load.bytes_in_use,
    );

    let tokens = engine.encode("The capital of France is").expect("encode");
    let config = GenerationConfig {
        max_tokens: 64,
        sampling: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
        ..GenerationConfig::default()
    };
    engine.generate(&tokens, &config, |_, _, _| {}).expect("generate");
    let after_gen = engine.stats_snapshot().gpu.expect("gpu sample after generation");
    println!(
        "[footprint] after 64 tokens: in_use={:.1}MB reserved={:.1}MB",
        after_gen.bytes_in_use as f64 / 1e6,
        after_gen.bytes_reserved as f64 / 1e6,
    );

    let removed = engine.clear_sessions(None).expect("clear sessions");
    assert!(removed >= 1, "the generation left a rolling session to drop");
    let after_clear = engine.stats_snapshot().gpu.expect("gpu sample after clear");
    println!(
        "[footprint] after session drop + cleanup: in_use={:.1}MB reserved={:.1}MB",
        after_clear.bytes_in_use as f64 / 1e6,
        after_clear.bytes_reserved as f64 / 1e6,
    );
    assert!(
        after_clear.bytes_in_use <= after_gen.bytes_in_use,
        "dropping the session must not grow live bytes ({} -> {})",
        after_gen.bytes_in_use,
        after_clear.bytes_in_use,
    );
    assert!(
        after_clear.bytes_reserved <= after_gen.bytes_reserved,
        "cleanup must never grow the pool ({} -> {})",
        after_gen.bytes_reserved,
        after_clear.bytes_reserved,
    );
}

/// The S2c refusal, witnessed: a budget the model cannot fit produces an
/// itemized error BEFORE any weight touches the device — not an
/// allocator panic afterwards.
#[test]
#[ignore = "requires COMBS_TEST_GGUF and a GPU"]
fn an_over_budget_model_is_refused_with_arithmetic() {
    let Ok(path) = std::env::var("COMBS_TEST_GGUF") else {
        eprintln!("skipping: set COMBS_TEST_GGUF");
        return;
    };
    // Process-wide by nature; this suite already runs one test per
    // process, so nothing else can observe the temporary value.
    std::env::set_var("COMBS_VRAM_BUDGET_MB", "1000");
    let source = combs_formats::open_model_source(&path).expect("open model");
    let err = Engine::load(&source, combs_core::init_device())
        .err()
        .expect("a 1000 MB budget cannot hold this model");
    std::env::remove_var("COMBS_VRAM_BUDGET_MB");
    let msg = err.to_string();
    assert!(
        msg.contains("COMBS_VRAM_BUDGET_MB=1000") && msg.contains("kv arena"),
        "the refusal must show its arithmetic; got: {msg}"
    );
    println!("[footprint] refusal: {msg}");
}
