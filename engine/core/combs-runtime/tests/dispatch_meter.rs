//! The dispatch meter, on a real GPU: a token's cost in dispatches is
//! MEASURED, not derived from an op count and a published microsecond.
//!
//! ```text
//! COMBS_TEST_GGUF=$HOME/.cache/combs/models/qwen3-0.6b-gguf/model.gguf \
//!   cargo test --release -p combs-runtime --test dispatch_meter -- --ignored --nocapture
//! ```
//!
//! It asserts only what must be true on any GPU backend (a token costs
//! dispatches, prefill costs some, submits never outnumber dispatches) and
//! PRINTS the numbers — the point of the instrument is the figure, and the
//! figure is a property of the model and the desk, not a constant to pin.

use combs_runtime::{Engine, GenerationConfig, SamplingParams};

#[test]
#[ignore = "requires COMBS_TEST_GGUF and a GPU"]
fn a_token_costs_measurable_dispatches() {
    let Ok(path) = std::env::var("COMBS_TEST_GGUF") else {
        eprintln!("skipping: set COMBS_TEST_GGUF");
        return;
    };
    let source = combs_formats::open_model_source(&path).expect("open model source");
    let engine = Engine::load(&source, combs_core::init_device()).expect("load engine");

    // A GGUF load repacks weights on the device and so dispatches; a dense
    // safetensors load does not. Both are correct — the load's count is
    // printed, never asserted; the decode's is what this probe stands on.
    let (before, _) = combs_core::gpu_counters();
    println!("[meter] load cost {before} dispatches");

    let tokens = engine.encode("Count to five.").expect("encode");
    let config = GenerationConfig {
        max_tokens: 16,
        sampling: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
        ..GenerationConfig::default()
    };
    let stats = engine
        .generate(&tokens, &config, |_, _, _| {})
        .expect("generate");

    println!(
        "[meter] {} prompt tokens, {} generated — prefill {} dispatches; decode {} dispatches, {} submits",
        stats.prompt_tokens, stats.generated_tokens, stats.prefill_dispatches, stats.dispatches, stats.submits
    );
    println!(
        "[meter] {:.1} dispatches/token, {:.2} submits/token, {:.2} tok/s → {:.1} us/dispatch implied",
        stats.dispatches_per_token(),
        stats.submits_per_token(),
        stats.decode_tokens_per_second(),
        if stats.dispatches_per_token() > 0.0 {
            1e6 / stats.decode_tokens_per_second().max(1e-9) / stats.dispatches_per_token()
        } else {
            0.0
        }
    );

    // what the seat costs the desk, beside what it costs the driver
    if let Some(m) = combs_core::gpu_memory(&combs_core::init_device()) {
        println!(
            "[meter] gpu pool: {:.0} MB in use, {:.0} MB reserved",
            m.bytes_in_use as f64 / 1048576.0,
            m.bytes_reserved as f64 / 1048576.0
        );
    }

    assert!(stats.generated_tokens > 0, "nothing decoded");
    assert!(stats.dispatches > 0, "the decode loop dispatched nothing");
    assert!(stats.prefill_dispatches > 0, "prefill dispatched nothing");
    assert!(
        stats.submits <= stats.dispatches,
        "more submits ({}) than dispatches ({}) — the meter counts the wrong thing",
        stats.submits,
        stats.dispatches
    );
    // a transformer token is tens of dispatches at least and thousands at
    // most; outside that band the meter is measuring something else
    let per = stats.dispatches_per_token();
    assert!(
        (10.0..10_000.0).contains(&per),
        "{per:.1} dispatches/token is outside any plausible band"
    );
}
