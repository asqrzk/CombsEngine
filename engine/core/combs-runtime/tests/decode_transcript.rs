//! Golden decode transcript — the gate on any change to the token loop.
//!
//! Greedy decoding is deterministic: the same model, prompt and config must
//! produce the same token ids, the same text pieces, and the same stop
//! behaviour, run after run. That makes a printed transcript a usable
//! before/after witness for refactors of the decode path, where "the tests
//! still pass" is far too coarse a check — a misordered stop test or an
//! off-by-one in the token budget changes the output without failing
//! anything else.
//!
//! Requires a real model and a GPU, so it is ignored by default:
//!
//! ```text
//! COMBS_TEST_GGUF=$HOME/.cache/combs/models/smollm2-360m-instruct-gguf/model.gguf \
//!   cargo test -p combs-runtime --test decode_transcript -- --ignored --nocapture
//! ```
//!
//! Capture the output before a change and diff it against the output after.
//! Every line is deterministic; nothing timing-dependent is printed.
//!
//! RUN ONE TEST PER PROCESS on large-vocabulary models. Each test loads
//! its own engine, the GPU pool never returns memory until the process
//! exits, and autotune's working set scales super-linearly with the
//! vocabulary: one qwen3-0.6b test peaks at 20.5 GB of footprint against
//! 3.2 GB for smollm2-360m. A wholesale ignored run against a large-vocab
//! model stacks eleven such spikes in one address space and has taken the
//! whole machine down. Use `--exact <name>` per run and let the process
//! exit in between; keep suite-wide sweeps on a smollm2-family model.

use combs_runtime::{Engine, GenerationConfig, SamplingParams};

fn engine() -> Option<Engine> {
    let path = std::env::var("COMBS_TEST_GGUF").ok()?;
    let source = combs_formats::open_model_source(&path).expect("open model source");
    Some(Engine::load(&source, combs_core::init_device()).expect("load engine"))
}

/// Greedy config: no temperature, no penalties, no seed dependence.
fn greedy(max_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_tokens,
        sampling: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
        ..GenerationConfig::default()
    }
}

/// Runs one case and prints `case | ids | text | outcome` deterministically.
fn transcript(engine: &Engine, case: &str, prompt: &str, config: &GenerationConfig) {
    let tokens = engine.encode(prompt).expect("encode");
    let mut ids: Vec<u32> = Vec::new();
    let mut text = String::new();
    let result = engine.generate(&tokens, config, |id, piece, _lp| {
        ids.push(id);
        text.push_str(piece);
    });
    let outcome = match &result {
        Ok(s) => format!(
            "ok generated={} prompt={} cached={} pages={}",
            s.generated_tokens, s.prompt_tokens, s.cached_tokens, s.cache_pages_used
        ),
        Err(e) => format!("err {e}"),
    };
    println!("[{case}] ids={ids:?}");
    println!("[{case}] text={text:?}");
    println!("[{case}] {outcome}");
}

#[test]
#[ignore = "requires COMBS_TEST_GGUF and a GPU"]
fn greedy_transcript() {
    let Some(engine) = engine() else {
        eprintln!("skipping: set COMBS_TEST_GGUF");
        return;
    };

    // 1. Plain greedy decode to the token budget.
    transcript(&engine, "budget", "The capital of France is", &greedy(24));

    // 2. Stop string: the piece carrying it must be truncated, not dropped.
    let mut cfg = greedy(48);
    cfg.stop_strings = vec![".".to_string()];
    transcript(&engine, "stop-string", "Count: one, two, three", &cfg);

    // 3. Stop token id: ends without emitting the token.
    let mut cfg = greedy(48);
    if let Some(im_end) = engine.im_end_id() {
        cfg.stop_token_ids = vec![im_end];
    }
    transcript(&engine, "stop-token", "Say hi.", &cfg);

    // 4. Session prefix reuse: the second run shares the first's prefill,
    //    and must produce the same continuation it would have cold.
    let mut cfg = greedy(16);
    cfg.session_id = Some("transcript".to_string());
    transcript(&engine, "session-cold", "The quick brown fox", &cfg);
    transcript(&engine, "session-warm", "The quick brown fox jumps", &cfg);

    // 5. Constrained JSON: the mask must hold under greedy sampling.
    let mut cfg = greedy(64);
    cfg.constraint = combs_runtime::ConstraintSpec::from_response_format(&serde_json::json!({
        "type": "json_object"
    }))
    .expect("response_format");
    transcript(&engine, "json-object", "Emit a JSON object with one key.", &cfg);

    // 6. A single token — the shortest possible budget, where an off-by-one
    //    in the loop bound is most visible.
    transcript(&engine, "one-token", "Hello", &greedy(1));

    // 7. Empty prompt is a request error, not a crash.
    let empty: Vec<u32> = Vec::new();
    let err = engine.generate(&empty, &greedy(4), |_, _, _| {}).unwrap_err();
    println!("[empty-prompt] err={err}");
}

#[test]
#[ignore = "requires COMBS_TEST_GGUF and a GPU; run with COMBS_SPEC=1"]
fn speculative_transcript_matches_plain() {
    let Some(engine) = engine() else {
        eprintln!("skipping: set COMBS_TEST_GGUF");
        return;
    };
    // Prompt-lookup speculation is only a speed trick: greedy output under
    // COMBS_SPEC=1 must equal greedy output without it, token for token.
    // Repetitive text is what triggers drafts at all.
    let prompt = "Repeat after me: alpha beta gamma. alpha beta gamma. alpha beta";
    transcript(&engine, "spec", prompt, &greedy(32));
}

/// The bytes door and the file door must produce the same generation.
///
/// This is the browser's whole claim in one test: a model handed over as
/// bytes — no filesystem, no mmap, no cached tokenizer next to it — decodes
/// to the same tokens as the same file opened the ordinary way. If these
/// ever diverge, the browser is quietly running a different model.
#[test]
#[ignore = "requires COMBS_TEST_GGUF and a GPU"]
fn bytes_loaded_model_matches_file_loaded() {
    let Ok(path) = std::env::var("COMBS_TEST_GGUF") else {
        eprintln!("skipping: set COMBS_TEST_GGUF");
        return;
    };
    let prompt = "The capital of France is";
    let config = greedy(24);

    let from_file = {
        let source = combs_formats::open_model_source(&path).expect("open file");
        let engine = Engine::load(&source, combs_core::init_device()).expect("engine");
        let tokens = engine.encode(prompt).expect("encode");
        let mut ids = Vec::new();
        engine
            .generate(&tokens, &config, |id, _, _| ids.push(id))
            .expect("generate");
        ids
    };

    let from_bytes = {
        let bytes = std::fs::read(&path).expect("read model");
        let source = combs_formats::open_model_source_bytes(bytes).expect("open bytes");
        let engine = Engine::load(&source, combs_core::init_device()).expect("engine");
        let tokens = engine.encode(prompt).expect("encode");
        let mut ids = Vec::new();
        engine
            .generate(&tokens, &config, |id, _, _| ids.push(id))
            .expect("generate");
        ids
    };

    println!("[bytes-parity] file={from_file:?}");
    println!("[bytes-parity] bytes={from_bytes:?}");
    assert!(!from_file.is_empty(), "the file path generated nothing");
    assert_eq!(
        from_file, from_bytes,
        "a model delivered as bytes decoded differently from the same file"
    );
}

/// The browser's engine, driven natively, must decode like the desktop's.
///
/// `LocalEngine` is what runs in a tab: no worker thread, no channel, one
/// awaited token at a time. Its decode logic is shared with `Engine`, but
/// "shared" is a claim about the code, and this is the claim about the
/// output. Run natively because that is where a GPU and a model are.
#[test]
#[ignore = "requires COMBS_TEST_GGUF and a GPU"]
fn local_engine_matches_threaded_engine() {
    use combs_runtime::{LocalEngine, StepEvent};

    let Ok(path) = std::env::var("COMBS_TEST_GGUF") else {
        eprintln!("skipping: set COMBS_TEST_GGUF");
        return;
    };
    let prompt = "The capital of France is";
    let config = greedy(24);

    let source = combs_formats::open_model_source(&path).expect("open model");

    let threaded = {
        let engine = Engine::load(&source, combs_core::init_device()).expect("engine");
        let tokens = engine.encode(prompt).expect("encode");
        let mut ids = Vec::new();
        let mut text = String::new();
        engine
            .generate(&tokens, &config, |id, piece, _| {
                ids.push(id);
                text.push_str(piece);
            })
            .expect("generate");
        (ids, text)
    };

    let local = pollster::block_on(async {
        let mut engine =
            LocalEngine::load(&source, combs_core::init_device()).expect("local engine");
        let tokens = engine.encode(prompt).expect("encode");
        engine.begin(&tokens, &config).expect("begin");
        let mut ids = Vec::new();
        let mut text = String::new();
        loop {
            match engine.step().await.expect("step") {
                StepEvent::Token { id, text: piece, .. } => {
                    ids.push(id);
                    text.push_str(&piece);
                }
                StepEvent::Done { tail, stats } => {
                    text.push_str(&tail);
                    assert_eq!(
                        stats.generated_tokens,
                        ids.len(),
                        "stats disagree with the tokens actually delivered"
                    );
                    break;
                }
            }
        }
        (ids, text)
    });

    println!("[local] threaded={:?}", threaded.0);
    println!("[local] local   ={:?}", local.0);
    println!("[local] text={:?}", local.1);
    assert!(!threaded.0.is_empty(), "the threaded engine generated nothing");
    assert_eq!(threaded.0, local.0, "token ids differ between the two drivers");
    assert_eq!(threaded.1, local.1, "text differs between the two drivers");
}

/// A Stop between tokens keeps what already arrived.
#[test]
#[ignore = "requires COMBS_TEST_GGUF and a GPU"]
fn local_engine_cancels_between_tokens() {
    use combs_runtime::{EngineError, LocalEngine, StepEvent};

    let Ok(path) = std::env::var("COMBS_TEST_GGUF") else {
        eprintln!("skipping: set COMBS_TEST_GGUF");
        return;
    };
    let source = combs_formats::open_model_source(&path).expect("open model");

    pollster::block_on(async {
        let mut engine =
            LocalEngine::load(&source, combs_core::init_device()).expect("local engine");
        let tokens = engine.encode("The capital of France is").expect("encode");
        engine.begin(&tokens, &greedy(256)).expect("begin");

        let mut delivered = 0usize;
        let outcome = loop {
            match engine.step().await {
                Ok(StepEvent::Token { .. }) => {
                    delivered += 1;
                    // Exactly what a Stop button does: set the flag between
                    // two awaited tokens.
                    if delivered == 4 {
                        engine.cancel();
                    }
                }
                Ok(StepEvent::Done { .. }) => break Ok(()),
                Err(e) => break Err(e),
            }
        };
        println!("[local-cancel] delivered={delivered} outcome={outcome:?}");
        assert!(
            matches!(outcome, Err(EngineError::Cancelled)),
            "expected a cancelled run, got {outcome:?}"
        );
        assert!(
            (4..256).contains(&delivered),
            "a cancelled run kept {delivered} tokens"
        );
    });
}

/// An abandoned generation must not wedge the engine.
///
/// A browser tab that navigates away mid-answer drops the future without
/// ever reaching a terminal step. If that left the single-flight slot
/// taken, every later request would be refused for a turn nobody is
/// waiting for.
#[test]
#[ignore = "requires COMBS_TEST_GGUF and a GPU"]
fn abandoned_generation_frees_the_engine() {
    use combs_runtime::{LocalEngine, StepEvent};

    let Ok(path) = std::env::var("COMBS_TEST_GGUF") else {
        eprintln!("skipping: set COMBS_TEST_GGUF");
        return;
    };
    let source = combs_formats::open_model_source(&path).expect("open model");

    pollster::block_on(async {
        let mut engine =
            LocalEngine::load(&source, combs_core::init_device()).expect("local engine");
        let tokens = engine.encode("The capital of France is").expect("encode");

        engine.begin(&tokens, &greedy(64)).expect("first begin");
        let _ = engine.step().await.expect("one token");
        // Starting another turn while one runs is refused, not queued.
        assert!(
            engine.begin(&tokens, &greedy(8)).is_err(),
            "a second generation should be refused while one is in flight"
        );

        assert!(engine.abandon(), "there was a generation to abandon");
        assert!(!engine.abandon(), "abandoning twice reports nothing left");

        // The engine is usable again, and the abandoned turn left no trace.
        engine.begin(&tokens, &greedy(4)).expect("begin after abandon");
        let mut delivered = 0;
        loop {
            match engine.step().await.expect("step") {
                StepEvent::Token { .. } => delivered += 1,
                StepEvent::Done { stats, .. } => {
                    assert_eq!(stats.generated_tokens, delivered);
                    break;
                }
            }
        }
        assert!(delivered > 0, "the recovered engine generated nothing");
        println!("[abandon] recovered and generated {delivered} tokens");
    });
}

/// A sampled turn with no seed — the path every default chat request takes.
///
/// Every other test here is greedy, which never asks the sampler for
/// randomness and so never touches the clock the RNG seeds from. That gap
/// is exactly where the browser build failed: it loaded, reported its
/// metadata, and aborted on the first sampled token. Greedy proves the
/// decode loop; this proves the request shape a user actually sends.
#[test]
#[ignore = "requires COMBS_TEST_GGUF and a GPU"]
fn unseeded_sampled_generation_runs() {
    use combs_runtime::{LocalEngine, StepEvent};

    let Ok(path) = std::env::var("COMBS_TEST_GGUF") else {
        eprintln!("skipping: set COMBS_TEST_GGUF");
        return;
    };
    let source = combs_formats::open_model_source(&path).expect("open model");

    // The console's defaults, verbatim: temperature and penalties on, and
    // no seed — so the sampler must invent one.
    let config = GenerationConfig {
        max_tokens: 16,
        sampling: SamplingParams {
            temperature: 0.7,
            top_p: Some(0.9),
            repetition_penalty: Some(1.1),
            seed: None,
            ..SamplingParams::default()
        },
        ..GenerationConfig::default()
    };

    pollster::block_on(async {
        let mut engine =
            LocalEngine::load(&source, combs_core::init_device()).expect("local engine");
        let tokens = engine.encode("The capital of France is").expect("encode");
        engine.begin(&tokens, &config).expect("begin");
        let mut text = String::new();
        let mut n = 0;
        loop {
            match engine.step().await.expect("step") {
                StepEvent::Token { text: piece, .. } => {
                    text.push_str(&piece);
                    n += 1;
                }
                StepEvent::Done { tail, .. } => {
                    text.push_str(&tail);
                    break;
                }
            }
        }
        println!("[sampled] {n} tokens: {text:?}");
        assert!(n > 0, "an unseeded sampled request generated nothing");
    });
}

/// Native decode throughput on the same model the browser runs, so the
/// browser number has something honest to be compared against.
///
/// Same file, same quantization, same sampling as the console's defaults.
/// Printed, not asserted: a throughput assertion would be a machine
/// benchmark masquerading as a correctness test.
#[test]
#[ignore = "requires COMBS_TEST_GGUF and a GPU; prints, asserts nothing"]
fn native_throughput_baseline() {
    let Ok(path) = std::env::var("COMBS_TEST_GGUF") else {
        eprintln!("skipping: set COMBS_TEST_GGUF");
        return;
    };
    let source = combs_formats::open_model_source(&path).expect("open model");
    let engine = Engine::load(&source, combs_core::init_device()).expect("engine");

    let config = GenerationConfig {
        max_tokens: 128,
        sampling: SamplingParams {
            temperature: 0.7,
            top_p: Some(0.9),
            repetition_penalty: Some(1.1),
            seed: Some(1),
            ..SamplingParams::default()
        },
        ..GenerationConfig::default()
    };
    let prompt = engine.wrap_chat(&[combs_runtime::ChatMessage {
        role: "user".to_string(),
        content: "What are some good software engineering practices?".to_string(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
    }]);
    let tokens = engine.encode(&prompt).expect("encode");

    // Warm the kernels first; the first run of any shape pays for its
    // compilation and would otherwise be reported as decode time.
    let _ = engine.generate(&tokens, &config, |_, _, _| {});
    let stats = engine.generate(&tokens, &config, |_, _, _| {}).expect("generate");

    println!(
        "[native] prompt={} generated={} ttft={:.0}ms decode={:.1} tok/s prefill={:.0} tok/s",
        stats.prompt_tokens,
        stats.generated_tokens,
        stats.ttft.as_secs_f64() * 1000.0,
        stats.decode_tokens_per_second(),
        stats.prefill_tokens_per_second(),
    );
}

/// Prints the device's own report of itself, native, for comparison with
/// what the same code reports inside a browser.
///
/// Must run in a process of its own: `device_caps` performs the cubecl
/// runtime setup, and cubecl 0.10 panics on a second one — any test that
/// has already touched the GPU poisons it. Hence the env gate on top of
/// `#[ignore]`, so a blanket `-- --ignored` does not sweep it up:
///
/// ```text
/// COMBS_CAPS_ONLY=1 cargo test --release -p combs-runtime \
///   --test decode_transcript native_device_caps -- --ignored --nocapture
/// ```
#[test]
#[ignore = "prints the native device caps; must run alone (COMBS_CAPS_ONLY=1)"]
fn native_device_caps() {
    if std::env::var("COMBS_CAPS_ONLY").as_deref() != Ok("1") {
        eprintln!("skipping: set COMBS_CAPS_ONLY=1 and run this test alone");
        return;
    }
    let caps = combs_core::device_caps(&combs_core::init_device());
    println!("[caps] {}", serde_json::to_string(&caps).unwrap());
}

/// A new generation must not inherit the previous one's ending.
///
/// A run that finishes on the same step it emits its last token parks the
/// terminal for one more step. A caller who takes that token and starts a
/// new turn without draining would otherwise be handed the old run's Done
/// — stats and all — as the first result of the new one.
#[test]
#[ignore = "requires COMBS_TEST_GGUF and a GPU"]
fn a_new_generation_does_not_inherit_the_previous_terminal() {
    use combs_runtime::{LocalEngine, StepEvent};

    let Ok(path) = std::env::var("COMBS_TEST_GGUF") else {
        eprintln!("skipping: set COMBS_TEST_GGUF");
        return;
    };
    let source = combs_formats::open_model_source(&path).expect("open model");

    pollster::block_on(async {
        let mut engine =
            LocalEngine::load(&source, combs_core::init_device()).expect("local engine");
        let tokens = engine.encode("The capital of France is").expect("encode");

        // Stop mid-run, then start again without draining the terminal.
        engine.begin(&tokens, &greedy(8)).expect("first begin");
        let _ = engine.step().await.expect("a token");
        engine.abandon();

        // The same shape via the outbox: run to the last token but not past
        // it, then begin again.
        engine.begin(&tokens, &greedy(2)).expect("second begin");
        let mut seen = 0;
        while seen < 2 {
            match engine.step().await.expect("step") {
                StepEvent::Token { .. } => seen += 1,
                StepEvent::Done { .. } => break,
            }
        }
        engine.begin(&tokens, &greedy(4)).expect("third begin");

        // The first result of the new run must be a token, not the
        // previous run's completion.
        match engine.step().await.expect("step") {
            StepEvent::Token { .. } => {}
            StepEvent::Done { stats, .. } => panic!(
                "a fresh generation returned the previous run's terminal \
                 (generated={})",
                stats.generated_tokens
            ),
        }
        let mut delivered = 1;
        loop {
            match engine.step().await.expect("step") {
                StepEvent::Token { .. } => delivered += 1,
                StepEvent::Done { stats, .. } => {
                    assert_eq!(stats.generated_tokens, delivered, "stats belong to this run");
                    break;
                }
            }
        }
        println!("[outbox] fresh run delivered {delivered} tokens of its own");
    });
}

/// A step dropped mid-readback must refuse, not silently repeat a token.
///
/// The logits that future was carrying are gone while `next` has already
/// been emitted, so resuming would emit it twice and feed the cache a
/// duplicate row — wrong output, no error. It is worse than that: the
/// abandoned readback is still outstanding in the GPU runtime, which
/// aborts on the next command instead of returning an error. So the
/// engine must retire itself, and this test holds it to that rather than
/// to a recovery it cannot perform.
#[test]
#[ignore = "requires COMBS_TEST_GGUF and a GPU"]
fn a_dropped_step_refuses_to_resume() {
    use combs_runtime::{EngineError, LocalEngine, StepEvent};
    use std::future::Future;

    let Ok(path) = std::env::var("COMBS_TEST_GGUF") else {
        eprintln!("skipping: set COMBS_TEST_GGUF");
        return;
    };
    let source = combs_formats::open_model_source(&path).expect("open model");

    pollster::block_on(async {
        let mut engine =
            LocalEngine::load(&source, combs_core::init_device()).expect("local engine");
        let tokens = engine.encode("The capital of France is").expect("encode");
        engine.begin(&tokens, &greedy(16)).expect("begin");

        // One completed step, so a token has been emitted and a readback
        // is outstanding.
        let first = match engine.step().await.expect("step") {
            StepEvent::Token { id, .. } => id,
            StepEvent::Done { .. } => panic!("expected a token"),
        };

        // Drive the next step as far as its readback, then drop it. An
        // *unpolled* future would prove nothing — an `async fn` body does
        // not run until polled — so it has to be polled once to reach the
        // await that takes the pending logits.
        let reached_await = {
            let mut fut = std::pin::pin!(engine.step());
            let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
            matches!(fut.as_mut().poll(&mut cx), std::task::Poll::Pending)
        };
        if !reached_await {
            eprintln!("skipping: the readback completed before the future could be dropped");
            return;
        }

        match engine.step().await {
            Err(EngineError::WorkerGone(msg)) => {
                assert!(msg.contains("dropped"), "unhelpful message: {msg}");
                println!("[dropped] {msg}");
            }
            Ok(StepEvent::Token { id, .. }) => panic!(
                "resumed after a dropped step and re-emitted token {id} (first was {first})"
            ),
            Ok(StepEvent::Done { .. }) => panic!("expected a refusal, got a completion"),
            Err(e) => panic!("expected a WorkerGone refusal, got {e}"),
        }

        // And it stays retired: a second attempt gets the same answer
        // rather than an abort from somewhere unrelated.
        let again = engine.begin(&tokens, &greedy(2));
        match again {
            Err(EngineError::WorkerGone(msg)) => {
                assert!(msg.contains("no longer be used"), "unhelpful: {msg}");
            }
            _ => panic!("a retired engine accepted a new generation"),
        }
    });
}
