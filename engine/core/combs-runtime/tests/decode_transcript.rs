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
//! COMBS_TEST_GGUF=$HOME/.cache/combs/models/qwen3-0.6b-gguf/model.gguf \
//!   cargo test -p combs-runtime --test decode_transcript -- --ignored --nocapture
//! ```
//!
//! Capture the output before a change and diff it against the output after.
//! Every line is deterministic; nothing timing-dependent is printed.

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
