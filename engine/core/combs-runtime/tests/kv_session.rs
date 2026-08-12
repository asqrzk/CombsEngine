//! Rolling-session KV prefix reuse: a second request whose prompt extends
//! the first request's history must serve most of its prompt from the KV
//! cache (`cached_tokens > 0`) and produce the same greedy continuation as
//! a cold (fresh-cache) request.
//!
//! Ignored by default; run with:
//! `COMBS_TEST_MODEL=/path/to/SmolLM2-135M cargo test -p combs-runtime --test kv_session -- --ignored`

use combs_formats::SafetensorsSource;
use combs_runtime::{Engine, GenerationConfig};

fn model_dir() -> String {
    std::env::var("COMBS_TEST_MODEL").unwrap_or_else(|_| "../../../models/SmolLM2-135M".to_string())
}

fn load_engine() -> Engine {
    let source = SafetensorsSource::load(model_dir()).expect("load source");
    let device = combs_core::init_device();
    Engine::load(&source, device).expect("load engine")
}

fn greedy(max_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_tokens,
        ..Default::default() // greedy; session_reuse on by default
    }
}

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn second_turn_reuses_prefix() {
    let engine = load_engine();
    let prompt1 = engine.encode("The capital of France is").unwrap();

    let mut reply1 = Vec::new();
    let stats1 = engine
        .generate(&prompt1, &greedy(16), |id, _, _| reply1.push(id))
        .expect("generate turn 1");
    assert_eq!(stats1.cached_tokens, 0, "first request is a cold start");
    assert!(!reply1.is_empty());

    // Turn 2: prompt1 + reply1 + a new suffix — the classic chat pattern.
    let suffix = engine.encode(" And the capital of Italy is").unwrap();
    let mut prompt2 = prompt1.clone();
    prompt2.extend_from_slice(&reply1);
    prompt2.extend_from_slice(&suffix);

    let mut reply2 = Vec::new();
    let stats2 = engine
        .generate(&prompt2, &greedy(16), |id, _, _| reply2.push(id))
        .expect("generate turn 2");
    assert!(
        stats2.cached_tokens >= prompt1.len(),
        "turn 2 should reuse at least the original prompt prefix: cached={} prompt1={}",
        stats2.cached_tokens,
        prompt1.len(),
    );
    assert!(stats2.cached_tokens < prompt2.len());
    assert!(!reply2.is_empty());
}

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn reused_prefix_matches_cold_generation() {
    let engine = load_engine();
    let prompt1 = engine.encode("Water is made of").unwrap();
    let mut reply1 = Vec::new();
    engine
        .generate(&prompt1, &greedy(8), |id, _, _| reply1.push(id))
        .expect("generate turn 1");

    let suffix = engine.encode(" Hydrogen is").unwrap();
    let mut prompt2 = prompt1.clone();
    prompt2.extend_from_slice(&reply1);
    prompt2.extend_from_slice(&suffix);

    // Warm (session reuse) generation.
    let mut warm = Vec::new();
    let stats = engine
        .generate(&prompt2, &greedy(24), |id, _, _| warm.push(id))
        .expect("warm generate");
    assert!(stats.cached_tokens > 0);

    // Cold generation of the same prompt on a fresh engine (no session).
    let cold_engine = load_engine();
    let mut cold = Vec::new();
    cold_engine
        .generate(&prompt2, &greedy(24), |id, _, _| cold.push(id))
        .expect("cold generate");

    assert_eq!(warm, cold, "prefix reuse must not change greedy output");
}

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn divergent_prompt_gets_no_reuse() {
    let engine = load_engine();
    let a = engine.encode("One two three").unwrap();
    engine.generate(&a, &greedy(4), |_, _, _| {}).expect("gen a");

    let b = engine.encode("Completely different words altogether").unwrap();
    let stats = engine.generate(&b, &greedy(4), |_, _, _| {}).expect("gen b");
    assert_eq!(stats.cached_tokens, 0, "unrelated prompt must cold-start");
}

fn agent_turn(
    engine: &Engine,
    prompt: &[u32],
    session: &str,
    max_tokens: usize,
) -> (Vec<u32>, combs_runtime::GenerationStats) {
    let mut ids = Vec::new();
    let stats = engine
        .generate(
            prompt,
            &GenerationConfig {
                max_tokens,
                session_id: Some(session.to_string()),
                ..Default::default()
            },
            |id, _, _| ids.push(id),
        )
        .expect("agent turn");
    (ids, stats)
}

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn named_sessions_reuse_when_interleaved() {
    let engine = load_engine();

    // Two agents alternate; each has its own named session (debate pattern).
    let pa = engine.encode("You are Alice, debating for local AI. Open the debate.").unwrap();
    let (ra, sa) = agent_turn(&engine, &pa, "alice", 8);
    assert_eq!(sa.cached_tokens, 0, "alice's first turn is cold");

    let pb = engine.encode("You are Bob, debating against local AI. Open the debate.").unwrap();
    let (rb, sb) = agent_turn(&engine, &pb, "bob", 8);
    assert_eq!(sb.cached_tokens, 0, "bob's first turn is cold");

    // Alice's second turn extends HER prompt — reuse despite Bob in between.
    let mut pa2 = pa.clone();
    pa2.extend_from_slice(&ra);
    pa2.extend_from_slice(&engine.encode(" Bob disagreed. Respond.").unwrap());
    let (_, sa2) = agent_turn(&engine, &pa2, "alice", 8);
    assert!(
        sa2.cached_tokens >= pa.len(),
        "alice's second turn should reuse her named session: cached={} prev_prompt={}",
        sa2.cached_tokens,
        pa.len(),
    );

    // Bob's second turn likewise.
    let mut pb2 = pb.clone();
    pb2.extend_from_slice(&rb);
    pb2.extend_from_slice(&engine.encode(" Alice disagreed. Respond.").unwrap());
    let (_, sb2) = agent_turn(&engine, &pb2, "bob", 8);
    assert!(
        sb2.cached_tokens >= pb.len(),
        "bob's second turn should reuse his named session: cached={} prev_prompt={}",
        sb2.cached_tokens,
        pb.len(),
    );
}
