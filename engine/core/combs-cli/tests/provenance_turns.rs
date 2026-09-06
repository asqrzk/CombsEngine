//! The record-depth promise, pinned: `turn` depth says "every unit of
//! work", and for weeks the only unit that emitted was the MOUNT — a
//! generation left no trace at any depth, which surfaced only when a
//! timeline tried to draw chats and found silence. This test drives
//! one real generation and asserts the turn records exist, so the
//! promise can never silently regress to mount-only again. Env-gated
//! on the cached smollm2 gguf; skips loudly on model-less CI.

#[test]
fn a_generation_leaves_turn_records_at_turn_depth() {
    let home = std::env::var("HOME").expect("HOME");
    let model = std::path::PathBuf::from(home)
        .join(".cache/combs/models/smollm2-360m-instruct-gguf/model.gguf");
    if !model.is_file() {
        eprintln!("skipping: smollm2-360m gguf not cached");
        return;
    }

    let out = std::process::Command::new(env!("CARGO_BIN_EXE_combs"))
        .args([
            "run",
            "--model",
            model.to_str().unwrap(),
            "--prompt",
            "hi",
            "--max-tokens",
            "4",
            "--temperature",
            "0",
            "--chat",
        ])
        .env("COMBS_PROVENANCE", "turn")
        .env("COMBS_PROVENANCE_FORMAT", "json")
        .output()
        .expect("spawn combs run");

    let stderr = String::from_utf8_lossy(&out.stderr);
    let start = stderr.lines().any(|l| {
        l.starts_with('{')
            && l.contains("\"kind\":\"turn.start\"")
            && l.contains("\"op\":\"generate\"")
    });
    let end = stderr.lines().find(|l| {
        l.starts_with('{')
            && l.contains("\"kind\":\"turn.end\"")
            && l.contains("\"op\":\"generate\"")
    });
    assert!(
        start,
        "no turn.start for the generation in stderr:\n{stderr}"
    );
    let end = end.unwrap_or_else(|| panic!("no turn.end for the generation in stderr:\n{stderr}"));
    assert!(
        end.contains("\"outcome\":\"done\""),
        "outcome missing: {end}"
    );
    assert!(
        end.contains("\"generated_tokens\""),
        "token count missing: {end}"
    );
}
