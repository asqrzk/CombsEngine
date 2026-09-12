//! The workhorse: what THIS model costs on THIS box.
//!
//! The estate has measured this by hand for months — a test invoked with
//! an environment variable, its printed lines copied into a private
//! ledger by a person. That is why an outside reader once quoted a
//! figure taken before the kernels existed. One command, one record, and
//! the numbers carry their own provenance.
//!
//! Two rules bind every figure here. The reported run is the WARM
//! second of a pair, because a cold run times compilation and pool
//! growth rather than the model. And each depth is its own child
//! process (see `child`), because the pool does not release until exit
//! and the dispatch counter cannot tell two subjects apart.

use combs_runtime::{Engine, GenerationConfig, SamplingParams};

use crate::child::Subject;
use crate::record::{Run, Value};

/// A prompt of roughly `want` tokens. The text is deliberately ordinary
/// prose: a pathological prompt would measure the tokenizer, not the
/// model.
fn prompt_of(want: usize) -> String {
    if want == 0 {
        return "Count to five.".to_string();
    }
    let unit = "the ledger records every seat and every measured number for this estate. ";
    unit.repeat(want / 12 + 2)
}

/// What a person means when they name a model.
///
/// A packed model is a directory holding one `.gguf`, and that is how
/// the cache stores it and how anyone will type it. The loader wants the
/// file. Resolving it here costs nothing and saves every caller the
/// discovery that a directory "is not a model"; a directory carrying a
/// `config.json` is already a model in the other sense and is left
/// alone. More than one `.gguf` is ambiguous, and ambiguity is returned
/// to the person rather than guessed at.
pub fn resolve(path: &str) -> Result<String, String> {
    let p = std::path::Path::new(path);
    if !p.is_dir() || p.join("config.json").exists() {
        return Ok(path.to_string());
    }
    let mut ggufs: Vec<std::path::PathBuf> = std::fs::read_dir(p)
        .map_err(|e| format!("cannot read {path}: {e}"))?
        .flatten()
        .map(|e| e.path())
        .filter(|q| q.extension().is_some_and(|x| x == "gguf"))
        .collect();
    ggufs.sort();
    match ggufs.len() {
        0 => Ok(path.to_string()),
        1 => Ok(ggufs.remove(0).to_string_lossy().into_owned()),
        n => Err(format!(
            "{path} holds {n} .gguf files — name the one you mean"
        )),
    }
}

/// What a model is CALLED, as opposed to where it happens to sit.
///
/// A record carries this, never the path. Two reasons, and the second is
/// the one that bit: a baseline naming a home directory's cache is
/// not portable and cannot be compared against anyone else's run — and
/// committing it would publish a home directory, which the hygiene gate
/// refuses by name and was right to.
pub fn identity(path: &str) -> String {
    let p = std::path::Path::new(path);
    // A packed seat is a directory holding one file; the DIRECTORY is
    // the name people use for it.
    let named = if p.extension().is_some_and(|x| x == "gguf") {
        p.parent().and_then(|d| d.file_name())
    } else {
        p.file_name()
    };
    named
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Measure one subject. Runs INSIDE a child process; never call it from
/// the parent.
pub fn run_one(subject: &Subject) -> Run {
    let mut notes = Vec::new();
    let resolved = match resolve(&subject.model) {
        Ok(r) => r,
        Err(e) => {
            return Run {
                id: subject.id.clone(),
                kind: "model".into(),
                subject: identity(&subject.model),
                config: subject.to_json_for_record(),
                warm: false,
                values: Vec::new(),
                notes: vec![format!("no measurement: {e}")],
            };
        }
    };
    let source = match combs_formats::open_model_source(&resolved) {
        Ok(s) => s,
        Err(e) => {
            return Run {
                id: subject.id.clone(),
                kind: "model".into(),
                subject: identity(&subject.model),
                config: subject.to_json_for_record(),
                warm: false,
                values: Vec::new(),
                notes: vec![format!("no measurement: cannot open the model: {e}")],
            };
        }
    };

    let load_started = std::time::Instant::now();
    let engine = match Engine::load(&source, combs_core::init_device()) {
        Ok(e) => e,
        Err(e) => {
            return Run {
                id: subject.id.clone(),
                kind: "model".into(),
                subject: identity(&subject.model),
                config: subject.to_json_for_record(),
                warm: false,
                values: Vec::new(),
                notes: vec![format!("no measurement: cannot load the model: {e}")],
            };
        }
    };
    let load_ms = load_started.elapsed().as_secs_f64() * 1e3;
    let (load_dispatches, _) = combs_core::gpu_counters();

    let prompt = prompt_of(subject.prompt_tokens);
    let tokens = match engine.encode(&prompt) {
        Ok(t) => t,
        Err(e) => {
            return Run {
                id: subject.id.clone(),
                kind: "model".into(),
                subject: identity(&subject.model),
                config: subject.to_json_for_record(),
                warm: false,
                values: Vec::new(),
                notes: vec![format!("no measurement: cannot encode the prompt: {e}")],
            };
        }
    };
    let config = GenerationConfig {
        max_tokens: subject.max_tokens,
        sampling: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
        ..GenerationConfig::default()
    };

    // Warm the pipeline, then measure ONCE.
    //
    // The obvious design — run the same prompt twice and keep the
    // second — is wrong here, and wrong in a way that produces a
    // beautiful number: the cache still holds the prompt on the second
    // pass, so prefill is SKIPPED and the rate reads as six figures.
    // The first version of this file printed 114,580 tokens a second at
    // depth 2000, which is what a measurement of nothing looks like.
    //
    // So the warm-up uses a SHORT, DIFFERENT prompt — it pays for
    // shader compilation, autotune and pool growth, none of which the
    // model pays twice — and the sessions are cleared before the
    // measured run, so the prompt is prefilled for real.
    let warm = engine
        .encode("Count to five.")
        .unwrap_or_else(|_| tokens.clone());
    let warmup = GenerationConfig {
        max_tokens: 4,
        ..config.clone()
    };
    if let Err(e) = engine.generate(&warm, &warmup, |_, _, _| {}) {
        notes.push(format!("the warm-up generation failed: {e}"));
    }
    if let Err(e) = engine.clear_sessions(None) {
        notes.push(format!(
            "sessions could not be cleared, so prefill may be measuring a cache hit: {e}"
        ));
    }
    let stats = match engine.generate(&tokens, &config, |_, _, _| {}) {
        Ok(s) => s,
        Err(e) => {
            return Run {
                id: subject.id.clone(),
                kind: "model".into(),
                subject: identity(&subject.model),
                config: subject.to_json_for_record(),
                warm: true,
                values: Vec::new(),
                notes: vec![format!("no measurement: the measured generation failed: {e}")],
            };
        }
    };

    let decode_tok_s = stats.decode_tokens_per_second();
    let disp_per_tok = stats.dispatches_per_token();
    let mut values = vec![
        Value::measured("load_ms", load_ms, "ms"),
        Value::measured("load_dispatches", load_dispatches as f64, "dispatches"),
        Value::measured("prompt_tokens", stats.prompt_tokens as f64, "tokens"),
        Value::measured("generated_tokens", stats.generated_tokens as f64, "tokens"),
        Value::measured("ttft_ms", stats.ttft.as_secs_f64() * 1e3, "ms"),
        Value::measured(
            "prefill_tok_s",
            stats.prefill_tokens_per_second(),
            "tokens/s",
        ),
        Value::measured("decode_tok_s", decode_tok_s, "tokens/s"),
        Value::measured("dispatches_per_token", disp_per_tok, "dispatches/token"),
        Value::measured("submits_per_token", stats.submits_per_token(), "submits/token"),
        Value::measured(
            "prefill_dispatches",
            stats.prefill_dispatches as f64,
            "dispatches",
        ),
    ];
    // Derived, and it says so: this is the number that turns a dispatch
    // count into a decision about whether a seat is launch-bound or
    // bandwidth-bound, and reading it as a measurement would be wrong.
    if decode_tok_s > 0.0 && disp_per_tok > 0.0 {
        values.push(Value::derived(
            "us_per_dispatch",
            1e6 / decode_tok_s / disp_per_tok,
            "us",
            "1e6 / decode_tok_s / dispatches_per_token",
        ));
    }
    // Sampled between generations, never during one: reading the pool is
    // a blocking submit on the compute stream.
    if let Some(mem) = combs_core::gpu_memory(&combs_core::init_device()) {
        values.push(Value::measured(
            "pool_in_use_mb",
            mem.bytes_in_use as f64 / 1_048_576.0,
            "MB",
        ));
        values.push(Value::measured(
            "pool_reserved_mb",
            mem.bytes_reserved as f64 / 1_048_576.0,
            "MB",
        ));
    }

    // The engine complains if it is dropped without one, and a bench
    // whose output is half diagnostics is harder to read than it needs
    // to be.
    engine.shutdown();

    Run {
        id: subject.id.clone(),
        kind: "model".into(),
        subject: identity(&subject.model),
        config: subject.to_json_for_record(),
        warm: true,
        values,
        notes,
    }
}

/// Print one run as a human line. The record is the artefact; this is
/// so a person watching a sweep can see it moving.
pub fn print_run(run: &Run) {
    let get = |name: &str| {
        run.values
            .iter()
            .find(|v| v.name == name)
            .map(|v| v.value)
            .unwrap_or(f64::NAN)
    };
    if run.values.is_empty() {
        println!("  {:<8} {}", run.id, run.notes.join("; "));
        return;
    }
    println!(
        "  {:<8} prompt {:>6.0} | prefill {:>7.1} tok/s | ttft {:>7.0} ms | decode {:>7.2} tok/s | {:>8.1} disp/tok | {:>6.1} us/disp | pool {:>7.0} MB",
        run.id,
        get("prompt_tokens"),
        get("prefill_tok_s"),
        get("ttft_ms"),
        get("decode_tok_s"),
        get("dispatches_per_token"),
        get("us_per_dispatch"),
        get("pool_in_use_mb"),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The prompt must actually reach the depth asked for, or every
    /// "at 2k" figure in the record is a figure at some other depth.
    #[test]
    fn a_prompt_reaches_roughly_the_depth_asked_for() {
        // ~12 tokens per repeated unit; the check is deliberately loose
        // because the real count comes from the tokenizer and is
        // RECORDED from the run, never assumed.
        let words = prompt_of(2000).split_whitespace().count();
        assert!(words >= 2000, "asked for 2000 tokens, got {words} words");
        assert!(words < 4000, "overshot badly: {words} words");
        assert_eq!(prompt_of(0), "Count to five.");
    }

    /// A path that is not a packed directory is handed back untouched —
    /// the resolver must never invent a file where the loader would
    /// have given a clear error.
    #[test]
    fn a_path_that_is_not_a_packed_directory_is_left_alone() {
        assert_eq!(resolve("/nowhere/at/all").unwrap(), "/nowhere/at/all");
    }
}
