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
    // A named arena seats the room the app would seat; none leaves the
    // engine's own default, and the record reports whichever it was.
    let loaded = match subject.arena {
        Some(tokens) => {
            let mut config = combs_runtime::CacheConfig::paged(tokens);
            if matches!(std::env::var("COMBS_KV").as_deref(), Ok("contiguous")) {
                config.kind = combs_runtime::CacheKind::Contiguous;
            }
            Engine::load_with_cache_config(&source, combs_core::init_device(), config)
        }
        None => Engine::load(&source, combs_core::init_device()),
    };
    let engine = match loaded {
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
    // The model as the ENGINE reports it after loading, not as the path
    // suggests. A directory named "14b" that held a 7b would otherwise be
    // recorded as a 14b — and the person reading the record has asked,
    // plainly, that nothing about the model be assumed.
    let meta = engine.metadata();
    let loaded = serde_json::json!({
        "file": std::path::Path::new(&resolved)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        "architecture": meta.architecture,
        "layers": meta.num_hidden_layers,
        "hidden": meta.hidden_size,
        "heads": meta.num_attention_heads,
        "kv_heads": meta.num_key_value_heads,
        "head_dim": meta.head_dim,
        "vocab": meta.vocab_size,
        "max_positions": meta.max_position_embeddings,
    });
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
        // How many decode steps the throughput figure is actually over.
        // `decode_tok_s` is (generated - 1)/decode_time — the first token
        // is prefill's — so `--tokens 16` times FIFTEEN steps while
        // llama-bench's `-n 16` times sixteen. The records carried the
        // ratio without carrying that; now they carry both.
        Value::measured(
            "decode_tokens_timed",
            stats.generated_tokens.saturating_sub(1) as f64,
            "tokens",
        ),
        // The bandwidth denominator, read from the device and not from
        // the file: the repack widens Q8_0's scale f16 -> f32 (+5.88%)
        // and Q4_K's superblock to 148 B from 144 (+2.78%), so a GB/s
        // over GGUF bytes under-reports the traffic.
        Value::measured(
            "packed_device_bytes",
            combs_models::packed_device_bytes() as f64,
            "bytes",
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
    // What a room costs and how large one this seat may hold: the two
    // figures a posture card is priced from. Before this they were read
    // off a running server's stats by hand.
    let cfg = engine.cache_config();
    let snap = engine.stats_snapshot();
    if cfg.page_size > 0 {
        values.push(Value::measured(
            "kv_bytes_per_token",
            snap.kv_page_bytes as f64 / cfg.page_size as f64,
            "bytes/token",
        ));
    }
    // A hybrid's recurrent state: fixed per session, beside the pages.
    // Arithmetic over the file's geometry, so it is labelled as such.
    if snap.recurrent_state_bytes > 0 {
        values.push(Value::derived(
            "recurrent_state_bytes",
            snap.recurrent_state_bytes as f64,
            "bytes/session",
            "DeltaNet layers x (value heads x key dim x value dim + (kernel - 1) x conv channels) x 4 B",
        ));
    }
    values.push(Value::measured(
        "arena_max_tokens",
        cfg.max_seq_len as f64,
        "tokens",
    ));
    values.push(Value::measured(
        "kv_int8",
        if cfg.quantize_kv { 1.0 } else { 0.0 },
        "bool",
    ));

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

    // What was ACTUALLY live, asked of the code that read each door,
    // and what ACTUALLY dispatched. Assumed configuration is how this
    // estate measured the wrong backend for an afternoon; a run that
    // cannot say what it ran is a number with no question attached.
    let mut doors: Vec<(String, String)> = combs_models::active_doors(
        &<combs_core::wgpu::WgpuRuntime as cubecl::prelude::Runtime>::client(
            &combs_core::init_device(),
        ),
    )
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    // An open door is not a road taken. Only THIS crate knows whether
    // the tensors reaching those kernels are f16, so only this crate may
    // say whether the storage road was walked.
    if let Some(entry) = doors.iter_mut().find(|(k, _)| k == "f16_storage_door") {
        entry.1 = match (entry.1 == "true", cfg!(feature = "f16")) {
            (true, true) => "open · taken (f16 tensors reach the kernels)".to_string(),
            (true, false) => "open · never taken (this build has no f16 tensors)".to_string(),
            (false, _) => "shut".to_string(),
        };
    }
    let census = combs_models::quant_census();

    // The engine complains if it is dropped without one, and a bench
    // whose output is half diagnostics is harder to read than it needs
    // to be.
    engine.shutdown();

    // The batched-matmul road's EFFECTIVE configuration — threshold,
    // kernel, operand width — read from the same function `combs serve`
    // stamps at startup. `env_requested` says what was asked; this says
    // what took effect. §110 blessed a 1.9x prefill door and seven
    // anchor sittings then ran with it off, unrecorded, because the
    // record carried only the ask.
    let matmul = combs_models::batched_matmul_summary();
    let config = record_config(subject, doors, census, matmul, loaded);

    Run {
        id: subject.id.clone(),
        kind: "model".into(),
        subject: identity(&subject.model),
        config,
        warm: true,
        values,
        notes,
    }
}

/// The record's `config`: the subject by name, then what was ACTUALLY
/// live in the measuring process — the doors, the kernel census, the
/// batched-matmul road in effect, and the model as loaded.
fn record_config(
    subject: &crate::child::Subject,
    doors: Vec<(String, String)>,
    census: String,
    matmul: String,
    loaded: serde_json::Value,
) -> serde_json::Value {
    let mut config = subject.to_json_for_record();
    if let Some(obj) = config.as_object_mut() {
        obj.insert(
            "doors".into(),
            serde_json::Value::Object(doors.into_iter().map(|(k, v)| (k, serde_json::Value::String(v))).collect()),
        );
        obj.insert("quant_dispatch".into(), serde_json::Value::String(census));
        obj.insert("matmul".into(), serde_json::Value::String(matmul));
        obj.insert("loaded".into(), loaded);
    }
    config
}

/// Several readings of one subject, merged into a Run whose values
/// carry the median and the spread.
///
/// The readings are separate CHILDREN, not a loop inside one: a loop
/// would share a warmed pool, a settled clock and one thermal state,
/// and would therefore agree with itself. The point is to catch the run
/// that disagrees.
pub fn merge(readings: Vec<Run>) -> Run {
    let mut first = match readings.first() {
        Some(r) => r.clone(),
        None => return Run {
            id: "none".into(),
            kind: "model".into(),
            subject: "none".into(),
            config: serde_json::Value::Null,
            warm: false,
            values: vec![],
            notes: vec!["no measurement: nothing to merge".into()],
        },
    };
    if readings.len() == 1 {
        return first;
    }
    let mut names: Vec<String> = Vec::new();
    for r in &readings {
        for v in &r.values {
            if !names.contains(&v.name) {
                names.push(v.name.clone());
            }
        }
    }
    let mut merged = Vec::new();
    for name in names {
        let group: Vec<crate::record::Value> = readings
            .iter()
            .flat_map(|r| r.values.iter())
            .filter(|v| v.name == name)
            .cloned()
            .collect();
        // A value missing from some readings is reported on the ones it
        // appeared in, and says so — never silently averaged over fewer.
        if group.len() < readings.len() {
            first.notes.push(format!(
                "{name}: {} of {} readings carried it",
                group.len(),
                readings.len()
            ));
        }
        if let Some(v) = crate::record::Value::from_samples(group) {
            merged.push(v);
        }
    }
    let failed = readings.iter().filter(|r| r.values.is_empty()).count();
    if failed > 0 {
        first
            .notes
            .push(format!("{failed} of {} readings produced nothing", readings.len()));
    }
    first.values = merged;
    first
}

/// Print one run as a human line. The record is the artefact; this is
/// so a person watching a sweep can see it moving.
/// The spread badge on the bench line: the figure people quote carries
/// its own range on the same line, so a range cannot be read as a number.
/// It is the WORST spread among the headline figures (prefill and
/// decode), named: §105's first cell had a steady decode and a ±121%
/// prefill, and a decode-only badge let the range reach the record and
/// never the screen.
pub fn spread_badge(run: &Run) -> String {
    let worst = ["decode_tok_s", "prefill_tok_s"]
        .iter()
        .filter_map(|name| {
            let v = run.values.iter().find(|v| &v.name == name)?;
            v.spread_pct.map(|sp| (sp, v.samples.unwrap_or(1), name.trim_end_matches("_tok_s")))
        })
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    match worst {
        Some((sp, n, which)) if sp > crate::record::SPREAD_BAND_PCT => {
            format!("  ±{sp:.0}% over {n} on {which} — A RANGE, NOT A NUMBER")
        }
        Some((sp, n, _)) => format!("  ±{sp:.0}% over {n}"),
        None => String::new(),
    }
}

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
    let tail = spread_badge(run);
    if let Some(m) = run.config.get("loaded") {
        println!(
            "  {:<8} loaded {} · {} · {} layers · {}q/{}kv heads × {} · hidden {} · file {}",
            "",
            m["architecture"].as_str().unwrap_or("?"),
            run.subject,
            m["layers"],
            m["heads"],
            m["kv_heads"],
            m["head_dim"],
            m["hidden"],
            m["file"].as_str().unwrap_or("?"),
        );
    }
    if let Some(req) = run.config.get("env_requested").and_then(|d| d.as_object()) {
        let asked: Vec<String> = req
            .iter()
            .map(|(k, v)| format!("{k}={}", v.as_str().unwrap_or("?")))
            .collect();
        println!("  {:<8} asked {}", "", asked.join(" "));
    }
    if let Some(doors) = run.config.get("doors").and_then(|d| d.as_object()) {
        let live: Vec<String> = doors
            .iter()
            .map(|(k, v)| format!("{k}={}", v.as_str().unwrap_or("?")))
            .collect();
        println!("  {:<8} doors {}", "", live.join(" "));
        if let Some(c) = run.config.get("quant_dispatch").and_then(|v| v.as_str()) {
            if !c.is_empty() {
                println!("  {:<8} dispatch {c}", "");
            }
        }
        if let Some(m) = run.config.get("matmul").and_then(|v| v.as_str()) {
            println!("  {:<8} matmul {m}", "");
        }
    }
    println!(
        "  {:<8} prompt {:>6.0} | prefill {:>7.1} tok/s | ttft {:>7.0} ms | decode {:>7.2} tok/s | {:>8.1} disp/tok | {:>6.1} us/disp | pool {:>7.0} MB{tail}",
        run.id,
        get("prompt_tokens"),
        get("prefill_tok_s"),
        get("ttft_ms"),
        get("decode_tok_s"),
        get("dispatches_per_token"),
        get("us_per_dispatch"),
        get("pool_in_use_mb"),
    );
    let kv = get("kv_bytes_per_token");
    if kv.is_finite() {
        let state = get("recurrent_state_bytes");
        println!(
            "  {:<8} kv {:>7.0} B/token{} | arena {:>6.0} tokens{}",
            "",
            kv,
            if get("kv_int8") == 1.0 { " (int8)" } else { "" },
            get("arena_max_tokens"),
            if state.is_finite() {
                format!(" | state {:.1} MB/session", state / 1_048_576.0)
            } else {
                String::new()
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The prompt must actually reach the depth asked for, or every
    /// "at 2k" figure in the record is a figure at some other depth.
    /// §105's first cell: decode steady at ±0.7%, prefill at ±121%. The
    /// badge printed "±1% over 2" and the range reached the record but
    /// never the screen. The badge must name the WORST headline spread.
    #[test]
    fn the_badge_names_the_worst_headline_spread_not_only_decode() {
        use crate::record::Value;
        let two = |name: &str, a: f64, b: f64| {
            Value::from_samples(vec![
                Value::measured(name, a, "tokens/s"),
                Value::measured(name, b, "tokens/s"),
            ])
            .unwrap()
        };
        let run = Run {
            id: "d4".into(),
            kind: "model".into(),
            subject: "seat".into(),
            config: serde_json::Value::Null,
            warm: true,
            values: vec![two("prefill_tok_s", 5.0, 22.0), two("decode_tok_s", 18.5, 18.6)],
            notes: vec![],
        };
        let badge = spread_badge(&run);
        assert!(badge.contains("A RANGE"), "a ±121% prefill is a range: {badge:?}");
        assert!(badge.contains("prefill"), "the badge names the figure that ranged: {badge:?}");
        let steady = Run { values: vec![two("prefill_tok_s", 22.0, 22.1), two("decode_tok_s", 18.5, 18.6)], ..run };
        let badge = spread_badge(&steady);
        assert!(!badge.contains("A RANGE"), "{badge:?}");
        assert!(badge.contains("over 2"), "{badge:?}");
    }

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

    /// The record must name the batched-matmul road IN EFFECT — threshold,
    /// kernel and operand width — not only the doors that were asked for.
    /// §110 blessed a 1.9x prefill door on 2026-09-13; seven anchor
    /// sittings then ran with it off and nothing in the record said so,
    /// because `env_requested` records the ask and nothing recorded the
    /// effect. The string is the one `combs serve` stamps at startup.
    #[test]
    fn a_record_config_names_the_effective_prefill_operand() {
        let subject = crate::child::Subject {
            id: "d4".into(),
            model: "some-seat".into(),
            prompt_tokens: 4,
            max_tokens: 16,
            arena: Some(16_384),
        };
        let live = combs_models::batched_matmul_summary();
        assert!(
            live.contains("operands") || live.contains("OFF"),
            "the summary names the operand width or says the road is off: {live:?}"
        );
        let config = record_config(
            &subject,
            vec![("wgsl".into(), "true".into())],
            "packed on device=1".into(),
            live.clone(),
            serde_json::json!({}),
        );
        assert_eq!(config["matmul"].as_str(), Some(live.as_str()), "{config}");
        assert_eq!(config["quant_dispatch"].as_str(), Some("packed on device=1"));
        assert_eq!(config["doors"]["wgsl"].as_str(), Some("true"));
        assert_eq!(config["arena"].as_u64(), Some(16_384), "the subject's own fields survive");
    }
}
