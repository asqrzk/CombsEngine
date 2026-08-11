//! Token→bytes table validation against real cached tokenizers.
//!
//! Ignored by default; run with:
//! `cargo test --release -p combs-runtime --test constraint_table -- --ignored --nocapture`
//! Set `COMBS_MODELS_DIR` to override the default cache location.

use combs_runtime::{ConstraintSpec, ConstraintState, TokenByteTable};
use tokenizers::Tokenizer;

fn models_dir() -> String {
    std::env::var("COMBS_MODELS_DIR").unwrap_or_else(|_| {
        format!(
            "{}/.cache/combs/models",
            std::env::var("HOME").expect("HOME set")
        )
    })
}

/// Every id whose table bytes are valid UTF-8 must match the tokenizer
/// decoder's own answer for "what does this token append mid-stream"
/// (anchor-diff: decode([a, id]) − decode([a])). Covers GPT-2 byte-level
/// BPE and SPM vocabs.
#[test]
#[ignore = "requires cached model tokenizers (~/.cache/combs/models)"]
fn table_matches_decoder_on_cached_vocabs() {
    let base = models_dir();
    let names = [
        "smollm2-135m",                  // GPT-2 byte-level BPE
        "qwen3-0.6b",                    // byte-level BPE, 151k vocab
        "phi-3.1-mini-4k-instruct-gguf", // SPM (sibling tokenizer.json)
        "gemma-3-1b-it",                 // SPM, 262k vocab
    ];
    let mut ran = 0;
    for name in names {
        let path = format!("{base}/{name}/tokenizer.json");
        if !std::path::Path::new(&path).exists() {
            eprintln!("skip {name}: {path} missing");
            continue;
        }
        let tok = Tokenizer::from_file(&path).expect("tokenizer loads");
        let t0 = std::time::Instant::now();
        let table = TokenByteTable::build(&tok);
        let build_ms = t0.elapsed().as_millis();

        let anchor = *tok.get_vocab(true).get("a").expect("'a' in vocab");
        let anchor_text = tok.decode(&[anchor], false).expect("anchor decodes");
        let n = tok.get_vocab_size(true);
        let step = (n / 2000).max(1);
        let mut checked = 0usize;
        for id in (0..n as u32).step_by(step) {
            let Some(bytes) = table.bytes(id) else { continue };
            let Ok(text) = std::str::from_utf8(bytes) else { continue };
            let full = tok.decode(&[anchor, id], false).expect("pair decodes");
            let Some(piece) = full.strip_prefix(anchor_text.as_str()) else {
                continue;
            };
            assert_eq!(piece, text, "{name}: id {id} table/decoder disagree");
            checked += 1;
        }
        println!("{name}: vocab {n}, table {build_ms}ms, {checked} ids cross-checked");
        assert!(checked > 500, "{name}: too few checkable ids ({checked})");
        ran += 1;
    }
    assert!(ran > 0, "no cached tokenizers found under {base}");
}

/// Drives the automaton with a real 151k vocab: every token of a legal
/// JSON document must survive the mask, and the per-step mask cost is
/// reported (budget: low single-digit ms).
#[test]
#[ignore = "requires cached model tokenizers (~/.cache/combs/models)"]
fn mask_allows_legal_json_and_reports_overhead() {
    let path = format!("{}/qwen3-0.6b/tokenizer.json", models_dir());
    if !std::path::Path::new(&path).exists() {
        eprintln!("skip: {path} missing");
        return;
    }
    let tok = Tokenizer::from_file(&path).expect("tokenizer loads");
    let table = TokenByteTable::build(&tok);
    let n = tok.get_vocab_size(true);

    let doc = r#"{"name": "combs", "age": 3, "tags": ["fast", "local"], "score": -2.5e-1, "ok": true, "extra": null}"#;
    let ids = tok
        .encode(doc, false)
        .expect("encodes")
        .get_ids()
        .to_vec();

    let schema = ConstraintSpec::JsonObject.compile().expect("compiles");
    let mut state = ConstraintState::new(schema, &table, vec![]);
    let mut logits = vec![0.0f32; n];
    let mut total = std::time::Duration::ZERO;
    for &id in &ids {
        logits.fill(0.0);
        let t0 = std::time::Instant::now();
        let allowed = state.mask(&mut logits);
        total += t0.elapsed();
        assert!(allowed > 0);
        assert!(
            logits[id as usize].is_finite(),
            "token {id} ({:?}) of legal JSON was masked out",
            tok.id_to_token(id)
        );
        state.advance(id);
    }
    assert!(state.accepting(), "document should end in an accept state");
    println!(
        "mask overhead: {:.2} ms/step over {} steps at vocab {n}",
        total.as_secs_f64() * 1000.0 / ids.len() as f64,
        ids.len()
    );
}
