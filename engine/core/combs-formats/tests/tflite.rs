//! TFLite source structural test: parses a .task file's flatbuffer
//! structure (metadata proto, tokenizer extraction, tensor inventory,
//! name mapping). Weight bytes are only touched when in range, so a
//! truncated recon chunk works for structure validation.
//!
//! Env-gated: `COMBS_TEST_TASK=/path/to/file.task cargo test --release \
//!   -p combs-formats --test tflite -- --nocapture`

use combs_formats::{ModelSource, TfliteSource};

#[test]
fn parses_task_structure() {
    let Ok(path) = std::env::var("COMBS_TEST_TASK") else {
        eprintln!("skipping: set COMBS_TEST_TASK");
        return;
    };
    let source = TfliteSource::load(&path).expect("load tflite source");
    let m = source.metadata();
    println!("architecture: {}", m.architecture);
    println!(
        "hidden={} layers={} heads={} kv={} head_dim={} inter={} vocab={}",
        m.hidden_size,
        m.num_hidden_layers,
        m.num_attention_heads,
        m.num_key_value_heads,
        m.head_dim,
        m.intermediate_size,
        m.vocab_size
    );
    println!(
        "window={:?} pattern={} rope_local={}",
        m.attention_pattern.sliding_window,
        m.attention_pattern.pattern,
        m.attention_pattern.rope_local_theta
    );
    let mut names = source.tensor_names();
    names.sort();
    println!("{} mapped tensors; first 12:", names.len());
    for n in names.iter().take(12) {
        println!("  {n}");
    }
    assert!(names.iter().any(|n| n == "model.embed_tokens.weight"));
    assert!(names.iter().any(|n| n.ends_with("self_attn.q_proj.weight")));
    assert!(names.iter().any(|n| n.ends_with("mlp.down_proj.weight")));
    // Tokenizer extraction + conversion must have run.
    let spec = source.tokenizer().expect("tokenizer");
    assert!(!spec.json_bytes().unwrap().is_empty());
    println!("tokenizer: {}", spec.json_path().expect("path-backed").display());
}

#[test]
fn parses_litertlm_sections() {
    let Ok(path) = std::env::var("COMBS_TEST_LITERTLM_HEAD") else {
        eprintln!("skipping: set COMBS_TEST_LITERTLM_HEAD");
        return;
    };
    let head = std::fs::read(&path).unwrap();
    let sections = combs_formats::litertlm_read_sections(&head).unwrap();
    println!("{} sections:", sections.len());
    for s in &sections {
        println!(
            "  type={} begin={:#x} end={:#x} ({} KB)",
            s.data_type,
            s.begin,
            s.end,
            (s.end - s.begin) / 1024
        );
    }
    assert!(!sections.is_empty());
    assert!(sections.iter().any(|s| s.data_type == 3), "TFLiteModel section");
    for s in &sections {
        assert_eq!(s.begin % (16 * 1024), 0, "16KB alignment");
    }
}
