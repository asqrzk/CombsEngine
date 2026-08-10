//! SentencePiece converter tests. The parity test is env-gated (like
//! COMBS_TEST_MODEL): point COMBS_TEST_SPM at a .model file and
//! COMBS_TEST_SPM_REF at the HF tokenizer.json converted from it.
//!
//!   COMBS_TEST_SPM=/tmp/spiece.model COMBS_TEST_SPM_REF=/tmp/t5-tokenizer.json \
//!     cargo test --release -p combs-formats --test spm

use std::path::Path;

use combs_formats::{ensure_tokenizer_json_from_spm, spm_added_tokens};

#[test]
fn rejects_non_spm_files() {
    let dir = tempfile::tempdir().unwrap();
    let bogus = dir.path().join("bogus.model");
    std::fs::write(&bogus, b"definitely not a sentencepiece protobuf\xFF\xFF").unwrap();
    assert!(ensure_tokenizer_json_from_spm(&bogus).is_err());
}

#[test]
fn parity_with_hf_conversion() {
    let (Some(spm), Some(reference)) = (
        std::env::var("COMBS_TEST_SPM").ok(),
        std::env::var("COMBS_TEST_SPM_REF").ok(),
    ) else {
        eprintln!("skipping: set COMBS_TEST_SPM + COMBS_TEST_SPM_REF");
        return;
    };
    let json_path = ensure_tokenizer_json_from_spm(Path::new(&spm)).unwrap();
    let ours: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).unwrap();
    let theirs: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&reference).unwrap()).unwrap();

    let our_vocab = ours["model"]["vocab"].as_array().unwrap();
    let their_vocab = theirs["model"]["vocab"].as_array().unwrap();
    // HF converters may append model-family extras (e.g. T5's 100
    // <extra_id_*> sentinels) — our spm vocab must be an exact PREFIX.
    assert!(our_vocab.len() <= their_vocab.len(), "vocab length");

    // Piece strings must match exactly; scores are f32→f64 round-trips.
    let mut score_drift = 0usize;
    for (o, t) in our_vocab.iter().zip(their_vocab.iter()) {
        assert_eq!(o[0], t[0], "piece mismatch");
        let ds = (o[1].as_f64().unwrap() - t[1].as_f64().unwrap()).abs();
        if ds > 1e-6 {
            score_drift += 1;
        }
    }
    assert!(score_drift == 0, "score drift in {score_drift} pieces");

    // unk id + special tokens.
    assert_eq!(ours["model"]["unk_id"], theirs["model"]["unk_id"]);
    let specials = spm_added_tokens(Path::new(&spm)).unwrap();
    assert!(!specials.is_empty(), "expected special tokens");
    for (id, text) in &specials {
        assert_eq!(&their_vocab[*id as usize][0].as_str().unwrap(), text);
    }

    // The converted file must load in the tokenizers crate and encode.
    let tokenizer = tokenizers::Tokenizer::from_file(&json_path).unwrap();
    let enc = tokenizer.encode("Hello world, this is a parity check.", false).unwrap();
    assert!(!enc.get_ids().is_empty());
    eprintln!(
        "parity OK: {} pieces, {} specials, sample encoding {} tokens",
        our_vocab.len(),
        specials.len(),
        enc.get_ids().len()
    );
}
