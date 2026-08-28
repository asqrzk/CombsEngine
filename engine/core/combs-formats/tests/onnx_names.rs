//! Real-export smoke: the ONNX source over an actual HF export must
//! surface HF-canonical tensor names and the sibling tokenizer +
//! chat template. Geometry-only — forward parity lives in
//! combs-models' cross-source test.
//!
//! Run:
//! ```sh
//! cargo test -p combs-formats --test onnx_names -- --ignored --nocapture
//! ```
//! (uses $HOME/.cache/combs/models/qwen3-0.6b-onnx/onnx/model_fp16.onnx,
//! or COMBS_TEST_ONNX.)

use combs_formats::{ModelSource, OnnxSource};

fn model_path() -> String {
    std::env::var("COMBS_TEST_ONNX").unwrap_or_else(|_| {
        format!(
            "{}/.cache/combs/models/qwen3-0.6b-onnx/onnx/model_fp16.onnx",
            std::env::var("HOME").expect("HOME")
        )
    })
}

#[test]
#[ignore = "requires a downloaded ONNX export (COMBS_TEST_ONNX)"]
fn real_export_surfaces_canonical_names() {
    let path = model_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} not present");
        return;
    }
    let source = OnnxSource::load(&path).expect("open");

    let meta = source.metadata();
    assert_eq!(meta.architecture, "qwen3", "architecture from sibling config");
    assert_eq!(meta.num_hidden_layers, 28);
    assert_eq!(meta.hidden_size, 1024);

    let names = source.tensor_names();
    println!("[onnx] {} tensors; first 12 sorted:", names.len());
    let mut sorted = names.clone();
    sorted.sort();
    for n in sorted.iter().take(12) {
        println!("  {n}");
    }
    // The names the llama-family loader will actually ask for.
    for want in [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.norm.weight",
    ] {
        assert!(
            names.iter().any(|n| n == want),
            "expected canonical name {want} — first names: {:?}",
            sorted.iter().take(20).collect::<Vec<_>>()
        );
    }

    let spec = source.tokenizer().expect("tokenizer");
    assert!(spec.chat_template.is_some(), "chat template from siblings");
    let reader = source.open_tensor("model.embed_tokens.weight").expect("embed");
    println!("[onnx] embed shape {:?} dtype {:?}", reader.shape(), reader.dtype());
}
