//! Phi-3 fused-projection loading, on synthetic safetensors checkpoints:
//! - `self_attn.qkv_proj.weight` (`[q|k|v]` rows) and `mlp.gate_up_proj.weight`
//!   (`[gate|up]` rows) split at load into the uniform decoder layout, with
//!   logits identical to an equivalent pre-split checkpoint;
//! - phi3's always-on sliding window (2047) loads unguarded and is a
//!   numerical no-op for sequences shorter than the window.

use std::io::Write;
use std::path::Path;

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use combs_formats::SafetensorsSource;
use combs_models::{CacheConfig, ModelRegistry};

type B = NdArray<f32>;

const HIDDEN: usize = 8;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = HIDDEN / HEADS;
const Q_ROWS: usize = HEADS * HEAD_DIM;
const KV_ROWS: usize = KV_HEADS * HEAD_DIM;
const INTERMEDIATE: usize = 16;
const VOCAB: usize = 16;

/// Deterministic values keyed on the SPLIT tensor name, so the fused
/// checkpoint (which concatenates these) and the split checkpoint hold the
/// same weights.
fn wave(name: &str, len: usize) -> Vec<f32> {
    let seed: usize = name.bytes().map(|b| b as usize).sum();
    (0..len)
        .map(|i| ((seed * 31 + i) as f32 * 0.618).sin() * 0.1)
        .collect()
}

fn concat(parts: &[(String, usize)]) -> Vec<f32> {
    let mut out = Vec::new();
    for (name, len) in parts {
        out.extend(wave(name, *len));
    }
    out
}

/// Writes `model.safetensors` + `config.json` + a stub `tokenizer.json`;
/// `fused` selects the phi on-disk layout (qkv_proj/gate_up_proj) vs the
/// split llama layout — with identical underlying values either way.
fn write_model_dir(dir: &Path, config: &str, fused: bool) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config).unwrap();
    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();

    let p = "model.layers.0";
    let q_name = format!("{p}.self_attn.q_proj.weight");
    let k_name = format!("{p}.self_attn.k_proj.weight");
    let v_name = format!("{p}.self_attn.v_proj.weight");
    let gate_name = format!("{p}.mlp.gate_proj.weight");
    let up_name = format!("{p}.mlp.up_proj.weight");

    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut push = |name: &str, shape: Vec<usize>, data: Option<Vec<f32>>| {
        let len: usize = shape.iter().product();
        let values = data.unwrap_or_else(|| wave(name, len));
        tensors.push((name.to_string(), shape, values));
    };

    push("model.embed_tokens.weight", vec![VOCAB, HIDDEN], None);
    push("model.norm.weight", vec![HIDDEN], Some(vec![1.0; HIDDEN]));
    push(&format!("{p}.input_layernorm.weight"), vec![HIDDEN], Some(vec![1.0; HIDDEN]));
    push(
        &format!("{p}.post_attention_layernorm.weight"),
        vec![HIDDEN],
        Some(vec![1.0; HIDDEN]),
    );
    if fused {
        push(
            &format!("{p}.self_attn.qkv_proj.weight"),
            vec![Q_ROWS + 2 * KV_ROWS, HIDDEN],
            Some(concat(&[
                (q_name.clone(), Q_ROWS * HIDDEN),
                (k_name.clone(), KV_ROWS * HIDDEN),
                (v_name.clone(), KV_ROWS * HIDDEN),
            ])),
        );
        push(
            &format!("{p}.mlp.gate_up_proj.weight"),
            vec![2 * INTERMEDIATE, HIDDEN],
            Some(concat(&[
                (gate_name.clone(), INTERMEDIATE * HIDDEN),
                (up_name.clone(), INTERMEDIATE * HIDDEN),
            ])),
        );
    } else {
        push(&q_name, vec![Q_ROWS, HIDDEN], None);
        push(&k_name, vec![KV_ROWS, HIDDEN], None);
        push(&v_name, vec![KV_ROWS, HIDDEN], None);
        push(&gate_name, vec![INTERMEDIATE, HIDDEN], None);
        push(&up_name, vec![INTERMEDIATE, HIDDEN], None);
    }
    push(&format!("{p}.self_attn.o_proj.weight"), vec![HIDDEN, HIDDEN], None);
    push(&format!("{p}.mlp.down_proj.weight"), vec![HIDDEN, INTERMEDIATE], None);

    // Hand-rolled safetensors: u64 header length, JSON header, raw LE data.
    let mut header_entries = Vec::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, shape, values) in &tensors {
        let start = data.len();
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let shape_json = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        header_entries.push(format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape_json}],\"data_offsets\":[{start},{}]}}",
            data.len()
        ));
    }
    let header = format!("{{{}}}", header_entries.join(","));

    let mut file = std::fs::File::create(dir.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(&data).unwrap();
}

/// Phi-3-mini-shaped config: fused projections on disk, always-on sliding
/// window (like every shipped mini), no `use_sliding_window` key.
fn config(model_type: &str, sliding: bool) -> String {
    let window = if sliding { r#", "sliding_window": 2047"# } else { "" };
    format!(
        r#"{{
        "model_type": "{model_type}",
        "hidden_size": {HIDDEN},
        "intermediate_size": {INTERMEDIATE},
        "num_hidden_layers": 1,
        "num_attention_heads": {HEADS},
        "num_key_value_heads": {KV_HEADS},
        "vocab_size": {VOCAB},
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": true,
        "bos_token_id": 0,
        "eos_token_id": 15{window}
    }}"#
    )
}

fn last_logits(dir: &Path) -> Vec<f32> {
    let device = Default::default();
    let source = SafetensorsSource::load(dir).expect("open source");
    let registry = ModelRegistry::<B>::new();
    let mut model = registry.load(&source, &device).expect("load model");
    let mut cache = model.create_kv_cache(&CacheConfig::contiguous(64));
    let tokens: Vec<i32> = vec![1, 2, 3, 4];
    let n = tokens.len();
    let embedded = model.embed(Tensor::from_data(TensorData::new(tokens, [1, n]), &device));
    let logits = model.prefill(embedded, cache.as_mut(), 0..n as u32);
    logits.into_data().to_vec().unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn fused_projections_match_split_checkpoint() {
    let root = std::env::temp_dir().join(format!("combs-phi-fused-{}", std::process::id()));
    let fused = root.join("fused");
    let split = root.join("split");
    write_model_dir(&fused, &config("phi3", true), true);
    write_model_dir(&split, &config("phi3", true), false);

    let a = last_logits(&fused);
    let b = last_logits(&split);
    assert!(
        max_abs_diff(&a, &b) < 1e-6,
        "fused qkv/gate_up split must reproduce the pre-split checkpoint"
    );

    // Same weights under plain llama (no window): phi3's sliding window is
    // far larger than the sequence, so the layouts must agree numerically —
    // the sliding plumbing may not perturb short contexts.
    let llama = root.join("llama");
    write_model_dir(&llama, &config("llama", false), false);
    let c = last_logits(&llama);
    assert!(
        max_abs_diff(&b, &c) < 1e-6,
        "dormant sliding window must be a numerical no-op"
    );

    std::fs::remove_dir_all(&root).ok();
}
