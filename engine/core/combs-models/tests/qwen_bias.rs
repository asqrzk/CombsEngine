//! Qwen2-shape loading rules, on synthetic safetensors checkpoints:
//! - q/k/v biases load by tensor presence — HF Qwen2 configs never emit an
//!   `attention_bias` flag, so gating on metadata would silently skip them;
//! - the registry rejects checkpoints whose sliding window is actually
//!   active (the llama block would run them unmasked and silently wrong).

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
const INTERMEDIATE: usize = 16;
const VOCAB: usize = 16;

/// Deterministic small values so logits are nontrivial but reproducible.
/// Keyed on the tensor NAME (not push order) so the same tensor gets the
/// same values whether or not bias tensors are interleaved into the file.
fn wave(name: &str, len: usize) -> Vec<f32> {
    let seed: usize = name.bytes().map(|b| b as usize).sum();
    (0..len)
        .map(|i| ((seed * 31 + i) as f32 * 0.618).sin() * 0.1)
        .collect()
}

/// Writes `model.safetensors` + `config.json` + a stub `tokenizer.json`.
/// `bias` = Some(v) writes q/k/v projection biases filled with `v`.
fn write_model_dir(dir: &Path, config: &str, bias: Option<f32>) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config).unwrap();
    // SafetensorsSource only requires the file to exist at load time.
    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();

    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut push = |name: &str, shape: Vec<usize>, data: Option<Vec<f32>>| {
        let len: usize = shape.iter().product();
        let values = data.unwrap_or_else(|| wave(name, len));
        tensors.push((name.to_string(), shape, values));
    };

    push("model.embed_tokens.weight", vec![VOCAB, HIDDEN], None);
    push("model.norm.weight", vec![HIDDEN], Some(vec![1.0; HIDDEN]));
    let p = "model.layers.0";
    push(&format!("{p}.input_layernorm.weight"), vec![HIDDEN], Some(vec![1.0; HIDDEN]));
    push(
        &format!("{p}.post_attention_layernorm.weight"),
        vec![HIDDEN],
        Some(vec![1.0; HIDDEN]),
    );
    push(&format!("{p}.self_attn.q_proj.weight"), vec![HEADS * HEAD_DIM, HIDDEN], None);
    push(&format!("{p}.self_attn.k_proj.weight"), vec![KV_HEADS * HEAD_DIM, HIDDEN], None);
    push(&format!("{p}.self_attn.v_proj.weight"), vec![KV_HEADS * HEAD_DIM, HIDDEN], None);
    push(&format!("{p}.self_attn.o_proj.weight"), vec![HIDDEN, HIDDEN], None);
    if let Some(v) = bias {
        push(&format!("{p}.self_attn.q_proj.bias"), vec![HEADS * HEAD_DIM], Some(vec![v; HEADS * HEAD_DIM]));
        push(&format!("{p}.self_attn.k_proj.bias"), vec![KV_HEADS * HEAD_DIM], Some(vec![v; KV_HEADS * HEAD_DIM]));
        push(&format!("{p}.self_attn.v_proj.bias"), vec![KV_HEADS * HEAD_DIM], Some(vec![v; KV_HEADS * HEAD_DIM]));
    }
    push(&format!("{p}.mlp.gate_proj.weight"), vec![INTERMEDIATE, HIDDEN], None);
    push(&format!("{p}.mlp.up_proj.weight"), vec![INTERMEDIATE, HIDDEN], None);
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

/// Qwen2-style config: declares a BOS and a (disabled) sliding window, and —
/// like the real HF export — has NO `attention_bias` key.
fn qwen_config(extra: &str) -> String {
    format!(
        r#"{{
        "model_type": "qwen2",
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
        "eos_token_id": 15{extra}
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
fn qwen_bias_loads_by_presence_without_config_flag() {
    let root = std::env::temp_dir().join(format!("combs-qwen-bias-{}", std::process::id()));
    let with_bias = root.join("bias");
    let without_bias = root.join("nobias");
    let zero_bias = root.join("zerobias");
    write_model_dir(&with_bias, &qwen_config(""), Some(0.5));
    write_model_dir(&without_bias, &qwen_config(""), None);
    write_model_dir(&zero_bias, &qwen_config(""), Some(0.0));

    let biased = last_logits(&with_bias);
    let unbiased = last_logits(&without_bias);
    let zeroed = last_logits(&zero_bias);

    // Real bias values must change the output even though the config never
    // said `attention_bias: true`...
    assert!(
        max_abs_diff(&biased, &unbiased) > 1e-3,
        "bias tensors were silently skipped"
    );
    // ...and zero-valued bias tensors must be a numerical no-op, proving the
    // divergence above comes from the bias path and nothing else.
    assert!(
        max_abs_diff(&zeroed, &unbiased) < 1e-6,
        "zero bias must match the biasless checkpoint"
    );
}

#[test]
fn registry_loads_sliding_window_layouts() {
    // Active sliding windows used to be refused; they now map onto the
    // per-layer attention layout (ArchSpec: first `max_window_layers`
    // global, rest sliding — mistral v0.1 all-layer like phi3), so every
    // configuration here must load.
    let root = std::env::temp_dir().join(format!("combs-sliding-guard-{}", std::process::id()));
    let device = burn::tensor::Device::<B>::default();
    let registry = ModelRegistry::<B>::new();

    // Qwen2 with sliding explicitly enabled.
    let qwen_on = root.join("qwen-sliding");
    write_model_dir(
        &qwen_on,
        &qwen_config(r#", "sliding_window": 4096, "use_sliding_window": true"#),
        None,
    );
    let source = SafetensorsSource::load(&qwen_on).unwrap();
    assert!(registry.load(&source, &device).is_ok());

    // Qwen2 with the (usual) disabled window.
    let qwen_off = root.join("qwen-nosliding");
    write_model_dir(
        &qwen_off,
        &qwen_config(r#", "sliding_window": 131072, "use_sliding_window": false"#),
        None,
    );
    let source = SafetensorsSource::load(&qwen_off).unwrap();
    assert!(registry.load(&source, &device).is_ok());

    // Mistral v0.1 style (window, no use_sliding_window key).
    let mistral = root.join("mistral-sliding");
    write_model_dir(
        &mistral,
        &qwen_config(r#", "sliding_window": 4096"#).replace("qwen2", "mistral"),
        None,
    );
    let source = SafetensorsSource::load(&mistral).unwrap();
    assert!(registry.load(&source, &device).is_ok());

    // Mistral v0.3+/Nemo style (no window): plain llama.
    let nemo = root.join("mistral-global");
    write_model_dir(&nemo, &qwen_config("").replace("qwen2", "mistral"), None);
    let source = SafetensorsSource::load(&nemo).unwrap();
    assert!(registry.load(&source, &device).is_ok());
}
