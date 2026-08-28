//! Cross-source parity: the SAME qwen3 checkpoint loaded from an ONNX
//! fp16 export and from the safetensors fp32 original must agree —
//! greedy tokens identical, logits within fp16-storage noise. This is
//! the gate that proves the ONNX container path (names, transposes,
//! dtypes, tokenizer siblings) feeds the llama family correctly.
//!
//! Run:
//! ```sh
//! cargo test -p combs-models --test onnx_forward --release -- --ignored --nocapture
//! ```
//! (uses $HOME/.cache/combs/models/{qwen3-0.6b,qwen3-0.6b-onnx}, or
//! COMBS_TEST_QWEN3 / COMBS_TEST_ONNX.)

use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use combs_formats::{open_model_source, ModelSource, SafetensorsSource};
use combs_models::{CacheConfig, GenerativeModel, KVCache, LlamaModel};

type B = NdArray<f32>;

fn home(rel: &str) -> String {
    format!("{}/{rel}", std::env::var("HOME").expect("HOME"))
}

fn greedy(
    model: &mut LlamaModel<B>,
    cache: &mut dyn KVCache<B>,
    prompt: &[u32],
    steps: usize,
) -> (Vec<u32>, Vec<f32>) {
    let device = Default::default();
    let embed = |model: &LlamaModel<B>, toks: &[u32]| {
        let data: Vec<i32> = toks.iter().map(|&t| t as i32).collect();
        model.embed(Tensor::<B, 2, Int>::from_data(
            TensorData::new(data, [1, toks.len()]),
            &device,
        ))
    };
    let logits = model.prefill(embed(model, prompt), cache, 0..prompt.len() as u32);
    let first: Vec<f32> = logits.clone().into_data().to_vec().unwrap();
    let mut out = Vec::new();
    let mut next = argmax(&first) as u32;
    out.push(next);
    for _ in 1..steps {
        let logits = model.decode(embed(model, &[next]), cache);
        let row: Vec<f32> = logits.into_data().to_vec().unwrap();
        next = argmax(&row) as u32;
        out.push(next);
    }
    (out, first)
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

#[test]
#[ignore = "requires the qwen3-0.6b safetensors + onnx checkouts"]
fn onnx_fp16_matches_the_safetensors_original() {
    let onnx_path = std::env::var("COMBS_TEST_ONNX")
        .unwrap_or_else(|_| home(".cache/combs/models/qwen3-0.6b-onnx/onnx/model_fp16.onnx"));
    let st_path = std::env::var("COMBS_TEST_QWEN3")
        .unwrap_or_else(|_| home(".cache/combs/models/qwen3-0.6b"));
    if !std::path::Path::new(&onnx_path).exists() {
        eprintln!("skipping: {onnx_path} not present");
        return;
    }

    let device = Default::default();
    let onnx_src = open_model_source(&onnx_path).expect("onnx source");
    let st_src = SafetensorsSource::load(&st_path).expect("safetensors source");

    // Same tokenizer text → same ids from both sources' tokenizers.
    let prompt = "The capital of France is Paris, and the capital of Italy is";
    let tok_a = tokenizers::Tokenizer::from_bytes(
        onnx_src.tokenizer().unwrap().json_bytes().unwrap(),
    )
    .unwrap();
    let tok_b = tokenizers::Tokenizer::from_bytes(
        st_src.tokenizer().unwrap().json_bytes().unwrap(),
    )
    .unwrap();
    let ids_a = tok_a.encode(prompt, false).unwrap().get_ids().to_vec();
    let ids_b = tok_b.encode(prompt, false).unwrap().get_ids().to_vec();
    assert_eq!(ids_a, ids_b, "sibling tokenizers must agree");

    let mut m_onnx =
        LlamaModel::<B>::load(onnx_src.as_ref(), &device).expect("load onnx");
    let mut m_st =
        LlamaModel::<B>::load(&st_src as &dyn ModelSource, &device).expect("load safetensors");

    let mut c_onnx = m_onnx.create_kv_cache(&CacheConfig::contiguous(512));
    let mut c_st = m_st.create_kv_cache(&CacheConfig::contiguous(512));
    let steps = 16;
    let (toks_onnx, logits_onnx) = greedy(&mut m_onnx, c_onnx.as_mut(), &ids_a, steps);
    let (toks_st, logits_st) = greedy(&mut m_st, c_st.as_mut(), &ids_b, steps);

    let max_diff = logits_onnx
        .iter()
        .zip(&logits_st)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("[onnx-forward] first-position max |logit diff| {max_diff:.4}");
    println!("[onnx-forward] onnx tokens: {toks_onnx:?}");
    println!("[onnx-forward] st   tokens: {toks_st:?}");
    // fp16 weight storage vs fp32: the same drift class the chunked
    // prefill bar (0.05) covers; greedy tokens must agree exactly.
    assert!(max_diff < 0.25, "logit drift beyond fp16 storage noise: {max_diff}");
    assert_eq!(toks_onnx, toks_st, "greedy tokens diverged");
}
