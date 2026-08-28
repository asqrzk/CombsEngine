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

/// The int4 export: (a) on REAL tensors, the Q4_0 repack must
/// dequantize bit-identically to the reference MatMulNBits semantics
/// (f16 scales make the whole path lossless); (b) the model loads and
/// greedy-decodes coherently against the fp16 export — Q4 rounding
/// legitimately moves logits, so tokens are REPORTED with a
/// first-token gate rather than promised identical.
#[test]
#[ignore = "requires the qwen3-0.6b onnx checkouts"]
fn onnx_q4f16_repack_and_decode() {
    let q4_path = std::env::var("COMBS_TEST_ONNX_Q4")
        .unwrap_or_else(|_| home(".cache/combs/models/qwen3-0.6b-onnx/onnx/model_q4f16.onnx"));
    let fp16_path = std::env::var("COMBS_TEST_ONNX")
        .unwrap_or_else(|_| home(".cache/combs/models/qwen3-0.6b-onnx/onnx/model_fp16.onnx"));
    if !std::path::Path::new(&q4_path).exists() {
        eprintln!("skipping: {q4_path} not present");
        return;
    }
    let device = Default::default();
    let q4_src = open_model_source(&q4_path).expect("q4 source");

    // (a) repacked kernel stream ≡ dequant fallback, on real weights.
    let mut checked = 0;
    for name in [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.13.mlp.down_proj.weight",
        "model.layers.27.self_attn.o_proj.weight",
    ] {
        let Some(qt) = q4_src.open_tensor_quant(name).expect("quant open") else {
            panic!("{name} should serve a Q4_0 stream");
        };
        let n: usize = qt.shape[0];
        let k: usize = qt.shape[1];
        let via_kernel_stream =
            combs_formats::quants::dequantize_q4_0(&qt.data, n * k).expect("q4_0 dequant");
        let dense: Vec<f32> = q4_src
            .open_tensor(name)
            .expect("dense fallback")
            .load_data()
            .expect("load")
            .to_vec()
            .expect("f32 vec");
        let worst = via_kernel_stream
            .iter()
            .zip(&dense)
            .map(|(a, b): (&f32, &f32)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(worst, 0.0, "{name}: repack path diverged from reference dequant");
        checked += 1;
    }
    println!("[onnx-q4] {checked} real tensors: repack ≡ dequant, bit-exact");

    // (b) coherence vs the fp16 export.
    let fp16_src = open_model_source(&fp16_path).expect("fp16 source");
    let tok = tokenizers::Tokenizer::from_bytes(
        q4_src.tokenizer().unwrap().json_bytes().unwrap(),
    )
    .unwrap();
    let prompt = "The capital of France is Paris, and the capital of Italy is";
    let ids = tok.encode(prompt, false).unwrap().get_ids().to_vec();

    let mut m_q4 = LlamaModel::<B>::load(q4_src.as_ref(), &device).expect("load q4");
    let mut c_q4 = m_q4.create_kv_cache(&CacheConfig::contiguous(512));
    let (toks_q4, logits_q4) = greedy(&mut m_q4, c_q4.as_mut(), &ids, 12);
    drop(m_q4);
    let mut m_fp = LlamaModel::<B>::load(fp16_src.as_ref(), &device).expect("load fp16");
    let mut c_fp = m_fp.create_kv_cache(&CacheConfig::contiguous(512));
    let (toks_fp, logits_fp) = greedy(&mut m_fp, c_fp.as_mut(), &ids, 12);

    let max_diff = logits_q4
        .iter()
        .zip(&logits_fp)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let agree = toks_q4.iter().zip(&toks_fp).take_while(|(a, b)| a == b).count();
    println!("[onnx-q4] first-position max |logit diff| {max_diff:.3} (Q4 rounding)");
    println!("[onnx-q4] q4   tokens: {toks_q4:?}");
    println!("[onnx-q4] fp16 tokens: {toks_fp:?}");
    println!("[onnx-q4] greedy agreement: {agree}/12");
    assert_eq!(
        argmax(&logits_q4),
        argmax(&logits_fp),
        "first greedy token flipped under Q4"
    );
}
