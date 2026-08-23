//! CPU reference check: NdArray single-shot vs chunked prefill must produce
//! (nearly) identical last-position logits on a long prompt. This isolates
//! model math from wgpu kernel behavior.
//!
//! Ignored by default; run with:
//! `COMBS_TEST_MODEL=/path cargo test -p combs-models --test prefill_cpu --release -- --ignored --nocapture`

use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use combs_formats::{ModelSource, SafetensorsSource};
use combs_models::{CacheConfig, GenerativeModel, LlamaModel};

type B = NdArray<f32>;

fn model_dir() -> String {
    std::env::var("COMBS_TEST_MODEL").unwrap_or_else(|_| "../../../models/SmolLM2-135M".to_string())
}

/// Last-position logits after prefilling `tokens` in chunks of `chunk`
/// (0 = single shot), on the CPU backend with the contiguous cache.
fn cpu_logits(model: &mut LlamaModel<B>, tokens: &[u32], chunk: usize) -> Vec<f32> {
    let device = Default::default();
    let mut cache = model.create_kv_cache(&CacheConfig::contiguous(4096));
    let data: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let embedded = model.embed(Tensor::from_data(
        TensorData::new(data, [1, tokens.len()]),
        &device,
    ));
    let chunk = if chunk == 0 { usize::MAX } else { chunk };
    let mut offset = 0;
    let mut logits = None;
    while offset < tokens.len() {
        let len = chunk.min(tokens.len() - offset);
        let input = embedded.clone().narrow(1, offset, len);
        logits = Some(model.prefill(
            input,
            cache.as_mut(),
            offset as u32..(offset + len) as u32,
        ));
        offset += len;
    }
    logits.unwrap().into_data().to_vec().unwrap()
}

#[test]
#[ignore = "requires a local model directory (COMBS_TEST_MODEL)"]
fn cpu_chunked_matches_cpu_single_shot() {
    let para = "The history of computing spans mechanical calculators, \
                vacuum-tube machines, transistorized mainframes, and modern \
                microprocessors, each generation shrinking cost and size while \
                multiplying speed and memory capacity by orders of magnitude. ";
    let mut prompt = para.repeat(14);
    prompt.push_str("In one sentence, summarize the above:");

    let source = SafetensorsSource::load(model_dir()).expect("load source");
    let tokenizer =
        tokenizers::Tokenizer::from_bytes(source.tokenizer().unwrap().json_bytes().unwrap())
            .unwrap();
    let tokens: Vec<u32> = tokenizer.encode(prompt, false).unwrap().get_ids().to_vec();
    assert!(tokens.len() >= 500);

    let device = Default::default();
    let mut model = LlamaModel::<B>::load(&source as &dyn ModelSource, &device)
        .expect("load model");

    let single = cpu_logits(&mut model, &tokens, 0);
    let chunked = cpu_logits(&mut model, &tokens, 128);

    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    let max_diff = single
        .iter()
        .zip(chunked.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("tokens: {}", tokens.len());
    println!("single argmax: {} (logit {:.4})", argmax(&single), single[argmax(&single)]);
    println!("chunk  argmax: {} (logit {:.4})", argmax(&chunked), chunked[argmax(&chunked)]);
    println!("max |logit diff|: {max_diff:.6}");
    // CPU matmuls reduce in shape-dependent order too, but the drift must
    // stay in f32-noise territory and the greedy pick must agree.
    assert!(max_diff < 0.05, "logit drift too large: {max_diff}");
    assert_eq!(argmax(&single), argmax(&chunked), "greedy token flipped on CPU");
}
