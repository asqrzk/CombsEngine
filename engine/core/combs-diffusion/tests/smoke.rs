//! Smoke test for the Burn-native diffusion scaffold.

use combs_diffusion::{DiffusionModel, PromptEmbed, StableDiffusionPipeline};
use combs_formats::{AttentionPattern, ModelMetadata, ModelSource, Result, SamplerConfig, TensorReader, TokenizerSpec};

/// Trivial in-memory model source so `StableDiffusionPipeline::load` can run
/// without a real model directory.
struct EmptySource;

impl ModelSource for EmptySource {
    fn metadata(&self) -> &ModelMetadata {
        // A static dummy metadata is enough for the scaffold load path.
        static META: std::sync::OnceLock<ModelMetadata> = std::sync::OnceLock::new();
        META.get_or_init(|| ModelMetadata {
            architecture: "stable-diffusion".to_string(),
            hidden_size: 0,
            intermediate_size: 0,
            num_hidden_layers: 0,
            num_attention_heads: 0,
            num_key_value_heads: 0,
            vocab_size: 0,
            max_position_embeddings: 0,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            tie_word_embeddings: false,
            head_dim: 0,
            attention_bias: false,
            bos_token_id: None,
            eos_token_ids: Vec::new(),
            vision: None,
            attention_pattern: AttentionPattern::default(),
        })
    }

    fn tensor_names(&self) -> Vec<String> {
        Vec::new()
    }

    fn open_tensor(&self, _name: &str) -> Result<TensorReader<'_>> {
        unimplemented!("scaffold does not load weights")
    }

    fn tokenizer(&self) -> Result<TokenizerSpec> {
        Err(combs_formats::FormatError::MissingFile(
            "tokenizer.json".to_string(),
        ))
    }

    fn sampler_defaults(&self) -> Option<SamplerConfig> {
        None
    }
}

#[test]
fn diffusion_pipeline_runs() {
    // Use ndarray backend so the test is CPU-only and fast.
    type B = burn::backend::NdArray;
    let device = Default::default();
    let source = EmptySource;
    let mut pipeline = StableDiffusionPipeline::<B>::load(&source, &device).unwrap();

    let embed = PromptEmbed {
        positive: burn::tensor::Tensor::zeros([1, 77, 768], &device),
        negative: None,
    };

    let (image, seed) = pipeline
        .generate(embed, 64, 64, 5, 7.5, Some(42), combs_diffusion::SchedulerKind::default())
        .unwrap();
    assert_eq!(seed, 42);

    let [batch, channels, height, width] = image.dims();
    assert_eq!(batch, 1);
    assert_eq!(channels, 3);
    assert_eq!(height, 64);
    assert_eq!(width, 64);
}
