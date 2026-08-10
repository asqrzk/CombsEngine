//! Larger, slower diffusion tests. Run with `COMBS_DIFFUSION_INTEGRATION=1`.

use combs_diffusion::{DiffusionModel, PromptEmbed, StableDiffusionPipeline};
use combs_formats::{AttentionPattern, ModelMetadata, ModelSource, Result, SamplerConfig, TensorReader, TokenizerSpec};

struct EmptySource;

impl ModelSource for EmptySource {
    fn metadata(&self) -> &ModelMetadata {
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
        unimplemented!("integration test does not load weights")
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
fn pipeline_512x512_runs() {
    if std::env::var("COMBS_DIFFUSION_INTEGRATION").is_err() {
        return;
    }

    type B = burn::backend::NdArray;
    let device = Default::default();
    let source = EmptySource;
    let mut pipeline = StableDiffusionPipeline::<B>::load(&source, &device).unwrap();

    let embed = PromptEmbed {
        positive: burn::tensor::Tensor::zeros([1, 77, 768], &device),
        negative: None,
    };

    let (image, _seed) = pipeline
        .generate(embed, 512, 512, 5, 7.5, Some(42), combs_diffusion::SchedulerKind::default())
        .unwrap();
    assert_eq!(image.dims(), [1, 3, 512, 512]);
}
