//! CLIP text encoder for Stable Diffusion 1.5 conditioning.
//!
//! Implements the text branch of OpenAI CLIP ViT-L/14 as used by SD 1.5:
//!   vocab_size = 49408, hidden_size = 768, intermediate_size = 3072,
//!   num_layers = 12, num_heads = 12, max_position_embeddings = 77,
//!   hidden_act = quick_gelu, layer_norm_eps = 1e-5.

use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear};
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, ElementConversion, Int, Tensor, TensorData};

use crate::weights::{load_embedding, load_layer_norm, load_linear};
use combs_formats::{ModelSource, Result};

const VOCAB_SIZE: usize = 49408;
const HIDDEN_SIZE: usize = 768;
const INTERMEDIATE_SIZE: usize = 3072;
const NUM_LAYERS: usize = 12;
const NUM_HEADS: usize = 12;
const MAX_POSITION_EMBEDDINGS: usize = 77;
const LAYER_NORM_EPS: f64 = 1e-5;

/// `quick_gelu(x) = x * sigmoid(1.702 * x)`, the activation used by CLIP.
fn quick_gelu<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    x.clone().mul(sigmoid(x.mul_scalar(1.702f32)))
}

/// Causal mask for CLIP self-attention: lower-triangular attend, upper-triangular
/// blocked with a large negative value.
fn causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 4> {
    // burn's triangle masks are complements: `tril_mask(offset 0)` is FALSE
    // on the lower triangle + diagonal and TRUE strictly above it — exactly
    // the future positions to block. (`triu_mask(1)` is the inverse and
    // silently produced an ANTI-causal mask — every token attended only to
    // its future; caught by the torch parity harness.)
    let mask: Tensor<B, 2, Bool> =
        Tensor::tril_mask([seq_len, seq_len], 0, device);
    Tensor::<B, 4>::zeros([1, 1, seq_len, seq_len], device)
        .mask_fill(mask.reshape([1, 1, seq_len, seq_len]), -1e9f32)
}

/// Token + learned positional embeddings.
pub struct ClipTextEmbeddings<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
}

impl<B: Backend> ClipTextEmbeddings<B> {
    pub fn new(device: &B::Device) -> Self {
        let token_embedding = EmbeddingConfig::new(VOCAB_SIZE, HIDDEN_SIZE)
            .init(device);
        let position_embedding =
            EmbeddingConfig::new(MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE).init(device);
        Self {
            token_embedding,
            position_embedding,
        }
    }

    pub fn load_from(source: &dyn ModelSource, device: &B::Device) -> Result<Self> {
        let token_embedding = load_embedding(
            source,
            "text_model.embeddings.token_embedding",
            VOCAB_SIZE,
            HIDDEN_SIZE,
            device,
        )?;
        let position_embedding = load_embedding(
            source,
            "text_model.embeddings.position_embedding",
            MAX_POSITION_EMBEDDINGS,
            HIDDEN_SIZE,
            device,
        )?;
        Ok(Self {
            token_embedding,
            position_embedding,
        })
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_batch, seq_len] = input_ids.dims();
        let token_embeds = self.token_embedding.forward(input_ids);

        let position_ids =
            Tensor::<B, 1, Int>::arange(0..seq_len as i64, &token_embeds.device());
        let position_ids = position_ids.reshape([1, seq_len]);
        let position_embeds = self.position_embedding.forward(position_ids);

        token_embeds + position_embeds
    }
}

/// Multi-head self-attention with bias on all projections.
pub struct ClipAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl<B: Backend> ClipAttention<B> {
    pub fn new(device: &B::Device) -> Self {
        let num_heads = NUM_HEADS;
        let head_dim = HIDDEN_SIZE / num_heads;
        Self {
            q_proj: burn::nn::LinearConfig::new(HIDDEN_SIZE, HIDDEN_SIZE)
                .with_bias(true)
                .init(device),
            k_proj: burn::nn::LinearConfig::new(HIDDEN_SIZE, HIDDEN_SIZE)
                .with_bias(true)
                .init(device),
            v_proj: burn::nn::LinearConfig::new(HIDDEN_SIZE, HIDDEN_SIZE)
                .with_bias(true)
                .init(device),
            out_proj: burn::nn::LinearConfig::new(HIDDEN_SIZE, HIDDEN_SIZE)
                .with_bias(true)
                .init(device),
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    pub fn load_from(source: &dyn ModelSource, prefix: &str, device: &B::Device) -> Result<Self> {
        let num_heads = NUM_HEADS;
        let head_dim = HIDDEN_SIZE / num_heads;
        let q_proj = load_linear(
            source,
            &format!("{prefix}.q_proj"),
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            true,
            device,
        )?;
        let k_proj = load_linear(
            source,
            &format!("{prefix}.k_proj"),
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            true,
            device,
        )?;
        let v_proj = load_linear(
            source,
            &format!("{prefix}.v_proj"),
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            true,
            device,
        )?;
        let out_proj = load_linear(
            source,
            &format!("{prefix}.out_proj"),
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            true,
            device,
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: &Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _] = hidden_states.dims();

        let q = self.q_proj.forward(hidden_states.clone());
        let k = self.k_proj.forward(hidden_states.clone());
        let v = self.v_proj.forward(hidden_states);

        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let k = k
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let v = v
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .permute([0, 2, 1, 3]);

        let scores = q
            .matmul(k.permute([0, 1, 3, 2]))
            .mul_scalar(self.scale)
            .add(attention_mask.clone());
        let attn_weights = softmax(scores, 3);
        let out = attn_weights.matmul(v);

        let out = out
            .permute([0, 2, 1, 3])
            .reshape([batch, seq_len, HIDDEN_SIZE]);
        self.out_proj.forward(out)
    }
}

/// CLIP MLP: fc1 -> quick_gelu -> fc2.
pub struct ClipMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> ClipMlp<B> {
    pub fn new(device: &B::Device) -> Self {
        let fc1 = burn::nn::LinearConfig::new(HIDDEN_SIZE, INTERMEDIATE_SIZE)
            .with_bias(true)
            .init(device);
        let fc2 = burn::nn::LinearConfig::new(INTERMEDIATE_SIZE, HIDDEN_SIZE)
            .with_bias(true)
            .init(device);
        Self { fc1, fc2 }
    }

    pub fn load_from(source: &dyn ModelSource, prefix: &str, device: &B::Device) -> Result<Self> {
        let fc1 = load_linear(
            source,
            &format!("{prefix}.fc1"),
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            true,
            device,
        )?;
        let fc2 = load_linear(
            source,
            &format!("{prefix}.fc2"),
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            true,
            device,
        )?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.fc2.forward(quick_gelu(self.fc1.forward(x)))
    }
}

/// One transformer encoder layer: pre-LN self-attention + pre-LN MLP.
pub struct ClipEncoderLayer<B: Backend> {
    layer_norm1: LayerNorm<B>,
    self_attn: ClipAttention<B>,
    layer_norm2: LayerNorm<B>,
    mlp: ClipMlp<B>,
}

impl<B: Backend> ClipEncoderLayer<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            layer_norm1: LayerNormConfig::new(HIDDEN_SIZE)
                .with_epsilon(LAYER_NORM_EPS)
                .init(device),
            self_attn: ClipAttention::new(device),
            layer_norm2: LayerNormConfig::new(HIDDEN_SIZE)
                .with_epsilon(LAYER_NORM_EPS)
                .init(device),
            mlp: ClipMlp::new(device),
        }
    }

    pub fn load_from(source: &dyn ModelSource, prefix: &str, device: &B::Device) -> Result<Self> {
        let layer_norm1 = load_layer_norm(
            source,
            &format!("{prefix}.layer_norm1"),
            HIDDEN_SIZE,
            device,
        )?;
        let self_attn = ClipAttention::load_from(
            source,
            &format!("{prefix}.self_attn"),
            device,
        )?;
        let layer_norm2 = load_layer_norm(
            source,
            &format!("{prefix}.layer_norm2"),
            HIDDEN_SIZE,
            device,
        )?;
        let mlp = ClipMlp::load_from(source, &format!("{prefix}.mlp"), device)?;
        Ok(Self {
            layer_norm1,
            self_attn,
            layer_norm2,
            mlp,
        })
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        attention_mask: &Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let residual = x.clone();
        let x = self.layer_norm1.forward(x);
        let x = residual.add(self.self_attn.forward(x, attention_mask));

        let residual = x.clone();
        let x = self.layer_norm2.forward(x);
        let x = residual.add(self.mlp.forward(x));
        x
    }
}

/// Full CLIP text transformer.
pub struct ClipTextModel<B: Backend> {
    embeddings: ClipTextEmbeddings<B>,
    encoder: Vec<ClipEncoderLayer<B>>,
    final_layer_norm: LayerNorm<B>,
}

impl<B: Backend> ClipTextModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let mut encoder = Vec::with_capacity(NUM_LAYERS);
        for _ in 0..NUM_LAYERS {
            encoder.push(ClipEncoderLayer::new(device));
        }
        Self {
            embeddings: ClipTextEmbeddings::new(device),
            encoder,
            final_layer_norm: LayerNormConfig::new(HIDDEN_SIZE)
                .with_epsilon(LAYER_NORM_EPS)
                .init(device),
        }
    }

    pub fn load_from(source: &dyn ModelSource, device: &B::Device) -> Result<Self> {
        let mut encoder = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            encoder.push(ClipEncoderLayer::load_from(
                source,
                &format!("text_model.encoder.layers.{i}"),
                device,
            )?);
        }
        Ok(Self {
            embeddings: ClipTextEmbeddings::load_from(source, device)?,
            encoder,
            final_layer_norm: load_layer_norm(
                source,
                "text_model.final_layer_norm",
                HIDDEN_SIZE,
                device,
            )?,
        })
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_, seq_len] = input_ids.dims();
        let mut hidden = self.embeddings.forward(input_ids);
        let mask = causal_mask(seq_len, &hidden.device());
        for layer in &self.encoder {
            hidden = layer.forward(hidden, &mask);
        }
        self.final_layer_norm.forward(hidden)
    }
}

/// Convert token ids into the `Int` tensor the embedding layer expects.
pub fn input_ids_to_tensor<B: Backend>(
    ids: &[u32],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let data: Vec<B::IntElem> = ids
        .iter()
        .map(|&id| (id as i64).elem::<B::IntElem>())
        .collect();
    let data = TensorData::new(data, [1, ids.len()]);
    Tensor::<B, 2, Int>::from_data(data, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_text_model_runs() {
        type B = burn::backend::NdArray;
        let device = Default::default();
        let model = ClipTextModel::<B>::new(&device);

        // Random token ids; shape must be [batch, seq_len].
        let input_ids = input_ids_to_tensor::<B>(&[0, 1, 2, 3, 4], &device);
        let out = model.forward(input_ids);
        assert_eq!(out.dims(), [1, 5, HIDDEN_SIZE]);
    }

    #[test]
    fn causal_mask_blocks_future() {
        type B = burn::backend::NdArray;
        let device = Default::default();
        let mask = causal_mask::<B>(3, &device);
        assert_eq!(mask.dims(), [1, 1, 3, 3]);
        // VALUES, not just shape (the original dims-only assertion let an
        // anti-causal mask ship): row i must be 0 for j <= i and blocked
        // for j > i.
        let v = mask.into_data().to_vec::<f32>().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let x = v[i * 3 + j];
                if j <= i {
                    assert_eq!(x, 0.0, "past ({i},{j}) must be visible");
                } else {
                    assert!(x <= -1e8, "future ({i},{j}) must be blocked, got {x}");
                }
            }
        }
    }
}
