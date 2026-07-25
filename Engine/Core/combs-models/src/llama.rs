//! Llama-family architecture (Llama, SmolLM2, …) on the [`GenerativeModel`]
//! contract.
//!
//! Expects HuggingFace weight names:
//! `model.embed_tokens.weight`,
//! `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`,
//! `model.layers.{i}.mlp.{gate,up,down}_proj.weight`,
//! `model.layers.{i}.{input,post_attention}_layernorm.weight`,
//! `model.norm.weight`, `lm_head.weight` (optional when tied).
//!
//! Weights are hand-rolled matmuls (`y = x @ W^T`) rather than burn `nn`
//! modules so they can be streamed straight from a [`ModelSource`] without
//! record files. The family has no biases; the loader tolerates optional
//! `*_proj.bias` tensors for related checkpoints that carry them.

use std::ops::Range;

use burn::tensor::{Device, Int, Tensor, backend::Backend};
use combs_formats::{ModelMetadata, ModelSource};

use crate::kv::{CacheConfig, CacheKind, ContiguousKVCache, KVCache, PagedKVCache};
use crate::matmul::safe_matmul;
use crate::norm::rms_norm;
use crate::rope::RotaryEmbedding;
use crate::traits::GenerativeModel;
use crate::{ModelError, Result};

/// One decoder layer's weights. All projections are `[out, in]` (HF layout).
struct LlamaLayer<B: Backend> {
    q: Tensor<B, 2>,
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    o: Tensor<B, 2>,
    q_bias: Option<Tensor<B, 1>>,
    k_bias: Option<Tensor<B, 1>>,
    v_bias: Option<Tensor<B, 1>>,
    o_bias: Option<Tensor<B, 1>>,
    gate: Tensor<B, 2>,
    up: Tensor<B, 2>,
    down: Tensor<B, 2>,
    input_norm: Tensor<B, 1>,
    post_norm: Tensor<B, 1>,
}

/// Llama-family causal LM.
pub struct LlamaModel<B: Backend> {
    metadata: ModelMetadata,
    embed: Tensor<B, 2>, // [vocab, hidden]
    lm_head: Option<Tensor<B, 2>>, // None => tied to `embed`
    final_norm: Tensor<B, 1>,
    layers: Vec<LlamaLayer<B>>,
    rotary: RotaryEmbedding<B>,
    /// 1 / sqrt(head_dim)
    scale: f64,
}

/// `y = x @ W^T (+ b)` for `[batch, seq, in] @ [out, in]`.
fn linear<B: Backend>(
    x: Tensor<B, 3>,
    w: &Tensor<B, 2>,
    bias: Option<&Tensor<B, 1>>,
) -> Tensor<B, 3> {
    // matmul is same-rank only, so the weight is batch-unsqueezed.
    // `safe_matmul`: at seq >= 512 with in >= 512 this shape enters the
    // broken wgpu/Metal matmul region (see combs_models::matmul docs).
    let out = safe_matmul(x, w.clone().transpose().unsqueeze_dim::<3>(0));
    match bias {
        Some(b) => {
            let [batch, seq, dim] = out.dims();
            out + b.clone().reshape([1, 1, dim]).expand([batch, seq, dim])
        }
        None => out,
    }
}

fn load_weight<B: Backend, const D: usize>(
    source: &dyn ModelSource,
    device: &Device<B>,
    name: &str,
) -> Result<Tensor<B, D>> {
    source
        .open_tensor(name)
        .map_err(|e| match e {
            combs_formats::FormatError::TensorNotFound(_) => {
                ModelError::MissingTensor(name.to_string())
            }
            other => ModelError::Format(other),
        })?
        .load_to_tensor::<B, D>(device)
        .map_err(ModelError::Format)
}

fn load_optional_bias<B: Backend>(
    source: &dyn ModelSource,
    device: &Device<B>,
    name: &str,
) -> Result<Option<Tensor<B, 1>>> {
    match source.open_tensor(name) {
        Ok(reader) => Ok(Some(
            reader.load_to_tensor::<B, 1>(device).map_err(ModelError::Format)?,
        )),
        Err(combs_formats::FormatError::TensorNotFound(_)) => Ok(None),
        Err(e) => Err(ModelError::Format(e)),
    }
}

impl<B: Backend> LlamaModel<B> {
    fn expect_shape(name: &str, got: &[usize], expected: &[usize]) -> Result<()> {
        if got == expected {
            Ok(())
        } else {
            Err(ModelError::BadShape {
                tensor: name.to_string(),
                expected: expected.to_vec(),
                got: got.to_vec(),
            })
        }
    }

    /// Shared trunk for prefill and decode: embeddings in, final-normed
    /// hidden states out. `pos` is the absolute position of the first input
    /// token.
    fn forward_hidden(
        &self,
        mut x: Tensor<B, 3>,
        cache: &mut dyn KVCache<B>,
        pos: usize,
    ) -> Tensor<B, 3> {
        let m = &self.metadata;
        let [_, seq, _] = x.dims();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // --- attention block ------------------------------------------------
            let h = rms_norm(x.clone(), layer.input_norm.clone(), m.rms_norm_eps);
            let q = linear(h.clone(), &layer.q, layer.q_bias.as_ref());
            let k = linear(h.clone(), &layer.k, layer.k_bias.as_ref());
            let v = linear(h, &layer.v, layer.v_bias.as_ref());

            let q = q
                .reshape([1, seq, m.num_attention_heads, m.head_dim])
                .swap_dims(1, 2);
            let k = k
                .reshape([1, seq, m.num_key_value_heads, m.head_dim])
                .swap_dims(1, 2);
            let v = v
                .reshape([1, seq, m.num_key_value_heads, m.head_dim])
                .swap_dims(1, 2);

            let q = self.rotary.apply(q, pos);
            let k = self.rotary.apply(k, pos);

            // The cache owns K/V layout, GQA expansion and causal masking.
            let ctx = cache.attention(layer_idx, q, k, v, pos, self.scale);
            let ctx = ctx
                .swap_dims(1, 2)
                .reshape([1, seq, m.num_attention_heads * m.head_dim]);
            let attn_out = linear(ctx, &layer.o, layer.o_bias.as_ref());
            x = x + attn_out;

            // --- MLP block (SwiGLU) ---------------------------------------------
            let h = rms_norm(x.clone(), layer.post_norm.clone(), m.rms_norm_eps);
            let gated = burn::tensor::activation::silu(linear(h.clone(), &layer.gate, None))
                * linear(h.clone(), &layer.up, None);
            let mlp_out = linear(gated, &layer.down, None);
            x = x + mlp_out;
        }

        rms_norm(x, self.final_norm.clone(), m.rms_norm_eps)
    }

    /// Logits of the last sequence position: `[1, hidden] -> [1, vocab]`.
    fn last_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 2> {
        let [_, seq, hidden_size] = hidden.dims();
        let last = hidden.narrow(1, seq - 1, 1).reshape([1, hidden_size]);
        let w = self.lm_head.as_ref().unwrap_or(&self.embed);
        safe_matmul(last, w.clone().transpose())
    }
}

impl<B: Backend> GenerativeModel<B> for LlamaModel<B> {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn load(source: &dyn ModelSource, device: &Device<B>) -> Result<Self> {
        let m = source.metadata().clone();

        let embed: Tensor<B, 2> = load_weight(source, device, "model.embed_tokens.weight")?;
        Self::expect_shape(
            "model.embed_tokens.weight",
            &embed.dims(),
            &[m.vocab_size, m.hidden_size],
        )?;

        let lm_head = if m.tie_word_embeddings {
            None
        } else {
            let w: Tensor<B, 2> = load_weight(source, device, "lm_head.weight")?;
            Self::expect_shape("lm_head.weight", &w.dims(), &[m.vocab_size, m.hidden_size])?;
            Some(w)
        };

        let final_norm: Tensor<B, 1> = load_weight(source, device, "model.norm.weight")?;

        let mut layers = Vec::with_capacity(m.num_hidden_layers);
        for i in 0..m.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let q: Tensor<B, 2> =
                load_weight(source, device, &format!("{p}.self_attn.q_proj.weight"))?;
            let k: Tensor<B, 2> =
                load_weight(source, device, &format!("{p}.self_attn.k_proj.weight"))?;
            let v: Tensor<B, 2> =
                load_weight(source, device, &format!("{p}.self_attn.v_proj.weight"))?;
            let o: Tensor<B, 2> =
                load_weight(source, device, &format!("{p}.self_attn.o_proj.weight"))?;
            Self::expect_shape(
                &format!("{p}.self_attn.q_proj.weight"),
                &q.dims(),
                &[m.num_attention_heads * m.head_dim, m.hidden_size],
            )?;
            Self::expect_shape(
                &format!("{p}.self_attn.k_proj.weight"),
                &k.dims(),
                &[m.num_key_value_heads * m.head_dim, m.hidden_size],
            )?;

            // Biases are absent in this model family but tolerated for
            // related checkpoints (e.g. some Qwen releases).
            let bias = |proj: &str| -> Result<Option<Tensor<B, 1>>> {
                if m.attention_bias || proj.starts_with("mlp") {
                    load_optional_bias(
                        source,
                        device,
                        &format!("{p}.{proj}.bias"),
                    )
                } else {
                    Ok(None)
                }
            };

            layers.push(LlamaLayer {
                q,
                k,
                v,
                o,
                q_bias: bias("self_attn.q_proj")?,
                k_bias: bias("self_attn.k_proj")?,
                v_bias: bias("self_attn.v_proj")?,
                o_bias: bias("self_attn.o_proj")?,
                gate: load_weight(source, device, &format!("{p}.mlp.gate_proj.weight"))?,
                up: load_weight(source, device, &format!("{p}.mlp.up_proj.weight"))?,
                down: load_weight(source, device, &format!("{p}.mlp.down_proj.weight"))?,
                input_norm: load_weight(source, device, &format!("{p}.input_layernorm.weight"))?,
                post_norm: load_weight(
                    source,
                    device,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?,
            });
        }

        let rotary =
            RotaryEmbedding::new(m.head_dim, m.rope_theta, m.max_position_embeddings, device);

        Ok(LlamaModel {
            scale: 1.0 / (m.head_dim as f64).sqrt(),
            metadata: m,
            embed,
            lm_head,
            final_norm,
            layers,
            rotary,
        })
    }

    fn create_kv_cache(&self, config: &CacheConfig) -> Box<dyn KVCache<B>> {
        match config.kind {
            CacheKind::Contiguous => {
                Box::new(ContiguousKVCache::<B>::new(self.metadata.num_hidden_layers))
            }
            CacheKind::Paged => Box::new(PagedKVCache::<B>::new(
                self.metadata.num_hidden_layers,
                *config,
            )),
        }
    }

    fn embed(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq] = tokens.dims();
        let flat = tokens.reshape([batch * seq]);
        self.embed
            .clone()
            .select(0, flat)
            .reshape([batch, seq, self.metadata.hidden_size])
    }

    fn prefill(
        &mut self,
        input: Tensor<B, 3>,
        cache: &mut dyn KVCache<B>,
        pos: Range<u32>,
    ) -> Tensor<B, 2> {
        let [_, seq, _] = input.dims();
        assert_eq!(
            seq,
            (pos.end - pos.start) as usize,
            "prefill pos range must match the input sequence length"
        );
        let hidden = self.forward_hidden(input, cache, pos.start as usize);
        self.last_logits(hidden)
    }

    fn decode(&mut self, input: Tensor<B, 3>, cache: &mut dyn KVCache<B>) -> Tensor<B, 2> {
        let pos = cache.seq_len();
        let hidden = self.forward_hidden(input, cache, pos);
        self.last_logits(hidden)
    }
}
