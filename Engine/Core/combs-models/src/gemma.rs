//! Gemma 3 text architecture (`gemma3_text`, `gemma3`) on the
//! [`GenerativeModel`] contract.
//!
//! Differences from the Llama block (all sourced from the HF config —
//! see `AttentionPattern` in combs-formats):
//! - sandwich norms: each sublayer output is normed AGAIN before the
//!   residual add (`post_attention_layernorm`, `post_feedforward_layernorm`),
//!   and all norms are zero-centered (`(1 + w)`, [`gemma_rms_norm`])
//! - QK-norm: per-head RMSNorm on q/k (over `head_dim`) before RoPE
//! - dual RoPE: global layers (every `pattern`-th) use `rope_theta`
//!   (1e6), local layers `rope_local_base_freq` (1e4)
//! - sliding-window attention on local layers (`window`, masked inside
//!   the KV cache — keys stay cached, older-than-window are masked out)
//! - embeddings scaled by `sqrt(hidden_size)`; MLP is gelu(tanh)-gated
//! - attention scale is `1/sqrt(query_pre_attn_scalar)` (defaults to
//!   `head_dim`, matching Llama)
//! - `head_dim` comes from the config explicitly (256 on gemma-3-1b —
//!   NOT `hidden_size / num_attention_heads`)
//!
//! Weight names (HF): `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`,
//! `.self_attn.{q_norm,k_norm}.weight`, `.mlp.{gate,up,down}_proj.weight`,
//! `.{input,post_attention,pre_feedforward,post_feedforward}_layernorm.weight`,
//! `model.embed_tokens.weight`, `model.norm.weight` (lm_head tied on 1b).

use std::ops::Range;

use burn::tensor::{Device, Int, Tensor, backend::Backend};
use combs_formats::{ModelMetadata, ModelSource};

use crate::kv::{CacheConfig, CacheKind, ContiguousKVCache, KVCache, PagedKVCache};
use crate::llama::{load_linear, load_tensor};
use crate::qlinear::Linear;
use crate::matmul::safe_matmul;
use crate::norm::gemma_rms_norm;
use crate::precision::{to_f32, to_float};
use crate::rope::RotaryEmbedding;
use crate::traits::GenerativeModel;
use crate::{ModelError, Result};

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

/// One decoder layer's weights (all projections `[out, in]`, HF layout);
/// projections are [`Linear`] — dense or packed-quant per stored format.
struct GemmaLayer<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    q_norm: Tensor<B, 1>, // [head_dim]
    k_norm: Tensor<B, 1>, // [head_dim]
    gate: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
    input_norm: Tensor<B, 1>,
    post_attn_norm: Tensor<B, 1>,
    pre_ff_norm: Tensor<B, 1>,
    post_ff_norm: Tensor<B, 1>,
}

/// Gemma 3 causal LM.
pub struct GemmaModel<B: Backend> {
    metadata: ModelMetadata,
    embed: Tensor<B, 2>, // [vocab, hidden]; dense for `select` + tied head
    lm_head: Option<Linear<B>>,
    final_norm: Tensor<B, 1>,
    layers: Vec<GemmaLayer<B>>,
    rope_global: RotaryEmbedding<B>,
    rope_local: RotaryEmbedding<B>,
    /// 1 / sqrt(query_pre_attn_scalar)
    scale: f64,
}

/// gelu with the tanh approximation (HF `gelu_pytorch_tanh`):
/// `0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))`.
fn gelu_tanh<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    // x^3 overflows f16 for moderately large activations; compute in f32.
    let out_dtype = x.dtype();
    let x = to_f32(x);
    const C: f64 = 0.797_884_560_802_865_4; // sqrt(2/pi)
    let inner = (x.clone() + x.clone().powf_scalar(3.0).mul_scalar(0.044715)).mul_scalar(C);
    let y = x * inner.tanh().add_scalar(1.0).mul_scalar(0.5);
    to_float(y, out_dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    #[test]
    fn gelu_tanh_matches_known_values() {
        let device = burn::tensor::Device::<NdArray<f32>>::default();
        let x: Tensor<NdArray<f32>, 3> = Tensor::from_data(
            TensorData::new(vec![0.0f32, 1.0, -1.0, 2.0], [1, 1, 4]),
            &device,
        );
        let y = gelu_tanh(x).into_data().to_vec::<f32>().unwrap();
        // Reference: torch.nn.functional.gelu(x, approximate='tanh').
        let expected = [0.0f32, 0.841_191_95, -0.158_808_05, 1.954_597_7];
        for (got, want) in y.iter().zip(expected) {
            assert!((got - want).abs() < 1e-5, "gelu_tanh: got {got}, want {want}");
        }
    }
}

impl<B: Backend> GemmaModel<B> {
    fn forward_hidden(&self, mut x: Tensor<B, 3>, cache: &mut dyn KVCache<B>, pos: usize) -> Tensor<B, 3> {
        let m = &self.metadata;
        let pattern = &m.attention_pattern;
        let [_, seq, _] = x.dims();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let global = pattern.is_global_layer(layer_idx);

            // --- attention block --------------------------------------
            let h = gemma_rms_norm(x.clone(), layer.input_norm.clone(), m.rms_norm_eps);
            let q = layer
                .q
                .forward(h.clone(), None)
                .reshape([1, seq, m.num_attention_heads, m.head_dim])
                .swap_dims(1, 2);
            let k = layer
                .k
                .forward(h.clone(), None)
                .reshape([1, seq, m.num_key_value_heads, m.head_dim])
                .swap_dims(1, 2);
            let v = layer
                .v
                .forward(h, None)
                .reshape([1, seq, m.num_key_value_heads, m.head_dim])
                .swap_dims(1, 2);

            // QK-norm (per head, over head_dim) then layer-type RoPE.
            let q = gemma_rms_norm(q, layer.q_norm.clone(), m.rms_norm_eps);
            let k = gemma_rms_norm(k, layer.k_norm.clone(), m.rms_norm_eps);
            let rope = if global { &self.rope_global } else { &self.rope_local };
            let q = rope.apply(q, pos);
            let k = rope.apply(k, pos);

            let window = if global { None } else { pattern.sliding_window };
            let ctx = cache.attention_opts(layer_idx, q, k, v, pos, self.scale, window);
            let ctx = ctx
                .swap_dims(1, 2)
                .reshape([1, seq, m.num_attention_heads * m.head_dim]);
            let attn_out = layer.o.forward(ctx, None);
            // Sandwich: post-norm on the sublayer output before residual.
            x = x + gemma_rms_norm(attn_out, layer.post_attn_norm.clone(), m.rms_norm_eps);

            // --- MLP block (gelu-tanh gated) ---------------------------
            let h = gemma_rms_norm(x.clone(), layer.pre_ff_norm.clone(), m.rms_norm_eps);
            let gated =
                gelu_tanh(layer.gate.forward(h.clone(), None)) * layer.up.forward(h, None);
            let mlp_out = layer.down.forward(gated, None);
            x = x + gemma_rms_norm(mlp_out, layer.post_ff_norm.clone(), m.rms_norm_eps);
        }

        gemma_rms_norm(x, self.final_norm.clone(), m.rms_norm_eps)
    }

    fn last_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 2> {
        let [_, seq, hidden_size] = hidden.dims();
        let last = hidden.narrow(1, seq - 1, 1); // [1, 1, hidden]
        let logits = match &self.lm_head {
            Some(head) => head.forward(last, None),
            None => {
                let last = last.reshape([1, hidden_size]);
                return safe_matmul(last, self.embed.clone().transpose());
            }
        };
        let [_, _, vocab] = logits.dims();
        logits.reshape([1, vocab])
    }
}

impl<B: Backend> GenerativeModel<B> for GemmaModel<B> {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn load(source: &dyn ModelSource, device: &Device<B>) -> Result<Self> {
        let m = source.metadata().clone();

        let embed: Tensor<B, 2> = load_tensor(source, device, "model.embed_tokens.weight")?;
        expect_shape(
            "model.embed_tokens.weight",
            &embed.dims(),
            &[m.vocab_size, m.hidden_size],
        )?;

        let lm_head = if m.tie_word_embeddings {
            None
        } else {
            // Some checkpoints (e.g. unsloth's gemma-3-1b) tie embeddings
            // without saying so in config.json — fall back to tied when
            // the tensor simply isn't there.
            match load_linear(source, device, "lm_head.weight") {
                Ok(w) => {
                    expect_shape("lm_head.weight", &w.dims(), &[m.vocab_size, m.hidden_size])?;
                    Some(w)
                }
                Err(ModelError::MissingTensor(_)) => None,
                Err(e) => return Err(e),
            }
        };

        let final_norm: Tensor<B, 1> = load_tensor(source, device, "model.norm.weight")?;

        let mut layers = Vec::with_capacity(m.num_hidden_layers);
        for i in 0..m.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let q = load_linear(source, device, &format!("{p}.self_attn.q_proj.weight"))?;
            let k = load_linear(source, device, &format!("{p}.self_attn.k_proj.weight"))?;
            expect_shape(
                &format!("{p}.self_attn.q_proj.weight"),
                &q.dims(),
                &[m.num_attention_heads * m.head_dim, m.hidden_size],
            )?;
            expect_shape(
                &format!("{p}.self_attn.k_proj.weight"),
                &k.dims(),
                &[m.num_key_value_heads * m.head_dim, m.hidden_size],
            )?;
            layers.push(GemmaLayer {
                q,
                k,
                v: load_linear(source, device, &format!("{p}.self_attn.v_proj.weight"))?,
                o: load_linear(source, device, &format!("{p}.self_attn.o_proj.weight"))?,
                q_norm: load_tensor(source, device, &format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: load_tensor(source, device, &format!("{p}.self_attn.k_norm.weight"))?,
                gate: load_linear(source, device, &format!("{p}.mlp.gate_proj.weight"))?,
                up: load_linear(source, device, &format!("{p}.mlp.up_proj.weight"))?,
                down: load_linear(source, device, &format!("{p}.mlp.down_proj.weight"))?,
                input_norm: load_tensor(source, device, &format!("{p}.input_layernorm.weight"))?,
                post_attn_norm: load_tensor(
                    source,
                    device,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?,
                pre_ff_norm: load_tensor(
                    source,
                    device,
                    &format!("{p}.pre_feedforward_layernorm.weight"),
                )?,
                post_ff_norm: load_tensor(
                    source,
                    device,
                    &format!("{p}.post_feedforward_layernorm.weight"),
                )?,
            });
        }

        let max_pos = m.max_position_embeddings;
        let rope_global = RotaryEmbedding::new(m.head_dim, m.rope_theta, max_pos, device);
        let rope_local =
            RotaryEmbedding::new(m.head_dim, m.attention_pattern.rope_local_theta, max_pos, device);
        let scale_div = m
            .attention_pattern
            .query_pre_attn_scalar
            .unwrap_or(m.head_dim as f64);

        Ok(GemmaModel {
            scale: 1.0 / scale_div.sqrt(),
            metadata: m,
            embed,
            lm_head,
            final_norm,
            layers,
            rope_global,
            rope_local,
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
        let x = self
            .embed
            .clone()
            .select(0, flat)
            .reshape([batch, seq, self.metadata.hidden_size]);
        // Gemma scales embeddings by sqrt(hidden_size).
        x.mul_scalar((self.metadata.hidden_size as f64).sqrt())
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
