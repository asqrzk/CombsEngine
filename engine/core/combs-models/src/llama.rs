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

use crate::archspec::{ArchSpec, LayerKind, NormFlavor};
use crate::kv::{CacheConfig, CacheKind, ContiguousKVCache, KVCache, PagedKVCache};
use crate::matmul::safe_matmul;
use crate::norm::{gemma_rms_norm, rms_norm};
use crate::precision::{to_f32, to_float};
use crate::qlinear::{Linear, try_quant_linear};
use crate::rope::RotaryEmbedding;
use crate::traits::GenerativeModel;
use crate::{ModelError, Result};

/// One decoder layer's weights. All projections are `[out, in]` (HF layout);
/// each is a [`Linear`] — dense tensor or packed-quant kernel dispatch,
/// decided per tensor at load from the source's stored format.
///
/// The optional norms cover the family's variants: `attn_out_norm` /
/// `mlp_out_norm` are gemma's sandwich norms, `q_norm`/`k_norm` the
/// per-head QK-norms (gemma-3, qwen-3). For plain llama they're all `None`
/// and the forward reduces exactly to the classic block.
struct LlamaLayer<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    q_bias: Option<Tensor<B, 1>>,
    k_bias: Option<Tensor<B, 1>>,
    v_bias: Option<Tensor<B, 1>>,
    o_bias: Option<Tensor<B, 1>>,
    q_norm: Option<Tensor<B, 1>>,
    k_norm: Option<Tensor<B, 1>>,
    gate: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
    input_norm: Tensor<B, 1>,
    /// Pre-MLP norm: llama's `post_attention_layernorm`, gemma's
    /// `pre_feedforward_layernorm` (same role, different name).
    pre_mlp_norm: Tensor<B, 1>,
    attn_out_norm: Option<Tensor<B, 1>>,
    mlp_out_norm: Option<Tensor<B, 1>>,
}

/// Llama-family causal LM, parameterized by the resolved [`ArchSpec`] —
/// llama, smollm2, qwen2, and mistral today; the gemma/qwen3/phi presets
/// migrate onto it stage by stage (roadmap wave 2).
pub struct LlamaModel<B: Backend> {
    metadata: ModelMetadata,
    spec: ArchSpec,
    /// `[vocab, hidden]`, dense or packed. When packed, lookup runs the
    /// dequant-gather kernel and a tied head shares the packed table
    /// through `lm_head` — so `lm_head: None` implies the dense arm.
    embed: crate::embed::Embedding<B>,
    lm_head: Option<Linear<B>>, // None => tied to the DENSE `embed`
    final_norm: Tensor<B, 1>,
    layers: Vec<LlamaLayer<B>>,
    rotary: RotaryEmbedding<B>,
    /// Local-theta tables for sliding layers (gemma dual-RoPE).
    rotary_local: Option<RotaryEmbedding<B>>,
    /// `1/sqrt(query_pre_attn_scalar || head_dim)`.
    scale: f64,
}

/// `y = x @ W^T (+ b)` for `[batch, seq, in] @ [out, in]`.
pub(crate) fn linear<B: Backend>(
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

pub(crate) fn load_tensor<B: Backend, const D: usize>(
    source: &dyn ModelSource,
    device: &Device<B>,
    name: &str,
) -> Result<Tensor<B, D>> {
    load_weight(source, device, name)
}

/// Loads a projection weight as a [`Linear`]: the packed-quant fast path
/// when the source stores it in a kernel-supported GGUF format *and* the
/// backend runs on wgpu, else the dense tensor (portable fallback).
pub(crate) fn load_linear<B: Backend>(
    source: &dyn ModelSource,
    device: &Device<B>,
    name: &str,
) -> Result<Linear<B>> {
    if let Some(op) = try_quant_linear::<B>(source, name, device)? {
        return Ok(Linear::Quant(op));
    }
    Ok(Linear::Dense(load_weight(source, device, name)?))
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

/// Loads a fused projection dense and splits its rows into `N` groups (phi
/// `qkv_proj` = `[q|k|v]`, `gate_up_proj` = `[gate|up]`, HF phi3 order).
/// Dense-only on purpose: fused checkpoints are safetensors; GGUF phi files
/// are row-sliced into split names by the format adapter before reaching
/// this loader.
fn split_fused_rows<B: Backend, const N: usize>(
    source: &dyn ModelSource,
    device: &Device<B>,
    name: &str,
    rows: [usize; N],
) -> Result<[Linear<B>; N]> {
    let w: Tensor<B, 2> = load_weight(source, device, name)?;
    let [total, cols] = w.dims();
    let expect: usize = rows.iter().sum();
    if total != expect {
        return Err(ModelError::BadShape {
            tensor: name.to_string(),
            expected: vec![expect, cols],
            got: vec![total, cols],
        });
    }
    let mut at = 0;
    Ok(rows.map(|r| {
        let part = w.clone().narrow(0, at, r);
        at += r;
        Linear::Dense(part)
    }))
}

/// Row-splits a fused projection's bias when present.
fn split_fused_bias<B: Backend, const N: usize>(
    source: &dyn ModelSource,
    device: &Device<B>,
    name: &str,
    rows: [usize; N],
) -> Result<Option<[Tensor<B, 1>; N]>> {
    let Some(b) = load_optional_bias(source, device, name)? else {
        return Ok(None);
    };
    let [total] = b.dims();
    let expect: usize = rows.iter().sum();
    if total != expect {
        return Err(ModelError::BadShape {
            tensor: name.to_string(),
            expected: vec![expect],
            got: vec![total],
        });
    }
    let mut at = 0;
    Ok(Some(rows.map(|r| {
        let part = b.clone().narrow(0, at, r);
        at += r;
        part
    })))
}

impl<B: Backend> LlamaModel<B> {
    pub(crate) fn expect_shape(name: &str, got: &[usize], expected: &[usize]) -> Result<()> {
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

    /// Norm in the spec's flavor (`x̂·w` vs gemma's `x̂·(1+w)`).
    fn norm<const D: usize>(&self, x: Tensor<B, D>, w: &Tensor<B, 1>) -> Tensor<B, D> {
        match self.spec.norm_flavor {
            NormFlavor::RmsNorm => rms_norm(x, w.clone(), self.metadata.rms_norm_eps),
            NormFlavor::GemmaRmsNorm => {
                gemma_rms_norm(x, w.clone(), self.metadata.rms_norm_eps)
            }
        }
    }

    /// RoPE tables for a layer: sliding layers rotate with the local theta
    /// when the spec defines one (gemma dual-RoPE), global layers with the
    /// (possibly scaled) global tables.
    fn rotary_for(&self, layer_idx: usize) -> &RotaryEmbedding<B> {
        match (self.spec.layers.get(layer_idx), &self.rotary_local) {
            (Some(LayerKind::Sliding(_)), Some(local)) => local,
            _ => &self.rotary,
        }
    }

    /// Shared trunk for prefill and decode: embeddings in, final-normed
    /// hidden states out. `pos` is the absolute position of the first input
    /// token.
    pub(crate) fn forward_hidden(
        &self,
        mut x: Tensor<B, 3>,
        cache: &mut dyn KVCache<B>,
        pos: usize,
    ) -> Tensor<B, 3> {
        let m = &self.metadata;
        let [_, seq, _] = x.dims();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let window = match self.spec.layers.get(layer_idx) {
                Some(LayerKind::Sliding(w)) => Some(*w),
                _ => None,
            };

            // --- attention block ------------------------------------------------
            let h = self.norm(x.clone(), &layer.input_norm);
            let q = layer.q.forward(h.clone(), layer.q_bias.as_ref());
            let k = layer.k.forward(h.clone(), layer.k_bias.as_ref());
            let v = layer.v.forward(h, layer.v_bias.as_ref());

            let mut q = q
                .reshape([1, seq, m.num_attention_heads, m.head_dim])
                .swap_dims(1, 2);
            let mut k = k
                .reshape([1, seq, m.num_key_value_heads, m.head_dim])
                .swap_dims(1, 2);
            let v = v
                .reshape([1, seq, m.num_key_value_heads, m.head_dim])
                .swap_dims(1, 2);

            // Per-head QK-norm (gemma-3 / qwen-3), over head_dim.
            if let Some(qn) = &layer.q_norm {
                q = self.norm(q, qn);
            }
            if let Some(kn) = &layer.k_norm {
                k = self.norm(k, kn);
            }

            let rotary = self.rotary_for(layer_idx);
            let q = rotary.apply(q, pos);
            let k = rotary.apply(k, pos);

            // The cache owns K/V layout, GQA expansion and masking
            // (causal + optional sliding window).
            let ctx = cache.attention_opts(layer_idx, q, k, v, pos, self.scale, window);
            let ctx = ctx
                .swap_dims(1, 2)
                .reshape([1, seq, m.num_attention_heads * m.head_dim]);
            let mut attn_out = layer.o.forward(ctx, layer.o_bias.as_ref());
            if let Some(n) = &layer.attn_out_norm {
                attn_out = self.norm(attn_out, n);
            }
            x = x + attn_out;

            // --- MLP block (gated) ----------------------------------------------
            let h = self.norm(x.clone(), &layer.pre_mlp_norm);
            let gated = crate::act::apply(self.spec.activation, layer.gate.forward(h.clone(), None))
                * layer.up.forward(h.clone(), None);
            let mut mlp_out = layer.down.forward(gated, None);
            if let Some(n) = &layer.mlp_out_norm {
                mlp_out = self.norm(mlp_out, n);
            }
            x = x + mlp_out;
        }

        self.norm(x, &self.final_norm)
    }

    /// Logits of every position: `[1, seq, hidden] -> [1, seq, vocab]`
    /// (the perplexity / speculative-decode head).
    pub(crate) fn all_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, seq, hidden_size] = hidden.dims();
        let logits: Tensor<B, 3> = match &self.lm_head {
            Some(head) => head.forward(hidden, None),
            None => {
                // Tied head over the dense table. A packed embedding
                // always installs a packed tied head, so this arm is
                // dense by construction.
                let table = self
                    .embed
                    .dense()
                    .expect("lm_head: None implies a dense embedding");
                let flat = hidden.reshape([seq, hidden_size]);
                let out = safe_matmul(flat, table.clone().transpose());
                let [_, vocab] = out.dims();
                out.reshape([1, seq, vocab])
            }
        };
        match self.spec.final_logit_softcap {
            Some(cap) => logits.div_scalar(cap as f32).tanh().mul_scalar(cap as f32),
            None => logits,
        }
    }

    /// Logits of the last sequence position: `[1, hidden] -> [1, vocab]`.
    pub(crate) fn last_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 2> {
        let [_, seq, hidden_size] = hidden.dims();
        let last = hidden.narrow(1, seq - 1, 1); // [1, 1, hidden]
        let logits: Tensor<B, 2> = match &self.lm_head {
            Some(head) => {
                let logits = head.forward(last, None);
                let [_, _, vocab] = logits.dims();
                logits.reshape([1, vocab])
            }
            None => {
                // Dense by construction — see `all_logits`.
                let table = self
                    .embed
                    .dense()
                    .expect("lm_head: None implies a dense embedding");
                let last = last.reshape([1, hidden_size]);
                safe_matmul(last, table.clone().transpose())
            }
        };
        match self.spec.final_logit_softcap {
            Some(cap) => logits.div_scalar(cap as f32).tanh().mul_scalar(cap as f32),
            None => logits,
        }
    }
}

impl<B: Backend> GenerativeModel<B> for LlamaModel<B> {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn load(source: &dyn ModelSource, device: &Device<B>) -> Result<Self> {
        // HF exports either the causal-LM wrapper (model.embed_tokens…) or
        // the bare base model (embed_tokens…, the sentence-transformers /
        // embedding-checkpoint layout); detect from the tensor names.
        let prefix = if source
            .tensor_names()
            .iter()
            .any(|n| n == "model.embed_tokens.weight" || n == "model.embed_tokens")
        {
            "model"
        } else {
            ""
        };
        Self::load_with_prefix(source, device, prefix)
    }

    fn create_kv_cache(&self, config: &CacheConfig) -> Box<dyn KVCache<B>> {
        match config.kind {
            CacheKind::Contiguous => {
                Box::new(ContiguousKVCache::<B>::new(self.metadata.num_hidden_layers))
            }
            // Sliding layers (per the resolved layout) keep a rolling tensor
            // instead of a paged arena; all-global specs pass all-None.
            CacheKind::Paged => Box::new(PagedKVCache::<B>::new_with_windows(
                self.metadata.num_hidden_layers,
                *config,
                self.spec.windows(),
            )),
        }
    }

    fn embed(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let embedded = self.embed.gather(tokens);
        if self.spec.embed_scale_sqrt_hidden {
            // Gemma scales embeddings by sqrt(hidden); computed in f32 —
            // the half-precision product was the f16 garbage-output bug.
            let out_dtype = embedded.dtype();
            let scale = (self.metadata.hidden_size as f64).sqrt();
            to_float(to_f32(embedded).mul_scalar(scale), out_dtype)
        } else {
            embedded
        }
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

    fn prefill_hidden(
        &mut self,
        input: Tensor<B, 3>,
        cache: &mut dyn KVCache<B>,
        pos: Range<u32>,
    ) -> Result<Tensor<B, 3>> {
        let [_, seq, _] = input.dims();
        assert_eq!(
            seq,
            (pos.end - pos.start) as usize,
            "prefill pos range must match the input sequence length"
        );
        Ok(self.forward_hidden(input, cache, pos.start as usize))
    }

    fn supports_hidden_states(&self) -> bool {
        true
    }

    fn prefill_all_logits(
        &mut self,
        input: Tensor<B, 3>,
        cache: &mut dyn KVCache<B>,
        pos: Range<u32>,
    ) -> Result<Tensor<B, 3>> {
        let hidden = self.prefill_hidden(input, cache, pos)?;
        Ok(self.all_logits(hidden))
    }

    fn decode(&mut self, input: Tensor<B, 3>, cache: &mut dyn KVCache<B>) -> Tensor<B, 2> {
        let pos = cache.seq_len();
        let hidden = self.forward_hidden(input, cache, pos);
        self.last_logits(hidden)
    }

    fn decode_all_logits(
        &mut self,
        input: Tensor<B, 3>,
        cache: &mut dyn KVCache<B>,
    ) -> Result<Tensor<B, 3>> {
        let pos = cache.seq_len();
        let hidden = self.forward_hidden(input, cache, pos);
        Ok(self.all_logits(hidden))
    }

    fn supports_decode_all_logits(&self) -> bool {
        true
    }
}

impl<B: Backend> LlamaModel<B> {
    /// Loads the text stack with weight names under `prefix` (e.g. `"model"`
    /// for plain Llama, `"model.text_model"` for Idefics3/SmolVLM, `""` for
    /// bare base-model exports). `lm_head.weight` always stays top-level.
    pub(crate) fn load_with_prefix(
        source: &dyn ModelSource,
        device: &Device<B>,
        prefix: &str,
    ) -> Result<Self> {
        let m = source.metadata().clone();
        let spec = ArchSpec::resolve(&m);
        // Dotted prefix, or nothing for bare exports.
        let prefix = if prefix.is_empty() {
            String::new()
        } else {
            format!("{prefix}.")
        };
        let prefix = prefix.as_str();

        // Packed first: keep the largest tensor of the model in its GGUF
        // packing, with lookup as a gather kernel — and hold onto the
        // packed tied-head op it comes paired with. Any refusal falls
        // back to the dense load, byte-for-byte today's behavior.
        let embed_name = format!("{prefix}embed_tokens.weight");
        let (embed, packed_head) =
            match crate::embed::try_quant_embedding::<B>(source, &embed_name, device)? {
                Some((e, head)) => (e, Some(head)),
                None => {
                    let t: Tensor<B, 2> = load_weight(source, device, &embed_name)?;
                    (crate::embed::Embedding::Dense(t), None)
                }
            };
        Self::expect_shape(
            "embed_tokens.weight",
            &embed.dims(),
            &[m.vocab_size, m.hidden_size],
        )?;

        // A tied head with a packed embedding uses the SAME packed table
        // through the quant-linear op — one copy in VRAM, and the decode
        // head's dense-vs-embed matmul becomes a packed gemv. With a
        // dense embedding, `None` keeps today's tied fallback matmul.
        let tied_head = || packed_head.map(Linear::Quant);
        let lm_head = if m.tie_word_embeddings {
            tied_head()
        } else {
            match load_linear(source, device, "lm_head.weight") {
                Ok(w) => {
                    Self::expect_shape(
                        "lm_head.weight",
                        &w.dims(),
                        &[m.vocab_size, m.hidden_size],
                    )?;
                    Some(w)
                }
                // Configs lie about tying (gemma3 omits the flag entirely
                // but ships no lm_head): presence decides, like the GGUF
                // output.weight rule — loudly, since a genuinely untied
                // checkpoint missing its head would be corrupt.
                Err(ModelError::MissingTensor(_)) => {
                    eprintln!(
                        "[load] lm_head.weight absent; falling back to tied embeddings"
                    );
                    tied_head()
                }
                Err(e) => return Err(e),
            }
        };

        let final_norm: Tensor<B, 1> =
            load_weight(source, device, &format!("{prefix}norm.weight"))?;

        // Sandwich-norm architectures (gemma) rename the pre-MLP norm and
        // add output norms around both residual adds.
        let pre_mlp_name = if spec.sandwich_norms {
            "pre_feedforward_layernorm"
        } else {
            "post_attention_layernorm"
        };

        let q_rows = m.num_attention_heads * m.head_dim;
        let kv_rows = m.num_key_value_heads * m.head_dim;
        let mut layers = Vec::with_capacity(m.num_hidden_layers);
        for i in 0..m.num_hidden_layers {
            let p = format!("{prefix}layers.{i}");
            // Phi-family checkpoints fuse the attention input projections
            // (`qkv_proj` = [q|k|v] rows); probe the split names first.
            let (q, k, v, fused_qkv_bias) =
                match load_linear(source, device, &format!("{p}.self_attn.q_proj.weight")) {
                    Ok(q) => (
                        q,
                        load_linear(source, device, &format!("{p}.self_attn.k_proj.weight"))?,
                        load_linear(source, device, &format!("{p}.self_attn.v_proj.weight"))?,
                        None,
                    ),
                    Err(ModelError::MissingTensor(_)) => {
                        let name = format!("{p}.self_attn.qkv_proj");
                        let [q, k, v] = split_fused_rows(
                            source,
                            device,
                            &format!("{name}.weight"),
                            [q_rows, kv_rows, kv_rows],
                        )?;
                        let b = split_fused_bias(
                            source,
                            device,
                            &format!("{name}.bias"),
                            [q_rows, kv_rows, kv_rows],
                        )?;
                        (q, k, v, b)
                    }
                    Err(e) => return Err(e),
                };
            let o = load_linear(source, device, &format!("{p}.self_attn.o_proj.weight"))?;
            Self::expect_shape(
                &format!("{p}.self_attn.q_proj.weight"),
                &q.dims(),
                &[q_rows, m.hidden_size],
            )?;
            Self::expect_shape(
                &format!("{p}.self_attn.k_proj.weight"),
                &k.dims(),
                &[kv_rows, m.hidden_size],
            )?;
            // Same fusion for the MLP input (`gate_up_proj` = [gate|up]).
            let (gate, up) =
                match load_linear(source, device, &format!("{p}.mlp.gate_proj.weight")) {
                    Ok(gate) => (
                        gate,
                        load_linear(source, device, &format!("{p}.mlp.up_proj.weight"))?,
                    ),
                    Err(ModelError::MissingTensor(_)) => {
                        let [gate, up] = split_fused_rows(
                            source,
                            device,
                            &format!("{p}.mlp.gate_up_proj.weight"),
                            [m.intermediate_size, m.intermediate_size],
                        )?;
                        (gate, up)
                    }
                    Err(e) => return Err(e),
                };

            // Bias loading is presence-driven: HF Qwen2 configs never emit
            // `attention_bias` (the bias is implicit in the modeling code),
            // so gating on metadata would silently skip real bias tensors.
            // Plain llama/smollm checkpoints have none and probe to `None`.
            let bias = |proj: &str| -> Result<Option<Tensor<B, 1>>> {
                load_optional_bias(source, device, &format!("{p}.{proj}.bias"))
            };
            let (q_bias, k_bias, v_bias) = match fused_qkv_bias {
                Some([qb, kb, vb]) => (Some(qb), Some(kb), Some(vb)),
                None => (
                    bias("self_attn.q_proj")?,
                    bias("self_attn.k_proj")?,
                    bias("self_attn.v_proj")?,
                ),
            };

            let optional_norm = |name: &str| -> Result<Option<Tensor<B, 1>>> {
                match source.open_tensor(&format!("{p}.{name}.weight")) {
                    Ok(reader) => Ok(Some(
                        reader.load_to_tensor::<B, 1>(device).map_err(ModelError::Format)?,
                    )),
                    Err(combs_formats::FormatError::TensorNotFound(_)) => Ok(None),
                    Err(e) => Err(ModelError::Format(e)),
                }
            };

            layers.push(LlamaLayer {
                q,
                k,
                v,
                o,
                q_bias,
                k_bias,
                v_bias,
                o_bias: bias("self_attn.o_proj")?,
                q_norm: if spec.qk_norm { optional_norm("self_attn.q_norm")? } else { None },
                k_norm: if spec.qk_norm { optional_norm("self_attn.k_norm")? } else { None },
                gate,
                up,
                down: load_linear(source, device, &format!("{p}.mlp.down_proj.weight"))?,
                input_norm: load_weight(source, device, &format!("{p}.input_layernorm.weight"))?,
                pre_mlp_norm: load_weight(
                    source,
                    device,
                    &format!("{p}.{pre_mlp_name}.weight"),
                )?,
                attn_out_norm: if spec.sandwich_norms {
                    optional_norm("post_attention_layernorm")?
                } else {
                    None
                },
                mlp_out_norm: if spec.sandwich_norms {
                    optional_norm("post_feedforward_layernorm")?
                } else {
                    None
                },
            });
            combs_core::progress::load(
                "weights",
                Some((i + 1) as u64),
                Some(m.num_hidden_layers as u64),
                None,
            );
        }

        let rotary = RotaryEmbedding::new_scaled(
            m.head_dim,
            spec.rope_theta,
            m.max_position_embeddings,
            &spec.rope_scaling,
            device,
        );
        let rotary_local = spec.rope_local_theta.map(|theta| {
            // Local (sliding) layers rotate unscaled at their own base.
            RotaryEmbedding::new(m.head_dim, theta, m.max_position_embeddings, device)
        });

        Ok(LlamaModel {
            scale: 1.0
                / spec
                    .query_pre_attn_scalar
                    .unwrap_or(m.head_dim as f64)
                    .sqrt(),
            metadata: m,
            spec,
            embed,
            lm_head,
            final_norm,
            layers,
            rotary,
            rotary_local,
        })
    }
}
