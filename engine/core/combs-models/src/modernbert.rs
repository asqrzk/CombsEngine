//! A ModernBERT encoder, read bidirectionally.
//!
//! Every generative model in this crate reads left to right and caches
//! what it has read. This one does neither: it sees the whole sequence
//! at once, attends in both directions, and keeps nothing — so there is
//! no KV arena, no position offset, and no decode step. What it produces
//! is one hidden state per token, which the System 1 heads then read at
//! the option markers.
//!
//! Four things differ from the decoder beside it, and each is a place a
//! plausible-looking port goes quietly wrong:
//!
//! * **Two rotary bases in one forward pass.** Layers that attend to the
//!   whole sequence turn at 160000; the sliding ones turn at 10000. A
//!   port with one base still produces numbers.
//! * **LayerNorm, not RMSNorm**, and with no bias on this checkpoint.
//! * **A sliding BAND, not a causal triangle.** A sliding layer sees
//!   `|i - j| <= local_window`, both directions.
//! * **GeGLU packed into one projection.** `mlp.Wi` is twice the
//!   intermediate size: the first half is the value, the second the
//!   gate. Reading them the other way round is a silent halving of the
//!   model's meaning.
//!
//! Layer 0 carries no `attn_norm`: the embedding norm already normalised
//! that vector, and a second pass over it would be a second norm.

use burn::tensor::{activation::softmax, backend::Backend, Device, Int, Tensor};
use combs_formats::{EncoderConfig, ModelSource};

use crate::llama::{linear, load_tensor};
use crate::matmul::safe_matmul;
use crate::norm::layer_norm_opt;
use crate::precision::{to_f32, to_float};
use crate::rope::RotaryEmbedding;
use crate::{ModelError, Result};

/// One encoder block.
pub struct ModernBertLayer<B: Backend> {
    /// Absent on layer 0 only.
    attn_norm: Option<Tensor<B, 1>>,
    wqkv: Tensor<B, 2>,
    wo: Tensor<B, 2>,
    mlp_norm: Tensor<B, 1>,
    wi: Tensor<B, 2>,
    wo_mlp: Tensor<B, 2>,
}

/// The encoder's geometry, as the model's own metadata states it.
///
/// Named rather than passed as four bare `usize`, because three of them
/// are interchangeable at the call site and a swapped pair builds a
/// model that loads, runs, and is wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderGeometry {
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub layers: usize,
}

impl EncoderGeometry {
    pub fn from_metadata(meta: &combs_formats::ModelMetadata) -> Self {
        Self {
            hidden: meta.hidden_size,
            heads: meta.num_attention_heads,
            intermediate: meta.intermediate_size,
            layers: meta.num_hidden_layers,
        }
    }
}

/// The encoder, and the geometry the heads above it need.
pub struct ModernBert<B: Backend> {
    tok_embeddings: Tensor<B, 2>,
    embed_norm: Tensor<B, 1>,
    layers: Vec<ModernBertLayer<B>>,
    final_norm: Tensor<B, 1>,
    rotary_global: RotaryEmbedding<B>,
    rotary_local: RotaryEmbedding<B>,
    cfg: EncoderConfig,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    intermediate: usize,
}

/// What a masked-out score is filled with.
///
/// Half of `f32::MIN` rather than all of it: the mask is ADDED to a
/// score, and two masks over the same position (a sliding band and a
/// padded key) would overflow at the full value. Both choices exp to
/// zero, so nothing downstream can tell them apart.
pub(crate) const MASKED: f32 = f32::MIN / 2.0;

impl<B: Backend> ModernBert<B> {
    /// Loads every tensor the encoder needs, under `prefix` — `encoder.`
    /// in a laya checkpoint, where a `ForMaskedLM` wrapper would have put
    /// it one level deeper.
    pub fn load(
        source: &dyn ModelSource,
        device: &Device<B>,
        prefix: &str,
        cfg: EncoderConfig,
        geom: EncoderGeometry,
    ) -> Result<Self> {
        let EncoderGeometry {
            hidden,
            heads,
            intermediate,
            layers: layer_count,
        } = geom;
        if !hidden.is_multiple_of(heads) {
            return Err(ModelError::Unsupported(format!(
                "hidden size {hidden} does not divide into {heads} heads"
            )));
        }
        let head_dim = hidden / heads;
        let w = |n: String| load_tensor::<B, 2>(source, device, &n);
        let v = |n: String| load_tensor::<B, 1>(source, device, &n);

        let mut layers = Vec::with_capacity(layer_count);
        for i in 0..layer_count {
            let l = format!("{prefix}layers.{i}");
            layers.push(ModernBertLayer {
                // Only layer 0 may be without one; anywhere else its
                // absence is a checkpoint that is missing a tensor.
                attn_norm: if i == 0 {
                    None
                } else {
                    Some(v(format!("{l}.attn_norm.weight"))?)
                },
                wqkv: w(format!("{l}.attn.Wqkv.weight"))?,
                wo: w(format!("{l}.attn.Wo.weight"))?,
                mlp_norm: v(format!("{l}.mlp_norm.weight"))?,
                wi: w(format!("{l}.mlp.Wi.weight"))?,
                wo_mlp: w(format!("{l}.mlp.Wo.weight"))?,
            });
        }

        // Positions are 0..s every time — nothing is cached, so nothing
        // is offset.
        let max_pos = 8192;
        Ok(Self {
            tok_embeddings: w(format!("{prefix}embeddings.tok_embeddings.weight"))?,
            embed_norm: v(format!("{prefix}embeddings.norm.weight"))?,
            layers,
            final_norm: v(format!("{prefix}final_norm.weight"))?,
            rotary_global: RotaryEmbedding::new(head_dim, cfg.global_rope_theta, max_pos, device),
            rotary_local: RotaryEmbedding::new(head_dim, cfg.local_rope_theta, max_pos, device),
            cfg,
            hidden,
            heads,
            head_dim,
            intermediate,
        })
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.cfg
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// One hidden state per token: `[batch, seq, hidden]`.
    ///
    /// `attention` is 1 where a token is real and 0 where it is padding.
    pub fn forward(&self, ids: Tensor<B, 2, Int>, attention: &Tensor<B, 2>) -> Tensor<B, 3> {
        self.forward_taps(ids, attention, &[])
            .expect("an empty tap list can name no layer out of range")
            .0
    }

    /// The same, plus a copy of the residual stream at each depth named
    /// in `taps` — the depth knob a later arm scores offline, measured
    /// before it is ever a default.
    ///
    /// **A tap index is the number of layers APPLIED**, which is the
    /// coordinate `GenerativeModel::prefill_taps` states, the one
    /// `llama.rs` keeps, and the one a `transformers` `hidden_states`
    /// stack uses — so a published tap index carries over verbatim. Tap
    /// `0` is the stream the first layer reads (here the token embedding
    /// AFTER this architecture's embedding norm, which is what HF's
    /// `ModernBertEmbeddings` returns), and tap `num_layers` is the
    /// stream the final norm has not yet touched. **No norm is applied
    /// to any tap**; the returned `.0` is the normed output as always.
    ///
    /// This used to key on the 0-based index of the layer that produced
    /// the stream, so `taps=[0]` meant ONE layer applied and the
    /// embedding stream had no index at all — an off-by-one against
    /// every other tap in the estate, and silent, because a tap past the
    /// stack was simply absent from the result rather than refused.
    ///
    /// Taps come back in the CALLER's order, one entry per index given,
    /// so a caller who asks `[28, 7]` gets them that way round.
    pub fn forward_taps(
        &self,
        ids: Tensor<B, 2, Int>,
        attention: &Tensor<B, 2>,
        taps: &[usize],
    ) -> Result<(Tensor<B, 3>, Vec<(usize, Tensor<B, 3>)>)> {
        let layers = self.layers.len();
        if let Some(&bad) = taps.iter().find(|&&t| t > layers) {
            return Err(ModelError::Unsupported(format!(
                "tap {bad} exceeds the {layers}-layer stack: a tap is the number of layers \
                 applied, so {layers} is the deepest and 0 is the embedding stream"
            )));
        }
        let [batch, seq] = ids.dims();
        let device = ids.device();

        // The two masks, both additive and both born in f32: an f16 mask
        // would overflow to -inf and take every score with it.
        let pad = key_padding_mask(attention, batch, seq);
        let band = self.band_mask(seq, &device);
        let global_mask = pad.clone().expand([batch, 1, seq, seq]);
        let sliding_mask = global_mask.clone() + band.expand([batch, 1, seq, seq]);

        let flat = ids.reshape([batch * seq]);
        let embedded =
            self.tok_embeddings
                .clone()
                .select(0, flat)
                .reshape([batch, seq, self.hidden]);
        let mut h = layer_norm_opt(embedded, self.embed_norm.clone(), None, self.cfg.norm_eps);

        // One slot per index the caller asked for, filled in their order.
        let mut caught: Vec<Option<Tensor<B, 3>>> = vec![None; taps.len()];
        for (slot, &t) in taps.iter().enumerate() {
            if t == 0 {
                caught[slot] = Some(h.clone());
            }
        }
        for (i, layer) in self.layers.iter().enumerate() {
            let global = self.cfg.is_global_layer(i);
            let mask = if global { &global_mask } else { &sliding_mask };
            let rotary = if global {
                &self.rotary_global
            } else {
                &self.rotary_local
            };

            // Attention, around a residual.
            let normed = match &layer.attn_norm {
                Some(w) => layer_norm_opt(h.clone(), w.clone(), None, self.cfg.norm_eps),
                None => h.clone(),
            };
            let qkv = linear(normed, &layer.wqkv, None);
            let split = |at: usize| {
                qkv.clone()
                    .narrow(2, at * self.hidden, self.hidden)
                    .reshape([batch, seq, self.heads, self.head_dim])
                    .swap_dims(1, 2)
            };
            let (q, k) = rotary.apply_qk(split(0), split(1), 0);
            let v = split(2);

            // Widened BEFORE the product, not after: see
            // `precision::attention_scores`. This used to read
            // `safe_matmul(q, kᵀ)` at the build's dtype with a `to_f32`
            // on the result, which is the cast arriving after the
            // arithmetic it protects.
            let scores = crate::precision::attention_scores(q, k.swap_dims(2, 3), self.head_dim);
            // The scores and the mask are f32 for stability, so what
            // comes out is f32 — and `wo` below is the build's dtype.
            // Handing an f32 activation to an f16 weight is a width
            // mismatch the backend refuses at dispatch, which is what
            // took the f16 seat down; whisper.rs:249 records that the
            // same mismatch elsewhere is NOT refused and simply answers
            // wrongly, which is the worse half of the lesson.
            let attended = safe_matmul(softmax(scores + mask.clone(), 3), to_f32(v));
            let attended =
                to_float(attended, h.dtype())
                    .swap_dims(1, 2)
                    .reshape([batch, seq, self.hidden]);
            h = h + linear(attended, &layer.wo, None);

            // GeGLU, around the next residual. `Wi` is twice the
            // intermediate size: value first, gate second.
            let normed = layer_norm_opt(h.clone(), layer.mlp_norm.clone(), None, self.cfg.norm_eps);
            let wide = linear(normed, &layer.wi, None);
            let value = wide.clone().narrow(2, 0, self.intermediate);
            let gate = wide.narrow(2, self.intermediate, self.intermediate);
            let activated = burn::tensor::activation::gelu(value) * gate;
            h = h + linear(activated, &layer.wo_mlp, None);

            let applied = i + 1;
            for (slot, &t) in taps.iter().enumerate() {
                if t == applied {
                    caught[slot] = Some(h.clone());
                }
            }
        }

        Ok((
            layer_norm_opt(h, self.final_norm.clone(), None, self.cfg.norm_eps),
            taps.iter()
                .zip(caught)
                .map(|(&t, h)| (t, h.expect("every tap in range is caught")))
                .collect(),
        ))
    }

    /// `[1, 1, seq, seq]`, zero inside the band and `MASKED` outside it.
    ///
    /// The band is `|i - j| <= local_window`, in BOTH directions — a
    /// causal triangle here would be a decoder's mask on an encoder, and
    /// the edge is where an off-by-one lives, which is why the tiny
    /// fixture's window is four.
    fn band_mask(&self, seq: usize, device: &Device<B>) -> Tensor<B, 4> {
        let w = self.cfg.local_window as i64;
        let mut m = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if (i as i64 - j as i64).abs() > w {
                    m[i * seq + j] = MASKED;
                }
            }
        }
        // Born F32 on EVERY build. `from_floats` otherwise takes the
        // device's dtype, and on the f16 build `f32::MIN / 2` is past
        // ±65,504 and becomes infinity — and then an f32 score plus an
        // f16 mask is a width mismatch this backend refuses at dispatch
        // (it took the whole f16 seat down) or, worse, does not refuse
        // and answers wrongly. The same lesson whisper.rs:239 records.
        Tensor::from_data(
            burn::tensor::TensorData::new(m, [1, 1, seq, seq]),
            (device, burn::tensor::DType::F32),
        )
    }
}

/// `[batch, 1, 1, seq]`: zero on a real token, `MASKED` on padding.
fn key_padding_mask<B: Backend>(
    attention: &Tensor<B, 2>,
    batch: usize,
    seq: usize,
) -> Tensor<B, 4> {
    let keep = to_f32(attention.clone());
    // 1 -> 0, 0 -> MASKED.
    (keep.neg() + 1.0)
        .mul_scalar(MASKED as f64)
        .reshape([batch, 1, 1, seq])
}
