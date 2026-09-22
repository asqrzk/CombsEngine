//! The typed-decision heads that sit on a ModernBERT encoder.
//!
//! The encoder gives one hidden state per token. These heads turn those
//! into an answer: a logit per option marker, and a decision about
//! whether to answer at all.
//!
//! Four things about the order, each of which a reasonable reading gets
//! backwards, and each of which changes the numbers without changing
//! their shape:
//!
//! * The question TYPE is added to every position, not to the first one.
//! * The markers are gathered AFTER the head, not before it. The head is
//!   what lets one option's marker see the others.
//! * The head runs over the WHOLE sequence — the state included — with
//!   the padding mask, not over the markers alone.
//! * The act features are taken from `softmax(RAW logits)`, before the
//!   calibration temperature, and the pooled vector they ride with is
//!   `h[0]` taken POST-head.
//!
//! The head is torch's own `TransformerEncoderLayer(norm_first=True)`
//! with its default ReLU, so it is spelled out here rather than
//! approximated: pre-norm around each residual, one fused QKV
//! projection, and no final norm, because `nn.TransformerEncoder` was
//! built without one.

use burn::tensor::{activation::softmax, backend::Backend, Device, Int, Tensor};
use combs_classify::sequence::DecisionBatch;
use combs_formats::{ModelMetadata, ModelSource};

use crate::llama::{linear, load_tensor};
use crate::matmul::safe_matmul;
use crate::modernbert::{EncoderGeometry, ModernBert};
use crate::norm::layer_norm_opt;
use crate::precision::{to_f32, to_float};
use crate::{ModelError, Result};

/// What a marker that does not exist scores. laya's own value: low
/// enough to vanish under softmax, and NOT `-inf`, which would make the
/// entropy of a padded row a NaN.
const INVALID: f32 = -1e4;

/// torch's LayerNorm default, which the head was built with.
const HEAD_EPS: f64 = 1e-5;

/// Features the act head reads beside the pooled vector: the top
/// probability, its margin over the runner-up, the normalised entropy,
/// and the option count scaled by 255.
pub const ACT_FEATURES: usize = 4;

/// What one batch of questions came back with.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionLogits {
    /// One row per question, one logit per marker slot; slots with no
    /// marker carry [`INVALID`].
    pub logits: Vec<Vec<f32>>,
    /// The act distribution per question, already softmaxed.
    pub act: Vec<Vec<f32>>,
}

/// Anything that can answer a batch of typed questions.
pub trait TypedDecisionModel: Send {
    fn metadata(&self) -> &ModelMetadata;
    fn decide(&self, batch: &DecisionBatch) -> Result<DecisionLogits>;
}

/// A Linear's weight `[out, in]` and its bias.
struct Dense<B: Backend> {
    w: Tensor<B, 2>,
    b: Tensor<B, 1>,
}

/// A LayerNorm's weight and its bias. Every norm in the torch head has
/// both, where none in the encoder below has either.
struct Norm<B: Backend> {
    w: Tensor<B, 1>,
    b: Tensor<B, 1>,
}

/// LayerNorm, Linear, GELU, Linear — `scorer.0` through `.3` under
/// torch's Sequential indices, where `.2` is the GELU and carries no
/// weight of its own.
struct Scorer<B: Backend> {
    norm: Norm<B>,
    hidden: Dense<B>,
    out: Dense<B>,
}

/// Linear, GELU, Linear — `act_head.0` and `.2`.
struct ActHead<B: Backend> {
    hidden: Dense<B>,
    out: Dense<B>,
}

/// One layer of torch's `TransformerEncoderLayer`, pre-norm.
struct TorchEncoderLayer<B: Backend> {
    norm1: Norm<B>,
    in_proj: Dense<B>,
    out_proj: Dense<B>,
    norm2: Norm<B>,
    linear1: Dense<B>,
    linear2: Dense<B>,
}

impl<B: Backend> TorchEncoderLayer<B> {
    fn forward(&self, x: Tensor<B, 3>, pad: &Tensor<B, 4>, heads: usize) -> Tensor<B, 3> {
        let [batch, seq, hidden] = x.dims();
        let head_dim = hidden / heads;

        let n = layer_norm_opt(
            x.clone(),
            self.norm1.w.clone(),
            Some(self.norm1.b.clone()),
            HEAD_EPS,
        );
        let qkv = linear(n, &self.in_proj.w, Some(&self.in_proj.b));
        let part = |at: usize| {
            qkv.clone()
                .narrow(2, at * hidden, hidden)
                .reshape([batch, seq, heads, head_dim])
                .swap_dims(1, 2)
        };
        // No rotary here: torch's MultiheadAttention has no positional
        // term of its own, and the encoder below already placed one.
        let scores = safe_matmul(part(0), part(1).swap_dims(2, 3)) / (head_dim as f64).sqrt();
        // Back to the build's width before `out_proj`: see the note in
        // modernbert.rs. An f32 activation into an f16 weight is a
        // dispatch the backend refuses.
        let attended = to_float(
            safe_matmul(softmax(to_f32(scores) + pad.clone(), 3), to_f32(part(2))),
            x.dtype(),
        )
        .swap_dims(1, 2)
        .reshape([batch, seq, hidden]);
        let x = x + linear(attended, &self.out_proj.w, Some(&self.out_proj.b));

        let n = layer_norm_opt(
            x.clone(),
            self.norm2.w.clone(),
            Some(self.norm2.b.clone()),
            HEAD_EPS,
        );
        let ff = burn::tensor::activation::relu(linear(n, &self.linear1.w, Some(&self.linear1.b)));
        x + linear(ff, &self.linear2.w, Some(&self.linear2.b))
    }
}

/// A laya typed-decision model: the encoder, the head over its output,
/// and the two readouts.
pub struct LayaModel<B: Backend> {
    encoder: ModernBert<B>,
    type_emb: Tensor<B, 2>,
    head: Vec<TorchEncoderLayer<B>>,
    scorer: Scorer<B>,
    act_head: ActHead<B>,
    metadata: ModelMetadata,
    hidden: usize,
    head_heads: usize,
}

impl<B: Backend> LayaModel<B> {
    /// Loads the whole checkpoint. `head_layers` comes from
    /// `rl_agent_config.json`, which this crate does not parse.
    ///
    /// `metadata` is handed over rather than read off the source: a laya
    /// checkpoint keeps its encoder config under `encoder/` while its
    /// weights sit at the root, so where the config lives is the
    /// caller's problem and the model only needs the geometry it states.
    pub fn load(
        source: &dyn ModelSource,
        device: &Device<B>,
        metadata: ModelMetadata,
        head_layers: usize,
    ) -> Result<Self> {
        let cfg = metadata.encoder.clone().ok_or_else(|| {
            ModelError::Unsupported(format!(
                "{} is not a bidirectional encoder: a typed-decision model scores markers, and a \
                 decoder has none",
                metadata.architecture
            ))
        })?;
        let geom = EncoderGeometry::from_metadata(&metadata);
        let hidden = geom.hidden;
        let encoder =
            ModernBert::load(source, device, crate::system_one::ENCODER_PREFIX, cfg, geom)?;

        let w = |n: String| load_tensor::<B, 2>(source, device, &n);
        let v = |n: String| load_tensor::<B, 1>(source, device, &n);
        let lin = |n: &str| -> Result<Dense<B>> {
            Ok(Dense {
                w: w(format!("{n}.weight"))?,
                b: v(format!("{n}.bias"))?,
            })
        };
        let norm = |n: &str| -> Result<Norm<B>> {
            Ok(Norm {
                w: v(format!("{n}.weight"))?,
                b: v(format!("{n}.bias"))?,
            })
        };

        let mut head = Vec::with_capacity(head_layers);
        for j in 0..head_layers {
            let l = format!("head.layers.{j}");
            head.push(TorchEncoderLayer {
                norm1: norm(&format!("{l}.norm1"))?,
                in_proj: Dense {
                    w: w(format!("{l}.self_attn.in_proj_weight"))?,
                    b: v(format!("{l}.self_attn.in_proj_bias"))?,
                },
                out_proj: lin(&format!("{l}.self_attn.out_proj"))?,
                norm2: norm(&format!("{l}.norm2"))?,
                linear1: lin(&format!("{l}.linear1"))?,
                linear2: lin(&format!("{l}.linear2"))?,
            });
        }

        Ok(Self {
            encoder,
            type_emb: w("type_emb.weight".to_string())?,
            head,
            scorer: Scorer {
                norm: norm("scorer.0")?,
                hidden: lin("scorer.1")?,
                out: lin("scorer.3")?,
            },
            act_head: ActHead {
                hidden: lin("act_head.0")?,
                out: lin("act_head.2")?,
            },
            metadata,
            hidden,
            // torch's own rule in laya: one head per 64 dimensions, at
            // least one.
            head_heads: (hidden / 64).max(1),
        })
    }

    /// The hidden state the scorer reads, and the pooled vector the act
    /// head reads — both AFTER the head, which is the whole point of
    /// there being one.
    fn encode(&self, batch: &DecisionBatch, device: &Device<B>) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let (rows, len) = (batch.rows, batch.len);
        let ids = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(
                batch
                    .input_ids
                    .iter()
                    .map(|&i| i as i64)
                    .collect::<Vec<_>>(),
                [rows * len],
            ),
            device,
        )
        .reshape([rows, len]);
        let attention = Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(
                batch
                    .attention
                    .iter()
                    .map(|&a| a as f32)
                    .collect::<Vec<_>>(),
                [rows * len],
            ),
            device,
        )
        .reshape([rows, len]);

        let mut h = self.encoder.forward(ids, &attention);

        // The question's type, added at EVERY position.
        let qt = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(
                batch.qtype.iter().map(|&q| q as i64).collect::<Vec<_>>(),
                [rows],
            ),
            device,
        );
        let type_vec = self
            .type_emb
            .clone()
            .select(0, qt)
            .reshape([rows, 1, self.hidden])
            .expand([rows, len, self.hidden]);
        h = h + type_vec;

        // The same additive padding mask the encoder used, rebuilt for
        // the head: torch takes a boolean `src_key_padding_mask` and
        // this is its additive twin.
        let pad = (to_f32(attention).neg() + 1.0)
            .mul_scalar(crate::modernbert::MASKED as f64)
            .reshape([rows, 1, 1, len])
            .expand([rows, 1, len, len]);
        for layer in &self.head {
            h = layer.forward(h, &pad, self.head_heads);
        }
        (h, pad)
    }
}

impl<B: Backend> TypedDecisionModel for LayaModel<B> {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn decide(&self, batch: &DecisionBatch) -> Result<DecisionLogits> {
        let device = self.type_emb.device();
        let (rows, kmax) = (batch.rows, batch.markers);
        let (h, _) = self.encode(batch, &device);

        // Gather the marker rows. A masked slot's position is a zero
        // nobody reads: its logit is overwritten below.
        let idx = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(
                batch
                    .marker_pos
                    .iter()
                    .map(|&p| p as i64)
                    .collect::<Vec<_>>(),
                [rows * kmax],
            ),
            &device,
        )
        .reshape([rows, kmax, 1])
        .expand([rows, kmax, self.hidden]);
        let markers = h.clone().gather(1, idx);

        // scorer: LayerNorm -> Linear -> GELU -> Linear -> one logit.
        let s = layer_norm_opt(
            markers,
            self.scorer.norm.w.clone(),
            Some(self.scorer.norm.b.clone()),
            HEAD_EPS,
        );
        let s = burn::tensor::activation::gelu(linear(
            s,
            &self.scorer.hidden.w,
            Some(&self.scorer.hidden.b),
        ));
        let scored = linear(s, &self.scorer.out.w, Some(&self.scorer.out.b));
        // WIDENED before it is read: on an f16 build every tensor here
        // is f16, and asking for f32 out of one is a type error rather
        // than a conversion — which took the whole f16 seat down at the
        // first request, listening and answering nothing but a refusal.
        let mut logits: Vec<f32> = to_f32(scored)
            .reshape([rows * kmax])
            .into_data()
            .to_vec()
            .map_err(|e| ModelError::Unsupported(format!("reading the head's logits: {e:?}")))?;
        for (i, valid) in batch.marker_mask.iter().enumerate() {
            if !valid {
                logits[i] = INVALID;
            }
        }

        // The act features, from the RAW logits — before any
        // calibration temperature, which is a property of the answer and
        // not of the decision to answer.
        let mut feats = Vec::with_capacity(rows * ACT_FEATURES);
        for r in 0..rows {
            let row = &logits[r * kmax..(r + 1) * kmax];
            let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = row.iter().map(|&l| (l - top).exp()).collect();
            let sum: f32 = e.iter().sum();
            let p: Vec<f32> = e.iter().map(|&v| v / sum).collect();

            // `k` is the count of REAL markers, floored at two: a
            // one-option question is certain by construction and
            // dividing by ln(1) would be a division by zero.
            let k = batch.marker_mask[r * kmax..(r + 1) * kmax]
                .iter()
                .filter(|&&m| m)
                .count()
                .max(2) as f32;
            let ent = -p.iter().map(|&v| v * v.max(1e-9).ln()).sum::<f32>() / k.ln();
            let mut sorted = p.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let top1 = sorted.first().copied().unwrap_or(0.0);
            let top2 = sorted.get(1).copied().unwrap_or(0.0);
            feats.extend_from_slice(&[top1, top1 - top2, ent, k / 255.0]);
        }

        // The pooled vector is position zero AFTER the head.
        let pooled = h.narrow(1, 0, 1).reshape([rows, self.hidden]);
        // Born F32, like every other host-built tensor here: `from_data`
        // without a dtype takes the DEVICE's, so on the f16 build this
        // was an f16 tensor concatenated onto an f32 one — a width
        // mismatch the backend refuses, and the refusal surfaces only at
        // the next read, several operations from its cause.
        let feat_t = Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(feats, [rows * ACT_FEATURES]),
            (&device, burn::tensor::DType::F32),
        )
        .reshape([rows, ACT_FEATURES]);
        // And back to the weights' own width before the act head.
        let joined = to_float(
            Tensor::cat(vec![to_f32(pooled), feat_t], 1),
            self.act_head.hidden.w.dtype(),
        )
        .reshape([rows, 1, self.hidden + ACT_FEATURES]);
        let a = burn::tensor::activation::gelu(linear(
            joined,
            &self.act_head.hidden.w,
            Some(&self.act_head.hidden.b),
        ));
        let act_logits = linear(a, &self.act_head.out.w, Some(&self.act_head.out.b));
        let n_act = act_logits.dims()[2];
        let act_flat: Vec<f32> = to_f32(softmax(act_logits.reshape([rows, n_act]), 1))
            .reshape([rows * n_act])
            .into_data()
            .to_vec()
            .map_err(|e| ModelError::Unsupported(format!("reading the act head: {e:?}")))?;

        Ok(DecisionLogits {
            logits: (0..rows)
                .map(|r| logits[r * kmax..(r + 1) * kmax].to_vec())
                .collect(),
            act: (0..rows)
                .map(|r| act_flat[r * n_act..(r + 1) * n_act].to_vec())
                .collect(),
        })
    }
}
