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
        // Widened before the product, as in modernbert.rs.
        let scores = crate::precision::attention_scores(part(0), part(1).swap_dims(2, 3), head_dim);
        // Back to the build's width before `out_proj`: see the note in
        // modernbert.rs. An f32 activation into an f16 weight is a
        // dispatch the backend refuses.
        let attended = to_float(
            safe_matmul(softmax(scores + pad.clone(), 3), to_f32(part(2))),
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

/// A float tensor built on the host, born F32 on EVERY build. `from_data`
/// without a width takes the DEVICE's — f16 on the f16 build — and a
/// host-built f16 beside an f32 is a width mismatch this backend refuses
/// at dispatch or, worse, does not. Every host-built float tensor in this
/// file goes through here; `width_tests` holds it to that.
fn host_f32<B: Backend, const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
    device: &Device<B>,
) -> Tensor<B, D> {
    Tensor::from_data(
        burn::tensor::TensorData::new(values, shape),
        (device, burn::tensor::DType::F32),
    )
}

/// How deep the head in these weights is: the number of `j` for which
/// `head.layers.{j}.norm1.weight` exists, counted contiguously from 0.
/// A gap ends the count (a checkpoint with layers 0 and 2 has one).
pub fn head_depth(names: &[String]) -> usize {
    (0..)
        .take_while(|j| {
            names
                .iter()
                .any(|n| n == &format!("head.layers.{j}.norm1.weight"))
        })
        .count()
}

impl<B: Backend> LayaModel<B> {
    /// Loads the whole checkpoint. The head's depth is read off the
    /// weights (`head.layers.{j}.*`); `head_layers` is what
    /// `rl_agent_config.json` states, if it states it, and a stated depth
    /// that disagrees with the weights is refused naming both.
    ///
    /// `metadata` is handed over rather than read off the source: a laya
    /// checkpoint keeps its encoder config under `encoder/` while its
    /// weights sit at the root, so where the config lives is the
    /// caller's problem and the model only needs the geometry it states.
    pub fn load(
        source: &dyn ModelSource,
        device: &Device<B>,
        metadata: ModelMetadata,
        head_layers: Option<usize>,
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

        // The head's depth is a fact of the weights: `head.layers.{j}.*`
        // exists for exactly the layers there are. A config may state it;
        // one that states it wrongly is refused rather than trusted,
        // because a two-layer head loaded over three layers of weights
        // answers — wrongly, and without a word.
        let counted = head_depth(&source.tensor_names());
        if counted == 0 {
            return Err(ModelError::Unsupported(
                "no `head.layers.*` in the weights: an encoder without laya's head cannot decide"
                    .into(),
            ));
        }
        let head_layers = match head_layers {
            Some(stated) if stated != counted => {
                return Err(ModelError::Unsupported(format!(
                    "rl_agent_config.json states head_layers = {stated} but the weights carry \
                     {counted} (head.layers.0..{counted}): the config disagrees with the checkpoint"
                )));
            }
            _ => counted,
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

    /// The head's depth as loaded — read off the weights.
    pub fn head_layers(&self) -> usize {
        self.head.len()
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
        // The padding mask, born F32 like every host-built tensor here.
        // It used to take the device's width — f16 on the f16 build —
        // where its siblings named theirs: harmless while `to_f32`
        // widened it downstream, and exactly the class that took the f16
        // seat down once (`feat_t`, below).
        let attention = host_f32::<B, 1>(
            batch.attention.iter().map(|&a| a as f32).collect(),
            [rows * len],
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
        let feat_t =
            host_f32::<B, 1>(feats, [rows * ACT_FEATURES], &device).reshape([rows, ACT_FEATURES]);
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

#[cfg(test)]
mod depth_tests {
    use super::head_depth;

    #[test]
    fn the_depth_is_the_contiguous_count_of_head_layers_in_the_weights() {
        let names = |js: &[usize]| -> Vec<String> {
            js.iter()
                .flat_map(|j| {
                    [
                        format!("head.layers.{j}.norm1.weight"),
                        format!("head.layers.{j}.in_proj.weight"),
                    ]
                })
                .chain(["scorer.0.weight".to_string()])
                .collect()
        };
        assert_eq!(head_depth(&names(&[])), 0);
        assert_eq!(head_depth(&names(&[0])), 1);
        assert_eq!(head_depth(&names(&[0, 1, 2])), 3);
        assert_eq!(head_depth(&names(&[0, 2])), 1, "a gap ends the count");
        assert_eq!(
            head_depth(&names(&[1])),
            0,
            "layer 1 without layer 0 is no head"
        );
    }
}

#[cfg(test)]
mod width_tests {
    /// Every float tensor this file builds on the host names its width,
    /// by going through `host_f32`. A bare `from_data` takes the
    /// device's — f16 on the f16 build — and the seat's own lib tests
    /// cannot see that: they pin the test device to f32 before any
    /// tensor exists, so a tensor taking the device's width is f32 there
    /// too. The source is read instead: outside `host_f32`, only integer
    /// tensors may call `from_data` bare.
    #[test]
    fn every_host_built_float_tensor_names_its_width() {
        let src = include_str!("decision.rs");
        let call = concat!("from_", "data(");
        let mut bare = Vec::new();
        let mut in_helper = false;
        for (i, line) in src.lines().enumerate() {
            if line.starts_with("fn host_f32") {
                in_helper = true;
            }
            if in_helper {
                if line == "}" {
                    in_helper = false;
                }
                continue;
            }
            // An integer tensor names `Int` on the line of its call.
            if !line.contains(call) || line.contains("Int>") {
                continue;
            }
            bare.push(format!("{}: {}", i + 1, line.trim()));
        }
        assert!(
            bare.is_empty(),
            "host-built float tensors that do not go through host_f32:\n{}",
            bare.join("\n")
        );
    }
}
