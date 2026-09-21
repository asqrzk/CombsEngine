//! What a System 1 checkpoint must carry, named before it is opened.
//!
//! A laya typed-decisions checkpoint is a ModernBERT encoder under
//! `encoder.`, a small torch transformer head run over the option
//! markers, and two `nn.Sequential` readouts on top of it. Nothing here
//! loads a tensor: this module *names* every tensor such a checkpoint
//! must carry, at the shape it must carry it, derived from the two
//! configs rather than discovered from the file — so a checkpoint that
//! dropped a tensor, or renamed one between revisions, is refused by
//! name instead of answering from a half-built head.
//!
//! The names are torch's own, under `nn.Sequential` INDICES: `scorer.0`
//! is the LayerNorm, `scorer.1` and `scorer.3` the two Linears, and the
//! `2` between them is the GELU, which carries no weight. Reading those
//! indices as words would be reading a checkpoint that does not exist.
//!
//! Two shapes here are not guesses and are worth saying out loud:
//! **layer 0 alone carries no `attn_norm`** (HF's ModernBERT normalises
//! the embeddings instead, so the first layer's pre-attention norm would
//! be a second norm over the same vector), and `temperature` is a
//! registered buffer the checkpoint carries — strict load demands it —
//! that inference never reads, because the calibration table in
//! `rl_agent_config.json` is the authority.

use combs_formats::ModelMetadata;

use crate::ModelError;

/// Prefix every encoder tensor sits under. Not `encoder.model.`: the
/// checkpoint stores the ModernBERT body one level up from where a
/// `ForMaskedLM` wrapper would put it.
pub const ENCODER_PREFIX: &str = "encoder.";

/// Question types the seat answers, in laya's own order: choice, score,
/// noul. `type_emb` holds one vector per type and `temperature` one
/// scalar per type.
pub const NUM_QUESTION_TYPES: usize = 3;

/// The two counts the head's shape depends on, which sit in
/// `rl_agent_config.json` rather than in the encoder's `config.json`.
///
/// Named fields rather than two bare `usize`, because swapping them
/// would build a manifest that is wrong and still plausible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadShape {
    /// Layers in the transformer head over the markers.
    pub layers: usize,
    /// Acts the head chooses between: one per act the config prices,
    /// plus answering, which costs nothing and is not in the table. This
    /// checkpoint prices one act and so has two.
    pub acts: usize,
}

/// Width of the act head's hidden layer.
pub const ACT_HIDDEN: usize = 256;

/// Features concatenated onto the pooled vector before the act head:
/// top-1 probability, the top-1/top-2 margin, normalised entropy, and
/// the option count scaled by 255. Four numbers — which is why the act
/// head's first Linear is `[256, hidden + 4]` and not `[256, hidden]`.
pub const ACT_FEATURES: usize = 4;

/// The torch head's feed-forward width, as a multiple of the model
/// dimension. laya passes `4 * d` where torch's own default is a flat
/// 2048, so the multiple is the checkpoint's fact, not torch's.
pub const HEAD_FFN_MULTIPLE: usize = 4;

/// Tensors a strict load demands and inference never reads. `temperature`
/// is a registered buffer: training wrote the per-type scalars into it,
/// and the seat reads the richer per-(type × option-count) table out of
/// `rl_agent_config.json` instead. Refusing a checkpoint for its absence
/// is right; reading it would be wrong.
pub const CARRIED_NOT_READ: &[&str] = &["temperature"];

/// The geometry of a typed-decision checkpoint.
///
/// It is assembled from two files, because the checkpoint keeps it in
/// two: `encoder/config.json` gives the body (hidden size, layer count,
/// vocabulary, which biases exist) and `rl_agent_config.json` gives the
/// head's depth. A caller that has read only the first cannot name the
/// manifest, and [`from_metadata`](Self::from_metadata) says so by
/// taking `head_layers` in the hand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayaShape {
    /// Model dimension — the encoder's and the head's alike.
    pub hidden_size: usize,
    /// The encoder MLP's inner width. `mlp.Wi` is TWICE this: GeGLU
    /// packs the value and its gate into one projection.
    pub intermediate_size: usize,
    /// Token-embedding rows.
    pub vocab_size: usize,
    /// Encoder layers.
    pub num_hidden_layers: usize,
    /// Whether the encoder's LayerNorms carry biases (ModernBERT-large:
    /// they do not).
    pub norm_bias: bool,
    /// Whether the encoder's attention projections carry biases.
    pub attention_bias: bool,
    /// Whether the encoder's MLP projections carry biases.
    pub mlp_bias: bool,
    /// Layers in the torch transformer head over the markers.
    pub head_layers: usize,
    /// The head's feed-forward width.
    pub head_ffn: usize,
    /// Acts the act head reads out.
    pub num_acts: usize,
}

impl LayaShape {
    /// Reads the encoder half from parsed metadata; the head's two
    /// counts come from `rl_agent_config.json`, which this crate does
    /// not parse.
    ///
    /// Refuses any metadata without an encoder section — a generative
    /// model has no markers to score.
    pub fn from_metadata(meta: &ModelMetadata, head: HeadShape) -> Result<Self, ModelError> {
        let enc = meta.encoder.as_ref().ok_or_else(|| {
            ModelError::Unsupported(format!(
                "{} is not a bidirectional encoder: a typed-decision checkpoint reads a whole \
                 sequence at once and has no decoder to name",
                meta.architecture
            ))
        })?;
        Ok(Self {
            hidden_size: meta.hidden_size,
            intermediate_size: meta.intermediate_size,
            vocab_size: meta.vocab_size,
            num_hidden_layers: meta.num_hidden_layers,
            norm_bias: enc.norm_bias,
            attention_bias: enc.attention_bias,
            mlp_bias: enc.mlp_bias,
            head_layers: head.layers,
            head_ffn: HEAD_FFN_MULTIPLE * meta.hidden_size,
            num_acts: head.acts,
        })
    }
}

/// Every tensor the checkpoint must carry, in load order: the encoder,
/// then the head, then the two readouts, then the type embedding, then
/// the buffer nobody reads.
///
/// Names are unique; the shapes are torch's own row-major `[out, in]`.
pub fn manifest(shape: &LayaShape) -> Vec<(String, Vec<usize>)> {
    let h = shape.hidden_size;
    let mut out = Vec::new();

    // --- the encoder, under `encoder.` ---
    let p = ENCODER_PREFIX;
    out.push((
        format!("{p}embeddings.tok_embeddings.weight"),
        vec![shape.vocab_size, h],
    ));
    norm(&mut out, &format!("{p}embeddings.norm"), h, shape.norm_bias);
    for i in 0..shape.num_hidden_layers {
        let l = format!("{p}layers.{i}");
        // Layer 0 normalises nothing before attention: the embedding
        // norm above already did it.
        if i > 0 {
            norm(&mut out, &format!("{l}.attn_norm"), h, shape.norm_bias);
        }
        linear(
            &mut out,
            &format!("{l}.attn.Wqkv"),
            [3 * h, h],
            shape.attention_bias,
        );
        linear(
            &mut out,
            &format!("{l}.attn.Wo"),
            [h, h],
            shape.attention_bias,
        );
        norm(&mut out, &format!("{l}.mlp_norm"), h, shape.norm_bias);
        // GeGLU: one projection wide enough for the value and its gate.
        linear(
            &mut out,
            &format!("{l}.mlp.Wi"),
            [2 * shape.intermediate_size, h],
            shape.mlp_bias,
        );
        linear(
            &mut out,
            &format!("{l}.mlp.Wo"),
            [h, shape.intermediate_size],
            shape.mlp_bias,
        );
    }
    norm(&mut out, &format!("{p}final_norm"), h, shape.norm_bias);

    // --- the torch head over the markers ---
    // `nn.TransformerEncoderLayer`: one fused QKV projection, an output
    // projection, two Linears and two LayerNorms — every one of them
    // with a bias, because torch's defaults say so and laya kept them.
    for j in 0..shape.head_layers {
        let l = format!("head.layers.{j}");
        out.push((format!("{l}.self_attn.in_proj_weight"), vec![3 * h, h]));
        out.push((format!("{l}.self_attn.in_proj_bias"), vec![3 * h]));
        linear(&mut out, &format!("{l}.self_attn.out_proj"), [h, h], true);
        linear(&mut out, &format!("{l}.linear1"), [shape.head_ffn, h], true);
        linear(&mut out, &format!("{l}.linear2"), [h, shape.head_ffn], true);
        norm(&mut out, &format!("{l}.norm1"), h, true);
        norm(&mut out, &format!("{l}.norm2"), h, true);
    }

    // --- the readouts, under Sequential indices ---
    // scorer: LayerNorm(0) -> Linear(1) -> GELU(2) -> Linear(3) -> one
    // logit per marker.
    norm(&mut out, "scorer.0", h, true);
    linear(&mut out, "scorer.1", [h, h], true);
    linear(&mut out, "scorer.3", [1, h], true);
    // act_head: Linear(0) -> GELU(1) -> Linear(2), reading the pooled
    // vector with the four decision features glued on.
    linear(&mut out, "act_head.0", [ACT_HIDDEN, h + ACT_FEATURES], true);
    linear(&mut out, "act_head.2", [shape.num_acts, ACT_HIDDEN], true);

    out.push(("type_emb.weight".to_string(), vec![NUM_QUESTION_TYPES, h]));
    out.push(("temperature".to_string(), vec![NUM_QUESTION_TYPES]));

    out
}

/// A LayerNorm: a weight, and a bias where the config says there is one.
fn norm(out: &mut Vec<(String, Vec<usize>)>, name: &str, hidden: usize, bias: bool) {
    out.push((format!("{name}.weight"), vec![hidden]));
    if bias {
        out.push((format!("{name}.bias"), vec![hidden]));
    }
}

/// A Linear: `[out, in]` and, where there is one, a bias of `[out]`.
fn linear(out: &mut Vec<(String, Vec<usize>)>, name: &str, dims: [usize; 2], bias: bool) {
    out.push((format!("{name}.weight"), vec![dims[0], dims[1]]));
    if bias {
        out.push((format!("{name}.bias"), vec![dims[0]]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use combs_formats::EncoderConfig;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    /// The header of the real `convaiinnovations/laya-typed-decisions`
    /// checkpoint, read out of its safetensors by HTTP range: 206 rows
    /// of name, dtype and shape. This fixture is the authority — the
    /// manifest is graded against it, never the other way round.
    fn header_tsv() -> BTreeMap<String, Vec<usize>> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/laya_typed_decisions.header.tsv");
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let mut col = l.split('\t');
                let name = col.next().expect("a name").to_string();
                let dtype = col.next().expect("a dtype");
                assert_eq!(dtype, "F16", "{name} is not F16");
                let dims = col
                    .next()
                    .expect("a shape")
                    .split('x')
                    .map(|d| d.parse::<usize>().expect("a dimension"))
                    .collect();
                (name, dims)
            })
            .collect()
    }

    /// ModernBERT-large's geometry as its own `config.json` states it
    /// (the fixture `combs-formats` grades its reader against), with the
    /// head depth `rl_agent_config.json` states.
    fn laya_typed_decisions() -> LayaShape {
        let mut m = ModelMetadata::diffusion_placeholder("modernbert");
        m.hidden_size = 1024;
        m.intermediate_size = 2624;
        m.vocab_size = 50368;
        m.num_hidden_layers = 28;
        m.num_attention_heads = 16;
        m.encoder = Some(EncoderConfig {
            global_attn_every_n_layers: 3,
            local_window: 64,
            global_rope_theta: 160_000.0,
            local_rope_theta: 10_000.0,
            norm_eps: 1e-5,
            norm_bias: false,
            attention_bias: false,
            mlp_bias: false,
            cls_id: 50281,
            sep_id: 50282,
            pad_id: 50283,
            layer_types: Vec::new(),
        });
        // `head_layers: 2` and one priced act — plus answering — as
        // `rl_agent_config.json` states them.
        LayaShape::from_metadata(&m, HeadShape { layers: 2, acts: 2 })
            .expect("an encoder has a decision shape")
    }

    #[test]
    fn the_manifest_names_exactly_what_the_checkpoint_carries() {
        let want = header_tsv();
        assert_eq!(want.len(), 206, "the fixture is 206 tensors");

        let named = manifest(&laya_typed_decisions());
        let got: BTreeMap<String, Vec<usize>> = named.iter().cloned().collect();
        assert_eq!(got.len(), named.len(), "the manifest names a tensor twice");

        let missing: Vec<&str> = want
            .keys()
            .filter(|k| !got.contains_key(*k))
            .map(|k| k.as_str())
            .collect();
        let extra: Vec<&str> = got
            .keys()
            .filter(|k| !want.contains_key(*k))
            .map(|k| k.as_str())
            .collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "the checkpoint carries {} the manifest does not name ({missing:?}) and the manifest \
             names {} the checkpoint does not carry ({extra:?})",
            missing.len(),
            extra.len()
        );

        for (name, dims) in &want {
            assert_eq!(got[name], *dims, "{name}");
        }
    }

    #[test]
    fn the_manifest_carries_the_buffer_inference_never_reads() {
        let named = manifest(&laya_typed_decisions());
        for carried in CARRIED_NOT_READ {
            assert!(
                named.iter().any(|(n, _)| n == carried),
                "a strict load demands {carried}"
            );
        }
        let t = named
            .iter()
            .find(|(n, _)| n == "temperature")
            .expect("temperature");
        assert_eq!(t.1, vec![NUM_QUESTION_TYPES]);
    }

    #[test]
    fn layer_zero_alone_carries_no_attention_norm() {
        let shape = laya_typed_decisions();
        let named = manifest(&shape);
        let has = |n: &str| named.iter().any(|(name, _)| name == n);
        assert!(!has("encoder.layers.0.attn_norm.weight"));
        for i in 1..shape.num_hidden_layers {
            assert!(
                has(&format!("encoder.layers.{i}.attn_norm.weight")),
                "layer {i}"
            );
        }
        // The embedding norm is what layer 0 leans on instead.
        assert!(has("encoder.embeddings.norm.weight"));
    }

    #[test]
    fn the_biases_the_config_denies_are_never_named() {
        let shape = laya_typed_decisions();
        let named = manifest(&shape);
        for (name, _) in &named {
            assert!(
                !name.starts_with(ENCODER_PREFIX) || !name.ends_with(".bias"),
                "{name}: ModernBERT-large carries no encoder bias"
            );
        }
        // Flip the three flags and every one of them appears.
        let with = LayaShape {
            norm_bias: true,
            attention_bias: true,
            mlp_bias: true,
            ..shape.clone()
        };
        let widened = manifest(&with);
        assert!(widened.len() > named.len());
        let has = |n: &str| widened.iter().any(|(name, _)| name == n);
        assert!(has("encoder.embeddings.norm.bias"));
        assert!(has("encoder.layers.3.attn.Wqkv.bias"));
        assert!(has("encoder.layers.3.mlp.Wi.bias"));
    }

    #[test]
    fn a_generative_config_has_no_decision_manifest() {
        let m = ModelMetadata::diffusion_placeholder("llama");
        let refused = LayaShape::from_metadata(&m, HeadShape { layers: 2, acts: 2 })
            .expect_err("a decoder scores no markers");
        assert!(
            refused.to_string().contains("llama"),
            "the refusal names the architecture: {refused}"
        );
    }
}
