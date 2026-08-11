//! Per-architecture resolution: turns parsed [`ModelMetadata`] into the
//! concrete knobs the universal decoder consumes — activation, norm
//! flavor, QK-norm, sandwich norms, embedding scale, logit softcaps, RoPE
//! configuration, and the per-layer attention layout.
//!
//! `combs-formats` stays parse-only; everything HF configs can't express
//! (gemma's sandwich norms, its `(1+w)` RMSNorm flavor, per-family layout
//! semantics) is encoded here, keyed on the architecture string. Adding an
//! architecture = adding a resolver arm, not decoder code.

use combs_formats::{Activation, ModelMetadata, RopeScaling};

/// RMSNorm flavor: plain (`x̂·w`) or gemma's zero-centered (`x̂·(1+w)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormFlavor {
    #[default]
    RmsNorm,
    GemmaRmsNorm,
}

/// Attention kind of one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Global,
    Sliding(usize),
}

/// Resolved architecture description consumed by the decoder.
#[derive(Debug, Clone)]
pub struct ArchSpec {
    pub activation: Activation,
    pub norm_flavor: NormFlavor,
    /// Per-head RMSNorm on Q and K after projection (gemma-3, qwen-3).
    pub qk_norm: bool,
    /// Extra post-attention / post-MLP norms around the residual adds
    /// (gemma-3 "sandwich" norms).
    pub sandwich_norms: bool,
    /// Multiply embeddings by `sqrt(hidden_size)` (gemma; computed in f32).
    pub embed_scale_sqrt_hidden: bool,
    /// Soft-capping of attention logits / final logits (gemma-2 family).
    pub attn_logit_softcap: Option<f64>,
    pub final_logit_softcap: Option<f64>,
    /// Attention scale divisor override (`1/sqrt(query_pre_attn_scalar)`
    /// instead of `1/sqrt(head_dim)`).
    pub query_pre_attn_scalar: Option<f64>,
    /// Global-layer RoPE base + scaling.
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,
    /// Sliding layers rotate with this base instead (gemma dual-RoPE).
    pub rope_local_theta: Option<f64>,
    /// One entry per transformer layer.
    pub layers: Vec<LayerKind>,
}

impl ArchSpec {
    pub fn resolve(meta: &ModelMetadata) -> Self {
        let pattern = &meta.attention_pattern;
        let n = meta.num_hidden_layers;
        let arch = meta.architecture.as_str();

        let layers: Vec<LayerKind> = match (arch, pattern.sliding_window) {
            (_, None) => vec![LayerKind::Global; n],
            // Gemma interleave: every `pattern`-th layer is global.
            ("gemma3" | "gemma3_text", Some(w)) => (0..n)
                .map(|i| {
                    if pattern.is_global_layer(i) {
                        LayerKind::Global
                    } else {
                        LayerKind::Sliding(w)
                    }
                })
                .collect(),
            // Qwen2/3 partition: the first `max_window_layers` layers are
            // full attention, the rest slide. HF defaults the split to the
            // layer count, i.e. nothing slides.
            ("qwen2" | "qwen3", Some(w)) => {
                let full = pattern.max_window_layers.unwrap_or(n);
                (0..n)
                    .map(|i| if i < full { LayerKind::Global } else { LayerKind::Sliding(w) })
                    .collect()
            }
            // Mistral v0.1 and phi-3: every layer slides (phi-3-mini ships
            // sliding_window 2047 alongside its 4k context).
            ("mistral" | "phi3", Some(w)) => vec![LayerKind::Sliding(w); n],
            // Unknown sliding semantics: run global rather than guess a
            // wrong mask (the registry guards refuse the risky cases).
            (_, Some(_)) => vec![LayerKind::Global; n],
        };

        let gemma = matches!(arch, "gemma3" | "gemma3_text");
        ArchSpec {
            // Gemma is forced: its GGUFs carry no activation key and the
            // family never uses silu.
            activation: if gemma { Activation::GeluTanh } else { meta.activation },
            norm_flavor: if gemma { NormFlavor::GemmaRmsNorm } else { NormFlavor::RmsNorm },
            qk_norm: matches!(arch, "gemma3" | "gemma3_text" | "qwen3"),
            sandwich_norms: gemma,
            embed_scale_sqrt_hidden: gemma,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            query_pre_attn_scalar: pattern.query_pre_attn_scalar,
            rope_theta: meta.rope_theta,
            rope_scaling: meta.rope_scaling.clone(),
            rope_local_theta: (gemma && pattern.sliding_window.is_some())
                .then_some(pattern.rope_local_theta),
            layers,
        }
    }

    /// Per-layer window sizes in the shape `PagedKVCache::new_with_windows`
    /// takes (`None` = global arena, `Some(w)` = rolling window).
    pub fn windows(&self) -> Vec<Option<usize>> {
        self.layers
            .iter()
            .map(|k| match k {
                LayerKind::Global => None,
                LayerKind::Sliding(w) => Some(*w),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use combs_formats::AttentionPattern;

    fn meta(arch: &str, layers: usize, pattern: AttentionPattern) -> ModelMetadata {
        let mut m = ModelMetadata::diffusion_placeholder(arch);
        m.num_hidden_layers = layers;
        m.attention_pattern = pattern;
        m
    }

    #[test]
    fn gemma_every_nth_layer_is_global() {
        let p = AttentionPattern {
            sliding_window: Some(512),
            pattern: 6,
            ..AttentionPattern::default()
        };
        let spec = ArchSpec::resolve(&meta("gemma3_text", 12, p));
        for (i, k) in spec.layers.iter().enumerate() {
            let expect = if (i + 1) % 6 == 0 { LayerKind::Global } else { LayerKind::Sliding(512) };
            assert_eq!(*k, expect, "layer {i}");
        }
        assert!(spec.qk_norm && spec.sandwich_norms && spec.embed_scale_sqrt_hidden);
        assert_eq!(spec.norm_flavor, NormFlavor::GemmaRmsNorm);
        assert_eq!(spec.rope_local_theta, Some(10_000.0));
    }

    #[test]
    fn qwen_first_n_layers_are_global() {
        let p = AttentionPattern {
            sliding_window: Some(4096),
            max_window_layers: Some(2),
            ..AttentionPattern::default()
        };
        let spec = ArchSpec::resolve(&meta("qwen2", 4, p));
        assert_eq!(
            spec.layers,
            vec![
                LayerKind::Global,
                LayerKind::Global,
                LayerKind::Sliding(4096),
                LayerKind::Sliding(4096)
            ]
        );
        // Default split = layer count: nothing slides.
        let p = AttentionPattern {
            sliding_window: Some(4096),
            ..AttentionPattern::default()
        };
        let spec = ArchSpec::resolve(&meta("qwen2", 4, p));
        assert_eq!(spec.layers, vec![LayerKind::Global; 4]);
    }

    #[test]
    fn mistral_slides_everywhere_and_llama_stays_global() {
        let p = AttentionPattern {
            sliding_window: Some(4096),
            ..AttentionPattern::default()
        };
        let spec = ArchSpec::resolve(&meta("mistral", 3, p));
        assert_eq!(spec.layers, vec![LayerKind::Sliding(4096); 3]);
        assert_eq!(spec.windows(), vec![Some(4096); 3]);

        let spec = ArchSpec::resolve(&meta("llama", 3, AttentionPattern::default()));
        assert_eq!(spec.layers, vec![LayerKind::Global; 3]);
        assert!(!spec.qk_norm && !spec.sandwich_norms);
        assert_eq!(spec.norm_flavor, NormFlavor::RmsNorm);
    }
}
