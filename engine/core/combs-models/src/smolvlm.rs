//! Idefics3 / SmolVLM architecture: SigLIP vision encoder + pixel-shuffle
//! connector + Llama-family text decoder (SmolLM2), on the same
//! [`GenerativeModel`] contract. The text stack is the shared Llama trunk
//! (weights under `model.text_model.*`); the vision tower is stateless and
//! runs once per image inside `embed_multimodal`, whose output replaces the
//! `<image>` token spans in the embedded prompt.
//!
//! Weight names (HF safetensors):
//! `model.vision_model.embeddings.{patch_embedding.weight,bias}`,
//! `model.vision_model.embeddings.position_embedding.weight`,
//! `model.vision_model.encoder.layers.{i}.{layer_norm1,self_attn,layer_norm2,mlp}.*`,
//! `model.vision_model.post_layernorm.{weight,bias}`,
//! `model.connector.modality_projection.proj.weight`,
//! `model.text_model.*` (Llama layout), `lm_head.weight`.

use std::ops::Range;

use burn::tensor::{Device, Int, Tensor, TensorData, activation::softmax, backend::Backend};
use combs_formats::{ModelMetadata, ModelSource, VisionConfig};

use crate::kv::{CacheConfig, KVCache};
use crate::llama::{LlamaModel, linear, load_tensor};
use crate::matmul::safe_matmul;
use crate::norm::layer_norm;
use crate::precision::{to_f32, to_float};
use crate::traits::GenerativeModel;
use crate::{ModelError, Result};

/// One SigLIP encoder layer's weights (all projections carry biases).
struct SiglipLayer<B: Backend> {
    ln1_w: Tensor<B, 1>,
    ln1_b: Tensor<B, 1>,
    q_w: Tensor<B, 2>,
    q_b: Tensor<B, 1>,
    k_w: Tensor<B, 2>,
    k_b: Tensor<B, 1>,
    v_w: Tensor<B, 2>,
    v_b: Tensor<B, 1>,
    o_w: Tensor<B, 2>,
    o_b: Tensor<B, 1>,
    ln2_w: Tensor<B, 1>,
    ln2_b: Tensor<B, 1>,
    fc1_w: Tensor<B, 2>,
    fc1_b: Tensor<B, 1>,
    fc2_w: Tensor<B, 2>,
    fc2_b: Tensor<B, 1>,
}

/// SigLIP vision transformer (fixed square input, full self-attention).
struct SiglipEncoder<B: Backend> {
    cfg: VisionConfig,
    /// Patch-embed conv weight flattened to `[hidden, channels*patch²]`.
    patch_w: Tensor<B, 2>,
    patch_b: Tensor<B, 1>,
    /// Learned absolute position embeddings `[num_patches, hidden]`.
    pos_embed: Tensor<B, 2>,
    layers: Vec<SiglipLayer<B>>,
    post_ln_w: Tensor<B, 1>,
    post_ln_b: Tensor<B, 1>,
    scale: f64,
}

/// GELU (tanh approximation), SigLIP's `gelu_pytorch_tanh`.
fn gelu_tanh<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let inner = (x.clone() + x.clone().powf_scalar(3.0).mul_scalar(0.044715))
        .mul_scalar((2.0f64 / std::f64::consts::PI).sqrt());
    // 0.5 * x * (1 + tanh(inner))
    x * inner.tanh().add_scalar(1.0).mul_scalar(0.5)
}

impl<B: Backend> SiglipEncoder<B> {
    fn load(source: &dyn ModelSource, device: &Device<B>, cfg: &VisionConfig) -> Result<Self> {
        let p = "model.vision_model";
        // Conv weight [hidden, channels, patch, patch] -> [hidden, channels*patch²].
        let conv: Tensor<B, 4> = load_tensor(
            source,
            device,
            &format!("{p}.embeddings.patch_embedding.weight"),
        )?;
        let patch_w = conv.reshape([
            cfg.hidden_size,
            3 * cfg.patch_size * cfg.patch_size,
        ]);
        let patch_b = load_tensor(source, device, &format!("{p}.embeddings.patch_embedding.bias"))?;
        let pos_embed: Tensor<B, 2> = load_tensor(
            source,
            device,
            &format!("{p}.embeddings.position_embedding.weight"),
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let lp = format!("{p}.encoder.layers.{i}");
            layers.push(SiglipLayer {
                ln1_w: load_tensor(source, device, &format!("{lp}.layer_norm1.weight"))?,
                ln1_b: load_tensor(source, device, &format!("{lp}.layer_norm1.bias"))?,
                q_w: load_tensor(source, device, &format!("{lp}.self_attn.q_proj.weight"))?,
                q_b: load_tensor(source, device, &format!("{lp}.self_attn.q_proj.bias"))?,
                k_w: load_tensor(source, device, &format!("{lp}.self_attn.k_proj.weight"))?,
                k_b: load_tensor(source, device, &format!("{lp}.self_attn.k_proj.bias"))?,
                v_w: load_tensor(source, device, &format!("{lp}.self_attn.v_proj.weight"))?,
                v_b: load_tensor(source, device, &format!("{lp}.self_attn.v_proj.bias"))?,
                o_w: load_tensor(source, device, &format!("{lp}.self_attn.out_proj.weight"))?,
                o_b: load_tensor(source, device, &format!("{lp}.self_attn.out_proj.bias"))?,
                ln2_w: load_tensor(source, device, &format!("{lp}.layer_norm2.weight"))?,
                ln2_b: load_tensor(source, device, &format!("{lp}.layer_norm2.bias"))?,
                fc1_w: load_tensor(source, device, &format!("{lp}.mlp.fc1.weight"))?,
                fc1_b: load_tensor(source, device, &format!("{lp}.mlp.fc1.bias"))?,
                fc2_w: load_tensor(source, device, &format!("{lp}.mlp.fc2.weight"))?,
                fc2_b: load_tensor(source, device, &format!("{lp}.mlp.fc2.bias"))?,
            });
        }

        Ok(SiglipEncoder {
            scale: 1.0 / (cfg.head_dim() as f64).sqrt(),
            cfg: cfg.clone(),
            patch_w,
            patch_b,
            pos_embed,
            layers,
            post_ln_w: load_tensor(source, device, &format!("{p}.post_layernorm.weight"))?,
            post_ln_b: load_tensor(source, device, &format!("{p}.post_layernorm.bias"))?,
        })
    }

    /// Full (non-causal) self-attention over the patch sequence.
    fn attention(&self, layer: &SiglipLayer<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let cfg = &self.cfg;
        let [batch, seq, _] = x.dims();
        let heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim();

        let q = linear(x.clone(), &layer.q_w, Some(&layer.q_b))
            .reshape([batch, seq, heads, head_dim])
            .swap_dims(1, 2);
        let k = linear(x.clone(), &layer.k_w, Some(&layer.k_b))
            .reshape([batch, seq, heads, head_dim])
            .swap_dims(1, 2);
        let v = linear(x, &layer.v_w, Some(&layer.v_b))
            .reshape([batch, seq, heads, head_dim])
            .swap_dims(1, 2);

        // K dims hit the broken wgpu/Metal matmul region (>=512) — safe_matmul.
        // Scores + softmax in f32 for f16 stability (no-op in f32 builds).
        let out_dtype = q.dtype();
        let (q, k, v) = (to_f32(q), to_f32(k), to_f32(v));
        let scores = safe_matmul(q, k.transpose()).mul_scalar(self.scale);
        let ctx = to_float(safe_matmul(softmax(scores, 3), v), out_dtype);
        let ctx = ctx.swap_dims(1, 2).reshape([batch, seq, heads * head_dim]);
        linear(ctx, &layer.o_w, Some(&layer.o_b))
    }

    /// `[1, channels, image, image] -> [1, num_patches, hidden]`.
    fn forward(&self, pixels: Tensor<B, 4>) -> Tensor<B, 3> {
        let cfg = &self.cfg;
        let p = cfg.patch_size;
        let [_, c, h, w] = pixels.dims();
        let (gh, gw) = (h / p, w / p);
        debug_assert_eq!(c, 3);
        debug_assert_eq!(h % p + w % p, 0);

        // Unfold into patches (equivalent to the stride-p conv): reshape to
        // [c, gh, p, gw, p] -> [gh, gw, c, p, p] -> [gh*gw, c*p*p].
        let patches = pixels
            .reshape([c, gh, p, gw, p])
            .swap_dims(0, 1)
            .swap_dims(1, 3)
            .swap_dims(2, 3)
            .reshape([gh * gw, c * p * p])
            .unsqueeze_dim::<3>(0);
        let mut x = linear(patches, &self.patch_w, Some(&self.patch_b));

        // Fixed square input: positional ids are exactly 0..num_patches.
        let np = gh * gw;
        x = x + self
            .pos_embed
            .clone()
            .narrow(0, 0, np)
            .reshape([1, np, cfg.hidden_size]);

        for layer in &self.layers {
            let h = layer_norm(
                x.clone(),
                layer.ln1_w.clone(),
                layer.ln1_b.clone(),
                cfg.layer_norm_eps,
            );
            x = x + self.attention(layer, h);
            let h = layer_norm(
                x.clone(),
                layer.ln2_w.clone(),
                layer.ln2_b.clone(),
                cfg.layer_norm_eps,
            );
            let mlp = linear(
                gelu_tanh(linear(h, &layer.fc1_w, Some(&layer.fc1_b))),
                &layer.fc2_w,
                Some(&layer.fc2_b),
            );
            x = x + mlp;
        }

        layer_norm(x, self.post_ln_w.clone(), self.post_ln_b.clone(), cfg.layer_norm_eps)
    }
}

/// Idefics3 connector: pixel-shuffle (space-to-depth, HF ordering) followed
/// by a bias-free linear projection into the text hidden size.
struct Connector<B: Backend> {
    scale: usize,
    proj: Tensor<B, 2>, // [text_hidden, vision_hidden * scale²]
}

impl<B: Backend> Connector<B> {
    /// `[1, patches, vision_hidden] -> [1, patches/s², vision_hidden*s²]`
    /// (matches HF `Idefics3Connector.pixel_shuffle` channel ordering:
    /// channel = ((row_in_group * s) + col_in_group) * C + c).
    fn pixel_shuffle(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let s = self.scale;
        let [b, seq, c] = x.dims();
        let side = (seq as f64).sqrt() as usize;
        assert_eq!(side * side, seq, "patch grid must be square");
        x.reshape([b, side, side, c])
            .reshape([b, side, side / s, c * s])
            .swap_dims(1, 2) // [b, side/s, side, c*s]
            .reshape([b, side / s, side / s, c * s * s])
            .swap_dims(1, 2) // [b, side/s, side/s, c*s²]
            .reshape([b, seq / (s * s), c * s * s])
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        linear(self.pixel_shuffle(x), &self.proj, None)
    }
}

/// SmolVLM (Idefics3): SigLIP + connector + Llama-family text decoder.
pub struct SmolVlmModel<B: Backend> {
    metadata: ModelMetadata,
    vision_cfg: VisionConfig,
    vision: SiglipEncoder<B>,
    connector: Connector<B>,
    text: LlamaModel<B>,
}

impl<B: Backend> SmolVlmModel<B> {
    /// Runs one image through the vision tower + connector:
    /// `[1, 3, H, W] -> [1, image_seq_len, text_hidden]`.
    fn image_features(&self, pixels: Tensor<B, 4>) -> Tensor<B, 3> {
        self.connector.forward(self.vision.forward(pixels))
    }
}

impl<B: Backend> GenerativeModel<B> for SmolVlmModel<B> {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn load(source: &dyn ModelSource, device: &Device<B>) -> Result<Self> {
        let metadata = source.metadata().clone();
        let vision_cfg = metadata
            .vision
            .clone()
            .ok_or_else(|| ModelError::MissingTensor("vision_config".to_string()))?;
        let vision = SiglipEncoder::load(source, device, &vision_cfg)?;
        let proj: Tensor<B, 2> = load_tensor(
            source,
            device,
            "model.connector.modality_projection.proj.weight",
        )?;
        LlamaModel::<B>::expect_shape(
            "model.connector.modality_projection.proj.weight",
            &proj.dims(),
            &[
                metadata.hidden_size,
                vision_cfg.hidden_size * vision_cfg.scale_factor * vision_cfg.scale_factor,
            ],
        )?;
        let text = LlamaModel::<B>::load_with_prefix(source, device, "model.text_model")?;
        Ok(SmolVlmModel {
            connector: Connector {
                scale: vision_cfg.scale_factor,
                proj,
            },
            metadata,
            vision_cfg,
            vision,
            text,
        })
    }

    fn create_kv_cache(&self, config: &CacheConfig) -> Box<dyn KVCache<B>> {
        self.text.create_kv_cache(config)
    }

    fn embed(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.text.embed(tokens)
    }

    fn embed_multimodal(
        &self,
        tokens: Tensor<B, 2, Int>,
        images: &[Tensor<B, 4>],
    ) -> Result<Tensor<B, 3>> {
        if images.is_empty() {
            return Ok(self.text.embed(tokens));
        }
        let [_, seq] = tokens.dims();
        let ids: Vec<i64> = tokens
            .clone()
            .into_data()
            .convert::<i64>()
            .to_vec()
            .map_err(|e| ModelError::BadShape {
                tensor: "tokens".to_string(),
                expected: vec![1, seq],
                got: vec![],
            })
            .unwrap_or_default();
        if ids.len() != seq {
            return Err(ModelError::BadShape {
                tensor: "tokens".to_string(),
                expected: vec![1, seq],
                got: vec![ids.len()],
            });
        }

        // Consecutive spans of the image token; one span per image, in order.
        let image_id = self.vision_cfg.image_token_id as i64;
        let span_len = self.vision_cfg.image_seq_len();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i < seq {
            if ids[i] == image_id {
                let start = i;
                while i < seq && ids[i] == image_id {
                    i += 1;
                }
                spans.push((start, i));
            } else {
                i += 1;
            }
        }
        if spans.len() != images.len() {
            return Err(ModelError::UnsupportedMedia(format!(
                "found {} image-token span(s) of len {span_len} but {} image(s) were provided",
                spans.len(),
                images.len()
            )));
        }

        let base = self.text.embed(tokens);
        let mut pieces: Vec<Tensor<B, 3>> = Vec::new();
        let mut cursor = 0;
        for (idx, (start, end)) in spans.iter().enumerate() {
            if end - start != span_len {
                return Err(ModelError::UnsupportedMedia(format!(
                    "image-token span has length {}, expected {span_len}",
                    end - start
                )));
            }
            if *start > cursor {
                pieces.push(base.clone().narrow(1, cursor, start - cursor));
            }
            pieces.push(self.image_features(images[idx].clone()));
            cursor = *end;
        }
        if cursor < seq {
            pieces.push(base.narrow(1, cursor, seq - cursor));
        }
        Ok(Tensor::cat(pieces, 1))
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
        let hidden = self.text.forward_hidden(input, cache, pos.start as usize);
        self.text.last_logits(hidden)
    }

    fn decode(&mut self, input: Tensor<B, 3>, cache: &mut dyn KVCache<B>) -> Tensor<B, 2> {
        let pos = cache.seq_len();
        let hidden = self.text.forward_hidden(input, cache, pos);
        self.text.last_logits(hidden)
    }

    fn prefill_hidden(
        &mut self,
        input: Tensor<B, 3>,
        cache: &mut dyn KVCache<B>,
        pos: Range<u32>,
    ) -> Result<Tensor<B, 3>> {
        self.text.prefill_hidden(input, cache, pos)
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
        self.text.prefill_all_logits(input, cache, pos)
    }
}

/// Builds the Idefics3 prompt expansion for one image:
/// `<fake_token_around_image><global-img><image>×image_seq_len<fake_token_around_image>`.
/// (`image_seq_len` = 64 for SmolVLM-256M.) The returned string is meant to
/// replace each `<image>` placeholder in the chat text.
pub fn image_prompt_expansion(image_seq_len: usize) -> String {
    let mut s = String::with_capacity(image_seq_len * 8 + 64);
    s.push_str("<fake_token_around_image><global-img>");
    for _ in 0..image_seq_len {
        s.push_str("<image>");
    }
    s.push_str("<fake_token_around_image>");
    s
}

/// Reads token data back to host ids (helper for tests).
#[allow(dead_code)]
fn token_ids<B: Backend>(tokens: Tensor<B, 2, Int>) -> Vec<i64> {
    tokens
        .into_data()
        .convert::<i64>()
        .to_vec()
        .unwrap_or_default()
}

/// Builds a `[1, 3, H, W]` pixel tensor from planar CHW f32 data (used by the
/// runtime to hand media to `embed_multimodal`).
pub fn pixels_to_tensor<B: Backend>(
    data: Vec<f32>,
    shape: [usize; 4],
    device: &Device<B>,
) -> Tensor<B, 4> {
    Tensor::from_data(TensorData::new(data, shape), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    type TestBackend = burn::backend::NdArray<f32>;

    #[test]
    fn pixel_shuffle_matches_hf_ordering() {
        // seq=16 (4x4 grid), s=2, C=1: channel groups must follow
        // ((row_in_group * s) + col_in_group) * C + c ordering.
        let device = Default::default();
        let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let x = Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [1, 16, 1]), &device);
        let conn = Connector::<TestBackend> {
            scale: 2,
            proj: Tensor::eye(4, &device),
        };
        let out = conn.pixel_shuffle(x);
        let got: Vec<f32> = out.into_data().to_vec().unwrap();
        // Grid (row-major ids): 0 1 2 3 / 4 5 6 7 / 8 9 10 11 / 12 13 14 15
        // Token 0 (top-left 2x2 group): rows 0-1, cols 0-1 → [0,1,4,5]
        // Token 1: rows 0-1, cols 2-3 → [2,3,6,7]
        // Token 2: rows 2-3, cols 0-1 → [8,9,12,13]
        // Token 3: rows 2-3, cols 2-3 → [10,11,14,15]
        let expected: Vec<f32> = vec![
            0.0, 1.0, 4.0, 5.0, //
            2.0, 3.0, 6.0, 7.0, //
            8.0, 9.0, 12.0, 13.0, //
            10.0, 11.0, 14.0, 15.0,
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn image_prompt_expansion_shape() {
        let s = image_prompt_expansion(64);
        assert!(s.starts_with("<fake_token_around_image><global-img>"));
        assert!(s.ends_with("<fake_token_around_image>"));
        assert_eq!(s.matches("<image>").count(), 64);
    }

    #[test]
    fn gelu_tanh_reference() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 1>::from_data(TensorData::new(vec![0.0f32, 1.0, -1.0], [3]), &device);
        let y: Vec<f32> = gelu_tanh(x).into_data().to_vec().unwrap();
        // gelu(0)=0, gelu(1)≈0.8412, gelu(-1)≈-0.1588 (tanh approx).
        assert!(y[0].abs() < 1e-5);
        assert!((y[1] - 0.8412).abs() < 1e-3, "gelu(1) = {}", y[1]);
        assert!((y[2] + 0.1588).abs() < 1e-3, "gelu(-1) = {}", y[2]);
    }
}
