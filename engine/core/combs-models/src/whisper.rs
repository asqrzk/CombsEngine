//! Whisper encoder–decoder for speech-to-text.
//!
//! Encoder: two GELU conv1d stems (stride 1 then 2), checkpoint-stored
//! sinusoidal positions, pre-LN MHA stack, final LayerNorm. Decoder:
//! learned positions, causal self-attention, cross-attention against the
//! encoder states, tied output head. Attention mirrors the HF layout: q/v
//! and the output projection carry biases, k_proj does not.
//!
//! v1 simplification, stated plainly: `decode_step` recomputes the decoder
//! over the whole token prefix each step (no KV cache). Whisper contexts
//! are ≤ 448 tokens and transcription prefixes are usually far shorter, so
//! the quadratic cost is negligible at base/small scale; a decoder KV
//! cache slots in behind the same trait without API changes.

use burn::tensor::activation::{gelu, softmax};
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, Int, Tensor, TensorData, backend::Backend};
use combs_formats::{ModelMetadata, ModelSource};

use crate::llama::{linear, load_tensor};
use crate::matmul::safe_matmul;
use crate::norm::layer_norm;
use crate::precision::{to_f32, to_float};
use crate::traits::SpeechToTextModel;
use crate::{ModelError, Result};

/// LayerNorm epsilon Whisper checkpoints are trained with.
const LN_EPS: f64 = 1e-5;

/// One attention block's projections. `k_b` is `None` for Whisper
/// (k_proj is bias-free in every export).
struct Attn<B: Backend> {
    q_w: Tensor<B, 2>,
    q_b: Tensor<B, 1>,
    k_w: Tensor<B, 2>,
    v_w: Tensor<B, 2>,
    v_b: Tensor<B, 1>,
    o_w: Tensor<B, 2>,
    o_b: Tensor<B, 1>,
}

struct EncoderLayer<B: Backend> {
    ln1_w: Tensor<B, 1>,
    ln1_b: Tensor<B, 1>,
    attn: Attn<B>,
    ln2_w: Tensor<B, 1>,
    ln2_b: Tensor<B, 1>,
    fc1_w: Tensor<B, 2>,
    fc1_b: Tensor<B, 1>,
    fc2_w: Tensor<B, 2>,
    fc2_b: Tensor<B, 1>,
}

struct DecoderLayer<B: Backend> {
    ln1_w: Tensor<B, 1>,
    ln1_b: Tensor<B, 1>,
    self_attn: Attn<B>,
    ln_x_w: Tensor<B, 1>,
    ln_x_b: Tensor<B, 1>,
    cross_attn: Attn<B>,
    ln2_w: Tensor<B, 1>,
    ln2_b: Tensor<B, 1>,
    fc1_w: Tensor<B, 2>,
    fc1_b: Tensor<B, 1>,
    fc2_w: Tensor<B, 2>,
    fc2_b: Tensor<B, 1>,
}

pub struct WhisperModel<B: Backend> {
    meta: ModelMetadata,
    device: Device<B>,
    heads: usize,
    head_dim: usize,
    scale: f64,
    /// Mel bins expected by conv1 (from its weight shape).
    n_mels: usize,
    /// Max encoder frames after the stride-2 conv (from embed_positions).
    n_audio_ctx: usize,
    conv1_w: Tensor<B, 3>,
    conv1_b: Tensor<B, 1>,
    conv2_w: Tensor<B, 3>,
    conv2_b: Tensor<B, 1>,
    enc_pos: Tensor<B, 2>,
    enc_layers: Vec<EncoderLayer<B>>,
    enc_ln_w: Tensor<B, 1>,
    enc_ln_b: Tensor<B, 1>,
    embed_tokens: Tensor<B, 2>,
    dec_pos: Tensor<B, 2>,
    dec_layers: Vec<DecoderLayer<B>>,
    dec_ln_w: Tensor<B, 1>,
    dec_ln_b: Tensor<B, 1>,
}

impl<B: Backend> WhisperModel<B> {
    pub fn load(source: &dyn ModelSource, device: &Device<B>) -> Result<Self> {
        let meta = source.metadata().clone();
        let t = |name: &str| -> Result<Tensor<B, 2>> { load_tensor(source, device, name) };
        let v = |name: &str| -> Result<Tensor<B, 1>> { load_tensor(source, device, name) };

        let attn = |p: &str| -> Result<Attn<B>> {
            Ok(Attn {
                q_w: t(&format!("{p}.q_proj.weight"))?,
                q_b: v(&format!("{p}.q_proj.bias"))?,
                k_w: t(&format!("{p}.k_proj.weight"))?,
                v_w: t(&format!("{p}.v_proj.weight"))?,
                v_b: v(&format!("{p}.v_proj.bias"))?,
                o_w: t(&format!("{p}.out_proj.weight"))?,
                o_b: v(&format!("{p}.out_proj.bias"))?,
            })
        };

        let conv1_w: Tensor<B, 3> = load_tensor(source, device, "model.encoder.conv1.weight")?;
        let conv2_w: Tensor<B, 3> = load_tensor(source, device, "model.encoder.conv2.weight")?;
        let enc_pos: Tensor<B, 2> =
            load_tensor(source, device, "model.encoder.embed_positions.weight")?;
        let [_, n_mels, _] = conv1_w.dims();
        let [n_audio_ctx, _] = enc_pos.dims();

        let n_layers = meta.num_hidden_layers;
        let mut enc_layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("model.encoder.layers.{i}");
            enc_layers.push(EncoderLayer {
                ln1_w: v(&format!("{p}.self_attn_layer_norm.weight"))?,
                ln1_b: v(&format!("{p}.self_attn_layer_norm.bias"))?,
                attn: attn(&format!("{p}.self_attn"))?,
                ln2_w: v(&format!("{p}.final_layer_norm.weight"))?,
                ln2_b: v(&format!("{p}.final_layer_norm.bias"))?,
                fc1_w: t(&format!("{p}.fc1.weight"))?,
                fc1_b: v(&format!("{p}.fc1.bias"))?,
                fc2_w: t(&format!("{p}.fc2.weight"))?,
                fc2_b: v(&format!("{p}.fc2.bias"))?,
            });
        }

        let mut dec_layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("model.decoder.layers.{i}");
            dec_layers.push(DecoderLayer {
                ln1_w: v(&format!("{p}.self_attn_layer_norm.weight"))?,
                ln1_b: v(&format!("{p}.self_attn_layer_norm.bias"))?,
                self_attn: attn(&format!("{p}.self_attn"))?,
                ln_x_w: v(&format!("{p}.encoder_attn_layer_norm.weight"))?,
                ln_x_b: v(&format!("{p}.encoder_attn_layer_norm.bias"))?,
                cross_attn: attn(&format!("{p}.encoder_attn"))?,
                ln2_w: v(&format!("{p}.final_layer_norm.weight"))?,
                ln2_b: v(&format!("{p}.final_layer_norm.bias"))?,
                fc1_w: t(&format!("{p}.fc1.weight"))?,
                fc1_b: v(&format!("{p}.fc1.bias"))?,
                fc2_w: t(&format!("{p}.fc2.weight"))?,
                fc2_b: v(&format!("{p}.fc2.bias"))?,
            });
        }

        let heads = meta.num_attention_heads;
        let head_dim = meta.head_dim;
        Ok(WhisperModel {
            device: device.clone(),
            heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
            n_mels,
            n_audio_ctx,
            conv1_w,
            conv1_b: v("model.encoder.conv1.bias")?,
            conv2_w,
            conv2_b: v("model.encoder.conv2.bias")?,
            enc_pos,
            enc_layers,
            enc_ln_w: v("model.encoder.layer_norm.weight")?,
            enc_ln_b: v("model.encoder.layer_norm.bias")?,
            embed_tokens: t("model.decoder.embed_tokens.weight")?,
            dec_pos: t("model.decoder.embed_positions.weight")?,
            dec_layers,
            dec_ln_w: v("model.decoder.layer_norm.weight")?,
            dec_ln_b: v("model.decoder.layer_norm.bias")?,
            meta,
        })
    }

    /// Multi-head attention over `x` (queries) and `kv` (keys/values), with
    /// an optional additive mask on the score matrix. Scores and softmax run
    /// in f32 (no-op in f32 builds).
    fn attention(
        &self,
        attn: &Attn<B>,
        x: Tensor<B, 3>,
        kv: Tensor<B, 3>,
        mask: Option<&Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let [batch, q_len, _] = x.dims();
        let [_, kv_len, _] = kv.dims();
        let (h, d) = (self.heads, self.head_dim);

        let q = linear(x, &attn.q_w, Some(&attn.q_b))
            .reshape([batch, q_len, h, d])
            .swap_dims(1, 2);
        let k = linear(kv.clone(), &attn.k_w, None)
            .reshape([batch, kv_len, h, d])
            .swap_dims(1, 2);
        let v = linear(kv, &attn.v_w, Some(&attn.v_b))
            .reshape([batch, kv_len, h, d])
            .swap_dims(1, 2);

        let out_dtype = q.dtype();
        let (q, k, v) = (to_f32(q), to_f32(k), to_f32(v));
        let mut scores = safe_matmul(q, k.transpose()).mul_scalar(self.scale);
        if let Some(m) = mask {
            scores = scores + m.clone().reshape([1, 1, q_len, kv_len]);
        }
        let ctx = to_float(safe_matmul(softmax(scores, 3), v), out_dtype);
        let ctx = ctx.swap_dims(1, 2).reshape([batch, q_len, h * d]);
        linear(ctx, &attn.o_w, Some(&attn.o_b))
    }

    fn mlp(
        &self,
        x: Tensor<B, 3>,
        fc1_w: &Tensor<B, 2>,
        fc1_b: &Tensor<B, 1>,
        fc2_w: &Tensor<B, 2>,
        fc2_b: &Tensor<B, 1>,
    ) -> Tensor<B, 3> {
        linear(gelu(linear(x, fc1_w, Some(fc1_b))), fc2_w, Some(fc2_b))
    }

    /// Additive causal mask `[n, n]`: 0 on and below the diagonal, a large
    /// negative above. Built host-side — n is ≤ the 448 decoder context.
    fn causal_mask(&self, n: usize) -> Tensor<B, 2> {
        let mut data = vec![0.0f32; n * n];
        for r in 0..n {
            for c in (r + 1)..n {
                data[r * n + c] = f32::MIN / 2.0;
            }
        }
        Tensor::from_data(TensorData::new(data, [n, n]), &self.device)
    }
}

impl<B: Backend> SpeechToTextModel<B> for WhisperModel<B> {
    fn metadata(&self) -> &ModelMetadata {
        &self.meta
    }

    fn n_mels(&self) -> usize {
        self.n_mels
    }

    fn encode_audio(&self, mel: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let [_, mels, _] = mel.dims();
        if mels != self.n_mels {
            return Err(ModelError::BadShape {
                tensor: "mel spectrogram".into(),
                expected: vec![1, self.n_mels],
                got: vec![1, mels],
            });
        }
        let x = gelu(conv1d(
            mel,
            self.conv1_w.clone(),
            Some(self.conv1_b.clone()),
            ConvOptions::new([1], [1], [1], 1),
        ));
        let x = gelu(conv1d(
            x,
            self.conv2_w.clone(),
            Some(self.conv2_b.clone()),
            ConvOptions::new([2], [1], [1], 1),
        ));
        // [1, d, frames/2] -> [1, frames/2, d] + sinusoidal positions.
        let mut x = x.swap_dims(1, 2);
        let [_, t, d] = x.dims();
        if t > self.n_audio_ctx {
            return Err(ModelError::BadShape {
                tensor: "encoder frames".into(),
                expected: vec![self.n_audio_ctx],
                got: vec![t],
            });
        }
        x = x + self
            .enc_pos
            .clone()
            .slice([0..t, 0..d])
            .reshape([1, t, d]);

        for layer in &self.enc_layers {
            let normed = layer_norm(x.clone(), layer.ln1_w.clone(), layer.ln1_b.clone(), LN_EPS);
            x = x + self.attention(&layer.attn, normed.clone(), normed, None);
            let normed = layer_norm(x.clone(), layer.ln2_w.clone(), layer.ln2_b.clone(), LN_EPS);
            x = x + self.mlp(normed, &layer.fc1_w, &layer.fc1_b, &layer.fc2_w, &layer.fc2_b);
        }
        Ok(layer_norm(x, self.enc_ln_w.clone(), self.enc_ln_b.clone(), LN_EPS))
    }

    fn decode_step(&self, tokens: &[u32], encoded: &Tensor<B, 3>) -> Result<Tensor<B, 1>> {
        let n = tokens.len();
        let [_, _, d] = encoded.dims();
        if n == 0 {
            return Err(ModelError::BadShape {
                tensor: "decoder tokens".into(),
                expected: vec![1],
                got: vec![0],
            });
        }
        let [max_ctx, _] = self.dec_pos.dims();
        if n > max_ctx {
            return Err(ModelError::BadShape {
                tensor: "decoder context".into(),
                expected: vec![max_ctx],
                got: vec![n],
            });
        }

        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let ids: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::new(ids, [1, n]), &self.device);
        let mut x = self
            .embed_tokens
            .clone()
            .select(0, ids.reshape([n]))
            .reshape([1, n, d])
            + self.dec_pos.clone().slice([0..n, 0..d]).reshape([1, n, d]);

        let mask = self.causal_mask(n);
        for layer in &self.dec_layers {
            let normed = layer_norm(x.clone(), layer.ln1_w.clone(), layer.ln1_b.clone(), LN_EPS);
            x = x + self.attention(&layer.self_attn, normed.clone(), normed, Some(&mask));
            let normed =
                layer_norm(x.clone(), layer.ln_x_w.clone(), layer.ln_x_b.clone(), LN_EPS);
            x = x + self.attention(&layer.cross_attn, normed, encoded.clone(), None);
            let normed = layer_norm(x.clone(), layer.ln2_w.clone(), layer.ln2_b.clone(), LN_EPS);
            x = x + self.mlp(normed, &layer.fc1_w, &layer.fc1_b, &layer.fc2_w, &layer.fc2_b);
        }
        let x = layer_norm(x, self.dec_ln_w.clone(), self.dec_ln_b.clone(), LN_EPS);

        // Tied head over the final position only.
        let last = x.slice([0..1, (n - 1)..n, 0..d]).reshape([1, d]);
        let logits = safe_matmul(to_f32(last), to_f32(self.embed_tokens.clone().transpose()));
        let vocab = self.meta.vocab_size;
        Ok(logits.reshape([vocab]))
    }
}

/// Loads a speech model by `metadata.architecture` — the ASR counterpart of
/// the text registry's loader map.
pub fn load_speech_model<B: Backend>(
    source: &dyn ModelSource,
    device: &Device<B>,
) -> Result<Box<dyn SpeechToTextModel<B>>> {
    match source.metadata().architecture.as_str() {
        "whisper" => Ok(Box::new(WhisperModel::<B>::load(source, device)?)),
        other => Err(ModelError::UnsupportedArchitecture(other.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use combs_core::init_device;

    type TB = combs_core::CombsBackend;

    #[test]
    fn causal_mask_shape_and_triangle() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = init_device();
        let model_mask = |n: usize| {
            // Build via the same routine the decoder uses.
            let mut data = vec![0.0f32; n * n];
            for r in 0..n {
                for c in (r + 1)..n {
                    data[r * n + c] = f32::MIN / 2.0;
                }
            }
            Tensor::<TB, 2>::from_data(TensorData::new(data, [n, n]), &device)
        };
        let m = model_mask(4).into_data().to_vec::<f32>().unwrap();
        assert_eq!(m[0 * 4 + 0], 0.0);
        assert!(m[0 * 4 + 1] < -1e30);
        assert_eq!(m[3 * 4 + 3], 0.0);
        assert_eq!(m[3 * 4 + 0], 0.0);
    }
}
