//! Qwen3.5 (`qwen35`): the hybrid family — three Gated DeltaNet layers to
//! every gated full-attention layer — on the [`GenerativeModel`] contract.
//!
//! Layout as llama.cpp's converter writes a GGUF (and as the parity fixture
//! reproduces it): every norm already carries its +1 (plain RMSNorm) except
//! the DeltaNet's gated norm; `linear_attn.A` holds -exp(A_log); V heads are
//! tiled, so key heads repeat by tiling, never interleaving. The attention
//! layer's q projection carries a per-head output gate beside each query,
//! and RoPE turns only the first `rotary_dim` of each head.
//!
//! The recurrent state lives in the session's [`HybridCache`]. The delta
//! rule is stepped token by token on both roads — correct first; the
//! chunked prefill form is its own rung, judged by time to first token.

use std::ops::Range;

use burn::tensor::{activation, backend::Backend, Device, Int, Tensor};
use combs_formats::{LinearAttention, ModelMetadata, ModelSource};

use crate::archspec::{ArchSpec, LayerKind};
use crate::kv::{
    CacheConfig, CacheKind, ContiguousKVCache, DeltaState, HybridCache, KVCache, PagedKVCache,
};
use crate::matmul::safe_matmul;
use crate::norm::rms_norm;
use crate::precision::{to_f32, to_float};
use crate::qlinear::Linear;
use crate::rope::RotaryEmbedding;
use crate::staged::WeightSupply;
use crate::traits::GenerativeModel;
use crate::{ModelError, Result};

/// The reference L2 normalization's epsilon (inside the rsqrt), fixed by
/// the family rather than the file.
const L2_EPS: f64 = 1e-6;

struct DeltaNet<B: Backend> {
    /// `[2 * key_dim + value_dim, hidden]`: q | k | v, V rows tiled.
    qkv: Linear<B>,
    /// The output gate `z`, `[value_dim, hidden]`.
    z: Linear<B>,
    /// `beta` logits and the decay's `alpha`, `[value_heads, hidden]` each.
    b: Linear<B>,
    a: Linear<B>,
    /// Depthwise causal conv over the q|k|v stream, `[channels, kernel]`.
    conv: Tensor<B, 2>,
    /// `-exp(A_log)` per value head.
    decay_a: Tensor<B, 1>,
    dt_bias: Tensor<B, 1>,
    /// Gated RMSNorm weight over `value_head_dim`.
    norm: Tensor<B, 1>,
    out: Linear<B>,
}

struct GatedAttention<B: Backend> {
    /// `[heads * 2 * head_dim, hidden]`: per head, the query then its gate.
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    q_norm: Tensor<B, 1>,
    k_norm: Tensor<B, 1>,
}

enum Mixer<B: Backend> {
    Delta(DeltaNet<B>),
    Attention(GatedAttention<B>),
}

struct Layer<B: Backend> {
    input_norm: Tensor<B, 1>,
    mixer: Mixer<B>,
    pre_mlp_norm: Tensor<B, 1>,
    gate: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
}

/// Qwen3.5 causal LM (text stack).
pub struct Qwen35Model<B: Backend> {
    metadata: ModelMetadata,
    spec: ArchSpec,
    la: LinearAttention,
    rotary_dim: usize,
    embed: crate::embed::Embedding<B>,
    lm_head: Option<Linear<B>>, // None => tied to the DENSE `embed`
    final_norm: Tensor<B, 1>,
    layers: Vec<Layer<B>>,
    rotary: RotaryEmbedding<B>,
}

fn silu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone() * activation::sigmoid(x)
}

/// `log(1 + e^x)`, stable for large |x|.
fn softplus<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone().clamp_min(0.0) + x.abs().neg().exp().log1p()
}

/// `x / sqrt(sum(x²) + eps)` over the last dim.
fn l2norm<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let inv = x
        .clone()
        .powf_scalar(2.0)
        .sum_dim(3)
        .add_scalar(L2_EPS)
        .sqrt()
        .recip();
    x * inv
}

/// `[1, seq, heads, d] -> [1, seq, heads * rep, d]` in tiled order
/// (h0 h1 … h0 h1 …), the order a converted file's V heads are in.
fn tile_heads<B: Backend>(x: Tensor<B, 4>, rep: usize) -> Tensor<B, 4> {
    if rep == 1 {
        return x;
    }
    let [b, seq, heads, d] = x.dims();
    x.unsqueeze_dim::<5>(2)
        .expand([b, seq, rep, heads, d])
        .reshape([b, seq, rep * heads, d])
}

impl<B: Backend> Qwen35Model<B> {
    fn eps(&self) -> f64 {
        self.metadata.rms_norm_eps
    }

    /// The decode step through the two WGSL dispatches (seq = 1). Asks the
    /// door first, so a refusal projects nothing and takes no state; `None`
    /// leaves the host road to run.
    fn delta_step_kernel(
        &self,
        d: &DeltaNet<B>,
        h: Tensor<B, 3>,
        cache: &mut dyn KVCache<B>,
        layer: usize,
    ) -> Option<Tensor<B, 3>> {
        let la = self.la;
        let [channels, kernel] = d.conv.dims();
        let g = crate::wgsl::StepGeometry {
            kh: la.key_heads,
            vh: la.value_heads,
            kd: la.key_head_dim,
            vd: la.value_head_dim,
            kernel,
            rms_eps: self.eps() as f32,
            l2_eps: L2_EPS as f32,
        };
        if g.channels() != channels || !crate::wgsl::deltanet_step_available::<B>(&g) {
            return None;
        }
        let out_dtype = h.dtype();
        let device = h.device();
        let hist = kernel - 1;
        let state_len = g.vh * g.kd * g.vd;
        let flat = |t: Tensor<B, 3>| {
            let n = t.dims().iter().product::<usize>();
            t.reshape([n])
        };
        let inputs_proj = (
            flat(d.qkv.forward(h.clone(), None)),
            flat(d.z.forward(h.clone(), None)),
            flat(d.b.forward(h.clone(), None)),
            flat(d.a.forward(h, None)),
        );
        let slot = cache
            .delta_state(layer)
            .expect("a qwen35 model runs on its hybrid cache");
        let (ring, state) = match slot.take() {
            Some(DeltaState { conv, state }) => {
                (conv.reshape([channels * hist]), state.reshape([state_len]))
            }
            None => (
                to_f32(Tensor::<B, 1>::zeros([channels * hist], &device)),
                to_f32(Tensor::<B, 1>::zeros([state_len], &device)),
            ),
        };
        let (qkv, z, b, a) = inputs_proj;
        let inputs = crate::wgsl::StepInputs {
            qkv,
            z,
            b,
            a,
            decay_a: d.decay_a.clone(),
            dt: d.dt_bias.clone(),
            normw: d.norm.clone(),
            conv_w: d.conv.clone().reshape([channels * kernel]),
            ring,
            state,
        };
        // The door said yes before the state left its slot; a refusal now is
        // a defect, and losing the state silently would be worse than loud.
        let (out, ring, state) = crate::wgsl::try_deltanet_step(inputs, g)
            .expect("the DeltaNet step runs where the door said it is available");
        *slot = Some(DeltaState {
            conv: ring.reshape([1, channels, hist]),
            state: state.reshape([1, g.vh, g.kd, g.vd]),
        });
        let y = to_float(out.reshape([1, 1, g.vh * g.vd]), out_dtype);
        Some(d.out.forward(y, None))
    }

    /// One Gated DeltaNet mixer over `h` (`[1, seq, hidden]`, normed),
    /// carrying the conv ring and the state matrix in `cache`.
    fn delta(
        &self,
        d: &DeltaNet<B>,
        h: Tensor<B, 3>,
        cache: &mut dyn KVCache<B>,
        layer: usize,
    ) -> Tensor<B, 3> {
        let la = self.la;
        let [_, seq, _] = h.dims();
        if seq == 1 {
            if let Some(y) = self.delta_step_kernel(d, h.clone(), cache, layer) {
                return y;
            }
        }
        let out_dtype = h.dtype();
        let (kh, vh, kd, vd) = (
            la.key_heads,
            la.value_heads,
            la.key_head_dim,
            la.value_head_dim,
        );
        let key_dim = kh * kd;
        let value_dim = vh * vd;
        let [channels, kernel] = d.conv.dims();

        let qkv = to_f32(d.qkv.forward(h.clone(), None));
        let z = to_f32(d.z.forward(h.clone(), None));
        let beta = activation::sigmoid(to_f32(d.b.forward(h.clone(), None)));
        let alpha = to_f32(d.a.forward(h, None));
        let device = qkv.device();

        let slot = cache
            .delta_state(layer)
            .expect("a qwen35 model runs on its hybrid cache");
        let (conv_prev, mut state) = match slot.take() {
            Some(DeltaState { conv, state }) => (conv, state),
            None => (
                to_f32(Tensor::<B, 3>::zeros([1, channels, kernel - 1], &device)),
                to_f32(Tensor::<B, 4>::zeros([1, vh, kd, vd], &device)),
            ),
        };

        // The causal conv sees the last kernel-1 columns, then this call's
        // tokens; torch's conv1d is a correlation, so no kernel flip.
        let xin = Tensor::cat(vec![conv_prev, qkv.swap_dims(1, 2)], 2);
        let w = to_f32(d.conv.clone());
        let mut mixed: Option<Tensor<B, 3>> = None;
        for j in 0..kernel {
            let col =
                xin.clone().narrow(2, j, seq) * w.clone().narrow(1, j, 1).reshape([1, channels, 1]);
            mixed = Some(match mixed {
                None => col,
                Some(acc) => acc + col,
            });
        }
        let mixed = silu(mixed.expect("kernel > 0")).swap_dims(1, 2);
        let conv_next = xin.narrow(2, seq, kernel - 1);

        let q = l2norm(
            mixed
                .clone()
                .narrow(2, 0, key_dim)
                .reshape([1, seq, kh, kd]),
        );
        let k = l2norm(
            mixed
                .clone()
                .narrow(2, key_dim, key_dim)
                .reshape([1, seq, kh, kd]),
        );
        let v = mixed
            .narrow(2, 2 * key_dim, value_dim)
            .reshape([1, seq, vh, vd]);
        let q = tile_heads(q, vh / kh).mul_scalar(1.0 / (kd as f64).sqrt());
        let k = tile_heads(k, vh / kh);
        let g = to_f32(d.decay_a.clone()).reshape([1, 1, vh])
            * softplus(alpha + to_f32(d.dt_bias.clone()).reshape([1, 1, vh]));
        let decay = g.exp();

        let mut outs = Vec::with_capacity(seq);
        for t in 0..seq {
            let q_t = q.clone().narrow(1, t, 1).reshape([1, vh, 1, kd]);
            let k_t = k.clone().narrow(1, t, 1).reshape([1, vh, kd]);
            let v_t = v.clone().narrow(1, t, 1).reshape([1, vh, vd]);
            let decay_t = decay.clone().narrow(1, t, 1).reshape([1, vh, 1, 1]);
            let beta_t = beta.clone().narrow(1, t, 1).reshape([1, vh, 1]);
            state = state * decay_t;
            let recalled = k_t
                .clone()
                .unsqueeze_dim::<4>(2)
                .matmul(state.clone())
                .reshape([1, vh, vd]);
            let delta = (v_t - recalled) * beta_t;
            state = state
                + k_t
                    .unsqueeze_dim::<4>(3)
                    .matmul(delta.unsqueeze_dim::<4>(2));
            outs.push(q_t.matmul(state.clone()).reshape([1, 1, vh, vd]));
        }
        let core = Tensor::cat(outs, 1);

        let gated = rms_norm(core, d.norm.clone(), self.eps()) * silu(z.reshape([1, seq, vh, vd]));
        *slot = Some(DeltaState {
            conv: conv_next,
            state,
        });
        d.out.forward(
            to_float(gated.reshape([1, seq, value_dim]), out_dtype),
            None,
        )
    }

    /// Partial RoPE: the first `rotary_dim` of each head turn, the rest pass.
    fn rotate(&self, q: Tensor<B, 4>, k: Tensor<B, 4>, pos: usize) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let hd = self.metadata.head_dim;
        let rd = self.rotary_dim;
        if rd == hd {
            return self.rotary.apply_qk(q, k, pos);
        }
        let (q_rot, q_pass) = (q.clone().narrow(3, 0, rd), q.narrow(3, rd, hd - rd));
        let (k_rot, k_pass) = (k.clone().narrow(3, 0, rd), k.narrow(3, rd, hd - rd));
        let (q_rot, k_rot) = self.rotary.apply_qk(q_rot, k_rot, pos);
        (
            Tensor::cat(vec![q_rot, q_pass], 3),
            Tensor::cat(vec![k_rot, k_pass], 3),
        )
    }

    /// One gated full-attention mixer; returns the gated context before the
    /// output projection.
    fn attend(
        &self,
        a: &GatedAttention<B>,
        h: Tensor<B, 3>,
        cache: &mut dyn KVCache<B>,
        layer: usize,
        pos: usize,
    ) -> Tensor<B, 3> {
        let m = &self.metadata;
        let [_, seq, _] = h.dims();
        let (heads, kvh, hd) = (m.num_attention_heads, m.num_key_value_heads, m.head_dim);
        let qg =
            a.q.forward(h.clone(), None)
                .reshape([1, seq, heads, 2 * hd]);
        let q = qg.clone().narrow(3, 0, hd).swap_dims(1, 2);
        let gate = qg.narrow(3, hd, hd).reshape([1, seq, heads * hd]);
        let k =
            a.k.forward(h.clone(), None)
                .reshape([1, seq, kvh, hd])
                .swap_dims(1, 2);
        let v =
            a.v.forward(h, None)
                .reshape([1, seq, kvh, hd])
                .swap_dims(1, 2);
        let q = rms_norm(q, a.q_norm.clone(), self.eps());
        let k = rms_norm(k, a.k_norm.clone(), self.eps());
        let (q, k) = self.rotate(q, k, pos);
        let ctx = cache.attention_opts(layer, q, k, v, pos, 1.0 / (hd as f64).sqrt(), None);
        let ctx = ctx.swap_dims(1, 2).reshape([1, seq, heads * hd]);
        let out_dtype = ctx.dtype();
        to_float(to_f32(ctx) * activation::sigmoid(to_f32(gate)), out_dtype)
    }

    fn forward_hidden(
        &self,
        mut x: Tensor<B, 3>,
        cache: &mut dyn KVCache<B>,
        pos: usize,
    ) -> Tensor<B, 3> {
        for (i, layer) in self.layers.iter().enumerate() {
            let h = rms_norm(x.clone(), layer.input_norm.clone(), self.eps());
            x = match &layer.mixer {
                Mixer::Delta(d) => {
                    let y = self.delta(d, h, cache, i);
                    x + y
                }
                Mixer::Attention(a) => {
                    let ctx = self.attend(a, h, cache, i, pos);
                    a.o.forward_add(ctx, None, x)
                }
            };
            let h = rms_norm(x.clone(), layer.pre_mlp_norm.clone(), self.eps());
            let gated = silu(layer.gate.forward(h.clone(), None)) * layer.up.forward(h, None);
            x = layer.down.forward_add(gated, None, x);
        }
        rms_norm(x, self.final_norm.clone(), self.eps())
    }

    fn head(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, seq, hidden_size] = hidden.dims();
        match &self.lm_head {
            Some(head) => head.forward(hidden, None),
            None => {
                let table = self
                    .embed
                    .dense()
                    .expect("lm_head: None implies a dense embedding");
                let out = safe_matmul(
                    hidden.reshape([seq, hidden_size]),
                    table.clone().transpose(),
                );
                let [_, vocab] = out.dims();
                out.reshape([1, seq, vocab])
            }
        }
    }

    fn last_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 2> {
        let [_, seq, _] = hidden.dims();
        let logits = self.head(hidden.narrow(1, seq - 1, 1));
        let [_, _, vocab] = logits.dims();
        logits.reshape([1, vocab])
    }

    fn load_from(supply: &mut WeightSupply<'_, B>, device: &Device<B>) -> Result<Self> {
        let m = supply.metadata().clone();
        let spec = ArchSpec::resolve(&m);
        let la = m.linear_attention.ok_or_else(|| {
            ModelError::Unsupported("qwen35 file carries no linear-attention geometry".into())
        })?;
        let shape = |name: &str, got: &[usize], want: &[usize]| {
            if got == want {
                Ok(())
            } else {
                Err(ModelError::BadShape {
                    tensor: name.to_string(),
                    expected: want.to_vec(),
                    got: got.to_vec(),
                })
            }
        };

        let (embed, packed_head) = supply.embedding(device, "model.embed_tokens.weight")?;
        shape(
            "embed_tokens.weight",
            &embed.dims(),
            &[m.vocab_size, m.hidden_size],
        )?;
        let lm_head = if m.tie_word_embeddings {
            packed_head.map(Linear::Quant)
        } else {
            Some(supply.linear(device, "lm_head.weight")?)
        };
        let final_norm = supply.weight(device, "model.norm.weight")?;

        let key_dim = la.key_heads * la.key_head_dim;
        let value_dim = la.value_heads * la.value_head_dim;
        let channels = 2 * key_dim + value_dim;
        let mut layers = Vec::with_capacity(m.num_hidden_layers);
        for (i, kind) in spec.layers.iter().enumerate() {
            let p = format!("model.layers.{i}");
            let mixer = match kind {
                LayerKind::DeltaNet => {
                    let l = format!("{p}.linear_attn");
                    let qkv = supply.linear(device, &format!("{l}.in_proj_qkv.weight"))?;
                    shape(
                        &format!("{l}.in_proj_qkv.weight"),
                        &qkv.dims(),
                        &[channels, m.hidden_size],
                    )?;
                    let conv: Tensor<B, 2> =
                        supply.weight(device, &format!("{l}.conv1d.weight"))?;
                    shape(
                        &format!("{l}.conv1d.weight"),
                        &conv.dims(),
                        &[channels, la.conv_kernel],
                    )?;
                    Mixer::Delta(DeltaNet {
                        qkv,
                        z: supply.linear(device, &format!("{l}.in_proj_z.weight"))?,
                        b: supply.linear(device, &format!("{l}.in_proj_b.weight"))?,
                        a: supply.linear(device, &format!("{l}.in_proj_a.weight"))?,
                        conv,
                        decay_a: supply.weight(device, &format!("{l}.A"))?,
                        dt_bias: supply.weight(device, &format!("{l}.dt_bias"))?,
                        norm: supply.weight(device, &format!("{l}.norm.weight"))?,
                        out: supply.linear(device, &format!("{l}.out_proj.weight"))?,
                    })
                }
                _ => {
                    let s = format!("{p}.self_attn");
                    let q = supply.linear(device, &format!("{s}.q_proj.weight"))?;
                    shape(
                        &format!("{s}.q_proj.weight"),
                        &q.dims(),
                        &[m.num_attention_heads * 2 * m.head_dim, m.hidden_size],
                    )?;
                    Mixer::Attention(GatedAttention {
                        q,
                        k: supply.linear(device, &format!("{s}.k_proj.weight"))?,
                        v: supply.linear(device, &format!("{s}.v_proj.weight"))?,
                        o: supply.linear(device, &format!("{s}.o_proj.weight"))?,
                        q_norm: supply.weight(device, &format!("{s}.q_norm.weight"))?,
                        k_norm: supply.weight(device, &format!("{s}.k_norm.weight"))?,
                    })
                }
            };
            layers.push(Layer {
                input_norm: supply.weight(device, &format!("{p}.input_layernorm.weight"))?,
                mixer,
                pre_mlp_norm: supply
                    .weight(device, &format!("{p}.post_attention_layernorm.weight"))?,
                gate: supply.linear(device, &format!("{p}.mlp.gate_proj.weight"))?,
                up: supply.linear(device, &format!("{p}.mlp.up_proj.weight"))?,
                down: supply.linear(device, &format!("{p}.mlp.down_proj.weight"))?,
            });
            combs_core::progress::load(
                "weights",
                Some((i + 1) as u64),
                Some(m.num_hidden_layers as u64),
                None,
            );
        }

        let rotary_dim = m.rotary_dim.unwrap_or(m.head_dim);
        let rotary = RotaryEmbedding::new_scaled(
            rotary_dim,
            spec.rope_theta,
            m.max_position_embeddings,
            &spec.rope_scaling,
            device,
        );
        Ok(Qwen35Model {
            metadata: m,
            spec,
            la,
            rotary_dim,
            embed,
            lm_head,
            final_norm,
            layers,
            rotary,
        })
    }
}

impl<B: Backend> GenerativeModel<B> for Qwen35Model<B> {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn load(source: &dyn ModelSource, device: &Device<B>) -> Result<Self> {
        Self::load_from(&mut WeightSupply::Source(source), device)
    }

    fn create_kv_cache(&self, config: &CacheConfig) -> Box<dyn KVCache<B>> {
        let attends: Vec<bool> = self
            .spec
            .layers
            .iter()
            .map(|k| !matches!(k, LayerKind::DeltaNet))
            .collect();
        let n = attends.iter().filter(|a| **a).count();
        let inner: Box<dyn KVCache<B>> = match config.kind {
            CacheKind::Contiguous => Box::new(ContiguousKVCache::<B>::new(n)),
            CacheKind::Paged => Box::new(PagedKVCache::<B>::new_with_windows(
                n,
                *config,
                vec![None; n],
            )),
        };
        Box::new(HybridCache::new(inner, &attends))
    }

    fn embed(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embed.gather(tokens)
    }

    fn prefill(
        &mut self,
        input: Tensor<B, 3>,
        cache: &mut dyn KVCache<B>,
        pos: Range<u32>,
    ) -> Tensor<B, 2> {
        let hidden = self.forward_hidden(input, cache, pos.start as usize);
        self.last_logits(hidden)
    }

    fn prefill_hidden(
        &mut self,
        input: Tensor<B, 3>,
        cache: &mut dyn KVCache<B>,
        pos: Range<u32>,
    ) -> Result<Tensor<B, 3>> {
        Ok(self.forward_hidden(input, cache, pos.start as usize))
    }

    fn prefill_all_logits(
        &mut self,
        input: Tensor<B, 3>,
        cache: &mut dyn KVCache<B>,
        pos: Range<u32>,
    ) -> Result<Tensor<B, 3>> {
        let hidden = self.forward_hidden(input, cache, pos.start as usize);
        Ok(self.head(hidden))
    }

    fn decode(&mut self, input: Tensor<B, 3>, cache: &mut dyn KVCache<B>) -> Tensor<B, 2> {
        let pos = cache.seq_len();
        let hidden = self.forward_hidden(input, cache, pos);
        self.last_logits(hidden)
    }
}
