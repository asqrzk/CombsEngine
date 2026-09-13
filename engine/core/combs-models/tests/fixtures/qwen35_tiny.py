#!/usr/bin/env python3
"""Writes the tiny Qwen3.5 parity fixture read by tests/qwen35_parity.rs.

A 4-layer model (three Gated DeltaNet layers, one gated full-attention
layer) with seeded random weights is built with transformers' Qwen3.5 code,
then converted to GGUF the way llama.cpp's converter writes a real file:
norm weights +1 (the gated DeltaNet norm excepted), A = -exp(A_log), the
conv1d kernel squeezed, and every V head reordered from grouped to tiled
order (qkv's V rows, z, a, b, A, dt_bias, the conv's V channels, out's
columns). The reference logits are recorded for a full prefill and for a
prefill of the first PREFIX tokens followed by one-token decodes.

Run from this directory with a Python that has torch and a transformers
release carrying Qwen3.5; the engine's tests only read the two outputs.
"""
import copy
import json
import struct

import numpy as np
import torch
import transformers
from transformers.models.qwen3_5 import configuration_qwen3_5 as qcfg
from transformers.models.qwen3_5 import modeling_qwen3_5 as qmod

HIDDEN, FFN, LAYERS, HEADS, KV_HEADS, HEAD_DIM = 64, 128, 4, 4, 2, 32
K_HEADS, V_HEADS, K_DIM, V_DIM, CONV = 2, 4, 16, 16, 4
VOCAB, CTX, EPS, THETA, PARTIAL = 128, 512, 1e-6, 1.0e7, 0.25
SECTIONS = [1, 1, 2]
IDS = [1, 5, 9, 42, 7, 100, 3, 77]
PREFIX = 5

cfg = qcfg.Qwen3_5TextConfig(
    hidden_size=HIDDEN, intermediate_size=FFN, num_hidden_layers=LAYERS,
    num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM,
    linear_num_key_heads=K_HEADS, linear_num_value_heads=V_HEADS,
    linear_key_head_dim=K_DIM, linear_value_head_dim=V_DIM, linear_conv_kernel_dim=CONV,
    vocab_size=VOCAB, max_position_embeddings=CTX, rms_norm_eps=EPS,
    tie_word_embeddings=True, full_attention_interval=4,
    rope_parameters={"rope_type": "default", "rope_theta": THETA,
                     "partial_rotary_factor": PARTIAL, "mrope_section": SECTIONS,
                     "mrope_interleaved": True},
)
torch.manual_seed(1234)
model = qmod.Qwen3_5ForCausalLM(cfg).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if p.ndim == 2:
            torch.nn.init.normal_(p, std=0.5 if "embed" in name else 0.15)
        elif name.endswith("norm.weight") or name.endswith("dt_bias"):
            p.add_(torch.randn_like(p) * 0.2)
        elif name.endswith("A_log"):
            p.copy_(torch.log(torch.empty_like(p).uniform_(0.5, 8.0)))
        elif name.endswith("conv1d.weight"):
            torch.nn.init.normal_(p, std=0.4)

ids = torch.tensor([IDS])
with torch.no_grad():
    full = model(ids).logits[0].float()
    out = model(ids[:, :PREFIX], use_cache=True)
    past = out.past_key_values
    steps = [out.logits[0, -1].float()]
    for t in range(PREFIX, len(IDS)):
        o = model(ids[:, t:t + 1], past_key_values=past, use_cache=True)
        past = o.past_key_values
        steps.append(o.logits[0, -1].float())
chunk_vs_recurrent = max(float((full[PREFIX - 1 + i] - s).abs().max()) for i, s in enumerate(steps))

# The f16 build stores every dense weight in f16, rounding each converted
# value once at load (a norm after its +1, A as -exp(A_log)). This twin keeps
# f32 arithmetic but holds the weights as that build does, so the f16 build
# is compared against what it actually holds.
twin = copy.deepcopy(model)
with torch.no_grad():
    for name, p in twin.named_parameters():
        if name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
            p.copy_((p + 1.0).half().float() - 1.0)
        elif name.endswith("A_log"):
            p.copy_(torch.log(-((-torch.exp(p)).half().float())))
        else:
            p.copy_(p.half().float())
    full16 = twin(ids).logits[0].float()
    out = twin(ids[:, :PREFIX], use_cache=True)
    past = out.past_key_values
    steps16 = [out.logits[0, -1].float()]
    for t in range(PREFIX, len(IDS)):
        o = twin(ids[:, t:t + 1], past_key_values=past, use_cache=True)
        past = o.past_key_values
        steps16.append(o.logits[0, -1].float())
weight_rounding_moves = float((full - full16).abs().max())


def reorder_v(t, dim, head_dim):
    """Grouped-by-K-head V heads to tiled order, as llama.cpp's converter."""
    shape = list(t.shape)
    dim %= len(shape)
    grouped = shape[:dim] + [K_HEADS, V_HEADS // K_HEADS, head_dim] + shape[dim + 1:]
    t = t.reshape(grouped)
    perm = list(range(len(grouped)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return np.ascontiguousarray(t.transpose(perm)).reshape(shape)


sd = {k: v.detach().float().numpy().copy() for k, v in model.state_dict().items()}
tensors = [("token_embd.weight", sd["model.embed_tokens.weight"]),
           ("output_norm.weight", sd["model.norm.weight"] + 1.0)]
qk = 2 * K_HEADS * K_DIM
for i in range(LAYERS):
    p = f"model.layers.{i}."
    b = f"blk.{i}."
    tensors += [(b + "attn_norm.weight", sd[p + "input_layernorm.weight"] + 1.0),
                (b + "post_attention_norm.weight", sd[p + "post_attention_layernorm.weight"] + 1.0)]
    if cfg.layer_types[i] == "linear_attention":
        la = p + "linear_attn."
        qkv = sd[la + "in_proj_qkv.weight"]
        qkv = np.concatenate([qkv[:qk], reorder_v(qkv[qk:], 0, V_DIM)], axis=0)
        conv = sd[la + "conv1d.weight"].squeeze(1)
        conv = np.concatenate([conv[:qk], reorder_v(conv[qk:], 0, V_DIM)], axis=0)
        tensors += [
            (b + "attn_qkv.weight", qkv),
            (b + "attn_gate.weight", reorder_v(sd[la + "in_proj_z.weight"], 0, V_DIM)),
            (b + "ssm_beta.weight", reorder_v(sd[la + "in_proj_b.weight"], 0, 1)),
            (b + "ssm_alpha.weight", reorder_v(sd[la + "in_proj_a.weight"], 0, 1)),
            (b + "ssm_a", -np.exp(reorder_v(sd[la + "A_log"][:, None], 0, 1)[:, 0])),
            (b + "ssm_dt.bias", reorder_v(sd[la + "dt_bias"][:, None], 0, 1)[:, 0]),
            (b + "ssm_conv1d.weight", conv),
            (b + "ssm_norm.weight", sd[la + "norm.weight"]),
            (b + "ssm_out.weight", reorder_v(sd[la + "out_proj.weight"], 1, V_DIM)),
        ]
    else:
        sa = p + "self_attn."
        tensors += [
            (b + "attn_q.weight", sd[sa + "q_proj.weight"]),
            (b + "attn_k.weight", sd[sa + "k_proj.weight"]),
            (b + "attn_v.weight", sd[sa + "v_proj.weight"]),
            (b + "attn_output.weight", sd[sa + "o_proj.weight"]),
            (b + "attn_q_norm.weight", sd[sa + "q_norm.weight"] + 1.0),
            (b + "attn_k_norm.weight", sd[sa + "k_norm.weight"] + 1.0),
        ]
    tensors += [(b + "ffn_gate.weight", sd[p + "mlp.gate_proj.weight"]),
                (b + "ffn_up.weight", sd[p + "mlp.up_proj.weight"]),
                (b + "ffn_down.weight", sd[p + "mlp.down_proj.weight"])]

U32, I32, F32, STR, ARR = 4, 5, 6, 8, 9
kv = [("general.architecture", STR, "qwen35"), ("general.alignment", U32, 32),
      ("qwen35.block_count", U32, LAYERS), ("qwen35.context_length", U32, CTX),
      ("qwen35.embedding_length", U32, HIDDEN), ("qwen35.feed_forward_length", U32, FFN),
      ("qwen35.attention.head_count", U32, HEADS), ("qwen35.attention.head_count_kv", U32, KV_HEADS),
      ("qwen35.attention.key_length", U32, HEAD_DIM), ("qwen35.attention.value_length", U32, HEAD_DIM),
      ("qwen35.attention.layer_norm_rms_epsilon", F32, EPS), ("qwen35.rope.freq_base", F32, THETA),
      ("qwen35.rope.dimension_count", U32, int(HEAD_DIM * PARTIAL)),
      ("qwen35.rope.dimension_sections", ARR, (I32, SECTIONS + [0])),
      ("qwen35.ssm.conv_kernel", U32, CONV), ("qwen35.ssm.state_size", U32, K_DIM),
      ("qwen35.ssm.group_count", U32, K_HEADS), ("qwen35.ssm.time_step_rank", U32, V_HEADS),
      ("qwen35.ssm.inner_size", U32, V_HEADS * V_DIM), ("qwen35.full_attention_interval", U32, 4),
      ("qwen35.vocab_size", U32, VOCAB),
      # a byte-level vocabulary is enough for a logits fixture: the engine
      # synthesizes its tokenizer from these keys and never tokenizes here
      ("tokenizer.ggml.model", STR, "gpt2"), ("tokenizer.ggml.pre", STR, "qwen35"),
      ("tokenizer.ggml.tokens", ARR, (STR, [f"<t{i}>" for i in range(VOCAB)])),
      ("tokenizer.ggml.token_type", ARR, (I32, [1] * VOCAB)),
      ("tokenizer.ggml.eos_token_id", U32, VOCAB - 1)]


def pstr(s):
    e = s.encode()
    return struct.pack("<Q", len(e)) + e


def pval(ty, v):
    if ty == U32:
        return struct.pack("<I", v)
    if ty == I32:
        return struct.pack("<i", v)
    if ty == F32:
        return struct.pack("<f", v)
    if ty == STR:
        return pstr(v)
    ety, items = v
    return struct.pack("<IQ", ety, len(items)) + b"".join(pval(ety, x) for x in items)


align = lambda n: (n + 31) // 32 * 32
head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
head += b"".join(pstr(k) + struct.pack("<I", ty) + pval(ty, v) for k, ty, v in kv)
data, offset = b"", 0
for name, arr in tensors:
    arr = np.ascontiguousarray(arr, dtype="<f4")
    offset = align(offset)
    head += pstr(name) + struct.pack("<I", arr.ndim) + b"".join(struct.pack("<Q", d) for d in arr.shape[::-1])
    head += struct.pack("<IQ", 0, offset)
    data += b"\0" * (offset - len(data)) + arr.tobytes()
    offset += arr.nbytes
blob = head + b"\0" * (align(len(head)) - len(head)) + data
open("qwen35_tiny.gguf", "wb").write(blob)
json.dump({
    "reference": f"transformers {transformers.__version__}, torch {torch.__version__}, float32, cpu",
    "ids": IDS, "prefix": PREFIX, "vocab": VOCAB,
    "logits": [round(float(x), 7) for x in full.flatten()],
    "step_logits": [[round(float(x), 7) for x in s] for s in steps],
    "reference_chunked_vs_recurrent_max_abs": chunk_vs_recurrent,
    "logits_f16_weights": [round(float(x), 7) for x in full16.flatten()],
    "step_logits_f16_weights": [[round(float(x), 7) for x in s] for s in steps16],
    "f16_weight_rounding_moves_logits_max_abs": weight_rounding_moves,
}, open("qwen35_tiny.json", "w"))
print(f"wrote qwen35_tiny.gguf ({len(blob)} B, {len(tensors)} tensors) and qwen35_tiny.json; "
      f"layer types {cfg.layer_types}; reference chunked-vs-recurrent max |d| {chunk_vs_recurrent:.2e}; "
      f"logit range {float(full.min()):.3f}..{float(full.max()):.3f}; "
      f"f16-rounded weights alone move the logits by max |d| {weight_rounding_moves:.3e}")
