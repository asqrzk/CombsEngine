#!/usr/bin/env python3
"""Reference activations for the combs-diffusion component-parity harness.

Loads the LOCAL stable-diffusion-v1-5 components with diffusers/transformers
(torch CPU, f32), feeds DETERMINISTIC inputs (sin patterns — never RNG), and
dumps f32 binaries + a manifest. The Rust side (combs-diffusion/tests/
parity.rs, gated on COMBS_DIFFUSION_PARITY_DIR) reads the same inputs from
the dump — no cross-language math — and compares per stage. The first
failing stage is the bug.

Usage:
  ~/venv-diff/bin/python tools/goldens/gen_diffusion_reference.py <out-dir>
"""
import json
import os
import sys

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.manual_seed(0)  # not used for inputs; silences any lazy-init noise

MODEL = os.path.expanduser("~/.cache/combs/models/stable-diffusion-v1-5")
OUT = sys.argv[1] if len(sys.argv) > 1 else "diffusion-parity"
os.makedirs(OUT, exist_ok=True)

from diffusers import AutoencoderKL, UNet2DConditionModel  # noqa: E402
from transformers import CLIPTextModel, CLIPTokenizerFast  # noqa: E402

unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=torch.float32)
vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
text = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=torch.float32)
unet.eval()
vae.eval()
text.eval()

manifest = []


def det(shape, scale=1.0, phase=0.0):
    """Deterministic full-range pattern; matches nothing, depends on nothing."""
    n = int(np.prod(shape))
    x = np.sin(np.arange(n, dtype=np.float64) * 0.0137 + phase) * scale
    return torch.tensor(x.reshape(shape), dtype=torch.float32)


def dump(name, t):
    a = t.detach().cpu().numpy()
    a.tofile(os.path.join(OUT, f"{name}.bin"))
    manifest.append({"name": name, "shape": list(a.shape), "dtype": str(a.dtype)})
    return t


# (a) timestep projection + MLP at t=951 (the first DDIM step of a 50-step
# run) — catches sinusoid frequency/direction and the time MLP.
t = torch.tensor([951], dtype=torch.int64)
tp = unet.time_proj(t).to(torch.float32)
dump("time_proj_951", tp)
dump("time_embedding_951", unet.time_embedding(tp))

# (b) CLIP: ids for a fixed prompt + last hidden state, plus internal
# activations via forward hooks (version-proof) to bisect a divergence.
tok = CLIPTokenizerFast.from_pretrained(MODEL, subfolder="tokenizer")
enc = tok(
    "a red apple on a wooden table",
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="pt",
)
ids = enc.input_ids
dump("clip_ids", ids.to(torch.int32))

acts = {}


def hook(name):
    def fn(_mod, _inp, out):
        acts[name] = out[0] if isinstance(out, tuple) else out

    return fn


def input_hook(name):
    def fn(_mod, inp, _out):
        acts[name] = inp[0]

    return fn


# transformers <5 wraps the transformer in `.text_model`; v5 flattens it.
tm = getattr(text, "text_model", text)
tm.embeddings.register_forward_hook(hook("clip_emb_out"))
for i in (0, 5, 11):
    tm.encoder.layers[i].register_forward_hook(hook(f"clip_layer{i}_out"))
l0 = tm.encoder.layers[0]
l0.layer_norm1.register_forward_hook(hook("clip_l0_ln1"))
l0.self_attn.register_forward_hook(hook("clip_l0_attn"))
l0.mlp.register_forward_hook(hook("clip_l0_mlp"))
# Attention internals: the three projections and the merged context that
# feeds out_proj (its INPUT), to bisect inside the attention module.
l0.self_attn.q_proj.register_forward_hook(hook("clip_l0_q"))
l0.self_attn.k_proj.register_forward_hook(hook("clip_l0_k"))
l0.self_attn.v_proj.register_forward_hook(hook("clip_l0_v"))
l0.self_attn.out_proj.register_forward_hook(input_hook("clip_l0_ctx"))

hidden = text(ids).last_hidden_state
dump("clip_hidden", hidden)
for name, act in acts.items():
    dump(name, act)

# (f) full UNet noise prediction on a fixed latent, conditioned on the
# REFERENCE hidden state (so UNet parity is independent of CLIP parity) —
# with per-block taps matching UNet2DConditionModel::forward_traced.
unet.conv_in.register_forward_hook(hook("unet_conv_in"))
unet.down_blocks[0].resnets[0].register_forward_hook(hook("unet_d0_res0"))
unet.down_blocks[0].attentions[0].register_forward_hook(hook("unet_d0_attn0"))
for i, blk in enumerate(unet.down_blocks):
    blk.register_forward_hook(hook(f"unet_down{i}"))
unet.mid_block.register_forward_hook(hook("unet_mid"))
for i, blk in enumerate(unet.up_blocks):
    blk.register_forward_hook(hook(f"unet_up{i}"))

lat = det((1, 4, 64, 64))
dump("latent_in", lat)
dump("unet_out_951", unet(lat, torch.tensor(951), encoder_hidden_states=hidden).sample)
for name in list(acts):
    if name.startswith("unet_"):
        dump(name, acts.pop(name))

# (g) VAE decode of a fixed latent (diffusers decode() includes
# post_quant_conv), pattern scaled to typical latent magnitude.
vlat = det((1, 4, 32, 32), scale=2.0, phase=1.0)
dump("vae_latent", vlat)
dump("vae_out", vae.decode(vlat).sample)

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print(f"wrote {len(manifest)} arrays to {OUT}")
