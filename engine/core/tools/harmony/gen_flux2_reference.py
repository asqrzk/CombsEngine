#!/usr/bin/env python3
"""Reference activations for the Flux2 transformer parity harness.

Instantiates a TINY randomly-initialized Flux2Transformer2DModel (seeded,
then SAVED — the Rust side loads the identical weights, so no
cross-language RNG dependency), feeds DETERMINISTIC inputs (sin patterns
— never RNG), and dumps f32 binaries + a manifest + the weights as
safetensors. The Rust side (combs-diffusion/tests/flux2_parity.rs, gated
on COMBS_FLUX2_PARITY_DIR) reads the same inputs from the dump and
compares per stage; the first failing stage is the bug.

Usage:
  ~/.cache/combs/ref-venv/bin/python tools/harmony/gen_flux2_reference.py <out-dir>
"""
import json
import os
import sys

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.manual_seed(0)

OUT = sys.argv[1] if len(sys.argv) > 1 else "flux2-parity"
os.makedirs(OUT, exist_ok=True)

from diffusers import Flux2Transformer2DModel  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

# Tiny, klein-shaped: double+single streams, no guidance embedding,
# 4-axis rope summing to head_dim.
CFG = dict(
    patch_size=1,
    in_channels=16,
    num_layers=2,
    num_single_layers=2,
    attention_head_dim=8,
    num_attention_heads=4,
    joint_attention_dim=24,
    mlp_ratio=3.0,
    axes_dims_rope=(2, 2, 2, 2),
    rope_theta=2000,
    guidance_embeds=False,
)
model = Flux2Transformer2DModel(**CFG)
model.eval()

sd = model.state_dict()
save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(OUT, "model.safetensors"))
with open(os.path.join(OUT, "state_dict.json"), "w") as f:
    json.dump({k: list(v.shape) for k, v in sd.items()}, f, indent=1)
print(f"[weights] {len(sd)} tensors saved")

manifest = []


def det(shape, scale=1.0, phase=0.0):
    n = int(np.prod(shape))
    x = np.sin(np.arange(n, dtype=np.float64) * 0.0137 + phase) * scale
    return torch.tensor(x.reshape(shape), dtype=torch.float32)


def dump(name, t):
    a = t.detach().cpu().numpy().astype(np.float32)
    a.tofile(os.path.join(OUT, f"{name}.bin"))
    manifest.append({"name": name, "shape": list(a.shape), "dtype": "float32"})
    return t


# Inputs: a 3x4 latent grid (12 image tokens, packed channels 16) and 5
# text tokens. ids follow the pipeline scheme: image (T=0, H=row, W=col,
# L=0), text (T=H=W=0, L=0..seq-1).
H, W, TXT = 3, 4, 5
img = det((1, H * W, CFG["in_channels"]), scale=0.7)
txt = det((1, TXT, CFG["joint_attention_dim"]), scale=0.5, phase=1.0)
img_ids = torch.tensor(
    [[0, h, w, 0] for h in range(H) for w in range(W)], dtype=torch.float32
)
txt_ids = torch.tensor([[0, 0, 0, l] for l in range(TXT)], dtype=torch.float32)
timestep = torch.tensor([0.75], dtype=torch.float32)  # pipeline passes t/1000

dump("input_img", img)
dump("input_txt", txt)
dump("input_img_ids", img_ids)
dump("input_txt_ids", txt_ids)

captures = {}


def hook(name):
    def fn(_mod, _inp, out):
        if isinstance(out, tuple):
            for i, o in enumerate(out):
                if torch.is_tensor(o):
                    captures[f"{name}_out{i}"] = o
        elif torch.is_tensor(out):
            captures[f"{name}_out0"] = out
    return fn


for i, blk in enumerate(model.transformer_blocks):
    blk.register_forward_hook(hook(f"double_{i}"))
for i, blk in enumerate(model.single_transformer_blocks):
    blk.register_forward_hook(hook(f"single_{i}"))
for name in ["x_embedder", "context_embedder", "time_guidance_embed", "pos_embed", "norm_out"]:
    mod = getattr(model, name, None)
    if mod is not None:
        mod.register_forward_hook(hook(name))

kwargs = dict(
    hidden_states=img,
    encoder_hidden_states=txt,
    timestep=timestep,
    img_ids=img_ids,
    txt_ids=txt_ids,
    return_dict=False,
)
try:
    out = model(**kwargs)[0]
except TypeError as e:
    print(f"[forward] retrying with guidance kwarg: {e}")
    out = model(**kwargs, guidance=None)[0]

dump("output", out)
for name, t in captures.items():
    dump(name, t)

# ---- tiny flux2 autoencoder: decode path + latent bn round-trip -----
from diffusers import AutoencoderKLFlux2  # noqa: E402

# The REAL klein autoencoder geometry (block widths, groups, 32-ch
# latent) so the shipped decoder path is exercised unmodified; only
# the weights are random. ~200 MB on disk, local-only.
VAE_CFG = dict(
    in_channels=3,
    out_channels=3,
    block_out_channels=(128, 256, 512, 512),
    layers_per_block=2,
    latent_channels=32,
    norm_num_groups=32,
    sample_size=64,
    patch_size=(2, 2),
)
vae = AutoencoderKLFlux2(**VAE_CFG)
vae.eval()
# Give the bn running stats non-trivial deterministic values (a fresh
# module has mean 0 / var 1, which would hide denorm bugs).
n_bn = 4 * VAE_CFG["latent_channels"]
vae.bn.running_mean.copy_(det((n_bn,), scale=0.3).squeeze())
vae.bn.running_var.copy_((det((n_bn,), scale=0.4, phase=2.0).squeeze() + 1.1).abs())

os.makedirs(os.path.join(OUT, "vae"), exist_ok=True)
vae_sd = vae.state_dict()
save_file(
    {k: v.contiguous() for k, v in vae_sd.items()},
    os.path.join(OUT, "vae", "model.safetensors"),
)

# Packed tokens for a 3x4 grid of patchified 16-channel latents.
tokens = det((1, 12, n_bn), scale=0.9, phase=3.0)
dump("vae_tokens", tokens)
grid = tokens.reshape(1, 12, n_bn).permute(0, 2, 1).reshape(1, n_bn, 3, 4)
mean = vae.bn.running_mean.view(1, -1, 1, 1)
std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
denormed = grid * std + mean
dump("vae_denormed", denormed)
unpatch = denormed.reshape(1, VAE_CFG["latent_channels"], 2, 2, 3, 4)
unpatch = unpatch.permute(0, 1, 4, 2, 5, 3).reshape(1, VAE_CFG["latent_channels"], 6, 8)
dump("vae_unpatchified", unpatch)
dump("vae_image", vae.decode(unpatch, return_dict=False)[0])

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump({"config": {k: list(v) if isinstance(v, tuple) else v for k, v in CFG.items()},
               "vae_config": {k: list(v) if isinstance(v, tuple) else v for k, v in VAE_CFG.items()},
               "tensors": manifest}, f, indent=1)
print(f"[dump] {len(manifest)} tensors -> {OUT}")
print("[names] first 40 weight keys:")
for k in list(sd.keys())[:40]:
    print("  ", k, list(sd[k].shape))
