#!/usr/bin/env python3
"""Reference inv_freq values for combs-models/src/rope.rs scaling harmony.

Pure-f64 reimplementation of HF modeling_rope_utils: llama3 piecewise and
YaRN NTK-by-parts. Re-run to audit the constants pinned in rope.rs tests.
"""
import math

def base_inv_freq(dim, base):
    return [base ** (-(2*i)/dim) for i in range(dim//2)]

def llama3(dim, base, factor, low_f, high_f, orig):
    out = []
    for f in base_inv_freq(dim, base):
        wavelen = 2*math.pi/f
        if wavelen < orig/high_f:
            out.append(f)
        elif wavelen > orig/low_f:
            out.append(f/factor)
        else:
            smooth = (orig/wavelen - low_f)/(high_f - low_f)
            out.append((1-smooth)*f/factor + smooth*f)
    return out

l3 = llama3(64, 500000.0, 32.0, 1.0, 4.0, 8192)
print("llama3 inv_freq[0,8,16,24,31]:", [round(l3[i], 12) for i in (0,8,16,24,31)])

def yarn(dim, base, factor, orig, beta_fast=32.0, beta_slow=1.0):
    def corr_dim(rot):
        return (dim * math.log(orig/(rot*2*math.pi))) / (2*math.log(base))
    low = max(math.floor(corr_dim(beta_fast)), 0)
    high = min(math.ceil(corr_dim(beta_slow)), dim-1)
    if low == high: high += 0.001
    inv = []
    for i in range(dim//2):
        pos_freq = base ** ((2*i)/dim)
        ramp = min(max((i - low)/(high - low), 0.0), 1.0)
        ef = 1 - ramp
        inv.append((1.0/(factor*pos_freq))*(1-ef) + (1.0/pos_freq)*ef)
    return inv, 0.1*math.log(factor) + 1.0, low, high

yi, attn, lo, hi = yarn(128, 1000000.0, 4.0, 32768)
print("yarn low/high:", lo, hi, "attention_factor:", round(attn, 12))
print("yarn inv_freq[0,16,32,48,63]:", [repr(yi[i]) for i in (0,16,32,48,63)])
