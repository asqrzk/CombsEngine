#!/usr/bin/env python3
"""Reference values for combs-diffusion/src/scheduler.rs golden tests.

Pure-f64 reimplementation of the diffusers formulas (scaled-linear betas,
"leading" timestep spacing with steps_offset=1, spaced-DDPM fixed_small
posterior, DDIM with set_alpha_to_one=False, DPM-Solver++ 2M midpoint with
final_sigmas_type="zero" and lower-order final step). Re-run to audit the
constants pinned in scheduler.rs tests.
"""
import math

N = 1000
bs, be = 0.00085, 0.012
betas = [(bs**0.5 + i/(N-1) * (be**0.5 - bs**0.5))**2 for i in range(N)]
ac = []
acc = 1.0
for b in betas:
    acc *= (1.0 - b)
    ac.append(acc)

print("alphas_cumprod[0..5]  =", [round(x, 12) for x in ac[:5]])
print("alphas_cumprod[-5..]  =", [round(x, 12) for x in ac[-5:]])

n = 20
ratio = N // n
ts = [i * ratio + 1 for i in range(n)][::-1]
print("leading timesteps(20) =", ts)

x = 1.0
for t in ts:
    at = ac[t]
    prev_t = t - ratio
    ap = ac[prev_t] if prev_t >= 0 else ac[0]
    x0 = (x - math.sqrt(1 - at) * 0.5) / math.sqrt(at)
    x = math.sqrt(ap) * x0 + math.sqrt(1 - ap) * 0.5
print("ddim scalar final     =", round(x, 12))

rows = []
for t in ts:
    at = ac[t]
    prev_t = t - ratio
    ap = ac[prev_t] if prev_t >= 0 else 1.0
    cur_alpha = at / ap
    cur_beta = 1 - cur_alpha
    c_x0 = math.sqrt(ap) * cur_beta / (1 - at)
    c_xt = math.sqrt(cur_alpha) * (1 - ap) / (1 - at)
    var = max(cur_beta * (1 - ap) / (1 - at), 1e-20)
    rows.append((c_x0, c_xt, var))
print("ddpm first row        =", [round(v, 12) for v in rows[0]])
print("ddpm mid row [10]     =", [round(v, 12) for v in rows[10]])

sig = [math.sqrt((1 - ac[t]) / ac[t]) for t in ts] + [0.0]
lam = [-math.log(s) if s > 0 else float("inf") for s in sig]
alp_t = [1.0 / math.sqrt(1 + s * s) for s in sig]
sig_t = [s / math.sqrt(1 + s * s) for s in sig]
x = 1.0
prev_x0 = prev_h = None
for i in range(n):
    x0 = (x - sig_t[i] * 0.5) / alp_t[i]
    h = lam[i + 1] - lam[i]
    if prev_x0 is None or i == n - 1:
        d = x0
    else:
        r0 = prev_h / h
        d = (1 + 1 / (2 * r0)) * x0 - (1 / (2 * r0)) * prev_x0
    ehm1 = math.exp(-h) - 1.0 if math.isfinite(h) else -1.0
    ratio_s = sig_t[i + 1] / sig_t[i] if sig_t[i] > 0 else 0.0
    x = ratio_s * x - alp_t[i + 1] * ehm1 * d
    prev_x0, prev_h = x0, h
print("dpmpp2m scalar final  =", round(x, 12))
