# Combs Engine — Mixed-Precision & Quantized Inference Plan

**Goal:** run models locally and natively at their real memory footprint —
f16 compute and true 4-bit (and K-quant) weights — with the right kernel chosen
per model/tensor automatically.

## Where we are today (verified)

- Backend is `Wgpu<f32, i32, u32>` + `fusion` (combs-core/src/lib.rs:20). **All**
  compute is f32.
- GGUF quant tensors are parsed with exact ggml layouts (combs-formats/gguf.rs)
  but **dequantized to f32 `Vec` on the CPU at load** → an 8× blow-up (Q4 7B:
  4 GB file → ~28 GB RAM).
- `bf16`/`f16` as the backend float **compiles but crashes at runtime** in
  `burn-fusion 0.21` (`index out of bounds` in stream ordering). Verified by
  building `--features bf16` and running — panics before first token.
- **No custom CubeCL kernels exist yet**; cubecl 0.10 is a transitive dep only.
  Metal/cubecl is already fragile: the autotuner trips Metal's 32 KB shared-mem
  limit for matmuls with M,K ≥ 512, silently returning garbage — worked around
  by `safe_matmul` M-slabbing (combs-models/matmul.rs). New kernels must respect
  the same limits.
- `burn 0.21` **has a partial quantization API**: `QuantizedTensor`,
  `dequantize`, and `q_matmul` — but `q_matmul` **dequantizes then calls the
  normal matmul** (burn-cubecl ops/qtensor.rs:281-305), so it's a *storage* win,
  not a fused peak-memory win; and `q_gather`/`q_select`/`q_slice`/`q_expand`
  are `unimplemented!()` (embedding lookup uses gather). Its scheme is
  int8/int4 affine-per-block, **not** ggml K-quant compatible.
- The only f32-CPU boundary in the hot path is `readback_logits`
  (combs-runtime/engine.rs) — the sampler runs in f32. Everything else flows
  through backend tensors, so dtype changes are contained.

## Two independent axes (and a note on the brief)

The requirement "*if f32 use 6-bit kernel, else f16 4-bit*" conflates two
separable axes. The clean design keeps them orthogonal:

- **Storage format** (per weight tensor): F32 / F16 / Q8 / Q6_K / Q5_K / Q4_K /
  Q4_0 … → picks the **dequant path/kernel** for that tensor.
- **Compute dtype** (whole model): f32 today, f16 later → accumulation/output
  precision of matmul + activations + KV.

A Q6_K weight uses the 6-bit dequant kernel; a Q4_K weight the 4-bit kernel —
independent of whether the model computes in f16 or f32. **Recommendation:**
drive the kernel off the tensor's real stored format (not the compute dtype),
and expose compute dtype as its own switch. Please confirm this reading — if you
literally want compute-dtype→bit-width coupling, we can add that as a policy on
top, but the kernel dispatch itself should key on storage format.

---

## Phase 0 — f16 compute path — ✅ DONE (2026-08-10)

**Value:** ~2× memory (weights + activations + KV) and faster. Shipped and
validated for the Llama/SmolLM2 family.

### What shipped
- **`--features f16`** builds an f16 engine (`cargo build --release --features
  f16`). Coherent output, **~2× faster prefill** (81 vs 39 tok/s on
  smollm2-360m), weights + KV stored as f16 (~half the memory) by construction.
- **Mechanism 1 — bypass fusion at the type level:** f16 uses the *unfused*
  `CubeBackend<WgpuRuntime, f16, i32, u32>` directly (burn-fusion 0.21 panics on
  reduced precision). The default build keeps `Wgpu<f32>` **fused and unchanged**
  — no workspace/feature surgery, no risk to the validated f32 path.
- **Mechanism 2 — f32-accumulated reductions** (`combs-models::precision`): the
  ops that overflow/underflow in f16 — RMS/LayerNorm `mean(x²)`, attention
  scores + softmax (`exp` > f16's 65504), Gemma's gelu `x³` — run in f32 and
  cast back. **No-op in the f32 build** (cast-to-current-dtype is free), so the
  default path is byte-for-byte identical.
- Two binaries now exist: `combs` (f32, default) and `combs-f16` (f16).

### Status of the follow-ups
- **Gemma-3**: emits empty output in **both f32 and f16** for a raw
  (no-BOS) prompt — so this is a pre-existing Gemma prompt/template issue
  (needs a leading `<bos>`), **not an f16 regression**. f16 is at parity with
  f32 here. Tracked separately from precision work.
- **SmolVLM vision encoder** — ✅ f16-guarded (`smolvlm.rs` scores+softmax now
  run in f32 via `to_f32`/`to_float`).
- **Reclaim freed VRAM** — ✅ `DEFAULT_KV_ARENA_CAP` is now dtype-aware: 32k in
  f32, **64k in f16** (KV per token is half, so the same memory buys 2× the
  context). `--context-size` still overrides.

### (original notes retained below)

**Value (orig):** ~2× memory (weights + activations + KV) and usually faster;
helps every model, not just GGUF. 7B: 28 → 14 GB.

### ⚠️ Empirical findings (probed 2026-08-10) — Phase 0 is NOT a quick win

Tested both reduced-precision backends end-to-end (built + ran smollm2-360m):

- **bf16 fused** → panics in `burn-fusion` (stream ordering).
- **bf16 unfused** → `cubek-matmul` refuses to launch: *"Types
  lhs/rhs/output=Scalar(Float(BF16)) not supported."* **cubecl 0.10's matmul
  has no bf16 path on Metal/wgpu.** Hard blocker — bf16 is off the table until
  cubecl adds it upstream.
- **f16 unfused** → matmul launches (Metal supports f16), **but output is
  garbage** (`ectableectable…`). f16 lacks the range/precision for the model's
  **un-guarded reductions** (RMSNorm sum-of-squares, softmax `exp`, QKᵀ scores).

**Conclusion:** f16 is *reachable* but needs the core model math made
f16-safe — accumulate every reduction in f32 (upcast → reduce → downcast) in
`norm.rs`, the attention block, and softmax, across `llama.rs`/`gemma.rs`/
`smolvlm.rs`. That's a **multi-day numerics port with real accuracy risk**, not
the "days, easy" win first estimated. bf16 (the safe-range choice) can't help
until cubecl supports it. **→ Reprioritize: do Phase 1A first** (quantized
weights keep the working f32 numerics and deliver the dominant memory win);
return to f16 as a deliberate numerics effort.

**Original blocker note:** burn-fusion crashes on bf16/f16.

**Approach (in order):**
1. Build burn **without `fusion`** for a reduced-precision backend
   (`Wgpu<f16>` unfused) and test that it runs. Fusion is a workspace feature;
   introduce a `bf16`/`f16` cargo feature that also drops `fusion`. Accept the
   perf cost from losing kernel fusion as the price of the memory win, measure
   it, decide.
2. If unfused f16 also breaks, isolate a minimal repro and **file/patch
   upstream** (burn-fusion / cubecl). Track the issue; f16 blocks on it.
3. Prefer **bf16 over f16** for the default reduced path — bf16 keeps f32's
   exponent range, so Gemma/attention/softmax won't overflow. Use f16 only where
   validated.

**Work:** cfg-gated backend alias (already prototyped), fix `readback_logits`
to `convert::<f32>()` (already prototyped), guard reductions (rms-norm, softmax,
attention scores) to **accumulate in f32** even when storing f16. Update the
Gemma bit-exact test to a tolerance test under reduced precision.

**Deliverable:** `combs serve --dtype bf16` (or `--features bf16` build) that
runs coherently with no NaNs and ~half the memory. **Effort: days.**

---

## Kernel architecture — best-practices from HuggingFace `kernels`

HF's `kernels` library (drop-in optimized kernels distributed per op+device) is a
proven design template. Principles we adopt for our CubeCL kernels:

1. **Swap behind a stable op interface.** Model code keeps calling `linear()`,
   `rms_norm()`, `attend()`. The kernel is chosen underneath — model code never
   changes. (We already have these seams.)
2. **Always fall back to a portable path.** HF: "falls back to standard PyTorch
   when no kernel is available." Ours: the portable burn-tensor path (and the
   `gguf.rs` CPU dequant) is the reference/fallback; the CubeCL kernel is the
   opt-in fast path. **This is what de-risks the burn dependency** — kernels are
   optional accelerators, never load-bearing for correctness.
3. **Device-aware dispatch.** HF maps each op to a different kernel per
   cuda/rocm/metal/xpu. Ours: a small registry picks the kernel by
   `(op, device, dtype, weight-format)`; unknown combos → fallback.
4. **Own your fusion (don't depend on an auto-fuser).** HF ships explicit fused
   modules (e.g. `RMSNorm+MLP`) with a companion layout class. This is the direct
   answer to "why depend on burn-fusion?": we write the specific fused kernels
   that matter (dequant+matmul first), which we control and can debug — instead
   of an opaque global fuser that panics on f16.
5. **Two-layer kernel = Layout + Compute.** HF splits `KernelNameLayout` (weight
   packing / checkpoint remap) from `KernelName` (the `forward` compute). Ours:
   a **WeightStore** packs GGUF quant blocks into the device buffer layout the
   kernel expects, separate from the **kernel** that consumes them. Testable
   independently.
6. **Validate against a reference with tolerance.** HF notes kernels differ from
   the reference by reordering/accumulation ("matches ~97%"). Ours: each CubeCL
   kernel is unit-tested against the `gguf.rs` CPU dequant + an f32 reference
   matmul, within tolerance.
7. **Priority ops = the same ones HF kernelizes:** Linear (quantized
   dequant-matmul first), Attention, RMSNorm, Rotary, activations.

### The abstraction (what we build)

```
trait LinearKernel {                     // the swappable op
    fn forward(&self, x: Tensor) -> Tensor;
}
// impls, chosen by the registry:
//   PortableLinear      — f32/f16 burn matmul            (fallback, exists)
//   PortableQuantLinear — dequant (CPU/burn) + matmul    (fallback, exists)
//   CubeQ4Linear        — fused Q4 dequant-matmul kernel (fast path, NEW)
struct WeightStore { /* packed bytes + scales on device, per format */ }
fn pick_linear(device, dtype, fmt) -> Box<dyn LinearKernel>  // registry + fallback
```

Model loaders build the op via `pick_linear(...)`; on an unsupported
device/format it returns the portable impl. New kernels register without
touching model code — exactly HF's model.

## Phase 1 — Quantized weights (the big weight-memory win)

Weights dominate memory, so keep them quantized in VRAM and dequantize on demand.
Two implementation tiers; do 1A first for a working path, then 1B for the real
fused win + ggml fidelity.

### 1A — Ride burn's quantization API (fast path to "it works")
- At load, **re-quantize** each GGUF/f16 weight into burn `QuantizedTensor`
  (int8 first — 4× smaller; int4 next — 8× if the scheme + wgpu path support it).
- Route Linear through `q_matmul`. Storage is quantized; the transient dequant
  is freed per layer, so **persistent** weight memory drops 4–8×.
- **Gaps to close:** `q_gather` is unimplemented → keep the **embedding table in
  f16/f32** (it's one tensor) or implement gather-on-quantized. Validate
  `q_matmul` correctness + that peak (quantized + one transient dequant) actually
  fits budget.
- **Caveat:** burn's scheme ≠ ggml K-quant, so we requantize from the
  CPU-dequantized f32 (small extra quality loss vs native K-quant). Acceptable
  for a first cut.
- **Effort: ~1 week.**

### 1B — Custom CubeCL fused dequant-matmul kernels (native, best)

**STATUS: Q4_0, Q4_K and Q6_K landed and validated on Metal**
(`combs-models/src/qmatmul.rs`) — everything a Q4_K_M model file needs:
- Layout layer: `repack_q4_0` / `repack_q4_k` / `repack_q6_k` turn GGUF
  block streams (18/144/210 B — none word-aligned) into GPU
  structure-of-arrays: packed quant bytes as `u32` words + f32 super-scales
  (5.0 / 4.63 / 6.63 bits per weight in VRAM; exact f16→f32 scale
  conversion preserves reference numerics). `Q40Weight`/`Q4KWeight`/
  `Q6KWeight` hold the packed handles device-side.
- Compute layer per format: a `*_dequant_kernel` (validation) + a
  `*_matmul_kernel` (fused production path, f32 accumulation). K-quants
  unpack the 6-bit scale/min pairs (`get_scale_min_k4`) and Q6_K's split
  4+2-bit values in-kernel, and use the ggml sum-split
  (`Σ(d·sc·q − dmin·m)x = d·sc·Σqx − dmin·m·Σx`) so scales apply once per
  sub-block.
- Validation per principle #6: dequant kernels match
  `combs_formats::quants::{dequantize_q4_0, dequantize_q4_k,
  dequantize_q6_k}` (the gguf.rs scalar paths, now public as the golden
  reference) — Q4_0 **bit-exact**, K-quants within 1e-6 relative (FMA
  contraction only); fused matmuls match a reference matmul within 1e-3
  for decode (m=1) and prefill (m>1) shapes.

Remaining in 1B:
- Q8_0/Q5_K kernels if models need them (same pattern, mechanical now).
- Store weights as raw packed bytes end-to-end (stop the CPU→f32 expansion in
  gguf.rs; upload the quant blocks as-is) — needs a raw-tensor accessor on
  `GgufSource` + Phase 2's per-tensor dispatch.
- Device-tensor forward (activation handle in/out, no host round-trip) behind
  the linear seam; then tiling/vectorized loads for throughput.
- **Must** respect Metal's 32 KB shared-mem limit (cf. the 512-matmul bug) —
  tile conservatively, reuse the `safe_matmul` M-slab lesson.
- **Must** respect Metal's 32 KB shared-mem limit (cf. the 512-matmul bug) —
  tile conservatively, reuse the `safe_matmul` M-slab lesson.
- **Effort: 2–4 weeks** (new CubeCL competency; per-format kernels + tiling +
  Metal validation).

---

## Phase 2 — Per-tensor kernel dispatch ("load the kernel per model")

**STATUS: LANDED for the Llama family.** Measured on Llama-3.2-1B Q4_K_M
(Metal): **1.72 GB GPU in use vs 5.01 GB** on the dense fallback (2.9×),
token-identical greedy output over 30 tokens, decode parity (21.7 vs 21.0
tok/s), faster TTFT (192 vs 541 ms). Remaining resident f32 is the tied
embedding (kept dense for `select` + tied head) plus KV/workspace.

How it's wired:
- `combs-formats`: `ModelSource::open_tensor_quant` hands out the raw packed
  bytes + `QuantFormat` (GGUF Q4_0/Q4_K/Q6_K); other formats/dtypes return
  `None`. **Gotcha fixed en route:** the `impl ModelSource for Box<T>`
  forwarder must forward *defaulted* methods too — a missing forward
  silently pinned every CLI call to the trait default and the quant path
  never engaged while concrete-type tests passed.
- `combs-models/qlinear.rs`: `Linear<B>` seam (`Dense(Tensor)` |
  `Quant(Box<dyn QuantLinearOp<B>>)`); `try_quant_linear` picks the kernel
  per tensor at load. Backend dispatch via safe `Any` downcasts:
  - default fused f32 backend → the matmul enters the fusion stream as a
    **custom operation** (burn's `OperationIr::Custom` escape hatch);
  - unfused f32 → direct launch; `--features f16` → activation cast-wrapped.
  Forwards are dtype-following (burn 0.21 resolves tensor dtypes from
  per-device defaults, so even an f32 backend can carry f16 tensors).
- `llama.rs`: the seven per-layer projections + untied `lm_head` load
  through `load_linear`; embeddings stay dense. Gemma/SmolVLM-vision still
  dense (follow-up).
- Escape hatch: `COMBS_NO_QUANT_KERNELS=1` forces the dense path;
  `COMBS_DEBUG_QUANT=1` logs the per-tensor decision.
- Shape guard: tensors whose row size isn't a block multiple (ggml itself
  stores those in 32-block formats — e.g. SmolLM2's hidden 960) fall back
  to dense per tensor, never error.

Follow-up round (landed): **Q5_0 + Q8_0 kernels** — the formats ggml falls
back to for rows not divisible by 256, so "Q4_K_M" files of models like
SmolLM2 (hidden 960: 176 Q5_0 + 17 Q8_0 tensors) now pack fully. Both GPU
dequants are bit-exact vs the CPU references. **Gemma** now loads its seven
projections + lm_head through the same `load_linear` factory.

Validated matrix (Metal, `combs run`, greedy — quant output token-identical
to the dense fallback in every case):
| Model | Path | GPU in use |
|---|---|---|
| Llama-3.2-1B Q4_K_M | packed, f32 fused | 1.72 GB |
| Llama-3.2-1B Q4_K_M | dense fallback | 5.01 GB |
| Llama-3.2-1B Q4_K_M | packed, f16 build | **1.16 GB** |
| SmolLM2-360M Q4_K_M (Q5_0/Q8_0 mix) | packed (all 224 tensors) | 427 MB |
| SmolLM2-360M Q4_K_M | dense fallback | 1.45 GB |
| Gemma-3-1B / SmolVLM safetensors | dense (regression-checked) | unchanged |

Still open: SmolVLM vision-tower linears through the factory (GGUF VLM
files are rare; text stack already covered via the shared Llama loader);
prefill-shape tiling for the fused kernels (per-element kernel is
decode-optimal; prefill is correct but untiled); Q4_1/Q5_1/Q5_K kernels if
models surface with them.

## Observability + theory follow-ups (2026-08-10)

- **Gemma f16 FIXED**: the sqrt(hidden) embedding scale was the last
  unguarded half-precision op (HF documents the same corruption in bf16);
  now computed in f32 (gemma.rs embed). f16 gemma output is identical to
  f32 — and decodes at 17.2 tok/s vs f32's 1.8 (unfused backend avoids
  whatever stalls fused f32 gemma; that slowness is still open).
- **GGUF Q/K RoPE permutation — audit inconclusive but empirically OK**:
  llama.cpp's convert permutes Q/K rows for the llama arch, our rope is HF
  rotate_half with no de-permutation in gguf.rs, yet GGUF vs safetensors
  quality is at parity (same first tokens, coherent text, dense-vs-quant
  token-identical). Either the pairing coincidentally composes or the
  files we use aren't permuted as assumed. Proper resolution: a
  perplexity comparison (HF perplexity.md methodology) safetensors vs
  GGUF on ~1k tokens — do before trusting GGUF for anything sensitive.
- **/v1/stats** (serve.rs) now reports: totals + cache hit rate, EWMA
  decode tok/s, last-generation timings, GPU allocator sample, per-session
  KV page tables + kv_bytes, weight/quant identity, device caps, build
  flags. `usage` carries timing/cache per request. The CombsLLM Monitor
  tab consumes this via the platform's /api/events SSE stream.

Next engine phase (planned, not started): layer-typed KV cache with the
(kv_length, kv_offset) mask contract + sliding-window layers for gemma's
5:1 local:global pattern (~5/6 KV memory at long context); then KIVI-style
KV cache quantization (fp16 residual window + group-64 int4 bulk,
full-attention layers only) behind COMBS_KV_QUANT=1.

---

## Phase 3 — Validation, targets, benchmarks

**Memory targets (7B, indicative):**

| Mode | Bytes/param (weights) | 7B weights | Fits |
|---|---|---|---|
| f32 (today) | 4 | ~28 GB | 32 GB+ only |
| f16/bf16 (Phase 0) | 2 | ~14 GB | 16 GB (tight) |
| int8 (Phase 1A) | 1 | ~7 GB | 16 GB comfortably |
| Q4 fused (Phase 1B) | ~0.5 | ~4 GB | 8 GB |

**Correctness:** per-format kernel unit tests vs the `gguf.rs` CPU dequant
reference; end-to-end coherence + perplexity vs the f32 baseline; NaN guards;
Gemma parity as a tolerance test under reduced precision.

**Perf:** tok/s per mode; ensure the quant path is fused enough not to regress
decode speed (1A's dequant-then-matmul may be slower — measure; 1B fixes it).
Log the chosen precision/kernel per tensor at load for transparency.

---

## Cross-cutting risks

- **burn-fusion + reduced precision** — the f16/bf16 blocker; may force
  fusion-off (perf) until an upstream fix lands. This gates Phase 0.
- **Metal 32 KB shared memory** — custom kernels must tile conservatively; the
  512-matmul bug is the cautionary precedent.
- **ggml-K-quant vs burn scheme mismatch** — 1A loses native K-quant fidelity;
  1B preserves it but is the heavy lift.
- **New maintenance surface** — hand-written CubeCL kernels are a new competency;
  keep the CPU dequant as the reference oracle for tests.

## Recommended sequencing

1. **Phase 0 (f16/bf16, fusion-off)** — days — biggest bang-for-effort; ships a
   real 2× win and de-risks the reduced-precision plumbing.
2. **Phase 1A (burn int8/int4)** — ~1 week — first working quantized path,
   4–8× weight storage, validates dispatch.
3. **Phase 2 (dispatch)** — days — wire per-tensor selection.
4. **Phase 1B (custom Q4_K/Q6_K kernels)** — 2–4 weeks — the native, fused,
   true-4-bit endgame; migrate hot tensors off 1A onto real kernels.

Phases 0→2 give a usable, memory-efficient engine in ~2 weeks; Phase 1B is the
"native quantized" finish that makes 7B run in ~4 GB.
