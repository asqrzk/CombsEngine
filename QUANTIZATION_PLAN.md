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

### Remaining f16 follow-ups (not blocking)
- **Gemma-3 f16** still emits empty output — needs 1–2 more f32 guards
  (embedding √hidden scale and/or a BOS/template check; gelu is already guarded).
  Llama-family + SmolLM2 are fully working.
- **SmolVLM vision encoder** (`smolvlm.rs` softmax) not yet f16-guarded — text
  path works; guard the vision attention with the same `to_f32`/`to_float`
  pattern before using f16 with images.
- **Reclaim the freed VRAM:** in f16, KV per token is half, so the
  `DEFAULT_KV_ARENA_CAP` (32k) could double to 64k for the same memory — a
  dtype-aware cap is an easy follow-up (this is the "use the other half for more
  KV / longer context" idea).

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
- Author `#[cube]` kernels that read **ggml-packed blocks directly**
  (Q4_0/Q8_0/Q4_K/Q5_K/Q6_K) from a raw `u32` storage buffer and do a **fused
  dequant-matmul**, accumulating in f32/f16 — the weight is *never* fully
  materialized. This is the llama.cpp/MLX approach and the only path to true
  ~0.5 B/param peak.
- Store weights as raw packed bytes (stop the CPU→f32 expansion in gguf.rs;
  upload the quant blocks as-is). The existing `gguf.rs` CPU dequant functions
  become the **golden reference** for kernel unit tests.
- One kernel per quant family, or a single kernel parameterized by a block-format
  descriptor (scale offset, bits, superblock size). Start with **Q4_K_M** (the
  most common) + Q6_K, then Q4_0/Q8_0.
- **Must** respect Metal's 32 KB shared-mem limit (cf. the 512-matmul bug) —
  tile conservatively, reuse the `safe_matmul` M-slab lesson.
- **Effort: 2–4 weeks** (new CubeCL competency; per-format kernels + tiling +
  Metal validation).

---

## Phase 2 — Per-tensor kernel dispatch ("load the kernel per model")

- The GGUF loader already knows each tensor's `ggml_type`; safetensors carry a
  dtype. Add a `LinearKind` / `WeightStore` enum and a factory:
  `build_linear(source, name) -> Box<dyn LinearLayer>` that returns an f32/f16
  `Linear`, a burn-`QuantizedLinear`, or a `Q4KLinear`/`Q6KLinear` bound to the
  matching custom kernel — chosen from the **stored format**.
- `GenerativeModel` construction (llama.rs / gemma.rs / smolvlm.rs) calls the
  factory instead of `load_weight` directly. Mixed models (some tensors f16,
  some Q4_K) fall out naturally — this is exactly "the kernel loaded per the
  model being used."
- Registry (`combs-models/registry.rs`) stays architecture-keyed; quantization
  is an orthogonal per-tensor concern layered under the loader.

**Effort: days**, once 1A/1B provide the layer types.

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
