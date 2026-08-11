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
  dequantize_q6_k}` (the gguf.rs scalar paths, now public as the harmony
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

**Layer-typed sliding-window KV — LANDED (2026-08-10).** The transformers
DynamicSlidingWindowLayer design mapped onto `PagedKVCache`:
- `new_with_windows(layers, config, windows)`: `Some(w)` layers keep a
  rolling `[1, n_kv, ≤w-1, d]` tensor (the `-w+1` rule: next decode step
  appends 1 ⇒ exactly `w` visible keys) and **never allocate a paged
  arena**; `None` layers unchanged. Keys stored post-RoPE with absolute
  positions; on eviction only the mask is re-based
  (`kv_offset = pos + seq - full_len`, queries shifted by it into
  `attend`) — RoPE is never re-applied.
- Rollback: all-or-nothing. Un-evicted sliding layers truncate exactly;
  once evicted, `popn` refuses (returns 0) and the engine's reuse gate
  (now `popped == need`, else fresh cache) rebuilds — HF's "prefix caching
  disables under sliding windows" rule. Pure history extensions still
  reuse (need == 0).
- Gemma wires its `AttentionPattern` into `create_kv_cache`; llama stays
  all-global and byte-identical.
- Proof: unit parity tests (sliding vs mask-only global — prefill chunk >
  window, eviction, GQA, rollback/replay) all exact; E2E gemma-3-1b on a
  900+-token prompt is **token-identical** between the sliding paged cache
  and the contiguous mask-only reference; post-generation worker-stream
  GPU sample shows 0.25 GB = exactly the 4 global-layer arenas (4×67 MB) —
  the 22 sliding layers allocate nothing (was 26×67 MB = 1.74 GB).
- Caveat noted: cubecl `memory_usage()` is per-stream — the serve stats
  GPU number covers the worker stream (KV + activations), not the
  load-stream weights; the `combs run` post-load print is the converse.

**KV cache quantization — LANDED (2026-08-10), `COMBS_KV_QUANT=1`.**
int8, group-32 along head_dim, per-token (self-contained — nothing is
ever re-quantized), global layers only; sliding layers stay float (they
hold ≤ w-1 tokens). Contrary to the earlier note, no CubeCL kernel was
needed: 4 signed bytes pack into one Int element with an exact-fit scheme
— lanes 0..2 offset-binary (q+128 ∈ [1,255]), lane 3 signed, so the
extreme corner is exactly i32::MAX; unpack emulates floor-division on the
truncating int div with a sign-corrected high-lane remainder. Pure burn
tensor ops ⇒ works on fused wgpu, unfused f16, and ndarray alike.
- At int8 the KIVI residual window is unnecessary (llama.cpp ships q8_0
  KV the same way); it becomes required if int4 is added later.
- Proof: pack/unpack round-trip is bit-exact on grid values across all
  four lanes; dequant error ≤ scale/2 per element; quantized paged cache
  tracks fp within 1e-2 across page boundaries/prefill/decode with
  identical popn/page semantics. E2E: gemma int8-KV answers correctly and
  the 900-token sliding+quant run is token-identical to full precision;
  llama diverges only at a greedy near-tie (equally coherent).
- Measured: llama-3.2-1B worker-stream KV memory 2048 MB → 576 MB
  (3.6×, matching the theoretical 32/9 exactly); /v1/stats reports
  kv.quantized and the corrected page_bytes (1048576 → 294912).

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

---

## 2026-08-11 — Universal-engine wave 1.1: Qwen2/2.5 + Mistral (LANDED)

Per `CombsLLM/docs/ROADMAP.md` v2 wave 1.1. What shipped:

- **Registry**: `qwen2` + `mistral` → llama loader, both guarded by
  `reject_active_sliding` (an *active* window would run unmasked and silently
  wrong on the llama block — refused loudly until the wave-2 AttentionLayout).
  Metadata parse nulls `sliding_window` when `use_sliding_window: false`
  (qwen configs carry the key even when sliding is off); `max_window_layers`
  stored raw for wave 2.
- **Presence-driven bias loading** (llama.rs): q/k/v/o bias tensors are probed
  by existence, not gated on `metadata.attention_bias` — HF Qwen2 configs
  never emit the flag, and GGUF hardcoded it false. Proven behaviorally:
  synthetic checkpoints with real vs zero vs absent biases (zero == absent
  bit-for-bit, real diverges).
- **`add_bos` honored end-to-end**: `TokenizerSpec.add_bos` from
  `tokenizer.ggml.add_bos_token` / `tokenizer_config.json::add_bos_token`;
  the engine's BOS prepend skips when `Some(false)`. Qwen GGUF declares a BOS
  id (`<|endoftext|>`) but `add_bos_token=false` — previously every prompt
  would have been silently prefixed with it.
- **Per-family GGUF pretokenizer** (`tokenizer.ggml.pre`): qwen2 splits digit
  runs into single digits (`\p{N}` vs the GPT-2/llama `\p{N}{1,3}`); the
  synthesized-tokenizer cache regenerates when its regex disagrees (the old
  cache poisoning class, now covered for regex drift too).
- **Split-GGUF guards**: `GgufSource::load` rejects llama.cpp `gguf-split`
  shards with "shard N of M" instead of a baffling missing-tensor error +
  wrong tied-head inference; `combs pull` skips `-NNNNN-of-NNNNN` shard files
  and explains when a repo has nothing else. (The cached qwen "model.gguf"
  turned out to be shard 1 of 2 — 81 of 339 tensors — which is why the model
  never worked. Multi-file GGUF loading + pull resume/retry are wave-2
  backlog.)

**Proof**: new tests — qwen GGUF fixture (add_bos + digit split + poisoned
cache regen), bias presence proof, sliding guards, split-shard rejection,
metadata gate; full formats/models/runtime suites green. **Token-identity
regression**: llama-3.2-1b GGUF, smollm2-360m safetensors + GGUF byte-identical
greedy output vs the pre-change binary (only timing footers differ). **E2E**:
qwen2.5-coder-7b-instruct Q4_K_M (bartowski single-file, 339 tensors) loads as
`qwen2 (28 layers, hidden 3584, separate lm_head)`, 6.68 GB in use after load,
and greedily generates a correct Python IPv4 validator through the ChatML wrap.
Decode 1.1 tok/s on fused-f32 per-element kernels — slow as expected at 7B;
prefill tiling + the f16-default gate (wave 4) are the scheduled fixes.

---

## 2026-08-11 — Wave 1.2: checkpoint chat templates + GGUF RoPE de-permutation (LANDED)

Two fixes that turned out to be one story: llama-family quality over chat.

**Chat templates (minijinja).** The checkpoint's own Jinja template
(tokenizer_config.json AND GGUF `tokenizer.chat_template`, now read) is
evaluated under the transformers contract ({messages, add_generation_prompt,
bos_token, eos_token} + raise_exception + strftime_now, trim_blocks +
lstrip_blocks) in `combs-runtime/src/template.rs`; `Engine::wrap_chat`
prefers it and falls back to the token-sniffed ChatML/Gemma wraps on any
render failure (log-once, never a 500). ONE template path now serves
`combs serve`, `combs run --chat` (wrap moved after Engine::load — llama
`--chat` used to be a hard error), and the FFI (drifted `apply_chatml`
deleted). `render_str` per call: parsing 4 KB of Jinja is microseconds
against a generation and keeps renders thread-safe with zero 'static
gymnastics. `COMBS_CHAT_DATE` pins `strftime_now` for reproducibility.
Proof: 12 harmony fixtures (llama-3.2/qwen2.5/gemma-3/smollm2 real templates
× 3 message sets) byte-equal vs a Python-jinja2 reference generated under
transformers' environment settings; the gemma template render proved
byte-identical to the old hardcoded wrap before switching. smollm2 prompts
now include its template's default system prompt (transformers-faithful
change, documented).

**GGUF RoPE de-permutation — the real root cause.** With correct llama3
formatting in place, llama-3.2-1b GGUF still produced garbage; probes showed
ANY structured prompt degraded while trivial continuations survived.
Diagnosis: llama.cpp's convert permutes each head's attn_q/attn_k rows into
interleaved-pairs RoPE layout for llama-family archs; this engine applies HF
rotate-half and never de-interleaved. qwen2 GGUF (unpermuted arch) working
perfectly while llama GGUF garbled was the tell. Fix in gguf.rs:
`rope_depermute_src_rows` (transformers' `_reverse_permute_weights` map,
hf[j] = ggml[2j] | ggml[2(j-d/2)+1] per head) applied in BOTH open paths —
dense dequant reorders rows post-dequant; `open_tensor_quant` reorders the
packed stream row-chunk-wise (every supported block format stores whole
blocks per row, so the reorder is exact; `QuantTensor.data` became
`Cow<[u8]>`). Gated on arch ∈ {llama, mistral}, tensors attn_q/attn_k
(weight + bias).

**Proof.** Unit: forward-permute/inverse round-trip. E2E: llama-3.2-1b
`--chat` went from token salad to "Mmap is a system call in Unix-like
operating systems that allows a program to map a file or a block of memory
into a virtual address space…"; **smollm2-360m Q4_K_M GGUF now generates
token-identical text to its f32 safetensors twin** (48-token greedy — the
old lazy 4-token bailout was permutation damage in miniature); serve HTTP
multi-turn with a context-dependent follow-up answers correctly (the
recorded "fluent but off" failure class). gemma chat unchanged through the
template path. **This closes the long-standing GGUF Q/K RoPE permutation
audit**: the mismatch was real, the earlier "empirical parity" was a
shallow-prompt illusion, and parity is now proven at the strongest level
(quant-vs-dense token identity). The wave-4 perplexity harness remains
scheduled as ongoing QA, no longer as the audit's resolution.

---

## 2026-08-11 — Wave 1.3: diffusion correctness — CFG, seeds, schedulers (LANDED)

The three dead knobs are real now. `guidance_scale` and `seed` were parsed
end-to-end and silently dropped; the negative embedding was computed then
discarded; DDPM@20 under-denoised.

- **Classifier-free guidance**: one batched UNet pass over [uncond; cond]
  (batch 2), split, `uncond + scale·(cond − uncond)`; `scale ≤ 1` skips the
  uncond half. Fix surfaced: the resnet time projection reshapes to
  [batch, C, 1, 1], so the timestep embedding is now built batch-sized.
- **Seeded host-side noise** (`noise.rs`): xorshift64* + Box–Muller (same
  no-dep RNG family as the sampler), uploaded via `from_data` — backend
  RNGs are global and non-reproducible; host noise gives byte-identical
  images per seed on any backend. Entropy-drawn seeds are echoed in the
  serve-images response (`"seed"`), the CLI print, and the Create flow
  (which auto-fills the empty seed field for replay).
- **Scheduler trait** + three implementations sharing the SD 1.5
  scaled-linear schedule and diffusers "leading" spacing (offset 1), all
  computed in f64: spaced-posterior DDPM (fixed_small), DDIM
  (set_alpha_to_one=false), and **DPM++ 2M** (multistep midpoint, epsilon,
  final sigma zero, first-order final step) — **the new default @ 20
  steps**, fixing the under-denoise. `scheduler` field on the HTTP API,
  `--scheduler` CLI flag, picker in the Create flow.

**Proof**: harmony tests against Python-computed diffusers-formula references
(alphas_cumprod to 1e-9, DDPM posterior coefficients to 1e-9, DDIM and
DPM++ 2M 20-step scalar chains to f32 tolerance); seeded-noise stream
reproducibility + Gaussian moments; E2E on SD 1.5 @ 512×512×20: same seed →
byte-identical PNG (sha-equal), different seed differs, cfg 7.5 vs 1.0
differs (red channel +9 toward the prompt), negative-prompt path active.
The 4×3 visual QA grid (seeds × scales × schedulers) remains a documented
manual pass via the Create flow's new picker.

---

## 2026-08-11 — Wave 1.4: `combs serve-audio` persistent speech worker (LANDED)

TTS leaves subprocess-per-request land. `generate_audio.rs` refactored
around a load-once `TtsEngine` (ONNX session + phoneme vocab resident,
per-voice style tables cached on first use; `encode_wav` split out for
in-memory serving); `combs generate-audio` is now a thin caller. New
`combs serve-audio --model <dir|kokoro-82m> --port 8083` mirrors the
serve-images pattern: `/health`, `/v1/stats` (busy/totals/durations/voices),
`GET /v1/audio/voices`, `POST /v1/audio/speech` (`{input|text, voice?,
speed?, lang?}` → binary `audio/wav`, OpenAI-shaped), mutex single-flight.
STT (`/v1/audio/transcriptions`) joins this worker in wave 4.

Platform: `ensureAudioWorker()` lazy-start (image-worker clone),
worker-first `generateAudio` with subprocess fallback, worker-first voice
listing, `audio` surface in `surfaces()` (Monitor + `ps` sampling pick it
up automatically), `audioPort`/`COMBS_AUDIO_PORT` config persisted, audio
worker stopped on shutdown and on TTS model change.

**Proof**: first-ever HTTP-surface test in combs-cli
(`tests/serve_audio.rs`, env-gated on the kokoro cache): spawns the worker,
asserts health, 55-voice listing incl. af_heart, real synthesis returning
RIFF/WAVE @ 24 kHz > 10 KB, and 400 on empty input. Live E2E: two speech
requests at 1.6 s / 1.4 s against the resident engine (the subprocess path
re-paid session build + espeak per call); `/v1/stats` live. Deno checks
green.

---

## 2026-08-11 — Reference review vs transformers-main: 3 critical diffusion bugs fixed; audio hardened

A structured review against the on-disk HF transformers source found the
REAL reasons images always under-delivered — none of them in the (harmony-
tested) scheduler:

1. **Upsample2D tiled instead of upsampling**: burn's `repeat_dim` is TILE
   (`[r0,r1,r0,r1]`), not neighbor duplication — every UNet/VAE upsample
   (6 stacked) turned the feature map into a 2×2 mosaic. Fixed with
   `interpolate(..., Nearest)`; harmony test added (a 2×2→4×4 fixture that
   fails under tiling).
2. **Timestep embedding frequencies wrong in scale AND direction**: the
   sweep was missing its `ln(10000)` factor and ran inverted — every
   channel sat near cos(0), so the UNet barely saw the timestep. Fixed to
   the diffusers `Timesteps` convention (freq_shift 0, flip_sin_to_cos);
   harmony test added.
3. **VAE `post_quant_conv` never applied**: AutoencoderKL's 1×1 latent
   projection was skipped entirely. Now loaded and applied (warns when a
   decoder-only extract lacks it).

Plus: CLIP >77-token truncation now preserves EOS; `steps` clamped to
1..=1000 (ratio-0 degenerated all timesteps to 1 → NaN in DPM++); DPM++
multistep guards sequential step order; serve-images validates size/steps/
guidance with 400s instead of silent fallbacks; non-finite pixels fail
loudly instead of returning a black 200. Harmony-generator scripts for the
scheduler/rope/chat-template constants are checked in under
`tools/harmony/` (they regenerate the pinned values byte-identically).

**Honest status**: seeded determinism still byte-exact, CFG effective, PNG
entropy dropped ~40% (mosaic gone) — but a SmolVLM look at the outputs
shows scenes still incoherent; at least one defect remains in the UNet
path. Next diffusion step: a component-parity harness against torch-cpu
reference activations (time_proj → resnet → attention → single UNet step →
VAE) to isolate it. Audio hardening from the same review: espeak resolved
once at load (was re-probed per sentence) with a 30 s watchdog and `--`
argv guard; over-budget sentences chunk at word boundaries instead of
silently truncating; vocab-miss drops are counted and warned; the speech
endpoint recovers poisoned mutexes, clamps speed to 0.25–4.0, and caps
request bodies at 1 MB. The full Whisper-port checklist (mel constants,
conv stem, sinusoid layout, forced prefix, seek loop, harmony ladder) is
recorded in the wave-4 planning notes.

## 2026-08-11 — Wave 2 stage C: qwen3 + phi3 presets (LANDED)

(Also the retro-note for stages A/B, whose commits `ec5ae73`/`0d4cf5b`
landed just before this entry: extended metadata + RoPE scaling
linear/llama3/yarn with formula harmony + the ArchSpec resolver; then
llama.rs parameterized on ArchSpec with the byte-identity gate green over
smollm2 safetensors+GGUF and llama-3.2 GGUF.)

**qwen3** — registry entry riding the W2-B decoder unchanged: ArchSpec
already set `qk_norm` for the family, the loader already probed
`self_attn.{q,k}_norm.weight`. Qwen3-0.6B E2E (safetensors, bf16 1.5 GB):
raw greedy coding prompt and `--chat` (its Jinja template, `<think>` mode)
both coherent. The 0.6B config has explicit `head_dim: 128` against
`hidden/heads = 64` — the decoupled-head-dim path is now exercised for
real. (Qwen3-1.7B+ ship sharded safetensors — blocked on multi-file
loading, same backlog as split-GGUF. Qwen3 GGUF waits for the W2-E
tensor map: `attn_q_norm`/`attn_k_norm`.)

**phi3** — three mechanisms, all load-time (zero forward-pass changes):
1. *Fused projections*: HF phi stores `qkv_proj` (`[q|k|v]` rows) and
   `gate_up_proj` (`[gate|up]` rows). Safetensors: probe-then-split dense
   via `narrow` in the llama loader (biases analogous). GGUF stores fused
   `attn_qkv`/`ffn_up` too (verified in a metadata dump of the Q4_K_M
   file): the adapter's `fused_slice` serves the split HF names as packed
   row ranges — exact for every kernel dtype (whole superblocks per row),
   `Cow::Borrowed` off the mmap. Synthetic tests both sides: a fused
   safetensors checkpoint reproduces its pre-split twin bit-for-bit, and
   a fused GGUF fixture serves the five split names with the right rows.
2. *All-layer sliding window*: every shipped mini activates
   `sliding_window: 2047`; ArchSpec maps phi3 → all-`Sliding(w)` (HF
   semantics) and the GGUF reader now parses `attention.sliding_window`.
   E2E at 3126 prompt tokens (past the window, real KV eviction on all 32
   layers): answers the tail question correctly ("2, 3, and 5.") — the
   W2-B sliding plumbing's first beyond-window proof. A synthetic test
   pins dormant-window ≡ global for short contexts.
3. *EOG token scan* (llama.cpp `special_eog_ids` equivalent): phi GGUFs
   declare eos 32000 (`<|endoftext|>`) but chat turns end with `<|end|>`
   (32007) — generation never stopped. Control tokens matching the known
   end-of-turn set (`<|end|>`, `<|eot_id|>`, `<|eom_id|>`, `<|im_end|>`,
   `<end_of_turn>`, `<|end_of_text|>`, `<EOT>`) now join `eos_token_ids`.

Plus **LongRope parse + short-context tables**: `rope_scaling.type =
"longrope"` (phi 128k variants) parses — factors, top-level
`original_max_position_embeddings`, ratio-derived factor — and
`scaled_inv_freq` builds the short-factor tables with the HF attention
temperature `sqrt(1 + ln(factor)/ln(orig))`; harmony-tested. The long-table
runtime switch lands with the first beyond-original-context preset.

E2E: Phi-3.1-mini-4k-instruct Q4_K_M (bartowski single file, 2.4 GB;
`phi-3.1-mini` preset): chat "In one sentence, what is mmap?" → one
correct sentence, stops at `<|end|>` (44 < 80 max tokens); raw docstring
continuation coherent; 5.78 GB in use after load; 5.6–9.4 tok/s decode.

Known gaps recorded: Q5_K has no GPU kernel — phi's Q5_K `attn_qkv`
dequantizes to dense f32 at load (≈3.6 GB extra; a Q5_K kernel is queued
with the wave-4 perf work). SPM-vocab GGUFs (tokenizer.ggml.model =
"llama") skip the BPE synthesis and need a sibling tokenizer.json — the
pull preset stages the original Microsoft one automatically
(`GGUF_TOKENIZER_COMPANIONS`); a Unigram/byte-fallback synthesis is the
principled follow-up.

## 2026-08-11 — Wave 2 stage D: gemma3 on the universal decoder, gemma.rs deleted (LANDED)

The registry's gemma3/gemma3_text entries now construct `LlamaModel`;
ArchSpec supplies the whole family profile ((1+w) norm flavor, per-head
qk norms, sandwich norms, gelu_tanh, sqrt(hidden) f32 embed scale,
query_pre_attn_scalar, dual-RoPE local theta, every-Nth-global layout
into per-layer cache windows). gemma.rs is deleted — one decoder remains.

Gates (all against gemma.rs behavior on gemma-3-1b-it, fused-f32):
1. token identity — greedy chat + raw byte-identical;
2. logit parity — top-8 last-position logits identical at the printed
   4-decimal precision (" Paris" 19.4067 top);
3. f16 smoke — `--features f16` build, chat output word-identical.
Sweep: smollm2 safetensors+GGUF, llama-3.2 GGUF raw+chat, phi-3.1 chat
all token-identical; formats+models suites green.

Gate 1's first run FAILED usefully, catching two loader truths:
- gemma3 configs omit `tie_word_embeddings` (the HF config class
  defaults it true) — our parse defaulted false and llama.rs then
  demanded the lm_head the checkpoint doesn't have. Metadata now carries
  the per-family default (unit-tested); gemma.rs had been masking this
  with a silent probe-fallback.
- `lm_head.weight` absence now falls back to tied embeddings with a loud
  eprintln (the GGUF `output.weight` presence rule, ported) instead of a
  fatal MissingTensor.

gemma-3 GGUF still waits on the W2-E arch-aware tensor map (`ffn_norm`
name collision, qk-norm names, `attention.key_length`, sliding keys).

## 2026-08-11 — Wave 2 stage E: arch-aware GGUF tensor map — WAVE 2 COMPLETE

`map_tensor_name` takes the architecture: gemma3's `ffn_norm` maps to
`pre_feedforward_layernorm` (llama's same-named tensor is the pre-MLP
post-attention norm — the collision that motivated this stage), its
`post_attention_norm`/`post_ffw_norm` are the sandwich norms, and
`attn_{q,k}_norm` resolve for every family (the loader probes them only
when the spec asks). `attention.key_length` now feeds `head_dim` —
mandatory for gemma3 (256 vs hidden/heads 288) and qwen3 GGUF (128 vs
64); verified present-and-equal (llama-3.2) or absent (rest) across the
cached set, so nothing shifted. Unmapped tensors warn loudly at load
(count + first name) instead of dropping silently, with a known-skip
list (`rope_freqs.weight`) and a fused-source exemption (phi3
`attn_qkv` is consumed by the row-slicer, not the name map).

**The bug the first gemma-GGUF load exposed**: llama.cpp's gemma
converters bake `(1+w)` into every stored norm weight (their graph runs
plain `x̂·w`); the engine keeps HF semantics (`x̂·(1+w)` via the gemma
norm flavor), so the model computed `x̂·(2+w)` across all ~157 norms —
multilingual token soup with a perfect prompt render. Proven with a
numpy cross-check (GGUF norm == safetensors norm + 1.0 exactly, max
diff 0.0) and fixed by removing the offset at the adapter boundary —
the same normalize-at-the-adapter rule as the RoPE de-permutation.
Synthetic gemma3 fixture asserts the name collision mapping, key_length
head_dim, sliding key, `<end_of_turn>` EOG join, and the −1 offset.

E2E: gemma-3-1b-it Q4_K_M GGUF (`gemma-3-1b-gguf` preset,
tokenizer.json staged from the safetensors twin — gated repo companion)
loads warning-free, one-sentence chat answer, stops at `<end_of_turn>`,
raw "The capital of France is → Paris." — near-identical to safetensors
modulo Q4 quant noise. Qwen3-0.6B Q8_0 GGUF (`qwen3-0.6b-gguf` preset)
is **token-identical to its safetensors twin for all 120 generated
tokens** (sole rendered diff: the HF tokenizer prints `<think>`, the
GGUF detok drops it as a control token). Regression sweep
token-identical across smollm2 ×2, llama-3.2 raw+chat, phi-3.1, gemma
safetensors; formats+models suites green.

Wave 2 status: A (metadata/RoPE/ArchSpec), B (universal decoder,
byte-identity), C (qwen3+phi3, fused split, sliding E2E, EOG, LongRope),
D (gemma migrated, gemma.rs deleted), E (arch-aware GGUF map) — ALL
LANDED. One decoder, five architectures, two formats, quant + dense,
per-layer attention layouts, scaled RoPE — the universal decoder holds.

## 2026-08-11 — Diffusion component parity vs torch: two root causes found and fixed

The queued parity harness landed: `tools/harmony/gen_diffusion_reference.py`
dumps deterministic-input reference activations from the LOCAL SD-1.5 via
diffusers/transformers (torch-cpu f32) — time embedding, CLIP (with
per-layer and attention-internal taps via forward hooks), UNet (per-block
taps), VAE decode — and `combs-diffusion/tests/parity.rs` (env-gated on
`COMBS_DIFFUSION_PARITY_DIR`) replays the same dumped inputs through our
components on NdArray, comparing stage by stage. The UNet gained
`forward_traced` (the plain forward delegates to it) so per-block taps
can't drift. The first failing stage names the bug; two were found, both
burn-API-semantics traps in the `repeat_dim` family:

1. **CLIP ran anti-causal.** burn's triangle masks are complements:
   `triu_mask(offset 1)` returns TRUE at-and-below the diagonal (the doc
   example makes it explicit), so `mask_fill` blocked the PAST and kept
   the FUTURE — every text token attended only to later tokens, in all 12
   layers. The old unit test asserted only the mask's SHAPE, which is how
   it shipped. Fixed with `tril_mask(0)` (TRUE strictly above the
   diagonal); the unit test now asserts values. Bisect evidence:
   embeddings bit-exact, q/k/v exact, manual narrow-per-head attention
   matched torch only under the corrected mask (reference ctx[0] == v[0]
   proved torch causal). After: attention 2.7e-6, last hidden 1.7e-5.

2. **Downsample convs shifted the scene half a pixel per level.**
   `load_conv2d` hardcoded `PaddingConfig2d::Same`; for the stride-2 3×3
   downsample convs burn-Same places the single required pad bottom/right
   (TF convention) while torch/diffusers pad (1,1) symmetrically — same
   output size, spatially shifted content, compounding across the three
   down levels. All convs now carry explicit k/2 padding (identical math
   for stride 1, corrected for stride 2). Isolation evidence: conv_in,
   down0's resnet0 and spatial transformer all exact on reference inputs,
   while the down0 block tap (which includes its downsampler) diverged at
   7.15.

Full suite after both fixes — every stage within bounds: time 5e-5/1e-6,
CLIP exact→1.7e-5, UNet per-block ≤2e-4 with **noise_pred 2.1e-6** (was
0.247), VAE decode 2.9e-5 (first-ever VAE verification). The unused
combs-models/combs-core deps are dropped from combs-diffusion.
