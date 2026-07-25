# Combs Engine — L0 Rust Core + L1 FFI Boundary

Cross-platform edge AI inference engine: Rust core (Burn 0.21 + CubeCL/wgpu),
a stable C-ABI/JSON-FFI boundary, and a Deno/TypeScript application layer
(`../Js`). This directory is the L0+L1 cargo workspace.

**Phases 1–5 complete.** Models load from HuggingFace safetensors dirs OR
GGUF files (F32/F16/BF16/Q4_0/Q8_0) via `open_model_source` and generate on
GPU (wgpu/Metal) via CLI, the C FFI, or the bundled OpenAI-compatible
server — and the FFI library cross-builds for macOS (arm64/x86_64), iOS and
Android from the same code (wgpu abstracts Metal/Vulkan/DX12). The CLI also
scaffolds Svelte 5 chat/debate apps (`combs chew-chat-ui`/`chew-debate-ui`).

## Crate layout

| Crate | Role |
|---|---|
| `combs-core` | Backend type aliases (`CombsBackend` = fused wgpu/CubeCL, f32), device helpers + `DeviceCaps`, `BufferPool` memory facade, `quant` (q4 dequant ops) |
| `combs-formats` | Format-agnostic `ModelSource` trait + `open_model_source` dispatcher; safetensors adapter (HF dirs, mmap zero-copy) and **GGUF adapter** (v2/v3, ggml→HF name/dim mapping, Q4_0/Q8_0 dequant, GGUF-metadata tokenizer synthesis) |
| `combs-models` | `GenerativeModel` contract (`embed`/`prefill`/`decode`/`create_kv_cache`), attention-facing `KVCache` trait with `PagedKVCache` (page arena + page table + free-page allocator, `popn` rollback) and `ContiguousKVCache` baseline, `CacheConfig`, Llama family, `ModelRegistry` |
| `combs-runtime` | `Engine` (single-flight mpsc request queue + worker thread, chunked prefill, cancel flag, context budget vs cache capacity), `Sampler`s (greedy + seeded multinomial) with composable `LogitsProcessor`s (penalties, temperature, top-k/top-p), stop token/string detection, incremental detokenization, stats (TTFT, prefill/decode tok/s, cache pages used) |
| `combs-cli` | `combs` binary (`run`, `serve`, `devices`) |
| `combs-ffi` | L1 boundary: stable C ABI + JSON FFI (`cdylib`/`staticlib`), process-shared device, streaming callback, per-request cancellation |
| `include/combs.h` | Hand-maintained C header mirror of the FFI API |
| `xtask` | Build entrypoint (`cargo xtask …`) incl. cross-platform matrix + `dist/` bundling |

## Commands

```sh
# Build / test
cargo xtask build [--release]
cargo test --workspace

# Generate (streams tokens to stdout, stats to stderr)
cargo xtask run -- run --model <model-dir> --prompt "The capital of France is" --max-tokens 40
# or directly:
cargo run --release -p combs-cli -- run --model <model-dir> --prompt "..." [--max-tokens N] [--temperature T] [--top-k K] [--top-p P] [--repetition-penalty R] [--frequency-penalty F] [--presence-penalty F] [--seed S] [--stop STR ...] [--chat] [--prefill-chunk-size N]

# KV cache selection: COMBS_KV=paged (default) | contiguous
COMBS_KV=contiguous cargo run --release -p combs-cli -- run --model <model-dir> --prompt "..."

# Serve (OpenAI-compatible HTTP + SSE)
cargo run --release -p combs-cli -- serve --model <model-dir> --port 8080

# Cross-platform builds (macos/ios/android full; windows/linux check)
cargo xtask matrix
cargo xtask target ios-arm64
cargo xtask bundle        # -> dist/<platform>/{lib,combs.h}, cleans cross trees

# Integration test against a real model (ignored by default)
COMBS_TEST_MODEL=../../models/SmolLM2-135M cargo test --release -p combs-runtime -- --ignored
# cubecl wgpu/Metal matmul cliff repro (see Known issues)
cargo test --release -p combs-models --test attn_cliff -- --ignored --nocapture
```

A model dir is a HuggingFace layout: `config.json`, `model.safetensors`
(or shards + `model.safetensors.index.json`), `tokenizer.json`,
optional `generation_config.json` / `tokenizer_config.json`.

## Design notes (Phase 2)

- **Attention lives behind the cache**: `KVCache::attention(layer, q, k, v,
  pos, scale)` appends K/V and returns the attention output, so the cache
  owns K/V layout, GQA expansion and causal masking. Model code
  (`llama.rs`) no longer knows how K/V is stored.
- **`PagedKVCache`** (default): per-layer arenas of
  `[num_pages, n_kv, page_size=16, head_dim]`, a single-sequence page table
  and a LIFO free-page allocator. Appends write one `slice_assign` per
  touched page; attention gathers active pages (`select` + reshape +
  `narrow`) into a contiguous window for the standard matmul path — the
  Phase 1 O(seq) `cat` per token is gone. `popn(n)` rolls back tokens and
  frees trailing pages. A fused no-gather CubeCL kernel is a later task.
- **`ContiguousKVCache`** remains as the correctness baseline
  (`COMBS_KV=contiguous`); a CPU (NdArray) test suite cross-validates the
  two token-for-token across chunked/unaligned appends, `popn` rollback
  and `reset`.
- **Chunked prefill**: `GenerationConfig::prefill_chunk_size` (default 512;
  0 = single shot). Any chunk size is safe: matmuls route through
  `safe_matmul` (see Known issues), which slabs the M dimension around the
  wgpu/Metal kernel bug below.
- **Engine queue (LiteRT-LM ExecutionQueue pattern)**: the model lives on
  one worker thread; `Engine` is `Send + Sync`, `generate` is `&self` and
  queues requests serially (single-flight), streaming pieces back over a
  channel. `generate_cancellable` takes an `Arc<AtomicBool>` abort flag
  checked per decode step. Context budget is enforced against the cache
  capacity (`CacheConfig::max_seq_len`).
- **Traits first**: the runtime talks to models only via `GenerativeModel`
  and to files only via `ModelSource`. New architectures = one registry line;
  new formats (GGUF/ONNX/litertlm) = new `ModelSource` impl.
- **Fused attention path**: `attend()` routes through burn's
  `module::attention` (flash-capable; its bottom-right causal alignment
  exactly matches the `pos`-offset masking chunked prefill needs — verified
  token-identical at 1261 tokens). `COMBS_ATTN=manual` forces the reference
  scores/mask/softmax path. Without the `autotune` cargo feature both paths
  resolve to the same kernels; we measured `autotune` ON = **3.7x slower
  decode** at this model scale (per-process shape-anchor tuning rounds, no
  persistent tune cache), so it stays OFF. A custom zero-gather
  paged-decode CubeCL kernel + persistent tune cache is future work.
- **Sampling**: CPU-side logits-processor chain (repetition/frequency/
  presence penalties → temperature → top-k → top-p) with greedy and seeded
  multinomial samplers (`--seed` reproducible); model `generation_config.json`
  defaults are honored (explicit flags win); stop tokens + boundary-safe
  stop strings; context-length guard against cache capacity.
- **Quantization**: `combs_core::quant::dequantize_q4` (GGUF q4_0 layout,
  portable `remainder`/`div` nibble extraction — no bitwise ops needed) and
  `combs_models::QuantizedLinear` keep weights packed in VRAM (4x footprint
  cut) and dequantize on-device before the matmul; a fused dequant-matmul
  kernel is future work. The GGUF `ModelSource` adapter lands in Phase 5.
- **f32 compute everywhere** (BF16/F16 weights are widened on load).
- **Memory**: cubecl's pooled allocator is used as-is; `combs_core::BufferPool`
  is a documented no-op facade reserved for arena management.
- RoPE uses the half-split (`rotate_half`) convention; GQA expands KV heads
  inside the cache impls; causal masking only where seq > 1.

## Known issues

- **cubecl 0.10 matmul silently no-ops on wgpu/Metal when M >= 512 AND
  K >= 512.** For any such matmul shape (any N, any tensor rank, contiguous
  or transposed rhs, fused or unfused cubecl backend — fusion is *not*
  involved), cubecl's matmul autotuner picks a tile configuration that
  requests 40960 B of shared memory while Metal exposes at most 32768 B. The
  dispatch fails wgpu validation, the error is only reported through wgpu's
  async error channel (never surfaced by cubecl/burn in release paths), the
  kernel never runs, and the zero-initialized output buffer is returned
  unwritten — deterministic garbage logits for any >= 512-token prefill.
  The cliff is exact (511 correct, 512 zeroed; verified against the NdArray
  CPU reference with identity-matrix inputs, where the GPU output is all
  zeros instead of B).
  - **Workaround (landed)**: `combs_models::matmul::safe_matmul` slabs any
    matmul with M >= 512 and K >= 512 into <= 256-row chunks along M and
    `cat`s the results — exact (each output row is still one K-reduction),
    zero cost below the boundary. All model matmuls (`linear`, lm_head,
    attention P@V) route through it.
  - **Repro / bisection**: `cargo test --release -p combs-models --test
    attn_cliff -- --ignored --nocapture` — minimal attend repro (511 vs 512),
    op bisection (softmax/mask/matmul stages), exact shape-boundary sweep,
    identity-matrix no-op proof, and the post-workaround KV-cache regression
    test (`kv_cache_attention_matches_cpu_above_cliff`).
