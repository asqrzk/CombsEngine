# Combs Engine

[![ci](https://github.com/asqrzk/CombsEngine/actions/workflows/ci.yml/badge.svg)](https://github.com/asqrzk/CombsEngine/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/asqrzk/CombsEngine)](https://github.com/asqrzk/CombsEngine/releases)
[![npm](https://img.shields.io/npm/v/@combs-edge/combs-engine?label=npm)](https://www.npmjs.com/package/@combs-edge/combs-engine)
[![PyPI](https://img.shields.io/pypi/v/combs-engine?label=pypi)](https://pypi.org/project/combs-engine/)
[![crates.io](https://img.shields.io/crates/v/combs-mesh?label=crates.io)](https://crates.io/crates/combs-mesh)
[![JSR](https://img.shields.io/jsr/v/@combs/core?label=jsr)](https://jsr.io/@combs/core)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

> Local-first AI inference engine + agent framework — one Rust GPU core, everywhere.

Combs Engine runs large language models **on-device** with a single Rust core compiled for macOS, iOS, Android, Linux, Windows, and Web (WASM) — and exposes it through a C ABI to a full TypeScript agent framework, an OpenAI-compatible server, native mobile shells, and a one-command UI scaffolder.

## Install

| Channel | Package | Command |
|---|---|---|
| **npm** (CLI + engine binary) | `@combs-edge/combs-engine` | `npm i -g @combs-edge/combs-engine` |
| **npm** (JS client for `combs serve`) | `@combs-edge/combs-client` | `npm i @combs-edge/combs-client` |
| **npm** (zero-trust crypto) | `@combs-edge/combs-zerotrust` | `npm i @combs-edge/combs-zerotrust` |
| **npm** (CombsMesh emoji FFI lib) | `@combs-edge/combs-mesh` | `npm i @combs-edge/combs-mesh` |
| **PyPI** (CLI wrapper) | `combs-engine` | `pip install combs-engine` |
| **crates.io** (CLI binary, built from source) | `combs-cli` | `cargo install combs-cli` |
| **crates.io** (Rust core) | `combs-runtime`, `combs-diffusion`, `combs-ffi`, `combs-mesh`, … | `cargo add combs-mesh` |
| **JSR** (Deno/TS framework) | `@combs/core`, `@combs/graph`, `@combs/agents`, `@combs/mesh`, … | `deno add @combs/core` |
| **GitHub Releases** | prebuilt binaries per platform | [releases](https://github.com/asqrzk/CombsEngine/releases) |

Platform-by-platform from-scratch instructions (macOS / Linux / Windows, binaries or source): [INSTALL.md](INSTALL.md).

Or build from source:

```bash
# Build the CLI
cargo xtask build --release

# Run a model (HF directory or .gguf file — auto-detected)
./target/release/combs run --model ~/.cache/combs/models/smollm2-135m --chat

# Scaffold a full chat UI (Svelte 5, dark/light, auth, permissions)
./target/release/combs chew chat-ui my-app --yes

# ...or a multi-agent debate UI
./target/release/combs chew debate-ui my-debate --topic "Pineapple on pizza" --turns 6

# Serve an OpenAI-compatible API
./target/release/combs serve --model ~/.cache/combs/models/smollm2-135m --port 11434
```

---

## The Stack

Four layers. Compute lives exclusively in **L0** — every layer above is thin orchestration over the same engine.

```
┌─────────────────────────────────────────────────────────────┐
│ L3 — Platform Shells                                        │
│     Svelte 5 UI template · Android (JNI/Kotlin) ·           │
│     iOS (Swift) · Web Worker transport                      │
├─────────────────────────────────────────────────────────────┤
│ L2 — Deno / TypeScript Framework   (engine/js)              │
│     @combs/core · @combs/graph · @combs/agents ·            │
│     @combs/runtime · @combs/flows · @combs/telemetry ·      │
│     @combs/observe · @combs/zerotrust · @combs/mesh         │
├─────────────────────────────────────────────────────────────┤
│ L1 — C ABI / JSON-FFI                (combs-ffi)            │
│     combs_engine_create · combs_chat_completion_stream ·    │
│     OpenAI-compatible server (`combs serve`)                │
├─────────────────────────────────────────────────────────────┤
│ L0 — Rust GPU Core                   (engine/core)          │
│     Burn 0.21 · CubeCL · wgpu (Metal/Vulkan/DX12/WebGPU)    │
│     Paged KV cache · full sampler · GGUF/SafeTensors        │
└─────────────────────────────────────────────────────────────┘
```

| Layer | Location | What it is |
|---|---|---|
| L0 Core | [`engine/core`](engine/core) | Cargo workspace: `combs-core`, `combs-formats`, `combs-media`, `combs-models`, `combs-runtime`, `combs-ffi`, `combs-mesh`, `combs-mesh-ffi`, `combs-cli`, `xtask` |
| L2 Framework | [`engine/js`](engine/js) | Deno workspace of 9 `@combs/*` packages |
| L3 UI | [`engine/ui`](engine/ui) | Svelte 5 + Vite template consumed by `combs chew` |
| L3 Android | [`engine/android`](engine/android) | JNI glue + Kotlin API over the FFI `.so` |
| L3 iOS | [`engine/ios`](engine/ios) | Swift wrapper over the FFI static lib |

## What's Available

### L0 — Rust Core (`engine/core`)

- **Inference engine** — one universal decoder on Burn 0.21 + CubeCL/wgpu serving Llama/SmolLM2, Qwen2.5, Qwen3, Phi-3, and Gemma-3 (sliding-window + dual-RoPE variants included), plus SmolVLM vision input, Stable Diffusion 1.5 image generation, Kokoro TTS, and **Whisper speech-to-text** (WAV → 16 kHz log-mel → encoder-decoder transcription). New architectures = a registry line + an `ArchSpec` preset.
- **Model formats** — SafeTensors (Hugging Face layout, causal-LM or bare base exports) and **GGUF v3** (F32 / F16 / Q4_0 / Q5_0 / Q8_0 / Q4_K / Q5_K / Q6_K) through a single `ModelSource` adapter trait. `combs run --model anything.gguf` just works.
- **LoRA adapters** — `combs pull` detects adapter repos structurally (diffusers / kohya / PEFT layouts probed from the safetensors header, base family fingerprinted) and caches them beside models; `--lora <file|cache-id> --lora-scale s` on `serve`, `serve-images`, and `generate-image` fuses the adapter into dense or quantized bases at load — zero added latency after the merge, and `/v1/stats` echoes exactly what was fused.
- **Honest loading** — a pre-flight fit check refuses models whose
  largest allocation exceeds the adapter's binding cap (naming the
  tensor, the need, and the limit) instead of serving corrupt weights;
  load-phase panics abort the process; uploads are synced at load so
  `/v1/stats.gpu` is truthful from startup. `combs devices` prints the
  allocation ceilings that decide fit.
- **Graph compute (`combs-graphkit`)** — looped, convergent graph
  algorithms (spreading activation, decayed k-hop) with a CPU
  reference and a Burn/wgpu GPU path, exposed through a single
  JSON-op C ABI (`combsgraph_op_json`).
- **Generation observability** — spawning with `COMBS_PROGRESS=json` emits structured load stages (open / per-layer weights / bind) on stderr for any supervisor to parse; the image worker's `/v1/stats` reports a live `generation` object during a run — step k/N, phase (encode / denoise / decode / png), measured `eta_ms` — and with `--preview-every N` it serves the emerging image at `/v1/preview`. Generations are single-flight; the pipeline mutex serializes concurrent requests.
- **Memory** — paged KV cache (page-16 arenas, LIFO free-list, prefix-safe `popn`), chunked prefill, quantized linear layers with weights packed in VRAM.
- **Sampling** — temperature, top-k, top-p, min-p, repetition/frequency/presence penalties **bounded to a recent window** (`repeat_last_n`, default 128; `0` restores whole-context) with stop tokens always exempt — so long chats can't penalize their own ability to stop — plus logit_bias, per-token logprobs, seeded (byte-identical) generation, stop strings & stop tokens; opt-in prompt-lookup speculative decoding for greedy runs (`COMBS_SPEC=1`).
- **Streaming that survives UTF-8** — the incremental detokenizer holds back split multi-byte characters (every emoji, CJK on byte-level vocabs) until they complete, so streamed text never shows replacement chars or drops characters; buffered and streamed output are byte-identical.
- **CLI (`combs`)** — `run` · `serve` · `perplexity` · `transcribe` · `pull` · `devices` · `chew` · `generate-image` · `generate-audio` · `serve-images` · `serve-audio` · `convert`.
- **OpenAI-compatible server** — `/v1/chat/completions` (SSE + non-streaming, native **tool calling** through each model's own chat template, `response_format` json_object/json_schema constrained output, logprobs, `n` choices), `/v1/completions` (raw prompt / FIM), `/v1/embeddings` (pooling detection, matryoshka `dimensions`, base64), `/v1/models` + `/v1/model/info` (per-model capability advertisement), `/v1/sessions` (list + release KV sessions), `/v1/stats` (totals with cancelled-vs-error accounting, per-session KV pages with measured per-layer arena state, attention geometry, last-request timings), `/health` — plus `/v1/images/generations`, `/v1/audio/speech`, and `/v1/audio/transcriptions` (multipart or raw WAV) on the media workers.
- **`xtask`** — cross-platform build orchestrator: `cargo xtask matrix` shows live toolchain detection; `cargo xtask bundle` produces `dist/host/{combs, combs-f32}` (the default `combs` computes in f16, with image generation pinned to f32 internally; `combs-f32` is the full-precision build) plus `dist/<platform>/{lib, combs.h}` for every target.
- **CombsMesh emoji engine (`combs-mesh`)** — binary `.cmse` block format (10 block types: text/image/todo/functions/api/lifecycle/character/emotion/encryption/orchestration), Unicode PUA tag-character encoding, AES-256-GCM/ChaCha20 crypto with HKDF subkeys, CPU + wgpu sprite renderers, content-addressed registry, wasm32-clean with wasm-bindgen bindings; C ABI via `combs-mesh-ffi` (`combsmesh_*` + `combsmesh_op_json`).

### L2 — TypeScript Framework (`engine/js`)

| Package | Purpose |
|---|---|
| `@combs/core` | `Combs.init("smollm2-135m")` — presets, model cache, device planner, FFI / Remote / Worker engines |
| `@combs/graph` | LangGraph-equivalent: StateGraph, channels, Pregel runner, checkpointers (Memory/Deno KV/SQLite), human-in-the-loop interrupts, streaming |
| `@combs/agents` | Tools & ToolNode, ReAct agents, structured output, memory, MCP client, skills loader |
| `@combs/runtime` | Agent HTTP/WS servers, orchestrator, agent pool, KV task queue, session stores |
| `@combs/flows` | `createWorkflow` (steps/loops/checks), `createRoleplayChat`, `addMemory` |
| `@combs/telemetry` | Scoped logging, OpenTelemetry-shaped spans, metrics |
| `@combs/observe` | Realtime observability bus (isomorphic core) — EventBus, instrument middleware (`wrapEngine`/`instrumentFetch`/`span`), sinks (memory/NDJSON/WebSocket), redaction; powers the Control Tower |
| `@combs/mesh` | CombsMesh emoji client — FFI `Mesh` wrapper, pure-TS Unicode PUA codec (byte-parity with Rust), MCP server mode, `MeshPeer` WS connector with sha256-verified fetch |
| `@combs/memory` | Knowledge-graph memory — SQLite entities/relations with a usage-driven lifecycle, ranked recall, repository ingestion (`graphify`), at-rest crypto door, embeddings-hybrid retrieval, and an MCP stdio server exposing the graph as tools |

### L3 — UI & Shells

- **`combs chew`** — scaffolder (interactive or fully flag-driven) that stamps out a ready-to-build Svelte 5 app:
  - 🔐 first-run keypair auth ritual + device passkey (WebAuthn) for permission approvals
  - 🛡️ fine-grained network/storage permission grants ("allow once / this session / always")
  - 📊 realtime network + storage monitor
  - 🗼 **Control Tower** — realtime observability view (sources, runs, context, network, permissions) fed by `@combs/observe`
  - 🌗 dark/light themes, responsive; **chat**, **debate**, **roleplay** (two engine processes), and **multi-turn** (chat + Control Tower) views
- **Android** — JNI bridge + Kotlin `CombsEngine` API.
- **iOS** — Swift wrapper over the C ABI.
- **Web** — `combs-wasm` runs full models in a browser tab over WebGPU: streamed GGUF mount (no whole-file buffering), an always-on batched-matmul value canary, and a worker transport. Measured: a 4.7 GB Qwen2.5-Coder-7B mounts in ~3 s at a 1.49 GB linear-memory peak and chats, gated headless in CI-style runs.

## Performance

Measured on Apple Silicon (wgpu → Metal), SmolLM2-135M-Instruct:

| Metric | Value |
|---|---|
| Decode | ~35 tok/s |
| Prefill (1261-token prompt) | ~660 tok/s |
| TTFT | ~200 ms |
| FFI streaming (via Deno) | ~31 tok/s |

## Model Formats

| Format | Status |
|---|---|
| SafeTensors (HF layout) | ✅ full |
| GGUF — F32 / F16 | ✅ full |
| GGUF — Q4_0 / Q5_0 / Q8_0 | ✅ full, fused GPU kernels |
| GGUF — Q4_K / Q5_K / Q6_K | ✅ full, fused GPU kernels |

Model presets **must** use instruction-tuned variants (e.g. `SmolLM2-135M-Instruct`) — base models echo prompts in chat mode.

## Platform Matrix

| Target | Artifact | Status |
|---|---|---|
| macOS arm64 / x86_64 | `libcombs_ffi.dylib` | ✅ built & tested |
| iOS arm64 | `libcombs_ffi.a` | ✅ builds |
| Android arm64 | `libcombs_ffi.so` | ✅ builds (NDK) |
| Linux / Windows x86_64 | — | ✅ `cargo check` clean (linking needs native toolchain) |
| Web (wasm32) | `combs_wasm` module (34 MB) | ✅ 7B mounts + chats in a tab (streamed, suite-gated) |

Run `cargo xtask matrix` for live detection of your local toolchains.

## Development

```bash
# Rust — full workspace suite (release profile saves disk)
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"   # rustup is keg-only on macOS
cargo test --release --workspace

# TypeScript — 33 tests
cd Js && deno task test

# Cross-build everything into dist/
cargo xtask bundle
```

## Contributing

Contributions are welcome! To get started:

1. **Fork** the repository and create a branch from `main`.
2. **Keep the layering rule**: L2/L3 only orchestrate — never move compute out of the Rust core (L0).
3. **Match the existing style** — minimal, no external UI/component deps, equivalence-tested.
4. **Add tests** — CPU unit tests with the NdArray backend; GPU tests behind `#[ignore]` + `COMBS_TEST_MODEL`.
5. **Run the full suite** (`cargo test --release --workspace` and `deno task test`) before opening a PR.
6. Open a **pull request** with a clear description of the change.

Good first contributions: additional GGUF K-quant dequant kernels, new model architectures for the registry, platform shell improvements.

Please report bugs and request features via **GitHub Issues**.

## License

[MIT](LICENSE) © Combs Engine contributors — free to use, modify, and distribute, commercially or otherwise.
