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
| **crates.io** (Rust core) | `combs-runtime`, `combs-ffi`, `combs-mesh`, … | `cargo add combs-mesh` |
| **JSR** (Deno/TS framework) | `@combs/core`, `@combs/graph`, `@combs/agents`, `@combs/mesh`, … | `deno add @combs/core` |
| **GitHub Releases** | prebuilt binaries per platform | [releases](https://github.com/asqrzk/CombsEngine/releases) |

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
│ L2 — Deno / TypeScript Framework   (Engine/Js)              │
│     @combs/core · @combs/graph · @combs/agents ·            │
│     @combs/runtime · @combs/flows · @combs/telemetry ·      │
│     @combs/observe · @combs/zerotrust · @combs/mesh         │
├─────────────────────────────────────────────────────────────┤
│ L1 — C ABI / JSON-FFI                (combs-ffi)            │
│     combs_engine_create · combs_chat_completion_stream ·    │
│     OpenAI-compatible server (`combs serve`)                │
├─────────────────────────────────────────────────────────────┤
│ L0 — Rust GPU Core                   (Engine/Core)          │
│     Burn 0.21 · CubeCL · wgpu (Metal/Vulkan/DX12/WebGPU)    │
│     Paged KV cache · full sampler · GGUF/SafeTensors        │
└─────────────────────────────────────────────────────────────┘
```

| Layer | Location | What it is |
|---|---|---|
| L0 Core | [`Core/`](Core) | Cargo workspace: `combs-core`, `combs-formats`, `combs-media`, `combs-models`, `combs-runtime`, `combs-ffi`, `combs-mesh`, `combs-mesh-ffi`, `combs-cli`, `xtask` |
| L2 Framework | [`Js/`](Js) | Deno workspace of 9 `@combs/*` packages |
| L3 UI | [`Ui/`](Ui) | Svelte 5 + Vite template consumed by `combs chew` |
| L3 Android | [`Android/`](Android) | JNI glue + Kotlin API over the FFI `.so` |
| L3 iOS | [`Ios/`](Ios) | Swift wrapper over the FFI static lib |

## What's Available

### L0 — Rust Core (`Engine/Core`)

- **Inference engine** — Llama-family architectures on Burn 0.21 + CubeCL/wgpu; validated with SmolLM2 on Apple Silicon (Metal). New architectures = one line in the `ModelRegistry`.
- **Model formats** — SafeTensors (Hugging Face layout) and **GGUF v3** (Q8_0 / Q4_0 / F16 / F32) through a single `ModelSource` adapter trait. `combs run --model anything.gguf` just works.
- **Memory** — paged KV cache (page-16 arenas, LIFO free-list, prefix-safe `popn`), chunked prefill, quantized linear layers with weights packed in VRAM.
- **Sampling** — temperature, top-k, top-p, repetition/frequency/presence penalties, seeded (byte-identical) generation, stop strings & stop tokens.
- **CLI (`combs`)** — `run` · `serve` · `devices` · `chew` · `pull` · `convert`.
- **OpenAI-compatible server** — `POST /v1/chat/completions` (SSE streaming + non-streaming), `GET /v1/models`, `GET /health`.
- **`xtask`** — cross-platform build orchestrator: `cargo xtask matrix` shows live toolchain detection; `cargo xtask bundle` produces `dist/<platform>/{lib, combs.h}` for every target.
- **CombsMesh emoji engine (`combs-mesh`)** — binary `.cmse` block format (10 block types: text/image/todo/functions/api/lifecycle/character/emotion/encryption/orchestration), Unicode PUA tag-character encoding, AES-256-GCM/ChaCha20 crypto with HKDF subkeys, CPU + wgpu sprite renderers, content-addressed registry, wasm32-clean with wasm-bindgen bindings; C ABI via `combs-mesh-ffi` (`combsmesh_*` + `combsmesh_op_json`).

### L2 — TypeScript Framework (`Engine/Js`)

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

### L3 — UI & Shells

- **`combs chew`** — scaffolder (interactive or fully flag-driven) that stamps out a ready-to-build Svelte 5 app:
  - 🔐 first-run keypair auth ritual + device passkey (WebAuthn) for permission approvals
  - 🛡️ fine-grained network/storage permission grants ("allow once / this session / always")
  - 📊 realtime network + storage monitor
  - 🗼 **Control Tower** — realtime observability view (sources, runs, context, network, permissions) fed by `@combs/observe`
  - 🌗 dark/light themes, responsive; **chat**, **debate**, **roleplay** (two engine processes), and **multi-turn** (chat + Control Tower) views
- **Android** — JNI bridge + Kotlin `CombsEngine` API.
- **iOS** — Swift wrapper over the C ABI.
- **Web** — `combs-wasm` skeleton + `WorkerEngine` transport (WebGPU enablement in progress).

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
| GGUF — Q8_0 / Q4_0 | ✅ full |
| GGUF — Q4_K / Q5_K / Q6_K | 🚧 planned |

Model presets **must** use instruction-tuned variants (e.g. `SmolLM2-135M-Instruct`) — base models echo prompts in chat mode.

## Platform Matrix

| Target | Artifact | Status |
|---|---|---|
| macOS arm64 / x86_64 | `libcombs_ffi.dylib` | ✅ built & tested |
| iOS arm64 | `libcombs_ffi.a` | ✅ builds |
| Android arm64 | `libcombs_ffi.so` | ✅ builds (NDK) |
| Linux / Windows x86_64 | — | ✅ `cargo check` clean (linking needs native toolchain) |
| Web (wasm32) | — | 🚧 skeleton, checks clean |

Run `cargo xtask matrix` for live detection of your local toolchains.

## Development

```bash
# Rust — 53 tests (release profile saves disk)
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"   # rustup is keg-only on macOS
cargo test --release --workspace

# TypeScript — 33 tests
cd Js && deno task test

# Cross-build everything into dist/
cargo xtask bundle
```

Repo conventions live in the root [`AGENTS.md`](../AGENTS.md).

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
