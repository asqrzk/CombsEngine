# Combs Engine

> Local-first AI inference engine + agent framework — one Rust GPU core, everywhere.

Combs Engine runs large language models **on-device** with a single Rust core compiled for macOS, iOS, Android, Linux, Windows, and Web (WASM) — and exposes it through a C ABI to a full TypeScript agent framework, an OpenAI-compatible server, native mobile shells, and a one-command UI scaffolder.

```bash
# Scaffold a full chat UI — one command does everything:
# copies the template, writes config, runs npm install, starts the dev server
combs chew chat-ui my-app --yes

# ...or a multi-agent debate UI
combs chew debate-ui my-debate --topic "Pineapple on pizza" --turns 6

# ...or a two-role roleplay UI — each role runs on its OWN engine process
# (the UI asks for the roles, then spawns a second `combs serve` on a new port)
combs chew roleplay-ui my-roleplay --model smollm2-135m

# Download + cache a model preset (Hugging Face → ~/.cache/combs/models)
combs pull smollm2-135m

# Run a model directly (preset id, HF directory, or .gguf file)
combs run --model smollm2-135m --prompt "Hello" --chat

# Serve an OpenAI-compatible API
combs serve --model smollm2-135m --port 8080
```

> `combs chew` is self-contained: the Svelte template is **embedded in the
> binary** (extracted to `~/.cache/combs/ui-template/<version>` on first use,
> override with `COMBS_UI_TEMPLATE`), so it works from any install — no repo
> checkout needed. It also **downloads the model if it isn't cached** (via
> `combs pull`) and **starts `combs serve` for you** before launching the UI.
> Use `--no-install` / `--no-start` to skip the automatic steps.
>
> Model weights are cached as plaintext in `~/.cache/combs/models`: the
> engine mmaps them directly (encrypting public weights would break mmap
> for zero threat-model gain). Zero-trust encryption-at-rest covers
> everything crossing the UI proxy — chats, downloads, agent data.

---

## Installation

Pick whatever fits your stack — every option installs the same `combs` binary:

| Channel | Command | Notes |
|---|---|---|
| **Prebuilt binary** | download from [GitHub Releases](https://github.com/asqrzk/CombsEngine/releases) | macOS arm64/x86_64, Linux x86_64, Windows x86_64 — built natively per-OS by CI |
| **npm** | `npm install -g combs-engine` | wrapper downloads the matching release binary ([source](Packages/npm/combs-engine)) |
| **npm** (JS client) | `npm install combs-client` | zero-dep browser/Node client for `combs serve` — used by the chew UI template ([source](Packages/npm/combs-client)) |
| **pip** | `pip install combs-engine` | same, for Python environments ([source](Packages/pypi/combs-engine)) |
| **cargo** | `cargo install --path Engine/Core/combs-cli` | builds from source — works on **any** OS with a Rust toolchain |
| **JSR** (TS libs) | `deno add @combs/core @combs/graph ...` | the 6 framework packages, published via `deno publish` in `Js/` |

### Building from source (including non-macOS)

The engine builds natively on any platform wgpu supports (Metal, Vulkan,
DX12) — no cross-compilation is involved when you build **on** the machine
you target:

```bash
# Linux / Windows / macOS — same steps everywhere
git clone https://github.com/asqrzk/CombsEngine.git
cd CombsEngine/Engine/Core
cargo build --release -p combs-cli        # CLI:  ./target/release/combs[.exe]
cargo build --release -p combs-ffi        # C ABI: libcombs_ffi.{so,dylib,dll}
```

Prerequisites: a stable Rust toolchain (`rustup`) — that's it on Linux and
Windows. The `xtask` cross-compile helpers (`cargo xtask bundle`, iOS/Android
targets) are a macOS-developer convenience; they are **not** required to use
the engine. CI (`.github/workflows/release.yml`) builds every desktop
platform natively on its own runner for exactly this reason.

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
│     @combs/runtime · @combs/flows · @combs/telemetry        │
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
| L0 Core | [`Core/`](Core) | Cargo workspace: `combs-core`, `combs-formats`, `combs-models`, `combs-runtime`, `combs-ffi`, `combs-cli`, `xtask` |
| L2 Framework | [`Js/`](Js) | Deno workspace of 6 `@combs/*` packages |
| L3 UI | [`Ui/`](Ui) | Svelte 5 + Vite template consumed by `combs chew` |
| L3 Android | [`Android/`](Android) | JNI glue + Kotlin API over the FFI `.so` |
| L3 iOS | [`Ios/`](Ios) | Swift wrapper over the FFI static lib |

## What's Available

### L0 — Rust Core (`Engine/Core`)

- **Inference engine** — Llama-family architectures on Burn 0.21 + CubeCL/wgpu; validated with SmolLM2 on Apple Silicon (Metal). New architectures = one line in the `ModelRegistry`.
- **Model formats** — SafeTensors (Hugging Face layout) and **GGUF v3** (Q8_0 / Q4_0 / F16 / F32) through a single `ModelSource` adapter trait. `combs run --model anything.gguf` just works.
- **Memory** — paged KV cache (page-16 arenas, LIFO free-list, prefix-safe `popn`), chunked prefill, quantized linear layers with weights packed in VRAM.
- **Sampling** — temperature, top-k, top-p, min-p, repetition/frequency/presence penalties windowed over recent history (`repeat_last_n`, stop tokens exempt), seeded (byte-identical) generation, stop strings & stop tokens, UTF-8-safe incremental detokenization for streaming.
- **CLI (`combs`)** — `run` · `serve` · `perplexity` · `transcribe` · `pull` (HF download → local cache; presets, adapters, or any repo) · `devices` · `chew` · `generate-image` · `generate-audio` · `serve-images` · `serve-audio` · `convert`.
- **OpenAI-compatible server** — `POST /v1/chat/completions` (SSE streaming + non-streaming), `GET /v1/models`, `GET /v1/sessions` (list/release KV sessions), `GET /v1/stats` (cancelled-vs-error totals, per-layer KV arena state per session), `GET /health`.
- **`xtask`** — cross-platform build orchestrator: `cargo xtask matrix` shows live toolchain detection; `cargo xtask bundle` produces `dist/<platform>/{lib, combs.h}` for every target.

### L2 — TypeScript Framework (`Engine/Js`)

| Package | Purpose |
|---|---|
| `@combs/core` | `Combs.init("smollm2-135m")` — presets, model cache, device planner, FFI / Remote / Worker engines |
| `@combs/graph` | LangGraph-equivalent: StateGraph, channels, Pregel runner, checkpointers (Memory/Deno KV/SQLite), human-in-the-loop interrupts, streaming |
| `@combs/agents` | Tools & ToolNode, ReAct agents, structured output, memory, MCP client, skills loader |
| `@combs/runtime` | Agent HTTP/WS servers, orchestrator, agent pool, KV task queue, session stores |
| `@combs/flows` | `createWorkflow` (steps/loops/checks), `createRoleplayChat`, `addMemory` |
| `@combs/telemetry` | Scoped logging, OpenTelemetry-shaped spans, metrics |
| `@combs/zerotrust` | Zero-trust crypto core: identity (ECDSA+ECDH), sealed envelopes, at-rest keystore, capability tokens — pure WebCrypto, browser/Node/Deno |

### L3 — UI & Shells

- **`combs chew`** — scaffolder (interactive or fully flag-driven) that stamps out a ready-to-run Svelte 5 app, then installs deps and starts the dev server for you:
  - 🔐 first-run keypair ritual (ECDSA P-256 signing + ECDH P-256 encryption keys) — **always on, no configuration escapes it**; a model must always be chosen explicitly (`--model`, mandatory with `--yes`)
  - 🪪 **device passkey** (WebAuthn via [SimpleWebAuthn](https://github.com/MasterKale/SimpleWebAuthn)): every permission approval — model downloads, agent internet access, storage — is confirmed with Touch ID / Windows Hello / a security key, verified server-side by the proxy
  - 🛡️ **backend permission proxy** (`server/proxy.mjs`): the browser never touches the internet or disk directly — every request goes through `POST /api/relay`, every file write through `/api/files/*`, and the proxy enforces the grants server-side. The frontend only renders the dialog and forwards decisions
  - 🔒 **zero-trust storage** ([`combs-zerotrust`](Packages/npm/combs-zerotrust)): everything written to disk or localStorage is AES-256-GCM encrypted with per-blob wrapped keys + SHA-256 integrity hashes; tampered data is rejected on read (409), never served
  - 🤖 **E2E agent channels**: agent subprocesses exchange public keys once (permission-checked) and then communicate only in sealed envelopes (ephemeral ECDH → HKDF → AES-GCM + hash + ECDSA signature) — no re-permission for replies, tampering rejected
  - 🌐 **sandboxed internet for agents** (token1/token2): agents never get raw web access — a per-agent sandbox proxy with allowlists, secret-leak guardrails and an encrypted audit log brokers it, with responses re-encrypted to the main tab
  - 📊 realtime network + storage monitor fed by the proxy (`/api/monitor`) — the backend sees every byte, so it's the source of truth
  - 📦 talks to `combs serve` via the official [`combs-client`](Packages/npm/combs-client) npm package
  - 🎭 **roleplay mode**: define two roles in the UI → the proxy spawns a second `combs serve` subprocess on its own port (permission-gated `system:subprocess`) — separate process, separate GPU device per role; engines cleaned up on exit
  - 🌗 dark/light themes, responsive, chat & multi-agent debate & roleplay views
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
| macOS arm64 / x86_64 | `libcombs_ffi.dylib` | ✅ built & tested (CI release binaries) |
| Linux / Windows x86_64 | `libcombs_ffi.so` / `.dll` | ✅ built natively by CI release workflow |
| iOS arm64 | `libcombs_ffi.a` | ✅ builds (from macOS via `xtask`) |
| Android arm64 | `libcombs_ffi.so` | ✅ builds (NDK, from macOS via `xtask`) |
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

### Releasing

- **CLI binaries**: `git tag v0.x.0 && git push origin v0.x.0` → the release workflow builds macOS/Linux/Windows natively and attaches `combs-<version>-<platform>-<arch>` archives to the GitHub Release.
- **npm**: `Packages/npm/combs-client` and `Packages/npm/combs-zerotrust` (the libraries the chew UIs depend on) plus `Packages/npm/combs-engine` (CLI wrapper; postinstall fetches the release asset). Full step-by-step for every channel: **[Packages/RELEASING.md](Packages/RELEASING.md)**.
- **PyPI**: bump `Packages/pypi/combs-engine/pyproject.toml`, `python -m build && twine upload dist/*`.
- **JSR (TS packages)**: bump each `Js/<pkg>/deno.json` version, then `deno publish` from `Js/` (workspace-aware, publishes all six `@combs/*` packages).

Please report bugs and request features via **GitHub Issues**.

## License

[MIT](LICENSE) © Combs Engine contributors — free to use, modify, and distribute, commercially or otherwise.
