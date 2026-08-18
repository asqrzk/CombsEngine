# Running Combs from scratch

Every path below ends at the same place: a `combs` binary serving an
OpenAI-compatible API on your machine. Pick the shortest route for your
platform, or build from source — the build needs nothing beyond a
stable Rust toolchain on all three OSes.

## 1. The fast path (no toolchain)

### macOS / Linux

```bash
# npm (recommended — postinstall fetches the prebuilt binary for your platform)
npm install -g @combs-edge/combs-engine

# or PyPI (first `combs` run downloads the binary into ~/.cache/combs/bin)
pip install combs-engine

# or grab the archive directly
# https://github.com/asqrzk/CombsEngine/releases
tar -xzf combs-<version>-macos-arm64.tar.gz     # or linux-x86_64
./combs-<version>-macos-arm64/combs --help
```

### Windows

```powershell
# npm (Node 18+)
npm install -g @combs-edge/combs-engine

# or PyPI
pip install combs-engine

# or the release zip
Expand-Archive combs-<version>-windows-x86_64.zip
.\combs-<version>-windows-x86_64\combs.exe --help
```

## 2. From source (all platforms)

Prerequisites: [Rust](https://rustup.rs) (stable) and git. Nothing
else — CI builds macOS (arm64 + Intel), Linux, and Windows from exactly
these steps with no extra system packages.

```bash
git clone https://github.com/asqrzk/CombsEngine.git
cd CombsEngine/engine/core

# CLI + FFI libraries (what the release ships)
cargo build --release -p combs-cli -p combs-ffi -p combs-mesh-ffi

./target/release/combs --help          # combs.exe on Windows
```

On a case-sensitive filesystem (Linux), paths matter: build from the
repo root layout as cloned — the CLI embeds the UI template from
`engine/ui/template` at build time.

## 3. Get a model and run

```bash
# Pull a preset from Hugging Face into the local cache
combs pull smollm2-135m            # small, quick first run
combs pull qwen2.5-coder-7b        # 4.4 GB, needs ~6 GB free memory

# One-shot chat in the terminal
combs run --model ~/.cache/combs/models/smollm2-135m --chat

# Serve the OpenAI-compatible API
combs serve --model ~/.cache/combs/models/smollm2-135m --port 11434
curl http://127.0.0.1:11434/v1/models
```

Any OpenAI client works against `http://127.0.0.1:<port>/v1` — chat
completions (SSE streaming or buffered), tool calling, JSON-schema
constrained output, embeddings, plus `/v1/stats` for live engine truth
(KV sessions, per-layer arena state, timings).

## 4. GPU notes per platform

The core runs on wgpu, so it uses whatever your OS exposes:

| OS | Backend | Needs |
|---|---|---|
| macOS | Metal | nothing — works out of the box on Apple Silicon and Intel |
| Linux | Vulkan | Vulkan drivers (`mesa-vulkan-drivers` on Debian/Ubuntu, `vulkan-radeon`/`nvidia` per GPU) |
| Windows | DX12 / Vulkan | a current GPU driver |

No adapter at all (headless server without Vulkan)? The engine cannot
run GPU inference there; kernels require a device. CPU fallback is on
the roadmap, not shipped.

## 5. The web platform (optional)

[CombsLLM](https://github.com/asqrzk/CombsLLM) puts a full chat UI,
live engine monitor, and model manager on top of `combs serve`:

```bash
git clone https://github.com/asqrzk/CombsLLM.git
cd CombsLLM
deno task hive     # starts the engine (if not already healthy) + platform
# → http://localhost:8787  (first run: create your passkey)
```

Needs [Deno](https://deno.com) 2.x. `COMBS_BIN` / `COMBS_MODEL`
override which binary and model the hive boots.

## Troubleshooting

- **Build fails with `No space left on device`** — a release build of
  the workspace peaks well over 10 GB of target-dir intermediates.
  `cargo clean` reclaims it.
- **`UI template not found`** — the source tree was flattened or
  partially copied; build from a full clone so `engine/ui/template`
  exists relative to `engine/core`.
- **Server seems to hang on first request** — large models memory-map
  and warm up on first prefill; watch `/v1/stats` (`uptime_s` ticking,
  then `totals.requests` moving) rather than assuming a hang.
- **Port already in use** — another engine instance is listening; pick
  another `--port` or stop the old process.
- **Linux: `No possible adapter available`** — Vulkan drivers are
  missing; install your distribution's Vulkan package for the GPU.
