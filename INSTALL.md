# Running Combs from scratch

Truly from scratch: a fresh machine with no toolchains. Pick a route,
install its prerequisites for your OS, then follow the route. Source
builds are exercised by CI on real macOS (arm64 + Intel), Linux, and
Windows runners with exactly the commands shown here.

## 0. Choose your route

| I want… | Route | Needs |
|---|---|---|
| the `combs` CLI, quickest | **A. npm install** | Node 18+ |
| the CLI via Python tooling | **B. pip install** | Python 3.9+ |
| the CLI, no runtimes at all | **C. release archive** | nothing |
| to build it myself | **D. from source** | Rust + git |
| the full web platform (chat UI, monitor) | **E. CombsLLM** | Deno 2 + one of A–D |

---

## 1. Prerequisites, per platform

Install only what your chosen route needs.

### macOS

Option 1 — **Homebrew** (install brew first from [brew.sh](https://brew.sh) if missing):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install node        # route A
brew install python      # route B
brew install rustup git  # route D  (then: rustup default stable)
brew install deno        # route E
```

Option 2 — official installers, no package manager:

- Node: [nodejs.org](https://nodejs.org) → macOS installer (.pkg)
- Python: [python.org/downloads](https://www.python.org/downloads/)
- Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Deno: `curl -fsSL https://deno.land/install.sh | sh`
- git: ships with Xcode Command Line Tools — `xcode-select --install`

### Linux (Debian/Ubuntu shown; Fedora/Arch equivalents noted)

Option 1 — distribution packages where they're current, vendor scripts where they're not:

```bash
sudo apt update && sudo apt install -y git curl build-essential   # dnf groupinstall "Development Tools" / pacman -S base-devel git

# Node 18+ (distro node is often too old — use NodeSource or nvm)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt install -y nodejs
# ...or nvm (no sudo, any distro):
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
nvm install --lts

# Python (route B) — usually preinstalled; else:
sudo apt install -y python3 python3-pip                            # dnf install python3-pip / pacman -S python-pip

# Rust (route D) — rustup, not distro rust (distro versions lag):
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Deno (route E):
curl -fsSL https://deno.land/install.sh | sh

# GPU runtime (all routes — see §6): Vulkan drivers
sudo apt install -y mesa-vulkan-drivers                            # or your GPU vendor's driver
```

### Windows

Option 1 — **winget** (preinstalled on Windows 10 21H2+ / 11), from PowerShell:

```powershell
winget install Git.Git                 # routes D, E
winget install OpenJS.NodeJS.LTS       # route A
winget install Python.Python.3.12      # route B
winget install Rustlang.Rustup         # route D — see MSVC note below
winget install DenoLand.Deno           # route E
```

Option 2 — **Chocolatey** ([chocolatey.org/install](https://chocolatey.org/install)):

```powershell
choco install git nodejs-lts python rustup.install deno
```

Option 3 — official installers, no package manager: [git-scm.com](https://git-scm.com), [nodejs.org](https://nodejs.org), [python.org](https://www.python.org/downloads/windows/), [rustup.rs](https://rustup.rs) (`rustup-init.exe`), [deno.com](https://deno.com).

> **MSVC note (route D only):** Rust's default Windows toolchain links
> with MSVC. If you don't have Visual Studio, install the free **Build
> Tools for Visual Studio** with the "Desktop development with C++"
> workload — `winget install Microsoft.VisualStudio.2022.BuildTools`,
> then select the C++ workload in its installer. `rustup-init` prompts
> for this too. Reopen the terminal after installing so PATH updates.

---

## 2. Route A — npm

```bash
npm install -g @combs-edge/combs-engine
combs --help
```

The postinstall step downloads the prebuilt binary for your OS/arch
from GitHub Releases. Works identically in PowerShell on Windows.

## 3. Route B — pip

```bash
pip install combs-engine          # py -m pip install combs-engine on Windows
combs --help
```

The first `combs` invocation downloads the platform binary into
`~/.cache/combs/bin` (`%USERPROFILE%\.cache\combs\bin` on Windows).

## 4. Route C — release archive (no runtimes)

Download from [releases](https://github.com/asqrzk/CombsEngine/releases):

```bash
# macOS (arm64 shown; use x86_64 on Intel)
tar -xzf combs-<version>-macos-arm64.tar.gz
./combs-<version>-macos-arm64/combs --help

# Linux
tar -xzf combs-<version>-linux-x86_64.tar.gz
./combs-<version>-linux-x86_64/combs --help
```

```powershell
# Windows
Expand-Archive combs-<version>-windows-x86_64.zip
.\combs-<version>-windows-x86_64\combs.exe --help
```

Each archive also carries the C FFI libraries (`libcombs_ffi.*` /
`combs_ffi.dll` + `combs.h`, and the CombsMesh pair) for embedding.

macOS Gatekeeper may quarantine a downloaded binary; clear it with
`xattr -d com.apple.quarantine <path>/combs` if launch is blocked.

## 5. Route D — from source (all platforms)

Prerequisites: Rust (stable, via rustup) and git — §1. No other system
packages: CI builds all three OSes from exactly these steps.

```bash
git clone https://github.com/asqrzk/CombsEngine.git
cd CombsEngine/engine/core

cargo build --release -p combs-cli -p combs-ffi -p combs-mesh-ffi

./target/release/combs --help        # target\release\combs.exe on Windows
```

Notes:

- First build compiles the full dependency tree — expect several
  minutes and >10 GB of `target/` intermediates (`cargo clean` reclaims
  it later).
- Build from a full clone: the CLI embeds the UI template from
  `engine/ui/template` at build time, and Linux filesystems are
  case-sensitive about it.
- Run the test suite with `cargo test --release --workspace`. On
  machines with no GPU adapter, GPU kernel tests skip themselves and
  say so.

## 6. Get a model and run (all routes)

```bash
# Pull a preset from Hugging Face into the local cache
combs pull smollm2-135m            # small, quick first run
combs pull qwen2.5-coder-7b        # 4.4 GB download, wants ~6 GB free memory

# One-shot chat in the terminal
combs run --model ~/.cache/combs/models/smollm2-135m --chat

# Serve the OpenAI-compatible API
combs serve --model ~/.cache/combs/models/smollm2-135m --port 11434
curl http://127.0.0.1:11434/v1/models
```

(Windows: the cache lives under `%USERPROFILE%\.cache\combs\models`;
`curl` is built into PowerShell 5+.)

Any OpenAI client works against `http://127.0.0.1:<port>/v1` — chat
completions (SSE streaming or buffered), tool calling, JSON-schema
constrained output, embeddings, plus `/v1/stats` for live engine truth
(KV sessions, per-layer arena state, timings).

### GPU requirements

The core runs on wgpu and uses whatever your OS exposes:

| OS | Backend | Needs |
|---|---|---|
| macOS | Metal | nothing — Apple Silicon and Intel both work out of the box |
| Linux | Vulkan | Vulkan drivers: `mesa-vulkan-drivers` (Debian/Ubuntu), `vulkan-radeon` / vendor NVIDIA driver as appropriate |
| Windows | DX12 / Vulkan | a current GPU driver (standard on any desktop) |

No adapter at all (a headless server without Vulkan)? GPU inference
cannot run there; kernels require a device. CPU fallback is on the
roadmap, not shipped.

## 7. Route E — the CombsLLM web platform (optional)

[CombsLLM](https://github.com/asqrzk/CombsLLM) puts a full chat UI,
live engine monitor, and model manager on top of `combs serve`. Needs
Deno 2 (§1) plus a `combs` binary from any route above.

```bash
git clone https://github.com/asqrzk/CombsLLM.git
cd CombsLLM
deno task hive     # starts the engine (if not already healthy) + platform
# → http://localhost:8787  (first run: create your passkey)
```

`COMBS_BIN` / `COMBS_MODEL` override which binary and model the hive
boots. To point at an engine you started yourself, set
`COMBS_ENGINE_URL`.

## 8. Troubleshooting

- **`combs: command not found` after npm install** — the npm global bin
  dir isn't on PATH (`npm bin -g` shows it); with nvm, re-run
  `nvm use --lts`. On Windows, reopen the terminal.
- **Rust build fails immediately on Windows with `link.exe` not
  found** — the MSVC Build Tools C++ workload is missing (§1 note).
- **Build fails with `No space left on device`** — the release build
  peaks well over 10 GB of target-dir intermediates. `cargo clean`.
- **`UI template not found`** — partial or flattened source tree; build
  from a full clone so `engine/ui/template` exists.
- **Linux: `No possible adapter available`** — Vulkan drivers missing;
  install your distro's Vulkan package for your GPU.
- **Server seems to hang on first request** — big models memory-map and
  warm up on first prefill; watch `/v1/stats` (`totals.requests`
  moving) before assuming a hang.
- **Port already in use** — another engine is listening; change
  `--port` or stop the old process.
- **macOS: "cannot be opened because the developer cannot be
  verified"** — quarantine flag on a downloaded archive; `xattr -d
  com.apple.quarantine <path>/combs`.
