# Combs Engine — L2 Application Layer (Deno/TypeScript)

`@combs/core` is the user-facing layer over the native engine: one
`EngineClient` contract, multiple transports, model presets, a device
planner, and a local model cache. Everything is configurable, and every
default can be overridden at four levels: built-in defaults → preset →
`combs.config.json` → per-call options.

## Quick start

```sh
# Native library (from Engine/Core): cargo xtask build --release
deno task test                                   # unit + FFI integration tests
deno run --allow-ffi --allow-read --allow-env --allow-net examples/chat.ts
```

```ts
import { Combs } from "@combs/core";

// Preset id → download (if needed) → device plan → load.
const engine = await Combs.init("smollm2-135m");
for await (const ev of engine.stream({ messages: [{ role: "user", content: "Hi" }] })) {
  if (ev.type === "delta") console.log(ev.text);
}
engine.close();

// Or a local model directory, with overrides:
const e2 = await Combs.init({
  model: "/path/to/model",
  engine: { max_seq_len: 4096, kv_cache: "paged" },
});

// Or a remote `combs serve` instance (same contract):
const e3 = Combs.remote("http://localhost:8080");
```

## Architecture

| Module | Role |
|---|---|
| `types.ts` | `EngineClient` contract + all shared types |
| `ffi.ts` | `Deno.dlopen` bindings (platform-aware lib resolution) |
| `engine.ts` | `FfiEngine` — native in-process engine (`nonblocking` + `UnsafeCallback.threadSafe` streaming) |
| `remote.ts` | `RemoteEngine` — OpenAI-compatible HTTP/SSE client |
| `cache.ts` | `ModelCache` — streaming downloads into `~/.cache/combs/models` |
| `presets.ts` | `ModelPreset` registry (smollm2 family; llama-arch only for now) |
| `planner.ts` | `DevicePlanner` — device caps → `EngineConfig` (KV budget, prefill chunking, mobile detection) |
| `config.ts` | layered config (defaults → preset → `combs.config.json` → per-call) |
| `combs.ts` | `Combs.init()` high-level orchestration |

## Configuration

`combs.config.json` (cwd, `$COMBS_CONFIG`, or `~/.config/combs/config.json`):

```json
{
  "modelStore": "/data/models",
  "libraryPath": "/opt/combs/lib/libcombs_ffi.dylib",
  "engine": { "max_seq_len": 4096, "prefill_chunk_size": 256 },
  "sampling": { "temperature": 0.7, "top_p": 0.9 }
}
```

Env vars: `COMBS_LIB` (native library path), `COMBS_HOME` (store root),
`COMBS_CONFIG` (config file), `COMBS_KV`/`COMBS_ATTN` (core engine knobs).

## Cross-platform

The same code runs anywhere Deno runs; the native library is resolved per
platform from `Engine/Core/dist/<platform>/` (see `cargo xtask matrix`):
macOS arm64/x86_64, iOS, Android (Linux/Windows check-only until a linker is
available). The browser transport (Web Worker + WASM/WebGPU) lands in Phase 5
behind the same `EngineClient` contract.
