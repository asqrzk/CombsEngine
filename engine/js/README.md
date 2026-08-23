# Combs Engine — L2 Application Layer (Deno/TypeScript)

Eight packages forming the user-facing layer over the native engine:

| Package | Role |
|---|---|
| `@combs/core` | Engine client (FFI/remote transports), presets, ModelCache, DevicePlanner, layered config |
| `@combs/graph` | LangGraph-equivalent graph engine: channels/reducers, StateGraph, superstep runner (concurrency, retry, abort), checkpointers (Memory/Deno KV/SQLite), HITL interrupt+resume, Command/Send, stream modes, time travel |
| `@combs/agents` | Tools + ToolRegistry, ToolNode, createReactAgent, structured output (schema→prompt→validate→retry), memory stores (KV/SQLite), MCP client (stdio/WS), skills loader |
| `@combs/runtime` | Parallelism infra: findFreePort, secure tokens, KeyedMutex/Semaphore, KV task queue, SQLite SessionStore, AgentServer (HTTP+WS per agent with auth), Orchestrator (delegation hub), AgentPool (subprocess spawning) |
| `@combs/flows` | High-level factories: `createWorkflow` (loops/checks), `createRoleplayChat` (multi-agent turn-taking), `addMemory`/`withMemory` |
| `@combs/telemetry` | Color scoped logging, OTel-shaped spans (console/JSONL/OTLP exporters), metrics — all flag-driven (`COMBS_LOG_LEVEL`, `COMBS_TELEMETRY`) |
| `@combs/observe` | Realtime observability bus (isomorphic, zero runtime deps): EventBus, instrument middleware (`wrapEngine`/`instrumentFetch`/`span`), sinks (memory/NDJSON/WebSocket), secret redaction — powers the Control Tower |

## Quick start

```sh
# Native library (from Engine/Core): cargo xtask build --release
deno task test                                   # all 33 tests
deno run --allow-ffi --allow-read --allow-env --allow-net --unstable-kv examples/chat.ts
deno run --allow-ffi --allow-read --allow-env --allow-net --unstable-kv examples/agent_demo.ts
deno run --allow-ffi --allow-read --allow-write --allow-env --allow-net examples/orchestration_demo.ts
```

```ts
import { Combs } from "@combs/core";
import { createReactAgent, tool } from "@combs/agents";

const engine = await Combs.init("smollm2-135m");   // preset → cache → planner → engine
const agent = createReactAgent({ engine, tools: [myTool], systemPrompt: "..." });
const out = await agent.invoke({ messages: [{ role: "user", content: "..." }] },
                               { threadId: "conv-1" });   // checkpointed, resumable
```

## Orchestration (agents on ports)

```ts
import { createAgentServer, Orchestrator } from "@combs/runtime";

// Each server: free port found, auth token minted, HTTP + WS endpoints.
const a = await createAgentServer({ name: "poet", handler });
const orch = new Orchestrator();
await orch.register({ name: "poet", url: a.url, token: a.token });
const result = await orch.delegate("poet", { text: "..." });   // WS, serialized per agent
```

`AgentPool` spawns the same servers as isolated subprocesses (own engine
instance per process; model written to the shared model store first).

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

Env: `COMBS_LIB`, `COMBS_HOME`, `COMBS_CONFIG`, `COMBS_KV`, `COMBS_ATTN`,
`COMBS_LOG_LEVEL`, `COMBS_TELEMETRY`, `COMBS_TELEMETRY_FILE`.

Note: Deno KV APIs (checkpoints, memory, queues) require `--unstable-kv`.


## Cross-platform

The same code runs anywhere Deno runs; the native library is resolved per
platform from `Engine/Core/dist/<platform>/` (see `cargo xtask matrix`):
macOS arm64/x86_64, iOS, Android (Linux/Windows check-only until a linker is
available).

The browser is the fourth transport, behind the same `EngineClient`
contract: `WorkerEngine` talks to `combs.worker.js`, which hosts the engine
compiled to WebAssembly on WebGPU. Build it with `cargo xtask web` — the
module and its glue are generated, never committed.
