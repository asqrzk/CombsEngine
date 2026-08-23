/**
 * @combs/core — Combs Engine application layer (L2).
 *
 * One `EngineClient` contract, three transports:
 * - `FfiEngine` — in-process native engine via Deno FFI (desktop/server)
 * - `RemoteEngine` — OpenAI-compatible HTTP/SSE (`combs serve`)
 * - `WorkerEngine` — browser Web Worker hosting the engine as WASM on
 *   WebGPU (`combs.worker.js`; build with `cargo xtask web`)
 *
 * High-level entry: {@link Combs} (presets + cache + planner + engine).
 */

export { Combs } from "./src/combs.ts";
export type { CombsInitOptions } from "./src/combs.ts";
export { FfiEngine, queryDeviceCaps } from "./src/engine.ts";
export { RemoteEngine } from "./src/remote.ts";
export { WorkerEngine } from "./src/worker.ts";
export { ModelCache } from "./src/cache.ts";
export { planEngineConfig, needsWeightSharding } from "./src/planner.ts";
export { findPreset, listPresets, PRESETS } from "./src/presets.ts";
export { loadConfigFile, mergeSampling } from "./src/config.ts";
export type {
  ChatMessage,
  ChatRequest,
  CombsConfig,
  DeviceCaps,
  EngineClient,
  EngineConfig,
  EngineMetadata,
  GenerationStats,
  ModelPreset,
  SamplingParams,
  StreamEvent,
} from "./src/types.ts";
