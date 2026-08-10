/**
 * @combs/observe — realtime observability bus + instrument middleware.
 *
 * Isomorphic core: no Deno.*, no node:*, no window. Runtime specifics are
 * injected via ports (clock / ids / sink append+broadcast). Producers publish
 * ObsEvents to an EventBus; sinks and the Control Tower consume them.
 *
 * ```ts
 * import { EventBus, wrapEngine, instrumentFetch, span, MemorySink } from "@combs/observe";
 * const bus = new EventBus({ sinks: [new MemorySink()] });
 * const engine = wrapEngine(realEngine, bus, "engine:A");
 * ```
 */
export type { ObsEvent, ObsKind, Severity, SourceDescriptor } from "./src/types.ts";
export type { ClockPort, IdPort, SinkPort } from "./src/ports.ts";
export { defaultIds, wallClock } from "./src/ports.ts";
export { EventBus, getBus, defaultRedactor } from "./src/bus.ts";
export type { BusOptions, Redactor, Subscriber } from "./src/bus.ts";
export { instrumentFetch, span, wrapEngine } from "./src/instrument.ts";
export type { EngineLike, FetchLike } from "./src/instrument.ts";
export { MemorySink } from "./src/sinks/memory.ts";
export { NdjsonSink } from "./src/sinks/ndjson.ts";
export { WebSocketSink } from "./src/sinks/websocket.ts";
