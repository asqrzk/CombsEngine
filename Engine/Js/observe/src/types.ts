/**
 * @combs/observe — shared event types (runtime-agnostic).
 *
 * These types are the contract between producers (proxy, engines, agents,
 * orchestrator, UI) and consumers (sinks, the Control Tower). They contain no
 * runtime-specific shapes — everything is plain JSON-serializable data.
 */

/** Severity for log-style events. */
export type Severity = "debug" | "info" | "warn" | "error";

/** The category of an observed event. */
export type ObsKind =
  | "span.start" // a unit of work began (engine.stream, proxy.relay, agent.delegate)
  | "span.end" // ... and finished (carries output/status/duration)
  | "event" // a point-in-time fact (permission decided, engine spawned)
  | "metric" // a numeric sample (bytes, tokens/s, KV pages)
  | "log" // a human message
  | "context"; // a snapshot of working context (messages, transcript, KV state)

/** A single observed event. Producers build these; the bus routes them. */
export interface ObsEvent {
  /** Unique event id (from the injected IdPort). */
  id: string;
  /** Milliseconds from the injected ClockPort (monotonic where available). */
  ts: number;
  /** Origin, e.g. "proxy", "engine:A", "engine:B", "agent:coder", "orchestrator", "ui". */
  source: string;
  kind: ObsKind;
  /** Dot-named operation, e.g. "engine.stream", "proxy.relay", "agent.delegate". */
  name: string;
  /** Correlation: all spans in one run share a traceId. */
  traceId?: string;
  spanId?: string;
  parentSpanId?: string;
  /** Scalar annotations (status codes, counts, model, port, ...). */
  attrs?: Record<string, string | number | boolean | undefined>;
  /** Redacted request payload / prompt / arguments. */
  input?: unknown;
  /** Result / generated text / generation stats. */
  output?: unknown;
  /** Working context: the messages sent, the transcript window, KV usage. */
  context?: unknown;
  status?: "ok" | "error";
  error?: string;
  severity?: Severity;
}

/** A component that can be observed (an engine, an agent, the proxy, a window). */
export interface SourceDescriptor {
  /** Stable source id used in ObsEvent.source. */
  id: string;
  /** What it is: "proxy" | "engine" | "agent" | "orchestrator" | "window". */
  type: string;
  /** Human label (model name, agent name, ...). */
  label?: string;
  /** Free-form state (port, pid, ready, ...). */
  state?: Record<string, string | number | boolean | undefined>;
}
