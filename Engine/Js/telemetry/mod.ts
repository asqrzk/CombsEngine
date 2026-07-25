/** @combs/telemetry — logging, tracing, metrics (flag-driven). */
export { getLogger, Logger } from "./src/logger.ts";
export type { LogLevel } from "./src/logger.ts";
export {
  ConsoleSpanExporter,
  getTracer,
  JsonlSpanExporter,
  metrics,
  Metrics,
  OtlpSpanExporter,
  Span,
  Tracer,
} from "./src/tracer.ts";
export type { Attributes, SpanEvent, SpanExporter, SpanRecord } from "./src/tracer.ts";
