/**
 * Tracing + metrics: OTel-shaped spans with pluggable exporters.
 *
 * ```ts
 * const tracer = getTracer("combs.graph");
 * await tracer.span("superstep", { step: 3 }, async (span) => {
 *   span.event("node.start", { node: "planner" });
 *   // ...
 * });
 * ```
 */

export type AttributeValue = string | number | boolean | undefined;
export type Attributes = Record<string, AttributeValue>;

export interface SpanEvent {
  name: string;
  time: number;
  attributes?: Attributes;
}

export interface SpanRecord {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  name: string;
  startTime: number;
  endTime?: number;
  durationMs?: number;
  attributes: Attributes;
  events: SpanEvent[];
  status: "ok" | "error" | "unset";
  error?: string;
}

export class Span {
  readonly record: SpanRecord;
  constructor(
    record: SpanRecord,
    private readonly exporter: SpanExporter,
  ) {
    this.record = record;
  }

  get spanId(): string {
    return this.record.spanId;
  }

  setAttribute(key: string, value: AttributeValue): this {
    this.record.attributes[key] = value;
    return this;
  }

  event(name: string, attributes?: Attributes): this {
    this.record.events.push({ name, time: Date.now(), attributes });
    return this;
  }

  setError(err: unknown): this {
    this.record.status = "error";
    this.record.error = err instanceof Error ? err.message : String(err);
    return this;
  }

  end(): void {
    if (this.record.endTime) return;
    this.record.endTime = Date.now();
    this.record.durationMs = this.record.endTime - this.record.startTime;
    if (this.record.status === "unset") this.record.status = "ok";
    this.exporter.export(this.record);
  }
}

export interface SpanExporter {
  export(span: SpanRecord): void;
}

/** Dev exporter: one line per span. */
export class ConsoleSpanExporter implements SpanExporter {
  export(span: SpanRecord): void {
    const status = span.status === "error" ? "ERROR" : "ok";
    console.error(
      `[span] ${span.name} ${span.durationMs}ms ${status}` +
        (span.error ? ` (${span.error})` : ""),
    );
  }
}

/** JSONL exporter: appends one JSON span per line (Grafana/Loki friendly). */
export class JsonlSpanExporter implements SpanExporter {
  constructor(readonly path: string) {}
  export(span: SpanRecord): void {
    try {
      Deno.writeTextFileSync(this.path, JSON.stringify(span) + "\n", { append: true });
    } catch {
      // Telemetry must never break the app.
    }
  }
}

/** OTLP/HTTP exporter (traces): posts OTLP JSON to a collector. */
export class OtlpSpanExporter implements SpanExporter {
  private queue: SpanRecord[] = [];
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(readonly endpoint: string, readonly flushMs = 2000) {}

  export(span: SpanRecord): void {
    this.queue.push(span);
    if (this.timer === null) {
      this.timer = setTimeout(() => this.flush(), this.flushMs);
    }
  }

  async flush(): Promise<void> {
    this.timer = null;
    const batch = this.queue.splice(0);
    if (batch.length === 0) return;
    const body = {
      resourceSpans: [{
        resource: { attributes: [{ key: "service.name", value: { stringValue: "combs" } }] },
        scopeSpans: [{
          scope: { name: "@combs/telemetry" },
          spans: batch.map((s) => ({
            traceId: s.traceId.replaceAll("-", ""),
            spanId: s.spanId.replaceAll("-", "").slice(0, 16),
            name: s.name,
            startTimeUnixNano: String(s.startTime * 1e6),
            endTimeUnixNano: String((s.endTime ?? s.startTime) * 1e6),
            attributes: Object.entries(s.attributes)
              .filter(([, v]) => v !== undefined)
              .map(([key, v]) => ({ key, value: { stringValue: String(v) } })),
            status: { code: s.status === "error" ? 2 : 1 },
          })),
        }],
      }],
    };
    try {
      await fetch(`${this.endpoint}/v1/traces`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch {
      // Telemetry must never break the app.
    }
  }
}

export class Tracer {
  constructor(
    readonly scope: string,
    private readonly exporter: SpanExporter,
  ) {}

  /** Starts a span and returns it (manual `.end()`). */
  startSpan(name: string, attributes: Attributes = {}, parent?: Span): Span {
    return new Span(
      {
        traceId: parent?.record.traceId ?? crypto.randomUUID(),
        spanId: crypto.randomUUID().slice(0, 16),
        parentSpanId: parent?.spanId,
        name: `${this.scope}.${name}`,
        startTime: Date.now(),
        attributes,
        events: [],
        status: "unset",
      },
      this.exporter,
    );
  }

  /** Runs `fn` inside a span, ending it automatically (error captured). */
  async span<T>(
    name: string,
    attributes: Attributes,
    fn: (span: Span) => Promise<T> | T,
  ): Promise<T> {
    const span = this.startSpan(name, attributes);
    try {
      const result = await fn(span);
      span.end();
      return result;
    } catch (err) {
      span.setError(err);
      span.end();
      throw err;
    }
  }
}

/** No-op exporter when telemetry is off. */
class NullExporter implements SpanExporter {
  export(): void {}
}

let defaultTracer: Tracer | null = null;

/** The process-wide tracer, configured by COMBS_TELEMETRY. */
export function getTracer(scope: string): Tracer {
  if (!defaultTracer) {
    defaultTracer = new Tracer(scope, exporterFromEnv());
  }
  return new Tracer(scope, defaultTracer["exporter"]);
}

function exporterFromEnv(): SpanExporter {
  const mode = Deno.env.get("COMBS_TELEMETRY") ?? "off";
  if (mode === "console") return new ConsoleSpanExporter();
  if (mode === "jsonl") {
    return new JsonlSpanExporter(
      Deno.env.get("COMBS_TELEMETRY_FILE") ?? "combs-telemetry.jsonl",
    );
  }
  if (mode.startsWith("otlp:")) return new OtlpSpanExporter(mode.slice(5));
  return new NullExporter();
}

/** Simple counter/gauge registry (printed via logger or exported by spans). */
export class Metrics {
  private counters = new Map<string, number>();
  private gauges = new Map<string, number>();

  inc(name: string, by = 1): void {
    this.counters.set(name, (this.counters.get(name) ?? 0) + by);
  }
  gauge(name: string, value: number): void {
    this.gauges.set(name, value);
  }
  snapshot(): Record<string, number> {
    return { ...Object.fromEntries(this.counters), ...Object.fromEntries(this.gauges) };
  }
}

export const metrics = new Metrics();
