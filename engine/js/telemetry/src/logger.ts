/**
 * @combs/telemetry — logging, tracing and metrics for the Combs stack.
 *
 * Everything is flag-driven (env or code):
 * - `COMBS_LOG_LEVEL` = debug | info | warn | error | off (default: info)
 * - `COMBS_LOG_COLOR` = 1 | 0 (default: auto from TTY)
 * - `COMBS_TELEMETRY` = console | jsonl | otlp:<endpoint> | off (default: off)
 * - `COMBS_TELEMETRY_FILE` = path for the jsonl exporter
 *
 * The surface mirrors OpenTelemetry's shape (tracer/span/attributes) so an
 * OTLP exporter can be swapped in without call-site changes.
 */

export type LogLevel = "debug" | "info" | "warn" | "error" | "off";

const LEVEL_ORDER: Record<LogLevel, number> = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
  off: 100,
};

const COLORS = {
  reset: "\x1b[0m",
  dim: "\x1b[2m",
  debug: "\x1b[36m", // cyan
  info: "\x1b[32m", // green
  warn: "\x1b[33m", // yellow
  error: "\x1b[31m", // red
  scope: "\x1b[35m", // magenta
} as const;

function envLevel(): LogLevel {
  const raw = (Deno.env.get("COMBS_LOG_LEVEL") ?? "info").toLowerCase();
  return (raw in LEVEL_ORDER ? raw : "info") as LogLevel;
}

function envColor(): boolean {
  const raw = Deno.env.get("COMBS_LOG_COLOR");
  if (raw === "1") return true;
  if (raw === "0") return false;
  try {
    return Deno.stdout.isTerminal();
  } catch {
    return false;
  }
}

/** Scoped, leveled, color logger. Cheap to construct — create one per module. */
export class Logger {
  constructor(
    readonly scope: string,
    readonly minLevel: LogLevel = envLevel(),
    readonly useColor: boolean = envColor(),
  ) {}

  child(scope: string): Logger {
    return new Logger(`${this.scope}:${scope}`, this.minLevel, this.useColor);
  }

  private write(level: LogLevel, msg: string, data?: Record<string, unknown>): void {
    if (LEVEL_ORDER[level] < LEVEL_ORDER[this.minLevel]) return;
    const ts = new Date().toISOString();
    const extra = data ? ` ${JSON.stringify(data)}` : "";
    if (this.useColor) {
      const c = COLORS[level as keyof typeof COLORS] ?? "";
      console.error(
        `${COLORS.dim}${ts}${COLORS.reset} ${c}${level.toUpperCase().padEnd(5)}${COLORS.reset} ` +
          `${COLORS.scope}${this.scope}${COLORS.reset} ${msg}${extra}`,
      );
    } else {
      console.error(`${ts} ${level.toUpperCase().padEnd(5)} ${this.scope} ${msg}${extra}`);
    }
  }

  debug(msg: string, data?: Record<string, unknown>): void {
    this.write("debug", msg, data);
  }
  info(msg: string, data?: Record<string, unknown>): void {
    this.write("info", msg, data);
  }
  warn(msg: string, data?: Record<string, unknown>): void {
    this.write("warn", msg, data);
  }
  error(msg: string, data?: Record<string, unknown>): void {
    this.write("error", msg, data);
  }

  /** Convenience: time an async operation and log its duration. */
  async time<T>(label: string, fn: () => Promise<T>): Promise<T> {
    const start = performance.now();
    try {
      return await fn();
    } finally {
      this.debug(`${label} done`, { ms: Math.round(performance.now() - start) });
    }
  }
}

/** Creates a logger for a scope (respects env flags). */
export function getLogger(scope: string): Logger {
  return new Logger(scope);
}
