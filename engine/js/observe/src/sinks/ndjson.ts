/**
 * @combs/observe — NDJSON sink (one JSON event per line).
 *
 * The sink itself only SERIALIZES to lines and hands them to an injected
 * `append` function — the runtime decides where lines go (Deno.writeTextFile,
 * node fs.appendFile, console, a rolling buffer). Keeps the core file-system
 * free.
 */

import type { SinkPort } from "../ports.ts";
import type { ObsEvent } from "../types.ts";

export class NdjsonSink implements SinkPort {
  constructor(private readonly append: (line: string) => void) {}
  write(event: ObsEvent): void {
    try {
      this.append(JSON.stringify(event) + "\n");
    } catch {
      // never break the app
    }
  }
  /** Convenience: build a sink that appends to an in-memory string array. */
  static toArray(lines: string[]): NdjsonSink {
    return new NdjsonSink((l) => lines.push(l));
  }
}
