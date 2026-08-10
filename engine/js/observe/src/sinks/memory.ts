/**
 * @combs/observe — in-memory ring sink (default, browser-safe).
 */

import type { SinkPort } from "../ports.ts";
import type { ObsEvent } from "../types.ts";

/** Retains the last N events in memory. The Control Tower uses this in the
 *  browser; tests use it to assert on emitted events. */
export class MemorySink implements SinkPort {
  readonly events: ObsEvent[] = [];
  constructor(readonly capacity = 2000) {}
  write(event: ObsEvent): void {
    this.events.push(event);
    if (this.events.length > this.capacity) {
      this.events.splice(0, this.events.length - this.capacity);
    }
  }
  /** Most recent first. */
  latest(n = 50): ObsEvent[] {
    return this.events.slice(-n).reverse();
  }
  byTrace(traceId: string): ObsEvent[] {
    return this.events.filter((e) => e.traceId === traceId);
  }
}
