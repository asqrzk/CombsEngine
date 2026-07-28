/**
 * @combs/observe — the EventBus.
 *
 * Producers publish ObsEvents; subscribers (sinks, the Control Tower relay,
 * tests) receive them in order. A bounded ring buffer retains recent events so
 * late subscribers (e.g. a just-opened tower tab) can catch up via snapshot().
 *
 * The bus is the single choke point — redaction runs here so no producer has
 * to remember to strip secrets.
 */

import type { IdPort, ClockPort, SinkPort } from "./ports.ts";
import { defaultIds, wallClock } from "./ports.ts";
import type { ObsEvent } from "./types.ts";

export type Subscriber = (event: ObsEvent) => void;

/** Redacts obvious secret material before an event is published. */
export type Redactor = (event: ObsEvent) => ObsEvent;

const SECRET_RE =
  /(api[_-]?key|authorization|bearer|password|secret|token|sk-[a-z0-9]{8,})/gi;

/** Default redactor: masks values whose key looks secret, and scrubs
 *  secret-looking strings inside shallow string fields. */
export const defaultRedactor: Redactor = (event) => {
  const scrub = (v: unknown): unknown => {
    if (typeof v === "string") return v.replace(SECRET_RE, (m) => "•".repeat(Math.min(m.length, 12)));
    if (Array.isArray(v)) return v.map(scrub);
    if (v && typeof v === "object") {
      const out: Record<string, unknown> = {};
      for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
        out[k] = SECRET_RE.test(k) ? "***" : scrub(val);
        SECRET_RE.lastIndex = 0;
      }
      return out;
    }
    return v;
  };
  return {
    ...event,
    input: event.input !== undefined ? scrub(event.input) : undefined,
    output: event.output !== undefined ? scrub(event.output) : undefined,
    context: event.context !== undefined ? scrub(event.context) : undefined,
  };
};

export interface BusOptions {
  clock?: ClockPort;
  ids?: IdPort;
  /** Ring capacity (default 2000). */
  capacity?: number;
  /** Redactor applied to every event (default: defaultRedactor; pass e => e to disable). */
  redactor?: Redactor;
  /** Sinks that receive every event alongside subscribers. */
  sinks?: SinkPort[];
}

export class EventBus {
  private readonly clock: ClockPort;
  private readonly ids: IdPort;
  private readonly capacity: number;
  private readonly redactor: Redactor;
  private readonly sinks: SinkPort[];
  private ring: ObsEvent[] = [];
  private subscribers = new Set<Subscriber>();

  constructor(opts: BusOptions = {}) {
    this.clock = opts.clock ?? wallClock;
    this.ids = opts.ids ?? defaultIds;
    this.capacity = opts.capacity ?? 2000;
    this.redactor = opts.redactor ?? defaultRedactor;
    this.sinks = opts.sinks ?? [];
  }

  /** Publishes an event (id/ts filled if absent), redacted, fanned out. */
  publish(partial: Omit<ObsEvent, "id" | "ts"> & Partial<Pick<ObsEvent, "id" | "ts">>): ObsEvent {
    let event: ObsEvent = {
      id: partial.id ?? this.ids.id(),
      ts: partial.ts ?? this.clock.now(),
      ...partial,
    } as ObsEvent;
    try {
      event = this.redactor(event);
    } catch {
      // redaction must never break publishing
    }
    this.ring.push(event);
    if (this.ring.length > this.capacity) this.ring.splice(0, this.ring.length - this.capacity);
    for (const sink of this.sinks) {
      try {
        sink.write(event);
      } catch {
        // a sink must never break the bus
      }
    }
    for (const sub of this.subscribers) {
      try {
        sub(event);
      } catch {
        // a subscriber must never break the bus
      }
    }
    return event;
  }

  /** Subscribes to live events; returns an unsubscribe function. */
  subscribe(fn: Subscriber): () => void {
    this.subscribers.add(fn);
    return () => this.subscribers.delete(fn);
  }

  /** Recent retained events (oldest → newest), optionally filtered. */
  snapshot(filter?: (e: ObsEvent) => boolean): ObsEvent[] {
    return filter ? this.ring.filter(filter) : [...this.ring];
  }

  clear(): void {
    this.ring = [];
  }

  /** Convenience publishers. */
  spanStart(source: string, name: string, extra: Partial<ObsEvent> = {}): ObsEvent {
    return this.publish({ source, kind: "span.start", name, ...extra });
  }
  spanEnd(source: string, name: string, extra: Partial<ObsEvent> = {}): ObsEvent {
    return this.publish({ source, kind: "span.end", name, ...extra });
  }
  event(source: string, name: string, extra: Partial<ObsEvent> = {}): ObsEvent {
    return this.publish({ source, kind: "event", name, ...extra });
  }
  metric(source: string, name: string, value: number, extra: Partial<ObsEvent> = {}): ObsEvent {
    return this.publish({ ...extra, source, kind: "metric", name, attrs: { value, ...extra.attrs } });
  }
  context(source: string, name: string, context: unknown, extra: Partial<ObsEvent> = {}): ObsEvent {
    return this.publish({ source, kind: "context", name, context, ...extra });
  }
  log(source: string, name: string, message: string, extra: Partial<ObsEvent> = {}): ObsEvent {
    return this.publish({ source, kind: "log", name, output: message, ...extra });
  }
}

/** A process-local default bus (optional; callers may construct their own). */
let defaultBus: EventBus | null = null;
export function getBus(opts?: BusOptions): EventBus {
  if (!defaultBus) defaultBus = new EventBus(opts);
  return defaultBus;
}
