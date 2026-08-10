/**
 * @combs/observe — runtime ports.
 *
 * The core has ZERO runtime dependencies: it never touches Deno.*, node:*, or
 * window. Anything that differs per runtime (time, id generation, where events
 * are persisted) is injected through one of these ports. A runtime adapter
 * (Deno / Node / browser) supplies concrete implementations at composition time.
 */

/** Supplies monotonic-ish millisecond timestamps. */
export interface ClockPort {
  now(): number;
}

/** Generates unique ids (events, spans, traces). */
export interface IdPort {
  id(): string;
}

/** Terminal destination for events leaving the process. */
export interface SinkPort {
  /** Receives one event. Must never throw — observability must not break the app. */
  write(event: import("./types.ts").ObsEvent): void;
  /** Optional flush for buffered sinks. */
  flush?(): void | Promise<void>;
}

// ---------------------------------------------------------------------------
// Defaults (portable — no runtime globals beyond the ECMAScript standard).
// ---------------------------------------------------------------------------

/** Wall-clock milliseconds. Monotonic enough for ordering within a process. */
export const wallClock: ClockPort = { now: () => Date.now() };

let counter = 0;
/** RFC4122-ish unique id without crypto.getRandomValues dependency: time +
 *  counter + random. Sufficient for local correlation ids. */
export const defaultIds: IdPort = {
  id: () => {
    counter = (counter + 1) & 0xffffff;
    const rand = Math.floor(Math.random() * 0xffffffff);
    return `${Date.now().toString(36)}-${counter.toString(36)}-${rand.toString(36)}`;
  },
};
