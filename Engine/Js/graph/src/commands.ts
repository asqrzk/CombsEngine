/**
 * Commands, control-flow primitives and the HITL interrupt sentinel.
 */

import type { GraphMessage } from "./channels.ts";

/** Pseudo-nodes. */
export const START = "__start__";
export const END = "__end__";

/** Map-style dynamic fan-out: schedule `node` with per-task input `args`. */
export class Send {
  constructor(
    public readonly node: string,
    public readonly args: unknown,
  ) {}
}

/**
 * Node return value combining a state update with routing.
 * - `update`: merged into state through the channel reducers.
 * - `goto`: next node(s) to trigger (or Send packets for fan-out).
 * - `resume`: used when resuming an interrupted graph.
 */
export interface Command {
  update?: Record<string, unknown>;
  goto?: string | string[] | Send | Send[];
  resume?: unknown;
}

/** Type guard: node results are either partial state or a Command. */
export function isCommand(value: unknown): value is Command {
  return (
    typeof value === "object" &&
    value !== null &&
    ("goto" in value || "resume" in value) &&
    !("role" in value)
  );
}

/** One pending human-in-the-loop interruption. */
export interface Interrupt {
  node: string;
  value: unknown;
}

/**
 * Thrown by `ctx.interrupt()` to pause the graph. Caught by the runner,
 * persisted into the checkpoint, and surfaced to the caller as an
 * `{ __interrupt__: [...] }` event. Resume by invoking the graph again with
 * `{ threadId, resume: value }`.
 */
export class GraphInterrupt extends Error {
  constructor(public readonly interrupts: Interrupt[]) {
    super(`graph interrupted at ${interrupts.map((i) => i.node).join(", ")}`);
    this.name = "GraphInterrupt";
  }
}

/** Event tag used in the stream when the graph pauses for humans. */
export const INTERRUPT = "__interrupt__";

/** The canonical agent state shape (messages channel). */
export interface AgentState extends Record<string, unknown> {
  messages: GraphMessage[];
}
