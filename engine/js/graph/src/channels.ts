/**
 * Channels: the state-merge model.
 *
 * Each state key maps to a channel that decides how concurrent node writes
 * fold into state. This is what makes parallel node execution deterministic:
 * writes from one superstep are collected and applied through reducers in a
 * fixed order.
 *
 * - `lastValue()`  — no reducer; exactly one writer per superstep (LangGraph
 *   semantics: two writes in the same superstep raise an error, surfacing
 *   races instead of hiding them).
 * - `append()`     — list channel; each write is appended.
 * - `binaryOp()`   — custom reducer `(current, write) => merged`.
 * - `ephemeral()`  — one-shot trigger channel (used for edge routing).
 */

export class InvalidUpdateError extends Error {
  constructor(channel: string, count: number) {
    super(
      `channel "${channel}" received ${count} writes in one superstep but has ` +
        `no reducer (lastValue). Add a reducer or restructure the graph.`,
    );
    this.name = "InvalidUpdateError";
  }
}

export interface Channel<V = unknown, W = unknown> {
  /** Folds one superstep's writes. Returns true if the value changed. */
  update(writes: W[]): boolean;
  /** Current value. */
  get(): V;
  /** Serializable snapshot. */
  checkpoint(): unknown;
  /** Restore from a snapshot. */
  fromCheckpoint(data: unknown): void;
  /** End-of-superstep hook (ephemeral channels clear themselves). */
  finish(): void;
}

class LastValueChannel<T> implements Channel<T, T> {
  private value: T | undefined;
  constructor(private readonly initial?: T) {
    this.value = initial;
  }
  update(writes: T[]): boolean {
    if (writes.length === 0) return false;
    if (writes.length > 1) throw new InvalidUpdateError("lastValue", writes.length);
    this.value = writes[0];
    return true;
  }
  get(): T {
    return this.value as T;
  }
  checkpoint(): unknown {
    return this.value;
  }
  fromCheckpoint(data: unknown): void {
    this.value = data as T;
  }
  finish(): void {}
}

class AppendChannel<T> implements Channel<T[], T[]> {
  private value: T[];
  constructor(initial: T[] = []) {
    this.value = [...initial];
  }
  /** Each write is a list of items; the channel extends with it (reducer
   * semantics: a node returning `{ log: ["a"] }` appends `"a"`). */
  update(writes: T[][]): boolean {
    if (writes.length === 0) return false;
    for (const w of writes) this.value.push(...w);
    return true;
  }
  get(): T[] {
    return this.value;
  }
  checkpoint(): unknown {
    return this.value;
  }
  fromCheckpoint(data: unknown): void {
    this.value = [...(data as T[])];
  }
  finish(): void {}
}

class BinaryOpChannel<T> implements Channel<T, T> {
  private value: T;
  constructor(
    initial: T,
    private readonly reducer: (current: T, write: T) => T,
  ) {
    this.value = initial;
  }
  update(writes: T[]): boolean {
    if (writes.length === 0) return false;
    for (const w of writes) this.value = this.reducer(this.value, w);
    return true;
  }
  get(): T {
    return this.value;
  }
  checkpoint(): unknown {
    return this.value;
  }
  fromCheckpoint(data: unknown): void {
    this.value = data as T;
  }
  finish(): void {}
}

class EphemeralChannel<T> implements Channel<T | undefined, T> {
  private value: T | undefined;
  private touched = false;
  update(writes: T[]): boolean {
    if (writes.length === 0) return false;
    this.value = writes[writes.length - 1];
    this.touched = true;
    return true;
  }
  get(): T | undefined {
    return this.value;
  }
  get seen(): boolean {
    return this.touched;
  }
  checkpoint(): unknown {
    return this.value;
  }
  fromCheckpoint(data: unknown): void {
    this.value = data as T;
  }
  finish(): void {
    this.value = undefined;
    this.touched = false;
  }
}

/** Channel factory type used in StateGraph state specs. */
export type ChannelFactory = () => Channel;

/** Last-write-wins, single-writer-per-superstep channel. */
export function lastValue<T>(initial?: T): ChannelFactory {
  return () => new LastValueChannel(initial);
}

/** List-accumulating channel. */
export function append<T>(initial: T[] = []): ChannelFactory {
  return () => new AppendChannel(initial);
}

/** Custom reducer channel. */
export function binaryOp<T>(initial: T, reducer: (current: T, write: T) => T): ChannelFactory {
  return () => new BinaryOpChannel(initial, reducer);
}

/** One-shot trigger channel (edge routing internals). */
export function ephemeral<T>(): ChannelFactory {
  return () => new EphemeralChannel<T>();
}

/** The canonical messages channel: appends chat messages. */
export interface GraphMessage {
  role: string;
  content: string;
  name?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export function messages(): ChannelFactory {
  return append<GraphMessage>([]);
}
