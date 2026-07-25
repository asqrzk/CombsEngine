/**
 * The graph builder: nodes, edges, conditional routing, compilation.
 */

import type { Channel, ChannelFactory } from "./channels.ts";
import type { Checkpointer } from "./checkpoint.ts";
import { END, Send, START } from "./commands.ts";
import { PregelRunner } from "./runner.ts";

/** What a node receives beyond state. */
export interface NodeContext<S> {
  /** Run configuration (threadId, signal, user metadata). */
  config: RunConfig;
  /** Emit a custom stream event (consumed with streamMode "custom"). */
  writer: (chunk: unknown) => void;
  /**
   * Human-in-the-loop: pauses the graph (persisted in the checkpoint).
   * On resume (`{ threadId, resume }` in the next invoke), this returns the
   * resume value. Deterministic: nodes re-execute from the top on resume.
   */
  interrupt: (value: unknown) => Promise<unknown>;
  /** Input attached to a `Send` packet (dynamic fan-out), if any. */
  sendArgs?: unknown;
  /** Abort signal for the whole run. */
  signal?: AbortSignal;
}

/** User run configuration. */
export interface RunConfig {
  threadId?: string;
  /** Resume value for an interrupted graph. */
  resume?: unknown;
  maxConcurrency?: number;
  signal?: AbortSignal;
  /** Free-form metadata (traced, passed through to nodes). */
  metadata?: Record<string, unknown>;
}

/** A graph node: receives state, returns a partial update or a Command. */
export type NodeFn<S> = (
  state: S,
  ctx: NodeContext<S>,
) => Promise<Partial<S> | import("./commands.ts").Command | void> | Partial<S> | import("./commands.ts").Command | void;

export interface RetryPolicy {
  maxAttempts: number;
  backoffMs?: number;
}

export interface NodeSpec<S> {
  fn: NodeFn<S>;
  retry?: RetryPolicy;
}

type Router<S> = (state: S) => string | string[] | Send | Send[];

type EdgeSpec<S> =
  | { kind: "static"; targets: string[] }
  | { kind: "conditional"; targets: Router<S> };

export interface CompileOptions {
  checkpointer?: Checkpointer;
  interruptBefore?: string[];
  interruptAfter?: string[];
}

/**
 * The declarative builder. `S` is the state shape; each key's channel
 * factory decides how parallel writes merge.
 */
export class StateGraph<S extends Record<string, unknown>> {
  private nodes = new Map<string, NodeSpec<S>>();
  private edges = new Map<string, EdgeSpec<S>>();
  private startEdges: string[] = [];

  constructor(readonly channels: { [K in keyof S]: ChannelFactory }) {}

  addNode(name: string, fn: NodeFn<S>, opts: { retry?: RetryPolicy } = {}): this {
    if (name === START || name === END) throw new Error(`reserved node name: ${name}`);
    this.nodes.set(name, { fn, retry: opts.retry });
    return this;
  }

  /** `addEdge(a, b)` or `addEdge([a, b], c)` (c waits for ALL sources). */
  addEdge(from: string | string[], to: string): this {
    if (Array.isArray(from)) {
      for (const f of from) this.addEdge(f, `${"__join__"}${from.join("+")}__${to}`);
      this.edges.set(`${"__join__"}${from.join("+")}__${to}`, {
        kind: "static",
        targets: [to],
      });
      return this;
    }
    if (from === START) {
      this.startEdges.push(to);
      return this;
    }
    this.edges.set(from, { kind: "static", targets: [to] });
    return this;
  }

  /** Routes after `from` based on state; return node name(s), END, or Send. */
  addConditionalEdges(from: string, router: Router<S>, mapping?: Record<string, string>): this {
    if (mapping) {
      const inner = router;
      router = (state) => {
        const route = inner(state);
        const mapOne = (r: string) => mapping[r] ?? r;
        if (typeof route === "string") return mapOne(route);
        if (Array.isArray(route)) {
          return route.map((r) => (typeof r === "string" ? mapOne(r) : r)) as string[] | Send[];
        }
        return route;
      };
    }
    this.edges.set(from, { kind: "conditional", targets: router });
    return this;
  }

  /** Validates and freezes the graph into an executable CompiledGraph. */
  compile(opts: CompileOptions = {}): CompiledGraph<S> {
    for (const target of this.startEdges) this.assertNode(target);
    for (const [from, edge] of this.edges) {
      if (!from.startsWith("__join__")) this.assertNode(from);
      if (edge.kind === "static") {
        for (const t of edge.targets) {
          if (t.startsWith("__join__")) continue;
          this.assertNode(t);
        }
      }
    }
    for (const name of this.nodes.keys()) {
      const hasInbound = this.startEdges.includes(name) ||
        [...this.edges.values()].some(
          (e) => e.kind === "static" && e.targets.includes(name),
        );
      // Conditional targets can't be validated statically — warn only.
      if (!hasInbound && this.nodes.size > 1) {
        console.warn(`[combs:graph] node "${name}" has no static inbound edges (ok if reached via conditional edges)`);
      }
    }
    return new CompiledGraph(this, opts);
  }

  private assertNode(name: string): void {
    if (name !== END && !this.nodes.has(name)) {
      throw new Error(`edge references unknown node "${name}"`);
    }
  }

  /** Internal accessors for the runner. */
  get nodeSpecs(): ReadonlyMap<string, NodeSpec<S>> {
    return this.nodes;
  }
  get edgeSpecs(): ReadonlyMap<string, EdgeSpec<S>> {
    return this.edges;
  }
  get entryPoints(): readonly string[] {
    return this.startEdges;
  }
}

/** A compiled, executable graph. */
export class CompiledGraph<S extends Record<string, unknown>> {
  constructor(
    readonly graph: StateGraph<S>,
    readonly options: CompileOptions,
  ) {}

  /** Fresh channel instances for a run. */
  makeChannels(): Record<string, Channel> {
    const out: Record<string, Channel> = {};
    for (const [key, factory] of Object.entries(this.graph.channels)) {
      out[key] = factory();
    }
    return out;
  }

  /** Runs the graph to completion and returns the final state. */
  async invoke(input: Partial<S> = {}, config: RunConfig = {}): Promise<S> {
    const runner = new PregelRunner(this, config);
    return await runner.run(input);
  }

  /** Streams graph events (see StreamMode in runner.ts). */
  stream(input: Partial<S> = {}, config: RunConfig = {}): AsyncIterable<import("./runner.ts").StreamEvent<S>> {
    const runner = new PregelRunner(this, config);
    return runner.stream(input);
  }

  /** Current state snapshot of a thread (requires a checkpointer). */
  async getState(threadId: string): Promise<import("./runner.ts").StateSnapshot<S> | undefined> {
    const cp = await this.options.checkpointer?.get(threadId);
    if (!cp) return undefined;
    const channels = this.makeChannels();
    for (const [k, v] of Object.entries(cp.channelValues)) channels[k]?.fromCheckpoint(v);
    const state = {} as S;
    for (const key of Object.keys(channels)) {
      (state as Record<string, unknown>)[key] = channels[key].get();
    }
    return { values: state, step: cp.step, pendingInterrupts: cp.pendingInterrupts };
  }

  /** Lists checkpoints of a thread, newest first (time travel). */
  async getStateHistory(threadId: string): Promise<import("./checkpoint.ts").Checkpoint[]> {
    return (await this.options.checkpointer?.list(threadId)) ?? [];
  }

  /** Applies values to a thread's state as if a node produced them (human
   * editing / fork-from-checkpoint), creating a new checkpoint. */
  async updateState(threadId: string, values: Partial<S>): Promise<void> {
    const cp = await this.options.checkpointer?.get(threadId);
    if (!cp) throw new Error("updateState requires a checkpointer and an existing thread");
    const channels = this.makeChannels();
    for (const [k, v] of Object.entries(cp.channelValues)) channels[k]?.fromCheckpoint(v);
    const byChannel = new Map<string, unknown[]>();
    for (const [k, v] of Object.entries(values)) {
      byChannel.set(k, [v]);
    }
    for (const [k, writes] of byChannel) {
      if (channels[k]?.update(writes)) cp.versions[k] = (cp.versions[k] ?? 0) + 1;
      cp.channelValues[k] = channels[k].checkpoint();
    }
    cp.step += 1;
    await this.options.checkpointer!.put(threadId, cp);
  }
}
