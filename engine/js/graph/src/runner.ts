/**
 * The superstep runner (Pregel model, LangGraph semantics):
 *
 *   loop {
 *     apply completed tasks' writes through the channel reducers
 *     checkpoint
 *     compute the next frontier (static edges, conditional routers, Sends,
 *     join barriers)
 *     run frontier tasks concurrently (bounded), collecting results
 *   } until the frontier is empty
 *
 * Human-in-the-loop: a node's `ctx.interrupt(v)` throws a sentinel; the
 * runner persists it in the checkpoint and halts. Resuming with
 * `{ threadId, resume }` re-executes the interrupted node from the top and
 * returns the resume value from `ctx.interrupt`.
 */

import type { Channel } from "./channels.ts";
import type { Checkpoint } from "./checkpoint.ts";
import {
  Command,
  END,
  GraphInterrupt,
  INTERRUPT,
  isCommand,
  Send,
  START,
} from "./commands.ts";
import type { CompiledGraph, NodeContext, RunConfig } from "./state.ts";

export type StreamMode = "values" | "updates" | "custom" | "interrupt" | "debug";

export type StreamEvent<S> =
  | { mode: "values"; step: number; data: S }
  | { mode: "updates"; node: string; data: unknown }
  | { mode: "custom"; node: string; data: unknown }
  | { mode: "interrupt"; data: { node: string; value: unknown }[] }
  | { mode: "debug"; data: { step: number; frontier: string[]; done: string[] } };

export interface StateSnapshot<S> {
  values: S;
  step: number;
  pendingInterrupts?: { node: string; value: unknown }[];
}

/** Thrown by ctx.interrupt; caught by the runner. */
class InterruptSentinel extends Error {
  constructor(readonly value: unknown) {
    super("interrupt");
  }
}

interface Task {
  node: string;
  sendArgs?: unknown;
}

const JOIN_RE = /^__join__(.+)__([a-zA-Z0-9_-]+)$/;

export class PregelRunner<S extends Record<string, unknown>> {
  private channels: Record<string, Channel>;
  private versions: Record<string, number> = {};
  private completed = new Set<string>();
  private resumeQueue: unknown[] = [];
  private resumeFrontier: Task[] = [];
  private step = 0;
  private pendingInterrupts: { node: string; value: unknown }[] = [];
  private restored = false;
  /** Skip the interruptBefore check once after resuming past a breakpoint. */
  private skipBeforeOnce = false;

  constructor(
    private readonly compiled: CompiledGraph<S>,
    private readonly config: RunConfig,
  ) {
    this.channels = compiled.makeChannels();
  }

  /** Runs to completion, returning the final state. */
  async run(input: Partial<S>): Promise<S> {
    let state: S | undefined;
    for await (const event of this.stream(input)) {
      if (event.mode === "values") state = event.data;
    }
    if (this.pendingInterrupts.length > 0) {
      throw new GraphInterrupt(this.pendingInterrupts);
    }
    return state ?? this.readState();
  }

  /** Streams events through the whole run. */
  async *stream(input: Partial<S>): AsyncIterable<StreamEvent<S>> {
    await this.restore();
    this.seedInput(input);

    let frontier: Task[] = this.initialFrontier();
    while (frontier.length > 0) {
      this.config.signal?.throwIfAborted();
      yield { mode: "debug", data: { step: this.step, frontier: frontier.map((t) => t.node), done: [...this.completed] } };

      // interruptBefore: pause instead of executing these nodes.
      const before = new Set(this.compiled.options.interruptBefore ?? []);
      if (!this.skipBeforeOnce && frontier.some((t) => before.has(t.node))) {
        this.pendingInterrupts = frontier
          .filter((t) => before.has(t.node))
          .map((t) => ({ node: t.node, value: { before: true } }));
        await this.checkpoint();
        yield { mode: "interrupt", data: this.pendingInterrupts };
        return;
      }
      this.skipBeforeOnce = false;

      const results = await this.executeFrontier(frontier);
      const interrupted = results.find((r) => r.interrupt);
      if (interrupted) {
        this.pendingInterrupts = [{ node: interrupted.task.node, value: interrupted.interrupt }];
        await this.checkpoint();
        yield { mode: "interrupt", data: this.pendingInterrupts };
        return;
      }
      // Drain custom events emitted by tasks in this superstep.
      for (const ev of this.customEvents.splice(0)) yield ev;

      // 1. Gather this superstep's writes grouped by channel (this is what
      // lets lastValue detect two writers in one superstep).
      const channelWrites = new Map<string, unknown[]>();
      const routings: { task: Task; result?: unknown }[] = [];
      for (const { task, result } of results) {
        this.completed.add(task.node);
        let update: Record<string, unknown> | undefined;
        if (isCommand(result)) {
          update = result.update;
        } else if (result && typeof result === "object") {
          update = result as Record<string, unknown>;
        }
        if (update) {
          for (const [key, value] of Object.entries(update)) {
            const list = channelWrites.get(key) ?? [];
            list.push(value);
            channelWrites.set(key, list);
          }
        }
        routings.push({ task, result });
        yield { mode: "updates", node: task.node, data: result };
      }
      for (const [key, writes] of channelWrites) {
        const channel = this.channels[key];
        if (!channel) throw new Error(`unknown state key "${key}"`);
        channel.update(writes);
      }

      // 2. Routing decisions see the fully merged superstep state.
      const next = new Map<string, Task[]>();
      const push = (task: Task) => {
        if (task.node === END) return;
        const list = next.get(task.node) ?? [];
        // A node triggered by multiple parents in one superstep runs ONCE
        // (explicit Sends are separate tasks and always run).
        if (task.sendArgs === undefined && list.some((t) => t.sendArgs === undefined)) {
          next.set(task.node, list);
          return;
        }
        list.push(task);
        next.set(task.node, list);
      };
      for (const { task, result } of routings) {
        const goto = isCommand(result) ? result.goto : undefined;
        if (goto !== undefined) {
          for (const t of normalizeGoto(goto)) push(t);
          continue;
        }
        const edge = this.compiled.graph.edgeSpecs.get(task.node);
        if (edge?.kind === "static") {
          for (const target of edge.targets) {
            this.resolveTarget(target, push);
          }
        } else if (edge?.kind === "conditional") {
          const route = edge.targets(this.readState());
          for (const t of normalizeGoto(route)) push(t);
        }
      }

      this.bumpVersionsAndFinish();
      await this.checkpoint();
      yield { mode: "values", step: this.step, data: this.readState() };

      // interruptAfter: pause after executing these nodes.
      const after = new Set(this.compiled.options.interruptAfter ?? []);
      const hitAfter = results.filter((r) => after.has(r.task.node));
      if (hitAfter.length > 0) {
        this.pendingInterrupts = hitAfter.map((r) => ({ node: r.task.node, value: { after: true } }));
        this.resumeFrontier = [...next.values()].flat();
        await this.checkpoint();
        yield { mode: "interrupt", data: this.pendingInterrupts };
        return;
      }

      frontier = [...next.values()].flat();
    }
  }

  // --- execution -----------------------------------------------------------

  private async executeFrontier(
    frontier: Task[],
  ): Promise<{ task: Task; result?: unknown; interrupt?: unknown }[]> {
    const max = this.config.maxConcurrency ?? 16;
    const out: { task: Task; result?: unknown; interrupt?: unknown }[] = new Array(frontier.length);
    let cursor = 0;

    const worker = async () => {
      while (cursor < frontier.length) {
        const i = cursor++;
        const task = frontier[i];
        try {
          const result = await this.runTask(task);
          out[i] = { task, result };
        } catch (err) {
          if (err instanceof InterruptSentinel) {
            out[i] = { task, interrupt: err.value };
          } else {
            throw err;
          }
        }
      }
    };
    await Promise.all(Array.from({ length: Math.min(max, frontier.length) }, worker));
    return out;
  }

  private async runTask(task: Task): Promise<unknown> {
    const spec = this.compiled.graph.nodeSpecs.get(task.node);
    if (!spec) throw new Error(`unknown node "${task.node}"`);
    const retry = spec.retry;
    const attempts = retry?.maxAttempts ?? 1;

    const ctx: NodeContext<S> = {
      config: this.config,
      writer: (chunk) => this.customEvents.push({ mode: "custom", node: task.node, data: chunk }),
      interrupt: (value) => {
        if (this.resumeQueue.length > 0) {
          return Promise.resolve(this.resumeQueue.shift());
        }
        return Promise.reject(new InterruptSentinel(value));
      },
      sendArgs: task.sendArgs,
      signal: this.config.signal,
    };

    let lastErr: unknown;
    for (let attempt = 1; attempt <= attempts; attempt++) {
      try {
        return await spec.fn(this.readState(), ctx);
      } catch (err) {
        if (err instanceof InterruptSentinel) throw err;
        lastErr = err;
        if (attempt < attempts) {
          await new Promise((r) => setTimeout(r, (retry?.backoffMs ?? 100) * attempt));
        }
      }
    }
    throw lastErr;
  }

  /** Custom events emitted between supersteps (drained by stream()). */
  private customEvents: StreamEvent<S>[] = [];

  // --- frontier computation -------------------------------------------------

  private initialFrontier(): Task[] {
    // Resume: re-run interrupted nodes (interruptBefore / ctx.interrupt),
    // or continue with the stored frontier (interruptAfter).
    if (this.restored && this.pendingInterrupts.length > 0) {
      const isAfter = this.pendingInterrupts.every(
        (i) => typeof i.value === "object" && i.value !== null && (i.value as { after?: boolean }).after,
      );
      const isBefore = this.pendingInterrupts.every(
        (i) => typeof i.value === "object" && i.value !== null && (i.value as { before?: boolean }).before,
      );
      this.pendingInterrupts = [];
      if (isAfter && this.resumeFrontier.length > 0) {
        const frontier = this.resumeFrontier;
        this.resumeFrontier = [];
        return frontier;
      }
      // Resuming past an interruptBefore breakpoint: run the held nodes
      // without re-triggering the same breakpoint.
      if (isBefore) this.skipBeforeOnce = true;
      return this.pendingInterruptsNodes ?? [];
    }
    return this.compiled.graph.entryPoints.map((node) => ({ node }));
  }

  private pendingInterruptsNodes: Task[] | null = null;

  /** Static target resolution with join barriers (`__join__a+b__c`). */
  private resolveTarget(target: string, push: (t: Task) => void): void {
    const join = JOIN_RE.exec(target);
    if (!join) {
      push({ node: target });
      return;
    }
    const [, sourcesRaw] = join;
    const sources = sourcesRaw.split("+");
    const seenKey = `__joinseen__${target}`;
    const seen = (this.joinSeen.get(seenKey) ?? new Set<string>());
    for (const done of this.completed) {
      if (sources.includes(done)) seen.add(done);
    }
    this.joinSeen.set(seenKey, seen);
    if (sources.every((s) => seen.has(s))) {
      push({ node: target.replace(JOIN_RE, "$2") });
    }
  }

  private joinSeen = new Map<string, Set<string>>();

  // --- state plumbing --------------------------------------------------------

  private seedInput(input: Partial<S>): void {
    if (this.restored) return; // resume: channels already restored
    this.applyUpdate(input as Record<string, unknown>);
    this.bumpVersionsAndFinish();
  }

  private applyUpdate(update: Record<string, unknown>): void {
    for (const [key, value] of Object.entries(update)) {
      const channel = this.channels[key];
      if (!channel) throw new Error(`unknown state key "${key}"`);
      channel.update([value]);
    }
  }

  private bumpVersionsAndFinish(): void {
    for (const [key, channel] of Object.entries(this.channels)) {
      const snap = channel.checkpoint();
      if (snap !== this.lastSnapshots[key]) {
        this.versions[key] = (this.versions[key] ?? 0) + 1;
        this.lastSnapshots[key] = snap;
      }
      channel.finish();
    }
  }

  private lastSnapshots: Record<string, unknown> = {};

  private readState(): S {
    const state: Record<string, unknown> = {};
    for (const [key, channel] of Object.entries(this.channels)) {
      state[key] = channel.get();
    }
    return state as S;
  }

  // --- checkpointing ----------------------------------------------------------

  private async restore(): Promise<void> {
    const cp = this.config.threadId
      ? await this.compiled.options.checkpointer?.get(this.config.threadId)
      : undefined;
    if (!cp) return;
    this.restored = true;
    for (const [key, value] of Object.entries(cp.channelValues)) {
      this.channels[key]?.fromCheckpoint(value);
      this.lastSnapshots[key] = this.channels[key]?.checkpoint();
    }
    this.versions = { ...cp.versions };
    this.completed = new Set(cp.completedNodes ?? []);
    this.resumeQueue = [...cp.resumeQueue];
    if (this.config.resume !== undefined) this.resumeQueue.push(this.config.resume);
    this.pendingInterrupts = cp.pendingInterrupts ?? [];
    this.pendingInterruptsNodes = this.pendingInterrupts.map((i) => ({ node: i.node }));
    this.resumeFrontier = cp.resumeFrontier ?? [];
    this.step = cp.step;
  }

  private async checkpoint(): Promise<void> {
    this.step += 1;
    const checkpointer = this.compiled.options.checkpointer;
    if (!checkpointer || !this.config.threadId) return;
    const channelValues: Record<string, unknown> = {};
    for (const [key, channel] of Object.entries(this.channels)) {
      channelValues[key] = channel.checkpoint();
    }
    const cp: Checkpoint & { completedNodes?: string[] } = {
      step: this.step,
      channelValues,
      versions: { ...this.versions },
      versionsSeen: {},
      resumeQueue: [...this.resumeQueue],
      pendingInterrupts: this.pendingInterrupts.length > 0 ? [...this.pendingInterrupts] : undefined,
      completedNodes: [...this.completed],
      resumeFrontier: this.resumeFrontier.length > 0 ? [...this.resumeFrontier] : undefined,
    };
    await checkpointer.put(this.config.threadId, cp);
  }
}

function normalizeGoto(goto: Command["goto"] | string | string[] | Send | Send[]): Task[] {
  const arr = Array.isArray(goto) ? goto : [goto];
  return (arr as (string | Send)[]).map((item) =>
    typeof item === "string" ? { node: item } : { node: item.node, sendArgs: item.args }
  );
}

export { END, INTERRUPT, Send, START };
