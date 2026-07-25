/**
 * WorkerEngine: browser Web Worker transport (EXPERIMENTAL).
 *
 * web-llm pattern: the engine (combs-wasm module) lives in a Web Worker;
 * the main thread talks to it with a `{kind, id, payload}` envelope —
 * pull-based streaming, one pending RPC per id. The worker script
 * (`combs.worker.js`) hosts the wasm module and implements the same
 * event shapes as the FFI engine, so app code is transport-agnostic.
 *
 * Status: the protocol side is complete; the worker-side wasm engine is
 * the Phase 5+ follow-up (see Engine/Core/combs-wasm).
 */

import type {
  ChatRequest,
  EngineClient,
  EngineMetadata,
  GenerationStats,
  StreamEvent,
} from "./types.ts";

interface WorkerRequest {
  kind: "load" | "chat" | "cancel" | "metadata" | "next";
  id: string;
  payload?: unknown;
}

interface WorkerReply {
  kind: "ready" | "metadata" | "delta" | "done" | "error" | "event";
  id: string;
  payload?: unknown;
}

export class WorkerEngine implements EngineClient {
  readonly kind = "worker" as const;
  private pending = new Map<string, {
    resolve: (v: unknown) => void;
    reject: (e: Error) => void;
    events: StreamEvent[];
    waiting?: () => void;
  }>();

  private constructor(private worker: Worker) {
    worker.onmessage = (ev: MessageEvent<WorkerReply>) => {
      const msg = ev.data;
      const slot = this.pending.get(msg.id);
      if (!slot) return;
      switch (msg.kind) {
        case "event":
          slot.events.push(msg.payload as StreamEvent);
          slot.waiting?.();
          break;
        case "done":
          this.pending.delete(msg.id);
          slot.resolve(msg.payload);
          break;
        case "error":
          this.pending.delete(msg.id);
          slot.reject(new Error(String(msg.payload)));
          break;
        case "ready":
        case "metadata":
          this.pending.delete(msg.id);
          slot.resolve(msg.payload);
          break;
      }
    };
  }

  /** Loads the wasm engine in the worker. */
  static async load(workerUrl: string | URL, config: Record<string, unknown>): Promise<WorkerEngine> {
    const engine = new WorkerEngine(new Worker(workerUrl, { type: "module" }));
    const id = crypto.randomUUID();
    await engine.rpc({ kind: "load", id, payload: config });
    return engine;
  }

  private rpc(req: WorkerRequest): Promise<unknown> {
    return new Promise((resolve, reject) => {
      this.pending.set(req.id, { resolve, reject, events: [] });
      this.worker.postMessage(req);
    });
  }

  async metadata(): Promise<EngineMetadata> {
    const id = crypto.randomUUID();
    return (await this.rpc({ kind: "metadata", id })) as EngineMetadata;
  }

  async *stream(request: ChatRequest, requestId?: string): AsyncIterable<StreamEvent> {
    const id = requestId ?? crypto.randomUUID();
    const queue: StreamEvent[] = [];
    let wakeFn: (() => void) | null = null;
    let failure: Error | null = null;

    const slot = {
      resolve: (_v: unknown) => wakeFn?.(),
      reject: (e: Error) => {
        failure = e;
        wakeFn?.();
      },
      events: queue,
      waiting: undefined as (() => void) | undefined,
    };
    Object.defineProperty(slot, "waiting", {
      get: () => wakeFn ?? undefined,
      set: (fn) => {
        wakeFn = fn as (() => void) | null;
      },
    });
    this.pending.set(id, slot);
    this.worker.postMessage({ kind: "chat", id, payload: request } satisfies WorkerRequest);

    try {
      while (true) {
        while (queue.length > 0) {
          const event = queue.shift()!;
          yield event;
          if (event.type === "done" || event.type === "error") return;
        }
        if (failure) throw failure;
        await new Promise<void>((resolve) => {
          wakeFn = resolve;
          setTimeout(resolve, 50);
        });
        wakeFn = null;
        if (!this.pending.has(id)) {
          while (queue.length > 0) {
            const event = queue.shift()!;
            yield event;
            if (event.type === "done" || event.type === "error") return;
          }
          return;
        }
      }
    } finally {
      this.pending.delete(id);
    }
  }

  async complete(
    request: ChatRequest,
  ): Promise<{ text: string; stats: GenerationStats; finishReason: string }> {
    let text = "";
    for await (const event of this.stream(request)) {
      if (event.type === "delta") text += event.text;
      if (event.type === "done") {
        return { text, stats: event.stats, finishReason: event.finish_reason };
      }
      if (event.type === "error") throw new Error(event.message);
    }
    throw new Error("stream ended without a done event");
  }

  cancel(requestId: string): void {
    this.worker.postMessage({ kind: "cancel", id: requestId } satisfies WorkerRequest);
  }

  close(): void {
    this.worker.terminate();
  }
}
