/**
 * FfiEngine: runs the native engine in-process via Deno FFI.
 *
 * `combs_chat_completion` blocks and streams events through a callback on
 * the calling thread; it is bound `nonblocking: true` (runs on Deno's
 * blocking pool) and the callback is `Deno.UnsafeCallback.threadSafe` so
 * events safely cross into the async world as an AsyncIterable.
 */
import { cstring, deviceCaps, lastError, openLibrary, readOwnedCString } from "./ffi.ts";
import type { CombsLib } from "./ffi.ts";
import type {
  ChatRequest,
  DeviceCaps,
  EngineClient,
  EngineConfig,
  EngineMetadata,
  GenerationStats,
  StreamEvent,
} from "./types.ts";

export class FfiEngine implements EngineClient {
  readonly kind = "ffi" as const;
  private handle: Deno.PointerValue;
  private closed = false;

  private constructor(
    private lib: CombsLib,
    handle: Deno.PointerValue,
    private meta: EngineMetadata,
  ) {
    this.handle = handle;
  }

  static async load(config: EngineConfig, libraryPath?: string): Promise<FfiEngine> {
    const lib = openLibrary(libraryPath);
    const configBuf = cstring(JSON.stringify(config));
    const handle = lib.symbols.combs_engine_create(Deno.UnsafePointer.of(configBuf));
    if (!handle) {
      const err = lastError(lib);
      lib.close();
      throw new Error(`engine creation failed: ${err}`);
    }
    const metaPtr = lib.symbols.combs_engine_metadata_json(handle);
    const meta = JSON.parse(readOwnedCString(lib, metaPtr)) as EngineMetadata;
    return new FfiEngine(lib, handle, meta);
  }

  metadata(): Promise<EngineMetadata> {
    return Promise.resolve(this.meta);
  }

  async *stream(request: ChatRequest, requestId?: string): AsyncIterable<StreamEvent> {
    this.assertOpen();
    const id = requestId ?? crypto.randomUUID();
    const queue: (StreamEvent | "end")[] = [];
    let wake: (() => void) | null = null;
    let failure: unknown = null;

    // threadSafe: the native thread must wake Deno's event loop.
    const callback = Deno.UnsafeCallback.threadSafe(
      { parameters: ["pointer", "pointer"], result: "void" },
      (eventPtr: Deno.PointerValue, _userData: Deno.PointerValue) => {
        try {
          if (!eventPtr) return;
          const json = new Deno.UnsafePointerView(eventPtr).getCString();
          queue.push(JSON.parse(json) as StreamEvent);
        } catch (e) {
          failure = e;
        }
        wake?.();
      },
    );

    const cbPtr = callback.pointer;
    // Keep the request buffers alive until the native call settles.
    const reqBuf = cstring(JSON.stringify(request));
    const idBuf = cstring(id);
    const rc = this.lib.symbols.combs_chat_completion(
      this.handle,
      Deno.UnsafePointer.of(reqBuf),
      Deno.UnsafePointer.of(idBuf),
      cbPtr,
      null,
    );

    try {
      while (true) {
        while (queue.length > 0) {
          const event = queue.shift()!;
          if (event === "end") return;
          yield event;
          if (event.type === "done" || event.type === "error") {
            return;
          }
        }
        if (failure) throw failure;
        // Suspend until the callback wakes us or the FFI call settles.
        const settled = await Promise.race([
          rc.then(
            () => "settled",
            (e: unknown) => {
              failure = e;
              return "settled";
            },
          ),
          new Promise<string>((resolve) => {
            wake = () => resolve("event");
            // Also poll lightly in case a wake is missed between drains.
            setTimeout(() => resolve("tick"), 50);
          }),
        ]);
        wake = null;
        if (settled === "settled") {
          // Drain any events queued before completion, then finish.
          while (queue.length > 0) {
            const event = queue.shift()!;
            if (event === "end") return;
            yield event;
            if (event.type === "done" || event.type === "error") {
              return;
            }
          }
          return;
        }
      }
    } finally {
      callback.close();
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
    const idBuf = cstring(requestId);
    this.lib.symbols.combs_cancel(Deno.UnsafePointer.of(idBuf));
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.lib.symbols.combs_engine_destroy(this.handle);
    this.handle = null;
    this.lib.close();
  }

  private assertOpen(): void {
    if (this.closed) throw new Error("engine is closed");
  }
}

/** Static device capabilities (no engine needed). */
export function queryDeviceCaps(libraryPath?: string): DeviceCaps {
  const lib = openLibrary(libraryPath);
  try {
    return deviceCaps(lib);
  } finally {
    lib.close();
  }
}
