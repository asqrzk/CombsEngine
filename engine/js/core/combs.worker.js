/**
 * combs.worker.js — the engine's home in a browser.
 *
 * `WorkerEngine` (src/worker.ts) speaks a `{kind, id, payload}` envelope to
 * this script; this script speaks wasm-bindgen to the engine. Everything
 * heavy — the model bytes, the weights, the KV cache, decoding — lives on
 * this side of the postMessage boundary, so the page's own thread stays
 * free to render the tokens it is being handed.
 *
 * The event payloads are the engine's own JSON shapes, identical to the
 * ones the native FFI emits. A client written against `EngineClient` cannot
 * tell which transport it got, which is the point.
 *
 * Serve this file, ./pkg/combs_wasm.js and ./pkg/combs_wasm_bg.wasm from
 * the same directory. Generate pkg/ with `cargo xtask web`.
 */

import init, {
  combs_cancel,
  combs_chat_completion,
  combs_device_caps,
  combs_engine_create,
  combs_engine_destroy,
  combs_engine_metadata,
  combs_engine_stats,
  combs_model_abort,
  combs_model_append,
  combs_model_finish,
  combs_model_open,
} from "./pkg/combs_wasm.js";

/** The one engine this worker hosts. One worker, one model, one flight. */
let engineId = null;
let ready = null;
/** The module's linear memory, captured at init — progress and stats
 * report its true byteLength so mount headroom is a measured fact. */
let wasmMemory = null;

/**
 * What WebGPU swallows, surfaced. A shader that fails validation in a
 * browser does not error the call that dispatches it — the pipeline is
 * invalid, every dispatch through it is a silent no-op, and the output
 * buffer stays zero-initialized. From the outside that is a model that
 * "runs" and emits garbage. These interceptors catch the compile errors
 * and device errors as they happen, log them loudly, and expose them on
 * the `stats` reply so a page can tell a broken kernel from a bad model.
 */
const gpuErrors = [];
const noteGpuError = (kind, detail) => {
  const entry = { kind, detail: String(detail).slice(0, 500), at: Date.now() };
  gpuErrors.push(entry);
  if (gpuErrors.length > 20) gpuErrors.shift();
  console.error(`[combs.worker] ${kind}: ${entry.detail}`);
};
if (globalThis.GPUDevice) {
  const origCSM = GPUDevice.prototype.createShaderModule;
  GPUDevice.prototype.createShaderModule = function (desc) {
    const mod = origCSM.call(this, desc);
    mod.getCompilationInfo?.().then((info) => {
      const errs = info.messages.filter((m) => m.type === 'error');
      if (errs.length) {
        noteGpuError('shader-compile-error', `${desc.label ?? '(unlabeled)'}: ${errs[0].message}`);
      }
    });
    return mod;
  };
}
if (globalThis.GPUAdapter) {
  const origRD = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (...args) {
    const device = await origRD.apply(this, args);
    device.addEventListener?.('uncapturederror', (ev) => {
      noteGpuError('uncaptured-error', ev.error?.message ?? ev.error);
    });
    return device;
  };
}

const post = (kind, id, payload) => self.postMessage({ kind, id, payload });
const fail = (id, error) => post("error", id, String(error?.message ?? error));

/**
 * Loads the wasm module and creates the engine.
 *
 * The model can arrive two ways: `modelBytes` (an ArrayBuffer the page
 * already has — transfer it, do not copy a hundred megabytes) or
 * `modelUrl` (fetched here, which also keeps the download off the page's
 * thread). Everything else in the payload is the engine config.
 */
async function load(id, payload = {}) {
  const {
    modelUrl,
    modelBytes,
    modelBlob,
    cacheName,
    cacheKey,
    expectedLen,
    wasmUrl,
    ...config
  } = payload;
  const exports = await init(wasmUrl ? { module_or_path: wasmUrl } : undefined);
  wasmMemory = exports?.memory ?? wasmMemory;

  // A worker hosts one engine. Creating a second without freeing the
  // first would leave hundreds of megabytes of weights and KV arenas
  // reachable only by an id nobody holds any more — the engine table is
  // keyed by id, so an overwritten id is a leak, not a replacement.
  // Freed BEFORE the mount reserves its buffer, so both never coexist.
  const previous = engineId;
  engineId = null;
  if (previous !== null) combs_engine_destroy(previous);

  let created;
  if (modelBytes) {
    // The one-shot path: bytes the page already holds (small models,
    // parity fixtures). Kept verbatim — it is also the reference the
    // streamed path must match byte-for-byte.
    created = await combs_engine_create(
      JSON.stringify(config),
      new Uint8Array(modelBytes),
    );
  } else {
    created = await mountStreamed(id, config, {
      modelBlob,
      cacheName,
      cacheKey,
      modelUrl,
      expectedLen,
    });
  }
  engineId = created;
  post("ready", id, JSON.parse(combs_engine_metadata(created)));
}

/**
 * The big-model path: never materialize the file as one ArrayBuffer
 * (Chrome caps page/worker-heap ArrayBuffers around 2.1 GB — the wall
 * this exists to pass). The bytes stream straight from their source
 * into wasm linear memory through the chunk-append mount; the wasm
 * side owns the single full-size buffer under its 4 GiB ceiling.
 */
async function mountStreamed(id, config, source) {
  const { stream, total } = await openModelStream(source);
  const expected = Number(source.expectedLen ?? total ?? 0);
  if (!Number.isFinite(expected) || expected <= 0) {
    throw new Error("mount needs `expectedLen` (or a source that knows its size)");
  }
  const handle = await combs_model_open(JSON.stringify(config), expected, "buffer");
  try {
    const reader = stream.getReader();
    let loaded = 0;
    let lastProgress = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      combs_model_append(handle, value);
      loaded += value.byteLength;
      if (loaded - lastProgress >= 32 * 1024 * 1024) {
        lastProgress = loaded;
        post("progress", id, {
          loaded,
          total: expected,
          wasm_memory_bytes: wasmMemory?.buffer?.byteLength ?? null,
        });
      }
    }
    if (loaded !== expected) {
      throw new Error(`model stream ended at ${loaded} of ${expected} bytes`);
    }
    return await combs_model_finish(handle);
  } catch (error) {
    // Idempotent: frees the partial buffer even if finish consumed the
    // handle before failing further down.
    combs_model_abort(handle);
    throw error;
  }
}

/** A readable byte stream for each way a model can arrive. */
async function openModelStream({ modelBlob, cacheName, cacheKey, modelUrl }) {
  if (modelBlob) {
    // A user-picked local file: disk-backed and structured-cloned in,
    // streamed without ever loading it whole.
    return { stream: modelBlob.stream(), total: modelBlob.size };
  }
  if (cacheName && cacheKey) {
    const cache = await caches.open(cacheName);
    const hit = await cache.match(cacheKey);
    if (!hit || !hit.body) {
      throw new Error(`model not in CacheStorage ${cacheName}: ${cacheKey}`);
    }
    const len = Number(hit.headers.get("content-length"));
    return { stream: hit.body, total: Number.isFinite(len) && len > 0 ? len : null };
  }
  if (modelUrl) {
    const response = await fetch(modelUrl);
    if (!response.ok || !response.body) {
      throw new Error(`fetching model: ${response.status} ${response.statusText}`);
    }
    const len = Number(response.headers.get("content-length"));
    return { stream: response.body, total: Number.isFinite(len) && len > 0 ? len : null };
  }
  throw new Error("load needs `modelBytes`, `modelBlob`, `cacheName`+`cacheKey`, or `modelUrl`");
}

/**
 * Runs one completion, forwarding every engine event to the main thread as
 * it happens. The terminal `done`/`error` event is what ends the consumer's
 * loop, so it is forwarded like any other.
 */
async function chat(id, request) {
  if (engineId === null) throw new Error("no engine loaded");
  await combs_chat_completion(
    engineId,
    id,
    JSON.stringify(request ?? {}),
    (json) => post("event", id, JSON.parse(json)),
  );
  post("done", id, null);
}

self.onmessage = async (event) => {
  const { kind, id, payload } = event.data ?? {};
  try {
    switch (kind) {
      case "load":
        // Serialize loads: a second one arriving mid-download would race
        // the first for the single engine slot. Chained on SETTLED, not
        // success — a failed load already reported to its own requester,
        // and letting its rejection poison every later load turned one
        // refused mount into a dead worker.
        ready = (ready ?? Promise.resolve())
          .catch(() => {})
          .then(() => load(id, payload));
        await ready;
        break;
      case "metadata":
        if (engineId === null) throw new Error("no engine loaded");
        post("metadata", id, JSON.parse(combs_engine_metadata(engineId)));
        break;
      case "chat":
        await chat(id, payload);
        break;
      case "cancel":
        // Nothing to reply to: the running request reports its own
        // cancellation through its event stream.
        combs_cancel(id);
        break;
      case "stats": {
        if (engineId === null) throw new Error("no engine loaded");
        const stats = JSON.parse(combs_engine_stats(engineId));
        stats.gpu_errors = gpuErrors;
        stats.wasm_memory_bytes = wasmMemory?.buffer?.byteLength ?? null;
        post("metadata", id, stats);
        break;
      }
      case "caps":
        post("metadata", id, JSON.parse(await combs_device_caps()));
        break;
      case "close":
        if (engineId !== null) combs_engine_destroy(engineId);
        engineId = null;
        post("done", id, null);
        break;
      default:
        throw new Error(`unknown request kind: ${kind}`);
    }
  } catch (error) {
    fail(id, error);
  }
};
