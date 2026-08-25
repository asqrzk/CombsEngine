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
} from "./pkg/combs_wasm.js";

/** The one engine this worker hosts. One worker, one model, one flight. */
let engineId = null;
let ready = null;

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
  const { modelUrl, modelBytes, wasmUrl, ...config } = payload;
  await init(wasmUrl ? { module_or_path: wasmUrl } : undefined);

  let bytes;
  if (modelBytes) {
    bytes = new Uint8Array(modelBytes);
  } else if (modelUrl) {
    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`fetching model: ${response.status} ${response.statusText}`);
    }
    bytes = new Uint8Array(await response.arrayBuffer());
  } else {
    throw new Error("load needs `modelBytes` or `modelUrl`");
  }

  // A worker hosts one engine. Creating a second without freeing the
  // first would leave hundreds of megabytes of weights and KV arenas
  // reachable only by an id nobody holds any more — the engine table is
  // keyed by id, so an overwritten id is a leak, not a replacement.
  const previous = engineId;
  engineId = null;
  if (previous !== null) combs_engine_destroy(previous);

  const created = await combs_engine_create(JSON.stringify(config), bytes);
  engineId = created;
  post("ready", id, JSON.parse(combs_engine_metadata(created)));
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
        // the first for the single engine slot.
        ready = (ready ?? Promise.resolve()).then(() => load(id, payload));
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
