/**
 * A stand-in for combs.worker.js that speaks the same envelope without a
 * GPU, a model, or WebAssembly.
 *
 * The real worker's job is to translate between the `{kind, id, payload}`
 * protocol and the engine. This one keeps the first half and replaces the
 * second with a script, so the protocol can be tested where the engine
 * cannot run — which is everywhere except a browser with WebGPU.
 */

const PIECES = ["Par", "is", " is", " the", " capital"];

let cancelled = new Set();
/** What `load` was configured with — metadata must report it afterwards. */
let configured = null;
/** How many engines this worker has created and never freed. */
let live = 0;

const post = (kind, id, payload) => self.postMessage({ kind, id, payload });

const METADATA = {
  architecture: "llama",
  vocab_size: 49152,
  max_position_embeddings: 8192,
  max_seq_len: 4096,
  page_size: 16,
  eos_token_ids: [2],
  im_end_id: null,
};

async function chat(id) {
  for (let i = 0; i < PIECES.length; i++) {
    // A tick between tokens, so a cancel posted from the main thread has
    // somewhere to land — exactly what the real engine's await provides.
    await new Promise((r) => setTimeout(r, 1));
    if (cancelled.has(id)) {
      post("event", id, {
        type: "done",
        finish_reason: "cancelled",
        stats: { prompt_tokens: 5, generated_tokens: i, ttft_ms: 1,
          decode_tokens_per_second: 0, prefill_tokens_per_second: 0,
          cache_pages_used: 1 },
      });
      post("done", id, null);
      return;
    }
    post("event", id, { type: "delta", text: PIECES[i], token_id: 100 + i });
  }
  post("event", id, {
    type: "done",
    finish_reason: "stop",
    stats: { prompt_tokens: 5, generated_tokens: PIECES.length, ttft_ms: 1,
      decode_tokens_per_second: 42, prefill_tokens_per_second: 500,
      cache_pages_used: 1 },
  });
  post("done", id, null);
}

self.onmessage = async (event) => {
  const { kind, id, payload } = event.data ?? {};
  switch (kind) {
    case "load":
      if (payload?.fail) return post("error", id, "load refused by the test");
      // The real worker accepts the model two ways and refuses neither
      // silently; the test worker holds it to the same rule.
      if (!payload?.modelBytes && !payload?.modelUrl) {
        return post("error", id, "load needs `modelBytes` or `modelUrl`");
      }
      if (live > 0) live -= 1; // the reload frees what it replaces
      live += 1;
      configured = {
        ...METADATA,
        max_seq_len: payload.max_seq_len ?? METADATA.max_seq_len,
      };
      return post("ready", id, configured);
    case "live":
      return post("metadata", id, { live });
    case "stats":
      return post("metadata", id, {
        kind: "paged",
        quantized: false,
        max_seq_len: configured?.max_seq_len ?? METADATA.max_seq_len,
        page_size: 16,
        page_bytes: 368640,
        sessions: [{ id: "(anonymous)", history_tokens: 42, pages_used: 3 }],
      });
    case "metadata":
      // The same answer `load` gave: an engine that reported one context
      // window at load and another when asked would be lying about which
      // one it allocated.
      return post("metadata", id, configured ?? METADATA);
    case "chat":
      return await chat(id);
    case "cancel":
      cancelled.add(id);
      return;
    default:
      return post("error", id, `unknown request kind: ${kind}`);
  }
};
