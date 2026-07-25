/*
 * combs.h — Combs Engine stable C ABI (L1 boundary)
 *
 * Hand-maintained mirror of `combs-ffi/src/lib.rs` (keep in sync; the API
 * is intentionally small and stable). All payloads are JSON strings.
 *
 * Threading: `combs_chat_completion` BLOCKS the calling thread and invokes
 * the stream callback on that same thread — call it from a worker thread
 * (Deno: dlopen `nonblocking: true` + `Deno.UnsafeCallback.threadSafe`;
 * Kotlin/Swift: a background executor).
 */
#ifndef COMBS_H
#define COMBS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* Opaque engine handle. */
typedef struct CombsEngine CombsEngine;

/* Streaming callback: `event_json` is a NUL-terminated JSON event
 * ({"type":"delta"|"done"|"error", ...}); `user_data` is the opaque pointer
 * passed to combs_chat_completion. */
typedef void (*CombsStreamCallback)(const char *event_json, void *user_data);

/* Device capabilities JSON (name, backend, buffer/compute limits, features).
 * Free with combs_string_free. Returns NULL on error. */
char *combs_device_caps_json(void);

/* Last error on this thread (borrowed; NULL if none). */
const char *combs_last_error(void);

/* Frees a string returned by a combs_*_json function. */
void combs_string_free(char *s);

/* Creates an engine from a JSON config:
 *   {"model_dir": "path", "max_seq_len": 8192, "page_size": 16,
 *    "kv_cache": "paged", "prefill_chunk_size": 512}
 * Only `model_dir` is required. Returns NULL on error (see
 * combs_last_error). */
CombsEngine *combs_engine_create(const char *config_json);

/* Destroys an engine handle. */
void combs_engine_destroy(CombsEngine *engine);

/* Engine metadata JSON (architecture, vocab, context, eos ids, ...).
 * Free with combs_string_free. */
char *combs_engine_metadata_json(const CombsEngine *engine);

/* Runs a chat completion; blocks until done, streaming events to `cb`.
 * `request_json`:
 *   {"prompt": "..."} or {"messages": [{"role","content"}, ...]},
 *   plus optional: max_tokens, temperature, top_k, top_p,
 *   repetition_penalty, frequency_penalty, presence_penalty, seed,
 *   stop: ["..."], stop_token_ids: [...], prefill_chunk_size.
 * `request_id` identifies the request for combs_cancel.
 * Returns 0 on success, -1 on error. */
int combs_chat_completion(CombsEngine *engine,
                          const char *request_json,
                          const char *request_id,
                          CombsStreamCallback cb,
                          void *user_data);

/* Requests cancellation of an in-flight completion.
 * Returns 0 if found, 1 if no such request, -1 on error. */
int combs_cancel(const char *request_id);

#ifdef __cplusplus
}
#endif

#endif /* COMBS_H */
