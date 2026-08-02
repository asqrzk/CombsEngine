/*
 * combsmesh.h — CombsMesh emoji engine stable C ABI.
 *
 * Conventions (mirrors combs.h):
 *  - All functions are panic-fenced; errors land in a thread-local slot
 *    read via combsmesh_last_error().
 *  - int32_t return codes: 0 = ok, 1 = error.
 *  - Strings returned by combsmesh_op_json / combsmesh_infer are owned by
 *    the library: free them with combsmesh_string_free().
 *  - Byte buffers written to out-params (encrypt/decrypt/render) are owned
 *    by the library: free them with combsmesh_bytes_free(ptr, len) using
 *    the exact pointer and length returned. Never free either with free().
 *  - combsmesh_last_error() returns a BORROWED pointer: do not free it;
 *    it is invalidated by the next FFI call on the same thread.
 */
#ifndef COMBSMESH_H
#define COMBSMESH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Last error on this thread (borrowed, NUL-terminated) or NULL. */
const char* combsmesh_last_error(void);

/* Frees a string returned by combsmesh_op_json / combsmesh_infer. */
void combsmesh_string_free(char* p);

/* Frees a byte buffer written to an out-param by this library. */
void combsmesh_bytes_free(uint8_t* p, size_t len);

/*
 * Initializes the process-wide keyring (HKDF master key).
 * key == NULL or key_len == 0 generates a random master key.
 * Returns 0 on success, 1 on error.
 */
int32_t combsmesh_init(const uint8_t* key, size_t key_len);

/* Shuts the library down, zeroizing the master key. Always returns 0. */
int32_t combsmesh_shutdown(void);

/*
 * Encrypts a memory blob (AES-256-GCM, 12-byte nonce prepended).
 * Requires a prior combsmesh_init. On success *out/*out_len receive a
 * buffer to free with combsmesh_bytes_free. Returns 0 on success.
 */
int32_t combsmesh_encrypt_memory(const uint8_t* data, size_t len,
                                 uint8_t** out, size_t* out_len);

/* Decrypts a blob produced by combsmesh_encrypt_memory. Same contract. */
int32_t combsmesh_decrypt_memory(const uint8_t* data, size_t len,
                                 uint8_t** out, size_t* out_len);

/*
 * Renders frame `frame` of the emoji in `cmse` (a .cmse binary) to a
 * tightly packed frame_width * frame_height * 4 RGBA8 buffer (free with
 * combsmesh_bytes_free). Encrypted containers are accepted when the
 * process keyring holds the right key. Returns 0 on success.
 */
int32_t combsmesh_render_sprite(const uint8_t* cmse, size_t len,
                                uint32_t frame,
                                uint8_t** out_rgba, size_t* out_len);

/*
 * Runs inference on `prompt`; *out receives a string to free with
 * combsmesh_string_free. Requires the `engine` build feature and a model
 * loaded via combsmesh_op_json {"op":"engine_load","model_dir":"..."};
 * otherwise returns 1 with an "unsupported" error.
 */
int32_t combsmesh_infer(const char* prompt, char** out);

/*
 * JSON-FFI escape hatch: one stable symbol for every op that does not
 * need a dedicated C signature. Request: {"op": "...", ...}; response:
 * an op-specific JSON object. Free the returned string with
 * combsmesh_string_free. Returns NULL on error.
 *
 * Ops:
 *   build             {name, description?, blocks?}      → {emoji, binary_b64, unicode}
 *   from_binary       {binary_b64}                       → {emoji}
 *   to_unicode        {emoji}                            → {unicode}
 *   from_unicode      {unicode}                          → {emoji}
 *   registry_register {binary_b64, name?}                → {hash}
 *   registry_resolve  {name_or_hash}                     → {emoji, binary_b64}
 *   registry_list     {}                                 → {entries: [{name, hash, path, bytes}]}
 *   render            {binary_b64, frame?}               → {rgba_b64, width, height}
 *   engine_load       {model_dir, max_seq_len?}          → {loaded}   (engine feature)
 */
char* combsmesh_op_json(const char* request_json);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* COMBSMESH_H */
