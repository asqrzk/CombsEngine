package ai.combs

/**
 * CombsEngine — Kotlin wrapper over the combs native core (libcombs_ffi.so).
 *
 * Mirrors the LiteRT-LM Engine/Conversation shape: create with a JSON
 * config, stream chat completions, cancel by request id.
 *
 * ```kotlin
 * val engine = CombsEngine.create("""{"model_dir": "/data/models/smollm2"}""")
 * engine.chatCompletion("""{"messages":[{"role":"user","content":"hi"}]}""") { event ->
 *     // {"type":"delta","text":"..."} / {"type":"done",...} / {"type":"error",...}
 * }
 * engine.close()
 * ```
 */
class CombsEngine private constructor(private val handle: Long) : AutoCloseable {

    fun interface StreamCallback {
        fun onEvent(eventJson: String)
    }

    companion object {
        init {
            System.loadLibrary("combs_jni")
            System.loadLibrary("combs_ffi")
        }

        /** Device capabilities JSON (buffer limits, backend, features). */
        @JvmStatic
        external fun nativeDeviceCaps(): String?

        @JvmStatic
        private external fun nativeCreate(configJson: String): Long

        @JvmStatic
        private external fun nativeDestroy(handle: Long)

        @JvmStatic
        private external fun nativeMetadata(handle: Long): String?

        @JvmStatic
        private external fun nativeChatCompletion(
            handle: Long,
            requestJson: String,
            requestId: String,
            callback: StreamCallback,
        ): Int

        @JvmStatic
        private external fun nativeCancel(requestId: String): Int

        @JvmStatic
        external fun nativeLastError(): String?

        /**
         * Creates an engine. `configJson`:
         * `{"model_dir": "...", "max_seq_len": 4096, "kv_cache": "paged"}`
         */
        fun create(configJson: String): CombsEngine {
            val handle = nativeCreate(configJson)
            check(handle != 0L) { "combs engine creation failed: ${nativeLastError()}" }
            return CombsEngine(handle)
        }
    }

    /** Model metadata JSON (architecture, vocab, context, eos ids). */
    fun metadata(): String = nativeMetadata(handle) ?: "{}"

    /**
     * Runs a chat completion (BLOCKS the calling thread — call from a
     * background executor). Streams JSON events to `callback`.
     */
    fun chatCompletion(requestJson: String, requestId: String, callback: StreamCallback) {
        val rc = nativeChatCompletion(handle, requestJson, requestId, callback)
        check(rc == 0) { "chat completion failed: ${nativeLastError()}" }
    }

    /** Requests cancellation of an in-flight completion. */
    fun cancel(requestId: String) {
        nativeCancel(requestId)
    }

    override fun close() {
        nativeDestroy(handle)
    }
}
