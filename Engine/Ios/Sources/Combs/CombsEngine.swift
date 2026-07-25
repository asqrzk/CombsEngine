import Foundation

/// CombsEngine — Swift wrapper over the combs native core (libcombs_ffi).
///
/// Thin glue over the stable C ABI (combs.h); all inference lives in the
/// Rust core (Metal via wgpu).
///
/// ```swift
/// let engine = try CombsEngine(configJson: #"{"model_dir": ".../smollm2-135m"}"#)
/// try engine.chatCompletion(#"{"messages":[{"role":"user","content":"hi"}]}"#,
///                           requestId: "req-1") { event in
///     print(event) // delta / done / error JSON
/// }
/// ```
public final class CombsEngine {
    public typealias StreamHandler = (String) -> Void

    private let handle: OpaquePointer

    /// Unmanaged callback context passed through the C ABI.
    private class CallbackBox {
        let handler: StreamHandler
        init(_ handler: @escaping StreamHandler) { self.handler = handler }
    }

    /// Device capabilities JSON (buffer limits, backend, features).
    public static var deviceCaps: String {
        guard let caps = combs_device_caps_json() else { return "{}" }
        defer { combs_string_free(caps) }
        return String(cString: caps)
    }

    /// Last error on this thread.
    public static var lastError: String {
        guard let err = combs_last_error() else { return "" }
        return String(cString: err)
    }

    public init(configJson: String) throws {
        var handle: OpaquePointer?
        configJson.withCString { ptr in
            handle = combs_engine_create(ptr)
        }
        guard let handle else {
            throw NSError(
                domain: "CombsEngine",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "engine creation failed: \(CombsEngine.lastError)"]
            )
        }
        self.handle = handle
    }

    deinit {
        combs_engine_destroy(handle)
    }

    /// Model metadata JSON (architecture, vocab, context, eos ids).
    public var metadata: String {
        guard let md = combs_engine_metadata_json(handle) else { return "{}" }
        defer { combs_string_free(md) }
        return String(cString: md)
    }

    /// Runs a chat completion (BLOCKS — call from a background queue).
    /// Streams JSON events to `handler`.
    public func chatCompletion(
        _ requestJson: String,
        requestId: String,
        handler: @escaping StreamHandler
    ) throws {
        let box = CallbackBox(handler)
        let userData = Unmanaged.passRetained(box).toOpaque()
        defer { Unmanaged<CallbackBox>.fromOpaque(userData).release() }

        let callback: CombsStreamCallback = { eventJson, userData in
            guard let eventJson, let userData else { return }
            let box = Unmanaged<CallbackBox>.fromOpaque(userData).takeUnretainedValue()
            box.handler(String(cString: eventJson))
        }

        let rc = requestJson.withCString { reqPtr in
            requestId.withCString { idPtr in
                combs_chat_completion(handle, reqPtr, idPtr, callback, userData)
            }
        }
        guard rc == 0 else {
            throw NSError(
                domain: "CombsEngine",
                code: Int(rc),
                userInfo: [NSLocalizedDescriptionKey: "chat completion failed: \(CombsEngine.lastError)"]
            )
        }
    }

    /// Requests cancellation of an in-flight completion.
    public func cancel(requestId: String) {
        requestId.withCString { ptr in
            _ = combs_cancel(ptr)
        }
    }
}
