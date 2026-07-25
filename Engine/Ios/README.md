# Combs Engine — iOS shell

Thin Swift wrapper over the native core (`libcombs_ffi.a`, Metal via wgpu).
All inference lives in the Rust core; this is only glue.

## Build

```sh
# 1. Native static library (from Engine/Core):
cargo xtask target ios-arm64          # -> target/aarch64-apple-ios/release/libcombs_ffi.a

# 2. In Xcode:
#    - Add libcombs_ffi.a to "Link Binary With Libraries"
#    - Add Engine/Core/include/combs.h to the bridging header:
#        #include "combs.h"
#    - Add Engine/Ios/Sources/Combs/CombsEngine.swift to your target
```

## Usage

```swift
let engine = try CombsEngine(
    configJson: #"{"model_dir": "\(docs)/models/smollm2-135m"}"#
)
DispatchQueue.global().async {
    try? engine.chatCompletion(
        #"{"messages":[{"role":"user","content":"Hello"}]}"#,
        requestId: "req-1"
    ) { eventJson in
        // delta / done / error events
    }
}
engine.cancel(requestId: "req-1")
```

`.gguf` model files work as well: `{"model_dir": ".../model-q8_0.gguf"}`.
