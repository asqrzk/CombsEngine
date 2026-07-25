# Combs Engine — Android shell

Thin JNI + Kotlin wrapper over the native core (`libcombs_ffi.so`, Vulkan
via wgpu). All inference logic lives in the Rust core; this is only glue.

## Build

```sh
# 1. Native library (from Engine/Core):
cargo xtask target android-arm64        # -> target/aarch64-linux-android/release/libcombs_ffi.so

# 2. Copy into your app's jniLibs:
cp target/aarch64-linux-android/release/libcombs_ffi.so \
   <app>/src/main/jniLibs/arm64-v8a/

# 3. Compile the JNI shim (jni/combs_jni.c) with the NDK:
$ANDROID_HOME/ndk/<ver>/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android24-clang \
  -shared -fPIC -o <app>/src/main/jniLibs/arm64-v8a/libcombs_jni.so \
  Engine/Android/jni/combs_jni.c -I$JAVA_HOME/include -I$JAVA_HOME/include/darwin

# 4. Add combs-android/ as a module (or copy CombsEngine.kt) and load.
```

## Usage

```kotlin
val engine = CombsEngine.create("""{"model_dir": "${filesDir}/models/smollm2-135m"}""")
Thread {
    engine.chatCompletion(
        """{"messages":[{"role":"user","content":"Hello"}]}""",
        "req-1",
    ) { eventJson -> /* delta/done/error events */ }
}.start()
engine.cancel("req-1")
engine.close()
```

Model files: download on-device into `filesDir` (the Deno/Kotlin layer or a
simple downloader), then point `model_dir` at them. `.gguf` files work too
(`{"model_dir": ".../model-q8_0.gguf"}`).
