# CombsMesh spec — resolution notes

How the original CombsMesh emoji-engine spec maps to this implementation:
what was adopted verbatim, what was adapted, what was rejected, and why.

## (a) Crypto: `ring` → RustCrypto — ADAPTED

The spec asks for `ring`. We use `aes-gcm` + `chacha20poly1305` + `hkdf` +
`sha2` + `zeroize` — the **same algorithms and wire layout** (AES-256-GCM /
ChaCha20-Poly1305, HKDF-SHA256 with `info = "combsmesh-emoji-encryption"`,
random 12-byte nonce prepended), but pure Rust:

- `ring`'s asm breaks `wasm32-unknown-unknown` (a stated spec target);
- C/asm deps break iOS/Android cross builds (the repo's oniguruma lesson);
- RustCrypto algorithm-matches `@combs/zerotrust`'s WebCrypto
  (AES-GCM/HKDF), so Rust ↔ JS envelopes can interop later.

No spec change needed unless byte-level interop with a ring-based
implementation is required (it would still work — the layout is identical).

## (b) UniFFI → hand-written JSON-FFI — REJECTED (UniFFI)

The repo's entire cross-platform story already works without UniFFI:
`combs-ffi` exposes hand-written C symbols bound trivially from Deno
(`Deno.dlopen`), Android (JNI) and iOS (Swift). UniFFI would add proc-macro
version coupling, a second binding philosophy, and still not cover WASM.
`combs-mesh-ffi` therefore mirrors `combs-ffi`: panic-fenced exports,
thread-local error slot, `combsmesh_` prefix, `include/combsmesh.h`
maintained by hand (kept honest by ABI tests calling the extern fns
directly).

## (c) Sprite rendering: CPU v1, trait seam — ADAPTED

v1 ships `CpuRenderer` only (frame extraction + src-over alpha compositing
in u16 math, zero deps, deterministic). GPU/wgpu rendering and SVG
rasterization are deliberately deferred; the `Renderer` trait
(`render_frame`, `compose`) is the seam a wgpu renderer slots into later
without API churn.

## (d) `combsmesh_op_json` escape hatch — ADDED

Beyond the spec's fixed symbol list we add one stable JSON-FFI symbol (the
MLC `json_ffi` pattern the engine already uses): `build`, `from_binary`,
`to_unicode`, `from_unicode`, `registry_register`, `registry_resolve`,
`registry_list`, `render`, `engine_load`. Every future op (block inspect,
multi-emoji compose, ...) lands here without ABI churn.

## (e) `infer` optional via the `engine` feature — ADAPTED

The spec bakes inference into the engine core. Here the core crate is pure
data (no burn/wgpu dep); `CombsEngineCore::infer` returns
`EngineError::Unsupported` in the default build (`DefaultEngine`). The
`engine` feature on `combs-mesh-ffi` provides `RuntimeEngine`, an adapter
over `combs_runtime::Engine` (model dir supplied via the `engine_load` op;
single non-streaming greedy generation; reuses the existing single-flight
queue, KV sessions and sampler — no engine changes).

## (f) Layout: `combsmesh/` → `Engine/Core/combs-mesh` — ADAPTED

The spec's top-level `combsmesh/` directory is realized as
`Engine/Core/combs-mesh` + `Engine/Core/combs-mesh-ffi` inside the existing
cargo workspace (same module structure), so it ships through the same
xtask/release/npm pipeline and stays extractable to a future standalone
repo (no workspace deps in the core crate beyond pure-Rust crates).

## Smaller deviations

- **Directory entries are 16 bytes** (tag 3 + flags 1 + offset 4 + len 4 +
  crc 4); the spec text undercounted at 12.
- **Emoji name round-trips via the `txt` block** — the v1 header has no
  name field; `EmojiBuilder` always emits one.
- **`KeyRing::encrypt` returns `Result`** (the spec sketched an infallible
  signature) — AEAD APIs are fallible and the codebase bans unwraps on
  library paths.
- **CRC via `crc32fast`** rather than a hand-rolled table (same IEEE CRC32
  on the wire).
- **Binary artifact name** is `libcombsmesh_ffi.*` (crate `[lib] name =
  "combsmesh_ffi"`), matching the `combsmesh_` symbol prefix.
