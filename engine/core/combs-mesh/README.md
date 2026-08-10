# combs-mesh — CombsMesh emoji engine core

An emoji that *does* things: a typed bag of blocks (text, sprite atlas,
todos, functions, APIs, lifecycle, character, emotion, encryption,
orchestration) with two serializations — a `.cmse` binary container and a
Unicode envelope string that survives text-only channels — plus AEAD
encryption at rest, a content-addressed registry, and CPU sprite rendering.

## Architecture position

Pure-data **L0 sibling** in `Engine/Core`: no burn, no wgpu, no C
dependencies — wasm32-unknown-unknown compiles clean. Inference is *not*
in this crate; the optional `engine` feature (used by `combs-mesh-ffi`)
adapts `combs-runtime` behind the `CombsEngineCore` trait.

```
combs-mesh        pure Rust: format, unicode codec, crypto, render, registry
combs-mesh-ffi    cdylib: combsmesh_* C ABI + combsmesh_op_json (JSON-FFI)
@combs/mesh (L2)  Deno client + MCP server + MeshPeer connector
```

## Protocol family

| Surface | Owner | Spec |
|---|---|---|
| `.cmse` binary + PUA envelope | this crate | this README (frozen wire format) |
| `/mesh/v1` peer federation | `@combs/mesh` `peer.ts` | `documentations/mesh-protocol.md` |
| MCP connector | `@combs/mesh` `mcp.ts` | mesh-protocol.md §related surfaces |
| Sealed envelopes | `@combs/zerotrust` | Engine/Js/zerotrust |

Product repos (e.g. the CombsMesh repo: SHARD runtime + M2 fabric crate)
consume all of the above via published packages — they never reimplement
the wire format, codec, or crypto.

## Binary format v1 (`.cmse`)

```
offset  size  field
0       4     magic b"CMSE"
4       2     u16 LE version (= 1)
6       2     u16 LE flags (bit0 = at least one block encrypted)
8       4     u32 LE block_count
12      16×N  directory: [u8;3] tag | u8 flags (bit0 = block encrypted) |
              u32 offset | u32 len | u32 crc32(IEEE, of stored payload)
12+16N  ..    payloads: serde_json of the block struct; encrypted blocks
              store nonce(12) || ciphertext, CRC'd as stored
```

10 block types, identified by 3-byte tags (`txt img tdo fnc api lfc chr
emo enc orc`); the tag index (0..10) is part of the wire format.

## Unicode envelope

For text channels (chat, MCP tool results), each block encodes as:

```
[TAG char U+E0061+idx] [2 × plane-15 length chars] [ceil(len/2) × plane-16 data chars]
```

- Plane 15 (U+F0000..U+FFFFF, PUA-A) holds the 24-bit payload length as two
  12-bit chunks in the block type's sub-range (16 sub-ranges of 4096).
- Plane 16 (U+100000..U+10FFFF, PUA-B) carries the payload, 2 bytes (BE
  u16) per codepoint.

All codepoints are valid scalar values; decoders skip non-marker text, so
envelopes ride inside ordinary prose.

## Crypto

RustCrypto (`aes-gcm`, `chacha20poly1305`, `hkdf`, `sha2`, `zeroize`) —
**not** `ring`: pure Rust keeps wasm32 and mobile cross-builds clean, and
the algorithms match `@combs/zerotrust`'s WebCrypto stack for future
Rust ↔ JS interop. Master key → HKDF-SHA256 subkey
(`info = "combsmesh-emoji-encryption"`) → AEAD with a random 12-byte nonce
prepended to each ciphertext. An `enc` block names the algorithm and which
block types to encrypt at rest; the writer applies it when a `KeyRing` is
supplied.

## Quick start

```rust
use combs_mesh::{EmojiBuilder, EmojiExporter};

let emoji = EmojiBuilder::new("my-emoji")
    .description("An emoji that does things")
    .add_todo("task1", "Build the thing")
    .add_image_rgba(64, 64, vec![0u8; 64 * 64 * 4])
    .with_agent_lifecycle()
    .build();

let binary  = EmojiExporter::to_binary(&emoji)?;   // .cmse container
let unicode = EmojiExporter::to_unicode(&emoji)?;  // text-channel string
let back    = EmojiExporter::from_binary(&binary)?;
```

## Registry

Content-addressed store at `$COMBS_HOME/mesh` (default
`~/.cache/combs/mesh`): `<sha256-of-binary>.cmse` + `index.json`.
`Registry::register / resolve / list / remove`; a missing or corrupt index
is rebuilt from the directory.

## Feature flags

| feature  | effect |
|----------|--------|
| (none)   | pure data + crypto + CPU render; `infer` → `Unsupported` |
| `engine` | enables the combs-runtime dependency for the `RuntimeEngine` adapter (implemented in `combs-mesh-ffi`) |
| `gpu`    | adds `WgpuRenderer` (raw wgpu sprite rendering) |
| `wasm`   | adds `#[wasm_bindgen]` browser bindings (`src/wasm.rs`) |

### `gpu` — WgpuRenderer

Implements the same `Renderer` trait as `CpuRenderer` over raw wgpu
(same wgpu 29 the engine uses). It owns its OWN instance/device — raw
wgpu permits multiple devices per adapter; the repo's process-global
device rule is a cubecl constraint and does not apply here (see
`src/render/gpu.rs` docs). CPU/GPU pixel parity is covered by
`tests/gpu.rs` (ignored by default, run with `-- --ignored`).

### `wasm` — browser bindings

Thin `#[wasm_bindgen]` free functions (`mesh_version`, `mesh_init`,
`emoji_build`, `emoji_to_unicode`, `emoji_from_binary`,
`emoji_from_unicode`, `mesh_encrypt`, `mesh_decrypt`) over the existing
APIs. The crate stays an rlib: a downstream cdylib (like `combs-wasm`
for the engine) turns it into a `.wasm`. Check:
`cargo check -p combs-mesh --target wasm32-unknown-unknown --features wasm`.

## C ABI

See `combs-mesh-ffi/include/combsmesh.h`: `combsmesh_init/shutdown`,
`encrypt/decrypt_memory`, `render_sprite`, `infer`, and the
`combsmesh_op_json` escape hatch (build/encode/registry/render/engine_load).
