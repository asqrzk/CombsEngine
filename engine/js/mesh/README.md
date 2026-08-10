# @combs/mesh

CombsMesh emoji engine client for Deno (L2) — FFI client, pure-TS unicode
codec, MCP server, and the MeshPeer connector.

## Mesh (FFI)

Typed client over `libcombsmesh_ffi` (crypto, render, registry, op
wrappers). Resolution: `COMBS_MESH_LIB` → `COMBS_LIB` →
`Engine/Core/dist/<plat>` → `Engine/Core/target/release`.

```ts
import { Mesh } from "@combs/mesh";
const mesh = Mesh.open();
mesh.init(); // random master key; mesh.init(keyBytes) for a fixed one
const built = mesh.buildEmoji({ name: "my-emoji", description: "demo" });
const hash = mesh.registryRegister(built.binary);
const frame = mesh.renderSprite(built.binary, 0); // {rgba, width, height}
mesh.close();
```

## unicode.ts (browser, no FFI)

Pure-TS port of the Rust `codepoint.rs` envelope codec — byte-identical
output (covered by the FFI parity test), safe for browser/no-FFI paths.

```ts
import { decodeBlocks, encodeBlocks } from "@combs/mesh";
const s = encodeBlocks([{ tag: "txt", payload: { name: "n", description: "d", specs: [] } }]);
const blocks = decodeBlocks(s); // round-trip, envelopes may ride inside prose
```

## MCP server

Serves emojis as MCP tools/resources to any agent (complements
`@combs/agents`' MCP client). Tools: `emoji_build`, `emoji_render`,
`emoji_list`, `emoji_to_unicode`; resources at `emoji://<name_or_hash>`.

```ts
import { EmojiMcpServer, Mesh } from "@combs/mesh";
await new EmojiMcpServer(Mesh.open()).serveStdio(); // newline-delimited JSON-RPC
```

## MeshPeer

Peer-to-peer connector: serves a registry over HTTP+WS with optional
bearer token; the client verifies sha256 integrity of fetched binaries.

```ts
import { MeshPeer } from "@combs/mesh";
const peer = MeshPeer.serve({ mesh, token: "secret" }); // GET /mesh/v1/{list,emoji/:id,announce,ws}
const { emojiJson, binary } = await MeshPeer.fetch(peer.baseUrl(), "my-emoji", { token: "secret" });
// throws on sha256 mismatch
```

## Tests

```sh
cd Engine/Js
deno task test   # mesh: unicode unit tests + FFI integration + peer test
                 # (skip FFI parts with COMBS_SKIP_INTEGRATION=1)
```
