# @combs-edge/combs-mesh

> **Unpublished skeleton** — this package is wired into the release
> pipeline but will only become installable after the next release tag
> (`v*`) produces GitHub Release assets containing `libcombsmesh_ffi.*` +
> `combsmesh.h`. Until then, build from source
> (`cd Engine/Core && cargo build --release -p combs-mesh-ffi`) and point
> `COMBS_MESH_LIB` at the artifact.

CombsMesh emoji engine native library for your platform. `postinstall`
downloads the same GitHub Release asset as `combs-engine` and keeps only
`libcombsmesh_ffi.*` + `combsmesh.h` in `vendor/`.

```js
const { libPath, headerPath, requireLib } = require("@combs-edge/combs-mesh");

requireLib(); // absolute path to libcombsmesh_ffi (throws with guidance)
libPath();    // same, or null
headerPath(); // absolute path to combsmesh.h, or null
```

Environment:

- `COMBS_MESH_LIB` — explicit library path override.
- `COMBS_RELEASE_REPO` — override the release repo (default `asqrzk/CombsEngine`).
- `COMBS_SKIP_BINARY_DOWNLOAD=1` — skip the postinstall download.

For a typed client, see `@combs/mesh` (Deno/JSR, `Engine/Js/mesh`) which
`Deno.dlopen`s this exact library.
