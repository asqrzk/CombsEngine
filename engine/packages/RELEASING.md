# Releasing Combs Engine packages

All channels ship the **same version number** — pick one, bump it
everywhere first, then release in the order below (GitHub Release first,
because the npm/PyPI wrappers download its assets at install time).

## Before anything

- **Pick a version no channel has burned.** Channels can drift: 0.2.1
  existed only on crates.io (combs-mesh), so 0.2.2 became the first
  number available everywhere. Check `cargo search combs-mesh`,
  `npm view @combs-edge/combs-engine version`, and the GitHub releases
  page before choosing.
- **CI must be green on `main` first — and CI only runs on `main`.**
  Integration work meets CI for the first time at merge, so run the full
  local equivalent before merging: `cargo test --release --workspace`
  and `cargo check --workspace --all-targets` (the second catches
  crates like combs-ffi that a package-scoped test never compiles).

## 0. Bump versions

| File | Field |
|---|---|
| `Engine/Core/Cargo.toml` (workspace) | `[workspace.package] version` |
| `Engine/Packages/npm/combs-engine/package.json` | `version` |
| `Engine/Packages/npm/combs-mesh/package.json` | `version` |
| `Engine/Packages/npm/combs-client/package.json` | `version` |
| `Engine/Packages/npm/combs-zerotrust/package.json` | `version` |
| `Engine/Packages/pypi/combs-engine/pyproject.toml` | `version` + `src/combs_engine/__init__.py` |
| `engine/js/<pkg>/deno.json` (all ten, incl. `mesh` and `memory`) | `version` |

> `combs-zerotrust` is dual-published: source of truth is
> `Engine/Js/zerotrust/mod.js` — before publishing, copy it over
> `Engine/Packages/npm/combs-zerotrust/index.js` (keep them byte-identical)
> and `deno publish` ships `@combs/zerotrust` to JSR from the same file.

Then verify: `cargo test --release --workspace` and `cd Engine/Js && deno task test`.

## 1. GitHub Release (CLI binaries for every OS)

CI builds macOS (arm64 + Intel), Linux, and Windows **natively** — no
cross-compiling, no local toolchain needed:

```bash
git tag v0.2.0
git push origin v0.2.0
```

Watch: https://github.com/asqrzk/CombsEngine/actions — the `release`
workflow attaches `combs-0.2.0-macos-arm64.tar.gz`,
`combs-0.2.0-macos-x86_64.tar.gz`, `combs-0.2.0-linux-x86_64.tar.gz`,
`combs-0.2.0-windows-x86_64.zip` (each with the `combs` binary,
`libcombs_ffi.*` + `combs.h`, and `libcombsmesh_ffi.*` + `combsmesh.h`)
to the GitHub Release.

## 2. npm — `combs-client` + `combs-zerotrust` (the JS libraries)

The Svelte template depends on BOTH of these, so publish them before
announcing (scaffolded apps `npm install` them).

```bash
cp Engine/Js/zerotrust/mod.js Engine/Packages/npm/combs-zerotrust/index.js
cd Engine/Packages/npm/combs-zerotrust && npm publish --access public
cd ../combs-client && npm publish --access public
```

## 3. npm — `combs-engine` (the CLI wrapper) + `combs-mesh` (the mesh library)

```bash
cd Engine/Packages/npm/combs-engine
npm publish --access public
cd ../combs-mesh && npm publish --access public
```

Users then: `npm install -g combs-engine` → postinstall downloads the
step-1 release asset for their platform; `@combs-edge/combs-mesh`
downloads the same asset and keeps only `libcombsmesh_ffi.*` +
`combsmesh.h`. (One-time setup: `npm login`.)

## 4. PyPI — `combs-engine`

```bash
cd Engine/Packages/pypi/combs-engine
python3 -m pip install --upgrade build twine
rm -rf dist && python3 -m build
twine upload dist/*
```

Users then: `pip install combs-engine` → first `combs` run downloads the
step-1 release asset into `~/.cache/combs/bin`. (One-time setup: PyPI
account + `twine configure` or API token.)

## 5. JSR — the TypeScript framework (`@combs/*`)

Publishes all workspace packages at once (incl. `@combs/mesh`):

```bash
cd Engine/Js
deno publish --allow-slow-types   # interactive terminal: opens the browser once
```

`--allow-slow-types` is currently required (three public-API functions
lack explicit return types); drop the flag once those are fixed. Run
`deno publish --dry-run --allow-slow-types` first — it validates all
ten packages without auth.

(One-time: create the `combs` scope at https://jsr.io/new — then users can
`deno add @combs/core @combs/graph ...` or `npx jsr add @combs/core`.)

## 6. crates.io (the Rust library channel)

Eight crates ship, in dependency order (each waits for the previous to
be indexed; `cargo publish` handles the wait automatically). Cargo
reads `CARGO_REGISTRY_TOKEN` from the environment:

```bash
cd Engine/Core
for c in combs-core combs-formats combs-media combs-models \
         combs-runtime combs-ffi combs-mesh combs-mesh-ffi; do
  cargo publish -p $c || break
done
```

`combs-cli` is NOT published: its `build.rs` needs the UI template
vendored into the crate (`vendor/ui-template`) and the staging tooling
does not exist yet. Rust users get the binary from the GitHub Release
or `npm install -g combs-engine` instead.

## If the release workflow fails

Safe to delete and re-push the tag ONLY while nothing consumed it: the
failed run published no release object and no assets, and npm/PyPI have
not shipped (their installs reference the assets). Then:

```bash
git push origin :refs/tags/vX.Y.Z
git tag -d vX.Y.Z && git tag vX.Y.Z && git push origin vX.Y.Z
```

Known platform trap (fixed in combs-cli/Cargo.toml, keep it that way):
onnxruntime ships no prebuilt libraries for Intel macOS, so `ort` is a
target-specific dependency there with `load-dynamic` — the CLI builds
everywhere and Intel-Mac TTS needs a locally installed onnxruntime.

## Quick checklist per release

1. ☐ Version free on every channel; CI green on main
2. ☐ Bump all versions (table above) + zerotrust byte-copy
3. ☐ `cargo test --release --workspace` and `cargo check --workspace --all-targets` green
4. ☐ `deno task test` green
5. ☐ `git tag vX.Y.Z && git push origin vX.Y.Z` → wait for all 4 release assets
6. ☐ `cargo publish` the eight crates in dependency order (can run while assets build)
7. ☐ `npm publish` zerotrust, client, engine, mesh (2FA per your account settings)
8. ☐ `twine upload` combs-engine
9. ☐ `deno publish --allow-slow-types` from Engine/Js — interactive terminal
10. ☐ Verify: `npm view`, `cargo search`, pypi.org, jsr.io meta, release assets page
