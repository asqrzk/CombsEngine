# Releasing Combs Engine packages

All channels ship the **same version number** — pick one (e.g. `0.2.0`),
bump it everywhere first, then release in the order below (GitHub Release
first, because the npm/PyPI wrappers download its assets at install time).

## 0. Bump versions

| File | Field |
|---|---|
| `Engine/Core/Cargo.toml` (workspace) | `[workspace.package] version` |
| `Engine/Packages/npm/combs-engine/package.json` | `version` |
| `Engine/Packages/npm/combs-client/package.json` | `version` |
| `Engine/Packages/npm/combs-zerotrust/package.json` | `version` |
| `Engine/Packages/pypi/combs-engine/pyproject.toml` | `version` + `src/combs_engine/__init__.py` |
| `Engine/Js/<pkg>/deno.json` (all seven) | `version` |

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
`libcombs_ffi.*` and `combs.h`) to the GitHub Release.

## 2. npm — `combs-client` + `combs-zerotrust` (the JS libraries)

The Svelte template depends on BOTH of these, so publish them before
announcing (scaffolded apps `npm install` them):

```bash
cp Engine/Js/zerotrust/mod.js Engine/Packages/npm/combs-zerotrust/index.js
cd Engine/Packages/npm/combs-zerotrust && npm publish --access public
cd ../combs-client && npm publish --access public
```

## 3. npm — `combs-engine` (the CLI wrapper)

```bash
cd Engine/Packages/npm/combs-engine
npm publish --access public
```

Users then: `npm install -g combs-engine` → postinstall downloads the
step-1 release asset for their platform. (One-time setup: `npm login`.)

## 4. PyPI — `combs-engine`

```bash
cd Engine/Packages/pypi/combs-engine
python3 -m pip install --upgrade build twine
python3 -m build
twine upload dist/*
```

Users then: `pip install combs-engine` → first `combs` run downloads the
step-1 release asset into `~/.cache/combs/bin`. (One-time setup: PyPI
account + `twine configure` or API token.)

## 5. JSR — the TypeScript framework (`@combs/*`)

Publishes all six workspace packages at once:

```bash
cd Engine/Js
deno publish        # first run opens a browser to authorize the @combs scope
```

(One-time: create the `combs` scope at https://jsr.io/new — then users can
`deno add @combs/core @combs/graph ...` or `npx jsr add @combs/core`.)

## 6. crates.io (optional, for Rust developers)

Published crates must be self-contained, so stage the UI template inside
the CLI crate first (its `build.rs` picks `vendor/ui-template` over the
repo path automatically):

```bash
cd Engine/Core
mkdir -p combs-cli/vendor
cp -R ../Ui/template combs-cli/vendor/ui-template
cargo publish -p combs-core -p combs-formats -p combs-models -p combs-runtime \
              -p combs-ffi -p combs-cli --dry-run   # then for real, in dependency order
rm -rf combs-cli/vendor
```

Users then: `cargo install combs-cli`.

## Quick checklist per release

1. ☐ Bump all versions (table above)
2. ☐ `cargo test --release --workspace` green
3. ☐ `deno task test` green
4. ☐ `git tag vX.Y.Z && git push origin vX.Y.Z` → CI release assets
5. ☐ `npm publish` combs-client, then combs-engine
6. ☐ `twine upload` combs-engine
7. ☐ `deno publish` from Engine/Js
