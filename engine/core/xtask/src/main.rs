//! `cargo xtask` — single build entrypoint for Combs Engine.
//!
//! Besides the host build, this orchestrates the cross-platform matrix.
//! wgpu already abstracts Metal/Vulkan/DX12/WebGPU, so the *same* Rust code
//! compiles for every target; what differs is the toolchain glue:
//!
//! | target          | triple                    | artifact              | build |
//! |-----------------|---------------------------|-----------------------|-------|
//! | macos-arm64     | aarch64-apple-darwin      | libcombs_ffi.dylib    | full  |
//! | macos-x86_64    | x86_64-apple-darwin       | libcombs_ffi.dylib    | full  |
//! | ios-arm64       | aarch64-apple-ios         | libcombs_ffi.a        | full  |
//! | android-arm64   | aarch64-linux-android     | libcombs_ffi.so       | full (needs NDK) |
//! | windows-x86_64  | x86_64-pc-windows-msvc    | combs_ffi.dll         | check (needs MSVC to link) |
//! | linux-x86_64    | x86_64-unknown-linux-gnu  | libcombs_ffi.so       | check (needs a cross linker) |
//!
//! `xtask target <name>` runs a full build when the toolchain is available,
//! otherwise a `cargo check` (type/borrow verification without linking).
//! `xtask bundle` assembles `dist/<target>/` with the library + combs.h.
//!
//! The browser is its own command rather than a row above, because it does
//! not produce a C library and cannot be bundled like one: `xtask web`
//! builds `combs-wasm` for `wasm32-unknown-unknown` and, when
//! `wasm-bindgen` is installed, emits the loadable ES module beside the
//! worker script in `js/core/pkg/`.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Context, Result};
use clap::Parser;

#[derive(clap::Parser)]
#[command(name = "xtask", about = "Combs Engine build orchestrator")]
struct Cli {
    #[command(subcommand)]
    command: XCommand,
}

#[derive(clap::Subcommand)]
enum XCommand {
    /// Build the workspace (host target).
    Build {
        /// Build in release mode.
        #[arg(long)]
        release: bool,
    },
    /// Run the `combs` CLI (release mode), passing args through.
    Run {
        /// Arguments forwarded to `combs`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Build (or check) one cross-compilation target.
    Target {
        /// Target name: macos-arm64 | macos-x86_64 | ios-arm64 |
        /// android-arm64 | windows-x86_64 | linux-x86_64 | web-wasm32
        name: String,
        /// Force cargo check even when a full build is possible.
        #[arg(long)]
        check: bool,
    },
    /// Run every public gate: formatting, lints, and the test tiers.
    ///
    /// One command so a change is checked the same way by everyone,
    /// rather than by whichever subset each person remembers.
    Gates {
        /// Skip the GPU-backed tiers (formats and models take minutes
        /// and need an adapter); leaves formatting, lints and the
        /// backend-free crates.
        #[arg(long)]
        quick: bool,
        /// Treat formatting and lints as failures rather than as the
        /// standing debt they currently are.
        #[arg(long)]
        strict: bool,
    },
    /// Build the browser engine: combs-wasm for wasm32-unknown-unknown,
    /// plus the wasm-bindgen ES module in js/core/pkg/.
    Web {
        /// Build without optimizations (much faster, much larger).
        #[arg(long)]
        debug: bool,
        /// Also copy the worker script and pkg/ into this directory, for a
        /// host application that serves them (e.g. the platform's engine/).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Serve in half precision. Writes to the same pkg/, so a build
        /// made with this flag REPLACES the f32 bundle until one is made
        /// without it — the module logs which it is on load.
        #[arg(long)]
        f16: bool,
    },
    /// Show the platform matrix and detected toolchains.
    Matrix,
    /// Assemble dist/<target>/ artifacts (library + include/combs.h).
    Bundle,
}

/// One row of the platform matrix.
struct Target {
    name: &'static str,
    triple: &'static str,
    /// Library file name inside target/<triple>/release/.
    artifact: &'static str,
    /// CombsMesh library file name (combs-mesh-ffi, same target dir).
    mesh_artifact: &'static str,
    /// `false` = cargo check only on this host (no cross linker).
    full_build: fn(&Ctx) -> bool,
    /// Extra env for the cargo invocation.
    env: fn(&Ctx) -> Vec<(String, String)>,
}

struct Ctx {
    root: PathBuf,
}

fn android_ndk() -> Option<PathBuf> {
    let home = std::env::var("ANDROID_HOME")
        .or_else(|_| std::env::var("ANDROID_SDK_ROOT"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("Library/Android/sdk")
        });
    let ndk_dir = home.join("ndk");
    let mut versions: Vec<PathBuf> = std::fs::read_dir(&ndk_dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    versions.sort();
    versions.pop()
}

fn android_linker() -> Option<String> {
    let ndk = android_ndk()?;
    let prebuilt = ndk.join("toolchains/llvm/prebuilt");
    let host = std::fs::read_dir(&prebuilt)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.is_dir())?;
    let clang = host.join("bin/aarch64-linux-android24-clang");
    clang.exists().then(|| clang.to_string_lossy().into_owned())
}

const TARGETS: &[Target] = &[
    Target {
        name: "macos-arm64",
        triple: "aarch64-apple-darwin",
        artifact: "libcombs_ffi.dylib",
        mesh_artifact: "libcombsmesh_ffi.dylib",
        full_build: |_| true,
        env: |_| vec![],
    },
    Target {
        name: "macos-x86_64",
        triple: "x86_64-apple-darwin",
        artifact: "libcombs_ffi.dylib",
        mesh_artifact: "libcombsmesh_ffi.dylib",
        full_build: |_| true,
        env: |_| vec![],
    },
    Target {
        name: "ios-arm64",
        triple: "aarch64-apple-ios",
        artifact: "libcombs_ffi.a",
        mesh_artifact: "libcombsmesh_ffi.a",
        full_build: |_| true,
        env: |_| vec![],
    },
    Target {
        name: "android-arm64",
        triple: "aarch64-linux-android",
        artifact: "libcombs_ffi.so",
        mesh_artifact: "libcombsmesh_ffi.so",
        full_build: |_| android_linker().is_some(),
        env: |_| {
            android_linker()
                .map(|l| vec![("CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER".to_string(), l)])
                .unwrap_or_default()
        },
    },
    Target {
        name: "windows-x86_64",
        triple: "x86_64-pc-windows-msvc",
        artifact: "combs_ffi.dll",
        mesh_artifact: "combsmesh_ffi.dll",
        // No MSVC linker on this host: check only.
        full_build: |_| false,
        env: |_| vec![],
    },
    Target {
        name: "linux-x86_64",
        triple: "x86_64-unknown-linux-gnu",
        artifact: "libcombs_ffi.so",
        mesh_artifact: "libcombsmesh_ffi.so",
        // No cross linker on this host: check only.
        full_build: |_| false,
        env: |_| vec![],
    },
];

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent dir")
        .to_path_buf()
}

fn cargo() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string())
}

fn run(mut cmd: Command) -> Result<()> {
    let status = cmd.status().context("spawning cargo")?;
    if !status.success() {
        bail!("command failed with {status}");
    }
    Ok(())
}

/// Builds the reduced-precision `combs` CLI variant. It lives in its own
/// `target-f16/` tree so the two feature sets never invalidate each other's
/// incremental artifacts.
fn build_f16_cli(ctx: &Ctx) -> Result<PathBuf> {
    let mut cmd = Command::new(cargo());
    cmd.current_dir(&ctx.root)
        .env("CARGO_TARGET_DIR", ctx.root.join("target-f16"))
        .args(["build", "--release", "-p", "combs-cli", "--features", "f16"]);
    eprintln!("== combs (f16 variant, target-f16/) ==");
    run(cmd)?;
    Ok(ctx.root.join("target-f16/release/combs"))
}

fn build_target(ctx: &Ctx, target: &Target, force_check: bool) -> Result<PathBuf> {
    let full = !force_check && (target.full_build)(ctx);
    let mut cmd = Command::new(cargo());
    cmd.current_dir(&ctx.root);
    if full {
        cmd.args(["build", "--release"]);
    } else {
        cmd.args(["check", "--release"]);
    }
    // combs-mesh-ffi ships in the same dist/ dirs. Its `engine` feature is
    // deliberately OFF here (inference-free artifact by default; enable with
    // `--features combs-mesh-ffi/engine` for an inference-capable build).
    cmd.args([
        "-p",
        "combs-ffi",
        "-p",
        "combs-mesh-ffi",
        "--target",
        target.triple,
    ]);
    for (k, v) in (target.env)(ctx) {
        cmd.env(k, v);
    }
    eprintln!(
        "== {} ({}): {} ==",
        target.name,
        target.triple,
        if full { "full build" } else { "cargo check" }
    );
    run(cmd)?;
    Ok(ctx
        .root
        .join("target")
        .join(target.triple)
        .join("release")
        .join(target.artifact))
}

/// Builds the browser engine and its JS glue.
///
/// The wasm module is useless to a page on its own — it needs the
/// wasm-bindgen shim that knows how to pass strings and closures across
/// the boundary. When `wasm-bindgen` is missing this says so and stops,
/// rather than leaving a `pkg/` that looks built and imports nothing.
/// Every gate that can live in the repository.
///
/// Deliberately only the public half: formatting, lints and tests. The
/// checks that read a private vocabulary live with that vocabulary, in
/// tooling this repository does not carry and must not name — a
/// committed file that points at them would leak their existence, which
/// is the one thing they exist to prevent.
///
/// Formatting and lints REPORT by default rather than fail. The
/// workspace does not currently satisfy either, by a wide margin, and a
/// gate that can never pass is a gate everyone learns to ignore —
/// which is worse than one that says plainly, every time it runs, how
/// large the debt is. `--strict` makes them binding, and is what a
/// cleanup change should be judged by.
fn cmd_gates(ctx: &Ctx, quick: bool, strict: bool) -> Result<()> {
    let mut failed: Vec<String> = Vec::new();
    let mut debt: Vec<String> = Vec::new();

    let output = |args: &[&str]| -> (bool, String) {
        let out = Command::new(cargo())
            .current_dir(&ctx.root)
            .args(args)
            .output();
        match out {
            Ok(o) => {
                let mut text = String::from_utf8_lossy(&o.stdout).into_owned();
                text.push_str(&String::from_utf8_lossy(&o.stderr));
                (o.status.success(), text)
            }
            Err(e) => (false, e.to_string()),
        }
    };

    eprintln!("== fmt ==");
    let (fmt_ok, fmt_out) = output(&["fmt", "--check"]);
    if !fmt_ok {
        let n = fmt_out
            .lines()
            .filter(|l| l.starts_with("Diff in "))
            .count();
        eprintln!("   {n} formatting differences");
        if strict {
            failed.push("fmt".into())
        } else {
            debt.push(format!("fmt: {n}"))
        }
    }

    eprintln!("== clippy ==");
    let (_, clippy_out) = output(&["clippy", "--workspace", "--all-targets"]);
    let n = clippy_out
        .lines()
        .filter(|l| l.starts_with("warning: ") || l.starts_with("error: "))
        .count();
    if n > 0 {
        eprintln!("   {n} lint findings");
        if strict {
            failed.push("clippy".into())
        } else {
            debt.push(format!("clippy: {n}"))
        }
    }

    let mut tier = |name: &str, args: &[&str]| {
        eprintln!("== {name} ==");
        let mut cmd = Command::new(cargo());
        cmd.current_dir(&ctx.root).args(args);
        if run(cmd).is_err() {
            failed.push(name.to_string());
        }
    };
    tier(
        "core",
        &["test", "--release", "-p", "combs-core", "-p", "combs-media"],
    );
    if !quick {
        // These need a GPU adapter and take minutes; a machine without
        // one skips their GPU cases rather than failing them.
        tier("formats", &["test", "--release", "-p", "combs-formats"]);
        tier(
            "models",
            &["test", "--release", "-p", "combs-models", "--lib"],
        );
        tier("runtime", &["test", "--release", "-p", "combs-runtime"]);
        tier(
            "diffusion",
            &["test", "--release", "-p", "combs-diffusion", "--lib"],
        );
    }

    if !debt.is_empty() {
        eprintln!(
            "\nstanding debt (not failing; --strict makes it): {}",
            debt.join(", ")
        );
    }
    if failed.is_empty() {
        eprintln!("all gates passed");
        Ok(())
    } else {
        bail!("gates failed: {}", failed.join(", "))
    }
}

fn cmd_web(ctx: &Ctx, release: bool, out_dir: Option<PathBuf>, f16: bool) -> Result<()> {
    const TRIPLE: &str = "wasm32-unknown-unknown";

    let mut cmd = Command::new(cargo());
    cmd.current_dir(&ctx.root)
        .args(["build", "-p", "combs-wasm", "--target", TRIPLE]);
    if release {
        cmd.arg("--release");
    }
    if f16 {
        cmd.args(["--features", "f16"]);
    }
    eprintln!(
        "== web-wasm32 ({TRIPLE}): {} {} ==",
        if release { "release" } else { "debug" },
        if f16 { "f16" } else { "f32" }
    );
    run(cmd)?;

    let wasm = ctx
        .root
        .join("target")
        .join(TRIPLE)
        .join(if release { "release" } else { "debug" })
        .join("combs_wasm.wasm");
    if !wasm.exists() {
        bail!("expected artifact missing: {}", wasm.display());
    }
    let bytes = std::fs::metadata(&wasm).map(|m| m.len()).unwrap_or(0);
    eprintln!("built {} ({:.1} MB)", wasm.display(), bytes as f64 / 1e6);
    check_wasm_max_memory(&wasm)?;

    let out = ctx.root.join("../js/core/pkg");
    let mut bindgen_args = vec!["--target", "web", "--out-name", "combs_wasm"];
    if release {
        // The name section is 7.7 MB of function names a release visitor
        // never sees; debug builds keep it so stack traces stay readable.
        bindgen_args.push("--remove-name-section");
        bindgen_args.push("--remove-producers-section");
    }
    bindgen_args.push("--out-dir");
    let bindgen = Command::new("wasm-bindgen")
        .args(&bindgen_args)
        .arg(&out)
        .arg(&wasm)
        .current_dir(&ctx.root)
        .status();
    match bindgen {
        Ok(status) if status.success() => {
            eprintln!("bindings -> {}", out.display());
            // The glue masks pointers with `>>> 0` so addresses past
            // 2 GiB stay unsigned in JS. A bindgen that stopped
            // emitting them would corrupt every big-model mount —
            // loudly is the only acceptable way for that to surface.
            let glue = out.join("combs_wasm.js");
            match std::fs::read_to_string(&glue) {
                Ok(text) if text.contains(">>> 0") => {}
                Ok(_) => eprintln!(
                    "WARNING: {} carries no '>>> 0' pointer masks — \
                     mounts beyond 2 GiB will read the WRONG addresses",
                    glue.display()
                ),
                Err(e) => eprintln!("WARNING: cannot read {}: {e}", glue.display()),
            }
            if release {
                // wasm-opt -Oz takes ~30s and buys ~12% on this module.
                // Optional on purpose: a missing binaryen must not fail
                // the build, and debug builds keep the unoptimized module.
                let module = out.join("combs_wasm_bg.wasm");
                let opt = Command::new("wasm-opt")
                    .args([
                        "-Oz",
                        "--enable-bulk-memory",
                        "--enable-nontrapping-float-to-int",
                    ])
                    .arg(&module)
                    .arg("-o")
                    .arg(&module)
                    .status();
                match opt {
                    Ok(s) if s.success() => {
                        let bytes = std::fs::metadata(&module).map(|m| m.len()).unwrap_or(0);
                        eprintln!("wasm-opt -Oz -> {:.1} MB", bytes as f64 / 1e6);
                    }
                    Ok(s) => eprintln!("wasm-opt failed ({s}); shipping unoptimized"),
                    Err(_) => eprintln!(
                        "wasm-opt not found; shipping unoptimized (brew install binaryen)"
                    ),
                }
            }
            if let Some(dest) = out_dir {
                copy_web_bundle(ctx, &out, &dest)?;
                eprintln!("bundle -> {}", dest.display());
            } else {
                eprintln!(
                    "serve js/core/combs.worker.js and js/core/pkg/ from the same directory, \
                     or re-run with --out <dir> to copy both somewhere that already is"
                );
            }
            Ok(())
        }
        Ok(status) => bail!("wasm-bindgen failed: {status}"),
        Err(e) => bail!(
            "wasm-bindgen not found ({e}). Install the version matching the \
             wasm-bindgen crate in Cargo.lock:\n  \
             cargo install wasm-bindgen-cli --version <locked> --locked"
        ),
    }
}

/// Copies the worker script and the generated package into a host
/// application's static directory. Both must land together and stay
/// together: the worker imports `./pkg/combs_wasm.js` by relative path.
/// Postcondition on the built module: linear memory must declare the
/// full 4 GiB maximum (65536 pages). The linker defaults to 2 GiB, and
/// a silently-reverted `--max-memory` link flag would cap every mount
/// at half the address space — this reads the wasm memory section
/// directly so the ceiling is a checked fact, not a build-config hope.
fn check_wasm_max_memory(path: &std::path::Path) -> Result<()> {
    const WANT_PAGES: u64 = 65536;
    let bytes = std::fs::read(path)?;
    anyhow::ensure!(
        bytes.len() > 8 && &bytes[0..4] == b"\0asm",
        "not a wasm module"
    );
    let mut pos = 8usize; // magic + version
    let varint = |bytes: &[u8], pos: &mut usize| -> Result<u64> {
        let mut out = 0u64;
        for shift in (0..64).step_by(7) {
            let b = *bytes
                .get(*pos)
                .ok_or_else(|| anyhow::anyhow!("truncated wasm"))?;
            *pos += 1;
            out |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(out);
            }
        }
        bail!("varint overflow");
    };
    while pos < bytes.len() {
        let id = bytes[pos];
        pos += 1;
        let size = varint(&bytes, &mut pos)? as usize;
        let body_end = pos + size;
        if id == 5 {
            // memory section: count, then per-memory limits.
            let mut p = pos;
            let count = varint(&bytes, &mut p)?;
            anyhow::ensure!(count >= 1, "wasm module declares no memory");
            let flags = varint(&bytes, &mut p)?;
            let min = varint(&bytes, &mut p)?;
            let max = if flags & 1 == 1 {
                Some(varint(&bytes, &mut p)?)
            } else {
                None
            };
            match max {
                Some(m) if m == WANT_PAGES => {
                    eprintln!("memory: min {min} pages, max {m} pages (4 GiB ceiling ok)");
                    return Ok(());
                }
                Some(m) => bail!(
                    "wasm memory max is {m} pages ({:.2} GiB) — expected {WANT_PAGES}; \
                     the --max-memory link flag in .cargo/config.toml got lost",
                    m as f64 * 65536.0 / 1e9
                ),
                None => bail!(
                    "wasm memory declares NO maximum — expected {WANT_PAGES} pages; \
                     the --max-memory link flag in .cargo/config.toml got lost"
                ),
            }
        }
        pos = body_end;
    }
    bail!("no memory section found in {}", path.display());
}

fn copy_web_bundle(ctx: &Ctx, pkg: &std::path::Path, dest: &std::path::Path) -> Result<()> {
    let worker = ctx.root.join("../js/core/combs.worker.js");
    std::fs::create_dir_all(dest.join("pkg"))?;
    std::fs::copy(&worker, dest.join("combs.worker.js"))
        .with_context(|| format!("copying {}", worker.display()))?;
    for entry in std::fs::read_dir(pkg)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            std::fs::copy(entry.path(), dest.join("pkg").join(entry.file_name()))?;
        }
    }
    Ok(())
}

fn cmd_matrix() {
    let ctx = Ctx {
        root: workspace_root(),
    };
    println!("platform matrix (wgpu abstracts Metal/Vulkan/DX12/WebGPU):");
    for t in TARGETS {
        let mode = if (t.full_build)(&ctx) {
            "full build"
        } else {
            "cargo check (no cross linker on this host)"
        };
        println!(
            "  {:<15} {:<26} -> {:<20} [{}]",
            t.name, t.triple, t.artifact, mode
        );
    }
    println!(
        "  {:<15} {:<26} -> {:<20} [{}]",
        "web-wasm32", "wasm32-unknown-unknown", "combs_wasm.wasm", "cargo xtask web"
    );
    match android_linker() {
        Some(l) => println!("android NDK linker: {l}"),
        None => println!("android NDK linker: NOT FOUND (android-arm64 will be check-only)"),
    }
}

fn cmd_bundle(ctx: &Ctx) -> Result<()> {
    let dist = ctx.root.join("dist");
    std::fs::create_dir_all(&dist)?;

    // Host CLI, both float variants. `combs` is the f16 build (fleet-wide
    // perplexity deltas under 0.2% and faster throughout; diffusion is
    // pinned to f32 internally either way); `combs-f32` ships alongside
    // for full-precision runs.
    {
        let mut cmd = Command::new(cargo());
        cmd.current_dir(&ctx.root)
            .args(["build", "--release", "-p", "combs-cli"]);
        eprintln!("== combs-f32 (host) ==");
        run(cmd)?;
        let f32_bin = ctx.root.join("target/release/combs");
        let f16_bin = build_f16_cli(ctx)?;
        let out = dist.join("host");
        std::fs::create_dir_all(&out)?;
        std::fs::copy(&f16_bin, out.join("combs"))?;
        std::fs::copy(&f32_bin, out.join("combs-f32"))?;
        eprintln!("bundled combs (f16) + combs-f32 -> {}", out.display());
    }

    for t in TARGETS {
        if !(t.full_build)(ctx) {
            eprintln!("skip {} (check-only target)", t.name);
            continue;
        }
        let artifact = build_target(ctx, t, false)?;
        if !artifact.exists() {
            bail!("expected artifact missing: {}", artifact.display());
        }
        let mesh_artifact = artifact.with_file_name(t.mesh_artifact);
        if !mesh_artifact.exists() {
            bail!("expected artifact missing: {}", mesh_artifact.display());
        }
        let out = dist.join(t.name);
        std::fs::create_dir_all(&out)?;
        std::fs::copy(&artifact, out.join(t.artifact))?;
        std::fs::copy(ctx.root.join("include/combs.h"), out.join("combs.h"))?;
        std::fs::copy(&mesh_artifact, out.join(t.mesh_artifact))?;
        std::fs::copy(
            ctx.root.join("combs-mesh-ffi/include/combsmesh.h"),
            out.join("combsmesh.h"),
        )?;
        eprintln!("bundled {} -> {}", t.artifact, out.display());
        eprintln!("bundled {} -> {}", t.mesh_artifact, out.display());
        // Cross-target build trees are several GB each; drop them after the
        // artifact is safely in dist/ (the plain `target/release` host build
        // is unaffected; re-bundling just recompiles).
        let dir = ctx.root.join("target").join(t.triple);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
            eprintln!("cleaned {}", dir.display());
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let ctx = Ctx {
        root: workspace_root(),
    };
    match cli.command {
        XCommand::Build { release } => {
            let mut cmd = Command::new(cargo());
            cmd.current_dir(&ctx.root).args(["build", "--workspace"]);
            if release {
                cmd.arg("--release");
            }
            run(cmd)?;
            // Release builds also produce the f16 CLI variant; `combs`
            // itself stays the f32 build.
            if release {
                let bin = build_f16_cli(&ctx)?;
                eprintln!("f16 variant: {}", bin.display());
            }
            Ok(())
        }
        XCommand::Run { args } => {
            let mut cmd = Command::new(cargo());
            cmd.current_dir(&ctx.root)
                .args(["run", "--release", "-p", "combs-cli", "--"])
                .args(args);
            run(cmd)
        }
        XCommand::Target { name, check } => {
            let target = TARGETS
                .iter()
                .find(|t| t.name == name)
                .with_context(|| format!("unknown target {name:?}; see `cargo xtask matrix`"))?;
            let artifact = build_target(&ctx, target, check)?;
            if artifact.exists() {
                eprintln!("artifact: {}", artifact.display());
            }
            Ok(())
        }
        XCommand::Gates { quick, strict } => cmd_gates(&ctx, quick, strict),
        XCommand::Web { debug, out, f16 } => cmd_web(&ctx, !debug, out, f16),
        XCommand::Matrix => {
            cmd_matrix();
            Ok(())
        }
        XCommand::Bundle => cmd_bundle(&ctx),
    }
}

#[cfg(test)]
mod build_support {
    /// combs-cli's build script stamps times with a committed copy of
    /// combs-core's formatter (a packaged tarball cannot include across
    /// crates). One implementation stays the law: this fails the
    /// workspace tests the moment the copy drifts by a byte.
    #[test]
    fn the_cli_build_stamp_formatter_matches_combs_core() {
        let original = include_str!("../../combs-core/src/timefmt.rs");
        let copy = include_str!("../../combs-cli/build/timefmt.rs");
        assert_eq!(
            original, copy,
            "combs-cli/build/timefmt.rs has drifted from combs-core/src/timefmt.rs — re-copy it"
        );
    }
}
