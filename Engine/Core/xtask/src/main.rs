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
//! | web-wasm32      | wasm32-unknown-unknown    | combs_wasm.wasm       | Phase 5 (combs-wasm crate) |
//!
//! `xtask target <name>` runs a full build when the toolchain is available,
//! otherwise a `cargo check` (type/borrow verification without linking).
//! `xtask bundle` assembles `dist/<target>/` with the library + combs.h.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, bail};
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
            PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join("Library/Android/sdk")
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
        full_build: |_| true,
        env: |_| vec![],
    },
    Target {
        name: "macos-x86_64",
        triple: "x86_64-apple-darwin",
        artifact: "libcombs_ffi.dylib",
        full_build: |_| true,
        env: |_| vec![],
    },
    Target {
        name: "ios-arm64",
        triple: "aarch64-apple-ios",
        artifact: "libcombs_ffi.a",
        full_build: |_| true,
        env: |_| vec![],
    },
    Target {
        name: "android-arm64",
        triple: "aarch64-linux-android",
        artifact: "libcombs_ffi.so",
        full_build: |_| android_linker().is_some(),
        env: |_| {
            android_linker()
                .map(|l| {
                    vec![(
                        "CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER".to_string(),
                        l,
                    )]
                })
                .unwrap_or_default()
        },
    },
    Target {
        name: "windows-x86_64",
        triple: "x86_64-pc-windows-msvc",
        artifact: "combs_ffi.dll",
        // No MSVC linker on this host: check only.
        full_build: |_| false,
        env: |_| vec![],
    },
    Target {
        name: "linux-x86_64",
        triple: "x86_64-unknown-linux-gnu",
        artifact: "libcombs_ffi.so",
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

fn build_target(ctx: &Ctx, target: &Target, force_check: bool) -> Result<PathBuf> {
    let full = !force_check && (target.full_build)(ctx);
    let mut cmd = Command::new(cargo());
    cmd.current_dir(&ctx.root);
    if full {
        cmd.args(["build", "--release"]);
    } else {
        cmd.args(["check", "--release"]);
    }
    cmd.args(["-p", "combs-ffi", "--target", target.triple]);
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
        println!("  {:<15} {:<26} -> {:<20} [{}]", t.name, t.triple, t.artifact, mode);
    }
    println!("  {:<15} {:<26} -> {:<20} [{}]", "web-wasm32", "wasm32-unknown-unknown", "combs_wasm.wasm", "Phase 5 (combs-wasm crate)");
    match android_linker() {
        Some(l) => println!("android NDK linker: {l}"),
        None => println!("android NDK linker: NOT FOUND (android-arm64 will be check-only)"),
    }
}

fn cmd_bundle(ctx: &Ctx) -> Result<()> {
    let dist = ctx.root.join("dist");
    std::fs::create_dir_all(&dist)?;
    for t in TARGETS {
        if !(t.full_build)(ctx) {
            eprintln!("skip {} (check-only target)", t.name);
            continue;
        }
        let artifact = build_target(ctx, t, false)?;
        if !artifact.exists() {
            bail!("expected artifact missing: {}", artifact.display());
        }
        let out = dist.join(t.name);
        std::fs::create_dir_all(&out)?;
        std::fs::copy(&artifact, out.join(t.artifact))?;
        std::fs::copy(ctx.root.join("include/combs.h"), out.join("combs.h"))?;
        eprintln!("bundled {} -> {}", t.artifact, out.display());
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
            run(cmd)
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
        XCommand::Matrix => {
            cmd_matrix();
            Ok(())
        }
        XCommand::Bundle => cmd_bundle(&ctx),
    }
}
