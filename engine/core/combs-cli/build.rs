//! Build manifest: the facts about how this binary was produced,
//! captured BY the build and compiled INTO the binary, so they cannot
//! drift from it the way a hand-written note or a remembered command
//! line can. Every question of the form "which build actually ran, with
//! what turned on?" is answerable from the artifact itself
//! (`combs build-info`), from a running worker's startup log, and from
//! `target/<profile>/build-manifest.json` beside the binary.
//!
//! Cargo re-runs a directive-free build script only when a file in ITS
//! OWN package changes, which is not enough: editing another crate in
//! the workspace relinks this binary while leaving the stamp describing
//! the previous one. That was caught in the act — a rebuilt binary
//! still reporting the day before's commit and time — so the script
//! now names what it depends on: every crate's sources, and git's HEAD
//! and index so a commit or a staging change re-stamps too. The
//! declaration lives in [`watched_paths`]; a new crate belongs in it.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// Same formatter the binary logs with — one implementation, so a
// build stamp and a runtime line can never disagree about the time.
include!("../combs-core/src/timefmt.rs");

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn rustc_version() -> String {
    std::env::var("RUSTC")
        .ok()
        .and_then(|rustc| Command::new(rustc).arg("--version").output().ok())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// Seconds since the epoch — a stable, dependency-free timestamp. The
/// manifest carries it raw so no formatting choice can misread it.
fn built_at_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// What this stamp is a statement about. Cargo watches directories
/// recursively, so one entry per crate covers it; the git files make a
/// commit or a staging change re-run the script even when no source
/// moved. Paths that do not exist are emitted anyway — cargo treats a
/// missing watched path as "changed", which errs toward re-stamping.
fn watched_paths() -> Vec<String> {
    let mut paths = vec![
        "build.rs".to_string(),
        "src".to_string(),
        "../vendor".to_string(),
    ];
    let workspace = std::path::Path::new("..");
    if let Ok(entries) = std::fs::read_dir(workspace) {
        let mut crates: Vec<String> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.join("Cargo.toml").is_file())
            .filter_map(|p| p.join("src").to_str().map(str::to_string))
            .collect();
        crates.sort();
        paths.extend(crates);
    }
    for git_path in ["../../../.git/HEAD", "../../../.git/index"] {
        paths.push(git_path.to_string());
    }
    paths
}

fn main() {
    embed_template();

    for path in watched_paths() {
        println!("cargo:rerun-if-changed={path}");
    }
    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let short = git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let branch = git(&["rev-parse", "--abbrev-ref", "HEAD"]).unwrap_or_else(|| "unknown".into());
    // Any tracked modification, staged or not, makes this build
    // unreproducible from the commit alone — the flag says so plainly.
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
        .map(|s| !s.is_empty())
        .unwrap_or(false);

    // Cargo hands the enabled features in as CARGO_FEATURE_<NAME>.
    let mut features: Vec<String> = std::env::vars()
        .filter_map(|(k, _)| k.strip_prefix("CARGO_FEATURE_").map(|f| f.to_lowercase()))
        .collect();
    features.sort();

    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "unknown".into());
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".into());
    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "unknown".into());
    let opt_level = std::env::var("OPT_LEVEL").unwrap_or_else(|_| "unknown".into());
    let debug = std::env::var("DEBUG").unwrap_or_else(|_| "unknown".into());
    let rustc = rustc_version();
    let built_at = built_at_epoch();
    let feature_list = features.join(",");

    for (key, value) in [
        ("COMBS_BUILD_COMMIT", commit.as_str()),
        ("COMBS_BUILD_COMMIT_SHORT", short.as_str()),
        ("COMBS_BUILD_BRANCH", branch.as_str()),
        ("COMBS_BUILD_DIRTY", if dirty { "true" } else { "false" }),
        ("COMBS_BUILD_PROFILE", profile.as_str()),
        ("COMBS_BUILD_TARGET", target.as_str()),
        ("COMBS_BUILD_RUSTC", rustc.as_str()),
        ("COMBS_BUILD_FEATURES", feature_list.as_str()),
        ("COMBS_BUILD_OPT_LEVEL", opt_level.as_str()),
        ("COMBS_BUILD_DEBUG", debug.as_str()),
    ] {
        println!("cargo:rustc-env={key}={value}");
    }
    println!("cargo:rustc-env=COMBS_BUILD_AT={built_at}");
    println!(
        "cargo:rustc-env=COMBS_BUILD_AT_RFC3339={}",
        rfc3339(built_at)
    );

    // A sidecar beside the artifact, for anyone holding the directory
    // rather than the binary (a pod's deploy step, an archived build).
    // OUT_DIR is target/<profile>/build/<pkg>-<hash>/out.
    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        let artifact_dir = std::path::Path::new(&out_dir)
            .ancestors()
            .nth(3)
            .map(|p| p.to_path_buf());
        if let Some(dir) = artifact_dir {
            let json = format!(
                concat!(
                    "{{\n",
                    "  \"name\": \"combs\",\n",
                    "  \"version\": \"{version}\",\n",
                    "  \"built_at_epoch\": {built_at},\n",
                    "  \"built_at\": \"{built_at_rfc}\",\n",
                    "  \"profile\": \"{profile}\",\n",
                    "  \"opt_level\": \"{opt_level}\",\n",
                    "  \"debug\": \"{debug}\",\n",
                    "  \"target\": \"{target}\",\n",
                    "  \"features\": \"{features}\",\n",
                    "  \"rustc\": \"{rustc}\",\n",
                    "  \"git\": {{\n",
                    "    \"commit\": \"{commit}\",\n",
                    "    \"short\": \"{short}\",\n",
                    "    \"branch\": \"{branch}\",\n",
                    "    \"dirty_as_of_build\": {dirty}\n",
                    "  }}\n",
                    "}}\n"
                ),
                version = version,
                built_at = built_at,
                built_at_rfc = rfc3339(built_at),
                profile = profile,
                opt_level = opt_level,
                debug = debug,
                target = target,
                features = feature_list,
                rustc = rustc,
                commit = commit,
                short = short,
                branch = branch,
                dirty = dirty,
            );
            let _ = std::fs::write(dir.join("build-manifest.json"), json);
        }
    }
}

// ---------------------------------------------------------------------
// The UI template embed — the OTHER half of this build script, restored.
// The build-manifest rewrite replaced this file wholesale and dropped
// it; chew.rs still includes $OUT_DIR/template_manifest.rs, so every
// fresh fingerprint (CI clones, a toolchain change) failed to compile
// combs-cli while stale local OUT_DIRs papered over the hole for weeks.
// Both halves now live here and neither may replace the other again.

const SKIP_DIRS: &[&str] = &["node_modules", "dist", ".svelte-kit", ".vite", "data"];
/// Runtime secrets/state the proxy creates — never embed these in a binary.
const SKIP_FILES: &[&str] = &[
    "master.key",
    "permissions.json",
    "manifest.json",
    "authn.json",
    "package-lock.json",
];

fn embed_template() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendored = manifest.join("vendor/ui-template");
    let repo = manifest.join("../../ui/template");
    let root = if vendored.is_dir() { vendored } else { repo };
    assert!(root.is_dir(), "UI template not found at {}", root.display());

    let mut entries: Vec<(String, PathBuf)> = Vec::new();
    collect(&root, &root, &mut entries);
    entries.sort();

    let mut code = String::from(
        "/// (relative path, contents) for every file in the embedded UI template.\n\
         pub static TEMPLATE_FILES: &[(&str, &[u8])] = &[\n",
    );
    for (rel, abs) in &entries {
        println!("cargo:rerun-if-changed={}", abs.display());
        // Forward slashes work in include_bytes! on every platform.
        let abs_fwd = abs.display().to_string().replace('\\', "/");
        code.push_str(&format!("    ({rel:?}, include_bytes!(\"{abs_fwd}\")),\n"));
    }
    code.push_str("];\n");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("template_manifest.rs");
    fs::write(out, code).unwrap();
}

fn collect(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) {
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.file_type().unwrap().is_dir() {
            if !SKIP_DIRS.contains(&name.as_str()) {
                collect(root, &path, out);
            }
        } else if !SKIP_FILES.contains(&name.as_str()) {
            let rel = path
                .strip_prefix(root)
                .unwrap()
                .display()
                .to_string()
                .replace('\\', "/");
            out.push((rel, path));
        }
    }
}
