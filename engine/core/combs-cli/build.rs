//! Build manifest: the facts about how this binary was produced,
//! captured BY the build and compiled INTO the binary, so they cannot
//! drift from it the way a hand-written note or a remembered command
//! line can. Every question of the form "which build actually ran, with
//! what turned on?" is answerable from the artifact itself
//! (`combs build-info`), from a running worker's startup log, and from
//! `target/<profile>/build-manifest.json` beside the binary.
//!
//! Deliberately no `cargo:rerun-if-changed` lines: without them cargo
//! re-runs this script whenever any file in the package changes, which
//! keeps the manifest honest for the crate being built. The one limit
//! worth knowing is recorded in the manifest itself
//! (`git.dirty_as_of_build`): the working-tree state is read when this
//! script runs, so a source edit elsewhere in the workspace can leave
//! that flag describing an earlier moment than the compile.

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

fn main() {
    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let short = git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let branch =
        git(&["rev-parse", "--abbrev-ref", "HEAD"]).unwrap_or_else(|| "unknown".into());
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
    println!("cargo:rustc-env=COMBS_BUILD_AT_RFC3339={}", rfc3339(built_at));

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
