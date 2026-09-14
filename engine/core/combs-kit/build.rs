//! Which code produced a number.
//!
//! The binary's own modification time answers "is this stale" and
//! nothing else: a binary can be newer than the sources and still have
//! been built from a different branch, or from a tree with uncommitted
//! work in it. A record that cannot name the code behind it cannot be
//! checked by a stranger, which is the whole point of the record.
//!
//! The watched-path list is the part that matters and the part the
//! first attempt got wrong: a build script that watches only itself is
//! NOT re-run when another crate changes, so its stamp goes stale while
//! the binary is rebuilt around it. This watches every crate's sources
//! and git's own files, the same way the CLI's stamp does.

use std::process::Command;

// Where the source's identity comes from when there is no git to ask — a
// packed archive names its commit through COMBS_SOURCE_COMMIT. A byte copy
// of combs-cli/build/stamp.rs; xtask's `the_kit_source_resolver_matches_combs_cli`
// fails the gate's xtask tier the moment the two differ.
include!("build/stamp.rs");

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    Some(text.trim().to_string())
}

fn watched_paths() -> Vec<String> {
    let mut paths = vec!["build.rs".to_string(), "build/stamp.rs".to_string(), "src".to_string()];
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
    // A commit or a staging change re-stamps even when no source moved.
    for git_path in ["../../../.git/HEAD", "../../../.git/index"] {
        paths.push(git_path.to_string());
    }
    paths
}

fn main() {
    for path in watched_paths() {
        println!("cargo:rerun-if-changed={path}");
    }
    println!("cargo:rerun-if-env-changed=COMBS_SOURCE_COMMIT");
    println!("cargo:rerun-if-env-changed=COMBS_SOURCE_REF");
    // Any tracked modification, staged or not, makes this build
    // unreproducible from the commit alone — the flag says so plainly,
    // and every number it takes carries the flag. A packed archive has no
    // git; whoever builds it hands the commit in as COMBS_SOURCE_COMMIT.
    let SourceIdentity { short, dirty, .. } = source_identity(
        git(&["rev-parse", "HEAD"]),
        git(&["rev-parse", "--short", "HEAD"]),
        git(&["rev-parse", "--abbrev-ref", "HEAD"]),
        git(&["status", "--porcelain", "--untracked-files=no"]).map(|s| !s.is_empty()),
        std::env::var("COMBS_SOURCE_COMMIT").ok(),
        std::env::var("COMBS_SOURCE_REF").ok(),
    );
    println!("cargo:rustc-env=COMBS_KIT_COMMIT={short}");
    println!("cargo:rustc-env=COMBS_KIT_DIRTY={dirty}");
}
