//! Which code produced this toolkit.
//!
//! A byte copy of the resolver combs-cli and combs-kit already share, for
//! the same reason they share it: a binary that cannot name its commit
//! cannot be told apart from another build, and `git archive` has no git
//! to ask (the pod road hands the commit in as COMBS_SOURCE_COMMIT).
//!
//! The watched-path list is the part that is easy to get wrong: a build
//! script that watches only itself is NOT re-run when another crate
//! changes, so its stamp goes stale while the binary is rebuilt around it.

use std::process::Command;

// A byte copy of combs-cli/build/stamp.rs; xtask's drift guard fails the
// gate the moment the copies differ.
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
    let mut paths = vec![
        "build.rs".to_string(),
        "build/stamp.rs".to_string(),
        "src".to_string(),
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
    let SourceIdentity { short, dirty, .. } = source_identity(
        git(&["rev-parse", "HEAD"]),
        git(&["rev-parse", "--short", "HEAD"]),
        git(&["rev-parse", "--abbrev-ref", "HEAD"]),
        git(&["status", "--porcelain", "--untracked-files=no"]).map(|s| !s.is_empty()),
        std::env::var("COMBS_SOURCE_COMMIT").ok(),
        std::env::var("COMBS_SOURCE_REF").ok(),
    );
    println!("cargo:rustc-env=COMBS_TK_COMMIT={short}");
    println!("cargo:rustc-env=COMBS_TK_DIRTY={dirty}");
}
