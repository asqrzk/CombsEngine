//! The build's own account of itself, compiled in by `build.rs`.
//!
//! Exists because an investigation that has to ASSUME which binary ran,
//! with which features, from which commit, is an investigation resting
//! on nothing. Every long-running command prints [`summary`] at startup
//! and serves [`manifest`] from its stats route, so any measurement can
//! be traced back to the artifact that produced it.

use serde_json::{Value, json};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const COMMIT: &str = env!("COMBS_BUILD_COMMIT");
pub const COMMIT_SHORT: &str = env!("COMBS_BUILD_COMMIT_SHORT");
pub const BRANCH: &str = env!("COMBS_BUILD_BRANCH");
pub const DIRTY: &str = env!("COMBS_BUILD_DIRTY");
pub const PROFILE: &str = env!("COMBS_BUILD_PROFILE");
pub const TARGET: &str = env!("COMBS_BUILD_TARGET");
pub const RUSTC: &str = env!("COMBS_BUILD_RUSTC");
pub const FEATURES: &str = env!("COMBS_BUILD_FEATURES");
pub const OPT_LEVEL: &str = env!("COMBS_BUILD_OPT_LEVEL");
pub const DEBUG: &str = env!("COMBS_BUILD_DEBUG");
pub const BUILT_AT: &str = env!("COMBS_BUILD_AT");
pub const BUILT_AT_RFC3339: &str = env!("COMBS_BUILD_AT_RFC3339");

/// Which float width the engine serves at — the single fact most often
/// assumed and least often checked (an f16 twin looks identical from
/// the outside and fails in ways that look like model bugs).
pub const SERVING_DTYPE: &str = if cfg!(feature = "f16") { "f16" } else { "f32" };

fn features_list() -> Vec<&'static str> {
    FEATURES.split(',').filter(|f| !f.is_empty()).collect()
}

/// The full manifest, for `/v1/stats` and `combs build-info --json`.
pub fn manifest() -> Value {
    json!({
        "name": "combs",
        "version": VERSION,
        "serving_dtype": SERVING_DTYPE,
        "built_at": BUILT_AT_RFC3339,
        "built_at_epoch": BUILT_AT.parse::<u64>().unwrap_or(0),
        "profile": PROFILE,
        "opt_level": OPT_LEVEL,
        "debug": DEBUG,
        "target": TARGET,
        "features": features_list(),
        "rustc": RUSTC,
        "git": {
            "commit": COMMIT,
            "short": COMMIT_SHORT,
            "branch": BRANCH,
            "dirty_as_of_build": DIRTY == "true",
        },
    })
}

/// One line for a startup log: enough to tell two builds apart at a
/// glance, including the two that have cost real time — an f16 twin
/// mistaken for the f32 build, and a stale binary mistaken for a fresh
/// one.
pub fn summary() -> String {
    let dirty = if DIRTY == "true" { "+dirty" } else { "" };
    let features = if FEATURES.is_empty() {
        String::new()
    } else {
        format!(" [{FEATURES}]")
    };
    format!(
        "combs {VERSION} {SERVING_DTYPE} ({PROFILE}, {COMMIT_SHORT}{dirty}, {TARGET}){features}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manifest must describe THIS binary, not a placeholder: if
    /// build.rs stopped running, these would be empty and every record
    /// citing them would be fiction.
    #[test]
    fn manifest_is_populated_by_the_build() {
        assert!(!VERSION.is_empty());
        assert!(!RUSTC.is_empty() && RUSTC != "unknown", "rustc: {RUSTC}");
        assert!(matches!(PROFILE, "debug" | "release"), "profile: {PROFILE}");
        assert!(matches!(DIRTY, "true" | "false"), "dirty: {DIRTY}");
        assert!(BUILT_AT.parse::<u64>().is_ok_and(|t| t > 1_700_000_000));
        assert!(matches!(SERVING_DTYPE, "f16" | "f32"));
        let m = manifest();
        assert_eq!(m["serving_dtype"], SERVING_DTYPE);
        assert!(m["git"]["dirty_as_of_build"].is_boolean());
        assert!(summary().contains(VERSION));
    }

    /// The f16 twin and the f32 build differ in exactly the way that has
    /// caused confusion before; the summary must say which one it is.
    #[test]
    fn summary_names_the_serving_width() {
        assert!(summary().contains(SERVING_DTYPE));
    }
}
