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
pub const SERVING_DTYPE: &str = combs_core::BACKEND.dtype;

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

/// The same facts a stats route should carry: which binary, and how
/// this surface is configured. Assembled here because this crate owns
/// JSON; `combs_core::provenance` owns the emitting.
pub fn stats(role: &str, config: &[(&str, String)]) -> Value {
    let mut cfg = serde_json::Map::new();
    for (k, v) in config {
        cfg.insert((*k).to_string(), Value::String(v.clone()));
    }
    json!({ "role": role, "build": manifest(), "config": Value::Object(cfg) })
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
    // The build TIME is in here because a commit is not an identity:
    // two builds from the same dirty tree are otherwise indistinguishable
    // in a log, and an A/B that unknowingly compared a fresh binary with
    // a stale one would look exactly like a real finding.
    format!(
        "combs {VERSION} {SERVING_DTYPE} ({PROFILE}, {COMMIT_SHORT}{dirty}, \
         built {BUILT_AT_RFC3339}, {TARGET}){features}"
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
        // A build's identity must include when it was built.
        assert!(summary().contains(BUILT_AT_RFC3339), "{}", summary());
    }

    /// The f16 twin and the f32 build differ in exactly the way that has
    /// caused confusion before; the summary must say which one it is.
    #[test]
    fn summary_names_the_serving_width() {
        assert!(summary().contains(SERVING_DTYPE));
    }
}

/// The identity a build stamps comes from one resolver shared with
/// build.rs: git where the source is a checkout, the packed stamp where it
/// is an archive (a pod builds from `git archive`), "unknown" otherwise.
#[cfg(test)]
mod source_identity_tests {
    include!("../build/stamp.rs");

    fn s(v: &str) -> Option<String> {
        Some(v.to_string())
    }

    #[test]
    fn a_checkout_is_named_by_git() {
        let id = source_identity(
            s("7f91a84aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            s("7f91a84"),
            s("w/qwen35"),
            Some(true),
            s("0123456789abcdef0123456789abcdef01234567"),
            s("elsewhere"),
        );
        assert_eq!(id.commit, "7f91a84aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        assert_eq!(id.short, "7f91a84");
        assert_eq!(id.branch, "w/qwen35");
        assert!(id.dirty);
    }

    #[test]
    fn an_archive_without_git_is_named_by_its_stamp() {
        let id = source_identity(
            None,
            None,
            None,
            None,
            s("7F91A84C0FFEE1234567890ABCDEF1234567890A"),
            s("w/qwen35"),
        );
        assert_eq!(id.commit, "7f91a84c0ffee1234567890abcdef1234567890a");
        assert_eq!(id.short, "7f91a84");
        assert_eq!(id.branch, "w/qwen35");
        assert!(!id.dirty, "an archive of a commit is that commit, unmodified");
    }

    #[test]
    fn a_stamp_that_is_not_a_commit_is_refused() {
        for bad in ["", "not-a-sha", "7f91a8", "$(rm -rf /)", "7f91a84c0ffee1234567890abcdef1234567890a0"] {
            let id = source_identity(None, None, None, None, s(bad), s("w/qwen35"));
            assert_eq!(id.commit, "unknown", "{bad:?}");
            assert_eq!(id.short, "unknown", "{bad:?}");
        }
    }

    #[test]
    fn nothing_to_go_on_says_unknown() {
        let id = source_identity(None, None, None, None, None, None);
        assert_eq!(id.commit, "unknown");
        assert_eq!(id.short, "unknown");
        assert_eq!(id.branch, "unknown");
        assert!(!id.dirty);
    }
}
