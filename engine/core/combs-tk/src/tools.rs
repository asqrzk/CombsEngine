//! Finding the binaries this toolkit drives, and refusing to guess.
//!
//! The ladder is the app's, deliberately: `COMBS_BIN` (or `COMBS_KIT_BIN`)
//! first so an operator can always name the binary, then the cache the
//! installer writes, then the usual install roots, then PATH. A toolkit
//! that searched a different set from the app would answer questions
//! about a different engine than the one the person is running.
//!
//! And a binary nobody can name is refused rather than driven. `combs`
//! carries its own identity — version, dtype, commit, dirty — and every
//! number or status this toolkit prints is about a specific build. The
//! app learned this the expensive way: for six days its sidecars ran an
//! engine a week older than the tree while every report looked current.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Which tool: they share a ladder but not a name, and each has its own
/// override so a person can drive a freshly built pair without installing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    Engine,
    Kit,
}

impl Tool {
    pub fn binary(self) -> &'static str {
        match self {
            Tool::Engine => "combs",
            Tool::Kit => "combs-kit",
        }
    }

    pub fn env_var(self) -> &'static str {
        match self {
            Tool::Engine => "COMBS_BIN",
            Tool::Kit => "COMBS_KIT_BIN",
        }
    }

    /// What to tell someone who does not have it. Never "not found" alone
    /// — the next command is the useful half.
    pub fn how_to_get_it(self) -> &'static str {
        match self {
            Tool::Engine => {
                "build it with `cd engine/core && cargo build --release -p combs-cli`, \
                 install it with `cargo xtask install-bin`, or name it with COMBS_BIN"
            }
            Tool::Kit => {
                "build it with `cd engine/core && cargo build --release -p combs-kit`, \
                 or name it with COMBS_KIT_BIN"
            }
        }
    }
}

/// Every place a binary may live, in the order the app searches.
pub fn candidates(tool: Tool, home: Option<&Path>, target_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(t) = target_dir {
        out.push(t.join(tool.binary()));
    }
    if let Some(h) = home {
        out.push(h.join(".cache/combs/bin").join(tool.binary()));
        out.push(h.join(".cargo/bin").join(tool.binary()));
        out.push(h.join(".local/bin").join(tool.binary()));
    }
    out.push(PathBuf::from("/opt/homebrew/bin").join(tool.binary()));
    out.push(PathBuf::from("/usr/local/bin").join(tool.binary()));
    out
}

/// Resolve a tool to a path, or say where it looked.
pub fn find(tool: Tool) -> Result<PathBuf, String> {
    if let Ok(p) = std::env::var(tool.env_var()) {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
        return Err(format!(
            "{} names {}, which is not a file",
            tool.env_var(),
            p.display()
        ));
    }
    let home = std::env::var("HOME").ok().map(PathBuf::from);
    // A developer tree: the binary beside the sources beats an installed
    // one, because someone who just built is asking about what they built.
    let target = std::env::current_dir()
        .ok()
        .map(|d| d.join("target/release"))
        .filter(|p| p.is_dir());
    for cand in candidates(tool, home.as_deref(), target.as_deref()) {
        if cand.is_file() {
            return Ok(cand);
        }
    }
    if let Ok(out) = Command::new("which").arg(tool.binary()).output() {
        if out.status.success() {
            let p = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
            if p.is_file() {
                return Ok(p);
            }
        }
    }
    Err(format!(
        "no `{}` on this machine — {}",
        tool.binary(),
        tool.how_to_get_it()
    ))
}

/// What a binary says about itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Build {
    pub version: String,
    pub dtype: String,
    pub commit: String,
    pub dirty: bool,
}

impl Build {
    pub fn line(&self) -> String {
        format!(
            "combs {} · {} · {}{}",
            self.version,
            self.dtype,
            self.commit,
            if self.dirty { " (dirty)" } else { "" }
        )
    }
}

/// Parse `combs build-info --json`. Pure, so the shapes are tested
/// without a binary to run — including the shapes that must be REFUSED:
/// a build nobody can name is not a build this toolkit will report on.
pub fn parse_build(text: &str) -> Result<Build, String> {
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("build-info is not JSON: {e}"))?;
    let version = v
        .get("version")
        .and_then(|x| x.as_str())
        .ok_or("build-info names no version")?
        .to_string();
    let dtype = v
        .get("serving_dtype")
        .and_then(|x| x.as_str())
        .ok_or("build-info names no serving dtype")?
        .to_string();
    let git = v.get("git").ok_or("build-info carries no git block")?;
    let commit = git
        .get("short")
        .and_then(|x| x.as_str())
        .ok_or("build-info names no commit")?
        .to_string();
    let dirty = git
        .get("dirty_as_of_build")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    Ok(Build { version, dtype, commit, dirty })
}

/// Ask a binary who it is. A tool that cannot answer is not driven.
pub fn build_of(bin: &Path) -> Result<Build, String> {
    let out = Command::new(bin)
        .args(["build-info", "--json"])
        .output()
        .map_err(|e| format!("{} could not be run: {e}", bin.display()))?;
    if !out.status.success() {
        return Err(format!(
            "{} did not answer `build-info --json` — it may not be a combs binary",
            bin.display()
        ));
    }
    parse_build(&String::from_utf8_lossy(&out.stdout))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ladder_prefers_an_explicit_target_then_the_cache() {
        let home = PathBuf::from("/home/someone");
        let target = PathBuf::from("/work/target/release");
        let c = candidates(Tool::Engine, Some(&home), Some(&target));
        assert_eq!(c[0], target.join("combs"));
        assert_eq!(c[1], home.join(".cache/combs/bin/combs"));
        assert!(c.iter().any(|p| p.ends_with("usr/local/bin/combs")));
    }

    #[test]
    fn the_kit_has_its_own_name_and_override() {
        assert_eq!(Tool::Kit.binary(), "combs-kit");
        assert_eq!(Tool::Kit.env_var(), "COMBS_KIT_BIN");
        let c = candidates(Tool::Kit, None, None);
        assert!(c.iter().all(|p| p.ends_with("combs-kit")));
    }

    #[test]
    fn a_build_names_its_commit_and_dtype() {
        let b = parse_build(
            r#"{"name":"combs","version":"0.3.0","serving_dtype":"f16",
                "git":{"short":"ab89d50","dirty_as_of_build":false}}"#,
        )
        .expect("a well-formed manifest parses");
        assert_eq!(b.commit, "ab89d50");
        assert_eq!(b.dtype, "f16");
        assert!(!b.dirty);
        assert!(b.line().contains("ab89d50"));
    }

    /// A dirty build is reported as dirty. Its numbers cannot be
    /// reproduced from the commit alone, and a reader must be told.
    #[test]
    fn a_dirty_build_says_so() {
        let b = parse_build(
            r#"{"version":"0.3.0","serving_dtype":"f32",
                "git":{"short":"deadbee","dirty_as_of_build":true}}"#,
        )
        .unwrap();
        assert!(b.dirty);
        assert!(b.line().contains("dirty"));
    }

    /// The refusals: a binary that cannot name itself is not driven, and
    /// the message says which part was missing rather than "invalid".
    #[test]
    fn a_build_nobody_can_name_is_refused_by_name() {
        assert!(parse_build("not json at all").unwrap_err().contains("not JSON"));
        assert!(parse_build(r#"{"serving_dtype":"f16","git":{"short":"a"}}"#)
            .unwrap_err()
            .contains("version"));
        assert!(parse_build(r#"{"version":"0.3.0","git":{"short":"a"}}"#)
            .unwrap_err()
            .contains("dtype"));
        assert!(parse_build(r#"{"version":"0.3.0","serving_dtype":"f16"}"#)
            .unwrap_err()
            .contains("git"));
    }

    /// Never "not found" alone: the next command is the useful half.
    #[test]
    fn a_missing_tool_says_how_to_get_it() {
        for t in [Tool::Engine, Tool::Kit] {
            assert!(t.how_to_get_it().contains("cargo build"));
            assert!(t.how_to_get_it().contains(t.env_var()));
        }
    }
}
