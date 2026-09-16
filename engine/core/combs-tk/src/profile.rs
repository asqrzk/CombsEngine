//! # A profile: one file that reproduces a whole setup.
//!
//! `--profile networky` has to mean the same thing on every desk and on
//! every day, so everything a run needs is in one place a person can
//! copy and edit: which tools the model may call, the roots its reads
//! are bounded by, which dotfiles it may open, the words it works under.
//! The toolkit had no configuration file at all before this — it read
//! three environment variables and nothing else — so this is new
//! surface, built once.
//!
//! A profile is data, and data is refused by name when it is wrong. A
//! profile with no roots would let a read go anywhere; a tool name that
//! is not shaped like a tool name is a typo the model would then spend
//! a turn on. Both are caught at parse, before any seat is taken.
//!
//! What a profile does NOT do: it does not make a tool exist. It names
//! what the model may call; the toolbox decides what is built. A name
//! here that the toolbox has not built is refused at the call, by name,
//! rather than at parse — so a profile can be written ahead of the tools
//! it describes and stay true as they land.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::fence::Policy;

/// One setup, as a person would write it down.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Profile {
    pub name: String,
    #[serde(default)]
    pub about: String,
    /// Directories a read may resolve inside. `~` means the person's
    /// home and is expanded at use, so a profile is the same file on
    /// every machine.
    pub roots: Vec<String>,
    /// Where a relative operand is resolved from. Defaults to the first
    /// root.
    #[serde(default)]
    pub cwd: Option<String>,
    /// Dot-components a read may open, by name. The shell setup files
    /// are the point of the networky profile; nothing else beginning
    /// with a dot is, and history is never here unless a person puts it.
    #[serde(default)]
    pub dotfiles: Vec<String>,
    /// The tools the model may call. Deny by default: a tool not here
    /// is refused by name.
    pub tools: Vec<String>,
    /// The words the model works under.
    #[serde(default)]
    pub system: String,
}

/// The profile shipped in the crate: a person's own machine, seen
/// through read-only probes, many at once.
pub const NETWORKY: &str = r#"{
  "name": "networky",
  "about": "see this machine: its network, its files, its shell setup, through read-only probes fanned out at once",
  "roots": ["~"],
  "cwd": "~",
  "dotfiles": [".zshrc", ".zshenv", ".zprofile", ".bashrc", ".bash_profile", ".profile"],
  "tools": [
    "shell.run", "shell.config",
    "net.interfaces", "net.routes", "net.listeners", "net.neighbors",
    "net.dns", "net.lookup", "net.ping", "net.trace", "net.scan",
    "fs.find", "fs.read", "fs.size",
    "tk.system", "tk.models", "tk.scout", "tk.board"
  ],
  "system": "You are Al. You look at one machine - its network, its files, its shell setup - with read-only probes, many at once, and you say what you found and what you could not determine. You never state a fact a probe could have read for you. Every finding names the probe that found it."
}"#;

/// A tool is `area.verb` — two lowercase words with a dot between them.
/// Anything else is either a typo the model would spend a turn
/// discovering, or a command line someone hoped would be run.
fn is_tool_name(s: &str) -> bool {
    let Some((area, verb)) = s.split_once('.') else {
        return false;
    };
    let word = |part: &str| {
        !part.is_empty()
            && part
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    };
    word(area) && word(verb)
}

/// Read a profile from its text, refusing by name what is wrong.
pub fn parse(text: &str) -> Result<Profile, String> {
    let p: Profile =
        serde_json::from_str(text).map_err(|e| format!("this is not a profile: {e}"))?;
    if p.name.trim().is_empty() {
        return Err("a profile with no name cannot be asked for by name".into());
    }
    // A read bounded by nothing is not bounded, and the roots are the
    // only control this fence has over reading at all.
    if p.roots.is_empty() {
        return Err(format!(
            "the profile {:?} names no roots, and a read bounded by nothing is not bounded",
            p.name
        ));
    }
    if p.tools.is_empty() {
        return Err(format!(
            "the profile {:?} names no tools, so nothing could be called under it",
            p.name
        ));
    }
    for t in &p.tools {
        if !is_tool_name(t) {
            return Err(format!(
                "{t:?} is not a tool name: a tool is area.verb, and a profile names what may be called, never a command line"
            ));
        }
    }
    Ok(p)
}

/// `~` and `~/x` against a given home; anything else untouched. Pure,
/// so a test never reads this machine's home.
pub fn expand_home(s: &str, home: &Path) -> PathBuf {
    if s == "~" {
        return home.to_path_buf();
    }
    if let Some(rest) = s.strip_prefix("~/") {
        return home.join(rest);
    }
    // `~someone` is another person's home. It is left as the literal it
    // is, so the fence refuses it, rather than this quietly resolving
    // to somewhere the profile never named.
    PathBuf::from(s)
}

impl Profile {
    /// May the model call this tool under this profile?
    ///
    /// Deny by default: a tool absent from the list is refused by name
    /// at the call, so a person reads "shell.run is not in this
    /// profile" and not "invalid".
    pub fn allows(&self, tool: &str) -> bool {
        self.tools.iter().any(|t| t == tool)
    }

    /// The fence's ground, with every root made absolute against the
    /// home it was given.
    pub fn policy(&self, home: &Path) -> Policy {
        let roots: Vec<PathBuf> = self.roots.iter().map(|r| expand_home(r, home)).collect();
        let cwd = self
            .cwd
            .as_deref()
            .map(|c| expand_home(c, home))
            // A profile that names no cwd works from its first root,
            // which is the only place it has said it may read.
            .or_else(|| roots.first().cloned())
            .unwrap_or_else(|| home.to_path_buf());
        Policy {
            roots,
            cwd,
            dotfiles: self.dotfiles.clone(),
        }
    }
}

/// Where a profile by that name may live.
///
/// The person's own copy wins over the one shipped in the crate, so
/// editing a profile never means editing the binary. Pure over the home
/// it is given, so a test never reads this machine's.
///
/// A name is a name: anything that could climb, or carry a separator,
/// yields nowhere to look rather than a path somewhere it was never
/// meant to reach.
pub fn search_paths(name: &str, home: &Path) -> Vec<PathBuf> {
    // A name is a name. Anything empty, hidden, or carrying a separator
    // is not one, and yields nowhere to look rather than a path built
    // out of it — the same reason the fence refuses a verb with a slash.
    let plain = !name.is_empty()
        && !name.starts_with('.')
        && !name.contains('/')
        && !name.contains('\\')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.');
    if !plain {
        return Vec::new();
    }
    vec![home.join("tk/profiles").join(format!("{name}.json"))]
}

/// The profile by that name: the person's own if they have written one,
/// else the one shipped in the crate.
pub fn load(name: &str, home: &Path) -> Result<Profile, String> {
    for p in search_paths(name, home) {
        if let Ok(text) = std::fs::read_to_string(&p) {
            return parse(&text).map_err(|e| format!("{}: {e}", p.display()));
        }
    }
    if name == "networky" {
        return parse(NETWORKY);
    }
    Err(format!(
        "no profile named {name:?}; the toolkit ships `networky`, and a person's own live under tk/profiles"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A profile is looked for under the same home as the runs and the
    /// models, so a person has one directory to look in.
    #[test]
    fn a_persons_own_profile_is_looked_for_under_the_same_home() {
        let paths = search_paths("networky", Path::new("/home/someone/.cache/combs"));
        assert_eq!(
            paths,
            vec![PathBuf::from(
                "/home/someone/.cache/combs/tk/profiles/networky.json"
            )]
        );
    }

    /// A name that could climb is not a name, and yields nowhere to look
    /// rather than somewhere it was never meant to reach.
    #[test]
    fn a_profile_name_that_could_climb_finds_nowhere() {
        let home = Path::new("/home/someone/.cache/combs");
        for bad in ["../../etc/passwd", "net/work", "", ".", ".."] {
            assert!(
                search_paths(bad, home).is_empty(),
                "{bad:?} should find nowhere to look"
            );
        }
    }

    /// With no file of the person's own, the shipped profile answers;
    /// a name nobody ships is refused naming what was asked for.
    #[test]
    fn the_shipped_profile_answers_when_the_person_has_written_none() {
        let p = load("networky", Path::new("/nowhere/at/all")).expect("the shipped one");
        assert_eq!(p.name, "networky");
        let e = load("nosuch", Path::new("/nowhere/at/all")).unwrap_err();
        assert!(
            e.contains("nosuch"),
            "the refusal must name what was asked for: {e}"
        );
    }

    /// The shipped profile must parse, or nothing ships.
    #[test]
    fn the_shipped_networky_profile_parses_and_names_its_tools() {
        let p = parse(NETWORKY).expect("the shipped profile parses");
        assert_eq!(p.name, "networky");
        assert!(p.allows("shell.run"), "the shell is the point of it");
        assert!(p.allows("net.routes"));
        assert!(!p.allows("tk.measure"), "the device is never handed to a model by default");
        assert!(!p.allows("rm"), "a bare verb is not a tool name");
    }

    /// The name the driver gave the agent lives in exactly one place,
    /// and this is it.
    #[test]
    fn the_shipped_profile_names_al() {
        let p = parse(NETWORKY).expect("the shipped profile parses");
        assert!(
            p.system.starts_with("You are Al."),
            "the agent is named Al, and the profile is where the name lives: {:?}",
            p.system
        );
    }

    /// No roots means a read may go anywhere, which is not a profile, it
    /// is the absence of one.
    #[test]
    fn a_profile_without_roots_is_refused_by_name() {
        let e = parse(r#"{"name":"x","roots":[],"tools":["shell.run"]}"#).unwrap_err();
        assert!(e.contains("roots"), "the refusal must say what is missing: {e}");
    }

    /// A tool is `area.verb`. Anything else is a typo the model would
    /// spend a turn on, or a shell line someone hoped would run.
    #[test]
    fn a_tool_name_that_is_not_shaped_like_one_is_refused_by_name() {
        let e = parse(r#"{"name":"x","roots":["~"],"tools":["rm -rf /"]}"#).unwrap_err();
        assert!(e.contains("rm -rf /"), "the refusal must name the offender: {e}");
        let e = parse(r#"{"name":"x","roots":["~"],"tools":["shell"]}"#).unwrap_err();
        assert!(e.contains("shell"), "a name with no area is not a tool: {e}");
    }

    /// `~` is the person's home wherever the profile is read.
    #[test]
    fn a_tilde_expands_against_the_home_it_was_given() {
        let home = Path::new("/home/someone");
        assert_eq!(expand_home("~", home), PathBuf::from("/home/someone"));
        assert_eq!(expand_home("~/Projects", home), PathBuf::from("/home/someone/Projects"));
        assert_eq!(expand_home("/srv/data", home), PathBuf::from("/srv/data"));
        assert_eq!(
            expand_home("~someone/x", home),
            PathBuf::from("~someone/x"),
            "another person's home is not expanded, it is left as a literal that the fence will refuse"
        );
    }

    /// The fence gets absolute roots and the dotfiles the profile named
    /// — nothing it did not.
    #[test]
    fn the_policy_carries_expanded_roots_and_the_named_dotfiles() {
        let p = parse(NETWORKY).expect("the shipped profile parses");
        let pol = p.policy(Path::new("/home/someone"));
        assert_eq!(pol.roots, vec![PathBuf::from("/home/someone")]);
        assert_eq!(pol.cwd, PathBuf::from("/home/someone"));
        assert!(pol.dotfiles.contains(&".zshenv".to_string()));
        assert!(
            !pol.dotfiles.contains(&".ssh".to_string()),
            "keys are never opened by a profile that did not name them"
        );
    }

    /// A profile that names no cwd works from its first root.
    #[test]
    fn the_cwd_defaults_to_the_first_root() {
        let p = parse(r#"{"name":"x","roots":["~/work","~/other"],"tools":["shell.run"]}"#)
            .expect("parses");
        let pol = p.policy(Path::new("/home/someone"));
        assert_eq!(pol.cwd, PathBuf::from("/home/someone/work"));
    }
}
