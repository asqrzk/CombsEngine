//! # The fence: a shell line is refused, not parsed.
//!
//! ## Why this does not tokenize first
//!
//! Eight ways of writing `ls` were run on this desk, and all eight
//! executed it in both zsh and bash: `l""s`, `'l's`, `l\s`, `$'ls'`,
//! `$'\x6cs'`, `${x:-ls}`, `$(echo ls)` and `eval ls`. So a fence that
//! splits a line into words and compares the first against an allow-list
//! is already defeated, eight ways, before `$IFS` is even reached — and
//! unquoting before comparing means reimplementing two shells' quote
//! removal in a crate that may take no parser crate.
//!
//! The inversion: gate the CHARACTER SET first. A line that survives
//! cannot contain any character that changes what a word means, so what
//! is left is not a shell line at all — it is whitespace-separated
//! literal words joined by pipes, and `split_whitespace` on that is not
//! an approximation of a tokenizer but an exact one. The hard case is
//! made unrepresentable rather than parsed.
//!
//! That one rule also moots `$IFS`, env prefixes, brace expansion,
//! globs, process substitution and every redirection shape at once.
//! `eval ls` needs no refused character, which is why a deny-list of
//! verbs stands beside the gate rather than instead of it.
//!
//! ## What this does not do, said plainly
//!
//! It stops writes. The roots are the only control on READS, so a
//! `grep -r SECRET <root>` is allowed by design. It is lexical, so a
//! path that passes containment can become a symlink out of the roots
//! before exec — the caller re-checks at spawn, and that check lives
//! outside this module deliberately, so the pure part stays testable
//! with no fixture tree. It is a fence, not a sandbox.

use std::path::PathBuf;

/// What the fence decided, and in its own words why.
///
/// A refusal names the rule that refused. "invalid" sends a person to
/// the source; naming the character and its position sends them to their
/// own line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Allow,
    /// Safe to run only with a person's consent. At the tool boundary
    /// this becomes a refusal, because there is no person inside a tool
    /// call to consent.
    Ask(String),
    Refuse(String),
}

/// The ground a line is judged against.
#[derive(Debug, Clone)]
pub struct Policy {
    /// Directories a path operand may resolve inside.
    pub roots: Vec<PathBuf>,
    /// Where a relative operand is resolved from.
    pub cwd: PathBuf,
    /// Dot-components allowed by name. Everything else beginning with a
    /// dot asks, because a person's secrets live in dotfiles.
    pub dotfiles: Vec<String>,
}

/// The longest line this will consider. Past it, it is a program.
pub const MAX_LINE: usize = 512;

/// The most stages a pipeline may have.
pub const MAX_STAGES: usize = 3;

/// Everything a surviving line may contain besides letters and digits.
/// Space and `|` join words and stages; the rest can appear inside a
/// path, a host or a flag without changing what any word means.
const ALLOWED_PUNCT: &[char] = &['-', '_', '.', '/', ':', ',', '@', '%', '+'];

/// Verbs that are refused whatever they carry, each with the reason.
/// A shell is here because a shell as an allowed verb is a fence with a
/// door in it; `eval` because it needs no forbidden character at all.
const NEVER: &[(&str, &str)] = &[
    ("sh", "a shell would run a second line this fence never read"),
    ("bash", "a shell would run a second line this fence never read"),
    ("zsh", "a shell would run a second line this fence never read"),
    ("dash", "a shell would run a second line this fence never read"),
    ("ksh", "a shell would run a second line this fence never read"),
    ("fish", "a shell would run a second line this fence never read"),
    ("eval", "eval builds a command out of text after the fence has read it"),
    ("exec", "exec replaces this process with one the fence never judged"),
    ("source", "source runs a file the fence never read"),
    ("sudo", "privilege is not a reading"),
    ("su", "privilege is not a reading"),
    ("doas", "privilege is not a reading"),
    ("rm", "it destroys"),
    ("mv", "it writes"),
    ("cp", "it writes"),
    ("dd", "it writes"),
    ("chmod", "it writes"),
    ("chown", "it writes"),
    ("chflags", "it writes"),
    ("ln", "it writes"),
    ("truncate", "it writes"),
    ("tee", "it writes"),
    ("mkdir", "it writes"),
    ("rmdir", "it writes"),
    ("touch", "it writes"),
    ("sed", "sed -i edits in place, and the flag is one character"),
    ("perl", "perl -i edits in place, and perl runs anything"),
    ("awk", "awk runs a program of its own"),
    ("python", "it runs a program of its own"),
    ("python3", "it runs a program of its own"),
    ("ruby", "it runs a program of its own"),
    ("node", "it runs a program of its own"),
    ("curl", "it reaches the network"),
    ("wget", "it reaches the network"),
    ("ssh", "it reaches the network"),
    ("scp", "it reaches the network"),
    ("ftp", "it reaches the network"),
    ("telnet", "it reaches the network"),
    ("nc", "a socket in both directions is not a reading"),
    ("route", "route takes a subcommand, and the routing table is writable through it"),
    ("security", "it reads and writes the keychain"),
    ("kill", "process control is not a reading"),
    ("killall", "process control is not a reading"),
    ("launchctl", "process control is not a reading"),
    ("pmset", "process control is not a reading"),
    ("shutdown", "process control is not a reading"),
    ("reboot", "process control is not a reading"),
];

/// What a verb's bare operands are allowed to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operand {
    /// None at all.
    Nothing,
    /// Paths, bounded by the policy's roots.
    Paths,
    /// A hostname or an address literal.
    Hosts,
    /// An interface name, which is letters then digits — so `down`,
    /// `up`, `create` and `destroy` are not interface names.
    Iface,
    /// Free words: a pattern, a preference name, a service name.
    Names,
    /// A pattern first, then paths.
    PatternThenPaths,
}

/// One allowed verb: which dash-tokens it may carry, which of those take
/// a value, how many bare operands it may have and of what kind, and any
/// flag it must carry to be allowed at all.
struct Verb {
    name: &'static str,
    flags: &'static [&'static str],
    valued: &'static [&'static str],
    operands: Operand,
    max_operands: usize,
    /// A flag without which the verb only asks. `ping` without a bounded
    /// count is a timeout, not an answer.
    requires: Option<&'static str>,
    /// `head -5` is a real form, and a bare number is not a path.
    numeric_flag: bool,
}

const VERBS: &[Verb] = &[
    Verb { name: "netstat", flags: &["-a", "-n", "-an", "-r", "-rn", "-i", "-b", "-v", "-s", "-p", "-f", "-l", "-na"], valued: &["-p", "-f"], operands: Operand::Nothing, max_operands: 0, requires: None, numeric_flag: false },
    Verb { name: "lsof", flags: &["-n", "-P", "-nP", "-i", "-iTCP", "-iUDP", "-sTCP:LISTEN", "-c", "-p", "-t"], valued: &["-c", "-p"], operands: Operand::Nothing, max_operands: 0, requires: None, numeric_flag: false },
    Verb { name: "ifconfig", flags: &["-a", "-l", "-u", "-d", "-m", "-v", "-L"], valued: &[], operands: Operand::Iface, max_operands: 1, requires: None, numeric_flag: false },
    Verb { name: "arp", flags: &["-a", "-n", "-an", "-l", "-i"], valued: &["-i"], operands: Operand::Hosts, max_operands: 1, requires: None, numeric_flag: false },
    Verb { name: "scutil", flags: &["--dns", "--proxy", "--nwi", "-r", "-W"], valued: &[], operands: Operand::Hosts, max_operands: 2, requires: None, numeric_flag: false },
    Verb { name: "dscacheutil", flags: &["-q", "-a", "-configuration", "-statistics", "-cachedump", "-buckets", "-entries"], valued: &["-q", "-a"], operands: Operand::Names, max_operands: 2, requires: None, numeric_flag: false },
    Verb { name: "networksetup", flags: &["-listallhardwareports", "-listallnetworkservices", "-listnetworkserviceorder", "-getinfo", "-getdnsservers", "-getcurrentlocation"], valued: &[], operands: Operand::Names, max_operands: 1, requires: None, numeric_flag: false },
    Verb { name: "ping", flags: &["-c", "-W", "-t", "-n", "-q", "-i"], valued: &["-c", "-W", "-t", "-i"], operands: Operand::Hosts, max_operands: 1, requires: Some("-c"), numeric_flag: false },
    Verb { name: "traceroute", flags: &["-n", "-w", "-q", "-m"], valued: &["-w", "-q", "-m"], operands: Operand::Hosts, max_operands: 1, requires: None, numeric_flag: false },
    Verb { name: "dig", flags: &["+short", "+noall", "+answer", "-t", "-x", "-4", "-6"], valued: &["-t", "-x"], operands: Operand::Hosts, max_operands: 2, requires: None, numeric_flag: false },
    Verb { name: "host", flags: &["-t", "-a"], valued: &["-t"], operands: Operand::Hosts, max_operands: 2, requires: None, numeric_flag: false },
    Verb { name: "grep", flags: &["-i", "-n", "-r", "-R", "-c", "-l", "-v", "-E", "-F", "-w", "-A", "-B", "-C", "-e", "-h"], valued: &["-A", "-B", "-C", "-e"], operands: Operand::PatternThenPaths, max_operands: 8, requires: None, numeric_flag: false },
    Verb { name: "rg", flags: &["-i", "-n", "-l", "-w", "-F", "-e", "-t", "-g", "--files", "--json", "--no-heading"], valued: &["-e", "-t", "-g"], operands: Operand::PatternThenPaths, max_operands: 8, requires: None, numeric_flag: false },
    Verb { name: "find", flags: &["-name", "-iname", "-type", "-maxdepth", "-mindepth", "-path", "-size", "-newer"], valued: &["-name", "-iname", "-type", "-maxdepth", "-mindepth", "-path", "-size", "-newer"], operands: Operand::Paths, max_operands: 1, requires: None, numeric_flag: false },
    Verb { name: "ls", flags: &["-l", "-a", "-la", "-al", "-h", "-lh", "-t", "-r", "-R", "-1", "-A"], valued: &[], operands: Operand::Paths, max_operands: 4, requires: None, numeric_flag: false },
    Verb { name: "cat", flags: &["-n", "-b"], valued: &[], operands: Operand::Paths, max_operands: 4, requires: None, numeric_flag: false },
    Verb { name: "head", flags: &["-n", "-c"], valued: &["-n", "-c"], operands: Operand::Paths, max_operands: 4, requires: None, numeric_flag: true },
    Verb { name: "tail", flags: &["-n", "-c"], valued: &["-n", "-c"], operands: Operand::Paths, max_operands: 4, requires: None, numeric_flag: true },
    Verb { name: "wc", flags: &["-l", "-w", "-c"], valued: &[], operands: Operand::Paths, max_operands: 4, requires: None, numeric_flag: false },
    Verb { name: "uname", flags: &["-a", "-s", "-r", "-m", "-v"], valued: &[], operands: Operand::Nothing, max_operands: 0, requires: None, numeric_flag: false },
    Verb { name: "hostname", flags: &[], valued: &[], operands: Operand::Nothing, max_operands: 0, requires: None, numeric_flag: false },
    Verb { name: "id", flags: &["-u", "-g", "-n"], valued: &[], operands: Operand::Nothing, max_operands: 0, requires: None, numeric_flag: false },
    Verb { name: "whoami", flags: &[], valued: &[], operands: Operand::Nothing, max_operands: 0, requires: None, numeric_flag: false },
    Verb { name: "uptime", flags: &[], valued: &[], operands: Operand::Nothing, max_operands: 0, requires: None, numeric_flag: false },
    Verb { name: "sw_vers", flags: &["-productVersion", "-buildVersion", "-productName"], valued: &[], operands: Operand::Nothing, max_operands: 0, requires: None, numeric_flag: false },
];

fn is_flag(tok: &str) -> bool {
    (tok.starts_with('-') || tok.starts_with('+')) && tok.len() > 1
}

fn looks_numeric(tok: &str) -> bool {
    !tok.is_empty() && tok.chars().all(|c| c.is_ascii_digit())
}

fn looks_like_host(tok: &str) -> bool {
    !tok.is_empty()
        && tok
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == ':')
}

fn looks_like_iface(tok: &str) -> bool {
    let letters = tok.chars().take_while(|c| c.is_ascii_lowercase()).count();
    let digits = tok.chars().skip(letters).count();
    letters > 0 && digits > 0 && tok.chars().skip(letters).all(|c| c.is_ascii_digit())
}

/// Judge one path operand lexically. Deliberately lexical: resolving
/// symlinks here would need a fixture tree and would still be stale by
/// the time anything ran, so the caller re-checks at spawn.
fn judge_path(tok: &str, policy: &Policy) -> Verdict {
    if tok.split('/').any(|c| c == "..") {
        return Verdict::Refuse(format!(
            "{tok} climbs out of its root with `..`; name the path you mean"
        ));
    }
    let full = if tok.starts_with('/') {
        PathBuf::from(tok)
    } else {
        policy.cwd.join(tok)
    };
    if !policy.roots.iter().any(|r| full.starts_with(r)) {
        let roots: Vec<String> = policy
            .roots
            .iter()
            .map(|r| r.display().to_string())
            .collect();
        return Verdict::Refuse(format!(
            "{tok} is outside the roots this run may read: {}",
            roots.join(", ")
        ));
    }
    for part in full.components() {
        let name = part.as_os_str().to_string_lossy();
        if name.starts_with('.') && name.len() > 1 && !policy.dotfiles.contains(&name.to_string()) {
            return Verdict::Ask(format!(
                "{tok} reaches into {name}, and a person's secrets live in dotfiles"
            ));
        }
    }
    Verdict::Allow
}

/// Judge one stage of a pipeline: a verb and its words.
fn judge_stage(words: &[&str], policy: &Policy) -> Verdict {
    let verb = words[0];
    if verb.contains('/') {
        return Verdict::Refuse(format!(
            "{verb} names a binary by path; a verb here is a bare name resolved against a fixed PATH"
        ));
    }
    if let Some((_, why)) = NEVER.iter().find(|(n, _)| *n == verb) {
        return Verdict::Refuse(format!("{verb} is never allowed: {why}"));
    }
    let Some(spec) = VERBS.iter().find(|v| v.name == verb) else {
        return Verdict::Ask(format!("{verb} is not on the read-only list"));
    };

    let mut asks: Option<String> = None;
    let mut operands = 0usize;
    let mut seen_flags: Vec<&str> = Vec::new();
    let mut i = 1;
    while i < words.len() {
        let tok = words[i];
        if is_flag(tok) {
            if spec.numeric_flag && looks_numeric(&tok[1..]) {
                i += 1;
                continue;
            }
            if !spec.flags.contains(&tok) {
                return Verdict::Refuse(format!(
                    "{verb} may not carry {tok} here: its flags are an allow-list, because a deny-list stops only what someone already thought of"
                ));
            }
            seen_flags.push(tok);
            if spec.valued.contains(&tok) {
                i += 2;
                continue;
            }
            i += 1;
            continue;
        }
        operands += 1;
        if operands > spec.max_operands {
            return Verdict::Refuse(format!(
                "{verb} takes at most {} operand(s) here, and {tok} is one too many",
                spec.max_operands
            ));
        }
        let v = match spec.operands {
            Operand::Nothing => Verdict::Refuse(format!("{verb} takes no operands here, and {tok} is one")),
            Operand::Paths => judge_path(tok, policy),
            Operand::PatternThenPaths => {
                if operands == 1 {
                    Verdict::Allow
                } else {
                    judge_path(tok, policy)
                }
            }
            Operand::Hosts => {
                if looks_like_host(tok) {
                    Verdict::Allow
                } else {
                    Verdict::Refuse(format!("{tok} is not a hostname or an address"))
                }
            }
            Operand::Iface => {
                if looks_like_iface(tok) {
                    Verdict::Allow
                } else {
                    Verdict::Refuse(format!(
                        "{tok} is not an interface name, and {verb}'s parameters are not readings"
                    ))
                }
            }
            Operand::Names => Verdict::Allow,
        };
        match v {
            Verdict::Refuse(why) => return Verdict::Refuse(why),
            Verdict::Ask(why) => {
                asks.get_or_insert(why);
            }
            Verdict::Allow => {}
        }
        i += 1;
    }

    if let Some(must) = spec.requires {
        if !seen_flags.contains(&must) {
            return Verdict::Ask(format!(
                "{verb} without {must} is unbounded, and an unbounded probe is a timeout rather than an answer"
            ));
        }
    }
    match asks {
        Some(why) => Verdict::Ask(why),
        None => Verdict::Allow,
    }
}

/// Judge one shell line.
pub fn decide_shell(line: &str, policy: &Policy) -> Verdict {
    if line.len() > MAX_LINE {
        return Verdict::Refuse(format!(
            "the line is {} characters; past {MAX_LINE} it is a program, not a line",
            line.len()
        ));
    }
    // The character gate, first and before any splitting. Everything a
    // shell could reinterpret is gone before a word exists.
    for (at, c) in line.char_indices() {
        let ok = c.is_ascii_alphanumeric() || c == ' ' || c == '|' || ALLOWED_PUNCT.contains(&c);
        if !ok {
            let named = if c.is_control() {
                format!("a control character (U+{:04X})", c as u32)
            } else if !c.is_ascii() {
                format!("{c:?} (U+{:04X}), outside printable ASCII", c as u32)
            } else {
                format!("{c:?}")
            };
            return Verdict::Refuse(format!(
                "{named} at byte {at} changes what the words around it mean; this fence refuses shell syntax rather than parsing it"
            ));
        }
    }

    let stages: Vec<&str> = line.split('|').collect();
    if stages.len() > MAX_STAGES {
        return Verdict::Refuse(format!(
            "{} stages; at most {MAX_STAGES} are allowed",
            stages.len()
        ));
    }
    let mut asks: Option<String> = None;
    for stage in &stages {
        let words: Vec<&str> = stage.split_whitespace().collect();
        if words.is_empty() {
            return Verdict::Refuse("a pipeline stage is empty".into());
        }
        // Any refusal anywhere outranks every ask anywhere, so a line
        // that both leaves its roots and destroys refuses on the
        // destroying.
        match judge_stage(&words, policy) {
            Verdict::Refuse(why) => return Verdict::Refuse(why),
            Verdict::Ask(why) => {
                asks.get_or_insert(why);
            }
            Verdict::Allow => {}
        }
    }
    match asks {
        Some(why) => Verdict::Ask(why),
        None => Verdict::Allow,
    }
}

/// The argv that actually runs an approved line.
///
/// Half the fence lives here: a fence whose caller spawns `zsh -c` has
/// no fence at all. Measured on this desk — `zsh -c` sources `~/.zshenv`
/// and `zsh -f -c` does not, and `.zshenv` can define a FUNCTION named
/// `ls`, at which point an allow-list allow-lists nothing.
pub fn shell_argv(line: &str) -> Vec<String> {
    // Each approved verb is run as a binary. Measured: `zsh -f -c`
    // still lists two compiled-in aliases, and zsh expands aliases at
    // PARSE time for the whole `-c` string — so `unalias -a` prepended
    // to that same string cannot protect the line it rides on. Proven
    // from the other side too: an alias defined earlier in one `-c`
    // string never intercepted a later word. `command` does.
    let fenced = line
        .split('|')
        .map(|stage| {
            let s = stage.trim();
            if s.is_empty() {
                String::new()
            } else {
                format!("command {s}")
            }
        })
        .collect::<Vec<_>>()
        .join(" | ");
    vec!["zsh".into(), "-f".into(), "-c".into(), fenced]
}

/// The environment an approved line runs in: these names and no others.
///
/// Measured, and it corrects the obvious recipe: `bash -c` honours
/// `BASH_ENV`, and `bash --noprofile --norc -c` STILL sourced it. Flags
/// are not the defence; a cleared environment is. The doors are absent
/// by construction rather than by removal, which is why this builds a
/// list instead of filtering one.
pub fn fenced_env() -> Vec<(String, String)> {
    vec![
        ("PATH".into(), "/usr/bin:/bin:/usr/sbin:/sbin".into()),
        ("HOME".into(), std::env::var("HOME").unwrap_or_default()),
        // A fixed locale, so a reading does not change shape with the
        // caller's language and become unparseable to its own ladder.
        ("LANG".into(), "C".into()),
        ("TZ".into(), std::env::var("TZ").unwrap_or_else(|_| "UTC".into())),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> Policy {
        Policy {
            roots: vec![PathBuf::from("/tmp/work")],
            cwd: PathBuf::from("/tmp/work"),
            dotfiles: vec![],
        }
    }

    fn refused(line: &str) -> String {
        match decide_shell(line, &policy()) {
            Verdict::Refuse(why) => why,
            other => panic!("{line:?} was not refused: {other:?}"),
        }
    }

    /// The measurement this whole design rests on. Every one of these
    /// ran `ls` in both zsh and bash on this desk, so every one must
    /// refuse — seven on a character the gate forbids, and `eval ls` on
    /// the verb, since it needs no forbidden character at all.
    #[test]
    fn every_quoting_form_that_runs_ls_is_refused() {
        for form in [
            "l\"\"s",
            "'l's",
            "l\\s",
            "$'ls'",
            "$'\\x6cs'",
            "${x:-ls}",
            "$(echo ls)",
            "eval ls",
        ] {
            let why = refused(form);
            assert!(
                !why.is_empty(),
                "{form:?} refused without saying why"
            );
        }
    }

    /// Chaining turns one approved verb into two, the second unread.
    #[test]
    fn a_line_that_chains_is_refused() {
        for line in ["ls; rm -rf /", "ls && rm x", "ls || rm x", "ls &"] {
            refused(line);
        }
    }

    /// A newline is a chain that does not look like one.
    #[test]
    fn a_newline_is_refused() {
        refused("ls\nrm -rf /");
        refused("ls\rrm -rf /");
    }

    /// Redirection writes without a write verb.
    #[test]
    fn redirection_is_refused() {
        for line in ["ls > /tmp/work/x", "ls >> /tmp/work/x", "grep a < /tmp/work/x"] {
            refused(line);
        }
    }

    /// An env prefix rewrites the meaning of every word after it, `IFS`
    /// most of all.
    #[test]
    fn an_env_prefix_is_refused() {
        refused("IFS=x ls");
        refused("LD_PRELOAD=/tmp/x ls");
    }

    /// A glob is the filesystem writing the argument list.
    #[test]
    fn a_glob_or_a_home_shorthand_is_refused() {
        refused("ls *");
        refused("ls ?");
        refused("ls ~");
        refused("ls [abc]");
    }

    /// Arguments are free text, and a zero-width character inside a path
    /// is invisible in the prompt a person reads. A path a person cannot
    /// see is not a path this runs.
    #[test]
    fn anything_outside_printable_ascii_is_refused() {
        refused("ls /tmp/work/\u{200B}secret");
        refused("ls /tmp/work/café");
    }

    /// A shell as an allowed verb is a fence with a door in it, and
    /// `zsh -fc ls` needs no refused character at all.
    #[test]
    fn a_shell_is_never_a_verb() {
        for line in ["zsh -fc ls", "bash -c ls", "sh -c ls", "dash -c ls"] {
            refused(line);
        }
    }

    /// A verb is a bare name resolved against a fixed PATH. Anything
    /// carrying a slash names a binary of the caller's choosing.
    #[test]
    fn a_verb_carrying_a_path_is_refused() {
        refused("./ls");
        refused("/bin/ls");
        refused("../ls");
    }

    /// Past this it is a program, not a line.
    #[test]
    fn a_line_past_the_cap_is_refused() {
        let long = format!("ls {}", "a".repeat(MAX_LINE));
        refused(&long);
    }

    /// Pipes are allowed because `lsof | grep` is the shape of every
    /// real answer — but bounded, and never into a shell.
    #[test]
    fn a_pipeline_is_bounded_and_never_empty() {
        refused("netstat -an | grep LISTEN | head -5 | grep x");
        refused("netstat -an | | grep LISTEN");
        refused("netstat -an |");
        refused("netstat -an | sh");
    }

    /// The verbs that destroy, whatever flags they carry.
    #[test]
    fn a_verb_that_writes_is_refused() {
        for verb in [
            "rm x", "mv a b", "cp a b", "dd if=a", "chmod 777 x", "chown me x",
            "ln -s a b", "truncate -s 0 x", "tee x", "mkdir x", "sudo ls", "su",
        ] {
            refused(verb);
        }
    }

    /// The sharpest rule here, because every one of these verbs reads as
    /// read-only until you look: measured on this desk, `scutil --set`
    /// writes a preference, `ifconfig` documents `down`, `dscacheutil`
    /// carries `-flushcache`, BSD `find` has four write-or-exec
    /// primaries, `arp -d` deletes an entry, and `rg --pre` runs a
    /// command of its own. Flags are an allow-list for exactly this
    /// reason: a deny-list stops only what someone already thought of.
    #[test]
    fn a_read_only_verb_carrying_a_write_flag_is_refused() {
        for line in [
            "scutil --set HostName x",
            "ifconfig en0 down",
            "dscacheutil -flushcache",
            "find /tmp/work -delete",
            "find /tmp/work -exec rm {} +",
            "arp -d 10.0.0.1",
            "rg --pre=curl pattern /tmp/work",
            "rg --hostname-bin=curl pattern /tmp/work",
            "sed -i s/a/b/ /tmp/work/x",
            "networksetup -setairportpower en0 off",
        ] {
            refused(line);
        }
    }

    /// Reaching the network is how a fenced line stops being local.
    #[test]
    fn egress_is_refused() {
        for line in ["curl http://x", "ssh host", "scp a b", "nc -l 9000"] {
            refused(line);
        }
    }

    /// Credentials and process control are not readings.
    #[test]
    fn credentials_and_process_control_are_refused() {
        for line in ["security find-generic-password", "kill 1", "killall combs", "launchctl list"] {
            refused(line);
        }
    }

    /// A path that climbs out refuses outright: sharper than
    /// normalising, and it makes containment a prefix test.
    #[test]
    fn a_path_that_climbs_or_leaves_the_roots_is_refused() {
        refused("grep a /tmp/work/../etc/passwd");
        refused("grep a /etc/passwd");
    }

    /// Precedence is fixed: any refusal anywhere beats every ask
    /// anywhere, so a line that both leaves the roots and destroys
    /// refuses on the destroying.
    #[test]
    fn a_refusal_anywhere_beats_an_ask_anywhere() {
        let v = decide_shell("rm /etc/passwd", &policy());
        assert!(
            matches!(v, Verdict::Refuse(_)),
            "a refusal must outrank an ask: {v:?}"
        );
    }

    /// A verb nobody listed is not a verb nobody may run — it is one
    /// only a person may approve.
    #[test]
    fn a_verb_that_is_merely_unknown_asks() {
        match decide_shell("vmstat", &policy()) {
            Verdict::Ask(why) => assert!(!why.is_empty(), "an ask must say what it is asking about"),
            other => panic!("an unknown verb should ask, not {other:?}"),
        }
    }

    /// An unbounded probe is a timeout, not an answer.
    #[test]
    fn a_ping_without_a_bound_asks() {
        match decide_shell("ping 1.1.1.1", &policy()) {
            Verdict::Ask(_) => {}
            other => panic!("an unbounded ping should ask, not {other:?}"),
        }
        assert_eq!(
            decide_shell("ping -c 3 1.1.1.1", &policy()),
            Verdict::Allow,
            "a bounded ping is the ordinary reading"
        );
    }

    /// The composition the fence exists to permit: two read-only verbs
    /// with a pipe between them. Refusing this would push a person
    /// toward blanket consent for ordinary readings, which is how
    /// consent stops meaning anything.
    #[test]
    fn a_pipe_between_two_read_only_verbs_is_allowed() {
        assert_eq!(
            decide_shell("netstat -an | grep LISTEN", &policy()),
            Verdict::Allow
        );
    }

    /// The spawn recipe is half the fence. `zsh -c` sources `~/.zshenv`
    /// and `-f` is what stops it; an interactive shell would load
    /// `.zshrc` and its aliases; and zsh expands aliases at PARSE time
    /// for the whole `-c` string, so `unalias -a` prepended to that same
    /// string cannot protect the line it rides on. `command` can, and
    /// does.
    #[test]
    fn the_argv_never_lets_a_startup_file_run() {
        let argv = shell_argv("netstat -an");
        assert!(
            argv.iter().any(|a| a == "-f"),
            "the recipe must skip the startup files: {argv:?}"
        );
        assert!(
            !argv.iter().any(|a| a.contains('i')  && a.starts_with('-')),
            "an interactive shell loads the aliases the fence exists to defeat: {argv:?}"
        );
        let line = argv.last().expect("a line to run");
        assert!(
            line.starts_with("command "),
            "each approved verb must be run as a binary, not whatever an alias made of it: {line:?}"
        );
    }

    /// The environment carries these names and no others. A loader
    /// variable or a shell door in the child is the fence's own
    /// bypass — and the one that actually fooled me was `BASH_ENV`,
    /// which `--noprofile --norc` does not stop.
    #[test]
    fn the_environment_carries_no_loader_and_no_shell_door() {
        let env = fenced_env();
        let keys: Vec<&str> = env.iter().map(|(k, _)| k.as_str()).collect();
        for door in [
            "BASH_ENV", "ENV", "ZDOTDIR", "RIPGREP_CONFIG_PATH", "IFS",
            "DYLD_INSERT_LIBRARIES", "LD_PRELOAD", "PYTHONSTARTUP",
        ] {
            assert!(
                !keys.contains(&door),
                "{door} reaches the child: {keys:?}"
            );
        }
        let mut names: Vec<&str> = keys.clone();
        names.sort_unstable();
        assert_eq!(
            names,
            vec!["HOME", "LANG", "PATH", "TZ"],
            "the environment must carry exactly the names the fence allows"
        );
    }
}
