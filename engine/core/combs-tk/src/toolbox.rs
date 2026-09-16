//! # The toolbox: capabilities, not command names.
//!
//! ## Why a ladder rather than a command
//!
//! `ip` and `ss` are absent from this desk; `ifconfig` and `netstat` are
//! absent or deprecated on a modern Linux. A tool that names one of them
//! names a MACHINE rather than a capability, and stops working the first
//! time it is carried anywhere. So each capability carries an ordered
//! list of implementations, the first present one answers, and the
//! record says WHICH answered — because the routes read by `netstat -rn`
//! and by `ip route` are one fact reported two ways, and a reader needs
//! to know which way it was read before comparing two records.
//!
//! When no rung is present the refusal names the capability and every
//! implementation it looked for. "not found" sends a person to the
//! source; naming what was sought sends them to their package manager.
//!
//! ## The unit belongs to the flag, not to the tool
//!
//! Measured on this desk from the tools' own manual pages: `ping -W` is
//! in MILLISECONDS while `ping -i` is in SECONDS — in the same command —
//! and `traceroute -w` is in SECONDS. A ladder that passed one timeout
//! figure to both would be wrong by a thousand locally, before any
//! second platform is involved. So the figure is written beside the flag
//! that carries it, per platform, and never computed from a single
//! "timeout" the caller supplies.
//!
//! ## What this does not do
//!
//! It chooses and composes; it does not run. What it hands back is a
//! line for the fence to judge, and the fence refuses it like any other.
//! A capability here is never a way around that.

/// One way to answer a capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rung {
    /// The bare verb, resolved against the fence's own fixed PATH.
    pub verb: &'static str,
    /// The words that go with it, units already correct for this
    /// platform.
    pub args: &'static [&'static str],
}

/// A capability, and every way this estate knows to answer it.
#[derive(Debug, Clone, Copy)]
pub struct Capability {
    /// `area.verb`, the same shape a profile names.
    pub name: &'static str,
    /// What this reads, in the words the model is offered. A tool with
    /// no description is a name the model must guess at, and a guessed
    /// tool is a wasted round.
    pub about: &'static str,
    /// Whether answering it can change anything. Everything here is a
    /// reading; the field exists so that stops being an assumption.
    pub read_only: bool,
    pub rungs: &'static [Rung],
    /// The one argument this capability needs from the CALL rather than
    /// from its rung: a host to ping, a name to look up.
    ///
    /// It is named here and nowhere else, because a second list of which
    /// tools need a host would be a second truth — and two truths drift
    /// the first time one of them is updated alone.
    ///
    /// The operand is appended and the WHOLE line is then judged by the
    /// fence, exactly as a typed line is. So a host carrying a semicolon
    /// is refused at the character gate rather than composed into a
    /// second command: a capability is not a way around the fence, and
    /// an argument arriving from a model is not trusted for being one.
    pub takes: Option<&'static str>,
}

/// The rung that will answer, and the line it becomes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chosen {
    pub tool: String,
    /// Which implementation answered — the half a record cannot omit.
    pub verb: String,
    /// The line, for the fence to judge like any other.
    pub line: String,
}

/// How long to wait for a ping reply, in the unit `-W` actually means
/// here. Milliseconds on this desk, seconds on Linux — read from the
/// manual pages rather than assumed, because the two differ by a
/// thousand and the flag is spelled the same on both.
#[cfg(target_os = "macos")]
pub const PING_WAIT: &str = "500";
#[cfg(not(target_os = "macos"))]
pub const PING_WAIT: &str = "1";

/// Every capability the toolbox can answer, each with its ladder.
pub const CAPABILITIES: &[Capability] = &[
    Capability {
        name: "net.interfaces",
        about: "every network interface on this machine, with its addresses, flags and MTU",
        read_only: true,
        rungs: &[
            Rung { verb: "ifconfig", args: &["-a"] },
            Rung { verb: "ip", args: &["-o", "addr"] },
        ],
        takes: None,
    },
    Capability {
        name: "net.routes",
        about: "the routing table: which interface carries the default route, and every route beside it",
        read_only: true,
        rungs: &[
            Rung { verb: "netstat", args: &["-rn"] },
            Rung { verb: "ip", args: &["route"] },
        ],
        takes: None,
    },
    Capability {
        name: "net.listeners",
        about: "every TCP port this machine is listening on, and which process holds each one",
        read_only: true,
        rungs: &[
            Rung { verb: "lsof", args: &["-nP", "-iTCP", "-sTCP:LISTEN"] },
            Rung { verb: "ss", args: &["-ltn"] },
        ],
        takes: None,
    },
    Capability {
        name: "net.neighbors",
        about: "the neighbour table: the machines this one has recently spoken to on its local network",
        read_only: true,
        rungs: &[
            Rung { verb: "arp", args: &["-an"] },
            Rung { verb: "ip", args: &["neigh"] },
        ],
        takes: None,
    },
    Capability {
        name: "net.dns",
        about: "the DNS resolvers this machine uses, per scope",
        read_only: true,
        rungs: &[
            Rung { verb: "scutil", args: &["--dns"] },
            Rung { verb: "resolvectl", args: &["status"] },
        ],
        takes: None,
    },
    Capability {
        name: "net.trace",
        about: "the hops between this machine and a host you name",
        read_only: true,
        rungs: &[Rung { verb: "traceroute", args: &["-n", "-w", "1", "-q", "1", "-m", "12"] }],
        takes: Some("host"),
    },
    // The count is bounded at the rung, not left to the caller: an
    // unbounded probe is a timeout rather than an answer, and the fence
    // asks about a ping that carries no `-c` for exactly that reason.
    // The wait figure beside `-W` is the unit that flag actually means
    // here — milliseconds on this desk, seconds elsewhere.
    Capability {
        name: "net.ping",
        about: "whether a host you name answers, and how long one round trip took",
        read_only: true,
        rungs: &[Rung { verb: "ping", args: &["-c", "1", "-W", PING_WAIT] }],
        takes: Some("host"),
    },
    Capability {
        name: "net.lookup",
        about: "the addresses a name resolves to",
        read_only: true,
        rungs: &[
            Rung { verb: "dig", args: &["+short"] },
            Rung { verb: "host", args: &[] },
        ],
        takes: Some("host"),
    },
];

/// The capability by that name, if the toolbox knows it.
pub fn find(name: &str) -> Option<&'static Capability> {
    CAPABILITIES.iter().find(|c| c.name == name)
}

/// Choose the rung that will answer, or say why none can.
///
/// `present` is passed in rather than read, so the rule is testable on a
/// machine that has the opposite set of tools installed to this one —
/// which is the whole reason a ladder exists.
pub fn resolve(name: &str, present: impl Fn(&str) -> bool) -> Result<Chosen, String> {
    let Some(cap) = find(name) else {
        // Not knowing a capability and not being able to answer one are
        // different failures, and a person can only act on the second.
        return Err(format!(
            "{name} is not a capability this toolbox knows"
        ));
    };
    for r in cap.rungs {
        if present(r.verb) {
            let mut line = String::from(r.verb);
            for a in r.args {
                line.push(' ');
                line.push_str(a);
            }
            return Ok(Chosen {
                tool: cap.name.to_string(),
                verb: r.verb.to_string(),
                line,
            });
        }
    }
    // Name every rung that was looked for. "not found" sends a person to
    // the source; this sends them to their package manager.
    let looked: Vec<&str> = cap.rungs.iter().map(|r| r.verb).collect();
    Err(format!(
        "{name} cannot be answered on this machine: none of {} is installed",
        looked.join(", ")
    ))
}

/// Is this capability shaped the way the rest of the estate expects?
///
/// A function rather than a loop inside a test, because a check written
/// inline over a table someone hand-wrote correct cannot fail, and a
/// check that cannot fail is worth less than no check. This one is given
/// deliberately broken capabilities in its own tests before it is
/// believed about the real ones — the same way the dependency guard
/// proves itself against a manifest that smuggles a crate in.
pub fn well_formed(c: &Capability) -> Result<(), String> {
    let Some((area, verb)) = c.name.split_once('.') else {
        return Err(format!("{} is not area.verb", c.name));
    };
    if area.is_empty() || verb.is_empty() {
        return Err(format!("{} is not area.verb", c.name));
    }
    if !c.read_only {
        return Err(format!("{} is in the toolbox but is not a reading", c.name));
    }
    if c.rungs.is_empty() {
        return Err(format!("{} is a ladder with no rungs", c.name));
    }
    // A tool with no words is a name the model must guess at, and a
    // guessed tool is a round spent finding out what it was.
    if c.about.trim().is_empty() {
        return Err(format!("{} does not say what it reads", c.name));
    }
    for r in c.rungs {
        // A verb carrying a path names a binary of someone's choosing;
        // the fence refuses that, and a ladder must not emit one.
        if r.verb.contains('/') {
            return Err(format!("{} names a binary by path: {}", c.name, r.verb));
        }
        for a in r.args {
            // The fence tokenises on whitespace, so a word with a space
            // inside it is two words wearing one coat.
            if a.contains(' ') {
                return Err(format!("{} carries a word with a space in it: {a:?}", c.name));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The negative control. Before the walk below is believed about the
    /// shipped table, the check must be shown to REJECT each way a
    /// capability can be wrong.
    #[test]
    fn the_shape_check_catches_a_capability_that_is_wrong() {
        const SLASHED: &[Rung] = &[Rung { verb: "/bin/ls", args: &[] }];
        const SPACED: &[Rung] = &[Rung { verb: "ls", args: &["-l /etc"] }];
        const FINE: &[Rung] = &[Rung { verb: "ls", args: &["-l"] }];

        let cases: [(Capability, &str); 5] = [
            (Capability { name: "noarea", about: "reads a thing", read_only: true, rungs: FINE, takes: None }, "area.verb"),
            (Capability { name: "a.b", about: "reads a thing", read_only: false, rungs: FINE, takes: None }, "not a reading"),
            (Capability { name: "a.b", about: "reads a thing", read_only: true, rungs: &[], takes: None }, "no rungs"),
            (Capability { name: "a.b", about: "reads a thing", read_only: true, rungs: SLASHED, takes: None }, "by path"),
            (Capability { name: "a.b", about: "", read_only: true, rungs: FINE, takes: None }, "does not say what it reads"),
        ];
        for (c, want) in cases {
            let e = well_formed(&c)
                .expect_err(&format!("{:?} should not be well formed", c.name));
            assert!(e.contains(want), "the refusal must say {want:?}: {e}");
        }
        let spaced = Capability { name: "a.b", about: "reads a thing", read_only: true, rungs: SPACED, takes: None };
        assert!(
            well_formed(&spaced).is_err(),
            "a word with a space in it is two words wearing one coat"
        );
        // And the positive control: a sound one passes, so the check is
        // not simply refusing everything.
        let good = Capability { name: "net.things", about: "reads a thing", read_only: true, rungs: FINE, takes: None };
        assert_eq!(well_formed(&good), Ok(()));
    }

    /// Only these exist on this desk; `ip`, `ss` and `resolvectl` do
    /// not. Measured, not assumed.
    fn bsd(verb: &str) -> bool {
        ["ifconfig", "netstat", "lsof", "arp", "scutil", "ping", "traceroute"].contains(&verb)
    }

    /// The opposite machine: a modern Linux with none of the BSD set.
    fn gnu(verb: &str) -> bool {
        ["ip", "ss", "resolvectl", "ping", "traceroute"].contains(&verb)
    }

    /// The same capability is answered by different implementations on
    /// different machines, and the record says which answered.
    #[test]
    fn the_first_present_rung_answers_and_is_named() {
        let here = resolve("net.routes", bsd).expect("this desk can read its routes");
        assert_eq!(here.verb, "netstat");
        assert_eq!(here.line, "netstat -rn");

        let there = resolve("net.routes", gnu).expect("a linux box can read its routes");
        assert_eq!(there.verb, "ip");
        assert_eq!(there.line, "ip route");
    }

    /// A ladder with nothing present refuses by name, and names every
    /// implementation it looked for — "not found" sends a person to the
    /// source; this sends them to their package manager.
    #[test]
    fn a_capability_with_no_implementation_refuses_by_name() {
        let e = resolve("net.routes", |_| false).unwrap_err();
        assert!(e.contains("net.routes"), "the refusal must name the capability: {e}");
        assert!(e.contains("netstat"), "it must name what it looked for: {e}");
        assert!(e.contains("ip"), "it must name every rung it looked for: {e}");
    }

    /// A capability nobody has written is not the same as one nothing
    /// can answer, and the words say which.
    #[test]
    fn a_capability_the_toolbox_does_not_know_is_refused_by_name() {
        let e = resolve("net.telepathy", bsd).unwrap_err();
        assert!(e.contains("net.telepathy"), "the refusal must name it: {e}");
    }

    /// The unit belongs to the flag. `-W` is milliseconds here and
    /// seconds on Linux, so the figure beside it is not the same number
    /// on both — a ladder that shared one would be wrong by a thousand.
    #[test]
    fn the_ping_wait_is_the_unit_its_flag_actually_means() {
        if cfg!(target_os = "macos") {
            assert_eq!(PING_WAIT, "500", "-W is milliseconds on this desk");
        } else {
            assert_eq!(PING_WAIT, "1", "-W is seconds elsewhere");
        }
    }

    /// Every shipped capability passes the check that was just shown to
    /// reject a broken one.
    ///
    /// This was two loops restating the rule inline, which is exactly
    /// what made them unfailable: a check written over a table someone
    /// hand-wrote correct agrees with itself and proves nothing. One
    /// rule, in one place, believed only after it was made to fail.
    #[test]
    fn every_shipped_capability_is_well_formed() {
        for c in CAPABILITIES {
            assert_eq!(well_formed(c), Ok(()), "{} is not well formed", c.name);
        }
    }
}
