//! What this toolkit started, and how it knows a process is still its own.
//!
//! The rule here is written in scar tissue from elsewhere in this estate:
//! **never sweep by command line.** A `pkill -f` matching on arguments
//! once executed the very process doing the matching, and a sweep keyed
//! on a name will happily kill a colleague's engine, a probe's server, or
//! a second person's session on a shared box.
//!
//! So a run is remembered as a file: the pid, the port, the model and the
//! commit of the binary that was started. A process counts as ours only
//! when the record exists AND the pid is still alive AND the port it
//! claimed still answers. Any of those failing makes it stale, and a
//! stale record is reported as stale rather than acted on — the toolkit
//! never signals a pid it did not itself record.

use std::path::{Path, PathBuf};
use std::process::Child;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// One engine this toolkit started.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Run {
    /// What it serves: `text`, `images`, `audio`.
    pub role: String,
    pub pid: u32,
    pub port: u16,
    /// The model as the person names it, never an absolute path — a
    /// record that carries someone's home directory cannot be shared.
    /// Always written through [`identity`].
    pub model: String,
    /// The binary's commit, so a status line can say which engine is up.
    pub commit: String,
    pub dtype: String,
    /// Where this run's output is being written.
    pub log: PathBuf,
    /// Unix seconds. Passed in rather than read, so the caller owns the
    /// clock and the tests are not at the mercy of one.
    pub started: u64,
}

/// A model named the way a person names it, never the way a filesystem
/// does. A packed seat is a directory holding one `model.gguf`, so the
/// DIRECTORY is the name; anything else is its own final component.
///
/// The rule is combs-kit's `model::identity`, deliberately: a record or
/// a status line that carries someone's home directory cannot be shared,
/// pasted into a report, or compared against the same seat on another
/// machine — and it blows out every column it appears in.
pub fn identity(path: &str) -> String {
    let p = std::path::Path::new(path);
    let named = if p.extension().is_some_and(|x| x == "gguf") {
        p.parent().and_then(|d| d.file_name())
    } else {
        p.file_name()
    };
    named
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Where runs are remembered. Under the same root the cache uses, so a
/// person has one directory to look in and one to clear.
pub fn root() -> Result<PathBuf, String> {
    let home = std::env::var("COMBS_HOME")
        .map(PathBuf::from)
        .or_else(|_| {
            std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .map(|h| PathBuf::from(h).join(".cache/combs"))
        })
        .map_err(|_| "cannot locate a home directory (set COMBS_HOME)".to_string())?;
    Ok(home.join("tk/runs"))
}

pub fn save(dir: &Path, run: &Run) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    let path = dir.join(format!("{}.json", run.role));
    let text = serde_json::to_string_pretty(run).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| format!("writing {}: {e}", path.display()))
}

pub fn load_all(dir: &Path) -> Vec<Run> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<Run> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .filter_map(|t| serde_json::from_str(&t).ok())
        .collect();
    out.sort_by(|a, b| a.role.cmp(&b.role));
    out
}

pub fn forget(dir: &Path, role: &str) {
    let _ = std::fs::remove_file(dir.join(format!("{role}.json")));
}

/// What a recorded run actually is right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Liveness {
    /// Recorded, alive, and answering on its port.
    Up,
    /// The process is alive but the port does not answer — still loading,
    /// or wedged. Never treated as down: killing a loading engine because
    /// it has not bound yet is how a slow seat becomes a crash loop.
    Starting,
    /// The record outlived the process. Nothing to signal.
    Stale,
}

/// Judge a recorded run from two independent readings. Pure, so the rule
/// is testable without processes or sockets: a record alone is never
/// enough, and a port alone is never enough either — another program may
/// hold the port we once used.
pub fn liveness(pid_alive: bool, port_answers: bool) -> Liveness {
    match (pid_alive, port_answers) {
        (true, true) => Liveness::Up,
        (true, false) => Liveness::Starting,
        (false, _) => Liveness::Stale,
    }
}

/// Is this pid still running? Signal 0 asks without delivering anything.
#[cfg(unix)]
pub fn pid_alive(pid: u32) -> bool {
    // SAFETY: kill with signal 0 performs error checking only and
    // delivers nothing; it cannot affect the target process.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

#[cfg(not(unix))]
pub fn pid_alive(_pid: u32) -> bool {
    false
}

/// Does something answer on this port? A connect, not a request: the
/// engine binds only after the model is loaded, so a refused connection
/// means "not ready yet", and an accepted one means the listener is up.
pub fn port_answers(port: u16) -> bool {
    use std::net::{Ipv4Addr, SocketAddr, TcpStream};
    use std::time::Duration;
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

/// A free port, chosen by the OS. Asking for zero and reading back what
/// was bound is the only way that does not race a scan-then-bind gap.
pub fn free_port() -> Result<u16, String> {
    use std::net::{Ipv4Addr, TcpListener};
    let l = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).map_err(|e| e.to_string())?;
    l.local_addr().map(|a| a.port()).map_err(|e| e.to_string())
}

/// What a just-started engine is doing.
#[derive(Debug, PartialEq, Eq)]
pub enum Start {
    /// The port answers: it is serving.
    Ready,
    /// It ended before it ever bound, carrying its exit code.
    Died(Option<i32>),
    /// Still loading when the wait ran out. The engine is LEFT RUNNING:
    /// a large seat takes a while, and killing one for being slow is how
    /// a slow seat becomes a crash loop.
    Waiting,
}

/// Wait for a just-spawned engine to answer, and say what happened.
///
/// The rule that matters here is WHICH PROCESS IS ASKED, and how. For a
/// process this toolkit did not spawn, [`pid_alive`] is the honest
/// reading. For our own child it is not: a child that has exited and has
/// not been reaped is a zombie, and `kill(pid, 0)` calls a zombie alive
/// until somebody waits on it. So the child is asked through `try_wait`,
/// which reads the truth and reaps it in one move.
pub fn wait_for_start(
    child: &mut Child,
    port: u16,
    deadline: Duration,
    mut tick: impl FnMut(Duration),
) -> Start {
    let began = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Start::Died(status.code()),
            // Still running — or the wait itself failed, which is not
            // evidence of death, so it falls through to the port and the
            // deadline rather than declaring an engine dead on a guess.
            Ok(None) | Err(_) => {}
        }
        if port_answers(port) {
            return Start::Ready;
        }
        if began.elapsed() >= deadline {
            return Start::Waiting;
        }
        tick(began.elapsed());
        std::thread::sleep(Duration::from_millis(400));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(role: &str, port: u16) -> Run {
        Run {
            role: role.into(),
            pid: 4242,
            port,
            model: "qwen3-0.6b-gguf".into(),
            commit: "ab89d50".into(),
            dtype: "f16".into(),
            log: PathBuf::from("/tmp/x.log"),
            started: 1_700_000_000,
        }
    }

    /// A status line and a saved record must never carry the operator's
    /// home directory: it cannot be shared, and it is not what the seat
    /// is called. The rule matches combs-kit's, so the two agree on the
    /// name of the same model.
    #[test]
    fn a_model_is_named_never_pathed() {
        // A synthetic home, never this machine's: a tracked file that
        // carries a real absolute path is refused by the hygiene gate,
        // and rightly — it names whoever built it.
        assert_eq!(
            identity("/home/someone/.cache/combs/models/qwen3-0.6b-gguf/model.gguf"),
            "qwen3-0.6b-gguf"
        );
        // A directory seat (safetensors) is named by itself.
        assert_eq!(identity("/var/models/qwen3-0.6b"), "qwen3-0.6b");
        // Already a bare name: unchanged.
        assert_eq!(identity("qwen3-4b-gguf"), "qwen3-4b-gguf");
        for p in [
            "/home/someone/.cache/combs/models/qwen3-0.6b-gguf/model.gguf",
            "/home/other/models/x/model.gguf",
        ] {
            assert!(
                !identity(p).contains('/'),
                "an identity must not carry a path: {}",
                identity(p)
            );
        }
    }

    #[test]
    fn a_run_round_trips_through_its_file() {
        let dir = std::env::temp_dir().join(format!("combs-tk-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let r = run("text", 8080);
        save(&dir, &r).expect("save");
        let back = load_all(&dir);
        assert_eq!(back.len(), 1);
        assert_eq!(back[0], r);
        forget(&dir, "text");
        assert!(load_all(&dir).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A record is never enough on its own, and neither is a port: the
    /// toolkit signals only a pid it recorded AND finds alive.
    #[test]
    fn liveness_needs_both_readings() {
        assert_eq!(liveness(true, true), Liveness::Up);
        assert_eq!(liveness(false, true), Liveness::Stale);
        assert_eq!(liveness(false, false), Liveness::Stale);
    }

    /// A live process that has not bound yet is STARTING, never down —
    /// the engine binds only after the model loads, and a big seat takes
    /// half a minute. Killing it for that would be a crash loop.
    #[test]
    fn a_loading_engine_is_starting_not_down() {
        assert_eq!(liveness(true, false), Liveness::Starting);
    }

    /// A port nobody serves. Port 1 is privileged and unused, and a
    /// connect to it on loopback is refused at once — measured, 500 of
    /// 500. A port BORROWED from [`free_port`] cannot do this job: tests
    /// asking for ports at the same time collide on one number, which is
    /// what made every earlier shape of these tests flake.
    const NOBODY_SERVES: u16 = 1;

    /// The positive control for [`port_answers`], which every wait test
    /// leans on: a listener this test HOLDS is seen on its own port.
    ///
    /// It keeps the listener rather than releasing the port and taking
    /// it again, because taking it again is a race — another test's
    /// `free_port` wins the number in between and this one dies with
    /// "address already in use". Measured: three runs in five.
    #[test]
    fn a_listener_is_seen_on_the_port_it_holds() {
        use std::net::{Ipv4Addr, TcpListener};
        let held = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("the OS can lend a port");
        let p = held
            .local_addr()
            .expect("a bound listener has an address")
            .port();
        assert!(port_answers(p), "a real listener must be seen on its port");
        drop(held);
    }

    /// What `free_port` can promise is a port the OS lent at the moment
    /// it was asked. It cannot promise the number is still free a moment
    /// later — on a machine running other programs nothing can — so that
    /// is not asserted here.
    #[test]
    fn a_lent_port_is_a_real_port() {
        assert!(free_port().expect("the OS can lend a port") > 0);
    }

    /// This process is alive; a pid that cannot exist is not.
    #[test]
    fn pid_liveness_reads_the_os() {
        assert!(pid_alive(std::process::id()));
        assert!(!pid_alive(0x7FFF_FFFF));
    }

    /// A child that exits before it ever binds is DEAD, and the toolkit
    /// must read it as dead at once. The pid cannot say so: an unreaped
    /// child is a zombie, and `kill(pid, 0)` calls a zombie alive — so
    /// the pid road waits on a corpse. It did, for thirteen minutes,
    /// against an engine that had already refused to load and written
    /// why. The exit code must survive the reading too: it is what a
    /// caller reports.
    #[test]
    fn a_child_that_exited_is_dead_not_starting() {
        let mut child = std::process::Command::new("sh")
            .args(["-c", "exit 3"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn a child that exits at once");
        let port = NOBODY_SERVES;
        let out = wait_for_start(&mut child, port, Duration::from_secs(2), |_| {});
        assert_eq!(
            out,
            Start::Died(Some(3)),
            "a child that has exited must read as dead, and carry its code"
        );
    }

    /// The other half of the same rule: an engine that is merely slow is
    /// never killed and never called dead. The wait gives up; the engine
    /// keeps loading.
    #[test]
    fn a_slow_engine_is_left_running_when_the_wait_gives_up() {
        let mut child = std::process::Command::new("sh")
            .args(["-c", "sleep 30"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn a child that outlives the wait");
        let port = NOBODY_SERVES;
        let out = wait_for_start(&mut child, port, Duration::from_millis(600), |_| {});
        assert_eq!(out, Start::Waiting);
        assert!(
            pid_alive(child.id()),
            "the toolkit must never kill an engine that is still loading"
        );
        let _ = child.kill();
        let _ = child.wait();
    }
}
