//! # Asking the engine, and reading what the asking cost.
//!
//! ## Why this is not in `cmds.rs`
//!
//! `cmds.rs` drives BINARIES: it finds `combs`, spawns it, watches a log.
//! This half drives a SOCKET — it speaks the engine's own wire and reads
//! its accounting. The two share a display and nothing else, and the
//! rules worth testing here are rules about numbers, not about
//! processes. So the decisions live in pure functions with their tests
//! beside them, and the body below is the thin part that holds a socket.
//!
//! ## The number this module exists to refuse
//!
//! The engine reports `timing.prefill_tok_s`, and it is a trap: it
//! divides the WHOLE prompt by the time to first token, cached tokens
//! included. On a warm session that reads as over a thousand tokens a
//! second for work the engine did not do. Every rate printed here is
//! computed from what was actually prefilled, and the one rate that
//! cannot honestly be computed — the decode rate of a one-token answer —
//! is printed as a dash rather than a zero that reads as slow.

use crate::api::Cost;
use crate::state::{Liveness, Run};
use crate::ui;

/// Which seat answers, and when none does, what to type instead.
///
/// A record is not a seat. `ps` learned this the hard way: a record, a
/// pid and a port must all agree before a process counts as ours, and a
/// pid alone calls a zombie alive. So the caller passes the liveness it
/// measured and this only decides.
pub fn seat<'a>(
    runs: &'a [Run],
    role: &str,
    live: impl Fn(&Run) -> Liveness,
) -> Result<&'a Run, String> {
    let mine: Vec<&Run> = runs.iter().filter(|r| r.role == role).collect();
    if mine.is_empty() {
        return Err(format!(
            "no {role} engine is recorded — combs-tk up <model> --role {role}"
        ));
    }
    // A seat that is merely recorded is not a seat, so the three states
    // are told apart by name: waiting on a loading engine is patience,
    // and clearing a stale record is work.
    let mut starting = false;
    for r in mine {
        match live(r) {
            Liveness::Up => return Ok(r),
            Liveness::Starting => starting = true,
            Liveness::Stale => {}
        }
    }
    if starting {
        Err(format!(
            "the {role} engine is still starting — combs-tk ps, then ask again"
        ))
    } else {
        Err(format!(
            "the {role} engine's record is stale: its process is gone — combs-tk up <model> --role {role}"
        ))
    }
}

/// The answer without the model's thinking.
///
/// Qwen3 emits a `<think>` block even when told not to think — an empty
/// one, which reads as a stutter at the head of every reply. An unclosed
/// block means the model is thinking still, so everything after the open
/// tag is folded too rather than printed as if it were the answer.
pub fn fold_think(text: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(OPEN) {
        out.push_str(&rest[..at]);
        let after = &rest[at + OPEN.len()..];
        match after.find(CLOSE) {
            Some(end) => rest = &after[end + CLOSE.len()..],
            // Still thinking: nothing past the open tag is an answer yet.
            None => return out.trim().to_string(),
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

/// What a finish reason costs at the exit.
///
/// A truncated answer is not a success. It looks like one — the bytes
/// arrived, the socket closed cleanly — and a script that believes it
/// will carry half an answer forward as though it were whole.
pub fn code_for(finish: Option<&str>) -> i32 {
    match finish {
        Some("length") => 3,
        _ => 0,
    }
}

/// The one line a person reads after an answer.
///
/// Every rate here is computed from work the engine actually did, and a
/// rate that cannot be computed is a dash. A zero in that place reads as
/// "slow" when the truth is "there was nothing to measure".
pub fn ledger(cost: &Cost, dispatches: Option<f64>) -> String {
    fn rate(v: Option<f64>) -> String {
        match v {
            Some(r) => format!("{r:.1} t/s"),
            None => "—".to_string(),
        }
    }
    let mut line = format!(
        "prompt {} · skipped {} · computed {} · prefill {} ({} computed) · answer {} tok / {} / {}",
        cost.skipped + cost.computed,
        cost.skipped,
        cost.computed,
        ui::ms(cost.prefill_ms),
        rate(cost.computed_tok_s),
        cost.answer_tokens,
        ui::ms(cost.answer_ms),
        rate(cost.decode_tok_s),
    );
    // The engine reports zero dispatches a token for a one-token answer,
    // and that is not "no work" — the prefill plainly ran. It is a
    // figure that means nothing at this length, so it reads as a dash
    // like every other rate that cannot honestly be computed.
    match dispatches {
        Some(d) if d > 0.0 => line.push_str(&format!(" · {d:.0} disp/tok")),
        _ => line.push_str(" · — disp/tok"),
    }
    line
}

/// The spread of a set of readings, and whether it may be headlined.
#[derive(Debug, Clone, PartialEq)]
pub struct Spread {
    pub median: f64,
    pub low: f64,
    pub high: f64,
    /// How far the readings ranged, as a percentage of the median.
    pub pct: f64,
    /// How many readings it rests on. Carried because a percentage
    /// cannot say this by itself, and zero percent of one reading looks
    /// exactly like perfect agreement.
    pub readings: usize,
}

impl Spread {
    /// Past this band the readings are a range, not a number, and the
    /// range is what gets printed.
    pub const BAND: f64 = 10.0;

    /// Whether these readings agree well enough to quote one number.
    ///
    /// One reading has nothing to agree with. It still gets reported —
    /// it is all there is — but it is never called steady: against a
    /// live engine the lone reading is the cold one, which prefilled at
    /// 6.5 tokens a second where the warm pair beside it read above 100.
    pub fn steady(&self) -> bool {
        self.readings >= 2 && self.pct <= Self::BAND
    }
}

/// The readings that count: every one but the first.
///
/// Run one of a model pays for a cold cache and a cold autotune, and it
/// has told four separate lies in this estate's records. A single
/// reading is still reported — it is all there is — but it is never
/// called warm.
pub fn warm(values: &[f64]) -> &[f64] {
    if values.len() > 1 {
        &values[1..]
    } else {
        values
    }
}

/// The median and the range, or None when there is nothing to measure.
pub fn spread(values: &[f64]) -> Option<Spread> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    // These readings come from the engine's own accounting and are never
    // NaN, but a total order costs nothing and does not rest on that.
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    let median = if v.len() % 2 == 1 {
        v[mid]
    } else {
        (v[mid - 1] + v[mid]) / 2.0
    };
    let low = v[0];
    let high = v[v.len() - 1];
    let pct = if median > 0.0 {
        (high - low) / median * 100.0
    } else {
        0.0
    };
    Some(Spread {
        median,
        low,
        high,
        pct,
        readings: v.len(),
    })
}

/// A thought folded out of an answer while it is still arriving.
///
/// [`fold_think`] needs the whole text and a stream does not have it. A
/// reader watching tokens land must never see a sentence that a later
/// `</think>` would take back, so anything that might still turn out to
/// be a thought is held until it is known — including a tag chopped
/// across two deltas, which is the ordinary case at a 16-byte chunk.
#[derive(Debug, Default)]
pub struct Folder {
    held: String,
    thinking: bool,
}

impl Folder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Take a piece of the stream; hand back only what is safe to print.
    pub fn push(&mut self, delta: &str) -> String {
        const OPEN: &str = "<think>";
        const CLOSE: &str = "</think>";
        self.held.push_str(delta);
        let mut out = String::new();
        loop {
            let tag = if self.thinking { CLOSE } else { OPEN };
            match self.held.find(tag) {
                Some(at) => {
                    // Everything before an opening tag is answer; before a
                    // closing tag it is thought, and is dropped.
                    if !self.thinking {
                        out.push_str(&self.held[..at]);
                    }
                    self.held = self.held[at + tag.len()..].to_string();
                    self.thinking = !self.thinking;
                }
                None => {
                    // No tag yet — but the tail may be the front of one,
                    // which at a small chunk size is the ordinary case.
                    let keep = partial_tag(&self.held, tag);
                    let cut = self.held.len() - keep;
                    if !self.thinking {
                        out.push_str(&self.held[..cut]);
                    }
                    self.held = self.held[cut..].to_string();
                    return out;
                }
            }
        }
    }

    /// The stream ended. Whatever is left, unless it is a thought that
    /// never closed — the model was still thinking when it ran out.
    pub fn finish(&mut self) -> String {
        if self.thinking {
            self.held.clear();
            return String::new();
        }
        std::mem::take(&mut self.held)
    }
}

/// How many bytes at the end of `s` could still be the front of `tag`.
///
/// Held back rather than printed, because a delta boundary falls
/// wherever the engine's chunking puts it and `<thi` + `nk>` is one tag,
/// not four characters of answer.
fn partial_tag(s: &str, tag: &str) -> usize {
    let max = (tag.len() - 1).min(s.len());
    (1..=max)
        .rev()
        .find(|&n| s.is_char_boundary(s.len() - n) && tag.starts_with(&s[s.len() - n..]))
        .unwrap_or(0)
}

// ---- the half that holds a socket ----
//
// Everything above is a rule and has its test beside it. Everything
// below finds a seat, opens a connection and renders — it is proven by
// running it against a live engine, because that is the only thing that
// can prove it.

use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::api::{self, Ask, Message};
use crate::fence::{self, Verdict};
use crate::net;
use crate::profile::{self, Profile};
use crate::render;
use crate::state;
use crate::tools::{self, Tool};
use crate::wire::Frame;

/// A wedged engine must not hold the toolkit forever, but this is an
/// inactivity timeout and the engine sends a keepalive comment every
/// 100 ms of idle decode — so it fires only when the engine has truly
/// stopped talking, never because an answer was long.
const QUIET_FOR: Duration = Duration::from_secs(300);

fn fail(msg: &str) -> i32 {
    eprintln!("{} {}", ui::bad("combs-tk:"), msg);
    1
}

/// Does this verb exist where the fenced child would look for it?
///
/// Asked against the fence's OWN pinned PATH rather than the caller's,
/// because the child is given that PATH and nothing else. Answering from
/// the ambient environment would let a ladder choose a rung the child
/// then cannot find — a refusal at exec, long after the decision, and
/// nowhere near the rule that made it.
fn on_fenced_path(verb: &str) -> bool {
    fence::fenced_env()
        .iter()
        .find(|(k, _)| k == "PATH")
        .map(|(_, path)| {
            path.split(':')
                .any(|dir| std::path::Path::new(dir).join(verb).is_file())
        })
        .unwrap_or(false)
}

/// One generation and what it cost.
struct Reading {
    text: String,
    finish: Option<String>,
    usage: Option<api::Usage>,
    client_ms: f64,
}

/// Run one generation, printing the answer as it lands.
///
/// The engine's own clock starts after the request leaves its queue, so
/// the wall measured here is always the longer of the two. Both are
/// reported: the difference is the queue, and hiding it would make a
/// busy engine look fast.
fn generate(port: u16, body: &[u8], show: bool) -> Result<Reading, String> {
    let mut folder = Folder::new();
    let mut text = String::new();
    let mut finish: Option<String> = None;
    let mut usage: Option<api::Usage> = None;
    let mut broke: Option<String> = None;
    let began = Instant::now();

    net::stream(port, "/v1/chat/completions", body, QUIET_FOR, |f| match f {
        Frame::Content(piece) => {
            let safe = folder.push(piece);
            if !safe.is_empty() {
                text.push_str(&safe);
                if show {
                    print!("{safe}");
                    let _ = std::io::stdout().flush();
                }
            }
        }
        Frame::Finish(why) => finish = Some(why.clone()),
        Frame::Usage(v) => usage = api::parse_usage(v),
        // An in-band error arrives at HTTP 200, so the status line cannot
        // be trusted to report it and the caller must be told here.
        Frame::Error(e) => broke = Some(e.clone()),
        Frame::Calls(_) | Frame::Done => {}
    })?;

    let tail = folder.finish();
    if !tail.is_empty() {
        text.push_str(&tail);
        if show {
            print!("{tail}");
        }
    }
    if show {
        println!();
        let _ = std::io::stdout().flush();
    }
    if let Some(e) = broke {
        return Err(format!("the engine failed after it had begun answering: {e}"));
    }
    Ok(Reading {
        text,
        finish,
        usage,
        client_ms: began.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Ask the seat that is up, and price what the asking cost.
#[allow(clippy::too_many_arguments)]
pub fn ask(
    prompt: &str,
    role: &str,
    session: Option<&str>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    no_think: bool,
    repeat: usize,
    json_out: bool,
) -> i32 {
    let dir = match state::root() {
        Ok(d) => d,
        Err(e) => return fail(&e),
    };
    let runs = state::load_all(&dir);
    let seated = match seat(&runs, role, |r| {
        state::liveness(state::pid_alive(r.pid), state::port_answers(r.port))
    }) {
        Ok(r) => r,
        // Exit 2, not 1: nothing is wrong with the asking, there is
        // simply nothing to ask. A script can tell those apart only by
        // the code.
        Err(e) => {
            eprintln!("{} {}", ui::bad("combs-tk:"), e);
            return 2;
        }
    };

    let content = if no_think {
        format!("{prompt} /no_think")
    } else {
        prompt.to_string()
    };
    let wanted = Ask {
        messages: vec![Message { role: "user".into(), content }],
        max_tokens,
        temperature,
        session: session.map(str::to_string),
        stream: true,
        ..Ask::default()
    };
    let body = match api::chat_body(&wanted) {
        Ok(b) => b,
        Err(e) => return fail(&e),
    };
    let bytes = body.to_string().into_bytes();

    let repeat = repeat.max(1);
    if !json_out {
        println!("{}", ui::head(&render::seat_header("ask", &seated.model)));
        println!("  {}", ui::dim(&format!("{} on port {}", seated.dtype, seated.port)));
    }

    let mut readings: Vec<Reading> = Vec::new();
    for i in 0..repeat {
        // The answer is shown once: a person wants to read it, not read
        // it three times. The readings after it are measurement.
        let r = match generate(seated.port, &bytes, !json_out && i == 0) {
            Ok(r) => r,
            Err(e) => return fail(&e),
        };
        if !json_out {
            match &r.usage {
                Some(u) => println!("  {}", ui::dim(&ledger(&api::cost(u), u.dispatches_per_token))),
                None => println!(
                    "  {}",
                    ui::warn("this generation reported no usage — the numbers below are missing, not zero")
                ),
            }
        }
        readings.push(r);
    }

    let rates: Vec<f64> = readings
        .iter()
        .filter_map(|r| r.usage.as_ref().and_then(|u| api::cost(u).decode_tok_s))
        .collect();
    let steady = spread(warm(&rates));

    if !json_out && repeat > 1 {
        if let Some(s) = &steady {
            let line = if s.steady() {
                ui::good(&format!(
                    "decode {:.1} t/s, median of {} warm readings",
                    s.median, s.readings
                ))
            } else if s.readings < 2 {
                // Two generations leave exactly one warm reading, and one
                // reading has nothing to agree with. Say that, rather
                // than calling zero percent a range.
                ui::warn(&format!(
                    "decode {:.1} t/s from one warm reading — a spread needs two: --repeat 3",
                    s.median
                ))
            } else {
                ui::warn(&format!(
                    "decode {:.1}–{:.1} t/s over {} warm readings: a {:.0}% spread is a range, not a number",
                    s.low, s.high, s.readings, s.pct
                ))
            };
            println!("  {line}");
        }
    }

    let finish = readings.last().and_then(|r| r.finish.clone());
    let code = code_for(finish.as_deref());

    if json_out {
        let record = json!({
            "schema": "combs-tk.ask.v1",
            "seat": seated.model,
            "role": seated.role,
            "port": seated.port,
            "build": { "dtype": seated.dtype, "commit": seated.commit },
            "readings": readings.iter().map(|r| {
                let c = r.usage.as_ref().map(api::cost);
                json!({
                    "prompt": r.usage.as_ref().map(|u| u.prompt_tokens),
                    "skipped": c.as_ref().map(|c| c.skipped),
                    "computed": c.as_ref().map(|c| c.computed),
                    "prefill_ms": c.as_ref().map(|c| c.prefill_ms),
                    "computed_tok_s": c.as_ref().and_then(|c| c.computed_tok_s),
                    "answer_tokens": c.as_ref().map(|c| c.answer_tokens),
                    "answer_ms": c.as_ref().map(|c| c.answer_ms),
                    "decode_tok_s": c.as_ref().and_then(|c| c.decode_tok_s),
                    "dispatches_per_token": r.usage.as_ref().and_then(|u| u.dispatches_per_token),
                    "session_id": r.usage.as_ref().and_then(|u| u.session_id.clone()),
                    "client_ms": r.client_ms,
                    "finish": r.finish,
                    "answer": r.text,
                })
            }).collect::<Vec<_>>(),
            "warm_decode_tok_s": steady.as_ref().map(|s| json!({
                "median": s.median,
                "low": s.low,
                "high": s.high,
                "spread_pct": s.pct,
                // Carried because a percentage cannot say it: zero
                // percent of one reading looks exactly like agreement.
                "readings": s.readings,
                "steady": s.steady(),
            })),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&record).unwrap_or_else(|_| record.to_string())
        );
    } else if code == 3 {
        eprintln!(
            "{} {}",
            ui::bad("combs-tk:"),
            "the answer stopped at max_tokens: it is half an answer, not a short one"
        );
    }
    code
}

/// Fetch a model through the engine's own puller.
pub fn pull(source: &str) -> i32 {
    let bin = match tools::find(Tool::Engine) {
        Ok(b) => b,
        Err(e) => return fail(&e),
    };
    // A pull moves gigabytes and reports as it goes. Capturing its output
    // would hide the one thing worth watching.
    match Command::new(&bin).arg("pull").arg(source).status() {
        Ok(s) if s.success() => 0,
        Ok(s) => fail(&format!(
            "combs pull {source} exited {}",
            s.code().unwrap_or(-1)
        )),
        Err(e) => fail(&format!("{} could not be run: {e}", bin.display())),
    }
}

/// What answers this tool on this machine, or why nothing does.
///
/// Asked of the toolbox rather than kept in a list beside it. A parallel
/// list is a second truth, and two truths drift the first time a
/// capability lands and nobody remembers the other place — which is how
/// the cache home came to be resolved twice in this same crate.
fn answers_here(tool: &str) -> Result<String, String> {
    availability(tool, on_fenced_path)
}

/// What this profile allows, and which of those exist here.
pub fn tool_list(profile_name: &str, json_out: bool) -> i32 {
    let home = match state::home() {
        Ok(h) => h,
        Err(e) => return fail(&e),
    };
    let p = match profile::load(profile_name, &home) {
        Ok(p) => p,
        Err(e) => return fail(&e),
    };
    if json_out {
        let record = json!({
            "schema": "combs-tk.tools.v1",
            "profile": p.name,
            "about": p.about,
            "roots": p.roots,
            "tools": p.tools.iter().map(|t| {
                let by = answers_here(t);
                json!({
                    "name": t,
                    "built": by.is_ok(),
                    // Which implementation would answer HERE. Two records
                    // of the same capability are not comparable without
                    // it: `netstat -rn` and `ip route` are one fact read
                    // two ways.
                    "answered_by": by.as_ref().ok(),
                    // And when nothing answers, WHY — because "never
                    // taught" and "nothing installed" send a person to
                    // different places, and only one is a package manager.
                    "unavailable": by.as_ref().err(),
                })
            }).collect::<Vec<_>>(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&record).unwrap_or_else(|_| record.to_string())
        );
        return 0;
    }
    println!("{}", ui::head(&format!("{} — {}", p.name, p.about)));
    println!("  {:<10} {}", "roots", p.roots.join(", "));
    for t in &p.tools {
        let state = match answers_here(t) {
            Ok(by) => ui::good(&format!("built · {by}")),
            // Say WHICH failure it is. "nothing here answers it" sent a
            // person to install `ping` and `dig`, which were already
            // installed — the toolbox had simply never been taught those
            // capabilities, and that is not a thing anyone can install.
            Err(why) if why.contains("not a capability") => {
                ui::dim("not taught to the toolbox yet")
            }
            Err(why) => ui::warn(&why),
        };
        println!("  {t:<18} {state}");
    }
    0
}

/// Judge one tool call, and — unless this is a dry run — take it.
///
/// The exit code is the contract. A dry run exits 1 having spawned
/// nothing, because a dry run that exited 0 would read as "it ran"; a
/// refusal exits 1; an ask without a person's consent exits 1, because
/// only a person can consent and there is nobody inside a tool call.
pub fn tool_call(
    tool: &str,
    args_json: &str,
    profile_name: &str,
    dry_run: bool,
    yes: bool,
) -> i32 {
    let home = match state::home() {
        Ok(h) => h,
        Err(e) => return fail(&e),
    };
    let p = match profile::load(profile_name, &home) {
        Ok(p) => p,
        Err(e) => return fail(&e),
    };
    let args: Value = match serde_json::from_str(args_json) {
        Ok(v) => v,
        Err(e) => return fail(&format!("--args is not JSON: {e}")),
    };
    // The roots are the person's own home, not the cache home the
    // profiles live under.
    let person = std::env::var("HOME").unwrap_or_default();
    let call = match judge_call(tool, &args, &p, Path::new(&person), on_fenced_path) {
        Ok(c) => c,
        Err(e) => return fail(&e),
    };

    let names: Vec<&str> = call.env.iter().map(|(k, _)| k.as_str()).collect();
    let show = || {
        println!("  {:<10} {}", "argv", call.argv.join(" "));
        println!("  {:<10} {}", "env", names.join(" "));
    };

    match &call.verdict {
        Verdict::Refuse(why) => {
            println!("{}", ui::head(&format!("tool call {tool}")));
            println!("  {:<10} {}", "verdict", ui::bad("refused"));
            println!("  {:<10} {why}", "because");
            show();
            println!("  {}", ui::dim("nothing was spawned"));
            1
        }
        Verdict::Ask(why) if !yes => {
            println!("{}", ui::head(&format!("tool call {tool}")));
            println!("  {:<10} {}", "verdict", ui::warn("asks"));
            println!("  {:<10} {why}", "because");
            show();
            println!(
                "  {}",
                ui::dim("nothing was spawned — pass --yes to take it yourself")
            );
            1
        }
        _ if dry_run => {
            println!("{}", ui::head(&format!("tool call {tool} — dry run")));
            let word = match &call.verdict {
                Verdict::Allow => ui::good("allowed"),
                _ => ui::warn("asks, and --yes was given"),
            };
            println!("  {:<10} {word}", "verdict");
            show();
            println!("  {}", ui::dim("nothing was spawned"));
            1
        }
        _ => {
            let mut cmd = Command::new(&call.argv[0]);
            cmd.args(&call.argv[1..]);
            // The doors are absent by construction: the child is given
            // this list and nothing of ours.
            cmd.env_clear();
            for (k, v) in &call.env {
                cmd.env(k, v);
            }
            match cmd.output() {
                Ok(out) => {
                    print!("{}", String::from_utf8_lossy(&out.stdout));
                    let err = String::from_utf8_lossy(&out.stderr);
                    if !err.trim().is_empty() {
                        eprint!("{err}");
                    }
                    out.status.code().unwrap_or(1)
                }
                Err(e) => fail(&format!("{} could not be run: {e}", call.argv[0])),
            }
        }
    }
}

/// What a tool call would do, decided before anything is spawned.
///
/// The decision is kept apart from the doing so a person can see it
/// without taking it. That is what makes every refusal the fence makes
/// reproducible by hand: no model in the loop, and nothing run.
#[derive(Debug)]
pub struct Call {
    pub verdict: Verdict,
    pub argv: Vec<String>,
    pub env: Vec<(String, String)>,
}

/// Judge one tool call against a profile. Pure: it spawns nothing, reads
/// no clock, and touches no socket.
///
/// `present` answers whether a verb exists on this machine, and is
/// passed in rather than read for the same reason the ladder takes it:
/// the rule has to be testable on a box with the opposite set of tools
/// installed, which is the whole reason a ladder exists at all.
pub fn judge_call(
    tool: &str,
    args: &Value,
    p: &Profile,
    home: &Path,
    present: impl Fn(&str) -> bool,
) -> Result<Call, String> {
    let _ = &present;
    // Deny by default, and name the door that was shut. "invalid" sends
    // a person to the source; naming the tool sends them to the profile.
    if !p.allows(tool) {
        return Err(format!(
            "{tool} is not in the {} profile: what a profile does not name cannot be called",
            p.name
        ));
    }
    match tool {
        "shell.run" => {
            let line = args
                .get("line")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{tool} needs a `line` to judge, and this call carries none"))?;
            let policy = p.policy(home);
            Ok(Call {
                verdict: fence::decide_shell(line, &policy),
                argv: fence::shell_argv(line),
                env: fence::fenced_env(),
            })
        }
        // A capability: resolved through its ladder, then judged like any
        // other line. The ladder chooses WHICH implementation answers —
        // `netstat -rn` here, `ip route` on a box that has neither — and
        // refuses by name when nothing on this machine can answer at all.
        //
        // What comes back is a line the fence judges exactly as it judges
        // a typed one. A capability is never a way around the fence.
        _ => {
            let chosen = crate::toolbox::resolve(tool, present)?;
            let mut line = chosen.line;
            // An operand comes from the CALL, never from the rung: a host
            // to ping, a name to look up. It is appended and then the
            // WHOLE line is judged, so an argument arriving from a model
            // is not trusted for having arrived through a tool — a host
            // carrying a chain dies at the character gate rather than
            // becoming a second command.
            if let Some(key) = crate::toolbox::find(tool).and_then(|c| c.takes) {
                let operand = args
                    .get(key)
                    .and_then(Value::as_str)
                    .filter(|s| !s.trim().is_empty())
                    .ok_or_else(|| {
                        format!("{tool} needs a `{key}`, and this call carries none")
                    })?;
                line.push(' ');
                line.push_str(operand.trim());
            }
            let policy = p.policy(home);
            Ok(Call {
                verdict: fence::decide_shell(&line, &policy),
                argv: fence::shell_argv(&line),
                env: fence::fenced_env(),
            })
        }
    }
}

/// What answers this tool here — or, failing that, WHY nothing does.
///
/// Pure, so the two failures can be told apart without a machine that
/// has the opposite tools installed. The distinction is the whole point:
/// "the toolbox has never heard of this" and "nothing that answers it is
/// installed" are different problems, and collapsing them sends a person
/// to install software they already have.
pub fn availability(tool: &str, present: impl Fn(&str) -> bool) -> Result<String, String> {
    // The shell is not a ladder — the fence itself is what answers it.
    if tool == "shell.run" {
        return Ok("the fence".into());
    }
    // The ladder already words its two failures differently. Carrying
    // them out rather than flattening them to None is the whole fix:
    // "never taught" and "nothing installed" send a person to different
    // places, and only one of them is their package manager.
    crate::toolbox::resolve(tool, present).map(|c| c.verb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A capability that needs a host takes it from the call, and the
    /// host lands at the end of the line the ladder built.
    #[test]
    fn a_capability_that_needs_a_host_takes_it_from_the_call() {
        let c = judge_call(
            "net.ping",
            &serde_json::json!({"host": "1.1.1.1"}),
            &networky(),
            Path::new("/home/someone"),
            bsd,
        )
        .expect("a judged ping");
        let line = c.argv.last().expect("a line to run");
        assert!(line.contains("ping -c"), "{line:?}");
        assert!(line.ends_with("1.1.1.1"), "the host must reach the line: {line:?}");
        assert_eq!(c.verdict, Verdict::Allow, "a bounded ping at a literal address is the ordinary reading");
    }

    /// Needing a host and being given none is a typo, not a run, and the
    /// refusal names the argument that is missing.
    #[test]
    fn a_capability_that_needs_a_host_and_is_given_none_is_refused_by_name() {
        let e = judge_call(
            "net.ping",
            &serde_json::json!({}),
            &networky(),
            Path::new("/home/someone"),
            bsd,
        )
        .unwrap_err();
        assert!(e.contains("host"), "the refusal must name the argument: {e}");
    }

    /// The rule this whole design rests on: an argument arriving from a
    /// model is not trusted for having arrived through a tool. The
    /// composed line goes through the fence like any typed one, so a
    /// host carrying a chain is refused at the character gate rather
    /// than becoming a second command.
    #[test]
    fn an_operand_cannot_smuggle_shell_syntax_past_the_fence() {
        for hostile in ["1.1.1.1; rm -rf /", "1.1.1.1 && rm x", "$(rm -rf /)", "a | sh"] {
            let c = judge_call(
                "net.ping",
                &serde_json::json!({ "host": hostile }),
                &networky(),
                Path::new("/home/someone"),
                bsd,
            )
            .expect("a judged call");
            assert!(
                matches!(c.verdict, Verdict::Refuse(_)),
                "{hostile:?} was not refused: {:?}",
                c.verdict
            );
        }
    }

    /// A capability with a rung installed says which one answers.
    #[test]
    fn availability_names_the_implementation_that_answers() {
        assert_eq!(availability("net.routes", bsd), Ok("netstat".into()));
        assert_eq!(availability("net.interfaces", bsd), Ok("ifconfig".into()));
    }

    /// The fence is not a ladder, but it is what answers `shell.run`.
    #[test]
    fn the_fence_is_what_answers_a_shell_line() {
        assert!(availability("shell.run", bsd).is_ok());
    }

    /// A tool nothing has taught the toolbox is NOT the same as one
    /// whose implementations are missing — and the words must say which,
    /// or a person goes and installs something they already have.
    #[test]
    fn a_tool_the_toolbox_never_learned_is_told_apart_from_a_missing_one() {
        let unknown = availability("net.scan", bsd).unwrap_err();
        assert!(
            unknown.contains("not a capability"),
            "an unlearned tool must say so: {unknown}"
        );

        let missing = availability("net.routes", |_| false).unwrap_err();
        assert!(
            missing.contains("installed"),
            "a missing implementation must say so: {missing}"
        );
        assert!(
            missing.contains("netstat"),
            "and name what it looked for: {missing}"
        );
        assert_ne!(
            unknown, missing,
            "the two failures must not read the same"
        );
    }

    fn networky() -> Profile {
        profile::parse(profile::NETWORKY).expect("the shipped profile parses")
    }

    /// What this desk actually has. Measured, not assumed: `ip`, `ss`,
    /// `resolvectl`, `nmap` and `rg` are all absent here.
    fn bsd(verb: &str) -> bool {
        ["ifconfig", "netstat", "lsof", "arp", "scutil", "ping", "traceroute"].contains(&verb)
    }

    /// A capability is resolved through its ladder and comes back as a
    /// line the fence has judged — not as "not built yet".
    #[test]
    fn a_capability_resolves_through_its_ladder_and_is_judged() {
        let c = judge_call(
            "net.routes",
            &serde_json::json!({}),
            &networky(),
            Path::new("/home/someone"),
            bsd,
        )
        .expect("this desk can read its own routes");
        assert_eq!(c.verdict, Verdict::Allow);
        let line = c.argv.last().expect("a line to run");
        assert!(
            line.contains("netstat -rn"),
            "the ladder's chosen line must be the one that runs: {line:?}"
        );
    }

    /// The same capability on a machine with the opposite toolset picks
    /// the other rung — that is the whole reason a ladder exists.
    #[test]
    fn the_other_machines_rung_answers_there() {
        let gnu = |v: &str| ["ip", "ss", "resolvectl"].contains(&v);
        let c = judge_call(
            "net.routes",
            &serde_json::json!({}),
            &networky(),
            Path::new("/home/someone"),
            gnu,
        )
        .expect("a linux box can read its own routes");
        let line = c.argv.last().expect("a line to run");
        assert!(line.contains("ip route"), "{line:?}");
    }

    /// A capability nothing on this machine can answer is refused by
    /// name, and the refusal names what it looked for — so a person is
    /// sent to their package manager, not to the source.
    #[test]
    fn a_capability_with_no_implementation_here_is_refused_by_name() {
        let e = judge_call(
            "net.routes",
            &serde_json::json!({}),
            &networky(),
            Path::new("/home/someone"),
            |_| false,
        )
        .unwrap_err();
        assert!(e.contains("net.routes"), "the refusal must name it: {e}");
        assert!(e.contains("netstat"), "and what it looked for: {e}");
    }

    /// Deny by default, and the refusal names the door that was shut
    /// rather than calling the whole thing invalid.
    #[test]
    fn a_tool_the_profile_does_not_name_is_refused_by_name() {
        let e = judge_call(
            "tk.measure",
            &serde_json::json!({}),
            &networky(),
            Path::new("/home/someone"),
            bsd,
        )
        .unwrap_err();
        assert!(e.contains("tk.measure"), "the refusal must name the tool: {e}");
    }

    /// A tool the profile allows but the toolbox has never heard of is
    /// refused at the CALL, by name — so a profile may be written ahead
    /// of its tools and stay true as they land.
    ///
    /// `net.scan` rather than `net.routes`: the toolbox knows the routes
    /// now, so naming that one here would test the opposite of what it
    /// says the moment the ladder is wired.
    #[test]
    fn a_tool_that_is_named_but_not_built_is_refused_at_the_call() {
        let e = judge_call(
            "net.scan",
            &serde_json::json!({}),
            &networky(),
            Path::new("/home/someone"),
            bsd,
        )
        .unwrap_err();
        assert!(e.contains("net.scan"), "the refusal must name the tool: {e}");
    }

    /// The verdict travels with the argv it would have run, so a dry run
    /// shows both the answer and the thing it answered about.
    #[test]
    fn a_judged_call_carries_the_argv_it_did_not_run() {
        let c = judge_call(
            "shell.run",
            &serde_json::json!({"line": "netstat -an | grep LISTEN"}),
            &networky(),
            Path::new("/home/someone"),
            bsd,
        )
        .expect("a judged call");
        assert_eq!(c.verdict, Verdict::Allow);
        assert!(c.argv.contains(&"-f".to_string()), "{:?}", c.argv);
        assert!(
            c.argv.last().expect("a line to run").starts_with("command "),
            "{:?}",
            c.argv
        );
        let keys: Vec<&str> = c.env.iter().map(|(k, _)| k.as_str()).collect();
        assert!(
            !keys.contains(&"BASH_ENV"),
            "a shell door reached the child: {keys:?}"
        );
    }

    /// A refused line is still a judged call: someone reproducing a
    /// refusal wants to see what would have run.
    #[test]
    fn a_refused_line_is_still_a_judged_call() {
        let c = judge_call(
            "shell.run",
            &serde_json::json!({"line": "rm -rf /"}),
            &networky(),
            Path::new("/home/someone"),
            bsd,
        )
        .expect("a judged call");
        assert!(matches!(c.verdict, Verdict::Refuse(_)), "{:?}", c.verdict);
    }

    /// A tool that takes a line and is given none is a typo, not a run.
    #[test]
    fn shell_run_without_a_line_is_refused_by_name() {
        let e = judge_call(
            "shell.run",
            &serde_json::json!({}),
            &networky(),
            Path::new("/home/someone"),
            bsd,
        )
        .unwrap_err();
        assert!(
            e.contains("line"),
            "the refusal must name the missing field: {e}"
        );
    }

    /// Fed one byte at a time, a thought must still leave no trace: the
    /// reader sees the answer and never the reasoning.
    #[test]
    fn a_thought_never_reaches_the_reader_however_it_is_chopped() {
        let whole = "<think>the capital is Paris</think>Paris.";
        for chunk in [1usize, 3, 7, 40] {
            let mut f = Folder::new();
            let mut out = String::new();
            let bytes: Vec<char> = whole.chars().collect();
            for piece in bytes.chunks(chunk) {
                out.push_str(&f.push(&piece.iter().collect::<String>()));
            }
            out.push_str(&f.finish());
            assert_eq!(out, "Paris.", "a thought leaked at chunk size {chunk}");
        }
    }

    /// A tag split across two deltas is still a tag. At a small chunk
    /// size this is the ordinary case, not the edge.
    #[test]
    fn a_tag_split_across_deltas_is_still_a_tag() {
        let mut f = Folder::new();
        let mut out = String::new();
        for d in ["<thi", "nk>hidden</thi", "nk>Hi"] {
            out.push_str(&f.push(d));
        }
        out.push_str(&f.finish());
        assert_eq!(out, "Hi");
    }

    /// Text before a thought is the answer and prints; the thought after
    /// it does not.
    #[test]
    fn what_is_not_a_thought_still_reaches_the_reader() {
        let mut f = Folder::new();
        let mut out = String::new();
        out.push_str(&f.push("Hello "));
        out.push_str(&f.push("<think>hidden</think>"));
        out.push_str(&f.push("world"));
        out.push_str(&f.finish());
        assert!(out.contains("Hello"), "the answer was swallowed: {out:?}");
        assert!(out.contains("world"), "the answer was swallowed: {out:?}");
        assert!(!out.contains("hidden"), "the thought leaked: {out:?}");
    }

    /// A model still thinking when the stream ends has said nothing, and
    /// a half-thought printed as an answer is how it looks like it lost
    /// its mind.
    #[test]
    fn a_thought_that_never_closed_prints_nothing() {
        let mut f = Folder::new();
        let mut out = String::new();
        out.push_str(&f.push("<think>still going"));
        out.push_str(&f.finish());
        assert_eq!(out, "");
    }

    fn run(role: &str, port: u16) -> Run {
        Run {
            role: role.into(),
            pid: 1,
            port,
            model: "qwen3-0.6b-gguf".into(),
            commit: "abc1234".into(),
            dtype: "f16".into(),
            log: PathBuf::from("/dev/null"),
            started: 0,
        }
    }

    /// The refusal has to name the command that fixes it. "no seat" sends
    /// a person to the source; `combs-tk up <model>` sends them to an
    /// engine.
    #[test]
    fn no_seat_names_the_command_that_makes_one() {
        let runs = vec![];
        let e = seat(&runs, "text", |_| Liveness::Up).unwrap_err();
        assert!(
            e.contains("combs-tk up"),
            "a refusal must name the command that makes a seat: {e}"
        );
    }

    /// A record whose engine is still coming up is not an error to be
    /// retried blindly — it is a wait, and the words say which.
    #[test]
    fn a_starting_engine_is_told_apart_from_a_missing_one() {
        let runs = vec![run("text", 9001)];
        let e = seat(&runs, "text", |_| Liveness::Starting).unwrap_err();
        assert!(
            e.contains("starting"),
            "an engine still coming up must be named as starting: {e}"
        );
        let stale = seat(&runs, "text", |_| Liveness::Stale).unwrap_err();
        assert!(
            stale.contains("stale") || stale.contains("no longer"),
            "a dead record must be named as stale: {stale}"
        );
    }

    /// The seat answering is the one whose role was asked for, not the
    /// first record on disk.
    #[test]
    fn the_seat_is_the_one_whose_role_was_asked_for() {
        let runs = vec![run("vision", 9001), run("text", 9002)];
        let got = seat(&runs, "text", |_| Liveness::Up).expect("a live text seat");
        assert_eq!(got.port, 9002, "the wrong seat answered");
    }

    /// The empty thought Qwen3 emits under `/no_think` must not reach the
    /// reader as a stutter at the head of the answer.
    #[test]
    fn an_empty_thought_leaves_no_trace_in_the_answer() {
        assert_eq!(fold_think("<think>\n\n</think>Paris."), "Paris.");
    }

    /// A thought with content is folded whole, and a thought still being
    /// thought folds everything after it — printing a half-thought as the
    /// answer is how a reasoning model looks like it lost its mind.
    #[test]
    fn a_thought_is_folded_whether_or_not_it_finished() {
        assert_eq!(
            fold_think("<think>the capital is Paris</think>Paris."),
            "Paris."
        );
        assert_eq!(fold_think("<think>still going"), "");
        assert_eq!(fold_think("no thought at all"), "no thought at all");
    }

    /// Truncation is the one finish reason that must not exit zero: the
    /// answer is half an answer and a script cannot tell by looking.
    #[test]
    fn a_truncated_answer_is_not_a_success() {
        assert_eq!(code_for(Some("stop")), 0);
        assert_eq!(code_for(Some("tool_calls")), 0);
        assert_eq!(
            code_for(Some("length")),
            3,
            "a truncated answer must exit non-zero, and 3 is truncation"
        );
    }

    /// The ledger prints what was computed, never the engine's own
    /// prefill rate — which counts cached tokens as work and reads over a
    /// thousand tokens a second on a warm session.
    #[test]
    fn the_ledger_names_the_work_that_was_actually_done() {
        let c = Cost {
            skipped: 17,
            computed: 1,
            prefill_ms: 10.7,
            computed_tok_s: Some(93.4),
            answer_tokens: 24,
            answer_ms: 251.0,
            decode_tok_s: Some(91.5),
        };
        let line = ledger(&c, Some(916.0));
        for want in ["prompt 18", "skipped 17", "computed 1", "24 tok"] {
            assert!(line.contains(want), "the ledger lost {want}: {line}");
        }
    }

    /// A one-token answer has no decode rate: the engine's figure covers
    /// tokens two to N. A zero there reads as "slow" when the truth is
    /// "not measurable".
    #[test]
    fn a_rate_that_cannot_be_measured_is_a_dash_and_not_a_zero() {
        let c = Cost {
            skipped: 0,
            computed: 18,
            prefill_ms: 140.0,
            computed_tok_s: Some(128.5),
            answer_tokens: 1,
            answer_ms: 0.0,
            decode_tok_s: None,
        };
        let line = ledger(&c, None);
        assert!(
            line.contains('—'),
            "an unmeasurable rate must read as a dash: {line}"
        );
        assert!(
            !line.contains("0.0 t/s"),
            "an unmeasurable rate must not read as zero: {line}"
        );
    }

    /// Run one is never the reading. It pays for a cold cache and a cold
    /// autotune, and it has lied in this estate's records four separate
    /// times.
    #[test]
    fn the_first_reading_is_dropped_when_there_is_another() {
        assert_eq!(warm(&[1.0, 2.0, 3.0]).to_vec(), vec![2.0, 3.0]);
        assert_eq!(warm(&[5.0]).to_vec(), vec![5.0], "one reading is all there is");
    }

    /// A spread past the band is a range, and the printer must be able to
    /// tell so without re-deriving it.
    #[test]
    fn a_spread_past_the_band_is_not_steady() {
        let tight = spread(&[100.0, 101.0, 99.0]).expect("three readings");
        assert_eq!(tight.median, 100.0);
        assert!(tight.steady(), "2% is not a range: {tight:?}");

        let wide = spread(&[100.0, 180.0, 140.0]).expect("three readings");
        assert_eq!(wide.median, 140.0);
        assert!(
            !wide.steady(),
            "a 57% spread is a range, not a number: {wide:?}"
        );
        assert_eq!(wide.low, 100.0);
        assert_eq!(wide.high, 180.0);
    }

    /// Nothing to measure is not a zero.
    #[test]
    fn no_readings_have_no_spread() {
        assert_eq!(spread(&[]), None);
    }

    /// One reading has nothing to agree with. A spread of zero percent
    /// across a single generation reads as "steady" when the truth is
    /// "unmeasured" — and against a live engine the lone reading is the
    /// cold one, which prefilled at 6.5 t/s where the warm pair read
    /// above 100. That is the reading that must never be headlined.
    #[test]
    fn one_reading_is_not_something_to_call_steady() {
        let alone = spread(&[93.2]).expect("one reading is still a reading");
        assert_eq!(alone.median, 93.2, "the number is still reported");
        assert!(
            !alone.steady(),
            "a single reading cannot be steady: {alone:?}"
        );
        let pair = spread(&[99.8, 98.6]).expect("two readings");
        assert!(pair.steady(), "two close readings are steady: {pair:?}");
    }

    /// The engine reports zero dispatches a token for a one-token
    /// answer. That is not "no work" — the prefill plainly ran — it is a
    /// figure that means nothing at this length, and printing it as a
    /// number invites someone to set it beside a real one.
    #[test]
    fn a_dispatch_count_of_zero_is_not_a_measurement() {
        let c = Cost {
            skipped: 3,
            computed: 14,
            prefill_ms: 2500.0,
            computed_tok_s: Some(5.6),
            answer_tokens: 1,
            answer_ms: 0.0,
            decode_tok_s: None,
        };
        let line = ledger(&c, Some(0.0));
        assert!(
            !line.contains("0 disp/tok"),
            "zero dispatches printed as a measurement: {line}"
        );
        assert!(
            line.contains("— disp/tok"),
            "an unmeasurable dispatch count must read as a dash: {line}"
        );
    }
}
