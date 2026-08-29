//! Every activity says what it is: which build, configured how, doing
//! what, for how long, ending in what.
//!
//! The rule this enforces is that a run must never be RECONSTRUCTED
//! from memory. An A/B whose arms only *appeared* to differ in one
//! variable produced a wrong conclusion here, and an f16 build has been
//! mistaken for the f32 one more than once — both are failures of the
//! record, not of the reasoning.
//!
//! It lives in the lowest crate on purpose: mounts, adapters and cache
//! evictions happen far below the CLI, and an observability layer that
//! only the top layer can reach leaves exactly the events that are
//! hardest to reconstruct unrecorded.
//!
//! ```text
//! [combs/image] 2026-08-29T02:19:42Z config device=IntegratedGpu dtype=f32
//! [combs/image] 2026-08-29T02:19:43Z turn.start op=generate size=512x512
//! [combs/image] 2026-08-29T02:19:43Z turn.end op=generate outcome=done took=PT38.4S
//! ```
//!
//! Volume is a setting, not a fact of life: `COMBS_PROVENANCE` selects
//! the depth and `COMBS_PROVENANCE_FORMAT=json` emits one object per
//! line for a supervising process to parse. No dependency is taken for
//! that — the JSON is assembled with the same escaping `progress` uses,
//! keeping this crate's dependency surface unchanged.

use std::sync::{OnceLock, RwLock};

/// Field pairs, so the readable line and the JSON object are built from
/// one source and cannot disagree about what ran.
pub type Fields<'a> = &'a [(&'a str, String)];

/// How much a surface says about itself (`COMBS_PROVENANCE`).
///
/// The default is neither silence nor a firehose: the startup lines
/// answer "which build, configured how?" — the question whose absence
/// has cost the most time — for two lines per process. Everything
/// per-turn and below is opt-in.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Level {
    /// Say nothing.
    Off,
    /// Build + configuration, once per process (default).
    Startup,
    /// Also every unit of work: parameters, outcome, duration.
    Turn,
    /// Also lifecycle events beneath a turn — mounts, adapters, caches.
    Debug,
}

impl Level {
    /// `None` for anything unrecognised, so a typo falls back to the
    /// default rather than silently choosing a depth nobody asked for.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "none" => Some(Level::Off),
            "startup" | "1" => Some(Level::Startup),
            "turn" | "2" => Some(Level::Turn),
            "debug" | "3" | "all" => Some(Level::Debug),
            _ => None,
        }
    }
}

/// The configured level, read once per process.
pub fn level() -> Level {
    static L: OnceLock<Level> = OnceLock::new();
    *L.get_or_init(|| {
        std::env::var("COMBS_PROVENANCE")
            .ok()
            .and_then(|v| Level::parse(&v))
            .unwrap_or(Level::Startup)
    })
}

/// Whether this depth of record is wanted.
pub fn enabled(required: Level) -> bool {
    level() >= required
}

fn json_format() -> bool {
    static J: OnceLock<bool> = OnceLock::new();
    *J.get_or_init(|| std::env::var("COMBS_PROVENANCE_FORMAT").as_deref() == Ok("json"))
}

/// The build's own account of itself, handed down from the binary that
/// generated it (this crate cannot know its own git commit; the CLI's
/// build script does). Registered once at startup.
struct Build {
    summary: String,
    manifest_json: String,
}

fn build_slot() -> &'static RwLock<Option<Build>> {
    static B: OnceLock<RwLock<Option<Build>>> = OnceLock::new();
    B.get_or_init(|| RwLock::new(None))
}

/// Register the build manifest so every later line can cite it.
/// `manifest_json` is a serialized object; the caller owns
/// serialization so this crate takes no JSON dependency.
pub fn set_build(summary: String, manifest_json: String) {
    if let Ok(mut slot) = build_slot().write() {
        *slot = Some(Build { summary, manifest_json });
    }
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn render(fields: Fields<'_>) -> String {
    fields
        .iter()
        .map(|(k, v)| {
            if v.contains(' ') {
                format!("{k}=\"{v}\"")
            } else {
                format!("{k}={v}")
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    crate::timefmt::rfc3339(secs)
}

/// The single exit point: every provenance line leaves through here, so
/// format, level and stream stay one decision rather than a convention
/// each call site has to remember.
fn emit(role: &str, kind: &str, fields: Fields<'_>, build: Option<&str>) {
    let at = now_rfc3339();
    if json_format() {
        let mut line = format!(
            "{{\"at\":\"{at}\",\"role\":\"{}\",\"kind\":\"{}\"",
            escape(role),
            escape(kind)
        );
        for (k, v) in fields {
            line.push_str(&format!(",\"{}\":\"{}\"", escape(k), escape(v)));
        }
        if let Some(manifest) = build {
            line.push_str(&format!(",\"build\":{manifest}"));
        }
        line.push('}');
        eprintln!("{line}");
    } else {
        let tail = render(fields);
        let sep = if tail.is_empty() { "" } else { " " };
        eprintln!("[combs/{role}] {at} {kind}{sep}{tail}");
    }
}

/// The first thing a process says: which binary this is. Emitted once
/// from `main` for every command, so the build line always precedes the
/// work — including work that starts before a surface is configured
/// (an engine mounts before the server that will serve it exists).
pub fn process(command: &str) {
    if !enabled(Level::Startup) {
        return;
    }
    let (summary, manifest) = read_build();
    emit(command, "build", &[("summary", summary)], manifest.as_deref());
}

/// How this surface is configured, once it knows.
pub fn startup(role: &str, config: Fields<'_>) {
    if !enabled(Level::Startup) {
        return;
    }
    emit(role, "config", config, None);
}

fn read_build() -> (String, Option<String>) {
    match build_slot().read() {
        Ok(slot) => match slot.as_ref() {
            Some(b) => (b.summary.clone(), Some(b.manifest_json.clone())),
            None => ("unregistered".to_string(), None),
        },
        Err(_) => ("unregistered".to_string(), None),
    }
}

/// A lifecycle event underneath a turn — a model mounted, an adapter
/// fused, a cache session evicted. Silent unless that depth was asked
/// for, so callers may place them freely.
pub fn event(role: &str, what: &str, fields: Fields<'_>) {
    if !enabled(Level::Debug) {
        return;
    }
    emit(role, what, fields, None);
}

/// Open a unit of work. The returned guard closes the record with the
/// outcome and duration — and closes it even if the work unwinds, so a
/// turn cannot vanish from the log without saying so.
pub fn turn(role: &str, op: &'static str, params: Fields<'_>) -> Turn {
    if enabled(Level::Turn) {
        let mut f = vec![("op", op.to_string())];
        f.extend(params.iter().map(|(k, v)| (*k, v.clone())));
        emit(role, "turn.start", &f, None);
    }
    Turn { role: role.to_string(), op, started: std::time::Instant::now(), closed: false }
}

/// The open half of a turn record.
pub struct Turn {
    role: String,
    op: &'static str,
    started: std::time::Instant,
    closed: bool,
}

impl Turn {
    fn close(&mut self, outcome: &str, fields: Fields<'_>) {
        self.closed = true;
        if !enabled(Level::Turn) {
            return;
        }
        let ms = self.started.elapsed().as_millis();
        let mut f = vec![
            ("op", self.op.to_string()),
            ("outcome", outcome.to_string()),
            ("took", crate::timefmt::iso_duration(ms)),
            ("ms", ms.to_string()),
        ];
        f.extend(fields.iter().map(|(k, v)| (*k, v.clone())));
        emit(&self.role, "turn.end", &f, None);
    }

    /// Work finished: what came out.
    pub fn ok(mut self, output: Fields<'_>) {
        self.close("done", output);
    }

    /// Work failed: why, in the same shape as a success.
    pub fn failed(mut self, error: &str) {
        self.close("failed", &[("error", error.to_string())]);
    }
}

impl Drop for Turn {
    fn drop(&mut self) {
        if !self.closed {
            // An unfinished unit of work is itself a finding; silence
            // would hide exactly the runs worth looking at.
            self.close("abandoned", &[]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_order_and_parse() {
        assert!(Level::Off < Level::Startup);
        assert!(Level::Startup < Level::Turn);
        assert!(Level::Turn < Level::Debug);
        for (raw, want) in [
            ("off", Level::Off),
            ("STARTUP", Level::Startup),
            (" turn ", Level::Turn),
            ("all", Level::Debug),
        ] {
            assert_eq!(Level::parse(raw), Some(want), "{raw}");
        }
        assert_eq!(Level::parse("chatty"), None);
    }

    #[test]
    fn fields_with_spaces_stay_one_token() {
        let f = [("a", "1".to_string()), ("b", "two words".to_string())];
        assert_eq!(render(&f), "a=1 b=\"two words\"");
    }

    /// The JSON form must survive the characters that actually appear
    /// in these records — quotes and backslashes in paths and errors.
    #[test]
    fn json_escaping_covers_paths_and_messages() {
        assert_eq!(escape(r#"say "hi"#), r#"say \"hi"#);
        assert_eq!(escape(r"C:\models"), r"C:\\models");
        assert!(escape("line\nbreak").contains("\\u000a"));
    }

    /// A turn closes exactly once, whether finished or dropped.
    #[test]
    fn a_dropped_turn_still_closes() {
        drop(turn("test", "unit", &[("a", "1".to_string())]));
        let mut t = turn("test", "unit", &[]);
        t.close("done", &[]);
        assert!(t.closed);
        std::mem::forget(t);
    }

    #[test]
    fn build_registration_round_trips() {
        set_build("combs test f32".into(), "{\"serving_dtype\":\"f32\"}".into());
        let slot = build_slot().read().unwrap();
        let b = slot.as_ref().expect("registered");
        assert_eq!(b.summary, "combs test f32");
        assert!(b.manifest_json.contains("serving_dtype"));
    }
}
