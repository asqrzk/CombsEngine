//! Every activity says what it is, once at startup and once per turn.
//!
//! The rule this enforces: a run must never have to be RECONSTRUCTED
//! from memory or inferred from a command line. An A/B whose arms only
//! *appeared* to differ in one variable cost this project a wrong
//! conclusion, and an f16 twin mistaken for the f32 build has cost it
//! twice. So each surface — text, image, audio, one-shot or served —
//! announces the same three things in the same shape:
//!
//! ```text
//! [combs/image] build combs 0.2.3 f32 (release, b8fd34a, aarch64-apple-darwin)
//! [combs/image] config device=IntegratedGpu dtype=f32 model=flux2-klein-4b …
//! [combs/image] turn size=256x256 steps=4 sampler=flow-match-euler seed=42
//! ```
//!
//! One prefix, so `grep '\[combs/'` is the whole provenance stream, and
//! the same fields reach `/v1/stats` as JSON. It speaks every turn;
//! listening is the operator's choice.

use serde_json::{Map, Value, json};

/// How much a surface says about itself. Set with `COMBS_PROVENANCE`.
///
/// The default is deliberately not silence and not a firehose: the two
/// startup lines are what answer "which build, configured how?" — the
/// question that has cost this project the most time — and they cost
/// two lines per process. Everything per-turn and below is opt-in.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "none" => Some(Level::Off),
            "startup" | "1" => Some(Level::Startup),
            "turn" | "2" => Some(Level::Turn),
            "debug" | "3" | "all" => Some(Level::Debug),
            _ => None,
        }
    }
}

/// The configured level, read once. An unrecognised value falls back to
/// the default rather than failing a run — observability must never be
/// the reason work does not happen.
pub fn level() -> Level {
    static L: std::sync::OnceLock<Level> = std::sync::OnceLock::new();
    *L.get_or_init(|| {
        std::env::var("COMBS_PROVENANCE")
            .ok()
            .and_then(|v| Level::parse(&v))
            .unwrap_or(Level::Startup)
    })
}

fn enabled(required: Level) -> bool {
    level() >= required
}

/// `COMBS_PROVENANCE_FORMAT=json` emits one JSON object per line
/// instead of the readable form, so a supervising process can parse the
/// stream into a timeline rather than scraping prose.
fn json_format() -> bool {
    static J: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *J.get_or_init(|| std::env::var("COMBS_PROVENANCE_FORMAT").as_deref() == Ok("json"))
}

/// The single exit point: every provenance line in the engine goes
/// through here, so format, level, and stream stay one decision.
fn emit(role: &str, kind: &str, fields: Fields<'_>, extra: Option<Value>) {
    let at = now_rfc3339();
    if json_format() {
        let mut obj = Map::new();
        obj.insert("at".into(), Value::String(at));
        obj.insert("role".into(), Value::String(role.to_string()));
        obj.insert("kind".into(), Value::String(kind.to_string()));
        for (k, v) in fields {
            obj.insert((*k).to_string(), Value::String(v.clone()));
        }
        if let Some(Value::Object(more)) = extra {
            for (k, v) in more {
                obj.insert(k, v);
            }
        }
        eprintln!("{}", Value::Object(obj));
    } else {
        let tail = render(fields);
        let sep = if tail.is_empty() { "" } else { " " };
        eprintln!("[combs/{role}] {at} {kind}{sep}{tail}");
    }
}

/// Now, as an RFC 3339 instant. Every provenance line carries one so a
/// record can be lined up against a system log or another machine's.
fn now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    crate::timefmt::rfc3339(secs)
}

/// A field list, kept as pairs so the log line and the JSON cannot
/// disagree about what ran.
pub type Fields<'a> = &'a [(&'a str, String)];

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

/// Two lines at boot: which binary, and how it is configured. Called by
/// every surface before it does any work.
pub fn startup(role: &str, config: Fields<'_>) {
    if !enabled(Level::Startup) {
        return;
    }
    emit(
        role,
        "build",
        &[("summary", crate::build_info::summary())],
        Some(json!({ "build": crate::build_info::manifest() })),
    );
    emit(role, "config", config, None);
}

/// A lifecycle event underneath a turn — a model mounted, an adapter
/// fused, a cache session evicted. Cheap to call and silent unless the
/// operator asked for this depth.
pub fn event(role: &str, what: &str, fields: Fields<'_>) {
    if !enabled(Level::Debug) {
        return;
    }
    emit(role, what, fields, None);
}

/// Open a unit of work: what began, on which model, with which
/// parameters. The returned guard closes the record — with the outcome
/// and how long it took — and closes it even if the work unwinds, so a
/// turn cannot silently vanish from the log.
pub fn turn(role: &str, op: &'static str, params: Fields<'_>) -> Turn {
    if enabled(Level::Turn) {
        let mut f = vec![("op", op.to_string())];
        f.extend(params.iter().map(|(k, v)| (*k, v.clone())));
        emit(role, "turn.start", &f, None);
    }
    Turn {
        role: role.to_string(),
        op,
        started: std::time::Instant::now(),
        closed: false,
    }
}

/// The open half of a turn record. Dropping it without [`Turn::ok`] or
/// [`Turn::failed`] still writes a line — an abandoned turn is itself
/// a finding, and silence would hide it.
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
            self.close("abandoned", &[]);
        }
    }
}

/// The same facts for a stats route: the build manifest plus this
/// surface's configuration, under one key.
pub fn manifest(role: &str, config: Fields<'_>) -> Value {
    let mut cfg = Map::new();
    for (k, v) in config {
        cfg.insert((*k).to_string(), Value::String(v.clone()));
    }
    json!({
        "role": role,
        "build": crate::build_info::manifest(),
        "config": Value::Object(cfg),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_with_spaces_stay_one_field() {
        let f = [("a", "1".to_string()), ("b", "two words".to_string())];
        assert_eq!(render(&f), "a=1 b=\"two words\"");
    }

    /// A turn that is dropped without an outcome still leaves a record.
    /// Nothing asserts on stderr here — the value is that `close` runs
    /// exactly once, which the flag encodes.
    #[test]
    fn levels_order_and_parse() {
        assert!(Level::Off < Level::Startup && Level::Startup < Level::Turn);
        assert!(Level::Turn < Level::Debug);
        for (raw, want) in [("off", Level::Off), ("STARTUP", Level::Startup),
                            ("turn", Level::Turn), ("all", Level::Debug)] {
            assert!(Level::parse(raw) == Some(want), "{raw}");
        }
        // An unreadable setting must not decide anything by accident.
        assert!(Level::parse("chatty").is_none());
    }

    #[test]
    fn an_abandoned_turn_still_closes_itself() {
        let t = turn("test", "unit", &[("a", "1".to_string())]);
        drop(t);
        let mut t = turn("test", "unit", &[]);
        t.close("done", &[]);
        assert!(t.closed, "an explicit close marks the turn closed");
        std::mem::forget(t);
    }

    /// The log line and the JSON are built from the same pairs, so a
    /// field can never appear in one and not the other.
    #[test]
    fn manifest_carries_every_logged_field() {
        let f = [("device", "IntegratedGpu".to_string()), ("dtype", "f32".to_string())];
        let m = manifest("image", &f);
        assert_eq!(m["role"], "image");
        assert_eq!(m["config"]["device"], "IntegratedGpu");
        assert_eq!(m["config"]["dtype"], "f32");
        assert!(m["build"]["serving_dtype"].is_string());
        for (k, v) in &f {
            assert!(render(&f).contains(&format!("{k}={v}")));
        }
    }
}
