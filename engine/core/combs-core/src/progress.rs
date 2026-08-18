//! Structured load/pull progress on stderr, for supervising processes.
//!
//! Gated on `COMBS_PROGRESS=json` so interactive CLI output is unchanged;
//! when enabled, each event is one line: `@combs {"ev":...}`. Stderr is
//! the only channel that exists while `combs serve` is still loading
//! weights (the HTTP port binds after the load completes), which is why
//! this is not an endpoint.

use std::sync::OnceLock;

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("COMBS_PROGRESS").is_ok_and(|v| v == "json"))
}

/// Escapes `"` and `\` plus control chars; file names are the only
/// free-form strings emitted, so full JSON string escaping is overkill.
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

pub fn load(stage: &str, done: Option<u64>, total: Option<u64>, ms: Option<u64>) {
    if !enabled() {
        return;
    }
    let mut line = format!("@combs {{\"ev\":\"load\",\"stage\":\"{}\"", escape(stage));
    if let Some(d) = done {
        line.push_str(&format!(",\"done\":{d}"));
    }
    if let Some(t) = total {
        line.push_str(&format!(",\"total\":{t}"));
    }
    if let Some(m) = ms {
        line.push_str(&format!(",\"ms\":{m}"));
    }
    line.push('}');
    eprintln!("{line}");
}

pub fn pull(file: &str, done: u64, total: u64) {
    if !enabled() {
        return;
    }
    eprintln!(
        "@combs {{\"ev\":\"pull\",\"file\":\"{}\",\"done\":{done},\"total\":{total}}}",
        escape(file)
    );
}

#[cfg(test)]
mod tests {
    use super::escape;

    #[test]
    fn escape_handles_quotes_backslashes_controls() {
        assert_eq!(escape(r#"a"b\c"#), r#"a\"b\\c"#);
        assert_eq!(escape("x\ny"), "x\\u000ay");
        assert_eq!(escape("plain-file.safetensors"), "plain-file.safetensors");
    }
}
