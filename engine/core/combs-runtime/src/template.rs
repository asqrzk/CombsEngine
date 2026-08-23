//! Chat-template rendering: evaluates the checkpoint's own Jinja template
//! (via minijinja) under the transformers contract — context is
//! `{messages, tools, add_generation_prompt, bos_token, eos_token}` with
//! `raise_exception`, `strftime_now`, and transformers' `tojson` filter
//! (plain `json.dumps` semantics: UTF-8 passthrough, `", "`/`": "`
//! separators, optional `indent`, NO HTML escaping) — so llama3/qwen/gemma
//! checkpoints get their real prompt format instead of a sniffed builtin.
//!
//! The template is re-parsed per render (`render_str`): parsing a few KB of
//! Jinja is microseconds against a generation, it needs no `'static`
//! template storage, and a fresh `Environment` per call keeps `&self`
//! renders trivially thread-safe across serve's request threads.

use minijinja::{Environment, Error as JinjaError, ErrorKind, context};

/// A checkpoint-provided chat template plus the token strings its context
/// exposes. Falls back at the call site (`Engine::wrap_chat`) when absent
/// or when rendering fails.
pub struct ChatTemplate {
    source: String,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    pub fn new(source: String, bos_token: String, eos_token: String) -> Self {
        Self { source, bos_token, eos_token }
    }

    /// Renders `(role, content)` messages with the assistant turn left open
    /// (`add_generation_prompt: true`). Errors surface the minijinja message
    /// (including template-raised `raise_exception` texts).
    pub fn render(
        &self,
        messages: &[crate::ChatMessage],
        tools: Option<&serde_json::Value>,
    ) -> Result<String, String> {
        let mut env = Environment::new();
        // transformers compiles templates with trim_blocks + lstrip_blocks.
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<String, JinjaError> {
                Err(JinjaError::new(ErrorKind::InvalidOperation, msg))
            },
        );
        env.add_function("strftime_now", |fmt: String| strftime_now(&fmt));
        env.add_filter("tojson", tojson_filter);

        let msgs: Vec<serde_json::Value> =
            messages.iter().map(|m| m.to_template_value()).collect();
        // transformers passes `tools=None` when absent; jinja `none` keeps
        // `{% if tools %}` / `tools is not none` branches false.
        let tools_value = tools.cloned().unwrap_or(serde_json::Value::Null);

        env.render_str(
            &self.source,
            context! {
                messages => msgs,
                tools => tools_value,
                add_generation_prompt => true,
                bos_token => self.bos_token,
                eos_token => self.eos_token,
            },
        )
        .map_err(|e| e.to_string())
    }
}

/// `strftime_now` as HF templates use it (llama-3.x: `%d %b %Y`), over
/// std::time only. `COMBS_CHAT_DATE` overrides the output verbatim so
/// harmony tests and reproducible runs don't depend on the wall clock.
fn strftime_now(fmt: &str) -> String {
    if let Ok(pinned) = std::env::var("COMBS_CHAT_DATE") {
        return pinned;
    }
    let secs = crate::time::SystemTime::now()
        .duration_since(crate::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, m, d) = civil_from_days((secs / 86400) as i64);
    let (hh, mm, ss) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);
    const ABBR: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    const FULL: [&str; 12] = [
        "January", "February", "March", "April", "May", "June", "July", "August",
        "September", "October", "November", "December",
    ];
    let mi = (m as usize).saturating_sub(1).min(11);
    fmt.replace("%d", &format!("{d:02}"))
        .replace("%m", &format!("{m:02}"))
        .replace("%B", FULL[mi])
        .replace("%b", ABBR[mi])
        .replace("%Y", &y.to_string())
        .replace("%H", &format!("{hh:02}"))
        .replace("%M", &format!("{mm:02}"))
        .replace("%S", &format!("{ss:02}"))
}

/// Days-since-epoch → (year, month, day), Howard Hinnant's civil algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// transformers' `tojson` filter: python `json.dumps` semantics —
/// `", "`/`": "` separators when compact, `indent=N` pretty printing, raw
/// UTF-8 (`ensure_ascii=False`), no HTML escaping, insertion order (maps
/// arrive alphabetized via serde_json, matching how references are
/// authored).
fn tojson_filter(
    value: minijinja::Value,
    kwargs: minijinja::value::Kwargs,
) -> Result<String, JinjaError> {
    let json: serde_json::Value = serde_json::to_value(&value).map_err(|e| {
        JinjaError::new(ErrorKind::InvalidOperation, format!("tojson: {e}"))
    })?;
    let indent: Option<usize> = kwargs.get("indent")?;
    kwargs.assert_all_used()?;
    let mut out = String::new();
    write_python_json(&json, indent, 0, &mut out);
    Ok(out)
}

/// Serializes like python's `json.dumps`: compact form uses `", "` and
/// `": "`; `indent` switches to newline-per-item with `": "` after keys
/// and `,` line separators — byte-compatible with the jinja2 reference
/// renders.
fn write_python_json(v: &serde_json::Value, indent: Option<usize>, level: usize, out: &mut String) {
    match v {
        serde_json::Value::Array(items) if !items.is_empty() => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if indent.is_none() {
                        out.push(' ');
                    }
                }
                if let Some(n) = indent {
                    out.push('\n');
                    out.push_str(&" ".repeat(n * (level + 1)));
                }
                write_python_json(item, indent, level + 1, out);
            }
            if let Some(n) = indent {
                out.push('\n');
                out.push_str(&" ".repeat(n * level));
            }
            out.push(']');
        }
        serde_json::Value::Object(map) if !map.is_empty() => {
            out.push('{');
            for (i, (k, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if indent.is_none() {
                        out.push(' ');
                    }
                }
                if let Some(n) = indent {
                    out.push('\n');
                    out.push_str(&" ".repeat(n * (level + 1)));
                }
                out.push_str(&serde_json::to_string(k).unwrap());
                out.push_str(": ");
                write_python_json(item, indent, level + 1, out);
            }
            if let Some(n) = indent {
                out.push('\n');
                out.push_str(&" ".repeat(n * level));
            }
            out.push('}');
        }
        // Scalars and empty containers: serde_json compact form matches
        // python exactly ("[]", "{}", strings as UTF-8 with the same
        // escape set, true/false/null, integer formatting).
        other => out.push_str(&serde_json::to_string(other).unwrap()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tojson_matches_python_dumps() {
        let v: serde_json::Value = serde_json::json!({
            "b": [1, 2], "a": "héllo <tag>", "empty": {}
        });
        let mut compact = String::new();
        write_python_json(&v, None, 0, &mut compact);
        // serde_json maps are BTree-ordered: a, b, empty.
        assert_eq!(compact, r#"{"a": "héllo <tag>", "b": [1, 2], "empty": {}}"#);
        let mut pretty = String::new();
        write_python_json(&v, Some(4), 0, &mut pretty);
        assert_eq!(
            pretty,
            "{\n    \"a\": \"héllo <tag>\",\n    \"b\": [\n        1,\n        2\n    ],\n    \"empty\": {}\n}"
        );
    }

    #[test]
    fn civil_dates_are_correct() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        // 2026-08-11 = 20676 days after epoch.
        assert_eq!(civil_from_days(20_676), (2026, 8, 11));
    }

    #[test]
    fn render_reports_template_raises() {
        let t = ChatTemplate::new(
            "{{ raise_exception('roles must alternate') }}".to_string(),
            String::new(),
            String::new(),
        );
        let err = t
            .render(&[crate::ChatMessage::text("user", "hi")], None)
            .unwrap_err();
        assert!(err.contains("roles must alternate"), "got: {err}");
    }
}
