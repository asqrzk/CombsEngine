//! Chat-template rendering: evaluates the checkpoint's own Jinja template
//! (via minijinja) under the transformers contract — context is
//! `{messages, add_generation_prompt, bos_token, eos_token}` with
//! `raise_exception` and `strftime_now` available — so llama3/qwen/gemma
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
    pub fn render(&self, messages: &[(String, String)]) -> Result<String, String> {
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

        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|(role, content)| {
                serde_json::json!({ "role": role, "content": content })
            })
            .collect();

        env.render_str(
            &self.source,
            context! {
                messages => msgs,
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
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let err = t.render(&[("user".into(), "hi".into())]).unwrap_err();
        assert!(err.contains("roles must alternate"), "got: {err}");
    }
}
