//! The chat contract as pure functions: what a request carries, what a
//! reply's usage means, and how that usage splits into the two costs a
//! person actually waits for.
//!
//! Every rule here was measured against a live engine before it was
//! written, and two exist only because of what the measurement showed.
//! The engine reads its request as an untyped value, so a wrongly typed
//! field is dropped in silence rather than refused — the client checks
//! first and refuses by name. And the engine's own `prefill_tok_s`
//! divides the WHOLE prompt by the time to first token, cached tokens
//! included, so on a warm turn it overstates by whatever the cache
//! saved: sixteen of seventeen tokens skipped read as 1,336 tokens a
//! second of "prefill". The split below counts only what was computed.

use serde_json::{json, Value};

/// One turn of a transcript.
///
/// A tool round-trip needs three things the plain shape cannot carry:
/// the calls an assistant turn asked for, and — on the result coming
/// back — which call it answers and which tool answered it. The engine
/// reads all three (`tool_calls`, `tool_call_id`, `name`), and the chat
/// template renders them, so leaving them out would make a tool result
/// arrive as anonymous prose.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// An assistant turn's calls, in the engine's own wire shape.
    pub tool_calls: Option<Value>,
    /// On a `tool` result: the call it answers.
    pub tool_call_id: Option<String>,
    /// On a `tool` result: the tool that answered.
    pub name: Option<String>,
}

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Self { role: "system".into(), content: text.into(), ..Self::default() }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self { role: "user".into(), content: text.into(), ..Self::default() }
    }

    /// The assistant turn that asked for tools.
    ///
    /// Content is an empty STRING and never null: the engine reads
    /// `content` with a serde default, so absent or `""` is fine while a
    /// null is a 400 — a scar this estate has already paid for once.
    pub fn assistant_calls(calls: Value) -> Self {
        Self {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: Some(calls),
            ..Self::default()
        }
    }

    /// What one tool said, tied to the call that asked for it.
    pub fn tool_result(id: impl Into<String>, name: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: text.into(),
            tool_call_id: Some(id.into()),
            name: Some(name.into()),
            ..Self::default()
        }
    }
}

/// What a caller asks for: the fields the engine reads, and none it
/// ignores.
#[derive(Debug, Clone, Default)]
pub struct Ask {
    pub messages: Vec<Message>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub session: Option<String>,
    pub stream: bool,
    pub tools: Option<Value>,
    pub response_format: Option<Value>,
}

/// The longest session name the engine keeps. Past it the engine
/// truncates in silence, and two conversations can then share one
/// cache — so the client refuses before the wire.
pub const SESSION_MAX: usize = 64;

/// The request body, or the reason it was not built. A field the caller
/// did not set is absent, never null: the engine's defaults are the
/// model's own, and a null would replace them with nothing.
pub fn chat_body(ask: &Ask) -> Result<Value, String> {
    if ask.messages.is_empty() {
        return Err("nothing to ask: `messages` is empty".into());
    }
    if let Some(s) = &ask.session {
        let n = s.chars().count();
        if n > SESSION_MAX {
            return Err(format!(
                "the session name is {n} characters; the engine keeps {SESSION_MAX} and would truncate the rest in silence"
            ));
        }
    }
    if let Some(t) = ask.temperature {
        if !t.is_finite() || t < 0.0 {
            return Err(format!("temperature must be a finite number at or above zero, not {t}"));
        }
    }
    if ask.max_tokens == Some(0) {
        return Err("max_tokens 0 asks for nothing; omit it for the model's own default".into());
    }
    let messages: Vec<Value> = ask
        .messages
        .iter()
        .map(|m| {
            let mut v = json!({ "role": m.role, "content": m.content });
            if let Some(c) = &m.tool_calls {
                v["tool_calls"] = c.clone();
            }
            if let Some(id) = &m.tool_call_id {
                v["tool_call_id"] = json!(id);
            }
            if let Some(n) = &m.name {
                v["name"] = json!(n);
            }
            v
        })
        .collect();
    let mut body = json!({
        "model": "combs",
        "messages": messages,
        "stream": ask.stream,
    });
    if let Some(n) = ask.max_tokens {
        body["max_tokens"] = json!(n);
    }
    if let Some(t) = ask.temperature {
        body["temperature"] = json!(t);
    }
    if let Some(s) = &ask.session {
        body["session_id"] = json!(s);
    }
    if let Some(t) = &ask.tools {
        body["tools"] = t.clone();
    }
    if let Some(r) = &ask.response_format {
        body["response_format"] = r.clone();
    }
    Ok(body)
}

/// What the engine reported about one generation, read from its own
/// `usage` object and nothing else.
#[derive(Debug, Clone, PartialEq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// The prefix served from the session's cache: tokens never prefilled.
    pub cached_tokens: u64,
    /// To the first token: prefill, plus the first readback and sample.
    pub ttft_ms: f64,
    pub total_ms: f64,
    /// The engine's own decode rate, over tokens two to N.
    pub decode_tok_s: f64,
    pub dispatches_per_token: Option<f64>,
    pub session_id: Option<String>,
}

/// None when the object lacks what the split needs: a reading built on
/// guessed zeros is worse than none.
pub fn parse_usage(u: &Value) -> Option<Usage> {
    let timing = u.get("timing")?;
    let cached = u
        .get("cache")
        .and_then(|c| c.get("cached_tokens"))
        .and_then(Value::as_u64)
        .or_else(|| {
            u.get("prompt_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .and_then(Value::as_u64)
        })?;
    Some(Usage {
        prompt_tokens: u.get("prompt_tokens")?.as_u64()?,
        completion_tokens: u.get("completion_tokens")?.as_u64()?,
        cached_tokens: cached,
        ttft_ms: timing.get("ttft_ms")?.as_f64()?,
        total_ms: timing.get("total_ms")?.as_f64()?,
        decode_tok_s: timing.get("decode_tok_s")?.as_f64()?,
        dispatches_per_token: u
            .get("dispatch")
            .and_then(|d| d.get("per_token"))
            .and_then(Value::as_f64),
        session_id: u.get("session_id").and_then(Value::as_str).map(str::to_string),
    })
}

/// The two costs a person waits for, and what each was made of.
#[derive(Debug, Clone, PartialEq)]
pub struct Cost {
    /// Prompt tokens the cache already held.
    pub skipped: u64,
    /// Prompt tokens actually prefilled.
    pub computed: u64,
    /// Wall time to the first token. Includes the first readback and
    /// sample, and the reading says so where it is printed.
    pub prefill_ms: f64,
    /// Computed tokens over the prefill wall — never the engine's own
    /// rate, which counts skipped tokens as work. None when nothing was
    /// computed, because a rate over zero tokens is not a rate.
    pub computed_tok_s: Option<f64>,
    pub answer_tokens: u64,
    /// Wall time after the first token.
    pub answer_ms: f64,
    /// The engine's decode rate, which covers tokens two to N — so a
    /// one-token answer has none, rather than a zero that reads as slow.
    pub decode_tok_s: Option<f64>,
}

pub fn cost(u: &Usage) -> Cost {
    let skipped = u.cached_tokens.min(u.prompt_tokens);
    let computed = u.prompt_tokens - skipped;
    let prefill_ms = u.ttft_ms;
    let computed_tok_s = if computed > 0 && prefill_ms > 0.0 {
        Some(computed as f64 / (prefill_ms / 1000.0))
    } else {
        None
    };
    Cost {
        skipped,
        computed,
        prefill_ms,
        computed_tok_s,
        answer_tokens: u.completion_tokens,
        answer_ms: (u.total_ms - u.ttft_ms).max(0.0),
        decode_tok_s: if u.completion_tokens > 1 { Some(u.decode_tok_s) } else { None },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ask(text: &str) -> Ask {
        Ask { messages: vec![Message::user(text)], ..Ask::default() }
    }

    /// An assistant turn that asked for tools must carry them, or the
    /// template renders a turn that called nothing and the model sees
    /// its own request vanish.
    #[test]
    fn an_assistant_turn_carries_the_calls_it_made() {
        let calls = json!([{ "id": "call_0", "type": "function",
            "function": { "name": "net.routes", "arguments": "{}" } }]);
        let a = Ask {
            messages: vec![Message::user("what routes"), Message::assistant_calls(calls.clone())],
            ..Ask::default()
        };
        let b = chat_body(&a).expect("builds");
        assert_eq!(b["messages"][1]["tool_calls"], calls, "the calls must reach the engine");
        assert_eq!(
            b["messages"][1]["content"], json!(""),
            "content is an empty string, never null: the engine 400s on a null"
        );
    }

    /// A result must name the call it answers. Without the id the model
    /// gets an anonymous block of text and cannot tell which probe spoke.
    #[test]
    fn a_tool_result_names_the_call_it_answers() {
        let a = Ask {
            messages: vec![Message::tool_result("call_0", "net.routes", "default utun4")],
            ..Ask::default()
        };
        let b = chat_body(&a).expect("builds");
        assert_eq!(b["messages"][0]["role"], "tool");
        assert_eq!(b["messages"][0]["tool_call_id"], "call_0");
        assert_eq!(b["messages"][0]["name"], "net.routes");
        assert_eq!(b["messages"][0]["content"], "default utun4");
    }

    /// The body carries what the engine reads and nothing it ignores.
    /// `stream_options` is not sent: usage arrives on the final frame
    /// without it, and the engine would drop the field in silence.
    #[test]
    fn a_body_carries_only_what_the_engine_reads() {
        let mut a = ask("hi");
        a.max_tokens = Some(24);
        a.temperature = Some(0.0);
        a.session = Some("tk:main".into());
        a.stream = true;
        let b = chat_body(&a).expect("a plain ask builds");
        assert_eq!(b["messages"][0]["role"], "user");
        assert_eq!(b["messages"][0]["content"], "hi");
        assert_eq!(b["max_tokens"], 24);
        assert_eq!(b["temperature"], 0.0);
        assert_eq!(b["session_id"], "tk:main");
        assert_eq!(b["stream"], true);
        assert!(b.get("stream_options").is_none(), "stream_options is not a field the engine reads");
        assert!(b.get("n").is_none(), "n is never sent: the multi-choice road drops timing");
    }

    /// A field the caller did not set is absent, not sent as null: the
    /// engine's defaults are the model's own, and a null would replace
    /// them with nothing.
    #[test]
    fn an_unset_field_is_absent_not_null() {
        let b = chat_body(&ask("hi")).expect("builds");
        for k in ["max_tokens", "temperature", "session_id", "tools", "response_format"] {
            assert!(b.get(k).is_none(), "{k} should be absent when unset, got {}", b[k]);
        }
    }

    /// The engine truncates a long session name in silence, and two
    /// conversations could then share one cache. The client refuses
    /// first, naming the limit.
    #[test]
    fn a_session_name_past_the_limit_is_refused_by_name() {
        let mut a = ask("hi");
        a.session = Some("s".repeat(SESSION_MAX + 1));
        let err = chat_body(&a).expect_err("a name past the limit must be refused");
        assert!(err.contains("64"), "the refusal names the limit: {err}");
        assert!(err.contains("session"), "the refusal names the field: {err}");
    }

    /// An empty transcript is a 400 from the engine; the client says so
    /// before spending a connection on it.
    #[test]
    fn an_empty_transcript_is_refused_before_the_wire() {
        let err = chat_body(&Ask::default()).expect_err("nothing to ask");
        assert!(err.contains("messages"), "the refusal names the field: {err}");
    }

    /// The usage object as one engine actually wrote it, verbatim.
    const USAGE: &str = r#"{"cache":{"cached_tokens":16,"pages_used":2},"completion_tokens":8,
        "dispatch":{"decode_total":7648,"per_token":956.0,"prefill_total":2385,"submits_per_token":61.0},
        "prompt_tokens":17,"prompt_tokens_details":{"cached_tokens":16},"session_id":"tk:probe",
        "timing":{"decode_tok_s":86.2,"prefill_tok_s":1336.0,"total_ms":100.0,"ttft_ms":13.0},"total_tokens":25}"#;

    #[test]
    fn usage_is_read_from_the_engines_own_fields() {
        let v: Value = serde_json::from_str(USAGE).unwrap();
        let u = parse_usage(&v).expect("a real usage object parses");
        assert_eq!(u.prompt_tokens, 17);
        assert_eq!(u.completion_tokens, 8);
        assert_eq!(u.cached_tokens, 16);
        assert_eq!(u.ttft_ms, 13.0);
        assert_eq!(u.total_ms, 100.0);
        assert_eq!(u.decode_tok_s, 86.2);
        assert_eq!(u.dispatches_per_token, Some(956.0));
        assert_eq!(u.session_id.as_deref(), Some("tk:probe"));
    }

    /// A usage object missing what the split needs is not a usage
    /// object; a reading built on guessed zeros is worse than none.
    #[test]
    fn a_usage_without_timing_is_not_a_usage() {
        let v: Value = serde_json::from_str(r#"{"prompt_tokens":3,"completion_tokens":1}"#).unwrap();
        assert!(parse_usage(&v).is_none());
    }

    /// Sixteen of seventeen tokens came from the cache. The engine's
    /// own rate calls that 1,336 tokens a second of prefill; the honest
    /// reading is one token computed, sixteen skipped.
    #[test]
    fn prefill_cost_counts_only_what_was_computed() {
        let v: Value = serde_json::from_str(USAGE).unwrap();
        let c = cost(&parse_usage(&v).unwrap());
        assert_eq!(c.skipped, 16);
        assert_eq!(c.computed, 1);
        assert_eq!(c.prefill_ms, 13.0);
        assert_eq!(c.answer_tokens, 8);
        assert_eq!(c.answer_ms, 87.0);
        let rate = c.computed_tok_s.expect("one token was computed, so a rate exists");
        assert!((rate - 1000.0 / 13.0).abs() < 1e-6, "one token in 13 ms, not 1336 t/s: got {rate}");
        assert_eq!(c.decode_tok_s, Some(86.2));
    }

    /// Nothing computed means no rate — not an infinite one, not a zero
    /// that reads as slow.
    #[test]
    fn a_fully_cached_prompt_has_no_prefill_rate() {
        let u = Usage {
            prompt_tokens: 17,
            completion_tokens: 8,
            cached_tokens: 17,
            ttft_ms: 9.0,
            total_ms: 80.0,
            decode_tok_s: 90.0,
            dispatches_per_token: None,
            session_id: None,
        };
        let c = cost(&u);
        assert_eq!(c.computed, 0);
        assert_eq!(c.computed_tok_s, None);
    }

    /// The engine's decode rate covers tokens two to N, so a one-token
    /// answer has no rate. Printing its zero would read as a stall.
    #[test]
    fn a_one_token_answer_has_no_decode_rate() {
        let u = Usage {
            prompt_tokens: 5,
            completion_tokens: 1,
            cached_tokens: 0,
            ttft_ms: 50.0,
            total_ms: 50.0,
            decode_tok_s: 0.0,
            dispatches_per_token: None,
            session_id: None,
        };
        assert_eq!(cost(&u).decode_tok_s, None);
    }
}
