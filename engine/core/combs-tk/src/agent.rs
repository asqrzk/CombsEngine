//! # The loop: a model that can reach for a tool, and a verdict at the end.
//!
//! ## Why the deciding is a pure function
//!
//! Everything a run does between turns is a decision — dispatch these
//! calls, answer with this text, stop because the rounds are gone — and
//! every one of those decisions is a rule that can be wrong. So the
//! deciding lives here, over values, with no socket and no clock, and
//! the part that holds a connection only carries out what this says.
//!
//! ## What the engine actually sends, measured
//!
//! A tool call arrives as `tool_calls: [{id, type: "function",
//! function: {name, arguments}}]` and `arguments` is a STRING carrying
//! JSON, not an object — so it is parsed twice, and a model that emits
//! malformed JSON there is a refusal rather than a panic.
//!
//! `finish_reason` puts `length` above `tool_calls`, so a call that was
//! truncated mid-emission still arrives WITH its calls and a finish of
//! `length`. Treating that as "the model stopped" would silently drop a
//! tool the model asked for; it is still a call, and the run says the
//! answer was cut.
//!
//! ## What it refuses
//!
//! Exhaustion is never dressed as success. A run that used its rounds
//! without reaching an answer ends as exhausted and exits non-zero,
//! because a half-finished investigation that reports "done" is worse
//! than one that reports nothing.

use serde_json::Value;

/// One tool the model asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: Value,
}

/// What came back from one turn of the engine.
#[derive(Debug, Clone, Default)]
pub struct Turn {
    pub content: String,
    pub calls: Vec<ToolCall>,
    /// `stop`, `length` or `tool_calls`.
    pub finish: Option<String>,
    /// An in-band failure, which arrives at HTTP 200 and so cannot be
    /// read from the status line.
    pub error: Option<String>,
}

/// Where the run is.
#[derive(Debug, Clone)]
pub struct State {
    pub rounds_used: usize,
    pub max_rounds: usize,
}

/// What to do next.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Run these, append what they said, and ask again.
    Dispatch(Vec<ToolCall>),
    /// The model answered. `cut` when the answer stopped at the token
    /// limit rather than finishing.
    Answer { text: String, cut: bool },
    /// The rounds are gone and no answer was reached. Never success.
    Exhausted,
    /// The run cannot continue, and why.
    Failed(String),
}

/// Read the engine's `tool_calls` array into calls this loop can carry.
///
/// `arguments` is a string of JSON rather than an object, so it is
/// parsed twice — and a model that writes nonsense there is refused by
/// name rather than crashing the run.
pub fn parse_calls(raw: &Value) -> Result<Vec<ToolCall>, String> {
    let items = raw
        .as_array()
        .ok_or_else(|| "tool_calls is not an array".to_string())?;
    let mut out = Vec::with_capacity(items.len());
    for (i, it) in items.iter().enumerate() {
        let f = it.get("function").unwrap_or(it);
        let name = f
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("tool call {i} names no function"))?;
        let id = it
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("call")
            .to_string();
        // `arguments` is a STRING of JSON, not an object, so it parses
        // twice — and a model writing nonsense there is refused by name
        // rather than taking the run down with it.
        let args = match f.get("arguments") {
            None => Value::Object(Default::default()),
            Some(Value::Object(o)) => Value::Object(o.clone()),
            Some(Value::String(s)) if s.trim().is_empty() => Value::Object(Default::default()),
            Some(Value::String(s)) => serde_json::from_str(s)
                .map_err(|e| format!("{name} was called with arguments that are not JSON: {e}"))?,
            Some(other) => {
                return Err(format!(
                    "{name} was called with arguments that are neither an object nor JSON text: {other}"
                ))
            }
        };
        out.push(ToolCall { id, name: name.to_string(), args });
    }
    Ok(out)
}

/// Decide what one turn means.
pub fn next(state: &State, turn: &Turn) -> Action {
    // An in-band failure arrives at HTTP 200, so nothing upstream of
    // here can have reported it.
    if let Some(e) = &turn.error {
        return Action::Failed(e.clone());
    }
    let cut = turn.finish.as_deref() == Some("length");
    if !turn.calls.is_empty() {
        // `length` outranks `tool_calls`, so a call truncated mid-
        // emission still arrives carrying its calls. Reading that as
        // "the model stopped" would silently drop a tool it asked for.
        if state.rounds_used >= state.max_rounds {
            // Never dressed as success: a half-finished investigation
            // reporting "done" is worse than one reporting nothing.
            return Action::Exhausted;
        }
        return Action::Dispatch(turn.calls.clone());
    }
    if turn.content.trim().is_empty() {
        // Measured on this desk: a thinking model given too little room
        // spends every token inside a `<think>` that never closes, so
        // folding the thought away leaves nothing and the turn arrives
        // empty carrying `length`. That is the caller's budget, not the
        // model's silence, and the two want different words — one is
        // fixed by raising a number, the other is a finding about the
        // model.
        if cut {
            return Action::Failed(
                "the turn ran out of room before it said anything — every token went into thinking; raise --max-tokens, or put /no_think in the goal"
                    .into(),
            );
        }
        return Action::Failed(
            "the model returned neither an answer nor a tool call; saying nothing is not a finding"
                .into(),
        );
    }
    // A model that wrote a tool call this engine did not parse has not
    // answered: it asked for a probe, in a dialect nothing here reads.
    //
    // Measured against a live seat: a qwen3.5-family file writes
    // `<tool_call><function=name>` because its own template tells it to,
    // the engine detects the `<tool_call>` wrapper as the JSON dialect,
    // finds no JSON inside, and returns `tool_calls: null` with the
    // markup left in the content. Reading that as an answer reported
    // exit 0 over a run that had called no tool and answered nothing —
    // the worst kind of pass, because it looks exactly like success.
    if unparsed_call_markup(&turn.content) {
        return Action::Failed(
            "the model wrote a tool call in a dialect this engine does not parse: the reply carried \
             tool-call markup as text and no structured calls, so no probe ran. This seat cannot \
             drive tools until the engine reads that dialect."
                .into(),
        );
    }
    Action::Answer { text: turn.content.clone(), cut }
}

/// Does this reply carry tool-call markup that never became a call?
///
/// Deliberately narrow: only the wrappers a model emits when it is
/// ASKING for a tool. Prose naming a probe — which this profile's whole
/// purpose invites, since every finding must name what found it — is an
/// answer and must stay one.
fn unparsed_call_markup(text: &str) -> bool {
    text.contains("<tool_call>") || text.contains("<function=")
}

/// Is this a verdict a person can act on?
///
/// `unknowns` is required, and the requirement is the point: a verdict
/// that names nothing it could not determine is not trusted. "Nothing
/// withheld" is a valid answer where silence is not.
pub fn check_verdict(v: &Value) -> Result<(), String> {
    for field in ["verdict", "why"] {
        if v.get(field).and_then(Value::as_str).is_none() {
            return Err(format!("the verdict carries no `{field}`"));
        }
    }
    let findings = v
        .get("findings")
        .and_then(Value::as_array)
        .ok_or_else(|| "the verdict carries no `findings` list".to_string())?;
    for (i, f) in findings.iter().enumerate() {
        if f.get("what").and_then(Value::as_str).is_none() {
            return Err(format!("finding {i} says no `what`"));
        }
        // A finding without the probe that found it is an assertion, and
        // this run deals in readings.
        if f.get("evidence").and_then(Value::as_str).is_none() {
            return Err(format!(
                "finding {i} carries no `evidence`: name the probe that found it"
            ));
        }
    }
    // The honesty lock. The field must be there; an empty list is a
    // valid answer — "nothing withheld" — but silence is not.
    match v.get("unknowns") {
        None => Err(
            "the verdict names no `unknowns`: one that names nothing it could not determine is not trusted, and an empty list is how to say nothing was withheld"
                .into(),
        ),
        Some(u) if !u.is_array() => Err("`unknowns` must be a list".into()),
        Some(_) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fresh() -> State {
        State { rounds_used: 0, max_rounds: 6 }
    }

    /// The shape the engine actually sends, with `arguments` as a string
    /// of JSON rather than an object.
    #[test]
    fn a_tool_call_is_read_in_the_shape_the_engine_sends_it() {
        let raw = json!([{
            "id": "call_0",
            "type": "function",
            "function": { "name": "net.routes", "arguments": "{\"host\":\"1.1.1.1\"}" }
        }]);
        let calls = parse_calls(&raw).expect("the engine's own shape");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[0].name, "net.routes");
        assert_eq!(calls[0].args["host"], json!("1.1.1.1"));
    }

    /// Empty arguments are ordinary: most readings take none.
    #[test]
    fn a_call_with_no_arguments_is_still_a_call() {
        let raw = json!([{
            "id": "call_1",
            "function": { "name": "net.routes", "arguments": "{}" }
        }]);
        let calls = parse_calls(&raw).expect("a call with no arguments");
        assert_eq!(calls[0].args, json!({}));
    }

    /// A model that writes nonsense where JSON belongs is refused by
    /// name — it must not take the run down with it.
    #[test]
    fn arguments_that_are_not_json_are_refused_by_name() {
        let raw = json!([{
            "id": "call_2",
            "function": { "name": "net.routes", "arguments": "{not json" }
        }]);
        let e = parse_calls(&raw).unwrap_err();
        assert!(e.contains("net.routes"), "the refusal must name the call: {e}");
    }

    /// The measured precedence: `length` outranks `tool_calls`, so a
    /// truncated emission still arrives carrying its calls. Reading that
    /// as "the model stopped" would silently drop a tool it asked for.
    #[test]
    fn a_truncated_tool_call_is_still_a_call() {
        let turn = Turn {
            calls: vec![ToolCall { id: "c".into(), name: "net.routes".into(), args: json!({}) }],
            finish: Some("length".into()),
            ..Turn::default()
        };
        match next(&fresh(), &turn) {
            Action::Dispatch(c) => assert_eq!(c.len(), 1),
            other => panic!("a truncated call is still a call, not {other:?}"),
        }
    }

    /// Content and no calls is the answer.
    #[test]
    fn content_with_no_calls_is_the_answer() {
        let turn = Turn {
            content: "the default route is on a VPN".into(),
            finish: Some("stop".into()),
            ..Turn::default()
        };
        match next(&fresh(), &turn) {
            Action::Answer { text, cut } => {
                assert!(text.contains("VPN"));
                assert!(!cut, "a stop is not a cut");
            }
            other => panic!("expected an answer, got {other:?}"),
        }
    }

    /// An answer that stopped at the token limit is still an answer, but
    /// it is a CUT one and the run must say so rather than presenting
    /// half a paragraph as the whole finding.
    #[test]
    fn an_answer_that_ran_out_of_room_is_marked_cut() {
        let turn = Turn {
            content: "the default route is on".into(),
            finish: Some("length".into()),
            ..Turn::default()
        };
        match next(&fresh(), &turn) {
            Action::Answer { cut, .. } => assert!(cut, "a truncated answer must be marked cut"),
            other => panic!("expected a cut answer, got {other:?}"),
        }
    }

    /// Exhaustion is never success. A run that spent its rounds without
    /// reaching an answer says so, and exits non-zero.
    #[test]
    fn rounds_spent_without_an_answer_is_exhaustion_not_success() {
        let spent = State { rounds_used: 6, max_rounds: 6 };
        let turn = Turn {
            calls: vec![ToolCall { id: "c".into(), name: "net.routes".into(), args: json!({}) }],
            finish: Some("tool_calls".into()),
            ..Turn::default()
        };
        assert_eq!(next(&spent, &turn), Action::Exhausted);
    }

    /// An in-band error arrives at HTTP 200, so the status line cannot
    /// report it and this must.
    #[test]
    fn an_error_frame_ends_the_run_carrying_its_reason() {
        let turn = Turn {
            error: Some("the arena is full".into()),
            ..Turn::default()
        };
        match next(&fresh(), &turn) {
            Action::Failed(why) => assert!(why.contains("arena"), "{why}"),
            other => panic!("expected a failure, got {other:?}"),
        }
    }

    /// Measured on this desk: a thinking model given too little room
    /// spends every token inside an unclosed `<think>`, so the fold
    /// leaves nothing and the turn arrives empty with `length`. That is
    /// not silence — it is a turn that ran out of room before it said
    /// anything — and the two need different words, because one is the
    /// model's fault and the other is the caller's budget.
    #[test]
    fn a_turn_cut_off_before_it_spoke_says_so_rather_than_calling_it_silence() {
        let turn = Turn { finish: Some("length".into()), ..Turn::default() };
        match next(&fresh(), &turn) {
            Action::Failed(why) => {
                assert!(
                    why.contains("room") || why.contains("cut"),
                    "it must say the turn ran out of room, not that it said nothing: {why}"
                );
                assert!(
                    why.contains("max-tokens") || why.contains("max_tokens"),
                    "it must name the knob that fixes it: {why}"
                );
            }
            other => panic!("a cut-off turn is still a failure, not {other:?}"),
        }
    }

    /// A turn carrying neither content nor calls is not an answer — it
    /// is a model that said nothing, and saying nothing is not a finding.
    #[test]
    fn a_turn_that_says_nothing_at_all_is_a_failure() {
        let turn = Turn { finish: Some("stop".into()), ..Turn::default() };
        assert!(
            matches!(next(&fresh(), &turn), Action::Failed(_)),
            "an empty turn must not read as an answer"
        );
    }

    /// Measured on this desk against a real seat: a qwen3.5-family file
    /// writes its tool calls in its own dialect — `<tool_call>` wrapping
    /// `<function=name>` — which this engine does not parse, so the
    /// reply arrives with `tool_calls: null` and the markup sitting in
    /// the content as text.
    ///
    /// Calling that an answer reported exit 0 over a run that had called
    /// no tool and answered nothing. A model asking for a probe in a
    /// dialect nothing here reads has not answered; it has asked, and
    /// been misheard.
    #[test]
    fn tool_markup_the_engine_did_not_parse_is_not_an_answer() {
        let turn = Turn {
            content: "I'll read the routing table.\n</think>\n\n<tool_call>\n<function=net.routes>\n</function>\n</tool_call>".into(),
            finish: Some("stop".into()),
            ..Turn::default()
        };
        match next(&fresh(), &turn) {
            Action::Failed(why) => {
                assert!(
                    why.contains("tool call") || why.contains("tool_call"),
                    "it must say a tool call went unparsed: {why}"
                );
            }
            other => panic!("unparsed tool markup is not an answer, got {other:?}"),
        }
    }

    /// The control: ordinary prose that merely MENTIONS a tool by name is
    /// an answer, not a misparse. Without this the guard could refuse
    /// every reply that discusses its own probes — which is exactly what
    /// this profile asks a model to do.
    #[test]
    fn prose_that_names_a_probe_is_still_an_answer() {
        let turn = Turn {
            content: "The default route is on utun4, as net.routes reported.".into(),
            finish: Some("stop".into()),
            ..Turn::default()
        };
        match next(&fresh(), &turn) {
            Action::Answer { text, .. } => assert!(text.contains("utun4")),
            other => panic!("naming a probe in prose is an answer, got {other:?}"),
        }
    }

    /// The honesty lock: a verdict must name what it could not
    /// determine. "Nothing withheld" is a valid answer; silence is not.
    #[test]
    fn a_verdict_that_names_no_unknowns_is_refused() {
        let missing = json!({
            "verdict": "the default route is on a VPN",
            "why": "netstat -rn names utun4 first",
            "findings": [{"what": "utun4 holds the default", "evidence": "net.routes"}]
        });
        let e = check_verdict(&missing).unwrap_err();
        assert!(e.contains("unknowns"), "the refusal must name the field: {e}");

        let empty_is_fine = json!({
            "verdict": "the default route is on a VPN",
            "why": "netstat -rn names utun4 first",
            "findings": [{"what": "utun4 holds the default", "evidence": "net.routes"}],
            "unknowns": []
        });
        assert_eq!(check_verdict(&empty_is_fine), Ok(()));
    }

    /// A finding without its evidence is an assertion, and this run
    /// deals in readings.
    #[test]
    fn a_finding_without_evidence_is_refused() {
        let v = json!({
            "verdict": "something",
            "why": "because",
            "findings": [{"what": "utun4 holds the default"}],
            "unknowns": []
        });
        let e = check_verdict(&v).unwrap_err();
        assert!(e.contains("evidence"), "the refusal must name what is missing: {e}");
    }
}
