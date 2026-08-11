//! Chat message model and tool-call machinery.
//!
//! [`ChatMessage`] replaces the old `(role, content)` tuples with the full
//! OpenAI/HF message shape: assistant messages can carry `tool_calls`,
//! tool results arrive as `role: "tool"` with a `tool_call_id`/`name`.
//! Serde accepts the OpenAI wire format directly (including the
//! `{"type":"function","function":{...}}` nesting), and
//! [`ChatMessage::to_template_value`] emits the HF chat-template
//! convention — notably `function.arguments` as a JSON OBJECT, where the
//! OpenAI wire uses a string (transformers' documented divergence).

use serde::{Deserialize, Serialize};

/// One tool invocation, OpenAI wire shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Correlation id (`call_0`, ...). Empty when the model family does
    /// not use ids; omitted from template dicts in that case.
    #[serde(default)]
    pub id: String,
    pub function: ToolFunction,
}

/// The function half of a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    /// Arguments as a JSON object where possible. The OpenAI wire sends a
    /// JSON *string*; [`ChatMessage::to_template_value`] parses it before
    /// templating (HF templates `tojson` an object).
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// One chat message. `content` defaults to empty (tool-call-only
/// assistant messages have none).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// Plain text message.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }
    }

    /// The dict this message renders as inside a chat template, following
    /// the transformers conventions: `arguments` normalized to an object,
    /// empty ids omitted, `tool_call_id`/`name` present only when set.
    pub fn to_template_value(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("role".into(), self.role.clone().into());
        m.insert("content".into(), self.content.clone().into());
        if !self.tool_calls.is_empty() {
            let calls: Vec<serde_json::Value> = self
                .tool_calls
                .iter()
                .map(|tc| {
                    let mut f = serde_json::Map::new();
                    f.insert("name".into(), tc.function.name.clone().into());
                    f.insert(
                        "arguments".into(),
                        normalize_arguments(&tc.function.arguments),
                    );
                    let mut c = serde_json::Map::new();
                    c.insert("type".into(), "function".into());
                    if !tc.id.is_empty() {
                        c.insert("id".into(), tc.id.clone().into());
                    }
                    c.insert("function".into(), serde_json::Value::Object(f));
                    serde_json::Value::Object(c)
                })
                .collect();
            m.insert("tool_calls".into(), calls.into());
        }
        if let Some(id) = &self.tool_call_id {
            m.insert("tool_call_id".into(), id.clone().into());
        }
        if let Some(name) = &self.name {
            m.insert("name".into(), name.clone().into());
        }
        serde_json::Value::Object(m)
    }
}

impl From<(String, String)> for ChatMessage {
    fn from((role, content): (String, String)) -> Self {
        ChatMessage::text(role, content)
    }
}

/// OpenAI sends `arguments` as a JSON string; HF templates expect an
/// object (they `tojson` it themselves). Parse strings that hold JSON;
/// anything else passes through unchanged.
fn normalize_arguments(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::String(s) => {
            serde_json::from_str(s).unwrap_or_else(|_| v.clone())
        }
        other => other.clone(),
    }
}

/// How a model family phrases tool calls in generated text, detected from
/// the chat template SOURCE (the same file that teaches the model the
/// phrasing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallStyle {
    /// `<tool_call>{"name":…,"arguments":{…}}</tool_call>`, repeated for
    /// parallel calls (qwen2.5/qwen3/hermes/smollm3/ernie).
    Hermes,
    /// The whole response is one bare JSON object
    /// `{"name":…,"parameters":{…}}` (llama-3.x; single call by template
    /// contract).
    LlamaJson,
    /// No known phrasing — everything streams as content.
    None,
}

impl ToolCallStyle {
    pub fn detect(template_source: Option<&str>) -> Self {
        match template_source {
            Some(t) if t.contains("<tool_call>") => ToolCallStyle::Hermes,
            Some(t) if t.contains("ipython") && t.contains("\"parameters\"") => {
                ToolCallStyle::LlamaJson
            }
            _ => ToolCallStyle::None,
        }
    }
}

/// A parsed unit of generated output: plain text to stream, or a complete
/// tool call.
#[derive(Debug, Clone)]
pub enum ToolEvent {
    Content(String),
    Call(ToolCall),
}

/// Streaming tool-call parser sitting between detokenization and emission.
/// Text passes through untouched except while a marker prefix is plausible
/// (only the ambiguous tail is held back) or a call body is being
/// buffered. [`ToolCallParser::finish`] flushes any unterminated buffer as
/// plain content — malformed output degrades to text, it is never
/// swallowed.
pub struct ToolCallParser {
    style: ToolCallStyle,
    buf: String,
    in_call: bool,
    started: bool,
    buffer_all: bool,
    calls: usize,
}

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

impl ToolCallParser {
    pub fn new(style: ToolCallStyle) -> Self {
        ToolCallParser {
            style,
            buf: String::new(),
            in_call: false,
            started: false,
            buffer_all: false,
            calls: 0,
        }
    }

    /// Number of complete tool calls parsed so far.
    pub fn calls_seen(&self) -> usize {
        self.calls
    }

    /// Feeds one detokenized piece; returns events ready to emit.
    pub fn push(&mut self, piece: &str) -> Vec<ToolEvent> {
        match self.style {
            ToolCallStyle::None => vec![ToolEvent::Content(piece.to_string())],
            ToolCallStyle::Hermes => {
                self.buf.push_str(piece);
                self.drain_hermes()
            }
            ToolCallStyle::LlamaJson => {
                if self.buffer_all {
                    self.buf.push_str(piece);
                    return Vec::new();
                }
                if !self.started {
                    self.buf.push_str(piece);
                    let trimmed = self.buf.trim_start();
                    if trimmed.is_empty() {
                        return Vec::new(); // still only whitespace
                    }
                    self.started = true;
                    if trimmed.starts_with('{') {
                        self.buffer_all = true;
                        return Vec::new();
                    }
                    let flushed = std::mem::take(&mut self.buf);
                    return vec![ToolEvent::Content(flushed)];
                }
                vec![ToolEvent::Content(piece.to_string())]
            }
        }
    }

    /// Flushes at end of generation. Unterminated call bodies and
    /// unparseable buffers come back as plain content.
    pub fn finish(&mut self) -> Vec<ToolEvent> {
        match self.style {
            ToolCallStyle::None => Vec::new(),
            ToolCallStyle::Hermes => {
                let mut out = Vec::new();
                if self.in_call {
                    let body = std::mem::take(&mut self.buf);
                    out.push(ToolEvent::Content(format!("{OPEN}{body}")));
                    self.in_call = false;
                } else if !self.buf.is_empty() {
                    out.push(ToolEvent::Content(std::mem::take(&mut self.buf)));
                }
                out
            }
            ToolCallStyle::LlamaJson => {
                if !self.buffer_all {
                    return Vec::new();
                }
                let body = std::mem::take(&mut self.buf);
                match parse_call_json(body.trim()) {
                    Some(call) => {
                        self.calls += 1;
                        vec![ToolEvent::Call(call)]
                    }
                    None => vec![ToolEvent::Content(body)],
                }
            }
        }
    }

    fn drain_hermes(&mut self) -> Vec<ToolEvent> {
        let mut out = Vec::new();
        loop {
            if self.in_call {
                let Some(end) = self.buf.find(CLOSE) else { break };
                let body: String = self.buf.drain(..end + CLOSE.len()).collect();
                let inner = &body[..end];
                self.in_call = false;
                match parse_call_json(inner.trim()) {
                    Some(mut call) => {
                        if call.id.is_empty() {
                            call.id = format!("call_{}", self.calls);
                        }
                        self.calls += 1;
                        out.push(ToolEvent::Call(call));
                    }
                    // Unparseable body degrades to visible text.
                    None => out.push(ToolEvent::Content(format!("{OPEN}{body}"))),
                }
            } else if let Some(start) = self.buf.find(OPEN) {
                if start > 0 {
                    let prefix: String = self.buf.drain(..start).collect();
                    out.push(ToolEvent::Content(prefix));
                }
                self.buf.drain(..OPEN.len());
                self.in_call = true;
            } else {
                // Emit everything except the longest tail that could still
                // grow into the open marker.
                let hold = ambiguous_tail(&self.buf, OPEN);
                let safe = self.buf.len() - hold;
                if safe > 0 {
                    let prefix: String = self.buf.drain(..safe).collect();
                    out.push(ToolEvent::Content(prefix));
                }
                break;
            }
        }
        out
    }
}

/// Longest suffix of `buf` that is a proper prefix of `marker` (ASCII
/// marker ⇒ any match ends on a char boundary).
fn ambiguous_tail(buf: &str, marker: &str) -> usize {
    let max = (marker.len() - 1).min(buf.len());
    for l in (1..=max).rev() {
        if marker.as_bytes()[..l] == buf.as_bytes()[buf.len() - l..] {
            return l;
        }
    }
    0
}

/// Accepts `{"name": …, "arguments": {…}}` (hermes) and
/// `{"name": …, "parameters": {…}}` (llama).
fn parse_call_json(s: &str) -> Option<ToolCall> {
    let v: serde_json::Value = serde_json::from_str(s).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = v
        .get("arguments")
        .or_else(|| v.get("parameters"))
        .cloned()
        .unwrap_or(serde_json::Value::Object(Default::default()));
    Some(ToolCall {
        id: String::new(),
        function: ToolFunction { name, arguments },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_shape_deserializes_and_arguments_normalize() {
        let msg: ChatMessage = serde_json::from_str(
            r#"{
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "get_weather",
                                 "arguments": "{\"location\": \"Paris\"}"}
                }]
            }"#,
        )
        .unwrap();
        assert_eq!(msg.content, "");
        let v = msg.to_template_value();
        // String arguments become an object for the template.
        assert_eq!(
            v["tool_calls"][0]["function"]["arguments"]["location"],
            "Paris"
        );
        assert_eq!(v["tool_calls"][0]["type"], "function");
    }

    fn feed(parser: &mut ToolCallParser, pieces: &[&str]) -> (String, Vec<ToolCall>) {
        let mut text = String::new();
        let mut calls = Vec::new();
        for p in pieces {
            for ev in parser.push(p) {
                match ev {
                    ToolEvent::Content(c) => text.push_str(&c),
                    ToolEvent::Call(c) => calls.push(c),
                }
            }
        }
        for ev in parser.finish() {
            match ev {
                ToolEvent::Content(c) => text.push_str(&c),
                ToolEvent::Call(c) => calls.push(c),
            }
        }
        (text, calls)
    }

    #[test]
    fn style_detection_from_template_source() {
        assert_eq!(
            ToolCallStyle::detect(Some("… <tool_call>\\n{\"name\": …")),
            ToolCallStyle::Hermes
        );
        assert_eq!(
            ToolCallStyle::detect(Some(
                "… ipython … {\"name\": function name, \"parameters\": …"
            )),
            ToolCallStyle::LlamaJson
        );
        assert_eq!(ToolCallStyle::detect(Some("plain chatml")), ToolCallStyle::None);
        assert_eq!(ToolCallStyle::detect(None), ToolCallStyle::None);
    }

    #[test]
    fn hermes_single_call_split_across_pieces() {
        let mut p = ToolCallParser::new(ToolCallStyle::Hermes);
        let (text, calls) = feed(
            &mut p,
            &["<tool", "_call>\n{\"name\": \"get_weather\", \"argu",
              "ments\": {\"location\": \"Paris\"}}\n</tool_call>"],
        );
        assert_eq!(text, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments["location"], "Paris");
        assert_eq!(calls[0].id, "call_0");
    }

    #[test]
    fn hermes_parallel_calls_and_surrounding_text() {
        let mut p = ToolCallParser::new(ToolCallStyle::Hermes);
        let (text, calls) = feed(
            &mut p,
            &["Let me check.\n<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>",
              "<tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call> done"],
        );
        assert_eq!(text, "Let me check.\n done");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.name, "b");
        assert_eq!(calls[1].id, "call_1");
    }

    #[test]
    fn hermes_false_alarm_prefix_streams_through() {
        let mut p = ToolCallParser::new(ToolCallStyle::Hermes);
        let (text, calls) = feed(&mut p, &["a <tool", "box> is not a call"]);
        assert_eq!(text, "a <toolbox> is not a call");
        assert!(calls.is_empty());
    }

    #[test]
    fn hermes_unterminated_call_flushes_as_text() {
        let mut p = ToolCallParser::new(ToolCallStyle::Hermes);
        let (text, calls) = feed(&mut p, &["<tool_call>{\"name\": \"trunc"]);
        assert_eq!(text, "<tool_call>{\"name\": \"trunc");
        assert!(calls.is_empty());
    }

    #[test]
    fn hermes_malformed_json_degrades_to_text() {
        let mut p = ToolCallParser::new(ToolCallStyle::Hermes);
        let (text, calls) = feed(&mut p, &["<tool_call>not json</tool_call>"]);
        assert_eq!(text, "<tool_call>not json</tool_call>");
        assert!(calls.is_empty());
    }

    #[test]
    fn llama_whole_body_json_parses_at_finish() {
        let mut p = ToolCallParser::new(ToolCallStyle::LlamaJson);
        let (text, calls) = feed(
            &mut p,
            &["\n\n{\"name\": \"get_weather\", ",
              "\"parameters\": {\"location\": \"Paris\"}}"],
        );
        assert_eq!(text, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments["location"], "Paris");
    }

    #[test]
    fn llama_plain_text_streams_untouched() {
        let mut p = ToolCallParser::new(ToolCallStyle::LlamaJson);
        let (text, calls) = feed(&mut p, &["  Hello", " there"]);
        assert_eq!(text, "  Hello there");
        assert!(calls.is_empty());
    }

    #[test]
    fn llama_bad_json_flushes_as_text() {
        let mut p = ToolCallParser::new(ToolCallStyle::LlamaJson);
        let (text, calls) = feed(&mut p, &["{\"nam", "e\": oops"]);
        assert_eq!(text, "{\"name\": oops");
        assert!(calls.is_empty());
    }

    #[test]
    fn tool_result_roundtrip() {
        let msg: ChatMessage = serde_json::from_str(
            r#"{"role": "tool", "tool_call_id": "call_0", "name": "get_weather", "content": "22C"}"#,
        )
        .unwrap();
        let v = msg.to_template_value();
        assert_eq!(v["role"], "tool");
        assert_eq!(v["tool_call_id"], "call_0");
        assert_eq!(v["name"], "get_weather");
        assert_eq!(v["content"], "22C");
    }
}
