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
///
/// This is the KEY into [`crate::dialect`], where each family's reading
/// lives in its own file. Adding a family means adding a variant here and
/// an implementation there — never an arm inside somebody else's parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallStyle {
    /// `<tool_call><function=NAME><parameter=KEY>…</parameter></function></tool_call>`
    /// (qwen3.5/qwen3.8). Must be asked for BEFORE `Hermes`: its
    /// templates carry `<tool_call>` too.
    XmlFunction,
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
    /// Walks the dialect registry in its order, most specific first, and
    /// takes the first family that says this template teaches it.
    pub fn detect(template_source: Option<&str>) -> Self {
        let Some(t) = template_source else {
            return ToolCallStyle::None;
        };
        for dialect in crate::dialect::registry() {
            if dialect.teaches(t) {
                return dialect.style();
            }
        }
        ToolCallStyle::None
    }

    /// The name a card or a door uses for this style.
    pub fn name(self) -> &'static str {
        crate::dialect::for_style(self).name()
    }
}

/// A parsed unit of generated output: plain text to stream, or a complete
/// tool call.
#[derive(Debug, Clone)]
pub enum ToolEvent {
    Content(String),
    Call(ToolCall),
}

/// Streaming tool-call parser sitting between detokenization and
/// emission. Text passes through untouched except while a marker prefix
/// is plausible (only the ambiguous tail is held back) or a call body is
/// being buffered. [`ToolCallParser::finish`] flushes any unterminated
/// buffer as plain content — malformed output degrades to text, it is
/// never swallowed.
///
/// The READING belongs to the dialect ([`crate::dialect`]); this is the
/// one interface every caller holds, whichever family is seated. The
/// signatures here are the ones the serving layer already had, so
/// teaching the engine a new family moves nothing at a call site.
pub struct ToolCallParser {
    style: ToolCallStyle,
    inner: Box<dyn crate::dialect::DialectParser>,
}

impl ToolCallParser {
    pub fn new(style: ToolCallStyle) -> Self {
        ToolCallParser {
            style,
            inner: crate::dialect::for_style(style).parser(),
        }
    }

    /// Which family's reading this parser is doing.
    pub fn style(&self) -> ToolCallStyle {
        self.style
    }

    /// Number of complete tool calls parsed so far.
    pub fn calls_seen(&self) -> usize {
        self.inner.calls_seen()
    }

    /// Feeds one detokenized piece; returns events ready to emit.
    pub fn push(&mut self, piece: &str) -> Vec<ToolEvent> {
        self.inner.push(piece)
    }

    /// Flushes at end of generation. Unterminated call bodies and
    /// unparseable buffers come back as plain content.
    pub fn finish(&mut self) -> Vec<ToolEvent> {
        self.inner.finish()
    }
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
