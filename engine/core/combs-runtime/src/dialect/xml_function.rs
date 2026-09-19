
//! `<tool_call><function=NAME><parameter=KEY>…</parameter></function></tool_call>`
//! — the qwen3.5 / qwen3.8 form.
//!
//! The grammar comes from the template's own instruction to the model:
//! reasoning may come BEFORE a call and never after; a value MAY SPAN
//! LINES; and a value that is a mapping or a sequence is written through
//! `tojson` while everything else is written through `|string`.
//!
//! ## Reading a value back
//!
//! The exact inverse of that last rule: a value is parsed as JSON only
//! when it parses AND is an object or an array; anything else stays the
//! string it was written as. The honest consequence, said rather than
//! hidden: a numeric argument was written by `|string` and comes back as
//! the string `"22"`, because the template does not distinguish the
//! number 22 from the string "22" once it is on the wire. Guessing the
//! other way would silently retype a version string or a zero-padded
//! identifier, which is the worse failure.
//!
//! ## Why the whole block is buffered
//!
//! A call is emitted only once its closing marker has arrived, so a
//! marker split across detokenized pieces — the ordinary case at the
//! engine's chunk size — is never mistaken for prose. Text before a call
//! streams as it arrives, minus the longest tail that could still grow
//! into the open marker. Anything that does not parse comes back out as
//! the text it was: malformed output degrades, it is never swallowed.

use super::{DialectParser, Off, Thinking, ToolDialect, ambiguous_tail};
use crate::toolcall::{ToolCall, ToolCallStyle, ToolEvent, ToolFunction};

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
const FN_OPEN: &str = "<function=";
const FN_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";

pub struct XmlFunction;

impl ToolDialect for XmlFunction {
    fn name(&self) -> &'static str {
        "xml-function"
    }

    fn style(&self) -> ToolCallStyle {
        ToolCallStyle::XmlFunction
    }

    fn teaches(&self, template_source: &str) -> bool {
        // Both this family and the hermes family write `<tool_call>`;
        // only this one writes `<function=`. That is the whole
        // discriminator, and it is why this dialect is asked first.
        template_source.contains(FN_OPEN)
    }

    fn parser(&self) -> Box<dyn DialectParser> {
        Box::new(XmlFunctionParser::default())
    }

    fn thinking(&self) -> Thinking {
        // This template appends `<think>\n` to the generation prompt
        // unless `enable_thinking` is false, so the reply begins INSIDE a
        // thought and carries only the closing tag.
        Thinking {
            open_in_prompt: true,
            open: "<think>",
            close: "</think>",
            off: Off::Kwarg("enable_thinking"),
        }
    }
}

#[derive(Default)]
pub struct XmlFunctionParser {
    buf: String,
    in_call: bool,
    calls: usize,
}

impl DialectParser for XmlFunctionParser {
    fn push(&mut self, piece: &str) -> Vec<ToolEvent> {
        self.buf.push_str(piece);
        self.drain()
    }

    fn finish(&mut self) -> Vec<ToolEvent> {
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

    fn calls_seen(&self) -> usize {
        self.calls
    }
}

impl XmlFunctionParser {
    fn drain(&mut self) -> Vec<ToolEvent> {
        let mut out = Vec::new();
        loop {
            if self.in_call {
                let Some(end) = self.buf.find(CLOSE) else { break };
                let block: String = self.buf.drain(..end + CLOSE.len()).collect();
                let inner = &block[..end];
                self.in_call = false;
                match parse_function_block(inner) {
                    Some(mut call) => {
                        call.id = format!("call_{}", self.calls);
                        self.calls += 1;
                        out.push(ToolEvent::Call(call));
                    }
                    // Unreadable body degrades to visible text, markup and
                    // all — the reader sees what the model actually wrote.
                    None => out.push(ToolEvent::Content(format!("{OPEN}{block}"))),
                }
            } else if let Some(start) = self.buf.find(OPEN) {
                if start > 0 {
                    let prefix: String = self.buf.drain(..start).collect();
                    out.push(ToolEvent::Content(prefix));
                }
                self.buf.drain(..OPEN.len());
                self.in_call = true;
            } else {
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

/// Reads what sits between `<tool_call>` and `</tool_call>`:
/// `<function=NAME>` then zero or more `<parameter=KEY>VALUE</parameter>`
/// then `</function>`. The whitespace the template writes around the
/// markers is tolerated, never required. Anything else is not a call, and
/// saying so is how prose stays prose.
fn parse_function_block(inner: &str) -> Option<ToolCall> {
    let rest = inner.trim();
    let rest = rest.strip_prefix(FN_OPEN)?;
    let name_end = rest.find('>')?;
    let name = &rest[..name_end];
    if name.is_empty() || name.contains('<') || name.contains('\n') {
        return None;
    }
    let mut rest = &rest[name_end + 1..];

    // The block must close, and nothing may follow the close.
    let fn_close = rest.rfind(FN_CLOSE)?;
    if !rest[fn_close + FN_CLOSE.len()..].trim().is_empty() {
        return None;
    }
    rest = &rest[..fn_close];

    let mut args = serde_json::Map::new();
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }
        // Junk between parameters is not tolerated: a body we cannot
        // account for is a body we should hand back as text.
        let after_open = rest.strip_prefix(PARAM_OPEN)?;
        let key_end = after_open.find('>')?;
        let key = &after_open[..key_end];
        if key.is_empty() || key.contains('<') || key.contains('\n') {
            return None;
        }
        let value_start = &after_open[key_end + 1..];
        // The template writes exactly one newline after the marker and
        // one before the close; strip those and no more, so a value that
        // legitimately begins with a space or a blank line survives.
        let value_start = value_start.strip_prefix('\n').unwrap_or(value_start);
        let close = value_start.find(PARAM_CLOSE)?;
        let raw = &value_start[..close];
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        args.insert(key.to_string(), value_from_text(raw));
        rest = &value_start[close + PARAM_CLOSE.len()..];
    }

    Some(ToolCall {
        id: String::new(),
        function: ToolFunction {
            name: name.to_string(),
            arguments: serde_json::Value::Object(args),
        },
    })
}

/// A value is JSON only when it parses AND is an object or an array —
/// the exact inverse of the template's `tojson`-or-`|string` rule.
pub(crate) fn value_from_text(text: &str) -> serde_json::Value {
    let trimmed = text.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if v.is_object() || v.is_array() {
                return v;
            }
        }
    }
    serde_json::Value::String(text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toolcall::{ToolCallParser, ToolCallStyle};

    fn feed(pieces: &[&str]) -> (String, Vec<ToolCall>) {
        let mut p = ToolCallParser::new(ToolCallStyle::XmlFunction);
        let mut text = String::new();
        let mut calls = Vec::new();
        for piece in pieces {
            for ev in p.push(piece) {
                match ev {
                    ToolEvent::Content(c) => text.push_str(&c),
                    ToolEvent::Call(c) => calls.push(c),
                }
            }
        }
        for ev in p.finish() {
            match ev {
                ToolEvent::Content(c) => text.push_str(&c),
                ToolEvent::Call(c) => calls.push(c),
            }
        }
        (text, calls)
    }

    /// The exact bytes a seated qwen3.8 returned when it was offered
    /// tools: a closing think tag, then a call with no parameters. The
    /// engine read it as prose and reported `tool_calls: null` — an
    /// answer that had in fact called nothing.
    const LIVE_REPLY: &str = "I should look at the routing table.\n</think>\n\n<tool_call>\n<function=net.routes>\n</function>\n</tool_call>";

    #[test]
    fn the_live_reply_parses_to_one_call() {
        let (text, calls) = feed(&[LIVE_REPLY]);
        assert_eq!(calls.len(), 1, "text came back as {text:?}");
        assert_eq!(calls[0].function.name, "net.routes");
        assert_eq!(calls[0].function.arguments, serde_json::json!({}));
        assert_eq!(calls[0].id, "call_0");
        assert!(
            text.starts_with("I should look at the routing table."),
            "the prose before the call must stream as content, got {text:?}"
        );
        assert!(
            !text.contains("<function="),
            "no call markup may reach the reader as text, got {text:?}"
        );
    }

    /// The template's own worked example, plus the one rule a naive
    /// parser gets wrong: the second value spans three lines.
    #[test]
    fn a_multiline_parameter_survives_and_a_json_value_becomes_an_object() {
        let reply = "<tool_call>\n<function=example_function_name>\n\
             <parameter=example_parameter_1>\nvalue_1\n</parameter>\n\
             <parameter=example_parameter_2>\nThis is the value for the second parameter\n\
             that can span\nmultiple lines\n</parameter>\n\
             <parameter=ports>\n[80, 443]\n</parameter>\n\
             </function>\n</tool_call>";
        let (_, calls) = feed(&[reply]);
        assert_eq!(calls.len(), 1);
        let a = &calls[0].function.arguments;
        assert_eq!(a["example_parameter_1"], "value_1");
        assert_eq!(
            a["example_parameter_2"],
            "This is the value for the second parameter\nthat can span\nmultiple lines"
        );
        assert_eq!(a["ports"], serde_json::json!([80, 443]));
    }

    #[test]
    fn two_blocks_in_one_reply_are_two_calls() {
        let reply = "<tool_call>\n<function=net.routes>\n</function>\n</tool_call>\n\
             <tool_call>\n<function=net.dns>\n</function>\n</tool_call>";
        let (text, calls) = feed(&[reply]);
        assert_eq!(calls.len(), 2, "text came back as {text:?}");
        assert_eq!(calls[0].function.name, "net.routes");
        assert_eq!(calls[1].function.name, "net.dns");
        assert_eq!(calls[1].id, "call_1");
    }

    /// A marker split across detokenized pieces is the ordinary case at
    /// the engine's chunk size, not an edge one.
    #[test]
    fn a_call_split_across_pieces_at_every_boundary() {
        let whole = "before <tool_call>\n<function=net.ping>\n<parameter=host>\n1.1.1.1\n</parameter>\n</function>\n</tool_call>";
        for cut in 1..whole.len() {
            if !whole.is_char_boundary(cut) {
                continue;
            }
            let (_, calls) = feed(&[&whole[..cut], &whole[cut..]]);
            assert_eq!(calls.len(), 1, "split at byte {cut} lost the call");
            assert_eq!(calls[0].function.name, "net.ping");
            assert_eq!(calls[0].function.arguments["host"], "1.1.1.1");
        }
    }

    #[test]
    fn an_unterminated_call_flushes_as_text() {
        let (text, calls) = feed(&["<tool_call>\n<function=net.rou"]);
        assert!(calls.is_empty());
        assert_eq!(
            text, "<tool_call>\n<function=net.rou",
            "malformed output degrades to text, it is never swallowed"
        );
    }

    /// The negative control. Without it the grammar could be loosened
    /// until it also matched prose, and the tests above would still pass.
    #[test]
    fn prose_naming_a_function_is_not_a_call() {
        let prose = "The helper is written <function=example> in the docs, but I am not calling it.";
        let (text, calls) = feed(&[prose]);
        assert!(calls.is_empty(), "prose became a call");
        assert_eq!(text, prose);
    }

    #[test]
    fn a_value_is_json_only_when_the_template_would_have_written_json() {
        assert_eq!(value_from_text("value_1"), "value_1");
        assert_eq!(value_from_text("[80, 443]"), serde_json::json!([80, 443]));
        assert_eq!(value_from_text("{\"a\": 1}"), serde_json::json!({"a": 1}));
        // Written by `|string`, so it comes back as the string it was.
        assert_eq!(value_from_text("22"), "22");
        assert_eq!(value_from_text("3.8"), "3.8");
    }
}
