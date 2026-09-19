//! `<tool_call>{"name": …, "arguments": {…}}</tool_call>`, repeated for
//! parallel calls — qwen2.5, qwen3, hermes, smollm3, ernie.
//!
//! Moved here verbatim from the single-parser era; the behaviour is
//! pinned by the parser tests that predate the move, which is what makes
//! the move provable rather than merely plausible.

use super::{DialectParser, Off, Thinking, ToolDialect, ambiguous_tail, parse_call_json};
use crate::toolcall::{ToolCallStyle, ToolEvent};

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

pub struct Hermes;

impl ToolDialect for Hermes {
    fn name(&self) -> &'static str {
        "hermes"
    }

    fn style(&self) -> ToolCallStyle {
        ToolCallStyle::Hermes
    }

    fn teaches(&self, template_source: &str) -> bool {
        template_source.contains(OPEN)
    }

    fn parser(&self) -> Box<dyn DialectParser> {
        Box::new(HermesParser::default())
    }

    fn thinking(&self) -> Thinking {
        // qwen3's template honours `enable_thinking: false` by writing an
        // empty thought into the prompt, but has no else-branch — so with
        // thinking on, the model opens AND closes the tag itself.
        // `/no_think` does not appear in the template source at all; it is
        // a training convention and is named here as one.
        Thinking {
            open_in_prompt: false,
            open: "<think>",
            close: "</think>",
            off: Off::Kwarg("enable_thinking"),
        }
    }
}

#[derive(Default)]
pub struct HermesParser {
    buf: String,
    in_call: bool,
    calls: usize,
}

impl DialectParser for HermesParser {
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

impl HermesParser {
    fn drain(&mut self) -> Vec<ToolEvent> {
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
