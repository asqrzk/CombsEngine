//! The whole response is one bare JSON object
//! `{"name": …, "parameters": {…}}` — llama-3.x, one call by template
//! contract.
//!
//! Moved here verbatim from the single-parser era; the behaviour is
//! pinned by the parser tests that predate the move.

use super::{DialectParser, Off, Thinking, ToolDialect, parse_call_json};
use crate::toolcall::{ToolCallStyle, ToolEvent};

pub struct LlamaJson;

impl ToolDialect for LlamaJson {
    fn name(&self) -> &'static str {
        "llama-json"
    }

    fn style(&self) -> ToolCallStyle {
        ToolCallStyle::LlamaJson
    }

    fn teaches(&self, template_source: &str) -> bool {
        template_source.contains("ipython") && template_source.contains("\"parameters\"")
    }

    fn parser(&self) -> Box<dyn DialectParser> {
        Box::new(LlamaJsonParser::default())
    }

    fn thinking(&self) -> Thinking {
        Thinking {
            open_in_prompt: false,
            open: "<think>",
            close: "</think>",
            off: Off::None,
        }
    }
}

#[derive(Default)]
pub struct LlamaJsonParser {
    buf: String,
    started: bool,
    buffer_all: bool,
    calls: usize,
}

impl DialectParser for LlamaJsonParser {
    fn push(&mut self, piece: &str) -> Vec<ToolEvent> {
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

    fn finish(&mut self) -> Vec<ToolEvent> {
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

    fn calls_seen(&self) -> usize {
        self.calls
    }
}
