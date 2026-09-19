//! How each model family phrases a tool call, and how this engine reads
//! it back.
//!
//! One implementation per family, one file per implementation. Tomorrow's
//! model is a new file, a line in [`registry`], and a card key — never an
//! arm grafted onto a parser that belongs to someone else.
//!
//! ## Why the registry is ORDERED
//!
//! Detection reads the chat template's source, because that is the same
//! file that taught the model its phrasing. The order is most specific
//! first, and it is load-bearing rather than tidy: the qwen3 and qwen3.8
//! templates BOTH contain `<tool_call>`, and only qwen3.8 contains
//! `<function=`. Asking the broader dialect first hands qwen3.8 a parser
//! that cannot read it — which is exactly the defect this module exists
//! to close. A model then writes a perfectly good call, the parser finds
//! no JSON inside it, the block degrades to visible text, and the reply
//! comes back `finish_reason: stop` with `tool_calls: null`: a refusal
//! wearing an answer's clothes.
//!
//! A new dialect goes at the position where nothing above it would claim
//! its templates.

use crate::toolcall::{ToolCall, ToolCallStyle, ToolEvent, ToolFunction};

pub mod choose;
pub mod hermes;
pub mod llama_json;
pub mod xml_function;

pub use choose::{Chose, Chosen, choose, door};

/// Where a family keeps its thinking tags, and how a caller turns
/// thinking off.
///
/// The field that matters to a reader is [`Thinking::open_in_prompt`].
/// qwen3.8's template appends `<think>\n` to the GENERATION PROMPT unless
/// `enable_thinking` is false, so its output begins already inside a
/// thought and emits only the closing tag. qwen3's template has no such
/// branch: the model writes both tags itself. A client that folds a
/// thought out of a reply has to know which of those it is looking at, or
/// it folds from an opening tag that never arrives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Thinking {
    /// True when the template opens the thought in the prompt, so only
    /// the closing tag appears in generated text.
    pub open_in_prompt: bool,
    pub open: &'static str,
    pub close: &'static str,
    pub off: Off,
}

/// How thinking is turned off for a family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Off {
    /// A chat-template keyword argument the caller sets, e.g.
    /// `enable_thinking: false`. Both qwen templates implement this; it
    /// is the only road that is a template CONTRACT rather than a habit.
    Kwarg(&'static str),
    /// A phrase in the user's own text. Trained, not templated —
    /// `/no_think` appears in NEITHER qwen template's source, so it works
    /// only as far as the training took it.
    SoftSwitch(&'static str),
    /// This family does not think.
    None,
}

/// A fresh streaming parser for one generation.
///
/// The contract the serving layer already holds: feed detokenized pieces
/// to [`DialectParser::push`], drain what it returns, and call
/// [`DialectParser::finish`] once at the end. An unterminated or
/// malformed call comes back from `finish` as plain content — malformed
/// output degrades to text, it is never swallowed.
pub trait DialectParser: Send {
    fn push(&mut self, piece: &str) -> Vec<ToolEvent>;
    fn finish(&mut self) -> Vec<ToolEvent>;
    fn calls_seen(&self) -> usize;
}

/// One model family's tool-call phrasing.
pub trait ToolDialect: Send + Sync {
    /// The name a card or a door uses.
    fn name(&self) -> &'static str;

    /// The key this dialect answers to.
    fn style(&self) -> ToolCallStyle;

    /// Whether this template teaches this dialect. Asked in the
    /// registry's order, most specific first.
    fn teaches(&self, template_source: &str) -> bool;

    fn parser(&self) -> Box<dyn DialectParser>;

    fn thinking(&self) -> Thinking;
}

/// Every dialect this engine can read, MOST SPECIFIC FIRST.
pub fn registry() -> &'static [&'static dyn ToolDialect] {
    &[
        &xml_function::XmlFunction,
        &hermes::Hermes,
        &llama_json::LlamaJson,
    ]
}

/// The implementation behind a key.
pub fn for_style(style: ToolCallStyle) -> &'static dyn ToolDialect {
    match style {
        ToolCallStyle::XmlFunction => &xml_function::XmlFunction,
        ToolCallStyle::Hermes => &hermes::Hermes,
        ToolCallStyle::LlamaJson => &llama_json::LlamaJson,
        ToolCallStyle::None => &Plain,
    }
}

/// The dialect named by a card or a door, or `None` when nobody teaches
/// that name. Refusing by name is the caller's job — it knows whether the
/// name came from a card file or an environment door.
pub fn by_name(name: &str) -> Option<&'static dyn ToolDialect> {
    if name == Plain.name() {
        return Some(&Plain);
    }
    registry().iter().copied().find(|d| d.name() == name)
}

/// Every name this engine knows, for a refusal that tells the person what
/// they could have written instead.
pub fn known_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = registry().iter().map(|d| d.name()).collect();
    names.push(Plain.name());
    names
}

/// No known phrasing: everything streams as content.
pub struct Plain;

impl ToolDialect for Plain {
    fn name(&self) -> &'static str {
        "none"
    }
    fn style(&self) -> ToolCallStyle {
        ToolCallStyle::None
    }
    fn teaches(&self, _template_source: &str) -> bool {
        false
    }
    fn parser(&self) -> Box<dyn DialectParser> {
        Box::new(PlainParser)
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

struct PlainParser;

impl DialectParser for PlainParser {
    fn push(&mut self, piece: &str) -> Vec<ToolEvent> {
        vec![ToolEvent::Content(piece.to_string())]
    }
    fn finish(&mut self) -> Vec<ToolEvent> {
        Vec::new()
    }
    fn calls_seen(&self) -> usize {
        0
    }
}

/// Longest suffix of `buf` that is a proper prefix of `marker` (ASCII
/// marker ⇒ any match ends on a char boundary). Held back so a marker
/// split across two detokenized pieces is still recognised, and so a
/// false alarm streams through intact.
pub(crate) fn ambiguous_tail(buf: &str, marker: &str) -> usize {
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
pub(crate) fn parse_call_json(s: &str) -> Option<ToolCall> {
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
    fn the_registry_asks_the_most_specific_dialect_first() {
        // Not a style preference: qwen3 and qwen3.8 both write
        // `<tool_call>`, so a broader dialect placed first would claim
        // qwen3.8's templates and hand them a parser that cannot read
        // them.
        let names: Vec<&str> = registry().iter().map(|d| d.name()).collect();
        let xml = names.iter().position(|n| *n == "xml-function");
        let hermes = names.iter().position(|n| *n == "hermes");
        assert!(
            xml < hermes,
            "xml-function must be asked before hermes, got {names:?}"
        );
    }

    #[test]
    fn every_dialect_answers_to_exactly_one_key() {
        for d in registry() {
            assert_eq!(
                for_style(d.style()).name(),
                d.name(),
                "{} and its key disagree",
                d.name()
            );
        }
        assert_eq!(for_style(ToolCallStyle::None).name(), "none");
    }

    #[test]
    fn a_name_nobody_teaches_is_not_invented() {
        assert!(by_name("xml-function").is_some());
        assert!(by_name("hermes").is_some());
        assert!(by_name("none").is_some());
        assert!(by_name("harmony").is_none());
        assert!(known_names().contains(&"xml-function"));
    }
}
