//! Which dialect a seat speaks, and who decided.
//!
//! Four layers, highest first: a DOOR set for this process, a CARD
//! travelling beside the weights, the TEMPLATE sniffed from the file, and
//! finally nothing at all. Each layer is a deliberate act by someone
//! further from the model than the last, so each outranks the one below.
//!
//! The whole thing is pure — sources come in as arguments, not as
//! environment reads — because a precedence rule that can only be tested
//! by setting a process-wide variable cannot be tested inside a parallel
//! suite at all. This crate has paid for that lesson elsewhere.
//!
//! Whoever decided is carried out alongside the answer and echoed on the
//! engine's stats, because a configuration that is read and never said is
//! a configuration nobody can debug.

use super::{by_name, known_names};
use crate::toolcall::ToolCallStyle;

/// Who decided the dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chose {
    /// An environment door set for this process.
    Door,
    /// A card sitting beside the weights.
    Card,
    /// Sniffed from the chat template the model was taught with.
    Template,
    /// Nothing named one, and no template taught one.
    Nothing,
}

impl Chose {
    pub fn as_str(self) -> &'static str {
        match self {
            Chose::Door => "door",
            Chose::Card => "card",
            Chose::Template => "template",
            Chose::Nothing => "none",
        }
    }
}

/// The dialect a seat speaks, and who said so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chosen {
    pub style: ToolCallStyle,
    pub from: Chose,
}

/// Resolve the precedence.
///
/// A name that no dialect answers to is refused BY NAME, and the refusal
/// lists every name that would have worked — a person who mistypes a
/// dialect should not have to go reading source to find the spelling.
pub fn choose(
    door: Option<&str>,
    card: Option<&str>,
    template: Option<&str>,
) -> Result<Chosen, String> {
    if let Some(name) = door {
        let name = name.trim();
        if !name.is_empty() {
            let d = by_name(name).ok_or_else(|| refusal("the door", name))?;
            return Ok(Chosen {
                style: d.style(),
                from: Chose::Door,
            });
        }
    }
    if let Some(name) = card {
        let name = name.trim();
        if !name.is_empty() {
            let d = by_name(name).ok_or_else(|| refusal("the card beside the weights", name))?;
            return Ok(Chosen {
                style: d.style(),
                from: Chose::Card,
            });
        }
    }
    let style = ToolCallStyle::detect(template);
    Ok(Chosen {
        style,
        from: if style == ToolCallStyle::None {
            Chose::Nothing
        } else {
            Chose::Template
        },
    })
}

/// The door, read once at load.
///
/// Kept OUTSIDE [`choose`] deliberately: a precedence rule that reads the
/// environment cannot be tested inside a parallel suite, because the
/// environment is process-wide. The rule stays pure and takes its sources
/// as arguments; this is the only place that touches the process.
pub fn door() -> Option<String> {
    std::env::var("COMBS_TOOL_DIALECT").ok()
}

fn refusal(who: &str, name: &str) -> String {
    let mut known = known_names();
    known.sort_unstable();
    format!(
        "{who} names the tool dialect {name:?}, which this engine does not read. \
         It knows: {}",
        known.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const XML_TEMPLATE: &str = "… <tool_call>\n<function=name>\n …";
    const HERMES_TEMPLATE: &str = "… <tool_call>\n{\"name\": …";

    #[test]
    fn a_door_outranks_a_card_outranks_the_template() {
        // The template alone.
        let c = choose(None, None, Some(XML_TEMPLATE)).unwrap();
        assert_eq!(c.style, ToolCallStyle::XmlFunction);
        assert_eq!(c.from, Chose::Template);

        // A card overrides what the template says.
        let c = choose(None, Some("hermes"), Some(XML_TEMPLATE)).unwrap();
        assert_eq!(c.style, ToolCallStyle::Hermes);
        assert_eq!(c.from, Chose::Card);

        // A door overrides the card.
        let c = choose(Some("none"), Some("hermes"), Some(XML_TEMPLATE)).unwrap();
        assert_eq!(c.style, ToolCallStyle::None);
        assert_eq!(c.from, Chose::Door);
    }

    #[test]
    fn nothing_named_and_nothing_taught_is_said_as_nothing() {
        let c = choose(None, None, Some("plain chatml")).unwrap();
        assert_eq!(c.style, ToolCallStyle::None);
        assert_eq!(c.from, Chose::Nothing);

        let c = choose(None, None, None).unwrap();
        assert_eq!(c.from, Chose::Nothing);
    }

    #[test]
    fn an_empty_layer_is_skipped_rather_than_refused() {
        // An unset door reaches us as an empty string often enough that
        // treating it as a refusal would break a spawn that meant nothing
        // by it.
        let c = choose(Some(""), None, Some(HERMES_TEMPLATE)).unwrap();
        assert_eq!(c.style, ToolCallStyle::Hermes);
        assert_eq!(c.from, Chose::Template);
        let c = choose(Some("  "), Some(" "), Some(HERMES_TEMPLATE)).unwrap();
        assert_eq!(c.from, Chose::Template);
    }

    #[test]
    fn a_dialect_nobody_teaches_is_refused_naming_the_ones_that_are() {
        let err = choose(Some("harmony"), None, Some(XML_TEMPLATE)).unwrap_err();
        assert!(err.contains("harmony"), "got {err:?}");
        assert!(err.contains("the door"), "the refusal must say WHO named it: {err:?}");
        assert!(err.contains("xml-function"), "it must list what would work: {err:?}");
        assert!(err.contains("hermes"), "got {err:?}");

        let err = choose(None, Some("harmony"), None).unwrap_err();
        assert!(err.contains("card"), "got {err:?}");
    }

    #[test]
    fn the_source_reads_back_as_a_word_for_the_stats_line() {
        assert_eq!(Chose::Door.as_str(), "door");
        assert_eq!(Chose::Card.as_str(), "card");
        assert_eq!(Chose::Template.as_str(), "template");
        assert_eq!(Chose::Nothing.as_str(), "none");
    }
}
