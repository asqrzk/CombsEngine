//! The per-model card: what a weight file cannot say about itself.
//!
//! A card is a small JSON file named `card.json` sitting beside the
//! weights, in the same place and by the same rule as the tokenizer
//! sidecar. It travels with the model, so a preset that fetches a model
//! can write one, and a person can write one by hand for a model nobody
//! has carded yet.
//!
//! ## What belongs here, and what does not
//!
//! Only facts the engine cannot read out of the file. The tool-call
//! dialect is the first: a chat template teaches the model a phrasing,
//! and sniffing that template is a good guess but only a guess. A card
//! turns the guess into a statement, and a door turns the statement into
//! an override for the one machine that needs it.
//!
//! The dialect is kept here as a bare STRING on purpose. This crate
//! reads files; it knows nothing about which dialects exist. The layer
//! that does know refuses an unknown name and says which names it knows —
//! a refusal that names the alternatives is worth more than a type that
//! cannot express the mistake.

use serde::Deserialize;
use std::path::Path;

/// Configuration that travels beside the weights.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCard {
    /// The tool-call dialect this family speaks, by name.
    #[serde(default)]
    pub tool_dialect: Option<String>,
    /// How this family handles thinking, when it differs from what the
    /// dialect's own default says.
    #[serde(default)]
    pub thinking: Option<CardThinking>,
}

/// The thinking half of a card.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CardThinking {
    /// True when the chat template opens the thought in the PROMPT, so
    /// generated text carries only the closing tag.
    #[serde(default)]
    pub open_in_prompt: Option<bool>,
    /// The chat-template keyword argument that turns thinking off.
    #[serde(default)]
    pub off_kwarg: Option<String>,
}

impl ModelCard {
    /// Reads a card from its text, refusing by name what is wrong.
    ///
    /// Unknown keys are refused rather than ignored: a card is written by
    /// hand often enough that a typo must be told about, and a silently
    /// ignored key reads as a setting that did not take.
    pub fn parse(text: &str) -> Result<ModelCard, String> {
        let card: ModelCard = serde_json::from_str(text)
            .map_err(|e| format!("this is not a model card: {e}"))?;
        if let Some(d) = &card.tool_dialect {
            if d.trim().is_empty() {
                return Err(
                    "the card names an empty tool_dialect; leave the key out to let the \
                     template decide"
                        .into(),
                );
            }
        }
        Ok(card)
    }

    /// The card beside a weight file, if one is there.
    ///
    /// Same sibling rule as the tokenizer sidecar. A card that exists but
    /// cannot be read is an error, never a silent absence: someone wrote
    /// it meaning it to be used.
    pub fn beside(weights: &Path) -> Result<Option<ModelCard>, String> {
        let path = weights.with_file_name("card.json");
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("the card beside these weights could not be read: {e}"))?;
        ModelCard::parse(&text).map(Some)
    }

    /// True when this card says nothing — the same as having no card.
    pub fn is_empty(&self) -> bool {
        self == &ModelCard::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_card_names_a_dialect_and_how_the_family_thinks() {
        let card = ModelCard::parse(
            r#"{"tool_dialect": "xml-function",
                "thinking": {"open_in_prompt": true, "off_kwarg": "enable_thinking"}}"#,
        )
        .unwrap();
        assert_eq!(card.tool_dialect.as_deref(), Some("xml-function"));
        let t = card.thinking.unwrap();
        assert_eq!(t.open_in_prompt, Some(true));
        assert_eq!(t.off_kwarg.as_deref(), Some("enable_thinking"));
    }

    #[test]
    fn an_empty_card_is_the_same_as_no_card() {
        assert!(ModelCard::parse("{}").unwrap().is_empty());
    }

    #[test]
    fn a_key_nobody_reads_is_refused_by_name_rather_than_ignored() {
        let err = ModelCard::parse(r#"{"tool_dialekt": "hermes"}"#).unwrap_err();
        assert!(
            err.contains("tool_dialekt"),
            "the refusal must name the offending key, got {err:?}"
        );
        // And it must say what it would have accepted, or the person is
        // left guessing at a spelling.
        assert!(
            err.contains("tool_dialect"),
            "the refusal must name the keys it knows, got {err:?}"
        );
    }

    #[test]
    fn an_empty_dialect_is_refused_rather_than_read_as_none() {
        let err = ModelCard::parse(r#"{"tool_dialect": "  "}"#).unwrap_err();
        assert!(err.contains("empty"), "got {err:?}");
    }

    #[test]
    fn text_that_is_not_a_card_is_refused_as_such() {
        assert!(ModelCard::parse("not json at all").is_err());
    }
}
