//! The semantic tether: what an option MEANS, so that no decision is
//! ever taken on the strength of a label.
//!
//! The hazard this exists to close, in one sentence: a classifier whose
//! options are identifiers like `escalate_user_planning_unsubscription`
//! scores by TOPIC, so "translate the phrase 'I'm considering
//! unsubscription' into Arabic" lands on the escalation because the
//! words overlap — an action nobody asked for.
//!
//! So an identifier is never scored. It is an opaque routing key the
//! caller reads back. What a scorer reads is the tether: one sentence
//! saying what the option means, and the anchors saying what it is NOT.
//! The mention-vs-use case then has somewhere to land that is not the
//! action — a translation request matches "quoting or translating a
//! sentence about it" rather than the escalation's own meaning.
//!
//! The tether carries the implication too (`requires`), because "is the
//! person REQUESTING this?" is part of what an action option means
//! rather than a separate question bolted on beside it. `requires` is
//! NOT scored text: it names a question the gate rules on, and its name
//! would put an identifier's shape back in front of the scorer.

use serde::{Deserialize, Serialize};

/// What one option means, and what it is not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tether {
    /// One sentence in the caller's own words: what is true of the state
    /// when this option is the right answer. Required, and refused empty
    /// — an option with nothing but an identifier has no meaning to
    /// score against.
    pub means: String,
    /// The near misses this option must NOT absorb: quoting, translating,
    /// defining, asking about, a past tense, a negation, a hypothetical.
    /// Each is scored beside `means`, so a mention lands here.
    #[serde(default)]
    pub not: Vec<String>,
    /// Questions that must hold before this option may fire, by name.
    /// The usual one is a speech-act question ("is the person requesting
    /// this, as opposed to quoting or describing it?").
    #[serde(default)]
    pub requires: Vec<String>,
}

/// An option: an opaque identifier the caller routes on, and the tether
/// that is the only thing a scorer ever sees.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Opt {
    pub id: String,
    pub tether: Tether,
}

impl Opt {
    /// What a scorer reads for this option.
    ///
    /// The identifier is NOT in it, by construction, and neither are the
    /// `requires` names — both carry an identifier's shape, and an
    /// identifier is a topic, not a meaning. The test beside this
    /// function is what keeps that true.
    pub fn scored_text(&self) -> String {
        let mut s = self.tether.means.trim().trim_end_matches('.').to_string();
        s.push('.');
        let anchors: Vec<&str> =
            self.tether.not.iter().map(|n| n.trim()).filter(|n| !n.is_empty()).collect();
        if !anchors.is_empty() {
            s.push_str(" Not this when the state is instead: ");
            s.push_str(&anchors.join("; "));
            s.push('.');
        }
        s
    }
}

/// A meaning compared for sameness: case and spacing are not meaning,
/// and neither is a trailing full stop.
fn same_meaning(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_end_matches(['.', '!', '?'])
        .to_string()
}

/// Refuse a question's options by name, before anything is scored.
///
/// `declared` is every question name in the same request, so an
/// implication naming a question nobody asked is caught here rather than
/// reading, later and silently, as "the implication did not hold".
///
/// `max_options` of 0 means no cap.
pub fn validate(opts: &[Opt], declared: &[String], max_options: usize) -> Result<(), String> {
    if opts.is_empty() {
        return Err("a question with no options has no answer space".to_string());
    }
    if max_options > 0 && opts.len() > max_options {
        return Err(format!(
            "{} options arrived, past the maximum of {max_options}",
            opts.len()
        ));
    }
    for (i, o) in opts.iter().enumerate() {
        if o.id.trim().is_empty() {
            return Err(format!("option {i} carries no identifier"));
        }
        if o.tether.means.trim().is_empty() {
            return Err(format!(
                "option \"{}\" carries no tether: an identifier is not a meaning",
                o.id
            ));
        }
        if let Some(prev) = opts[..i].iter().find(|p| p.id == o.id) {
            return Err(format!("option \"{}\" appears twice", prev.id));
        }
        if let Some(prev) = opts[..i]
            .iter()
            .find(|p| same_meaning(&p.tether.means) == same_meaning(&o.tether.means))
        {
            return Err(format!(
                "options \"{}\" and \"{}\" tether to the same meaning, so choosing between them could only be a guess",
                prev.id, o.id
            ));
        }
        for r in &o.tether.requires {
            if !declared.iter().any(|d| d == r) {
                return Err(format!(
                    "option \"{}\" requires \"{r}\", which this request does not declare",
                    o.id
                ));
            }
        }
    }
    Ok(())
}
