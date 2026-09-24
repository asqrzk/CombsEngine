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
#[serde(deny_unknown_fields)]
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
    /// A short word a scorer MAY read in a label slot, for the road that
    /// has one: laya's sequence renders `label: criterion`, and its own
    /// presets put a meaning there (`billing`, `refund`), never an
    /// identifier. Absent, that road uses an ordinal instead — an
    /// identifier is never a label, which is the whole rule.
    #[serde(default)]
    pub label: Option<String>,
}

/// An option: an opaque identifier the caller routes on, and the tether
/// that is the only thing a scorer ever sees.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    ///
    /// And nothing of ours is in it unless the anchors are: an option
    /// with no contrary reaches the scorer as the caller wrote it, byte
    /// for byte, on every road. The rule is "no punctuation of ours where
    /// there are no words of ours", and it is measured, not aesthetic:
    /// until 2026-09-24 this function closed every meaning with a full
    /// stop the means road did not add, and on a 16-option catalogue
    /// with no contraries that one byte moved 70 of 173 answers — every
    /// anchors-vs-means contrast taken before then was a punctuation
    /// contrast wearing the anchors' name (§139a). A normaliser would
    /// hide that sensitivity, not remove it: the caller's text is the
    /// caller's, stop or no stop.
    ///
    /// With anchors there ARE words of ours, and they carry our
    /// punctuation: one stop closes the meaning before the clause (the
    /// caller's own trailing stops fold into it rather than doubling),
    /// and one closes the clause.
    pub fn scored_text(&self) -> String {
        let anchors: Vec<&str> =
            self.tether.not.iter().map(|n| n.trim()).filter(|n| !n.is_empty()).collect();
        if anchors.is_empty() {
            return self.tether.means.clone();
        }
        let mut s = self.tether.means.trim().trim_end_matches('.').to_string();
        s.push_str(". Not this when the state is instead: ");
        s.push_str(&anchors.join("; "));
        s.push('.');
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

/// An identifier and a label reduced to the same shape, so one wearing
/// the other's clothes is caught.
///
/// `request_refund`, `Request-Refund` and `request refund` are one key:
/// the separator an identifier happens to use is not a meaning, and a
/// scorer reads the words either way.
fn label_key(text: &str) -> String {
    text.to_lowercase()
        .split(|c: char| c == '_' || c == '-' || c.is_whitespace())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whether a label is a meaning-bearing word rather than a routing key.
///
/// The shape of an identifier is what is refused: `_`, a digit, and the
/// `:` / `[` / `]` the sequence itself uses — a `:` in a label forges a
/// second `label: criterion` slot in the text the model reads. What is
/// allowed is what laya's own presets use: one to three ordinary words,
/// hyphens inside a word, short enough to be a name.
fn looks_like_an_identifier(label: &str) -> Option<String> {
    const MAX: usize = 24;
    if label.chars().count() > MAX {
        return Some(format!(
            "it is {} characters, past the {MAX} a name fits in",
            label.chars().count()
        ));
    }
    if let Some(c) = label.chars().find(|c| matches!(c, '_' | ':' | '[' | ']')) {
        return Some(match c {
            '_' => "it is spelled with underscores, which is how identifiers are spelled".to_string(),
            _ => format!("it carries a {c:?}, which the option text uses to separate a label from its criterion"),
        });
    }
    if label.chars().any(|c| c.is_ascii_digit()) {
        return Some("it carries a digit, and a version or an index is not a meaning".to_string());
    }
    let words: Vec<&str> = label.split_whitespace().filter(|w| !w.is_empty()).collect();
    if words.len() > 3 {
        return Some(format!(
            "it is {} words: a label names the option, the criterion explains it",
            words.len()
        ));
    }
    if !label
        .chars()
        .all(|c| c.is_alphabetic() || c == '-' || c.is_whitespace())
    {
        return Some("it is not made of words".to_string());
    }
    None
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
        if let Some(label) = o.tether.label.as_deref() {
            let label = label.trim();
            if label.is_empty() {
                return Err(format!(
                    "option \"{}\" carries an empty label: leave it out rather than score nothing",
                    o.id
                ));
            }
            // The whole point of the label slot is that a scorer MAY
            // read it. So it has to be a meaning, and the rule cannot be
            // `label == id`: `escalate_unsubscription` on
            // `escalate_user_planning_unsubscription` is the same topic
            // in a shorter coat, and word overlap is word overlap.
            if let Some(why) = looks_like_an_identifier(label) {
                return Err(format!(
                    "option \"{}\" carries the label {label:?}, which reads as an identifier: \
                     {why}. A label is a short meaning-bearing word a scorer may read — laya's \
                     own presets use `refund`, `billing` — and an identifier is a topic that \
                     matches by word overlap.",
                    o.id
                ));
            }
            let key = label_key(label);
            if let Some(twin) = opts.iter().find(|p| label_key(&p.id) == key) {
                return Err(format!(
                    "option \"{}\" labels itself {label:?}, which is option \"{}\"'s identifier \
                     in another spelling: an identifier is a topic, not a meaning, and a scorer \
                     that reads it matches by word overlap",
                    o.id, twin.id
                ));
            }
            if let Some(prev) = opts[..i]
                .iter()
                .filter_map(|p| p.tether.label.as_deref().map(|l| (p, l)))
                .find(|(_, l)| label_key(l) == key)
            {
                return Err(format!(
                    "options \"{}\" and \"{}\" both label themselves {label:?}: two options a \
                     scorer cannot tell apart are one option",
                    prev.0.id, o.id
                ));
            }
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
