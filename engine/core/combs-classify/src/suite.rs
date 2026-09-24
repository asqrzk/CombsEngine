//! The invariant suite: what a rung must survive before anything is
//! wired to it.
//!
//! Six checks, and each one exists because a classifier can pass the
//! obvious test and still be useless:
//!
//! - SCHEMA — the answer is shaped like an answer at all: it names an
//!   option that was declared, over a probability that is one.
//! - CONTRAST — the mention does NOT fire the action. This is the check
//!   the whole tether exists for, and the one most likely to fail.
//! - MONOTONICITY — paraphrases of one state keep one answer, and their
//!   top probability does not wander further than `delta`. A rung that
//!   answers by topic drifts here.
//! - IMPLICATIONS — an action that fired had its speech act hold.
//! - CALIBRATION — the confidence means something, scored by ECE *and*
//!   Brier.
//! - NOUL — the speech act reads as the fixture says it should. The gate
//!   the whole design rests on had no gold of its own until this check:
//!   every item said which option it expected and nothing about whether
//!   it was a request at all.
//!
//! A check is only worth its name if it can be shown to CATCH
//! something, so every check here has a test that feeds it a canned
//! failure and requires it to refuse, and a positive control requiring
//! it not to refuse everything.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::tether::{Opt, Tether};
use crate::{bars, brier, ece, Kind};

/// One question as the fixture declares it. A `choice` carries options;
/// a `noul` carries one tether and no options.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QuestionSpec {
    pub name: String,
    pub kind: Kind,
    /// What the scorer is being asked, in the caller's own words.
    /// Optional because v1 carried none — which for a cosine is merely
    /// unused text, and for a trained cross-encoder is a prompt shape it
    /// was never fitted on.
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub options: Vec<Opt>,
    #[serde(default)]
    pub tether: Option<Tether>,
}

/// The fixture's first line: the suite's name, its drift bound, and the
/// questions every item is asked.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Header {
    pub suite: String,
    pub delta: f32,
    pub questions: Vec<QuestionSpec>,
    /// Why this fixture's noul is allowed a false side past the option
    /// cap. v1 and the v2 control copy joined four contraries into a
    /// 46-word false side that the 48-token cap cuts mid-clause; they are
    /// kept verbatim for continuity with the records taken on them, and
    /// they say so here. A current fixture carries no such note and is
    /// held to the bound.
    #[serde(default)]
    pub legacy_noul: Option<String>,
    /// Anything the fixture's author wants a reader to know first — a
    /// probe's provenance, a pending second hand.
    #[serde(default)]
    pub note: Option<String>,
}

/// What an item expects: which question, which option, and whether that
/// option should fire at all. `fires: false` is a contrast item — the
/// state mentions the option's topic and asks for nothing.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Expect {
    pub question: String,
    pub option: String,
    pub fires: bool,
    /// Options that are ALSO right for this item.
    ///
    /// Every item in v1 and v2 has exactly one right answer, which is
    /// what a seed suite should have. A genuinely ambiguous state does
    /// not: "shall I walk to the carwash" is a person asking for a view,
    /// and both "advise" and "none of these" are defensible readings of
    /// a catalogue that has no verb for it. Scoring such an item against
    /// one string measures the writer's taste, not the scorer.
    ///
    /// `option` stays the named answer — what a table prints and what
    /// the base-lookup reads — and this is what else would be accepted.
    /// Only meaningful where the option must fire; `parse` refuses it
    /// anywhere else, and refuses one that names the option again or an
    /// option the item forbids.
    #[serde(default)]
    pub any_of: Vec<String>,
}

impl Expect {
    /// Is `top` one of the answers this item accepts?
    pub fn accepts(&self, top: &str) -> bool {
        top == self.option || self.any_of.iter().any(|o| o == top)
    }

    /// Every option this item accepts, the named one first.
    pub fn accepted(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.option.as_str()).chain(self.any_of.iter().map(String::as_str))
    }
}

/// An option that must NEVER be the answer for this item, whatever the
/// right answer turns out to be.
///
/// v1 could only say "not this one", because a near miss had nowhere
/// correct to land — every contrast was scored purely by what it must
/// not do. v2 gives each one a right answer AND keeps the old bar as
/// this field, so relabelling an item cannot quietly drop what v1
/// asserted about it.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Never {
    pub question: String,
    pub option: String,
}

/// The near-miss modes, which are the noul tether's own `not` clauses:
/// a quotation or translation, a question about meaning, a description
/// of something past, and a refusal. An item named by its mode says
/// WHICH near miss defeated a scorer, where `contrast-of:` could only
/// say that one did.
pub const NEAR_MISS_KINDS: &[&str] = &[
    "contrast-of:",
    "translation-of:",
    "definition-of:",
    "past-of:",
    "negation-of:",
    // The fifth mode, and the one the others cannot cover: a person
    // asking what THEY should do. Translation, definition, past and
    // negation are all "not a request", so the `is_request` gate can
    // shut them; a deliberation IS a request — for a view — so only the
    // option's own tether can hold it. An item of this kind scored as a
    // plain positive before it was named here, which is a suite
    // measuring less than it says.
    "deliberation-of:",
];

/// One labelled state.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Item {
    pub id: String,
    /// `base` | `paraphrase-of:<id>` | one of [`NEAR_MISS_KINDS`] + `<id>`
    pub kind: String,
    pub state: String,
    pub expect: Expect,
    /// In the split a temperature may be fitted on.
    #[serde(default)]
    pub calib: bool,
    /// An option this item must never take (v2).
    #[serde(default)]
    pub never: Option<Never>,
    /// FNV-1a 64 of this state as suite v1 carried it, where the state
    /// was carried over verbatim. A suite that edits a state it claims
    /// to have inherited is a suite whose numbers cannot be compared
    /// with the run that measured the original.
    #[serde(default)]
    pub from_v1: Option<u64>,
    /// What each noul question should read for this state: `true` when
    /// the proposition holds. Keyed by question name, so a fixture with
    /// two nouls says both. `expect` names an OPTION and says nothing
    /// about the speech act; until this field every measurement of the
    /// gate was a measurement against nothing.
    #[serde(default)]
    pub noul: BTreeMap<String, bool>,
}

impl Item {
    /// The base this item is a paraphrase or contrast of, if any.
    pub fn of(&self) -> Option<&str> {
        self.kind.split_once(':').map(|(_, id)| id)
    }
    pub fn is_paraphrase(&self) -> bool {
        self.kind.starts_with("paraphrase-of:")
    }
    pub fn is_contrast(&self) -> bool {
        NEAR_MISS_KINDS.iter().any(|k| self.kind.starts_with(k))
    }

    /// Which near miss this is, by name — `translation`, `definition`,
    /// `past`, `negation`, or the untyped `contrast` v1 shipped.
    pub fn near_miss_mode(&self) -> Option<&str> {
        self.kind
            .split_once("-of:")
            .filter(|(mode, _)| NEAR_MISS_KINDS.iter().any(|k| k.starts_with(mode)))
            .map(|(mode, _)| mode)
            .filter(|mode| *mode != "paraphrase")
    }

    /// The (question, option) pair this item must never answer with.
    ///
    /// One rule for both suites: v2 says it outright in `never`, and v1
    /// says it by expecting its own option NOT to fire.
    pub fn forbidden(&self) -> Option<(&str, &str)> {
        if let Some(n) = &self.never {
            return Some((n.question.as_str(), n.option.as_str()));
        }
        if self.is_contrast() && !self.expect.fires {
            return Some((self.expect.question.as_str(), self.expect.option.as_str()));
        }
        None
    }
}

/// What a scorer actually answered for one question of one item.
#[derive(Debug, Clone)]
pub struct Answer {
    pub question: String,
    pub top: String,
    pub p_top: f32,
    pub confidence: f32,
    /// `act` | `abstain` | `refused`
    pub decision: String,
}

/// Every answer for one item.
#[derive(Debug, Clone)]
pub struct Observed {
    pub item: String,
    pub answers: Vec<Answer>,
}

/// A check's result: the claim it makes, whether it held, and the
/// evidence a person can act on — never a bare boolean.
#[derive(Debug, Clone, Serialize)]
pub struct Verdict {
    pub claim: String,
    pub holds: bool,
    pub evidence: String,
}

/// Read a fixture: one header line, then one item per line. Blank lines
/// are skipped; anything else is refused BY LINE NUMBER, because an
/// item dropped in silence is a suite that measures less than it says.
pub fn parse(jsonl: &str) -> Result<(Header, Vec<Item>), String> {
    let mut header: Option<Header> = None;
    let mut items: Vec<Item> = Vec::new();
    for (n, line) in jsonl.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("line {}: {e}", n + 1))?;
        if v.get("kind").and_then(|k| k.as_str()) == Some("header") {
            if header.is_some() {
                return Err(format!("line {}: a second header", n + 1));
            }
            header = Some(serde_json::from_value(v).map_err(|e| format!("line {}: {e}", n + 1))?);
            continue;
        }
        let item: Item = serde_json::from_value(v).map_err(|e| format!("line {}: {e}", n + 1))?;
        check_any_of(&item, n + 1)?;
        items.push(item);
    }
    let header = header.ok_or_else(|| "the fixture has no header line".to_string())?;
    if items.is_empty() {
        return Err("the fixture has no items".to_string());
    }
    for (n, item) in items.iter().enumerate() {
        for q in item.noul.keys() {
            let is_noul = header
                .questions
                .iter()
                .any(|hq| hq.name == *q && hq.kind == Kind::Noul);
            if !is_noul {
                return Err(format!(
                    "item {}: names a noul answer for {q:?}, which the header does not declare \
                     as a noul",
                    item.id
                ));
            }
        }
        let declared = header
            .questions
            .iter()
            .find(|q| q.name == item.expect.question);
        if let Some(q) = declared {
            if !q.options.is_empty() {
                for want in item.expect.accepted() {
                    if !q.options.iter().any(|o| o.id == want) {
                        return Err(format!(
                            "item {}: expects {want:?} on question {:?}, which declares no such \
                             option",
                            item.id, item.expect.question
                        ));
                    }
                }
            }
        } else {
            return Err(format!(
                "item {}: expects question {:?}, which the header does not declare",
                item.id, item.expect.question
            ));
        }
        let _ = n;
    }
    Ok((header, items))
}

/// An alternative answer is only an alternative where an answer is due.
fn check_any_of(item: &Item, line: usize) -> Result<(), String> {
    if item.expect.any_of.is_empty() {
        return Ok(());
    }
    if !item.expect.fires {
        return Err(format!(
            "line {line}: item {} lists any_of but expects its option NOT to fire: an alternative \
             answer is only an alternative where an answer is due",
            item.id
        ));
    }
    if let Some(dup) = item
        .expect
        .any_of
        .iter()
        .find(|o| **o == item.expect.option)
    {
        return Err(format!(
            "line {line}: item {} lists {dup:?} in any_of and as its option",
            item.id
        ));
    }
    if let Some(n) = &item.never {
        if let Some(clash) = item.expect.any_of.iter().find(|o| **o == n.option) {
            return Err(format!(
                "line {line}: item {} accepts {clash:?} in any_of and forbids it in never",
                item.id
            ));
        }
    }
    Ok(())
}

/// Did this option take this question, and act on it?
///
/// Takes the pair rather than an `Expect`, because the question a near
/// miss must NEVER answer is not the one it is expected to answer.
fn fired_option(obs: &Observed, question: &str, option: &str) -> bool {
    obs.answers
        .iter()
        .any(|a| a.question == question && a.top == option && a.decision == "act")
}

fn find<'a>(observed: &'a [Observed], id: &str) -> Option<&'a Observed> {
    observed.iter().find(|o| o.item == id)
}

fn answer_for<'a>(obs: &'a Observed, question: &str) -> Option<&'a Answer> {
    obs.answers.iter().find(|a| a.question == question)
}

/// Every answer names a declared option over a real probability, and
/// every item was answered at all.
pub fn check_schema(h: &Header, items: &[Item], observed: &[Observed]) -> Verdict {
    let mut bad: Vec<String> = Vec::new();
    for item in items {
        let Some(obs) = find(observed, &item.id) else {
            bad.push(format!("{}: no answer at all", item.id));
            continue;
        };
        for a in &obs.answers {
            let Some(q) = h.questions.iter().find(|q| q.name == a.question) else {
                bad.push(format!(
                    "{}: answered a question nobody asked: {}",
                    item.id, a.question
                ));
                continue;
            };
            if q.kind == Kind::Choice && !q.options.iter().any(|o| o.id == a.top) {
                bad.push(format!(
                    "{}: named an option nobody declared: {}",
                    item.id, a.top
                ));
            }
            if !a.p_top.is_finite() || !(0.0..=1.0).contains(&a.p_top) {
                bad.push(format!("{}: a top probability of {}", item.id, a.p_top));
            }
        }
    }
    Verdict {
        claim: "every answer names a declared option".to_string(),
        holds: bad.is_empty(),
        evidence: if bad.is_empty() {
            format!(
                "{} items, every answer inside the declared answer space",
                items.len()
            )
        } else {
            format!(
                "{} answers outside the answer space: {}",
                bad.len(),
                bad.join("; ")
            )
        },
    }
}

/// THE check. A contrast item mentions the option's topic and asks for
/// nothing; the option must not fire.
pub fn check_contrast(_h: &Header, items: &[Item], observed: &[Observed]) -> Verdict {
    let mut fired_ones: Vec<String> = Vec::new();
    let mut unanswered: Vec<String> = Vec::new();
    let mut n = 0usize;
    for item in items {
        let Some((question, option)) = item.forbidden() else {
            continue;
        };
        n += 1;
        match find(observed, &item.id) {
            Some(obs) if fired_option(obs, question, option) => fired_ones.push(item.id.clone()),
            Some(_) => {}
            None => unanswered.push(item.id.clone()),
        }
    }
    let holds = fired_ones.is_empty() && unanswered.is_empty();
    Verdict {
        claim: "a mention never fires the action it mentions".to_string(),
        holds,
        evidence: if holds {
            format!("{n} contrast pairs, none of them fired")
        } else {
            let mut s = format!(
                "{} of {n} mentions fired: {}",
                fired_ones.len(),
                fired_ones.join(", ")
            );
            if !unanswered.is_empty() {
                s.push_str(&format!("; unanswered: {}", unanswered.join(", ")));
            }
            s
        },
    }
}

/// Paraphrases keep their base's top option, and their top probability
/// stays within `delta` of it.
pub fn check_monotonicity(h: &Header, items: &[Item], observed: &[Observed]) -> Verdict {
    let mut changed: Vec<String> = Vec::new();
    let mut drifted: Vec<String> = Vec::new();
    let mut max_drift = 0.0f32;
    let mut n = 0usize;
    for item in items.iter().filter(|i| i.is_paraphrase()) {
        let Some(base_id) = item.of() else { continue };
        let (Some(p), Some(b)) = (find(observed, &item.id), find(observed, base_id)) else {
            continue;
        };
        let q = item.expect.question.as_str();
        let (Some(pa), Some(ba)) = (answer_for(p, q), answer_for(b, q)) else {
            continue;
        };
        n += 1;
        if pa.top != ba.top {
            changed.push(format!(
                "{} answered {} where {} answered {}",
                item.id, pa.top, base_id, ba.top
            ));
            continue;
        }
        let d = (pa.p_top - ba.p_top).abs();
        if d > max_drift {
            max_drift = d;
        }
        if d > h.delta {
            drifted.push(format!("{} drifted {d:.2}", item.id));
        }
    }
    let holds = changed.is_empty() && drifted.is_empty();
    let mut evidence = format!(
        "{n} paraphrases, max drift {max_drift:.2} (bound {:.2})",
        h.delta
    );
    if !changed.is_empty() {
        evidence.push_str(&format!("; changed answer: {}", changed.join(", ")));
    }
    if !drifted.is_empty() {
        evidence.push_str(&format!("; past the bound: {}", drifted.join(", ")));
    }
    Verdict {
        claim: "paraphrases of one state get one answer".to_string(),
        holds,
        evidence,
    }
}

/// An option that fired had every question it requires hold.
///
/// Only items whose implication question was actually observed are
/// judged: a suite run that never asked the speech act says so rather
/// than passing quietly.
pub fn check_implications(h: &Header, items: &[Item], observed: &[Observed]) -> Verdict {
    let mut broken: Vec<String> = Vec::new();
    let mut judged = 0usize;
    for item in items {
        let Some(obs) = find(observed, &item.id) else {
            continue;
        };
        let Some(a) = answer_for(obs, &item.expect.question) else {
            continue;
        };
        if a.decision != "act" {
            continue;
        }
        let Some(q) = h.questions.iter().find(|q| q.name == item.expect.question) else {
            continue;
        };
        let Some(opt) = q.options.iter().find(|o| o.id == a.top) else {
            continue;
        };
        for r in &opt.tether.requires {
            let Some(req) = answer_for(obs, r) else {
                continue;
            };
            judged += 1;
            if req.decision != "act" {
                broken.push(format!(
                    "{} fired {} while {r} did not hold",
                    item.id, a.top
                ));
            }
        }
    }
    Verdict {
        claim: "an action that fired was actually requested".to_string(),
        holds: broken.is_empty(),
        evidence: if judged == 0 {
            "no implication was observed in this run: the speech act was never asked".to_string()
        } else if broken.is_empty() {
            format!("{judged} implications, every one held")
        } else {
            format!(
                "{} implications broken: {}",
                broken.len(),
                broken.join("; ")
            )
        },
    }
}

/// The confidence means something: ECE within its bar, Brier reported
/// beside it so an uninformative-but-honest rung cannot hide behind a
/// small ECE.
pub fn check_calibration(_h: &Header, items: &[Item], observed: &[Observed]) -> Verdict {
    let mut pairs: Vec<(f32, bool)> = Vec::new();
    for item in items {
        let Some(obs) = find(observed, &item.id) else {
            continue;
        };
        let Some(a) = answer_for(obs, &item.expect.question) else {
            continue;
        };
        let hit = if item.expect.fires {
            item.expect.accepts(&a.top)
        } else {
            a.top != item.expect.option
        };
        pairs.push((a.confidence, hit));
    }
    if pairs.is_empty() {
        return Verdict {
            claim: "the stated confidence matches the observed accuracy".to_string(),
            holds: false,
            evidence: "nothing was answered: there is no calibration to measure".to_string(),
        };
    }
    let e = ece(&pairs, 10);
    let b = brier(&pairs);
    let acc = pairs.iter().filter(|(_, hit)| *hit).count() as f32 / pairs.len() as f32;
    Verdict {
        claim: "the stated confidence matches the observed accuracy".to_string(),
        holds: e <= bars::MAX_ECE,
        evidence: format!(
            "ECE {e:.3} over {} items in 10 bins (bar {:.2}); Brier {b:.3}; accuracy {acc:.3}",
            pairs.len(),
            bars::MAX_ECE
        ),
    }
}

/// The speech act reads as the fixture says: a noul's `top` is `true`
/// exactly where the item's gold says the proposition holds.
///
/// Only items that carry a gold for an observed noul are judged, and an
/// item whose gold names a question this run never asked is counted as
/// unanswered rather than skipped — a run that forgot the gate says so.
pub fn check_noul(_h: &Header, items: &[Item], observed: &[Observed]) -> Verdict {
    let mut wrong: Vec<String> = Vec::new();
    let mut unanswered: Vec<String> = Vec::new();
    let mut n = 0usize;
    for item in items {
        for (q, gold) in &item.noul {
            n += 1;
            let Some(a) = find(observed, &item.id).and_then(|o| answer_for(o, q)) else {
                unanswered.push(format!("{}.{q}", item.id));
                continue;
            };
            let read = a.top == "true";
            if read != *gold {
                wrong.push(format!(
                    "{}.{q} read {} (p {:.2}) where the fixture says {}",
                    item.id, a.top, a.p_top, gold
                ));
            }
        }
    }
    let holds = wrong.is_empty() && unanswered.is_empty();
    Verdict {
        claim: "the speech act reads as the fixture says".to_string(),
        holds,
        evidence: if n == 0 {
            "no item carries a noul answer: the gate was never graded".to_string()
        } else if holds {
            format!("{n} noul readings, every one as the fixture says")
        } else {
            let mut s = format!(
                "{} of {n} noul readings wrong: {}",
                wrong.len(),
                wrong.join("; ")
            );
            if !unanswered.is_empty() {
                s.push_str(&format!("; unanswered: {}", unanswered.join(", ")));
            }
            s
        },
    }
}

/// Every check, in the order a reader wants them.
pub fn run_all(h: &Header, items: &[Item], observed: &[Observed]) -> Vec<Verdict> {
    vec![
        check_schema(h, items, observed),
        check_contrast(h, items, observed),
        check_monotonicity(h, items, observed),
        check_implications(h, items, observed),
        check_calibration(h, items, observed),
        check_noul(h, items, observed),
    ]
}
