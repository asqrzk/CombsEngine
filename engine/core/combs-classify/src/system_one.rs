//! Reading a typed question the way a trained decision model reads it.
//!
//! The cosine road scores an option by comparing embeddings, so the text
//! it compares is ours to choose. A trained cross-encoder is not like
//! that: it was fitted on ONE rendering of a question, and any other
//! rendering is off the distribution it was measured on. So everything
//! here is a port rather than a design — the strings, the bucket
//! boundaries, the softmax and the order the two halves of a `noul` come
//! back in are laya's, and where this file had an opinion it deferred.
//!
//! Nothing here holds a tensor or a tokenizer. It is the text and the
//! arithmetic on either side of the model: what goes in as option text,
//! and what a row of head logits means once it comes out.
//!
//! Read from `laya` 0.3.5 (`laya/common.py`, `laya/agent.py`), not from
//! a description of it. Three things that reading settled, each of which
//! a reasonable guess would have got wrong:
//!
//! * A `noul`'s options are rendered `[false, true]` and the proposition
//!   is index **1**. Our own wire puts the proposition first, so the two
//!   orders are opposite and [`engine_order`] is where they meet.
//! * A fitted temperature below 1 SHARPENS. This checkpoint ships
//!   `choice:11+` at 0.1006, which multiplies the logits tenfold and
//!   would publish a 0.24 top probability as 0.99 — so laya clamps to
//!   [[`TEMP_MIN`], [`TEMP_MAX`]] at load and says which values it
//!   refused. [`clamp_temperature`] is that guard, and it is not
//!   optional: without it the calibration makes a coin flip read as a
//!   certainty.
//! * For a `choice`, only `null` and `""` mean "this option has no
//!   description". `0` and `false` are criteria a caller may legitimately
//!   write, and dropping them would silently rename the option.

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;

use crate::Kind;

/// Question types, by the index `type_emb` and the temperature table are
/// keyed on. laya's own order, and the checkpoint's.
pub fn qtype(kind: Kind) -> usize {
    match kind {
        Kind::Choice => 0,
        Kind::Score => 1,
        Kind::Noul => 2,
    }
}

/// The name that index carries in a temperature bucket.
pub fn qtype_name(qtype: usize) -> &'static str {
    match qtype {
        0 => "choice",
        1 => "score",
        _ => "noul",
    }
}

/// What a `noul`'s false side reads as when the caller described only
/// the proposition.
pub const NOUL_FALSE: &str = "no, the statement does not hold";
/// What its true side reads as.
pub const NOUL_TRUE: &str = "yes, the statement holds";

/// The softest a fitted temperature may be.
pub const TEMP_MAX: f32 = 5.0;
/// The sharpest. Below this a "calibration" is amplifying the logits
/// rather than calibrating them; see this module's header.
pub const TEMP_MIN: f32 = 0.5;

/// A temperature that changes nothing, and what a value that is not a
/// number falls back to.
pub const NEUTRAL: f32 = 1.0;

/// `t` confined to `[TEMP_MIN, TEMP_MAX]`, with anything that is not a
/// finite number falling back to [`NEUTRAL`].
pub fn clamp_temperature(t: f64) -> f32 {
    if !t.is_finite() {
        return NEUTRAL;
    }
    (t as f32).clamp(TEMP_MIN, TEMP_MAX)
}

/// One criterion as the model reads it: a string passes through, and
/// anything structured becomes JSON, so a rubric written as an object
/// reads as an object rather than as a language's idea of one.
pub fn render_criterion(v: &Value) -> String {
    if let Value::String(s) = v {
        return s.clone();
    }
    // Neither failure is reachable — a `Value` always serialises and
    // serde_json always writes utf-8 — but this runs once per option per
    // request, and a request path with an `expect` on it is a request
    // path that can end the process. `to_string()` is the same text
    // without the house spacing, which is worse than the rubric and far
    // better than a panic.
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, Spaced);
    if v.serialize(&mut ser).is_err() {
        return v.to_string();
    }
    String::from_utf8(buf).unwrap_or_else(|_| v.to_string())
}

/// JSON with a space after every separator — python's `json.dumps`
/// default, which is what the rendering was fitted on. serde_json's own
/// compact form omits both spaces and would render a different string.
struct Spaced;

impl serde_json::ser::Formatter for Spaced {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

/// A question's option criteria, in the shape each kind carries them.
///
/// The order is the answer's order: a `choice`'s options come back in
/// the order they are given here, so a caller that reorders them
/// reorders the distribution too.
#[derive(Debug, Clone, PartialEq)]
pub enum Criteria {
    /// Label, and what that label means. A label with no description is
    /// scored as the bare label.
    Choice(Vec<(String, Value)>),
    /// One criterion per level, in level order.
    Score(Vec<Value>),
    /// The two sides of a proposition, either of which may be left to
    /// its default.
    Noul {
        when_false: Option<Value>,
        when_true: Option<Value>,
    },
}

impl Criteria {
    /// The question kind these criteria belong to.
    pub fn kind(&self) -> Kind {
        match self {
            Criteria::Choice(_) => Kind::Choice,
            Criteria::Score(_) => Kind::Score,
            Criteria::Noul { .. } => Kind::Noul,
        }
    }
}

/// `null` and `""` are "no description"; every other value, `0` and
/// `false` included, is a criterion the caller meant.
fn described(v: &Value) -> Option<&Value> {
    match v {
        Value::Null => None,
        Value::String(s) if s.is_empty() => None,
        other => Some(other),
    }
}

/// How an option's text reaches the scorer on the marker road.
///
/// laya's sequence renders `label: criterion` per option. What goes in
/// those two slots is ours to choose, and the choice is measurable — so
/// it is a door with a default that changes nothing, not a rewrite.
///
/// The hazard the tether exists to close (`tether.rs`) is an identifier
/// in front of a scorer: `escalate_user_planning_unsubscription` carries
/// the TOPIC, so "translate the phrase 'I'm considering unsubscription'"
/// matches it by word overlap. The cosine road never had this problem —
/// it embeds `scored_text()`, and two tests pin that the id is absent.
/// The marker road arrived with the id in the label slot and the `not`
/// anchors nowhere, so the two roads have never scored the same text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Rendering {
    /// `id: means` — what this seat shipped, kept as the default so a
    /// record taken against it stays comparable.
    #[default]
    IdMeans,
    /// `<label or ordinal>: means` — the identifier gone, the anchors
    /// still absent. Isolates what the identifier alone was worth.
    LabelMeans,
    /// `<label or ordinal>: means. Not this when the state is instead: …`
    /// — the text the cosine road scores, on the marker road at last.
    /// For an option with no contrary it is byte-identical to
    /// `label-means`: a rendering adds no punctuation where it adds no
    /// words (`Opt::scored_text`).
    LabelAnchors,
}

/// The two roads an option can reach a scorer by, kept apart on purpose.
///
/// The driver's words. The **normal** road LABELS the option — laya's
/// own convention, a word in the label slot and the criterion after it —
/// so whatever that word carries, topic included, is scored. The
/// **semantic** road scores the tether instead: what the option means
/// and what it is NOT, with the option never named at all.
///
/// They are not two settings of one thing. They disagree about whether
/// an option has a name the model may read, so a record must say which
/// road it ran and a seat must be startable on either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Road {
    /// The option is named; the name is scored.
    Normal,
    /// The option is described; the name is never scored.
    Semantic,
}

impl Road {
    pub fn as_str(self) -> &'static str {
        match self {
            Road::Normal => "normal",
            Road::Semantic => "semantic",
        }
    }
}

impl Rendering {
    /// The wire's own word for this rendering, and back.
    pub fn as_str(self) -> &'static str {
        match self {
            Rendering::IdMeans => "id-means",
            Rendering::LabelMeans => "label-means",
            Rendering::LabelAnchors => "label-anchors",
        }
    }

    /// Which of the two roads this rendering walks.
    ///
    /// `id-means` is on the NORMAL road: it labels the option, with the
    /// worst available label. That it is the accident this seat shipped
    /// does not make it a third road — it makes it the normal road with
    /// the routing key in the name slot.
    pub fn road(self) -> Road {
        match self {
            Rendering::IdMeans | Rendering::LabelMeans => Road::Normal,
            Rendering::LabelAnchors => Road::Semantic,
        }
    }

    /// Refused by name rather than defaulted: a caller who asks for a
    /// rendering that does not exist gets told, not silently measured
    /// against another one.
    ///
    /// `normal` and `semantic` name the two ROADS and are the words a
    /// configuration should use; the hyphenated names pick an exact
    /// rendering, which is what a measurement arm wants. `normal`
    /// resolves to `label-means` and never to `id-means`: the identifier
    /// is a legacy default a caller must now ask for by its own name.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "id-means" => Ok(Rendering::IdMeans),
            "label-means" | "normal" => Ok(Rendering::LabelMeans),
            "label-anchors" | "semantic" => Ok(Rendering::LabelAnchors),
            other => Err(format!(
                "unknown render {other:?}: a road (normal, semantic) or a \
                 rendering (id-means, label-means, label-anchors)"
            )),
        }
    }
}

/// What the scorer reads in an option's label slot.
///
/// Never the identifier unless the caller asked for `id-means`. A
/// tether that names a label gets that word; one that does not gets an
/// ordinal, which carries no topic at all.
fn label_of(opt: &crate::Opt, i: usize, rendering: Rendering) -> String {
    match rendering {
        Rendering::IdMeans => opt.id.clone(),
        _ => opt
            .tether
            .label
            .as_deref()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("option {}", i + 1)),
    }
}

/// One question's criteria as the model will read them.
///
/// The seat and any measurement of it call THIS, so a rendering cannot
/// be one thing in the server and another in the harness that grades it.
pub fn criteria_for(kind: Kind, options: &[crate::Opt], rendering: Rendering) -> Criteria {
    let text = |o: &crate::Opt| match rendering {
        Rendering::LabelAnchors => o.scored_text(),
        _ => o.tether.means.clone(),
    };
    match kind {
        Kind::Choice => Criteria::Choice(
            options
                .iter()
                .enumerate()
                .map(|(i, o)| (label_of(o, i, rendering), Value::String(text(o))))
                .collect(),
        ),
        Kind::Score => Criteria::Score(options.iter().map(|o| Value::String(text(o))).collect()),
        // A noul's two sides are already the proposition and its
        // contraries; `parse_request` folded them, and a label slot
        // there is laya's own `false:`/`true:`, not ours to fill.
        Kind::Noul => Criteria::Noul {
            when_true: options
                .first()
                .map(|o| Value::String(o.tether.means.clone())),
            when_false: options
                .get(1)
                .map(|o| Value::String(o.tether.means.clone())),
        },
    }
}

/// The option texts, in marker order — one string per marker the
/// sequence will carry.
pub fn render_options(criteria: &Criteria) -> Vec<String> {
    match criteria {
        Criteria::Choice(opts) => opts
            .iter()
            .map(|(label, crit)| match described(crit) {
                None => label.clone(),
                Some(v) => format!("{label}: {}", render_criterion(v)),
            })
            .collect(),
        Criteria::Score(levels) => levels
            .iter()
            .enumerate()
            .map(|(i, c)| format!("level {i}: {}", render_criterion(c)))
            .collect(),
        // Always two, always false first. The proposition is the second
        // marker, and everything downstream that says `true` means index
        // one.
        Criteria::Noul {
            when_false,
            when_true,
        } => {
            let side = |v: &Option<Value>, default: &str| -> String {
                v.as_ref()
                    .and_then(described)
                    .map(render_criterion)
                    .unwrap_or_else(|| default.to_string())
            };
            vec![
                format!("false: {}", side(when_false, NOUL_FALSE)),
                format!("true: {}", side(when_true, NOUL_TRUE)),
            ]
        }
    }
}

/// The calibration bucket a question of this type and option count falls
/// in. A single-option question buckets with the two-option ones — there
/// is no separate bucket for it, and it is certain by construction
/// anyway.
pub fn temp_bucket(qtype: usize, k: usize) -> String {
    let size = if k <= 2 {
        "2"
    } else if k <= 5 {
        "3-5"
    } else if k <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{}:{size}", qtype_name(qtype))
}

/// The fitted temperatures a checkpoint ships: one per question type,
/// and a finer one per (type x option count) where the fit had enough
/// items to make one.
///
/// Every value is clamped as it is read, and the ones that moved are
/// kept so a seat can say out loud which of its own calibration it
/// refused.
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    per_type: [f32; 3],
    by_bucket: BTreeMap<String, f32>,
    clamped: Vec<String>,
    /// The shipped number behind each clamped entry, by the same name,
    /// when it was a number at all — what a served temperature was
    /// clamped FROM, so an answer can say so.
    clamped_from: BTreeMap<String, f32>,
}

impl Calibration {
    /// No calibration: every temperature neutral.
    pub fn neutral() -> Self {
        Self {
            per_type: [NEUTRAL; 3],
            by_bucket: BTreeMap::new(),
            clamped: Vec::new(),
            clamped_from: BTreeMap::new(),
        }
    }

    /// The temperature for one question: its bucket's, or its type's
    /// when the fit had no bucket for it.
    pub fn temperature(&self, kind: Kind, k: usize) -> f32 {
        let q = qtype(kind);
        self.by_bucket
            .get(&temp_bucket(q, k))
            .copied()
            .unwrap_or(self.per_type[q])
    }

    /// The bucket that temperature came from, for the record.
    pub fn bucket(&self, kind: Kind, k: usize) -> String {
        temp_bucket(qtype(kind), k)
    }

    /// Which shipped values were refused, each as `name=value`. Empty
    /// when the checkpoint's calibration was usable as written.
    pub fn clamped(&self) -> &[String] {
        &self.clamped
    }

    /// The value a question's served temperature was clamped FROM, when
    /// it was: the bucket's shipped number if the bucket exists and was
    /// clamped, else the type's if the question fell back to it and
    /// that was clamped. `None` means the served temperature is what
    /// the checkpoint wrote.
    ///
    /// Why an answer needs it: `calibration.temperature` reads 0.5 on
    /// every 16-option question of this checkpoint, and 0.5 is not what
    /// laya fitted — `choice:11+` shipped at 0.1006, the fitter's own
    /// floor, and the loader clamped it. A record that only saw the
    /// served number could not tell a fitted 0.5 from a clamped one
    /// (§139a measured the clamp as the better of the two; the point is
    /// that the answer says which it is).
    pub fn clamped_from(&self, kind: Kind, k: usize) -> Option<f32> {
        let q = qtype(kind);
        let bucket = temp_bucket(q, k);
        if self.by_bucket.contains_key(&bucket) {
            return self.clamped_from.get(&bucket).copied();
        }
        self.clamped_from.get(&format!("temperature[{q}]")).copied()
    }
}

/// A laya agent's own `rl_agent_config.json`: the calibration, the two
/// length budgets the marker sequence is built to, and the two counts
/// the head's shape depends on.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentConfig {
    /// The whole sequence's token budget.
    pub max_len: usize,
    /// How much of it the instructions and the option markers may take
    /// before the state gets what is left.
    pub head_max_len: usize,
    /// Layers in the transformer head over the markers, when the config
    /// states them. Absent is NOT two: the weights carry the count
    /// (`head.layers.{j}.*`), the model reads it off them, and a config
    /// that states a different number is refused at load.
    pub head_layers: Option<usize>,
    /// Acts the head chooses between: one per priced act, plus the one
    /// that costs nothing — answering.
    pub num_acts: usize,
    pub calibration: Calibration,
}

impl AgentConfig {
    /// Parses the config, filling laya's own defaults for what it omits.
    pub fn parse(text: &str) -> Result<Self, String> {
        let v: Value =
            serde_json::from_str(text).map_err(|e| format!("this is not an agent config: {e}"))?;
        let usize_at = |key: &str, default: usize| -> usize {
            v.get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(default)
        };

        // A value the config does not carry was never refused — it
        // simply defaulted; only a value that is THERE and unusable is
        // worth naming.
        let mut clamped = Vec::new();
        let mut clamped_from = BTreeMap::new();
        let mut take = |name: String, raw: Option<&Value>| -> f32 {
            let Some(r) = raw else {
                return NEUTRAL;
            };
            match r.as_f64() {
                Some(x) if x.is_finite() && (x as f32) >= TEMP_MIN && (x as f32) <= TEMP_MAX => {
                    x as f32
                }
                usable => {
                    clamped.push(format!("{name}={r}"));
                    if let Some(x) = usable.filter(|x| x.is_finite()) {
                        clamped_from.insert(name, x as f32);
                    }
                    usable.map(clamp_temperature).unwrap_or(NEUTRAL)
                }
            }
        };

        let per_type_raw = v.get("temperature").and_then(Value::as_array);
        let mut per_type = [NEUTRAL; 3];
        for (i, slot) in per_type.iter_mut().enumerate() {
            *slot = take(
                format!("temperature[{i}]"),
                per_type_raw.and_then(|a| a.get(i)),
            );
        }

        let mut by_bucket = BTreeMap::new();
        if let Some(map) = v.get("temperature_by_options").and_then(Value::as_object) {
            for (name, raw) in map {
                by_bucket.insert(name.clone(), take(name.clone(), Some(raw)));
            }
        }

        Ok(Self {
            max_len: usize_at("max_len", 512),
            head_max_len: usize_at("head_max_len", 192),
            head_layers: match v.get("head_layers") {
                None => None,
                Some(raw) => Some(
                    raw.as_u64()
                        .filter(|&n| n >= 1)
                        .map(|n| n as usize)
                        .ok_or_else(|| {
                            format!(
                                "rl_agent_config.json: `head_layers` must be a positive integer, got {raw}"
                            )
                        })?,
                ),
            },
            // One act per price, plus answering, which is priced at
            // nothing and is not in the table.
            num_acts: v
                .get("act_costs")
                .and_then(Value::as_object)
                .map(|m| m.len())
                .unwrap_or(0)
                + 1,
            calibration: Calibration {
                per_type,
                by_bucket,
                clamped,
                clamped_from,
            },
        })
    }
}

/// The distribution one row of head logits stands for, tempered.
///
/// `logits` is already cut to the markers that survived — a marker the
/// sequence dropped has no logit to soften. The temperature is clamped
/// here as well as at load, because a caller may hand one over per
/// request and the guard belongs on every road into the softmax.
pub fn readout(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let t = clamp_temperature(temperature as f64);
    let z: Vec<f32> = logits.iter().map(|&l| l / t).collect();
    let top = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|&v| (v - top).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.iter().map(|&v| v / sum).collect()
}

/// The proposition's probability, out of a `noul`'s distribution in the
/// model's own order — index **one**, because the markers are rendered
/// `["false: ...", "true: ..."]`.
pub fn p_true(p: &[f32]) -> f32 {
    p.get(1).copied().unwrap_or(0.0)
}

/// The same distribution in the order this engine's wire carries it.
///
/// Our `/v1/classify` puts a noul's proposition FIRST — an implication
/// reads `distribution[0]` as the probability the proposition holds —
/// and the model puts it second. This is the one place those two
/// conventions meet, so that no caller has to know there are two.
pub fn engine_order(kind: Kind, p: &[f32]) -> Vec<f32> {
    match kind {
        Kind::Noul => vec![p_true(p), p.first().copied().unwrap_or(0.0)],
        _ => p.to_vec(),
    }
}

/// Confidence, by laya's own rule for each kind.
///
/// `choice` and `score` read the whole distribution as normalised
/// entropy — [`crate::confidence`], the same formula this crate already
/// held. A `noul` reads the calibrated probability itself: two outcomes
/// have no tail to weigh, and `max(p, 1-p)` is the distance from a coin
/// flip. That form is the same under either ordering, so this function
/// takes a distribution in whichever order the caller holds.
pub fn decision_confidence(kind: Kind, p: &[f32]) -> f32 {
    match kind {
        Kind::Noul => p.iter().copied().fold(0.0f32, f32::max).clamp(0.0, 1.0),
        _ => crate::confidence(p),
    }
}

#[cfg(test)]
mod tests {

    /// THE rule, on the road that broke it. `tether.rs` says an
    /// identifier is never scored, and the cosine road has two tests
    /// pinning it — but the marker road renders `label: criterion` and
    /// arrived with the id in the label slot, so the two roads have
    /// never scored the same text. Every rendering but the default must
    /// keep the identifier out; the default carries it and is named
    /// here so nobody mistakes it for safe.
    #[test]
    fn no_rendering_but_the_default_puts_the_identifier_in_front_of_the_scorer() {
        let opts = vec![
            crate::Opt {
                id: "escalate_user_planning_unsubscription".to_string(),
                tether: crate::Tether {
                    means: "the person is asking us to end their subscription now".to_string(),
                    not: vec!["quoting or translating a sentence about unsubscribing".to_string()],
                    requires: vec!["is_request".to_string()],
                    label: None,
                },
            },
            crate::Opt {
                id: "request_refund".to_string(),
                tether: crate::Tether {
                    means: "the person is asking us to return money".to_string(),
                    not: Vec::new(),
                    requires: Vec::new(),
                    label: Some("refund".to_string()),
                },
            },
        ];
        let texts = |r| render_options(&criteria_for(Kind::Choice, &opts, r));

        // The default: the identifier IS in front of the scorer, which
        // is what every number taken against this seat so far was.
        let shipped = texts(Rendering::IdMeans);
        assert!(
            shipped[0].starts_with("escalate_user_planning_unsubscription:"),
            "{shipped:?}"
        );

        for rendering in [Rendering::LabelMeans, Rendering::LabelAnchors] {
            let rendered = texts(rendering);
            for (o, text) in opts.iter().zip(&rendered) {
                assert!(
                    !text.contains(&o.id),
                    "{} put an identifier in front of the scorer: {text}",
                    rendering.as_str()
                );
            }
            // A tether with no label gets an ordinal, which carries no
            // topic; one that names a label gets that word.
            assert!(rendered[0].starts_with("option 1:"), "{rendered:?}");
            assert!(rendered[1].starts_with("refund:"), "{rendered:?}");
        }

        // Only `label-anchors` carries what the option is NOT — the
        // clause the cosine road has always scored.
        assert!(!texts(Rendering::LabelMeans)[0].contains("Not this when"));
        let anchored = texts(Rendering::LabelAnchors);
        assert!(
            anchored[0].contains("Not this when the state is instead: quoting or translating"),
            "{anchored:?}"
        );
        // `requires` is a question name, which has an identifier's
        // shape, and stays out of every rendering.
        for r in [
            Rendering::IdMeans,
            Rendering::LabelMeans,
            Rendering::LabelAnchors,
        ] {
            assert!(!texts(r)[0].contains("is_request"), "{r:?}");
        }
    }

    fn bare(id: &str, means: &str) -> crate::Opt {
        crate::Opt {
            id: id.to_string(),
            tether: crate::Tether {
                means: means.to_string(),
                not: Vec::new(),
                requires: Vec::new(),
                label: None,
            },
        }
    }

    /// With no contraries there are no words of ours to add, so there is
    /// no punctuation of ours either: the two label roads render an
    /// option byte for byte the same. Measured 2026-09-24 on CLINC
    /// banking (§139a): the one trailing full stop this road used to add
    /// moved 70 of 173 tops, so every anchors-vs-means contrast before
    /// this test was a punctuation contrast.
    #[test]
    fn label_means_and_label_anchors_render_byte_identically_when_no_option_has_a_contrary() {
        let opts = vec![
            bare(
                "freeze_account",
                "the person is asking about freeze account",
            ),
            bare("halt", "Halt the agent now."),
            bare("none_of_these", "something  else,  spaced oddly "),
        ];
        for kind in [Kind::Choice, Kind::Score] {
            let means = render_options(&criteria_for(kind, &opts, Rendering::LabelMeans));
            let anchors = render_options(&criteria_for(kind, &opts, Rendering::LabelAnchors));
            assert_eq!(
                means, anchors,
                "{kind:?}: the two label roads differ on options with no contrary"
            );
        }
    }

    /// A rendering never adds or removes a character of the caller's
    /// meaning: a full stop, a missing one, a doubled space are the
    /// caller's and reach the scorer as written, on every road. A
    /// normaliser would hide a sensitivity the model has rather than
    /// remove it; the author of the text owns the text.
    #[test]
    fn a_rendering_keeps_the_callers_meaning_byte_for_byte_when_it_adds_no_words() {
        for t in [
            "x",
            "x.",
            "x..",
            " x ",
            "x\n",
            "a  b",
            "Halt the agent now.",
        ] {
            let opt = bare("o", t);
            assert_eq!(opt.scored_text(), t, "scored_text altered {t:?}");
            for r in [
                Rendering::IdMeans,
                Rendering::LabelMeans,
                Rendering::LabelAnchors,
            ] {
                let rendered =
                    render_options(&criteria_for(Kind::Choice, std::slice::from_ref(&opt), r));
                let rest = rendered[0]
                    .split_once(": ")
                    .map(|(_, rest)| rest)
                    .unwrap_or(&rendered[0]);
                assert_eq!(rest, t, "{}: the meaning {t:?} was altered", r.as_str());
            }
        }
    }

    /// With contraries there ARE words of ours, and they carry our
    /// punctuation: one stop closes the caller's meaning before the
    /// clause (their own trailing stops fold into it, never doubled),
    /// one closes the clause; an anchor that is only whitespace is no
    /// anchor at all.
    #[test]
    fn an_anchored_text_joins_our_words_with_our_punctuation_and_keeps_the_callers() {
        let with = |means: &str, not: &[&str]| crate::Opt {
            id: "o".to_string(),
            tether: crate::Tether {
                means: means.to_string(),
                not: not.iter().map(|n| n.to_string()).collect(),
                requires: Vec::new(),
                label: None,
            },
        };
        let want = "the person wants money back. Not this when the state is instead: quoting it; asking what it is.";
        assert_eq!(
            with(
                "the person wants money back",
                &["quoting it", " asking what it is "]
            )
            .scored_text(),
            want
        );
        assert_eq!(
            with(
                "the person wants money back.",
                &["quoting it", "asking what it is"]
            )
            .scored_text(),
            want
        );
        assert_eq!(with("x", &["  "]).scored_text(), "x");
    }

    /// Every fixture this crate ships renders the same on both label
    /// roads wherever an option has no contrary — the property the
    /// 2026-09-24 records lacked. Walks `tests/data/*.jsonl` headers.
    #[test]
    fn every_shipped_fixture_renders_the_same_on_both_label_roads_where_it_has_no_contrary() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let mut seen = 0usize;
        for entry in std::fs::read_dir(&dir).expect("tests/data") {
            let path = entry.expect("an entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("a fixture");
            let header: Value =
                serde_json::from_str(text.lines().next().expect("a header line")).expect("json");
            for q in header["questions"].as_array().expect("questions") {
                let kind = match q["kind"].as_str() {
                    Some("choice") => Kind::Choice,
                    Some("score") => Kind::Score,
                    _ => continue,
                };
                let opts: Vec<crate::Opt> = q["options"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|o| serde_json::from_value(o.clone()).expect("an option"))
                    .filter(|o: &crate::Opt| o.tether.not.iter().all(|n| n.trim().is_empty()))
                    .collect();
                if opts.is_empty() {
                    continue;
                }
                seen += opts.len();
                assert_eq!(
                    render_options(&criteria_for(kind, &opts, Rendering::LabelMeans)),
                    render_options(&criteria_for(kind, &opts, Rendering::LabelAnchors)),
                    "{}: question {}",
                    path.display(),
                    q["name"]
                );
            }
        }
        assert!(
            seen > 0,
            "no shipped fixture option without a contrary was walked"
        );
    }

    /// A rendering nobody implemented is refused by name rather than
    /// quietly measured as the default.
    #[test]
    fn an_unknown_rendering_is_refused_by_name() {
        assert_eq!(
            Rendering::parse("label-anchors"),
            Ok(Rendering::LabelAnchors)
        );
        let why = Rendering::parse("anchors").expect_err("not a rendering");
        assert!(
            why.contains("anchors") && why.contains("label-anchors"),
            "{why}"
        );
        assert_eq!(Rendering::default(), Rendering::IdMeans);
    }

    /// The two roads are the configuration's vocabulary, and `normal`
    /// must never resolve to the identifier: that is a legacy default a
    /// caller now asks for by its own name or not at all.
    #[test]
    fn the_two_roads_are_nameable_and_normal_is_not_the_identifier() {
        assert_eq!(Rendering::parse("normal"), Ok(Rendering::LabelMeans));
        assert_eq!(Rendering::parse("semantic"), Ok(Rendering::LabelAnchors));
        assert_ne!(Rendering::parse("normal"), Ok(Rendering::IdMeans));

        assert_eq!(Rendering::IdMeans.road(), Road::Normal);
        assert_eq!(Rendering::LabelMeans.road(), Road::Normal);
        assert_eq!(Rendering::LabelAnchors.road(), Road::Semantic);

        // Exactly one rendering scores the tether, and it is the one the
        // cosine road has always scored.
        let semantic: Vec<_> = [
            Rendering::IdMeans,
            Rendering::LabelMeans,
            Rendering::LabelAnchors,
        ]
        .into_iter()
        .filter(|r| r.road() == Road::Semantic)
        .collect();
        assert_eq!(semantic, vec![Rendering::LabelAnchors]);

        // The refusal names both roads and every rendering, so a bad
        // configuration says what it could have been.
        let why = Rendering::parse("anchors").expect_err("not a rendering");
        for word in [
            "normal",
            "semantic",
            "id-means",
            "label-means",
            "label-anchors",
        ] {
            assert!(why.contains(word), "{why} omits {word}");
        }
    }

    use super::*;
    use serde_json::json;

    fn choice(pairs: &[(&str, Value)]) -> Criteria {
        Criteria::Choice(
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        )
    }

    #[test]
    fn an_option_with_no_description_is_scored_as_its_bare_label() {
        let c = choice(&[
            ("refund", json!("the person wants their money back")),
            ("other", Value::Null),
            ("blank", json!("")),
        ]);
        assert_eq!(
            render_options(&c),
            vec![
                "refund: the person wants their money back".to_string(),
                "other".to_string(),
                "blank".to_string(),
            ]
        );
    }

    #[test]
    fn zero_and_false_are_criteria_a_caller_meant() {
        // The hazard: a falsiness test instead of a null test drops
        // these two and renames both options to their labels.
        let c = choice(&[("none", json!(0)), ("no", json!(false))]);
        assert_eq!(
            render_options(&c),
            vec!["none: 0".to_string(), "no: false".to_string()]
        );
    }

    #[test]
    fn a_structured_criterion_reads_as_json_with_python_spacing() {
        let c = choice(&[("rubric", json!({"desc": "a", "n": 2}))]);
        assert_eq!(render_options(&c), vec![r#"rubric: {"desc": "a", "n": 2}"#]);
        assert_eq!(render_criterion(&json!([1, 2])), "[1, 2]");
        // A string is never quoted — it passes through as itself.
        assert_eq!(render_criterion(&json!("plain")), "plain");
    }

    #[test]
    fn a_score_renders_one_level_per_index() {
        let c = Criteria::Score(vec![
            json!("not at all"),
            json!("somewhat"),
            json!("entirely"),
        ]);
        assert_eq!(
            render_options(&c),
            vec![
                "level 0: not at all".to_string(),
                "level 1: somewhat".to_string(),
                "level 2: entirely".to_string(),
            ]
        );
    }

    #[test]
    fn a_noul_is_always_two_options_and_false_comes_first() {
        let bare = Criteria::Noul {
            when_false: None,
            when_true: None,
        };
        assert_eq!(
            render_options(&bare),
            vec![format!("false: {NOUL_FALSE}"), format!("true: {NOUL_TRUE}"),]
        );
        let described = Criteria::Noul {
            when_false: Some(json!("the person is quoting it")),
            when_true: Some(json!("the person is asking for it")),
        };
        assert_eq!(
            render_options(&described),
            vec![
                "false: the person is quoting it".to_string(),
                "true: the person is asking for it".to_string(),
            ]
        );
        // An empty side falls back rather than rendering "false: ".
        let half = Criteria::Noul {
            when_false: Some(json!("")),
            when_true: Some(json!("they are asking")),
        };
        assert_eq!(render_options(&half)[0], format!("false: {NOUL_FALSE}"));
    }

    #[test]
    fn the_bucket_boundaries_are_the_fitted_ones() {
        assert_eq!(qtype(Kind::Choice), 0);
        assert_eq!(qtype(Kind::Score), 1);
        assert_eq!(qtype(Kind::Noul), 2);
        // A one-option question shares the two-option bucket: the table
        // has no row below "2", and falling through to the type's
        // temperature instead would silently use a different number.
        assert_eq!(temp_bucket(0, 1), "choice:2");
        assert_eq!(temp_bucket(0, 2), "choice:2");
        assert_eq!(temp_bucket(0, 3), "choice:3-5");
        assert_eq!(temp_bucket(0, 5), "choice:3-5");
        assert_eq!(temp_bucket(0, 6), "choice:6-10");
        assert_eq!(temp_bucket(0, 10), "choice:6-10");
        assert_eq!(temp_bucket(0, 11), "choice:11+");
        assert_eq!(temp_bucket(1, 4), "score:3-5");
        assert_eq!(temp_bucket(2, 2), "noul:2");
    }

    fn shipped() -> AgentConfig {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/laya_rl_agent_config.json"
        ))
        .expect("the shipped calibration");
        AgentConfig::parse(&text).expect("it parses")
    }

    #[test]
    fn the_shipped_calibration_is_read_as_written() {
        let cfg = shipped();
        assert_eq!(cfg.max_len, 1024);
        assert_eq!(cfg.head_max_len, 256);
        assert_eq!(cfg.head_layers, Some(2));
        // `act_costs` prices one act; answering is the other.
        assert_eq!(cfg.num_acts, 2);

        let c = &cfg.calibration;
        // The per-type table, used where no bucket was fitted.
        assert!((c.temperature(Kind::Score, 7) as f64 - 1.0374259948730469).abs() < 1e-7);
        // The buckets that were.
        assert!((c.temperature(Kind::Choice, 4) as f64 - 1.7601518630981445).abs() < 1e-7);
        assert!((c.temperature(Kind::Choice, 8) as f64 - 1.0000158548355103).abs() < 1e-7);
        assert!((c.temperature(Kind::Choice, 2) as f64 - 1.9063563346862793).abs() < 1e-7);
        assert!((c.temperature(Kind::Score, 4) as f64 - 1.2514300346374512).abs() < 1e-7);
        assert!((c.temperature(Kind::Noul, 2) as f64 - 1.983399510383606).abs() < 1e-7);
    }

    #[test]
    fn the_bucket_that_would_sharpen_tenfold_is_refused_by_name() {
        let cfg = shipped();
        // `choice:11+` ships at 0.1006. Applied, it multiplies every
        // logit about tenfold: a 0.24 top probability would be published
        // as 0.99 and a caller gating on confidence would be told a coin
        // flip is a certainty.
        assert_eq!(cfg.calibration.temperature(Kind::Choice, 12), TEMP_MIN);
        assert_eq!(
            cfg.calibration.clamped().len(),
            1,
            "exactly one shipped value is outside the band: {:?}",
            cfg.calibration.clamped()
        );
        assert!(cfg.calibration.clamped()[0].starts_with("choice:11+="));

        // The guard itself, at every edge.
        assert_eq!(clamp_temperature(0.10058280825614929), TEMP_MIN);
        assert_eq!(clamp_temperature(0.0), TEMP_MIN);
        assert_eq!(clamp_temperature(-3.0), TEMP_MIN);
        assert_eq!(clamp_temperature(9.0), TEMP_MAX);
        assert_eq!(clamp_temperature(1.0), NEUTRAL);
        assert_eq!(clamp_temperature(f64::NAN), NEUTRAL);
        assert_eq!(clamp_temperature(f64::INFINITY), NEUTRAL);
        assert!((clamp_temperature(1.7601518630981445) as f64 - 1.7601518630981445).abs() < 1e-7);
    }

    /// A served temperature that the loader clamped names the number it
    /// was clamped from; one the checkpoint wrote as served names
    /// nothing. The bucket's own value first, the type's when the
    /// question fell back to it, and a value that was not a number has
    /// no number to name.
    #[test]
    fn a_clamped_bucket_names_the_value_it_was_clamped_from() {
        let cfg = shipped();
        let from = cfg
            .calibration
            .clamped_from(Kind::Choice, 16)
            .expect("choice:11+ was clamped");
        assert!((from as f64 - 0.10058280825614929).abs() < 1e-7, "{from}");
        assert_eq!(cfg.calibration.clamped_from(Kind::Choice, 4), None);
        assert_eq!(cfg.calibration.clamped_from(Kind::Noul, 2), None);
        assert_eq!(cfg.calibration.clamped_from(Kind::Score, 4), None);

        // No bucket at all: the type's value is what was served, and
        // what was clamped.
        let typed = AgentConfig::parse(r#"{"temperature": [12.0, 1.0, 1.0]}"#).expect("parses");
        assert_eq!(typed.calibration.temperature(Kind::Choice, 4), TEMP_MAX);
        assert_eq!(typed.calibration.clamped_from(Kind::Choice, 4), Some(12.0));
        assert_eq!(typed.calibration.clamped_from(Kind::Score, 4), None);

        // Not a number: refused by name, and nothing to be clamped from.
        let odd =
            AgentConfig::parse(r#"{"temperature_by_options": {"choice:2": "x"}}"#).expect("parses");
        assert_eq!(odd.calibration.clamped().len(), 1);
        assert_eq!(odd.calibration.clamped_from(Kind::Choice, 2), None);
        assert_eq!(odd.calibration.temperature(Kind::Choice, 2), NEUTRAL);
    }

    /// `head_layers` used to default to 2 in silence: a three-layer
    /// checkpoint without the key loaded two layers and answered wrongly
    /// without a word. Absent is now absent — the weights say how deep
    /// the head is — and a value that is there but unusable is refused
    /// by name, as every other key on this wire is.
    #[test]
    fn a_config_without_head_layers_names_none_not_two() {
        assert_eq!(AgentConfig::parse("{}").expect("parses").head_layers, None);
        assert_eq!(
            AgentConfig::parse(r#"{"head_layers": 3}"#)
                .expect("parses")
                .head_layers,
            Some(3)
        );
        for bad in [
            r#"{"head_layers": "two"}"#,
            r#"{"head_layers": 0}"#,
            r#"{"head_layers": -1}"#,
        ] {
            let why = AgentConfig::parse(bad).expect_err("an unusable head_layers is refused");
            assert!(
                why.contains("head_layers"),
                "the refusal names the key: {why}"
            );
        }
    }

    #[test]
    fn a_config_without_a_calibration_is_neutral_rather_than_refused() {
        let cfg = AgentConfig::parse("{}").expect("an empty config still names its defaults");
        assert_eq!(cfg.max_len, 512);
        assert_eq!(cfg.head_max_len, 192);
        assert_eq!(cfg.head_layers, None, "absent is absent, not two");
        assert_eq!(cfg.num_acts, 1);
        assert_eq!(cfg.calibration.temperature(Kind::Choice, 4), NEUTRAL);
        assert!(cfg.calibration.clamped().is_empty());
        assert!(AgentConfig::parse("not json").is_err());
    }

    #[test]
    fn the_readout_is_a_distribution_and_the_temperature_widens_it() {
        let logits = [3.0f32, 1.0, 0.0];
        let p = readout(&logits, 1.0);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[0] > p[1] && p[1] > p[2]);

        // Softer: every gap narrows, the order holds.
        let soft = readout(&logits, 5.0);
        assert!(soft[0] < p[0] && soft[2] > p[2]);

        // A flat row is uniform whatever the temperature.
        let flat = readout(&[2.0, 2.0, 2.0, 2.0], 1.0);
        for v in &flat {
            assert!((v - 0.25).abs() < 1e-6);
        }
        // A large logit does not overflow: the max is subtracted first.
        let big = readout(&[800.0, 0.0], 1.0);
        assert!(big[0].is_finite() && (big[0] - 1.0).abs() < 1e-6);
        assert!(readout(&[], 1.0).is_empty());
    }

    #[test]
    fn the_noul_flips_both_ways() {
        // The model's order: false first, the proposition second.
        let model = readout(&[0.2f32, 1.4], 1.0);
        assert!(model[1] > model[0], "the proposition is the second marker");
        assert_eq!(p_true(&model), model[1]);

        // The wire's order: the proposition first, because an
        // implication reads distribution[0] as "it holds".
        let wire = engine_order(Kind::Noul, &model);
        assert_eq!(wire[0], model[1]);
        assert_eq!(wire[1], model[0]);
        assert_eq!(wire[0], p_true(&model));
        assert!((wire.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        // Flipping twice is the identity, so nothing downstream can
        // double-apply it without the test noticing.
        assert_eq!(engine_order(Kind::Noul, &wire), model);

        // Neither other kind is touched.
        let three = readout(&[1.0f32, 2.0, 3.0], 1.0);
        assert_eq!(engine_order(Kind::Choice, &three), three);
        assert_eq!(engine_order(Kind::Score, &three), three);
    }

    #[test]
    fn confidence_reads_the_tail_for_a_choice_and_the_coin_for_a_noul() {
        let p = [0.7f32, 0.2, 0.1];
        assert_eq!(decision_confidence(Kind::Choice, &p), crate::confidence(&p));
        assert_eq!(decision_confidence(Kind::Score, &p), crate::confidence(&p));

        // A noul reads max(p, 1-p) — the distance from a coin flip —
        // and reads it the same in either ordering.
        let model = [0.25f32, 0.75];
        assert!((decision_confidence(Kind::Noul, &model) - 0.75).abs() < 1e-6);
        let wire = engine_order(Kind::Noul, &model);
        assert!((decision_confidence(Kind::Noul, &wire) - 0.75).abs() < 1e-6);
        // An even split is no confidence at all, not a half-certainty
        // about the entropy.
        assert!((decision_confidence(Kind::Noul, &[0.5, 0.5]) - 0.5).abs() < 1e-6);
    }
}
