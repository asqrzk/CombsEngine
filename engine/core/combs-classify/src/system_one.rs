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
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, Spaced);
    v.serialize(&mut ser).expect("a Value serialises");
    String::from_utf8(buf).expect("serde_json writes utf-8")
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
}

impl Calibration {
    /// No calibration: every temperature neutral.
    pub fn neutral() -> Self {
        Self {
            per_type: [NEUTRAL; 3],
            by_bucket: BTreeMap::new(),
            clamped: Vec::new(),
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
    /// Layers in the transformer head over the markers.
    pub head_layers: usize,
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
            head_layers: usize_at("head_layers", 2),
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
        assert_eq!(cfg.head_layers, 2);
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

    #[test]
    fn a_config_without_a_calibration_is_neutral_rather_than_refused() {
        let cfg = AgentConfig::parse("{}").expect("an empty config still names its defaults");
        assert_eq!(cfg.max_len, 512);
        assert_eq!(cfg.head_max_len, 192);
        assert_eq!(cfg.head_layers, 2);
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
