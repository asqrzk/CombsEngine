//! The record: one shape every subcommand writes, and the one rule that
//! makes it worth trusting.
//!
//! The disease this cures is real and current. The platform's measured
//! throughput survives only inside display strings in its catalog
//! ("~615 MB · 79 tok/s"); the engine's survives only in a private
//! markdown ledger. So an outside audit quoted a figure from before the
//! kernels existed, misjudged the gap by half, and nothing in the tree
//! could contradict it. A measurement that cannot be compared later is
//! a story, not a number.
//!
//! THE RULE: every value carries the tier of knowledge behind it, and a
//! value that is not measured must name its basis. There is no
//! constructor that lets a derived number be born labelled measured.

use serde::{Deserialize, Serialize};

/// How this number came to be known. The estate already holds this law
/// in two places and honours it in neither consistently: the platform's
/// capability tiers and the private toolkit's measured-vs-derived rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// This run read it off the machine.
    Measured,
    /// Arithmetic over measured values. `basis` carries the formula.
    Derived,
    /// A model or an extrapolation beyond what was measured.
    Estimated,
    /// Someone said so. `basis` carries who.
    Declared,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Value {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub tier: Tier,
    /// Required for every tier but `Measured`; the formula or the source.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub basis: Option<String>,
}

impl Value {
    pub fn measured(name: impl Into<String>, value: f64, unit: impl Into<String>) -> Self {
        Self { name: name.into(), value, unit: unit.into(), tier: Tier::Measured, basis: None }
    }

    pub fn derived(
        name: impl Into<String>,
        value: f64,
        unit: impl Into<String>,
        basis: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            value,
            unit: unit.into(),
            tier: Tier::Derived,
            basis: Some(basis.into()),
        }
    }

    pub fn estimated(
        name: impl Into<String>,
        value: f64,
        unit: impl Into<String>,
        basis: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            value,
            unit: unit.into(),
            tier: Tier::Estimated,
            basis: Some(basis.into()),
        }
    }

    /// True when this value is allowed to stand as written: a
    /// non-measured value without a basis is malformed, and the writer
    /// refuses the whole record rather than emit it.
    pub fn well_formed(&self) -> bool {
        match self.tier {
            Tier::Measured => true,
            _ => self.basis.as_ref().is_some_and(|b| !b.trim().is_empty()),
        }
    }
}

/// One subject measured once: a model at a depth, a kernel cell, a box.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub id: String,
    pub kind: String,
    pub subject: String,
    #[serde(default)]
    pub config: serde_json::Value,
    /// False marks a first, cold run. The house rule is that a reported
    /// timing is the warm second run of a pair; a cold one may be
    /// recorded but must say so.
    pub warm: bool,
    pub values: Vec<Value>,
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Whether this box was trusted at the moment of measurement. A machine
/// that fails the doctor can still be measured — the numbers are simply
/// stamped, because a software rasterizer serves plausible garbage at a
/// crawl and calls itself healthy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trust {
    pub doctor: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verdict {
    pub claim: String,
    pub holds: bool,
    pub evidence: String,
}

pub const SCHEMA: &str = "combs-kit.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub schema: String,
    pub kit: String,
    pub build: serde_json::Value,
    pub machine: serde_json::Value,
    pub trust: Trust,
    #[serde(default)]
    pub runs: Vec<Run>,
    #[serde(default)]
    pub verdicts: Vec<Verdict>,
}

impl Record {
    pub fn new(machine: serde_json::Value, build: serde_json::Value, trust: Trust) -> Self {
        Self {
            schema: SCHEMA.to_string(),
            kit: format!("combs-kit {}", env!("CARGO_PKG_VERSION")),
            build,
            machine,
            trust,
            runs: Vec::new(),
            verdicts: Vec::new(),
        }
    }

    /// Every malformed value in the record, named. The writer calls this
    /// and refuses rather than emit a number whose provenance is a lie.
    pub fn malformed(&self) -> Vec<String> {
        self.runs
            .iter()
            .flat_map(|r| {
                r.values
                    .iter()
                    .filter(|v| !v.well_formed())
                    .map(move |v| format!("{}/{}: {:?} without a basis", r.id, v.name, v.tier))
            })
            .collect()
    }

    pub fn write(&self, to: Option<&std::path::Path>) -> Result<(), String> {
        let bad = self.malformed();
        if !bad.is_empty() {
            return Err(format!(
                "refusing to write a record with unsourced values: {}",
                bad.join("; ")
            ));
        }
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        match to {
            Some(p) => std::fs::write(p, text).map_err(|e| format!("{}: {e}", p.display())),
            None => {
                println!("{text}");
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_with(v: Value) -> Run {
        Run {
            id: "r0".into(),
            kind: "probe".into(),
            subject: "none".into(),
            config: serde_json::Value::Null,
            warm: true,
            values: vec![v],
            notes: vec![],
        }
    }

    fn record_with(v: Value) -> Record {
        let mut r = Record::new(
            serde_json::json!({}),
            serde_json::json!({}),
            Trust { doctor: "not run".into(), detail: String::new() },
        );
        r.runs.push(run_with(v));
        r
    }

    /// The rule the whole record exists for: arithmetic cannot pass
    /// itself off as a reading.
    #[test]
    fn a_derived_value_carries_its_basis_and_a_measured_one_needs_none() {
        let m = Value::measured("decode", 31.9, "tok/s");
        assert!(m.well_formed());
        assert_eq!(m.tier, Tier::Measured);

        let d = Value::derived("us_per_dispatch", 16.2, "us", "1e6 / decode_tok_s / disp_per_tok");
        assert!(d.well_formed());
        assert_eq!(d.tier, Tier::Derived);
        assert!(d.basis.as_deref().is_some_and(|b| b.contains("decode_tok_s")));
    }

    #[test]
    fn a_record_refuses_to_write_an_unsourced_derived_value() {
        let mut bad = Value::derived("x", 1.0, "u", "formula");
        bad.basis = None; // only reachable by hand; the constructors cannot
        let rec = record_with(bad);
        let err = rec.write(None).unwrap_err();
        assert!(err.contains("unsourced"), "the refusal must name the problem: {err}");
        assert!(err.contains("r0/x"), "and the value: {err}");
    }

    #[test]
    fn a_record_round_trips_and_pins_its_schema() {
        let rec = record_with(Value::measured("decode", 31.9, "tok/s"));
        let text = serde_json::to_string(&rec).unwrap();
        let back: Record = serde_json::from_str(&text).unwrap();
        assert_eq!(back.schema, SCHEMA);
        assert_eq!(back.runs[0].values[0].value, 31.9);
        assert!(
            !text.contains("\"basis\""),
            "a measured value writes no empty basis key"
        );
    }

    /// A cold run may be recorded; it may not be recorded silently.
    #[test]
    fn warmth_is_part_of_the_record() {
        let mut rec = record_with(Value::measured("decode", 12.0, "tok/s"));
        rec.runs[0].warm = false;
        let text = serde_json::to_string(&rec).unwrap();
        assert!(text.contains("\"warm\":false"));
    }
}
