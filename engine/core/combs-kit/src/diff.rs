//! Did it get worse? The question a measurement exists to answer twice:
//! once when it is taken, and again when something changes.
//!
//! Two rules keep this honest. A diff across DIFFERENT MACHINES is
//! nearly meaningless and is refused; a diff across different BUILDS is
//! meaningful and common (f16 against f32 is the estate's live question)
//! so it is allowed and labelled loudly rather than refused. And the
//! band is wide by default, because this desk has produced a 2.4x
//! outlier on a single reading and a tool that cries regression at noise
//! is a tool people learn to ignore.

use crate::record::Record;
use crate::report::{better, Better};

pub struct Verdict {
    pub lines: Vec<String>,
    pub regressions: usize,
    pub refused: Option<String>,
}

fn machine_of(rec: &Record) -> String {
    format!(
        "{} / {}",
        rec.machine["cpu"].as_str().unwrap_or("?"),
        rec.machine["gpu"]["name"].as_str().unwrap_or("?"),
    )
}

fn build_of(rec: &Record) -> String {
    format!(
        "{}{}",
        rec.build["dtype"].as_str().unwrap_or("?"),
        if rec.build["fusion"].as_bool().unwrap_or(false) {
            "+fusion"
        } else {
            ""
        },
    )
}

/// `tolerance_pct` is how far a figure may move before it counts.
pub fn compare(a: &Record, b: &Record, tolerance_pct: f64) -> Verdict {
    let mut lines = Vec::new();
    let mut regressions = 0;

    if machine_of(a) != machine_of(b) {
        return Verdict {
            lines: vec![],
            regressions: 0,
            refused: Some(format!(
                "these records are from different machines ({} against {}), and \
                 comparing them would say more about the desks than the change",
                machine_of(a),
                machine_of(b)
            )),
        };
    }
    if build_of(a) != build_of(b) {
        lines.push(format!(
            "NOTE: different builds — {} against {}. That is a legitimate \
             comparison and the most common one here, but it is not a regression \
             check; it is a choice between two roads.",
            build_of(a),
            build_of(b)
        ));
    }
    if a.machine["power"] != b.machine["power"] {
        lines.push(format!(
            "WARNING: different power sources — {} against {}. A laptop on \
             battery throttles; treat every figure below as suspect.",
            a.machine["power"].as_str().unwrap_or("?"),
            b.machine["power"].as_str().unwrap_or("?"),
        ));
    }

    for run_a in &a.runs {
        let Some(run_b) = b.runs.iter().find(|r| r.id == run_a.id) else {
            lines.push(format!("{}: only in the first record", run_a.id));
            continue;
        };
        for va in &run_a.values {
            let Some(vb) = run_b.values.iter().find(|v| v.name == va.name) else {
                continue;
            };
            let dir = better(&va.name, &va.unit);
            if dir == Better::Neither || va.value == 0.0 {
                continue;
            }
            let pct = (vb.value - va.value) / va.value * 100.0;
            if pct.abs() < tolerance_pct {
                continue;
            }
            let worse = match dir {
                Better::Higher => pct < 0.0,
                Better::Lower => pct > 0.0,
                Better::Neither => false,
            };
            if worse {
                regressions += 1;
            }
            lines.push(format!(
                "{:<6} {:<22} {:>10.2} → {:>10.2} {:<16} {:+.1}%  {}",
                run_a.id,
                va.name,
                va.value,
                vb.value,
                va.unit,
                pct,
                if worse { "WORSE" } else { "better" },
            ));
        }
    }
    if lines.is_empty() {
        lines.push(format!(
            "nothing moved by more than {tolerance_pct}% — the two records agree"
        ));
    }
    Verdict { lines, regressions, refused: None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{Run, Trust, Value};

    fn rec(cpu: &str, dtype: &str, power: &str, values: Vec<Value>) -> Record {
        let mut r = Record::new(
            serde_json::json!({"cpu": cpu, "power": power, "gpu": {"name": "g"}}),
            serde_json::json!({"dtype": dtype, "fusion": true}),
            Trust { doctor: "pass".into(), detail: String::new() },
        );
        r.runs.push(Run {
            id: "d4".into(),
            kind: "model".into(),
            subject: "seat".into(),
            config: serde_json::Value::Null,
            warm: true,
            values,
            notes: vec![],
        });
        r
    }

    /// Slower decode is a regression; fewer dispatches is not. A diff
    /// that cannot tell those apart is useless at a gate.
    #[test]
    fn direction_decides_what_counts_as_worse() {
        let a = rec("m", "f32", "ac", vec![
            Value::measured("decode_tok_s", 100.0, "tokens/s"),
            Value::measured("dispatches_per_token", 1000.0, "dispatches/token"),
        ]);
        let b = rec("m", "f32", "ac", vec![
            Value::measured("decode_tok_s", 80.0, "tokens/s"),
            Value::measured("dispatches_per_token", 500.0, "dispatches/token"),
        ]);
        let v = compare(&a, &b, 5.0);
        assert_eq!(v.regressions, 1, "only the slower decode counts");
        assert!(v.lines.iter().any(|l| l.contains("decode_tok_s") && l.contains("WORSE")));
        assert!(v.lines.iter().any(|l| l.contains("dispatches_per_token") && l.contains("better")));
    }

    /// Different desks are refused. Comparing them would report on the
    /// machines rather than on the change.
    #[test]
    fn a_cross_machine_diff_is_refused_by_name() {
        let a = rec("M3 Pro", "f32", "ac", vec![Value::measured("decode_tok_s", 100.0, "tokens/s")]);
        let b = rec("Xeon", "f32", "ac", vec![Value::measured("decode_tok_s", 100.0, "tokens/s")]);
        let v = compare(&a, &b, 5.0);
        let why = v.refused.expect("refused");
        assert!(why.contains("different machines"));
        assert!(v.lines.is_empty());
    }

    /// Different builds are the estate's live question, so they are
    /// allowed — and labelled, because it is a choice, not a regression.
    #[test]
    fn a_cross_build_diff_is_allowed_and_labelled() {
        let a = rec("m", "f32", "ac", vec![Value::measured("decode_tok_s", 100.0, "tokens/s")]);
        let b = rec("m", "f16", "ac", vec![Value::measured("decode_tok_s", 30.0, "tokens/s")]);
        let v = compare(&a, &b, 5.0);
        assert!(v.refused.is_none());
        assert!(v.lines.iter().any(|l| l.contains("different builds")));
    }

    /// A battery run beside a mains run is the trap this desk has
    /// already fallen into; it must be shouted about, not silently
    /// tabulated.
    #[test]
    fn a_power_mismatch_is_shouted_about() {
        let a = rec("m", "f32", "ac", vec![Value::measured("decode_tok_s", 100.0, "tokens/s")]);
        let b = rec("m", "f32", "battery", vec![Value::measured("decode_tok_s", 60.0, "tokens/s")]);
        let v = compare(&a, &b, 5.0);
        assert!(v.lines.iter().any(|l| l.starts_with("WARNING: different power")));
    }

    #[test]
    fn agreement_inside_the_band_says_so_plainly() {
        let a = rec("m", "f32", "ac", vec![Value::measured("decode_tok_s", 100.0, "tokens/s")]);
        let b = rec("m", "f32", "ac", vec![Value::measured("decode_tok_s", 102.0, "tokens/s")]);
        let v = compare(&a, &b, 5.0);
        assert_eq!(v.regressions, 0);
        assert!(v.lines[0].contains("the two records agree"));
    }
}
