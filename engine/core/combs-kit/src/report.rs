//! Turning a record into the two things people actually read: a ledger
//! section, and a verdict on whether something got worse.
//!
//! The ledger's sections have been typed by hand for months, which is
//! why they drift in shape and why a number in one is hard to compare
//! with a number in another. A record already holds everything a section
//! needs; rendering it is not a convenience, it is what stops the
//! measurement and its write-up from disagreeing.

use crate::record::{Record, Tier};

/// Which way is better for a given figure. A diff that does not know
/// this cannot tell a win from a regression, and a tool that reports
/// "changed" for both is no use at a gate.
#[derive(PartialEq, Clone, Copy)]
pub enum Better {
    Higher,
    Lower,
    /// Neither: a count that describes the run rather than judging it.
    Neither,
}

pub fn better(name: &str, unit: &str) -> Better {
    match unit {
        "tokens/s" => Better::Higher,
        "ms" | "us" | "MB" | "dispatches/token" | "submits/token" => Better::Lower,
        "dispatches" if name.starts_with("prefill") || name.starts_with("load") => Better::Lower,
        // prompt_tokens, generated_tokens: what was asked for, not a score.
        _ => Better::Neither,
    }
}

/// Render a record as the house's ledger section.
pub fn markdown(rec: &Record, section: &str, claim: &str, date: &str, branch: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!("## §{section} — {claim} ({date}, {branch})\n\n"));

    let m = &rec.machine;
    let cpu = m["cpu"].as_str().unwrap_or("?");
    let gpu = m["gpu"]["name"].as_str().unwrap_or("?");
    let backend = m["gpu"]["backend"].as_str().unwrap_or("?");
    let ram = m["ram_bytes"].as_u64().map(|b| b / 1_073_741_824);
    let power = m["power"].as_str().unwrap_or("unknown");
    let dtype = rec.build["dtype"].as_str().unwrap_or("?");
    let fusion = rec.build["fusion"].as_bool().unwrap_or(false);
    out.push_str(&format!(
        "{cpu}{}, {gpu} / {backend}, {power} power; {dtype} build{}; \
         every figure the warm second run of a pair, each subject its own process.\n\n",
        ram.map(|g| format!(" {g} GB")).unwrap_or_default(),
        if fusion { " with fusion" } else { " without fusion" },
    ));
    // Trust is stated, never assumed.
    out.push_str(&format!(
        "Trust: doctor {} — {}\n\n",
        rec.trust.doctor, rec.trust.detail
    ));

    // One column per run, one row per figure, so depths sit side by side
    // the way a reader compares them.
    let mut names: Vec<&str> = Vec::new();
    for r in &rec.runs {
        for v in &r.values {
            if !names.contains(&v.name.as_str()) {
                names.push(&v.name);
            }
        }
    }
    if names.is_empty() {
        out.push_str("No measurements: every subject failed. The notes:\n\n");
        for r in &rec.runs {
            for n in &r.notes {
                out.push_str(&format!("- {}: {n}\n", r.id));
            }
        }
        return out;
    }

    out.push_str("| figure | unit |");
    for r in &rec.runs {
        out.push_str(&format!(" {} |", r.id));
    }
    out.push_str("\n|---|---|");
    for _ in &rec.runs {
        out.push_str("---|");
    }
    out.push('\n');
    for name in &names {
        let unit = rec
            .runs
            .iter()
            .flat_map(|r| &r.values)
            .find(|v| v.name == *name)
            .map(|v| v.unit.clone())
            .unwrap_or_default();
        out.push_str(&format!("| {name} | {unit} |"));
        for r in &rec.runs {
            match r.values.iter().find(|v| v.name == *name) {
                Some(v) => {
                    // A derived number is MARKED in the table, not only in
                    // the JSON. A reader must not have to open the record
                    // to learn that a figure is arithmetic. And a figure
                    // that MOVED across its readings is printed as the
                    // range it is, because a median alone would read as a
                    // fact — which is how this ledger published an 8.4x
                    // that was really 3.5x.
                    let mark = if v.tier == Tier::Measured { "" } else { "*" };
                    match v.spread_pct {
                        Some(sp) if sp > crate::record::SPREAD_BAND_PCT => {
                            out.push_str(&format!(" {:.2}{mark} ±{sp:.0}% |", v.value));
                        }
                        _ => out.push_str(&format!(" {:.2}{mark} |", v.value)),
                    }
                }
                None => out.push_str(" — |"),
            }
        }
        out.push('\n');
    }
    if rec
        .runs
        .iter()
        .flat_map(|r| &r.values)
        .any(|v| v.tier != Tier::Measured)
    {
        out.push_str("\n`*` derived, not measured:\n");
        let mut seen: Vec<&str> = Vec::new();
        for v in rec.runs.iter().flat_map(|r| &r.values) {
            if v.tier != Tier::Measured && !seen.contains(&v.name.as_str()) {
                seen.push(&v.name);
                out.push_str(&format!(
                    "- {} = {}\n",
                    v.name,
                    v.basis.as_deref().unwrap_or("(no basis recorded)")
                ));
            }
        }
    }
    let unsteady: Vec<String> = rec
        .runs
        .iter()
        .flat_map(|r| r.values.iter().map(move |v| (r, v)))
        .filter(|(_, v)| v.spread_pct.is_some_and(|sp| sp > crate::record::SPREAD_BAND_PCT))
        .map(|(r, v)| format!("- {}/{}: {}", r.id, v.name, v.steady().unwrap_err()))
        .collect();
    if !unsteady.is_empty() {
        out.push_str(
            "\nThese figures MOVED across their readings and are ranges, not \
             numbers. Do not quote them as facts:\n",
        );
        for line in unsteady {
            out.push_str(&line);
            out.push('\n');
        }
    }
    let failed: Vec<&crate::record::Run> =
        rec.runs.iter().filter(|r| r.values.is_empty()).collect();
    if !failed.is_empty() {
        out.push_str("\nSubjects that produced nothing, and why:\n");
        for r in failed {
            out.push_str(&format!("- {}: {}\n", r.id, r.notes.join("; ")));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{Run, Trust, Value};

    fn rec_with(values: Vec<Value>) -> Record {
        let mut r = Record::new(
            serde_json::json!({"cpu": "Apple M3 Pro", "ram_bytes": 19327352832u64,
                "power": "ac", "gpu": {"name": "Apple M3 Pro", "backend": "Metal"}}),
            serde_json::json!({"dtype": "f32", "fusion": true}),
            Trust { doctor: "not run".into(), detail: "x".into() },
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

    /// A derived figure must be marked in the TABLE. A reader comparing
    /// two sections will not open the JSON to find out that one of the
    /// columns is arithmetic.
    #[test]
    fn a_derived_figure_is_marked_and_its_basis_printed() {
        let rec = rec_with(vec![
            Value::measured("decode_tok_s", 31.9, "tokens/s"),
            Value::derived("us_per_dispatch", 16.2, "us", "1e6 / decode / disp"),
        ]);
        let md = markdown(&rec, "99", "a claim", "2026-09-12", "w/x");
        assert!(md.contains("| decode_tok_s | tokens/s | 31.90 |"));
        assert!(md.contains("16.20* |"), "derived values are starred: {md}");
        assert!(md.contains("1e6 / decode / disp"), "the basis is printed");
    }

    /// The machine line is not decoration: a section without the power
    /// source and the build cannot be compared with another.
    #[test]
    fn the_machine_line_carries_power_and_build() {
        let md = markdown(
            &rec_with(vec![Value::measured("decode_tok_s", 1.0, "tokens/s")]),
            "99",
            "c",
            "d",
            "b",
        );
        assert!(md.contains("ac power"));
        assert!(md.contains("f32 build with fusion"));
        assert!(md.contains("Trust: doctor not run"));
    }

    /// A record where everything failed must render as a record of
    /// failure, not as an empty table that reads like a clean run.
    #[test]
    fn a_record_with_no_measurements_says_so() {
        let mut rec = rec_with(vec![]);
        rec.runs[0].notes = vec!["no measurement: cannot open the model".into()];
        let md = markdown(&rec, "99", "c", "d", "b");
        assert!(md.contains("every subject failed"));
        assert!(md.contains("cannot open the model"));
    }

    #[test]
    fn direction_knows_a_win_from_a_regression() {
        assert!(better("decode_tok_s", "tokens/s") == Better::Higher);
        assert!(better("ttft_ms", "ms") == Better::Lower);
        assert!(better("dispatches_per_token", "dispatches/token") == Better::Lower);
        assert!(better("prompt_tokens", "tokens") == Better::Neither);
    }
}
