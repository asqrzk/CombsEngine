//! The System 1 seat, on the real checkpoint.
//!
//! Ignored by default: it wants the 842 MB laya checkpoint. Run it with
//!
//!     COMBS_TEST_LAYA=~/.cache/combs/models/laya-typed-decisions \
//!       cargo test --release -p combs-runtime --test laya_seat -- --ignored --nocapture
//!
//! `--dump <path>` is not a flag here; set `COMBS_LAYA_DUMP` to write the
//! answers as JSON for the python side to compare against.

use combs_classify::system_one::Criteria;
use combs_runtime::{DecisionEngine, DecisionQuestion};
use serde_json::json;

/// The option ids of one question, in the order the caller declared
/// them — which is the order its distribution comes back in.
fn option_ids(questions: &[DecisionQuestion], name: &str) -> Vec<String> {
    let q = questions
        .iter()
        .find(|q| q.name == name)
        .unwrap_or_else(|| panic!("no question {name}"));
    match &q.criteria {
        Criteria::Choice(opts) => opts.iter().map(|(id, _)| id.clone()).collect(),
        Criteria::Score(levels) => (0..levels.len()).map(|i| i.to_string()).collect(),
        Criteria::Noul { .. } => vec!["true".into(), "false".into()],
    }
}

fn seat() -> Option<DecisionEngine> {
    let dir = std::env::var("COMBS_TEST_LAYA").ok()?;
    Some(DecisionEngine::load(&dir).expect("the checkpoint seats"))
}

/// The three states and two questions the python reference runs too.
fn suite() -> (Vec<&'static str>, Vec<DecisionQuestion>) {
    let states = vec![
        "I want to cancel my subscription — please stop billing me.",
        "translate the phrase 'I'm considering unsubscription' to Arabic",
        "what's the difference between a refund and a chargeback?",
    ];
    let questions = vec![
        DecisionQuestion {
            name: "is_request".into(),
            instructions: "is the person asking us to do something, right now, in this message"
                .into(),
            criteria: Criteria::Noul {
                when_false: None,
                when_true: None,
            },
        },
        DecisionQuestion {
            name: "intent".into(),
            instructions: "which of these is the person asking for".into(),
            criteria: Criteria::Choice(vec![
                (
                    "unsubscribe".into(),
                    json!("the person is asking us to end their subscription now"),
                ),
                (
                    "refund".into(),
                    json!("the person is asking for money back"),
                ),
                (
                    "translate".into(),
                    json!("the person is asking us to put text into another language"),
                ),
                (
                    "explain".into(),
                    json!("the person is asking what something means or how it works"),
                ),
                ("none".into(), json!("none of these")),
            ]),
        },
    ];
    (states, questions)
}

#[test]
#[ignore = "needs the laya checkpoint (COMBS_TEST_LAYA)"]
fn the_seat_answers_the_real_checkpoint() {
    let Some(engine) = seat() else {
        panic!("set COMBS_TEST_LAYA to the checkpoint directory");
    };
    let (states, questions) = suite();
    let mut dump = Vec::new();

    for state in &states {
        let answers = engine.decide(state, &questions).expect("it answers");
        assert_eq!(answers.len(), questions.len());
        println!("\nstate: {state}");
        let mut row = serde_json::Map::new();
        row.insert("state".into(), json!(state));
        let mut out = Vec::new();
        for a in &answers {
            let sum: f32 = a.distribution.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "{} is not a distribution: {:?}",
                a.name,
                a.distribution
            );
            assert_eq!(a.distribution.len(), a.logits.len());
            assert!(
                (0.0..=1.0).contains(&a.p_act),
                "p_act {} is not a probability",
                a.p_act
            );
            println!(
                "  {:<11} T {:.4} ({:<10}) p_act {:.4}  {:?}",
                a.name,
                a.temperature,
                a.bucket,
                a.p_act,
                a.distribution
                    .iter()
                    .map(|p| (p * 1000.0).round() / 1000.0)
                    .collect::<Vec<_>>()
            );
            out.push(json!({
                "name": a.name,
                "distribution": a.distribution,
                "logits": a.logits,
                "temperature": a.temperature,
                "bucket": a.bucket,
                "p_act": a.p_act,
            }));
        }
        row.insert("answers".into(), json!(out));
        dump.push(serde_json::Value::Object(row));
    }

    // Against laya's own answers to exactly these questions, recorded by
    // tests/fixtures/laya_reference.py. The tiny fixture proves the
    // arithmetic; this proves 28 layers of it at the real rotary bases
    // and the real calibration land where laya lands.
    let expected: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../combs-models/tests/fixtures/laya_real_expected.json"
    )))
    .expect("the reference fixture is json");
    let mut worst = 0.0f32;
    let mut at = String::new();
    for (row, want_row) in dump.iter().zip(expected.as_array().expect("rows")) {
        assert_eq!(
            row["state"], want_row["state"],
            "the fixture asks other states"
        );
        for a in row["answers"].as_array().expect("answers") {
            let name = a["name"].as_str().expect("a name");
            let w = &want_row["answers"][name];
            let got: Vec<f32> = serde_json::from_value(a["distribution"].clone()).expect("p");
            // laya reports a noul as p_true alone; our wire puts the
            // proposition FIRST, so index zero is what it compares to.
            // BY NAME, never by position: serde_json's map is ordered
            // alphabetically and the option order is the caller's, so
            // comparing the two positionally lines `unsubscribe` up
            // against `explain` and calls a correct port wrong.
            let want: Vec<f32> = if w["type"] == "noul" {
                vec![w["noul"].as_f64().expect("noul") as f32]
            } else {
                let ids = option_ids(&questions, name);
                ids.iter()
                    .map(|id| {
                        w["probabilities"][id]
                            .as_f64()
                            .unwrap_or_else(|| panic!("the fixture has no {name}.{id}"))
                            as f32
                    })
                    .collect()
            };
            for (i, want_p) in want.iter().enumerate() {
                let d = (got[i] - want_p).abs();
                if d > worst {
                    worst = d;
                    at = format!("{name}[{i}] {:.6} vs {want_p:.6}", got[i]);
                }
            }
            let want_act = w["action"]["act_probability"].as_f64().expect("act") as f32;
            let got_act = a["p_act"].as_f64().expect("p_act") as f32;
            let d = (got_act - want_act).abs();
            if d > worst {
                worst = d;
                at = format!("{name}.p_act {got_act:.6} vs {want_act:.6}");
            }
        }
    }
    // Two bounds, because the two builds hold the weights differently.
    // The checkpoint is F16 on disk: the f32 build WIDENS it and should
    // reproduce laya's own f32 run to the floor this comparison can show
    // — laya rounds every published figure to four decimals, so 5e-5 is
    // that floor. The f16 build keeps the weights at half precision and
    // pays for it; 3e-3 is the same allowance qwen35_parity makes, and
    // the measured gap is printed either way so the bound can tighten.
    let tol = if combs_core::BACKEND.dtype == "f16" {
        3e-3
    } else {
        1e-4
    };
    println!(
        "\n  worst gap against laya's own answers: {worst:.2e} at {at}  (bound {tol:.0e} on the {} build)",
        combs_core::BACKEND.dtype
    );
    assert!(worst <= tol, "real-checkpoint parity: {worst:.2e} at {at}");

    if let Ok(path) = std::env::var("COMBS_LAYA_DUMP") {
        std::fs::write(&path, serde_json::to_string_pretty(&dump).expect("json"))
            .unwrap_or_else(|e| panic!("{path}: {e}"));
        println!("\nwrote {path}");
    }
}

/// Does a question answer the same alone as it does beside another?
///
/// The seat sends every question of a request as ONE batch and `collate`
/// pads every row to the longest, so the noul — always the short row —
/// has been padded to the choice's length in every number ever recorded,
/// and nothing had asked whether that moved a logit. The parity test
/// proves the key-padding mask EXISTS; this asks what the padding costs.
///
/// **Measured 2026-09-24, three states x two questions, on the real
/// checkpoint: f32 3.7e-6, f16 1.03e-2.** The f32 figure is the floor of
/// the comparison, so the mask and the padding are arithmetically right:
/// a padded position contributes nothing. What moves at f16 is the
/// REDUCTION ORDER — the same real terms are summed in a different order
/// when the row is a different length, and half precision keeps a
/// different amount of it. It is not fixable by masking; only fixed-class
/// padding or an f32 accumulator would remove it, and both cost more than
/// the effect is worth today.
///
/// So the test asserts what actually holds, at two strengths:
///
/// - **The top never changes.** That is what a caller reads, and it holds
///   on both builds.
/// - **The logits agree to the f32 floor on f32** (1e-4), which is the
///   evidence that the padding is sound.
/// - **On f16 the gap is a REGRESSION bar, not a guarantee**: 2e-2, which
///   keeps the probability it can move below ~5e-3 (a logit gap d moves a
///   two-way probability by at most d/4), an order of magnitude under
///   0.043 — the smallest top-to-runner-up margin the seed suites have
///   ever produced (`refund-p2`, s138). Past this bar, batch composition
///   could flip a decision, and the seat would need fixed-class padding
///   before any record mixed one-question and two-question calls.
///
/// The consequence for a record: on f16, a cell that asks one question
/// per call is NOT the same configuration as one that asks both, and the
/// record says which it took.
#[test]
#[ignore = "needs the laya checkpoint (COMBS_TEST_LAYA)"]
fn a_question_answers_the_same_alone_and_beside_another() {
    let Some(engine) = seat() else {
        panic!("set COMBS_TEST_LAYA to the checkpoint directory");
    };
    let (states, questions) = suite();
    let f16 = combs_core::BACKEND.dtype == "f16";
    // f32: the floor of the comparison. f16: a regression bar, derived
    // from the smallest margin the seeds have produced (see above).
    let bound = if f16 { 2e-2 } else { 1e-4 };
    let mut worst = 0.0f32;
    let mut at = String::new();
    for state in &states {
        let together = engine.decide(state, &questions).expect("both answer");
        for (i, q) in questions.iter().enumerate() {
            let alone = engine
                .decide(state, std::slice::from_ref(q))
                .expect("one answers");
            let (t, a) = (&together[i], &alone[0]);
            assert_eq!(t.name, a.name);
            assert_eq!(t.distribution.len(), a.distribution.len());
            let top = |d: &[f32]| {
                d.iter()
                    .enumerate()
                    .max_by(|x, y| x.1.total_cmp(y.1))
                    .map(|(i, _)| i)
            };
            assert_eq!(
                top(&t.distribution),
                top(&a.distribution),
                "{state:?} {}: the TOP changed beside another question, which is what a caller \
                 reads: {:?} vs {:?}",
                q.name,
                t.distribution,
                a.distribution
            );
            for (j, (x, y)) in t.logits.iter().zip(&a.logits).enumerate() {
                let d = (x - y).abs();
                if d > worst {
                    worst = d;
                    at = format!("{} logit {j} {x:.6} vs {y:.6} on {state:?}", q.name);
                }
            }
            let d = (t.p_act - a.p_act).abs();
            if d > worst {
                worst = d;
                at = format!(
                    "{}.p_act {:.6} vs {:.6} on {state:?}",
                    q.name, t.p_act, a.p_act
                );
            }
        }
    }
    println!(
        "\n  worst gap alone vs beside another: {worst:.2e} at {at}  (bound {bound:.0e} on the {} \
         build — the f32 figure is the floor, the f16 one a regression bar)",
        combs_core::BACKEND.dtype
    );
    assert!(
        worst <= bound,
        "batch composition moves an answer further than it did when this was measured: \
         {worst:.2e} at {at}"
    );
}

#[test]
#[ignore = "needs the laya checkpoint (COMBS_TEST_LAYA)"]
fn a_question_whose_options_cannot_fit_is_refused_by_name() {
    let Some(engine) = seat() else { return };
    // Enough options that their markers alone overrun max_len.
    let many: Vec<(String, serde_json::Value)> = (0..400)
        .map(|i| {
            (
                format!("o{i}"),
                json!(format!("meaning number {i} of this option")),
            )
        })
        .collect();
    let q = DecisionQuestion {
        name: "too_many".into(),
        instructions: "which".into(),
        criteria: Criteria::Choice(many),
    };
    let err = engine
        .decide("a state", &[q])
        .expect_err("it cannot be answered");
    let why = err.to_string();
    assert!(
        why.contains("too_many"),
        "the refusal does not name the question: {why}"
    );
    assert!(
        why.contains("max_len"),
        "the refusal does not say why: {why}"
    );
}

#[test]
#[ignore = "needs the laya checkpoint (COMBS_TEST_LAYA)"]
fn warming_makes_the_first_request_cost_what_the_second_does() {
    let Some(engine) = seat() else { return };
    let report = engine
        .warm(&combs_runtime::default_warm_shapes(engine.limits()))
        .expect("the seat warms");
    for p in &report.passes {
        println!("  warm {}x{}: {} ms", p.rows, p.len, p.ms);
    }
    println!(
        "  warm-up total: {} ms over {} passes",
        report.total_ms,
        report.passes.len()
    );
    let (states, questions) = suite();
    let time = |s: &str| {
        let t = std::time::Instant::now();
        engine.decide(s, &questions).expect("it answers");
        t.elapsed().as_millis() as u64
    };
    let first = time(states[0]);
    let second = time(states[0]);
    println!("  first request after warm-up: {first} ms; second: {second} ms");
    assert!(
        first <= 2 * second.max(1),
        "the first request after warm-up still paid for something: {first} ms against {second} ms"
    );
}
