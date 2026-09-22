//! What the option text is worth, on the seat that reads it.
//!
//! The cosine road scores `scored_text()` — the meaning plus the
//! anchors, with the identifier excluded by construction and two tests
//! pinning it (`combs-classify/tests/gate.rs`). The marker road arrived
//! rendering laya's `label: criterion` with the option's IDENTIFIER in
//! the label slot and the anchors nowhere. So §137 and §138 compared two
//! scorers that were never reading the same text, and "the trained
//! scorer classifies by topic" was measured on a scorer we handed a
//! topic to.
//!
//! This runs one seat over both seed suites under all three renderings,
//! so the identifier's worth and the anchors' worth are separated from
//! each other and from the model. Ignored by default: it wants the
//! 842 MB checkpoint.
//!
//!     COMBS_TEST_LAYA=~/.cache/combs/models/laya-typed-decisions \
//!       cargo test --release -p combs-runtime --test laya_renderings \
//!       -- --ignored --nocapture
//!
//! It is a PROBE, not a sitting: one pass per item, no interleaving, no
//! gates. It says which way is worth measuring properly, not what the
//! number is.

use std::collections::BTreeMap;

use combs_classify::system_one::{criteria_for, Rendering};
use combs_classify::{Kind, Opt, Tether};
use combs_runtime::{DecisionEngine, DecisionQuestion};
use serde_json::Value;

const NEAR_MISS: [&str; 5] = [
    "contrast-of:",
    "translation-of:",
    "definition-of:",
    "past-of:",
    "negation-of:",
];

struct Item {
    id: String,
    kind: String,
    state: String,
    question: String,
    option: String,
    fires: bool,
    never: Option<(String, String)>,
}

/// The suite's questions, as the wire declares them, and its items.
fn suite(path: &str) -> (Vec<(String, Kind, String, Vec<Opt>)>, Vec<Item>) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Value = serde_json::from_str(lines.next().expect("a header")).expect("json");
    let mut questions = Vec::new();
    for q in header["questions"].as_array().expect("questions") {
        let name = q["name"].as_str().expect("name").to_string();
        let instructions = q["instructions"].as_str().unwrap_or("").to_string();
        let opt_of = |v: &Value| Opt {
            id: v["id"].as_str().expect("id").to_string(),
            tether: Tether {
                means: v["tether"]["means"].as_str().unwrap_or("").to_string(),
                not: v["tether"]["not"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|s| s.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default(),
                requires: Vec::new(),
                // The suites name no label, so every non-default
                // rendering falls to an ordinal — the weakest form of
                // the change, and the honest one to measure first.
                label: None,
            },
        };
        match q["kind"].as_str().expect("kind") {
            "noul" => {
                let t = &q["tether"];
                let contrary = t["not"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join("; ")
                    })
                    .unwrap_or_default();
                questions.push((
                    name,
                    Kind::Noul,
                    instructions,
                    vec![
                        opt_of(&serde_json::json!({"id":"true","tether":{"means":t["means"]}})),
                        opt_of(&serde_json::json!({"id":"false","tether":{"means":contrary}})),
                    ],
                ));
            }
            _ => {
                let opts = q["options"]
                    .as_array()
                    .expect("options")
                    .iter()
                    .map(opt_of)
                    .collect();
                questions.push((name, Kind::Choice, instructions, opts));
            }
        }
    }
    let items = lines
        .map(|l| {
            let v: Value = serde_json::from_str(l).expect("json");
            Item {
                id: v["id"].as_str().expect("id").to_string(),
                kind: v["kind"].as_str().expect("kind").to_string(),
                state: v["state"].as_str().expect("state").to_string(),
                question: v["expect"]["question"].as_str().expect("q").to_string(),
                option: v["expect"]["option"].as_str().expect("o").to_string(),
                fires: v["expect"]["fires"].as_bool().unwrap_or(true),
                never: v["never"].as_object().map(|n| {
                    (
                        n["question"].as_str().unwrap_or("").to_string(),
                        n["option"].as_str().unwrap_or("").to_string(),
                    )
                }),
            }
        })
        .collect();
    (questions, items)
}

/// The (question, option) an item must NEVER answer with: v2 says it
/// outright, v1 says it by expecting its own option not to fire.
fn forbidden(it: &Item) -> Option<(String, String)> {
    if let Some(n) = &it.never {
        return Some(n.clone());
    }
    let near = NEAR_MISS.iter().any(|p| it.kind.starts_with(p));
    (near && !it.fires).then(|| (it.question.clone(), it.option.clone()))
}

#[test]
#[ignore = "needs the laya checkpoint (COMBS_TEST_LAYA)"]
fn what_the_option_text_is_worth() {
    let Ok(dir) = std::env::var("COMBS_TEST_LAYA") else {
        panic!("set COMBS_TEST_LAYA to the checkpoint directory");
    };
    let engine = DecisionEngine::load(&dir).expect("the checkpoint seats");
    engine
        .warm(&combs_runtime::default_warm_shapes(engine.limits()))
        .expect("the seat warms");

    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/../combs-classify/tests/data");
    println!(
        "\nseat: {} on the {} build\n",
        engine.model_id(),
        combs_core::BACKEND.dtype
    );

    for (suite_name, file) in [
        ("v1", "classify-seed.jsonl"),
        ("v2", "classify-seed-v2.jsonl"),
    ] {
        let (questions, items) = suite(&format!("{data}/{file}"));
        println!("== seed {suite_name}: {} items ==", items.len());
        println!(
            "{:<14} {:>8} {:>10} {:>12} {:>9} {:>9}",
            "rendering", "top-1", "held", "whole suite", "p50 ms", "tokens"
        );

        for rendering in [
            Rendering::IdMeans,
            Rendering::LabelMeans,
            Rendering::LabelAnchors,
        ] {
            let asked: Vec<DecisionQuestion> = questions
                .iter()
                .map(|(name, kind, instructions, opts)| DecisionQuestion {
                    name: name.clone(),
                    instructions: instructions.clone(),
                    criteria: criteria_for(*kind, opts, rendering),
                })
                .collect();
            let ids: BTreeMap<&str, Vec<String>> = questions
                .iter()
                .map(|(n, _, _, o)| (n.as_str(), o.iter().map(|o| o.id.clone()).collect()))
                .collect();

            let (mut bases, mut base_hits) = (0usize, 0usize);
            let (mut forbid, mut held) = (0usize, 0usize);
            let (mut scored, mut correct) = (0usize, 0usize);
            let mut fired: Vec<String> = Vec::new();
            let mut ms: Vec<u128> = Vec::new();
            let mut tokens = 0usize;

            for it in &items {
                let t = std::time::Instant::now();
                let decided = engine
                    .decide_with_shape(&it.state, &asked)
                    .unwrap_or_else(|e| panic!("{}: {e}", it.id));
                ms.push(t.elapsed().as_millis());
                tokens = decided.batch.rows * decided.batch.len;
                let top = |q: &str| -> &str {
                    let a = decided
                        .answers
                        .iter()
                        .find(|a| a.name == q)
                        .unwrap_or_else(|| panic!("no answer {q}"));
                    let best = a
                        .distribution
                        .iter()
                        .enumerate()
                        .fold(
                            (0usize, f32::MIN),
                            |b, (i, &v)| if v > b.1 { (i, v) } else { b },
                        )
                        .0;
                    ids[q][best].as_str()
                };

                if it.kind == "base" {
                    bases += 1;
                    base_hits += usize::from(top(&it.question) == it.option);
                }
                if let Some((q, opt)) = forbidden(it) {
                    forbid += 1;
                    if top(&q) == opt {
                        fired.push(it.id.clone());
                    } else {
                        held += 1;
                    }
                }
                scored += 1;
                let hit = top(&it.question) == it.option;
                correct += usize::from(if it.fires { hit } else { !hit });
            }
            ms.sort_unstable();
            println!(
                "{:<14} {:>4}/{:<3} {:>6}/{:<3} {:>8}/{:<3} {:>9} {:>9}",
                rendering.as_str(),
                base_hits,
                bases,
                held,
                forbid,
                correct,
                scored,
                ms[ms.len() / 2],
                tokens
            );
            if !fired.is_empty() {
                println!("{:<14} fired: {}", "", fired.join(" "));
            }
        }
        println!();
    }
}
