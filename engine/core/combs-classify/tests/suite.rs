//! The suite's checks, each shown to CATCH its own failure before it is
//! believed about a real rung, and each with a positive control so a
//! check that refuses everything cannot pass for a strict one.

use combs_classify::suite::{
    check_calibration, check_contrast, check_monotonicity, check_schema, parse, Answer, Header,
    Item, Observed,
};
use combs_classify::tether::validate;

const SEED: &str = include_str!("data/classify-seed.jsonl");

fn read() -> (Header, Vec<Item>) {
    parse(SEED).expect("the seed fixture parses")
}

/// One answer, with everything the checks read.
fn ans(question: &str, top: &str, p_top: f32, confidence: f32, decision: &str) -> Answer {
    Answer {
        question: question.to_string(),
        top: top.to_string(),
        p_top,
        confidence,
        decision: decision.to_string(),
    }
}

/// The answers a rung that works would give: every item's expected
/// option on top, firing exactly when the fixture says it should.
fn answers_that_work(items: &[Item]) -> Vec<Observed> {
    items
        .iter()
        .map(|i| Observed {
            item: i.id.clone(),
            answers: vec![if i.expect.fires {
                ans("intent", &i.expect.option, 0.86, 0.72, "act")
            } else {
                ans("intent", "none_of_these", 0.80, 0.66, "act")
            }],
        })
        .collect()
}

#[test]
fn the_seed_fixture_parses_and_carries_the_translation_contrast() {
    let (h, items) = read();
    assert_eq!(h.suite, "seed-v1");
    assert!(h.delta > 0.0, "the fixture must name its drift bound");
    assert!(items.len() >= 20, "the seed is thinner than it claims: {} items", items.len());

    let c1 = items.iter().find(|i| i.id == "unsub-c1").expect("the translation contrast is item 1's pair");
    assert!(c1.is_contrast(), "unsub-c1 must be a contrast: {}", c1.kind);
    assert_eq!(c1.of(), Some("unsub-b1"));
    assert!(!c1.expect.fires, "a translation request must not be expected to fire an escalation");
    assert!(c1.state.to_lowercase().contains("arabic"), "the state must be the translation one");

    // Every base has at least one paraphrase and exactly one contrast.
    for base in items.iter().filter(|i| i.kind == "base") {
        let paras = items.iter().filter(|i| i.is_paraphrase() && i.of() == Some(&base.id)).count();
        let contrasts = items.iter().filter(|i| i.is_contrast() && i.of() == Some(&base.id)).count();
        assert!(paras >= 2, "{} has {paras} paraphrases", base.id);
        assert_eq!(contrasts, 1, "{} has {contrasts} contrasts", base.id);
    }
}

#[test]
fn every_option_in_the_seed_carries_a_tether() {
    // A GUARD: the fixture is written correct, so this passes on its
    // first run. It earns its place by failing the day somebody adds an
    // option with a label and no meaning.
    let (h, _) = read();
    let declared: Vec<String> = h.questions.iter().map(|q| q.name.clone()).collect();
    for q in &h.questions {
        if q.options.is_empty() {
            assert!(q.tether.is_some(), "question {} carries neither options nor a tether", q.name);
            continue;
        }
        validate(&q.options, &declared, 64)
            .unwrap_or_else(|e| panic!("question {}: {e}", q.name));
    }
}

#[test]
fn the_contrast_check_catches_a_mention_that_fired() {
    let (h, items) = read();
    let mut observed = answers_that_work(&items);
    // The exact failure this suite exists to catch: the translation
    // request fires the escalation.
    let c1 = observed.iter_mut().find(|o| o.item == "unsub-c1").unwrap();
    c1.answers = vec![ans("intent", "escalate_user_planning_unsubscription", 0.74, 0.55, "act")];

    let v = check_contrast(&h, &items, &observed);
    assert!(!v.holds, "a mention that fired must be caught");
    assert!(v.evidence.contains("unsub-c1"), "the evidence must name the item: {}", v.evidence);
}

#[test]
fn the_contrast_check_passes_when_every_mention_stays_shut() {
    let (h, items) = read();
    let observed = answers_that_work(&items);
    let v = check_contrast(&h, &items, &observed);
    assert!(v.holds, "a rung that keeps every mention shut must pass: {}", v.evidence);
}

#[test]
fn the_monotonicity_check_catches_a_paraphrase_that_changed_its_answer() {
    let (h, items) = read();
    let mut observed = answers_that_work(&items);
    let p = observed.iter_mut().find(|o| o.item == "unsub-p1").unwrap();
    p.answers = vec![ans("intent", "request_refund", 0.81, 0.70, "act")];

    let v = check_monotonicity(&h, &items, &observed);
    assert!(!v.holds, "a paraphrase that changed its top option must be caught");
    assert!(v.evidence.contains("unsub-p1"), "the evidence must name the item: {}", v.evidence);
}

#[test]
fn the_monotonicity_check_catches_a_paraphrase_that_drifted_too_far() {
    let (h, items) = read();
    let mut observed = answers_that_work(&items);
    // Same top option, but the probability wandered well past delta.
    let p = observed.iter_mut().find(|o| o.item == "unsub-p2").unwrap();
    p.answers = vec![ans("intent", "escalate_user_planning_unsubscription", 0.30, 0.20, "act")];

    let v = check_monotonicity(&h, &items, &observed);
    assert!(!v.holds, "a drift of 0.56 past a bound of 0.15 must be caught");
    assert!(
        v.evidence.contains("unsub-p2"),
        "the evidence must name the item: {}",
        v.evidence
    );
    assert!(v.evidence.contains("0.15"), "the evidence must name the bound: {}", v.evidence);
}

#[test]
fn the_schema_check_catches_an_answer_naming_an_option_nobody_declared() {
    let (h, items) = read();
    let mut observed = answers_that_work(&items);
    observed[0].answers = vec![ans("intent", "fly_the_user_to_the_moon", 0.99, 0.99, "act")];

    let v = check_schema(&h, &items, &observed);
    assert!(!v.holds, "an undeclared option must be caught");
    assert!(
        v.evidence.contains("fly_the_user_to_the_moon"),
        "the evidence must name it: {}",
        v.evidence
    );
}

#[test]
fn the_calibration_check_reports_its_numbers_and_refuses_an_overconfident_rung() {
    let (h, items) = read();
    // Every answer certain, half of them wrong: the shape a rung ships
    // in before a temperature is fitted.
    let observed: Vec<Observed> = items
        .iter()
        .enumerate()
        .map(|(n, i)| Observed {
            item: i.id.clone(),
            answers: vec![if n % 2 == 0 {
                ans("intent", &i.expect.option, 0.99, 0.99, "act")
            } else {
                ans("intent", "none_of_these", 0.99, 0.99, "act")
            }],
        })
        .collect();

    let v = check_calibration(&h, &items, &observed);
    assert!(!v.holds, "a rung certain and wrong half the time must be caught");
    assert!(
        v.evidence.to_lowercase().contains("ece"),
        "the evidence must report ECE: {}",
        v.evidence
    );
    assert!(
        v.evidence.to_lowercase().contains("brier"),
        "ECE is never reported alone: {}",
        v.evidence
    );
}

#[test]
fn fixtures_stay_under_the_ceiling() {
    // The same ceilings the rest of the estate keeps: a fixture small
    // enough to review and to clone.
    const FILE_CEILING: u64 = 256 * 1024;
    const DIR_CEILING: u64 = 2 * 1024 * 1024;
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut total = 0u64;
    for entry in std::fs::read_dir(&dir).expect("tests/data exists") {
        let entry = entry.unwrap();
        let meta = entry.metadata().unwrap();
        if !meta.is_file() {
            continue;
        }
        assert!(
            meta.len() <= FILE_CEILING,
            "{} is {} bytes (ceiling {FILE_CEILING})",
            entry.path().display(),
            meta.len()
        );
        total += meta.len();
    }
    assert!(total <= DIR_CEILING, "tests/data totals {total} bytes (ceiling {DIR_CEILING})");
    assert!(total > 0, "tests/data unexpectedly empty — the ceiling test lost its subject");
}
