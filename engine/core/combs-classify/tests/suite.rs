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
    assert!(
        items.len() >= 20,
        "the seed is thinner than it claims: {} items",
        items.len()
    );

    let c1 = items
        .iter()
        .find(|i| i.id == "unsub-c1")
        .expect("the translation contrast is item 1's pair");
    assert!(c1.is_contrast(), "unsub-c1 must be a contrast: {}", c1.kind);
    assert_eq!(c1.of(), Some("unsub-b1"));
    assert!(
        !c1.expect.fires,
        "a translation request must not be expected to fire an escalation"
    );
    assert!(
        c1.state.to_lowercase().contains("arabic"),
        "the state must be the translation one"
    );

    // Every base has at least one paraphrase and exactly one contrast.
    for base in items.iter().filter(|i| i.kind == "base") {
        let paras = items
            .iter()
            .filter(|i| i.is_paraphrase() && i.of() == Some(&base.id))
            .count();
        let contrasts = items
            .iter()
            .filter(|i| i.is_contrast() && i.of() == Some(&base.id))
            .count();
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
            assert!(
                q.tether.is_some(),
                "question {} carries neither options nor a tether",
                q.name
            );
            continue;
        }
        validate(&q.options, &declared, 64).unwrap_or_else(|e| panic!("question {}: {e}", q.name));
    }
}

#[test]
fn the_contrast_check_catches_a_mention_that_fired() {
    let (h, items) = read();
    let mut observed = answers_that_work(&items);
    // The exact failure this suite exists to catch: the translation
    // request fires the escalation.
    let c1 = observed.iter_mut().find(|o| o.item == "unsub-c1").unwrap();
    c1.answers = vec![ans(
        "intent",
        "escalate_user_planning_unsubscription",
        0.74,
        0.55,
        "act",
    )];

    let v = check_contrast(&h, &items, &observed);
    assert!(!v.holds, "a mention that fired must be caught");
    assert!(
        v.evidence.contains("unsub-c1"),
        "the evidence must name the item: {}",
        v.evidence
    );
}

#[test]
fn the_contrast_check_passes_when_every_mention_stays_shut() {
    let (h, items) = read();
    let observed = answers_that_work(&items);
    let v = check_contrast(&h, &items, &observed);
    assert!(
        v.holds,
        "a rung that keeps every mention shut must pass: {}",
        v.evidence
    );
}

#[test]
fn the_monotonicity_check_catches_a_paraphrase_that_changed_its_answer() {
    let (h, items) = read();
    let mut observed = answers_that_work(&items);
    let p = observed.iter_mut().find(|o| o.item == "unsub-p1").unwrap();
    p.answers = vec![ans("intent", "request_refund", 0.81, 0.70, "act")];

    let v = check_monotonicity(&h, &items, &observed);
    assert!(
        !v.holds,
        "a paraphrase that changed its top option must be caught"
    );
    assert!(
        v.evidence.contains("unsub-p1"),
        "the evidence must name the item: {}",
        v.evidence
    );
}

#[test]
fn the_monotonicity_check_catches_a_paraphrase_that_drifted_too_far() {
    let (h, items) = read();
    let mut observed = answers_that_work(&items);
    // Same top option, but the probability wandered well past delta.
    let p = observed.iter_mut().find(|o| o.item == "unsub-p2").unwrap();
    p.answers = vec![ans(
        "intent",
        "escalate_user_planning_unsubscription",
        0.30,
        0.20,
        "act",
    )];

    let v = check_monotonicity(&h, &items, &observed);
    assert!(
        !v.holds,
        "a drift of 0.56 past a bound of 0.15 must be caught"
    );
    assert!(
        v.evidence.contains("unsub-p2"),
        "the evidence must name the item: {}",
        v.evidence
    );
    assert!(
        v.evidence.contains("0.15"),
        "the evidence must name the bound: {}",
        v.evidence
    );
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
    assert!(
        !v.holds,
        "a rung certain and wrong half the time must be caught"
    );
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
    assert!(
        total <= DIR_CEILING,
        "tests/data totals {total} bytes (ceiling {DIR_CEILING})"
    );
    assert!(
        total > 0,
        "tests/data unexpectedly empty — the ceiling test lost its subject"
    );
}

// --- suite v2: the near misses, typed ------------------------------------

const SEED_V2: &str = include_str!("data/classify-seed-v2.jsonl");

fn read_v2() -> (Header, Vec<Item>) {
    parse(SEED_V2).expect("the v2 fixture parses")
}

/// The same hash the engine stamps a state with, so a state carried from
/// one suite to the next can be shown to be the same bytes.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
fn v2_carries_every_v1_state_unedited() {
    // THE bar for comparing the two runs. v2 re-labels six of v1's items
    // and re-types all six; if it also re-worded one, the two suites
    // would be measuring different inputs while claiming to share them,
    // and no cell could be read across them.
    let (_, v1) = read();
    let (_, v2) = read_v2();
    let carried: Vec<&Item> = v2.iter().filter(|i| i.from_v1.is_some()).collect();
    assert_eq!(
        carried.len(),
        v1.len(),
        "v2 claims to carry {} of v1's {} states",
        carried.len(),
        v1.len()
    );

    for item in &carried {
        let original = v1.iter().find(|o| o.id == item.id).unwrap_or_else(|| {
            panic!("{} claims to come from v1, which has no such item", item.id)
        });
        assert_eq!(
            item.from_v1,
            Some(fnv1a(original.state.as_bytes())),
            "{}: the state was edited on its way from v1",
            item.id
        );
        assert_eq!(item.state, original.state, "{}", item.id);
    }
}

#[test]
fn every_near_miss_says_which_near_miss_it_is_and_what_it_must_never_answer() {
    let (h, items) = read_v2();
    assert_eq!(h.suite, "seed-v2");
    assert_eq!(items.len(), 30);

    // Every question tells the scorer what it is being asked. v1 carried
    // no instructions at all, which for a trained cross-encoder is a
    // prompt it was never fitted on.
    for q in &h.questions {
        let ins = q.instructions.as_deref().unwrap_or("");
        assert!(
            !ins.trim().is_empty(),
            "question {} carries no instructions",
            q.name
        );
    }

    let mut modes: Vec<&str> = items.iter().filter_map(|i| i.near_miss_mode()).collect();
    modes.sort_unstable();
    modes.dedup();
    assert_eq!(
        modes,
        vec!["definition", "negation", "past", "translation"],
        "v2 must name all four near-miss modes and retire the untyped one"
    );

    for item in items.iter().filter(|i| i.is_contrast()) {
        let base_id = item.of().expect("a near miss names its base");
        let base = items
            .iter()
            .find(|b| b.id == base_id)
            .expect("its base exists");
        let never = item
            .never
            .as_ref()
            .unwrap_or_else(|| panic!("{} carries no `never`", item.id));
        // The bar v1 asserted about this state, kept: the action its
        // base asks for is the one it must never take.
        assert_eq!(never.option, base.expect.option, "{}", item.id);
        assert_eq!(never.question, "intent");
        // And it now has somewhere right to land, which v1 never gave it.
        assert_ne!(item.expect.option, never.option, "{}", item.id);
        assert!(
            item.expect.fires,
            "{} should have a right answer, not only a wrong one",
            item.id
        );
        assert_eq!(item.forbidden(), Some(("intent", never.option.as_str())));
    }
}

#[test]
fn the_two_new_options_are_asked_for_in_their_own_right() {
    // An option that only ever appears as the correct answer to a
    // re-labelled near miss is never tested positively, so top-1 on the
    // bases would not cover it.
    let (h, items) = read_v2();
    let intent = h
        .questions
        .iter()
        .find(|q| q.name == "intent")
        .expect("intent");
    for id in ["translate_text", "explain_or_define"] {
        assert!(
            intent.options.iter().any(|o| o.id == id),
            "{id} is not declared"
        );
        let bases = items
            .iter()
            .filter(|i| i.kind == "base" && i.expect.option == id)
            .count();
        assert!(bases >= 1, "{id} has no base item");
    }
    for base in items.iter().filter(|i| i.kind == "base") {
        let paras = items
            .iter()
            .filter(|i| i.is_paraphrase() && i.of() == Some(&base.id))
            .count();
        assert!(paras >= 2, "{} has {paras} paraphrases", base.id);
    }
}

#[test]
fn the_contrast_check_reads_v2s_never_and_catches_a_scorer_that_takes_it() {
    let (h, items) = read_v2();

    // A rung that answers every item correctly.
    let good = answers_that_work(&items);
    let v = check_contrast(&h, &items, &good);
    assert!(v.holds, "{}", v.evidence);

    // The same rung, except the translation request is taken as an
    // unsubscription — the exact failure §126 measured, now expressed
    // against `never` rather than against `fires: false`.
    let mut bad = good.clone();
    let at = items
        .iter()
        .position(|i| i.id == "unsub-c1")
        .expect("unsub-c1");
    bad[at].answers = vec![ans(
        "intent",
        "escalate_user_planning_unsubscription",
        0.98,
        0.94,
        "act",
    )];
    let v = check_contrast(&h, &items, &bad);
    assert!(!v.holds, "the check did not catch a mention that fired");
    assert!(v.evidence.contains("unsub-c1"), "{}", v.evidence);
}

/// One fixture built by hand, so the rules below are shown to catch
/// their own failures without touching a shipped suite.
fn one_item(expect: &str, never: &str, kind: &str) -> String {
    format!(
        concat!(
            r#"{{"kind":"header","suite":"t","delta":0.15,"questions":[{{"name":"intent","kind":"choice","options":["#,
            r#"{{"id":"advise_or_decide","tether":{{"means":"the person is asking us what they should do, not to do it"}}}},"#,
            r#"{{"id":"schedule_call","tether":{{"means":"the person is asking us to book a call"}}}},"#,
            r#"{{"id":"none_of_these","tether":{{"means":"something else entirely"}}}}]}}]}}"#,
            "\n",
            r#"{{"id":"call-d1","kind":"{kind}","state":"shall i book that call or just email them","expect":{expect}{never}}}"#,
        ),
        kind = kind,
        expect = expect,
        never = never,
    )
}

/// A seed suite should have one right answer per item, and both shipped
/// ones do. A genuinely ambiguous state does not: a person asking what
/// THEY should do, against a catalogue with no verb for it, is read
/// defensibly as advice or as none-of-these, and scoring that against
/// one string measures the writer's taste. `any_of` names the
/// alternatives; `option` stays the named answer a table prints.
#[test]
fn an_item_with_two_acceptable_answers_is_hit_by_either_and_never_by_its_never() {
    let fixture = one_item(
        r#"{"question":"intent","option":"advise_or_decide","fires":true,"any_of":["none_of_these"]}"#,
        r#","never":{"question":"intent","option":"schedule_call"}"#,
        "deliberation-of:call-b1",
    );
    let (h, items) = parse(&fixture).expect("the fixture parses");
    let item = &items[0];
    assert!(item.expect.accepts("advise_or_decide"), "the named answer");
    assert!(item.expect.accepts("none_of_these"), "the alternative");
    assert!(!item.expect.accepts("schedule_call"), "what it forbids");
    assert_eq!(
        item.expect.accepted().collect::<Vec<_>>(),
        ["advise_or_decide", "none_of_these"],
        "the named answer first"
    );
    assert_eq!(item.forbidden(), Some(("intent", "schedule_call")));

    // Either accepted answer scores as a hit; the forbidden one does not.
    for top in ["advise_or_decide", "none_of_these"] {
        let obs = vec![Observed {
            item: item.id.clone(),
            answers: vec![ans("intent", top, 0.8, 0.8, "act")],
        }];
        let v = check_calibration(&h, &items, &obs);
        assert!(
            v.evidence.contains("accuracy 1.00"),
            "{top} should be a hit: {}",
            v.evidence
        );
    }
    let obs = vec![Observed {
        item: item.id.clone(),
        answers: vec![ans("intent", "schedule_call", 0.8, 0.8, "act")],
    }];
    let v = check_calibration(&h, &items, &obs);
    assert!(
        v.evidence.contains("accuracy 0.00"),
        "the forbidden option is not a hit: {}",
        v.evidence
    );
}

/// An alternative that names an option nobody declared, an option the
/// item already names, or an option it forbids, is a fixture that
/// measures something other than what it says.
#[test]
fn any_of_naming_an_undeclared_or_contradictory_option_is_refused() {
    let cases = [
        (
            r#"{"question":"intent","option":"advise_or_decide","fires":true,"any_of":["ask_pricing"]}"#,
            "",
            "declares no such option",
        ),
        (
            r#"{"question":"intent","option":"advise_or_decide","fires":true,"any_of":["advise_or_decide"]}"#,
            "",
            "as its option",
        ),
        (
            r#"{"question":"intent","option":"advise_or_decide","fires":true,"any_of":["schedule_call"]}"#,
            r#","never":{"question":"intent","option":"schedule_call"}"#,
            "forbids it in never",
        ),
        (
            r#"{"question":"intent","option":"schedule_call","fires":false,"any_of":["none_of_these"]}"#,
            "",
            "an alternative answer is only an alternative where an answer is due",
        ),
    ];
    for (expect, never, why) in cases {
        let fixture = one_item(expect, never, "deliberation-of:call-b1");
        let err = parse(&fixture).expect_err("this fixture must be refused");
        assert!(err.contains(why), "expected {why:?}, got {err}");
    }
}

/// Translation, definition, past and negation are all "not a request",
/// so the `is_request` gate can shut them. A deliberation IS a request —
/// for a view — and only the option's own tether can hold it. Before it
/// was named, an item of this kind scored as a plain positive: the
/// suite would have reported a clean pass on the very bar it exists to
/// apply, which is the near miss of the near-miss scorer.
#[test]
fn deliberation_is_a_near_miss_mode_beside_the_four() {
    let fixture = one_item(
        r#"{"question":"intent","option":"advise_or_decide","fires":true}"#,
        r#","never":{"question":"intent","option":"schedule_call"}"#,
        "deliberation-of:call-b1",
    );
    let (_, items) = parse(&fixture).expect("the fixture parses");
    let item = &items[0];
    assert!(item.is_contrast(), "a deliberation is a near miss");
    assert_eq!(item.near_miss_mode(), Some("deliberation"));
    assert_eq!(item.of(), Some("call-b1"));
    assert!(!item.is_paraphrase());
}
