//! Every rule the gate makes, named. A rule that cannot be shown to
//! refuse something is not a rule, so each test here carries the case it
//! must catch, and the crate's stubs are written to fail them first.

use combs_classify::sweep::{sweep_thresholds, Costs, Row};
use combs_classify::tether::{validate, Opt, Tether};
use combs_classify::{
    brier, check_distribution, confidence, decide, ece, expected_level, fit_temperature,
    softmax_scaled, Implication,
};

fn near(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// A tether with only a meaning — the common shape.
fn means(id: &str, text: &str) -> Opt {
    Opt {
        id: id.to_string(),
        tether: Tether {
            means: text.to_string(),
            not: Vec::new(),
            requires: Vec::new(),
            label: None,
        },
    }
}

#[test]
fn entropy_confidence_is_one_at_a_point_mass_and_zero_at_uniform() {
    let point = confidence(&[1.0, 0.0, 0.0, 0.0]);
    assert!(near(point, 1.0, 1e-4), "a point mass must be certain: got {point}");
    let flat = confidence(&[0.25, 0.25, 0.25, 0.25]);
    assert!(near(flat, 0.0, 1e-4), "a uniform distribution must carry no confidence: got {flat}");
    let one = confidence(&[1.0]);
    assert!(near(one, 1.0, 1e-4), "a single-option question is certain by construction: got {one}");
}

#[test]
fn a_confidence_below_its_threshold_abstains_naming_the_rule() {
    let d = decide(0.48, 0.60, &[]);
    let reason = d.reason().unwrap_or("").to_string();
    assert_eq!(d.word(), "abstain", "0.48 is under a floor of 0.60 and must not act");
    assert!(
        reason.contains("0.48") && reason.contains("0.60"),
        "the reason must name the confidence and the floor: {reason}"
    );
}

#[test]
fn an_implication_below_tau_abstains_by_name() {
    let imp = Implication { requires: "is_request".to_string(), p: 0.12, tau: 0.70 };
    let d = decide(0.99, 0.0, std::slice::from_ref(&imp));
    assert_eq!(d.word(), "abstain", "a confident answer whose implication fails must not act");
    let reason = d.reason().unwrap_or("").to_string();
    assert!(
        reason.contains("is_request"),
        "the reason must name the question that did not hold: {reason}"
    );
    // And the same answer acts once the implication holds.
    let held = Implication { requires: "is_request".to_string(), p: 0.91, tau: 0.70 };
    assert_eq!(decide(0.99, 0.0, &[held]).word(), "act");
}

#[test]
fn an_implication_with_no_floor_is_not_a_gate_and_says_so() {
    // Measured on the seed suite: with tau at zero every mention fired,
    // because `p >= 0` is true of everything. The rule was wired right
    // and the default made it inert — so whether a gate is a gate has
    // to be visible rather than inferred from a floor.
    let inert = Implication { requires: "is_request".to_string(), p: 0.02, tau: 0.0 };
    assert!(inert.holds(), "a floor of zero cannot refuse: every probability clears it");
    assert!(!inert.gated(), "and an inert implication must say so");
    assert_eq!(decide(0.9, 0.0, std::slice::from_ref(&inert)).word(), "act");

    let real = Implication { requires: "is_request".to_string(), p: 0.02, tau: 0.5 };
    assert!(real.gated(), "a floor above zero is a gate");
    assert_eq!(decide(0.9, 0.0, &[real]).word(), "abstain");
}

#[test]
fn a_distribution_that_does_not_sum_to_one_is_refused() {
    let err = check_distribution(&[0.5, 0.6]).unwrap_err();
    assert!(err.contains("1.1"), "the refusal must name what it summed to: {err}");
    assert!(check_distribution(&[0.25, 0.75]).is_ok(), "a real distribution must pass");
    assert!(check_distribution(&[]).is_err(), "an empty distribution is not a distribution");
    assert!(check_distribution(&[1.5, -0.5]).is_err(), "a negative probability is not one");
}

#[test]
fn expected_level_weights_the_index() {
    let got = expected_level(&[0.2, 0.5, 0.3]);
    assert!(near(got, 1.1, 1e-4), "0·0.2 + 1·0.5 + 2·0.3 is 1.1: got {got}");
}

#[test]
fn temperature_scaling_preserves_the_argmax_and_changes_the_spread() {
    let z = [2.0f32, 1.0, 0.5];
    let sharp = softmax_scaled(&z, 0.5);
    let flat = softmax_scaled(&z, 2.0);
    let top = |p: &Vec<f32>| {
        p.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
    };
    assert_eq!(top(&sharp).0, 0, "scaling must not move the argmax");
    assert_eq!(top(&flat).0, 0, "scaling must not move the argmax");
    assert!(
        top(&sharp).1 > top(&flat).1 + 1e-3,
        "a colder temperature must be sharper: {} against {}",
        top(&sharp).1,
        top(&flat).1
    );
    let sum: f32 = sharp.iter().sum();
    assert!(near(sum, 1.0, 1e-5), "a scaled distribution still sums to one: got {sum}");
}

#[test]
fn ece_is_zero_on_a_calibrated_table_and_not_on_an_overconfident_one() {
    // Ten items at 0.9 confidence, nine of them right: exactly calibrated.
    let mut honest: Vec<(f32, bool)> = (0..9).map(|_| (0.9f32, true)).collect();
    honest.push((0.9, false));
    let e_honest = ece(&honest, 10);
    assert!(e_honest <= 0.02, "a calibrated table must score near zero: got {e_honest}");

    // Ten items at 0.9 confidence, half of them right.
    let mut bragging: Vec<(f32, bool)> = (0..5).map(|_| (0.9f32, true)).collect();
    bragging.extend((0..5).map(|_| (0.9f32, false)));
    let e_brag = ece(&bragging, 10);
    assert!(e_brag > 0.3, "an overconfident table must be caught: got {e_brag}");
    assert!(brier(&bragging) > brier(&honest), "Brier must separate them too");
}

#[test]
fn fit_temperature_recovers_a_known_scale() {
    // Labels drawn from softmax(z / 2.0): the fit should find ~2.0.
    // A small deterministic generator, so the test never moves.
    let mut seed: u64 = 0x5eed_1234;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32) / ((1u64 << 31) as f32)
    };
    let mut logits = Vec::new();
    let mut labels = Vec::new();
    for _ in 0..4000 {
        let z: Vec<f32> = (0..3).map(|_| (next() - 0.5) * 6.0).collect();
        let p = {
            let m = z.iter().cloned().fold(f32::MIN, f32::max);
            let e: Vec<f32> = z.iter().map(|v| ((v - m) / 2.0).exp()).collect();
            let s: f32 = e.iter().sum();
            e.into_iter().map(|v| v / s).collect::<Vec<f32>>()
        };
        let mut u = next();
        let mut pick = p.len() - 1;
        for (i, &pi) in p.iter().enumerate() {
            if u < pi {
                pick = i;
                break;
            }
            u -= pi;
        }
        logits.push(z);
        labels.push(pick);
    }
    let t = fit_temperature(&logits, &labels);
    assert!(near(t, 2.0, 0.25), "the fit must recover the scale the labels were drawn at: got {t}");
}

#[test]
fn a_scorer_never_reads_an_option_identifier() {
    // THE tether's reason for existing. The identifier carries the topic
    // ("unsubscription"); a scorer that reads it matches a translation
    // request by word overlap. Only the meaning may be scored.
    let opt = Opt {
        id: "escalate_user_planning_unsubscription".to_string(),
        tether: Tether {
            label: None,
            means: "the person is asking us to end their subscription now".to_string(),
            not: vec!["quoting or translating a sentence about unsubscribing".to_string()],
            requires: vec!["is_request".to_string()],
        },
    };
    let text = opt.scored_text();
    assert!(
        !text.contains("escalate_user_planning_unsubscription"),
        "the identifier reached the scorer: {text}"
    );
    assert!(
        !text.contains('_'),
        "an identifier's shape reached the scorer: {text}"
    );
    assert!(
        text.contains("end their subscription"),
        "the meaning must be what is scored: {text}"
    );
    assert!(
        text.contains("translating"),
        "the near-miss anchors must be scored beside the meaning: {text}"
    );
}

#[test]
fn an_option_without_a_tether_is_refused_by_name() {
    let opts = vec![means("refund", "the person wants money back"), means("bare", "  ")];
    let err = validate(&opts, &[], 64).unwrap_err();
    assert!(err.contains("bare"), "the refusal must name the option: {err}");
    assert!(
        err.contains("meaning") || err.contains("tether"),
        "the refusal must say what is missing: {err}"
    );
}

#[test]
fn two_options_that_mean_the_same_thing_are_refused_by_name() {
    let opts = vec![
        means("cancel", "the person wants to end their subscription"),
        means("stop_billing", "the person wants to end their subscription"),
    ];
    let err = validate(&opts, &[], 64).unwrap_err();
    assert!(
        err.contains("cancel") && err.contains("stop_billing"),
        "the refusal must name both options: {err}"
    );
}

#[test]
fn an_implication_naming_an_undeclared_question_is_refused_by_name() {
    let opts = vec![Opt {
        id: "escalate".to_string(),
        tether: Tether {
            label: None,
            means: "the person is asking us to act".to_string(),
            not: Vec::new(),
            requires: vec!["is_request".to_string()],
        },
    }];
    let err = validate(&opts, &["something_else".to_string()], 64).unwrap_err();
    assert!(err.contains("is_request"), "the refusal must name the question: {err}");
    assert!(
        validate(&opts, &["is_request".to_string()], 64).is_ok(),
        "a declared implication must pass"
    );
}

#[test]
fn seventy_seven_options_refuse_by_name() {
    // The number is the one the field's own high-cardinality case uses.
    let opts: Vec<Opt> =
        (0..77).map(|i| means(&format!("label{i}"), &format!("meaning number {i}"))).collect();
    let err = validate(&opts, &[], 64).unwrap_err();
    assert!(err.contains("77"), "the refusal must name how many arrived: {err}");
    assert!(err.contains("64"), "the refusal must name the maximum: {err}");
}

#[test]
/// The label slot exists so a scorer MAY read it, which is exactly why
/// it cannot hold a routing key. `label == id` caught only the laziest
/// spelling: `escalate_unsubscription` on
/// `escalate_user_planning_unsubscription` is the same topic in a
/// shorter coat, and word overlap does not care about the difference.
fn a_label_cannot_be_an_identifier_in_disguise() {
    let opt = |id: &str, label: Option<&str>| Opt {
        id: id.to_string(),
        tether: Tether {
            means: format!("what {id} means, at some length"),
            not: Vec::new(),
            requires: Vec::new(),
            label: label.map(str::to_string),
        },
    };
    let refused = |opts: &[Opt]| -> String {
        combs_classify::tether::validate(opts, &[], 64).expect_err("refused")
    };

    // The shape of an identifier, however it is spelled.
    for bad in [
        "request_refund",
        "refund_v2",
        "refund2",
        "yes: refund",
        "a much longer label than any name",
        "cancel the subscription entirely now",
    ] {
        let why = refused(&[opt("some_option", Some(bad))]);
        assert!(why.contains("some_option"), "{bad}: {why}");
        assert!(why.contains(&format!("{bad:?}")), "{bad}: {why}");
    }

    // A sibling's identifier, in a separator that passes the shape rule
    // — these reach the sibling check and must name the option they are
    // wearing. (`REQUEST_REFUND` never gets that far: the underscore
    // rule above refuses it first, which is why it is not in this list.)
    for disguise in ["request refund", "Request-Refund"] {
        let opts = [opt("request_refund", None), opt("other", Some(disguise))];
        let why = combs_classify::tether::validate(&opts, &[], 64).expect_err("a sibling's id");
        assert!(why.contains("request_refund"), "{disguise}: {why}");
    }
    // Refused either way, whichever rule gets there first.
    let shouty = [
        opt("request_refund", None),
        opt("other", Some("REQUEST_REFUND")),
    ];
    assert!(combs_classify::tether::validate(&shouty, &[], 64).is_err());

    // Two options a scorer could not tell apart are one option.
    let twins = [opt("a", Some("refund")), opt("b", Some("Refund"))];
    assert!(combs_classify::tether::validate(&twins, &[], 64).is_err());

    // What laya's own presets look like still passes — including the
    // crate's own example, a `refund` label on a `request_refund` id,
    // which a token-of-the-id rule would have refused.
    for good in ["refund", "billing", "money back", "not-billing"] {
        let opts = [opt("request_refund", Some(good)), opt("other_thing", None)];
        combs_classify::tether::validate(&opts, &[], 64)
            .unwrap_or_else(|e| panic!("{good} is a meaning, not an identifier: {e}"));
    }
}

#[test]
fn the_ev_sweep_finds_the_threshold_that_minimises_it() {
    // A table where the cheap system is right exactly when its
    // confidence is at least 0.6, and the dear one is always right.
    let rows: Vec<Row> = (0..20)
        .map(|i| {
            let c = i as f32 / 20.0;
            Row { sys1_confidence: c, sys1_correct: c >= 0.6, sys2_correct: true }
        })
        .collect();
    let costs = Costs {
        c_sys1: 1.0,
        c_sys2: 10.0,
        l_sys1_ms: 10.0,
        l_sys2_ms: 100.0,
        w_lat_per_ms: 0.0,
        w_err: 1000.0,
    };
    let (t_star, curve) = sweep_thresholds(&rows, &costs);
    assert!(!curve.is_empty(), "the sweep must return its curve");
    assert!(
        near(t_star, 0.60, 0.051),
        "with errors dear, the line belongs where the cheap system stops being right: got {t_star}"
    );

    // With errors free, never fall back.
    let free = Costs { w_err: 0.0, ..costs };
    let (t_free, _) = sweep_thresholds(&rows, &free);
    assert!(near(t_free, 0.0, 1e-6), "with errors free the cheap system always answers: got {t_free}");
}

/// `bars.rs` says the gateway keeps the same question cap "by hand, in
/// two trees: a test pins each so a drift is a failure". Neither test
/// existed: the gateway's own tests pass 2 by environment and assert a
/// literal 16, and nothing on either side ever read the other's number.
/// This is our half. The gateway is built from its own image, so it
/// cannot read this tree; its half is a cited literal.
///
/// When the gateway tree is not beside us the test SAYS SO and passes —
/// a silent pass is how the first version of this rule was lost.
#[test]
fn the_gateways_question_cap_matches_this_crates_when_the_gateway_tree_is_beside_us() {
    let conf = match std::env::var("COMBS_INFERENCE_DIR") {
        Ok(dir) => std::path::PathBuf::from(dir).join("config/serve.conf"),
        Err(_) => std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../combs-inference/config/serve.conf"),
    };
    let Ok(text) = std::fs::read_to_string(&conf) else {
        eprintln!(
            "cross-tree pin unread: no gateway config at {} (set COMBS_INFERENCE_DIR to pin it)",
            conf.display()
        );
        return;
    };
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("MAX_QUESTIONS="))
        .unwrap_or_else(|| {
            panic!(
                "{} carries no MAX_QUESTIONS line: the gateway stopped capping what this crate caps",
                conf.display()
            )
        });
    let theirs: usize = line["MAX_QUESTIONS=".len()..]
        .split('#')
        .next()
        .unwrap_or_default()
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("{} is not a number in {}: {e}", line, conf.display()));
    assert_eq!(
        theirs,
        combs_classify::bars::MAX_QUESTIONS,
        "the gateway caps questions at {theirs} and this crate at {} ({}): a request \
         the gateway forwards would be refused by the seat, or one the seat would answer is \
         refused before it arrives",
        combs_classify::bars::MAX_QUESTIONS,
        conf.display()
    );
    eprintln!("cross-tree pin read: {} says {theirs}", conf.display());
}
