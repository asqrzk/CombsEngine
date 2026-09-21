//! The marker sequence, graded against laya's own.
//!
//! The reds beside `build_sequence` assert index arithmetic this port
//! derived. That is worth little on its own: a port and its author can
//! agree with each other and both be wrong about the thing being ported.
//! So the fixture here is laya 0.3.5's `build_sequence` run from ITS
//! source over a stub tokenizer, and this test asserts the two produce
//! the same ids and the same marker positions for nine questions that
//! between them take every truncation road in the function — the option
//! cap, the option floor, the instruction floor, the state's right cut,
//! and a marker pushed off the end.
//!
//! Regenerating it needs the laya clone; the fixture says which commit.

use combs_classify::sequence::*;
use combs_classify::Kind;

/// The same stub the fixture was generated with: one id per whitespace
/// word, and the literal `[MASK]` mapping to a real mask id so an
/// unstripped one would show up as a marker in the wrong place.
struct Words;

impl Encode for Words {
    fn ids(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|w| {
                if w == "[MASK]" {
                    50284
                } else {
                    1000 + w.len() as u32
                }
            })
            .collect()
    }
}

/// One question the fixture holds an answer for.
struct Case {
    name: &'static str,
    kind: Kind,
    instructions: String,
    options: Vec<String>,
    state: &'static str,
    max_len: usize,
    head_max_len: usize,
}

fn numbers(fixture: &serde_json::Value, case: &str, field: &str) -> Vec<i64> {
    fixture["cases"][case][field]
        .as_array()
        .unwrap_or_else(|| panic!("the fixture has no {case}.{field}"))
        .iter()
        .map(|v| v.as_i64().expect("a number"))
        .collect()
}

#[test]
fn every_truncation_road_lands_where_laya_lands() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/laya_sequences.json"
    ))
    .expect("the parity fixture");
    let fixture: serde_json::Value = serde_json::from_str(&text).expect("it is json");

    let specials = Specials {
        cls: 50281,
        sep: 50282,
        mask: 50284,
        pad: 50283,
        mask_text: "[MASK]".to_string(),
    };
    let long: String = (0..100).map(|i| format!("w{i} ")).collect();
    let head: String = (0..60).map(|i| format!("w{i} ")).collect();
    let o = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();

    let case = |name, kind, instructions: String, options, state, max_len, head_max_len| Case {
        name,
        kind,
        instructions,
        options,
        state,
        max_len,
        head_max_len,
    };
    let cases = vec![
        // The ordinary road: three short options, nothing cut.
        case(
            "markers",
            Kind::Choice,
            "which one".into(),
            o(&["a refund", "an exchange", "nothing"]),
            "the parcel arrived broken",
            512,
            192,
        ),
        // One option of a hundred words: the 48-token cap, marker excluded.
        case(
            "cap",
            Kind::Choice,
            "q".into(),
            vec![long.clone()],
            "s",
            512,
            192,
        ),
        // `[MASK]` in all three caller texts.
        case(
            "mask",
            Kind::Choice,
            "is [MASK] the answer".into(),
            o(&["yes [MASK] indeed", "no"]),
            "the user wrote [MASK] in their message",
            512,
            192,
        ),
        // A state longer than the room left: cut from the right.
        case(
            "right",
            Kind::Choice,
            "q".into(),
            o(&["one"]),
            "a bb ccc dddd eeeee ffffff",
            12,
            192,
        ),
        // Options that eat the head: they are cut to the floor and the
        // instructions keep their eight.
        case(
            "headfloor",
            Kind::Choice,
            head.clone(),
            o(&["one two three four five", "six seven eight", "nine ten"]),
            "s",
            512,
            16,
        ),
        // A long instruction with room to spare: cut to the budget.
        case(
            "headroom",
            Kind::Choice,
            head.clone(),
            o(&["a", "b"]),
            "s",
            512,
            192,
        ),
        // A sequence so short that two markers fall off the end.
        case(
            "dropped",
            Kind::Choice,
            "q".into(),
            o(&["one two three", "four five six", "seven"]),
            "s",
            8,
            192,
        ),
        case("short", Kind::Choice, "q".into(), o(&["a"]), "s", 512, 192),
        // A noul, whose type name changes the head text.
        case(
            "noul",
            Kind::Noul,
            "is it so".into(),
            o(&["false: no", "true: yes"]),
            "a much longer state than the other one had",
            512,
            192,
        ),
    ];

    for c in &cases {
        let name = c.name;
        let s = build_sequence(
            &Words,
            &specials,
            c.kind,
            &c.instructions,
            &c.options,
            c.state,
            Limits {
                max_len: c.max_len,
                head_max_len: c.head_max_len,
            },
        );
        let want_ids = numbers(&fixture, name, "ids");
        let want_markers = numbers(&fixture, name, "markers");
        assert_eq!(
            s.ids.iter().map(|&i| i as i64).collect::<Vec<_>>(),
            want_ids,
            "{name}: ids"
        );
        assert_eq!(
            s.markers.iter().map(|&m| m as i64).collect::<Vec<_>>(),
            want_markers,
            "{name}: markers"
        );
    }

    // The fixture is only worth what it covers: every case must be one
    // this test drives, so a case added there is a case run here.
    let covered = cases.len();
    let in_fixture = fixture["cases"].as_object().expect("cases").len();
    assert_eq!(
        covered, in_fixture,
        "the fixture holds {in_fixture} cases and this test drives {covered}"
    );
}
