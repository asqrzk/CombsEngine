//! The marker sequence a decision model reads, and the batch of them.
//!
//! One question becomes one sequence:
//!
//! ```text
//! [CLS] <type> question: <instructions> [SEP] ([MASK] <option>)* [SEP] <state> [SEP]
//! ```
//!
//! and the `[MASK]` before each option is the MARKER — the position
//! whose hidden state the scorer reads as that option's logit. So the
//! marker positions are the output of this file as much as the ids are:
//! a sequence whose markers are off by one scores the wrong tokens and
//! still answers.
//!
//! This is laya 0.3.5's `build_sequence` and `collate_items`, ported.
//! Nothing here holds a tokenizer — [`Encode`] is the seam, implemented
//! where one is owned — so the whole thing is text arithmetic and its
//! reds cost a second rather than a build.
//!
//! **Everything here truncates; nothing here refuses.** That is laya's
//! own shape and it matters, because the two failure modes read very
//! differently to a caller. An over-long instruction, an over-long
//! option and an over-long state are each cut to fit and the question is
//! still answered — on less text, which the record must say. The ONE
//! thing that cannot be recovered from is a marker pushed past
//! `max_len`: its option has no position to score, so [`Sequence`]
//! counts what it dropped and the caller refuses by the question's name.
//!
//! The three defences that are not obvious, each of which a plainer port
//! would have got wrong:
//!
//! * The mask token's TEXT is stripped from the instructions, from every
//!   option and from the state before any of them is tokenized.
//!   Otherwise a state containing the literal `[MASK]` injects a marker,
//!   and the option it scores is whatever happened to follow it.
//! * An option is tokenized with a LEADING SPACE. The tokenizer is
//!   byte-level: `"refund"` and `" refund"` are different tokens, and
//!   the model was fitted on the spaced one.
//! * The state is cut from the RIGHT by default (`Cut::Right`); the seat's
//!   `COMBS_CLASSIFY_STATE_CUT=left` keeps the end instead, laya's own
//!   `truncate_left`. That function exists in laya and
//!   defaults false, and its caller never passes it — so the beginning
//!   of a long state survives and the end is dropped.

use crate::system_one::{qtype, qtype_name};
use serde::{Deserialize, Serialize};

use crate::Kind;

/// Tokens one option may spend on itself. The marker is not counted
/// against it — it is spliced before the cap is applied.
pub const OPTION_TOKEN_CAP: usize = 48;

/// Below this many tokens left for the instructions, the options are cut
/// instead: an instruction shorter than this says nothing.
pub const MIN_OPTION_BUDGET: usize = 16;

/// The instructions always keep at least this much, even when the
/// options have eaten the whole head.
pub const MIN_HEAD_TOKENS: usize = 8;

/// An option always keeps at least this much, marker included.
pub const MIN_OPTION_TOKENS: usize = 4;

/// Anything that can turn text into token ids.
///
/// The seam exists so this crate links no tokenizer: it is implemented
/// where one is already owned. A stub implementation is what the reds
/// below run on, which is also why they cost no model.
pub trait Encode {
    /// Ids for `text`, with NO special tokens added — this file splices
    /// every special by number itself.
    fn ids(&self, text: &str) -> Vec<u32>;
}

/// The special token ids a marker sequence splices.
///
/// By NUMBER, never by writing the text: `[MASK]` is `lstrip: true` in
/// this tokenizer, so a spliced string would not round-trip to the id
/// the model was trained on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Specials {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub pad: u32,
    /// The mask token's text, which is what gets stripped out of every
    /// caller-supplied string.
    pub mask_text: String,
}

/// Which end of a state longer than its room survives.
///
/// The asymmetry with options is principled and stays: a cut option
/// changes the answer SPACE and is refused by name; a cut state changes
/// the EVIDENCE and is counted (`state_tokens_cut`). This only chooses
/// which half of the evidence the seat reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Cut {
    /// The beginning survives — laya's `agent.py` behaviour, and this
    /// seat's since it was seated: every sealed record was taken so.
    #[default]
    Right,
    /// The end survives — laya's own `truncate_left`, which its agent
    /// never passes. A pasted conversation's ask is usually at the end.
    Left,
}

impl Cut {
    /// The word a door or a config line uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Cut::Right => "right",
            Cut::Left => "left",
        }
    }

    /// The two words and nothing else, refused by name.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "right" => Ok(Cut::Right),
            "left" => Ok(Cut::Left),
            other => Err(format!(
                "unknown state cut {other:?}: `right` keeps the beginning of a long state, \
                 `left` keeps its end"
            )),
        }
    }
}

/// The two length budgets, from `rl_agent_config.json`, and which end of
/// a long state they leave standing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// The whole sequence.
    pub max_len: usize,
    /// The instructions and the markers, before the state gets what is
    /// left.
    pub head_max_len: usize,
    /// Which end of a state longer than its room survives.
    pub cut: Cut,
}

impl Default for Limits {
    /// laya's own defaults, which this checkpoint overrides with
    /// 1024/256.
    fn default() -> Self {
        Self {
            max_len: 512,
            head_max_len: 192,
            cut: Cut::Right,
        }
    }
}

/// One question, ready to read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sequence {
    pub ids: Vec<u32>,
    /// Where each surviving option's marker sits in `ids`.
    pub markers: Vec<usize>,
    /// How many options were asked for, which is how many markers there
    /// would be if none had been pushed off the end.
    pub options: usize,
    pub kind: Kind,
    /// Instruction tokens the head budget cut away.
    pub instruction_tokens_cut: usize,
    /// Option tokens cut, by the per-option cap and by the shared head
    /// budget together — the model read that much less of what the
    /// options meant.
    pub option_tokens_cut: usize,
    /// State tokens cut from the RIGHT to fit `max_len`, the final
    /// truncation included.
    pub state_tokens_cut: usize,
}

/// What one question cost and what it lost, on the wire beside its
/// answer — so a record can tell a silently truncated option from a
/// genuinely low-scoring one, which §138's could not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionUsage {
    pub tokens: usize,
    pub markers: usize,
    pub dropped_options: usize,
    pub instruction_tokens_cut: usize,
    pub option_tokens_cut: usize,
    pub state_tokens_cut: usize,
}

impl Sequence {
    /// Options whose marker fell past `max_len` and therefore have no
    /// position to score. Non-zero means the question cannot be
    /// answered and the caller must refuse it by name.
    pub fn dropped(&self) -> usize {
        self.options.saturating_sub(self.markers.len())
    }

    /// The counts a caller records beside the answer.
    pub fn usage(&self) -> QuestionUsage {
        QuestionUsage {
            tokens: self.ids.len(),
            markers: self.markers.len(),
            dropped_options: self.dropped(),
            instruction_tokens_cut: self.instruction_tokens_cut,
            option_tokens_cut: self.option_tokens_cut,
            state_tokens_cut: self.state_tokens_cut,
        }
    }
}

/// A batch of sequences, padded — plain numbers, so the crate that owns
/// tensors can lay them out and this one stays free of them.
///
/// `input_ids` and `attention` are `rows * len`; `marker_pos` and
/// `marker_mask` are `rows * markers`; `qtype` is one per row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionBatch {
    pub rows: usize,
    pub len: usize,
    pub markers: usize,
    pub input_ids: Vec<u32>,
    pub attention: Vec<u8>,
    pub marker_pos: Vec<u32>,
    pub marker_mask: Vec<bool>,
    pub qtype: Vec<u32>,
}

/// Builds one question's sequence, truncating wherever it must.
///
/// `options` are already rendered ([`crate::system_one::render_options`])
/// — this function splices and cuts them, it does not decide what they
/// say.
pub fn build_sequence(
    enc: &dyn Encode,
    specials: &Specials,
    kind: Kind,
    instructions: &str,
    options: &[String],
    state: &str,
    limits: Limits,
) -> Sequence {
    // A caller's own `[MASK]` would be a marker the scorer then reads as
    // an option. Out it goes, from all three texts, before anything is
    // tokenized.
    let strip = |t: &str| t.replace(&specials.mask_text, " ");

    let mut head_ids = enc.ids(&format!(
        "{} question: {}",
        qtype_name(qtype(kind)),
        strip(instructions)
    ));

    let mut option_tokens_cut = 0usize;
    let mut opt_ids: Vec<Vec<u32>> = options
        .iter()
        .map(|o| {
            let mut body = enc.ids(&format!(" {}", strip(o)));
            option_tokens_cut += body.len().saturating_sub(OPTION_TOKEN_CAP);
            body.truncate(OPTION_TOKEN_CAP);
            let mut v = Vec::with_capacity(body.len() + 1);
            v.push(specials.mask);
            v.extend(body);
            v
        })
        .collect();

    let spent = |o: &[Vec<u32>]| o.iter().map(Vec::len).sum::<usize>() as isize;
    let head_max = limits.head_max_len as isize;
    let mut budget = head_max - spent(&opt_ids);
    if budget < MIN_OPTION_BUDGET as isize {
        // The options have taken the head. Cut them all to an equal
        // share of what is left over the instruction's own floor, and
        // see what that frees.
        let per = MIN_OPTION_TOKENS
            .max(limits.head_max_len.saturating_sub(MIN_OPTION_BUDGET) / opt_ids.len().max(1));
        for o in &mut opt_ids {
            option_tokens_cut += o.len().saturating_sub(per);
            o.truncate(per);
        }
        budget = head_max - spent(&opt_ids);
    }
    let head_before = head_ids.len();
    head_ids.truncate(budget.max(MIN_HEAD_TOKENS as isize) as usize);
    let instruction_tokens_cut = head_before - head_ids.len();

    let mut ids = Vec::with_capacity(limits.max_len);
    ids.push(specials.cls);
    ids.extend_from_slice(&head_ids);
    ids.push(specials.sep);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend_from_slice(o);
    }
    ids.push(specials.sep);

    // What is left for the state, keeping one place for the final [SEP].
    let room = limits.max_len.saturating_sub(ids.len() + 1);
    let mut st = enc.ids(&strip(state));
    // Which end survives is the door's; the count is the same either way.
    let state_before = st.len();
    match limits.cut {
        Cut::Right => st.truncate(room),
        Cut::Left => {
            let gone = st.len().saturating_sub(room);
            st.drain(..gone);
        }
    }
    let mut state_tokens_cut = state_before - st.len();
    ids.extend_from_slice(&st);
    ids.push(specials.sep);

    // The final cut, which only bites when the head and the options
    // alone overflow `max_len`: what it removes is the state's tail
    // (the state is last), so it is counted with the state.
    let before_final = ids.len();
    ids.truncate(limits.max_len);
    state_tokens_cut += before_final - ids.len();
    let markers = markers
        .into_iter()
        .filter(|&m| m < limits.max_len)
        .collect();
    Sequence {
        ids,
        markers,
        options: options.len(),
        kind,
        instruction_tokens_cut,
        option_tokens_cut,
        state_tokens_cut,
    }
}

/// Pads a batch of sequences to one rectangle. `None` for an empty
/// batch — there is nothing to ask the model.
pub fn collate(items: &[Sequence], pad: u32) -> Option<DecisionBatch> {
    if items.is_empty() {
        return None;
    }
    let rows = items.len();
    let len = items.iter().map(|s| s.ids.len()).max().unwrap_or(0);
    let markers = items.iter().map(|s| s.markers.len()).max().unwrap_or(0);

    let mut input_ids = vec![pad; rows * len];
    let mut attention = vec![0u8; rows * len];
    let mut marker_pos = vec![0u32; rows * markers];
    let mut marker_mask = vec![false; rows * markers];
    let mut qtypes = Vec::with_capacity(rows);

    for (r, s) in items.iter().enumerate() {
        let at = r * len;
        input_ids[at..at + s.ids.len()].copy_from_slice(&s.ids);
        attention[at..at + s.ids.len()].fill(1);
        let mat = r * markers;
        for (j, &m) in s.markers.iter().enumerate() {
            marker_pos[mat + j] = m as u32;
            marker_mask[mat + j] = true;
        }
        qtypes.push(qtype(s.kind) as u32);
    }

    Some(DecisionBatch {
        rows,
        len,
        markers,
        input_ids,
        attention,
        marker_pos,
        marker_mask,
        qtype: qtypes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    const CLS: u32 = 50281;
    const SEP: u32 = 50282;
    const PAD: u32 = 50283;
    const MASK: u32 = 50284;

    /// One id per whitespace-separated word, and the literal `[MASK]`
    /// maps to the mask id — so a text this file failed to strip shows
    /// up as a marker id in the wrong place, which is exactly the
    /// hazard. Every text it is handed is recorded verbatim, leading
    /// space and all.
    #[derive(Default)]
    struct Words {
        seen: RefCell<Vec<String>>,
    }

    impl Encode for Words {
        fn ids(&self, text: &str) -> Vec<u32> {
            self.seen.borrow_mut().push(text.to_string());
            text.split_whitespace()
                .map(|w| {
                    if w == "[MASK]" {
                        MASK
                    } else {
                        1000 + w.len() as u32
                    }
                })
                .collect()
        }
    }

    fn specials() -> Specials {
        Specials {
            cls: CLS,
            sep: SEP,
            mask: MASK,
            pad: PAD,
            mask_text: "[MASK]".to_string(),
        }
    }

    fn opts(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn build(
        enc: &Words,
        instructions: &str,
        options: &[String],
        state: &str,
        limits: Limits,
    ) -> Sequence {
        build_sequence(
            enc,
            &specials(),
            Kind::Choice,
            instructions,
            options,
            state,
            limits,
        )
    }

    #[test]
    fn every_marker_points_at_its_own_option() {
        let enc = Words::default();
        let s = build(
            &enc,
            "which one",
            &opts(&["a refund", "an exchange", "nothing"]),
            "the parcel arrived broken",
            Limits::default(),
        );
        assert_eq!(s.markers.len(), 3);
        assert_eq!(s.dropped(), 0);
        for (i, &m) in s.markers.iter().enumerate() {
            assert_eq!(s.ids[m], MASK, "marker {i} does not sit on a mask");
        }
        // [CLS] + "choice question: which one" (4 words) + [SEP] = 6,
        // so the first marker is at index 6 and the options follow in
        // order without a gap.
        assert_eq!(s.markers, vec![6, 9, 12]);
        assert_eq!(s.ids[0], CLS);
        assert_eq!(s.ids[5], SEP);
        assert_eq!(*s.ids.last().unwrap(), SEP);
        assert_eq!(s.kind, Kind::Choice);
        assert_eq!(s.options, 3);
    }

    #[test]
    fn an_option_is_tokenized_with_a_leading_space() {
        let enc = Words::default();
        build(
            &enc,
            "which one",
            &opts(&["refund"]),
            "s",
            Limits::default(),
        );
        let seen = enc.seen.borrow();
        assert!(
            seen.iter().any(|t| t == " refund"),
            "the tokenizer is byte-level and ' refund' is not 'refund': {seen:?}"
        );
        // The head is spelled with the question type in front of it.
        assert_eq!(seen[0], "choice question: which one");
    }

    #[test]
    fn the_option_cap_counts_the_option_and_not_its_marker() {
        let enc = Words::default();
        let long = opts(&[&(0..100).map(|i| format!("w{i} ")).collect::<String>()]);
        let s = build(&enc, "q", &long, "s", Limits::default());
        // One marker plus 48 of its own tokens.
        let after = s.markers[0];
        let head_and_option = s.ids[after..].iter().take_while(|&&t| t != SEP).count();
        assert_eq!(head_and_option, 1 + OPTION_TOKEN_CAP);
    }

    #[test]
    fn a_caller_supplied_mask_never_becomes_a_marker() {
        let enc = Words::default();
        let s = build(
            &enc,
            "is [MASK] the answer",
            &opts(&["yes [MASK] indeed", "no"]),
            "the user wrote [MASK] in their message",
            Limits::default(),
        );
        // Exactly as many mask ids as there are markers, and each one
        // is a marker.
        let masks: Vec<usize> = s
            .ids
            .iter()
            .enumerate()
            .filter(|(_, &t)| t == MASK)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            masks, s.markers,
            "a caller's [MASK] survived into the sequence"
        );
        // And nothing the tokenizer was handed still carries the text.
        for text in enc.seen.borrow().iter() {
            assert!(!text.contains("[MASK]"), "unstripped: {text:?}");
        }
    }

    #[test]
    fn the_state_is_cut_from_the_right() {
        let enc = Words::default();
        // Words of distinct lengths, so an id says which word it was.
        let state = "a bb ccc dddd eeeee ffffff";
        let tight = Limits {
            max_len: 12,
            head_max_len: 192,
            cut: Cut::Right,
        };
        let s = build(&enc, "q", &opts(&["one"]), state, tight);
        assert_eq!(s.ids.len(), tight.max_len);
        // [CLS] q-head(3) [SEP] [MASK] one [SEP] = 8, leaving 3 for the
        // state before the final [SEP]: the FIRST three words survive.
        let tail = &s.ids[8..11];
        assert_eq!(
            tail,
            &[1001, 1002, 1003],
            "the beginning of the state is what survives"
        );
        assert_eq!(*s.ids.last().unwrap(), SEP);
    }

    #[test]
    fn the_instructions_are_cut_but_never_below_their_floor() {
        let enc = Words::default();
        let head = (0..60).map(|i| format!("w{i} ")).collect::<String>();
        // Two options of 40 tokens each leave 192-82 = 110 for the head.
        let room = Limits {
            max_len: 512,
            head_max_len: 192,
            cut: Cut::Right,
        };
        let s = build(&enc, &head, &opts(&["a", "b"]), "s", room);
        assert_eq!(s.markers.len(), 2);

        // Now a head budget so small the options eat it: three options
        // at 4 tokens each is 12, leaving 4 — under the 16 floor — so
        // they are cut and the head keeps its 8.
        let squeeze = Limits {
            max_len: 512,
            head_max_len: 16,
            cut: Cut::Right,
        };
        let t = build_sequence(
            &enc,
            &specials(),
            Kind::Choice,
            &head,
            &opts(&["one two three four five", "six seven eight", "nine ten"]),
            "s",
            squeeze,
        );
        assert_eq!(t.markers.len(), 3);
        let head_len = t.markers[0] - 2; // minus [CLS] and [SEP]
        assert_eq!(head_len, MIN_HEAD_TOKENS);
        // Every option was cut to the floor, marker included.
        for w in t.markers.windows(2) {
            assert_eq!(w[1] - w[0], MIN_OPTION_TOKENS);
        }
    }

    #[test]
    fn what_was_cut_is_counted_per_budget() {
        let enc = Words::default();
        let roomy = Limits {
            max_len: 512,
            head_max_len: 192,
            cut: Cut::Right,
        };
        // Nothing cut: every counter is zero and the usage is the shape.
        let s = build(&enc, "q", &opts(&["one two", "three"]), "a state", roomy);
        let u = s.usage();
        assert_eq!(
            (
                u.instruction_tokens_cut,
                u.option_tokens_cut,
                u.state_tokens_cut,
                u.dropped_options
            ),
            (0, 0, 0, 0)
        );
        assert_eq!(u.tokens, s.ids.len());
        assert_eq!(u.markers, 2);

        // An option past its own cap: the surplus is what the model
        // never read.
        let long = (0..100)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let s = build(&enc, "q", &opts(&[&long, "b"]), "s", roomy);
        assert_eq!(s.usage().option_tokens_cut, 100 - OPTION_TOKEN_CAP);
        assert_eq!(s.usage().instruction_tokens_cut, 0);

        // The squeeze from the test above: a 60-word head under a
        // 16-token head budget keeps its floor of 8, and the options
        // are cut to theirs.
        let head = (0..60).map(|i| format!("w{i} ")).collect::<String>();
        let squeeze = Limits {
            max_len: 512,
            head_max_len: 16,
            cut: Cut::Right,
        };
        let t = build(
            &enc,
            &head,
            &opts(&["one two three four five", "six seven eight", "nine ten"]),
            "s",
            squeeze,
        );
        let u = t.usage();
        // The head the model reads is the instructions PLUS the type
        // prefix this file splices in ("choice question:"), so 60 words
        // of instruction are 62 tokens before the budget cuts them.
        assert_eq!(u.instruction_tokens_cut, 62 - MIN_HEAD_TOKENS);
        // Each option is cut to the 4-token floor, marker included, so
        // three words survive: the five-word option loses two, and the
        // three- and two-word options lose nothing.
        assert_eq!(u.option_tokens_cut, 2);
        assert_eq!(u.state_tokens_cut, 0);

        // A state longer than the room: cut from the right, and counted.
        let state = (0..50)
            .map(|i| format!("s{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let tight = Limits {
            max_len: 24,
            head_max_len: 192,
            cut: Cut::Right,
        };
        let s = build(&enc, "q", &opts(&["a", "b"]), &state, tight);
        assert_eq!(s.dropped(), 0);
        // [CLS] + 3 head words + [SEP] + 2×([MASK] + 1) + [SEP] = 10,
        // leaving 24 - 10 - 1 = 13 for the state.
        assert_eq!(s.usage().state_tokens_cut, 50 - 13);
        assert_eq!(s.ids.len(), 24);
    }

    /// The same state, the same room, the OTHER end: a pasted
    /// conversation's ask is usually at its end, and `Cut::Left` keeps it
    /// — laya's own `truncate_left`, never passed by its agent. What is
    /// cut is counted identically; only which half survives changes.
    /// `Words` maps a word to 1000 + its length, so `s0`..`s9` are 1002
    /// and `s10`..`s49` are 1003: the right cut keeps ten 1002s, the left
    /// cut keeps none.
    #[test]
    fn the_state_is_cut_from_the_left_when_the_door_says_so() {
        let enc = Words::default();
        let state = (0..50)
            .map(|i| format!("s{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let left = Limits {
            max_len: 24,
            head_max_len: 192,
            cut: Cut::Left,
        };
        let s = build(&enc, "q", &opts(&["a", "b"]), &state, left);
        assert_eq!(s.dropped(), 0);
        assert_eq!(
            s.usage().state_tokens_cut,
            50 - 13,
            "the count is the cut's, not the end's"
        );
        assert_eq!(s.ids.len(), 24);
        // The 13 state tokens sit between the second [SEP] and the last.
        let state_ids = &s.ids[10..23];
        assert!(
            state_ids.iter().all(|&id| id == 1003),
            "the END of the state survives a left cut (s37..s49, all length 3): {state_ids:?}"
        );
        let right = Limits {
            cut: Cut::Right,
            ..left
        };
        let s = build(&enc, "q", &opts(&["a", "b"]), &state, right);
        assert_eq!(
            &s.ids[10..20],
            &[1002; 10],
            "the right cut keeps s0..s9: {:?}",
            &s.ids[10..23]
        );
    }

    #[test]
    fn a_marker_pushed_past_the_end_is_counted_not_hidden() {
        let enc = Words::default();
        // Room for the head and the first option's marker only.
        let tiny = Limits {
            max_len: 8,
            head_max_len: 192,
            cut: Cut::Right,
        };
        let s = build(
            &enc,
            "q",
            &opts(&["one two three", "four five six", "seven"]),
            "s",
            tiny,
        );
        assert!(s.ids.len() <= tiny.max_len);
        assert_eq!(s.options, 3);
        assert!(s.markers.len() < 3, "markers: {:?}", s.markers);
        assert_eq!(s.dropped(), 3 - s.markers.len());
        assert!(s.dropped() > 0, "this question cannot be answered");
        for &m in &s.markers {
            assert!(m < tiny.max_len);
        }
    }

    #[test]
    fn collate_pads_every_row_and_masks_what_is_padding() {
        let enc = Words::default();
        let short = build(&enc, "q", &opts(&["a"]), "s", Limits::default());
        let long = build_sequence(
            &enc,
            &specials(),
            Kind::Noul,
            "is it so",
            &opts(&["false: no", "true: yes"]),
            "a much longer state than the other one had",
            Limits::default(),
        );
        assert!(long.ids.len() > short.ids.len());

        let b = collate(&[short.clone(), long.clone()], PAD).expect("two rows");
        assert_eq!(b.rows, 2);
        assert_eq!(b.len, long.ids.len());
        assert_eq!(b.markers, 2);
        assert_eq!(b.qtype, vec![0, 2]);

        // Row 0 is the short one: its own ids, then padding that the
        // attention mask denies.
        assert_eq!(&b.input_ids[..short.ids.len()], &short.ids[..]);
        for i in short.ids.len()..b.len {
            assert_eq!(b.input_ids[i], PAD);
            assert_eq!(b.attention[i], 0);
        }
        for i in 0..short.ids.len() {
            assert_eq!(b.attention[i], 1);
        }
        // Row 1 fills its row completely.
        assert!(b.attention[b.len..].iter().all(|&a| a == 1));

        // One marker in row 0, two in row 1: the spare slot is masked
        // off and its position is a zero nobody reads.
        assert_eq!(&b.marker_mask, &[true, false, true, true]);
        assert_eq!(b.marker_pos[1], 0);
        assert_eq!(b.marker_pos[0] as usize, short.markers[0]);
        assert_eq!(b.marker_pos[2] as usize, long.markers[0]);
        assert_eq!(b.marker_pos[3] as usize, long.markers[1]);

        assert!(collate(&[], PAD).is_none());
    }
}
