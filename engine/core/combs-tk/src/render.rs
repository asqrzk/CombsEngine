//! One seam every command prints its subject through.
//!
//! A seat is named by its identity — the directory a person recognises —
//! and never by the path the caller happened to type. A header carrying
//! someone's home directory cannot be pasted into a report, cannot be
//! compared against the same seat on another machine, and blows out
//! every column beside it.
//!
//! The run records and `ps` already obeyed that rule. `fit` did not, and
//! the reason is worth keeping: the rule lived in each call site instead
//! of in one function, so every new place that printed a model had to
//! remember it. It lives here now, and remembering is no longer the
//! mechanism.

use crate::state;

/// The heading a command prints above its answer: what it did, and to
/// which seat.
pub fn seat_header(verb: &str, model: &str) -> String {
    format!("{verb} — {}", state::identity(model))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every header names the seat, never the caller's path — the same
    /// law the records keep. It failed in `fit` because it was
    /// remembered in three call sites rather than held in one.
    #[test]
    fn a_seat_is_named_by_identity_in_every_header() {
        let h = seat_header(
            "fit",
            "/home/someone/.cache/combs/models/qwen3-4b-gguf/model.gguf",
        );
        assert!(!h.contains('/'), "a header must not carry a path: {h}");
        assert!(
            h.contains("qwen3-4b-gguf"),
            "a header must name the seat: {h}"
        );
        assert!(h.starts_with("fit"), "a header must name what was done: {h}");
    }

    /// A bare name is already an identity and must survive untouched.
    #[test]
    fn a_name_that_is_already_a_name_is_left_alone() {
        assert_eq!(seat_header("measure", "qwen3-0.6b-gguf"), "measure — qwen3-0.6b-gguf");
    }
}
