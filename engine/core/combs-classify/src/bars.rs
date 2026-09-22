//! The bars a rung is judged against, each with the argument that set it.
//!
//! A number here is a decision someone can disagree with, so it carries
//! its reason at the site. A bar is TIGHTENED by evidence and never
//! widened to make a rung pass — a rung that cannot meet its bar is
//! recorded as refused, with the number it reached.

/// How far a paraphrase's top probability may move from its base before
/// the suite calls the rung topical rather than semantic.
///
/// The first registered value and UNMEASURED: 0.15 is a judgement, not a
/// reading. The suite reports the drift it actually observed beside the
/// verdict so the first record can tighten this.
pub const PARAPHRASE_MAX_DRIFT: f32 = 0.15;

/// Expected calibration error, over ten equal-width confidence bins,
/// past which a rung's probabilities are not trustworthy enough to gate
/// on. The blueprint this design follows ships at 0.466 uncalibrated and
/// reaches 0.081 after fitting one temperature per cell, so 0.10 is the
/// near side of "fitted", not an aspiration.
pub const MAX_ECE: f32 = 0.10;

/// The share of paraphrases that must keep their base's top option.
pub const MIN_AGREEMENT: f32 = 0.95;

/// Options in one question, past which the request is refused whole.
///
/// Structural, not taste: the constrained decoder's own enum cap is 64
/// literals, so a question wider than this cannot be served by the rung
/// that needs no new model at all.
pub const MAX_OPTIONS: usize = 64;

/// Questions in one request, past which the request is refused whole.
///
/// A System 1 seat answers every question of a request in ONE forward
/// pass, one row per question — so the question count is the batch
/// size, and nothing else bounded it (the cosine road's cap counts
/// embeddings, which that seat never makes). The gateway keeps the
/// same number under `MAX_QUESTIONS` (combs-inference `serve.conf`),
/// by hand, in two trees: a test pins each so a drift is a failure.
pub const MAX_QUESTIONS: usize = 16;

/// The default confidence floor and implication floor: ZERO, which is
/// today's behaviour — act on the top option, with the implication rule
/// still binding. A deployment raises them from its own sweep; a guessed
/// threshold would refuse work nobody measured.
pub const DEFAULT_CONFIDENCE: f32 = 0.0;
/// See [`DEFAULT_CONFIDENCE`].
pub const DEFAULT_TAU: f32 = 0.0;

/// Microseconds the gate itself may cost per call, median of a thousand.
/// The gate runs on every decision, so a slow gate is a tax on the whole
/// road; this is registered UNMEASURED and the kit's cell reads it.
pub const GATE_BUDGET_US: u64 = 1_000;
