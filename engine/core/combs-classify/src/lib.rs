//! The gate a typed decision passes before anyone acts on it.
//!
//! A typed decision is a question with a FIXED answer space: the caller
//! hands over the options, and the answer is an index into them plus a
//! distribution. Nothing can be fabricated, because there is no
//! vocabulary on the output side — a stronger guarantee than
//! schema-forced decoding, where the shape is constrained while tokens
//! are still being generated.
//!
//! What remains after that is four failure modes, and this crate is the
//! four rules that hold them:
//!
//! 1. A wrong option held with high confidence — CALIBRATION. One
//!    temperature per (kind, option count), scored with ECE *and*
//!    Brier, and `calibrated: false` said out loud when none is fitted.
//! 2. A label matched by TOPIC rather than by meaning — the TETHER
//!    (see [`tether`]), plus the implication rule below.
//! 3. A low-information state answered anyway — ABSTENTION by
//!    confidence, naming the rule that refused.
//! 4. Anything outside the option set — the fixed answer space itself.
//!
//! What the gate CANNOT do, said plainly: a calibrated classifier fed an
//! input from outside the distribution it was measured on can be
//! confidently wrong, and calibration measured on a fixture says nothing
//! about inputs off it. The defences there are an abstention head
//! trained on negatives and the caller's own fallback — not this crate.

pub mod bars;
pub mod suite;
pub mod sweep;
pub mod system_one;
pub mod tether;

pub use tether::{Opt, Tether};

use serde::{Deserialize, Serialize};

/// The three question kinds, which are the whole interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// One of N tethered options; the answer is a distribution over them.
    Choice,
    /// An ordered rubric; the answer is a distribution plus its
    /// expectation over the level indices.
    Score,
    /// One proposition; the answer is its probability. A speech-act noul
    /// beside an action option is what makes the implication rule work.
    Noul,
}

/// What the caller is told to do, and why when it is not "act".
///
/// The caller branches on this and never parses the reason — the reason
/// is for a person reading a log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Act,
    Abstain(String),
    Refused(String),
}

impl Decision {
    pub fn word(&self) -> &'static str {
        match self {
            Decision::Act => "act",
            Decision::Abstain(_) => "abstain",
            Decision::Refused(_) => "refused",
        }
    }
    pub fn reason(&self) -> Option<&str> {
        match self {
            Decision::Act => None,
            Decision::Abstain(r) | Decision::Refused(r) => Some(r),
        }
    }
}

/// One implication, resolved: the question it names, the probability
/// that question came back with, and the floor it had to clear.
///
/// This is the "if A is detected then B must hold" rule made a runtime
/// check. For an action option, B is the speech act: the translate-case
/// fails here by construction, because nothing was requested.
#[derive(Debug, Clone, PartialEq)]
pub struct Implication {
    pub requires: String,
    pub p: f32,
    pub tau: f32,
}

impl Implication {
    pub fn holds(&self) -> bool {
        self.p >= self.tau
    }

    /// Whether this implication can refuse ANYTHING.
    ///
    /// A floor of zero is not a gate: every probability clears it, so
    /// `holds` is true whatever the question answered. That matters
    /// because an option which declares `requires` is ASKING to be
    /// gated, and a caller reading `holds: true` deserves to know
    /// whether a thing was checked or merely not checked.
    pub fn gated(&self) -> bool {
        self.tau > 0.0
    }
}

/// The floors a deployment sets. Both default to ZERO, which is today's
/// behaviour — act on the top option, with the implication rule still
/// binding. A deployment raises them from its own sweep; a guessed
/// threshold would refuse work nobody measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    pub confidence: f32,
    pub tau: f32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self { confidence: bars::DEFAULT_CONFIDENCE, tau: bars::DEFAULT_TAU }
    }
}

/// How far a sum may sit from one before the distribution is refused.
/// f32 over sixty-four options accumulates a little; a real mistake
/// lands orders of magnitude away from this.
const SUM_TOLERANCE: f32 = 1e-5;

/// Refuse a distribution that is not one: empty, negative, not finite,
/// or not summing to one. Every answer passes through here before any
/// rule reads it, so a scorer's arithmetic mistake cannot become a
/// decision.
pub fn check_distribution(p: &[f32]) -> Result<(), String> {
    if p.is_empty() {
        return Err("an empty distribution is not a distribution".to_string());
    }
    let mut sum = 0.0f64;
    for (i, &v) in p.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("probability {i} is not a number"));
        }
        if v < 0.0 {
            return Err(format!("probability {i} is {v}, and a probability is never negative"));
        }
        sum += v as f64;
    }
    if (sum - 1.0).abs() > SUM_TOLERANCE as f64 {
        return Err(format!(
            "the distribution sums to {sum:.4}, not 1 (tolerance {SUM_TOLERANCE:e})"
        ));
    }
    Ok(())
}

/// Confidence as normalised entropy: `1 - H(p)/ln K`.
///
/// One at a point mass, zero at uniform, and it reads the WHOLE
/// distribution rather than the top two — a long tail of near-misses is
/// less certain than a single rival, and a margin cannot see that. A
/// single-option question is certain by construction.
pub fn confidence(p: &[f32]) -> f32 {
    let k = p.len();
    if k <= 1 {
        return 1.0;
    }
    let mut h = 0.0f64;
    for &v in p {
        let v = v as f64;
        if v > 0.0 {
            h -= v * v.ln();
        }
    }
    let c = 1.0 - h / (k as f64).ln();
    (c as f32).clamp(0.0, 1.0)
}

/// The expectation over level indices, for an ordinal `score` question.
pub fn expected_level(p: &[f32]) -> f32 {
    let mut acc = 0.0f64;
    for (i, &v) in p.iter().enumerate() {
        acc += i as f64 * v as f64;
    }
    acc as f32
}

/// The decision rule, and the only place act/abstain is decided.
///
/// Act iff the confidence clears its floor AND every implication holds.
/// Otherwise abstain, naming the FIRST rule that refused — one reason,
/// so a log line says one thing.
pub fn decide(confidence: f32, threshold: f32, implications: &[Implication]) -> Decision {
    if confidence < threshold {
        return Decision::Abstain(format!(
            "confidence {confidence:.2} < threshold {threshold:.2}"
        ));
    }
    for imp in implications {
        if !imp.holds() {
            return Decision::Abstain(format!(
                "implication {}: {:.2} < tau {:.2}",
                imp.requires, imp.p, imp.tau
            ));
        }
    }
    Decision::Act
}

/// `softmax(z / T)`. Lower T is sharper; T is fitted per cell, never
/// guessed, and T = 1 is the unfitted identity.
pub fn softmax_scaled(logits: &[f32], t: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let t = if t.is_finite() && t > 1e-6 { t } else { 1e-6 };
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f64> = logits.iter().map(|&v| (((v - m) / t) as f64).exp()).collect();
    let s: f64 = e.iter().sum();
    if s <= 0.0 || !s.is_finite() {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    e.into_iter().map(|v| (v / s) as f32).collect()
}

/// Expected calibration error over equal-width confidence bins.
///
/// Reported WITH Brier, never alone: ECE can be made small by a model
/// that is uninformative but honest about it, and Brier catches that.
pub fn ece(pairs: &[(f32, bool)], bins: usize) -> f32 {
    if pairs.is_empty() || bins == 0 {
        return 0.0;
    }
    let mut sum_conf = vec![0.0f64; bins];
    let mut hits = vec![0.0f64; bins];
    let mut count = vec![0.0f64; bins];
    for &(c, correct) in pairs {
        let c = c.clamp(0.0, 1.0);
        let mut b = (c * bins as f32) as usize;
        if b >= bins {
            b = bins - 1;
        }
        sum_conf[b] += c as f64;
        hits[b] += if correct { 1.0 } else { 0.0 };
        count[b] += 1.0;
    }
    let n = pairs.len() as f64;
    let mut e = 0.0f64;
    for b in 0..bins {
        if count[b] == 0.0 {
            continue;
        }
        let mean_conf = sum_conf[b] / count[b];
        let accuracy = hits[b] / count[b];
        e += (count[b] / n) * (mean_conf - accuracy).abs();
    }
    e as f32
}

/// Mean squared error between the stated confidence and the outcome.
pub fn brier(pairs: &[(f32, bool)]) -> f32 {
    if pairs.is_empty() {
        return 0.0;
    }
    let mut s = 0.0f64;
    for &(c, correct) in pairs {
        let y = if correct { 1.0f64 } else { 0.0 };
        let d = c as f64 - y;
        s += d * d;
    }
    (s / pairs.len() as f64) as f32
}

/// The negative log likelihood of `labels` under `softmax(z / T)`.
fn nll(logits: &[Vec<f32>], labels: &[usize], t: f32) -> f64 {
    let t = if t.is_finite() && t > 1e-3 { t } else { 1e-3 };
    let mut s = 0.0f64;
    let mut n = 0.0f64;
    for (z, &y) in logits.iter().zip(labels.iter()) {
        let Some(&zy) = z.get(y) else { continue };
        let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f64;
        for &v in z {
            denom += (((v - m) / t) as f64).exp();
        }
        if denom <= 0.0 || !denom.is_finite() {
            continue;
        }
        s -= ((zy - m) / t) as f64 - denom.ln();
        n += 1.0;
    }
    if n == 0.0 {
        return f64::MAX;
    }
    s / n
}

/// Fit one temperature by minimising negative log likelihood, by golden
/// section over `ln T` in a bounded range.
///
/// Pure and deterministic: the same table always gives the same T, so a
/// calibration can be reproduced from the record that reports it.
/// A table with nothing to fit stays at the unfitted identity.
pub fn fit_temperature(logits: &[Vec<f32>], labels: &[usize]) -> f32 {
    if logits.is_empty() || logits.len() != labels.len() {
        return 1.0;
    }
    let (mut a, mut b) = (0.05f32.ln(), 10.0f32.ln());
    let gr = (5.0f32.sqrt() - 1.0) / 2.0;
    let mut c = b - gr * (b - a);
    let mut d = a + gr * (b - a);
    let mut fc = nll(logits, labels, c.exp());
    let mut fd = nll(logits, labels, d.exp());
    for _ in 0..60 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = nll(logits, labels, c.exp());
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = nll(logits, labels, d.exp());
        }
        if (b - a).abs() < 1e-5 {
            break;
        }
    }
    ((a + b) / 2.0).exp()
}
