//! The System 1 seat: text in, typed answers out.
//!
//! This is where the three halves meet. `combs-classify` renders a
//! question and reads a distribution back; `combs-models` holds the
//! tensors; the tokenizer lives here, and so this is the only place
//! that can turn a person's words into markers and markers back into
//! an answer.
//!
//! A laya checkpoint's own layout is not flat — the weights sit at the
//! root, the encoder's config under `encoder/`, the tokenizer under
//! `tokenizer/`, and the calibration beside them — so [`DecisionEngine::load`]
//! resolves each of those and accepts a flat directory too, which is
//! what the tiny parity fixture is and what the seat's own tests run on.

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Mutex;
// clock: native — `mod system_one` is cfg'd off wasm in lib.rs.
use std::time::Instant;

use combs_classify::bars;
use combs_classify::sequence::{
    build_sequence, collate, Cut, DecisionBatch, Encode, Limits, QuestionUsage, Sequence, Specials,
};
use combs_classify::system_one::{
    engine_order, readout, render_options, AgentConfig, Calibration, Criteria,
};
use combs_classify::Kind;
use combs_core::CombsBackend;
use combs_formats::{ModelMetadata, SafetensorsSource};
use combs_models::{DecisionLogits, LayaModel, TypedDecisionModel};
use tokenizers::Tokenizer;

use crate::EngineError;

type Result<T> = std::result::Result<T, EngineError>;

/// The tokenizer, behind the seam `combs-classify` defines so that it
/// links none.
struct Tok(Tokenizer);

impl Tok {
    /// One request's view of the tokenizer: the same ids, plus a record
    /// of the first text it could not read, kept on THIS call's stack so
    /// two requests on two threads can never read each other's failure.
    fn view(&self) -> TokView<'_> {
        TokView {
            tok: &self.0,
            failed: Cell::new(None),
        }
    }
}

/// [`Encode`] for one request. The seam returns ids and nothing else, so
/// a text the tokenizer cannot read used to become an EMPTY text — the
/// question kept its markers and was answered on nothing, and the record
/// could not tell that from a short state (`state_tokens_cut` reads 0
/// either way). The failure is recorded here and the request is refused
/// by name once every text has been read.
struct TokView<'a> {
    tok: &'a Tokenizer,
    failed: Cell<Option<String>>,
}

impl TokView<'_> {
    /// The first text this request could not read, if any.
    fn failure(&self) -> Option<String> {
        self.failed.take()
    }
}

impl Encode for TokView<'_> {
    fn ids(&self, text: &str) -> Vec<u32> {
        // `false`: this file splices every special by NUMBER, so the
        // tokenizer must not add any of its own.
        match self.tok.encode(text, false) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(e) => {
                // `Cell::take` is destructive, so an earlier failure is
                // taken out and put back rather than peeked at — asking
                // `take().is_none()` would throw away the very thing it
                // was checking for. The FIRST unreadable text is the one
                // the refusal names.
                match self.failed.take() {
                    Some(first) => self.failed.set(Some(first)),
                    None => {
                        let shown: String = text.chars().take(80).collect();
                        self.failed.set(Some(format!("{shown:?}: {e}")));
                    }
                }
                Vec::new()
            }
        }
    }
}

/// One question, as the seat is asked it.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionQuestion {
    pub name: String,
    pub instructions: String,
    pub criteria: Criteria,
}

/// One question, answered.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionAnswer {
    pub name: String,
    pub kind: Kind,
    /// In THIS ENGINE's order, which for a noul puts the proposition
    /// first — the model puts it second, and [`engine_order`] is where
    /// the two conventions meet.
    pub distribution: Vec<f32>,
    /// The head's raw logits, in the same order as `distribution`, so a
    /// caller can key both by the same option ids.
    pub logits: Vec<f32>,
    /// The temperature applied, and the bucket it came from.
    pub temperature: f32,
    pub bucket: String,
    /// The shipped value that temperature was clamped from, when the
    /// loader refused the checkpoint's own number — `None` when what was
    /// applied is what was written.
    pub clamped_from: Option<f32>,
    /// What the act head made of the decision: the probability of
    /// answering rather than escalating.
    pub p_act: f32,
    /// What this question cost and what it lost.
    pub usage: QuestionUsage,
}

/// The padded rectangle one request became: every question a row,
/// every row the longest question's length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchShape {
    pub rows: usize,
    pub len: usize,
    pub markers: usize,
}

/// Every answer, and the shape of the batch that produced them.
#[derive(Debug, Clone, PartialEq)]
pub struct Decided {
    pub answers: Vec<DecisionAnswer>,
    pub batch: BatchShape,
}

/// One warm-up pass and what it cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarmPass {
    pub rows: usize,
    pub len: usize,
    /// The marker width this pass tuned. A rectangle is not one shape:
    /// the head's matmuls are sized by the markers, so two passes of the
    /// same rows and length at different widths are different kernels.
    pub markers: usize,
    pub ms: u64,
}

/// What warming the seat cost — so a record can say the first real
/// request paid no autotune, and how long the seat took to be ready.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WarmReport {
    pub passes: Vec<WarmPass>,
    pub total_ms: u64,
}

/// A legal rectangle of `rows` questions, `len` tokens each, `markers`
/// options apiece — `[CLS] [MASK]…[MASK] [SEP] [SEP]…` to the full
/// length, so the kernels see every shape a request could take before
/// one arrives. `None` when the shape cannot hold its own markers.
pub fn warm_batch(
    rows: usize,
    len: usize,
    markers: usize,
    specials: &Specials,
) -> Option<DecisionBatch> {
    if rows == 0 || markers == 0 || len < markers + 2 {
        return None;
    }
    let seqs: Vec<Sequence> = (0..rows)
        .map(|_| {
            let mut ids = Vec::with_capacity(len);
            ids.push(specials.cls);
            let mut marks = Vec::with_capacity(markers);
            for _ in 0..markers {
                marks.push(ids.len());
                ids.push(specials.mask);
            }
            ids.push(specials.sep);
            // Real tokens to the row's full length: what matters is the
            // shape the matmuls see, and a padded row costs the same.
            while ids.len() < len {
                ids.push(specials.sep);
            }
            Sequence {
                ids,
                markers: marks,
                options: markers,
                kind: Kind::Choice,
                instruction_tokens_cut: 0,
                option_tokens_cut: 0,
                state_tokens_cut: 0,
            }
        })
        .collect();
    collate(&seqs, specials.pad)
}

/// The shapes a seat warms by default, as `(rows, len, markers)`.
///
/// A rectangle alone was not the shape: every pass here used to warm two
/// markers, so the head's own matmuls were tuned at width two and
/// nothing else. Measured on the ae86e63 smoke, an untuned shape costs
/// **≈ 5 s on its first call** — and every seed question (7 and 9
/// options), every small one-question body, and any request past four
/// questions met one. Three gaps are closed here: the smallest row a
/// request can make, a real marker width beside the suites' rectangle,
/// and the widest head the wire admits (`MAX_QUESTIONS` rows of
/// `MAX_OPTIONS` markers) at a length that keeps that pass on the same
/// 4,096-token rectangle as the deepest one, rather than adding a
/// 16,384-token pass to every start.
///
/// Each length is clamped to the checkpoint's `max_len`, and a shape
/// that could not hold its own markers is dropped rather than refused —
/// a tiny checkpoint should still warm what it can.
pub fn default_warm_shapes(limits: Limits) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for (rows, len, markers) in [
        (1, 64, 2),
        (1, 256, 2),
        (2, 256, 9),
        (2, 512, 2),
        (4, 1024, 2),
        (bars::MAX_QUESTIONS, 256, bars::MAX_OPTIONS),
    ] {
        let len = len.min(limits.max_len);
        if len < markers + 2 {
            continue;
        }
        if !out.contains(&(rows, len, markers)) {
            out.push((rows, len, markers));
        }
    }
    out
}

/// Turns the head's raw output into answers, refusing a shape it cannot
/// read rather than indexing into it.
///
/// Every index here used to be taken on faith: `out.logits[i]` assumed a
/// row per question, `[..k]` assumed that row was at least as long as
/// the question's options, and `out.act[i]` assumed the same of the act
/// head. A model that returned fewer rows than it was asked about — a
/// loader that quietly built the wrong head, a kernel that dropped a
/// launch — panicked inside the seat, OUTSIDE `guarded_decide`, which
/// catches only the forward pass. The seat then died on a bug in its own
/// reading, and the permanent `panics` count that was supposed to carry
/// a fault forward never got the chance to record it.
///
/// Pure: no device, no lock, no tokenizer. A refusal names the question
/// and both numbers, because "the model returned the wrong shape" is
/// useless to whoever has to find out why.
fn read_answers(
    questions: &[DecisionQuestion],
    rendered: &[Vec<String>],
    sequences: &[Sequence],
    out: &DecisionLogits,
    calibration: &Calibration,
) -> Result<Vec<DecisionAnswer>> {
    let bad = |what: String| EngineError::Model(combs_models::ModelError::Unsupported(what));
    if out.logits.len() != questions.len() {
        return Err(bad(format!(
            "the head returned {} rows for {} questions",
            out.logits.len(),
            questions.len()
        )));
    }
    if out.act.len() != questions.len() {
        return Err(bad(format!(
            "the act head returned {} rows for {} questions",
            out.act.len(),
            questions.len()
        )));
    }
    let mut answers = Vec::with_capacity(questions.len());
    for (i, q) in questions.iter().enumerate() {
        let kind = q.criteria.kind();
        let k = rendered[i].len();
        if out.logits[i].len() < k {
            return Err(bad(format!(
                "question {:?}: the head returned {} logits for {} markers — an answer over \
                 fewer options is an answer to a different question",
                q.name,
                out.logits[i].len(),
                k
            )));
        }
        let logits = &out.logits[i][..k];
        let t = calibration.temperature(kind, k);
        let p = readout(logits, t);
        answers.push(DecisionAnswer {
            name: q.name.clone(),
            kind,
            distribution: engine_order(kind, &p),
            logits: engine_order(kind, logits),
            temperature: t,
            bucket: calibration.bucket(kind, k),
            clamped_from: calibration.clamped_from(kind, k),
            // Index zero is answering; the rest are the acts the
            // config prices.
            p_act: out.act[i].first().copied().unwrap_or(0.0),
            usage: sequences[i].usage(),
        });
    }
    Ok(answers)
}

/// Runs one batch through the model, surviving a fault and recording it.
///
/// Two things that used to be one. **The lock is recovered**: `decide`
/// takes `&self` and mutates nothing, so a Mutex poisoned by an earlier
/// panic guards a model still exactly as it was loaded — the Mutex is
/// here to serialise the GPU, not to guard mutation, and dying of
/// poison threw away a sound seat. **The panic is contained and
/// counted**: one malformed batch that trips an index refuses ITSELF
/// rather than ending a process that answers everything else.
///
/// Recovering silently would be the wrong half of the fix. A seat that
/// has ever faulted says so for the rest of its life, so a measurement
/// can never be drawn from a broken process without the record showing
/// it. A panic here is OUR code faulting before the model answers — it
/// is not the model inventing anything, and this seat cannot invent
/// anything: it scores one marker per provided option and decodes
/// nothing.
fn guarded_decide(
    model: &Mutex<Box<dyn TypedDecisionModel>>,
    panics: &AtomicU64,
    batch: &DecisionBatch,
) -> Result<DecisionLogits> {
    let model = model.lock().unwrap_or_else(|e| e.into_inner());
    // `AssertUnwindSafe` is honest precisely because `decide` takes
    // `&self`: there is no half-written state for an unwind to expose.
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| model.decide(batch))) {
        Ok(r) => r.map_err(EngineError::Model),
        Err(_) => {
            panics.fetch_add(1, AtomicOrdering::Relaxed);
            Err(EngineError::Model(combs_models::ModelError::Unsupported(
                "the model panicked answering this batch: the request is refused and the seat \
                 stays up, but this process has now faulted and says so in \
                 `/v1/stats.classify.panics` for the rest of its life"
                    .to_string(),
            )))
        }
    }
}

/// A seated typed-decision model.
pub struct DecisionEngine {
    // One at a time: the model holds device buffers sized for one batch.
    model: Mutex<Box<dyn TypedDecisionModel>>,
    /// How many requests this seat has panicked on, for the life of the
    /// process. Never reset.
    ///
    /// A panic here is OUR bug faulting before the model answers — it is
    /// not the model inventing anything, and this seat cannot invent
    /// anything (it scores one marker per provided option and decodes
    /// nothing). But a seat that has ever faulted must SAY so, or a
    /// measurement can be drawn from a process that broke and no reader
    /// can tell. Recovery without a record is the wrong half of the fix.
    panics: AtomicU64,
    tokenizer: Tok,
    calibration: Calibration,
    limits: Limits,
    specials: Specials,
    metadata: ModelMetadata,
    model_id: String,
}

/// The first of `names` that exists under `dir`.
fn first_of(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    names.iter().map(|n| dir.join(n)).find(|p| p.exists())
}

impl DecisionEngine {
    /// Seats the checkpoint in `dir`.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let bad = |what: String| EngineError::Model(combs_models::ModelError::Unsupported(what));

        let config_path =
            first_of(dir, &["encoder/config.json", "config.json"]).ok_or_else(|| {
                bad(format!(
                    "{}: no encoder/config.json and no config.json — a decision seat needs the \
                 encoder's geometry",
                    dir.display()
                ))
            })?;
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&config_path)
                .map_err(|e| bad(format!("{}: {e}", config_path.display())))?,
        )
        .map_err(|e| bad(format!("{}: {e}", config_path.display())))?;
        let metadata = ModelMetadata::from_hf_config(&config, None)
            .map_err(|e| bad(format!("{}: {e}", config_path.display())))?;
        let encoder = metadata.encoder.clone().ok_or_else(|| {
            bad(format!(
                "{} names architecture {:?}, which is not a bidirectional encoder: a decision seat \
                 scores option markers and a decoder has none",
                config_path.display(),
                metadata.architecture
            ))
        })?;

        let tok_path = first_of(dir, &["tokenizer/tokenizer.json", "tokenizer.json"])
            .ok_or_else(|| bad(format!("{}: no tokenizer.json", dir.display())))?;
        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| bad(format!("{}: {e}", tok_path.display())))?;

        // The calibration, and the two length budgets the marker
        // sequence is built to. A checkpoint without one is neutral
        // rather than refused — every temperature 1.0, said out loud.
        let agent = match first_of(dir, &["rl_agent_config.json"]) {
            Some(p) => {
                let text = std::fs::read_to_string(&p)
                    .map_err(|e| bad(format!("{}: {e}", p.display())))?;
                AgentConfig::parse(&text).map_err(|e| bad(format!("{}: {e}", p.display())))?
            }
            None => AgentConfig::parse("{}").expect("an empty config is defaults"),
        };
        for refused in agent.calibration.clamped() {
            tracing::warn!(
                event = "calibration_clamped",
                value = %refused,
                "this checkpoint ships a temperature outside the usable band; it is clamped, not \
                 applied as written — a fitted temperature below 1 sharpens the logits rather than \
                 softening them"
            );
        }

        // The mask token by NAME then by id: `[MASK]` is `lstrip: true`
        // in this tokenizer, so a spliced string would not round-trip,
        // and a guessed id would mark the wrong position.
        let mask_text = first_of(
            dir,
            &["tokenizer/tokenizer_config.json", "tokenizer_config.json"],
        )
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| match v.get("mask_token") {
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(o) => o
                .get("content")
                .and_then(|c| c.as_str())
                .map(str::to_string),
            None => None,
        })
        .unwrap_or_else(|| "[MASK]".to_string());
        let mask = tokenizer.token_to_id(&mask_text).ok_or_else(|| {
            bad(format!(
                "the tokenizer has no {mask_text:?}, so no option could be marked"
            ))
        })?;

        let source = SafetensorsSource::load_weights_only(dir, &metadata.architecture)
            .map_err(|e| bad(format!("{}: {e}", dir.display())))?;
        let device = combs_core::init_device();
        let model =
            LayaModel::<CombsBackend>::load(&source, &device, metadata.clone(), agent.head_layers)
                .map_err(EngineError::Model)?;

        Ok(Self {
            model: Mutex::new(Box::new(model)),
            panics: AtomicU64::new(0),
            tokenizer: Tok(tokenizer),
            calibration: agent.calibration,
            limits: Limits {
                max_len: agent.max_len,
                head_max_len: agent.head_max_len,
                cut: Cut::Right,
            },
            specials: Specials {
                cls: encoder.cls_id,
                sep: encoder.sep_id,
                mask,
                pad: encoder.pad_id,
                mask_text,
            },
            metadata,
            model_id: dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "laya".to_string()),
        })
    }

    /// A seat over any model and tokenizer, for the seat's own tests: the
    /// calibration neutral, laya's default limits, the parity fixture's
    /// specials, and the tiny encoder's geometry as its metadata.
    #[cfg(test)]
    fn for_tests(model: Box<dyn TypedDecisionModel>, tokenizer: Tokenizer) -> Self {
        let config: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../combs-models/tests/fixtures/laya_tiny_config.json"
        )))
        .expect("the tiny config is json");
        Self {
            model: Mutex::new(model),
            panics: AtomicU64::new(0),
            tokenizer: Tok(tokenizer),
            calibration: Calibration::neutral(),
            limits: Limits::default(),
            specials: Specials {
                cls: 1,
                sep: 2,
                mask: 3,
                pad: 0,
                mask_text: "[MASK]".to_string(),
            },
            metadata: ModelMetadata::from_hf_config(&config, None).expect("the tiny config reads"),
            model_id: "test".to_string(),
        }
    }

    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// The two length budgets this seat answers within.
    pub fn limits(&self) -> Limits {
        self.limits
    }

    /// Which end of a long state this seat keeps — the door's, read by
    /// the seat before the weights and handed here after them.
    pub fn with_state_cut(mut self, cut: Cut) -> Self {
        self.limits.cut = cut;
        self
    }

    /// Runs the model over `shapes` before any request arrives, so the
    /// first real one pays neither autotune nor a first-launch compile.
    /// The seat's first request measured 1,058 ms cold (s138); a router
    /// with a 1,500 ms budget should never be the one to pay it. Every
    /// pass syncs on its readback, so a launch error surfaces HERE
    /// rather than in a caller's answer.
    pub fn warm(&self, shapes: &[(usize, usize, usize)]) -> Result<WarmReport> {
        let bad = |what: String| EngineError::Model(combs_models::ModelError::Unsupported(what));
        // A previous panic must not end the seat. `decide` takes `&self`
        // and mutates nothing, so a lock poisoned by an earlier fault
        // guards a model that is still exactly as it was loaded; the
        // Mutex is here to serialise the GPU, not to guard mutation.
        // `panics` is what carries the fault forward, not a dead lock.
        let model = self.model.lock().unwrap_or_else(|e| e.into_inner());
        let t0 = Instant::now();
        let mut passes = Vec::with_capacity(shapes.len());
        for &(rows, len, markers) in shapes {
            let batch = warm_batch(rows, len, markers, &self.specials).ok_or_else(|| {
                bad(format!(
                    "a warm-up shape of {rows}x{len} cannot hold its own {markers} markers"
                ))
            })?;
            let t = Instant::now();
            model.decide(&batch).map_err(EngineError::Model)?;
            passes.push(WarmPass {
                rows,
                len,
                markers,
                ms: t.elapsed().as_millis() as u64,
            });
        }
        Ok(WarmReport {
            passes,
            total_ms: t0.elapsed().as_millis() as u64,
        })
    }

    /// Every question over one state, in ONE forward pass.
    ///
    /// That is the shape this seat exists for: a batch of questions
    /// costs one pass, where scoring by embedding similarity costs one
    /// embedding per distinct instruction.
    pub fn decide(
        &self,
        state: &str,
        questions: &[DecisionQuestion],
    ) -> Result<Vec<DecisionAnswer>> {
        Ok(self.decide_with_shape(state, questions)?.answers)
    }

    /// [`Self::decide`], plus the shape of the batch it ran — what a
    /// record needs to price a request in tokens rather than in
    /// questions.
    /// How many requests this seat has panicked on since it was loaded.
    ///
    /// Non-zero is permanent and deliberate: any record taken from this
    /// process afterwards carries the fact that it faulted.
    pub fn panics(&self) -> u64 {
        self.panics.load(AtomicOrdering::Relaxed)
    }

    pub fn decide_with_shape(
        &self,
        state: &str,
        questions: &[DecisionQuestion],
    ) -> Result<Decided> {
        if questions.is_empty() {
            return Ok(Decided {
                answers: Vec::new(),
                batch: BatchShape {
                    rows: 0,
                    len: 0,
                    markers: 0,
                },
            });
        }
        let bad = |what: String| EngineError::Model(combs_models::ModelError::Unsupported(what));

        let mut sequences: Vec<Sequence> = Vec::with_capacity(questions.len());
        let mut rendered: Vec<Vec<String>> = Vec::with_capacity(questions.len());
        let tok = self.tokenizer.view();
        for q in questions {
            let options = render_options(&q.criteria);
            let seq = build_sequence(
                &tok,
                &self.specials,
                q.criteria.kind(),
                &q.instructions,
                &options,
                state,
                self.limits,
            );
            // The one thing truncation cannot recover from: an option
            // whose marker fell past `max_len` has no position to score,
            // so the question is refused BY NAME rather than answered on
            // fewer options than the caller asked about.
            if seq.dropped() > 0 {
                return Err(bad(format!(
                    "question {:?}: {} of its {} options were pushed past max_len {} — the state \
                     and the options together do not fit, and an answer over the rest would be an \
                     answer to a different question",
                    q.name,
                    seq.dropped(),
                    seq.options,
                    self.limits.max_len
                )));
            }
            sequences.push(seq);
            rendered.push(options);
        }
        // A text the tokenizer could not read was tokenised as nothing.
        // An answer over the markers alone would be an answer to a
        // different question, so the request is refused, naming the text.
        if let Some(why) = tok.failure() {
            return Err(bad(format!(
                "a text in this request could not be tokenised, and an answer on nothing would \
                 be an answer to a different question: {why}"
            )));
        }

        let batch: DecisionBatch = collate(&sequences, self.specials.pad)
            .ok_or_else(|| bad("nothing to ask".to_string()))?;
        let shape = BatchShape {
            rows: batch.rows,
            len: batch.len,
            markers: batch.markers,
        };
        let out = guarded_decide(&self.model, &self.panics, &batch)?;
        let answers = read_answers(questions, &rendered, &sequences, &out, &self.calibration)?;
        Ok(Decided {
            answers,
            batch: shape,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model that faults on its first batch and answers every one
    /// after. Its `metadata` is never reached by `guarded_decide`, so it
    /// is left unimplemented rather than fabricated.
    struct PanicsOnce {
        calls: AtomicU64,
    }

    impl TypedDecisionModel for PanicsOnce {
        fn metadata(&self) -> &ModelMetadata {
            unimplemented!("guarded_decide never asks a model for its metadata")
        }

        fn decide(
            &self,
            _batch: &DecisionBatch,
        ) -> std::result::Result<DecisionLogits, combs_models::ModelError> {
            if self.calls.fetch_add(1, AtomicOrdering::Relaxed) == 0 {
                panic!("an index this batch tripped");
            }
            Ok(DecisionLogits {
                logits: vec![vec![0.5]],
                act: vec![vec![1.0]],
            })
        }
    }

    fn empty_batch() -> DecisionBatch {
        DecisionBatch {
            rows: 1,
            len: 1,
            markers: 1,
            input_ids: vec![0],
            attention: vec![1],
            marker_pos: vec![0],
            marker_mask: vec![true],
            qtype: vec![0],
        }
    }

    /// A head that returns the wrong shape must refuse BY NAME, not
    /// panic inside the seat. `guarded_decide` catches the forward pass
    /// only, so a bug in our own reading of the output used to kill the
    /// process — and the permanent fault count that exists to carry a
    /// fault forward never recorded it, because the recorder died too.
    #[test]
    fn a_ragged_answer_is_refused_by_name_and_never_indexed() {
        let q = DecisionQuestion {
            name: "intent".to_string(),
            instructions: String::new(),
            criteria: Criteria::Choice(vec![
                ("a".to_string(), serde_json::json!("the first")),
                ("b".to_string(), serde_json::json!("the second")),
            ]),
        };
        let rendered = vec![vec![
            "a: the first".to_string(),
            "b: the second".to_string(),
        ]];
        // Built by hand rather than tokenized: `read_answers` reads a
        // sequence only for its `usage()`, and this test is about the
        // shape the HEAD returned.
        let sequences = vec![Sequence {
            ids: vec![1, 3, 4, 3, 5, 2],
            markers: vec![1, 3],
            options: 2,
            kind: Kind::Choice,
            instruction_tokens_cut: 0,
            option_tokens_cut: 0,
            state_tokens_cut: 0,
        }];
        let cal = Calibration::neutral();
        let qs = vec![q];

        // One logit for two markers: the old code sliced `[..2]` and
        // panicked where nobody was catching.
        let ragged = DecisionLogits {
            logits: vec![vec![0.5]],
            act: vec![vec![1.0]],
        };
        let why = read_answers(&qs, &rendered, &sequences, &ragged, &cal)
            .expect_err("one logit for two markers")
            .to_string();
        assert!(why.contains("intent"), "{why}");
        assert!(why.contains("1 logits for 2 markers"), "{why}");

        // No row at all for the question that was asked.
        let empty = DecisionLogits {
            logits: vec![],
            act: vec![],
        };
        let why = read_answers(&qs, &rendered, &sequences, &empty, &cal)
            .expect_err("no rows")
            .to_string();
        assert!(why.contains("0 rows for 1 questions"), "{why}");

        // The act head short while the scorer is sound — still refused,
        // rather than answered with a p_act nobody computed.
        let no_act = DecisionLogits {
            logits: vec![vec![0.5, 0.5]],
            act: vec![],
        };
        let why = read_answers(&qs, &rendered, &sequences, &no_act, &cal)
            .expect_err("no act rows")
            .to_string();
        assert!(why.contains("act head"), "{why}");

        // And a sound shape still reads.
        let good = DecisionLogits {
            logits: vec![vec![1.0, 0.0]],
            act: vec![vec![1.0]],
        };
        let answers = read_answers(&qs, &rendered, &sequences, &good, &cal).expect("a sound head");
        assert_eq!(answers.len(), 1);
        assert_eq!(answers[0].distribution.len(), 2);
    }

    /// The seat survives its own fault, keeps answering, and CARRIES THE
    /// FACT. Recovering silently would be the wrong half: a record must
    /// never be drawn from a process that faulted without showing it.
    #[test]
    fn a_panic_refuses_its_own_request_and_is_carried_for_the_seats_life() {
        let model: Mutex<Box<dyn TypedDecisionModel>> = Mutex::new(Box::new(PanicsOnce {
            calls: AtomicU64::new(0),
        }));
        let panics = AtomicU64::new(0);

        // The faulting request refuses ITSELF...
        let first = guarded_decide(&model, &panics, &empty_batch());
        let why = first.expect_err("the batch panicked").to_string();
        assert!(why.contains("panicked"), "{why}");
        // ...and the refusal says where the fact now lives, so whoever
        // reads it knows the seat is marked and not merely annoyed.
        assert!(why.contains("panics"), "{why}");
        assert_eq!(panics.load(AtomicOrdering::Relaxed), 1);

        // The lock was poisoned by that unwind. The seat answers anyway:
        // `decide` takes `&self`, so the model behind a poisoned lock is
        // exactly the model that was loaded.
        let second = guarded_decide(&model, &panics, &empty_batch());
        assert!(second.is_ok(), "a poisoned lock ended a sound seat");

        // And the count does not reset when the seat recovers.
        assert_eq!(panics.load(AtomicOrdering::Relaxed), 1);
    }

    fn specials() -> Specials {
        Specials {
            cls: 1,
            sep: 2,
            mask: 3,
            pad: 0,
            mask_text: "[MASK]".to_string(),
        }
    }

    /// A word-level tokenizer over exactly `words`, with NO unknown token:
    /// any other word is an error, which is the case a byte-level
    /// tokenizer never produces and this seat had never been asked about.
    fn strict_tokenizer(words: &[&str]) -> Tokenizer {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
        let vocab = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), i as u32))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .build()
            .expect("a word-level model");
        let mut t = Tokenizer::new(model);
        t.with_pre_tokenizer(Some(WhitespaceSplit));
        t
    }

    /// A text the tokenizer cannot read used to come back as NO ids, and
    /// the question was answered on the markers alone.
    #[test]
    fn an_unreadable_text_is_recorded_not_read_as_empty() {
        let t = strict_tokenizer(&["hello", "there"]);
        let tok = Tok(t);
        let view = tok.view();
        assert_eq!(view.ids("hello there"), vec![0, 1]);
        assert_eq!(view.failure(), None, "a readable text records nothing");
        assert_eq!(view.ids("goodbye"), Vec::<u32>::new());
        let why = view.failure().expect("the failure is recorded");
        assert!(why.contains("goodbye"), "it names the text: {why}");
        assert_eq!(view.failure(), None, "and it is taken once");
        // The first failure is the one kept.
        view.ids("first");
        view.ids("second");
        assert!(view.failure().unwrap().contains("first"));
    }

    /// A model that answers every batch with a flat row per question.
    struct Flat;

    impl TypedDecisionModel for Flat {
        fn metadata(&self) -> &ModelMetadata {
            unimplemented!("the seat never asks a model for its metadata")
        }

        fn decide(
            &self,
            batch: &DecisionBatch,
        ) -> std::result::Result<DecisionLogits, combs_models::ModelError> {
            Ok(DecisionLogits {
                logits: vec![vec![0.0; batch.markers]; batch.rows],
                act: vec![vec![1.0]; batch.rows],
            })
        }
    }

    /// The seat refuses a request whose state the tokenizer cannot read,
    /// naming the text, instead of answering on nothing; the same words
    /// it CAN read still answer, and the refusal is not a fault.
    #[test]
    fn a_request_whose_state_cannot_be_read_is_refused_by_name() {
        let words = [
            "choice",
            "question:",
            "a:",
            "the",
            "first",
            "b:",
            "second",
            "hello",
        ];
        let engine = DecisionEngine::for_tests(Box::new(Flat), strict_tokenizer(&words));
        let q = DecisionQuestion {
            name: "intent".to_string(),
            instructions: String::new(),
            criteria: Criteria::Choice(vec![
                ("a".to_string(), serde_json::json!("the first")),
                ("b".to_string(), serde_json::json!("the second")),
            ]),
        };
        let ok = engine
            .decide("hello", &[q.clone()])
            .expect("a readable state answers");
        assert_eq!(ok.len(), 1);
        assert_eq!(ok[0].distribution.len(), 2);

        let why = engine
            .decide("goodbye", &[q])
            .expect_err("an unreadable state is refused")
            .to_string();
        assert!(why.contains("could not be tokenised"), "{why}");
        assert!(why.contains("goodbye"), "it names the text: {why}");
        assert_eq!(engine.panics(), 0, "a refusal is not a fault");
    }

    #[test]
    fn a_warm_batch_is_a_legal_rectangle() {
        let b = warm_batch(2, 8, 2, &specials()).expect("it fits");
        assert_eq!((b.rows, b.len, b.markers), (2, 8, 2));
        assert_eq!(b.input_ids.len(), 16);
        assert!(
            b.attention.iter().all(|&a| a == 1),
            "no padding in a warm row"
        );
        assert!(b.marker_mask.iter().all(|&m| m), "every marker is real");
        for r in 0..2 {
            assert_eq!(b.input_ids[r * 8], 1, "a row starts with [CLS]");
            for j in 0..2 {
                let pos = b.marker_pos[r * 2 + j] as usize;
                assert!(pos < 8);
                assert_eq!(b.input_ids[r * 8 + pos], 3, "a marker sits on a [MASK]");
            }
        }
        assert_eq!(b.qtype, vec![0, 0]);
        assert!(
            warm_batch(1, 3, 2, &specials()).is_none(),
            "three tokens cannot hold two markers"
        );
        assert!(warm_batch(0, 8, 2, &specials()).is_none());
    }

    #[test]
    fn default_warm_shapes_never_exceed_max_len_and_never_repeat() {
        let shapes = default_warm_shapes(Limits {
            max_len: 300,
            head_max_len: 192,
            cut: Cut::Right,
        });
        assert!(shapes.iter().all(|&(_, len, _)| len <= 300), "{shapes:?}");
        let mut seen = std::collections::HashSet::new();
        assert!(shapes.iter().all(|s| seen.insert(*s)), "{shapes:?}");
        assert_eq!(shapes[0], (1, 64, 2));
    }

    /// The warm set used to be four rectangles at two markers. The head
    /// is sized by the MARKERS, so a seven-option question met an
    /// untuned kernel and paid ~5 s for it (the ae86e63 smoke), and so
    /// did a one-question body below the smallest rectangle. The set now
    /// covers the smallest row a request can make, a real marker width,
    /// and the widest head the wire admits — and every shape in it can
    /// hold its own markers, or `warm` would refuse at start-up.
    #[test]
    fn default_warm_shapes_cover_the_widest_head_and_the_smallest_row() {
        let limits = Limits {
            max_len: 1024,
            head_max_len: 256,
            cut: Cut::Right,
        };
        let shapes = default_warm_shapes(limits);
        assert!(shapes.contains(&(1, 64, 2)), "the smallest row: {shapes:?}");
        assert!(
            shapes
                .iter()
                .any(|&(rows, _, markers)| rows == bars::MAX_QUESTIONS
                    && markers == bars::MAX_OPTIONS),
            "the widest head the wire admits: {shapes:?}"
        );
        assert!(
            shapes.iter().any(|&(_, _, markers)| markers > 2),
            "a real marker width beside the suites' rectangle: {shapes:?}"
        );
        let specials = specials();
        for &(rows, len, markers) in &shapes {
            assert!(len <= limits.max_len, "{rows}x{len} past max_len");
            assert!(
                warm_batch(rows, len, markers, &specials).is_some(),
                "{rows}x{len} at {markers} markers cannot hold them: warm would refuse at start-up"
            );
        }
        let mut seen = std::collections::HashSet::new();
        assert!(shapes.iter().all(|s| seen.insert(*s)), "{shapes:?}");
        // The widest head rides the deepest pass's own rectangle rather
        // than adding a 16,384-token pass to every start.
        let widest = shapes
            .iter()
            .find(|&&(rows, _, _)| rows == bars::MAX_QUESTIONS)
            .copied()
            .expect("the widest head");
        assert!(
            widest.0 * widest.1 <= 4 * 1024,
            "the widest pass costs {} tokens: every seat start pays it",
            widest.0 * widest.1
        );
    }
}
