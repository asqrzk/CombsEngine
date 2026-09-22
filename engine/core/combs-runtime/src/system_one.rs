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

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use combs_classify::sequence::{
    build_sequence, collate, DecisionBatch, Encode, Limits, QuestionUsage, Sequence, Specials,
};
use combs_classify::system_one::{
    engine_order, readout, render_options, AgentConfig, Calibration, Criteria,
};
use combs_classify::Kind;
use combs_core::CombsBackend;
use combs_formats::{ModelMetadata, SafetensorsSource};
use combs_models::{LayaModel, TypedDecisionModel};
use tokenizers::Tokenizer;

use crate::EngineError;

type Result<T> = std::result::Result<T, EngineError>;

/// The tokenizer, behind the seam `combs-classify` defines so that it
/// links none.
struct Tok(Tokenizer);

impl Encode for Tok {
    fn ids(&self, text: &str) -> Vec<u32> {
        // `false`: this file splices every special by NUMBER, so the
        // tokenizer must not add any of its own.
        match self.0.encode(text, false) {
            Ok(enc) => enc.get_ids().to_vec(),
            // A text the tokenizer cannot read is empty rather than
            // fatal: the question still has its markers, and an answer
            // on less text is what every other budget here produces too.
            Err(_) => Vec::new(),
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

/// A seated typed-decision model.
pub struct DecisionEngine {
    // One at a time: the model holds device buffers sized for one batch.
    model: Mutex<Box<dyn TypedDecisionModel>>,
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
            tokenizer: Tok(tokenizer),
            calibration: agent.calibration,
            limits: Limits {
                max_len: agent.max_len,
                head_max_len: agent.head_max_len,
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

    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
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
        for q in questions {
            let options = render_options(&q.criteria);
            let seq = build_sequence(
                &self.tokenizer,
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

        let batch: DecisionBatch = collate(&sequences, self.specials.pad)
            .ok_or_else(|| bad("nothing to ask".to_string()))?;
        let shape = BatchShape {
            rows: batch.rows,
            len: batch.len,
            markers: batch.markers,
        };
        let out = {
            let model = self
                .model
                .lock()
                .map_err(|_| bad("the seat is poisoned".to_string()))?;
            model.decide(&batch).map_err(EngineError::Model)?
        };

        let mut answers = Vec::with_capacity(questions.len());
        for (i, q) in questions.iter().enumerate() {
            let kind = q.criteria.kind();
            let k = rendered[i].len();
            let logits = &out.logits[i][..k];
            let t = self.calibration.temperature(kind, k);
            let p = readout(logits, t);
            answers.push(DecisionAnswer {
                name: q.name.clone(),
                kind,
                distribution: engine_order(kind, &p),
                logits: engine_order(kind, logits),
                temperature: t,
                bucket: self.calibration.bucket(kind, k),
                // Index zero is answering; the rest are the acts the
                // config prices.
                p_act: out.act[i].first().copied().unwrap_or(0.0),
                usage: sequences[i].usage(),
            });
        }
        Ok(Decided {
            answers,
            batch: shape,
        })
    }
}
