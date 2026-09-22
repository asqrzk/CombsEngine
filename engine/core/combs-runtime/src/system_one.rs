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
use std::time::Instant;

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

/// One warm-up pass and what it cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarmPass {
    pub rows: usize,
    pub len: usize,
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

/// The shapes a seat warms by default: one and two questions at the
/// lengths the seed suites take (a two-question request is two rows of
/// ~150–200 tokens), then the larger rectangles a longer state or more
/// questions would make — each clamped to the checkpoint's `max_len`.
pub fn default_warm_shapes(limits: Limits) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (rows, len) in [(1, 256), (2, 256), (2, 512), (4, 1024)] {
        let len = len.min(limits.max_len);
        if !out.contains(&(rows, len)) {
            out.push((rows, len));
        }
    }
    out
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

    /// The two length budgets this seat answers within.
    pub fn limits(&self) -> Limits {
        self.limits
    }

    /// Runs the model over `shapes` before any request arrives, so the
    /// first real one pays neither autotune nor a first-launch compile.
    /// The seat's first request measured 1,058 ms cold (s138); a router
    /// with a 1,500 ms budget should never be the one to pay it. Every
    /// pass syncs on its readback, so a launch error surfaces HERE
    /// rather than in a caller's answer.
    pub fn warm(&self, shapes: &[(usize, usize)]) -> Result<WarmReport> {
        let bad = |what: String| EngineError::Model(combs_models::ModelError::Unsupported(what));
        let model = self
            .model
            .lock()
            .map_err(|_| bad("the seat is poisoned".to_string()))?;
        let t0 = Instant::now();
        let mut passes = Vec::with_capacity(shapes.len());
        for &(rows, len) in shapes {
            let batch = warm_batch(rows, len, 2, &self.specials).ok_or_else(|| {
                bad(format!(
                    "a warm-up shape of {rows}x{len} cannot hold its own markers"
                ))
            })?;
            let t = Instant::now();
            model.decide(&batch).map_err(EngineError::Model)?;
            passes.push(WarmPass {
                rows,
                len,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn specials() -> Specials {
        Specials {
            cls: 1,
            sep: 2,
            mask: 3,
            pad: 0,
            mask_text: "[MASK]".to_string(),
        }
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
        });
        assert!(shapes.iter().all(|&(_, len)| len <= 300), "{shapes:?}");
        let mut seen = std::collections::HashSet::new();
        assert!(shapes.iter().all(|s| seen.insert(*s)), "{shapes:?}");
        assert_eq!(shapes[0], (1, 256));
    }
}
