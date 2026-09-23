//! `POST /v1/classify` — typed decisions on the seated model, gated.
//!
//! The rung this serves needs NO new weights: the state and each
//! option's tether are embedded by the model already loaded, and the
//! answer is a softmax over their cosine similarities. That is the
//! cheapest honest classifier this estate can offer, and it is honest
//! about what it is — `calibrated: false` until a temperature is
//! fitted, and a `backend` field naming the road it took.
//!
//! What never happens here: an option's IDENTIFIER is never embedded.
//! The scorer reads [`combs_classify::Opt::scored_text`], which is the
//! tether's meaning and its near-miss anchors, so a state that merely
//! MENTIONS a topic can land on an anchor instead of the action. The
//! identifier comes back in the answer as a routing key and nothing
//! else.
//!
//! The option vectors are cached by content, above the engine's
//! single-flight worker: a question asked twice embeds its options
//! once, and only the state is paid for on the second call.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use combs_classify::{
    check_distribution, confidence, decide, expected_level, softmax_scaled, tether, Decision,
    Implication, Kind, Opt, Thresholds,
};
use combs_runtime::{EmbedOptions, Engine};

use crate::http::{error_json, json_response, HttpResponse};

/// Cosine similarities sit in a narrow band (roughly 0.3..0.8 for a
/// sentence encoder), so a softmax straight over them is nearly
/// uniform. This divides them into a usable range. It is a SHAPE
/// parameter, not a calibration: `calibrated` stays false until a
/// temperature is fitted against labels.
const DEFAULT_TEMPERATURE: f32 = 0.05;

/// Options a single question may carry here. The constrained decoder's
/// own enum cap, so a question this road accepts is one the rung above
/// it could also serve.
const MAX_OPTIONS: usize = combs_classify::bars::MAX_OPTIONS;

/// Texts embedded in one call. The engine's embeddings road takes at
/// most this many, and a request past it is refused by name rather than
/// silently truncated.
const MAX_TEXTS: usize = 64;

/// Questions in one request, on either road. The System 1 seat runs one
/// row per question in a single pass and nothing else bounded it.
const MAX_QUESTIONS: usize = combs_classify::bars::MAX_QUESTIONS;

/// One question as the wire declares it.
pub(crate) struct Question {
    pub(crate) name: String,
    pub(crate) kind: Kind,
    pub(crate) instructions: String,
    /// `choice` and `score` carry options; a `noul` is folded into a
    /// two-option choice whose first option is the proposition.
    pub(crate) options: Vec<Opt>,
}

pub(crate) struct Request {
    pub(crate) state: String,
    pub(crate) questions: Vec<Question>,
    /// The caller's own temperature, when they sent one.
    ///
    /// `None` is "the caller said nothing", which the cosine road reads
    /// as [`DEFAULT_TEMPERATURE`] and the marker road refuses to guess
    /// at: that seat's temperature is the checkpoint's fitted
    /// calibration, so a caller's number could not be applied without
    /// making `calibrated: {source: "checkpoint"}` a false claim. The
    /// two roads cannot share a default they disagree about.
    pub(crate) temperature: Option<f32>,
    pub(crate) thresholds: Thresholds,
}

/// Which scorer answers `/v1/classify` on this build. One constant, so
/// the answer's own `backend` field and the stats route cannot drift
/// apart and describe two different roads. `embed-cosine` is the
/// bi-encoder the ledger's §126 refused by its own suite.
pub const BACKEND: &str = "embed-cosine";

/// What the classify road has cost since load. Read by `/v1/stats`, so
/// a sitting can say whether the classifier ran, how much of it the
/// cache answered, and which scorer answered it — questions §126 could
/// only answer by reading a console.
#[derive(Debug, Clone, Default)]
pub struct ClassifyStats {
    pub requests: u64,
    pub questions: u64,
    pub cache_entries: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_clears: u64,
}

/// The option-vector cache: content-keyed, so the same question asked
/// again costs one embedding of the state and nothing else.
#[derive(Default)]
pub struct Classifier {
    options: Mutex<HashMap<u64, Arc<Vec<Vec<f32>>>>>,
    counts: Mutex<ClassifyStats>,
}

impl Classifier {
    pub fn new() -> Self {
        Self::default()
    }
    fn get(&self, key: u64) -> Option<Arc<Vec<Vec<f32>>>> {
        let hit = self.options.lock().ok()?.get(&key).cloned();
        if let Ok(mut c) = self.counts.lock() {
            if hit.is_some() {
                c.cache_hits += 1;
            } else {
                c.cache_misses += 1;
            }
        }
        hit
    }
    fn put(&self, key: u64, vectors: Arc<Vec<Vec<f32>>>) {
        if let Ok(mut m) = self.options.lock() {
            // A bound, so a long-running server cannot grow a cache
            // without end; the next question re-embeds, which costs one
            // call and never a wrong answer.
            if m.len() >= 256 {
                m.clear();
                if let Ok(mut c) = self.counts.lock() {
                    c.cache_clears += 1;
                }
            }
            m.insert(key, vectors);
            if let Ok(mut c) = self.counts.lock() {
                c.cache_entries = m.len();
            }
        }
    }
    /// Counted where the request is parsed, so a refused body is not
    /// counted as a classification.
    fn counted(&self, questions: usize) {
        if let Ok(mut c) = self.counts.lock() {
            c.requests += 1;
            c.questions += questions as u64;
        }
    }
    /// A copy for the stats route; the lock is held for microseconds.
    pub fn stats(&self) -> ClassifyStats {
        self.counts.lock().map(|c| c.clone()).unwrap_or_default()
    }
}

/// FNV-1a over bytes. No new dependency, and the only hash this file
/// needs: one for the cache key, one for naming a state in the log
/// without writing the state down.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// FNV-1a over the strings that decide a vector. A collision costs a
/// wrong cache hit, so the key covers every input the embedding depends
/// on.
fn key_of(kind: Kind, instructions: &str, texts: &[String]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: &[u8]| {
        for &byte in b {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(match kind {
        Kind::Choice => b"choice",
        Kind::Score => b"score",
        Kind::Noul => b"noul",
    });
    eat(b"\x1f");
    eat(instructions.as_bytes());
    for t in texts {
        eat(b"\x1f");
        eat(t.as_bytes());
    }
    h
}

pub(crate) fn bad(msg: &str) -> (u16, Value) {
    (400, error_json("invalid_request", msg))
}

fn parse_kind(s: &str) -> Option<Kind> {
    match s {
        "choice" => Some(Kind::Choice),
        "score" => Some(Kind::Score),
        "noul" => Some(Kind::Noul),
        _ => None,
    }
}

fn parse_opt(v: &Value, question: &str, i: usize) -> Result<Opt, (u16, Value)> {
    serde_json::from_value::<Opt>(v.clone()).map_err(|e| {
        bad(&format!(
            "question {question:?} option {i}: {e} — an option is {{\"id\", \"tether\": {{\"means\", \"not\", \"requires\"}}}}"
        ))
    })
}

/// Read a request, refusing by name before anything is embedded.
pub(crate) fn parse_request(body: &str) -> Result<Request, (u16, Value)> {
    let v: Value = serde_json::from_str(body).map_err(|e| bad(&format!("bad JSON: {e}")))?;

    let state = match v.get("state") {
        Some(Value::String(s)) if !s.trim().is_empty() => s.clone(),
        Some(Value::Object(o)) if o.contains_key("image") || o.contains_key("image_url") => {
            return Err(bad(
                "this seat classifies text only: an image state needs an image encoder",
            ));
        }
        _ => return Err(bad("`state` must be a non-empty string")),
    };

    let Some(Value::Array(qs)) = v.get("questions") else {
        return Err(bad("`questions` must be an array of at least one question"));
    };
    if qs.is_empty() {
        return Err(bad("`questions` must be an array of at least one question"));
    }
    if qs.len() > MAX_QUESTIONS {
        return Err((
            400,
            error_json(
                "budget",
                &format!(
                    "this request asks {} questions and the road takes {MAX_QUESTIONS}: ask fewer at once",
                    qs.len()
                ),
            ),
        ));
    }

    let declared: Vec<String> = qs
        .iter()
        .filter_map(|q| q.get("name").and_then(Value::as_str).map(str::to_string))
        .collect();

    let mut questions = Vec::with_capacity(qs.len());
    for q in qs {
        let name = q
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| bad("every question needs a `name`"))?
            .to_string();
        let kind = q
            .get("type")
            .and_then(Value::as_str)
            .and_then(parse_kind)
            .ok_or_else(|| {
                bad(&format!(
                    "question {name:?}: `type` must be choice, score or noul"
                ))
            })?;
        let instructions = q
            .get("instructions")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();

        let options: Vec<Opt> = match kind {
            Kind::Noul => {
                // A noul is one proposition. Scoring it by similarity
                // needs something to be similar AGAINST, so its tether
                // must carry at least one contrary anchor; inventing a
                // negation here would be the gate guessing.
                let t: tether::Tether = q
                    .get("tether")
                    .ok_or_else(|| bad(&format!("question {name:?}: a noul needs a `tether`")))
                    .and_then(|t| {
                        serde_json::from_value(t.clone())
                            .map_err(|e| bad(&format!("question {name:?} tether: {e}")))
                    })?;
                if t.not.is_empty() {
                    return Err(bad(&format!(
                        "question {name:?}: a noul needs at least one contrary in `tether.not` to score against"
                    )));
                }
                let contrary = t.not.join("; ");
                vec![
                    Opt {
                        id: "true".to_string(),
                        tether: tether::Tether {
                            means: t.means.clone(),
                            not: Vec::new(),
                            requires: Vec::new(),
                            // laya renders a noul's sides as `true:`/
                            // `false:` itself; a label here would be a
                            // third word for the same slot.
                            label: None,
                        },
                    },
                    Opt {
                        id: "false".to_string(),
                        tether: tether::Tether {
                            means: contrary,
                            not: Vec::new(),
                            requires: Vec::new(),
                            label: None,
                        },
                    },
                ]
            }
            _ => {
                let arr = q
                    .get("options")
                    .or_else(|| q.get("levels"))
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        bad(&format!("question {name:?}: `options` must be an array"))
                    })?;
                let mut out = Vec::with_capacity(arr.len());
                for (i, o) in arr.iter().enumerate() {
                    out.push(parse_opt(o, &name, i)?);
                }
                out
            }
        };

        // The tether's own refusals: a label with no meaning, two
        // options meaning the same thing, an implication naming a
        // question nobody asked, and the option cap.
        tether::validate(&options, &declared, MAX_OPTIONS)
            .map_err(|e| bad(&format!("question {name:?}: {e}")))?;

        questions.push(Question {
            name,
            kind,
            instructions,
            options,
        });
    }

    let temperature = match v.get("temperature") {
        None | Some(Value::Null) => None,
        Some(t) => {
            let t = t
                .as_f64()
                .map(|t| t as f32)
                .ok_or_else(|| bad("`temperature` must be a positive number"))?;
            if !(t.is_finite() && t > 0.0) {
                return Err(bad("`temperature` must be a positive number"));
            }
            Some(t)
        }
    };

    let mut thresholds = Thresholds::default();
    if let Some(t) = v.get("thresholds") {
        if let Some(c) = t.get("confidence").and_then(Value::as_f64) {
            thresholds.confidence = c as f32;
        }
        if let Some(tau) = t.get("tau").and_then(Value::as_f64) {
            thresholds.tau = tau as f32;
        }
    }

    Ok(Request {
        state,
        questions,
        temperature,
        thresholds,
    })
}

/// The embeddings this model returns are L2-normalised, so the cosine
/// is the dot product. Said here because it is load-bearing: if that
/// ever stops being true, this is wrong rather than merely imprecise.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// `Instruct: …\nQuery: …` — the query-side convention the cached
/// embedding checkpoint was trained with. Options are embedded plainly:
/// they are documents, not queries.
fn state_text(instructions: &str, state: &str) -> String {
    if instructions.trim().is_empty() {
        state.to_string()
    } else {
        format!("Instruct: {}\nQuery: {}", instructions.trim(), state)
    }
}

/// Answer one classify request.
pub fn handle_classify(
    engine: &Arc<Engine>,
    model_id: &str,
    body: &str,
    classifier: &Arc<Classifier>,
) -> HttpResponse {
    if !engine.supports_embeddings() {
        return json_response(
            400,
            error_json(
                "invalid_request",
                "this model does not expose hidden states, so it cannot classify on this road",
            ),
        );
    }
    let req = match parse_request(body) {
        Ok(r) => r,
        Err((code, v)) => return json_response(code, v),
    };
    // Counted here and not at the route: a body the parser refused was
    // never a classification, and counting it would make a suite of
    // malformed requests look like a suite of answered ones.
    classifier.counted(req.questions.len());
    // The state itself NEVER reaches the log — it is the user's text. Its
    // length and a content hash are enough to tell two runs apart, to
    // find the call a record's figure came from, and to say nothing
    // about what a person wrote.
    let turn = combs_core::provenance::turn(
        "classify",
        "classify",
        &[
            ("backend", BACKEND.to_string()),
            ("questions", req.questions.len().to_string()),
            ("state_len", req.state.len().to_string()),
            ("state_fnv", format!("{:016x}", fnv1a(req.state.as_bytes()))),
            (
                "temperature",
                format!("{:.4}", req.temperature.unwrap_or(DEFAULT_TEMPERATURE)),
            ),
        ],
    );

    // One embedding of the state per distinct instruction, and one set
    // of option vectors per question (cached). Everything that must be
    // embedded is gathered first so the engine is asked once.
    let mut want: Vec<String> = Vec::new();
    let mut state_at: Vec<usize> = Vec::with_capacity(req.questions.len());
    let mut option_key: Vec<u64> = Vec::with_capacity(req.questions.len());
    let mut option_at: Vec<Option<(usize, usize)>> = Vec::with_capacity(req.questions.len());

    for q in &req.questions {
        let st = state_text(&q.instructions, &req.state);
        let at = match want.iter().position(|w| *w == st) {
            Some(i) => i,
            None => {
                want.push(st);
                want.len() - 1
            }
        };
        state_at.push(at);

        let texts: Vec<String> = q.options.iter().map(|o| o.scored_text()).collect();
        let key = key_of(q.kind, &q.instructions, &texts);
        option_key.push(key);
        if classifier.get(key).is_some() {
            option_at.push(None);
        } else {
            let start = want.len();
            want.extend(texts);
            option_at.push(Some((start, q.options.len())));
        }
    }

    if want.len() > MAX_TEXTS {
        return json_response(
            400,
            error_json(
                "budget",
                &format!(
                    "this request needs {} embeddings and the road takes {MAX_TEXTS}: ask fewer questions, or fewer options at once",
                    want.len()
                ),
            ),
        );
    }

    let out = match engine.embed_texts(&want, &EmbedOptions::default()) {
        Ok(o) => o,
        Err(e) => {
            turn.failed(&e.to_string());
            return json_response(500, error_json("engine_error", &e.to_string()));
        }
    };
    if out.vectors.len() != want.len() {
        let why = "the embeddings road returned a different count than it was asked for";
        turn.failed(why);
        return json_response(500, error_json("engine_error", why));
    }

    // Distributions first, so an implication can read a question that
    // comes later in the request than the option requiring it.
    let mut dist: Vec<Vec<f32>> = Vec::with_capacity(req.questions.len());
    // The scores BEFORE the temperature. A distribution alone cannot be
    // re-derived or re-calibrated after the fact: two different score
    // vectors can soften to the same probabilities, so a harness given
    // only `distribution` cannot check the softmax it was told was used.
    let mut raw: Vec<Vec<f32>> = Vec::with_capacity(req.questions.len());
    for (qi, q) in req.questions.iter().enumerate() {
        let vectors: Arc<Vec<Vec<f32>>> = match option_at[qi] {
            Some((start, n)) => {
                let v = Arc::new(out.vectors[start..start + n].to_vec());
                classifier.put(option_key[qi], v.clone());
                v
            }
            None => match classifier.get(option_key[qi]) {
                Some(v) => v,
                // A cache swept between the look and the read: embed
                // nothing again, refuse rather than answer from a guess.
                None => {
                    let why = "the option cache moved under this request; try again";
                    turn.failed(why);
                    return json_response(503, error_json("engine_error", why));
                }
            },
        };
        let state_vec = &out.vectors[state_at[qi]];
        let sims: Vec<f32> = vectors.iter().map(|v| cosine(state_vec, v)).collect();
        let p = softmax_scaled(&sims, req.temperature.unwrap_or(DEFAULT_TEMPERATURE));
        if let Err(e) = check_distribution(&p) {
            let why = format!("question {:?}: {e}", q.name);
            turn.failed(&why);
            return json_response(500, error_json("engine_error", &why));
        }
        dist.push(p);
        raw.push(sims);
    }

    // Each question's probability of its FIRST option, which for a noul
    // is the proposition itself — what an implication reads.
    let p_true: HashMap<&str, f32> = req
        .questions
        .iter()
        .enumerate()
        .map(|(i, q)| (q.name.as_str(), dist[i].first().copied().unwrap_or(0.0)))
        .collect();

    let mut answers = Vec::with_capacity(req.questions.len());
    for (qi, q) in req.questions.iter().enumerate() {
        let p = &dist[qi];
        let (top_i, top_p) =
            p.iter().enumerate().fold(
                (0usize, f32::MIN),
                |b, (i, &v)| if v > b.1 { (i, v) } else { b },
            );
        let conf = confidence(p);
        let mut sorted: Vec<f32> = p.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let margin = sorted.first().copied().unwrap_or(0.0) - sorted.get(1).copied().unwrap_or(0.0);

        let implications: Vec<Implication> = q.options[top_i]
            .tether
            .requires
            .iter()
            .map(|r| Implication {
                requires: r.clone(),
                p: p_true.get(r.as_str()).copied().unwrap_or(0.0),
                tau: req.thresholds.tau,
            })
            .collect();

        let decision = decide(conf, req.thresholds.confidence, &implications);

        let distribution: serde_json::Map<String, Value> = q
            .options
            .iter()
            .zip(p.iter())
            .map(|(o, &v)| (o.id.clone(), json!(v)))
            .collect();
        // The raw score each probability came from, by the same key. On
        // this road a cosine; on a trained scorer, its head's logit.
        // Named for what it is so nobody reads it as a probability.
        let similarities: serde_json::Map<String, Value> = q
            .options
            .iter()
            .zip(raw[qi].iter())
            .map(|(o, &v)| (o.id.clone(), json!(v)))
            .collect();

        let mut a = json!({
            "name": q.name,
            "type": match q.kind { Kind::Choice => "choice", Kind::Score => "score", Kind::Noul => "noul" },
            "top": q.options[top_i].id,
            "p_top": top_p,
            "distribution": Value::Object(distribution),
            "similarities": Value::Object(similarities),
            "confidence": conf,
            "margin": margin,
            "calibrated": false,
            "temperature": req.temperature.unwrap_or(DEFAULT_TEMPERATURE),
            "decision": decision.word(),
            "reason": match &decision { Decision::Act => Value::Null, d => json!(d.reason()) },
            "implications": implications.iter().map(|i| json!({
                "requires": i.requires, "p": i.p, "tau": i.tau, "holds": i.holds(), "gated": i.gated()
            })).collect::<Vec<_>>(),
        });
        if q.kind == Kind::Score {
            a["expected_level"] = json!(expected_level(p));
        }
        // At Debug, the numbers behind one answer — raw scores included,
        // so a suspect verdict can be re-derived from the log alone.
        combs_core::provenance::event(
            "classify",
            "classify.question",
            &[
                ("name", q.name.clone()),
                ("top", q.options[top_i].id.clone()),
                ("p_top", format!("{top_p:.6}")),
                ("confidence", format!("{conf:.6}")),
                ("margin", format!("{margin:.6}")),
                ("decision", decision.word().to_string()),
                (
                    "raw",
                    raw[qi]
                        .iter()
                        .map(|v| format!("{v:.6}"))
                        .collect::<Vec<_>>()
                        .join(","),
                ),
            ],
        );
        answers.push(a);
    }

    turn.ok(&[
        ("answers", answers.len().to_string()),
        ("prompt_tokens", out.prompt_tokens.to_string()),
        ("embeddings", want.len().to_string()),
    ]);
    json_response(
        200,
        json!({
            "model": model_id,
            "backend": BACKEND,
            "answers": answers,
            "usage": {
                "prompt_tokens": out.prompt_tokens,
                "questions": req.questions.len(),
                "embeddings": want.len(),
            },
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q_choice() -> String {
        r#"{"state":"I want to cancel my subscription",
            "questions":[{"name":"intent","type":"choice","instructions":"what does the person want",
              "options":[
                {"id":"escalate_user_planning_unsubscription","tether":{"means":"the person is asking us to end their subscription now","not":["quoting or translating a sentence about unsubscribing"]}},
                {"id":"none_of_these","tether":{"means":"something else entirely"}}
              ]}]}"#
            .to_string()
    }

    /// The seventeenth question is refused before any is scored, by
    /// name and with the number: a System 1 seat would have run all
    /// seventeen as one batch with nothing to stop it.
    /// A caller's temperature and the cosine road's default are not the
    /// same fact. Collapsing them at parse time cost the marker seat the
    /// ability to tell them apart — so it dropped the caller's value and
    /// echoed the checkpoint's under the same key.
    #[test]
    fn a_temperature_is_absent_when_the_caller_sent_none() {
        let with = r#"{"state":"x","temperature":2.0,"questions":[
            {"name":"q","type":"choice","options":[
              {"id":"a","tether":{"means":"the first thing"}},
              {"id":"b","tether":{"means":"the second thing"}}]}]}"#;
        assert_eq!(parse_request(with).expect("valid").temperature, Some(2.0));

        let without = r#"{"state":"x","questions":[
            {"name":"q","type":"choice","options":[
              {"id":"a","tether":{"means":"the first thing"}},
              {"id":"b","tether":{"means":"the second thing"}}]}]}"#;
        assert_eq!(parse_request(without).expect("valid").temperature, None);

        // An explicit null is "said nothing", not "said zero".
        let null = without.replace(r#""state":"x""#, r#""state":"x","temperature":null"#);
        assert_eq!(parse_request(&null).expect("valid").temperature, None);

        // And a nonsense value is still refused by name, as before.
        let bad = without.replace(r#""state":"x""#, r#""state":"x","temperature":0"#);
        let Err((_, v)) = parse_request(&bad) else {
            panic!("zero is not a temperature")
        };
        assert!(
            v["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("temperature")),
            "{v}"
        );
    }

    #[test]
    fn seventeen_questions_are_refused_before_any_is_scored() {
        // Each question is WELL FORMED, so the cap is what refuses the
        // body — not a tether a hostile caller forgot.
        let well_formed = |i: usize| {
            format!(
                r#"{{"name":"q{i}","type":"noul","instructions":"is it so",
                     "tether":{{"means":"the statement holds","not":["it does not"]}}}}"#
            )
        };
        let qs: Vec<String> = (0..17).map(well_formed).collect();
        let body = format!(r#"{{"state":"a state","questions":[{}]}}"#, qs.join(","));
        let Err((code, v)) = parse_request(&body) else {
            panic!("seventeen questions must be refused before any is scored");
        };
        assert_eq!(code, 400);
        assert_eq!(v["error"]["type"], "budget");
        let msg = v["error"]["message"].as_str().unwrap_or("");
        assert!(msg.contains("17") && msg.contains("16"), "{msg}");

        let sixteen: Vec<String> = (0..16).map(well_formed).collect();
        let body = format!(
            r#"{{"state":"a state","questions":[{}]}}"#,
            sixteen.join(",")
        );
        assert!(
            parse_request(&body).is_ok(),
            "sixteen is the cap, not past it"
        );
    }

    #[test]
    fn a_request_parses_and_never_embeds_an_identifier() {
        let r = parse_request(&q_choice()).expect("a well-formed request");
        assert_eq!(r.questions.len(), 1);
        for o in &r.questions[0].options {
            let text = o.scored_text();
            assert!(
                !text.contains(&o.id),
                "the identifier reached the scorer: {text}"
            );
        }
    }

    #[test]
    fn an_option_with_no_meaning_is_refused_before_anything_is_embedded() {
        let body = r#"{"state":"x","questions":[{"name":"q","type":"choice","options":[{"id":"a","tether":{"means":""}}]}]}"#;
        let (code, v) = parse_request(body).err().expect("a label with no meaning");
        assert_eq!(code, 400);
        assert!(v.to_string().contains("meaning"), "{v}");
    }

    #[test]
    fn a_noul_without_a_contrary_is_refused_by_name() {
        let body = r#"{"state":"x","questions":[{"name":"is_request","type":"noul","tether":{"means":"the person is asking for something"}}]}"#;
        let (code, v) = parse_request(body)
            .err()
            .expect("a noul with nothing to score against");
        assert_eq!(code, 400);
        assert!(v.to_string().contains("contrary"), "{v}");
    }

    #[test]
    fn an_implication_naming_an_unasked_question_is_refused() {
        let body = r#"{"state":"x","questions":[{"name":"intent","type":"choice","options":[
            {"id":"a","tether":{"means":"the person wants a thing","requires":["is_request"]}}]}]}"#;
        let (_, v) = parse_request(body)
            .err()
            .expect("an implication with no question");
        assert!(v.to_string().contains("is_request"), "{v}");
    }

    #[test]
    fn an_image_state_is_refused_by_name_on_a_text_seat() {
        let body = r#"{"state":{"image":"AAAA"},"questions":[{"name":"q","type":"choice","options":[{"id":"a","tether":{"means":"a thing"}}]}]}"#;
        let (code, v) = parse_request(body).err().expect("an image on a text seat");
        assert_eq!(code, 400);
        assert!(v.to_string().contains("image encoder"), "{v}");
    }

    #[test]
    fn the_option_cache_key_covers_every_text_it_depends_on() {
        let a = key_of(Kind::Choice, "ins", &["one".into(), "two".into()]);
        let b = key_of(Kind::Choice, "ins", &["one".into(), "three".into()]);
        let c = key_of(Kind::Choice, "other", &["one".into(), "two".into()]);
        let d = key_of(Kind::Noul, "ins", &["one".into(), "two".into()]);
        assert_ne!(a, b, "a changed option must change the key");
        assert_ne!(a, c, "changed instructions must change the key");
        assert_ne!(a, d, "a changed kind must change the key");
        assert_eq!(
            a,
            key_of(Kind::Choice, "ins", &["one".into(), "two".into()])
        );
    }

    /// A sitting must be able to say whether the classifier ran, and how
    /// much of it the cache answered. §126's run could only answer that
    /// by reading a console, so a record could not carry it.
    #[test]
    fn the_classifier_counts_what_it_did_and_what_the_cache_answered() {
        let c = Classifier::new();
        assert_eq!(c.stats().requests, 0, "a fresh classifier has done nothing");

        c.counted(3);
        c.counted(1);
        let s = c.stats();
        assert_eq!(
            (s.requests, s.questions),
            (2, 4),
            "requests and questions are counted apart"
        );

        let key = key_of(Kind::Choice, "ins", &["one".into()]);
        assert!(c.get(key).is_none(), "a cold key misses");
        c.put(key, Arc::new(vec![vec![0.0f32]]));
        assert!(c.get(key).is_some(), "a warm key hits");
        let s = c.stats();
        assert_eq!((s.cache_misses, s.cache_hits, s.cache_entries), (1, 1, 1));
        assert_eq!(s.cache_clears, 0, "nothing was evicted yet");

        // The bound is a real eviction, and the record says it happened
        // rather than leaving a later miss looking like a cold start.
        for i in 0..256u64 {
            c.put(key.wrapping_add(i + 1), Arc::new(vec![vec![0.0f32]]));
        }
        assert_eq!(
            c.stats().cache_clears,
            1,
            "the bound cleared the cache once"
        );
    }

    /// One name for the road, so the answer and the stats route cannot
    /// describe two different scorers. A record that says `embed-cosine`
    /// must not have been produced by anything else.
    #[test]
    fn one_constant_names_the_scorer_for_both_the_answer_and_the_stats() {
        assert_eq!(BACKEND, "embed-cosine");
        // FNV-1a's published vectors, because a hash named after an
        // algorithm should be that algorithm. The multiplier here read
        // `0x1000_0000_01b3` for one digit group too long — a fine hash,
        // and not the one two other places in this estate compute, which
        // is how a state fingerprint stops being comparable across
        // tools. Found by a suite fixture whose hashes were generated
        // outside Rust and disagreed.
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x8594_4171_f739_67e8);
        // The model card names the same road. A literal there would let
        // `/v1/models` and `/v1/classify` disagree about the scorer.
        assert!(
            !std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/serve.rs"))
                .expect("serve.rs")
                .contains("\"classify_backend\": if engine.supports_embeddings() { \"embed"),
            "serve.rs spells the backend as a literal instead of naming BACKEND"
        );
        let src = include_str!("classify.rs");
        assert!(
            !src.contains("\"backend\": \"embed"),
            "the answer must name the road through BACKEND, not a second literal"
        );
    }
}
