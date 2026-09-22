//! A seat that classifies and decodes nothing.
//!
//! `combs serve` sizes a KV arena, fits it against free memory, and
//! stands up eleven routes. A decision model needs none of that: it
//! reads a whole sequence at once and answers from it, so §126's own
//! finding applies — the generation-sized default arena refused a
//! classifier outright, and it was right to, because a classifier
//! decodes nothing and the room was never for it.
//!
//! So this is a separate, smaller server: health, the two identity
//! routes, stats, and `/v1/classify`. Everything else is refused BY
//! NAME rather than answered wrongly — a generative seat's
//! `supports_hidden_states()` says yes unconditionally, and a caller
//! that reached `/v1/chat/completions` here should be told this seat
//! cannot, not handed something shaped like an answer.
//!
//! The wire shape is byte-for-byte the one `classify.rs` produces, with
//! `backend` and the calibration block saying which road answered. That
//! is not tidiness: §137's control arm was scored by a client reading
//! exactly these keys, and a second arm that answered a different shape
//! could not be compared to it.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use combs_classify::system_one::Criteria;
use combs_classify::{decide, expected_level, Decision, Implication, Kind};
use combs_runtime::{DecisionEngine, DecisionQuestion};
use serde_json::{json, Value};

/// Does this directory hold a decision model rather than a decoder?
///
/// By the CONFIG, before anything is opened: a laya checkpoint's config
/// lives under `encoder/`, which a generative reader would not find at
/// all, and a flat one names `modernbert` as its architecture.
pub fn is_decision_seat(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    if dir.join("encoder/config.json").exists() {
        return true;
    }
    std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| {
            v.get("model_type")
                .and_then(|m| m.as_str())
                .map(str::to_string)
        })
        .is_some_and(|m| m == "modernbert")
}

/// What this seat calls its scorer, everywhere it says so.
pub const BACKEND: &str = "system-one";

#[derive(Default)]
struct Counters {
    requests: AtomicU64,
    questions: AtomicU64,
    errors: AtomicU64,
    last_ms: AtomicU64,
}

/// A request's options, as the model reads them.
///
/// `parse_request` has already folded a noul into a two-option choice
/// whose FIRST option is the proposition; laya renders `[false, true]`,
/// so this is where the two conventions meet on the way in, as
/// `engine_order` is where they meet on the way out.
fn criteria_of(q: &crate::classify::Question) -> Criteria {
    let means = |i: usize| json!(q.options[i].tether.means.clone());
    match q.kind {
        Kind::Choice => Criteria::Choice(
            q.options
                .iter()
                .map(|o| (o.id.clone(), json!(o.tether.means.clone())))
                .collect(),
        ),
        Kind::Score => Criteria::Score(
            q.options
                .iter()
                .map(|o| json!(o.tether.means.clone()))
                .collect(),
        ),
        Kind::Noul => Criteria::Noul {
            when_true: Some(means(0)),
            when_false: q.options.get(1).map(|_| means(1)),
        },
    }
}

fn json_response(code: u16, body: Value) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let text = serde_json::to_string(&body).unwrap_or_else(|_| "{}".into());
    tiny_http::Response::from_string(text)
        .with_status_code(code)
        .with_header(
            tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
                .expect("a literal header"),
        )
}

fn refuse(code: u16, kind: &str, message: &str) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    json_response(code, json!({"error": {"type": kind, "message": message}}))
}

/// Answers one `/v1/classify` body from the decision seat.
fn classify(engine: &DecisionEngine, counters: &Counters, body: &str) -> Value {
    let req = match crate::classify::parse_request(body) {
        Ok(r) => r,
        Err((code, v)) => {
            counters.errors.fetch_add(1, Ordering::Relaxed);
            return json!({"_status": code, "_body": v});
        }
    };
    let questions: Vec<DecisionQuestion> = req
        .questions
        .iter()
        .map(|q| DecisionQuestion {
            name: q.name.clone(),
            instructions: q.instructions.clone(),
            criteria: criteria_of(q),
        })
        .collect();

    let t0 = Instant::now();
    let answered = match engine.decide(&req.state, &questions) {
        Ok(a) => a,
        Err(e) => {
            counters.errors.fetch_add(1, Ordering::Relaxed);
            return json!({"_status": 400, "_body": {"error": {"type": "invalid_request", "message": e.to_string()}}});
        }
    };
    counters
        .last_ms
        .store(t0.elapsed().as_millis() as u64, Ordering::Relaxed);
    counters.requests.fetch_add(1, Ordering::Relaxed);
    counters
        .questions
        .fetch_add(req.questions.len() as u64, Ordering::Relaxed);

    // A question's probability of its FIRST option, which for a noul is
    // the proposition — exactly what `classify.rs` feeds an implication.
    let p_true: std::collections::HashMap<&str, f32> = req
        .questions
        .iter()
        .zip(&answered)
        .map(|(q, a)| {
            (
                q.name.as_str(),
                a.distribution.first().copied().unwrap_or(0.0),
            )
        })
        .collect();

    let mut out = Vec::with_capacity(answered.len());
    for (q, a) in req.questions.iter().zip(&answered) {
        let p = &a.distribution;
        let (top_i, top_p) =
            p.iter().enumerate().fold(
                (0usize, f32::MIN),
                |b, (i, &v)| if v > b.1 { (i, v) } else { b },
            );
        // laya's own rule: the whole distribution for a choice or a
        // score, and the distance from a coin flip for a noul.
        let conf = combs_classify::system_one::decision_confidence(q.kind, p);
        let mut sorted = p.clone();
        sorted.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
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

        let by_id = |vals: &[f32]| -> serde_json::Map<String, Value> {
            q.options
                .iter()
                .zip(vals)
                .map(|(o, &v)| (o.id.clone(), json!(v)))
                .collect()
        };
        let mut ans = json!({
            "name": q.name,
            "type": match q.kind { Kind::Choice => "choice", Kind::Score => "score", Kind::Noul => "noul" },
            "top": q.options[top_i].id,
            "p_top": top_p,
            "distribution": Value::Object(by_id(p)),
            // The head's logit per option — the same key the cosine road
            // fills with a cosine, named for what it is either way.
            "similarities": Value::Object(by_id(&a.logits)),
            "confidence": conf,
            "margin": margin,
            // Unlike the cosine road, this one IS calibrated, and says
            // by what.
            "calibrated": true,
            "calibration": {"source": "checkpoint", "temperature": a.temperature, "bucket": a.bucket},
            "temperature": a.temperature,
            "p_act": a.p_act,
            "decision": decision.word(),
            "reason": match &decision { Decision::Act => Value::Null, d => json!(d.reason()) },
            "implications": implications.iter().map(|i| json!({
                "requires": i.requires, "p": i.p, "tau": i.tau, "holds": i.holds(), "gated": i.gated()
            })).collect::<Vec<_>>(),
        });
        if q.kind == Kind::Score {
            ans["expected_level"] = json!(expected_level(p));
        }
        out.push(ans);
    }

    json!({"_status": 200, "_body": {
        "model": engine.model_id(),
        "backend": BACKEND,
        "answers": out,
        "usage": {"questions": req.questions.len(), "forward_passes": 1},
    }})
}

/// Serves a decision model on `addr` forever.
pub fn serve(engine: DecisionEngine, addr: &str) -> Result<()> {
    let server = tiny_http::Server::http(addr).map_err(|e| anyhow::anyhow!("bind {addr}: {e}"))?;
    let engine = Arc::new(engine);
    let counters = Arc::new(Counters::default());
    let started = Instant::now();
    let model_id = engine.model_id().to_string();

    combs_core::provenance::startup(
        "system-one",
        &[
            ("model", model_id.clone()),
            ("dtype", combs_core::BACKEND.dtype.to_string()),
            ("backend", combs_core::backend_line()),
            ("matmul", combs_models::batched_matmul_summary()),
            ("classify_backend", BACKEND.to_string()),
            // Said out loud because it is the whole difference: this seat
            // holds no KV arena at all.
            ("kv_arena", "none (this seat decodes nothing)".to_string()),
        ],
    );
    eprintln!("combs serve: listening on http://{addr} (system-one seat: {model_id})");

    let card = |id: &str, meta: &combs_formats::ModelMetadata| {
        json!({
            "id": id,
            "object": "model",
            "owned_by": "combs",
            "context_length": meta.encoder.as_ref().map_or(0, |_| meta.max_position_embeddings),
            "kind": "system-one",
            "capabilities": {
                "classify": true,
                "classify_backend": BACKEND,
                "embeddings": false,
                "completions": false,
                "tools": false,
            },
        })
    };

    for mut request in server.incoming_requests() {
        let path = request.url().split('?').next().unwrap_or("").to_string();
        let method = request.method().as_str().to_string();
        let meta = engine.metadata();
        let response = match (method.as_str(), path.as_str()) {
            ("GET", "/health") => json_response(200, json!({"status": "ok"})),
            ("GET", "/v1/models") => json_response(
                200,
                json!({"object": "list", "data": [card(&model_id, meta)]}),
            ),
            ("GET", "/v1/model/info") => json_response(200, card(&model_id, meta)),
            ("GET", "/v1/stats") => json_response(
                200,
                json!({
                    "uptime_s": started.elapsed().as_secs(),
                    // `totals` stays zero and is present on purpose: a
                    // reader comparing this seat to a generative one must
                    // see that nothing was generated, not infer it.
                    "totals": {"requests": 0, "tokens": 0},
                    "classify": {
                        "backend": BACKEND,
                        "requests": counters.requests.load(Ordering::Relaxed),
                        "questions": counters.questions.load(Ordering::Relaxed),
                        "errors": counters.errors.load(Ordering::Relaxed),
                        "last_ms": counters.last_ms.load(Ordering::Relaxed),
                    },
                }),
            ),
            ("POST", "/v1/classify") => {
                let mut body = String::new();
                if let Err(e) = request.as_reader().read_to_string(&mut body) {
                    refuse(400, "invalid_request", &format!("unreadable body: {e}"))
                } else {
                    let v = classify(&engine, &counters, &body);
                    json_response(
                        v["_status"].as_u64().unwrap_or(500) as u16,
                        v["_body"].clone(),
                    )
                }
            }
            ("POST", "/v1/chat/completions" | "/v1/completions" | "/v1/embeddings") => refuse(
                400,
                "unsupported",
                &format!(
                    "this is a System 1 seat: it scores option markers and decodes nothing, so \
                     {path} has no answer here. Seat a generative model for that route."
                ),
            ),
            _ => refuse(404, "not_found", &format!("no route {method} {path}")),
        };
        let _ = request.respond(response);
    }
    Ok(())
}
