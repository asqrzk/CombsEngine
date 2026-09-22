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
use combs_classify::system_one::{criteria_for, Criteria, Rendering};
use combs_classify::{decide, expected_level, Decision, Implication, Kind};
use combs_runtime::{DecisionEngine, DecisionQuestion, WarmReport};
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

/// The config line's fields, composed without a device so a test can read
/// what a seat would say of itself. `extra` is what the caller learned
/// with a device in hand (the WGSL doors) or from its own flags.
pub(crate) fn config_fields(
    model_id: &str,
    dtype: &str,
    fusion: bool,
    extra: &[(&'static str, String)],
) -> Vec<(&'static str, String)> {
    let mut fields = vec![
        ("model", model_id.to_string()),
        ("dtype", dtype.to_string()),
        // The DENSE road: this seat's linears are plain tensors through
        // `llama::linear` and burn's matmul. The quantized batched road
        // and its operand door are never entered here, and the line that
        // describes them said `f32 operands` under an f16 build — true of
        // a road not taken. What is printed is the road that ran.
        ("matmul", combs_models::dense_matmul_summary(dtype, fusion)),
        ("classify_backend", BACKEND.to_string()),
        // The one WGSL door on this seat's path: `RotaryEmbedding::apply_qk`
        // asks `wgsl::try_rope_qk` (rope.rs). The norms are LayerNorm,
        // which has no WGSL path, and attention is burn's — so whatever
        // `wgsl_doors` reports open, this is the door that is walked.
        ("doors_walked", "rope".to_string()),
        // Said out loud because it is the whole difference: this seat
        // holds no KV arena at all.
        ("kv_arena", "none (this seat decodes nothing)".to_string()),
    ];
    fields.extend(extra.iter().cloned());
    fields
}

/// Every WGSL door as the kernels read it, on one line — the fact the
/// kit prints and no seat did. The f16 storage door is composed the way
/// the kit composes it: a door open is not a road taken, and only the
/// build's dtype says whether f16 tensors ever reach those kernels.
#[cfg(not(feature = "cpu"))]
fn wgsl_doors_line() -> String {
    let device = combs_core::init_device();
    let client = <combs_core::wgpu::WgpuRuntime as cubecl::prelude::Runtime>::client(&device);
    combs_models::active_doors(&client)
        .into_iter()
        .map(|(k, v)| {
            let v = if k == "f16_storage_door" {
                match (v == "true", combs_core::BACKEND.dtype == "f16") {
                    (true, true) => "open · taken (f16 tensors reach the kernels)".to_string(),
                    (true, false) => {
                        "open · never taken (this build has no f16 tensors)".to_string()
                    }
                    (false, _) => "shut".to_string(),
                }
            } else {
                v
            };
            format!("{k}={v}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

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
///
/// WHAT the model reads is `combs_classify::system_one::criteria_for`,
/// shared with anything that measures this seat so a rendering cannot be
/// one thing here and another in the harness that grades it.
fn criteria_of(q: &crate::classify::Question, rendering: Rendering) -> Criteria {
    criteria_for(q.kind, &q.options, rendering)
}

/// What one request cost and what its budgets cut away.
///
/// `tokens` is the padded rectangle the model actually ran — rows times
/// len, not the sum of the questions' own lengths — because that is what
/// the matmuls are priced in. The per-question block says what each one
/// lost: an option cut past the 48-token cap or the shared head budget,
/// instructions cut to their floor, the state's tail dropped to fit.
fn usage_json(req: &crate::classify::Request, decided: &combs_runtime::Decided) -> Value {
    let per: Vec<Value> = req
        .questions
        .iter()
        .zip(&decided.answers)
        .map(|(q, a)| {
            let u = &a.usage;
            json!({
                "name": q.name,
                "tokens": u.tokens,
                "markers": u.markers,
                "dropped_options": u.dropped_options,
                "instruction_tokens_cut": u.instruction_tokens_cut,
                "option_tokens_cut": u.option_tokens_cut,
                "state_tokens_cut": u.state_tokens_cut,
            })
        })
        .collect();
    json!({
        "questions": req.questions.len(),
        "forward_passes": 1,
        "tokens": decided.batch.rows * decided.batch.len,
        "batch": {
            "rows": decided.batch.rows,
            "len": decided.batch.len,
            "markers": decided.batch.markers,
        },
        "per_question": per,
    })
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
    // Which text the options reach the scorer as. Default `id-means`,
    // which is what this seat has always done, so a record taken against
    // it stays comparable; an unknown value is refused by name.
    let rendering = match serde_json::from_str::<Value>(body)
        .ok()
        .as_ref()
        .and_then(|v| v.get("render"))
        .and_then(Value::as_str)
        .map(Rendering::parse)
    {
        None => Rendering::default(),
        Some(Ok(r)) => r,
        Some(Err(why)) => {
            counters.errors.fetch_add(1, Ordering::Relaxed);
            return json!({"_status": 400, "_body": {"error": {"type": "invalid_request", "message": why}}});
        }
    };
    let questions: Vec<DecisionQuestion> = req
        .questions
        .iter()
        .map(|q| DecisionQuestion {
            name: q.name.clone(),
            instructions: q.instructions.clone(),
            criteria: criteria_of(q, rendering),
        })
        .collect();

    let t0 = Instant::now();
    let decided = match engine.decide_with_shape(&req.state, &questions) {
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
    let answered = decided.answers.as_slice();

    // A question's probability of its FIRST option, which for a noul is
    // the proposition — exactly what `classify.rs` feeds an implication.
    let p_true: std::collections::HashMap<&str, f32> = req
        .questions
        .iter()
        .zip(answered)
        .map(|(q, a)| {
            (
                q.name.as_str(),
                a.distribution.first().copied().unwrap_or(0.0),
            )
        })
        .collect();

    let mut out = Vec::with_capacity(answered.len());
    for (q, a) in req.questions.iter().zip(answered) {
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
        // What the request COST and what it LOST. §138's record could not
        // tell a silently truncated option from a genuinely low-scoring
        // one, because the only usage on the wire was a question count:
        // the seat's cost is padded tokens (rows x len), and its budgets
        // cut instructions, options and the state's tail without a word.
        "render": rendering.as_str(),
        "usage": usage_json(&req, &decided),
    }})
}

/// Serves a decision model on `addr` forever. `warm` is what the seat
/// already ran before listening, said on the config line and in
/// `/v1/stats` so a record can tell a warmed seat from a cold one.
pub fn serve(engine: DecisionEngine, warm: WarmReport, addr: &str) -> Result<()> {
    let server = tiny_http::Server::http(addr).map_err(|e| anyhow::anyhow!("bind {addr}: {e}"))?;
    let engine = Arc::new(engine);
    let counters = Arc::new(Counters::default());
    let started = Instant::now();
    let model_id = engine.model_id().to_string();

    let mut extra: Vec<(&'static str, String)> = vec![
        ("backend", combs_core::backend_line()),
        (
            "warm_up",
            format!("{} passes, {} ms", warm.passes.len(), warm.total_ms),
        ),
    ];
    #[cfg(not(feature = "cpu"))]
    extra.push(("wgsl_doors", wgsl_doors_line()));
    let warm_json = json!({
        "passes": warm
            .passes
            .iter()
            .map(|p| json!({"rows": p.rows, "len": p.len, "ms": p.ms}))
            .collect::<Vec<_>>(),
        "total_ms": warm.total_ms,
    });
    combs_core::provenance::startup(
        "system-one",
        &config_fields(
            &model_id,
            combs_core::BACKEND.dtype,
            combs_core::BACKEND.fusion,
            &extra,
        ),
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
                    "warm_up": warm_json.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The seat's own line must describe the dense road it runs — never
    /// the quantized batched road, whose line once stood in for it and
    /// said `f32 operands` under an f16 build.
    #[test]
    fn the_config_line_never_names_the_batched_road() {
        let fields = config_fields(
            "laya-typed-decisions",
            "f16",
            false,
            &[("backend", "f16 · fusion off".to_string())],
        );
        let get = |k: &str| {
            fields
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.as_str())
                .unwrap_or_else(|| panic!("the config line has no field {k}"))
        };
        let matmul = get("matmul");
        assert!(matmul.starts_with("dense matmul"), "{matmul}");
        assert!(matmul.contains("f16 operands"), "{matmul}");
        assert!(matmul.ends_with("fusion off"), "{matmul}");
        for word in ["batched", "tuned kernel", " rows,"] {
            assert!(
                !matmul.contains(word),
                "the quantized road's words on a dense seat: {matmul}"
            );
        }
        assert_eq!(get("doors_walked"), "rope");
        assert_eq!(get("classify_backend"), "system-one");
        assert_eq!(get("dtype"), "f16");
        assert!(get("kv_arena").starts_with("none"), "{}", get("kv_arena"));
        assert_eq!(get("backend"), "f16 · fusion off");
    }

    /// Every question says what it cost and what it lost, and the
    /// request's token count is the padded rectangle the model ran —
    /// not the sum of the questions' own lengths.
    #[test]
    fn usage_names_every_question_and_carries_the_five_counts() {
        let body = r#"{"state":"I want to cancel my subscription",
            "questions":[
              {"name":"is_request","type":"noul","instructions":"is the person asking us to do something",
               "tether":{"means":"the person is asking us to do something for them now",
                         "not":["the person is quoting, translating or describing"]}},
              {"name":"intent","type":"choice","instructions":"which of these is the person asking for",
               "options":[{"id":"unsubscribe","tether":{"means":"the person is asking us to end their subscription now"}},
                          {"id":"none_of_these","tether":{"means":"something else entirely"}}]}
            ]}"#;
        let req = crate::classify::parse_request(body).expect("a well-formed request");
        let answers: Vec<combs_runtime::DecisionAnswer> = req
            .questions
            .iter()
            .map(|q| combs_runtime::DecisionAnswer {
                name: q.name.clone(),
                kind: q.kind,
                distribution: vec![0.5, 0.5],
                logits: vec![0.0, 0.0],
                temperature: 1.0,
                bucket: "choice:2".to_string(),
                p_act: 1.0,
                usage: combs_classify::QuestionUsage {
                    tokens: 120,
                    markers: q.options.len(),
                    dropped_options: 0,
                    instruction_tokens_cut: 3,
                    option_tokens_cut: 7,
                    state_tokens_cut: 0,
                },
            })
            .collect();
        let decided = combs_runtime::Decided {
            answers,
            batch: combs_runtime::BatchShape {
                rows: 2,
                len: 155,
                markers: 7,
            },
        };
        let u = usage_json(&req, &decided);
        assert_eq!(u["questions"], 2);
        assert_eq!(u["forward_passes"], 1);
        // The padded rectangle, which is what was multiplied — two rows
        // of 155 even though one question is shorter than the other.
        assert_eq!(u["tokens"], 310);
        assert_eq!(u["batch"]["rows"], 2);
        assert_eq!(u["batch"]["len"], 155);
        let per = u["per_question"]
            .as_array()
            .expect("one entry per question");
        assert_eq!(per.len(), 2);
        assert_eq!(per[0]["name"], req.questions[0].name.as_str());
        for k in [
            "tokens",
            "markers",
            "dropped_options",
            "instruction_tokens_cut",
            "option_tokens_cut",
            "state_tokens_cut",
        ] {
            assert!(per[0].get(k).is_some(), "usage has no {k}: {per:?}");
        }
        assert_eq!(per[0]["option_tokens_cut"], 7);
        assert_eq!(per[1]["tokens"], 120);
    }
}
