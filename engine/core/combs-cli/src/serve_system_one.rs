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
use combs_classify::sequence::Cut;
use combs_classify::system_one::{criteria_for, Criteria, Rendering};
use combs_classify::{decide, expected_level, Decision, Implication, Kind};
use combs_runtime::{DecisionEngine, DecisionQuestion, WarmReport};
use serde_json::{json, Value};

/// Does this directory hold a decision model rather than a decoder?
///
/// What a model directory is, by its CONFIG, before anything is opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeatKind {
    /// A laya typed-decision checkpoint: the encoder's config under
    /// `encoder/`, or a flat ModernBERT with `rl_agent_config.json`
    /// beside it.
    Decision,
    /// Anything else — the generative road opens it, or refuses it.
    Generative,
}

/// By the CONFIG, before anything is opened: a laya checkpoint's config
/// lives under `encoder/`, which a generative reader would not find at
/// all, and a flat one names `modernbert` as its architecture. A flat
/// ModernBERT WITHOUT laya's `rl_agent_config.json` is an encoder with
/// no head — it can embed, it cannot decide — and this seat has no road
/// for it: it is refused here by name, before a weight is read, rather
/// than at load after 842 MB with no road left.
pub fn seat_kind(dir: &Path) -> Result<SeatKind, String> {
    if !dir.is_dir() {
        return Ok(SeatKind::Generative);
    }
    if dir.join("encoder/config.json").exists() {
        return Ok(SeatKind::Decision);
    }
    let flat_modernbert = std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| {
            v.get("model_type")
                .and_then(|m| m.as_str())
                .map(str::to_string)
        })
        .is_some_and(|m| m == "modernbert");
    if !flat_modernbert {
        return Ok(SeatKind::Generative);
    }
    if dir.join("rl_agent_config.json").exists() {
        return Ok(SeatKind::Decision);
    }
    Err(format!(
        "{}: a ModernBERT encoder without laya's head (no rl_agent_config.json beside \
         config.json) — it can embed but not decide, and this seat has no embedding road for it",
        dir.display()
    ))
}

/// What this seat calls its scorer, everywhere it says so.
pub const BACKEND: &str = "system-one";

/// The door that starts the seat on one road or the other.
///
/// The two roads are kept apart deliberately (`Road`), so which one a
/// seat serves is a property of the SEAT, not of whoever happens to call
/// it: a supervisor spawning this process sets the door, the config line
/// says what it resolved to, and a record can read it back. A request
/// may still name `render` for itself — that is how an arm of a sitting
/// measures the other road without restarting anything — but the seat's
/// own answer is the door's.
pub const RENDER_DOOR: &str = "COMBS_CLASSIFY_RENDER";

/// Resolves the door's value, refused by name rather than defaulted.
///
/// Pure so the refusal can be tested without an environment. `None` is
/// the door unset, which keeps `id-means`: the rendering this seat has
/// always served, so a seat started with no door answers exactly as the
/// sealed record did.
pub(crate) fn rendering_door(value: Option<&str>) -> Result<Rendering, String> {
    match value {
        None => Ok(Rendering::default()),
        Some(v) if v.trim().is_empty() => Ok(Rendering::default()),
        Some(v) => Rendering::parse(v.trim()).map_err(|why| format!("{RENDER_DOOR}: {why}")),
    }
}

/// The door as the environment holds it.
pub fn seat_rendering() -> Result<Rendering, String> {
    rendering_door(std::env::var(RENDER_DOOR).ok().as_deref())
}

/// Which end of a state longer than its room the seat keeps. Read before
/// the weights and refused by name; unset keeps the beginning, which is
/// how every sealed record was taken.
pub const STATE_CUT_DOOR: &str = "COMBS_CLASSIFY_STATE_CUT";

/// Resolves the state-cut door's value, pure so the refusal can be tested
/// without an environment.
pub(crate) fn state_cut_door(value: Option<&str>) -> Result<Cut, String> {
    match value {
        None => Ok(Cut::Right),
        Some(v) if v.trim().is_empty() => Ok(Cut::Right),
        Some(v) => Cut::parse(v.trim()).map_err(|why| format!("{STATE_CUT_DOOR}: {why}")),
    }
}

/// The door as the environment holds it.
pub fn seat_state_cut() -> Result<Cut, String> {
    state_cut_door(std::env::var(STATE_CUT_DOOR).ok().as_deref())
}

/// The config line's fields, composed without a device so a test can read
/// what a seat would say of itself. `extra` is what the caller learned
/// with a device in hand (the WGSL doors) or from its own flags.
pub(crate) fn config_fields(
    model_id: &str,
    dtype: &str,
    fusion: bool,
    rendering: Rendering,
    context_size_given: bool,
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
        // A caller who passed --context-size is told once, HERE, on the
        // line a record copies — not shouted to stderr on every start
        // where a log reader must decide whether a warning matters. The
        // field is absent when nothing was passed, so a line that
        // carries it says a real flag was ignored.
        // WHICH ROAD THE OPTIONS REACH THE SCORER BY. Said on the line
        // because §137 and §138 compared arms that differed in exactly
        // this and no record could tell: one scored the option's name,
        // the other the tether. A seat that does not say it cannot be
        // compared to one that does.
        ("render", rendering.as_str().to_string()),
        ("road", rendering.road().as_str().to_string()),
    ];
    if context_size_given {
        fields.push((
            "context_size",
            "ignored (this seat holds no KV arena)".to_string(),
        ));
    }
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
    /// Panics caught in the SEAT — the composing, the routing, the
    /// reading of a body — as distinct from the runtime's `panics`,
    /// which counts faults inside the forward pass. Permanent, and
    /// reported beside the model's so a record can tell the two apart.
    seat_panics: AtomicU64,
    /// Answers refused because they were not answers: a distribution
    /// that does not sum, carries a NaN, or does not have one
    /// probability per option the caller sent.
    ///
    /// Permanent, like `panics`, and for the same reason. `lib.rs` says
    /// every answer passes `check_distribution`; this road never called
    /// it, so a degenerate readout could reach the wire and a record
    /// could be drawn from it with nothing to show. Non-zero means some
    /// request got a refusal it should never have been able to earn.
    envelope_refusals: AtomicU64,
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

/// Runs one request's whole answer under a caught unwind.
///
/// `guarded_decide` in the runtime catches the FORWARD PASS. Everything
/// after it — reading the logits, composing the wire body, looking an
/// option up by the index of the largest probability — ran unguarded in
/// the seat, so a bug in our own reading killed the process and took the
/// permanent fault count with it. The count exists precisely so a record
/// drawn from a seat that faulted says so; a counter that dies with the
/// process cannot do that.
///
/// The two guards are disjoint. A model panic is caught in the runtime,
/// returns `Err`, and never reaches here; this one catches what the seat
/// itself does wrong. They are counted apart for the same reason.
fn answer_guarded(counters: &Counters, compose: impl FnOnce() -> Value) -> Value {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(compose)) {
        Ok(v) => v,
        Err(_) => {
            counters.seat_panics.fetch_add(1, Ordering::Relaxed);
            counters.errors.fetch_add(1, Ordering::Relaxed);
            json!({"_status": 500, "_body": {"error": {"type": "panic", "message":
                "the seat panicked composing this answer: the request is refused and the seat \
                 stays up, but this process has now faulted and says so in \
                 `/v1/stats.classify.panics` for the rest of its life"}}})
        }
    }
}

/// Refuses a temperature this seat will not apply.
///
/// It used to be parsed, validated, dropped, and then the CHECKPOINT's
/// value echoed back under the same key — so a caller who sent 2.0 was
/// told 1.0000159 with no way to know their number had gone nowhere.
/// This seat's answer claims `calibrated: true` with
/// `source: "checkpoint"`; applying a caller's scale would make that
/// claim false, and ignoring one quietly makes the echo a lie. Any
/// temperature is re-derivable offline from the `similarities` the
/// answer already carries, which is what a sweep does.
fn refuse_unappliable_temperature(
    req: &crate::classify::Request,
    counters: &Counters,
) -> std::result::Result<(), Value> {
    if req.temperature.is_none() {
        return Ok(());
    }
    counters.errors.fetch_add(1, Ordering::Relaxed);
    Err(
        json!({"_status": 400, "_body": {"error": {"type": "invalid_request", "message":
        "this seat will not take a temperature: its scale is the checkpoint's fitted \
         calibration, reported per answer under `calibration`. Sweep offline from the \
         `similarities` in the answer instead."}}}),
    )
}

/// Refuses a body that tries to choose its own rendering.
///
/// Pure, and it names the door rather than the field: a caller who wants
/// the other road needs a seat serving it, and the message says how.
fn refuse_render_key(body: &str) -> std::result::Result<(), String> {
    let Ok(v) = serde_json::from_str::<Value>(body) else {
        // Malformed JSON is `parse_request`'s refusal to make, not this
        // one's; it has already run by the time we are called.
        return Ok(());
    };
    if v.get("render").is_some() {
        return Err(format!(
            "a request cannot choose its own rendering: the road is the seat's, set by \
             {RENDER_DOOR} (normal | semantic | id-means | label-means | label-anchors) before \
             the weights load, and reported on the config line, in /v1/stats and in every \
             answer. Start a seat on the other road."
        ));
    }
    Ok(())
}

/// Turns one decided batch into the wire's `answers`, or refuses.
///
/// Pure, so the refusal can be tested without a checkpoint or a device.
/// Nothing here indexes on faith: a model that returned a ragged or
/// degenerate readout refuses the whole request by name rather than
/// answering from whatever happened to be in the vector. `lib.rs`
/// promises every answer passes [`check_distribution`]; this is the road
/// that keeps the promise.
///
/// The whole request is refused, not the one question: a body asks its
/// questions together and a caller cannot act on half an answer set.
fn compose_answers(
    req: &crate::classify::Request,
    decided: &combs_runtime::Decided,
    counters: &Counters,
) -> std::result::Result<Vec<Value>, (u16, Value)> {
    let refuse = |what: String| -> (u16, Value) {
        counters.envelope_refusals.fetch_add(1, Ordering::Relaxed);
        counters.errors.fetch_add(1, Ordering::Relaxed);
        (500, json!({"error": {"type": "envelope", "message": what}}))
    };
    let answered = decided.answers.as_slice();
    if answered.len() != req.questions.len() {
        return Err(refuse(format!(
            "the seat answered {} of {} questions: a request is answered whole or not at all",
            answered.len(),
            req.questions.len()
        )));
    }
    for (q, a) in req.questions.iter().zip(answered) {
        if a.distribution.len() != q.options.len() {
            return Err(refuse(format!(
                "question {:?}: {} probabilities for {} options — the answer is not over the \
                 options that were asked about",
                q.name,
                a.distribution.len(),
                q.options.len()
            )));
        }
        if a.logits.len() != a.distribution.len() {
            return Err(refuse(format!(
                "question {:?}: {} logits against {} probabilities — the two are keyed by the \
                 same option ids and cannot differ",
                q.name,
                a.logits.len(),
                a.distribution.len()
            )));
        }
        if let Err(why) = combs_classify::check_distribution(&a.distribution) {
            return Err(refuse(format!("question {:?}: {why}", q.name)));
        }
    }

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
        // Checked above: `p` is as long as `q.options`, so `top_i` is a
        // real option and the two indexings below cannot panic.
        let top = &q.options[top_i];
        // laya's own rule: the whole distribution for a choice or a
        // score, and the distance from a coin flip for a noul.
        let conf = combs_classify::system_one::decision_confidence(q.kind, p);
        let mut sorted = p.clone();
        sorted.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
        let margin = sorted.first().copied().unwrap_or(0.0) - sorted.get(1).copied().unwrap_or(0.0);

        let implications: Vec<Implication> = top
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
            "top": top.id,
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
        // Only when the loader refused the checkpoint's own number: a
        // served 0.5 that was fitted at 0.5 says nothing, a served 0.5
        // that was written as 0.1006 says what it was clamped from.
        if let Some(from) = a.clamped_from {
            ans["calibration"]["clamped_from"] = json!(from);
        }
        out.push(ans);
    }
    Ok(out)
}

/// Answers one `/v1/classify` body from the decision seat.
fn classify(engine: &DecisionEngine, counters: &Counters, body: &str, seat: Rendering) -> Value {
    let req = match crate::classify::parse_request(body) {
        Ok(r) => r,
        Err((code, v)) => {
            counters.errors.fetch_add(1, Ordering::Relaxed);
            return json!({"_status": code, "_body": v});
        }
    };
    // The road is the SEAT's, and only the seat's.
    //
    // A request naming `render` is refused rather than obeyed. This was
    // a per-request override for one commit, on the argument that an arm
    // of a sitting could measure the other road without restarting —
    // which is exactly the hole: the config line, `/v1/stats` and every
    // record would say one road while a body quietly scored another. A
    // seat's line is the whole truth only if nothing else can move it,
    // and §137 against §138 is already one comparison of arms whose
    // option text differed with nothing in either record able to say so.
    // A sitting that wants both roads starts two seats, or restarts one.
    if let Err(why) = refuse_render_key(body) {
        counters.errors.fetch_add(1, Ordering::Relaxed);
        return json!({"_status": 400, "_body": {"error": {"type": "invalid_request", "message": why}}});
    }
    if let Err(v) = refuse_unappliable_temperature(&req, counters) {
        return v;
    }
    let rendering = seat;
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
    let out = match compose_answers(&req, &decided, counters) {
        Ok(v) => v,
        Err((code, body)) => return json!({"_status": code, "_body": body}),
    };

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
        "road": rendering.road().as_str(),
        "usage": usage_json(&req, &decided),
    }})
}

/// Serves a decision model on `addr` forever. `warm` is what the seat
/// already ran before listening, said on the config line and in
/// `/v1/stats` so a record can tell a warmed seat from a cold one.
/// The seat's `/v1/stats`, composed without a device so a test can read
/// what a record would carry. `model_panics` is the engine's count; the
/// seat's own is in `counters`.
fn stats_json(
    uptime_s: u64,
    counters: &Counters,
    model_panics: u64,
    rendering: Rendering,
    state_cut: Cut,
    warm_up: &Value,
) -> Value {
    json!({
        "uptime_s": uptime_s,
        // The generative seat's two keys, in its shapes: a sitting copies
        // this file into its record, and `build-info --refuse-dirty` can
        // only be graded from a record that names the build.
        "build_manifest": crate::build_info::manifest(),
        "build": {
            "dtype": combs_core::BACKEND.dtype,
            "fusion": combs_core::BACKEND.fusion,
            "cpu_backend": combs_core::BACKEND.cpu,
        },
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
            "render": rendering.as_str(),
            "road": rendering.road().as_str(),
            "state_cut": state_cut.as_str(),
            // Non-zero is PERMANENT and deliberate. A seat
            // that ever faulted carries the fact here for
            // the life of the process, because a sitting
            // copies this file into its record: a
            // measurement must never be drawn from a broken
            // process without the record showing it. A panic
            // is OUR bug faulting before the model answers —
            // never the model inventing anything, which this
            // seat cannot do.
            // The sum a reader should look at, and the
            // two halves so a fault can be placed: the
            // model's forward pass, or the seat's own
            // reading of it.
            "panics": model_panics
                + counters.seat_panics.load(Ordering::Relaxed),
            "faults": {
                "model": model_panics,
                "seat": counters.seat_panics.load(Ordering::Relaxed),
            },
            // Permanent, like `panics`: an answer that was
            // not a distribution never reached a caller, and
            // a record drawn from this seat says so.
            "envelope_refusals": counters
                .envelope_refusals
                .load(Ordering::Relaxed),
        },
        "warm_up": warm_up.clone(),
    })
}

/// What the warm-up cost, for the config line and `/v1/stats`.
///
/// `markers` is part of the shape, not decoration: the head's matmuls
/// are sized by the marker width, so a record that says only the
/// rectangle cannot tell whether the seat's first real request paid
/// autotune. Pure, so a test reads what a record would carry.
fn warm_json(warm: &WarmReport) -> Value {
    json!({
        "passes": warm
            .passes
            .iter()
            .map(|p| json!({"rows": p.rows, "len": p.len, "markers": p.markers, "ms": p.ms}))
            .collect::<Vec<_>>(),
        "total_ms": warm.total_ms,
    })
}

/// The response type every route builds.
type Resp = tiny_http::Response<std::io::Cursor<Vec<u8>>>;

/// One thread per request, so a GET never waits behind a forward pass.
/// This does NOT make the seat concurrent: the engine's model lock is
/// its only serialisation and stays, so two classifies still run one
/// after the other and throughput is 1/latency. What moves is the
/// WAITING — `/health`, `/v1/stats`, parsing and refusals answer during
/// a 208 ms classify, where the supervisor polls health every 250 ms.
///
/// The handler runs inside a guard: a panic anywhere in routing answers
/// `panicked`'s response instead of dropping the request (tiny_http
/// answers a dropped request with an empty 500, and nobody counts it).
/// Kept apart from `serve` so a test can drive the loop with a handler
/// of its own.
fn serve_on<H, P>(server: Arc<tiny_http::Server>, handle: H, panicked: P)
where
    H: Fn(&str, &str, &mut dyn std::io::Read) -> Resp + Send + Sync + 'static,
    P: Fn(&str, &str) -> Resp + Send + Sync + 'static,
{
    let handle = Arc::new(handle);
    let panicked = Arc::new(panicked);
    for mut request in server.incoming_requests() {
        let handle = handle.clone();
        let panicked = panicked.clone();
        std::thread::spawn(move || {
            let path = request.url().split('?').next().unwrap_or("").to_string();
            let method = request.method().as_str().to_string();
            let response = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                handle(&method, &path, request.as_reader())
            })) {
                Ok(response) => response,
                Err(_) => panicked(&method, &path),
            };
            let _ = request.respond(response);
        });
    }
}

pub fn serve(
    engine: DecisionEngine,
    warm: WarmReport,
    rendering: Rendering,
    context_size_given: bool,
    addr: &str,
) -> Result<()> {
    let server = tiny_http::Server::http(addr).map_err(|e| anyhow::anyhow!("bind {addr}: {e}"))?;
    let engine = Arc::new(engine);
    let counters = Arc::new(Counters::default());
    let started = Instant::now();
    let model_id = engine.model_id().to_string();

    let mut extra: Vec<(&'static str, String)> = vec![
        ("backend", combs_core::backend_line()),
        // Which end of a long state this seat keeps — the door's word.
        ("state_cut", engine.limits().cut.as_str().to_string()),
        (
            "warm_up",
            format!("{} passes, {} ms", warm.passes.len(), warm.total_ms),
        ),
    ];
    #[cfg(not(feature = "cpu"))]
    extra.push(("wgsl_doors", wgsl_doors_line()));
    let warm_json = warm_json(&warm);
    combs_core::provenance::startup(
        "system-one",
        &config_fields(
            &model_id,
            combs_core::BACKEND.dtype,
            combs_core::BACKEND.fusion,
            rendering,
            context_size_given,
            &extra,
        ),
    );
    eprintln!(
        "combs serve: listening on http://{addr} (system-one seat: {model_id}, \
         {} road, render {})",
        rendering.road().as_str(),
        rendering.as_str()
    );

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

    let server = Arc::new(server);
    let faults = counters.clone();
    let handle = move |method: &str, path: &str, reader: &mut dyn std::io::Read| -> Resp {
        let meta = engine.metadata();
        match (method, path) {
            ("GET", "/health") => json_response(200, json!({"status": "ok"})),
            ("GET", "/v1/models") => json_response(
                200,
                json!({"object": "list", "data": [card(&model_id, meta)]}),
            ),
            ("GET", "/v1/model/info") => json_response(200, card(&model_id, meta)),
            ("GET", "/v1/stats") => json_response(
                200,
                stats_json(
                    started.elapsed().as_secs(),
                    &counters,
                    engine.panics(),
                    rendering,
                    engine.limits().cut,
                    &warm_json,
                ),
            ),
            ("POST", "/v1/classify") => {
                let mut body = String::new();
                if let Err(e) = reader.read_to_string(&mut body) {
                    refuse(400, "invalid_request", &format!("unreadable body: {e}"))
                } else {
                    let v = answer_guarded(&counters, || {
                        classify(&engine, &counters, &body, rendering)
                    });
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
        }
    };
    serve_on(server, handle, move |method, path| {
        // Counted under the same key as a fault inside the answer, and
        // permanent for the same reason: a record drawn from this seat
        // must show it.
        faults.seat_panics.fetch_add(1, Ordering::Relaxed);
        refuse(
            500,
            "panic",
            &format!(
                "the seat faulted while answering {method} {path}; the fault is counted for the \
                 life of the process in /v1/stats.classify.faults.seat"
            ),
        )
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::Read;
    use std::time::Duration;

    /// `--context-size` is a generative flag; a decision seat holds no
    /// KV arena and ignores it. It used to be shouted to stderr on every
    /// start that passed it — a warning a log reader has to weigh, and
    /// one the supervisor passes unconditionally, so it fired on every
    /// gateway-spawned seat. It belongs on the line a record copies,
    /// once, beside the `kv_arena` field that says why: one fact, not an
    /// alarm. A line that carries it says a real flag was ignored; a
    /// line without it says none was passed.
    #[test]
    fn a_context_size_handed_to_a_decision_seat_is_said_on_the_line_not_shouted() {
        let field = |given: bool| {
            config_fields("m", "f16", false, Rendering::default(), given, &[])
                .into_iter()
                .find(|(k, _)| *k == "context_size")
                .map(|(_, v)| v)
        };
        assert_eq!(
            field(true).as_deref(),
            Some("ignored (this seat holds no KV arena)")
        );
        assert_eq!(field(false), None, "a flag nobody passed is not a field");
    }

    /// A warm-up pass is a rectangle AND a marker width; a record that
    /// says only the rectangle cannot tell whether the first real
    /// request paid autotune, which is ~5 s at an untuned shape.
    #[test]
    fn warm_up_says_how_many_markers_each_pass_had() {
        let report = WarmReport {
            passes: vec![
                combs_runtime::WarmPass {
                    rows: 1,
                    len: 64,
                    markers: 2,
                    ms: 7,
                },
                combs_runtime::WarmPass {
                    rows: 16,
                    len: 256,
                    markers: 64,
                    ms: 11,
                },
            ],
            total_ms: 18,
        };
        let json = warm_json(&report);
        assert_eq!(json["total_ms"], 18);
        assert_eq!(json["passes"][0]["markers"], 2);
        assert_eq!(json["passes"][1]["markers"], 64);
        assert_eq!(json["passes"][1]["rows"], 16);
        assert_eq!(json["passes"][1]["len"], 256);
        assert_eq!(json["passes"][1]["ms"], 11);
    }

    /// The state-cut door: two words; unset keeps the beginning every
    /// sealed record was taken with; anything else is refused naming the
    /// door.
    #[test]
    fn the_state_cut_door_takes_two_words_and_names_itself_otherwise() {
        assert_eq!(state_cut_door(None), Ok(Cut::Right));
        assert_eq!(state_cut_door(Some("")), Ok(Cut::Right));
        assert_eq!(state_cut_door(Some(" left ")), Ok(Cut::Left));
        let why = state_cut_door(Some("middle")).expect_err("refused");
        assert!(
            why.contains("COMBS_CLASSIFY_STATE_CUT") && why.contains("middle"),
            "{why}"
        );
    }

    /// A sitting copies `/v1/stats` into its record, and `build-info
    /// --refuse-dirty` can only be graded from a record that names the
    /// build. The generative seat has carried `build_manifest` and
    /// `build.dtype` for as long as the platform has read them; this seat
    /// carried neither, and s138 could not be graded. Same keys, same
    /// shapes, so `serve_stats`' assertions read either seat.
    #[test]
    fn the_seats_stats_carry_the_manifest_under_the_generative_seats_keys() {
        let counters = Counters::default();
        counters.requests.fetch_add(3, Ordering::Relaxed);
        let warm = json!({"total_ms": 1});
        let stats = stats_json(7, &counters, 0, Rendering::LabelAnchors, Cut::Left, &warm);
        let manifest = &stats["build_manifest"];
        assert!(
            manifest["git"]["short"].is_string(),
            "the git identity: {manifest}"
        );
        assert!(manifest["built_at"].is_string(), "built_at: {manifest}");
        assert_eq!(manifest["serving_dtype"], combs_core::BACKEND.dtype);
        let dtype = stats["build"]["dtype"].as_str().unwrap_or_default();
        assert!(
            dtype == "f16" || dtype == "f32",
            "the platform reads .build.dtype: {}",
            stats["build"]
        );
        assert_eq!(
            stats["build"]["dtype"], stats["build_manifest"]["serving_dtype"],
            "one build, said twice the same"
        );
        // The keys the sealed records already carry, unchanged.
        let classify = &stats["classify"];
        for key in [
            "backend",
            "requests",
            "questions",
            "errors",
            "last_ms",
            "render",
            "road",
            "panics",
            "faults",
            "envelope_refusals",
        ] {
            assert!(!classify[key].is_null(), "classify.{key} left the wire");
        }
        assert_eq!(classify["requests"], 3);
        assert_eq!(classify["road"], "semantic");
        assert_eq!(classify["state_cut"], "left");
        assert_eq!(stats["uptime_s"], 7);
        assert_eq!(stats["warm_up"]["total_ms"], 1);
    }

    fn http_get(port: u16, path: &str) -> Option<(u16, Vec<u8>)> {
        use std::io::Write;
        let raw = format!("GET {path} HTTP/1.0\r\nHost: localhost\r\nConnection: close\r\n\r\n");
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .ok()?;
        stream.write_all(raw.as_bytes()).ok()?;
        let mut response = Vec::new();
        stream.read_to_end(&mut response).ok()?;
        let header_end = response.windows(4).position(|w| w == b"\r\n\r\n")? + 4;
        let head = String::from_utf8_lossy(&response[..header_end]);
        let status: u16 = head.split_whitespace().nth(1)?.parse().ok()?;
        Some((status, response[header_end..].to_vec()))
    }

    fn http_post(port: u16, path: &str, body: &str) -> Option<(u16, Vec<u8>)> {
        use std::io::Write;
        let raw = format!(
            "POST {path} HTTP/1.0\r\nHost: localhost\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .ok()?;
        stream.write_all(raw.as_bytes()).ok()?;
        let mut response = Vec::new();
        stream.read_to_end(&mut response).ok()?;
        let header_end = response.windows(4).position(|w| w == b"\r\n\r\n")? + 4;
        let head = String::from_utf8_lossy(&response[..header_end]);
        let status: u16 = head.split_whitespace().nth(1)?.parse().ok()?;
        Some((status, response[header_end..].to_vec()))
    }

    /// The seat was serial at the ACCEPT loop: `/health` and `/v1/stats`
    /// queued behind a 208 ms classify, and the supervisor polls health
    /// every 250 ms. One thread per request moves the waiting, not the
    /// work: a GET 50 ms into a 600 ms POST comes back in well under the
    /// POST's time. Throughput is unchanged — the model lock still
    /// serialises the forward passes — and this test claims nothing else.
    #[test]
    fn health_answers_while_a_classify_holds_the_seat() {
        let server = Arc::new(tiny_http::Server::http("127.0.0.1:0").expect("a free port"));
        let port = server.server_addr().to_ip().expect("an ip listener").port();
        let listener = server.clone();
        let served = std::thread::spawn(move || {
            serve_on(
                listener,
                |method, path, _reader| {
                    if method == "POST" {
                        std::thread::sleep(Duration::from_millis(600));
                    }
                    json_response(200, json!({"path": path}))
                },
                |_, _| refuse(500, "panic", "unreachable here"),
            );
        });
        let post = std::thread::spawn(move || {
            let t = Instant::now();
            let status = http_post(port, "/v1/classify", "{}").map(|(s, _)| s);
            (t.elapsed().as_millis(), status)
        });
        std::thread::sleep(Duration::from_millis(50));
        let t = Instant::now();
        let get = http_get(port, "/health").map(|(s, _)| s);
        let get_ms = t.elapsed().as_millis();
        assert_eq!(get, Some(200));
        assert!(
            get_ms < 200,
            "GET /health took {get_ms} ms behind a 600 ms POST: the seat is still serial at the \
             accept loop"
        );
        let (post_ms, post_status) = post.join().expect("the POST thread");
        assert_eq!(post_status, Some(200));
        assert!(
            post_ms >= 600,
            "the POST answered in {post_ms} ms, before its handler could have"
        );
        server.unblock();
        served
            .join()
            .expect("the loop ends when the listener is unblocked");
    }

    /// A panic anywhere in routing used to drop the request, which
    /// tiny_http answers with an empty 500 that nobody counts. Inside the
    /// thread the guard answers first, and the loop goes on.
    #[test]
    fn a_panic_outside_the_answer_guard_is_answered_and_counted() {
        let server = Arc::new(tiny_http::Server::http("127.0.0.1:0").expect("a free port"));
        let port = server.server_addr().to_ip().expect("an ip listener").port();
        let listener = server.clone();
        let counted = Arc::new(AtomicU64::new(0));
        let seen = counted.clone();
        let served = std::thread::spawn(move || {
            serve_on(
                listener,
                |_method, path, _reader| {
                    if path == "/boom" {
                        panic!("a bug in routing");
                    }
                    json_response(200, json!({}))
                },
                move |method, path| {
                    seen.fetch_add(1, Ordering::Relaxed);
                    refuse(500, "panic", &format!("faulted on {method} {path}"))
                },
            );
        });
        let (status, body) = http_get(port, "/boom").expect("an answer, not a dropped request");
        assert_eq!(status, 500);
        assert!(
            String::from_utf8_lossy(&body).contains("faulted on GET /boom"),
            "{body:?}"
        );
        assert_eq!(counted.load(Ordering::Relaxed), 1);
        let (status, _) = http_get(port, "/health").expect("the next request answers");
        assert_eq!(status, 200, "one fault does not take the loop down");
        server.unblock();
        served.join().expect("the loop ends");
    }

    /// A flat ModernBERT without laya's head used to be routed to this
    /// seat by its `model_type` alone and die at load, after the read,
    /// with no road left. It is an encoder — it can embed, not decide —
    /// and it is refused here by name before a weight is opened.
    #[test]
    fn an_encoder_without_a_head_is_refused_before_load() {
        let root = std::env::temp_dir().join(format!("combs-seat-kind-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let flat = root.join("flat");
        std::fs::create_dir_all(&flat).expect("a scratch dir");
        std::fs::write(flat.join("config.json"), r#"{"model_type":"modernbert"}"#).expect("config");
        let why = seat_kind(&flat).expect_err("an encoder without a head is refused");
        assert!(
            why.contains("rl_agent_config.json") && why.contains("embed"),
            "the refusal names what is missing and what the model can do: {why}"
        );
        std::fs::write(flat.join("rl_agent_config.json"), "{}").expect("agent config");
        assert_eq!(
            seat_kind(&flat),
            Ok(SeatKind::Decision),
            "with laya's head beside it, a seat"
        );

        let laya = root.join("laya");
        std::fs::create_dir_all(laya.join("encoder")).expect("a scratch dir");
        std::fs::write(
            laya.join("encoder/config.json"),
            r#"{"model_type":"modernbert"}"#,
        )
        .expect("config");
        assert_eq!(
            seat_kind(&laya),
            Ok(SeatKind::Decision),
            "laya's own layout is a seat"
        );

        let generative = root.join("gen");
        std::fs::create_dir_all(&generative).expect("a scratch dir");
        std::fs::write(generative.join("config.json"), r#"{"model_type":"qwen3"}"#)
            .expect("config");
        assert_eq!(seat_kind(&generative), Ok(SeatKind::Generative));
        assert_eq!(
            seat_kind(&root.join("missing")),
            Ok(SeatKind::Generative),
            "not a dir: the generative road decides"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The seat parsed a temperature, validated it, threw it away, and
    /// echoed the checkpoint's under the same key — so a caller who sent
    /// 2.0 was told 1.0000159 and could not tell their number had gone
    /// nowhere. Refusing says so; ignoring made the echo a lie.
    #[test]
    fn a_temperature_the_seat_will_not_apply_is_refused_not_echoed() {
        let body = r#"{"state":"x","temperature":2.0,"questions":[
            {"name":"q","type":"choice","options":[
              {"id":"a","tether":{"means":"the first thing"}},
              {"id":"b","tether":{"means":"the second thing"}}]}]}"#;
        let req = crate::classify::parse_request(body).expect("a well-formed request");
        assert_eq!(req.temperature, Some(2.0), "the parser keeps what was sent");

        // The refusal names the road out: the answer already carries the
        // logits a sweep would re-derive a temperature from.
        let counters = Counters::default();
        let v = refuse_unappliable_temperature(&req, &counters).expect_err("a temperature");
        assert_eq!(v["_status"], 400);
        let msg = v["_body"]["error"]["message"].as_str().expect("a message");
        assert!(msg.contains("checkpoint"), "{msg}");
        assert!(msg.contains("similarities"), "{msg}");
        assert_eq!(counters.errors.load(Ordering::Relaxed), 1);

        // Without one, nothing is refused.
        let plain = body.replace(r#""temperature":2.0,"#, "");
        let req = crate::classify::parse_request(&plain).expect("a well-formed request");
        assert!(refuse_unappliable_temperature(&req, &counters).is_ok());
        assert_eq!(counters.errors.load(Ordering::Relaxed), 1);
    }

    /// The runtime's guard covers the forward pass; this one covers
    /// everything the seat does with the result. Before it, a bug in our
    /// own reading killed the process — and with it the permanent count
    /// that exists so a record can say the seat ever faulted.
    #[test]
    fn a_panic_in_the_seat_answers_a_refusal_and_is_carried() {
        let counters = Counters::default();

        let v = answer_guarded(&counters, || panic!("an index the composer tripped"));
        assert_eq!(v["_status"], 500);
        assert_eq!(v["_body"]["error"]["type"], "panic");
        assert!(
            v["_body"]["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("panics")),
            "the refusal must say where the fact now lives: {v}"
        );
        assert_eq!(counters.seat_panics.load(Ordering::Relaxed), 1);
        assert_eq!(counters.errors.load(Ordering::Relaxed), 1);

        // The seat keeps answering, and recovering does not reset the
        // count — a process that faulted once says so for its life.
        let v = answer_guarded(&counters, || json!({"_status": 200}));
        assert_eq!(v["_status"], 200);
        assert_eq!(counters.seat_panics.load(Ordering::Relaxed), 1);
    }

    /// The road a seat serves is a property of the SEAT. For one commit
    /// a body could name `render` and flip it for that request, which
    /// made the config line, `/v1/stats` and the record say one road
    /// while another was scored — the §137-against-§138 defect, rebuilt
    /// on the road meant to close it. A body that asks is now told where
    /// the road actually lives.
    #[test]
    fn a_request_cannot_change_the_seats_road() {
        let with = r#"{"state":"x","render":"semantic","questions":[]}"#;
        let why = refuse_render_key(with).expect_err("a body naming render");
        assert!(why.contains(RENDER_DOOR), "{why}");
        assert!(why.contains("semantic") && why.contains("normal"), "{why}");
        assert!(why.contains("Start a seat"), "{why}");

        // Even the road the seat is already on is refused: the answer is
        // "not here", not "only if it agrees".
        assert!(refuse_render_key(r#"{"render":"id-means"}"#).is_err());

        // A body that does not ask passes, and malformed JSON is
        // `parse_request`'s refusal, not this one's.
        assert!(refuse_render_key(r#"{"state":"x","questions":[]}"#).is_ok());
        assert!(refuse_render_key("{not json").is_ok());
    }

    /// Builds a request and a matching `Decided` whose distributions the
    /// caller chooses, so a degenerate readout can be put in front of the
    /// composer without a checkpoint or a device.
    fn decided_with(dists: &[Vec<f32>]) -> (crate::classify::Request, combs_runtime::Decided) {
        let body = r#"{"state":"I want to cancel my subscription",
            "questions":[
              {"name":"intent","type":"choice","instructions":"which of these is the person asking for",
               "options":[{"id":"unsubscribe","tether":{"means":"the person is asking us to end their subscription now"}},
                          {"id":"none_of_these","tether":{"means":"something else entirely"}}]}
            ]}"#;
        let req = crate::classify::parse_request(body).expect("a well-formed request");
        let answers: Vec<combs_runtime::DecisionAnswer> = req
            .questions
            .iter()
            .zip(dists)
            .map(|(q, d)| combs_runtime::DecisionAnswer {
                name: q.name.clone(),
                kind: q.kind,
                distribution: d.clone(),
                logits: vec![0.0; d.len()],
                temperature: 1.0,
                bucket: "choice:2".to_string(),
                clamped_from: None,
                p_act: 1.0,
                usage: combs_classify::QuestionUsage {
                    tokens: 120,
                    markers: q.options.len(),
                    dropped_options: 0,
                    instruction_tokens_cut: 0,
                    option_tokens_cut: 0,
                    state_tokens_cut: 0,
                },
            })
            .collect();
        let decided = combs_runtime::Decided {
            answers,
            batch: combs_runtime::BatchShape {
                rows: 1,
                len: 120,
                markers: 2,
            },
        };
        (req, decided)
    }

    /// `calibration.temperature` on a 16-option question of this
    /// checkpoint reads 0.5, and 0.5 is not what laya fitted: `choice:11+`
    /// shipped at 0.1006 and the loader clamped it. An answer says so
    /// ONLY when it happened, so a served temperature that is the
    /// checkpoint's own carries no extra key.
    #[test]
    fn a_clamped_bucket_says_what_it_was_clamped_from_in_the_answer() {
        let counters = Counters::default();
        let (req, mut decided) = decided_with(&[vec![0.75, 0.25]]);
        let out = compose_answers(&req, &decided, &counters).expect("a distribution");
        assert!(
            out[0]["calibration"].get("clamped_from").is_none(),
            "nothing was clamped, so nothing is said: {}",
            out[0]["calibration"]
        );
        decided.answers[0].clamped_from = Some(0.1006);
        let out = compose_answers(&req, &decided, &counters).expect("a distribution");
        let from = out[0]["calibration"]["clamped_from"]
            .as_f64()
            .expect("the shipped value");
        assert!((from - 0.1006).abs() < 1e-6, "{from}");
        assert_eq!(out[0]["calibration"]["source"], "checkpoint");
    }

    /// `lib.rs` says every answer passes `check_distribution`; this road
    /// never called it, so a readout that had gone degenerate could reach
    /// a caller — and a measurement could be drawn from it with nothing
    /// in the record to show. Now the request is refused whole, by name,
    /// and the seat carries the count for the rest of its life.
    #[test]
    fn an_answer_that_is_not_a_distribution_is_refused_by_name_and_counted() {
        let counters = Counters::default();

        // Sums to 1.4.
        let (req, decided) = decided_with(&[vec![0.7, 0.7]]);
        let (code, body) =
            compose_answers(&req, &decided, &counters).expect_err("not a distribution");
        assert_eq!(code, 500);
        assert_eq!(body["error"]["type"], "envelope");
        let msg = body["error"]["message"].as_str().expect("a message");
        assert!(msg.contains("intent"), "{msg}");
        assert!(msg.contains("sums to"), "{msg}");

        // A NaN, which is what a zero-sum softmax leaves behind.
        let (req, decided) = decided_with(&[vec![f32::NAN, 0.5]]);
        let (_, body) = compose_answers(&req, &decided, &counters).expect_err("a NaN");
        assert!(
            body["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("not a number")),
            "{body}"
        );

        // Three probabilities for two options: the old code would have
        // answered, keyed only the first two, and never said so.
        let (req, decided) = decided_with(&[vec![0.2, 0.3, 0.5]]);
        let (_, body) = compose_answers(&req, &decided, &counters).expect_err("ragged");
        let msg = body["error"]["message"].as_str().expect("a message");
        assert!(msg.contains("3 probabilities for 2 options"), "{msg}");

        assert_eq!(counters.envelope_refusals.load(Ordering::Relaxed), 3);

        // And a sound answer still composes, keyed by the caller's ids.
        let (req, decided) = decided_with(&[vec![0.75, 0.25]]);
        let out = compose_answers(&req, &decided, &counters).expect("a distribution");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["top"], "unsubscribe");
        assert_eq!(out[0]["distribution"]["none_of_these"], 0.25);
        assert_eq!(counters.envelope_refusals.load(Ordering::Relaxed), 3);
    }

    /// The two roads are a SEAT's configuration, not a caller's habit.
    /// A record that cannot say which road produced it is the defect
    /// §137 and §138 fell into, so the door resolves, the line says it,
    /// and a value nobody spelled right is refused rather than defaulted
    /// into the other road's numbers.
    #[test]
    fn the_seat_is_startable_on_either_road_and_says_which() {
        assert_eq!(rendering_door(None), Ok(Rendering::IdMeans));
        assert_eq!(rendering_door(Some("")), Ok(Rendering::IdMeans));
        assert_eq!(
            rendering_door(Some("  semantic ")),
            Ok(Rendering::LabelAnchors)
        );
        assert_eq!(rendering_door(Some("normal")), Ok(Rendering::LabelMeans));
        assert_eq!(
            rendering_door(Some("label-anchors")),
            Ok(Rendering::LabelAnchors)
        );

        // Refused BY NAME, and the refusal names the door so an operator
        // knows which environment variable to fix.
        let why = rendering_door(Some("tether")).expect_err("not a road");
        assert!(why.contains(RENDER_DOOR), "{why}");
        assert!(why.contains("semantic") && why.contains("normal"), "{why}");

        // Both roads reach the config line, and the line carries the
        // road beside the exact rendering: `normal` and `id-means` are
        // the same road and different text, and a record needs both.
        for (r, road) in [
            (Rendering::IdMeans, "normal"),
            (Rendering::LabelMeans, "normal"),
            (Rendering::LabelAnchors, "semantic"),
        ] {
            let fields = config_fields("m", "f16", false, r, false, &[]);
            let get = |k: &str| {
                fields
                    .iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| v.as_str())
                    .unwrap_or_else(|| panic!("no field {k}"))
            };
            assert_eq!(get("render"), r.as_str());
            assert_eq!(get("road"), road);
        }
    }

    /// The seat's own line must describe the dense road it runs — never
    /// the quantized batched road, whose line once stood in for it and
    /// said `f32 operands` under an f16 build.
    #[test]
    fn the_config_line_never_names_the_batched_road() {
        let fields = config_fields(
            "laya-typed-decisions",
            "f16",
            false,
            Rendering::default(),
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
                clamped_from: None,
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
