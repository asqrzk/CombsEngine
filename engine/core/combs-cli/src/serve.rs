//! `combs serve` — OpenAI-compatible HTTP + SSE server over the local engine.
//!
//! Endpoints:
//! - `GET  /health` → `{"status":"ok"}`
//! - `GET  /v1/models` → the loaded model id
//! - `POST /v1/chat/completions` → OpenAI chat completion, `stream: true`
//!   for SSE chunks terminated by `data: [DONE]`. Vision models accept
//!   OpenAI array content parts (`image_url` with base64 data: URLs);
//!   images are preprocessed via combs-media and spliced into the
//!   model's `<image>` token spans.
//!
//! CORS: `Access-Control-Allow-Origin: *` on every response + OPTIONS
//! preflight, so browsers can call the engine directly (static hosting).
//!
//! tiny_http is synchronous and lightweight (no async runtime); the
//! engine's single-flight queue serializes concurrent requests. SSE
//! streaming is a channel-backed `Read` fed by the engine's token callback.

use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use serde_json::{json, Value};

use combs_media::{ImagePreprocessor, PixelBatch, SiglipPreprocessor};
use combs_models::image_prompt_expansion;
use combs_runtime::{Engine, GenerationConfig, GenerationStats};

use crate::http::{cors_header, error_json, json_response, respond_preflight, HttpResponse};

/// Server-process counters for `/v1/stats` (the engine snapshot carries the
/// per-generation numbers; these cover the HTTP surface).
struct ServeCounters {
    started: Instant,
    in_flight: AtomicU64,
}

/// Serves `engine` on `addr` (`host:port`) forever.
/// `default_prefill_chunk` overrides the engine's chunked-prefill size for
/// every request (from `combs serve --prefill-chunk-size`).
/// `static_info` is load-time identity for `/v1/stats` — `{weights, device}`,
/// built once in `cmd_serve` (device caps must be captured before the engine
/// initializes the cubecl runtime; see `cmd_serve`).
pub fn serve(
    engine: Arc<Engine>,
    model_id: String,
    addr: &str,
    default_prefill_chunk: Option<usize>,
    static_info: Value,
) -> Result<()> {
    let server = tiny_http::Server::http(addr).map_err(|e| anyhow::anyhow!("bind {addr}: {e}"))?;
    let counters = Arc::new(ServeCounters {
        started: Instant::now(),
        in_flight: AtomicU64::new(0),
    });
    let static_info = Arc::new(static_info);
    eprintln!("combs serve: listening on http://{addr} (model: {model_id})");

    for mut request in server.incoming_requests() {
        let engine = engine.clone();
        let model_id = model_id.clone();
        let counters = counters.clone();
        let static_info = static_info.clone();
        std::thread::spawn(move || {
            let url = request.url().to_string();
            let method = request.method().as_str().to_string();
            // CORS preflight — browsers probe before cross-origin POSTs.
            if method == "OPTIONS" {
                respond_preflight(request);
                return;
            }
            let response = match (method.as_str(), url.as_str()) {
                ("GET", "/health") => json_response(200, json!({"status": "ok"})),
                ("GET", "/v1/models") => json_response(
                    200,
                    json!({"object": "list", "data": [model_card(&engine, &model_id)]}),
                ),
                ("GET", "/v1/model/info") => model_info(&engine, &model_id),
                ("GET", "/v1/stats") => stats_response(&engine, &model_id, &counters, &static_info),
                ("GET", "/v1/sessions") => sessions_response(&engine),
                ("DELETE", p) if p == "/v1/sessions" || p.starts_with("/v1/sessions/") => {
                    clear_sessions_response(&engine, p)
                }
                ("POST", "/v1/chat/completions") => {
                    let mut body = String::new();
                    if request.as_reader().read_to_string(&mut body).is_err() {
                        json_response(400, error_json("invalid_request", "unreadable body"))
                    } else {
                        handle_chat(&engine, &model_id, &body, default_prefill_chunk, &counters)
                    }
                }
                ("POST", "/v1/completions") => {
                    let mut body = String::new();
                    if request.as_reader().read_to_string(&mut body).is_err() {
                        json_response(400, error_json("invalid_request", "unreadable body"))
                    } else {
                        handle_completions(
                            &engine,
                            &model_id,
                            &body,
                            default_prefill_chunk,
                            &counters,
                        )
                    }
                }
                ("POST", "/v1/embeddings") => {
                    let mut body = String::new();
                    if request.as_reader().read_to_string(&mut body).is_err() {
                        json_response(400, error_json("invalid_request", "unreadable body"))
                    } else {
                        handle_embeddings(&engine, &model_id, &body)
                    }
                }
                _ => json_response(404, error_json("not_found", "unknown endpoint")),
            };
            let _ = request.respond(response);
        });
    }
    Ok(())
}

/// `GET /v1/stats` — the engine's observability snapshot: rolling totals,
/// last-generation timings, GPU allocator state, live KV sessions, weight
/// identity, and build flags. Reads the worker-maintained snapshot; never
/// waits on generation.
fn stats_response(
    engine: &Arc<Engine>,
    model_id: &str,
    counters: &ServeCounters,
    static_info: &Value,
) -> HttpResponse {
    let snap = engine.stats_snapshot();
    let cc = engine.cache_config();
    let meta = engine.metadata();
    let hit_rate = if snap.prompt_tokens_total > 0 {
        snap.cached_tokens_total as f64 / snap.prompt_tokens_total as f64
    } else {
        0.0
    };
    let sessions: Vec<Value> = snap
        .sessions
        .iter()
        .map(|s| {
            json!({
                "id": s.id,
                "history_tokens": s.history_len,
                "pages": s.pages.map(|p| json!({
                    "used": p.pages_used,
                    "free": p.pages_free,
                    "total": p.num_pages,
                    "page_size": p.page_size,
                    "seq_len": p.seq_len,
                    "kv_bytes": p.pages_used as u64 * snap.kv_page_bytes,
                    // Arenas are allocated lazily, one per layer, so this
                    // is a measured count of layers actually holding KV —
                    // not a derivation from the layer total.
                    "layers_materialized": p.layers_materialized,
                    "layers_total": p.layers_total,
                    "layers_sliding": p.layers_sliding,
                })),
            })
        })
        .collect();
    json_response(
        200,
        json!({
            "object": "engine.stats",
            "model": model_id,
            "architecture": &meta.architecture,
            "uptime_s": counters.started.elapsed().as_secs(),
            "in_flight": counters.in_flight.load(Ordering::Relaxed),
            "totals": {
                "requests": snap.requests_total,
                "errors": snap.errors_total,
                "cancelled": snap.cancelled_total,
                "prompt_tokens": snap.prompt_tokens_total,
                "generated_tokens": snap.generated_tokens_total,
                "cached_tokens": snap.cached_tokens_total,
                "cache_hit_rate": hit_rate,
            },
            "throughput": {
                "decode_tok_s_ewma": snap.decode_tok_s_ewma,
                "last": snap.last.as_ref().map(|l| json!({
                    "prompt_tokens": l.prompt_tokens,
                    "generated_tokens": l.generated_tokens,
                    "cached_tokens": l.cached_tokens,
                    "ttft_ms": l.ttft_ms,
                    "prefill_tok_s": l.prefill_tok_s,
                    "decode_tok_s": l.decode_tok_s,
                    "total_ms": l.total_ms,
                    "cache_pages_used": l.cache_pages_used,
                })),
            },
            "gpu": snap.gpu.map(|g| json!({
                "bytes_in_use": g.bytes_in_use,
                "bytes_reserved": g.bytes_reserved,
                "bytes_padding": g.bytes_padding,
                "number_allocs": g.number_allocs,
            })),
            "kv": {
                "kind": format!("{:?}", cc.kind).to_lowercase(),
                "quantized": cc.quantize_kv,
                "max_seq_len": cc.max_seq_len,
                "page_size": cc.page_size,
                "page_bytes": snap.kv_page_bytes,
                "arena_bytes": cc.num_pages() as u64 * snap.kv_page_bytes,
                "bytes_per_token": snap.kv_page_bytes / cc.page_size.max(1) as u64,
                "max_sessions": snap.max_sessions,
                "evictions": snap.session_evictions,
                "sessions": sessions,
            },
            "attention": attention_json(engine),
            "weights": static_info.get("weights").cloned().unwrap_or(Value::Null),
            "device": static_info.get("device").cloned().unwrap_or(Value::Null),
            "build": {
                "dtype": if cfg!(feature = "f16") { "f16" } else { "f32" },
                "kv_env": std::env::var("COMBS_KV").ok(),
                "attn_env": std::env::var("COMBS_ATTN").ok(),
                "quant_kernels": std::env::var_os("COMBS_NO_QUANT_KERNELS").is_none(),
            },
        }),
    )
}

fn model_card(engine: &Arc<Engine>, model_id: &str) -> Value {
    json!({
        "id": model_id,
        "object": "model",
        "created": now_unix(),
        "owned_by": "combs",
        "context_length": engine.metadata().max_position_embeddings,
        "tools": engine.supports_tools(),
        "capabilities": capabilities_json(engine),
    })
}

/// What this engine+model can do, detected from the artifacts (template ⇒
/// tools, hidden-state path ⇒ embeddings, pooling config ⇒ pooling
/// default) and the build (constraints) — advertised so clients
/// self-configure instead of hardcoding per deployment.
fn capabilities_json(engine: &Arc<Engine>) -> Value {
    json!({
        "tools": engine.supports_tools(),
        "constraints": ["json_object", "json_schema"],
        "embeddings": engine.supports_embeddings(),
        "pooling": match engine.default_pooling() {
            combs_runtime::Pooling::Last => "last",
            combs_runtime::Pooling::Mean => "mean",
        },
        "logprobs": true,
        "min_p": true,
        "completions": true,
    })
}

/// `POST /v1/embeddings` — OpenAI shape: `input` string | [string] (≤ 64),
/// `encoding_format` float | base64, `dimensions` matryoshka truncation,
/// plus a `pooling` extension (`last` | `mean`) overriding the detected
/// checkpoint default.
fn handle_embeddings(engine: &Arc<Engine>, model_id: &str, body: &str) -> HttpResponse {
    let req: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            return json_response(
                400,
                error_json("invalid_request", &format!("bad JSON: {e}")),
            );
        }
    };
    if !engine.supports_embeddings() {
        return json_response(
            400,
            error_json(
                "invalid_request",
                "this model does not expose hidden states for embeddings",
            ),
        );
    }
    let texts: Vec<String> = match req.get("input") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => {
            let mut out = Vec::with_capacity(a.len());
            for v in a {
                match v.as_str() {
                    Some(s) => out.push(s.to_string()),
                    None => {
                        return json_response(
                            400,
                            error_json("invalid_request", "`input` array entries must be strings"),
                        );
                    }
                }
            }
            out
        }
        _ => {
            return json_response(
                400,
                error_json(
                    "invalid_request",
                    "`input` must be a string or an array of strings",
                ),
            );
        }
    };
    if texts.is_empty() || texts.len() > 64 {
        return json_response(
            400,
            error_json("invalid_request", "`input` must contain 1..=64 texts"),
        );
    }
    let mut opts = combs_runtime::EmbedOptions::default();
    if let Some(d) = req.get("dimensions").and_then(Value::as_u64) {
        opts.dimensions = Some(d as usize);
    }
    match req.get("pooling").and_then(Value::as_str) {
        None => {}
        Some("last") => opts.pooling = Some(combs_runtime::Pooling::Last),
        Some("mean") => opts.pooling = Some(combs_runtime::Pooling::Mean),
        Some(other) => {
            return json_response(
                400,
                error_json("invalid_request", &format!("unknown pooling: {other:?}")),
            );
        }
    }
    let as_base64 = match req.get("encoding_format").and_then(Value::as_str) {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return json_response(
                400,
                error_json(
                    "invalid_request",
                    &format!("unknown encoding_format: {other:?}"),
                ),
            );
        }
    };

    let out = match engine.embed_texts(&texts, &opts) {
        Ok(o) => o,
        Err(e) => {
            let (code, kind) = engine_error_parts(&e);
            return json_response(code, error_json(kind, &e.to_string()));
        }
    };
    let data: Vec<Value> = out
        .vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let embedding = if as_base64 {
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                json!(B64.encode(&bytes))
            } else {
                json!(v)
            };
            json!({"object": "embedding", "index": i, "embedding": embedding})
        })
        .collect();
    json_response(
        200,
        json!({
            "object": "list",
            "data": data,
            "model": model_id,
            "usage": {
                "prompt_tokens": out.prompt_tokens,
                "total_tokens": out.prompt_tokens,
            },
        }),
    )
}

/// OpenAI wire shape for completed tool calls (`arguments` re-serialized
/// to a JSON string).
fn tool_calls_json(calls: &[combs_runtime::ToolCall]) -> Value {
    Value::Array(
        calls
            .iter()
            .map(|c| {
                json!({
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": serde_json::to_string(&c.function.arguments)
                            .unwrap_or_else(|_| "{}".to_string()),
                    },
                })
            })
            .collect(),
    )
}

/// `GET /v1/sessions` — live KV sessions from the stats snapshot
/// (non-blocking; the worker refreshes it after every generation and
/// every clear).
fn sessions_response(engine: &Arc<Engine>) -> HttpResponse {
    let snap = engine.stats_snapshot();
    let data: Vec<Value> = snap
        .sessions
        .iter()
        .map(|s| {
            json!({
                "id": s.id,
                "history_tokens": s.history_len,
                "pages_used": s.pages.map(|p| p.pages_used),
            })
        })
        .collect();
    json_response(200, json!({ "object": "list", "data": data }))
}

/// `DELETE /v1/sessions` (all) / `DELETE /v1/sessions/{id}` — drops KV
/// sessions and frees their arenas. `(anonymous)` addresses the
/// empty-key session. Single-flight: waits behind a running generation.
fn clear_sessions_response(engine: &Arc<Engine>, path: &str) -> HttpResponse {
    let id = path
        .strip_prefix("/v1/sessions/")
        .filter(|s| !s.is_empty())
        .map(|s| if s == "(anonymous)" { "" } else { s });
    match engine.clear_sessions(id) {
        Ok(cleared) => json_response(200, json!({ "object": "sessions.cleared", "cleared": cleared })),
        Err(e) => json_response(500, error_json("engine_error", &e.to_string())),
    }
}

/// Compact per-layer window summary: 0 = global arena, w = sliding.
fn window_layers(windows: &[Option<usize>]) -> (Vec<usize>, usize) {
    let layers: Vec<usize> = windows.iter().map(|w| w.unwrap_or(0)).collect();
    let global = windows.iter().filter(|w| w.is_none()).count();
    (layers, global)
}

/// Attention identity: GQA head geometry plus the resolved per-layer
/// window layout. Static per checkpoint.
fn attention_json(engine: &Arc<Engine>) -> Value {
    let meta = engine.metadata();
    let windows = engine.attention_windows();
    let (layers, global_layers) = window_layers(&windows);
    json!({
        "num_attention_heads": meta.num_attention_heads,
        "num_key_value_heads": meta.num_key_value_heads,
        "head_dim": meta.head_dim,
        "num_layers": meta.num_hidden_layers,
        "gqa_ratio": meta.num_attention_heads / meta.num_key_value_heads.max(1),
        "sliding_window": meta.attention_pattern.sliding_window,
        "global_layers": global_layers,
        "layers": layers,
    })
}

/// Rich model descriptor: real context budget (KV arena), architecture,
/// and generation defaults — everything a client needs to size requests.
fn model_info(engine: &Arc<Engine>, model_id: &str) -> HttpResponse {
    let meta = engine.metadata();
    let cc = engine.cache_config();
    let dc = engine.default_config();
    json_response(
        200,
        json!({
            "id": model_id,
            "object": "model.info",
            "architecture": &meta.architecture,
            "context_length": meta.max_position_embeddings,
            "kv_cache": {
                "kind": format!("{:?}", cc.kind).to_lowercase(),
                "quantized": cc.quantize_kv,
                "max_seq_len": cc.max_seq_len,
                "page_size": cc.page_size,
            },
            "attention": attention_json(engine),
            "vocab_size": meta.vocab_size,
            "vision": meta.vision.is_some(),
            "tools": engine.supports_tools(),
            "capabilities": capabilities_json(engine),
            "defaults": {
                "max_tokens": dc.max_tokens,
                "temperature": dc.sampling.temperature,
                "top_p": dc.sampling.top_p,
                "top_k": dc.sampling.top_k,
            },
        }),
    )
}

/// The OpenAI-compatible `usage` object, extended with the timing and
/// cache stats the engine computes anyway (additive fields — existing
/// consumers that read only the token counts are unaffected). This is the
/// per-request half of the observability surface; `/v1/stats` is the
/// rolling half.
fn usage_json(stats: &GenerationStats, session_id: Option<&str>) -> Value {
    json!({
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.generated_tokens,
        "total_tokens": stats.prompt_tokens + stats.generated_tokens,
        "prompt_tokens_details": {"cached_tokens": stats.cached_tokens},
        "timing": {
            "ttft_ms": stats.ttft.as_secs_f64() * 1000.0,
            "prefill_tok_s": stats.prefill_tokens_per_second(),
            "decode_tok_s": stats.decode_tokens_per_second(),
            "total_ms": stats.total_time.as_secs_f64() * 1000.0,
        },
        "cache": {
            "pages_used": stats.cache_pages_used,
            "cached_tokens": stats.cached_tokens,
        },
        "session_id": session_id,
    })
}

/// Maps engine errors to (HTTP status, error type): context overflow is a
/// client-fixable 400, everything else stays a 500.
fn engine_error_parts(e: &combs_runtime::EngineError) -> (u16, &'static str) {
    match e {
        combs_runtime::EngineError::ContextTooLong { .. } => (400, "context_length_exceeded"),
        combs_runtime::EngineError::Constraint(_) => (400, "constraint_error"),
        combs_runtime::EngineError::Model(combs_models::ModelError::Unsupported(_)) => {
            (400, "invalid_request")
        }
        _ => (500, "engine_error"),
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Applies the sampling/stop/constraint request fields shared by the chat
/// and completions endpoints. `Err` is a client-fixable message (400).
fn apply_sampling_params(req: &Value, config: &mut GenerationConfig) -> Result<(), String> {
    // `max_completion_tokens` is the current OpenAI name; `max_tokens` the
    // legacy alias (the explicit new name wins when both are present).
    if let Some(mt) = req
        .get("max_completion_tokens")
        .or_else(|| req.get("max_tokens"))
        .and_then(Value::as_u64)
    {
        config.max_tokens = mt as usize;
    }
    if let Some(t) = req.get("temperature").and_then(Value::as_f64) {
        config.sampling.temperature = t as f32;
    }
    if let Some(k) = req.get("top_k").and_then(Value::as_u64) {
        config.sampling.top_k = Some(k as usize);
    }
    if let Some(p) = req.get("top_p").and_then(Value::as_f64) {
        config.sampling.top_p = Some(p as f32);
    }
    if let Some(p) = req.get("min_p").and_then(Value::as_f64) {
        config.sampling.min_p = Some(p as f32);
    }
    if let Some(rp) = req.get("repetition_penalty").and_then(Value::as_f64) {
        config.sampling.repetition_penalty = Some(rp as f32);
    }
    if let Some(fp) = req.get("frequency_penalty").and_then(Value::as_f64) {
        config.sampling.frequency_penalty = Some(fp as f32);
    }
    if let Some(pp) = req.get("presence_penalty").and_then(Value::as_f64) {
        config.sampling.presence_penalty = Some(pp as f32);
    }
    // Penalty window (llama.cpp's name for it). Negative or 0 means the
    // whole context — the unbounded behavior that suppresses common words
    // and the stop token itself in long chats. Omitted = engine default.
    if let Some(n) = req.get("repeat_last_n").and_then(Value::as_i64) {
        config.sampling.penalty_last_n = Some(if n <= 0 { 0 } else { n as usize });
    }
    if let Some(seed) = req.get("seed").and_then(Value::as_u64) {
        config.sampling.seed = Some(seed);
    }
    if let Some(bias) = req.get("logit_bias") {
        let obj = bias
            .as_object()
            .ok_or("logit_bias must be an object of token-id strings to numbers")?;
        let mut map = std::collections::HashMap::with_capacity(obj.len());
        for (k, v) in obj {
            let id: u32 = k
                .parse()
                .map_err(|_| format!("logit_bias key is not a token id: {k:?}"))?;
            let b = v
                .as_f64()
                .ok_or_else(|| format!("logit_bias value for {k:?} is not a number"))?;
            map.insert(id, b as f32);
        }
        config.sampling.logit_bias = Some(map);
    }
    if let Some(stop) = req.get("stop") {
        let stops: Vec<String> = match stop {
            Value::String(s) => vec![s.clone()],
            Value::Array(a) => a
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect(),
            _ => vec![],
        };
        config.stop_strings = stops;
    }
    // Structured output: parse AND compile the schema here so malformed
    // requests are clean 400s before anything is enqueued on the worker.
    if let Some(rf) = req.get("response_format") {
        config.constraint = combs_runtime::ConstraintSpec::from_response_format(rf)
            .and_then(|spec| {
                if let Some(s) = &spec {
                    s.compile()?;
                }
                Ok(spec)
            })
            .map_err(|msg| format!("response_format: {msg}"))?;
    }
    Ok(())
}

/// Multi-choice count: OpenAI `n`, capped at 4 sequential generations.
fn parse_n_choices(req: &Value, stream: bool) -> Result<usize, String> {
    let n = req.get("n").and_then(Value::as_u64).unwrap_or(1) as usize;
    if n == 0 || n > 4 {
        return Err("n must be 1..=4".to_string());
    }
    if stream && n > 1 {
        return Err("n > 1 is not supported with stream".to_string());
    }
    Ok(n)
}

/// Per-choice sampling config: distinct seeds keep multinomial choices
/// distinct while an explicit seed stays reproducible per index.
fn choice_config(config: &GenerationConfig, index: usize) -> GenerationConfig {
    let mut cfg = config.clone();
    if index > 0 {
        if let Some(seed) = cfg.sampling.seed {
            cfg.sampling.seed = Some(seed.wrapping_add(index as u64));
        }
    }
    cfg
}

/// One OpenAI `logprobs.content[]` entry for a sampled token.
fn logprob_entry_json(
    engine: &Arc<Engine>,
    piece: &str,
    lp: &combs_runtime::TokenLogprobs,
) -> Value {
    let top: Vec<Value> = lp
        .top
        .iter()
        .map(|(id, l)| {
            let text = engine.token_piece(*id);
            json!({"token": text, "logprob": l, "bytes": text.as_bytes()})
        })
        .collect();
    json!({
        "token": piece,
        "logprob": lp.logprob,
        "bytes": piece.as_bytes(),
        "top_logprobs": top,
    })
}

/// `usage` for multi-choice responses: prompt counted once, completions
/// summed. Single-choice keeps the full timing/cache detail.
fn usage_json_choices(all: &[GenerationStats], session_id: Option<&str>) -> Value {
    if all.len() == 1 {
        return usage_json(&all[0], session_id);
    }
    let prompt = all.first().map(|s| s.prompt_tokens).unwrap_or(0);
    let completion: usize = all.iter().map(|s| s.generated_tokens).sum();
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {
            "cached_tokens": all.first().map(|s| s.cached_tokens).unwrap_or(0),
        },
        "session_id": session_id,
    })
}

fn handle_chat(
    engine: &Arc<Engine>,
    model_id: &str,
    body: &str,
    default_prefill_chunk: Option<usize>,
    counters: &Arc<ServeCounters>,
) -> HttpResponse {
    let req: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            return json_response(
                400,
                error_json("invalid_request", &format!("bad JSON: {e}")),
            );
        }
    };

    let messages = req
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if messages.is_empty() {
        return json_response(
            400,
            error_json("invalid_request", "`messages` must be a non-empty array"),
        );
    }
    // OpenAI tools: pass schemas through to the chat template ("auto"),
    // drop them on tool_choice "none", reject on models whose template
    // can't express tools (rendering would silently omit them).
    let tool_choice_none = req
        .get("tool_choice")
        .and_then(Value::as_str)
        .is_some_and(|c| c == "none");
    let tools: Option<Value> = req
        .get("tools")
        .filter(|t| t.as_array().is_some_and(|a| !a.is_empty()) && !tool_choice_none)
        .cloned();
    if tools.is_some() && !engine.supports_tools() {
        return json_response(
            400,
            error_json(
                "invalid_request",
                "this model's chat template has no tool support — use a \
                 tool-trained model (qwen2.5/qwen3/llama-3.x)",
            ),
        );
    }
    let (prompt, images) = match build_prompt(engine, &messages, tools.as_ref()) {
        Ok(v) => v,
        Err(msg) => {
            return json_response(400, error_json("invalid_request", &msg));
        }
    };

    let tokens = match engine.encode(&prompt) {
        Ok(t) => t,
        Err(e) => {
            return json_response(
                500,
                error_json("engine_error", &format!("tokenization failed: {e}")),
            );
        }
    };

    let mut config: GenerationConfig = engine.default_config();
    if let Some(chunk) = default_prefill_chunk {
        config.prefill_chunk_size = chunk;
    }
    if let Err(msg) = apply_sampling_params(&req, &mut config) {
        return json_response(400, error_json("invalid_request", &msg));
    }
    // OpenAI chat logprobs: `logprobs: true` + `top_logprobs` 0..=20.
    if req.get("logprobs").and_then(Value::as_bool) == Some(true) {
        let n = req
            .get("top_logprobs")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .min(20) as usize;
        config.sampling.logprobs = Some(n);
    }
    if let Some(id) = engine.im_end_id() {
        config.stop_token_ids.push(id);
    }
    if let Some(id) = engine.end_turn_id() {
        config.stop_token_ids.push(id);
    }
    // Optional named KV session (e.g. one per debate agent) — requests with
    // the same id share a rolling prefix-reuse session.
    if let Some(sid) = req.get("session_id").and_then(Value::as_str) {
        let sid: String = sid.chars().take(64).collect();
        if !sid.is_empty() {
            config.session_id = Some(sid);
        }
    }

    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let n_choices = match parse_n_choices(&req, stream) {
        Ok(n) => n,
        Err(msg) => return json_response(400, error_json("invalid_request", &msg)),
    };
    let completion_id = format!("chatcmpl-combs-{}", now_unix());
    // The parser engages only when the request carries tools — tool-less
    // requests stream byte-identically to before.
    let parser_style = if tools.is_some() {
        engine.tool_call_style()
    } else {
        combs_runtime::ToolCallStyle::None
    };

    if !stream {
        let want_logprobs = config.sampling.logprobs.is_some();
        let mut choices: Vec<Value> = Vec::with_capacity(n_choices);
        let mut all_stats: Vec<GenerationStats> = Vec::with_capacity(n_choices);
        counters.in_flight.fetch_add(1, Ordering::Relaxed);
        for index in 0..n_choices {
            let cfg = choice_config(&config, index);
            let mut parser = combs_runtime::ToolCallParser::new(parser_style);
            let mut text = String::new();
            let mut calls: Vec<combs_runtime::ToolCall> = Vec::new();
            let mut lp_entries: Vec<Value> = Vec::new();
            let result = generate_maybe_media(
                engine,
                &tokens,
                images.clone(),
                &cfg,
                Arc::new(AtomicBool::new(false)),
                |_id, piece, lp| {
                    if let Some(lp) = lp {
                        lp_entries.push(logprob_entry_json(engine, piece, lp));
                    }
                    for ev in parser.push(piece) {
                        match ev {
                            combs_runtime::ToolEvent::Content(c) => text.push_str(&c),
                            combs_runtime::ToolEvent::Call(c) => calls.push(c),
                        }
                    }
                },
            );
            for ev in parser.finish() {
                match ev {
                    combs_runtime::ToolEvent::Content(c) => text.push_str(&c),
                    combs_runtime::ToolEvent::Call(c) => calls.push(c),
                }
            }
            let stats = match result {
                Ok(stats) => stats,
                Err(e) => {
                    counters.in_flight.fetch_sub(1, Ordering::Relaxed);
                    let (status, kind) = engine_error_parts(&e);
                    return json_response(status, error_json(kind, &e.to_string()));
                }
            };
            let finish = if stats.generated_tokens >= cfg.max_tokens {
                "length"
            } else if !calls.is_empty() {
                "tool_calls"
            } else {
                "stop"
            };
            let mut message = json!({"role": "assistant", "content": text});
            if !calls.is_empty() {
                if text.trim().is_empty() {
                    message["content"] = Value::Null;
                }
                message["tool_calls"] = tool_calls_json(&calls);
            }
            let logprobs = if want_logprobs {
                json!({"content": lp_entries})
            } else {
                Value::Null
            };
            choices.push(json!({
                "index": index,
                "message": message,
                "logprobs": logprobs,
                "finish_reason": finish,
            }));
            all_stats.push(stats);
        }
        counters.in_flight.fetch_sub(1, Ordering::Relaxed);
        return json_response(
            200,
            json!({
                "id": completion_id,
                "object": "chat.completion",
                "created": now_unix(),
                "model": model_id,
                "choices": choices,
                "usage": usage_json_choices(&all_stats, config.session_id.as_deref()),
            }),
        );
    }

    // SSE: channel-backed reader fed by the engine callback thread.
    let (tx, rx) = mpsc::channel::<String>();
    let engine = engine.clone();
    let model_id = model_id.to_string();
    let chunk_id = completion_id.clone();
    let counters = counters.clone();
    std::thread::spawn(move || {
        counters.in_flight.fetch_add(1, Ordering::Relaxed);
        // Client-disconnect abort: a failed SSE send means the socket is
        // gone — flag the engine to stop at its next decode step.
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_flag = cancel.clone();
        let send_chunk = |delta: Value,
                          finish: Option<&str>,
                          usage: Option<Value>,
                          logprobs: Option<Value>|
         -> bool {
            let mut chunk = json!({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": now_unix(),
                "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            });
            if let Some(lp) = logprobs {
                chunk["choices"][0]["logprobs"] = lp;
            }
            if let Some(u) = usage {
                chunk["usage"] = u;
            }
            let ok = tx.send(format!("data: {chunk}\n\n")).is_ok();
            if !ok {
                cancel_flag.store(true, Ordering::Relaxed);
            }
            ok
        };
        let _ = send_chunk(json!({"role": "assistant"}), None, None, None);
        let mut parser = combs_runtime::ToolCallParser::new(parser_style);
        let mut call_index = 0usize;
        let mut emit_event = |ev: combs_runtime::ToolEvent| match ev {
            combs_runtime::ToolEvent::Content(c) => {
                let _ = send_chunk(json!({"content": c}), None, None, None);
            }
            combs_runtime::ToolEvent::Call(c) => {
                // OpenAI wire: arguments as a JSON *string*; one complete
                // call per chunk.
                let args = serde_json::to_string(&c.function.arguments)
                    .unwrap_or_else(|_| "{}".to_string());
                let _ = send_chunk(
                    json!({"tool_calls": [{
                        "index": call_index,
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.function.name, "arguments": args},
                    }]}),
                    None,
                    None,
                    None,
                );
                call_index += 1;
            }
        };
        let result = generate_maybe_media(
            &engine,
            &tokens,
            images,
            &config,
            cancel,
            |_id, piece, lp| {
                for ev in parser.push(piece) {
                    emit_event(ev);
                }
                // Logprob entries ride their own chunk so tool-call
                // buffering can never displace them.
                if let Some(lp) = lp {
                    let entry = json!({"content": [logprob_entry_json(&engine, piece, lp)]});
                    let _ = send_chunk(json!({}), None, None, Some(entry));
                }
            },
        );
        for ev in parser.finish() {
            emit_event(ev);
        }
        match result {
            Ok(stats) => {
                let finish = if stats.generated_tokens >= config.max_tokens {
                    "length"
                } else if parser.calls_seen() > 0 {
                    "tool_calls"
                } else {
                    "stop"
                };
                let usage = usage_json(&stats, config.session_id.as_deref());
                let _ = send_chunk(json!({}), Some(finish), Some(usage), None);
            }
            Err(combs_runtime::EngineError::Cancelled) => {
                // Client hung up: the worker stopped within one decode step
                // and the session snapshot is already consistent. No error
                // frame (nobody is listening) — but loud on stderr, because
                // a silent abort would hide client-side bugs.
                eprintln!(
                    "combs serve: [cancelled] chat stream {chunk_id} — client disconnected, generation aborted"
                );
            }
            Err(e) => {
                let (_, kind) = engine_error_parts(&e);
                let _ = tx.send(format!(
                    "data: {}\n\n",
                    json!({"error": {"type": kind, "message": e.to_string()}})
                ));
            }
        }
        counters.in_flight.fetch_sub(1, Ordering::Relaxed);
        let _ = tx.send("data: [DONE]\n\n".to_string());
    });

    tiny_http::Response::empty(200)
        .with_data(
            Box::new(ChannelReader::new(rx)) as Box<dyn Read + Send>,
            None,
        )
        .with_header(tiny_http::Header::from_bytes("Content-Type", "text/event-stream").unwrap())
        .with_header(tiny_http::Header::from_bytes("Cache-Control", "no-cache").unwrap())
        .with_header(cors_header())
}

/// `POST /v1/completions` — the legacy raw-prompt endpoint: no chat
/// template, no turn stops, same sampling surface as chat. Unlocks
/// client-side FIM for coder models. `logprobs` here is the legacy int
/// form (top-n; presence turns capture on).
fn handle_completions(
    engine: &Arc<Engine>,
    model_id: &str,
    body: &str,
    default_prefill_chunk: Option<usize>,
    counters: &Arc<ServeCounters>,
) -> HttpResponse {
    let req: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            return json_response(
                400,
                error_json("invalid_request", &format!("bad JSON: {e}")),
            );
        }
    };
    let Some(prompt) = req.get("prompt").and_then(Value::as_str) else {
        return json_response(
            400,
            error_json("invalid_request", "`prompt` must be a string"),
        );
    };

    let mut config: GenerationConfig = engine.default_config();
    if let Some(chunk) = default_prefill_chunk {
        config.prefill_chunk_size = chunk;
    }
    if let Err(msg) = apply_sampling_params(&req, &mut config) {
        return json_response(400, error_json("invalid_request", &msg));
    }
    // Same named-session semantics as chat completions — raw-prompt
    // callers were previously forced onto the anonymous session.
    if let Some(sid) = req.get("session_id").and_then(Value::as_str) {
        let sid: String = sid.chars().take(64).collect();
        if !sid.is_empty() {
            config.session_id = Some(sid);
        }
    }
    if let Some(n) = req.get("logprobs").and_then(Value::as_u64) {
        config.sampling.logprobs = Some(n.min(20) as usize);
    }
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let n_choices = match parse_n_choices(&req, stream) {
        Ok(n) => n,
        Err(msg) => return json_response(400, error_json("invalid_request", &msg)),
    };

    let tokens = match engine.encode(prompt) {
        Ok(t) => t,
        Err(e) => {
            return json_response(
                500,
                error_json("engine_error", &format!("tokenization failed: {e}")),
            );
        }
    };
    let completion_id = format!("cmpl-combs-{}", now_unix());

    if !stream {
        let want_logprobs = config.sampling.logprobs.is_some();
        let mut choices: Vec<Value> = Vec::with_capacity(n_choices);
        let mut all_stats: Vec<GenerationStats> = Vec::with_capacity(n_choices);
        counters.in_flight.fetch_add(1, Ordering::Relaxed);
        for index in 0..n_choices {
            let cfg = choice_config(&config, index);
            let mut text = String::new();
            let mut lp = LegacyLogprobs::default();
            let result = engine.generate(&tokens, &cfg, |_id, piece, token_lp| {
                if let Some(l) = token_lp {
                    lp.push(engine, &text, piece, l);
                }
                text.push_str(piece);
            });
            let stats = match result {
                Ok(stats) => stats,
                Err(e) => {
                    counters.in_flight.fetch_sub(1, Ordering::Relaxed);
                    let (status, kind) = engine_error_parts(&e);
                    return json_response(status, error_json(kind, &e.to_string()));
                }
            };
            let finish = if stats.generated_tokens >= cfg.max_tokens {
                "length"
            } else {
                "stop"
            };
            choices.push(json!({
                "index": index,
                "text": text,
                "logprobs": if want_logprobs { lp.into_json() } else { Value::Null },
                "finish_reason": finish,
            }));
            all_stats.push(stats);
        }
        counters.in_flight.fetch_sub(1, Ordering::Relaxed);
        return json_response(
            200,
            json!({
                "id": completion_id,
                "object": "text_completion",
                "created": now_unix(),
                "model": model_id,
                "choices": choices,
                "usage": usage_json_choices(&all_stats, None),
            }),
        );
    }

    // SSE: text_completion chunks, one per piece.
    let (tx, rx) = mpsc::channel::<String>();
    let engine = engine.clone();
    let model_id = model_id.to_string();
    let counters = counters.clone();
    std::thread::spawn(move || {
        counters.in_flight.fetch_add(1, Ordering::Relaxed);
        // Same client-disconnect abort contract as the chat stream.
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_flag = cancel.clone();
        let send_chunk = |text: &str, finish: Option<&str>, usage: Option<Value>| -> bool {
            let mut chunk = json!({
                "id": completion_id,
                "object": "text_completion",
                "created": now_unix(),
                "model": model_id,
                "choices": [{"index": 0, "text": text, "finish_reason": finish}],
            });
            if let Some(u) = usage {
                chunk["usage"] = u;
            }
            let ok = tx.send(format!("data: {chunk}\n\n")).is_ok();
            if !ok {
                cancel_flag.store(true, Ordering::Relaxed);
            }
            ok
        };
        let result = engine.generate_cancellable(&tokens, &config, cancel, |_id, piece, _lp| {
            let _ = send_chunk(piece, None, None);
        });
        match result {
            Ok(stats) => {
                let finish = if stats.generated_tokens >= config.max_tokens {
                    "length"
                } else {
                    "stop"
                };
                let _ = send_chunk("", Some(finish), Some(usage_json(&stats, None)));
            }
            Err(combs_runtime::EngineError::Cancelled) => {
                // Client hung up — no error frame, session already
                // consistent; loud on stderr so aborts are never invisible.
                eprintln!(
                    "combs serve: [cancelled] completion stream {completion_id} — client disconnected, generation aborted"
                );
            }
            Err(e) => {
                let (_, kind) = engine_error_parts(&e);
                let _ = tx.send(format!(
                    "data: {}\n\n",
                    json!({"error": {"type": kind, "message": e.to_string()}})
                ));
            }
        }
        counters.in_flight.fetch_sub(1, Ordering::Relaxed);
        let _ = tx.send("data: [DONE]\n\n".to_string());
    });

    tiny_http::Response::empty(200)
        .with_data(
            Box::new(ChannelReader::new(rx)) as Box<dyn Read + Send>,
            None,
        )
        .with_header(tiny_http::Header::from_bytes("Content-Type", "text/event-stream").unwrap())
        .with_header(tiny_http::Header::from_bytes("Cache-Control", "no-cache").unwrap())
        .with_header(cors_header())
}

/// Accumulator for the legacy completions `logprobs` object
/// (tokens / token_logprobs / top_logprobs / text_offset arrays).
#[derive(Default)]
struct LegacyLogprobs {
    tokens: Vec<Value>,
    token_logprobs: Vec<Value>,
    top_logprobs: Vec<Value>,
    text_offset: Vec<Value>,
}

impl LegacyLogprobs {
    fn push(
        &mut self,
        engine: &Arc<Engine>,
        text_so_far: &str,
        piece: &str,
        lp: &combs_runtime::TokenLogprobs,
    ) {
        self.tokens.push(json!(piece));
        self.token_logprobs.push(json!(lp.logprob));
        let mut alts = serde_json::Map::new();
        for (id, l) in &lp.top {
            alts.insert(engine.token_piece(*id), json!(l));
        }
        self.top_logprobs.push(Value::Object(alts));
        self.text_offset.push(json!(text_so_far.len()));
    }

    fn into_json(self) -> Value {
        json!({
            "tokens": self.tokens,
            "token_logprobs": self.token_logprobs,
            "top_logprobs": self.top_logprobs,
            "text_offset": self.text_offset,
        })
    }
}

/// Route to the media-aware engine path when the request carries images
/// (image turns bypass the KV session cache engine-side by design).
fn generate_maybe_media(
    engine: &Arc<Engine>,
    tokens: &[u32],
    images: Vec<PixelBatch>,
    config: &GenerationConfig,
    cancel: Arc<AtomicBool>,
    on_token: impl FnMut(u32, &str, Option<&combs_runtime::TokenLogprobs>),
) -> combs_runtime::Result<GenerationStats> {
    if images.is_empty() {
        engine.generate_cancellable(tokens, config, cancel, on_token)
    } else {
        engine.generate_media_cancellable(tokens, images, config, cancel, on_token)
    }
}

/// Builds the chat prompt in the model's own template (ChatML or Gemma
/// turns, via [`Engine::wrap_chat`]), extracting OpenAI array content
/// parts. `image_url` parts (base64 data: URLs) are decoded and
/// preprocessed in order; each image's `<image>`-token span is spliced
/// into the message text where the part appears (spans before the
/// question, matching the `combs run --image` convention). Returns
/// (prompt, pixel_batches).
fn build_prompt(
    engine: &Arc<Engine>,
    messages: &[Value],
    tools: Option<&Value>,
) -> Result<(String, Vec<PixelBatch>), String> {
    let vision = engine.metadata().vision.clone();
    let mut images: Vec<PixelBatch> = Vec::new();
    let mut msgs: Vec<combs_runtime::ChatMessage> = Vec::with_capacity(messages.len());
    for m in messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        // Tool-protocol roles pass through — templates dispatch on them
        // (llama's tool results use "ipython"); only unknown roles coerce.
        let role = match role {
            "system" | "user" | "assistant" | "tool" | "ipython" => role,
            _ => "user",
        };
        let mut content = String::new();
        match m.get("content") {
            Some(Value::String(s)) => content = s.clone(),
            Some(Value::Array(parts)) => {
                for part in parts {
                    match part.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            if let Some(t) = part.get("text").and_then(Value::as_str) {
                                content.push_str(t);
                            }
                        }
                        Some("image_url") => {
                            let url = part
                                .get("image_url")
                                .and_then(|u| u.get("url"))
                                .and_then(Value::as_str)
                                .unwrap_or("");
                            let v = vision.as_ref().ok_or(
                                "this model has no vision tower — use a vision model (e.g. smolvlm) for image input",
                            )?;
                            let bytes = decode_data_image(url)?;
                            let batch = SiglipPreprocessor::new(v.image_size)
                                .preprocess(&bytes)
                                .map_err(|e| format!("image preprocessing failed: {e}"))?;
                            images.push(batch);
                            content.push_str(&image_prompt_expansion(v.image_seq_len()));
                        }
                        _ => {} // unknown parts ignored (forward-compat)
                    }
                }
            }
            _ => {}
        }
        // OpenAI wire fields beyond role/content: assistant tool_calls
        // (arguments arrive as JSON strings — ChatMessage normalizes them
        // to objects for the template, the HF convention) and tool-result
        // correlation ids.
        let tool_calls: Vec<combs_runtime::ToolCall> = m
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(|calls| {
                calls
                    .iter()
                    .filter_map(|c| serde_json::from_value(c.clone()).ok())
                    .collect()
            })
            .unwrap_or_default();
        msgs.push(combs_runtime::ChatMessage {
            role: role.to_string(),
            content,
            tool_calls,
            tool_call_id: m
                .get("tool_call_id")
                .and_then(Value::as_str)
                .map(str::to_string),
            name: m.get("name").and_then(Value::as_str).map(str::to_string),
        });
    }
    // Vision models (Idefics3/SmolVLM) were trained on a specific
    // `<|im_start|>User:<image-tokens>prompt\nAssistant:` format with no
    // `<|im_end|>` separators. The generic `wrap_chat` ChatML path works
    // for text-only models but degrades vision output, so we build the
    // vision prompt explicitly when images are present.
    let prompt = if images.is_empty() {
        engine.wrap_chat_with_tools(&msgs, tools)
    } else {
        let mut prompt = String::from("<|im_start|>User:");
        for m in &msgs {
            match m.role.as_str() {
                "system" => {
                    prompt.push_str(&m.content);
                    prompt.push('\n');
                }
                "user" => prompt.push_str(&m.content),
                "assistant" => {
                    prompt.push_str("\n<|im_start|>Assistant: ");
                    prompt.push_str(&m.content);
                    prompt.push('\n');
                }
                _ => {}
            }
        }
        prompt.push_str("\nAssistant:");
        prompt
    };
    Ok((prompt, images))
}

/// Decodes a `data:<mime>;base64,<payload>` image URL. Remote http(s)
/// URLs are rejected — fetch client-side and inline as a data URL.
fn decode_data_image(url: &str) -> Result<Vec<u8>, String> {
    let payload = url
        .strip_prefix("data:")
        .and_then(|rest| rest.split_once(',').map(|(_, data)| data))
        .ok_or("image_url must be a base64 data: URL (fetch remote URLs client-side)")?;
    B64.decode(payload.trim())
        .map_err(|e| format!("bad image base64: {e}"))
}

/// `Read` over an mpsc channel of SSE strings: blocks until the next chunk
/// arrives; EOF when the sender closes (after `[DONE]`).
struct ChannelReader {
    rx: mpsc::Receiver<String>,
    buf: std::collections::VecDeque<u8>,
}

impl ChannelReader {
    fn new(rx: mpsc::Receiver<String>) -> Self {
        ChannelReader {
            rx,
            buf: std::collections::VecDeque::new(),
        }
    }
}

impl Read for ChannelReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        while self.buf.is_empty() {
            match self.rx.recv() {
                Ok(chunk) => self.buf.extend(chunk.as_bytes()),
                Err(_) => return Ok(0), // channel closed: EOF
            }
        }
        let n = out.len().min(self.buf.len());
        for slot in out.iter_mut().take(n) {
            *slot = self.buf.pop_front().expect("buf non-empty");
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::window_layers;

    #[test]
    fn window_layer_summary_encodes_global_as_zero() {
        let (layers, global) = window_layers(&[None, Some(512), None, Some(512)]);
        assert_eq!(layers, vec![0, 512, 0, 512]);
        assert_eq!(global, 2);
    }

    #[test]
    fn window_layer_summary_all_global() {
        let (layers, global) = window_layers(&[None; 3]);
        assert_eq!(layers, vec![0, 0, 0]);
        assert_eq!(global, 3);
    }
}
