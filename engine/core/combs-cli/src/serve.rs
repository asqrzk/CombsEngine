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
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use serde_json::{Value, json};

use combs_media::{ImagePreprocessor, PixelBatch, SiglipPreprocessor};
use combs_models::image_prompt_expansion;
use combs_runtime::{Engine, GenerationConfig, GenerationStats};

use crate::http::{HttpResponse, cors_header, error_json, json_response, respond_preflight};

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
                ("GET", "/v1/stats") => {
                    stats_response(&engine, &model_id, &counters, &static_info)
                }
                ("POST", "/v1/chat/completions") => {
                    let mut body = String::new();
                    if request.as_reader().read_to_string(&mut body).is_err() {
                        json_response(400, error_json("invalid_request", "unreadable body"))
                    } else {
                        handle_chat(&engine, &model_id, &body, default_prefill_chunk, &counters)
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
                "max_sessions": snap.max_sessions,
                "evictions": snap.session_evictions,
                "sessions": sessions,
            },
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
    })
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
                "max_seq_len": cc.max_seq_len,
                "page_size": cc.page_size,
            },
            "vocab_size": meta.vocab_size,
            "vision": meta.vision.is_some(),
            "tools": engine.supports_tools(),
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
        combs_runtime::EngineError::ContextTooLong { .. } => {
            (400, "context_length_exceeded")
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
    if let Some(mt) = req.get("max_tokens").and_then(Value::as_u64) {
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
    if let Some(rp) = req.get("repetition_penalty").and_then(Value::as_f64) {
        config.sampling.repetition_penalty = Some(rp as f32);
    }
    if let Some(fp) = req.get("frequency_penalty").and_then(Value::as_f64) {
        config.sampling.frequency_penalty = Some(fp as f32);
    }
    if let Some(pp) = req.get("presence_penalty").and_then(Value::as_f64) {
        config.sampling.presence_penalty = Some(pp as f32);
    }
    if let Some(seed) = req.get("seed").and_then(Value::as_u64) {
        config.sampling.seed = Some(seed);
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
    let completion_id = format!("chatcmpl-combs-{}", now_unix());
    // The parser engages only when the request carries tools — tool-less
    // requests stream byte-identically to before.
    let parser_style = if tools.is_some() {
        engine.tool_call_style()
    } else {
        combs_runtime::ToolCallStyle::None
    };

    if !stream {
        let mut parser = combs_runtime::ToolCallParser::new(parser_style);
        let mut text = String::new();
        let mut calls: Vec<combs_runtime::ToolCall> = Vec::new();
        counters.in_flight.fetch_add(1, Ordering::Relaxed);
        let result = generate_maybe_media(engine, &tokens, images, &config, |_id, piece| {
            for ev in parser.push(piece) {
                match ev {
                    combs_runtime::ToolEvent::Content(c) => text.push_str(&c),
                    combs_runtime::ToolEvent::Call(c) => calls.push(c),
                }
            }
        });
        for ev in parser.finish() {
            match ev {
                combs_runtime::ToolEvent::Content(c) => text.push_str(&c),
                combs_runtime::ToolEvent::Call(c) => calls.push(c),
            }
        }
        counters.in_flight.fetch_sub(1, Ordering::Relaxed);
        return match result {
            Ok(stats) => {
                let finish = if stats.generated_tokens >= config.max_tokens {
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
                json_response(
                    200,
                    json!({
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": now_unix(),
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "message": message,
                            "finish_reason": finish,
                        }],
                        "usage": usage_json(&stats, config.session_id.as_deref()),
                    }),
                )
            }
            Err(e) => {
                let (status, kind) = engine_error_parts(&e);
                json_response(status, error_json(kind, &e.to_string()))
            }
        };
    }

    // SSE: channel-backed reader fed by the engine callback thread.
    let (tx, rx) = mpsc::channel::<String>();
    let engine = engine.clone();
    let model_id = model_id.to_string();
    let chunk_id = completion_id.clone();
    let counters = counters.clone();
    std::thread::spawn(move || {
        counters.in_flight.fetch_add(1, Ordering::Relaxed);
        let send_chunk = |delta: Value, finish: Option<&str>, usage: Option<Value>| -> bool {
            let mut chunk = json!({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": now_unix(),
                "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            });
            if let Some(u) = usage {
                chunk["usage"] = u;
            }
            tx.send(format!("data: {chunk}\n\n")).is_ok()
        };
        let _ = send_chunk(json!({"role": "assistant"}), None, None);
        let mut parser = combs_runtime::ToolCallParser::new(parser_style);
        let mut call_index = 0usize;
        let mut emit_event = |ev: combs_runtime::ToolEvent| match ev {
            combs_runtime::ToolEvent::Content(c) => {
                let _ = send_chunk(json!({"content": c}), None, None);
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
                );
                call_index += 1;
            }
        };
        let result = generate_maybe_media(&engine, &tokens, images, &config, |_id, piece| {
            for ev in parser.push(piece) {
                emit_event(ev);
            }
        });
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
                let _ = send_chunk(json!({}), Some(finish), Some(usage));
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
        .with_data(Box::new(ChannelReader::new(rx)) as Box<dyn Read + Send>, None)
        .with_header(tiny_http::Header::from_bytes("Content-Type", "text/event-stream").unwrap())
        .with_header(tiny_http::Header::from_bytes("Cache-Control", "no-cache").unwrap())
        .with_header(cors_header())
}

/// Route to the media-aware engine path when the request carries images
/// (image turns bypass the KV session cache engine-side by design).
fn generate_maybe_media(
    engine: &Arc<Engine>,
    tokens: &[u32],
    images: Vec<PixelBatch>,
    config: &GenerationConfig,
    on_token: impl FnMut(u32, &str),
) -> combs_runtime::Result<GenerationStats> {
    if images.is_empty() {
        engine.generate(tokens, config, on_token)
    } else {
        engine.generate_with_media(tokens, images, config, on_token)
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
