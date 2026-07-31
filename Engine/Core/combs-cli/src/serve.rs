//! `combs serve` — OpenAI-compatible HTTP + SSE server over the local engine.
//!
//! Endpoints:
//! - `GET  /health` → `{"status":"ok"}`
//! - `GET  /v1/models` → the loaded model id
//! - `POST /v1/chat/completions` → OpenAI chat completion, `stream: true`
//!   for SSE chunks terminated by `data: [DONE]`.
//!
//! tiny_http is synchronous and lightweight (no async runtime); the
//! engine's single-flight queue serializes concurrent requests. SSE
//! streaming is a channel-backed `Read` fed by the engine's token callback.

use std::io::Read;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde_json::{Value, json};

use combs_runtime::{Engine, GenerationConfig};

/// Response type used everywhere: a boxed reader so streaming and buffered
/// responses share one concrete type.
type HttpResponse = tiny_http::Response<Box<dyn Read + Send>>;

/// Serves `engine` on `addr` (`host:port`) forever.
pub fn serve(engine: Arc<Engine>, model_id: String, addr: &str) -> Result<()> {
    let server = tiny_http::Server::http(addr).map_err(|e| anyhow::anyhow!("bind {addr}: {e}"))?;
    eprintln!("combs serve: listening on http://{addr} (model: {model_id})");

    for mut request in server.incoming_requests() {
        let engine = engine.clone();
        let model_id = model_id.clone();
        std::thread::spawn(move || {
            let url = request.url().to_string();
            let method = request.method().as_str().to_string();
            let response = match (method.as_str(), url.as_str()) {
                ("GET", "/health") => json_response(200, json!({"status": "ok"})),
                ("GET", "/v1/models") => json_response(
                    200,
                    json!({"object": "list", "data": [model_card(&model_id)]}),
                ),
                ("POST", "/v1/chat/completions") => {
                    let mut body = String::new();
                    if request.as_reader().read_to_string(&mut body).is_err() {
                        json_response(400, error_json("invalid_request", "unreadable body"))
                    } else {
                        handle_chat(&engine, &model_id, &body)
                    }
                }
                _ => json_response(404, error_json("not_found", "unknown endpoint")),
            };
            let _ = request.respond(response);
        });
    }
    Ok(())
}

fn model_card(model_id: &str) -> Value {
    json!({
        "id": model_id,
        "object": "model",
        "created": now_unix(),
        "owned_by": "combs",
    })
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn error_json(kind: &str, message: &str) -> Value {
    json!({"error": {"type": kind, "message": message}})
}

fn json_response(status: u16, body: Value) -> HttpResponse {
    let data = body.to_string().into_bytes();
    let len = data.len();
    tiny_http::Response::empty(status)
        .with_data(
            Box::new(std::io::Cursor::new(data)) as Box<dyn Read + Send>,
            Some(len),
        )
        .with_header(tiny_http::Header::from_bytes("Content-Type", "application/json").unwrap())
}

fn handle_chat(engine: &Arc<Engine>, model_id: &str, body: &str) -> HttpResponse {
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
    let prompt = apply_chatml(&messages);

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
    if let Some(mt) = req.get("max_tokens").and_then(Value::as_u64) {
        config.max_tokens = mt as usize;
    }
    if let Some(t) = req.get("temperature").and_then(Value::as_f64) {
        config.sampling.temperature = t as f32;
    }
    if let Some(p) = req.get("top_p").and_then(Value::as_f64) {
        config.sampling.top_p = Some(p as f32);
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

    if !stream {
        let mut text = String::new();
        let result = engine.generate(&tokens, &config, |_id, piece| text.push_str(piece));
        return match result {
            Ok(stats) => json_response(
                200,
                json!({
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": now_unix(),
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": if stats.generated_tokens >= config.max_tokens { "length" } else { "stop" },
                    }],
                    "usage": {
                        "prompt_tokens": stats.prompt_tokens,
                        "completion_tokens": stats.generated_tokens,
                        "total_tokens": stats.prompt_tokens + stats.generated_tokens,
                        "prompt_tokens_details": {"cached_tokens": stats.cached_tokens},
                    },
                }),
            ),
            Err(e) => json_response(500, error_json("engine_error", &e.to_string())),
        };
    }

    // SSE: channel-backed reader fed by the engine callback thread.
    let (tx, rx) = mpsc::channel::<String>();
    let engine = engine.clone();
    let model_id = model_id.to_string();
    let chunk_id = completion_id.clone();
    std::thread::spawn(move || {
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
        let result = engine.generate(&tokens, &config, |_id, piece| {
            let _ = send_chunk(json!({"content": piece}), None, None);
        });
        match result {
            Ok(stats) => {
                let finish = if stats.generated_tokens >= config.max_tokens {
                    "length"
                } else {
                    "stop"
                };
                let usage = json!({
                    "prompt_tokens": stats.prompt_tokens,
                    "completion_tokens": stats.generated_tokens,
                    "total_tokens": stats.prompt_tokens + stats.generated_tokens,
                    "prompt_tokens_details": {"cached_tokens": stats.cached_tokens},
                });
                let _ = send_chunk(json!({}), Some(finish), Some(usage));
            }
            Err(e) => {
                let _ = tx.send(format!(
                    "data: {}\n\n",
                    json!({"error": {"type": "engine_error", "message": e.to_string()}})
                ));
            }
        }
        let _ = tx.send("data: [DONE]\n\n".to_string());
    });

    tiny_http::Response::empty(200)
        .with_data(Box::new(ChannelReader::new(rx)) as Box<dyn Read + Send>, None)
        .with_header(tiny_http::Header::from_bytes("Content-Type", "text/event-stream").unwrap())
        .with_header(tiny_http::Header::from_bytes("Cache-Control", "no-cache").unwrap())
}

/// Applies the ChatML template (same convention as the FFI boundary).
fn apply_chatml(messages: &[Value]) -> String {
    let mut out = String::new();
    for m in messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let role = match role {
            "system" | "user" | "assistant" => role,
            _ => "user",
        };
        let content = m.get("content").and_then(Value::as_str).unwrap_or("");
        out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    out.push_str("<|im_start|>assistant\n");
    out
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
