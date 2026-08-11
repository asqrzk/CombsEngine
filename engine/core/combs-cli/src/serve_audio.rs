//! `combs serve-audio` — persistent speech worker (mirrors `serve-images`).
//!
//! Loads the Kokoro TTS engine ONCE (ONNX session + vocab; voice style
//! tables cache on first use) and serves speech requests, removing the
//! per-request espeak+ONNX cold start of the `combs generate-audio`
//! subprocess. Requests are serialized through a mutex. STT
//! (`/v1/audio/transcriptions`) joins this worker in roadmap wave 4.
//!
//! Endpoints:
//! - `GET  /health` → `{"status":"ok"}`
//! - `GET  /v1/stats` → busy/requests/errors/last-duration/last-bytes
//! - `GET  /v1/audio/voices` → `{"voices": [...], "default": "af_heart"}`
//! - `POST /v1/audio/speech` → `audio/wav` bytes (OpenAI-style binary
//!   response); body `{input | text, voice?, speed?, lang?}`

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::generate_audio::{TtsEngine, encode_wav};
use crate::http::{error_json, json_response, respond_preflight};

#[derive(Default)]
struct AudioStats {
    requests_total: std::sync::atomic::AtomicU64,
    errors_total: std::sync::atomic::AtomicU64,
    last_duration_ms: std::sync::atomic::AtomicU64,
    last_bytes: std::sync::atomic::AtomicU64,
}

pub fn cmd_serve_audio(model: PathBuf, port: u16) -> Result<()> {
    let model_dir = super::resolve_model_arg(&model)?;

    eprintln!("[serve-audio] loading TTS engine...");
    let engine = TtsEngine::load(&model_dir).context("loading TTS engine")?;
    let voices = engine.voices().unwrap_or_default();
    let engine = Arc::new(Mutex::new(engine));

    let addr = format!("0.0.0.0:{port}");
    let server =
        tiny_http::Server::http(&addr).map_err(|e| anyhow::anyhow!("bind {addr}: {e}"))?;
    let model_id = model_dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "combs-tts".to_string());
    eprintln!(
        "[serve-audio] listening on http://{addr} (model: {model_id}, {} voices)",
        voices.len()
    );

    let stats = Arc::new(AudioStats::default());
    for mut request in server.incoming_requests() {
        let engine = engine.clone();
        let stats = stats.clone();
        let model_id = model_id.clone();
        let voices = voices.clone();
        std::thread::spawn(move || {
            let url = request.url().to_string();
            let method = request.method().as_str().to_string();
            if method == "OPTIONS" {
                respond_preflight(request);
                return;
            }
            match (method.as_str(), url.as_str()) {
                ("GET", "/health") => {
                    let _ = request.respond(json_response(200, json!({"status": "ok"})));
                }
                ("GET", "/v1/stats") => {
                    use std::sync::atomic::Ordering::Relaxed;
                    let _ = request.respond(json_response(
                        200,
                        json!({
                            "object": "audio_worker.stats",
                            "model": model_id,
                            // try_lock fails iff a synthesis holds the engine.
                            "busy": engine.try_lock().is_err(),
                            "voices": voices.len(),
                            "requests_total": stats.requests_total.load(Relaxed),
                            "errors_total": stats.errors_total.load(Relaxed),
                            "last_duration_ms": stats.last_duration_ms.load(Relaxed),
                            "last_bytes": stats.last_bytes.load(Relaxed),
                        }),
                    ));
                }
                ("GET", "/v1/audio/voices") => {
                    let _ = request.respond(json_response(
                        200,
                        json!({"voices": voices, "default": "af_heart"}),
                    ));
                }
                ("POST", "/v1/audio/speech") => {
                    // Cap the body — an unbounded read_to_string hands any
                    // client an arbitrary-allocation primitive.
                    let mut body = String::new();
                    let ok = {
                        use std::io::Read;
                        request
                            .as_reader()
                            .take(1_000_000)
                            .read_to_string(&mut body)
                            .is_ok()
                    };
                    if !ok {
                        let _ = request.respond(json_response(
                            400,
                            error_json("invalid_request", "unreadable body"),
                        ));
                        return;
                    }
                    handle_speech(request, &engine, &body, &stats);
                }
                _ => {
                    let _ = request
                        .respond(json_response(404, error_json("not_found", "unknown endpoint")));
                }
            }
        });
    }
    Ok(())
}

fn handle_speech(
    request: tiny_http::Request,
    engine: &Arc<Mutex<TtsEngine>>,
    body: &str,
    stats: &AudioStats,
) {
    let req: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            let _ = request.respond(json_response(
                400,
                error_json("invalid_request", &format!("bad JSON: {e}")),
            ));
            return;
        }
    };
    // OpenAI uses `input`; the platform's older subprocess contract used
    // `text` — accept both.
    let text = req
        .get("input")
        .or_else(|| req.get("text"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if text.trim().is_empty() {
        let _ = request.respond(json_response(
            400,
            error_json("invalid_request", "`input` is required"),
        ));
        return;
    }
    let voice = req.get("voice").and_then(Value::as_str).unwrap_or("af_heart");
    // OpenAI's contract range; also keeps a hostile value from producing
    // hours of audio under the mutex.
    let speed = (req.get("speed").and_then(Value::as_f64).unwrap_or(1.0) as f32)
        .clamp(0.25, 4.0);
    let lang = req.get("lang").and_then(Value::as_str);

    let started = std::time::Instant::now();
    let result = {
        // Single in-flight synthesis: the ONNX session is stateful. A
        // panicked previous request must not brick the endpoint forever —
        // recover the poisoned mutex.
        let mut engine = engine.lock().unwrap_or_else(|e| e.into_inner());
        engine.synthesize(text, voice, speed, lang)
    };

    use std::sync::atomic::Ordering::Relaxed;
    stats.requests_total.fetch_add(1, Relaxed);
    stats
        .last_duration_ms
        .store(started.elapsed().as_millis() as u64, Relaxed);

    match result {
        Ok(samples) => {
            let wav = encode_wav(&samples);
            stats.last_bytes.store(wav.len() as u64, Relaxed);
            eprintln!(
                "[serve-audio] spoke {} chars in {:.1}s ({} bytes, voice {voice})",
                text.chars().count(),
                started.elapsed().as_secs_f64(),
                wav.len()
            );
            let response = tiny_http::Response::from_data(wav)
                .with_header(
                    tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"audio/wav"[..])
                        .expect("static header"),
                )
                .with_header(
                    tiny_http::Header::from_bytes(
                        &b"Access-Control-Allow-Origin"[..],
                        &b"*"[..],
                    )
                    .expect("static header"),
                );
            let _ = request.respond(response);
        }
        Err(e) => {
            stats.errors_total.fetch_add(1, Relaxed);
            eprintln!("[serve-audio] failed: {e:#}");
            let _ = request
                .respond(json_response(500, error_json("engine_error", &format!("{e:#}"))));
        }
    }
}
