//! `combs serve-audio` — persistent speech worker (mirrors `serve-images`).
//!
//! Loads the Kokoro TTS engine ONCE (ONNX session + vocab; voice style
//! tables cache on first use) and serves speech requests, removing the
//! per-request espeak+ONNX cold start of the `combs generate-audio`
//! subprocess. With `--transcribe-model`, a Whisper speech-to-text engine
//! loads alongside it. Each engine serializes its requests through a mutex.
//!
//! Endpoints:
//! - `GET  /health` → `{"status":"ok"}`
//! - `GET  /v1/stats` → busy/requests/errors/last-duration/last-bytes
//! - `GET  /v1/audio/voices` → `{"voices": [...], "default": "af_heart"}`
//! - `POST /v1/audio/speech` → `audio/wav` bytes (OpenAI-style binary
//!   response); body `{input | text, voice?, speed?, lang?}`
//! - `POST /v1/audio/transcriptions` → `{"text": …}`; multipart form-data
//!   (`file` part) or a raw WAV body

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

pub fn cmd_serve_audio(
    model: PathBuf,
    port: u16,
    transcribe_model: Option<PathBuf>,
    language: String,
) -> Result<()> {
    let model_dir = super::resolve_model_arg(&model)?;

    combs_core::provenance::startup(
        "audio",
        &[
            ("dtype", crate::build_info::SERVING_DTYPE.to_string()),
            ("model", model.display().to_string()),
            ("transcribe", transcribe_model.as_ref().map_or("none".into(), |p| p.display().to_string())),
            ("language", language.clone()),
        ],
    );
    eprintln!("[serve-audio] loading TTS engine...");
    let engine = TtsEngine::load(&model_dir).context("loading TTS engine")?;
    let voices = engine.voices().unwrap_or_default();
    let engine = Arc::new(Mutex::new(engine));

    let stt = match transcribe_model {
        Some(p) => {
            let dir = super::resolve_model_arg(&p)?;
            eprintln!("[serve-audio] loading speech-to-text engine...");
            let e = combs_runtime::SpeechEngine::load(&dir, &language)
                .map_err(|e| anyhow::anyhow!("loading speech model: {e}"))?;
            Some(Arc::new(Mutex::new(e)))
        }
        None => None,
    };

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
        let stt = stt.clone();
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
                ("GET", "/v1/model/info") => {
                    let _ = request.respond(json_response(
                        200,
                        json!({
                            "model": model_id,
                            "kind": "tts",
                            // Kokoro synthesizes 24 kHz mono WAV; the rate
                            // is the model's, not a server option.
                            "sample_rate_hz": 24000,
                            "voices": voices.len(),
                            "default_voice": "af_heart",
                            "transcribe": stt.is_some(),
                        }),
                    ));
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
                ("POST", "/v1/audio/transcriptions") => {
                    handle_transcription(request, stt.as_deref(), &stats);
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

/// POST /v1/audio/transcriptions — multipart form-data (`file` part) or a
/// raw WAV body. Response: OpenAI-style `{"text": "..."}`.
fn handle_transcription(
    mut request: tiny_http::Request,
    stt: Option<&Mutex<combs_runtime::SpeechEngine>>,
    stats: &AudioStats,
) {
    use std::sync::atomic::Ordering::Relaxed;
    let Some(engine) = stt else {
        let _ = request.respond(json_response(
            501,
            error_json(
                "not_configured",
                "no transcription model loaded (start serve-audio with --transcribe-model)",
            ),
        ));
        return;
    };
    let content_type = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Content-Type"))
        .map(|h| h.value.as_str().to_string())
        .unwrap_or_default();
    // Cap the body: 64 MB covers ~35 min of 16-bit 16 kHz mono.
    let mut body = Vec::new();
    let ok = {
        use std::io::Read;
        request
            .as_reader()
            .take(64_000_000)
            .read_to_end(&mut body)
            .is_ok()
    };
    if !ok || body.is_empty() {
        let _ = request.respond(json_response(
            400,
            error_json("invalid_request", "unreadable body"),
        ));
        return;
    }
    let wav = match multipart_boundary(&content_type) {
        Some(b) => match multipart_file(&body, &b) {
            Some(f) => f,
            None => {
                let _ = request.respond(json_response(
                    400,
                    error_json("invalid_request", "multipart body has no file part"),
                ));
                return;
            }
        },
        None => body,
    };

    stats.requests_total.fetch_add(1, Relaxed);
    let start = std::time::Instant::now();
    let result = engine.lock().unwrap().transcribe_wav(&wav);
    match result {
        Ok(text) => {
            stats
                .last_duration_ms
                .store(start.elapsed().as_millis() as u64, Relaxed);
            stats.last_bytes.store(text.len() as u64, Relaxed);
            let _ = request.respond(json_response(200, json!({ "text": text })));
        }
        Err(e) => {
            stats.errors_total.fetch_add(1, Relaxed);
            let _ = request.respond(json_response(
                500,
                error_json("transcription_failed", &e.to_string()),
            ));
        }
    }
}

/// Extracts the boundary parameter from a multipart/form-data Content-Type.
fn multipart_boundary(content_type: &str) -> Option<String> {
    if !content_type
        .to_ascii_lowercase()
        .starts_with("multipart/form-data")
    {
        return None;
    }
    content_type.split(';').find_map(|p| {
        p.trim()
            .strip_prefix("boundary=")
            .map(|b| b.trim_matches('"').to_string())
    })
}

/// Minimal multipart parser: returns the bytes of the `file` part, or the
/// first part carrying a filename. Enough for OpenAI-style clients; not a
/// general RFC 2046 implementation.
fn multipart_file(body: &[u8], boundary: &str) -> Option<Vec<u8>> {
    let delim = format!("--{boundary}");
    let delim = delim.as_bytes();
    let find = |hay: &[u8], needle: &[u8], from: usize| -> Option<usize> {
        hay.get(from..)?
            .windows(needle.len())
            .position(|w| w == needle)
            .map(|p| p + from)
    };
    let mut fallback: Option<Vec<u8>> = None;
    let mut pos = find(body, delim, 0)?;
    loop {
        pos += delim.len();
        if body.get(pos..pos + 2) == Some(b"--".as_slice()) {
            break; // closing delimiter
        }
        let head_start = find(body, b"\r\n", pos)? + 2;
        let head_end = find(body, b"\r\n\r\n", head_start)?;
        let headers = String::from_utf8_lossy(&body[head_start..head_end]).to_ascii_lowercase();
        let content_start = head_end + 4;
        let next = find(body, delim, content_start)?;
        // Content ends before the \r\n that precedes the next delimiter.
        let content_end = next.saturating_sub(2).max(content_start);
        let content = body[content_start..content_end].to_vec();
        if headers.contains("name=\"file\"") {
            return Some(content);
        }
        if fallback.is_none() && headers.contains("filename=") {
            fallback = Some(content);
        }
        pos = next;
    }
    fallback
}
