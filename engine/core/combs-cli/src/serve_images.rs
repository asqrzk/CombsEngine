//! `combs serve-images` — persistent image-generation worker.
//!
//! Loads a Stable Diffusion checkpoint ONCE and serves OpenAI-style
//! `POST /v1/images/generations` requests, removing the 10–30 s cold start
//! of the per-request `combs generate-image` subprocess. Generations are
//! serialized through a mutex — one at a time on VRAM-limited hardware.
//!
//! Endpoints:
//! - `GET  /health` → `{"status":"ok"}`
//! - `POST /v1/images/generations` → `{created, data: [{b64_json}]}`
//!   body: `{prompt, negative_prompt?, size?: "WxH", width?, height?,
//!           steps?, guidance_scale?, seed?}`

use std::io::Read;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use serde_json::{Value, json};

use combs_diffusion::{
    DiffusionArchitecture, DiffusionModel, SchedulerKind, load_diffusion_model,
};

use crate::generate_image::tensor_to_rgb_image;
use crate::http::{HttpResponse, error_json, json_response, respond_preflight};

type SharedPipeline = Arc<Mutex<Box<dyn DiffusionModel<combs_core::CombsBackend>>>>;

/// Worker observability for `/v1/stats`: rolling counters written after
/// each generation (u64 atomics; durations stored as milliseconds).
#[derive(Default)]
struct ImageStats {
    requests_total: std::sync::atomic::AtomicU64,
    errors_total: std::sync::atomic::AtomicU64,
    last_duration_ms: std::sync::atomic::AtomicU64,
    last_bytes: std::sync::atomic::AtomicU64,
}

pub fn cmd_serve_images(model: PathBuf, port: u16) -> Result<()> {
    let model_dir = super::resolve_model_arg(&model)?;
    let architecture =
        DiffusionArchitecture::detect(&model_dir).context("detecting diffusion architecture")?;

    eprintln!("[serve-images] loading diffusion pipeline...");
    let device = combs_core::init_device();
    let pipeline = load_diffusion_model::<combs_core::CombsBackend>(
        architecture,
        &model_dir,
        &device,
    )
    .context("loading diffusion pipeline")?;
    let pipeline: SharedPipeline = Arc::new(Mutex::new(Box::new(pipeline)));

    let addr = format!("0.0.0.0:{port}");
    let server =
        tiny_http::Server::http(&addr).map_err(|e| anyhow::anyhow!("bind {addr}: {e}"))?;
    let model_id = model_dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "combs-diffusion".to_string());
    eprintln!("[serve-images] listening on http://{addr} (model: {model_id})");

    let stats = Arc::new(ImageStats::default());
    for mut request in server.incoming_requests() {
        let pipeline = pipeline.clone();
        let stats = stats.clone();
        let model_id = model_id.clone();
        std::thread::spawn(move || {
            let url = request.url().to_string();
            let method = request.method().as_str().to_string();
            if method == "OPTIONS" {
                respond_preflight(request);
                return;
            }
            let response = match (method.as_str(), url.as_str()) {
                ("GET", "/health") => json_response(200, json!({"status": "ok"})),
                ("GET", "/v1/stats") => {
                    use std::sync::atomic::Ordering::Relaxed;
                    json_response(
                        200,
                        json!({
                            "object": "image_worker.stats",
                            "model": model_id,
                            // try_lock fails iff a generation holds the pipeline.
                            "busy": pipeline.try_lock().is_err(),
                            "requests_total": stats.requests_total.load(Relaxed),
                            "errors_total": stats.errors_total.load(Relaxed),
                            "last_duration_ms": stats.last_duration_ms.load(Relaxed),
                            "last_bytes": stats.last_bytes.load(Relaxed),
                        }),
                    )
                }
                ("POST", "/v1/images/generations") => {
                    let mut body = String::new();
                    if request.as_reader().read_to_string(&mut body).is_err() {
                        json_response(400, error_json("invalid_request", "unreadable body"))
                    } else {
                        handle_generate(&pipeline, &body, &stats)
                    }
                }
                _ => json_response(404, error_json("not_found", "unknown endpoint")),
            };
            let _ = request.respond(response);
        });
    }
    Ok(())
}

fn handle_generate(pipeline: &SharedPipeline, body: &str, stats: &ImageStats) -> HttpResponse {
    let req: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            return json_response(
                400,
                error_json("invalid_request", &format!("bad JSON: {e}")),
            );
        }
    };
    let prompt = match req.get("prompt").and_then(Value::as_str) {
        Some(p) if !p.trim().is_empty() => p,
        _ => return json_response(400, error_json("invalid_request", "`prompt` is required")),
    };
    let negative = req.get("negative_prompt").and_then(Value::as_str);
    let (mut width, mut height) = (512u32, 512u32);
    if let Some(size) = req.get("size").and_then(Value::as_str) {
        if let Some((w, h)) = size.split_once('x') {
            if let (Ok(w), Ok(h)) = (w.parse::<u32>(), h.parse::<u32>()) {
                width = w;
                height = h;
            }
        }
    }
    if let Some(w) = req.get("width").and_then(Value::as_u64) {
        width = w as u32;
    }
    if let Some(h) = req.get("height").and_then(Value::as_u64) {
        height = h as u32;
    }
    let steps = req.get("steps").and_then(Value::as_u64).unwrap_or(20) as usize;
    let guidance =
        req.get("guidance_scale").and_then(Value::as_f64).unwrap_or(7.5) as f32;
    let seed = req.get("seed").and_then(Value::as_u64);
    let scheduler = match req.get("scheduler").and_then(Value::as_str) {
        None => SchedulerKind::default(),
        Some(s) => match SchedulerKind::parse(s) {
            Some(kind) => kind,
            None => {
                return json_response(
                    400,
                    error_json(
                        "invalid_request",
                        &format!("unknown scheduler {s:?} (ddpm | ddim | dpm++2m)"),
                    ),
                );
            }
        },
    };

    eprintln!(
        "[serve-images] generate: {prompt} ({width}x{height}, {steps} steps, {}, cfg {guidance})",
        scheduler.name()
    );

    // Single in-flight generation: the pipeline holds VRAM-resident state.
    let mut pipeline = pipeline.lock().unwrap();
    let started = std::time::Instant::now();
    let result = (|| -> Result<(Vec<u8>, u64)> {
        let embed = pipeline.encode_prompt(prompt, negative)?;
        let (image, effective_seed) =
            pipeline.generate(embed, width, height, steps, guidance, seed, scheduler)?;
        let img = tensor_to_rgb_image(&image)?;
        let mut buf: Vec<u8> = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)?;
        Ok((buf, effective_seed))
    })();

    {
        use std::sync::atomic::Ordering::Relaxed;
        stats.requests_total.fetch_add(1, Relaxed);
        stats
            .last_duration_ms
            .store(started.elapsed().as_millis() as u64, Relaxed);
        if let Ok((png, _)) = &result {
            stats.last_bytes.store(png.len() as u64, Relaxed);
        } else {
            stats.errors_total.fetch_add(1, Relaxed);
        }
    }

    match result {
        Ok((png, effective_seed)) => {
            eprintln!(
                "[serve-images] done in {:.1}s ({} bytes, seed {effective_seed})",
                started.elapsed().as_secs_f64(),
                png.len()
            );
            json_response(
                200,
                json!({
                    "created": SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                    // Echoed so the UI can display and replay the seed even
                    // when the request left it unset.
                    "seed": effective_seed,
                    "scheduler": scheduler.name(),
                    "data": [{ "b64_json": B64.encode(png) }],
                }),
            )
        }
        Err(e) => {
            eprintln!("[serve-images] failed: {e}");
            json_response(500, error_json("engine_error", &e.to_string()))
        }
    }
}
