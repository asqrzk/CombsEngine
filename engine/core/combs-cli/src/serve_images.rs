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

type SharedPipeline = Arc<Mutex<Box<dyn DiffusionModel<combs_core::CombsBackendF32>>>>;

/// Worker observability for `/v1/stats`: rolling counters written during
/// and after each generation (u64 atomics; durations stored as
/// milliseconds). `total_steps == 0` means no generation in flight —
/// the stats route reports `generation: null` then. `phase` maps
/// 1=encode (CLIP prompt), 2=denoise (UNet loop), 3=decode (VAE),
/// 4=png (pack + base64); 0 outside a run.
#[derive(Default)]
struct ImageStats {
    requests_total: std::sync::atomic::AtomicU64,
    errors_total: std::sync::atomic::AtomicU64,
    last_duration_ms: std::sync::atomic::AtomicU64,
    last_bytes: std::sync::atomic::AtomicU64,
    current_step: std::sync::atomic::AtomicU64,
    total_steps: std::sync::atomic::AtomicU64,
    started_at_ms: std::sync::atomic::AtomicU64,
    phase: std::sync::atomic::AtomicU8,
}

pub fn cmd_serve_images(
    model: PathBuf,
    port: u16,
    lora: Option<PathBuf>,
    lora_scale: f32,
) -> Result<()> {
    let t_load = std::time::Instant::now();
    let model_dir = super::resolve_model_arg(&model)?;
    let architecture =
        DiffusionArchitecture::detect(&model_dir).context("detecting diffusion architecture")?;
    let open_ms = t_load.elapsed().as_millis() as u64;
    combs_core::progress::load("open", None, None, Some(open_ms));

    eprintln!("[serve-images] loading diffusion pipeline...");
    let device = combs_core::init_device();
    let lora_spec = lora.as_ref().map(|path| combs_diffusion::LoraSpec {
        path: path.clone(),
        scale: lora_scale,
    });
    let pipeline = combs_diffusion::loader::load_diffusion_model_with_lora::<
        combs_core::CombsBackendF32,
    >(architecture, &model_dir, &device, lora_spec.as_ref())
    .context("loading diffusion pipeline")?;
    let weights_ms = t_load.elapsed().as_millis() as u64 - open_ms;
    combs_core::progress::load("weights_done", None, None, Some(weights_ms));
    let pipeline: SharedPipeline = Arc::new(Mutex::new(Box::new(pipeline)));

    let lora_info = match &lora_spec {
        Some(spec) => json!({
            "file": spec.path.file_name().map(|f| f.to_string_lossy().into_owned()),
            "scale": spec.scale,
        }),
        None => serde_json::Value::Null,
    };
    let load_ms = t_load.elapsed().as_millis() as u64;
    combs_core::progress::load("bind", None, None, Some(load_ms));
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
        let lora_info = lora_info.clone();
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
                    let gen_total = stats.total_steps.load(Relaxed);
                    let generation = if gen_total == 0 {
                        Value::Null
                    } else {
                        let now_ms = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .map(|d| d.as_millis() as u64)
                            .unwrap_or(0);
                        json!({
                            "step": stats.current_step.load(Relaxed),
                            "total": gen_total,
                            "phase": match stats.phase.load(Relaxed) {
                                1 => Value::from("encode"),
                                2 => Value::from("denoise"),
                                3 => Value::from("decode"),
                                4 => Value::from("png"),
                                _ => Value::Null,
                            },
                            "elapsed_ms":
                                now_ms.saturating_sub(stats.started_at_ms.load(Relaxed)),
                        })
                    };
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
                            "generation": generation,
                            "lora": lora_info,
                            "load": {"ms": load_ms, "open_ms": open_ms, "weights_ms": weights_ms},
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
    // Validate dimensions/steps/guidance BEFORE taking the pipeline mutex:
    // a silent fallback here returns the wrong image with a 200, and an
    // absurd size would stall every queued request.
    if let Some(size) = req.get("size").and_then(Value::as_str) {
        let parsed = size
            .split_once('x')
            .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)));
        if parsed.is_none() {
            return json_response(
                400,
                error_json("invalid_request", &format!("bad size {size:?} (expected WxH)")),
            );
        }
    }
    for (name, v) in [("width", width), ("height", height)] {
        if v % 8 != 0 || !(64..=1024).contains(&v) {
            return json_response(
                400,
                error_json(
                    "invalid_request",
                    &format!("{name} must be a multiple of 8 in 64..=1024, got {v}"),
                ),
            );
        }
    }
    let steps = req.get("steps").and_then(Value::as_u64).unwrap_or(20) as usize;
    if !(1..=1000).contains(&steps) {
        return json_response(
            400,
            error_json("invalid_request", &format!("steps must be 1..=1000, got {steps}")),
        );
    }
    let guidance =
        req.get("guidance_scale").and_then(Value::as_f64).unwrap_or(7.5) as f32;
    if !guidance.is_finite() || !(0.0..=50.0).contains(&guidance) {
        return json_response(
            400,
            error_json("invalid_request", "guidance_scale must be finite in 0..=50"),
        );
    }
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
    {
        // Single-flight (mutex above), so these are unambiguous.
        use std::sync::atomic::Ordering::Relaxed;
        stats.current_step.store(0, Relaxed);
        stats.total_steps.store(steps as u64, Relaxed);
        stats.phase.store(1, Relaxed);
        stats.started_at_ms.store(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            Relaxed,
        );
    }
    let mut on_step = |step: usize, total: usize| {
        use std::sync::atomic::Ordering::Relaxed;
        stats.current_step.store(step as u64, Relaxed);
        if step == total {
            // After the final scheduler step, what's left inside
            // generate() is the VAE decode.
            stats.phase.store(3, Relaxed);
            eprintln!(
                "[serve-images] denoise done ({total} steps) at {:.1}s",
                started.elapsed().as_secs_f64()
            );
        }
    };
    let result = (|| -> Result<(Vec<u8>, u64)> {
        let embed = pipeline.encode_prompt(prompt, negative)?;
        stats.phase.store(2, std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "[serve-images] prompt encoded at {:.1}s",
            started.elapsed().as_secs_f64()
        );
        let (image, effective_seed) = pipeline.generate(
            embed,
            width,
            height,
            steps,
            guidance,
            seed,
            scheduler,
            Some(&mut on_step),
        )?;
        stats.phase.store(4, std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "[serve-images] vae decoded at {:.1}s",
            started.elapsed().as_secs_f64()
        );
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
        stats.total_steps.store(0, Relaxed);
        stats.current_step.store(0, Relaxed);
        stats.phase.store(0, Relaxed);
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
