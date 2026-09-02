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
//!           steps?, guidance_scale?, seed?, scheduler?, preview_every?}`

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use serde_json::{Value, json};

use combs_diffusion::{DiffusionArchitecture, DiffusionModel, SchedulerKind};

use crate::generate_image::tensor_to_rgb_image;
use crate::http::{HttpResponse, bytes_response, error_json, json_response, respond_preflight};

type SharedPipeline = Arc<Mutex<Box<dyn DiffusionModel<combs_core::CombsBackendF32>>>>;
/// Latest mid-run preview: (completed step, PNG bytes). Cleared when a new
/// generation stamps its counters; survives after completion until then.
type PreviewStash = Arc<Mutex<Option<(u64, Vec<u8>)>>>;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

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
    // Completion time of denoise step 1 — the ETA baseline that excludes
    // prompt-encode and first-step warmup.
    first_step_at_ms: std::sync::atomic::AtomicU64,
    // Last preview's step number (0 = none this run).
    preview_step: std::sync::atomic::AtomicU64,
    // The cadence the RUNNING generation actually uses — a per-request
    // override may differ from the spawn-time flag the top level reports.
    preview_cadence: std::sync::atomic::AtomicU64,
    phase: std::sync::atomic::AtomicU8,
}

pub fn cmd_serve_images(
    model: PathBuf,
    port: u16,
    lora: Option<PathBuf>,
    lora_scale: f32,
    preview_every: usize,
    llm: Option<PathBuf>,
    vae: Option<PathBuf>,
) -> Result<()> {
    let t_load = std::time::Instant::now();
    let model_dir = super::resolve_model_arg(&model)?;
    let open_ms = t_load.elapsed().as_millis() as u64;
    combs_core::progress::load("open", None, None, Some(open_ms));

    eprintln!("[serve-images] loading diffusion pipeline...");
    let device = combs_core::init_device();
    // Captured BEFORE the pipeline primes the cubecl runtime: device_caps
    // performs that setup itself and calling it afterwards panics with
    // "Service already initialized" (the same ordering `cmd_run` keeps).
    let device_type = combs_core::device_caps(&device).device_type;
    let lora_spec = lora.as_ref().map(|path| combs_diffusion::LoraSpec {
        path: path.clone(),
        scale: lora_scale,
    });
    let pipeline = match (&llm, &vae) {
        (Some(llm), Some(vae)) => {
            anyhow::ensure!(lora_spec.is_none(), "LoRA is not wired for recipe pipelines yet");
            combs_diffusion::loader::load_flux2_klein_recipe_split::<
                combs_core::CombsBackendF32,
                combs_core::CombsBackendF32,
            >(&model_dir, llm, vae, &device, &device)
            .context("loading flux2-klein recipe")?
        }
        (None, None) => {
            let architecture = DiffusionArchitecture::detect(&model_dir)
                .context("detecting diffusion architecture")?;
            combs_diffusion::loader::load_diffusion_model_with_lora::<
                combs_core::CombsBackendF32,
            >(architecture, &model_dir, &device, lora_spec.as_ref())
            .context("loading diffusion pipeline")?
        }
        _ => anyhow::bail!("--llm and --vae come as a pair (the recipe needs both)"),
    };
    let weights_ms = t_load.elapsed().as_millis() as u64 - open_ms;
    combs_core::progress::load("weights_done", None, None, Some(weights_ms));
    // Resolved once here so the stats route and the response echo never
    // need the pipeline mutex (which a running generation holds).
    let fixed_sampler = pipeline.fixed_sampler();
    let working_set = pipeline.working_set();
    let pipeline: SharedPipeline = Arc::new(Mutex::new(pipeline));
    // The baseline every later fit estimate is measured against: what
    // this process holds with weights resident and nothing running.
    let resident_at_load = crate::fit::process_footprint_bytes().unwrap_or(0);
    // Only where the accelerator draws on host memory does host free
    // memory decide whether a canvas fits (see `image_fit_refusal`).
    let unified_memory = crate::fit::draws_on_host_memory(&device_type);
    let preflight = ImagePreflight {
        // This pipeline's own measured curve, or None when nobody has
        // measured it — then the question goes unasked rather than
        // answered with another pipeline's numbers.
        working_set: working_set.filter(|_| {
            unified_memory && resident_at_load > 0 && !crate::fit::preflight_disabled()
        }),
        resident_at_load,
        largest_completed_pixels: Arc::new(std::sync::atomic::AtomicU64::new(0)),
    };
    // What this process will actually do, recorded rather than assumed:
    // an A/B that infers its own configuration proves nothing.
    let provenance_config = Arc::new(vec![
        ("device", device_type.clone()),
        ("dtype", crate::build_info::SERVING_DTYPE.to_string()),
        ("matmul", combs_models::batched_matmul_summary()),
        ("preview_every", preview_every.to_string()),
        ("sampler", fixed_sampler.unwrap_or("caller's choice").to_string()),
        ("lora", lora_spec.as_ref().map_or("none".into(), |s| format!("{:?} x{}", s.path.file_name().unwrap_or_default(), s.scale))),
    ]);
    combs_core::provenance::startup("image", &provenance_config);
    match preflight.working_set {
        Some(_) => eprintln!(
            "[serve-images] memory pre-flight armed (resident {} MB, {device_type})",
            resident_at_load / (1024 * 1024)
        ),
        None => eprintln!(
            "[serve-images] memory pre-flight OFF ({})",
            if crate::fit::preflight_disabled() {
                "disabled by COMBS_IMAGE_PREFLIGHT=0"
            } else if !unified_memory {
                "accelerator does not draw on host memory"
            } else if resident_at_load == 0 {
                "no footprint probe on this platform"
            } else {
                "this pipeline's working set has not been measured"
            }
        ),
    }

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
    let preview: PreviewStash = Arc::new(Mutex::new(None));
    if preview_every > 0 {
        eprintln!("[serve-images] previews on: every {preview_every} steps");
    }
    for mut request in server.incoming_requests() {
        let pipeline = pipeline.clone();
        let stats = stats.clone();
        let preview = preview.clone();
        let model_id = model_id.clone();
        let lora_info = lora_info.clone();
        let fixed_sampler = fixed_sampler.clone();
        let preflight = preflight.clone();
        let provenance_config = provenance_config.clone();
        std::thread::spawn(move || {
            let url = request.url().to_string();
            let method = request.method().as_str().to_string();
            if method == "OPTIONS" {
                respond_preflight(request);
                return;
            }
            let response = match (method.as_str(), url.as_str()) {
                ("GET", "/health") => json_response(200, json!({"status": "ok"})),
                // Identity without the pipeline mutex: everything here
                // was resolved at load, so a running generation never
                // blocks the platform's capability merge (CP-3).
                ("GET", "/v1/model/info") => json_response(
                    200,
                    json!({
                        "model": model_id,
                        "kind": "diffusion",
                        "dtype": crate::build_info::SERVING_DTYPE,
                        "lora": lora_info,
                        "fixed_sampler": fixed_sampler,
                        // The measured working-set curve when one exists
                        // for THIS pipeline — the platform sizes canvas
                        // offers from it; null means unmeasured, not
                        // unlimited.
                        "working_set": preflight.working_set.as_ref().map(|ws| json!({
                            "fixed_bytes": ws.fixed_bytes,
                            "bytes_per_pixel": ws.bytes_per_pixel,
                            "measured_max_pixels": ws.measured_max_pixels,
                        })),
                    }),
                ),
                ("GET", "/v1/stats") => {
                    use std::sync::atomic::Ordering::Relaxed;
                    let gen_total = stats.total_steps.load(Relaxed);
                    let generation = if gen_total == 0 {
                        Value::Null
                    } else {
                        let now = now_ms();
                        let step = stats.current_step.load(Relaxed);
                        let phase = stats.phase.load(Relaxed);
                        // Measured pace: (t_now - t_step1) / (step - 1),
                        // so encode + first-step warmup never skew it.
                        let first = stats.first_step_at_ms.load(Relaxed);
                        let eta_ms = if phase == 2 && step >= 2 && first > 0 {
                            let per = now.saturating_sub(first) / (step - 1);
                            Value::from(per * (gen_total - step))
                        } else {
                            Value::Null
                        };
                        let preview_step = stats.preview_step.load(Relaxed);
                        json!({
                            "step": step,
                            "total": gen_total,
                            "phase": match phase {
                                1 => Value::from("encode"),
                                2 => Value::from("denoise"),
                                3 => Value::from("decode"),
                                4 => Value::from("png"),
                                _ => Value::Null,
                            },
                            "eta_ms": eta_ms,
                            "preview_step": if preview_step > 0 {
                                Value::from(preview_step)
                            } else {
                                Value::Null
                            },
                            "preview_every": stats.preview_cadence.load(Relaxed),
                            "elapsed_ms":
                                now.saturating_sub(stats.started_at_ms.load(Relaxed)),
                        })
                    };
                    json_response(
                        200,
                        json!({
                            "object": "image_worker.stats",
                            "model": model_id,
                            "provenance": crate::build_info::stats("image", &provenance_config),
                            // try_lock fails iff a generation holds the pipeline.
                            "busy": pipeline.try_lock().is_err(),
                            "requests_total": stats.requests_total.load(Relaxed),
                            "errors_total": stats.errors_total.load(Relaxed),
                            "last_duration_ms": stats.last_duration_ms.load(Relaxed),
                            "last_bytes": stats.last_bytes.load(Relaxed),
                            "generation": generation,
                            "preview_every": preview_every,
                            // Which sampler actually runs: pipelines with a
                            // built-in schedule ignore the request's choice.
                            "sampler": match fixed_sampler {
                                Some(name) => json!({"fixed": name}),
                                None => json!({
                                    "choices": ["ddpm", "ddim", "dpm++2m"],
                                }),
                            },
                            "lora": lora_info,
                            "load": {"ms": load_ms, "open_ms": open_ms, "weights_ms": weights_ms},
                        }),
                    )
                }
                ("GET", "/v1/preview") => {
                    match preview.lock().unwrap().clone() {
                        Some((step, png)) => bytes_response(200, "image/png", png)
                            .with_header(
                                tiny_http::Header::from_bytes(
                                    "X-Combs-Preview-Step",
                                    step.to_string(),
                                )
                                .unwrap(),
                            ),
                        None => json_response(
                            404,
                            error_json("not_found", "no preview available"),
                        ),
                    }
                }
                ("POST", "/v1/images/generations") => {
                    let mut body = String::new();
                    if request.as_reader().read_to_string(&mut body).is_err() {
                        json_response(400, error_json("invalid_request", "unreadable body"))
                    } else {
                        handle_generate(
                            &pipeline,
                            &body,
                            &stats,
                            &preview,
                            preview_every,
                            fixed_sampler,
                            &preflight,
                        )
                    }
                }
                _ => json_response(404, error_json("not_found", "unknown endpoint")),
            };
            let _ = request.respond(response);
        });
    }
    Ok(())
}

/// Everything the pre-flight needs that outlives a single request. A
/// `None` working set means the check is off, for one of the reasons
/// logged at startup.
#[derive(Clone)]
struct ImagePreflight {
    working_set: Option<combs_diffusion::WorkingSet>,
    resident_at_load: u64,
    /// Largest canvas served to completion — the pool is only credited
    /// against shapes it has actually held.
    largest_completed_pixels: Arc<std::sync::atomic::AtomicU64>,
}

impl ImagePreflight {
    fn refusal(&self, width: u32, height: u32) -> Option<String> {
        crate::fit::image_refusal(
            &crate::fit::PreflightContext {
                working_set: self.working_set,
                resident_at_load: self.resident_at_load,
                largest_completed_pixels: self
                    .largest_completed_pixels
                    .load(std::sync::atomic::Ordering::Relaxed),
            },
            width,
            height,
        )
    }
}

fn handle_generate(
    pipeline: &SharedPipeline,
    body: &str,
    stats: &ImageStats,
    preview: &PreviewStash,
    preview_every: usize,
    fixed_sampler: Option<&'static str>,
    preflight: &ImagePreflight,
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
    let requested_scheduler = req.get("scheduler").and_then(Value::as_str);
    // Everything user-facing reports the sampler that RUNS. Fixed-schedule
    // pipelines accept ANY scheduler string and drop it — including the
    // very name the response echoes ("flow-match-euler"), which
    // SchedulerKind cannot parse: rejecting it would 400 the replay of our
    // own response. Choice-taking pipelines keep strict validation.
    let scheduler = match requested_scheduler {
        None => SchedulerKind::default(),
        Some(s) => match SchedulerKind::parse(s) {
            Some(kind) => kind,
            None if fixed_sampler.is_some() => SchedulerKind::default(),
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
    let sampler_name = fixed_sampler.unwrap_or_else(|| scheduler.name());
    if let (Some(fixed), Some(requested)) = (fixed_sampler, requested_scheduler) {
        eprintln!("[serve-images] scheduler {requested:?} ignored — pipeline runs {fixed}");
    }
    // Preview cadence: the request may override the worker-wide flag, so a
    // short distilled run (4 steps) can still ask for previews every step.
    // The field never rejects — this endpoint tolerated unknown keys before
    // it existed, so an uninterpretable value keeps the flag and says so.
    let preview_every = match req.get("preview_every") {
        None => preview_every,
        Some(v) => match v.as_u64() {
            Some(n) if n <= 1000 => n as usize,
            _ => {
                eprintln!(
                    "[serve-images] preview_every {v} ignored — expected an integer in 0..=1000"
                );
                preview_every
            }
        },
    };

    // A canvas whose working set the machine cannot hand out does not
    // fail — it wedges the whole system while the kernel thrashes. The
    // only honest moment to say no is here: before the mutex, before
    // the first allocation.
    if let Some(err) = preflight.refusal(width, height) {
        combs_core::provenance::turn("image", "generate", &[("size", format!("{width}x{height}"))])
            .failed(&err);
        return json_response(507, error_json("insufficient_memory", &err));
    }

    let turn = combs_core::provenance::turn(
        "image",
        "generate",
        &[
            ("size", format!("{width}x{height}")),
            ("steps", steps.to_string()),
            ("sampler", sampler_name.to_string()),
            ("cfg", guidance.to_string()),
            ("seed", seed.map_or("entropy".into(), |s| s.to_string())),
            ("preview_every", preview_every.to_string()),
            ("prompt_chars", prompt.chars().count().to_string()),
        ],
    );

    // Single in-flight generation: the pipeline holds VRAM-resident state.
    let mut pipeline = pipeline.lock().unwrap();
    let started = std::time::Instant::now();
    {
        // Single-flight (mutex above), so these are unambiguous.
        use std::sync::atomic::Ordering::Relaxed;
        stats.current_step.store(0, Relaxed);
        stats.total_steps.store(steps as u64, Relaxed);
        stats.first_step_at_ms.store(0, Relaxed);
        stats.preview_step.store(0, Relaxed);
        stats.preview_cadence.store(preview_every as u64, Relaxed);
        stats.phase.store(1, Relaxed);
        stats.started_at_ms.store(now_ms(), Relaxed);
    }
    *preview.lock().unwrap() = None;
    let mut on_step = |step: usize, total: usize| {
        use std::sync::atomic::Ordering::Relaxed;
        stats.current_step.store(step as u64, Relaxed);
        if step == 1 {
            stats.first_step_at_ms.store(now_ms(), Relaxed);
        }
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
    let mut on_preview = |step: usize, image| {
        let png = tensor_to_rgb_image(&image).ok().and_then(|img| {
            let mut buf: Vec<u8> = Vec::new();
            img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
                .ok()?;
            Some(buf)
        });
        if let Some(png) = png {
            eprintln!(
                "[serve-images] preview at step {step} ({} bytes, {:.1}s)",
                png.len(),
                started.elapsed().as_secs_f64()
            );
            *preview.lock().unwrap() = Some((step as u64, png));
            stats
                .preview_step
                .store(step as u64, std::sync::atomic::Ordering::Relaxed);
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
            combs_diffusion::GenerationHooks {
                on_step: Some(&mut on_step),
                preview_every,
                on_preview: Some(&mut on_preview),
            },
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
            // This shape is now proven to fit, so the pool may be
            // credited against canvases up to this size.
            preflight
                .largest_completed_pixels
                .fetch_max(width as u64 * height as u64, Relaxed);
        } else {
            stats.errors_total.fetch_add(1, Relaxed);
        }
    }

    match result {
        Ok((png, effective_seed)) => {
            turn.ok(&[
                ("bytes", png.len().to_string()),
                ("seed", effective_seed.to_string()),
            ]);
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
                    "scheduler": sampler_name,
                    "data": [{ "b64_json": B64.encode(png) }],
                }),
            )
        }
        Err(e) => {
            turn.failed(&e.to_string());
            json_response(500, error_json("engine_error", &e.to_string()))
        }
    }
}
