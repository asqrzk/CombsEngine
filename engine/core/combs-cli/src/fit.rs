//! Pre-flight fit check: refuse a model whose largest single allocation
//! exceeds the adapter's per-binding cap BEFORE `Engine::load` enqueues
//! uploads. cubecl materializes buffers on its own compute threads where
//! an oversize allocation can only panic — never surface as a `Result` —
//! so the only honest moment to say "this model does not fit this
//! device" is before the first enqueue.

/// One tensor's contribution to the fit decision.
pub struct FitRow {
    pub name: String,
    /// Logical element count (dense materialization is `elements × 4`,
    /// the f32 widening every non-packed tensor goes through on load).
    pub elements: u64,
    /// Packed byte length + rank, set by the caller ONLY when the
    /// quant-linear path truly applies (rank-2 linear, block-aligned k,
    /// kernels enabled, not an embedding — embeddings always dequantize
    /// dense). Known limit: GGUF fused projections (phi3 attn_qkv) are
    /// measured per stored tensor, not per served slice.
    pub packed: Option<(u64, usize)>,
}

impl FitRow {
    /// Bytes this tensor materializes on the device. Unknown eligibility
    /// resolves to the LARGER dense size — over-refusing beats serving a
    /// model with holes in it.
    pub fn device_bytes(&self) -> u64 {
        match self.packed {
            Some((bytes, 2)) => bytes,
            _ => self.elements.saturating_mul(4),
        }
    }
}

pub struct FitReport {
    pub worst_name: String,
    pub worst_bytes: u64,
    pub tensors: usize,
}

/// Largest single device allocation across the model's tensors.
pub fn fit_report(rows: impl IntoIterator<Item = FitRow>) -> Option<FitReport> {
    let mut worst: Option<(String, u64)> = None;
    let mut tensors = 0usize;
    for row in rows {
        tensors += 1;
        let bytes = row.device_bytes();
        if worst.as_ref().map(|(_, b)| bytes > *b).unwrap_or(true) {
            worst = Some((row.name, bytes));
        }
    }
    worst.map(|(worst_name, worst_bytes)| FitReport {
        worst_name,
        worst_bytes,
        tensors,
    })
}

/// The refusal, worded to act on: device, cap, offending tensor, need.
pub fn check_fit(
    report: &FitReport,
    limit: u64,
    device_name: &str,
    device_type: &str,
) -> Result<(), String> {
    if report.worst_bytes <= limit {
        return Ok(());
    }
    Err(format!(
        "device {device_name} ({device_type}) caps allocations at {limit} bytes; \
         tensor {} needs {} — model does not fit this adapter \
         (checked {} tensors; a CPU-fallback adapter usually means the \
         container GPU is not reachable)",
        report.worst_name, report.worst_bytes, report.tensors
    ))
}

// ---------------------------------------------------------------------
// Host-memory pre-flight for image generation
// ---------------------------------------------------------------------
//
// The device-cap check above catches one tensor too large for a single
// binding. Image generation fails a different way: every tensor fits,
// but the TRANSIENT working set of a large canvas pushes the process
// past what the machine can hand out — and on unified memory the
// kernel's only remaining move is to thrash. The symptom is a frozen
// desktop, not an error return (a 512-pixel run took this machine down
// on 2026-08-28). The working set is predictable from the canvas size,
// so the honest moment to refuse is before the pipeline mutex is taken.

/// Beyond the measured range the per-pixel term inflates by 5/4 —
/// applied only to the pixels PAST the measurement, so the estimate is
/// continuous at the boundary instead of jumping by a constant.
const EXTRAPOLATION_SAFETY: (u64, u64) = (5, 4);

/// Cushion a generation should leave behind when it genuinely asks the
/// machine for new memory. Reclaiming inactive pages is real but not
/// instantaneous and a burst of GPU allocations outruns it. Capped at
/// the growth actually requested (see [`check_image_fit`]): demanding
/// two spare gigabytes to allocate nothing would refuse runs that
/// cannot fail.
pub const DEFAULT_HEADROOM_BYTES: u64 = 2 * 1024 * 1024 * 1024;

pub struct ImageDelta {
    pub bytes: u64,
    /// Canvas is larger than anything measured — the estimate carries
    /// the safety inflation and says so in any refusal.
    pub extrapolated: bool,
}

/// Predicted working-set growth for a canvas of `pixels` output pixels
/// under a pipeline's own measured curve.
pub fn estimate_image_delta(
    model: &combs_diffusion::WorkingSet,
    pixels: u64,
) -> ImageDelta {
    let measured = pixels.min(model.measured_max_pixels);
    let excess = pixels.saturating_sub(model.measured_max_pixels);
    let inflated_excess =
        excess.saturating_mul(EXTRAPOLATION_SAFETY.0) / EXTRAPOLATION_SAFETY.1;
    let variable = model
        .bytes_per_pixel
        .saturating_mul(measured.saturating_add(inflated_excess));
    ImageDelta {
        bytes: model.fixed_bytes.saturating_add(variable),
        extrapolated: excess > 0,
    }
}

/// What the refusal decision is made of. Split out so the arithmetic is
/// testable without a live process or a live machine.
pub struct ImageFitInputs {
    pub width: u64,
    pub height: u64,
    /// Process footprint once weights were resident, before any run.
    pub resident_at_load: u64,
    /// Process footprint right now.
    pub current_footprint: u64,
    /// Largest canvas this process has already completed, in pixels.
    /// A warm pool only counts toward a canvas no larger than one it
    /// has already served: the allocator hands back pages sized for
    /// the shapes it has seen, and a bigger canvas needs bigger
    /// buffers than those pages can satisfy.
    pub largest_completed_pixels: u64,
    pub available: u64,
    pub headroom: u64,
}

/// Refusal worded to act on: what it needs, what is free, what to do.
pub fn check_image_fit(
    model: &combs_diffusion::WorkingSet,
    inputs: &ImageFitInputs,
) -> Result<(), String> {
    let pixels = inputs.width.saturating_mul(inputs.height);
    let delta = estimate_image_delta(model, pixels);
    let estimated_peak = inputs.resident_at_load.saturating_add(delta.bytes);
    // Growth this process has already proven it can hold counts against
    // the request — but only for a canvas it has actually served at
    // least this large. Scaling up is priced as if the pool were cold.
    let already_held = if pixels <= inputs.largest_completed_pixels {
        inputs.current_footprint
    } else {
        inputs.resident_at_load.max(
            inputs
                .current_footprint
                .min(inputs.resident_at_load.saturating_add(
                    estimate_image_delta(model, inputs.largest_completed_pixels).bytes,
                )),
        )
    };
    let additional = estimated_peak.saturating_sub(already_held);
    // Never ask for more cushion than the growth being requested: a run
    // that allocates nothing new introduces no new risk to absorb.
    let need = additional.saturating_add(inputs.headroom.min(additional));
    if need <= inputs.available {
        return Ok(());
    }
    let mb = |b: u64| b / (1024 * 1024);
    let note = if delta.extrapolated {
        " (canvas is larger than any measured run — the estimate is \
         extrapolated and deliberately cautious)"
    } else {
        ""
    };
    Err(format!(
        "not enough free memory for a {}x{} generation: needs about {} MB more \
         than this process already holds (estimated peak {} MB), plus {} MB of \
         headroom, but only {} MB is free{note} — close other applications or \
         generate a smaller image",
        inputs.width,
        inputs.height,
        mb(additional),
        mb(estimated_peak),
        mb(inputs.headroom.min(additional)),
        mb(inputs.available),
    ))
}

/// True where the accelerator draws on host memory, so host free memory
/// is what decides whether a canvas fits. A discrete card holds the
/// working set in its own VRAM and this whole model says nothing about
/// it — that case wants a VRAM probe, and until there is one, silence
/// beats a wrong refusal.
pub fn draws_on_host_memory(device_type: &str) -> bool {
    matches!(device_type, "IntegratedGpu" | "Cpu" | "Other")
}

/// Everything the decision needs that is not the canvas itself.
pub struct PreflightContext {
    /// The loaded pipeline's own measured curve; `None` disables the
    /// check rather than borrowing another pipeline's numbers.
    pub working_set: Option<combs_diffusion::WorkingSet>,
    pub resident_at_load: u64,
    pub largest_completed_pixels: u64,
}

/// `Some(message)` when this canvas should not be attempted here and
/// now. Shared by the worker and the one-shot CLI so a generation
/// cannot escape the check by taking the other door. Any missing
/// measurement skips: refusing work we cannot price would be worse than
/// the risk it guards.
pub fn image_refusal(ctx: &PreflightContext, width: u32, height: u32) -> Option<String> {
    let model = ctx.working_set.as_ref()?;
    if ctx.resident_at_load == 0 {
        return None;
    }
    let available = env_mb("COMBS_IMAGE_PREFLIGHT_FREE_MB").or_else(available_memory_bytes)?;
    let current_footprint = process_footprint_bytes()?;
    check_image_fit(
        model,
        &ImageFitInputs {
            width: width as u64,
            height: height as u64,
            resident_at_load: ctx.resident_at_load,
            current_footprint,
            largest_completed_pixels: ctx.largest_completed_pixels,
            available,
            headroom: env_mb("COMBS_IMAGE_PREFLIGHT_HEADROOM_MB")
                .unwrap_or(DEFAULT_HEADROOM_BYTES),
        },
    )
    .err()
}

/// `COMBS_IMAGE_PREFLIGHT=0` closes the check entirely.
pub fn preflight_disabled() -> bool {
    matches!(std::env::var("COMBS_IMAGE_PREFLIGHT").as_deref(), Ok("0"))
}

fn env_mb(key: &str) -> Option<u64> {
    std::env::var(key).ok()?.parse::<u64>().ok().map(|mb| mb * 1024 * 1024)
}

/// Memory the machine can hand out, or `None` where we have no probe we
/// trust — an unknown platform skips the check rather than guessing.
#[cfg(target_os = "macos")]
pub fn available_memory_bytes() -> Option<u64> {
    // free + speculative + inactive: macOS keeps "free" deliberately
    // small, and inactive pages are what a large allocation actually
    // draws on. Optimistic by design — DEFAULT_HEADROOM_BYTES is the
    // counterweight.
    let mut stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // libc marks mach_host_self deprecated in favour of the mach2
    // crate; one call does not justify a new dependency, and the port
    // name it returns is stable.
    #[allow(deprecated)]
    let ok = unsafe {
        libc::host_statistics64(
            libc::mach_host_self(),
            libc::HOST_VM_INFO64,
            &mut stats as *mut _ as *mut libc::integer_t,
            &mut count,
        )
    };
    if ok != 0 {
        return None;
    }
    let page = unsafe { libc::vm_page_size } as u64;
    Some(
        (stats.free_count as u64 + stats.speculative_count as u64
            + stats.inactive_count as u64)
            * page,
    )
}

#[cfg(target_os = "linux")]
pub fn available_memory_bytes() -> Option<u64> {
    parse_kb_field(&std::fs::read_to_string("/proc/meminfo").ok()?, "MemAvailable:")
}

/// Pull a `kB`-suffixed field out of a /proc table. Kept off the
/// platform gate so the parsing is tested everywhere, not only where it
/// runs ("MemAvailable:" is the kernel's own estimate of what a fresh
/// allocation can get — exactly this question).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_kb_field(table: &str, field: &str) -> Option<u64> {
    table
        .lines()
        .find_map(|l| l.strip_prefix(field))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|kb| kb.parse::<u64>().ok())
        .map(|kb| kb * 1024)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn available_memory_bytes() -> Option<u64> {
    None
}

/// This process's physical footprint — the number Activity Monitor
/// shows, which unlike RSS counts GPU allocations on unified memory.
#[cfg(target_os = "macos")]
pub fn process_footprint_bytes() -> Option<u64> {
    let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
    let ok = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as libc::c_int,
            libc::RUSAGE_INFO_V2,
            &mut info as *mut _ as *mut libc::rusage_info_t,
        )
    };
    (ok == 0).then_some(info.ri_phys_footprint)
}

/// No probe off macOS yet: Linux's VmRSS omits GPU allocations (GTT
/// pages an integrated driver maps are not charged to the process), and
/// the whole model is denominated in a footprint that INCLUDES them.
/// Reporting VmRSS here would make every run after the first look cold
/// and refuse it. Skipping is the honest answer until someone measures
/// the platform properly.
#[cfg(not(target_os = "macos"))]
pub fn process_footprint_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(name: &str, elements: u64, packed: Option<(u64, usize)>) -> FitRow {
        FitRow { name: name.into(), elements, packed }
    }

    #[test]
    fn packed_rank2_uses_packed_bytes() {
        // Q8_0 math from the gguf fixtures: (64*8/32)*34 packed vs 64*8*4 dense.
        let r = row("w", 64 * 8, Some(((64 * 8 / 32) * 34, 2)));
        assert_eq!(r.device_bytes(), (64 * 8 / 32) * 34);
    }

    #[test]
    fn packed_non_rank2_falls_back_dense() {
        let r = row("w", 100, Some((10, 1)));
        assert_eq!(r.device_bytes(), 400);
    }

    #[test]
    fn dense_is_elements_times_four() {
        // A large vocab embed: 152064 × 3584 f32.
        let r = row("token_embd.weight", 152_064 * 3_584, None);
        assert_eq!(r.device_bytes(), 2_179_989_504);
    }

    #[test]
    fn report_finds_the_worst_and_refusal_names_it() {
        let report = fit_report(vec![
            row("small", 10, None),
            row("token_embd.weight", 152_064 * 3_584, None),
            row("mid", 1000, None),
        ])
        .unwrap();
        assert_eq!(report.worst_name, "token_embd.weight");
        assert_eq!(report.tensors, 3);
        let err = check_fit(&report, 134_217_728, "llvmpipe", "Cpu").unwrap_err();
        assert!(err.contains("llvmpipe (Cpu)"));
        assert!(err.contains("token_embd.weight"));
        assert!(err.contains("2179989504"));
        assert!(check_fit(&report, u64::MAX, "M3", "IntegratedGpu").is_ok());
    }

    #[test]
    fn packed_embed_counts_dense_and_refuses() {
        // A Q6_K token_embd is packed rank-2 but LOADS dense — the
        // walk passes packed: None for embeds, so the dense size
        // drives the refusal.
        let report = fit_report(vec![row("model.embed_tokens.weight", 152_064 * 3_584, None)])
            .unwrap();
        assert!(check_fit(&report, 2_147_483_647, "llvmpipe", "Cpu").is_err());
    }

    #[test]
    fn empty_model_yields_no_report() {
        assert!(fit_report(vec![]).is_none());
    }

    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * MB;
    const RESIDENT: u64 = 8_445 * MB;

    /// klein's own curve, as the pipeline reports it.
    fn klein() -> combs_diffusion::WorkingSet {
        combs_diffusion::WorkingSet {
            fixed_bytes: 1_478_130_074,
            bytes_per_pixel: 5_829,
            measured_max_pixels: 512 * 512,
        }
    }

    fn inputs(
        width: u64,
        height: u64,
        current: u64,
        largest_completed_pixels: u64,
        available: u64,
    ) -> ImageFitInputs {
        ImageFitInputs {
            width,
            height,
            resident_at_load: RESIDENT,
            current_footprint: current,
            largest_completed_pixels,
            available,
            headroom: DEFAULT_HEADROOM_BYTES,
        }
    }

    /// The curve must reproduce the runs it was fitted to (§56): within
    /// a megabyte of 1774 MB at 256px and 2867 MB at 512px.
    #[test]
    fn delta_model_reproduces_the_measured_runs() {
        let small = estimate_image_delta(&klein(), 256 * 256);
        assert!(small.bytes.abs_diff(1_774 * MB) < MB, "256px: {} MB", small.bytes / MB);
        assert!(!small.extrapolated);
        let large = estimate_image_delta(&klein(), 512 * 512);
        assert!(large.bytes.abs_diff(2_867 * MB) < MB, "512px: {} MB", large.bytes / MB);
        assert!(!large.extrapolated);
    }

    /// Past the measured range only the EXCESS pixels inflate, so the
    /// curve stays continuous at the boundary — a canvas one pixel
    /// larger than the last measured one must not jump hundreds of
    /// megabytes.
    #[test]
    fn extrapolation_is_continuous_at_the_boundary() {
        let at = estimate_image_delta(&klein(), 512 * 512).bytes;
        let just_past = estimate_image_delta(&klein(), 512 * 512 + 1);
        assert!(just_past.extrapolated);
        assert!(just_past.bytes - at < 16 * 1024, "jumped {} bytes", just_past.bytes - at);
        let big = estimate_image_delta(&klein(), 1024 * 1024);
        let straight = klein().fixed_bytes + klein().bytes_per_pixel * 1024 * 1024;
        assert!(big.bytes > straight, "extrapolation must be cautious");
        assert!(big.bytes < straight * 2, "…but not absurd");
    }

    /// The failing case from 2026-08-28: worker loaded, desktop busy, a
    /// cold 512px request. Refused, with the numbers and the way out.
    #[test]
    fn cold_pool_on_a_busy_machine_is_refused_with_numbers() {
        let err = check_image_fit(&klein(), &inputs(512, 512, RESIDENT, 0, 2 * GB)).unwrap_err();
        assert!(err.contains("512x512"), "{err}");
        assert!(err.contains("MB more"), "growth is named: {err}");
        // The peak this predicts is the one the crashing run actually
        // reached (11312 MB measured, §56) — a live check on the curve.
        assert!(err.contains("estimated peak 1131"), "{err}");
        assert!(err.contains("smaller image"), "{err}");
        assert!(!err.contains("extrapolated"), "512px is measured: {err}");
    }

    /// A pool already holding this exact shape asks the machine for
    /// nothing, so no cushion is demanded either — the check must not
    /// block a run that cannot fail. (Found by review: a flat headroom
    /// refused every repeat generation on the very machine this
    /// feature targets.)
    #[test]
    fn a_warm_pool_asking_for_nothing_is_never_refused() {
        let warm = RESIDENT + 2_867 * MB;
        for available in [0, MB, 512 * MB, 2 * GB] {
            assert!(
                check_image_fit(&klein(), &inputs(512, 512, warm, 512 * 512, available)).is_ok(),
                "warm repeat refused at {} MB free",
                available / MB
            );
        }
    }

    /// Scaling UP is priced cold even with a warm pool: pages sized for
    /// a small canvas cannot satisfy a larger one's buffers.
    #[test]
    fn a_bigger_canvas_gets_no_credit_for_a_smaller_run() {
        let warm_from_256 = RESIDENT + 1_774 * MB;
        let up = check_image_fit(
            &klein(),
            &inputs(512, 512, warm_from_256, 256 * 256, 1500 * MB),
        );
        assert!(up.is_err(), "scaling up on a small pool must not be waved through");
        // …while repeating the smaller shape stays allowed.
        assert!(
            check_image_fit(&klein(), &inputs(256, 256, warm_from_256, 256 * 256, 100 * MB))
                .is_ok()
        );
    }

    #[test]
    fn ample_free_memory_allows_a_cold_run() {
        assert!(check_image_fit(&klein(), &inputs(512, 512, RESIDENT, 0, 8 * GB)).is_ok());
    }

    /// Non-square canvases price by area and report their real shape.
    #[test]
    fn refusal_reports_the_actual_canvas() {
        let err = check_image_fit(&klein(), &inputs(1024, 512, RESIDENT, 0, GB)).unwrap_err();
        assert!(err.contains("1024x512"), "{err}");
        assert!(err.contains("extrapolated"), "beyond measured area: {err}");
    }

    /// Only pipelines that carry their own measurements are checked;
    /// nobody inherits klein's curve.
    #[test]
    fn an_unmeasured_pipeline_is_not_checked() {
        let ctx = PreflightContext {
            working_set: None,
            resident_at_load: RESIDENT,
            largest_completed_pixels: 0,
        };
        assert!(image_refusal(&ctx, 1024, 1024).is_none());
    }

    /// A discrete card keeps its working set in VRAM, where host free
    /// memory says nothing about it.
    #[test]
    fn host_memory_only_decides_for_host_backed_accelerators() {
        assert!(draws_on_host_memory("IntegratedGpu"));
        assert!(draws_on_host_memory("Cpu"));
        assert!(!draws_on_host_memory("DiscreteGpu"));
        assert!(!draws_on_host_memory("VirtualGpu"));
    }

    #[test]
    fn proc_tables_parse_kb_fields() {
        let meminfo = "MemTotal:       32000000 kB\nMemFree:  100 kB\nMemAvailable:   12345678 kB\n";
        assert_eq!(parse_kb_field(meminfo, "MemAvailable:"), Some(12_345_678 * 1024));
        assert_eq!(parse_kb_field("VmRSS:\t  4096 kB\n", "VmRSS:"), Some(4 * MB));
        assert_eq!(parse_kb_field(meminfo, "Nope:"), None);
        assert_eq!(parse_kb_field("MemAvailable:   what kB\n", "MemAvailable:"), None);
    }

    /// The live probes must agree with reality on this machine: a
    /// footprint in a plausible range, and free memory under total.
    #[test]
    fn live_probes_return_plausible_numbers() {
        let footprint = process_footprint_bytes().expect("footprint probe");
        assert!(footprint > 1024 * 1024, "implausibly small: {footprint}");
        assert!(footprint < 64 * GB, "implausibly large: {footprint}");
        let available = available_memory_bytes().expect("available probe");
        assert!(available > 0);
        assert!(available < 1024 * GB);
    }
}
