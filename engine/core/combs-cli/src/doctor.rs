//! `combs doctor` — one command that answers whether this machine's
//! stack can be trusted with real work.
//!
//! Why it exists: `/health` is not a gate. A Vulkan stack with no ICD
//! hands you a CPU rasterizer that answers every request cheerfully and
//! serves garbage, and the provisioning scripts could only grep an
//! adapter name for "discrete|nvidia" — which llvmpipe passes on a box
//! that has an NVIDIA card its container cannot reach.
//!
//! What it is NOT: new probe code. Every canary here already exists as
//! library code and is already trusted; this sequences them, gives them
//! one verdict vocabulary, and turns the answer into an exit code a
//! shell script can gate on. Two definitions of a canary is how one
//! silently stops matching the other.
//!
//! The refusals are the product. A doctor that cannot say no is not a
//! gate.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use serde_json::{Value, json};

use crate::build_info;

#[derive(clap::Args)]
pub struct DoctorArgs {
    /// Emit the verdict as JSON instead of human lines.
    #[arg(long)]
    pub json: bool,
    /// Model to prove end-to-end (preset id or path). Without it the
    /// e2e check reports `skip`, which is never `pass`.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// Run the canaries only; do not generate even if --model is given.
    #[arg(long)]
    pub skip_e2e: bool,
    /// Tokens to generate for the e2e check.
    #[arg(long, default_value_t = 16)]
    pub e2e_tokens: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum Status {
    Pass,
    Fail,
    Skip,
}

impl Status {
    fn word(self) -> &'static str {
        match self {
            Status::Pass => "pass",
            Status::Fail => "FAIL",
            Status::Skip => "skip",
        }
    }
}

struct Check {
    name: &'static str,
    status: Status,
    detail: String,
    ms: u128,
}

/// Adapters that render on the CPU while presenting as a GPU. The name
/// list is the fast path; `device_type == "Cpu"` is the structural
/// backstop behind it, because a rasterizer we have never heard of is
/// still a rasterizer.
const SOFTWARE_RASTERIZERS: &[&str] = &["llvmpipe", "swiftshader", "lavapipe", "softpipe", "warp"];

fn is_software_rasterizer(name: &str, device_type: &str) -> Option<String> {
    let lowered = name.to_ascii_lowercase();
    for needle in SOFTWARE_RASTERIZERS {
        if lowered.contains(needle) {
            return Some(format!(
                "software rasterizer refused: adapter is \"{name}\" ({needle}) — \
                 this renders on the CPU and will serve plausible garbage at a crawl"
            ));
        }
    }
    if device_type.eq_ignore_ascii_case("Cpu") {
        return Some(format!(
            "software rasterizer refused: adapter \"{name}\" reports device type Cpu"
        ));
    }
    None
}

/// A stable, dependency-free digest so two machines can be compared by
/// one short string. FNV-1a — a checksum for agreement, not a security
/// primitive, and named as such wherever it is printed.
fn transcript_digest(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, u128) {
    let start = Instant::now();
    let value = f();
    (value, start.elapsed().as_millis())
}

pub fn run(args: DoctorArgs) -> Result<()> {
    let mut checks: Vec<Check> = Vec::new();
    let mut device_json = json!(null);

    // --- device -----------------------------------------------------
    // On the CPU floor there is no adapter to enumerate, and asking wgpu
    // anyway would report a GPU this build never touches — which is how
    // a doctor comes to say "trusted" about hardware it is not using.
    #[cfg(feature = "cpu")]
    {
        checks.push(Check {
            name: "device",
            status: Status::Pass,
            detail: "cpu backend (ndarray) — this build computes on the CPU by design".into(),
            ms: 0,
        });
        device_json = json!({ "name": "cpu (ndarray)", "backend": "cpu", "device_type": "Cpu" });
        for name in ["wgsl", "batched"] {
            checks.push(Check {
                name,
                status: Status::Skip,
                detail: "not run: these probe a GPU compiler, and this build has none".into(),
                ms: 0,
            });
        }
        return finish(args, checks, device_json);
    }

    #[cfg(not(feature = "cpu"))]
    {
    // Ask whether there is an adapter AT ALL before touching the
    // runtime. Measured on arm64 Linux in a container with no GPU:
    // `device_caps` panicked twice — "No possible adapter available"
    // inside cubecl, then "an adapter exists" in our own expect — so
    // the one machine doctor exists to diagnose was the one machine it
    // died on. `gpu_available` enumerates without initializing, which
    // is why it can answer here.
    if !combs_core::gpu_available() {
        checks.push(Check {
            name: "device",
            status: Status::Fail,
            detail: concat!(
                "no GPU adapter is visible to wgpu on this machine — nothing to run on. ",
                "A build with --features cpu will serve here; this one will not.",
            )
            .into(),
            ms: 0,
        });
        for name in ["wgsl", "batched"] {
            checks.push(Check {
                name,
                status: Status::Skip,
                detail: "not run: there is no device to probe".into(),
                ms: 0,
            });
        }
        return finish(args, checks, json!({ "name": "none", "backend": "none" }));
    }
    // `device_caps` is what primes the runtime, and cubecl 0.10 panics
    // on a second init in one process — so this runs once and every
    // later check shares the default device it primed.
    let device = combs_core::init_device();
    let (caps, ms) = timed(|| combs_core::device_caps(&device));
    device_json = json!({
        "name": caps.name,
        "backend": caps.backend,
        "device_type": caps.device_type,
        "driver": caps.driver,
        "max_storage_buffer_binding_size": caps.max_storage_buffer_binding_size,
        "max_buffer_size": caps.max_buffer_size,
    });
    let device_ok = match is_software_rasterizer(&caps.name, &caps.device_type) {
        Some(refusal) => {
            checks.push(Check { name: "device", status: Status::Fail, detail: refusal, ms });
            false
        }
        None => {
            checks.push(Check {
                name: "device",
                status: Status::Pass,
                detail: format!("{} · {} · {}", caps.name, caps.backend, caps.device_type),
                ms,
            });
            true
        }
    };

    // --- the canaries -----------------------------------------------
    // Skipped when the device is already refused: probing a rasterizer
    // tells us nothing we do not know and costs minutes.
    if device_ok {
        let (result, ms) = timed(|| cubecl::future::block_on(combs_models::wgsl_probe_report()));
        match result {
            Ok(()) => checks.push(Check {
                name: "wgsl",
                status: Status::Pass,
                detail: "echo, shared memory and per-format quant gemv canaries agree".into(),
                ms,
            }),
            Err(e) => {
                checks.push(Check { name: "wgsl", status: Status::Fail, detail: e, ms })
            }
        }

        let (result, ms) = timed(|| cubecl::future::block_on(combs_models::batched_probe_report()));
        match result {
            Ok(()) => checks.push(Check {
                name: "batched",
                status: Status::Pass,
                detail: "batched matmul values match the host reference".into(),
                ms,
            }),
            Err(e) => {
                checks.push(Check { name: "batched", status: Status::Fail, detail: e, ms })
            }
        }
    } else {
        for name in ["wgsl", "batched"] {
            checks.push(Check {
                name,
                status: Status::Skip,
                detail: "not run: the device was refused".into(),
                ms: 0,
            });
        }
    }

    }

    #[cfg(not(feature = "cpu"))]
    return finish(args, checks, device_json);
}

/// Everything after the device and canary checks: the end-to-end run,
/// then the report. Shared so the CPU floor and the wgpu build cannot
/// drift into printing different shapes.
fn finish(args: DoctorArgs, mut checks: Vec<Check>, device_json: Value) -> Result<()> {
    let device_ok = !checks.iter().any(|c| c.name == "device" && c.status == Status::Fail);
    // --- end to end --------------------------------------------------
    match (&args.model, args.skip_e2e, device_ok) {
        (Some(model), false, true) => {
            let (result, ms) = timed(|| e2e(model, args.e2e_tokens));
            match result {
                Ok(digest) => checks.push(Check {
                    name: "e2e",
                    status: Status::Pass,
                    detail: format!("greedy transcript digest {digest} (fnv-1a, not a checksum of trust)"),
                    ms,
                }),
                Err(e) => checks.push(Check {
                    name: "e2e",
                    status: Status::Fail,
                    detail: format!("{e:#}"),
                    ms,
                }),
            }
        }
        (_, _, false) => checks.push(Check {
            name: "e2e",
            status: Status::Skip,
            detail: "not run: the device was refused".into(),
            ms: 0,
        }),
        (None, _, _) => checks.push(Check {
            name: "e2e",
            status: Status::Skip,
            detail: "not run: pass --model to prove generation on this machine".into(),
            ms: 0,
        }),
        (Some(_), true, _) => checks.push(Check {
            name: "e2e",
            status: Status::Skip,
            detail: "not run: --skip-e2e".into(),
            ms: 0,
        }),
    }

    let failed = checks.iter().any(|c| c.status == Status::Fail);

    if args.json {
        let rows: Vec<Value> = checks
            .iter()
            .map(|c| {
                json!({
                    "name": c.name,
                    "status": c.status.word().to_ascii_lowercase(),
                    "detail": c.detail,
                    "ms": c.ms,
                })
            })
            .collect();
        println!(
            "{:#}",
            json!({
                "verdict": if failed { "fail" } else { "pass" },
                "checks": rows,
                "device": device_json,
                "build": build_info::manifest(),
            })
        );
    } else {
        println!("combs doctor — {}", build_info::summary());
        for c in &checks {
            println!("  {:<8} {:<5} {}  ({} ms)", c.name, c.status.word(), c.detail, c.ms);
        }
        println!(
            "\nverdict: {}",
            if failed {
                "REFUSED — this machine is not trusted for real work"
            } else {
                "trusted"
            }
        );
    }

    if failed {
        std::process::exit(1);
    }
    Ok(())
}

/// A tiny greedy generation, hashed. Greedy so the digest means
/// something: two machines that agree here agree about arithmetic.
fn e2e(model: &PathBuf, max_tokens: usize) -> Result<String> {
    use combs_runtime::{Engine, GenerationConfig};

    let resolved = crate::resolve_model_arg(model)?;
    let source = combs_formats::open_model_source(&resolved)?;
    let device = combs_core::init_device();
    let engine = Engine::load(&source, device)?;

    let prompt = "The capital of France is";
    let tokens = engine.encode(prompt)?;
    let mut config: GenerationConfig = engine.default_config();
    config.max_tokens = max_tokens;
    // Greedy, unseeded, no penalties: anything else makes the digest a
    // statement about the sampler rather than about the machine.
    config.sampling.temperature = 0.0;
    config.sampling.top_k = None;
    config.sampling.top_p = None;
    config.sampling.repetition_penalty = None;
    config.sampling.frequency_penalty = None;
    config.sampling.presence_penalty = None;
    config.sampling.seed = None;

    let mut text = String::new();
    let mut emit = |_id: u32, piece: &str, _lp: Option<&combs_runtime::TokenLogprobs>| {
        text.push_str(piece);
    };
    engine.generate(&tokens, &config, &mut emit)?;
    if text.trim().is_empty() {
        anyhow::bail!("the model generated nothing");
    }
    Ok(transcript_digest(&text))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The refusal predicate is the one piece of doctor that must work
    // on a machine we cannot force into a bad state, so it is tested
    // against the adapter strings those stacks actually report.
    #[test]
    fn names_the_rasterizer_it_refuses() {
        let refusal =
            is_software_rasterizer("llvmpipe (LLVM 15.0.7, 256 bits)", "Cpu").expect("refused");
        assert!(refusal.contains("llvmpipe"), "the adapter must be named: {refusal}");
        assert!(is_software_rasterizer("SwiftShader Device (LLVM 10)", "Cpu").is_some());
        assert!(is_software_rasterizer("llvmpipe", "DiscreteGpu").is_some());
    }

    #[test]
    fn an_unknown_rasterizer_is_caught_by_its_device_type() {
        let refusal = is_software_rasterizer("Mesa Something New", "Cpu")
            .expect("device type Cpu is the backstop");
        assert!(refusal.contains("Cpu"));
    }

    #[test]
    fn real_gpus_pass() {
        assert!(is_software_rasterizer("Apple M3 Pro", "IntegratedGpu").is_none());
        assert!(is_software_rasterizer("NVIDIA GeForce RTX 4090", "DiscreteGpu").is_none());
        // The name that once passed provision.sh's grep on a box whose
        // container could not reach the card.
        assert!(is_software_rasterizer("NVIDIA A100", "DiscreteGpu").is_none());
    }

    #[test]
    fn the_digest_is_stable_and_distinguishing() {
        assert_eq!(transcript_digest("Paris."), transcript_digest("Paris."));
        assert_ne!(transcript_digest("Paris."), transcript_digest("Paris!"));
        assert_eq!(transcript_digest("Paris.").len(), 16);
    }
}
