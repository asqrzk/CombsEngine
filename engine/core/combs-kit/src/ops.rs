//! The operations matrix: what this engine can do, where, and WHY NOT.
//!
//! The question this answers is the one no page in the estate answered:
//! "can I do X on Windows?" — and, when the answer is no, the reason
//! rather than a shrug. A developer hitting a silent stand-down learns
//! nothing; a developer told "there is no host-memory probe on this
//! platform, so the gate stands down by name" can decide what to do.
//!
//! Four states, and `Unproven` is the important one. It is an honest
//! answer that the estate has never sat on that box, and it is what
//! keeps this table from quietly becoming marketing. A row may only
//! claim `Works` with evidence naming a ledger section or a test.

pub enum State {
    /// Proven here. `evidence` names a ledger section or a test.
    Works,
    /// The code refuses it by name. `evidence` names file:line.
    Refused,
    /// The stack cannot express it at all. `evidence` gives the reason.
    Unavailable,
    /// Nobody has sat on that box. Said, never guessed.
    Unproven,
}

pub struct Op {
    pub id: &'static str,
    pub title: &'static str,
    /// Platform, backend, or build this row speaks about. "any" when the
    /// answer does not vary.
    pub wher: &'static str,
    pub state: State,
    pub reason: &'static str,
    pub evidence: &'static str,
}

pub const OPS: &[Op] = &[
    Op {
        id: "llm-memory-preflight",
        title: "refuse to serve a model that will not fit host memory",
        wher: "macos",
        state: State::Works,
        reason: "the Mach probe reads free, speculative and inactive pages, which is what a large allocation actually draws on",
        evidence: "combs-core/src/host.rs available_memory_bytes",
    },
    Op {
        id: "llm-memory-preflight",
        title: "refuse to serve a model that will not fit host memory",
        wher: "linux",
        state: State::Works,
        reason: "MemAvailable is the kernel's own estimate of what a fresh allocation can get, which is exactly this question",
        evidence: "combs-core/src/host.rs available_memory_bytes",
    },
    Op {
        id: "llm-memory-preflight",
        title: "refuse to serve a model that will not fit host memory",
        wher: "windows",
        state: State::Unavailable,
        reason: "no host-memory probe exists on this platform, so the gate STANDS DOWN BY NAME rather than guess. You will not get a refusal here; you will get whatever the machine does when it runs out",
        evidence: "combs-core/src/host.rs available_memory_bytes; the fit.standdown provenance event",
    },
    Op {
        id: "llm-memory-preflight",
        title: "refuse to serve a model that will not fit host memory",
        wher: "discrete gpu, any platform",
        state: State::Unavailable,
        reason: "the model prices only hosts that draw on host memory; a discrete card holds the working set in its own VRAM and this arithmetic says nothing about it. A pod with a real card therefore has NO fit gate at all",
        evidence: "combs-cli/src/fit.rs draws_on_host_memory",
    },
    Op {
        id: "image-warm-pool-credit",
        title: "credit memory a previous render already holds, so a second image of the same size is not refused",
        wher: "macos",
        state: State::Works,
        reason: "the footprint probe counts GPU allocations on unified memory, which is what the credit is denominated in",
        evidence: "combs-core/src/host.rs process_footprint_bytes",
    },
    Op {
        id: "image-warm-pool-credit",
        title: "credit memory a previous render already holds",
        wher: "linux, windows",
        state: State::Unavailable,
        reason: "Linux VmRSS omits the GPU pages an integrated driver maps, so reporting it would make every run after the first look cold and refuse it. Skipping is the honest answer until someone measures the platform properly",
        evidence: "combs-core/src/host.rs process_footprint_bytes",
    },
    Op {
        id: "f16-storage-kernels",
        title: "hand-written kernels that read and write f16 directly",
        wher: "browser",
        state: State::Unavailable,
        reason: "a WGSL `enable` directive is rejected by the browser's validator, and the invalid pipeline then dispatches as a SILENT NO-OP — the model decodes zeros with no error anywhere. The kernel contract forbids `enable` for exactly this reason",
        evidence: "combs-models/src/wgsl/mod.rs, the contract header",
    },
    Op {
        id: "f16-storage-kernels",
        title: "hand-written kernels that read and write f16 directly",
        wher: "macos / metal",
        state: State::Works,
        reason: "rmsnorm, attention and the three projection gemvs read and write f16 in place with f32 arithmetic; perplexity identical to four decimals, five of five transcripts byte-identical; decode doubles at 2.2k. On by default here",
        evidence: "MEASUREMENTS §96, §97, §98, §100",
    },
    Op {
        id: "f16-storage-kernels",
        title: "hand-written kernels that read and write f16 directly",
        wher: "vulkan / dx12 (linux, windows, android)",
        state: State::Unproven,
        reason: "the kernels exist and the adapter check is the same, but a shader `enable` directive is exactly what a second driver may handle differently. Off by default there until one sitting; COMBS_WGSL_F16=1 opens it for the sitting",
        evidence: "MEASUREMENTS §97; the matrix's own rule that a default is a promise to every box",
    },
    Op {
        id: "fused-f16-backend",
        title: "run the f16 backend through burn-fusion, as f32 does",
        wher: "any",
        state: State::Refused,
        reason: "the fused reduce's autotune resolves a global argument that was never registered, and that failure then desynchronises the fusion queue so every later operation panics. The worker dies before the first token",
        evidence: "MEASUREMENTS §91; UPSTREAM-ISSUES #6",
    },
    Op {
        id: "bf16-anywhere",
        title: "compute in bfloat16",
        wher: "any",
        state: State::Unavailable,
        reason: "the WGSL compiler rejects it, and the failed autotune poisons the process rather than falling back",
        evidence: "MEASUREMENTS §65",
    },
    Op {
        id: "f8-f4-compute",
        title: "compute in 8-bit or 4-bit float",
        wher: "any",
        state: State::Unavailable,
        reason: "reachable only as NVIDIA tensor-core paths through CUDA or PTX, and this engine speaks one kernel language to every platform by design. Storage at 8 and 4 bits is a different question and already exists as the quantization road",
        evidence: "the plan's F.8 verdict",
    },
    Op {
        id: "dx12",
        title: "run on Direct3D 12 instead of Vulkan on Windows",
        wher: "windows",
        state: State::Unavailable,
        reason: "nothing selects it: the graphics API is chosen automatically and resolves to Vulkan on Windows. There is no door today, so DX12 has never been measured",
        evidence: "combs-core/src/lib.rs AutoGraphicsApi",
    },
    Op {
        id: "int8-kv",
        title: "quantize the key-value cache to int8",
        wher: "any, head_dim divisible by 32",
        state: State::Works,
        reason: "global layers quantize; the arena shrinks by roughly the ratio the cache module documents",
        evidence: "MEASUREMENTS §88",
    },
    Op {
        id: "int8-kv",
        title: "quantize the key-value cache to int8",
        wher: "any, head_dim not divisible by 32",
        state: State::Refused,
        reason: "refused out loud at load, with the head dim and the group named, and the flag corrected so the arena is priced at what it actually uses. It used to no-op silently AND keep reporting the quantized size — a room that read as fitting and did not",
        evidence: "combs-runtime/src/engine.rs resolve_kv_quant; the kv.quant.refused event",
    },
    Op {
        id: "software-rasterizer",
        title: "serve on llvmpipe, SwiftShader, lavapipe, softpipe or WARP",
        wher: "any",
        state: State::Refused,
        reason: "refused by adapter name AND by device type, because such a stack answers every request cheerfully and serves garbage at a crawl. A container missing its driver mount is the usual cause",
        evidence: "combs-cli/src/doctor.rs SOFTWARE_RASTERIZERS",
    },
    Op {
        id: "nvidia-vulkan",
        title: "the engine on an NVIDIA card over Vulkan",
        wher: "linux + nvidia",
        state: State::Unproven,
        reason: "the estate has never run on a real NVIDIA GPU; the one pod session ran on a CPU rasterizer and was refused. The provisioning runbook exists and gates on the doctor, but nobody has sat through it",
        evidence: "deployment-boundaries; the platform's pod postmortem",
    },
    Op {
        id: "amd-radv",
        title: "the engine on an AMD card over Vulkan",
        wher: "linux + amd",
        state: State::Unproven,
        reason: "no sitting yet",
        evidence: "deployment-boundaries",
    },
    Op {
        id: "android",
        title: "the engine on an Android device",
        wher: "android",
        state: State::Unproven,
        reason: "a compile lane exists and needs the NDK; no runtime record exists. Mobile GPUs are f16-first, so the f16 storage kernels are what would make this fast",
        evidence: "xtask android-arm64",
    },
    Op {
        id: "prefill-quantized-kernel",
        title: "prefill straight from packed weights",
        wher: "any",
        state: State::Unavailable,
        reason: "no quantized prefill kernel exists: every batched call dequantizes the whole weight to a dense transient and runs the tuned GEMM. This is the largest untaken lever on time-to-first-token",
        evidence: "combs-models/src/qlinear.rs, the batched threshold",
    },
    Op {
        id: "q5k-gemv",
        title: "decode a Q5_K model through a hand-written gemv",
        wher: "any",
        state: State::Works,
        reason: "the Q5_K gemv landed beside the Q4_K and Q6_K ones — the packed block's high bits are folded in place and the accumulator stays f32 — so a Q5_K projection no longer falls to the untiled cube path. It earns its place on the hybrid seats, whose fused qkv and output projections are Q5_K even inside a file sold as Q4_K_M",
        evidence: "combs-models/src/wgsl/decode_gemv_q5k.wgsl, and the gemv canary list in wgsl/mod.rs",
    },
    Op {
        id: "qwen35-hybrid",
        title: "run a gated-deltanet hybrid (the qwen3.5 / qwen3.8 family)",
        wher: "any",
        state: State::Works,
        reason: "24 recurrent layers beside 8 attention layers, with the recurrent state priced per session rather than per token — so KV is a quarter of a dense seat of the same size and the token barely slows with depth. Two caveats, both measured: prefill steps the recurrence per token in generic ops, so time-to-first-token is the worst on the roster, and the decode step kernel is opt-in because the shipped launch geometry made it slower, not faster",
        evidence: "combs-models/src/qwen35.rs and wgsl/deltanet.rs; the arch is read in combs-formats",
    },
    Op {
        id: "speculative-decode",
        title: "draft several tokens and verify them in one pass",
        wher: "greedy requests only",
        state: State::Works,
        reason: "prompt-lookup drafting with a multi-token greedy verify and a total rollback, opt-in behind its door. It pays on copy-heavy work and costs a little on ordinary prose, and it is refused outright on a hybrid seat because that cache cannot roll back",
        evidence: "combs-runtime/src/spec.rs and the verify path in step.rs",
    },
    Op {
        id: "q2k-q3k-iq",
        title: "load a Q2_K, Q3_K or IQ-family model at all",
        wher: "any",
        state: State::Refused,
        reason: "refused at header parse rather than half-supported. They would need a host dequant path first, so the perplexity cost could be measured before any kernel is written",
        evidence: "combs-formats gguf header parse",
    },
];

impl State {
    pub fn word(&self) -> &'static str {
        match self {
            State::Works => "works",
            State::Refused => "refused",
            State::Unavailable => "unavailable",
            State::Unproven => "unproven",
        }
    }
}

/// Rows for one operation id, or an empty slice when nothing is known
/// by that name — which the caller must not confuse with "no".
pub fn rows_for(id: &str) -> Vec<&'static Op> {
    OPS.iter().filter(|o| o.id == id).collect()
}

pub fn print_all(filter: Option<&str>) {
    let mut last = "";
    for op in OPS {
        if let Some(w) = filter {
            if !op.wher.contains(w) {
                continue;
            }
        }
        if op.id != last {
            println!("\n{} — {}", op.id, op.title);
            last = op.id;
        }
        println!("  {:<12} {:<32} {}", op.state.word(), op.wher, op.reason);
        println!("  {:<12} {:<32} evidence: {}", "", "", op.evidence);
    }
    println!();
}

/// One operation, answered. Exit 0 when every matching row works, 1 when
/// one does not, 2 when the name is unknown — because a question we did
/// not understand must never be answered "no".
pub fn answer(id: &str, wher: Option<&str>) -> i32 {
    let rows: Vec<&Op> = rows_for(id)
        .into_iter()
        .filter(|o| wher.is_none_or(|w| o.wher.contains(w) || o.wher == "any"))
        .collect();
    if rows.is_empty() {
        eprintln!(
            "combs-kit: no operation known as {id:?}{}. `combs-kit ops` lists them.",
            wher.map(|w| format!(" on {w:?}")).unwrap_or_default()
        );
        return 2;
    }
    let mut all_work = true;
    for op in &rows {
        println!("{}: {} on {}", op.id, op.state.word(), op.wher);
        println!("  {}", op.reason);
        println!("  evidence: {}", op.evidence);
        if !matches!(op.state, State::Works) {
            all_work = false;
        }
    }
    i32::from(!all_work)
}

pub fn as_json() -> serde_json::Value {
    serde_json::Value::Array(
        OPS.iter()
            .map(|o| {
                serde_json::json!({
                    "id": o.id,
                    "title": o.title,
                    "where": o.wher,
                    "state": o.state.word(),
                    "reason": o.reason,
                    "evidence": o.evidence,
                })
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole value of the table: a row that claims something without
    /// pointing at why is worse than no row, because it reads as
    /// authority. This is the test that keeps it from becoming
    /// marketing.
    #[test]
    fn every_row_carries_a_reason_and_evidence() {
        for op in OPS {
            assert!(!op.reason.trim().is_empty(), "{}/{}: no reason", op.id, op.wher);
            assert!(
                !op.evidence.trim().is_empty(),
                "{}/{}: no evidence",
                op.id,
                op.wher
            );
            assert!(!op.title.trim().is_empty(), "{}: no title", op.id);
        }
    }

    /// The driver's own question, pinned: a refactor that silently drops
    /// the stand-down must break a test rather than a person's trust.
    #[test]
    fn the_windows_memory_gate_says_why_it_cannot() {
        let rows = rows_for("llm-memory-preflight");
        let win = rows
            .iter()
            .find(|o| o.wher == "windows")
            .expect("the windows row exists");
        assert!(matches!(win.state, State::Unavailable));
        assert!(
            win.reason.contains("no host-memory probe"),
            "the reason must name the cause: {}",
            win.reason
        );
        assert!(
            win.reason.contains("STANDS DOWN"),
            "and must say the gate stands down rather than passes: {}",
            win.reason
        );
    }

    /// A question we did not understand is not a "no".
    #[test]
    fn an_unknown_operation_exits_two() {
        assert_eq!(answer("no-such-operation", None), 2);
    }

    /// int8 KV on an unsuitable head dim was the estate's most
    /// dangerous door: it did nothing, said nothing, and kept reporting
    /// the memory it was not saving. It is refused out loud now, and
    /// the table must say so rather than describe the old silence.
    #[test]
    fn int8_kv_on_a_bad_head_dim_is_a_loud_refusal_now() {
        let kv = rows_for("int8-kv");
        let bad = kv
            .iter()
            .find(|o| o.wher.contains("not divisible"))
            .expect("the unsuitable-head-dim row exists");
        assert!(matches!(bad.state, State::Refused), "silence is not a state we keep");
        assert!(bad.reason.contains("out loud"), "{}", bad.reason);
    }
}
