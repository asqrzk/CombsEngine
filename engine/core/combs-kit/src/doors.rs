//! Every environment door the engine reads, in one place for the first
//! time.
//!
//! About forty `COMBS_*` strings are read at about forty points across
//! four crates, and no page listed them until this one. A developer
//! asking "what can I even turn?" had to grep. `combs-kit doors` answers
//! it, and the drift test below keeps the answer true: it walks the
//! engine's own sources and fails the build when a door exists there and
//! not here, so the list cannot rot the way a hand-written page would.
//!
//! Data in code rather than a TOML file on purpose: the engine takes no
//! TOML dependency (its provenance module hand-escapes JSON rather than
//! add serde there), and a const table is data a test can walk.

/// What sort of thing this is. Without it the table reads as sixty-odd
/// knobs, when a dozen are build stamps the build system sets and
/// several are investigation cells that exist for one afternoon.
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Kind {
    /// A knob a developer may turn to change how the engine runs.
    Knob,
    /// Where something lives: a binary, a home, a template.
    Path,
    /// Set by the build, read for the stamp. Not a knob.
    Stamp,
    /// A diagnostic or an investigation cell, not a production door.
    Cell,
}

pub struct Door {
    pub kind: Kind,
    pub name: &'static str,
    pub default: &'static str,
    pub does: &'static str,
    pub read_at: &'static str,
    /// The ledger section proving this door moves a number, when one
    /// exists. `None` is an honest answer: the door is offered but its
    /// effect has never been measured here.
    pub measured: Option<&'static str>,
}

pub const DOORS: &[Door] = &[
    Door {
        kind: Kind::Knob,
        name: "COMBS_WGSL",
        default: "on",
        does: "the master door over every hand-written WGSL kernel; 0 sends every kernel back to the burn path",
        read_at: "combs-models/src/wgsl/mod.rs",
        measured: Some("MEASUREMENTS §90, §92"),
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_WGSL_GEMV",
        default: "on",
        does: "the packed-weight decode gemv kernels (q8_0, q4_k, q6_k)",
        read_at: "combs-models/src/wgsl/mod.rs",
        measured: Some("MEASUREMENTS §9"),
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_WGSL_GEMV_GEOMETRY",
        default: "16x16",
        does: "how every decode gemv workgroup is cut into output rows and lanes: 8x32, 16x16 or 32x8; any other value is refused aloud and 16x16 is used",
        read_at: "combs-models/src/wgsl/mod.rs",
        measured: None,
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_WGSL_RMSNORM",
        default: "on",
        does: "the fused RMSNorm kernel; accepts f16 since §92",
        read_at: "combs-models/src/wgsl/rmsnorm.rs",
        measured: Some("MEASUREMENTS §92"),
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_WGSL_DELTANET",
        default: "off",
        does: "1 runs the Gated DeltaNet decode step as two WGSL dispatches (qwen35); opt-in: it cuts dispatches but decodes slower than the generic-op road today",
        read_at: "combs-models/src/wgsl/deltanet.rs",
        measured: Some("MEASUREMENTS §113"),
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_WGSL_ROPE",
        default: "on",
        does: "the fused rope kernel for q and k in one dispatch; accepts f16 since §92",
        read_at: "combs-models/src/wgsl/rope.rs",
        measured: Some("MEASUREMENTS §92"),
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_WGSL_ATTN",
        default: "on",
        does: "paged decode attention, including the split-K and int8 variants",
        read_at: "combs-models/src/wgsl/decode_attn.rs",
        measured: Some("MEASUREMENTS §9"),
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_WGSL_PREFILL",
        default: "on",
        does: "paged causal prefill attention",
        read_at: "combs-models/src/wgsl/decode_attn.rs",
        measured: None,
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_NO_QUANT_KERNELS",
        default: "off",
        does: "forces the portable dense path: weights dequantized at load. Costs the memory win; isolates kernel bugs",
        read_at: "combs-models/src/qlinear.rs",
        measured: None,
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_KV_SCRUB",
        default: "on",
        does: "clear every KV buffer the engine frees before the pool hands it on — a session's arena and sliding window at its end, the old arena at growth, the old window at every sliding step — with one constant fill each; 0 leaves the freed bytes for the next allocation to read",
        read_at: "combs-models/src/wgsl/scrub.rs, combs-models/src/kv.rs",
        measured: None,
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_KV_EVENTS",
        default: "off",
        does: "index the KV pages the live sessions hold and publish them on /v1/kv/events, as a sequenced log a cache-aware router's indexer applies; off indexes nothing and the route refuses by name",
        read_at: "combs-runtime/src/kvevents.rs, combs-runtime/src/engine.rs",
        measured: None,
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_KV_QUANT",
        default: "off",
        does: "int8 KV on global layers. No-ops silently when head_dim does not divide by 32",
        read_at: "combs-models/src/kv.rs",
        measured: Some("MEASUREMENTS §88"),
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_KV",
        default: "paged",
        does: "paged or contiguous KV cache; contiguous turns off prefix reuse and speculative decode",
        read_at: "combs-runtime",
        measured: None,
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_QMATMUL_OPERAND",
        default: "f32",
        does: "f16 operands into the batched prefill matmul, f32 accumulate",
        read_at: "combs-models/src/qlinear.rs",
        measured: Some("MEASUREMENTS §65"),
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_QMATMUL_BATCHED",
        default: "on",
        does: "the dense-transient prefill path for m >= 8",
        read_at: "combs-models/src/qlinear.rs",
        measured: None,
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_MEM_POOLS",
        default: "exclusive",
        does: "exclusive pages, or cubecl's sub-slice pools",
        read_at: "combs-core/src/lib.rs",
        measured: Some("MEASUREMENTS §37"),
    },
    Door {
        kind: Kind::Knob,
        name: "CUBECL_WGPU_MAX_TASKS",
        default: "16 (the house default; cubecl's own is 32)",
        does: "dispatches per queue submit. The decode step ends in a blocking readback, so coarse batching serializes encode against execute",
        read_at: "combs-core/src/lib.rs",
        measured: Some("MEASUREMENTS §90 — 16 measured best here; the optimum is a property of the driver"),
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_TRACE_DECODE",
        default: "off",
        does: "prints the decode step's time split and its dispatches per token",
        read_at: "combs-runtime/src/step.rs",
        measured: None,
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_PROVENANCE",
        default: "startup",
        does: "off, startup, turn or debug: how much of its own history the engine narrates",
        read_at: "combs-core/src/provenance.rs",
        measured: None,
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_PROGRESS",
        default: "off",
        does: "json emits load and pull progress on stderr, the only channel open while weights load",
        read_at: "combs-core/src/progress.rs",
        measured: None,
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_LLM_PREFLIGHT",
        default: "on where a memory probe exists",
        does: "refuse to serve a model that will not fit host memory. Stands down BY NAME where the platform has no probe",
        read_at: "combs-cli/src/fit.rs",
        measured: None,
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_SPEC",
        default: "off",
        does: "speculative decoding; activates only under greedy sampling with penalties off",
        read_at: "combs-runtime/src/step.rs",
        measured: None,
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_LOAD_F16",
        default: "off",
        does: "load dense weights as f16 instead of widening every dtype to f32 at the load choke point",
        read_at: "combs-formats",
        measured: None,
    },
    Door {
        kind: Kind::Knob,
        name: "COMBS_WGSL_F16",
        default: "on where it is blessed (Metal; Vulkan on PCI vendor 0x10de); off on every other adapter; 0/1 wins",
        does: "let the hand-written kernels store activations in f16 instead of casting around an f32 launch; arithmetic stays f32 either way. Attention reads the f16 arena in place, which is where decode doubles at depth",
        read_at: "combs-models/src/wgsl/mod.rs f16_storage_enabled, STORAGE_BLESSED",
        measured: Some("MEASUREMENTS §96, §97, §98, §115, §116d"),
    },

    // ── knobs the table lacked until the drift test named them ──
    Door { kind: Kind::Knob, name: "COMBS_ATTN", default: "fused", does: "manual sends attention back to the hand-rolled path instead of the fused one", read_at: "combs-models/src/kv.rs:27", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_PACKED_EMBED", default: "on", does: "0 dequantizes the embedding table instead of gathering from packed bytes", read_at: "combs-models/src/embed.rs:260", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_PERSISTENT_LOAD", default: "on", does: "0 turns off the persistent load path in the device context", read_at: "combs-core/src/lib.rs:618", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_STAGE_REPACK", default: "device", does: "host stages the weight repack through host memory instead of on the device", read_at: "combs-models/src/qmatmul.rs:1239", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_REPACK_SLAB", default: "engine's own", does: "slab size for the device-side weight repack", read_at: "combs-models/src/qmatmul.rs", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_QMATMUL_STRATEGY", default: "autotune", does: "cube forces the cube matmul strategy instead of letting autotune choose", read_at: "combs-models/src/qlinear.rs:218", measured: Some("MEASUREMENTS §70") },
    Door { kind: Kind::Knob, name: "COMBS_QMATMUL_CUBE_K", default: "engine's own", does: "the k split for the cube matmul path", read_at: "combs-models/src/qlinear.rs", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_QMATMUL_PAD_ROWS", default: "off", does: "pads rows into the batched matmul rather than refusing a ragged shape", read_at: "combs-models/src/qlinear.rs", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_NO_TILED_MATMUL", default: "off", does: "1 disables the tiled quant matmul and takes the untiled cube path", read_at: "combs-models/src/qmatmul.rs:165", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_GPU_SAMPLE", default: "off", does: "1 samples the next token on the device instead of reading logits back", read_at: "combs-runtime/src/step.rs", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_THINK_GUARD", default: "on", does: "0 removes the think guard, for A/B runs against a model that reasons", read_at: "combs-runtime/src/step.rs", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_VRAM_BUDGET_MB", default: "unset", does: "refuse to mount a model whose weights and arena exceed this budget; the refusal shows its arithmetic", read_at: "combs-runtime/src/engine.rs:484", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_FIT_LIMIT", default: "the adapter's own", does: "override the per-binding allocation limit the fit check enforces", read_at: "combs-cli/src/fit.rs", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_LLM_PREFLIGHT_FREE_MB", default: "the probe's reading", does: "pretend this much host memory is free, to exercise the llm gate", read_at: "combs-cli/src/fit.rs", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_LLM_PREFLIGHT_HEADROOM_MB", default: "2048", does: "headroom the llm gate insists on above the working set", read_at: "combs-cli/src/fit.rs", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_IMAGE_PREFLIGHT", default: "on", does: "0 turns off the image working-set refusal", read_at: "combs-cli/src/fit.rs", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_IMAGE_PREFLIGHT_FREE_MB", default: "the probe's reading", does: "pretend this much host memory is free, to exercise the image gate", read_at: "combs-cli/src/fit.rs", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_IMAGE_PREFLIGHT_HEADROOM_MB", default: "2048", does: "headroom the image gate insists on", read_at: "combs-cli/src/fit.rs", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_KV_IDLE_SWEEP_S", default: "off", does: "seconds of idleness after which a session's kv pages are swept", read_at: "combs-cli/src/serve.rs", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_IMAGE_STATS_GPU", default: "off", does: "sample gpu memory in the image server's stats; it is a blocking submit, so it is off by default", read_at: "combs-cli/src/serve_images.rs:302", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_CHAT_DATE", default: "the build's own", does: "pin the date a chat template renders, so a template test is reproducible", read_at: "combs-runtime", measured: None },
    Door { kind: Kind::Knob, name: "COMBS_PROVENANCE_FORMAT", default: "human", does: "json writes provenance as one object per line", read_at: "combs-core/src/provenance.rs", measured: None },

    // ── paths ──
    Door { kind: Kind::Path, name: "COMBS_HOME", default: "the platform's own", does: "where the engine keeps models and run state", read_at: "combs-cli", measured: None },
    Door { kind: Kind::Path, name: "COMBS_BIN", default: "the binary on PATH", does: "which combs binary a harness should drive", read_at: "combs-cli", measured: None },
    Door { kind: Kind::Path, name: "COMBS_ESPEAK_NG", default: "espeak-ng on PATH", does: "the espeak binary the voice road shells to for phonemes", read_at: "combs-cli/src/generate_audio.rs:399", measured: None },
    Door { kind: Kind::Path, name: "COMBS_UI_TEMPLATE", default: "the embedded template", does: "serve a UI template from disk instead of the embedded one", read_at: "combs-cli/src/chew.rs:459", measured: None },

    // ── diagnostics and investigation cells ──
    Door { kind: Kind::Cell, name: "COMBS_DEBUG_PROMPT", default: "off", does: "print the assembled prompt text and its token ids", read_at: "combs-cli/src/main.rs", measured: None },
    Door { kind: Kind::Cell, name: "COMBS_DEBUG_QUANT", default: "off", does: "name every tensor's quant dispatch decision at load", read_at: "combs-models/src/qlinear.rs", measured: None },
    Door { kind: Kind::Cell, name: "COMBS_QMATMUL_DEBUG", default: "off", does: "print the first eight matmul calls with their shapes and dtypes", read_at: "combs-models/src/qlinear.rs", measured: None },
    Door { kind: Kind::Cell, name: "COMBS_K6_CELL", default: "off", does: "unlocks the own-process investigation cells in the matmul tests", read_at: "combs-models/src/qlinear.rs", measured: None },
    Door { kind: Kind::Cell, name: "COMBS_WTILE_BENCH", default: "off", does: "runs the klein-shaped weight-tile bench. Explicitly not a gate", read_at: "combs-models/src/qmatmul.rs", measured: None },
    Door { kind: Kind::Cell, name: "COMBS_SPIKE_N", default: "2000", does: "launches per arm in the replay spike, an own-process ignored test", read_at: "combs-models/src/wgsl/mod.rs", measured: None },
    Door { kind: Kind::Cell, name: "COMBS_SPIKE_REPS", default: "3", does: "repetitions per arm in the replay spike, an own-process ignored test", read_at: "combs-models/src/wgsl/mod.rs", measured: None },

    // ── build stamps: set by the build, read for the manifest ──
    Door { kind: Kind::Stamp, name: "COMBS_BUILD_AT", default: "the build's own", does: "build time, epoch seconds", read_at: "combs-cli/src/build_info.rs", measured: None },
    Door { kind: Kind::Stamp, name: "COMBS_BUILD_AT_RFC3339", default: "the build's own", does: "build time, readable", read_at: "combs-cli/src/build_info.rs", measured: None },
    Door { kind: Kind::Stamp, name: "COMBS_BUILD_BRANCH", default: "the build's own", does: "git branch at build", read_at: "combs-cli/src/build_info.rs", measured: None },
    Door { kind: Kind::Stamp, name: "COMBS_BUILD_COMMIT", default: "the build's own", does: "git commit at build", read_at: "combs-cli/src/build_info.rs", measured: None },
    Door { kind: Kind::Stamp, name: "COMBS_BUILD_COMMIT_SHORT", default: "the build's own", does: "short git commit", read_at: "combs-cli/src/build_info.rs", measured: None },
    Door { kind: Kind::Stamp, name: "COMBS_BUILD_DEBUG", default: "the build's own", does: "whether debug assertions are on", read_at: "combs-cli/src/build_info.rs", measured: None },
    Door { kind: Kind::Stamp, name: "COMBS_BUILD_DIRTY", default: "the build's own", does: "whether the tree was dirty at build", read_at: "combs-cli/src/build_info.rs", measured: None },
    Door { kind: Kind::Stamp, name: "COMBS_BUILD_FEATURES", default: "the build's own", does: "cargo features compiled in", read_at: "combs-cli/src/build_info.rs", measured: None },
    Door { kind: Kind::Stamp, name: "COMBS_BUILD_OPT_LEVEL", default: "the build's own", does: "optimization level", read_at: "combs-cli/src/build_info.rs", measured: None },
    Door { kind: Kind::Stamp, name: "COMBS_BUILD_PROFILE", default: "the build's own", does: "debug or release", read_at: "combs-cli/src/build_info.rs", measured: None },
    Door { kind: Kind::Stamp, name: "COMBS_BUILD_RUSTC", default: "the build's own", does: "the compiler that built it", read_at: "combs-cli/src/build_info.rs", measured: None },
    Door { kind: Kind::Stamp, name: "COMBS_BUILD_TARGET", default: "the build's own", does: "the target triple", read_at: "combs-cli/src/build_info.rs", measured: None },
];

/// Doors the kit knows about but the engine does not read under a
/// `COMBS_` name — cubecl's own knob is the one that matters, and it is
/// ours by measurement even though it is not ours by name.
pub const FOREIGN_DOORS: &[&str] = &["CUBECL_WGPU_MAX_TASKS"];

pub fn print_table() {
    let width = DOORS.iter().map(|d| d.name.len()).max().unwrap_or(0);
    for d in DOORS {
        let mark = match d.measured {
            Some(section) => format!(" [measured: {section}]"),
            None => String::new(),
        };
        println!("{:width$}  {}  — {}{}", d.name, d.default, d.does, mark, width = width);
        println!("{:width$}  read at {}", "", d.read_at, width = width);
    }
    println!();
    println!("{} doors; {} with a measured effect recorded.", DOORS.len(),
             DOORS.iter().filter(|d| d.measured.is_some()).count());
}

pub fn as_json() -> serde_json::Value {
    serde_json::Value::Array(
        DOORS
            .iter()
            .map(|d| {
                serde_json::json!({
                    "name": d.name,
                    "default": d.default,
                    "does": d.does,
                    "read_at": d.read_at,
                    "measured": d.measured,
                })
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn engine_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the kit sits in the engine workspace")
            .to_path_buf()
    }

    fn doors_in_sources() -> BTreeSet<String> {
        let mut found = BTreeSet::new();
        let crates = ["combs-core", "combs-models", "combs-runtime", "combs-cli", "combs-formats"];
        for c in crates {
            let dir = engine_root().join(c).join("src");
            walk(&dir, &mut found);
        }
        found
    }

    fn walk(dir: &std::path::Path, found: &mut BTreeSet<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, found);
            } else if p.extension().is_some_and(|x| x == "rs") {
                let Ok(text) = std::fs::read_to_string(&p) else { continue };
                for (i, _) in text.match_indices("COMBS_") {
                    let rest = &text[i..];
                    let end = rest
                        .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
                        .unwrap_or(rest.len());
                    let name = &rest[..end];
                    // Test-only and probe-only doors are not the
                    // developer's knobs; they are the battery's.
                    if name.len() > 6 && !name.starts_with("COMBS_TEST") {
                        found.insert(name.to_string());
                    }
                }
            }
        }
    }

    /// The list cannot rot: a door added to the engine without a line
    /// here fails this test by name.
    #[test]
    fn every_door_the_engine_reads_is_listed() {
        let listed: BTreeSet<String> = DOORS.iter().map(|d| d.name.to_string()).collect();
        let found = doors_in_sources();
        assert!(!found.is_empty(), "the source walk found nothing — the path is wrong, not the engine");
        let missing: Vec<&String> = found.difference(&listed).collect();
        assert!(
            missing.is_empty(),
            "{} door(s) the engine reads are not in the table: {:?}",
            missing.len(),
            missing
        );
    }

    #[test]
    fn a_measured_door_names_its_section() {
        for d in DOORS {
            if let Some(m) = d.measured {
                assert!(m.contains('§'), "{}: a measured door must name its ledger section", d.name);
            }
            assert!(!d.does.is_empty(), "{}: every door says what it does", d.name);
            assert!(!d.read_at.is_empty(), "{}: every door says where it is read", d.name);
        }
    }
}
