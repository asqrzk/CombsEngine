//! The Combs Kernel: the engine's owned GPU path.
//!
//! This module is the hand-written-WGSL half of the suite — launched on
//! the runtime's own stream through the template seam below; the
//! `#[cube]` quant family in `qmatmul.rs` is the other half. Together
//! they carry the whole decode hot path: packed embed gather, rms_norm,
//! rope, decode attention (paged / sliding / int8), and every quant
//! projection. Hand WGSL exists for the places where owning the exact
//! text matters — reductions and attention whose selection must be
//! fixed by shape, not autotuned, and whose source must be auditable
//! against the two validators it will meet (naga natively, Tint in a
//! browser).
//!
//! ## The contract every kernel here honors
//!
//! - Entry point named `main`; buffers bound at `@group(0) @binding(i)`
//!   in the exact order pushed into [`KernelArguments`].
//! - Launch scalars travel as raw u64 slots in a uniform buffer appended
//!   as the LAST binding. In WGSL that is a struct of `vec2<u32>` fields,
//!   low word in `.x`; f32 scalars cross as bits (`to_bits` host-side,
//!   `bitcast<f32>` in the shader). This layout is not assumed — the
//!   scalar-echo probe pins it by test on every backend before any other
//!   kernel's door opens.
//! - The runtime injects no bounds checks into template kernels: every
//!   access self-guards. No `enable` directives, ever — a browser rejects
//!   the shader and then dispatches the invalid pipeline as a silent
//!   no-op, which is how a model decodes zeros with no error. No
//!   subgroup operations for the same reason. No infinity literals (WGSL
//!   forbids them); large finite sentinels instead.
//! - One-dimensional workgroups of [`WORKGROUP`] threads; barriers only
//!   in uniform control flow — guards mask values, they never skip a
//!   barrier.
//! - **Every storage buffer is declared `read_write`**, including pure
//!   inputs. The pool sub-slices allocations, so two bindings routinely
//!   share one `wgpu::Buffer` — and wgpu rejects a dispatch using one
//!   buffer as both read-only and read-write. The shared-memory probe
//!   found this on day one; uniform declarations retire the whole class.
//! - f32 compute throughout; dtype casts happen outside the launch.
//!
//! Kernels are accelerators, never load-bearing: every integration point
//! offers the kernel through a `try_*` door that returns `None` into the
//! existing burn path, and every door reads its environment once per
//! process.

use burn_cubecl::template::{KernelSource, SourceKernel, SourceTemplate};

/// Which element type a kernel's STORAGE uses. Its arithmetic is f32
/// either way — that is not a tuning choice but a measured law (§66:
/// f16 operands with an f32 accumulator match the host to 1.5e-6, while
/// accumulating IN f16 diverges 3.4e-3 and overflows at a sum of
/// 14,467). What this changes is what crosses the bus and what sits in
/// the arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Storage {
    F32,
    F16,
}

impl Storage {
    /// Line one of the shader. WGSL requires `enable` directives before
    /// anything else, and a browser REJECTS them — then dispatches the
    /// invalid pipeline as a silent no-op, which is how a model decodes
    /// zeros with no error. The door below is what keeps this empty in
    /// a browser build.
    fn enable_line(self) -> &'static str {
        match self {
            Storage::F32 => "",
            Storage::F16 => "enable f16;\n",
        }
    }

    fn act(self) -> &'static str {
        match self {
            Storage::F32 => "f32",
            Storage::F16 => "f16",
        }
    }

    fn act4(self) -> &'static str {
        match self {
            Storage::F32 => "vec4<f32>",
            Storage::F16 => "vec4<f16>",
        }
    }
}

/// Where the f16 storage road is blessed, as data: a graphics backend and,
/// on it, the PCI vendor whose adapters were measured (`None` = every
/// adapter on that backend), with the evidence. A default is a promise to
/// every box of that kind, so a row enters only with perplexity identical
/// and greedy transcripts byte-identical against f32.
pub(crate) const STORAGE_BLESSED: &[(&str, Option<u32>, &str)] = &[
    // perplexity identical to four decimals, five of five transcripts
    // byte-identical; decode doubles at depth
    ("Metal", None, "§97, §98"),
    // perplexity identical with the door shut and open and within 0.09% of
    // f32, greedy text byte-identical to f32; decode 2.0-3.9x the door shut
    ("Vulkan", Some(0x10de), "§115, §116d"),
];

/// The storage door's default, as a pure decision so the walk is the
/// test.
///
/// `COMBS_WGSL_F16=0` shuts it and `=1` opens it anywhere the adapter
/// admits the type. Unset, it is OPEN on an adapter whose backend and
/// vendor [`STORAGE_BLESSED`] names and SHUT on every other: a shader
/// `enable` directive is exactly the kind of thing a second driver may
/// handle differently, and a default is a promise to every box at once.
pub(crate) fn storage_door_default(backend: &str, vendor_id: u32, env: Option<&str>) -> bool {
    match env {
        Some("0") => false,
        Some("1") => true,
        _ => STORAGE_BLESSED
            .iter()
            .any(|(b, v, _)| backend.eq_ignore_ascii_case(b) && v.map_or(true, |v| v == vendor_id)),
    }
}

/// How a gemv workgroup is cut: `rows` output rows per workgroup,
/// `lanes` lanes cooperating on each. Their product is the workgroup
/// size, so they trade against each other — more rows means fewer
/// workgroups and more work per lane, more lanes the reverse. That is
/// exactly the question §102 left open, which is why this is a
/// parameter and not a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Geometry {
    pub rows: u32,
    pub lanes: u32,
}

impl Geometry {
    /// The shape every measurement in this ledger was taken at, and the
    /// production default until a sweep beats it by a margin worth the
    /// change.
    pub const DEFAULT: Geometry = Geometry { rows: 16, lanes: 16 };

    /// The cuts of a 256-thread workgroup worth asking about. Fewer
    /// than 8 lanes leaves the block-strided read too scattered; more
    /// than 32 leaves fewer than four rows a group.
    pub const SWEEP: &'static [Geometry] = &[
        Geometry { rows: 8, lanes: 32 },
        Geometry { rows: 16, lanes: 16 },
        Geometry { rows: 32, lanes: 8 },
    ];

    pub fn workgroup(&self) -> u32 {
        self.rows * self.lanes
    }
}

/// The cut `COMBS_WGSL_GEMV_GEOMETRY` names, or the default when it is unset.
///
/// Only the swept cuts are accepted, spelled as the doors report prints
/// them (`8x32`, `16x16`, `32x8`), and a refusal names every cut it would
/// have taken: a mistyped value must not quietly run a cut nobody measured.
pub(crate) fn geometry_door(env: Option<&str>) -> Result<Geometry, String> {
    let Some(asked) = env else {
        return Ok(Geometry::DEFAULT);
    };
    let label = |g: &Geometry| format!("{}x{}", g.rows, g.lanes);
    match Geometry::SWEEP.iter().find(|g| label(*g) == asked) {
        Some(g) => Ok(*g),
        None => {
            let cuts: Vec<String> = Geometry::SWEEP.iter().map(label).collect();
            Err(format!(
                "COMBS_WGSL_GEMV_GEOMETRY={asked:?} is not a swept cut ({})",
                cuts.join(", ")
            ))
        }
    }
}

/// The workgroup cut every production decode gemv launches with.
///
/// The best cut belongs to the adapter, not to the engine: a sweep that
/// finds no cut ahead of 16x16 on one GPU can find another cut well ahead
/// on a different one. So the default stays 16x16 and
/// `COMBS_WGSL_GEMV_GEOMETRY` chooses, read once per process. A value
/// outside the sweep is refused aloud and the default is used; the doors
/// report names the cut actually taken either way.
pub fn gemv_geometry() -> Geometry {
    static CUT: std::sync::OnceLock<Geometry> = std::sync::OnceLock::new();
    *CUT.get_or_init(|| {
        match geometry_door(std::env::var("COMBS_WGSL_GEMV_GEOMETRY").ok().as_deref()) {
            Ok(g) => g,
            Err(why) => {
                eprintln!(
                    "[wgsl] {why}; the decode gemv takes {}x{}",
                    Geometry::DEFAULT.rows,
                    Geometry::DEFAULT.lanes
                );
                Geometry::DEFAULT
            }
        }
    })
}

/// May this box store kernel activations in f16?
///
/// Four conditions, and each is load-bearing. Never in a browser (the
/// `enable` above). Never when the master WGSL door is shut. Open by
/// default only where [`STORAGE_BLESSED`] names the adapter (`COMBS_WGSL_F16`
/// wins both ways). And only when the
/// ADAPTER admits the type — asked through the same registration the
/// compiler itself consults, rather than a feature string we parse
/// ourselves. cubecl requests every adapter feature when it creates the
/// device, so a type the properties admit is a type the shader may
/// `enable`.
pub(crate) fn f16_storage_enabled(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
) -> bool {
    if cfg!(target_family = "wasm") || !wgsl_enabled() {
        return false;
    }
    static ASKED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ASKED.get_or_init(|| {
        let adapter = combs_core::adapter_caps();
        storage_door_default(&adapter.backend, adapter.vendor_id, std::env::var("COMBS_WGSL_F16").ok().as_deref())
    }) {
        return false;
    }
    client
        .properties()
        .supports_type(cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16))
}

/// The adapter half of [`f16_storage_enabled`], without the opt-in env
/// door — so a test can ask "can this box do it at all?" and skip by
/// name, rather than silently pass because the door happened to be shut.
#[cfg(test)]
pub(crate) fn f16_storage_enabled_for_test(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
) -> bool {
    !cfg!(target_family = "wasm")
        && client
            .properties()
            .supports_type(cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16))
}

/// What is ACTUALLY live, asked of the code that reads each door.
///
/// A measurement that does not carry its own configuration is a number
/// with no question attached, and this estate has already published one
/// taken on the wrong backend because nothing on screen said which.
/// Every door here is read through the SAME function the kernels read,
/// so the report cannot drift from the behaviour: if a door is reported
/// on, a kernel took it.
///
/// `f16_storage` needs the device (it asks the adapter), so it is the
/// one door that must be asked with a client in hand.
pub fn active_doors(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
) -> Vec<(&'static str, String)> {
    vec![
        ("wgsl", wgsl_enabled().to_string()),
        ("gemv", gemv_enabled().to_string()),
        ("rmsnorm", rmsnorm::rmsnorm_enabled().to_string()),
        ("rope", rope::rope_enabled().to_string()),
        ("attn", decode_attn::attn_enabled().to_string()),
        // The DOOR only. Whether it is ever WALKED THROUGH depends on the
        // dtype of the tensors that arrive, which this crate cannot see:
        // it has no f16 feature of its own, so asking `cfg!` here
        // answered about the wrong thing and printed "f32 build" under an
        // f16 one. The caller knows its own dtype and composes the
        // sentence; this reports the fact it actually holds.
        ("f16_storage_door", f16_storage_enabled(client).to_string()),
        ("gemv_geometry", {
            let cut = gemv_geometry();
            format!("{}x{}", cut.rows, cut.lanes)
        }),
    ]
}

/// The seam's dtype bridge.
///
/// Every kernel here computes and stores f32 — the contract line above
/// ("f32 compute throughout; dtype casts happen outside the launch"),
/// which is what lets one text pass both validators and lets a browser
/// build exist at all. A backend whose activations are f16 therefore
/// casts on the way in and back on the way out: two or three small
/// elementwise launches, in exchange for a whole multi-op Burn chain.
/// On a matching dtype both are no-ops, so the f32 path is untouched.
pub(crate) fn as_f32<B: burn::tensor::backend::Backend, const D: usize>(
    x: burn::tensor::Tensor<B, D>,
) -> burn::tensor::Tensor<B, D> {
    match x.dtype() {
        burn::tensor::DType::F32 => x,
        _ => x.cast(burn::tensor::FloatDType::F32),
    }
}

/// The return leg of [`as_f32`]: put a kernel's f32 output back into the
/// dtype the caller handed in, so the tensor rejoins the model's stream
/// as the type every other op there expects. Mixing them is not merely
/// slow — an f16 tensor against an f32 broadcast silently returns
/// power-of-two-wrong values on this backend.
pub(crate) fn from_f32<B: burn::tensor::backend::Backend, const D: usize>(
    out: burn::tensor::Tensor<B, D>,
    dtype: burn::tensor::DType,
) -> burn::tensor::Tensor<B, D> {
    match dtype {
        burn::tensor::DType::F16 => out.cast(burn::tensor::FloatDType::F16),
        burn::tensor::DType::BF16 => out.cast(burn::tensor::FloatDType::BF16),
        _ => out,
    }
}


use cubecl::prelude::{CubeCount, CubeDim, KernelId, Runtime};
use cubecl::server::{Binding, KernelArguments, MetadataBindingInfo};

use burn::backend::wgpu::WgpuRuntime;

/// Threads per workgroup for every kernel in this module. Must match the
/// literal `@workgroup_size(256)` in each `.wgsl` file — the two cannot
/// drift, because [`launch`] passes this value as the task's `CubeDim`
/// and the runtime validates launches against it.
pub(crate) const WORKGROUP: u32 = 256;

/// Master door for the WGSL kernel suite. `COMBS_WGSL=0` restores the
/// burn path everywhere; per-kernel doors layer on top. Read once — a
/// door that flips mid-process would let autotuned and fixed paths
/// interleave within one generation, which no test covers.
pub(crate) fn wgsl_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("COMBS_WGSL").as_deref(), Ok("0")))
}

/// Declares a [`KernelSource`] over an included `.wgsl` file.
macro_rules! wgsl_kernel {
    ($name:ident, $file:expr) => {
        pub(crate) struct $name;

        impl KernelSource for $name {
            fn source(&self) -> SourceTemplate {
                SourceTemplate::new(include_str!($file))
            }

            fn id(&self) -> KernelId {
                KernelId::new::<Self>().cube_dim(CubeDim::new_1d(WORKGROUP))
            }
        }
    };
}

/// A kernel whose source is rendered for one STORAGE type.
///
/// Substitution, not a second file: the audited body — tiling, the
/// reduction's barrier discipline, the guards — must not fork, or the
/// two copies drift and only one of them stays reviewed. The f32
/// rendering must come out byte-identical to the pre-change file, and a
/// unit below holds that line.
/// A kernel rendered for one storage type AND one workgroup cut. Same
/// substitution discipline as `wgsl_kernel_dual!`: one audited body,
/// the parameters in the id so renderings cannot share a cache entry.
macro_rules! wgsl_kernel_gemv {
    ($name:ident, $file:expr) => {
        pub(crate) struct $name(pub(crate) Storage, pub(crate) Geometry);

        impl KernelSource for $name {
            fn source(&self) -> SourceTemplate {
                SourceTemplate::new(include_str!($file))
                    .register("f16_enable", self.0.enable_line())
                    .register("act", self.0.act())
                    .register("act4", self.0.act4())
                    .register("rows", self.1.rows.to_string())
                    .register("lanes", self.1.lanes.to_string())
            }

            fn id(&self) -> KernelId {
                KernelId::new::<Self>()
                    .cube_dim(CubeDim::new_1d(self.1.workgroup()))
                    .info((self.0, self.1))
            }
        }
    };
}

macro_rules! wgsl_kernel_dual {
    ($name:ident, $file:expr) => {
        pub(crate) struct $name(pub(crate) Storage);

        impl KernelSource for $name {
            fn source(&self) -> SourceTemplate {
                SourceTemplate::new(include_str!($file))
                    .register("f16_enable", self.0.enable_line())
                    .register("act", self.0.act())
                    .register("act4", self.0.act4())
            }

            fn id(&self) -> KernelId {
                // The storage rides the id, or the two renderings would
                // collide in the pipeline cache and whichever compiled
                // first would serve both.
                KernelId::new::<Self>()
                    .cube_dim(CubeDim::new_1d(WORKGROUP))
                    .info(self.0)
            }
        }
    };
}

wgsl_kernel!(ProbeEcho, "probe_echo.wgsl");
wgsl_kernel!(ProbeSmem, "probe_smem.wgsl");
wgsl_kernel_dual!(RmsNorm, "rmsnorm.wgsl");
wgsl_kernel_dual!(DecodeAttn, "decode_attn.wgsl");
wgsl_kernel!(DecodeAttnQ8, "decode_attn_q8.wgsl");
wgsl_kernel_dual!(DecodeAttnSplit, "decode_attn_split.wgsl");
wgsl_kernel_dual!(DecodeAttnCombine, "decode_attn_combine.wgsl");
wgsl_kernel!(PrefillAttn, "prefill_attn.wgsl");
wgsl_kernel!(RopeQk, "rope.wgsl");
wgsl_kernel!(DeltanetConvStep, "deltanet_conv_step.wgsl");
wgsl_kernel!(DeltanetStateStep, "deltanet_state_step.wgsl");
wgsl_kernel_gemv!(DecodeGemvQ8, "decode_gemv_q8.wgsl");
wgsl_kernel_gemv!(DecodeGemvQ4K, "decode_gemv_q4k.wgsl");
wgsl_kernel_gemv!(DecodeGemvQ5K, "decode_gemv_q5k.wgsl");
wgsl_kernel_gemv!(DecodeGemvQ6K, "decode_gemv_q6k.wgsl");
wgsl_kernel!(ScrubFill, "scrub_fill.wgsl");

/// Door for the split-K decode gemv (`COMBS_WGSL_GEMV=0` restores the
/// cube untiled kernel); read once, like every door here.
pub(crate) fn gemv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        wgsl_enabled() && !matches!(std::env::var("COMBS_WGSL_GEMV").as_deref(), Ok("0"))
    })
}

mod decode_attn;
mod deltanet;
mod decode_attn_q8;
mod rmsnorm;
mod rope;
pub(crate) mod scrub;
#[cfg(test)]
mod scrub_spike;

pub(crate) use decode_attn::{try_decode_attention, try_prefill_attention, try_sliding_decode_attention};
pub(crate) use decode_attn_q8::{QuantArena, try_decode_attention_q8};
pub(crate) use deltanet::{StepGeometry, StepInputs, deltanet_step_available, try_deltanet_step};
pub(crate) use rmsnorm::try_rms_norm;
pub(crate) use rope::try_rope_qk;

/// Which split-K decode gemv kernel to launch — the qmatmul family
/// names its format, this seam owns the kernel objects.
pub(crate) enum GemvKernel {
    /// Q8_0 packed words + per-block scales.
    Q8,
    /// Q4_K superblocks (qs, [d, dmin] pairs, packed 6-bit scales).
    Q4K,
    /// Q5_K superblocks: Q4_K's plus the high-bit words (qh).
    Q5K,
    /// Q6_K superblocks (ql, qh, i8 scales, d).
    Q6K,
}

/// [`launch`] for the decode gemv, callable from the qmatmul family
/// (which is hardwired to WgpuRuntime, same as this seam).
pub(crate) fn launch_gemv(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    kernel: GemvKernel,
    storage: Storage,
    geom: Geometry,
    workgroups: u32,
    buffers: Vec<Binding>,
    scalars: Vec<u64>,
) {
    let count = CubeCount::Static(workgroups, 1, 1);
    match kernel {
        GemvKernel::Q8 => launch_at(client, DecodeGemvQ8(storage, geom), geom.workgroup(), count, buffers, scalars),
        GemvKernel::Q4K => launch_at(client, DecodeGemvQ4K(storage, geom), geom.workgroup(), count, buffers, scalars),
        GemvKernel::Q5K => launch_at(client, DecodeGemvQ5K(storage, geom), geom.workgroup(), count, buffers, scalars),
        GemvKernel::Q6K => launch_at(client, DecodeGemvQ6K(storage, geom), geom.workgroup(), count, buffers, scalars),
    }
}

/// Launches one WGSL kernel on the runtime's stream.
///
/// The task joins the same encoder and the same submit as the surrounding
/// burn operations — ordering with neighbouring ops needs no flush and no
/// fence. Checked execution mode is used deliberately: template kernels
/// receive no injected guards either way, and Checked avoids the
/// unchecked entry point's extra obligations.
pub(crate) fn launch(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    kernel: impl KernelSource,
    count: CubeCount,
    buffers: Vec<Binding>,
    scalars: Vec<u64>,
) {
    launch_at(client, kernel, WORKGROUP, count, buffers, scalars);
}

/// `launch` for a kernel whose workgroup size is not the house default.
/// The size must match the `@workgroup_size` the rendering carries, or
/// the dispatch is a silent mismatch.
pub(crate) fn launch_at(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    kernel: impl KernelSource,
    workgroup: u32,
    count: CubeCount,
    buffers: Vec<Binding>,
    scalars: Vec<u64>,
) {
    let task: <<WgpuRuntime as Runtime>::Server as cubecl::server::ComputeServer>::Kernel =
        Box::new(SourceKernel::new(kernel, CubeDim::new_1d(workgroup)));
    client.launch(
        task,
        count,
        KernelArguments::new()
            .with_buffers(buffers)
            .with_info(MetadataBindingInfo::custom(scalars)),
    );
}

/// The floor cost of asking this driver to do anything: `n` launches
/// of the scalar-echo kernel — one workgroup, five words written — then
/// one sync. Returns the wall time and the dispatch-counter delta, so
/// the caller can prove the instrument counted exactly what it timed
/// (delta must equal `n`; anything else means another launch landed in
/// the window). Every µs/dispatch this ledger has ever quoted was
/// arithmetic over a whole decode step; this reads the launch alone.
pub fn dispatch_latency(n: usize) -> (std::time::Duration, u64) {
    use cubecl::prelude::*;
    let device = Default::default();
    let client = WgpuRuntime::client(&device);
    let out = client.empty(5 * core::mem::size_of::<u32>());
    let scalars = vec![1u64, 0xAABB_CCDD_0011_2233, 7, f32::to_bits(1.5) as u64];
    // Warm: the first launch compiles the pipeline, which is not a
    // launch cost.
    launch(&client, ProbeEcho, CubeCount::Static(1, 1, 1), vec![out.clone().binding()], scalars.clone());
    cubecl::future::block_on(client.sync()).expect("sync after warm-up");
    let (d0, _) = combs_core::gpu_counters();
    let started = std::time::Instant::now();
    for _ in 0..n {
        launch(&client, ProbeEcho, CubeCount::Static(1, 1, 1), vec![out.clone().binding()], scalars.clone());
    }
    cubecl::future::block_on(client.sync()).expect("sync after the launches");
    let took = started.elapsed();
    let (d1, _) = combs_core::gpu_counters();
    (took, d1 - d0)
}

/// Host→device and device→host at one size, timed to completion.
/// Returns (h2d, d2h). Device→device is not offered: the runtime has no
/// plain copy kernel, and timing a tensor op as if it were one would
/// report the op. Said here rather than approximated.
pub fn bus_cell(bytes: usize) -> (std::time::Duration, std::time::Duration) {
    use cubecl::prelude::*;
    let device = Default::default();
    let client = WgpuRuntime::client(&device);
    let payload: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
    // Warm both directions once.
    let h = client.create_from_slice(&payload);
    let _ = cubecl::future::block_on(client.read_async(vec![h]));
    let started = std::time::Instant::now();
    let h = client.create_from_slice(&payload);
    cubecl::future::block_on(client.sync()).expect("sync after upload");
    let h2d = started.elapsed();
    let started = std::time::Instant::now();
    let back = cubecl::future::block_on(client.read_async(vec![h])).expect("readback");
    let d2h = started.elapsed();
    assert_eq!(back[0].len(), bytes, "the bus returned a different size");
    (h2d, d2h)
}

/// The Q4_K decode gemv at one shape and one workgroup cut, `reps`
/// times, timed to completion — and CHECKED once against the host
/// before it is timed, because a cut that computes the wrong answer
/// quickly is not a faster kernel. Returns (per-rep time, max relative
/// error against the host dequant reference).
pub fn gemv_cell_at(n_out: usize, k: usize, reps: usize, rows: u32, lanes: u32) -> (std::time::Duration, f32) {
    use cubecl::prelude::*;
    let geom = Geometry { rows, lanes };
    let device = Default::default();
    let client = WgpuRuntime::client(&device);
    let blocks = n_out * k / 256;
    let mut data: Vec<u8> = (0..blocks * 144).map(|i| ((i * 37 + 11) % 251) as u8).collect();
    for b in 0..blocks {
        data[b * 144..b * 144 + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
        data[b * 144 + 2..b * 144 + 4].copy_from_slice(&0x3400u16.to_le_bytes());
    }
    let w = crate::qmatmul::Q4KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k)
        .expect("a synthetic q4_k weight packs");
    let x: Vec<f32> = (0..k).map(|i| (((i as f32 * 0.311).sin() * 1.4) - 0.1) / 32.0).collect();
    let x_h = client.create_from_slice(f32::as_bytes(&x));

    // Correctness at this cut, once, before any timing.
    let out = w.decode_gemv_wgsl_at(&client, x_h.clone(), Storage::F32, None, geom);
    let bytes = cubecl::future::block_on(client.read_async(vec![out])).expect("readback");
    let got = f32::from_bytes(&bytes[0]);
    let dense = combs_formats::quants::dequantize_q4_k(&data, n_out * k).expect("reference");
    let mut worst = 0.0f32;
    for r in 0..n_out {
        let expect: f32 = (0..k).map(|c| x[c] * dense[r * k + c]).sum();
        let rel = (got[r] - expect).abs() / expect.abs().max(1.0);
        if rel > worst {
            worst = rel;
        }
    }

    let _ = w.decode_gemv_wgsl_at(&client, x_h.clone(), Storage::F32, None, geom);
    cubecl::future::block_on(client.sync()).expect("sync after warm-up");
    let started = std::time::Instant::now();
    for _ in 0..reps {
        let _ = w.decode_gemv_wgsl_at(&client, x_h.clone(), Storage::F32, None, geom);
    }
    cubecl::future::block_on(client.sync()).expect("sync after the reps");
    (started.elapsed() / reps as u32, worst)
}

/// The Q4_K decode gemv at a real projection shape, `reps` times, timed
/// to completion — the kernel the packed seats spend most of a token in.
/// Returns the wall time per rep; the caller turns it into bytes read
/// per second against the packed size, which is the roofline question.
pub fn gemv_cell(n_out: usize, k: usize, reps: usize) -> std::time::Duration {
    gemv_cell_at(n_out, k, reps, Geometry::DEFAULT.rows, Geometry::DEFAULT.lanes).0
}

#[allow(dead_code)]
fn gemv_cell_unused(n_out: usize, k: usize, reps: usize) -> std::time::Duration {
    use cubecl::prelude::*;
    let device = Default::default();
    let client = WgpuRuntime::client(&device);
    let blocks = n_out * k / 256;
    let mut data: Vec<u8> = (0..blocks * 144).map(|i| ((i * 37 + 11) % 251) as u8).collect();
    for b in 0..blocks {
        data[b * 144..b * 144 + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
        data[b * 144 + 2..b * 144 + 4].copy_from_slice(&0x3400u16.to_le_bytes());
    }
    let w = crate::qmatmul::Q4KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k)
        .expect("a synthetic q4_k weight packs");
    let x: Vec<f32> = (0..k).map(|i| (((i as f32 * 0.311).sin() * 1.4) - 0.1) / 32.0).collect();
    let x_h = client.create_from_slice(f32::as_bytes(&x));
    let _ = w.decode_gemv_wgsl(&client, x_h.clone(), Storage::F32, None);
    cubecl::future::block_on(client.sync()).expect("sync after warm-up");
    let started = std::time::Instant::now();
    for _ in 0..reps {
        let _ = w.decode_gemv_wgsl(&client, x_h.clone(), Storage::F32, None);
    }
    cubecl::future::block_on(client.sync()).expect("sync after the reps");
    started.elapsed() / reps as u32
}

/// Runs both probes against the current device and reports the first
/// contract violation, if any.
///
/// Async because a browser can only read buffers back asynchronously;
/// the sync tests below remain the native gate. The wasm surface exposes
/// this so the SAME checks that naga passed can be demanded of Tint in
/// the actual browser before any WGSL door opens there — a shader that
/// fails validation in a browser dispatches as a silent no-op, so the
/// probe checks values, not absence of errors.
pub async fn probe_report() -> Result<(), String> {
    use cubecl::prelude::*;

    let device = Default::default();
    let client = WgpuRuntime::client(&device);

    // --- scalar layout ----------------------------------------------------
    let out = client.empty(5 * core::mem::size_of::<u32>());
    launch(
        &client,
        ProbeEcho,
        CubeCount::Static(1, 1, 1),
        vec![out.clone().binding()],
        vec![1, 0xAABB_CCDD_0011_2233, 7, f32::to_bits(1.5) as u64],
    );
    let bytes = client
        .read_async(vec![out])
        .await
        .map_err(|e| format!("echo readback failed: {e:?}"))?;
    let words = u32::from_bytes(&bytes[0]).to_vec();
    let expect = [1u32, 0, 0x0011_2233, 7, f32::to_bits(3.0)];
    if words[..5] != expect {
        return Err(format!(
            "scalar layout mismatch: got {words:?}, expected {expect:?} — \
             the uniform u64-slot contract does not hold on this backend"
        ));
    }

    // --- shared memory + barriers -----------------------------------------
    let n = 300usize;
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let in_h = client.create_from_slice(f32::as_bytes(&input));
    let out_h = client.empty(n * core::mem::size_of::<f32>());
    launch(
        &client,
        ProbeSmem,
        CubeCount::Static(2, 1, 1),
        vec![in_h.binding(), out_h.clone().binding()],
        vec![n as u64],
    );
    let bytes = client
        .read_async(vec![out_h])
        .await
        .map_err(|e| format!("smem readback failed: {e:?}"))?;
    let got = f32::from_bytes(&bytes[0]);
    for i in 0..n {
        let base = (i / 256) * 256;
        let mirrored = base + (255 - i % 256);
        let staged = if mirrored < n { mirrored as f32 } else { 0.0 };
        if got[i] != staged + 1.0 {
            return Err(format!(
                "smem mirror wrong at {i}: got {}, expected {} — workgroup \
                 memory or barriers are broken on this backend",
                got[i],
                staged + 1.0
            ));
        }
    }

    // --- rmsnorm canary (K1) ----------------------------------------------
    // Two ragged rows through the production kernel, gemma flavor to
    // exercise every scalar slot, checked against a host reduction.
    let (rows, n) = (2usize, 300usize);
    let xs: Vec<f32> = (0..rows * n).map(|i| (i as f32 * 0.37).sin() * 2.0).collect();
    let ws: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32 * 0.11).cos()).collect();
    let (eps, flavor) = (1e-6f32, 1.0f32);
    let x_h = client.create_from_slice(f32::as_bytes(&xs));
    let w_h = client.create_from_slice(f32::as_bytes(&ws));
    let y_h = client.empty(rows * n * core::mem::size_of::<f32>());
    launch(
        &client,
        RmsNorm(Storage::F32),
        CubeCount::Static(rows as u32, 1, 1),
        vec![x_h.binding(), w_h.binding(), y_h.clone().binding()],
        vec![
            rows as u64,
            n as u64,
            f32::to_bits(eps) as u64,
            f32::to_bits(flavor) as u64,
        ],
    );
    let bytes = client
        .read_async(vec![y_h])
        .await
        .map_err(|e| format!("rmsnorm readback failed: {e:?}"))?;
    let got = f32::from_bytes(&bytes[0]);
    for r in 0..rows {
        let row = &xs[r * n..(r + 1) * n];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for c in 0..n {
            let expect = row[c] * inv_rms * (ws[c] + flavor);
            let g = got[r * n + c];
            if (g - expect).abs() > 1e-4 * expect.abs().max(1.0) {
                return Err(format!(
                    "rmsnorm canary wrong at [{r},{c}]: got {g}, expected \
                     {expect} — the K1 kernel must stay doored off here"
                ));
            }
        }
    }

    // --- decode-attention canaries (K3a paged, K3b contiguous) ------------
    // Paged fixtures with a shuffled table, GQA 2:1, ragged totals,
    // checked against a host softmax — the same reference the harmony
    // matrix uses natively. d = 256 is gemma3's head dim and exercises
    // the full-workgroup column path; the contiguous case adds mode 1
    // with a window.
    // The totals straddle the 256-key tile boundary on purpose: the
    // multi-tile online-softmax correction is exactly the code a
    // single-tile canary would leave unproven, and gemma3's second turn
    // (context > 256) is where an unproven tile shows up as garbage.
    for (mode, d, window, n_kv, total) in [
        (0usize, 64usize, 0usize, 2usize, 21usize),
        (0, 256, 0, 2, 21),
        (1, 256, 5, 2, 21),
        (0, 256, 0, 2, 500),
        (1, 256, 512, 1, 500),
        (0, 128, 0, 1, 300),
    ] {
        decode_attn_canary(&client, mode, d, window, n_kv, total).await?;
    }

    // --- split-KV decode-attention canary (K8) ----------------------------
    // The two-pass deep path (segment partials + combine), value-checked
    // at a multi-segment total with a ragged last segment and d = 256.
    split_attn_canary(&client, 256, 1300).await?;
    split_attn_canary(&client, 128, 700).await?;

    // --- quantized decode-attention canaries (K3c) ------------------------
    // Grid-exact packed lanes (the i32::MAX corner word included) through
    // the q8 kernel's bitcast/shift unpack — constructs no other canary
    // exercises, value-checked against a host softmax at gemma3's d too.
    // Two checks per head dim: total=1 makes the softmax weight exactly
    // 1.0, so the output must be the dequantized V row to the last bit —
    // an unpack bug (sign extension, lane order, scale indexing) cannot
    // hide, and accumulation order cannot interfere. The long case then
    // bounds cross-compiler accumulation noise against an f64 reference
    // (Tint and naga contract fma differently; the quant grid's dynamic
    // range makes 1e-3-relative too tight for a 300-key online softmax).
    for d in [64usize, 256] {
        decode_attn_q8_canary(&client, d, 1).await?;
        decode_attn_q8_canary(&client, d, 300).await?;
    }

    // --- tiled quant-matmul canaries --------------------------------------
    // Prefill (m > 1) takes the shared-memory TILED kernels while decode
    // takes the untiled gemv — a browser model can therefore decode
    // perfectly while its prefill lies. Natively tiled and untiled are
    // bit-identical; hold Tint to the same bar, per format, and check
    // untiled against the CPU dequant reference so a both-wrong tie
    // cannot slip through.
    for format in ["q8_0", "q4_k", "q6_k"] {
        tiled_matmul_canary(&client, format).await?;
    }

    // --- split-K decode gemv canaries -------------------------------------
    // The same fixtures at m = 1 through the WGSL gemv vs the cube
    // untiled kernel: the decode path's dominant kernels, value-checked
    // per format on this compiler before the door means anything here.
    for format in ["q8_0", "q4_k", "q5_k", "q6_k"] {
        gemv_canary(&client, format).await?;
    }

    // --- compiled-tanh canary ---------------------------------------------
    // Goes through cubecl's WGSL compiler (not this module's hand-written
    // kernels) on purpose: it proves the safe-tanh workaround is active
    // on THIS target. Metal returns NaN for tanh past 43.0, and a build
    // that misses the clamp turns every GeluTanh model into token-0 soup.
    let xs = [-88.0f32, -50.0, -10.0, 0.5, 10.0, 43.5, 50.0, 88.0];
    let out_h = crate::qmatmul::tanh_canary_device(&client, &xs);
    let bytes = client
        .read_async(vec![out_h])
        .await
        .map_err(|e| format!("tanh readback failed: {e:?}"))?;
    let got = f32::from_bytes(&bytes[0]);
    for (i, (&x, &g)) in xs.iter().zip(got.iter()).enumerate() {
        let expect = x.tanh();
        if !g.is_finite() || (g - expect).abs() > 1e-6 {
            return Err(format!(
                "tanh canary wrong at [{i}] (x = {x}): got {g}, expected                  {expect} — the safe-tanh workaround is not active on this                  target and GeluTanh models will decode garbage"
            ));
        }
    }

    // --- rope canary (K2) -------------------------------------------------
    // Two heads of q, one of k, a mid-table position — both outputs
    // checked against the host rotate_half formula.
    let (n_q, n_kv, seq, d, pos, max_position) = (2usize, 1usize, 3usize, 64usize, 40usize, 64usize);
    let half = d / 2;
    let table = |t: usize, salt: f64| ((t as f64) * 0.031 + salt).sin() as f32;
    let cos_v: Vec<f32> = (0..max_position * d).map(|t| table(t, 0.0)).collect();
    let sin_v: Vec<f32> = (0..max_position * d).map(|t| table(t, 2.0)).collect();
    let xq: Vec<f32> = (0..n_q * seq * d).map(|i| table(i, 5.0)).collect();
    let xk: Vec<f32> = (0..n_kv * seq * d).map(|i| table(i, 7.0)).collect();
    let q_h = client.create_from_slice(f32::as_bytes(&xq));
    let k_h = client.create_from_slice(f32::as_bytes(&xk));
    let c_h = client.create_from_slice(f32::as_bytes(&cos_v));
    let s_h = client.create_from_slice(f32::as_bytes(&sin_v));
    let oq_h = client.empty(xq.len() * core::mem::size_of::<f32>());
    let ok_h = client.empty(xk.len() * core::mem::size_of::<f32>());
    let elems = (n_q + n_kv) * seq * d;
    launch(
        &client,
        RopeQk,
        CubeCount::Static(elems.div_ceil(WORKGROUP as usize) as u32, 1, 1),
        vec![
            q_h.binding(),
            k_h.binding(),
            c_h.binding(),
            s_h.binding(),
            oq_h.clone().binding(),
            ok_h.clone().binding(),
        ],
        vec![n_q as u64, n_kv as u64, seq as u64, d as u64, pos as u64],
    );
    let bytes = client
        .read_async(vec![oq_h, ok_h])
        .await
        .map_err(|e| format!("rope readback failed: {e:?}"))?;
    for (which, x, heads, got) in [
        ("q", &xq, n_q, f32::from_bytes(&bytes[0])),
        ("k", &xk, n_kv, f32::from_bytes(&bytes[1])),
    ] {
        for h in 0..heads {
            for r in 0..seq {
                for c in 0..d {
                    let i = (h * seq + r) * d + c;
                    let mate = if c < half { -x[i + half] } else { x[i - half] };
                    let t = (pos + r) * d + c;
                    let expect = x[i] * cos_v[t] + mate * sin_v[t];
                    if (got[i] - expect).abs() > 1e-5 * expect.abs().max(1.0) {
                        return Err(format!(
                            "rope canary wrong at {which}[{i}]: got {}, \
                             expected {expect} — the K2 kernel must stay \
                             doored off here",
                            got[i]
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// One split-KV decode-attention canary: the deep-context two-pass
/// pipeline (segment partials, then combine) against a host softmax,
/// with the same fixtures the single-pass canary uses.
async fn split_attn_canary(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    d: usize,
    total: usize,
) -> Result<(), String> {
    use cubecl::prelude::*;

    let (n_q, n_kv, page_size, seg_len) = (4usize, 2usize, 16usize, 512usize);
    let segs = total.div_ceil(seg_len);
    let pages = total.div_ceil(page_size);
    let num_pages = pages + 2;
    let table: Vec<i32> = (0..pages).map(|p| ((p + 2) % num_pages) as i32).collect();
    let val = |i: usize, salt: f32| ((i as f32 * 0.437 + salt).sin() * 1.5) as f32;
    let qv: Vec<f32> = (0..n_q * d).map(|i| val(i, 42.5)).collect();
    let mut ak = vec![0.0f32; num_pages * n_kv * page_size * d];
    let mut av = vec![0.0f32; num_pages * n_kv * page_size * d];
    let key = |g: usize, j: usize, c: usize| val(c + (g * total + j) * d, 0.71);
    let value = |g: usize, j: usize, c: usize| val(c + (g * total + j) * d, 90.0);
    for j in 0..total {
        let phys = table[j / page_size] as usize;
        for g in 0..n_kv {
            let base = ((phys * n_kv + g) * page_size + j % page_size) * d;
            for c in 0..d {
                ak[base + c] = key(g, j, c);
                av[base + c] = value(g, j, c);
            }
        }
    }
    let scale = 1.0 / (d as f32).sqrt();
    let q_h = client.create_from_slice(f32::as_bytes(&qv));
    let k_h = client.create_from_slice(f32::as_bytes(&ak));
    let v_h = client.create_from_slice(f32::as_bytes(&av));
    let t_h = client.create_from_slice(i32::as_bytes(&table));
    let m_h = client.empty(n_q * segs * core::mem::size_of::<f32>());
    let s_h = client.empty(n_q * segs * core::mem::size_of::<f32>());
    let o_h = client.empty(n_q * segs * d * core::mem::size_of::<f32>());
    let out_h = client.empty(n_q * d * core::mem::size_of::<f32>());
    launch(
        client,
        DecodeAttnSplit(Storage::F32),
        CubeCount::Static(n_q as u32, segs as u32, 1),
        vec![
            q_h.binding(),
            k_h.binding(),
            v_h.binding(),
            t_h.binding(),
            m_h.clone().binding(),
            s_h.clone().binding(),
            o_h.clone().binding(),
        ],
        vec![
            n_q as u64,
            n_kv as u64,
            d as u64,
            page_size as u64,
            total as u64,
            seg_len as u64,
            segs as u64,
            f32::to_bits(scale) as u64,
        ],
    );
    launch(
        client,
        DecodeAttnCombine(Storage::F32),
        CubeCount::Static(n_q as u32, 1, 1),
        vec![
            m_h.binding(),
            s_h.binding(),
            o_h.binding(),
            out_h.clone().binding(),
        ],
        vec![n_q as u64, d as u64, segs as u64],
    );
    let bytes = client
        .read_async(vec![out_h])
        .await
        .map_err(|e| format!("split-attn readback failed: {e:?}"))?;
    let got = f32::from_bytes(&bytes[0]);
    for h in 0..n_q {
        let g = h / (n_q / n_kv);
        let qi = &qv[h * d..(h + 1) * d];
        let sc: Vec<f64> = (0..total)
            .map(|j| {
                (0..d).map(|c| qi[c] as f64 * key(g, j, c) as f64).sum::<f64>()
                    * scale as f64
            })
            .collect();
        let m = sc.iter().cloned().fold(f64::MIN, f64::max);
        let ps: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
        let sum: f64 = ps.iter().sum();
        for c in 0..d {
            let expect: f64 = (0..total)
                .map(|j| ps[j] / sum * value(g, j, c) as f64)
                .sum();
            let gv = got[h * d + c] as f64;
            if !gv.is_finite() || (gv - expect).abs() > 5e-3 * expect.abs().max(1.0) {
                return Err(format!(
                    "split-attn canary (d {d}, total {total}) wrong at                      [{h},{c}]: got {gv}, expected {expect} — the K8 path                      must stay doored off here"
                ));
            }
        }
    }
    Ok(())
}

/// One split-K gemv canary: the tiled canary's synthetic weights at
/// m = 1 through the WGSL kernel, checked against the CPU dequant
/// reference (the cube kernel is checked by the tiled canary already).
async fn gemv_canary(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    format: &str,
) -> Result<(), String> {
    use cubecl::prelude::*;

    let (n_out, k) = (211usize, 1024usize);
    let synth = |block_bytes: usize, scale_offsets: &[usize]| -> Vec<u8> {
        let blocks = match format {
            "q8_0" => n_out * k / 32,
            _ => n_out * k / 256,
        };
        let mut data: Vec<u8> = (0..blocks * block_bytes)
            .map(|i| ((i * 37 + 11) % 251) as u8)
            .collect();
        for b in 0..blocks {
            for (which, &off) in scale_offsets.iter().enumerate() {
                let bits: u16 = if which == 0 { 0x3C00 } else { 0x3400 };
                data[b * block_bytes + off..b * block_bytes + off + 2]
                    .copy_from_slice(&bits.to_le_bytes());
            }
        }
        data
    };
    let x: Vec<f32> = (0..k)
        .map(|i| ((i as f32 * 0.311).sin() * 1.4) - 0.1)
        .collect();
    let x_h = client.create_from_slice(f32::as_bytes(&x));
    let (dense, out_h) = match format {
        "q8_0" => {
            let data = synth(34, &[0]);
            let w = crate::qmatmul::Q80Weight::<WgpuRuntime>::from_gguf_bytes(
                client, &data, n_out, k,
            )
            .map_err(|e| format!("q8_0 pack failed: {e}"))?;
            let dense = combs_formats::quants::dequantize_q8_0(&data, n_out * k)
                .map_err(|e| format!("q8_0 reference failed: {e}"))?;
            (dense, w.decode_gemv_wgsl(client, x_h, Storage::F32, None))
        }
        "q4_k" => {
            let data = synth(144, &[0, 2]);
            let w = crate::qmatmul::Q4KWeight::<WgpuRuntime>::from_gguf_bytes(
                client, &data, n_out, k,
            )
            .map_err(|e| format!("q4_k pack failed: {e}"))?;
            let dense = combs_formats::quants::dequantize_q4_k(&data, n_out * k)
                .map_err(|e| format!("q4_k reference failed: {e}"))?;
            (dense, w.decode_gemv_wgsl(client, x_h, Storage::F32, None))
        }
        "q5_k" => {
            let data = synth(176, &[0, 2]);
            let w = crate::qmatmul::Q5KWeight::<WgpuRuntime>::from_gguf_bytes(
                client, &data, n_out, k,
            )
            .map_err(|e| format!("q5_k pack failed: {e}"))?;
            let dense = combs_formats::quants::dequantize_q5_k(&data, n_out * k)
                .map_err(|e| format!("q5_k reference failed: {e}"))?;
            (dense, w.decode_gemv_wgsl(client, x_h, Storage::F32, None))
        }
        _ => {
            let data = synth(210, &[208]);
            let w = crate::qmatmul::Q6KWeight::<WgpuRuntime>::from_gguf_bytes(
                client, &data, n_out, k,
            )
            .map_err(|e| format!("q6_k pack failed: {e}"))?;
            let dense = combs_formats::quants::dequantize_q6_k(&data, n_out * k)
                .map_err(|e| format!("q6_k reference failed: {e}"))?;
            (dense, w.decode_gemv_wgsl(client, x_h, Storage::F32, None))
        }
    };
    let bytes = client
        .read_async(vec![out_h])
        .await
        .map_err(|e| format!("{format} gemv canary readback failed: {e:?}"))?;
    let got = f32::from_bytes(&bytes[0]);
    for r in 0..n_out {
        let expect: f32 = (0..k).map(|c| x[c] * dense[r * k + c]).sum();
        let g = got[r];
        if !g.is_finite() || (g - expect).abs() > 1e-3 * expect.abs().max(1.0) {
            return Err(format!(
                "{format} gemv canary wrong at row {r}: got {g}, expected                  {expect} — the split-K gemv must stay doored off here"
            ));
        }
    }
    Ok(())
}

/// One quantized decode-attention canary: chosen int8 lanes packed
/// exactly like `kv_quantize` (lanes 0..2 offset-binary, lane 3 signed
/// in the top byte — including the `i32::MAX` corner), power-of-two
/// scales so the host dequant is exact, a shuffled page table, and a
/// host softmax over the dequantized rows as the reference.
async fn decode_attn_q8_canary(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    d: usize,
    total: usize,
) -> Result<(), String> {
    use cubecl::prelude::*;

    let (n_q, n_kv, page_size) = (4usize, 2usize, 16usize);
    let (dw, dg) = (d / 4, d / 32);
    let pages = total.div_ceil(page_size);
    let num_pages = pages + 2;
    let table: Vec<i32> = (0..pages).map(|p| ((p + 1) % num_pages) as i32).collect();
    let lanes = [-127i32, -1, 0, 1, 63, 127, -64, 5];
    let scales = [0.5f32, 0.25, 1.0, 0.125];
    let pack = |q0: i32, q1: i32, q2: i32, q3: i32| -> i32 {
        (q0 + 128) | ((q1 + 128) << 8) | ((q2 + 128) << 16) | (q3 << 24)
    };
    let mut build = |salt: usize| {
        let mut packed = vec![0i32; num_pages * n_kv * page_size * dw];
        let mut scal = vec![0.0f32; num_pages * n_kv * page_size * dg];
        let mut dense = vec![vec![0.0f32; d]; n_kv * total];
        for j in 0..total {
            let phys = table[j / page_size] as usize;
            for g in 0..n_kv {
                let row = (phys * n_kv + g) * page_size + j % page_size;
                for grp in 0..dg {
                    let sc = scales[(j + g + grp + salt) % scales.len()];
                    scal[row * dg + grp] = sc;
                    for w in 0..8 {
                        let word = if j == 0 && g == 0 && grp == 0 && w == 0 {
                            // The exact-i32::MAX corner: all-255 low
                            // bytes, +127 top lane.
                            pack(127, 127, 127, 127)
                        } else {
                            let pick = |l: usize| {
                                lanes[(j * 31 + g * 7 + grp * 5 + w * 3 + l + salt)
                                    % lanes.len()]
                            };
                            pack(pick(0), pick(1), pick(2), pick(3))
                        };
                        packed[row * dw + grp * 8 + w] = word;
                        let base = grp * 32 + w * 4;
                        let out = &mut dense[g * total + j];
                        out[base] = ((word & 0xff) - 128) as f32 * sc;
                        out[base + 1] = (((word >> 8) & 0xff) - 128) as f32 * sc;
                        out[base + 2] = (((word >> 16) & 0xff) - 128) as f32 * sc;
                        out[base + 3] = (word >> 24) as f32 * sc;
                    }
                }
            }
        }
        (packed, scal, dense)
    };
    let (kp, ks, kd) = build(1);
    let (vp, vs, vd) = build(9);
    let qv: Vec<f32> = (0..n_q * d)
        .map(|i| ((i as f32 * 0.437 + 4.2).sin() * 1.5))
        .collect();
    let scale = 1.0 / (d as f32).sqrt();

    let q_h = client.create_from_slice(f32::as_bytes(&qv));
    let kp_h = client.create_from_slice(i32::as_bytes(&kp));
    let ks_h = client.create_from_slice(f32::as_bytes(&ks));
    let vp_h = client.create_from_slice(i32::as_bytes(&vp));
    let vs_h = client.create_from_slice(f32::as_bytes(&vs));
    let t_h = client.create_from_slice(i32::as_bytes(&table));
    let o_h = client.empty(n_q * d * core::mem::size_of::<f32>());
    launch(
        client,
        DecodeAttnQ8,
        CubeCount::Static(n_q as u32, 1, 1),
        vec![
            q_h.binding(),
            kp_h.binding(),
            ks_h.binding(),
            vp_h.binding(),
            vs_h.binding(),
            t_h.binding(),
            o_h.clone().binding(),
        ],
        vec![
            n_q as u64,
            n_kv as u64,
            d as u64,
            page_size as u64,
            total as u64,
            0,
            0,
            f32::to_bits(scale) as u64,
        ],
    );
    let bytes = client
        .read_async(vec![o_h])
        .await
        .map_err(|e| format!("q8 decode-attn readback failed: {e:?}"))?;
    let got = f32::from_bytes(&bytes[0]);
    for h in 0..n_q {
        let g = h / (n_q / n_kv);
        if total == 1 {
            // Single visible key: weight is exactly 1.0 and the output is
            // the dequantized V row, bit for bit.
            for c in 0..d {
                let expect = vd[g][c];
                let gv = got[h * d + c];
                if gv.to_bits() != expect.to_bits() {
                    return Err(format!(
                        "q8 decode-attn canary (d {d}) unpack wrong at [{h},{c}]: \
                         got {gv}, expected exactly {expect} — the K3c kernel \
                         must stay doored off here"
                    ));
                }
            }
            continue;
        }
        let qi = &qv[h * d..(h + 1) * d];
        let sc: Vec<f64> = (0..total)
            .map(|j| {
                qi.iter()
                    .zip(&kd[g * total + j])
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>()
                    * scale as f64
            })
            .collect();
        let m = sc.iter().cloned().fold(f64::MIN, f64::max);
        let ps: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
        let sum: f64 = ps.iter().sum();
        for c in 0..d {
            let expect: f64 = (0..total)
                .map(|j| ps[j] / sum * vd[g * total + j][c] as f64)
                .sum();
            let gv = got[h * d + c] as f64;
            if !gv.is_finite() || (gv - expect).abs() > 5e-3 * expect.abs().max(1.0) {
                return Err(format!(
                    "q8 decode-attn canary (d {d}) wrong at [{h},{c}]: got \
                     {gv}, expected {expect} — the K3c kernel must stay \
                     doored off here"
                ));
            }
        }
    }
    Ok(())
}

/// One tiled-vs-untiled quant matmul canary: synthetic packed weights,
/// a prefill-shaped input (m = 32), both kernel variants read back and
/// the untiled one checked against the CPU dequant reference.
async fn tiled_matmul_canary(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    format: &str,
) -> Result<(), String> {
    use cubecl::prelude::*;

    let (n_out, k, m) = (48usize, 256usize, 32usize);
    // Deterministic packed bytes with sane f16 scales patched in at each
    // block's scale offsets (0x3C00 = 1.0, 0x3400 = 0.25).
    let synth = |block_bytes: usize, scale_offsets: &[usize]| -> Vec<u8> {
        let blocks = match format {
            "q8_0" => n_out * k / 32,
            _ => n_out * k / 256,
        };
        let mut data: Vec<u8> = (0..blocks * block_bytes)
            .map(|i| ((i * 37 + 11) % 251) as u8)
            .collect();
        for b in 0..blocks {
            for (which, &off) in scale_offsets.iter().enumerate() {
                let bits: u16 = if which == 0 { 0x3C00 } else { 0x3400 };
                data[b * block_bytes + off..b * block_bytes + off + 2]
                    .copy_from_slice(&bits.to_le_bytes());
            }
        }
        data
    };
    let (x_host, dense, tiled_h, untiled_h) = match format {
        "q8_0" => {
            let data = synth(34, &[0]);
            let w = crate::qmatmul::Q80Weight::<WgpuRuntime>::from_gguf_bytes(
                client, &data, n_out, k,
            )
            .map_err(|e| format!("q8_0 pack failed: {e}"))?;
            let dense = combs_formats::quants::dequantize_q8_0(&data, n_out * k)
                .map_err(|e| format!("q8_0 reference failed: {e}"))?;
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i as f32 * 0.377).sin() * 1.3) - 0.2)
                .collect();
            let x_h = client.create_from_slice(f32::as_bytes(&x));
            let t = w.matmul_device_with(client, x_h.clone(), m, true);
            let u = w.matmul_device_with(client, x_h, m, false);
            (x, dense, t, u)
        }
        "q4_k" => {
            let data = synth(144, &[0, 2]);
            let w = crate::qmatmul::Q4KWeight::<WgpuRuntime>::from_gguf_bytes(
                client, &data, n_out, k,
            )
            .map_err(|e| format!("q4_k pack failed: {e}"))?;
            let dense = combs_formats::quants::dequantize_q4_k(&data, n_out * k)
                .map_err(|e| format!("q4_k reference failed: {e}"))?;
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i as f32 * 0.377).sin() * 1.3) - 0.2)
                .collect();
            let x_h = client.create_from_slice(f32::as_bytes(&x));
            let t = w.matmul_device_with(client, x_h.clone(), m, true);
            let u = w.matmul_device_with(client, x_h, m, false);
            (x, dense, t, u)
        }
        _ => {
            // Q6_K has no tiled variant — prefill and decode share the
            // one kernel, so the reference check alone covers it.
            let data = synth(210, &[208]);
            let w = crate::qmatmul::Q6KWeight::<WgpuRuntime>::from_gguf_bytes(
                client, &data, n_out, k,
            )
            .map_err(|e| format!("q6_k pack failed: {e}"))?;
            let dense = combs_formats::quants::dequantize_q6_k(&data, n_out * k)
                .map_err(|e| format!("q6_k reference failed: {e}"))?;
            let x: Vec<f32> = (0..m * k)
                .map(|i| ((i as f32 * 0.377).sin() * 1.3) - 0.2)
                .collect();
            let x_h = client.create_from_slice(f32::as_bytes(&x));
            let t = w.matmul_device(client, x_h.clone(), m);
            let u = w.matmul_device(client, x_h, m);
            (x, dense, t, u)
        }
    };
    let bytes = client
        .read_async(vec![tiled_h, untiled_h])
        .await
        .map_err(|e| format!("{format} tiled canary readback failed: {e:?}"))?;
    let tiled = f32::from_bytes(&bytes[0]);
    let untiled = f32::from_bytes(&bytes[1]);
    for i in 0..m * n_out {
        if tiled[i].to_bits() != untiled[i].to_bits() {
            return Err(format!(
                "{format} tiled canary: tiled[{i}] = {} differs from                  untiled {} — the tiled prefill kernel computes different                  values on this target",
                tiled[i], untiled[i]
            ));
        }
    }
    for r in 0..m {
        for c in 0..n_out {
            let expect: f32 = (0..k)
                .map(|j| x_host[r * k + j] * dense[c * k + j])
                .sum();
            let g = untiled[r * n_out + c];
            if !g.is_finite() || (g - expect).abs() > 1e-3 * expect.abs().max(1.0) {
                return Err(format!(
                    "{format} tiled canary: untiled[{r},{c}] = {g}, host                      reference {expect} — the quant matmul itself is wrong                      on this target"
                ));
            }
        }
    }
    Ok(())
}

/// One decode-attention canary case: build the fixture, launch, check
/// every output element against a host softmax.
async fn decode_attn_canary(
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    mode: usize,
    d: usize,
    window: usize,
    n_kv: usize,
    total: usize,
) -> Result<(), String> {
    use cubecl::prelude::*;

    let (n_q, page_size) = (4usize, 16usize);
    let pages = total.div_ceil(page_size);
    let num_pages = pages + 2;
    // A rotation is collision-free for every page count — the stride-3
    // shuffle this used to be collides whenever gcd(3, num_pages) > 1.
    let table: Vec<i32> = (0..pages).map(|p| ((p + 2) % num_pages) as i32).collect();
    let val = |i: usize, salt: f32| ((i as f32 * 0.437 + salt).sin() * 1.5) as f32;
    let qv: Vec<f32> = (0..n_q * d).map(|i| val(i, 42.5)).collect();
    let mut ak = vec![0.0f32; num_pages * n_kv * page_size * d];
    let mut av = vec![0.0f32; num_pages * n_kv * page_size * d];
    let key = |g: usize, j: usize, c: usize| val(c + (g * total + j) * d, 0.71);
    let value = |g: usize, j: usize, c: usize| val(c + (g * total + j) * d, 90.0);
    if mode == 0 {
        for j in 0..total {
            let phys = table[j / page_size] as usize;
            for g in 0..n_kv {
                let base = ((phys * n_kv + g) * page_size + j % page_size) * d;
                for c in 0..d {
                    ak[base + c] = key(g, j, c);
                    av[base + c] = value(g, j, c);
                }
            }
        }
    } else {
        ak = (0..n_kv * total * d)
            .map(|i| key(i / (total * d), (i / d) % total, i % d))
            .collect();
        av = (0..n_kv * total * d)
            .map(|i| value(i / (total * d), (i / d) % total, i % d))
            .collect();
    }
    let scale = 0.125f32;
    let q_h = client.create_from_slice(f32::as_bytes(&qv));
    let k_h = client.create_from_slice(f32::as_bytes(&ak));
    let v_h = client.create_from_slice(f32::as_bytes(&av));
    let t_h = client.create_from_slice(i32::as_bytes(&table));
    let o_h = client.empty(n_q * d * core::mem::size_of::<f32>());
    launch(
        client,
        DecodeAttn(Storage::F32),
        CubeCount::Static(n_q as u32, 1, 1),
        vec![
            q_h.binding(),
            k_h.binding(),
            v_h.binding(),
            t_h.binding(),
            o_h.clone().binding(),
        ],
        vec![
            n_q as u64,
            n_kv as u64,
            d as u64,
            page_size as u64,
            total as u64,
            window as u64,
            mode as u64,
            f32::to_bits(scale) as u64,
        ],
    );
    let bytes = client
        .read_async(vec![o_h])
        .await
        .map_err(|e| format!("decode-attn readback failed: {e:?}"))?;
    let got = f32::from_bytes(&bytes[0]);
    for h in 0..n_q {
        let g = h / (n_q / n_kv);
        let qi = &qv[h * d..(h + 1) * d];
        let visible: Vec<usize> = (0..total)
            .filter(|&j| window == 0 || j + window >= total)
            .collect();
        let scores: Vec<f32> = visible
            .iter()
            .map(|&j| {
                (0..d).map(|c| qi[c] * key(g, j, c)).sum::<f32>() * scale
            })
            .collect();
        let m = scores.iter().cloned().fold(f32::MIN, f32::max);
        let ps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let sum: f32 = ps.iter().sum();
        for c in 0..d {
            let expect: f32 = visible
                .iter()
                .enumerate()
                .map(|(idx, &j)| ps[idx] / sum * value(g, j, c))
                .sum();
            let gv = got[h * d + c];
            if (gv - expect).abs() > 1e-3 * expect.abs().max(1.0) {
                return Err(format!(
                    "decode-attn canary (mode {mode}, d {d}, window \
                     {window}) wrong at [{h},{c}]: got {gv}, expected \
                     {expect} — the K3 kernel must stay doored off here"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::prelude::*;

    /// The exact probe the browser runs, on the native compiler — when
    /// the Chrome probe fails a canary that passes here, the divergence
    /// is Tint's, not the fixture's.
    /// The default is a promise to every box at once, so it opens only
    /// where it has been blessed and the switch wins in both directions.
    #[test]
    fn the_storage_door_defaults_open_only_where_it_is_blessed() {
        const BLESSED_VULKAN_VENDOR: u32 = 0x10de;
        const OTHER_VENDOR: u32 = 0x1002;
        const THIRD_VENDOR: u32 = 0x8086;
        assert!(storage_door_default("Metal", 0, None), "blessed on Metal: open by default");
        assert!(
            storage_door_default("Vulkan", BLESSED_VULKAN_VENDOR, None),
            "blessed on Vulkan for this vendor: open by default"
        );
        assert!(!storage_door_default("Vulkan", OTHER_VENDOR, None), "unproven for another vendor on the same backend: shut");
        assert!(!storage_door_default("Vulkan", THIRD_VENDOR, None), "unproven: shut");
        assert!(!storage_door_default("Dx12", BLESSED_VULKAN_VENDOR, None), "a blessing is a backend and a vendor together: shut");
        assert!(!storage_door_default("Gl", BLESSED_VULKAN_VENDOR, None), "unproven backend: shut");
        assert!(!storage_door_default("Metal", 0, Some("0")), "the off switch wins where it is blessed");
        assert!(storage_door_default("Vulkan", OTHER_VENDOR, Some("1")), "the on switch wins where it is not");
    }

    #[test]
    fn every_blessed_row_names_its_evidence() {
        for (backend, _, evidence) in super::STORAGE_BLESSED {
            assert!(evidence.starts_with('§'), "the {backend} row names no measurement");
        }
    }

    /// The door takes exactly the cuts the sweep asks about, spelled as
    /// the doors report prints them, and a refusal names every cut it
    /// would have taken — a mistyped value must not quietly run a cut
    /// nobody measured.
    #[test]
    fn the_geometry_door_takes_only_the_swept_cuts() {
        assert_eq!(geometry_door(None), Ok(Geometry::DEFAULT), "no value: the default cut");
        for (text, rows, lanes) in [("8x32", 8, 32), ("16x16", 16, 16), ("32x8", 32, 8)] {
            assert_eq!(geometry_door(Some(text)), Ok(Geometry { rows, lanes }), "{text}");
        }
        for bad in ["", "8X32", " 8x32", "8x32 ", "4x64", "64x4", "8x16", "16", "banana"] {
            let why = geometry_door(Some(bad)).expect_err(bad);
            for cut in ["8x32", "16x16", "32x8"] {
                assert!(why.contains(cut), "the refusal of {bad:?} names {cut}: {why}");
            }
        }
    }

    /// Every cut renders a shader whose workgroup size matches the cut,
    /// and the default rendering is still the audited 16x16 text.
    #[test]
    fn every_gemv_cut_renders_its_own_workgroup_size() {
        use burn_cubecl::template::KernelSource;
        for g in Geometry::SWEEP {
            let t = DecodeGemvQ4K(Storage::F32, *g).source().complete();
            assert!(
                t.contains(&format!("const ROWS: u32 = {}u;", g.rows)),
                "cut {g:?} did not render its rows"
            );
            assert!(
                t.contains(&format!("const LANES: u32 = {}u;", g.lanes)),
                "cut {g:?} did not render its lanes"
            );
            assert!(!t.contains("{{"), "cut {g:?} left a placeholder");
            assert_eq!(g.workgroup(), 256, "a cut must fill the workgroup");
        }
        let default = DecodeGemvQ4K(Storage::F32, Geometry::DEFAULT).source().complete();
        assert!(default.contains("const ROWS: u32 = 16u;"));
        assert!(default.contains("@compute @workgroup_size(16 * 16)"));
    }

    /// The dispatch-latency cell counts exactly what it times: the
    /// counter delta across the window equals the launches issued.
    /// Anything else and the number is someone else's launches.
    ///
    /// Own process, like every counter test (§92's lesson, repeated
    /// once here before it was honoured): the counter is process-wide,
    /// so under the parallel suite a sibling's launches land in the
    /// window and the assertion's own message comes true. The LIVE
    /// proof is `combs-kit micro`'s dispatch cell, which runs in a
    /// child and prints counted against issued (§102: 2,000 / 2,000).
    ///
    /// ```text
    /// cargo test --release -p combs-models --lib -- --ignored \
    ///   --test-threads=1 counts_exactly_what_it_times
    /// ```
    #[test]
    #[ignore = "gpu: reads the process-wide dispatch counter, so it needs its own process"]
    fn the_empty_launch_cell_counts_exactly_what_it_times() {
        if crate::skip_no_gpu() {
            return;
        }
        let (took, delta) = dispatch_latency(64);
        assert_eq!(delta, 64, "the window held launches that were not ours");
        assert!(took.as_micros() > 0, "sixty-four launches took no time at all");
    }

// ======================================================================
// Paste INSIDE `mod tests` of engine/core/combs-models/src/wgsl/mod.rs
// (see where_to_insert). Two #[ignore] tests plus three private helpers.
//
//   BLOCK 1  replay_spike_cubecl_only          compiles at 571c733 as-is
//   BLOCK 2  replay_spike_today_vs_raw_replay  needs ONE line in
//            combs-models/Cargo.toml under [dev-dependencies]:  wgpu = "29"
//
// Why BLOCK 2 needs the line (read, not guessed): the raw wgpu VALUES are
// reachable from this crate — `init_setup` returns `WgpuSetup { device,
// queue, adapter }` (vendor/cubecl-wgpu/src/runtime.rs:210-220, re-exported
// by burn-wgpu) and `client.get_resource` returns `WgpuResource { buffer,
// offset, size }` — but the wgpu TYPE PATHS are not: nothing on the chain
// re-exports the crate (vendor/cubecl-wgpu/src/lib.rs has no `pub use
// wgpu`; burn-wgpu's lib.rs re-exports cubecl-wgpu items only; combs-core's
// `pub use burn::backend::wgpu` is burn's module, not `::wgpu`). A pipeline,
// a bind group, a query set and a mappable buffer are struct literals
// (`ShaderModuleDescriptor`, `ComputePipelineDescriptor`, `BindGroupEntry`,
// `QuerySetDescriptor`, `BufferDescriptor`) and a struct literal needs the
// path. The dev-dependency mirrors combs-core's own `wgpu = "29"` (same
// resolved 29.0.4, features already unified through combs-core, no new
// crate compiled, never part of a non-test build).
//
// Own process, both tests: they perform the one cubecl initialization
// themselves (`init_setup`, the doorway to the raw device/queue) and a
// second initialization on the same device panics ("Can't create a new
// client on an already registered server", cubecl-runtime client.rs:56-59).
// `skip_no_gpu` cannot be used here: `pin_test_device` → `set_default_dtypes`
// → `B::supports_dtype` → `R::client(device)` loads the client and would
// initialize lazily with cubecl's own defaults (burn-backend device.rs:260-
// 272, vendor/burn-cubecl/src/backend.rs:102-104). So the gate is
// `combs_core::gpu_available` (a fresh instance, no cubecl), then
// `init_setup`, then the pin.
// ======================================================================

    /// The engine's runtime options as `combs_core::combs_runtime_options`
    /// builds them (private there): exclusive pages, 16 tasks a submit,
    /// the person's `CUBECL_WGPU_MAX_TASKS` winning when set.
    fn spike_runtime_options() -> burn::backend::wgpu::RuntimeOptions {
        use burn::backend::wgpu::{MemoryConfiguration, RuntimeOptions};
        let tasks_max = match std::env::var("CUBECL_WGPU_MAX_TASKS") {
            Ok(_) => RuntimeOptions::default().tasks_max,
            Err(_) => combs_core::SUBMIT_TASKS,
        };
        let memory_config =
            if matches!(std::env::var("COMBS_MEM_POOLS").as_deref(), Ok("subslices")) {
                MemoryConfiguration::default()
            } else {
                MemoryConfiguration::ExclusivePages
            };
        RuntimeOptions { tasks_max, memory_config }
    }

    /// The synthetic Q4_K table `gemv_cell_at` uses (unit `d`, quarter
    /// `dmin`, pseudo-random nibbles) and the activation beside it.
    fn spike_fixture(n_out: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
        let blocks = n_out * k / 256;
        let mut data: Vec<u8> = (0..blocks * 144).map(|i| ((i * 37 + 11) % 251) as u8).collect();
        for b in 0..blocks {
            data[b * 144..b * 144 + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
            data[b * 144 + 2..b * 144 + 4].copy_from_slice(&0x3400u16.to_le_bytes());
        }
        let x: Vec<f32> = (0..k).map(|i| (((i as f32 * 0.311).sin() * 1.4) - 0.1) / 32.0).collect();
        (data, x)
    }

    fn spike_env(name: &str, default: usize) -> usize {
        std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    }

    /// Microseconds per unit.
    fn spike_us(d: std::time::Duration, per: usize) -> f64 {
        d.as_secs_f64() * 1e6 / per as f64
    }

    // ------------------------------------------------------------------
    // BLOCK 1 — the road with no Cargo change: arm (a) and the pieces of
    // it the cubecl client can price without naming a wgpu type.
    //
    // Run it as its own process THREE times: `CUBECL_WGPU_MAX_TASKS` unset
    // (the house 16), `=1024`, and `=2048` (≥ N: every launch joins ONE
    // pass and ONE submit at the sync — cubecl's own single-flush road,
    // with the caveat printed below about MAX_PENDING_WRITES). The runtime
    // reads the door once, at init.
    //
    //   cargo test --release -p combs-models --lib -- \
    //     wgsl::tests::replay_spike_cubecl_only --exact --ignored --nocapture --test-threads=1
    // ------------------------------------------------------------------
    #[test]
    #[ignore = "gpu: initializes the device itself and reads the process-wide counters, so it needs its own process"]
    fn replay_spike_cubecl_only() {
        use burn::backend::wgpu::{graphics::AutoGraphicsApi, init_setup, WgpuDevice};
        use cubecl::prelude::*;
        use std::time::Instant;

        if !combs_core::gpu_available() {
            return;
        }
        let device = WgpuDevice::default();
        let options = spike_runtime_options();
        let tasks_max = options.tasks_max;
        let setup = init_setup::<AutoGraphicsApi>(&device, options);
        crate::pin_test_device();
        let client = WgpuRuntime::client(&device);
        let info = setup.adapter.get_info();
        eprintln!(
            "[spike] {} · {:?} · {:?} · driver {} ({}) · tasks_max {} · runtime timing {}",
            info.name, info.backend, info.device_type, info.driver, info.driver_info,
            tasks_max, client.properties().timing_method
        );
        if tasks_max >= 64 {
            eprintln!(
                "[spike] note: tasks_max >= 64 — every launch's uniform is a write_to_buffer, and at \
                 MAX_PENDING_WRITES (64, vendor stream.rs) the stream submits an empty encoder and polls \
                 Wait; this configuration is not one flush"
            );
        }

        let (n_out, k) = (2048usize, 1024usize);
        let n = spike_env("COMBS_SPIKE_N", 2000);
        let reps = spike_env("COMBS_SPIKE_REPS", 3);
        let geom = Geometry::DEFAULT;
        let wg = (n_out as u32).div_ceil(geom.rows).max(1);
        let (data, x) = spike_fixture(n_out, k);

        let w = crate::qmatmul::Q4KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k)
            .expect("a synthetic q4_k weight packs");
        let x_h = client.create_from_slice(f32::as_bytes(&x));
        // The same table again as bare handles, for the launches that
        // bypass the wrapper (its handles are private to qmatmul).
        let (qs, dd, scales) = crate::qmatmul::repack_q4_k(&data).expect("repack");
        let qs_h = client.create_from_slice(u32::as_bytes(&qs));
        let dd_h = client.create_from_slice(f32::as_bytes(&dd));
        let sc_h = client.create_from_slice(u32::as_bytes(&scales));
        let out_fixed = client.empty(n_out * core::mem::size_of::<f32>());

        // Warm: the first launch compiles the pipeline, not a launch cost.
        let _ = w.decode_gemv_wgsl_at(&client, x_h.clone(), Storage::F32, None, geom);
        cubecl::future::block_on(client.sync()).expect("sync after warm-up");

        for rep in 0..reps {
            // (a) today's road: N launches of decode_gemv_wgsl_at — a fresh
            // `client.empty` output each, six bindings, three scalars —
            // then one sync.
            let (d0, s0) = combs_core::gpu_counters();
            let t0 = Instant::now();
            for _ in 0..n {
                let _ = w.decode_gemv_wgsl_at(&client, x_h.clone(), Storage::F32, None, geom);
            }
            let a_issue = t0.elapsed();
            cubecl::future::block_on(client.sync()).expect("sync after the launches");
            let a_wall = t0.elapsed();
            let (d1, s1) = combs_core::gpu_counters();
            assert_eq!(d1 - d0, n as u64, "the window held launches that were not ours");

            // (a-fixed) the same launch with ONE preallocated output: the
            // per-launch `client.empty` (page scan + a handle) priced by
            // difference.
            let t0 = Instant::now();
            for _ in 0..n {
                launch_gemv(
                    &client,
                    GemvKernel::Q4K,
                    Storage::F32,
                    geom,
                    wg,
                    vec![
                        x_h.clone().binding(),
                        qs_h.clone().binding(),
                        dd_h.clone().binding(),
                        sc_h.clone().binding(),
                        out_fixed.clone().binding(),
                        out_fixed.clone().binding(),
                    ],
                    vec![n_out as u64, k as u64, 0],
                );
            }
            let f_issue = t0.elapsed();
            cubecl::future::block_on(client.sync()).expect("sync after the fixed launches");
            let f_wall = t0.elapsed();

            // The GPU span the runtime's own timestamp pair reads. It covers
            // the FIRST pass after start_profile only (vendor timings.rs:
            // init_query_set drains init_tokens once and `current` never
            // moves), i.e. min(n, tasks_max) back-to-back dispatches. With
            // CUBECL_WGPU_MAX_TASKS >= n that pass is the whole window.
            let (w_ref, x_ref, c2) = (&w, &x_h, client.clone());
            let ((), prof) = client
                .profile(
                    move || {
                        for _ in 0..n {
                            let _ = w_ref.decode_gemv_wgsl_at(&c2, x_ref.clone(), Storage::F32, None, geom);
                        }
                    },
                    "replay-spike-a",
                )
                .expect("profile");
            let method = prof.timing_method();
            let span = cubecl::future::block_on(prof.resolve()).duration();
            let in_pass = n.min(tasks_max);

            // Two pieces alone: an output allocation freed at once (the
            // in-situ pattern), and one blocking round trip to the device
            // thread (what get_resource costs; a submit_blocking rendezvous).
            let t0 = Instant::now();
            for _ in 0..n {
                let _ = client.empty(n_out * core::mem::size_of::<f32>());
            }
            let empty_each = t0.elapsed();
            let t0 = Instant::now();
            for _ in 0..n {
                let _ = client.get_resource(out_fixed.clone()).expect("a handle resolves");
            }
            let resolve_each = t0.elapsed();

            eprintln!("[spike] rep {rep} · n {n} · tasks_max {tasks_max}");
            eprintln!(
                "[spike]   (a) today        cpu issue {:.2} us/launch · wall {:.2} us/launch · {} submits · {:.3} ms",
                spike_us(a_issue, n), spike_us(a_wall, n), s1 - s0, a_wall.as_secs_f64() * 1e3
            );
            eprintln!(
                "[spike]   (a-fixed)        cpu issue {:.2} us/launch · wall {:.2} us/launch (one output handle, no client.empty)",
                spike_us(f_issue, n), spike_us(f_wall, n)
            );
            eprintln!(
                "[spike]   (a) gpu span     {:.1} us over the first pass of {in_pass} dispatches = {:.2} us/dispatch ({method} timing)",
                span.as_secs_f64() * 1e6, spike_us(span, in_pass)
            );
            eprintln!(
                "[spike]   pieces           client.empty {:.2} us · get_resource round trip {:.2} us",
                spike_us(empty_each, n), spike_us(resolve_each, n)
            );
        }
    }

    // ------------------------------------------------------------------
    // BLOCK 2 — the spike itself. Needs `wgpu = "29"` in [dev-dependencies].
    //
    // (a) today's road: N × decode_gemv_wgsl_at through the cubecl client,
    //     one sync at the end; the meter proves the count.
    // (b) the replay ceiling: the SAME render (DecodeGemvQ4K(F32, 16x16)
    //     .source().complete()) compiled once with raw wgpu (naga), the
    //     weights uploaded once through cubecl and resolved with
    //     client.get_resource, ONE constant uniform, ONE bind group, N ×
    //     (set_bind_group + dispatch) recorded into ONE pass of ONE
    //     encoder, one submit, one poll — a TIMESTAMP_QUERY pair around the
    //     pass reads the GPU span beside the wall.
    // (c) (b) with a fresh bind group per dispatch — entries +
    //     get_bind_group_layout(0) + create_bind_group, what the vendored
    //     stream does per launch (stream.rs register_pipeline) — so
    //     (c) − (b) prices bind-group creation alone.
    //
    //   cargo test --release -p combs-models --lib -- \
    //     wgsl::tests::replay_spike_today_vs_raw_replay --exact --ignored --nocapture --test-threads=1
    // ------------------------------------------------------------------
    #[test]
    #[ignore = "gpu: initializes the device itself and reads the process-wide counters, so it needs its own process"]
    fn replay_spike_today_vs_raw_replay() {
        use burn::backend::wgpu::{graphics::AutoGraphicsApi, init_setup, WgpuDevice};
        use burn_cubecl::template::KernelSource;
        use cubecl::prelude::*;
        use std::time::{Duration, Instant};

        if !combs_core::gpu_available() {
            return;
        }
        let device = WgpuDevice::default();
        let options = spike_runtime_options();
        let tasks_max = options.tasks_max;
        let setup = init_setup::<AutoGraphicsApi>(&device, options);
        crate::pin_test_device();
        let client = WgpuRuntime::client(&device);
        let dev = &setup.device;
        let queue = &setup.queue;
        let stamps = dev.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let info = setup.adapter.get_info();
        eprintln!(
            "[spike] {} · {:?} · {:?} · driver {} ({}) · tasks_max {} · timestamp_query {}",
            info.name, info.backend, info.device_type, info.driver, info.driver_info, tasks_max, stamps
        );

        let (n_out, k) = (2048usize, 1024usize);
        let n = spike_env("COMBS_SPIKE_N", 2000);
        let reps = spike_env("COMBS_SPIKE_REPS", 3);
        let geom = Geometry::DEFAULT;
        let wg = (n_out as u32).div_ceil(geom.rows).max(1);
        let packed = (n_out * k / 256) * 144;
        let (data, x) = spike_fixture(n_out, k);

        // ---- (a) today's road -------------------------------------------
        let w = crate::qmatmul::Q4KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k)
            .expect("a synthetic q4_k weight packs");
        let x_h = client.create_from_slice(f32::as_bytes(&x));
        // Warm (compiles the pipeline), keep the answer as the reference,
        // and check it against the host once — a road that agrees with
        // this one then agrees with the truth.
        let out_ref = w.decode_gemv_wgsl_at(&client, x_h.clone(), Storage::F32, None, geom);
        let reference: Vec<f32> = f32::from_bytes(&client.read_one_unchecked(out_ref)).to_vec();
        let dense = combs_formats::quants::dequantize_q4_k(&data, n_out * k).expect("reference");
        let mut worst = 0.0f32;
        for r in 0..n_out {
            let expect: f32 = (0..k).map(|c| x[c] * dense[r * k + c]).sum();
            worst = worst.max((reference[r] - expect).abs() / expect.abs().max(1.0));
        }
        assert!(worst < 1e-4, "today's road disagrees with the host: rel {worst}");
        cubecl::future::block_on(client.sync()).expect("sync after warm-up");

        // ---- (b) the raw road -------------------------------------------
        // The table once more as bare handles (the wrapper's are private),
        // uploaded through cubecl and resolved to (buffer, offset, size).
        let (qs, dd, scales) = crate::qmatmul::repack_q4_k(&data).expect("repack");
        let qs_h = client.create_from_slice(u32::as_bytes(&qs));
        let dd_h = client.create_from_slice(f32::as_bytes(&dd));
        let sc_h = client.create_from_slice(u32::as_bytes(&scales));
        let xb_h = client.create_from_slice(f32::as_bytes(&x));
        let out_h = client.empty(n_out * core::mem::size_of::<f32>());
        // Everything cubecl has pending lands before a foreign submit
        // touches these buffers.
        cubecl::future::block_on(client.sync()).expect("sync before the raw road");
        // Binding order is the shader's: x, qs, dd, scales, out, residual
        // (bound to out when nothing is fused), then the uniform at 6.
        // The ManagedResource keeps each page alive for the whole run.
        let resolved: Vec<_> = [&xb_h, &qs_h, &dd_h, &sc_h, &out_h, &out_h]
            .into_iter()
            .map(|h| client.get_resource(h.clone()).expect("a handle resolves to its buffer"))
            .collect();

        let source = DecodeGemvQ4K(Storage::F32, geom).source().complete();
        // SAFETY: the same trust the vendored server extends to this render
        // (backend/base.rs create_module: bounds checks off, loop bounding
        // on in Checked mode, which is how launch_at runs it); every access
        // in the shader self-guards, per the module contract above.
        let module = unsafe {
            dev.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("replay-spike q4k gemv"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                },
                wgpu::ShaderRuntimeChecks {
                    bounds_checks: false,
                    force_loop_bounding: true,
                    ..wgpu::ShaderRuntimeChecks::unchecked()
                },
            )
        };
        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("replay-spike q4k gemv"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        });
        // The scalars as the kernel contract lays them out: u64 slots, low
        // word first — n_out, k, fuse = 0. One constant uniform, written once.
        let params: Vec<u8> = [n_out as u64, k as u64, 0u64]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let uniform = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("replay-spike params"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform, 0, &params);
        let entries = || {
            resolved
                .iter()
                .enumerate()
                .map(|(i, r)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: r.resource().as_wgpu_bind_resource(),
                })
                .chain(std::iter::once(wgpu::BindGroupEntry {
                    binding: 6,
                    resource: uniform.as_entire_binding(),
                }))
                .collect::<Vec<_>>()
        };
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries(),
        });

        let query_set = stamps.then(|| {
            dev.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("replay-spike"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            })
        });
        let resolve_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("replay-spike resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let map_buf = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("replay-spike map"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let period = queue.get_timestamp_period() as f64;

        // One raw run → (record, finish, submit, poll, gpu span). `fresh`
        // rebuilds the bind group per dispatch, as the vendored stream does.
        let run_raw = |fresh: bool| -> (Duration, Duration, Duration, Duration, Option<Duration>) {
            let t0 = Instant::now();
            let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("replay-spike"),
            });
            {
                let timestamp_writes = query_set.as_ref().map(|qs| wgpu::ComputePassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("replay-spike"),
                    timestamp_writes,
                });
                pass.set_pipeline(&pipeline);
                for _ in 0..n {
                    if fresh {
                        let layout = pipeline.get_bind_group_layout(0);
                        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &layout,
                            entries: &entries(),
                        });
                        pass.set_bind_group(0, &bg, &[]);
                    } else {
                        pass.set_bind_group(0, &bind_group, &[]);
                    }
                    pass.dispatch_workgroups(wg, 1, 1);
                }
            }
            if let Some(qs) = &query_set {
                encoder.resolve_query_set(qs, 0..2, &resolve_buf, 0);
                encoder.copy_buffer_to_buffer(&resolve_buf, 0, &map_buf, 0, Some(16));
            }
            let recorded = t0.elapsed();
            let command = encoder.finish();
            let finished = t0.elapsed();
            let index = queue.submit([command]);
            let submitted = t0.elapsed();
            if stamps {
                map_buf.slice(..).map_async(wgpu::MapMode::Read, |_| ());
            }
            dev.poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None })
                .expect("poll");
            let polled = t0.elapsed();
            let gpu = stamps.then(|| {
                let ticks: Vec<u64> = map_buf
                    .slice(..)
                    .get_mapped_range()
                    .chunks_exact(8)
                    .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                map_buf.unmap();
                Duration::from_nanos((ticks[1].saturating_sub(ticks[0]) as f64 * period) as u64)
            });
            (recorded, finished - recorded, submitted - finished, polled - submitted, gpu)
        };

        // Warm the raw road once: first-use costs are not a dispatch cost.
        let _ = run_raw(false);

        let gpu_line = |g: Option<Duration>| match g {
            Some(g) => format!(
                "{:.2} us/dispatch ({:.1} GB/s over {} packed bytes)",
                spike_us(g, n),
                packed as f64 / (g.as_secs_f64() / n as f64) / 1e9,
                packed
            ),
            None => "n/a (no TIMESTAMP_QUERY)".to_string(),
        };

        for rep in 0..reps {
            // (a)
            let (d0, s0) = combs_core::gpu_counters();
            let t0 = Instant::now();
            for _ in 0..n {
                let _ = w.decode_gemv_wgsl_at(&client, x_h.clone(), Storage::F32, None, geom);
            }
            let a_issue = t0.elapsed();
            cubecl::future::block_on(client.sync()).expect("sync after the launches");
            let a_wall = t0.elapsed();
            let (d1, s1) = combs_core::gpu_counters();
            assert_eq!(d1 - d0, n as u64, "the window held launches that were not ours");
            // (a)'s GPU span from the runtime's own timestamp pair — the
            // first pass only, min(n, tasks_max) dispatches (see BLOCK 1).
            let (w_ref, x_ref, c2) = (&w, &x_h, client.clone());
            let ((), prof) = client
                .profile(
                    move || {
                        for _ in 0..n {
                            let _ = w_ref.decode_gemv_wgsl_at(&c2, x_ref.clone(), Storage::F32, None, geom);
                        }
                    },
                    "replay-spike-a",
                )
                .expect("profile");
            let method = prof.timing_method();
            let a_span = cubecl::future::block_on(prof.resolve()).duration();
            let in_pass = n.min(tasks_max);

            // (b) and (c)
            let (b_rec, b_fin, b_sub, b_poll, b_gpu) = run_raw(false);
            let (c_rec, c_fin, c_sub, c_poll, c_gpu) = run_raw(true);
            let b_wall = b_rec + b_fin + b_sub + b_poll;
            let c_wall = c_rec + c_fin + c_sub + c_poll;

            eprintln!("[spike] rep {rep} · n {n} · tasks_max {tasks_max} · workgroups {wg} x 256");
            eprintln!(
                "[spike]   (a) today        cpu issue {:.2} us/launch · wall {:.2} us/launch · {} submits · {:.3} ms",
                spike_us(a_issue, n), spike_us(a_wall, n), s1 - s0, a_wall.as_secs_f64() * 1e3
            );
            eprintln!(
                "[spike]   (a) gpu span     {:.1} us over the first pass of {in_pass} dispatches = {:.2} us/dispatch ({method} timing)",
                a_span.as_secs_f64() * 1e6, spike_us(a_span, in_pass)
            );
            eprintln!(
                "[spike]   (b) raw replay   cpu record {:.2} + finish {:.2} + submit {:.2} = {:.2} us/dispatch · poll {:.2} · wall {:.2} us/dispatch · {:.3} ms",
                spike_us(b_rec, n), spike_us(b_fin, n), spike_us(b_sub, n),
                spike_us(b_rec + b_fin + b_sub, n), spike_us(b_poll, n), spike_us(b_wall, n),
                b_wall.as_secs_f64() * 1e3
            );
            eprintln!("[spike]   (b) gpu span     {}", gpu_line(b_gpu));
            if let Some(g) = b_gpu {
                eprintln!(
                    "[spike]   (b) host residual (wall - gpu span) {:.2} us/dispatch",
                    spike_us(b_wall.saturating_sub(g), n)
                );
            }
            eprintln!(
                "[spike]   (c) raw, fresh bind group each   cpu {:.2} us/dispatch · wall {:.2} us/dispatch · gpu span {} · (c)-(b) cpu {:.2} us/dispatch",
                spike_us(c_rec + c_fin + c_sub, n), spike_us(c_wall, n), gpu_line(c_gpu),
                spike_us((c_rec + c_fin + c_sub).saturating_sub(b_rec + b_fin + b_sub), n)
            );
            let a_us = spike_us(a_wall, n);
            let b_us = spike_us(b_wall, n);
            eprintln!(
                "[spike]   bar: (b) wall {b_us:.2} us/dispatch [<= 3.00] · (a) wall {a_us:.2} us/dispatch [>= 13.00] -> {}",
                match (b_us <= 3.0, a_us >= 13.0) {
                    (true, true) => "both bars met: the graph pays",
                    (false, true) => "(b) above 3 us: read its gpu span and host residual before deciding",
                    (true, false) => "(a) below 13 us: today's road is not host-bound at this shape on this driver",
                    (false, false) => "neither bar met",
                }
            );
        }

        // The raw road computed the same answer (the last run wrote out_h).
        let got: Vec<f32> = f32::from_bytes(&client.read_one_unchecked(out_h.clone())).to_vec();
        assert_eq!(got.len(), reference.len());
        let mut bit_identical = true;
        let mut worst_raw = 0.0f32;
        for r in 0..n_out {
            if got[r].to_bits() != reference[r].to_bits() {
                bit_identical = false;
            }
            worst_raw = worst_raw.max((got[r] - reference[r]).abs() / reference[r].abs().max(1.0));
        }
        assert!(worst_raw <= 1e-5, "the raw road disagrees with today's road: rel {worst_raw}");
        eprintln!(
            "[spike] raw road output vs today's road: {} (worst rel {worst_raw:e}); today's vs host rel {worst:e}",
            if bit_identical { "bit-identical" } else { "within 1e-5" }
        );
    }

    /// The four projection shaders render for f32 EXACTLY as the audited
    /// files did — no `enable`, no f16, no unfilled placeholder — and for
    /// f16 with the directive on line one and the partials still f32.
    /// Narrowing the accumulator is the one thing §66 forbids outright.
    #[test]
    fn the_gemv_renderings_hold_the_storage_contract() {
        use burn_cubecl::template::KernelSource;
        let f32s = [
            DecodeGemvQ8(Storage::F32, Geometry::DEFAULT).source().complete(),
            DecodeGemvQ4K(Storage::F32, Geometry::DEFAULT).source().complete(),
            DecodeGemvQ5K(Storage::F32, Geometry::DEFAULT).source().complete(),
            DecodeGemvQ6K(Storage::F32, Geometry::DEFAULT).source().complete(),
        ];
        for (i, t) in f32s.iter().enumerate() {
            assert!(!t.contains("enable"), "gemv {i}: an f32 shader enables nothing");
            assert!(!t.contains("f16"), "gemv {i}: no f16 in the f32 rendering");
            assert!(!t.contains("{{"), "gemv {i}: a placeholder reached the compiler");
            assert!(t.contains("x: array<vec4<f32>>"), "gemv {i}: f32 bindings");
            assert!(t.contains("residual: array<f32>"), "gemv {i}: the residual binding follows the storage");
            assert!(t.contains("fuse: vec2<u32>"), "gemv {i}: the fold flag rides the scalars");
        }
        let f16s = [
            DecodeGemvQ8(Storage::F16, Geometry::DEFAULT).source().complete(),
            DecodeGemvQ4K(Storage::F16, Geometry::DEFAULT).source().complete(),
            DecodeGemvQ5K(Storage::F16, Geometry::DEFAULT).source().complete(),
            DecodeGemvQ6K(Storage::F16, Geometry::DEFAULT).source().complete(),
        ];
        for (i, t) in f16s.iter().enumerate() {
            assert!(t.starts_with("enable f16;"), "gemv {i}: the directive is line one");
            assert!(t.contains("x: array<vec4<f16>>"), "gemv {i}: f16 activations");
            assert!(t.contains("out: array<f16>"), "gemv {i}: f16 output");
            assert!(t.contains("residual: array<f16>"), "gemv {i}: an f16 out takes an f16 residual");
            assert!(
                t.contains("var<workgroup> partial: array<f32, 256>"),
                "gemv {i}: the partials stay f32 in every rendering"
            );
            assert!(!t.contains("{{"), "gemv {i}: a placeholder reached the compiler");
        }
    }

    /// The residual folded into the epilogue equals the separate add, row
    /// for row, per format. Same weights, same activation, same kernel;
    /// the ONLY difference is the flag, so any drift is the fold's.
    #[test]
    fn a_folded_residual_equals_the_separate_add_per_format() {
        if crate::skip_no_gpu() {
            return;
        }
        use cubecl::prelude::*;
        let device = cubecl::wgpu::WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        let (n_out, k) = (211usize, 1024usize);
        let synth = |format: &str, block_bytes: usize, scale_offsets: &[usize]| -> Vec<u8> {
            let blocks = match format {
                "q8_0" => n_out * k / 32,
                _ => n_out * k / 256,
            };
            let mut data: Vec<u8> = (0..blocks * block_bytes)
                .map(|i| ((i * 37 + 11) % 251) as u8)
                .collect();
            for b in 0..blocks {
                for (which, &off) in scale_offsets.iter().enumerate() {
                    let bits: u16 = if which == 0 { 0x3C00 } else { 0x3400 };
                    data[b * block_bytes + off..b * block_bytes + off + 2]
                        .copy_from_slice(&bits.to_le_bytes());
                }
            }
            data
        };
        let x: Vec<f32> = (0..k).map(|i| (((i as f32 * 0.311).sin() * 1.4) - 0.1) / 32.0).collect();
        let res: Vec<f32> = (0..n_out).map(|r| (r as f32 * 0.73).cos() * 5.0).collect();
        for format in ["q8_0", "q4_k", "q6_k"] {
            let x_h = client.create_from_slice(f32::as_bytes(&x));
            let x_h2 = client.create_from_slice(f32::as_bytes(&x));
            let r_h = client.create_from_slice(f32::as_bytes(&res));
            let (plain, folded) = match format {
                "q8_0" => {
                    let w = crate::qmatmul::Q80Weight::<WgpuRuntime>::from_gguf_bytes(&client, &synth(format, 34, &[0]), n_out, k).unwrap();
                    (w.decode_gemv_wgsl(&client, x_h, Storage::F32, None), w.decode_gemv_wgsl(&client, x_h2, Storage::F32, Some(r_h)))
                }
                "q4_k" => {
                    let w = crate::qmatmul::Q4KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &synth(format, 144, &[0, 2]), n_out, k).unwrap();
                    (w.decode_gemv_wgsl(&client, x_h, Storage::F32, None), w.decode_gemv_wgsl(&client, x_h2, Storage::F32, Some(r_h)))
                }
                _ => {
                    let w = crate::qmatmul::Q6KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &synth(format, 210, &[208]), n_out, k).unwrap();
                    (w.decode_gemv_wgsl(&client, x_h, Storage::F32, None), w.decode_gemv_wgsl(&client, x_h2, Storage::F32, Some(r_h)))
                }
            };
            let bytes = cubecl::future::block_on(client.read_async(vec![plain, folded])).unwrap();
            let a = f32::from_bytes(&bytes[0]);
            let b = f32::from_bytes(&bytes[1]);
            for r in 0..n_out {
                let want = a[r] + res[r];
                assert!(
                    (b[r] - want).abs() <= 1e-5 * want.abs().max(1.0),
                    "{format} fold wrong at row {r}: folded {} vs separate {} + residual {}",
                    b[r], a[r], res[r]
                );
            }
        }
    }

    /// The f16-STORAGE gemv per format, against the host dequant
    /// reference on an activation ROUNDED to f16 first — so operand
    /// rounding is not scored as kernel error. The dot is f32 over f16
    /// operands with one f16 store rounding at the end: rel 1e-3, where a
    /// real mistake lands near 1e0. Skips by name where the adapter does
    /// not admit f16.
    #[test]
    fn f16_storage_gemv_matches_the_host_on_rounded_input() {
        if crate::skip_no_gpu() {
            return;
        }
        use cubecl::prelude::*;
        let device = cubecl::wgpu::WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        if !f16_storage_enabled_for_test(&client) {
            eprintln!("skipped: this adapter does not admit f16 in a shader");
            return;
        }
        let (n_out, k) = (211usize, 1024usize);
        let synth = |format: &str, block_bytes: usize, scale_offsets: &[usize]| -> Vec<u8> {
            let blocks = match format {
                "q8_0" => n_out * k / 32,
                _ => n_out * k / 256,
            };
            let mut data: Vec<u8> = (0..blocks * block_bytes)
                .map(|i| ((i * 37 + 11) % 251) as u8)
                .collect();
            for b in 0..blocks {
                for (which, &off) in scale_offsets.iter().enumerate() {
                    let bits: u16 = if which == 0 { 0x3C00 } else { 0x3400 };
                    data[b * block_bytes + off..b * block_bytes + off + 2]
                        .copy_from_slice(&bits.to_le_bytes());
                }
            }
            data
        };
        // The activation, rounded the way the f16 binding will hold it —
        // and SCALED DOWN, because the synthetic weights (unit scales,
        // 6-bit values, k = 1024) drive the f32 canary's dot products to
        // ~80,000, past f16's 65,504. The first run of this test stored
        // infinity there and was RIGHT to: that is what an f16 store does.
        // It is not a kernel defect — a real model already lives within
        // f16 on this backend, so an activation that large would already
        // have broken before any projection — it is a fixture that only
        // f32 could hold. The scale keeps the reference honest and the
        // outputs inside the format the kernel is being asked to write.
        let x: Vec<f32> = (0..k)
            .map(|i| f32::from(burn::tensor::f16::from_f32((((i as f32 * 0.311).sin() * 1.4) - 0.1) / 32.0)))
            .collect();
        let x_bytes: Vec<u8> = x
            .iter()
            .flat_map(|v| burn::tensor::f16::from_f32(*v).to_le_bytes())
            .collect();
        for format in ["q8_0", "q4_k", "q5_k", "q6_k"] {
            let x_h = client.create_from_slice(&x_bytes);
            let (dense, out_h) = match format {
                "q8_0" => {
                    let data = synth(format, 34, &[0]);
                    let w = crate::qmatmul::Q80Weight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k).unwrap();
                    (combs_formats::quants::dequantize_q8_0(&data, n_out * k).unwrap(), w.decode_gemv_wgsl(&client, x_h, Storage::F16, None))
                }
                "q4_k" => {
                    let data = synth(format, 144, &[0, 2]);
                    let w = crate::qmatmul::Q4KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k).unwrap();
                    (combs_formats::quants::dequantize_q4_k(&data, n_out * k).unwrap(), w.decode_gemv_wgsl(&client, x_h, Storage::F16, None))
                }
                "q5_k" => {
                    let data = synth(format, 176, &[0, 2]);
                    let w = crate::qmatmul::Q5KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k).unwrap();
                    (combs_formats::quants::dequantize_q5_k(&data, n_out * k).unwrap(), w.decode_gemv_wgsl(&client, x_h, Storage::F16, None))
                }
                _ => {
                    let data = synth(format, 210, &[208]);
                    let w = crate::qmatmul::Q6KWeight::<WgpuRuntime>::from_gguf_bytes(&client, &data, n_out, k).unwrap();
                    (combs_formats::quants::dequantize_q6_k(&data, n_out * k).unwrap(), w.decode_gemv_wgsl(&client, x_h, Storage::F16, None))
                }
            };
            let bytes = cubecl::future::block_on(client.read_async(vec![out_h])).unwrap();
            let got: Vec<f32> = bytes[0]
                .chunks_exact(2)
                .map(|c| f32::from(burn::tensor::f16::from_le_bytes([c[0], c[1]])))
                .collect();
            assert_eq!(got.len(), n_out, "{format}: the f16 output has one row per output");
            for r in 0..n_out {
                let expect: f32 = (0..k).map(|c| x[c] * dense[r * k + c]).sum();
                let g = got[r];
                assert!(
                    g.is_finite() && (g - expect).abs() <= 1e-3 * expect.abs().max(1.0),
                    "{format} f16 gemv wrong at row {r}: got {g}, expected {expect}"
                );
            }
        }
    }

    #[test]
    fn probe_report_passes_natively() {
        if crate::skip_no_gpu() {
            return;
        }
        cubecl::future::block_on(probe_report()).expect("all canaries green on naga");
    }

    /// The host->kernel scalar layout, proven rather than assumed: u64
    /// slots as vec2<u32>, low word first, f32 via bit round-trip.
    #[test]
    fn scalar_layout_is_what_every_kernel_assumes() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        let client = WgpuRuntime::client(&device);

        let out = client.empty(5 * core::mem::size_of::<u32>());
        launch(
            &client,
            ProbeEcho,
            CubeCount::Static(1, 1, 1),
            vec![out.clone().binding()],
            vec![
                1,
                0xAABB_CCDD_0011_2233,
                7,
                f32::to_bits(1.5) as u64,
            ],
        );
        let bytes = client.read_one_unchecked(out);
        let words = u32::from_bytes(&bytes);
        assert_eq!(words[0], 1, "slot 0 low word");
        assert_eq!(words[1], 0, "slot 0 high word");
        assert_eq!(words[2], 0x0011_2233, "slot 1 low word");
        assert_eq!(words[3], 7, "slot 2 low word");
        assert_eq!(
            f32::from_bits(words[4]),
            3.0,
            "f32 must survive the bits round-trip"
        );
    }

    /// Workgroup memory and barriers produce correct cross-thread values,
    /// including in a ragged final window. Wrong values here mean staged
    /// kernels cannot be trusted at all on this backend.
    #[test]
    fn shared_memory_mirror_survives_a_ragged_tail() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = Default::default();
        let client = WgpuRuntime::client(&device);

        // 300 elements: one full workgroup and one ragged 44-lane tail.
        let n = 300usize;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let in_h = client.create_from_slice(f32::as_bytes(&input));
        let out_h = client.empty(n * core::mem::size_of::<f32>());

        launch(
            &client,
            ProbeSmem,
            CubeCount::Static(2, 1, 1),
            vec![in_h.binding(), out_h.clone().binding()],
            vec![n as u64],
        );

        let bytes = client.read_one_unchecked(out_h);
        let got = f32::from_bytes(&bytes);
        for i in 0..n {
            let base = (i / 256) * 256;
            let lane = i % 256;
            let mirrored = base + (255 - lane);
            let staged = if mirrored < n { mirrored as f32 } else { 0.0 };
            assert_eq!(
                got[i],
                staged + 1.0,
                "lane {i}: expected the mirror of its window"
            );
        }
    }
}
