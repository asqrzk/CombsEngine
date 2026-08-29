//! # combs-core
//!
//! Backend type aliases, device helpers, and the memory-pool facade for the
//! Combs Engine L0 Rust core.
//!
//! Phase 1 runs entirely on the wgpu backend (Metal on Apple Silicon) with
//! f32 compute. f16 compute and custom CubeCL kernels are later-phase
//! optimizations; the aliases here are the single place to change.

pub use burn::backend::wgpu;

pub mod progress;
pub mod provenance;
pub mod quant;
pub mod timefmt;

use burn::backend::wgpu::{
    MemoryConfiguration, RuntimeOptions, WgpuDevice, WgpuSetup, graphics::AutoGraphicsApi,
};

// `AutoGraphicsApi` already resolves to `BrowserWebGpu` under
// `target_family = "wasm"`, so the graphics API needs no cfg of ours —
// only the *initialization* differs: cubecl's sync `init_setup` panics on
// wasm by construction ("Creating a wgpu setup synchronously is
// unsupported"), and `init_setup_async` is the form that works on every
// target. Every probe below therefore exists twice: the sync original,
// native-only and unchanged for the CLI/FFI/serve callers, and an `_async`
// twin compiled everywhere, which is what the browser bindings call.

/// The default inference backend: autotuned, fusing wgpu/CubeCL backend
/// (`Fusion<CubeBackend<WgpuRuntime, f32, i32, u32>>`).
///
/// `--features f16` switches to an **unfused f16** backend, which ~halves
/// weight + KV + activation memory (e.g. a 3B model drops from ~12 GB to
/// ~6 GB) and is typically faster. The numerically sensitive reductions
/// (RMS/LayerNorm, attention scores + softmax, gelu) run in f32 regardless
/// of backend (see `combs-models::precision`), so f16 output stays coherent.
///
/// Note: f16 uses the **unfused** `CubeBackend` type directly — burn-fusion
/// 0.21 panics on reduced-precision tensors, so we bypass the fusion layer
/// for f16 while keeping f32 fused. bf16 is unavailable (cubecl's matmul has
/// no bf16 path on Metal/wgpu).
#[cfg(not(feature = "f16"))]
pub type CombsBackend = burn::backend::Wgpu<f32, i32, u32>;

/// Always-f32 backend on the same wgpu runtime. The diffusion pipeline is
/// pinned to it in every build: SD-1.5's UNet/VAE collapse to black output
/// under f16 (range, not rounding), so image generation computes in f32
/// regardless of the text stack's dtype.
pub type CombsBackendF32 = burn::backend::Wgpu<f32, i32, u32>;
#[cfg(feature = "f16")]
pub type CombsBackend = burn::backend::wgpu::CubeBackend<
    burn::backend::wgpu::WgpuRuntime,
    burn::tensor::f16,
    i32,
    u32,
>;

/// The default device handle type.
pub type CombsDevice = WgpuDevice;

/// The one device context this process holds: the device handle, the
/// retained [`WgpuSetup`] (adapter, device, queue — the doorway for any
/// pass that must speak wgpu directly), and the capabilities read at
/// initialization. cubecl permits exactly one setup per device; retaining
/// it here is what lets `device_caps`/`device_info` be called freely
/// where a second `init_setup` used to panic.
struct DeviceContext {
    /// The retained setup — `Some` when this context performed the cubecl
    /// initialization itself (the doorway to the raw device/queue for any
    /// future foreign pass). `None` when something else initialized the
    /// runtime lazily first; capabilities are then read from a fresh
    /// adapter query, which answers everything the planner asks without
    /// touching cubecl's registration.
    setup: Option<WgpuSetup>,
    caps: DeviceCaps,
}

static CONTEXT: std::sync::OnceLock<DeviceContext> = std::sync::OnceLock::new();

/// Runtime options for the engine's device: the seam where memory-pool
/// policy is chosen. Native defaults to exclusive pages — the footprint
/// proof measured it at identical decode speed with reserved tracking
/// live bytes almost exactly (721 vs 1862 MB after a qwen3-0.6b load
/// under sub-slices; MEASUREMENTS §37). `COMBS_MEM_POOLS=subslices`
/// restores cubecl's sub-slice pools; `=exclusive` names the default
/// explicitly. The wasm build has no environment and takes cubecl's
/// wasm default, which is already exclusive pages.
fn combs_runtime_options() -> RuntimeOptions {
    #[cfg(not(target_family = "wasm"))]
    if matches!(
        std::env::var("COMBS_MEM_POOLS").as_deref(),
        Ok("subslices")
    ) {
        return RuntimeOptions::default();
    }
    RuntimeOptions {
        memory_config: MemoryConfiguration::ExclusivePages,
        ..RuntimeOptions::default()
    }
}

#[cfg(not(target_family = "wasm"))]
fn context() -> &'static DeviceContext {
    CONTEXT.get_or_init(|| {
        let device = WgpuDevice::default();
        // cubecl accepts exactly one initialization per device and offers
        // no way to ask whether one happened; a process that touched a
        // tensor before asking for capabilities has already initialized
        // lazily, and a second setup panics. Catch that one case and
        // answer from a plain adapter query instead — same numbers, no
        // retained setup.
        match std::panic::catch_unwind(|| {
            burn::backend::wgpu::init_setup::<AutoGraphicsApi>(
                &device,
                combs_runtime_options(),
            )
        }) {
            Ok(setup) => {
                let caps = caps_from_setup(&setup);
                DeviceContext {
                    setup: Some(setup),
                    caps,
                }
            }
            Err(_) => {
                let caps = cubecl::future::block_on(caps_from_fresh_adapter());
                DeviceContext {
                    setup: None,
                    caps,
                }
            }
        }
    })
}

/// Capability query with no cubecl involvement: a fresh instance and
/// adapter answer the planner's questions identically — limits, features
/// and subgroup sizes are adapter properties, not runtime state.
async fn caps_from_fresh_adapter() -> DeviceCaps {
    let instance = ::wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&::wgpu::RequestAdapterOptions {
            power_preference: ::wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("an adapter exists: the runtime already initialized on one");
    let info = adapter.get_info();
    let limits = adapter.limits();
    let features = adapter.features();
    DeviceCaps {
        name: info.name.clone(),
        backend: format!("{:?}", info.backend),
        device_type: format!("{:?}", info.device_type),
        driver: format!("{} ({})", info.driver, info.driver_info),
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size as u64,
        max_buffer_size: limits.max_buffer_size,
        max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
        max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
        features: format!("{:?}", features),
        subgroup_min_size: info.subgroup_min_size,
        subgroup_max_size: info.subgroup_max_size,
    }
}

/// [`context`] for callers that can await — the only form a browser has.
async fn context_async() -> &'static DeviceContext {
    if let Some(ctx) = CONTEXT.get() {
        return ctx;
    }
    let device = WgpuDevice::default();
    let setup = burn::backend::wgpu::init_setup_async::<AutoGraphicsApi>(
        &device,
        combs_runtime_options(),
    )
    .await;
    let caps = caps_from_setup(&setup);
    // A racing initializer on native would mean two setups were built;
    // only the first is kept, and cubecl treats the duplicate as the same
    // registered device. On wasm there is one thread and no race.
    let _ = CONTEXT.set(DeviceContext {
        setup: Some(setup),
        caps,
    });
    CONTEXT.get().expect("just set")
}

/// Returns the default wgpu device (best available GPU; on macOS this is
/// the Metal device). Honors cubecl's `CUBECL_WGPU_DEFAULT_DEVICE`
/// override. Deliberately lazy: the runtime initializes on first use (or
/// on the first capability query, which retains the setup — see
/// [`DeviceContext`]), so call order never becomes a correctness rule.
pub fn init_device() -> CombsDevice {
    WgpuDevice::default()
}

/// True when wgpu can see at least one adapter. Cached after the first
/// probe. Initializing a cubecl device on an adapterless machine (e.g. a
/// CI runner) panics in a worker thread, so GPU-dependent tests check
/// this first and skip rather than fail.
///
/// Native only — see [`gpu_available_async`] for the browser.
#[cfg(not(target_family = "wasm"))]
pub fn gpu_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // `::wgpu` is the raw crate — the bare name resolves to this
    // module's `burn::backend::wgpu` re-export.
    *AVAILABLE.get_or_init(|| {
        let instance = ::wgpu::Instance::default();
        let adapters =
            cubecl::future::block_on(instance.enumerate_adapters(::wgpu::Backends::all()));
        !adapters.is_empty()
    })
}

/// True when wgpu can reach an adapter, asked the way every platform can
/// answer it: `request_adapter`. The browser reports no adapters through
/// `enumerate_adapters` by construction, so the sync probe above is not
/// merely blocked there, it is wrong there; this one is correct on both.
pub async fn gpu_available_async() -> bool {
    let instance = ::wgpu::Instance::default();
    instance
        .request_adapter(&::wgpu::RequestAdapterOptions::default())
        .await
        .is_ok()
}

/// Basic information about a wgpu adapter.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Human-readable adapter name (e.g. "Apple M3 Pro").
    pub name: String,
    /// Graphics backend in use (e.g. "Metal").
    pub backend: String,
    /// Device type (e.g. "IntegratedGpu").
    pub device_type: String,
    /// Driver name + info string.
    pub driver: String,
}

/// Hardware capabilities consumed by the application-layer device planner
/// (sharding, KV budget, prefill chunk sizing). Serialized to JSON across
/// the FFI boundary.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DeviceCaps {
    /// Human-readable adapter name (e.g. "Apple M3 Pro").
    pub name: String,
    /// Graphics backend in use ("Metal", "Vulkan", "Dx12", "Gl", ...).
    pub backend: String,
    /// Device type ("IntegratedGpu", "DiscreteGpu", ...).
    pub device_type: String,
    /// Driver name + info string.
    pub driver: String,
    /// `max_storage_buffer_binding_size`: the hard cap on a single GPU
    /// buffer — the sharding limit on mobile devices.
    pub max_storage_buffer_binding_size: u64,
    /// `max_buffer_size`: the largest single allocation the driver allows.
    pub max_buffer_size: u64,
    /// Largest compute workgroup dimension.
    pub max_compute_workgroup_size_x: u32,
    /// `max_compute_invocations_per_workgroup`.
    pub max_compute_invocations_per_workgroup: u32,
    /// Debug dump of the adapter's enabled feature set (wgpu 29 no longer
    /// exposes WebGPU extension features like `SHADER_F16` through the
    /// public adapter API, so we surface the raw list for the planner).
    pub features: String,
    /// Smallest subgroup ("plane") width the adapter reports.
    ///
    /// Reported, not assumed, because it is a place the browser and the
    /// desktop genuinely disagree: WebGPU exposes no subgroup information
    /// today, so an adapter there reports zero — and cubecl substitutes a
    /// nominal 8/128 range, which some reduction kernels then size
    /// themselves against. A model whose reductions are correct on Metal
    /// and wrong in a tab is a model to check this number for first.
    pub subgroup_min_size: u32,
    /// Largest subgroup width the adapter reports. See
    /// [`DeviceCaps::subgroup_min_size`].
    pub subgroup_max_size: u32,
}

/// The adapter's limits and features, read once at setup and answered
/// from the retained context thereafter — callable as often as anyone
/// likes, where it used to panic on the second call.
#[cfg(not(target_family = "wasm"))]
pub fn device_caps(device: &CombsDevice) -> DeviceCaps {
    let _ = device; // one device per process; the context holds it
    context().caps.clone()
}

/// [`device_caps`] for callers that can await — the only form available in
/// a browser, and identical in result natively.
pub async fn device_caps_async(device: &CombsDevice) -> DeviceCaps {
    let _ = device;
    context_async().await.caps.clone()
}

/// Reads the capability set off an already-initialized setup. Both
/// `device_caps` forms share this so the two can never drift.
fn caps_from_setup(setup: &WgpuSetup) -> DeviceCaps {
    let info = setup.adapter.get_info();
    let limits = setup.adapter.limits();
    let features = setup.adapter.features();
    // Which adapter was actually selected — the first question asked
    // whenever output is wrong or slow, and the one a container can
    // answer differently from the host it claims to be.
    crate::provenance::event(
        "device",
        "device.select",
        &[
            ("name", info.name.clone()),
            ("type", format!("{:?}", info.device_type)),
            ("backend", format!("{:?}", info.backend)),
            ("driver", format!("{} ({})", info.driver, info.driver_info)),
            ("max_buffer_mb", (limits.max_buffer_size >> 20).to_string()),
            (
                "max_binding_mb",
                ((limits.max_storage_buffer_binding_size as u64) >> 20).to_string(),
            ),
        ],
    );
    DeviceCaps {
        name: info.name.clone(),
        backend: format!("{:?}", info.backend),
        device_type: format!("{:?}", info.device_type),
        driver: format!("{} ({})", info.driver, info.driver_info),
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size as u64,
        max_buffer_size: limits.max_buffer_size,
        max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
        max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
        features: format!("{:?}", features),
        subgroup_min_size: info.subgroup_min_size,
        subgroup_max_size: info.subgroup_max_size,
    }
}

/// GPU allocator state from cubecl's memory manager (authoritative — process
/// RSS is meaningless for unified-memory GPU accounting).
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct GpuMemory {
    /// Bytes referenced by live handles.
    pub bytes_in_use: u64,
    /// Bytes reserved by the pool (in-use + cached slabs).
    pub bytes_reserved: u64,
    /// Bytes lost to alignment padding.
    pub bytes_padding: u64,
    /// Live allocation count.
    pub number_allocs: u64,
}

/// Samples the GPU allocator. `memory_usage()` is `submit_blocking` on the
/// compute stream — call from the engine worker between generations (or
/// rate-limited), not from request threads during a long prefill.
#[cfg(not(target_family = "wasm"))]
pub fn gpu_memory(device: &CombsDevice) -> Option<GpuMemory> {
    let client =
        <burn::backend::wgpu::WgpuRuntime as cubecl::prelude::Runtime>::client(device);
    client.memory_usage().ok().map(|m| GpuMemory {
        bytes_in_use: m.bytes_in_use,
        bytes_reserved: m.bytes_reserved,
        bytes_padding: m.bytes_padding,
        number_allocs: m.number_allocs,
    })
}

/// Browser build: `memory_usage()` is a blocking submit on the compute
/// stream, which a single-threaded wasm page cannot perform. Reporting
/// `None` — "not measured" — keeps every call site source-identical and
/// keeps the number honest; when the platform grows a non-blocking
/// allocator query, only this arm changes.
#[cfg(target_family = "wasm")]
pub fn gpu_memory(_device: &CombsDevice) -> Option<GpuMemory> {
    None
}

/// Adapter identity, answered from the retained context.
#[cfg(not(target_family = "wasm"))]
pub fn device_info(device: &CombsDevice) -> DeviceInfo {
    let _ = device;
    let ctx = context();
    match &ctx.setup {
        Some(setup) => info_from_setup(setup),
        None => DeviceInfo {
            name: ctx.caps.name.clone(),
            backend: ctx.caps.backend.clone(),
            device_type: ctx.caps.device_type.clone(),
            driver: ctx.caps.driver.clone(),
        },
    }
}

/// [`device_info`] for callers that can await.
pub async fn device_info_async(device: &CombsDevice) -> DeviceInfo {
    let _ = device;
    let ctx = context_async().await;
    match &ctx.setup {
        Some(setup) => info_from_setup(setup),
        None => DeviceInfo {
            name: ctx.caps.name.clone(),
            backend: ctx.caps.backend.clone(),
            device_type: ctx.caps.device_type.clone(),
            driver: ctx.caps.driver.clone(),
        },
    }
}

/// Reads adapter identity off an already-initialized setup.
fn info_from_setup(setup: &WgpuSetup) -> DeviceInfo {
    let info = setup.adapter.get_info();
    DeviceInfo {
        name: info.name.clone(),
        backend: format!("{:?}", info.backend),
        device_type: format!("{:?}", info.device_type),
        driver: format!("{} ({})", info.driver, info.driver_info),
    }
}

/// Facade over the GPU buffer pool, backed by cubecl's allocator.
///
/// The pool itself is cubecl's (slab allocation, reuse, configured through
/// [`burn::backend::wgpu::MemoryConfiguration`]); this facade is the
/// engine's two levers over it. Without [`BufferPool::cleanup`] the pool
/// NEVER returns freed blocks to the driver — the failure mode behind two
/// machine crashes — so the runtime calls it at the moments memory
/// genuinely becomes garbage: after load transients, after session
/// eviction.
#[derive(Debug, Default, Clone, Copy)]
pub struct BufferPool;

impl BufferPool {
    /// Creates a new pool facade.
    pub fn new() -> Self {
        BufferPool
    }

    /// Runs `task` with the pool in persistent-allocation mode, so every
    /// buffer it creates is exempt from later [`BufferPool::cleanup`]
    /// calls — meant for weight loading. On by default natively: the
    /// footprint proof showed reserved landing exactly on live bytes
    /// with no transient pinned (712.7 = 712.7 MB vs 721.5 doored off;
    /// MEASUREMENTS §39). `COMBS_PERSISTENT_LOAD=0` opts out. The wasm
    /// build keeps the plain path until the mode is proven in a browser.
    pub fn pin_persistent<R>(
        &self,
        device: &CombsDevice,
        task: impl FnOnce() -> R + Send,
    ) -> R
    where
        R: Send,
    {
        let door = {
            #[cfg(not(target_family = "wasm"))]
            {
                !matches!(std::env::var("COMBS_PERSISTENT_LOAD").as_deref(), Ok("0"))
            }
            #[cfg(target_family = "wasm")]
            {
                false
            }
        };
        if !door {
            return task();
        }
        let client =
            <burn::backend::wgpu::WgpuRuntime as cubecl::prelude::Runtime>::client(device);
        match client.memory_persistent_allocation((), |_| task()) {
            Ok(out) => out,
            Err(e) => panic!("persistent-mode load failed to submit: {e:?}"),
        }
    }

    /// Asks the pool to release free blocks back to the driver — a plain
    /// submit, safe on every target. The allocator decides what is
    /// actually beneficial to free; persistent allocations are exempt.
    pub fn cleanup(&self, device: &CombsDevice) {
        let client =
            <burn::backend::wgpu::WgpuRuntime as cubecl::prelude::Runtime>::client(device);
        client.memory_cleanup();
    }
}
