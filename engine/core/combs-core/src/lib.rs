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
pub mod quant;

use burn::backend::wgpu::{RuntimeOptions, WgpuDevice, WgpuSetup, graphics::AutoGraphicsApi};

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

/// Returns the default wgpu device (best available GPU; on macOS this is the
/// Metal device). Honors cubecl's `CUBECL_WGPU_DEFAULT_DEVICE` override.
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

/// Queries the adapter for its limits/features and returns [`DeviceCaps`].
///
/// Like [`device_info`], this performs the cubecl runtime setup for the
/// device, so it is safe (and cheap) to use the device afterwards.
#[cfg(not(target_family = "wasm"))]
pub fn device_caps(device: &CombsDevice) -> DeviceCaps {
    caps_from_setup(&burn::backend::wgpu::init_setup::<AutoGraphicsApi>(
        device,
        RuntimeOptions::default(),
    ))
}

/// [`device_caps`] for callers that can await — the only form available in
/// a browser, and identical in result natively.
pub async fn device_caps_async(device: &CombsDevice) -> DeviceCaps {
    caps_from_setup(
        &burn::backend::wgpu::init_setup_async::<AutoGraphicsApi>(
            device,
            RuntimeOptions::default(),
        )
        .await,
    )
}

/// Reads the capability set off an already-initialized setup. Both
/// `device_caps` forms share this so the two can never drift.
fn caps_from_setup(setup: &WgpuSetup) -> DeviceCaps {
    let info = setup.adapter.get_info();
    let limits = setup.adapter.limits();
    let features = setup.adapter.features();
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

/// Initializes the wgpu runtime for `device` and returns adapter information.
///
/// Note: this performs the cubecl runtime setup for the device (the same setup
/// burn performs lazily on first tensor use), so it is safe to use the device
/// for compute afterwards.
/// NOTE: like [`device_caps`], this primes the cubecl runtime — only
/// ONE such probe may run per process (a second `init_setup` panics in
/// cubecl 0.10). Prefer `device_caps`, which carries a superset.
#[cfg(not(target_family = "wasm"))]
pub fn device_info(device: &CombsDevice) -> DeviceInfo {
    info_from_setup(&burn::backend::wgpu::init_setup::<AutoGraphicsApi>(
        device,
        RuntimeOptions::default(),
    ))
}

/// [`device_info`] for callers that can await.
pub async fn device_info_async(device: &CombsDevice) -> DeviceInfo {
    info_from_setup(
        &burn::backend::wgpu::init_setup_async::<AutoGraphicsApi>(
            device,
            RuntimeOptions::default(),
        )
        .await,
    )
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

/// Facade over the GPU buffer pool.
///
/// # Phase 1 status: documented no-op
///
/// The plan's hand-rolled slab/coalescing pool was deliberately replaced by
/// cubecl's built-in allocator, which already does pooled slab allocation and
/// reuse (configured through [`burn::backend::wgpu::MemoryConfiguration`]).
/// burn 0.21 does not publicly expose cubecl 0.10's
/// `ComputeClient::memory_cleanup`, and `memory_persistent_allocations` does
/// not exist in cubecl 0.10's public API at all, so this facade currently does
/// nothing. It exists so that the runtime can call `pool.cleanup()` /
/// `pool.pin_persistent()` today and Phase 2 can back those calls with real
/// handles (persistent KV/weight arenas) without changing call sites.
#[derive(Debug, Default, Clone, Copy)]
pub struct BufferPool;

impl BufferPool {
    /// Creates a new pool facade.
    pub fn new() -> Self {
        BufferPool
    }

    /// Pin long-lived allocations (weights, KV arena) so the pool never
    /// releases them. No-op in Phase 1 — cubecl's pooled allocator keeps
    /// freed blocks for reuse anyway.
    pub fn pin_persistent(&self) {
        // no-op: see type-level docs.
    }

    /// Release cached free blocks back to the driver. No-op in Phase 1 —
    /// cubecl 0.10's `memory_cleanup` is not reachable through burn's public
    /// API.
    pub fn cleanup(&self) {
        // no-op: see type-level docs.
    }
}
