//! # combs-core
//!
//! Backend type aliases, device helpers, and the memory-pool facade for the
//! Combs Engine L0 Rust core.
//!
//! Phase 1 runs entirely on the wgpu backend (Metal on Apple Silicon) with
//! f32 compute. f16 compute and custom CubeCL kernels are later-phase
//! optimizations; the aliases here are the single place to change.

pub use burn::backend::wgpu;

pub mod quant;

use burn::backend::wgpu::{RuntimeOptions, WgpuDevice, WgpuSetup, graphics::AutoGraphicsApi};

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
}

/// Queries the adapter for its limits/features and returns [`DeviceCaps`].
///
/// Like [`device_info`], this performs the cubecl runtime setup for the
/// device, so it is safe (and cheap) to use the device afterwards.
pub fn device_caps(device: &CombsDevice) -> DeviceCaps {
    let setup: WgpuSetup =
        burn::backend::wgpu::init_setup::<AutoGraphicsApi>(device, RuntimeOptions::default());
    let info = setup.adapter.get_info();
    let limits = setup.adapter.limits();
    let features = setup.adapter.features();
    DeviceCaps {
        name: info.name,
        backend: format!("{:?}", info.backend),
        device_type: format!("{:?}", info.device_type),
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size as u64,
        max_buffer_size: limits.max_buffer_size,
        max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
        max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
        features: format!("{:?}", features),
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

/// Initializes the wgpu runtime for `device` and returns adapter information.
///
/// Note: this performs the cubecl runtime setup for the device (the same setup
/// burn performs lazily on first tensor use), so it is safe to use the device
/// for compute afterwards.
pub fn device_info(device: &CombsDevice) -> DeviceInfo {
    let setup: WgpuSetup =
        burn::backend::wgpu::init_setup::<AutoGraphicsApi>(device, RuntimeOptions::default());
    let info = setup.adapter.get_info();
    DeviceInfo {
        name: info.name,
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
