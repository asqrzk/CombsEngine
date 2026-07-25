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

/// The default inference backend: autotuned, fusing wgpu/CubeCL backend.
///
/// With burn 0.21 + the `fusion` feature this expands to
/// `Fusion<CubeBackend<WgpuRuntime, f32, i32, u32>>`.
pub type CombsBackend = burn::backend::Wgpu<f32, i32, u32>;

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
