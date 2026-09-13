//! # combs-models
//!
//! Model architecture registry. The runtime drives any architecture through
//! the fixed [`GenerativeModel`] contract (MLC's `embed/prefill/decode/
//! create_kv_cache` function set); architectures register themselves in the
//! [`ModelRegistry`]. Phase 1 ships the Llama family (incl. SmolLM2).

mod act;
mod archspec;
mod embed;
mod forward_probe;
mod kv;
mod llama;
mod qwen35;
mod matmul;
mod norm;
mod precision;
mod qkernel;
mod qlinear;
mod qmatmul;
mod quant_linear;
mod registry;
mod wgsl;
mod rope;
mod smolvlm;
pub mod staged;
mod traits;
mod whisper;

pub use archspec::{ArchSpec, LayerKind, NormFlavor};
pub use kv::{
    CacheConfig, CacheKind, ContiguousKVCache, DeltaState, HybridCache, KVCache, PageStats,
    PagedKVCache,
};
pub use llama::LlamaModel;
pub use qwen35::Qwen35Model;
pub use norm::rms_norm;
pub use staged::flush_device;
pub use qlinear::{batched_matmul_summary, quant_census, Linear, QuantLinearOp, try_quant_linear};
pub use qmatmul::{
    Q4KWeight, Q5KWeight, Q6KWeight, Q40Weight, Q50Weight, Q80Weight, dequantize_q4_0_gpu,
    dequantize_q4_k_gpu, dequantize_q5_0_gpu, dequantize_q5_k_gpu, dequantize_q6_k_gpu,
    dequantize_q8_0_gpu,
    repack_q4_0, repack_q4_k, repack_q5_0, repack_q6_k, repack_q8_0,
};
pub use quant_linear::QuantizedLinear;
pub use registry::ModelRegistry;
pub use forward_probe::{batched_probe_report, forward_probe_report};
pub use wgsl::probe_report as wgsl_probe_report;
// The kit's micro cells: the launch floor, the bus, and the gemv against
// its packed bytes. Public for the same reason the probe is — a stranger's
// box must be able to answer them.
pub use wgsl::{active_doors, bus_cell, dispatch_latency, gemv_cell, gemv_cell_at};
pub use rope::RotaryEmbedding;
pub use smolvlm::{SmolVlmModel, image_prompt_expansion, pixels_to_tensor};
pub use whisper::{WhisperModel, load_speech_model};
pub use traits::{GenerativeModel, SpeechToTextModel};

/// Errors produced while constructing or running models.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// A format-adapter error.
    #[error(transparent)]
    Format(#[from] combs_formats::FormatError),

    /// No registered architecture matches the source metadata.
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    /// Media input (image/audio) was passed to a model that cannot take it.
    #[error("unsupported media input: {0}")]
    UnsupportedMedia(String),

    /// The model does not implement an optional capability (e.g. hidden
    /// states for embeddings).
    #[error("unsupported operation: {0}")]
    Unsupported(String),

    /// A required weight tensor is missing from the source.
    #[error("missing weight tensor: {0}")]
    MissingTensor(String),

    /// A weight tensor has an unexpected shape.
    #[error("bad shape for {tensor}: expected {expected:?}, got {got:?}")]
    BadShape {
        /// Tensor name.
        tensor: String,
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        got: Vec<usize>,
    },
}

/// Convenient result alias for this crate.
pub type Result<T> = std::result::Result<T, ModelError>;

/// Test-only guard: true (after logging) when the machine has no wgpu
/// adapter, so GPU-dependent tests can skip instead of panicking inside
/// cubecl's device worker. Real GPU coverage is unaffected wherever an
/// adapter exists — CI's macOS runners included.
#[cfg(test)]
pub(crate) fn skip_no_gpu() -> bool {
    if combs_core::gpu_available() {
        pin_test_device();
        return false;
    }
    eprintln!("skipped: no wgpu adapter on this machine");
    true
}

/// Test-only: pin the shared test device's default dtypes to F32 before any
/// tensor exists. burn 0.21 keeps ONE default dtype per device type for the
/// whole process (keyed by the device id and `WgpuDevice`'s type, write-once),
/// shared by every CubeBackend float width, so whichever backend resolves it
/// first decides it for every test after: an f16-backend test that ran first
/// turned the f32 tests' tensors f16 (19 failures in one gate run, a clean
/// pass in the run before). Every GPU test pins through `skip_no_gpu`; a test
/// that means another width names it at creation. Production-faithful f16
/// defaults are the integration tests' job, each binary its own process.
#[cfg(test)]
pub(crate) fn pin_test_device() {
    static PIN: std::sync::Once = std::sync::Once::new();
    PIN.call_once(|| {
        let device = burn::backend::wgpu::WgpuDevice::default();
        let _ = burn::tensor::set_default_dtypes::<crate::qlinear::UnfusedF32>(
            &device,
            burn::tensor::FloatDType::F32,
            burn::tensor::IntDType::I32,
        );
        let got =
            burn::tensor::get_device_settings::<crate::qlinear::UnfusedF32>(&device).float_dtype;
        assert_eq!(
            got,
            burn::tensor::FloatDType::F32,
            "the test device's default dtype was decided before the pin: a GPU test touched the device without skip_no_gpu"
        );
    });
}

#[cfg(test)]
mod device_default_tests {
    use crate::qlinear::{UnfusedF16, UnfusedF32};
    use burn::tensor::{DType, Int, Tensor, TensorData};

    /// burn keeps one default dtype per device type for the whole process,
    /// decided by whichever backend resolves it first. A GPU test that
    /// touches an f16-backend tensor first must not decide the dtype of
    /// every f32 test after it. Its own process (`--ignored --exact`): it
    /// has to be the first thing to touch the device.
    #[test]
    #[ignore = "gpu: must be the first to touch the device, so it needs its own process"]
    fn an_f16_backend_tensor_cannot_turn_the_f32_tests_f16() {
        if crate::skip_no_gpu() {
            return;
        }
        let device = burn::backend::wgpu::WgpuDevice::default();
        let _ = Tensor::<UnfusedF16, 1, Int>::zeros([1], &device);
        let x = Tensor::<UnfusedF32, 1>::from_data(TensorData::new(vec![0.5f32], [1]), &device);
        assert_eq!(
            x.dtype(),
            DType::F32,
            "an f16-backend tensor decided the f32 tests' default dtype"
        );
    }
}
