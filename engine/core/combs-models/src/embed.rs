//! The embedding table: dense, or packed with an on-device gather.
//!
//! On a quantized checkpoint the embedding is the largest tensor in the
//! model, and until now it was the one tensor that ALWAYS dequantized to
//! dense f32 at load — 622 MB for a 151k-vocab model whose every other
//! weight stayed packed. Worse, a tied lm_head then ran a dense matmul
//! over that table every step: the single largest op of decode, on a
//! model that is nominally "quantized".
//!
//! [`Embedding::Packed`] keeps the table in its GGUF packing. Lookup is a
//! dequant-gather kernel (bit-exact with the CPU reference, per row); a
//! tied head shares the same `Arc<QuantWeight>` through the existing
//! quant-linear op, so the table exists exactly once in VRAM and the
//! head's matmul reads packed bytes.
//!
//! Fallback discipline as everywhere: any reason at all — foreign
//! backend, unpacked source, a format without a gather kernel, the env
//! door — yields `None` from [`try_quant_embedding`] and the caller loads
//! the dense table exactly as before.

use std::any::{Any, TypeId};
use std::sync::Arc;

use burn::backend::wgpu::{CubeTensor, WgpuDevice, WgpuRuntime};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Device, FloatDType, Int, Shape, Tensor, TensorPrimitive};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl_fusion::CubeFusionHandle;
use burn_fusion::stream::{Operation, OperationStreams};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr, TensorStatus};
use combs_formats::ModelSource;

use crate::qlinear::{
    FusedF32, InnerF32, QuantLinearOp, UnfusedF16, UnfusedF32, quant_linear_from_weight,
};
use crate::qmatmul::QuantWeight;
use crate::{ModelError, Result};

/// A backend-specific packed-embedding lookup. Boxed into
/// [`Embedding::Packed`] at load time by [`try_quant_embedding`].
pub trait PackedEmbedOp<B: Backend>: Send + Sync {
    /// `tokens: [1, seq]` int ids → `[1, seq, hidden]` dequantized rows.
    fn gather(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3>;
    /// `[vocab, hidden]`, matching the dense table's `dims()`.
    fn dims(&self) -> [usize; 2];
    /// Bytes the packed table occupies in VRAM.
    fn vram_bytes(&self) -> usize;
}

/// The embedding table of a loaded model.
pub enum Embedding<B: Backend> {
    /// Dense `[vocab, hidden]` — the portable path, and the only one a
    /// tied head can matmul directly.
    Dense(Tensor<B, 2>),
    /// Packed GGUF blocks with an on-device gather; a tied head shares
    /// the same packed weight through [`crate::qlinear::Linear::Quant`].
    Packed(Box<dyn PackedEmbedOp<B>>),
}

impl<B: Backend> Embedding<B> {
    /// `[vocab, hidden]`.
    pub fn dims(&self) -> [usize; 2] {
        match self {
            Embedding::Dense(t) => t.dims(),
            Embedding::Packed(op) => op.dims(),
        }
    }

    /// Row lookup: `tokens [1, seq]` → `[1, seq, hidden]`.
    pub fn gather(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        match self {
            Embedding::Dense(t) => {
                let [_, seq] = tokens.dims();
                let flat = tokens.reshape([seq]);
                let [_, hidden] = t.dims();
                t.clone().select(0, flat).reshape([1, seq, hidden])
            }
            Embedding::Packed(op) => op.gather(tokens),
        }
    }

    /// The dense table, where one exists. A packed embedding deliberately
    /// has no dense view — materializing one would silently un-win the
    /// memory this type exists to save — so callers that genuinely need
    /// the dense tensor (the tied-head fallback matmul) must only reach
    /// it on the Dense arm, which the load path guarantees by pairing
    /// every packed embedding with a packed tied head.
    pub fn dense(&self) -> Option<&Tensor<B, 2>> {
        match self {
            Embedding::Dense(t) => Some(t),
            Embedding::Packed(_) => None,
        }
    }
}

/// The concrete packed op: a [`QuantWeight`] shared with the tied head.
struct CubeQuantEmbed {
    w: Arc<QuantWeight>,
}

impl CubeQuantEmbed {
    fn dims(&self) -> [usize; 2] {
        [self.w.n_out(), self.w.k()]
    }

    /// Shared unfused path: contiguous int ids in, f32 rows out. Token
    /// ids are non-negative i32 on device; the kernel reads the same
    /// bytes as u32.
    fn gather_cube(&self, ids: CubeTensor<WgpuRuntime>, seq: usize) -> CubeTensor<WgpuRuntime> {
        let ids = into_contiguous(ids);
        let out_h = self
            .w
            .gather_rows_device(&ids.client, ids.handle.clone(), seq)
            .expect("constructed only for formats with a gather kernel");
        CubeTensor::new_contiguous(
            ids.client.clone(),
            ids.device.clone(),
            Shape::from([1, seq, self.w.k()]),
            out_h,
            DType::F32,
        )
    }
}

impl PackedEmbedOp<UnfusedF32> for CubeQuantEmbed {
    fn gather(&self, tokens: Tensor<UnfusedF32, 2, Int>) -> Tensor<UnfusedF32, 3> {
        let [_, seq] = tokens.dims();
        let prim = tokens.into_primitive();
        let out = self.gather_cube(prim, seq);
        Tensor::from_primitive(TensorPrimitive::Float(out))
    }

    fn dims(&self) -> [usize; 2] {
        CubeQuantEmbed::dims(self)
    }

    fn vram_bytes(&self) -> usize {
        self.w.vram_bytes()
    }
}

impl PackedEmbedOp<UnfusedF16> for CubeQuantEmbed {
    fn gather(&self, tokens: Tensor<UnfusedF16, 2, Int>) -> Tensor<UnfusedF16, 3> {
        // Rows come back f32 (the kernel's dtype); the activation cast to
        // the backend's float follows the quant-linear precedent.
        let [_, seq] = tokens.dims();
        let prim = tokens.into_primitive();
        let out = self.gather_cube(prim, seq);
        Tensor::<UnfusedF16, 3>::from_primitive(TensorPrimitive::Float(out)).cast(FloatDType::F16)
    }

    fn dims(&self) -> [usize; 2] {
        CubeQuantEmbed::dims(self)
    }

    fn vram_bytes(&self) -> usize {
        self.w.vram_bytes()
    }
}

/// The fusion custom op: int ids in, f32 rows out.
struct EmbedGatherOp {
    desc: CustomOpIr,
    w: Arc<QuantWeight>,
    seq: usize,
}

impl core::fmt::Debug for EmbedGatherOp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "EmbedGatherOp {{ table: [{}, {}], seq: {} }}",
            self.w.n_out(),
            self.w.k(),
            self.seq
        )
    }
}

impl Operation<FusionCubeRuntime<WgpuRuntime>> for EmbedGatherOp {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle<WgpuRuntime>>) {
        let ([input], [output]) = self.desc.as_fixed::<1, 1>();
        let ids: CubeTensor<WgpuRuntime> = handles.get_int_tensor::<InnerF32>(input);
        let ids = into_contiguous(ids);
        let out_h = self
            .w
            .gather_rows_device(&ids.client, ids.handle.clone(), self.seq)
            .expect("constructed only for formats with a gather kernel");
        let out = CubeTensor::new_contiguous(
            ids.client.clone(),
            ids.device.clone(),
            Shape::from([1, self.seq, self.w.k()]),
            out_h,
            DType::F32,
        );
        handles.register_float_tensor::<InnerF32>(&output.id, out);
    }
}

impl PackedEmbedOp<FusedF32> for CubeQuantEmbed {
    fn gather(&self, tokens: Tensor<FusedF32, 2, Int>) -> Tensor<FusedF32, 3> {
        let [_, seq] = tokens.dims();
        let prim = tokens.into_primitive();
        let client = prim.client.clone();

        let mut streams = OperationStreams::default();
        streams.tensor(&prim);
        let input_ir = prim.into_ir();
        let out_ir = TensorIr {
            id: client.create_empty_handle(),
            shape: Shape::from([1, seq, self.w.k()]),
            status: TensorStatus::NotInit,
            dtype: DType::F32,
        };
        let desc = CustomOpIr::new("combs_embed_gather", &[input_ir], &[out_ir]);
        let op = EmbedGatherOp {
            desc: desc.clone(),
            w: self.w.clone(),
            seq,
        };
        let mut outputs = client.register(streams, OperationIr::Custom(desc), op);
        let out = outputs.pop().expect("custom op declares one output");
        Tensor::from_primitive(TensorPrimitive::Float(out))
    }

    fn dims(&self) -> [usize; 2] {
        CubeQuantEmbed::dims(self)
    }

    fn vram_bytes(&self) -> usize {
        self.w.vram_bytes()
    }
}

/// Boxes `op` as a `PackedEmbedOp<B>` iff `B` is `T` — the same runtime
/// type-equality bridge the quant-linear seam uses.
fn cast_embed<B: Backend, T: Backend>(
    op: Box<dyn PackedEmbedOp<T>>,
) -> Option<Box<dyn PackedEmbedOp<B>>> {
    let any: Box<dyn Any> = Box::new(op);
    any.downcast::<Box<dyn PackedEmbedOp<B>>>().ok().map(|b| *b)
}

/// Tries to keep the embedding table packed: gather kernel for lookup,
/// and — because the table is shared — a quant-linear op ready to serve
/// as a tied lm_head off the same VRAM copy. `None` → dense fallback,
/// identical to today. Errors only on malformed packed data.
pub fn try_quant_embedding<B: Backend>(
    source: &dyn ModelSource,
    name: &str,
    device: &Device<B>,
) -> Result<Option<(Embedding<B>, Box<dyn QuantLinearOp<B>>)>> {
    // Doors: the shared kill switch for all quant kernels, plus this
    // seam's own (the packed table changes the tied head's arithmetic
    // from dense matmul to the gemv kernel; the door isolates that).
    if std::env::var_os("COMBS_NO_QUANT_KERNELS").is_some_and(|v| v != "0") {
        return Ok(None);
    }
    if std::env::var_os("COMBS_PACKED_EMBED").is_some_and(|v| v == "0") {
        return Ok(None);
    }
    let supported = [
        TypeId::of::<FusedF32>(),
        TypeId::of::<UnfusedF32>(),
        TypeId::of::<UnfusedF16>(),
    ];
    if !supported.contains(&TypeId::of::<B>()) {
        return Ok(None);
    }
    let device_any: &dyn Any = device;
    let Some(wgpu_device) = device_any.downcast_ref::<WgpuDevice>() else {
        return Ok(None);
    };
    let Some(qt) = source.open_tensor_quant(name).map_err(ModelError::Format)? else {
        return Ok(None);
    };
    let &[vocab, hidden] = qt.shape.as_slice() else {
        return Ok(None);
    };

    let client = <WgpuRuntime as cubecl::prelude::Runtime>::client(wgpu_device);
    let Ok(w) = QuantWeight::from_quant_tensor(&client, qt.format, &qt.data, vocab, hidden) else {
        return Ok(None);
    };
    if !w.supports_gather() {
        // No gather kernel for this format yet (Q8_0 only in the first
        // landing) — the dense path is correct, just bigger.
        debug_embed(name, "no gather kernel for this format; dense");
        return Ok(None);
    }
    let w = Arc::new(w);

    let Some(head) = quant_linear_from_weight::<B>(w.clone()) else {
        return Ok(None);
    };
    let embed = CubeQuantEmbed { w };
    let op = if TypeId::of::<B>() == TypeId::of::<FusedF32>() {
        cast_embed::<B, FusedF32>(Box::new(embed))
    } else if TypeId::of::<B>() == TypeId::of::<UnfusedF32>() {
        cast_embed::<B, UnfusedF32>(Box::new(embed))
    } else {
        cast_embed::<B, UnfusedF16>(Box::new(embed))
    };
    if op.is_some() {
        debug_embed(name, "packed table + gather kernel + tied gemv head");
    }
    Ok(op.map(|op| (Embedding::Packed(op), head)))
}

/// Mirror of qlinear's `COMBS_DEBUG_QUANT` reporting for the embedding
/// seam — silent unless asked.
fn debug_embed(name: &str, outcome: &str) {
    if std::env::var_os("COMBS_DEBUG_QUANT").is_some() {
        eprintln!("quant-embed {name}: {outcome}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source with no packed representation — the trait's own default
    /// for `open_tensor_quant`. Nothing else is ever queried before the
    /// packed path bows out, and the panics prove it.
    struct NoQuantSource;
    impl ModelSource for NoQuantSource {
        fn metadata(&self) -> &combs_formats::ModelMetadata {
            unimplemented!("not queried by try_quant_embedding")
        }
        fn tensor_names(&self) -> Vec<String> {
            unimplemented!("not queried by try_quant_embedding")
        }
        fn open_tensor(
            &self,
            _: &str,
        ) -> combs_formats::Result<combs_formats::TensorReader<'_>> {
            unimplemented!("not queried by try_quant_embedding")
        }
        fn tokenizer(&self) -> combs_formats::Result<combs_formats::TokenizerSpec> {
            unimplemented!("not queried by try_quant_embedding")
        }
        fn sampler_defaults(&self) -> Option<combs_formats::SamplerConfig> {
            unimplemented!("not queried by try_quant_embedding")
        }
    }

    /// The safetensors guarantee, by construction: a source without packed
    /// tensors yields `None`, and the caller loads the dense embedding
    /// exactly as it did before this seam existed. Hermetic — the check
    /// happens before any device client is created.
    #[test]
    fn a_source_without_packed_tensors_stays_dense() {
        let device = Default::default();
        let got = try_quant_embedding::<FusedF32>(
            &NoQuantSource,
            "model.embed_tokens.weight",
            &device,
        )
        .expect("the dense fallback is not an error");
        assert!(got.is_none(), "no packed bytes must mean a dense embedding");
    }
}
