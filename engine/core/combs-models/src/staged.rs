//! Weights staged as their bytes arrive, and the seam that lets the
//! model loader read from either them or a whole file.
//!
//! A classic load walks the MODEL: it asks for `layers.0.self_attn.q_proj`
//! and the source finds those bytes wherever they are. A streamed load
//! walks the FILE: bytes arrive in offset order and each tensor must be
//! turned into its device object while it is briefly in hand, because the
//! window moves on and those bytes are gone. The two orders cannot both
//! drive the same loader, so the loader is given a supply it can ask,
//! and the supply knows which world it is in.
//!
//! The staging path calls the SAME loaders the classic path calls —
//! [`crate::qlinear::try_quant_linear`], [`crate::embed::try_quant_embedding`],
//! the dense fallback. Every presence-driven decision (a packed weight
//! the kernels refuse, an absent bias, a tied head) therefore resolves
//! exactly as it always did; nothing about which path a weight takes is
//! re-decided here, only when.

use std::collections::HashMap;

use burn::tensor::backend::Backend;
use burn::tensor::{Device, Tensor};
use combs_formats::{ModelMetadata, ModelSource};

use crate::embed::Embedding;
use crate::qlinear::{Linear, QuantLinearOp, try_quant_linear};
use crate::{ModelError, Result};

/// Run `f` with device allocations marked persistent, on the backends
/// that draw the distinction.
///
/// A staged weight lives as long as the model; the buffers used to
/// build it do not. Drawn from one pool those two lifetimes interleave,
/// and a mount lays down some hundreds of long-lived blocks between
/// transients, which is the shape that fragments an allocator. On a
/// backend without the notion this is simply a call.
pub(crate) fn persistent_scope<B: Backend, R: Send>(
    device: &Device<B>,
    f: impl FnOnce() -> R + Send,
) -> R {
    use std::any::Any;
    let device_any: &dyn Any = device;
    let Some(wgpu_device) = device_any.downcast_ref::<burn::backend::wgpu::WgpuDevice>() else {
        return f();
    };
    let client =
        <burn::backend::wgpu::WgpuRuntime as cubecl::prelude::Runtime>::client(wgpu_device);
    let mut out: Option<R> = None;
    let _ = client.memory_persistent_allocation(&mut out, move |slot: &mut Option<R>| {
        *slot = Some(f());
    });
    // The scope runs its task unconditionally today. If that ever stops
    // being true, a mount silently producing no weights is far worse
    // than one that says so here.
    out.expect("the persistent-allocation scope did not run its task")
}

/// Submit whatever device work is queued. Uploads are enqueued rather
/// than executed, so without this the buffers behind them cannot be
/// recycled and the queue grows with the model instead of with the
/// window. A no-op on backends that do not queue.
pub fn flush_device<B: Backend>(device: &Device<B>) {
    use std::any::Any;
    let device_any: &dyn Any = device;
    let Some(wgpu_device) = device_any.downcast_ref::<burn::backend::wgpu::WgpuDevice>() else {
        return;
    };
    let client =
        <burn::backend::wgpu::WgpuRuntime as cubecl::prelude::Runtime>::client(wgpu_device);
    let _ = client.flush();
}

/// A tensor already on the device, kept flat with its shape beside it.
///
/// Flat because the loader asks for ranks the map cannot know in
/// advance; a reshape on a device tensor moves no bytes, so carrying
/// rank at the boundary costs nothing and spares the alternative, which
/// is a rank-indexed enum every caller has to unwrap.
struct Staged<B: Backend> {
    flat: Tensor<B, 1>,
    shape: Vec<usize>,
}

impl<B: Backend> Staged<B> {
    fn to_rank<const D: usize>(&self, name: &str) -> Result<Tensor<B, D>> {
        let dims: [usize; D] = self.shape.clone().try_into().map_err(|_| {
            ModelError::Unsupported(format!(
                "{name} was staged with shape {:?}; the loader asked for rank {D}",
                self.shape
            ))
        })?;
        Ok(self.flat.clone().reshape(dims))
    }
}

/// Everything a model needs, uploaded as it arrived and keyed by the
/// name the model will ask for.
pub struct StagedWeights<B: Backend> {
    metadata: ModelMetadata,
    /// Names staged, in arrival order. Absence from this is what makes
    /// a [`ModelError::MissingTensor`] honest: the file has been walked
    /// entire, so "not staged" means "not in the file".
    names: Vec<String>,
    linears: HashMap<String, Linear<B>>,
    dense: HashMap<String, Staged<B>>,
    embed: Option<(Embedding<B>, Option<Box<dyn QuantLinearOp<B>>>)>,
}

impl<B: Backend> StagedWeights<B> {
    pub fn new(metadata: ModelMetadata) -> Self {
        StagedWeights {
            metadata,
            names: Vec::new(),
            linears: HashMap::new(),
            dense: HashMap::new(),
            embed: None,
        }
    }

    /// Upload one tensor from a source that currently holds its bytes.
    ///
    /// `source` may be a window over an image that has mostly gone; all
    /// that is required is that THIS tensor's payload is readable right
    /// now. What the tensor becomes — packed op, dense tensor, packed
    /// embedding with its tied head — is decided by the same helpers the
    /// whole-file loader uses, so a weight cannot take one path when
    /// streamed and another when read from disk.
    pub fn stage(
        &mut self,
        source: &dyn ModelSource,
        device: &Device<B>,
        name: &str,
    ) -> Result<()> {
        if self.names.iter().any(|n| n == name) {
            return Ok(());
        }
        persistent_scope::<B, _>(device, || self.stage_inner(source, device, name))
    }

    fn stage_inner(
        &mut self,
        source: &dyn ModelSource,
        device: &Device<B>,
        name: &str,
    ) -> Result<()> {
        // The embedding is staged as a pair: a packed table hands back
        // the tied head that shares it, one copy in VRAM rather than
        // two. Recognized by name because that pairing is the only
        // place in the model where one tensor yields two objects.
        if name.ends_with("embed_tokens.weight") || name.ends_with("embed_tokens") {
            let staged = match crate::embed::try_quant_embedding::<B>(source, name, device)? {
                Some((e, head)) => (e, Some(head)),
                None => (Embedding::Dense(self.dense_tensor(source, device, name)?), None),
            };
            self.embed = Some(staged);
            self.names.push(name.to_string());
            return Ok(());
        }
        if let Some(op) = try_quant_linear::<B>(source, name, device)? {
            self.linears.insert(name.to_string(), Linear::Quant(op));
            self.names.push(name.to_string());
            return Ok(());
        }
        let reader = source.open_tensor(name).map_err(|e| match e {
            combs_formats::FormatError::TensorNotFound(_) => {
                ModelError::MissingTensor(name.to_string())
            }
            other => ModelError::Format(other),
        })?;
        let shape = reader.shape().to_vec();
        let count: usize = shape.iter().product();
        // Read at the tensor's own rank, then flatten: a reshape on a
        // device tensor moves no bytes, and carrying rank in the map
        // would mean a rank-indexed enum at every call site.
        let flat: Tensor<B, 1> = match shape.len() {
            1 => reader.load_to_tensor::<B, 1>(device).map_err(ModelError::Format)?,
            2 => reader
                .load_to_tensor::<B, 2>(device)
                .map_err(ModelError::Format)?
                .reshape([count]),
            other => {
                return Err(ModelError::Unsupported(format!(
                    "{name}: staging a rank-{other} tensor is not supported"
                )));
            }
        };
        self.dense.insert(name.to_string(), Staged { flat, shape });
        self.names.push(name.to_string());
        Ok(())
    }

    fn dense_tensor(
        &self,
        source: &dyn ModelSource,
        device: &Device<B>,
        name: &str,
    ) -> Result<Tensor<B, 2>> {
        source
            .open_tensor(name)
            .map_err(|e| match e {
                combs_formats::FormatError::TensorNotFound(_) => {
                    ModelError::MissingTensor(name.to_string())
                }
                other => ModelError::Format(other),
            })?
            .load_to_tensor::<B, 2>(device)
            .map_err(ModelError::Format)
    }

    /// Names staged so far — what the loader's architecture probes read
    /// instead of a source's tensor list.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Metadata is fixed at the header but two facts about the model are
    /// only knowable once the file has been walked: whether it shipped
    /// an untied head, and hence whether the embedding is tied. The
    /// whole-file loader learns this from the tensor table; a streamed
    /// one learns it here, at the same moment and from the same fact.
    pub fn seal(&mut self) {
        self.metadata.tie_word_embeddings =
            !self.names.iter().any(|n| n == "lm_head.weight");
    }
}

/// Where the loader gets its weights. `Source` is the classic path,
/// unchanged in behaviour; `Staged` is the streamed one, answering from
/// what has already been uploaded.
///
/// Both answer [`ModelError::MissingTensor`] for a weight that is not
/// there, which is what keeps the loader's presence-driven fallbacks —
/// an absent bias, an absent `lm_head`, a fused projection standing in
/// for three split ones — working identically on either path.
pub(crate) enum WeightSupply<'a, B: Backend> {
    Source(&'a dyn ModelSource),
    Staged(&'a mut StagedWeights<B>),
}

impl<'a, B: Backend> WeightSupply<'a, B> {
    pub(crate) fn metadata(&self) -> &ModelMetadata {
        match self {
            WeightSupply::Source(s) => s.metadata(),
            WeightSupply::Staged(s) => s.metadata(),
        }
    }

    pub(crate) fn tensor_names(&self) -> Vec<String> {
        match self {
            WeightSupply::Source(s) => s.tensor_names(),
            WeightSupply::Staged(s) => s.names().to_vec(),
        }
    }

    /// A dense weight at the rank the caller expects.
    pub(crate) fn weight<const D: usize>(
        &mut self,
        device: &Device<B>,
        name: &str,
    ) -> Result<Tensor<B, D>> {
        match self {
            WeightSupply::Source(s) => crate::llama::load_tensor::<B, D>(*s, device, name),
            WeightSupply::Staged(staged) => staged
                .dense
                .get(name)
                .ok_or_else(|| ModelError::MissingTensor(name.to_string()))?
                .to_rank::<D>(name),
        }
    }

    /// A projection: packed op when the kernels took it, dense
    /// otherwise. The staged path does not re-decide which — that was
    /// settled when the bytes were in hand.
    pub(crate) fn linear(&mut self, device: &Device<B>, name: &str) -> Result<Linear<B>> {
        match self {
            WeightSupply::Source(s) => crate::llama::load_linear::<B>(*s, device, name),
            WeightSupply::Staged(staged) => {
                if let Some(lin) = staged.linears.remove(name) {
                    return Ok(lin);
                }
                let dense = staged
                    .dense
                    .get(name)
                    .ok_or_else(|| ModelError::MissingTensor(name.to_string()))?;
                Ok(Linear::Dense(dense.to_rank::<2>(name)?))
            }
        }
    }

    /// A weight the model may or may not have. `None` is an answer, not
    /// a failure — several architectures are told apart by exactly this.
    pub(crate) fn optional(
        &mut self,
        device: &Device<B>,
        name: &str,
    ) -> Result<Option<Tensor<B, 1>>> {
        match self.weight::<1>(device, name) {
            Ok(t) => Ok(Some(t)),
            Err(ModelError::MissingTensor(_)) => Ok(None),
            Err(ModelError::Format(combs_formats::FormatError::TensorNotFound(_))) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// The embedding, with the tied head that shares its packed table
    /// when there is one.
    pub(crate) fn embedding(
        &mut self,
        device: &Device<B>,
        name: &str,
    ) -> Result<(Embedding<B>, Option<Box<dyn QuantLinearOp<B>>>)> {
        match self {
            WeightSupply::Source(s) => {
                match crate::embed::try_quant_embedding::<B>(*s, name, device)? {
                    Some((e, head)) => Ok((e, Some(head))),
                    None => Ok((
                        Embedding::Dense(crate::llama::load_tensor::<B, 2>(*s, device, name)?),
                        None,
                    )),
                }
            }
            WeightSupply::Staged(staged) => staged
                .embed
                .take()
                .ok_or_else(|| ModelError::MissingTensor(name.to_string())),
        }
    }
}
