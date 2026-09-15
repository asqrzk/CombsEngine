//! The cache scrub: the bytes a KV buffer holds when the engine frees it — a session's arena and sliding window
//! at the session's end, the old arena at growth, the old window at every sliding step — are cleared with one
//! constant fill before the pool hands the buffer to the next allocation. The pool returns freed pages with their
//! bytes intact (measured), and multiplying by zero keeps sign bits and NaN, so a fill it is.
//!
//! Door: `COMBS_KV_SCRUB` — on unless it is `0`.

use std::any::TypeId;

use burn::backend::wgpu::{CubeTensor, WgpuRuntime};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use cubecl::prelude::*;

use super::rmsnorm::cast_any;
use super::{launch, ScrubFill};
use crate::qlinear::{FusedF32, UnfusedF16, UnfusedF32};

/// The door's parse, pure: on unless the variable is exactly `0`.
pub(crate) fn kv_scrub_door(env: Option<&str>) -> bool {
    env != Some("0")
}

/// `COMBS_KV_SCRUB`, read once per process.
pub fn kv_scrub_enabled() -> bool {
    static DOOR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DOOR.get_or_init(|| kv_scrub_door(std::env::var("COMBS_KV_SCRUB").ok().as_deref()))
}

/// The fill's grid over `words`: one invocation a word, 256 a group, a second grid dimension past 65,535 groups.
/// Returns the count and the groups in one row (the kernel's second scalar).
pub(crate) fn fill_grid(words: usize) -> (CubeCount, u64) {
    let groups = words.div_ceil(256).max(1);
    if groups <= 65_535 {
        (CubeCount::Static(groups as u32, 1, 1), groups as u64)
    } else {
        (CubeCount::Static(65_535, groups.div_ceil(65_535) as u32, 1), 65_535)
    }
}

/// One constant write of zero over every whole word of the allocation under the tensor, launched on the stream
/// every other op rides — so it lands after whatever last read the buffer and before the pool hands it on. Only
/// whole words: a trailing half-word (an odd byte count, which no allocation here has) is left, never written past.
pub(crate) fn scrub_buffer(t: &CubeTensor<WgpuRuntime>) {
    // The whole allocation under the tensor, whatever view of it the tensor is: a window narrowed out of a
    // longer buffer dies together with the rows it dropped, and a fill sized by the view would leave them.
    let mut whole = t.handle.clone();
    whole.offset_start = None;
    whole.offset_end = None;
    let words = (whole.size() / 4) as usize;
    if words == 0 || words > u32::MAX as usize {
        return;
    }
    let (count, row) = fill_grid(words);
    launch(&t.client, ScrubFill, count, vec![whole.binding()], vec![words as u64, row]);
}

/// Clears the buffer under a float tensor before it dies. Returns whether a fill was launched: false on a backend
/// whose buffers this cannot reach (the tensor is dropped as it was).
pub(crate) fn scrub_float<B: Backend, const D: usize>(t: Tensor<B, D>) -> bool {
    if TypeId::of::<B>() == TypeId::of::<UnfusedF32>() {
        let t: Tensor<UnfusedF32, D> = cast_any(t).expect("the backend was checked");
        scrub_buffer(&t.into_primitive().tensor());
        return true;
    }
    if TypeId::of::<B>() == TypeId::of::<UnfusedF16>() {
        let t: Tensor<UnfusedF16, D> = cast_any(t).expect("the backend was checked");
        scrub_buffer(&t.into_primitive().tensor());
        return true;
    }
    if TypeId::of::<B>() == TypeId::of::<FusedF32>() {
        // The fused backend hands out a lazy tensor; resolving it drains the fusion stream and gives the buffer.
        let t: Tensor<FusedF32, D> = cast_any(t).expect("the backend was checked");
        let ft = t.into_primitive().tensor();
        scrub_buffer(&ft.client.clone().resolve_tensor_float::<UnfusedF32>(ft));
        return true;
    }
    false
}

/// `scrub_float` for an int tensor (the packed halves of a quantized arena).
pub(crate) fn scrub_int<B: Backend, const D: usize>(t: Tensor<B, D, Int>) -> bool {
    if TypeId::of::<B>() == TypeId::of::<UnfusedF32>() {
        let t: Tensor<UnfusedF32, D, Int> = cast_any(t).expect("the backend was checked");
        scrub_buffer(&t.into_primitive());
        return true;
    }
    if TypeId::of::<B>() == TypeId::of::<UnfusedF16>() {
        let t: Tensor<UnfusedF16, D, Int> = cast_any(t).expect("the backend was checked");
        scrub_buffer(&t.into_primitive());
        return true;
    }
    if TypeId::of::<B>() == TypeId::of::<FusedF32>() {
        let t: Tensor<FusedF32, D, Int> = cast_any(t).expect("the backend was checked");
        let ft = t.into_primitive();
        scrub_buffer(&ft.client.clone().resolve_tensor_int::<UnfusedF32>(ft));
        return true;
    }
    false
}

/// A dying pair of K and V tensors: both scrubbed, the count of buffers reached returned.
pub(crate) fn scrub_pair<B: Backend, const D: usize>(k: Tensor<B, D>, v: Tensor<B, D>) -> usize {
    scrub_float(k) as usize + scrub_float(v) as usize
}

#[cfg(test)]
mod tests {
    use super::kv_scrub_door;

    #[test]
    fn the_scrub_door_is_on_unless_told_zero() {
        assert!(kv_scrub_door(None));
        assert!(kv_scrub_door(Some("1")));
        assert!(kv_scrub_door(Some("")));
        assert!(!kv_scrub_door(Some("0")));
    }
}
