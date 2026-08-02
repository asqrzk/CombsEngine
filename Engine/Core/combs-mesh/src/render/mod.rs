//! Sprite rendering.
//!
//! [`Renderer`] is the seam: frame extraction + alpha compositing onto an
//! RGBA8 canvas. v1 ships only [`CpuRenderer`] (zero-dep byte math); a
//! wgpu renderer can slot in behind the same trait later (deliberately
//! deferred — the mesh core carries no GPU dependency).

mod cpu;
#[cfg(feature = "gpu")]
pub mod gpu;

pub use cpu::CpuRenderer;
#[cfg(feature = "gpu")]
pub use gpu::WgpuRenderer;

use crate::blocks::SpriteAtlas;
use crate::error::Result;

/// A sprite renderer.
pub trait Renderer {
    /// Extracts frame `frame_index` as `frame_width * frame_height * 4`
    /// RGBA8 bytes.
    fn render_frame(&self, atlas: &SpriteAtlas, frame_index: u32) -> Result<Vec<u8>>;

    /// Composites `layers` — `(atlas, frame_index, x, y)` — onto a
    /// transparent `width`×`height` canvas (src-over alpha blending),
    /// returning `width * height * 4` RGBA8 bytes. Layers are painted in
    /// order (later = on top); out-of-canvas pixels are clipped.
    fn compose(
        &self,
        layers: &[(&SpriteAtlas, u32, i32, i32)],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>>;
}
