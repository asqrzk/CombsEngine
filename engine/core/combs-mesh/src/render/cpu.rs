//! Zero-dependency CPU renderer: frame slicing + src-over alpha blending
//! in integer math (no floating point, deterministic across platforms).

use crate::blocks::SpriteAtlas;
use crate::engine::sprites;
use crate::error::{MeshError, Result};
use crate::render::Renderer;

/// CPU [`Renderer`] implementation.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuRenderer;

impl CpuRenderer {
    /// Creates a renderer (stateless).
    #[must_use]
    pub fn new() -> Self {
        CpuRenderer
    }
}

impl Renderer for CpuRenderer {
    fn render_frame(&self, atlas: &SpriteAtlas, frame_index: u32) -> Result<Vec<u8>> {
        sprites::extract_frame(atlas, frame_index)
    }

    fn compose(
        &self,
        layers: &[(&SpriteAtlas, u32, i32, i32)],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>> {
        if width == 0 || height == 0 {
            return Err(MeshError::InvalidBlock("canvas must be non-empty".into()));
        }
        let mut canvas = vec![0u8; width as usize * height as usize * 4];
        for &(atlas, frame_index, x, y) in layers {
            let frame = sprites::extract_frame(atlas, frame_index)?;
            blit_src_over(
                &mut canvas,
                width,
                height,
                &frame,
                atlas.frame_width,
                atlas.frame_height,
                x,
                y,
            );
        }
        Ok(canvas)
    }
}

/// Paints `src` onto `dst` at `(ox, oy)` with src-over alpha blending.
#[allow(clippy::too_many_arguments)]
fn blit_src_over(
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    src: &[u8],
    src_w: u32,
    src_h: u32,
    ox: i32,
    oy: i32,
) {
    for sy in 0..src_h as i32 {
        let dy = oy + sy;
        if dy < 0 || dy >= dst_h as i32 {
            continue;
        }
        for sx in 0..src_w as i32 {
            let dx = ox + sx;
            if dx < 0 || dx >= dst_w as i32 {
                continue;
            }
            let si = (sy as u32 * src_w + sx as u32) as usize * 4;
            let di = (dy as u32 * dst_w + dx as u32) as usize * 4;
            blend_pixel(&mut dst[di..di + 4], &src[si..si + 4]);
        }
    }
}

/// src-over: out_a = sa + da(1-sa); out_c = (sc·sa + dc·da(1-sa)) / out_a.
/// Integer math with rounding; fully opaque/transparent fast paths.
fn blend_pixel(dst: &mut [u8], src: &[u8]) {
    let sa = src[3] as u16;
    if sa == 0 {
        return;
    }
    if sa == 255 {
        dst.copy_from_slice(src);
        return;
    }
    // u32 throughout: sc·sa·255 can reach ~8.3M (u16 wraps — found by the
    // GPU parity test).
    let da = dst[3] as u32;
    let sa32 = sa as u32;
    let inv = 255 - sa32;
    let out_a = sa32 + (da * inv + 127) / 255;
    if out_a == 0 {
        dst.iter_mut().for_each(|b| *b = 0);
        return;
    }
    for c in 0..3 {
        let sc = src[c] as u32;
        let dc = dst[c] as u32;
        let premul = sc * sa32 * 255 + dc * da * inv;
        dst[c] = ((premul / out_a + 127) / 255) as u8;
    }
    dst[3] = out_a as u8;
}
