//! Sprite atlas helpers: frame geometry and frame extraction. Used by the
//! renderers; kept renderer-agnostic so a future GPU renderer reuses them.

use crate::blocks::SpriteAtlas;
use crate::error::{MeshError, Result};

/// Returns `(x, y, width, height)` of frame `frame_index` in the atlas.
/// Frames are laid out row-major: `cols = atlas.width / frame_width`.
pub fn frame_rect(atlas: &SpriteAtlas, frame_index: u32) -> Result<(u32, u32, u32, u32)> {
    atlas.validate()?;
    if frame_index >= atlas.frame_count {
        return Err(MeshError::InvalidBlock(format!(
            "frame index {frame_index} out of range ({} frames)",
            atlas.frame_count
        )));
    }
    let cols = atlas.width / atlas.frame_width;
    let x = (frame_index % cols) * atlas.frame_width;
    let y = (frame_index / cols) * atlas.frame_height;
    Ok((x, y, atlas.frame_width, atlas.frame_height))
}

/// Extracts frame `frame_index` as a tightly packed
/// `frame_width * frame_height * 4` RGBA8 buffer.
pub fn extract_frame(atlas: &SpriteAtlas, frame_index: u32) -> Result<Vec<u8>> {
    let (x, y, w, h) = frame_rect(atlas, frame_index)?;
    let stride = atlas.width as usize * 4;
    let row_bytes = w as usize * 4;
    let mut out = Vec::with_capacity(row_bytes * h as usize);
    for row in 0..h as usize {
        let start = (y as usize + row) * stride + x as usize * 4;
        out.extend_from_slice(&atlas.rgba[start..start + row_bytes]);
    }
    Ok(out)
}
