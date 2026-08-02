//! `img` — a sprite atlas of RGBA8 pixels with fixed-size frames.

use serde::{Deserialize, Serialize};

use crate::error::{MeshError, Result};

/// An image block: an optional label plus the atlas itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageBlock {
    /// Optional label (may be empty).
    #[serde(default)]
    pub name: String,
    /// The pixel data + frame geometry.
    pub atlas: SpriteAtlas,
}

/// A sprite atlas: `width`×`height` RGBA8 pixels holding `frame_count`
/// frames of `frame_width`×`frame_height`, laid out row-major.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpriteAtlas {
    /// Atlas width in pixels.
    pub width: u32,
    /// Atlas height in pixels.
    pub height: u32,
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
    /// Number of frames in the atlas.
    pub frame_count: u32,
    /// RGBA8 pixels, `width * height * 4` bytes.
    pub rgba: Vec<u8>,
}

impl SpriteAtlas {
    /// Checks pixel length and frame geometry.
    pub fn validate(&self) -> Result<()> {
        let expected = self.width as usize * self.height as usize * 4;
        if self.rgba.len() != expected {
            return Err(MeshError::InvalidBlock(format!(
                "atlas {}x{} expects {expected} RGBA bytes, got {}",
                self.width,
                self.height,
                self.rgba.len()
            )));
        }
        if self.frame_width == 0 || self.frame_height == 0 {
            return Err(MeshError::InvalidBlock(
                "frame dimensions must be non-zero".into(),
            ));
        }
        if self.frame_width > self.width || self.frame_height > self.height {
            return Err(MeshError::InvalidBlock(format!(
                "frame {}x{} exceeds atlas {}x{}",
                self.frame_width, self.frame_height, self.width, self.height
            )));
        }
        let capacity = (self.width / self.frame_width) * (self.height / self.frame_height);
        if self.frame_count == 0 || self.frame_count > capacity {
            return Err(MeshError::InvalidBlock(format!(
                "frame_count {} does not fit {} frames in atlas",
                self.frame_count, capacity
            )));
        }
        Ok(())
    }
}
