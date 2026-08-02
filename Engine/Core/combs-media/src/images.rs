//! Image preprocessing (SigLIP / Idefics3 single-image path).

use crate::{MediaError, Result};

/// A preprocessed image: planar CHW f32, normalized, ready for
/// `Tensor::from_data(TensorData::new(data, [1, c, h, w]))`.
#[derive(Debug, Clone)]
pub struct PixelBatch {
    /// Image width (pixels, after resize+pad).
    pub width: usize,
    /// Image height (pixels, after resize+pad).
    pub height: usize,
    /// Channel count (3 = RGB).
    pub channels: usize,
    /// Planar CHW data, normalized with the preprocessor's mean/std.
    pub data: Vec<f32>,
}

impl PixelBatch {
    /// `[batch=1, channels, height, width]` tensor shape.
    pub fn shape(&self) -> [usize; 4] {
        [1, self.channels, self.height, self.width]
    }
}

/// Turns encoded image bytes (PNG/JPEG/WebP) into normalized pixel batches.
pub trait ImagePreprocessor: Send + Sync {
    /// Decodes and preprocesses one image.
    fn preprocess(&self, bytes: &[u8]) -> Result<PixelBatch>;
}

/// SigLIP / Idefics3 single-image preprocessing (SmolVLM-256M/500M):
/// RGB → resize longest edge to `image_size` (aspect preserved, bilinear) →
/// pad to a square with 0.5 → rescale 1/255 → normalize mean/std 0.5.
/// (Padding at the normalization mean maps to 0 after normalization.)
pub struct SiglipPreprocessor {
    image_size: usize,
    mean: [f32; 3],
    std: [f32; 3],
}

impl SiglipPreprocessor {
    /// Preprocessor for a square `image_size` (SmolVLM: 512).
    pub fn new(image_size: usize) -> Self {
        SiglipPreprocessor {
            image_size,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }
}

impl ImagePreprocessor for SiglipPreprocessor {
    fn preprocess(&self, bytes: &[u8]) -> Result<PixelBatch> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| MediaError::Decode(e.to_string()))?
            .to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        if w == 0 || h == 0 {
            return Err(MediaError::BadShape("empty image".to_string()));
        }

        // Longest edge -> image_size, aspect preserved.
        let size = self.image_size as f64;
        let scale = size / w.max(h) as f64;
        let new_w = ((w as f64 * scale).round() as usize).max(1);
        let new_h = ((h as f64 * scale).round() as usize).max(1);
        let resized = image::imageops::resize(
            &img,
            new_w as u32,
            new_h as u32,
            image::imageops::FilterType::Triangle,
        );

        // Pad to square with 0.5 (≈128/255), top-left anchored.
        let pad_byte = 128u8;
        let mut canvas = image::RgbImage::from_pixel(
            self.image_size as u32,
            self.image_size as u32,
            image::Rgb([pad_byte, pad_byte, pad_byte]),
        );
        image::imageops::overlay(&mut canvas, &resized, 0, 0);

        // Planar CHW, rescale 1/255, normalize.
        let n = self.image_size * self.image_size;
        let mut data = vec![0f32; 3 * n];
        for (x, y, pixel) in canvas.enumerate_pixels() {
            let idx = y as usize * self.image_size + x as usize;
            for c in 0..3 {
                let v = pixel[c] as f32 / 255.0;
                data[c * n + idx] = (v - self.mean[c]) / self.std[c];
            }
        }

        Ok(PixelBatch {
            width: self.image_size,
            height: self.image_size,
            channels: 3,
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png_bytes(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
        let img = image::RgbImage::from_pixel(w, h, image::Rgb(rgb));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[test]
    fn output_shape_and_range() {
        let pp = SiglipPreprocessor::new(512);
        let out = pp.preprocess(&png_bytes(64, 32, [255, 0, 0])).unwrap();
        assert_eq!(out.shape(), [1, 3, 512, 512]);
        assert_eq!(out.data.len(), 3 * 512 * 512);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in &out.data {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(lo >= -1.0 && hi <= 1.0, "normalized range [{lo}, {hi}]");
        // A pure-red image must saturate the R channel somewhere near +1.
        let n = 512 * 512;
        let r_max = out.data[..n].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(r_max > 0.9, "red channel max {r_max}");
    }

    #[test]
    fn padding_is_zero_after_normalization() {
        let pp = SiglipPreprocessor::new(512);
        // Wide image: bottom half is padding.
        let out = pp.preprocess(&png_bytes(64, 16, [0, 255, 0])).unwrap();
        let n = 512 * 512;
        // Sample a pixel deep in the padded region (bottom-right).
        let pad_idx = n - 1;
        for c in 0..3 {
            assert!(
                out.data[c * n + pad_idx].abs() < 0.02,
                "pad should normalize to ~0, got {}",
                out.data[c * n + pad_idx]
            );
        }
    }

    #[test]
    fn rejects_garbage() {
        let pp = SiglipPreprocessor::new(512);
        assert!(pp.preprocess(b"not an image").is_err());
    }
}
