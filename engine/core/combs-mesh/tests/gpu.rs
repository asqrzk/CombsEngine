//! CPU/GPU renderer parity: `CpuRenderer` vs `WgpuRenderer` (feature
//! `gpu`) must produce identical pixels — harmony compare on a fixed atlas.
//!
//! GPU tests are ignored by default (CI may have no GPU); run with:
//! `cargo test --release -p combs-mesh --features gpu --test gpu -- --ignored`
//!
//! The atlas is chosen so blending is exact in both implementations:
//! alpha 0/255 pixels (identity src-over) and one alpha-128 frame whose
//! blends land on exact 8-bit values. Arbitrary semi-transparent blends
//! may differ by ±1 (CPU integer vs GPU float rounding).

#![cfg(feature = "gpu")]

use combs_mesh::render::WgpuRenderer;
use combs_mesh::{CpuRenderer, Renderer, SpriteAtlas};

/// 4x2 atlas of four 2x1 frames: opaque red, opaque green, alpha-128 blue,
/// fully transparent.
fn atlas() -> SpriteAtlas {
    let mut rgba = vec![0u8; 4 * 2 * 4];
    let px = |x: usize, y: usize, p: [u8; 4], rgba: &mut Vec<u8>| {
        let i = (y * 4 + x) * 4;
        rgba[i..i + 4].copy_from_slice(&p);
    };
    px(0, 0, [255, 0, 0, 255], &mut rgba); // frame 0: red
    px(1, 0, [255, 0, 0, 255], &mut rgba);
    px(2, 0, [0, 255, 0, 255], &mut rgba); // frame 1: green
    px(3, 0, [0, 255, 0, 255], &mut rgba);
    px(0, 1, [0, 0, 255, 128], &mut rgba); // frame 2: blue @ 128
    px(1, 1, [0, 0, 255, 128], &mut rgba);
    // frame 3 (x=2..3, y=1): fully transparent, stays zeroed.
    SpriteAtlas {
        width: 4,
        height: 2,
        frame_width: 2,
        frame_height: 1,
        frame_count: 4,
        rgba,
    }
}

#[test]
#[ignore = "gpu"]
fn render_frame_parity() {
    let atlas = atlas();
    let cpu = CpuRenderer::new();
    let gpu = WgpuRenderer::new();
    for frame in 0..4 {
        let c = cpu.render_frame(&atlas, frame).expect("cpu frame");
        let g = gpu.render_frame(&atlas, frame).expect("gpu frame");
        assert_eq!(c, g, "frame {frame} pixels differ");
    }
}

#[test]
#[ignore = "gpu"]
fn compose_parity() {
    let atlas = atlas();
    let cpu = CpuRenderer::new();
    let gpu = WgpuRenderer::new();
    let layers: &[(&SpriteAtlas, u32, i32, i32)] = &[
        (&atlas, 0, 0, 0),    // opaque red at origin
        (&atlas, 1, 2, 1),    // opaque green, offset
        (&atlas, 2, 1, 0),    // alpha-128 blue over red
        (&atlas, 1, -1, -1),  // clipped negative offset
        (&atlas, 0, 100, 50), // fully off-canvas, skipped
        (&atlas, 3, 0, 2),    // transparent layer
    ];
    let c = cpu.compose(layers, 8, 8).expect("cpu compose");
    let g = gpu.compose(layers, 8, 8).expect("gpu compose");
    assert_eq!(c, g, "compose pixels differ");
}

#[test]
#[ignore = "gpu"]
fn compose_empty_canvas_layers() {
    let atlas = atlas();
    let gpu = WgpuRenderer::new();
    let cpu = CpuRenderer::new();
    let layers: &[(&SpriteAtlas, u32, i32, i32)] = &[(&atlas, 0, 50, 50)];
    let c = cpu.compose(layers, 4, 4).expect("cpu");
    let g = gpu.compose(layers, 4, 4).expect("gpu");
    assert_eq!(c, g);
    assert!(c.iter().all(|&b| b == 0), "expected a transparent canvas");
}
