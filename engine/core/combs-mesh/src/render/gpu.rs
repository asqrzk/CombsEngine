//! GPU [`Renderer`] over **raw wgpu** (feature `gpu`).
//!
//! Device ownership note (do not "optimize" this away): this renderer owns
//! its OWN `wgpu::Instance`/`Adapter`/`Device`, separate from the engine's
//! process-global cubecl device. The repo's one-device rule is a *cubecl*
//! constraint (one cubecl runtime per adapter); raw wgpu explicitly permits
//! multiple devices per adapter, and sharing the cubecl device from here
//! would couple a data crate to the inference stack.
//!
//! Implementation: one WGSL module, three pipelines over a positioned
//! quad — `fs_copy` (frame extraction, replace), `fs_premul` (per-layer
//! draw, premultiplied src-over = [`BlendState::PREMULTIPLIED_ALPHA_BLENDING`]),
//! `fs_unpremul` (resolve back to straight alpha). Compositing accumulates
//! in premultiplied space (mathematically identical to the CPU renderer's
//! src-over), then un-premultiplies at the end. Pipelines/sampler are
//! created once behind a `OnceLock`; textures/buffers are per call (sprite
//! atlases are tiny; document before optimizing).

use std::sync::OnceLock;

use crate::blocks::SpriteAtlas;
use crate::engine::sprites;
use crate::error::{MeshError, Result};
use crate::render::Renderer;

/// GPU renderer. Construction is cheap and infallible; the device is
/// requested lazily on first render (failures surface as [`MeshError`]).
pub struct WgpuRenderer {
    state: OnceLock<std::result::Result<GpuState, String>>,
}

struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_copy: wgpu::RenderPipeline,
    pipeline_premul: wgpu::RenderPipeline,
    pipeline_unpremul: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
}

/// Per-draw transform: frame UV rect + target NDC rect (8 × f32).
#[repr(C)]
#[derive(Clone, Copy)]
struct Xform([f32; 8]);

impl Xform {
    /// Raw bytes for `write_buffer` (repr(C) over 8 f32 — no padding, no
    /// extra bytemuck dependency).
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: Xform is repr(C) over 8 plain f32 values.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr().cast::<u8>(), 32) }
    }
}

const SHADER: &str = r#"
struct Xform {
    frame_off: vec2<f32>,
    frame_scale: vec2<f32>,
    ndc_min: vec2<f32>,
    ndc_max: vec2<f32>,
};
@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;
@group(0) @binding(2) var<uniform> xf: Xform;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let bx = f32(i & 1u);
    let by = f32((i >> 1u) & 1u);
    var out: VsOut;
    out.pos = vec4<f32>(
        mix(xf.ndc_min.x, xf.ndc_max.x, bx),
        mix(xf.ndc_min.y, xf.ndc_max.y, by),
        0.0, 1.0);
    out.uv = vec2<f32>(bx, 1.0 - by);
    return out;
}

fn sample(uv01: vec2<f32>) -> vec4<f32> {
    return textureSample(t_src, s_src, uv01 * xf.frame_scale + xf.frame_off);
}

@fragment
fn fs_copy(in: VsOut) -> @location(0) vec4<f32> {
    return sample(in.uv);
}

@fragment
fn fs_premul(in: VsOut) -> @location(0) vec4<f32> {
    let c = sample(in.uv);
    return vec4<f32>(c.rgb * c.a, c.a);
}

@fragment
fn fs_unpremul(in: VsOut) -> @location(0) vec4<f32> {
    let c = sample(in.uv);
    if c.a > 0.0 {
        return vec4<f32>(c.rgb / c.a, c.a);
    }
    return c;
}
"#;

impl WgpuRenderer {
    /// Creates a renderer; the GPU device is requested on first use.
    #[must_use]
    pub fn new() -> Self {
        WgpuRenderer {
            state: OnceLock::new(),
        }
    }

    fn state(&self) -> Result<&GpuState> {
        self.state
            .get_or_init(GpuState::init)
            .as_ref()
            .map_err(|e| MeshError::InvalidBlock(format!("gpu: {e}")))
    }
}

impl Default for WgpuRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuState {
    fn init() -> std::result::Result<GpuState, String> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .map_err(|e| format!("no suitable adapter: {e}"))?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .map_err(|e| format!("device request failed: {e}"))?;

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("combs-mesh-sprite"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("combs-mesh-sprite-bgl"),
                    entries: &[
                        bgl_entry(0, wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        }),
                        bgl_entry(1, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                        bgl_entry(2, wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(32),
                        }),
                    ],
                });

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("combs-mesh-sprite-layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

            let pipeline = |fs: &str, blend: wgpu::BlendState| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(fs),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(fs),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: Some(blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
            };

            Ok(GpuState {
                pipeline_copy: pipeline("fs_copy", wgpu::BlendState::REPLACE),
                pipeline_premul: pipeline("fs_premul", wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                pipeline_unpremul: pipeline("fs_unpremul", wgpu::BlendState::REPLACE),
                sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                    mag_filter: wgpu::FilterMode::Nearest,
                    min_filter: wgpu::FilterMode::Nearest,
                    ..Default::default()
                }),
                bind_group_layout,
                device,
                queue,
            })
        })
    }

    fn texture(&self, width: u32, height: u32, usage: wgpu::TextureUsages) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage,
            view_formats: &[],
        })
    }

    fn draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        target: &wgpu::TextureView,
        source: &wgpu::TextureView,
        xform: Xform,
        load: wgpu::LoadOp<wgpu::Color>,
    ) {
        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&uniform, 0, xform.as_bytes());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..4, 0..1);
    }

    /// Reads back a `width`×`height` RGBA8 texture (256-byte row padding
    /// is stripped here).
    fn readback(
        &self,
        encoder: wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>> {
        let row_bytes = width * 4;
        let padded = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (padded * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = encoder;
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| MeshError::InvalidBlock(format!("gpu: poll failed: {e}")))?;
        rx.recv()
            .map_err(|_| MeshError::InvalidBlock("gpu: map callback dropped".into()))?
            .map_err(|e| MeshError::InvalidBlock(format!("gpu: map failed: {e}")))?;

        let mapped = slice.get_mapped_range();
        let mut out = Vec::with_capacity((row_bytes * height) as usize);
        for row in 0..height as usize {
            out.extend_from_slice(&mapped[row * padded as usize..][..row_bytes as usize]);
        }
        drop(mapped);
        buffer.unmap();
        Ok(out)
    }

    fn upload_atlas(&self, atlas: &SpriteAtlas) -> Result<wgpu::Texture> {
        atlas.validate()?;
        let texture = self.texture(
            atlas.width,
            atlas.height,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas.width * 4),
                rows_per_image: Some(atlas.height),
            },
            wgpu::Extent3d {
                width: atlas.width,
                height: atlas.height,
                depth_or_array_layers: 1,
            },
        );
        Ok(texture)
    }
}

fn bgl_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty,
        count: None,
    }
}

fn full_ndx(frame_off: [f32; 2], frame_scale: [f32; 2]) -> Xform {
    Xform([
        frame_off[0],
        frame_off[1],
        frame_scale[0],
        frame_scale[1],
        -1.0,
        -1.0,
        1.0,
        1.0,
    ])
}

impl Renderer for WgpuRenderer {
    fn render_frame(&self, atlas: &SpriteAtlas, frame_index: u32) -> Result<Vec<u8>> {
        let state = self.state()?;
        let (fx, fy, fw, fh) = sprites::frame_rect(atlas, frame_index)?;
        let source = state.upload_atlas(atlas)?;
        let source_view = source.create_view(&Default::default());
        let target = state.texture(
            fw,
            fh,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let target_view = target.create_view(&Default::default());

        let mut encoder = state.device.create_command_encoder(&Default::default());
        let xform = full_ndx(
            [fx as f32 / atlas.width as f32, fy as f32 / atlas.height as f32],
            [fw as f32 / atlas.width as f32, fh as f32 / atlas.height as f32],
        );
        state.draw(
            &mut encoder,
            &state.pipeline_copy,
            &target_view,
            &source_view,
            xform,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        state.readback(encoder, &target, fw, fh)
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
        let state = self.state()?;
        let accum = state.texture(
            width,
            height,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let accum_view = accum.create_view(&Default::default());

        let mut encoder = state.device.create_command_encoder(&Default::default());
        let mut first = true;
        for &(atlas, frame_index, x, y) in layers {
            let (fx, fy, fw, fh) = sprites::frame_rect(atlas, frame_index)?;
            // Visible region = frame rect clipped to the canvas.
            let vis_x0 = x.clamp(0, width as i32) as u32;
            let vis_y0 = y.clamp(0, height as i32) as u32;
            let vis_x1 = (x + fw as i32).clamp(0, width as i32) as u32;
            let vis_y1 = (y + fh as i32).clamp(0, height as i32) as u32;
            if vis_x0 >= vis_x1 || vis_y0 >= vis_y1 {
                continue;
            }
            let source = state.upload_atlas(atlas)?;
            let source_view = source.create_view(&Default::default());
            let (aw, ah) = (atlas.width as f32, atlas.height as f32);
            let sub_fx = fx + (vis_x0 as i32 - x) as u32;
            let sub_fy = fy + (vis_y0 as i32 - y) as u32;
            let xform = Xform([
                sub_fx as f32 / aw,
                sub_fy as f32 / ah,
                (vis_x1 - vis_x0) as f32 / aw,
                (vis_y1 - vis_y0) as f32 / ah,
                vis_x0 as f32 / width as f32 * 2.0 - 1.0,
                1.0 - vis_y1 as f32 / height as f32 * 2.0,
                vis_x1 as f32 / width as f32 * 2.0 - 1.0,
                1.0 - vis_y0 as f32 / height as f32 * 2.0,
            ]);
            let load = if first {
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
            } else {
                wgpu::LoadOp::Load
            };
            state.draw(
                &mut encoder,
                &state.pipeline_premul,
                &accum_view,
                &source_view,
                xform,
                load,
            );
            first = false;
        }
        if first {
            // No visible layers: still clear the canvas.
            let dummy = state.texture(1, 1, wgpu::TextureUsages::TEXTURE_BINDING);
            let dummy_view = dummy.create_view(&Default::default());
            state.draw(
                &mut encoder,
                &state.pipeline_premul,
                &accum_view,
                &dummy_view,
                full_ndx([0.0; 2], [0.0; 2]),
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
        }

        // Resolve: un-premultiply into the output texture.
        let target = state.texture(
            width,
            height,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let target_view = target.create_view(&Default::default());
        state.draw(
            &mut encoder,
            &state.pipeline_unpremul,
            &target_view,
            &accum_view,
            full_ndx([0.0, 0.0], [1.0, 1.0]),
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        state.readback(encoder, &target, width, height)
    }
}
