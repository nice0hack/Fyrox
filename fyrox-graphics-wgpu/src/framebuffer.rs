// Copyright (c) 2019-present Dmitry Stepanov and Fyrox Engine contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use crate::buffer::WgpuBuffer;
use crate::geometry_buffer::WgpuGeometryBuffer;
use crate::program::WgpuProgram;
use crate::sampler::WgpuSampler;
use crate::server::WgpuGraphicsServer;
use crate::texture::WgpuTexture;
use fyrox_graphics::{
    core::{color::Color, math::Rect},
    error::FrameworkError,
    framebuffer::{Attachment, BufferDataUsage, DrawCallStatistics, GpuFrameBuffer, GpuFrameBufferTrait, ReadTarget, ResourceBindGroup, ResourceBinding},
    geometry_buffer::GpuGeometryBuffer,
    gpu_program::GpuProgram,
    gpu_texture::{image_2d_size_bytes, CubeMapFace, GpuTexture, GpuTextureKind},
    CompareFunc, CullFace, DrawParameters, ElementRange, BlendMode,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::rc::Weak;

fn compare_func_to_wgpu(f: CompareFunc) -> wgpu::CompareFunction {
    match f {
        CompareFunc::Never => wgpu::CompareFunction::Never,
        CompareFunc::Less => wgpu::CompareFunction::Less,
        CompareFunc::Equal => wgpu::CompareFunction::Equal,
        CompareFunc::LessOrEqual => wgpu::CompareFunction::LessEqual,
        CompareFunc::Greater => wgpu::CompareFunction::Greater,
        CompareFunc::NotEqual => wgpu::CompareFunction::NotEqual,
        CompareFunc::GreaterOrEqual => wgpu::CompareFunction::GreaterEqual,
        CompareFunc::Always => wgpu::CompareFunction::Always,
    }
}

fn blend_mode_to_wgpu(m: BlendMode) -> wgpu::BlendOperation {
    match m {
        BlendMode::Add => wgpu::BlendOperation::Add,
        BlendMode::Subtract => wgpu::BlendOperation::Subtract,
        BlendMode::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
        BlendMode::Min => wgpu::BlendOperation::Min,
        BlendMode::Max => wgpu::BlendOperation::Max,
    }
}

fn blend_factor_to_wgpu(f: fyrox_graphics::BlendFactor) -> wgpu::BlendFactor {
    use fyrox_graphics::BlendFactor;
    match f {
        BlendFactor::Zero => wgpu::BlendFactor::Zero,
        BlendFactor::One => wgpu::BlendFactor::One,
        BlendFactor::SrcColor => wgpu::BlendFactor::Src,
        BlendFactor::OneMinusSrcColor => wgpu::BlendFactor::OneMinusSrc,
        BlendFactor::DstColor => wgpu::BlendFactor::Dst,
        BlendFactor::OneMinusDstColor => wgpu::BlendFactor::OneMinusDst,
        BlendFactor::SrcAlpha => wgpu::BlendFactor::SrcAlpha,
        BlendFactor::OneMinusSrcAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
        BlendFactor::DstAlpha => wgpu::BlendFactor::DstAlpha,
        BlendFactor::OneMinusDstAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
        BlendFactor::ConstantColor | BlendFactor::ConstantAlpha => wgpu::BlendFactor::Constant,
        BlendFactor::OneMinusConstantColor | BlendFactor::OneMinusConstantAlpha => wgpu::BlendFactor::OneMinusConstant,
        BlendFactor::SrcAlphaSaturate => wgpu::BlendFactor::SrcAlphaSaturated,
        BlendFactor::Src1Color => wgpu::BlendFactor::Src,
        BlendFactor::OneMinusSrc1Color => wgpu::BlendFactor::OneMinusSrc,
        BlendFactor::Src1Alpha => wgpu::BlendFactor::SrcAlpha,
        BlendFactor::OneMinusSrc1Alpha => wgpu::BlendFactor::OneMinusSrcAlpha,
    }
}

fn texture_format_for_attachment(tex: &GpuTexture) -> wgpu::TextureFormat {
    tex.as_any().downcast_ref::<WgpuTexture>().unwrap().format()
}

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct PipelineKey {
    program_ptr: usize,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    sample_count: u32,
    blend: bool,
    depth_test: bool,
    depth_write: bool,
    cull: u8,
}

pub struct WgpuFrameBuffer {
    server: Weak<WgpuGraphicsServer>,
    depth_attachment: Option<Attachment>,
    color_attachments: Vec<Attachment>,
    is_backbuffer: bool,
}

impl WgpuFrameBuffer {
    pub fn new(server: &WgpuGraphicsServer, depth: Option<Attachment>, colors: Vec<Attachment>) -> Result<Self, FrameworkError> {
        Ok(Self { server: server.weak_ref(), depth_attachment: depth, color_attachments: colors, is_backbuffer: false })
    }

    pub fn backbuffer(server: &WgpuGraphicsServer) -> Self {
        Self { server: server.weak_ref(), depth_attachment: None, color_attachments: Default::default(), is_backbuffer: true }
    }

    fn get_or_create_pipeline(&self, server: &WgpuGraphicsServer, program: &WgpuProgram, params: &DrawParameters, geo: &WgpuGeometryBuffer, cf: wgpu::TextureFormat, df: Option<wgpu::TextureFormat>) -> wgpu::RenderPipeline {
        let key = PipelineKey {
            program_ptr: program as *const WgpuProgram as usize, color_format: cf, depth_format: df, sample_count: server.msaa_sample_count,
            blend: params.blend.is_some(), depth_test: params.depth_test.is_some(), depth_write: params.depth_write,
            cull: match params.cull_face { Some(CullFace::Back) => 2, Some(CullFace::Front) => 1, None => 0 },
        };
        let key_hash = { let mut h = DefaultHasher::new(); key.hash(&mut h); h.finish() };
        { let cache = server.pipeline_cache.borrow(); if let Some(p) = cache.get(&key_hash) { return p.clone(); } }

        let blend_state = params.blend.as_ref().map(|bp| wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: blend_factor_to_wgpu(bp.func.sfactor), dst_factor: blend_factor_to_wgpu(bp.func.dfactor), operation: blend_mode_to_wgpu(bp.equation.rgb) },
            alpha: wgpu::BlendComponent { src_factor: blend_factor_to_wgpu(bp.func.alpha_sfactor), dst_factor: blend_factor_to_wgpu(bp.func.alpha_dfactor), operation: blend_mode_to_wgpu(bp.equation.alpha) },
        });

        let depth_stencil = if params.depth_test.is_some() || params.depth_write {
            Some(wgpu::DepthStencilState {
                format: df.unwrap_or(wgpu::TextureFormat::Depth32Float),
                depth_write_enabled: Some(params.depth_write),
                depth_compare: Some(params.depth_test.map(compare_func_to_wgpu).unwrap_or(wgpu::CompareFunction::Always)),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })
        } else { df.map(|f| wgpu::DepthStencilState { format: f, depth_write_enabled: Some(false), depth_compare: Some(wgpu::CompareFunction::Always), stencil: wgpu::StencilState::default(), bias: wgpu::DepthBiasState::default() }) };

        let cull = match params.cull_face { Some(CullFace::Back) => Some(wgpu::Face::Back), Some(CullFace::Front) => Some(wgpu::Face::Front), None => None };

        let topo = match geo.element_kind() {
            fyrox_graphics::ElementKind::Triangle => wgpu::PrimitiveTopology::TriangleList,
            fyrox_graphics::ElementKind::Line => wgpu::PrimitiveTopology::LineList,
            fyrox_graphics::ElementKind::Point => wgpu::PrimitiveTopology::PointList,
        };

        let pipeline = server.state.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("RP"), layout: Some(program.pipeline_layout()),
            vertex: wgpu::VertexState { module: program.vertex_module(), entry_point: Some("main"), buffers: geo.vertex_buffer_layouts(), compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: program.fragment_module(), entry_point: Some("main"), targets: &[Some(wgpu::ColorTargetState { format: cf, blend: blend_state, write_mask: wgpu::ColorWrites::ALL })], compilation_options: Default::default() }),
            primitive: wgpu::PrimitiveState { topology: topo, strip_index_format: None, front_face: wgpu::FrontFace::Ccw, cull_mode: cull, ..Default::default() },
            depth_stencil,
            multisample: wgpu::MultisampleState { count: server.msaa_sample_count, mask: !0, alpha_to_coverage_enabled: false },
            multiview_mask: None,
            cache: None,
        });

        server.pipeline_cache.borrow_mut().insert(key_hash, pipeline.clone());
        pipeline
    }

    fn do_draw(&self, instance_count: u32, geometry: &GpuGeometryBuffer, viewport: Rect<i32>, program: &GpuProgram, params: &DrawParameters, resources: &[ResourceBindGroup], element_range: ElementRange) -> Result<DrawCallStatistics, FrameworkError> {
        let server = self.server.upgrade().unwrap();
        let geo = geometry.as_any().downcast_ref::<WgpuGeometryBuffer>().unwrap();
        let prog = program.as_any().downcast_ref::<WgpuProgram>().unwrap();

        let (offset, count) = match element_range { ElementRange::Full => (0, geo.element_count()), ElementRange::Specific { offset, count } => (offset, count) };
        if offset + count > geo.element_count() { return Err(FrameworkError::InvalidElementRange { start: offset, end: offset + count, total: geo.element_count() }); }
        if count == 0 { return Ok(DrawCallStatistics { triangles: 0 }); }

        let surface_tex = if self.is_backbuffer {
            match server.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Some(t),
                other => return Err(FrameworkError::Custom(format!("Surface texture error: {other:?}"))),
            }
        } else { None };

        let cf = if self.is_backbuffer { server.surface_config.read().unwrap().format }
        else if let Some(fc) = self.color_attachments.first() { texture_format_for_attachment(&fc.texture) }
        else { wgpu::TextureFormat::Rgba8Unorm };

        let df = self.depth_attachment.as_ref().map(|a| texture_format_for_attachment(&a.texture));
        let pipeline = self.get_or_create_pipeline(&server, prog, params, geo, cf, df);

        let bind_group = create_bind_group(&server, prog, resources);
        let mut encoder = server.state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("DrawEnc") });

        let color_view = if self.is_backbuffer { surface_tex.as_ref().unwrap().texture.create_view(&wgpu::TextureViewDescriptor::default()) }
        else { self.color_attachments.first().unwrap().texture.as_any().downcast_ref::<WgpuTexture>().unwrap().wgpu_view().clone() };

        let depth_view = self.depth_attachment.as_ref().map(|a| a.texture.as_any().downcast_ref::<WgpuTexture>().unwrap().wgpu_view().clone());

        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("DrawRP"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &color_view, resolve_target: None, depth_slice: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: depth_view.as_ref().map(|v| wgpu::RenderPassDepthStencilAttachment { view: v, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }), stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }) }),
                ..Default::default()
            });

            rp.set_viewport(viewport.x() as f32, viewport.y() as f32, viewport.w() as f32, viewport.h() as f32, 0.0, 1.0);
            rp.set_pipeline(&pipeline);
            if let Some(bg) = &bind_group { rp.set_bind_group(0, bg, &[]); }
            for (i, vb) in geo.vertex_buffers().iter().enumerate() { rp.set_vertex_buffer(i as u32, vb.slice(..)); }
            rp.set_index_buffer(geo.element_buffer().slice(..), wgpu::IndexFormat::Uint32);

            let ipe = geo.element_kind().index_per_element();
            let idx_count = (count * ipe) as u32;
            let base_vert = (offset * ipe) as i32;
            rp.draw_indexed(0..idx_count, base_vert, 0..instance_count);
        }

        server.state.queue.submit(std::iter::once(encoder.finish()));
        Ok(DrawCallStatistics { triangles: count * instance_count as usize })
    }
}

impl GpuFrameBufferTrait for WgpuFrameBuffer {
    fn color_attachments(&self) -> &[Attachment] { &self.color_attachments }
    fn depth_attachment(&self) -> Option<&Attachment> { self.depth_attachment.as_ref() }
    fn set_cubemap_face(&self, i: usize, face: CubeMapFace, level: usize) {
        if let Some(a) = self.color_attachments.get(i) { a.set_cube_map_face(Some(face)); a.set_level(level); }
    }
    fn blit_to(&self, _dest: &GpuFrameBuffer, _sx0: i32, _sy0: i32, _sx1: i32, _sy1: i32, _dx0: i32, _dy0: i32, _dx1: i32, _dy1: i32, _c: bool, _d: bool, _s: bool) {
        log::warn!("blit_to not yet implemented for wgpu");
    }
    fn read_pixels(&self, read_target: ReadTarget) -> Option<Vec<u8>> {
        let server = self.server.upgrade()?;
        let texture = match read_target {
            ReadTarget::Depth | ReadTarget::Stencil => &self.depth_attachment.as_ref()?.texture,
            ReadTarget::Color(i) => &self.color_attachments.get(i)?.texture,
        };
        let wtex = texture.as_any().downcast_ref::<WgpuTexture>()?;
        if let GpuTextureKind::Rectangle { width, height } = texture.kind() {
            let pk = texture.pixel_kind();
            let total = image_2d_size_bytes(pk, width, height);
            let buf = server.state.device.create_buffer(&wgpu::BufferDescriptor { label: Some("ReadPx"), size: total as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let mut enc = server.state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ReadPxEnc") });
            let fmt = wtex.format();
            let bps = fmt.block_copy_size(None).unwrap_or(4);
            enc.copy_texture_to_buffer(wgpu::TexelCopyTextureInfo { texture: wtex.wgpu_texture(), mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All }, wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some((width as u32 * bps).max(256)), rows_per_image: Some(height as u32) } }, wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 });
            server.state.queue.submit(std::iter::once(enc.finish()));
            let slice = buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            server.state.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            rx.recv().ok()?.ok()?;
            let mapped = slice.get_mapped_range();
            let mut result = vec![0u8; total];
            result.copy_from_slice(&mapped);
            drop(mapped);
            buf.unmap();
            Some(result)
        } else { None }
    }
    fn clear(&self, _viewport: Rect<i32>, _color: Option<Color>, _depth: Option<f32>, _stencil: Option<i32>) {}
    fn draw(&self, geometry: &GpuGeometryBuffer, viewport: Rect<i32>, program: &GpuProgram, params: &DrawParameters, resources: &[ResourceBindGroup], element_range: ElementRange) -> Result<DrawCallStatistics, FrameworkError> {
        self.do_draw(1, geometry, viewport, program, params, resources, element_range)
    }
    fn draw_instances(&self, instance_count: usize, geometry: &GpuGeometryBuffer, viewport: Rect<i32>, program: &GpuProgram, params: &DrawParameters, resources: &[ResourceBindGroup], element_range: ElementRange) -> Result<DrawCallStatistics, FrameworkError> {
        self.do_draw(instance_count as u32, geometry, viewport, program, params, resources, element_range)
    }
}

fn create_bind_group(server: &WgpuGraphicsServer, program: &WgpuProgram, groups: &[ResourceBindGroup]) -> Option<wgpu::BindGroup> {
    let mut entries = Vec::new();
    for group in groups {
        for binding in group.bindings {
            match binding {
                ResourceBinding::Texture { texture, sampler, binding: loc } => {
                    let wt = texture.as_any().downcast_ref::<WgpuTexture>()?;
                    let ws = sampler.as_any().downcast_ref::<WgpuSampler>()?;
                    entries.push(wgpu::BindGroupEntry { binding: *loc as u32, resource: wgpu::BindingResource::TextureView(wt.wgpu_binding_view()) });
                    entries.push(wgpu::BindGroupEntry { binding: (*loc + 100) as u32, resource: wgpu::BindingResource::Sampler(ws.wgpu_sampler()) });
                }
                ResourceBinding::Buffer { buffer, binding: loc, data_usage } => {
                    let wb = buffer.as_any().downcast_ref::<WgpuBuffer>()?;
                    match data_usage {
                        BufferDataUsage::UseEverything => entries.push(wgpu::BindGroupEntry { binding: (*loc + 200) as u32, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: wb.wgpu_buffer(), offset: 0, size: None }) }),
                        BufferDataUsage::UseSegment { offset, size } => entries.push(wgpu::BindGroupEntry { binding: (*loc + 200) as u32, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: wb.wgpu_buffer(), offset: *offset as u64, size: Some(std::num::NonZeroU64::new(*size as u64).unwrap()) }) }),
                    }
                }
            }
        }
    }
    if entries.is_empty() { return None; }
    Some(server.state.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("BG"), layout: program.bind_group_layout(), entries: &entries }))
}
