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
    gpu_texture::{CubeMapFace, GpuTexture, GpuTextureKind},
    CompareFunc, CullFace, DrawParameters, ElementRange, BlendMode,
};
use std::cell::{Cell, RefCell};
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

fn stencil_action_to_wgpu(a: fyrox_graphics::StencilAction) -> wgpu::StencilOperation {
    match a {
        fyrox_graphics::StencilAction::Keep => wgpu::StencilOperation::Keep,
        fyrox_graphics::StencilAction::Zero => wgpu::StencilOperation::Zero,
        fyrox_graphics::StencilAction::Replace => wgpu::StencilOperation::Replace,
        fyrox_graphics::StencilAction::Incr => wgpu::StencilOperation::IncrementClamp,
        fyrox_graphics::StencilAction::IncrWrap => wgpu::StencilOperation::IncrementWrap,
        fyrox_graphics::StencilAction::Decr => wgpu::StencilOperation::DecrementClamp,
        fyrox_graphics::StencilAction::DecrWrap => wgpu::StencilOperation::DecrementWrap,
        fyrox_graphics::StencilAction::Invert => wgpu::StencilOperation::Invert,
    }
}

fn stencil_face_state(compare: CompareFunc, op: &fyrox_graphics::StencilOp) -> wgpu::StencilFaceState {
    wgpu::StencilFaceState {
        compare: compare_func_to_wgpu(compare),
        fail_op: stencil_action_to_wgpu(op.fail),
        depth_fail_op: stencil_action_to_wgpu(op.zfail),
        pass_op: stencil_action_to_wgpu(op.zpass),
    }
}

fn format_has_stencil(fmt: wgpu::TextureFormat) -> bool {
    matches!(fmt, wgpu::TextureFormat::Depth24PlusStencil8 | wgpu::TextureFormat::Depth32FloatStencil8)
}

fn cubemap_face_to_layer(face: CubeMapFace) -> u32 {
    match face {
        CubeMapFace::PositiveX => 0,
        CubeMapFace::NegativeX => 1,
        CubeMapFace::PositiveY => 2,
        CubeMapFace::NegativeY => 3,
        CubeMapFace::PositiveZ => 4,
        CubeMapFace::NegativeZ => 5,
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
    stencil: bool,
    has_color: bool,
    cull: u8,
    extra_vert_count: u8,
}

pub struct WgpuFrameBuffer {
    server: Weak<WgpuGraphicsServer>,
    depth_attachment: Option<Attachment>,
    color_attachments: Vec<Attachment>,
    is_backbuffer: bool,
    needs_clear: Cell<bool>,
    pending_clear_color: RefCell<wgpu::Color>,
    pending_clear_depth: RefCell<f32>,
}

impl WgpuFrameBuffer {
    pub fn new(server: &WgpuGraphicsServer, depth: Option<Attachment>, colors: Vec<Attachment>) -> Result<Self, FrameworkError> {
        Ok(Self { server: server.weak_ref(), depth_attachment: depth, color_attachments: colors, is_backbuffer: false, needs_clear: Cell::new(false), pending_clear_color: RefCell::new(wgpu::Color::BLACK), pending_clear_depth: RefCell::new(1.0) })
    }

    pub fn backbuffer(server: &WgpuGraphicsServer, depth: Option<Attachment>) -> Self {
        Self { server: server.weak_ref(), depth_attachment: depth, color_attachments: Default::default(), is_backbuffer: true, needs_clear: Cell::new(false), pending_clear_color: RefCell::new(wgpu::Color::BLACK), pending_clear_depth: RefCell::new(1.0) }
    }

    fn get_or_create_pipeline(&self, server: &WgpuGraphicsServer, program: &WgpuProgram, params: &DrawParameters, all_layouts: &[wgpu::VertexBufferLayout<'static>], cf: wgpu::TextureFormat, df: Option<wgpu::TextureFormat>, pipeline_layout: &wgpu::PipelineLayout, element_kind: fyrox_graphics::ElementKind, has_color: bool) -> wgpu::RenderPipeline {
        let needs_stencil = params.stencil_test.is_some() || params.stencil_op.zpass != fyrox_graphics::StencilAction::Keep || params.stencil_op.fail != fyrox_graphics::StencilAction::Keep || params.stencil_op.zfail != fyrox_graphics::StencilAction::Keep;
        let depth_fmt = df.unwrap_or(wgpu::TextureFormat::Depth32Float);
        let stencil_supported = format_has_stencil(depth_fmt);
        let effective_stencil = needs_stencil && stencil_supported;
        let key = PipelineKey {
            program_ptr: program as *const WgpuProgram as usize, color_format: cf, depth_format: df, sample_count: server.msaa_sample_count,
            blend: params.blend.is_some(), depth_test: params.depth_test.is_some(), depth_write: params.depth_write,
            stencil: effective_stencil,
            has_color,
            cull: match params.cull_face { Some(CullFace::Back) => 2, Some(CullFace::Front) => 1, None => 0 },
            extra_vert_count: all_layouts.len() as u8,
        };
        let key_hash = { let mut h = DefaultHasher::new(); key.hash(&mut h); h.finish() };
        { let cache = server.pipeline_cache.borrow(); if let Some(p) = cache.get(&key_hash) { return p.clone(); } }

        let blend_state = params.blend.as_ref().map(|bp| {
            let rgb_op = blend_mode_to_wgpu(bp.equation.rgb);
            let alpha_op = blend_mode_to_wgpu(bp.equation.alpha);
            let is_minmax_rgb = matches!(rgb_op, wgpu::BlendOperation::Min | wgpu::BlendOperation::Max);
            let is_minmax_alpha = matches!(alpha_op, wgpu::BlendOperation::Min | wgpu::BlendOperation::Max);
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: if is_minmax_rgb { wgpu::BlendFactor::One } else { blend_factor_to_wgpu(bp.func.sfactor) },
                    dst_factor: if is_minmax_rgb { wgpu::BlendFactor::One } else { blend_factor_to_wgpu(bp.func.dfactor) },
                    operation: rgb_op,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: if is_minmax_alpha { wgpu::BlendFactor::One } else { blend_factor_to_wgpu(bp.func.alpha_sfactor) },
                    dst_factor: if is_minmax_alpha { wgpu::BlendFactor::One } else { blend_factor_to_wgpu(bp.func.alpha_dfactor) },
                    operation: alpha_op,
                },
            }
        });

        let wgpu_stencil_state = if effective_stencil {
            let default_face = stencil_face_state(CompareFunc::Always, &params.stencil_op);
            let sf = params.stencil_test.as_ref().map(|st| stencil_face_state(st.func, &params.stencil_op))
                .unwrap_or(default_face);
            let read_mask = params.stencil_test.as_ref().map(|st| st.mask).unwrap_or(0xFFFF_FFFF);
            wgpu::StencilState {
                front: sf,
                back: sf,
                read_mask,
                write_mask: params.stencil_op.write_mask,
            }
        } else {
            wgpu::StencilState::default()
        };

        let depth_stencil = if params.depth_test.is_some() || params.depth_write || effective_stencil {
            Some(wgpu::DepthStencilState {
                format: depth_fmt,
                depth_write_enabled: Some(params.depth_write),
                depth_compare: Some(params.depth_test.map(compare_func_to_wgpu).unwrap_or(wgpu::CompareFunction::Always)),
                stencil: wgpu_stencil_state,
                bias: wgpu::DepthBiasState::default(),
            })
        } else {
            df.map(|f| wgpu::DepthStencilState {
                format: f,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            })
        };

        let cull = match params.cull_face { Some(CullFace::Back) => Some(wgpu::Face::Back), Some(CullFace::Front) => Some(wgpu::Face::Front), None => None };

        let topo = match element_kind {
            fyrox_graphics::ElementKind::Triangle => wgpu::PrimitiveTopology::TriangleList,
            fyrox_graphics::ElementKind::Line => wgpu::PrimitiveTopology::LineList,
            fyrox_graphics::ElementKind::Point => wgpu::PrimitiveTopology::PointList,
        };

        let color_target = [Some(wgpu::ColorTargetState { format: cf, blend: blend_state, write_mask: wgpu::ColorWrites::ALL })];
        let fragment_state = if has_color { Some(wgpu::FragmentState { module: program.fragment_module(), entry_point: Some("fs_main"), targets: &color_target, compilation_options: Default::default() }) } else { None };
        let pipeline = server.state.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("RP"), layout: Some(pipeline_layout),
            vertex: wgpu::VertexState { module: program.vertex_module(), entry_point: Some("vs_main"), buffers: all_layouts, compilation_options: Default::default() },
            fragment: fragment_state,
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
            // Only acquire a new frame if one isn't already stored from a prior draw call this frame.
            if server.current_frame.borrow().is_none() {
                match server.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                        *server.current_frame.borrow_mut() = Some(t);
                    }
                    wgpu::CurrentSurfaceTexture::Timeout => {
                        log::warn!("Surface texture timeout, skipping frame");
                        return Ok(DrawCallStatistics { triangles: 0 });
                    }
                    wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                        let config = server.surface_config.read().unwrap();
                        server.surface.configure(&server.state.device, &config);
                        log::warn!("Surface lost/outdated, reconfigured");
                        return Ok(DrawCallStatistics { triangles: 0 });
                    }
                    other => return Err(FrameworkError::Custom(format!("Surface texture error: {other:?}"))),
                }
            }
            // Create a view from the stored frame (doesn't consume the SurfaceTexture).
            let frame = server.current_frame.borrow();
            Some(frame.as_ref().unwrap().texture.create_view(&wgpu::TextureViewDescriptor::default()))
        } else { None };

        let cf = if self.is_backbuffer { server.surface_config.read().unwrap().format }
        else if let Some(fc) = self.color_attachments.first() { texture_format_for_attachment(&fc.texture) }
        else { wgpu::TextureFormat::Rgba8Unorm };

        let df = self.depth_attachment.as_ref().map(|a| texture_format_for_attachment(&a.texture));

        // Collect actual texture formats from resources for layout creation
        let mut texture_formats: Vec<(usize, wgpu::TextureFormat)> = Vec::new();
        for group in resources {
            for binding in group.bindings {
                if let ResourceBinding::Texture { texture, binding: loc, .. } = binding {
                    let wt = texture.as_any().downcast_ref::<WgpuTexture>().unwrap();
                    texture_formats.push((*loc, wt.format()));
                }
            }
        }
        let (_, pipeline_layout) = prog.get_or_create_layouts(&texture_formats);

        let (all_layouts, extra_vert_count) = build_vertex_layouts(geo);
        let has_color = self.is_backbuffer || !self.color_attachments.is_empty();
        let pipeline = self.get_or_create_pipeline(&server, prog, params, &all_layouts, cf, df, &pipeline_layout, geo.element_kind(), has_color);

        let bind_group = create_bind_group(&server, prog, resources);
        let mut encoder = server.state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("DrawEnc") });

        let color_view = if self.is_backbuffer {
            Some(surface_tex.unwrap())
        } else {
            self.color_attachments.first().map(|first_color| {
                if let Some(face) = first_color.cube_map_face() {
                    let wt = first_color.texture.as_any().downcast_ref::<WgpuTexture>().unwrap();
                    wt.wgpu_texture().create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: cubemap_face_to_layer(face),
                        array_layer_count: Some(1),
                        mip_level_count: Some(1),
                        ..Default::default()
                    })
                } else {
                    first_color.texture.as_any().downcast_ref::<WgpuTexture>().unwrap().wgpu_view().clone()
                }
            })
        };

        let depth_view = self.depth_attachment.as_ref().map(|a| {
            let wt = a.texture.as_any().downcast_ref::<WgpuTexture>().unwrap();
            if let Some(face) = a.cube_map_face() {
                wt.wgpu_texture().create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: cubemap_face_to_layer(face),
                    array_layer_count: Some(1),
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            } else {
                wt.wgpu_view().clone()
            }
        });

        {
            // Backbuffer clears once per frame (flag set by swap_buffers, consumed on first draw).
            // Offscreen FBOs clear when their clear() was called.
            let has_stencil_aspect = df.map(format_has_stencil).unwrap_or(false);
            let (color_load, depth_load, stencil_load) = if self.is_backbuffer && server.backbuffer_needs_clear.replace(false) {
                (wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }), wgpu::LoadOp::Clear(1.0), if has_stencil_aspect { wgpu::LoadOp::Clear(0) } else { wgpu::LoadOp::Load })
            } else if !self.is_backbuffer && self.needs_clear.replace(false) {
                (wgpu::LoadOp::Clear(*self.pending_clear_color.borrow()), wgpu::LoadOp::Clear(*self.pending_clear_depth.borrow()), if has_stencil_aspect { wgpu::LoadOp::Clear(0) } else { wgpu::LoadOp::Load })
            } else {
                (wgpu::LoadOp::Load, wgpu::LoadOp::Load, wgpu::LoadOp::Load)
            };
            let color_att = color_view.as_ref().map(|view| wgpu::RenderPassColorAttachment { view, resolve_target: None, depth_slice: None, ops: wgpu::Operations { load: color_load, store: wgpu::StoreOp::Store } });
            let color_att_ref: &[Option<wgpu::RenderPassColorAttachment<'_>>] = if color_att.is_some() { &[color_att] } else { &[] };
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("DrawRP"),
                color_attachments: color_att_ref,
                depth_stencil_attachment: depth_view.as_ref().map(|v| wgpu::RenderPassDepthStencilAttachment { view: v, depth_ops: Some(wgpu::Operations { load: depth_load, store: wgpu::StoreOp::Store }), stencil_ops: Some(wgpu::Operations { load: stencil_load, store: wgpu::StoreOp::Store }) }),
                ..Default::default()
            });

            rp.set_viewport(viewport.x() as f32, viewport.y() as f32, viewport.w() as f32, viewport.h() as f32, 0.0, 1.0);
            rp.set_pipeline(&pipeline);
            if let Some(st) = &params.stencil_test { rp.set_stencil_reference(st.ref_value); }
            if let Some(bg) = &bind_group { rp.set_bind_group(0, bg, &[]); }
            let vbs = geo.vertex_buffers();
            for (i, vb) in vbs.iter().enumerate() { rp.set_vertex_buffer(i as u32, vb.slice(..)); }
            let geo_buf_count = vbs.len() as u32;
            for i in 0..extra_vert_count { rp.set_vertex_buffer(geo_buf_count + i, server.dummy_vertex_buffer.slice(..)); }
            let eb = geo.element_buffer();
            rp.set_index_buffer(eb.slice(..), wgpu::IndexFormat::Uint32);

            let ipe = geo.element_kind().index_per_element();
            let start_idx = (offset * ipe) as u32;
            let end_idx = ((offset + count) * ipe) as u32;
            rp.draw_indexed(start_idx..end_idx, 0, 0..instance_count);
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
            let fmt = wtex.format();
            let bps = fmt.block_copy_size(None).unwrap_or(4) as usize;
            let bytes_per_row = (width * bps).max(256);
            let padded_total = bytes_per_row * (height - 1) + width * bps;
            let buf = server.state.device.create_buffer(&wgpu::BufferDescriptor { label: Some("ReadPx"), size: padded_total as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let mut enc = server.state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ReadPxEnc") });
            enc.copy_texture_to_buffer(wgpu::TexelCopyTextureInfo { texture: wtex.wgpu_texture(), mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All }, wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bytes_per_row as u32), rows_per_image: Some(height as u32) } }, wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 });
            server.state.queue.submit(std::iter::once(enc.finish()));
            let slice = buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
            server.state.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            rx.recv().ok()?.ok()?;
            let mapped = slice.get_mapped_range();
            let unpadded_row = width * bps;
            let mut result = vec![0u8; unpadded_row * height];
            if bytes_per_row == unpadded_row {
                result.copy_from_slice(&mapped);
            } else {
                for y in 0..height {
                    let src_off = y * bytes_per_row;
                    let dst_off = y * unpadded_row;
                    result[dst_off..dst_off + unpadded_row].copy_from_slice(&mapped[src_off..src_off + unpadded_row]);
                }
            }
            drop(mapped);
            buf.unmap();
            Some(result)
        } else { None }
    }
    fn clear(&self, _viewport: Rect<i32>, color: Option<Color>, depth: Option<f32>, _stencil: Option<i32>) {
        if let Some(c) = color {
            *self.pending_clear_color.borrow_mut() = wgpu::Color {
                r: c.r as f64 / 255.0, g: c.g as f64 / 255.0,
                b: c.b as f64 / 255.0, a: c.a as f64 / 255.0,
            };
        }
        if let Some(d) = depth {
            *self.pending_clear_depth.borrow_mut() = d;
        }
        self.needs_clear.set(true);
    }
    fn draw(&self, geometry: &GpuGeometryBuffer, viewport: Rect<i32>, program: &GpuProgram, params: &DrawParameters, resources: &[ResourceBindGroup], element_range: ElementRange) -> Result<DrawCallStatistics, FrameworkError> {
        self.do_draw(1, geometry, viewport, program, params, resources, element_range)
    }
    fn draw_instances(&self, instance_count: usize, geometry: &GpuGeometryBuffer, viewport: Rect<i32>, program: &GpuProgram, params: &DrawParameters, resources: &[ResourceBindGroup], element_range: ElementRange) -> Result<DrawCallStatistics, FrameworkError> {
        self.do_draw(instance_count as u32, geometry, viewport, program, params, resources, element_range)
    }
}

/// Expected vertex attribute locations that the standard shaders may need but
/// geometry might not provide (e.g. boneWeights, boneIndices, vertexSecondTexCoord).
const EXTRA_VERTEX_ATTRS: &[(u32, wgpu::VertexFormat)] = &[
    (4, wgpu::VertexFormat::Float32x4), // boneWeights
    (5, wgpu::VertexFormat::Float32x4), // boneIndices
    (6, wgpu::VertexFormat::Float32x2), // vertexSecondTexCoord
];

/// Builds the full vertex buffer layout list, adding dummy entries for attributes
/// the shader expects but the geometry doesn't provide. Returns (layouts, extra_count).
fn build_vertex_layouts(geo: &WgpuGeometryBuffer) -> (Vec<wgpu::VertexBufferLayout<'static>>, u32) {
    let geo_layouts = geo.vertex_buffer_layouts();
    let mut provided = std::collections::HashSet::new();
    for layout in geo_layouts {
        for attr in layout.attributes { provided.insert(attr.shader_location); }
    }
    let mut all: Vec<wgpu::VertexBufferLayout<'static>> = geo_layouts.to_vec();
    let mut extra = 0u32;
    for &(loc, fmt) in EXTRA_VERTEX_ATTRS {
        if !provided.contains(&loc) {
            all.push(wgpu::VertexBufferLayout {
                array_stride: 0,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: Box::leak(vec![wgpu::VertexAttribute { format: fmt, offset: 0, shader_location: loc }].into_boxed_slice()),
            });
            extra += 1;
        }
    }
    (all, extra)
}

fn is_filterable_format(fmt: wgpu::TextureFormat) -> bool {
    !matches!(fmt, wgpu::TextureFormat::R32Float | wgpu::TextureFormat::Rg32Float | wgpu::TextureFormat::Rgba32Float
        | wgpu::TextureFormat::R8Uint | wgpu::TextureFormat::R16Uint | wgpu::TextureFormat::R32Uint
        | wgpu::TextureFormat::R8Sint | wgpu::TextureFormat::R16Sint | wgpu::TextureFormat::R32Sint)
}

fn create_bind_group(server: &WgpuGraphicsServer, program: &WgpuProgram, groups: &[ResourceBindGroup]) -> Option<wgpu::BindGroup> {
    let mut entries = Vec::new();
    let mut texture_formats: Vec<(usize, wgpu::TextureFormat)> = Vec::new();
    for group in groups {
        for binding in group.bindings {
            match binding {
                ResourceBinding::Texture { texture, sampler, binding: loc } => {
                    let wt = texture.as_any().downcast_ref::<WgpuTexture>()?;
                    let ws = sampler.as_any().downcast_ref::<WgpuSampler>()?;
                    texture_formats.push((*loc, wt.format()));
                    entries.push(wgpu::BindGroupEntry { binding: *loc as u32, resource: wgpu::BindingResource::TextureView(wt.wgpu_binding_view()) });
                    let sampler_res = if is_filterable_format(wt.format()) {
                        wgpu::BindingResource::Sampler(ws.wgpu_sampler())
                    } else {
                        wgpu::BindingResource::Sampler(server.non_filtering_sampler())
                    };
                    entries.push(wgpu::BindGroupEntry { binding: (*loc + 100) as u32, resource: sampler_res });
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
    let (bgl, _) = program.get_or_create_layouts(&texture_formats);
    Some(server.state.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("BG"), layout: &bgl, entries: &entries }))
}
