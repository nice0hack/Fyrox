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

use std::cell::{Cell, RefCell};
use fyrox_core::futures::executor::block_on;
use fyrox_graphics::buffer::{GpuBuffer, GpuBufferDescriptor};
use fyrox_graphics::error::FrameworkError;
use fyrox_graphics::framebuffer::{Attachment, GpuFrameBuffer};
use fyrox_graphics::geometry_buffer::{GpuGeometryBuffer, GpuGeometryBufferDescriptor};
use fyrox_graphics::gpu_program::{GpuProgram, GpuShader, ShaderKind, ShaderResourceDefinition};
use fyrox_graphics::gpu_texture::{GpuTexture, GpuTextureDescriptor};
use fyrox_graphics::query::GpuQuery;
use fyrox_graphics::read_buffer::GpuAsyncReadBuffer;
use fyrox_graphics::sampler::{GpuSampler, GpuSamplerDescriptor};
use fyrox_graphics::server::{
    GraphicsServer, ServerCapabilities, ServerMemoryUsage, SharedGraphicsServer,
};
use fyrox_graphics::stats::PipelineStatistics;
use fyrox_graphics::{PolygonFace, PolygonFillMode};
use std::rc::{Rc, Weak};
use std::sync::{Arc, RwLock};
use raw_window_handle::HasDisplayHandle;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

pub struct WgpuState {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

pub struct WgpuGraphicsServer {
    pub state: Arc<WgpuState>,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: RwLock<wgpu::SurfaceConfiguration>,
}

impl WgpuGraphicsServer {
    pub fn new(
        vsync: bool,
        msaa_sample_count: Option<u8>,
        window_target: &ActiveEventLoop,
        mut window_attributes: WindowAttributes,
        named_objects: bool,
    ) -> Result<(Window, SharedGraphicsServer), FrameworkError> {
        let window = window_target
            .create_window(window_attributes)
            .map_err(|e| FrameworkError::Custom(format!("Failed to create window: {e}")))?;
        let size = window.inner_size();

        let display_handle = window_target
            .display_handle()
            .map_err(|e| FrameworkError::Custom(format!("Failed to get display handle: {e}")))?;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(Box::new(window_target.owned_display_handle())));

        // Создаем поверхность отрисовки (Surface)
        let surface = unsafe {
            let target = wgpu::SurfaceTargetUnsafe::from_window(&window)
                .map_err(|e| FrameworkError::Custom(format!("Failed to get window handle: {e}")))?;
            instance
                .create_surface_unsafe(target)
                .map_err(|e| FrameworkError::Custom(format!("Failed to create surface: {e}")))?
        };

        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .map_err(|e| FrameworkError::Custom(format!("No suitable WGPU adapter found: {e}")))?;

        // Инициализируем логическое устройство и очередь команд
        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we'll have to disable some.
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            },
        ))
        .map_err(|e| FrameworkError::Custom(format!("Failed to request device: {e}")))?;

        let surface_caps = surface.get_capabilities(&adapter);

        let Some(surface_format) = surface_caps.formats.first().copied() else {
            return Err(FrameworkError::Custom(
                "Surface has no supported formats".into(),
            ));
        };

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);

        let state = Self {
            state: Arc::new(WgpuState {
                instance,
                adapter,
                device,
                queue,
            }),
            surface,
            surface_config: RwLock::new(surface_config),
        };

        let shared = Rc::new(state);

        Ok((window, shared))
    }
}

impl GraphicsServer for WgpuGraphicsServer {
    fn create_buffer(&self, desc: GpuBufferDescriptor) -> Result<GpuBuffer, FrameworkError> {
        todo!()
    }

    fn create_texture(&self, desc: GpuTextureDescriptor) -> Result<GpuTexture, FrameworkError> {
        todo!()
    }

    fn create_sampler(&self, desc: GpuSamplerDescriptor) -> Result<GpuSampler, FrameworkError> {
        todo!()
    }

    fn create_frame_buffer(
        &self,
        depth_attachment: Option<Attachment>,
        color_attachments: Vec<Attachment>,
    ) -> Result<GpuFrameBuffer, FrameworkError> {
        todo!()
    }

    fn back_buffer(&self) -> GpuFrameBuffer {
        todo!()
    }

    fn create_query(&self) -> Result<GpuQuery, FrameworkError> {
        todo!()
    }

    fn create_shader(
        &self,
        name: String,
        kind: ShaderKind,
        source: String,
        resources: &[ShaderResourceDefinition],
        line_offset: isize,
    ) -> Result<GpuShader, FrameworkError> {
        todo!()
    }

    fn create_program(
        &self,
        name: &str,
        vertex_source: String,
        vertex_source_line_offset: isize,
        fragment_source: String,
        fragment_source_line_offset: isize,
        resources: &[ShaderResourceDefinition],
    ) -> Result<GpuProgram, FrameworkError> {
        todo!()
    }

    fn create_program_from_shaders(
        &self,
        name: &str,
        vertex_shader: &GpuShader,
        fragment_shader: &GpuShader,
        resources: &[ShaderResourceDefinition],
    ) -> Result<GpuProgram, FrameworkError> {
        todo!()
    }

    fn create_async_read_buffer(
        &self,
        name: &str,
        pixel_size: usize,
        pixel_count: usize,
    ) -> Result<GpuAsyncReadBuffer, FrameworkError> {
        todo!()
    }

    fn create_geometry_buffer(
        &self,
        desc: GpuGeometryBufferDescriptor,
    ) -> Result<GpuGeometryBuffer, FrameworkError> {
        todo!()
    }

    fn weak(&self) -> Weak<dyn GraphicsServer> {
        todo!()
    }

    fn flush(&self) {
        todo!()
    }

    fn finish(&self) {
        todo!()
    }

    fn invalidate_resource_bindings_cache(&self) {
        todo!()
    }

    fn pipeline_statistics(&self) -> PipelineStatistics {
        todo!()
    }

    fn swap_buffers(&self) -> Result<(), FrameworkError> {
        todo!()
    }

    fn set_frame_size(&self, new_size: (u32, u32)) {
        todo!()
    }

    fn capabilities(&self) -> ServerCapabilities {
        todo!()
    }

    fn set_polygon_fill_mode(&self, polygon_face: PolygonFace, polygon_fill_mode: PolygonFillMode) {
        todo!()
    }

    fn generate_mipmap(&self, texture: &GpuTexture) {
        todo!()
    }

    fn memory_usage(&self) -> ServerMemoryUsage {
        todo!()
    }

    fn push_debug_group(&self, name: &str) {
        todo!()
    }

    fn pop_debug_group(&self) {
        todo!()
    }
}
