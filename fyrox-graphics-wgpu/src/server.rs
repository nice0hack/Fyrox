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
use crate::framebuffer::WgpuFrameBuffer;
use crate::geometry_buffer::WgpuGeometryBuffer;
use crate::program::{WgpuProgram, WgpuShader};
use crate::query::WgpuQuery;
use crate::read_buffer::WgpuAsyncReadBuffer;
use crate::sampler::WgpuSampler;
use crate::texture::WgpuTexture;
use fyrox_core::futures::executor::block_on;
use fyrox_core::log::{Log, MessageKind};
use fyrox_graphics::buffer::{GpuBuffer, GpuBufferDescriptor};
use fyrox_graphics::error::FrameworkError;
use fyrox_graphics::framebuffer::{Attachment, GpuFrameBuffer};
use fyrox_graphics::geometry_buffer::{GpuGeometryBuffer, GpuGeometryBufferDescriptor};
use fyrox_graphics::gpu_program::{GpuProgram, GpuShader, ShaderKind, ShaderResourceDefinition};
use fyrox_graphics::gpu_texture::{GpuTexture, GpuTextureDescriptor, GpuTextureKind, GpuTextureTrait};
use fyrox_graphics::query::GpuQuery;
use fyrox_graphics::read_buffer::GpuAsyncReadBuffer;
use fyrox_graphics::sampler::{GpuSampler, GpuSamplerDescriptor};
use fyrox_graphics::server::{
    GraphicsServer, ServerCapabilities, ServerMemoryUsage, SharedGraphicsServer,
};
use fyrox_graphics::stats::PipelineStatistics;
use fyrox_graphics::{PolygonFace, PolygonFillMode};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::{Arc, RwLock};
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
    pub named_objects: bool,
    pub msaa_sample_count: u32,
    pub pipeline_cache: RefCell<HashMap<u64, wgpu::RenderPipeline>>,
    /// Caches `wgpu::BindGroup` objects by resource pointer key to avoid
    /// re-creating identical bind groups every draw call.
    pub bind_group_cache: RefCell<crate::bind_group_cache::BindGroupCache>,
    weak_self: RefCell<Option<Weak<WgpuGraphicsServer>>>,
    pub memory_usage: RefCell<ServerMemoryUsage>,
    pipeline_statistics: RefCell<PipelineStatistics>,
    /// Small buffer bound to extra vertex slots when geometry lacks attributes the shader expects.
    pub dummy_vertex_buffer: wgpu::Buffer,
    /// Non-filtering sampler for textures with non-filterable formats (e.g. R32Float).
    non_filtering_sampler: wgpu::Sampler,
    /// Holds the acquired surface frame between do_draw and swap_buffers.
    pub current_frame: RefCell<Option<wgpu::SurfaceTexture>>,
    /// Whether the backbuffer needs clearing at the start of the next frame.
    pub backbuffer_needs_clear: Cell<bool>,
    /// Cached depth-stencil texture for the backbuffer, with its (width, height).
    backbuffer_depth_stencil: RefCell<Option<(u32, u32, GpuTexture)>>,
    /// Current polygon fill mode, used to trigger pipeline recreation when it changes.
    pub polygon_fill_mode: Cell<fyrox_graphics::PolygonFillMode>,
    /// Current debug group label, if any.
    current_debug_group: RefCell<Option<String>>,
}

/// Controls which GPU is preferred when selecting a WGPU adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdapterPreference {
    /// Automatically select a suitable adapter. Prefers high-performance GPUs.
    #[default]
    Auto,
    /// Prefer the most powerful available GPU (discrete GPU if present).
    PreferHighPerformance,
    /// Prefer the least powerful GPU (integrated GPU) to conserve power.
    PreferLowPower,
}

fn power_preference(pref: AdapterPreference) -> wgpu::PowerPreference {
    match pref {
        AdapterPreference::Auto | AdapterPreference::PreferHighPerformance => {
            wgpu::PowerPreference::HighPerformance
        }
        AdapterPreference::PreferLowPower => wgpu::PowerPreference::LowPower,
    }
}

/// Attempts to find a suitable WGPU adapter, with fallback tiers.
async fn select_adapter<'a>(
    instance: &'a wgpu::Instance,
    surface: &'a wgpu::Surface<'a>,
    preference: AdapterPreference,
) -> Option<wgpu::Adapter> {
    // Try preferred adapter first.
    if let Ok(adapter) = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: power_preference(preference),
            compatible_surface: Some(surface),
            force_fallback_adapter: false,
        })
        .await
    {
        return Some(adapter);
    }

    // Fall back to LowPower if HighPerformance was requested.
    if preference == AdapterPreference::PreferHighPerformance {
        if let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: Some(surface),
                force_fallback_adapter: false,
            })
            .await
        {
            return Some(adapter);
        }
    }

    // Last resort: force fallback adapter.
    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: power_preference(preference),
            compatible_surface: Some(surface),
            force_fallback_adapter: true,
        })
        .await
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_maps_to_high_performance() {
        assert_eq!(
            power_preference(AdapterPreference::Auto),
            wgpu::PowerPreference::HighPerformance
        );
    }

    #[test]
    fn prefer_high_performance_maps_to_high_performance() {
        assert_eq!(
            power_preference(AdapterPreference::PreferHighPerformance),
            wgpu::PowerPreference::HighPerformance
        );
    }

    #[test]
    fn prefer_low_power_maps_to_low_power() {
        assert_eq!(
            power_preference(AdapterPreference::PreferLowPower),
            wgpu::PowerPreference::LowPower
        );
    }
}

impl WgpuGraphicsServer {
    pub fn new(
        vsync: bool,
        _msaa_sample_count: Option<u8>,
        window_target: &ActiveEventLoop,
        window_attributes: WindowAttributes,
        named_objects: bool,
        adapter_preference: AdapterPreference,
    ) -> Result<(Window, SharedGraphicsServer), FrameworkError> {
        let window = window_target
            .create_window(window_attributes)
            .map_err(|e| FrameworkError::Custom(format!("Failed to create window: {e}")))?;
        let size = window.inner_size();

        #[cfg(not(target_arch = "wasm32"))]
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(
            Box::new(window_target.owned_display_handle()),
        ));

        #[cfg(target_arch = "wasm32")]
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        #[cfg(not(target_arch = "wasm32"))]
        let surface = unsafe {
            let target = wgpu::SurfaceTargetUnsafe::from_window(&window)
                .map_err(|e| FrameworkError::Custom(format!("Failed to get window handle: {e}")))?;
            instance
                .create_surface_unsafe(target)
                .map_err(|e| FrameworkError::Custom(format!("Failed to create surface: {e}")))?
        };

        #[cfg(target_arch = "wasm32")]
        let surface = {
            use fyrox_core::wasm_bindgen::JsCast;
            use winit::platform::web::WindowExtWebSys;
            let canvas = window.canvas().unwrap();
            let web_window = fyrox_core::web_sys::window().unwrap();
            let document = web_window.document().unwrap();
            let body = document.body().unwrap();
            body.append_child(&canvas).expect("Append canvas to HTML body");
            instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas))
                .map_err(|e| FrameworkError::Custom(format!("Failed to create surface: {e}")))?
        };

        let adapter = block_on(select_adapter(&instance, &surface, adapter_preference))
            .ok_or_else(|| FrameworkError::Custom("No suitable WGPU adapter found".to_string()))?;

        let adapter_info = adapter.get_info();
        Log::writeln(
            MessageKind::Information,
            format!(
                "WGPU adapter: {} (backend: {:?})",
                adapter_info.name, adapter_info.backend
            ),
        );

        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            },
        ))
        .map_err(|e| FrameworkError::Custom(format!("Failed to request device: {e}")))?;

        let surface_caps = surface.get_capabilities(&adapter);
        // Linear working space: swapchain is a linear format. The OS compositor
        // does the final linear→sRGB encode for display. This matches the GL
        // backend (which never enables GL_FRAMEBUFFER_SRGB) and avoids the
        // wgpu surface-level auto-encode that produced double-encoding under
        // the previous sRGB-surface convention. See docs/wgpu-color-space.md.
        let surface_format = crate::color_space::surface_format_for(&surface_caps.formats)
            .ok_or_else(|| FrameworkError::Custom("Surface has no supported formats".into()))?;

        let present_mode = if vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync };

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // TODO: Force `msaa_sample_count` to 1 in the wgpu backend. Full MSAA support requires creating multisampled render targets and resolve targets, which is a larger feature.
        let msaa = _msaa_sample_count.unwrap_or(1).max(1) as u32;

        let non_filtering_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("NonFilteringSampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let dummy_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DummyVB"),
            size: 16, // enough for vec4f
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let server = Rc::new(Self {
            state: Arc::new(WgpuState { instance, adapter, device, queue }),
            surface,
            surface_config: RwLock::new(surface_config),
            named_objects,
            msaa_sample_count: msaa,
            pipeline_cache: RefCell::new(HashMap::new()),
            bind_group_cache: Default::default(),
            weak_self: RefCell::new(None),
            memory_usage: RefCell::new(ServerMemoryUsage::default()),
            pipeline_statistics: RefCell::new(PipelineStatistics::default()),
            dummy_vertex_buffer,
            non_filtering_sampler,
            polygon_fill_mode: Cell::new(fyrox_graphics::PolygonFillMode::Fill),
            current_frame: RefCell::new(None),
            backbuffer_needs_clear: Cell::new(true),
            backbuffer_depth_stencil: RefCell::new(None),
            current_debug_group: RefCell::new(None),
        });

        *server.weak_self.borrow_mut() = Some(Rc::downgrade(&server));

        Ok((window, server))
    }

    pub fn weak_ref(&self) -> Weak<WgpuGraphicsServer> {
        self.weak_self.borrow().clone().unwrap()
    }
    pub fn non_filtering_sampler(&self) -> &wgpu::Sampler {
        &self.non_filtering_sampler
    }
    pub fn current_debug_group_label(&self) -> Option<String> {
        self.current_debug_group.borrow().clone()
    }
}

impl GraphicsServer for WgpuGraphicsServer {
    fn create_buffer(&self, desc: GpuBufferDescriptor) -> Result<GpuBuffer, FrameworkError> {
        Ok(GpuBuffer(Rc::new(WgpuBuffer::new(self, desc)?)))
    }
    fn create_texture(&self, desc: GpuTextureDescriptor) -> Result<GpuTexture, FrameworkError> {
        Ok(GpuTexture(Rc::new(WgpuTexture::new(self, desc)?)))
    }
    fn create_sampler(&self, desc: GpuSamplerDescriptor) -> Result<GpuSampler, FrameworkError> {
        Ok(GpuSampler(Rc::new(WgpuSampler::new(self, desc)?)))
    }
    fn create_frame_buffer(&self, depth: Option<Attachment>, colors: Vec<Attachment>) -> Result<GpuFrameBuffer, FrameworkError> {
        Ok(GpuFrameBuffer(Rc::new(WgpuFrameBuffer::new(self, depth, colors)?)))
    }
    fn back_buffer(&self) -> GpuFrameBuffer {
        let config = self.surface_config.read().unwrap();
        let (w, h) = (config.width, config.height);
        drop(config);

        let mut cache = self.backbuffer_depth_stencil.borrow_mut();
        let needs_recreate = match cache.as_ref() {
            Some((cw, ch, _)) => *cw != w || *ch != h,
            None => true,
        };
        if needs_recreate {
            if w > 0 && h > 0 {
                match self.create_2d_render_target(
                    "BackbufferDepthStencil",
                    fyrox_graphics::gpu_texture::PixelKind::D24S8,
                    w as usize,
                    h as usize,
                ) {
                    Ok(tex) => { *cache = Some((w, h, tex)); }
                    Err(e) => {
                        log::warn!("Failed to create backbuffer depth-stencil: {e}");
                        *cache = None;
                    }
                }
            } else {
                *cache = None;
            }
        }
        let depth_attachment = cache.as_ref().map(|(_, _, tex)| Attachment::depth_stencil(tex.clone()));
        GpuFrameBuffer(Rc::new(WgpuFrameBuffer::backbuffer(self, depth_attachment)))
    }
    fn create_query(&self) -> Result<GpuQuery, FrameworkError> {
        Ok(GpuQuery(Rc::new(WgpuQuery::new(self)?)))
    }
    fn create_shader(&self, name: String, kind: ShaderKind, source: String, resources: &[ShaderResourceDefinition], line_offset: isize) -> Result<GpuShader, FrameworkError> {
        Ok(GpuShader(Rc::new(WgpuShader::new(self, name, kind, source, resources, line_offset)?)))
    }
    fn create_program(&self, name: &str, vs: String, vs_offset: isize, fs: String, fs_offset: isize, resources: &[ShaderResourceDefinition]) -> Result<GpuProgram, FrameworkError> {
        Ok(GpuProgram(Rc::new(WgpuProgram::from_source(self, name, vs, vs_offset, fs, fs_offset, resources)?)))
    }
    fn create_program_from_shaders(&self, name: &str, vs: &GpuShader, fs: &GpuShader, resources: &[ShaderResourceDefinition]) -> Result<GpuProgram, FrameworkError> {
        Ok(GpuProgram(Rc::new(WgpuProgram::from_shaders(self, name, vs, fs, resources)?)))
    }
    fn create_async_read_buffer(&self, name: &str, pixel_size: usize, pixel_count: usize) -> Result<GpuAsyncReadBuffer, FrameworkError> {
        Ok(GpuAsyncReadBuffer(Rc::new(WgpuAsyncReadBuffer::new(self, name, pixel_size, pixel_count)?)))
    }
    fn create_geometry_buffer(&self, desc: GpuGeometryBufferDescriptor) -> Result<GpuGeometryBuffer, FrameworkError> {
        Ok(GpuGeometryBuffer(Rc::new(WgpuGeometryBuffer::new(self, desc)?)))
    }
    fn weak(&self) -> Weak<dyn GraphicsServer> {
        self.weak_ref() as Weak<dyn GraphicsServer>
    }
    fn flush(&self) { self.state.queue.submit(std::iter::empty()); }
    fn finish(&self) { self.state.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok(); }
    fn invalidate_resource_bindings_cache(&self) { *self.pipeline_statistics.borrow_mut() = Default::default(); }
    fn pipeline_statistics(&self) -> PipelineStatistics { *self.pipeline_statistics.borrow() }
    fn swap_buffers(&self) -> Result<(), FrameworkError> {
        if let Some(frame) = self.current_frame.borrow_mut().take() {
            frame.present();
        }
        self.backbuffer_needs_clear.set(true);
        Ok(())
    }
    fn set_frame_size(&self, new_size: (u32, u32)) {
        if new_size.0 > 0 && new_size.1 > 0 {
            // Clear any pending frame before reconfiguring surface to avoid semaphore issues.
            // The semaphore from a pending frame may still be in use by the GPU.
            if let Some(frame) = self.current_frame.borrow_mut().take() {
                frame.present();
            }

            let mut config = self.surface_config.write().unwrap();
            let old_format = config.format;
            config.width = new_size.0;
            config.height = new_size.1;
            self.surface.configure(&self.state.device, &config);

            // Invalidate depth stencil cache - it is sized for the OLD surface dimensions.
            // Next call to back_buffer() will recreate it with correct size.
            *self.backbuffer_depth_stencil.borrow_mut() = None;

            // Invalidate pipeline cache if the surface format changed (e.g. DPI/monitor switch).
            if config.format != old_format {
                self.pipeline_cache.borrow_mut().clear();
            }
        }
    }
    fn capabilities(&self) -> ServerCapabilities {
        let limits = self.state.device.limits();
        ServerCapabilities {
            max_uniform_block_size: limits.max_uniform_buffer_binding_size as usize,
            uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment as usize,
            max_lod_bias: 16.0,
        }
    }
    fn set_polygon_fill_mode(&self, _face: PolygonFace, mode: PolygonFillMode) {
        if self.polygon_fill_mode.get() != mode {
            self.polygon_fill_mode.set(mode);
            self.pipeline_cache.borrow_mut().clear();
        }
    }
    fn generate_mipmap(&self, texture: &GpuTexture) {
        let Some(wgpu_tex) = texture.as_any().downcast_ref::<WgpuTexture>() else {
            log::warn!("generate_mipmap: texture is not a WgpuTexture, skipping");
            return;
        };

        let src_texture = wgpu_tex.wgpu_texture();
        let format = src_texture.format();
        let kind = wgpu_tex.kind();
        let mip_level_count = src_texture.mip_level_count();

        // Only 2D Rectangle textures are supported for now
        let GpuTextureKind::Rectangle { .. } = kind else {
            log::warn!("generate_mipmap: only 2D Rectangle textures are supported");
            return;
        };

        if mip_level_count <= 1 {
            return;
        }

        // Use wgpu's built-in TextureBlitter with linear filtering for proper downsampling
        let blitter = wgpu::util::TextureBlitterBuilder::new(&self.state.device, format)
            .sample_type(wgpu::FilterMode::Linear)
            .build();

        let mut encoder = self.state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mipmap_gen_encoder"),
        });

        for mip_level in 1..mip_level_count {
            let src_view = src_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("mipmap_gen_src_view"),
                format: None,
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip_level - 1,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None,
                ..Default::default()
            });

            let dst_view = src_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("mipmap_gen_dst_view"),
                format: None,
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip_level,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None,
                ..Default::default()
            });

            blitter.copy(&self.state.device, &mut encoder, &src_view, &dst_view);
        }

        let cmd_buffer = encoder.finish();
        self.state.queue.submit(std::iter::once(cmd_buffer));
    }
    fn memory_usage(&self) -> ServerMemoryUsage { self.memory_usage.borrow().clone() }
    fn push_debug_group(&self, name: &str) {
        *self.current_debug_group.borrow_mut() = Some(name.to_string());
    }
    fn pop_debug_group(&self) {
        *self.current_debug_group.borrow_mut() = None;
    }
}
