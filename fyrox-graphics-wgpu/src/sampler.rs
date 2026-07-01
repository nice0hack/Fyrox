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

use crate::server::WgpuGraphicsServer;
use fyrox_graphics::{
    error::FrameworkError,
    sampler::{GpuSamplerDescriptor, GpuSamplerTrait, MagnificationFilter, MinificationFilter, WrapMode},
};
use std::fmt::Debug;
use std::rc::Weak;

fn min_filter_to_wgpu(f: MinificationFilter) -> wgpu::FilterMode {
    match f {
        MinificationFilter::Nearest | MinificationFilter::NearestMipMapNearest | MinificationFilter::NearestMipMapLinear => wgpu::FilterMode::Nearest,
        _ => wgpu::FilterMode::Linear,
    }
}

fn mag_filter_to_wgpu(f: MagnificationFilter) -> wgpu::FilterMode {
    match f { MagnificationFilter::Nearest => wgpu::FilterMode::Nearest, _ => wgpu::FilterMode::Linear }
}

fn mipmap_filter_to_wgpu(f: MinificationFilter) -> wgpu::MipmapFilterMode {
    match f {
        MinificationFilter::NearestMipMapLinear | MinificationFilter::LinearMipMapLinear => wgpu::MipmapFilterMode::Linear,
        _ => wgpu::MipmapFilterMode::Nearest,
    }
}

fn wrap_mode_to_wgpu(m: WrapMode) -> wgpu::AddressMode {
    match m {
        WrapMode::Repeat => wgpu::AddressMode::Repeat,
        WrapMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        WrapMode::ClampToBorder => wgpu::AddressMode::ClampToBorder,
        WrapMode::MirroredRepeat | WrapMode::MirrorClampToEdge => wgpu::AddressMode::MirrorRepeat,
    }
}

pub struct WgpuSampler {
    _server: Weak<WgpuGraphicsServer>,
    sampler: wgpu::Sampler,
}

impl Debug for WgpuSampler { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.debug_struct("WgpuSampler").finish() } }
impl GpuSamplerTrait for WgpuSampler {}

impl WgpuSampler {
    pub fn new(server: &WgpuGraphicsServer, desc: GpuSamplerDescriptor) -> Result<Self, FrameworkError> {
        let sampler = server.state.device.create_sampler(&wgpu::SamplerDescriptor {
            label: if server.named_objects { Some("Sampler") } else { None },
            address_mode_u: wrap_mode_to_wgpu(desc.s_wrap_mode),
            address_mode_v: wrap_mode_to_wgpu(desc.t_wrap_mode),
            address_mode_w: wrap_mode_to_wgpu(desc.r_wrap_mode),
            mag_filter: mag_filter_to_wgpu(desc.mag_filter),
            min_filter: min_filter_to_wgpu(desc.min_filter),
            mipmap_filter: mipmap_filter_to_wgpu(desc.min_filter),
            lod_min_clamp: desc.min_lod.max(0.0),
            lod_max_clamp: desc.max_lod.max(0.0),
            anisotropy_clamp: desc.anisotropy.clamp(1.0, 16.0) as u16,
            compare: None,
            border_color: None,
        });
        Ok(Self { _server: server.weak_ref(), sampler })
    }

    pub fn wgpu_sampler(&self) -> &wgpu::Sampler { &self.sampler }
}
