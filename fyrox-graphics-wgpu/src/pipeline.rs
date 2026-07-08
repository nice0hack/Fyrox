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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//! Pipeline catalog — pre-built render pipeline variants for known rendering paths.
//!
//! Instead of creating pipelines lazily at draw time, the catalog pre-builds
//! the set of variants a program needs upfront. Each [`VariantKey`] names a
//! rendering path (GBuffer, forward opaque, forward alpha, shadow, etc.).
//! The catalog is owned by a program; callers look up by variant name rather
//! than computing a hash key.

use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{BlendState, BlendComponent, CompareFunction, Device, Face, FragmentState, PipelineLayout, PrimitiveState, RenderPipeline, ShaderModule, TextureFormat, VertexBufferLayout, VertexState};

/// Named pipeline variants used by Fyrox's rendering pipeline.
///
/// Each variant encodes fixed rendering semantics (blend, depth write, cull).
/// The program owns a `PipelineCatalog` that maps these keys to built pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VariantKey {
    /// Opaque geometry rendered into the G-buffer.
    GBufferOpaque,
    /// Alpha-tested / masked geometry rendered into the G-buffer.
    GBufferMasked,
    /// Opaque geometry in the forward pass.
    ForwardOpaque,
    /// Alpha-blended geometry in the forward pass.
    ForwardAlphaBlend,
    /// Alpha-tested geometry in the forward pass.
    ForwardMaskedAlphaBlend,
    /// Shadow-casting directional/point/spot light depth pass.
    ShadowDepth,
    /// Shadow depth pass with front-face culling (for two-sided shadows).
    ShadowDepthCulledFront,
}

impl VariantKey {
    /// Returns the blend state for this variant, if any.
    pub fn blend_state(self) -> Option<BlendState> {
        match self {
            Self::GBufferOpaque
            | Self::GBufferMasked
            | Self::ForwardOpaque
            | Self::ShadowDepth
            | Self::ShadowDepthCulledFront => None,
            Self::ForwardAlphaBlend => Some(BlendState {
                color: BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            Self::ForwardMaskedAlphaBlend => Some(BlendState {
                color: BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
        }
    }

    /// Returns whether depth should be written for this variant.
    pub fn depth_write(self) -> bool {
        match self {
            Self::ForwardAlphaBlend => false,
            _ => true,
        }
    }

    /// Returns the depth compare function for this variant.
    pub fn depth_compare(self) -> CompareFunction {
        CompareFunction::Less
    }

    /// Returns the cull mode for this variant.
    pub fn cull_mode(self) -> Option<Face> {
        match self {
            Self::ShadowDepthCulledFront => Some(Face::Front),
            Self::ShadowDepth => Some(Face::Back),
            _ => Some(Face::Back),
        }
    }

    /// Returns whether color should be written for this variant.
    pub fn color_write(self) -> bool {
        match self {
            Self::ShadowDepth | Self::ShadowDepthCulledFront => false,
            _ => true,
        }
    }
}

/// A catalog of pre-built render pipeline variants for a single program.
///
/// Owned by a `WgpuProgram`; built once when the program is initialized.
#[derive(Default)]
pub struct PipelineCatalog {
    by_variant: HashMap<VariantKey, Arc<RenderPipeline>>,
}

impl PipelineCatalog {
    /// Looks up a pipeline by variant key.
    pub fn get(&self, key: VariantKey) -> Option<Arc<RenderPipeline>> {
        self.by_variant.get(&key).cloned()
    }

    /// Returns an iterator over all variants currently in the catalog.
    pub fn variants(&self) -> impl Iterator<Item = VariantKey> + '_ {
        self.by_variant.keys().copied()
    }

    /// Pre-builds all standard Fyrox variants for the given program and layout.
    ///
    /// Variants: `GBufferOpaque`, `GBufferMasked`, `ForwardOpaque`,
    /// `ForwardAlphaBlend`, `ForwardMaskedAlphaBlend`, `ShadowDepth`,
    /// `ShadowDepthCulledFront`.
    ///
    /// If you only need a subset (e.g. no shadow casters), build them individually
    /// with [`build_variant`](PipelineCatalog::build_variant) instead.
    pub fn prebuild_all(
        &mut self,
        device: &Device,
        shader: &ShaderModule,
        layout: &PipelineLayout,
        vertex_buffers: &[VertexBufferLayout<'static>],
        color_format: TextureFormat,
        depth_format: Option<TextureFormat>,
        sample_count: u32,
        topology: wgpu::PrimitiveTopology,
    ) {
        let variants = [
            VariantKey::GBufferOpaque,
            VariantKey::GBufferMasked,
            VariantKey::ForwardOpaque,
            VariantKey::ForwardAlphaBlend,
            VariantKey::ForwardMaskedAlphaBlend,
            VariantKey::ShadowDepth,
            VariantKey::ShadowDepthCulledFront,
        ];

        for &variant in &variants {
            let pipeline = self.build_variant(
                device,
                shader,
                layout,
                vertex_buffers,
                color_format,
                depth_format,
                sample_count,
                topology,
                variant,
            );
            self.by_variant.insert(variant, pipeline);
        }
    }

    /// Builds a single pipeline variant and inserts it into the catalog.
    pub fn build_variant(
        &mut self,
        device: &Device,
        shader: &ShaderModule,
        layout: &PipelineLayout,
        vertex_buffers: &[VertexBufferLayout<'static>],
        color_format: TextureFormat,
        depth_format: Option<TextureFormat>,
        sample_count: u32,
        topology: wgpu::PrimitiveTopology,
        variant: VariantKey,
    ) -> Arc<RenderPipeline> {
        let pipeline = Self::build_pipeline(
            device,
            shader,
            layout,
            vertex_buffers,
            color_format,
            depth_format,
            sample_count,
            topology,
            variant,
        );
        let arc = Arc::new(pipeline);
        self.by_variant.insert(variant, arc.clone());
        arc
    }

    /// Constructs a `RenderPipeline` for the given variant.
    pub fn build_pipeline(
        device: &Device,
        shader: &ShaderModule,
        layout: &PipelineLayout,
        vertex_buffers: &[VertexBufferLayout<'static>],
        color_format: TextureFormat,
        depth_format: Option<TextureFormat>,
        sample_count: u32,
        topology: wgpu::PrimitiveTopology,
        variant: VariantKey,
    ) -> RenderPipeline {
        let blend = variant.blend_state();
        let color_write = if variant.color_write() {
            wgpu::ColorWrites::ALL
        } else {
            wgpu::ColorWrites::empty()
        };

        let vertex = VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: vertex_buffers,
            compilation_options: Default::default(),
        };

        let fragment = FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend,
                write_mask: color_write,
            })],
            compilation_options: Default::default(),
        };

        let primitive = PrimitiveState {
            topology,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: variant.cull_mode(),
            ..Default::default()
        };

        let depth_stencil = depth_format.map(|fmt| wgpu::DepthStencilState {
            format: fmt,
            depth_write_enabled: Some(variant.depth_write()),
            depth_compare: Some(variant.depth_compare()),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        let desc = wgpu::RenderPipelineDescriptor {
            label: Some(match variant {
                VariantKey::GBufferOpaque => "GBufferOpaque",
                VariantKey::GBufferMasked => "GBufferMasked",
                VariantKey::ForwardOpaque => "ForwardOpaque",
                VariantKey::ForwardAlphaBlend => "ForwardAlphaBlend",
                VariantKey::ForwardMaskedAlphaBlend => "ForwardMaskedAlphaBlend",
                VariantKey::ShadowDepth => "ShadowDepth",
                VariantKey::ShadowDepthCulledFront => "ShadowDepthCulledFront",
            }),
            layout: Some(layout),
            vertex,
            primitive,
            depth_stencil,
            fragment: Some(fragment),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        };

        device.create_render_pipeline(&desc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_key_properties_gbuffer_opaque() {
        let v = VariantKey::GBufferOpaque;
        assert!(v.blend_state().is_none());
        assert!(v.depth_write());
        assert_eq!(v.cull_mode(), Some(Face::Back));
        assert!(v.color_write());
    }

    #[test]
    fn variant_key_properties_forward_alpha_blend() {
        let v = VariantKey::ForwardAlphaBlend;
        assert!(v.blend_state().is_some());
        assert!(!v.depth_write());
        assert_eq!(v.cull_mode(), Some(Face::Back));
        assert!(v.color_write());
    }

    #[test]
    fn variant_key_properties_shadow_depth() {
        let v = VariantKey::ShadowDepth;
        assert!(v.blend_state().is_none());
        assert!(v.depth_write());
        assert_eq!(v.cull_mode(), Some(Face::Back));
        assert!(!v.color_write());
    }

    #[test]
    fn variant_key_properties_shadow_depth_culled_front() {
        let v = VariantKey::ShadowDepthCulledFront;
        assert_eq!(v.cull_mode(), Some(Face::Front));
    }

    #[test]
    fn pipeline_catalog_default_is_empty() {
        let catalog: PipelineCatalog = Default::default();
        assert!(catalog.get(VariantKey::ForwardOpaque).is_none());
    }
}
