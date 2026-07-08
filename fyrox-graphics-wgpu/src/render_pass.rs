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

//! Named render pass types and their descriptors.
//!
//! Fyrox uses a fixed set of named render passes with well-defined ordering
//! and clear semantics. `RenderPass` is the canonical name for each pass.
//! `PassDescriptor` provides the metadata (name string, clear values) needed
//! to configure a framebuffer for that pass.

/// A named render pass in Fyrox's rendering pipeline.
///
/// Each variant corresponds to a distinct rendering phase with a specific
/// role in the deferred/forward pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderPass {
    /// G-Buffer construction pass — renders scene geometry into the
    /// geometry buffer (albedo, normal, roughness, etc.).
    GBuffer,
    /// Shadow map rendering pass — renders depth from light POV.
    Shadow,
    /// Deferred lighting pass — reads G-Buffer and computes lighting.
    Lighting,
    /// Forward opaque pass — renders opaque geometry with lighting applied.
    ForwardOpaque,
    /// Forward transparent pass — alpha-blended geometry (glass, water, etc.).
    ForwardTransparent,
    /// Outline pass — renders selected-object outlines using scene depth.
    Outline,
    /// Post-processing pass — bloom, tone-mapping, FXAA, etc.
    PostProcess,
    /// UI / editor pass — renders UI elements on top of everything.
    UI,
}

/// Metadata for configuring a render pass.
///
/// `PassDescriptor::for_pass(p)` gives the canonical clear values and name
/// for each `RenderPass` variant.
#[derive(Debug, Clone)]
pub struct PassDescriptor {
    /// Human-readable pass name for debugging / GPU profiling.
    pub name: &'static str,
    /// RGBA clear color, if this pass clears the color attachment.
    pub clear_color: Option<[f32; 4]>,
    /// Depth clear value, if this pass clears the depth attachment.
    pub clear_depth: Option<f32>,
}

impl PassDescriptor {
    /// Returns the canonical `PassDescriptor` for a render pass.
    pub fn for_pass(pass: RenderPass) -> Self {
        match pass {
            RenderPass::GBuffer => Self {
                name: "GBuffer",
                clear_color: None,
                clear_depth: Some(1.0),
            },
            RenderPass::Shadow => Self {
                name: "Shadow",
                clear_color: None,
                clear_depth: Some(1.0),
            },
            RenderPass::Lighting => Self {
                name: "Lighting",
                clear_color: None,
                clear_depth: None,
            },
            RenderPass::ForwardOpaque => Self {
                name: "ForwardOpaque",
                clear_color: None,
                clear_depth: None,
            },
            RenderPass::ForwardTransparent => Self {
                name: "ForwardTransparent",
                clear_color: None,
                clear_depth: None,
            },
            RenderPass::Outline => Self {
                name: "Outline",
                clear_color: None,
                clear_depth: None,
            },
            RenderPass::PostProcess => Self {
                name: "PostProcess",
                clear_color: Some([0.0, 0.0, 0.0, 1.0]),
                clear_depth: None,
            },
            RenderPass::UI => Self {
                name: "UI",
                clear_color: Some([0.0, 0.0, 0.0, 1.0]),
                clear_depth: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gbuffer_clears_depth() {
        let p = PassDescriptor::for_pass(RenderPass::GBuffer);
        assert_eq!(p.clear_depth, Some(1.0));
        assert_eq!(p.clear_color, None);
        assert_eq!(p.name, "GBuffer");
    }

    #[test]
    fn lighting_does_not_clear() {
        let p = PassDescriptor::for_pass(RenderPass::Lighting);
        assert_eq!(p.clear_color, None);
        assert_eq!(p.clear_depth, None);
    }

    #[test]
    fn ui_clears_to_black() {
        let p = PassDescriptor::for_pass(RenderPass::UI);
        assert_eq!(p.clear_color, Some([0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn shadow_clears_depth() {
        let p = PassDescriptor::for_pass(RenderPass::Shadow);
        assert_eq!(p.clear_depth, Some(1.0));
        assert_eq!(p.clear_color, None);
    }

    #[test]
    fn render_pass_equality() {
        assert_eq!(RenderPass::GBuffer, RenderPass::GBuffer);
        assert_ne!(RenderPass::GBuffer, RenderPass::Shadow);
    }
}
