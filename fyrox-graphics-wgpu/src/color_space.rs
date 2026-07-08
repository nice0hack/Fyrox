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

//! Color-space convention for the wgpu backend.
//!
//! ## Decision
//!
//! Fyrox uses **linear working space** for all GPU compute and intermediate
//! storage. The final linear→sRGB encode happens exactly once, by the OS
//! compositor / display path, when the linear output reaches the screen.
//!
//! ## Why
//!
//! The OpenGL backend's default framebuffer was linear (`GL_FRAMEBUFFER_SRGB`
//! was never enabled). The new wgpu backend matches that: the swapchain is
//! configured as `Rgba8Unorm` or `Bgra8Unorm` (linear), so wgpu does NOT
//! auto-encode on write.
//!
//! ## Texture sampling
//!
//! | Texture tag          | wgpu format           | Sampler behavior                        |
//! |----------------------|----------------------|----------------------------------------|
//! | `Rgba8UnormSrgb`    | auto                 | Samples are auto-decoded to linear     |
//! | `Rgba8Unorm`        | auto                 | Samples return raw linear values       |
//!
//! The shader does not (and must not) call `S_SRGBToLinear` on any
//! `textureSample(...)` result.
//!
//! ## Shader output
//!
//! All fragment shaders write linear values to the swapchain. They do not
//! (and must not) call `S_LinearToSRGB`. The swapchain is linear; the OS
//! compositor handles the final encode.
//!
//! ## CPU-side authored sRGB values
//!
//! UI brush colors, vertex colors, decal colors are authored in sRGB
//! (0..255 bytes). They are converted to linear on the CPU before being
//! uploaded to a uniform buffer. The shader receives linear.
//!
//! ## Enforcement
//!
//! The `shader_lint` module (`crate::shader_lint`) runs on every compiled
//! shader and refuses to compile any WGSL containing `S_SRGBToLinear` or
//! `S_LinearToSRGB` calls.

use wgpu::TextureFormat;

/// Returns the preferred surface format for the given surface capabilities.
///
/// Prefers a **linear** format (`Rgba8Unorm` / `Bgra8Unorm`) over sRGB
/// variants. This matches the GL backend (which never enabled
/// `GL_FRAMEBUFFER_SRGB`) and avoids double-encoding under the linear
/// working-space convention.
///
/// Returns `None` if `caps` is empty.
pub fn surface_format_for(caps: &[TextureFormat]) -> Option<TextureFormat> {
    // Prefer linear (non-sRGB) over sRGB. The OS compositor does the final
    // linear→sRGB encode for display.
    caps.iter()
        .copied()
        .find(|f| !f.is_srgb())
        .or_else(|| caps.first().copied())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_linear_over_srgb() {
        let caps = vec![
            TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Rgba8Unorm,
            TextureFormat::Bgra8Unorm,
        ];
        assert_eq!(surface_format_for(&caps), Some(TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn falls_back_to_first_when_no_linear() {
        let caps = vec![TextureFormat::Rgba8UnormSrgb, TextureFormat::Bgra8UnormSrgb];
        // When no linear format is available, fall back to the first listed.
        // The fallback is suboptimal but not fatal.
        assert_eq!(
            surface_format_for(&caps),
            Some(TextureFormat::Rgba8UnormSrgb)
        );
    }

    #[test]
    fn empty_returns_none() {
        let caps: Vec<TextureFormat> = vec![];
        assert_eq!(surface_format_for(&caps), None);
    }

    #[test]
    fn single_linear_format_returned() {
        let caps = vec![TextureFormat::Rgba8Unorm];
        assert_eq!(surface_format_for(&caps), Some(TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn bgra_linear_preferred_over_rgba_srgb() {
        let caps = vec![TextureFormat::Rgba8UnormSrgb, TextureFormat::Bgra8Unorm];
        // Bgra8Unorm is linear and should be selected.
        assert_eq!(surface_format_for(&caps), Some(TextureFormat::Bgra8Unorm));
    }
}
